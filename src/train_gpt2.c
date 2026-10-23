/* train_gpt2.c — taped GPT-2 forward (recorded, out-of-place) + functional
 * fine-tuning loop.
 *
 * Forward mirrors gpt2.c op-for-op: wte+wpe embed, per block LayerNorm
 * (eps=1e-5) -> c_attn linear(3C, optional bias) -> qkv split (K=H, rep=1)
 * -> causal SDPA -> merge -> c_proj -> residual -> LayerNorm ->
 * c_fc -> gelu_new (tanh approx) -> c_proj -> residual; final LayerNorm,
 * tied lm head (dot with wte). Records every op into a Tape so the same
 * path trains and verifies.
 *
 * GPT-2 faithfulness needs two tape ops the NCP/Qwen paths never used:
 * t_layernorm_eps (GPT-2 epsilon is 1e-5, not 1e-6) and t_gelu_tanh
 * (HF gelu_new, not exact-erf). With those, the taped forward is
 * bit-comparable to gpt2.c (see gpt2_cmd_taped) and the tape backward is
 * exact per-op grads on both host and device.
 *
 * train_gpt2_run is a complete fine-tuning loop: random-row batches from
 * a (T+1)-id .nctrain shard, taped forward + CE, device backward, AdamW
 * with cosine schedule, NCK1 checkpoints with exact resume.
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GPT2_TRAIN_LN_EPS 1e-5

static float *G2W(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "train_gpt2: missing weight %s\n", name); exit(1); }
    return t->data;
}

static float *G2WO(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    return t ? t->data : NULL;
}

float *train_gpt2_block(Tape *tp, WeightStore *ws, const NCMConfig *c,
                        int li, float *x, int B, int T, const float *mask) {
    int C = c->n_embd, H = c->n_head, hd = C / H;
    char pre[64], kk[192];
    snprintf(pre, sizeof(pre), "h.%d.", li);
    float *ln = arena_alloc((size_t)B * T * C);
    snprintf(kk, sizeof(kk), "%sln_1.weight", pre);
    float *ln1w = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%sln_1.bias", pre);
    float *ln1b = G2WO(ws, kk);
    t_layernorm_eps(tp, x, ln1w, ln1b, ln, (int64_t)B * T, C, GPT2_TRAIN_LN_EPS);
    float *qkv = arena_alloc((size_t)B * T * 3 * C);
    snprintf(kk, sizeof(kk), "%sattn.c_attn.weight", pre);
    float *aw = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_attn.bias", pre);
    float *ab = G2WO(ws, kk);
    t_linear(tp, ln, aw, ab, qkv, (int64_t)B * T, 3 * C, C);
    float *qq = arena_alloc((size_t)B * H * T * hd);
    float *kkk = arena_alloc((size_t)B * H * T * hd);
    float *vv = arena_alloc((size_t)B * H * T * hd);
    t_qkvsplit(tp, qkv, qq, kkk, vv, B, T, C, H, hd, H, 1);
    float *ctx = arena_alloc((size_t)B * H * T * hd);
    t_sdpa(tp, qq, kkk, vv, mask, ctx, B, H, H, T, T, hd);
    float *mo = arena_alloc((size_t)B * T * C);
    t_headmerge(tp, ctx, mo, B, T, C, H, hd);
    snprintf(kk, sizeof(kk), "%sattn.c_proj.weight", pre);
    float *cw = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_proj.bias", pre);
    float *cb = G2WO(ws, kk);
    float *ao = arena_alloc((size_t)B * T * C);
    t_linear(tp, mo, cw, cb, ao, (int64_t)B * T, C, C);
    float *x1 = arena_alloc((size_t)B * T * C);
    t_add(tp, x, ao, x1, (size_t)B * T * C);
    float *mn = arena_alloc((size_t)B * T * C);
    snprintf(kk, sizeof(kk), "%sln_2.weight", pre);
    float *ln2w = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%sln_2.bias", pre);
    float *ln2b = G2WO(ws, kk);
    t_layernorm_eps(tp, x1, ln2w, ln2b, mn, (int64_t)B * T, C, GPT2_TRAIN_LN_EPS);
    int I = 4 * C;
    float *mm = arena_alloc((size_t)B * T * I);
    snprintf(kk, sizeof(kk), "%smlp.c_fc.weight", pre);
    float *fw = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%smlp.c_fc.bias", pre);
    float *fb = G2WO(ws, kk);
    t_linear(tp, mn, fw, fb, mm, (int64_t)B * T, I, C);
    float *m2 = arena_alloc((size_t)B * T * I);
    t_gelu_tanh(tp, mm, m2, (int64_t)B * T * I);
    float *mlpo = arena_alloc((size_t)B * T * C);
    snprintf(kk, sizeof(kk), "%smlp.c_proj.weight", pre);
    float *pw = G2W(ws, kk);
    snprintf(kk, sizeof(kk), "%smlp.c_proj.bias", pre);
    float *pb = G2WO(ws, kk);
    t_linear(tp, m2, pw, pb, mlpo, (int64_t)B * T, C, I);
    float *out = arena_alloc((size_t)B * T * C);
    t_add(tp, x1, mlpo, out, (size_t)B * T * C);
    return out;
}

/* Full taped forward: ids[B][T] host ints -> logits[B][T][V] (backend).
 * Builds its own causal mask (caller T-sized). */
float *train_gpt2_forward(Tape *tp, WeightStore *ws, const NCMConfig *c,
                          const int *ids, int B, int T) {
    int C = c->n_embd;
    Tensor *te = weights_get(ws, "wte.weight");
    if (!te) { fprintf(stderr, "train_gpt2: missing wte.weight\n"); exit(1); }
    int V = (int)te->shape[0];
    Tensor *pe_t = weights_get(ws, "wpe.weight");
    if (!pe_t) { fprintf(stderr, "train_gpt2: missing wpe.weight\n"); exit(1); }
    int npos = (int)pe_t->shape[0];
    if (T > npos) { fprintf(stderr, "train_gpt2: T=%d exceeds wpe rows=%d\n", T, npos); exit(1); }
    float *h = arena_alloc((size_t)B * T * C);
    /* token + position embeds (t_embed stages host idx itself) */
    float *xe = arena_alloc((size_t)B * T * C);
    t_embed(tp, te->data, ids, V, xe, (int64_t)B * T, C);
    int *ph = (int *)malloc((size_t)B * T * sizeof(int));
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) ph[(size_t)b * T + t] = t;
    float *pe = arena_alloc((size_t)B * T * C);
    t_embed(tp, pe_t->data, ph, npos, pe, (int64_t)B * T, C);
    free(ph);
    t_add(tp, xe, pe, h, (size_t)B * T * C);
    float *hm = (float *)malloc((size_t)T * T * sizeof(float));
    for (int tq = 0; tq < T; tq++)
        for (int tk = 0; tk < T; tk++)
            hm[(size_t)tq * T + tk] = (tk <= tq) ? 0.0f : -1e9f;
    float *mask = arena_alloc((size_t)T * T);
    mem_h2d(mask, hm, (size_t)T * T * sizeof(float));
    free(hm);
    for (int li = 0; li < c->n_layer; li++)
        h = train_gpt2_block(tp, ws, c, li, h, B, T, mask);
    float *lnf = arena_alloc((size_t)B * T * C);
    t_layernorm_eps(tp, h, G2W(ws, "ln_f.weight"), G2WO(ws, "ln_f.bias"),
                    lnf, (int64_t)B * T, C, GPT2_TRAIN_LN_EPS);
    float *logits = arena_alloc((size_t)B * T * V);
    /* plain dot-product head (wte is [V][C], same layout t_linear expects) */
    t_linear(tp, lnf, te->data, NULL, logits, (int64_t)B * T, V, C);
    return logits;
}

/* ---------------- functional fine-tuning loop ---------------- */

/* one step: taped forward + CE, device backward, global-norm clip 1.0,
 * AdamW update. xin/xout: host B*T ints. Returns mean loss. */
static double train_gpt2_step(WeightStore *ws, NCMConfig *c, AdamW *opt,
                              Tape *tp, const int *xin, const int *xout,
                              int B, int T, int V, double base_lr,
                              double warmup, double total) {
    tape_clear(tp);
    arena_reset();
    float *logits = train_gpt2_forward(tp, ws, c, xin, B, T);
    /* t_ce takes HOST targets (it copies them itself, like t_embed's idx) */
    float *loss1 = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
    t_ce(tp, logits, xout, NULL, loss1, probs, (int64_t)B * T, V, 1.0);
    PMap pm;
    pm.params = opt->params; pm.grads = opt->g;
    pm.sizes = opt->sizes; pm.n = opt->n;
    tape_backward(tp, &pm, 0);
    double g2 = 0;
    for (int i = 0; i < pm.n; i++) {
        int64_t n = pm.sizes[i];
        for (int64_t k = 0; k < n; k++)
            g2 += (double)pm.grads[i][k] * pm.grads[i][k];
    }
    double gnorm = sqrt(g2);
    if (gnorm > 1.0) {
        float s = (float)(1.0 / gnorm);
        for (int i = 0; i < pm.n; i++) {
            int64_t n = pm.sizes[i];
            for (int64_t k = 0; k < n; k++) pm.grads[i][k] *= s;
        }
    }
    float lh = 0;
    mem_d2h(&lh, loss1, sizeof(float));
    fprintf(stderr, "  [train-gpt2] loss=%.4f gnorm=%.3f\n", lh, gnorm);
    adamw_step(opt, base_lr, warmup, total);
    adamw_zero_grad(opt);
    return lh;
}

int train_gpt2_run(const NCMConfig *c_, const char *weights_path,
                   const char *data_path, int max_steps, double lr,
                   const TrainOpts *o_) {
    TrainOpts o;
    memset(&o, 0, sizeof(o));
    if (o_) o = *o_;
    int ckpt_every = o.ckpt_every > 0 ? o.ckpt_every : 10;
    const char *tokid = o.tokenizer_id ? o.tokenizer_id : "";
    NCMConfig c = *c_;
    WeightStore ws = {NULL, 0};

    int want_exact = 0;
    if (o.resume_path) {
        int probe = ckpt_probe(o.resume_path);
        if (probe == 1) want_exact = 1;
        else if (probe == 0) {
            fprintf(stderr,
                "[train-gpt2] LEGACY weights '%s': optimizer state RESET "
                "(fresh start, NOT an exact resume).\n", o.resume_path);
            weights_path = o.resume_path;
        } else {
            fprintf(stderr, "[train-gpt2] cannot read resume file '%s'\n", o.resume_path);
            return 1;
        }
    }

    if (weights_load(&ws, weights_path)) return 1;
    arena_init_default();
    Tensor *te = weights_get(&ws, "wte.weight");
    if (!te) { fprintf(stderr, "[train-gpt2] missing wte.weight\n"); return 1; }
    int V = (int)te->shape[0];
    AdamW *opt = adamw_new(&ws, NULL); /* fine-tune every parameter */
    Tape *tp = tape_new();
    if (!(lr > 0)) lr = 1e-5;
    double warmup = 100.0;
    long target = max_steps > 0 ? max_steps : 100;
    double sched_total = o.sched_total > 0 ? o.sched_total : (double)target;
    TrainData *d = train_data_load(data_path, 0);
    if (!d) return 1;
    int T = d->row_len - 1;
    if (T < 1) { fprintf(stderr, "[train-gpt2] row_len %d < 2\n", d->row_len); return 1; }
    uint64_t data_hash = ckpt_file_hash(data_path, NULL);
    uint64_t conf_hash = o.conf_path ? ckpt_file_hash(o.conf_path, NULL) : 0;

    long completed = 0;
    unsigned long long rng = 0x12345;

    if (want_exact) {
        CkptExpect ex;
        ex.batch_B = 1; ex.n_rows = d->n_rows; ex.row_len = d->row_len;
        ex.data_hash = data_hash; ex.conf_hash = conf_hash;
        ex.tokenizer_id = tokid;
        ex.base_lr = lr; ex.warmup = warmup; ex.total = sched_total;
        ex.b1 = opt->b1; ex.b2 = opt->b2;
        ex.eps = opt->eps; ex.wd = opt->wd;
        CkptRunState lrs;
        char err[512] = {0};
        int lrc = ckpt_load_training(o.resume_path, &ws, opt, &ex, &lrs, err, sizeof(err));
        if (lrc) {
            fprintf(stderr, "[train-gpt2] resume REFUSED: %s\n", err);
            return 1;
        }
        completed = lrs.global_step;
        rng = lrs.batch_rng;
        fprintf(stderr, "[train-gpt2] exact resume: global_step=%ld opt_step=%ld\n",
                completed, (long)opt->step);
        if (completed >= target) {
            printf("[train-gpt2] already complete (%ld >= %ld), nothing to do\n",
                   completed, target);
            return 0;
        }
    }

    printf("[train-gpt2] %ld rows, T=%d target=%ld schedule_total=%.0f%s\n",
           (long)d->n_rows, T, target, sched_total, want_exact ? " (resumed)" : "");
    int *xbuf = (int *)malloc((size_t)T * sizeof(int));
    int *ybuf = (int *)malloc((size_t)T * sizeof(int));
    char cfg_json[2048];
    snprintf(cfg_json, sizeof(cfg_json),
             "{\"lr\":%.17g,\"warmup\":%.17g,\"total\":%.17g,\"B\":1,"
             "\"arch\":\"gpt2\",\"tokenizer_id\":\"%s\",\"data\":\"%s\",\"conf\":\"%s\"}",
             lr, warmup, sched_total, tokid,
             data_path, o.conf_path ? o.conf_path : "");
    int rc = 0;
    while (completed < target) {
        rng = rng_next(&rng);
        int r = (int)(rng % (unsigned long long)d->n_rows);
        memcpy(xbuf, d->rows + r * d->row_len, (size_t)T * sizeof(int));
        memcpy(ybuf, d->rows + r * d->row_len + 1, (size_t)T * sizeof(int));
        double loss = train_gpt2_step(&ws, &c, opt, tp, xbuf, ybuf, 1, T, V,
                                      lr, warmup, sched_total);
        completed++;
        const char *save_path = NULL;
        char rot[256];
        if (completed % ckpt_every == 0 || completed == target) {
            if (o.ckpt_out) save_path = (char *)o.ckpt_out;
            else if (completed == target) save_path = "ckpt-gpt2-final.nckpt";
            else { snprintf(rot, sizeof(rot), "ckpt-gpt2-step%ld.nckpt", completed); save_path = rot; }
            CkptRunState rs;
            memset(&rs, 0, sizeof(rs));
            rs.global_step = completed; rs.batch_rng = rng;
            rs.base_lr = lr; rs.warmup = warmup; rs.total = sched_total;
            rs.batch_B = 1; rs.accum_pending = 0;
            rs.n_rows = d->n_rows; rs.row_len = d->row_len;
            rs.data_hash = data_hash; rs.data_path = data_path;
            rs.conf_hash = conf_hash; rs.tokenizer_id = tokid;
            rs.noise_sd = 0.0; rs.train_cfg_json = cfg_json;
            char err[512] = {0};
            struct timespec st0, st1;
            clock_gettime(CLOCK_MONOTONIC, &st0);
            int src_ = ckpt_save_atomic(&ws, opt, &rs, save_path, err, sizeof(err));
            clock_gettime(CLOCK_MONOTONIC, &st1);
            double sdt = (st1.tv_sec - st0.tv_sec) + (st1.tv_nsec - st0.tv_nsec) * 1e-9;
            if (src_) {
                fprintf(stderr, "[train-gpt2] CHECKPOINT SAVE FAILED: %s\n", err);
                rc = 1;
                break;
            }
            fprintf(stderr, "[train-gpt2] checkpoint saved in %.2fs\n", sdt);
        }
        printf("[train-gpt2] step %ld loss %.4f%s%s%s\n", completed - 1, loss,
               save_path ? " (saved " : "", save_path ? save_path : "",
               save_path ? ")" : "");
        fflush(stdout);
    }
    free(xbuf); free(ybuf);
    free(d->rows); free(d);
    return rc;
}
