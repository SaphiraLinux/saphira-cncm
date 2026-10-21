/* train_qwen.c — taped Qwen forward (recorded, out-of-place) + functional
 * fine-tuning loop.
 *
 * Forward mirrors qwen.c op-for-op (RMSNorm, q/k/v linears + bias,
 * headsplit, RoPE pair-duplicated tables, GQA SDPA with causal mask,
 * merge, o_proj, residual, RMSNorm, SwiGLU MLP), final RMSNorm, tied head.
 * Records every op into a Tape so the same path trains and verifies.
 *
 * GQA flows un-replicated: k/v stay at Hkv heads from headsplit through
 * rope(H,Hkv) and sdpa(H,Hkv). The NCP fused path instead replicates via
 * t_qkvsplit(rep); both are mathematically identical attention.
 *
 * train_qwen_run is a complete fine-tuning loop: random-row batches from
 * a (T+1)-id .nctrain shard, taped forward + CE, device backward,
 * AdamW with cosine schedule, NCK1 checkpoints with exact resume.
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float *QW(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "train_qwen: missing weight %s\n", name); exit(1); }
    return t->data;
}

/* Pair-duplicated RoPE tables, theta convention identical to qwen.c's
 * qwen_open (a = t * theta^(-2d/hd), duplicated over halves). Host fill;
 * caller stages to backend. */
void train_qwen_tables(float *co_out, float *si_out, int T, int hd, double theta) {
    int h = hd / 2;
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < h; d++) {
            double a = (double)t * pow(theta, -(double)(2 * d) / (double)hd);
            float co = (float)cos(a), si = (float)sin(a);
            co_out[(size_t)t * hd + d] = co;
            co_out[(size_t)t * hd + h + d] = co;
            si_out[(size_t)t * hd + d] = si;
            si_out[(size_t)t * hd + h + d] = si;
        }
    }
}

float *train_qwen_block(Tape *tp, WeightStore *ws, const NCMConfig *c,
                        int li, float *x, int B, int T,
                        const float *mask, const float *cos, const float *sin) {
    int C = c->n_embd, H = c->n_head, hd = C / H;
    int K = c->n_kv_heads > 0 ? c->n_kv_heads : H;
    char pre[128], kk[256];
    snprintf(pre, sizeof(pre), "model.layers.%d.", li);
    float *h1 = arena_alloc((size_t)B * T * C);
    snprintf(kk, sizeof(kk), "%sinput_layernorm.weight", pre);
    t_rmsnorm(tp, x, QW(ws, kk), h1, (int64_t)B * T, C);
    float *ql = arena_alloc((size_t)B * T * C);
    float *kl = arena_alloc((size_t)B * T * K * hd);
    float *vl = arena_alloc((size_t)B * T * K * hd);
    snprintf(kk, sizeof(kk), "%sself_attn.q_proj.weight", pre);
    float *qw = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%sself_attn.q_proj.bias", pre);
    float *qb = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%sself_attn.k_proj.weight", pre);
    float *kw = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%sself_attn.k_proj.bias", pre);
    float *kb = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%sself_attn.v_proj.weight", pre);
    float *vw = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%sself_attn.v_proj.bias", pre);
    float *vb = QW(ws, kk);
    t_linear(tp, h1, qw, qb, ql, (int64_t)B * T, C, C);
    t_linear(tp, h1, kw, kb, kl, (int64_t)B * T, K * hd, C);
    t_linear(tp, h1, vw, vb, vl, (int64_t)B * T, K * hd, C);
    float *qq = arena_alloc((size_t)B * H * T * hd);
    float *kkk = arena_alloc((size_t)B * K * T * hd);
    float *vv = arena_alloc((size_t)B * K * T * hd);
    t_headsplit(tp, ql, qq, B, T, C, H, hd);
    t_headsplit(tp, kl, kkk, B, T, K * hd, K, hd);
    t_headsplit(tp, vl, vv, B, T, K * hd, K, hd);
    float *qr = arena_alloc((size_t)B * H * T * hd);
    float *kr = arena_alloc((size_t)B * K * T * hd);
    t_rope(tp, qq, kkk, qr, kr, cos, sin, B, H, K, T, hd);
    float *ctx = arena_alloc((size_t)B * H * T * hd);
    t_sdpa(tp, qr, kr, vv, mask, ctx, B, H, K, T, T, hd);
    float *mo = arena_alloc((size_t)B * T * C);
    t_headmerge(tp, ctx, mo, B, T, C, H, hd);
    snprintf(kk, sizeof(kk), "%sself_attn.o_proj.weight", pre);
    float *ao = arena_alloc((size_t)B * T * C);
    t_linear(tp, mo, QW(ws, kk), NULL, ao, (int64_t)B * T, C, C);
    float *x1 = arena_alloc((size_t)B * T * C);
    t_add(tp, x, ao, x1, (size_t)B * T * C);
    float *h2 = arena_alloc((size_t)B * T * C);
    snprintf(kk, sizeof(kk), "%spost_attention_layernorm.weight", pre);
    t_rmsnorm(tp, x1, QW(ws, kk), h2, (int64_t)B * T, C);
    snprintf(kk, sizeof(kk), "%smlp.gate_proj.weight", pre);
    Tensor *tg = weights_get(ws, kk);
    if (!tg) { fprintf(stderr, "train_qwen: missing weight %s\n", kk); exit(1); }
    int I = (int)tg->shape[0];
    snprintf(kk, sizeof(kk), "%smlp.up_proj.weight", pre);
    float *upw = QW(ws, kk);
    snprintf(kk, sizeof(kk), "%smlp.down_proj.weight", pre);
    float *downw = QW(ws, kk);
    float *gt = arena_alloc((size_t)B * T * I);
    float *up = arena_alloc((size_t)B * T * I);
    t_linear(tp, h2, tg->data, NULL, gt, (int64_t)B * T, I, C);
    t_linear(tp, h2, upw, NULL, up, (int64_t)B * T, I, C);
    float *ga = arena_alloc((size_t)B * T * I);
    t_silu(tp, gt, ga, (int64_t)B * T * I);
    float *gu = arena_alloc((size_t)B * T * I);
    t_mul(tp, ga, up, gu, (int64_t)B * T * I);
    float *mlpo = arena_alloc((size_t)B * T * C);
    t_linear(tp, gu, downw, NULL, mlpo, (int64_t)B * T, C, I);
    float *out = arena_alloc((size_t)B * T * C);
    t_add(tp, x1, mlpo, out, (size_t)B * T * C);
    return out;
}

/* Full taped forward: ids[B][T] host ints -> logits[B][T][V] (backend).
 * Builds its own causal mask and RoPE tables (caller T-sized). */
float *train_qwen_forward(Tape *tp, WeightStore *ws, const NCMConfig *c,
                          const int *ids, int B, int T) {
    int C = c->n_embd, H = c->n_head, hd = C / H;
    Tensor *te = weights_get(ws, "model.embed_tokens.weight");
    if (!te) { fprintf(stderr, "train_qwen: missing model.embed_tokens.weight\n"); exit(1); }
    int V = (int)te->shape[0];
    double theta = c->rope_theta > 0 ? c->rope_theta : 1000000.0;
    float *h = arena_alloc((size_t)B * T * C);
    /* t_embed stages idx itself: pass the HOST array (cf. bwo_embed). */
    t_embed(tp, te->data, ids, V, h, (int64_t)B * T, C);
    float *hm = (float *)malloc((size_t)T * T * sizeof(float));
    for (int tq = 0; tq < T; tq++)
        for (int tk = 0; tk < T; tk++)
            hm[(size_t)tq * T + tk] = (tk <= tq) ? 0.0f : -1e9f;
    float *mask = arena_alloc((size_t)T * T);
    mem_h2d(mask, hm, (size_t)T * T * sizeof(float));
    free(hm);
    float *hco = (float *)malloc((size_t)T * hd * sizeof(float));
    float *hsi = (float *)malloc((size_t)T * hd * sizeof(float));
    train_qwen_tables(hco, hsi, T, hd, theta);
    float *cos = arena_alloc((size_t)T * hd);
    float *sin = arena_alloc((size_t)T * hd);
    mem_h2d(cos, hco, (size_t)T * hd * sizeof(float));
    mem_h2d(sin, hsi, (size_t)T * hd * sizeof(float));
    free(hco); free(hsi);
    for (int li = 0; li < c->n_layer; li++)
        h = train_qwen_block(tp, ws, c, li, h, B, T, mask, cos, sin);
    float *lnf = arena_alloc((size_t)B * T * C);
    t_rmsnorm(tp, h, QW(ws, "model.norm.weight"), lnf, (int64_t)B * T, C);
    float *logits = arena_alloc((size_t)B * T * V);
    /* plain dot-product head (E is [V][C], same layout t_linear expects);
     * t_tied is the NCP distance head and does NOT apply here. */
    t_linear(tp, lnf, te->data, NULL, logits, (int64_t)B * T, V, C);
    return logits;
}

/* ---------------- functional fine-tuning loop ---------------- */

/* one step: taped forward + CE, device backward, global-norm clip 1.0,
 * AdamW update. xin/xout: host B*T ints. Returns mean loss. */
static double train_qwen_step(WeightStore *ws, NCMConfig *c, AdamW *opt,
                              Tape *tp, const int *xin, const int *xout,
                              int B, int T, int V, double base_lr,
                              double warmup, double total) {
    tape_clear(tp);
    arena_reset();
    float *logits = train_qwen_forward(tp, ws, c, xin, B, T);
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
    fprintf(stderr, "  [train-qwen] loss=%.4f gnorm=%.3f\n", lh, gnorm);
    adamw_step(opt, base_lr, warmup, total);
    adamw_zero_grad(opt);
    return lh;
}

int train_qwen_run(const NCMConfig *c_, const char *weights_path,
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
                "[train-qwen] LEGACY weights '%s': optimizer state RESET "
                "(fresh start, NOT an exact resume).\n", o.resume_path);
            weights_path = o.resume_path;
        } else {
            fprintf(stderr, "[train-qwen] cannot read resume file '%s'\n", o.resume_path);
            return 1;
        }
    }

    if (weights_load(&ws, weights_path)) return 1;
    arena_init_default();
    Tensor *te = weights_get(&ws, "model.embed_tokens.weight");
    if (!te) { fprintf(stderr, "[train-qwen] missing model.embed_tokens.weight\n"); return 1; }
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
    if (T < 1) { fprintf(stderr, "[train-qwen] row_len %d < 2\n", d->row_len); return 1; }
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
            fprintf(stderr, "[train-qwen] resume REFUSED: %s\n", err);
            return 1;
        }
        completed = lrs.global_step;
        rng = lrs.batch_rng;
        fprintf(stderr, "[train-qwen] exact resume: global_step=%ld opt_step=%ld\n",
                completed, (long)opt->step);
        if (completed >= target) {
            printf("[train-qwen] already complete (%ld >= %ld), nothing to do\n",
                   completed, target);
            return 0;
        }
    }

    printf("[train-qwen] %ld rows, T=%d target=%ld schedule_total=%.0f%s\n",
           (long)d->n_rows, T, target, sched_total, want_exact ? " (resumed)" : "");
    int *xbuf = (int *)malloc((size_t)T * sizeof(int));
    int *ybuf = (int *)malloc((size_t)T * sizeof(int));
    char cfg_json[2048];
    snprintf(cfg_json, sizeof(cfg_json),
             "{\"lr\":%.17g,\"warmup\":%.17g,\"total\":%.17g,\"B\":1,"
             "\"tokenizer_id\":\"%s\",\"data\":\"%s\",\"conf\":\"%s\"}",
             lr, warmup, sched_total, tokid,
             data_path, o.conf_path ? o.conf_path : "");
    int rc = 0;
    while (completed < target) {
        rng = rng_next(&rng);
        int r = (int)(rng % (unsigned long long)d->n_rows);
        memcpy(xbuf, d->rows + r * d->row_len, (size_t)T * sizeof(int));
        memcpy(ybuf, d->rows + r * d->row_len + 1, (size_t)T * sizeof(int));
        double loss = train_qwen_step(&ws, &c, opt, tp, xbuf, ybuf, 1, T, V,
                                      lr, warmup, sched_total);
        completed++;
        const char *save_path = NULL;
        char rot[256];
        if (completed % ckpt_every == 0 || completed == target) {
            if (o.ckpt_out) save_path = (char *)o.ckpt_out;
            else if (completed == target) save_path = "ckpt-qwen-final.nckpt";
            else { snprintf(rot, sizeof(rot), "ckpt-qwen-step%ld.nckpt", completed); save_path = rot; }
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
                fprintf(stderr, "[train-qwen] CHECKPOINT SAVE FAILED: %s\n", err);
                rc = 1;
                break;
            }
            fprintf(stderr, "[train-qwen] checkpoint saved in %.2fs\n", sdt);
        }
        printf("[train-qwen] step %ld loss %.4f%s%s%s\n", completed - 1, loss,
               save_path ? " (saved " : "", save_path ? save_path : "",
               save_path ? ")" : "");
        fflush(stdout);
    }
    free(xbuf); free(ybuf);
    free(d->rows); free(d);
    return rc;
}
