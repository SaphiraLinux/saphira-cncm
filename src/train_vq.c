/* train_vq.c — VQ-VAE training: recording encoder/decoder, recon CE +
 * commitment loss, straight-through estimator, EMA codebook updates.
 *
 * Mirrors models/vqvae.py + models/vae.py forward; quantize step itself
 * carries no grads (argmin), straight-through routes decoder-input grads
 * back to the encoder output.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ncm.h"

/* ---- recording VAE block (mirrors vae_block, out-of-place) ---- */
static float *train_vae_block(Tape *tp, WeightStore *ws, const char *pre, float *x,
                              int B, int T, int C, int n_head, int use_qk_norm,
                              int attn_window) {
    int hd = C / n_head;
    float *ln = arena_alloc((size_t)B * T * C);
    char kw[320], kb[320];
    snprintf(kw, sizeof(kw), "%sln_1.weight", pre);
    snprintf(kb, sizeof(kb), "%sln_1.bias", pre);
    Tensor *tw = weights_get(ws, kw);
    Tensor *tb = weights_get(ws, kb);
    t_layernorm(tp, x, tw->data, tb ? tb->data : NULL, ln, (int64_t)B * T, C);

    snprintf(kw, sizeof(kw), "%sattn.c_attn.weight", pre);
    Tensor *tqkv = weights_get(ws, kw);
    snprintf(kb, sizeof(kb), "%sattn.c_attn.bias", pre);
    Tensor *bqkv = weights_get(ws, kb);
    float *qkv = arena_alloc((size_t)B * T * 3 * C);
    t_linear(tp, ln, tqkv->data, bqkv ? bqkv->data : NULL, qkv, (int64_t)B * T, C, 3 * C);

    float *qq = arena_alloc((size_t)B * n_head * T * hd);
    float *kkk = arena_alloc((size_t)B * n_head * T * hd);
    float *vv = arena_alloc((size_t)B * n_head * T * hd);
    /* host interleave is unrecorded -> use recorded split */
    t_qkvsplit(tp, qkv, qq, kkk, vv, B, T, C, n_head, hd, n_head, 1);
    if (use_qk_norm) {
        snprintf(kw, sizeof(kw), "%sattn.q_norm.weight", pre);
        float *qn = arena_alloc((size_t)B * n_head * T * hd);
        t_rmsnorm(tp, qq, weights_get(ws, kw)->data, qn, (int64_t)B * n_head * T, hd);
        qq = qn;
        snprintf(kw, sizeof(kw), "%sattn.k_norm.weight", pre);
        float *kn = arena_alloc((size_t)B * n_head * T * hd);
        t_rmsnorm(tp, kkk, weights_get(ws, kw)->data, kn, (int64_t)B * n_head * T, hd);
        kkk = kn;
    }
    float *mask = NULL;
    float *hmask = NULL;
    if (attn_window > 0) {
        hmask = (float *)malloc((size_t)T * T * sizeof(float));
        for (int i = 0; i < T; i++) for (int j = 0; j < T; j++)
            hmask[i * T + j] = (abs(i - j) <= attn_window) ? 0.0f : -INFINITY;
        mask = arena_alloc((size_t)T * T);
        mem_h2d(mask, hmask, (size_t)T * T * sizeof(float));
    }
    float *ctx = arena_alloc((size_t)B * n_head * T * hd);
    t_sdpa(tp, qq, kkk, vv, mask, ctx, B, n_head, T, T, hd);
    free(hmask);
    float *mo = arena_alloc((size_t)B * T * C);
    t_headmerge(tp, ctx, mo, B, T, C, n_head, hd);
    snprintf(kw, sizeof(kw), "%sattn.c_proj.weight", pre);
    Tensor *tcp = weights_get(ws, kw);
    snprintf(kb, sizeof(kb), "%sattn.c_proj.bias", pre);
    Tensor *bcp = weights_get(ws, kb);
    float *ao = arena_alloc((size_t)B * T * C);
    t_linear(tp, mo, tcp->data, bcp ? bcp->data : NULL, ao, (int64_t)B * T, C, C);
    float *x1 = arena_alloc((size_t)B * T * C);
    t_add(tp, x, ao, x1, (size_t)B * T * C);

    snprintf(kw, sizeof(kw), "%sln_2.weight", pre);
    tw = weights_get(ws, kw);
    snprintf(kb, sizeof(kb), "%sln_2.bias", pre);
    tb = weights_get(ws, kb);
    float *min = arena_alloc((size_t)B * T * C);
    t_layernorm(tp, x1, tw->data, tb ? tb->data : NULL, min, (int64_t)B * T, C);
    int H4;
    {
        snprintf(kw, sizeof(kw), "%smlp.c_fc.weight", pre);
        Tensor *tf = weights_get(ws, kw);
        H4 = (int)tf->shape[0];
    }
    snprintf(kw, sizeof(kw), "%smlp.c_fc.weight", pre);
    Tensor *tf = weights_get(ws, kw);
    snprintf(kb, sizeof(kb), "%smlp.c_fc.bias", pre);
    Tensor *bf = weights_get(ws, kb);
    float *m = arena_alloc((size_t)B * T * H4);
    t_linear(tp, min, tf->data, bf ? bf->data : NULL, m, (int64_t)B * T, H4, C);
    float *m2 = arena_alloc((size_t)B * T * H4);
    t_gelu(tp, m, m2, (int64_t)B * T * H4);
    snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
    Tensor *tpp = weights_get(ws, kw);
    snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
    Tensor *bp = weights_get(ws, kb);
    float *mlpo = arena_alloc((size_t)B * T * C);
    t_linear(tp, m2, tpp->data, bp ? bp->data : NULL, mlpo, (int64_t)B * T, C, H4);
    float *out = arena_alloc((size_t)B * T * C);
    t_add(tp, x1, mlpo, out, (size_t)B * T * C);
    return out;
}

/* ---------------- VQ-VAE training step ----------------
 * ids:[B][T] host. Returns total loss (recon CE + VQ commitment).
 * Codebooks frozen (EMA instead); everything else trains. */

struct VQTrainer {
    AdamW *opt;
    PMap *pm;
    VQVAEState *vqvae;
    VQState *vq;
    WeightStore *ws;
    NCMConfig *c;
    double lr, warmup, total;
};

static int is_frozen_vq(const char *name) {
    /* freeze codebooks (EMA handles them); train encoder/decoder/resi/convs */
    return strstr(name, "vq.embeddings.") != NULL;
}

VQTrainer *vqtrainer_new(VQVAEState *vqvae, WeightStore *ws, NCMConfig *c,
                         double lr, double wd, double warmup, double total) {
    VQTrainer *t = (VQTrainer *)calloc(1, sizeof(VQTrainer));
    t->vqvae = vqvae;
    t->vq = vqvae_vq(vqvae);
    t->ws = ws; t->c = c;
    t->lr = lr; t->warmup = warmup; t->total = total;
    /* AdamW over non-codebook VQ weights */
    int cap = ws->count, n = 0;
    const char **names = (const char **)malloc(cap * sizeof(char *));
    for (int i = 0; i < ws->count; i++) {
        if (strncmp(ws->items[i].name, "tok.", 4) != 0) continue;
        if (is_frozen_vq(ws->items[i].name)) continue;
        names[n++] = ws->items[i].name;
    }
    /* build AdamW manually over the named subset */
    t->opt = (AdamW *)calloc(1, sizeof(AdamW));
    t->opt->b1 = 0.9; t->opt->b2 = 0.95; t->opt->eps = 1e-8; t->opt->wd = wd;
    t->opt->n = n;
    t->opt->params = (float **)malloc(n * sizeof(float *));
    t->opt->sizes = (int64_t *)malloc(n * sizeof(int64_t));
    t->opt->m = (float **)malloc(n * sizeof(float *));
    t->opt->v = (float **)malloc(n * sizeof(float *));
    t->opt->g = (float **)malloc(n * sizeof(float *));
    t->opt->lr = (double *)malloc(n * sizeof(double));
    t->opt->step = 0;
    for (int i = 0; i < n; i++) {
        Tensor *tt = weights_get(ws, names[i]);
        t->opt->params[i] = tt->data;
        t->opt->sizes[i] = tt->n;
        t->opt->m[i] = (float *)calloc(tt->n, sizeof(float));
        t->opt->v[i] = (float *)calloc(tt->n, sizeof(float));
        t->opt->g[i] = (float *)calloc(tt->n, sizeof(float));
        t->opt->lr[i] = 1.0;
    }
    free(names);
    t->pm = pmap_from_adam(t->opt);
    return t;
}


/* ---- recording encoder (mirrors vqvae_encode_z) ---- */
static float *train_encode(Tape *tp, WeightStore *ws, NCMConfig *c,
                           const int *bids, int B, int T, int D) {
    char kw[256], kb[256];
    float *x = arena_alloc((size_t)B * T * D);
    t_embed(tp, weights_get(ws, "tok.encoder.wte.weight")->data, bids, c->vq_in_vocab, x, (int64_t)B * T, D);
    if (c->enc_positional) {
        /* wpe rows 0..T-1, broadcast over batch via gather */
        int *rows = (int *)malloc((size_t)B * T * sizeof(int));
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) rows[b * T + t] = t;
        float *pe = arena_alloc((size_t)B * T * D);
        t_gather_rows(tp, weights_get(ws, "tok.encoder.wpe.weight")->data, rows, B * T, D, pe);
        free(rows);
        float *x2 = arena_alloc((size_t)B * T * D);
        t_add(tp, x, pe, x2, (size_t)B * T * D);
        x = x2;
    }
    for (int i = 0; i < c->enc_conv_layers; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.encoder.conv_blocks.%d.", i);
        snprintf(kw, sizeof(kw), "%sln.weight", pre);
        float *ln = arena_alloc((size_t)B * T * D);
        t_layernorm(tp, x, weights_get(ws, kw)->data, NULL, ln, (int64_t)B * T, D);
        /* BCL staging for conv */
        float *bcl = arena_alloc((size_t)B * T * D);
        {
            float *h = (float*)backend_to_host(ln, (size_t)B * T * D * sizeof(float));
            float *hb = (float *)malloc((size_t)B * T * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                hb[((size_t)b * D + d) * T + t] = h[((size_t)b * T + t) * D + d];
            free(h);
            mem_h2d(bcl, hb, (size_t)B * T * D * sizeof(float));
            free(hb);
        }
        snprintf(kw, sizeof(kw), "%sc1.weight", pre);
        snprintf(kb, sizeof(kb), "%sc1.bias", pre);
        float *c1 = arena_alloc((size_t)B * T * D);
        t_conv(tp, bcl, weights_get(ws, kw)->data, weights_get(ws, kb)->data, c1,
               B, D, D, T, c->enc_conv_kernel, 1, c->enc_conv_kernel / 2);
        float *c1a = arena_alloc((size_t)B * T * D);
        t_gelu(tp, c1, c1a, (int64_t)B * T * D);
        snprintf(kw, sizeof(kw), "%sc2.weight", pre);
        snprintf(kb, sizeof(kb), "%sc2.bias", pre);
        float *c2 = arena_alloc((size_t)B * T * D);
        t_conv(tp, c1a, weights_get(ws, kw)->data, weights_get(ws, kb)->data, c2,
               B, D, D, T, c->enc_conv_kernel, 1, c->enc_conv_kernel / 2);
        /* back to BLD + residual */
        float *cb = arena_alloc((size_t)B * T * D);
        {
            float *h = (float*)backend_to_host(c2, (size_t)B * T * D * sizeof(float));
            float *hb = (float *)malloc((size_t)B * T * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                hb[((size_t)b * T + t) * D + d] = h[((size_t)b * D + d) * T + t];
            free(h);
            mem_h2d(cb, hb, (size_t)B * T * D * sizeof(float));
            free(hb);
        }
        float *xn = arena_alloc((size_t)B * T * D);
        t_add(tp, x, cb, xn, (size_t)B * T * D);
        x = xn;
    }
    int n_head_enc = c->vq_n_layers;
    for (int i = 0; i < c->enc_layers; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.encoder.blocks.%d.", i);
        x = train_vae_block(tp, ws, pre, x, B, T, D, n_head_enc, 0, c->enc_attn_window);
    }
    if (c->enc_layers > 0 || c->enc_conv_layers > 0) {
        float *ln = arena_alloc((size_t)B * T * D);
        t_layernorm(tp, x, weights_get(ws, "tok.encoder.ln_e.weight")->data, NULL, ln, (int64_t)B * T, D);
        x = ln;
    }
    /* compression down (released cfgs use 1; support general) */
    if (c->vq_compression > 1) {
        int f = c->vq_compression, T2 = T / f;
        float *bcl = arena_alloc((size_t)B * T * D);
        {
            float *h = (float*)backend_to_host(x, (size_t)B * T * D * sizeof(float));
            float *hb = (float *)malloc((size_t)B * T * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                hb[((size_t)b * D + d) * T + t] = h[((size_t)b * T + t) * D + d];
            free(h);
            mem_h2d(bcl, hb, (size_t)B * T * D * sizeof(float));
            free(hb);
        }
        float *cv = arena_alloc((size_t)B * T2 * D);
        t_conv(tp, bcl, weights_get(ws, "tok.encoder.down.weight")->data,
               weights_get(ws, "tok.encoder.down.bias")->data, cv, B, D, D, T, f, f, 0);
        float *z = arena_alloc((size_t)B * T2 * D);
        {
            float *h = (float*)backend_to_host(cv, (size_t)B * T2 * D * sizeof(float));
            float *hz = (float *)malloc((size_t)B * T2 * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T2; t++) for (int d = 0; d < D; d++)
                hz[((size_t)b * T2 + t) * D + d] = h[((size_t)b * D + d) * T2 + t];
            free(h);
            mem_h2d(z, hz, (size_t)B * T2 * D * sizeof(float));
            free(hz);
        }
        return z;
    }
    /* pre-quant norm */
    if (c->pre_quant_groupnorm > 0) {
        float *bcl = arena_alloc((size_t)B * T * D);
        {
            float *h = (float*)backend_to_host(x, (size_t)B * T * D * sizeof(float));
            float *hb = (float *)malloc((size_t)B * T * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                hb[((size_t)b * D + d) * T + t] = h[((size_t)b * T + t) * D + d];
            free(h);
            mem_h2d(bcl, hb, (size_t)B * T * D * sizeof(float));
            free(hb);
        }
        float *gn = arena_alloc((size_t)B * T * D);
        t_groupnorm(tp, bcl, gn, B, c->pre_quant_groupnorm, D, T);
        float *z = arena_alloc((size_t)B * T * D);
        {
            float *h = (float*)backend_to_host(gn, (size_t)B * T * D * sizeof(float));
            float *hz = (float *)malloc((size_t)B * T * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
                hz[((size_t)b * T + t) * D + d] = h[((size_t)b * D + d) * T + t];
            free(h);
            mem_h2d(z, hz, (size_t)B * T * D * sizeof(float));
            free(hz);
        }
        return z;
    }
    return x;
}

/* ---- recording decoder (mirrors vqvae_decode) ---- */
static float *train_decode(Tape *tp, WeightStore *ws, NCMConfig *c,
                           float *zhat, int B, int T, int D) {
    float *x = zhat;
    if (c->vq_compression > 1) { fprintf(stderr, "train: compression>1 decode unsupported\n"); exit(1); }
    if (c->dec_positional) {
        Tensor *wpe = weights_get(ws, "tok.decoder.wpe.weight");
        int *rows = (int *)malloc(T * sizeof(int));
        for (int t = 0; t < T; t++) rows[t] = t;
        /* broadcast wpe rows over batch: gather per batch */
        int *rb = (int *)malloc((size_t)B * T * sizeof(int));
        for (int b = 0; b < B; b++) memcpy(rb + b * T, rows, T * sizeof(int));
        free(rows);
        float *pe = arena_alloc((size_t)B * T * D);
        t_gather_rows(tp, wpe->data, rb, B * T, D, pe);
        free(rb);
        float *x2 = arena_alloc((size_t)B * T * D);
        t_add(tp, x, pe, x2, (size_t)B * T * D);
        x = x2;
    }
    x = train_vae_block(tp, ws, "tok.decoder.middle_block.", x, B, T, D, c->vq_n_layers, c->vq_qk_norm, 0);
    for (int i = 0; i < c->dec_layers - 1; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.decoder.extra_blocks.%d.", i);
        x = train_vae_block(tp, ws, pre, x, B, T, D, c->vq_n_layers, c->vq_qk_norm, 0);
    }
    {
        float *ln = arena_alloc((size_t)B * T * D);
        t_layernorm(tp, x, weights_get(ws, "tok.decoder.ln_f.weight")->data, NULL, ln, (int64_t)B * T, D);
        x = ln;
    }
    return x;
}

/* full decoder + output head -> logits [B][T][V] */
static float *train_decode_logits(Tape *tp, WeightStore *ws, NCMConfig *c,
                                  float *zhat, int B, int T, int D) {
    float *x = train_decode(tp, ws, c, zhat, B, T, D);
    int V = c->vq_in_vocab;
    float *logits = arena_alloc((size_t)B * T * V);
    t_linear(tp, x, weights_get(ws, "tok.decoder.output_layer.weight")->data, NULL, logits, (int64_t)B * T, V, D);
    return logits;
}

/* ---------------- VQ-VAE training step ---------------- */
double train_vq_step(VQTrainer *t, Tape *tp, const int *ids, int B) {
    NCMConfig *c = t->c;
    WeightStore *ws = t->ws;
    VQState *vq = t->vq;
    int T = c->vq_context_length, D = c->vq_embed_dim, V = c->vq_in_vocab;
    tape_clear(tp);
    arena_reset();
    int *bids = (int *)arena_alloc((size_t)B * T);
    mem_h2d(bids, ids, (size_t)B * T * sizeof(int));
    /* encoder (recorded) */
    float *z = train_encode(tp, ws, c, bids, B, T, D);
    /* quantize (no grad): codes host + zhat backend */
    int total = vq_total_codes(vq);
    (void)total;
    int *codes = (int *)malloc((size_t)B * vq_total_codes(vq) * sizeof(int));
    /* stage z to host for argmin? use backend vq_z_to_idx (unrecorded) */
    vq_z_to_idx(vq, z, B, codes);
    float **zhats = (float **)malloc(vq_nlevels(vq) * sizeof(float *));
    for (int i = 0; i < vq_nlevels(vq); i++)
        zhats[i] = arena_alloc((size_t)B * vq_Ltotal(vq) * D);
    vq_idx_to_zhats(vq, codes, B, zhats);
    /* straight-through: z_st = z + (zhat_last - z).detach() */
    float *delta = arena_alloc((size_t)B * vq_Ltotal(vq) * D);
    {
        float *hz = (float*)backend_to_host(z, (size_t)B * vq_Ltotal(vq) * D * sizeof(float));
        float *hh = (float*)backend_to_host(zhats[vq_nlevels(vq) - 1], (size_t)B * vq_Ltotal(vq) * D * sizeof(float));
        float *hd = (float *)malloc((size_t)B * vq_Ltotal(vq) * D * sizeof(float));
        for (size_t i = 0; i < (size_t)B * vq_Ltotal(vq) * D; i++) hd[i] = hh[i] - hz[i];
        free(hz); free(hh);
        mem_h2d(delta, hd, (size_t)B * vq_Ltotal(vq) * D * sizeof(float));
        free(hd);
    }
    float *z_st = arena_alloc((size_t)B * vq_Ltotal(vq) * D);
    t_add(tp, z, delta, z_st, (size_t)B * vq_Ltotal(vq) * D);
    /* decoder (recorded) */
    float *logits = train_decode_logits(tp, ws, c, z_st, B, T, D);
    /* recon CE */
    float *loss_r = arena_alloc(1);
    float *probs = arena_alloc((size_t)B * T * V);
    t_ce(tp, logits, ids, NULL, loss_r, probs, (int64_t)B * T, V, 1.0);
    /* commitment per level: 0.25/nlev * mse(z, zhat_lvl.detach) */
    double commit = 0;
    for (int lv = 0; lv < vq_nlevels(vq); lv++) {
        float *lp = arena_alloc(1);
        t_mse(tp, z, zhats[lv], lp, (size_t)B * vq_Ltotal(vq) * D, 0.25 / vq_nlevels(vq));
        float hl = 0;
        mem_d2h(&hl, lp, sizeof(float));
        commit += hl;
    }
    float hr = 0;
    mem_d2h(&hr, loss_r, sizeof(float));
    /* EMA codebook update (no grad) */
    if (!getenv("CNCM_NOEMA")) vq_ema_update(vq, z, codes, B, 0.99, 1e-5);
    free(codes);
    free(zhats);
    return hr + commit;
}

void vqtrainer_free(VQTrainer *t) {
    if (!t) return;
    adamw_free(t->opt);
    free(t->pm);
    free(t);
}

/* driver: token-id rows [B][T] -> recon + commitment, AdamW, EMA, checkpoint */
int train_vq_run(const NCMConfig *c_, const char *weights_path, const char *data_path, int max_steps) {
    NCMConfig c = *c_;
    WeightStore ws = {NULL, 0};
    if (weights_load(&ws, weights_path)) return 1;
    arena_init(1ULL << 30);
    parena_init(1ULL << 28);
    VQVAEState *vqvae = vqvae_load(&c, &ws);
    Tape *tp = tape_new();
    VQTrainer *t = vqtrainer_new(vqvae, &ws, &c, 3e-4, 0.0, 100, max_steps > 0 ? max_steps : 1000);
    /* token-id data uses same .nctrain container, row_len must equal T */
    TrainData *d = train_data_load(data_path, 0);
    if (!d) return 1;
    if (d->row_len != c.vq_context_length) {
        fprintf(stderr, "row_len %d != T %d (need token-id rows)\n", d->row_len, c.vq_context_length);
        return 1;
    }
    int T = c.vq_context_length;
    printf("[train-vq] %d rows x %d, steps=%d\n", d->n_rows, d->row_len, max_steps);
    int B = 4;
    unsigned long long rng = 0x12345;
    int *buf = (int *)malloc(B * T * sizeof(int));
    for (int step = 0; step < (max_steps > 0 ? max_steps : 200); step++) {
        for (int bb = 0; bb < B; bb++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            int r = (int)((rng >> 11) % (unsigned long long)d->n_rows);
            memcpy(buf + bb * T, d->rows + r * d->row_len, T * sizeof(int));
        }
        double loss = train_vq_step(t, tp, buf, B);
        /* backward + step */
        PMap *pm = pmap_from_adam(t->opt);
        extern void tape_backward(Tape *t, PMap *pm, float seed);
        tape_backward(tp, pm, 0);
        pmap_free(pm);
        /* grad clip */
        {
            double g2 = 0;
            for (int i = 0; i < t->opt->n; i++) {
                int64_t n = t->opt->sizes[i];
                for (int64_t k = 0; k < n; k++) g2 += (double)t->opt->g[i][k] * t->opt->g[i][k];
            }
            double gnorm = sqrt(g2), clip = 1.0;
            if (gnorm > clip) {
                double s = clip / gnorm;
                for (int i = 0; i < t->opt->n; i++) {
                    int64_t n = t->opt->sizes[i];
                    for (int64_t k = 0; k < n; k++) t->opt->g[i][k] = (float)(t->opt->g[i][k] * s);
                }
            }
            printf("  [train-vq] step %d loss=%.4f gnorm=%.3f\n", step, loss, gnorm);
            fflush(stdout);
        }
        adamw_step(t->opt, t->lr, t->warmup, t->total);
        adamw_zero_grad(t->opt);
        if (step % 10 == 0) {
            char path[256];
            snprintf(path, sizeof(path), "ckpt-vq-step%d.ncm", step);
                weights_save(&ws, path);
            printf("[train-vq] step %d loss %.4f (saved %s)\n", step, loss, path);
            fflush(stdout);
        }
    }
    weights_save(&ws, "ckpt-vq-final.ncm");
    free(buf); train_data_free(d);
    vqtrainer_free(t);
    tape_free(tp);
    return 0;
}
