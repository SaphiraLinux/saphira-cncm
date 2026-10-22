/* train.c — AdamW + recording NCP forward + training steps + checkpointing.
 *
 * Reference training (correctness first). Forward runs on the backend;
 * backward stages to host (see train_tape.c). CPU-adequate speed.
 */
#include "ncm.h"
#include "sudoku_size.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------- AdamW ---------------- */

/* AdamW struct in ncm.h */
AdamW *adamw_new(WeightStore *ws, const int *frozen) {
    AdamW *o = (AdamW *)calloc(1, sizeof(AdamW));
    o->b1 = 0.9; o->b2 = 0.95; o->eps = 1e-8; o->wd = 0.01;
    o->step = 0;
    int cap = ws->count;
    o->params = (float **)malloc(cap * sizeof(float *));
    o->m = (float **)malloc(cap * sizeof(float *));
    o->v = (float **)malloc(cap * sizeof(float *));
    o->g = (float **)malloc(cap * sizeof(float *));
    o->sizes = (int64_t *)malloc(cap * sizeof(int64_t));
    o->lr = (double *)malloc(cap * sizeof(double));
    for (int i = 0; i < ws->count; i++) {
        if (frozen && frozen[i]) continue;
        /* skip non-trainable: codebooks are frozen by caller */
        o->params[o->n] = ws->items[i].t->data;
        o->sizes[o->n] = ws->items[i].t->n;
        o->m[o->n] = (float *)calloc(ws->items[i].t->n, sizeof(float));
        o->v[o->n] = (float *)calloc(ws->items[i].t->n, sizeof(float));
        o->g[o->n] = (float *)calloc(ws->items[i].t->n, sizeof(float));
        o->lr[o->n] = 1.0;
        o->n++;
    }
    return o;
}

void adamw_zero_grad(AdamW *o) {
    for (int i = 0; i < o->n; i++)
        memset(o->g[i], 0, o->sizes[i] * sizeof(float));
}

#ifdef NCM_CUDA
/* Device AdamW update. Bit-identical to the host loop below: same ops,
 * same order per element (one thread per element, disjoint outputs). */
__global__ static void k_adamw(float *p, const float *g, float *m, float *v, int64_t n,
                               double lr, double b1, double b2, double b1t, double b2t,
                               double eps, double wd) {
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float gk = g[k];
    p[k] = p[k] - (float)(lr * wd * p[k]);
    float mk = (float)(b1 * m[k] + (1.0 - b1) * gk);
    float vk = (float)(b2 * v[k] + (1.0 - b2) * gk * gk);
    m[k] = mk;
    v[k] = vk;
    double mh = mk / b1t, vh = vk / b2t;
    p[k] = p[k] - (float)(lr * mh / (sqrt(vh) + eps));
}

/* Allocate device mirrors, seeded FROM the host moments (which carry
 * the true state both on fresh start (zeros) and on exact resume). */
static void adamw_dev_init(AdamW *o) {
    o->md = (Tensor **)calloc((size_t)o->n, sizeof(Tensor *));
    o->vd = (Tensor **)calloc((size_t)o->n, sizeof(Tensor *));
    o->gd = (Tensor **)calloc((size_t)o->n, sizeof(Tensor *));
    for (int i = 0; i < o->n; i++) {
        o->md[i] = tensor_alloc(1, &o->sizes[i]);
        o->vd[i] = tensor_alloc(1, &o->sizes[i]);
        o->gd[i] = tensor_alloc(1, &o->sizes[i]);
        mem_h2d(o->md[i]->data, o->m[i], (size_t)o->sizes[i] * sizeof(float));
        mem_h2d(o->vd[i]->data, o->v[i], (size_t)o->sizes[i] * sizeof(float));
    }
    o->dev_ready = 1;
}

#endif

/* Push device moments back to host (checkpoint save path). No-op on CPU. */
void adamw_sync_to_host(AdamW *o) {
    if (!o) return;
#ifdef NCM_CUDA
    if (!o->dev_ready) return; /* nothing ever ran on device */
    /* CUDA kernels are launched without explicit sync; the transfers below
     * are ordered after them on the default stream, so this is safe. */
    for (int i = 0; i < o->n; i++) {
        mem_d2h(o->m[i], o->md[i]->data, (size_t)o->sizes[i] * sizeof(float));
        mem_d2h(o->v[i], o->vd[i]->data, (size_t)o->sizes[i] * sizeof(float));
    }
#else
    (void)o;
#endif
}

/* single AdamW step over all params; lr = base * cosine(warmup,total) */
static double sched_cosine_warmup(long step, double warmup, double total);
void adamw_step(AdamW *o, double base_lr, double warmup, double total) {
    o->step++;
    double lr_t = base_lr * sched_cosine_warmup(o->step, warmup, total);
    double b1t = 1.0 - pow(o->b1, (double)o->step);
    double b2t = 1.0 - pow(o->b2, (double)o->step);
#ifdef NCM_CUDA
    {
        const char *ad = getenv("NCM_ADAMW");
        int on_device = !(ad && strcmp(ad, "cpu") == 0);
        if (on_device) {
            if (!o->dev_ready) adamw_dev_init(o);
            for (int i = 0; i < o->n; i++) {
                int64_t n = o->sizes[i];
                double lr = lr_t * o->lr[i];
                mem_h2d(o->gd[i]->data, o->g[i], (size_t)n * sizeof(float));
                int blocks = (int)((n + 255) / 256);
                k_adamw<<<blocks, 256>>>((float *)o->params[i], (const float *)o->gd[i]->data,
                                         (float *)o->md[i]->data, (float *)o->vd[i]->data, n,
                                         lr, o->b1, o->b2, b1t, b2t, o->eps, o->wd);
            }
            return;
        }
    }
#endif
    /* independent per param: same serial arithmetic in every thread */
#pragma omp parallel for schedule(static) if(o->n > 1)
    for (int i = 0; i < o->n; i++) {
        int64_t n = o->sizes[i];
        float *p_host = (float *)malloc(n * sizeof(float));
        mem_d2h(p_host, o->params[i], n * sizeof(float));
        double lr = lr_t * o->lr[i];
        for (int64_t k = 0; k < n; k++) {
            float g = o->g[i][k];
            /* decoupled weight decay */
            p_host[k] -= (float)(lr * o->wd * p_host[k]);
            o->m[i][k] = (float)(o->b1 * o->m[i][k] + (1 - o->b1) * g);
            o->v[i][k] = (float)(o->b2 * o->v[i][k] + (1 - o->b2) * g * g);
            double mh = o->m[i][k] / b1t, vh = o->v[i][k] / b2t;
            p_host[k] -= (float)(lr * mh / (sqrt(vh) + o->eps));
        }
        mem_h2d(o->params[i], p_host, n * sizeof(float));
        free(p_host);
    }
}

void adamw_free(AdamW *o) {
    if (!o) return;
    for (int i = 0; i < o->n; i++) { free(o->m[i]); free(o->v[i]); free(o->g[i]); }
#ifdef NCM_CUDA
    if (o->dev_ready) {
        for (int i = 0; i < o->n; i++) { tensor_free(o->md[i]); tensor_free(o->vd[i]); tensor_free(o->gd[i]); }
        free(o->md); free(o->vd); free(o->gd);
    }
#endif
    free(o->params); free(o->m); free(o->v); free(o->g);
    free(o->sizes); free(o->lr);
    free(o);
}

/* PMap view sharing the optimizer's grad buffers (no copy). */
PMap *pmap_from_adam(AdamW *o) {
    PMap *pm = (PMap *)malloc(sizeof(PMap));
    pm->params = o->params;
    pm->grads = o->g;
    pm->sizes = o->sizes;
    pm->n = o->n;
    return pm;
}
void pmap_free(PMap *pm) { free(pm); }

static double sched_cosine_warmup(long step, double warmup, double total) {
    double w = warmup, T = total;
    if (step < w) return step / (w > 0 ? w : 1);
    if (step >= T) return 0.0;
    double x = (step - w) / (T - w);
    return 0.5 * (1 + cos(3.141592653589793 * x));
}


/* ---------------- recording NCP forward (training) ---------------- */

/* live-viewer capture (train_ncp_loss fills these when view_cap_on; the
 * train loop copies them into snapshots). Read-only staging + host
 * argmax: no RNG, no math, no arena impact — numerics and determinism
 * are untouched. Viewer off: a single branch per level. */
static int view_cap_on = 0;
static double view_lvl_loss[VIEW_MAXLV];
static double view_lvl_agree[VIEW_MAXLV];

static int vq_total_dummy(NCMConfig *c);

static Tensor *TW(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "missing weight %s\n", name); exit(1); }
    return t;
}

/* ---------------- NCP training forward (recorded) ----------------
 * Mirrors ncp_forward (same math, tape-recorded, out-of-place).
 * x:[B][P+T][Dvq] backend from frozen encoder/VQ. l0q expanded host array
 * (or NULL when lvl_first==1). Returns mean CE over target levels (host).
 * All released checkpoints use RoPE; non-RoPE raises.
 */

static int *train_lvl_idx(NCMConfig *c, int lvl_first, int *out_T) {
    int T = 0;
    for (int k = lvl_first; k < c->n_qlevels; k++) T += c->levels[k];
    int *idx = (int *)malloc(T * sizeof(int));
    int o = 0;
    for (int k = 0; k < c->n_qlevels - lvl_first; k++)
        for (int t = 0; t < c->levels[k + lvl_first]; t++) idx[o++] = k;
    if (out_T) *out_T = T;
    return idx;
}

static float *train_build_mask(NCMConfig *c, int P, int T, int lvl_first) {
    int S = P + T;
    float *m = (float *)malloc((size_t)S * S * sizeof(float));
    int *lv = train_lvl_idx(c, lvl_first, NULL);
    for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) {
        float v = 0;
        if (i < P && j >= P) v = -INFINITY;
        else if (i >= P && j >= P) {
            if (c->attn_pattern_block_diag) v = (lv[i - P] == lv[j - P]) ? 0.0f : -INFINITY;
            else v = (lv[i - P] >= lv[j - P]) ? 0.0f : -INFINITY;
        }
        m[i * S + j] = v;
    }
    free(lv);
    return m;
}

/* one recorded transformer block, out-of-place. x:[B][S][C] backend. */
static float *train_block(Tape *tp, WeightStore *ws, NCMConfig *c,
                          int li, float *x, int B, int S,
                          const float *mask, const float *cos, const float *sin) {
    int C = c->n_embd, H = c->n_head, hd = C / H;
    int nkv = c->n_kv_heads;
    char pre[128], kk[256];
    snprintf(pre, sizeof(pre), "gen.transformer.blocks.%d.", li);
    float *ln = arena_alloc((size_t)B * S * C);
    snprintf(kk, sizeof(kk), "%sln_1.weight", pre);
    Tensor *tw = TW(ws, kk);
    snprintf(kk, sizeof(kk), "%sln_1.bias", pre);
    Tensor *tb = weights_get(ws, kk);
    t_layernorm(tp, x, tw->data, tb ? tb->data : NULL, ln, (int64_t)B * S, C);

    snprintf(kk, sizeof(kk), "%sattn.c_attn.weight", pre);
    Tensor *tqkv = TW(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_attn.bias", pre);
    Tensor *bqkv = weights_get(ws, kk);
    float *qkv = arena_alloc((size_t)B * S * (C + 2 * nkv * hd));
    t_linear(tp, ln, tqkv->data, bqkv ? bqkv->data : NULL, qkv, (int64_t)B * S, C + 2 * nkv * hd, C);

    float *qq = arena_alloc((size_t)B * H * S * hd);
    float *kkk = arena_alloc((size_t)B * H * S * hd);
    float *vv = arena_alloc((size_t)B * H * S * hd);
    t_qkvsplit(tp, qkv, qq, kkk, vv, B, S, C, H, hd, nkv, H / nkv);
    float *qqn = qq, *kkn = kkk;
    if (c->use_qk_norm) {
        snprintf(kk, sizeof(kk), "%sattn.q_norm.weight", pre);
        qqn = arena_alloc((size_t)B * H * S * hd);
        t_rmsnorm(tp, qq, TW(ws, kk)->data, qqn, (int64_t)B * H * S, hd);
        snprintf(kk, sizeof(kk), "%sattn.k_norm.weight", pre);
        kkn = arena_alloc((size_t)B * H * S * hd);
        t_rmsnorm(tp, kkk, TW(ws, kk)->data, kkn, (int64_t)B * H * S, hd);
    }
    float *qr = arena_alloc((size_t)B * H * S * hd);
    float *kr = arena_alloc((size_t)B * H * S * hd);
    if (!c->use_rope) { fprintf(stderr, "train: non-RoPE configs unsupported\n"); exit(1); }
    t_rope(tp, qqn, kkn, qr, kr, cos, sin, B, H, H, S, hd);
    float *ctx = arena_alloc((size_t)B * H * S * hd);
    t_sdpa(tp, qr, kr, vv, mask, ctx, B, H, H, S, S, hd);
    float *mo = arena_alloc((size_t)B * S * C);
    t_headmerge(tp, ctx, mo, B, S, C, H, hd);
    snprintf(kk, sizeof(kk), "%sattn.c_proj.weight", pre);
    Tensor *tcp = TW(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_proj.bias", pre);
    Tensor *bcp = weights_get(ws, kk);
    float *ao = arena_alloc((size_t)B * S * C);
    t_linear(tp, mo, tcp->data, bcp ? bcp->data : NULL, ao, (int64_t)B * S, C, C);
    float *x1 = arena_alloc((size_t)B * S * C);
    t_add(tp, x, ao, x1, (size_t)B * S * C);

    snprintf(kk, sizeof(kk), "%sln_2.weight", pre);
    tw = TW(ws, kk);
    snprintf(kk, sizeof(kk), "%sln_2.bias", pre);
    tb = weights_get(ws, kk);
    float *min = arena_alloc((size_t)B * S * C);
    t_layernorm(tp, x1, tw->data, tb ? tb->data : NULL, min, (int64_t)B * S, C);
    float *mlpo = arena_alloc((size_t)B * S * C);
    {
        char kw[256], kb[256];
        int gated = c->use_swiglu || c->use_geglu;
        if (gated) {
            snprintf(kw, sizeof(kw), "%smlp.w_gate.weight", pre);
            Tensor *tg = TW(ws, kw);
            int Hh = (int)tg->shape[0];
            snprintf(kb, sizeof(kb), "%smlp.w_gate.bias", pre);
            Tensor *bg = weights_get(ws, kb);
            snprintf(kw, sizeof(kw), "%smlp.w_up.weight", pre);
            Tensor *tu = TW(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.w_up.bias", pre);
            Tensor *bu = weights_get(ws, kb);
            float *g = arena_alloc((size_t)B * S * Hh);
            float *u = arena_alloc((size_t)B * S * Hh);
            t_linear(tp, min, tg->data, bg ? bg->data : NULL, g, (int64_t)B * S, Hh, C);
            t_linear(tp, min, tu->data, bu ? bu->data : NULL, u, (int64_t)B * S, Hh, C);
            float *ga = arena_alloc((size_t)B * S * Hh);
            if (c->use_swiglu) t_silu(tp, g, ga, (int64_t)B * S * Hh);
            else t_gelu(tp, g, ga, (int64_t)B * S * Hh);
            float *gu = arena_alloc((size_t)B * S * Hh);
            t_mul(tp, ga, u, gu, (int64_t)B * S * Hh);
            snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
            Tensor *tpp = TW(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
            Tensor *bp = weights_get(ws, kb);
            t_linear(tp, gu, tpp->data, bp ? bp->data : NULL, mlpo, (int64_t)B * S, C, Hh);
        } else {
            snprintf(kw, sizeof(kw), "%smlp.c_fc.weight", pre);
            Tensor *tf = TW(ws, kw);
            int Hh = (int)tf->shape[0];
            snprintf(kb, sizeof(kb), "%smlp.c_fc.bias", pre);
            Tensor *bf = weights_get(ws, kb);
            float *m = arena_alloc((size_t)B * S * Hh);
            t_linear(tp, min, tf->data, bf ? bf->data : NULL, m, (int64_t)B * S, Hh, C);
            if (c->use_relu2) {
                float *m2 = arena_alloc((size_t)B * S * Hh);
                t_relu2(tp, m, m2, (int64_t)B * S * Hh);
                m = m2;
            } else {
                float *m2 = arena_alloc((size_t)B * S * Hh);
                t_gelu(tp, m, m2, (int64_t)B * S * Hh);
                m = m2;
            }
            snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
            Tensor *tpp = TW(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
            Tensor *bp = weights_get(ws, kb);
            t_linear(tp, m, tpp->data, bp ? bp->data : NULL, mlpo, (int64_t)B * S, C, Hh);
        }
    }
    float *out = arena_alloc((size_t)B * S * C);
    t_add(tp, x1, mlpo, out, (size_t)B * S * C);
    return out;
}

/* recording NCP forward + per-level CE. codes:[B][total] per-board level-major
 * (host). prefix:[B][P] host ints. l0q: [B][l0][D] host or NULL (non-query).
 * Returns mean CE (host). Records everything trainable into tp. */
static double train_ncp_loss(Tape *tp, NCPState *m, WeightStore *ws, NCMConfig *c,
                             const float *x, const int *prefix_ids, const int *codes,
                             const float *l0q_host, int B, double *out_ppl) {
    int P = ncp_P(m), T = ncp_T(m), S = P + T, C = c->n_embd, D = c->vq_embed_dim;
    int lvl_first = ncp_lvlfirst(m);
    /* NOTE: x slicing is frozen (no grad needed upstream): plain staging. */
    float *zpre = arena_alloc((size_t)B * P * D);
    float *xt = arena_alloc((size_t)B * T * D);
    {
        float *h = (float*)backend_to_host(x, (size_t)(B * S * D) * sizeof(float));
        float *a = (float *)malloc((size_t)B * P * D * sizeof(float));
        float *b2 = (float *)malloc((size_t)B * T * D * sizeof(float));
        for (int bb = 0; bb < B; bb++) {
            memcpy(a + (size_t)bb * P * D, h + (size_t)bb * S * D, (size_t)P * D * sizeof(float));
            memcpy(b2 + (size_t)bb * T * D, h + ((size_t)bb * S + P) * D, (size_t)T * D * sizeof(float));
        }
        free(h);
        mem_h2d(zpre, a, (size_t)B * P * D * sizeof(float));
        mem_h2d(xt, b2, (size_t)B * T * D * sizeof(float));
        free(a); free(b2);
    }
    if (c->puc_enabled) {
        float *h = (float*)backend_to_host(xt, (size_t)(B * T * D) * sizeof(float));
        int off = 0;
        for (int k = 0; k < ncp_ntgt(m); k++) {
            int lk = c->levels[k + lvl_first];
            char kw[128], kb[128];
            snprintf(kw, sizeof(kw), "gen.post_upsample_convs.%d.weight", k);
            snprintf(kb, sizeof(kb), "gen.post_upsample_convs.%d.bias", k);
            Tensor *tw = TW(ws, kw), *tb = TW(ws, kb);
            float *cb = (float *)malloc((size_t)B * lk * D * sizeof(float));
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++) for (int d = 0; d < D; d++)
                cb[((size_t)bb * D + d) * lk + t] = h[((size_t)bb * T + off + t) * D + d];
            float *dbc = arena_alloc((size_t)B * lk * D);
            float *dout = arena_alloc((size_t)B * lk * D);
            mem_h2d(dbc, cb, (size_t)B * lk * D * sizeof(float));
            t_conv(tp, dbc, tw->data, tb->data, dout, B, D, D, lk, c->puc_kernel, 1, c->puc_kernel / 2);
            float *co = (float*)backend_to_host(dout, (size_t)(B * lk * D) * sizeof(float));
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++) for (int d = 0; d < D; d++)
                h[((size_t)bb * T + off + t) * D + d] += co[((size_t)bb * D + d) * lk + t];
            free(cb); free(co);
            off += lk;
        }
        mem_h2d(xt, h, (size_t)B * T * D * sizeof(float));
        free(h);
    }
    float *pemb = arena_alloc((size_t)B * P * C);
    {
        Tensor *tw = TW(ws, "gen.transformer.wte.weight");
        Tensor *tb = weights_get(ws, "gen.transformer.wte.bias");
        t_linear(tp, zpre, tw->data, tb ? tb->data : NULL, pemb, (int64_t)B * P, C, D);
    }
    if (!c->use_rope) { fprintf(stderr, "train: non-RoPE wpe path unsupported\n"); exit(1); }
    float *xe = arena_alloc((size_t)B * T * C);
    {
        Tensor *tw = TW(ws, "gen.transformer.wte.weight");
        Tensor *tb = weights_get(ws, "gen.transformer.wte.bias");
        t_linear(tp, xt, tw->data, tb ? tb->data : NULL, xe, (int64_t)B * T, C, D);
        int *lidx1 = train_lvl_idx(c, lvl_first, NULL);
        int *lidx = (int *)malloc((size_t)B * T * sizeof(int));
        for (int bb = 0; bb < B; bb++) memcpy(lidx + bb * T, lidx1, T * sizeof(int));
        free(lidx1);
        Tensor *twle = TW(ws, "gen.transformer.wle.weight");
        float *le = arena_alloc((size_t)B * T * C);
        t_embed(tp, twle->data, lidx, ncp_ntgt(m), le, (int64_t)B * T, C);
        free(lidx);
        float *xe2 = arena_alloc((size_t)B * T * C);
        t_add(tp, xe, le, xe2, (size_t)B * T * C);
        xe = xe2;
    }
    /* interleave-concat prefix + target per batch (recorded) */
    float *hfull = arena_alloc((size_t)B * S * C);
    t_join(tp, pemb, xe, hfull, B, P, T, C);

    /* mask (host-built) with per-batch pad bias */
    float *hm0 = train_build_mask(c, P, T, lvl_first);
    float *mask = arena_alloc((size_t)S * S);
    mem_h2d(mask, hm0, (size_t)S * S * sizeof(float));

    int any_pad = 0;
    if (c->prefix_pad_token_id >= 0 && prefix_ids)
        for (int bb = 0; bb < B; bb++) for (int s = 0; s < P; s++)
            if (prefix_ids[bb * P + s] == c->prefix_pad_token_id) any_pad = 1;
    /* run stack: single mask, or per-batch loop when pads present */
    float *h = hfull;
    if (any_pad) {
        /* accumulate per-batch outputs into fresh buffer */
        float *hout = arena_alloc((size_t)B * S * C);
        for (int bb = 0; bb < B; bb++) {
            float *hm = (float *)malloc((size_t)S * S * sizeof(float));
            memcpy(hm, hm0, (size_t)S * S * sizeof(float));
            for (int i = 0; i < S; i++) for (int j = 0; j < P; j++)
                if (prefix_ids[bb * P + j] == c->prefix_pad_token_id)
                    hm[(size_t)i * S + j] = -INFINITY;
            float *bm = arena_alloc((size_t)S * S);
            mem_h2d(bm, hm, (size_t)S * S * sizeof(float));
            free(hm);
            float *hb = arena_alloc((size_t)S * C);
            mem_d2d(hb, h + (size_t)bb * S * C, (size_t)S * C * sizeof(float));
            for (int li = 0; li < c->n_layer; li++)
                hb = train_block(tp, ws, c, li, hb, 1, S, bm, ncp_rope_cos(m), ncp_rope_sin(m));
            Tensor *tw = TW(ws, "gen.transformer.ln_f.weight");
            Tensor *tb = weights_get(ws, "gen.transformer.ln_f.bias");
            float *ln = arena_alloc((size_t)S * C);
            t_layernorm(tp, hb, tw->data, tb ? tb->data : NULL, ln, (int64_t)S, C);
            mem_d2d(hout + (size_t)bb * S * C, ln, (size_t)S * C * sizeof(float));
        }
        h = hout;
        free(hm0);
    } else {
        for (int li = 0; li < c->n_layer; li++)
            h = train_block(tp, ws, c, li, h, B, S, mask, ncp_rope_cos(m), ncp_rope_sin(m));
        Tensor *tw = TW(ws, "gen.transformer.ln_f.weight");
        Tensor *tb = weights_get(ws, "gen.transformer.ln_f.bias");
        float *ln = arena_alloc((size_t)B * S * C);
        t_layernorm(tp, h, tw->data, tb ? tb->data : NULL, ln, (int64_t)B * S, C);
        h = ln;
        free(hm0);
    }
    (void)l0q_host;
    /* heads per level + CE (weight 1/n_tgt each) */
    double loss_sum = 0;
    int off = 0;
    for (int k = 0; k < ncp_ntgt(m); k++) {
        int lk = c->levels[k + lvl_first];
        int V = c->vocab_size[k];
        /* slice target rows [off:off+lk] per batch (recorded gather) */
        float *slice = arena_alloc((size_t)B * lk * C);
        {
            int *rows = (int *)malloc((size_t)B * lk * sizeof(int));
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++)
                rows[bb * lk + t] = bb * S + P + off + t;
            t_gather_rows(tp, h, rows, B * lk, C, slice);
            free(rows);
        }
        float *lg = arena_alloc((size_t)B * lk * V);
        /* head */
        if (c->shared_output_head) {
            Tensor *tw = TW(ws, "gen.lm_heads.head.weight");
            Tensor *tb = NULL;
            char kb[128];
            if (c->shared_head_per_level_bias) {
                snprintf(kb, sizeof(kb), "gen.lm_heads.biases.%d", k);
                tb = TW(ws, kb);
            } else if (c->untied_head_bias) {
                tb = TW(ws, "gen.lm_heads.head.bias");
            }
            t_linear(tp, slice, tw->data, tb ? tb->data : NULL, lg, (int64_t)B * lk, V, C);
        } else if (c->tie_heads_to_codebook) {
            char kw[128];
            snprintf(kw, sizeof(kw), "gen.lm_heads.%d.proj.weight", k);
            Tensor *tpp = TW(ws, kw);
            int Dd = (int)tpp->shape[0];
            snprintf(kw, sizeof(kw), "gen.lm_heads.%d.proj.bias", k);
            Tensor *bp = weights_get(ws, kw);
            float *z = arena_alloc((size_t)B * lk * Dd);
            t_linear(tp, slice, tpp->data, bp ? bp->data : NULL, z, (int64_t)B * lk, Dd, C);
            /* codebook constants (frozen) */
            int lv = k + lvl_first;
            snprintf(kw, sizeof(kw), "tok.vq.embeddings.%d.weight", lv);
            Tensor *te = TW(ws, kw);
            float *hE = (float*)backend_to_host(te->data, (size_t)(V * Dd) * sizeof(float));
            double ds = 0;
            for (int v = 0; v < V; v++) {
                double s = 0;
                for (int d = 0; d < Dd; d++) s += (double)hE[v * Dd + d] * hE[v * Dd + d];
                ds += s;
            }
            ds /= V;
            free(hE);
            snprintf(kw, sizeof(kw), "gen.lm_heads.%d.log_inv_T", k);
            float lit = 0;
            mem_d2h(&lit, TW(ws, kw)->data, sizeof(float));
            snprintf(kw, sizeof(kw), "gen.lm_heads.%d.bias", k);
            Tensor *tbb = weights_get(ws, kw);
            t_tied(tp, z, te->data, tbb ? tbb->data : NULL, lg, (int64_t)B * lk, Dd, V, ds, exp(lit));
            /* NOTE: log_inv_T scalar itself is trainable; its grad needs the
             * head outputs. We route it manually below via t_scale-like path?
             * For now: freeze log_inv_T (documented). */
        } else {
            char kw[128], kb[128];
            snprintf(kw, sizeof(kw), "gen.lm_heads.%d.weight", k);
            snprintf(kb, sizeof(kb), "gen.lm_heads.%d.bias", k);
            Tensor *tw2 = TW(ws, kw);
            Tensor *tb2 = weights_get(ws, kb);
            t_linear(tp, slice, tw2->data, tb2 ? tb2->data : NULL, lg, (int64_t)B * lk, V, C);
        }
        /* targets for this level (per-board level-major codes) */
        int *tgt = (int *)malloc((size_t)B * lk * sizeof(int));
        {
            int bo = 0;
            for (int q = 0; q < k + lvl_first; q++) bo += c->levels[q];
            int tot = vq_total_dummy(c);
            for (int bb = 0; bb < B; bb++) for (int tti = 0; tti < lk; tti++)
                tgt[bb * lk + tti] = codes[bb * tot + bo + tti];
        }
        float *loss1 = arena_alloc(1);
        float *probs = arena_alloc((size_t)B * lk * V);
        t_ce(tp, lg, tgt, NULL, loss1, probs, (int64_t)B * lk, V, 1.0 / ncp_ntgt(m));
        float lh = 0;
        mem_d2h(&lh, loss1, sizeof(float));
        loss_sum += lh;
        if (view_cap_on && k < VIEW_MAXLV) {
            /* batch-averaged exact-code agreement for this level */
            view_lvl_loss[k] = lh;
            float *h = (float *)malloc((size_t)B * lk * V * sizeof(float));
            mem_d2h(h, lg, (size_t)B * lk * V * sizeof(float));
            long hit = 0;
            for (int bb = 0; bb < B; bb++) for (int tti = 0; tti < lk; tti++) {
                int best = 0;
                float bv = h[((size_t)bb * lk + tti) * V];
                for (int v = 1; v < V; v++) {
                    float x = h[((size_t)bb * lk + tti) * V + v];
                    if (x > bv) { bv = x; best = v; }
                }
                if (best == tgt[bb * lk + tti]) hit++;
            }
            free(h);
            view_lvl_agree[k] = (double)hit / ((double)B * lk);
        }

        free(tgt);
        off += lk;
    }
    if (out_ppl) *out_ppl = exp(loss_sum / ncp_ntgt(m));
    return loss_sum / ncp_ntgt(m); /* mean over levels (grads weighted likewise) */
}

/* total codes helper (levels sum) without VQState */
static int vq_total_dummy(NCMConfig *c) {
    int s = 0;
    for (int i = 0; i < c->n_qlevels; i++) s += c->levels[i];
    return s;
}

/* ---------------- NCP training step ---------------- */

typedef struct {
    AdamW *opt;
    PMap pm;
    NCPState *m;
    WeightStore *ws;
    NCMConfig *c;
    double lr, warmup, total;
    double sched_p[2];
} NCPTrainer;

static int is_frozen_ncp(const char *name) {
    return (strncmp(name, "tok.", 4) == 0) || (strncmp(name, "inference.", 10) == 0);
}

static NCPTrainer *ncptrainer_new(NCPState *m, WeightStore *ws, NCMConfig *c,
                                  double lr, double wd, double warmup, double total) {
    NCPTrainer *t = (NCPTrainer *)calloc(1, sizeof(NCPTrainer));
    t->m = m; t->ws = ws; t->c = c;
    t->lr = lr; t->warmup = warmup; t->total = total;
    t->sched_p[0] = warmup; t->sched_p[1] = total;
    /* PMap over trainable params */
    int cap = ws->count, n = 0;
    t->pm.params = (float **)malloc(cap * sizeof(float *));
    t->pm.grads = (float **)malloc(cap * sizeof(float *));
    t->pm.sizes = (int64_t *)malloc(cap * sizeof(int64_t));
    for (int i = 0; i < ws->count; i++) {
        if (is_frozen_ncp(ws->items[i].name)) continue;
        /* skip non-parameter buffers just in case (all skipped on export) */
        t->pm.params[n] = ws->items[i].t->data;
        t->pm.sizes[n] = ws->items[i].t->n;
        t->pm.grads[n] = (float *)calloc(ws->items[i].t->n, sizeof(float));
        n++;
    }
    t->pm.n = n;
    t->opt = adamw_new(ws, NULL);
    /* adamw_new builds its own list; align it to PMap by rebuilding: free and rebuild */
    adamw_free(t->opt);
    t->opt = (AdamW *)calloc(1, sizeof(AdamW));
    t->opt->b1 = 0.9; t->opt->b2 = 0.95; t->opt->eps = 1e-8; t->opt->wd = wd;
    t->opt->n = n;
    t->opt->params = (float **)malloc(n * sizeof(float *));
    t->opt->m = (float **)malloc(n * sizeof(float *));
    t->opt->v = (float **)malloc(n * sizeof(float *));
    t->opt->g = (float **)malloc(n * sizeof(float *));
    t->opt->sizes = (int64_t *)malloc(n * sizeof(int64_t));
    t->opt->lr = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        t->opt->params[i] = t->pm.params[i];
        t->opt->sizes[i] = t->pm.sizes[i];
        t->opt->m[i] = (float *)calloc(t->pm.sizes[i], sizeof(float));
        t->opt->v[i] = (float *)calloc(t->pm.sizes[i], sizeof(float));
        t->opt->g[i] = t->pm.grads[i]; /* share grad buffers with PMap */
        t->opt->lr[i] = 1.0;
    }
    return t;
}

/* forward only (records tape). prefix:[B][P] host ints, codes:[B][total]
 * host ints (per-board level-major). Returns mean CE loss. */
static double train_ncp_forward(NCPTrainer *t, Tape *tp, const int *prefix,
                                const int *codes, int B) {
    NCMConfig *c = t->c;
    NCPState *m = t->m;
    WeightStore *ws = t->ws;
    int P = ncp_P(m), T = ncp_T(m), S = P + T, D = c->vq_embed_dim;
    int lvl_first = ncp_lvlfirst(m);
    tape_clear(tp);
    arena_reset();
    /* build x = [zpre | l0q? | xt] on backend (frozen parts, unrecorded) */
    float *zpre = arena_alloc((size_t)B * P * D);
    {
        int *bids = (int *)arena_alloc((size_t)B * P);
        mem_h2d(bids, prefix, (size_t)B * P * sizeof(int));
        embedding_fwd(TW(t->ws, "tok.encoder.wte.weight")->data, bids, zpre, (int64_t)B * P, D);
    }
    /* xt from frozen VQ */
    VQState *vq = ncp_vq(m);
    int T2 = vq_total_dummy(c) - (lvl_first ? 0 : c->levels[0]);
    float *xt = arena_alloc((size_t)B * T2 * D);
    vq_idx_to_ncm_input(vq, codes, B, xt);
    int L0 = (lvl_first == 0) ? c->levels[0] : 0;
    float *l0q = NULL;
    if (L0) {
        /* recorded broadcast of learned queries (grads flow via gather) */
        Tensor *tq = TW(ws, "gen.l0_queries");
        int *rows = (int *)malloc((size_t)B * L0 * sizeof(int));
        for (int b = 0; b < B; b++) for (int t = 0; t < L0; t++) rows[b * L0 + t] = t;
        l0q = arena_alloc((size_t)B * L0 * D);
        t_gather_rows(tp, tq->data, rows, B * L0, D, l0q);
        free(rows);
    }
    float *xfull = arena_alloc((size_t)B * S * D);
    {
        float *a1 = (float *)malloc((size_t)B * P * D * sizeof(float));
        float *a2 = (float *)malloc((size_t)B * T2 * D * sizeof(float));
        mem_d2h(a1, zpre, (size_t)B * P * D * sizeof(float));
        mem_d2h(a2, xt, (size_t)B * T2 * D * sizeof(float));
        float *af = (float *)malloc((size_t)B * S * D * sizeof(float));
        for (int b = 0; b < B; b++) {
            memcpy(af + (size_t)b * S * D, a1 + (size_t)b * P * D, (size_t)P * D * sizeof(float));
            if (L0) {
                float *a0 = (float *)malloc((size_t)B * L0 * D * sizeof(float));
                mem_d2h(a0, l0q, (size_t)B * L0 * D * sizeof(float));
                memcpy(af + ((size_t)b * S + P) * D, a0 + (size_t)b * L0 * D, (size_t)L0 * D * sizeof(float));
                free(a0);
            }
            memcpy(af + ((size_t)b * S + P + L0) * D, a2 + (size_t)b * T2 * D, (size_t)T2 * D * sizeof(float));
        }
        free(a1); free(a2);
        mem_h2d(xfull, af, (size_t)B * S * D * sizeof(float));
        free(af);
    }
    double ppl = 0;
    double loss = train_ncp_loss(tp, m, ws, c, xfull, prefix, codes, l0q, B, &ppl);
    (void)ppl;
    (void)codes;
    return loss;
}

/* one training step = forward + backward + clip + AdamW step */
static double train_ncp_step(NCPTrainer *t, Tape *tp, const int *prefix,
                             const int *codes, int B) {
    double loss = train_ncp_forward(t, tp, prefix, codes, B);
    tape_backward(tp, &t->pm, 0);
    /* manual grad for l0_queries expansion (sum over batch of input grads).
     * d_xfull lives in activation map; fetch it. */
    /* clip + step */
    {
        double g2 = 0;
        for (int i = 0; i < t->pm.n; i++) {
            int64_t n = t->pm.sizes[i];
            for (int64_t k = 0; k < n; k++) g2 += (double)t->pm.grads[i][k] * t->pm.grads[i][k];
        }
        double gnorm = sqrt(g2);
        double clip = 1.0;
        if (gnorm > clip) {
            double s = clip / gnorm;
            for (int i = 0; i < t->pm.n; i++) {
                int64_t n = t->pm.sizes[i];
                for (int64_t k = 0; k < n; k++) t->pm.grads[i][k] = (float)(t->pm.grads[i][k] * s);
            }
        }
        fprintf(stderr, "  [train] loss=%.4f gnorm=%.3f\n", loss, gnorm);
    }
    adamw_step(t->opt, t->lr, t->warmup, t->total);
    adamw_zero_grad(t->opt);
    return loss;
}


/* ---------------- checkpoint save/load (.ncm) ---------------- */

int weights_save(WeightStore *ws, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return -1; }
    fwrite("NCM1", 1, 4, f);
    uint32_t count = ws->count;
    fwrite(&count, 4, 1, f);
    for (int i = 0; i < ws->count; i++) {
        Tensor *t = ws->items[i].t;
        unsigned char nl = (unsigned char)strlen(ws->items[i].name);
        fwrite(&nl, 1, 1, f);
        fwrite(ws->items[i].name, 1, nl, f);
        unsigned char ndim = (unsigned char)t->ndim;
        fwrite(&ndim, 1, 1, f);
        for (int d = 0; d < t->ndim; d++) {
            uint32_t dd = (uint32_t)t->shape[d];
            fwrite(&dd, 4, 1, f);
        }
        uint64_t nbytes = (uint64_t)t->n * 4;
        fwrite(&nbytes, 8, 1, f);
        float *h = (float *)malloc(nbytes);
        mem_d2h(h, t->data, nbytes);
        fwrite(h, 1, nbytes, f);
        free(h);
    }
    fclose(f);
    return 0;
}

/* ---------------- train entry ---------------- */

static unsigned long long train_rng = 0x123456789ABCDEFULL;
static double train_randn(void) {
    /* Box-Muller */
    double u1 = ((train_rng = train_rng * 6364136223846793005ULL + 1442695040888963407ULL) >> 11) * (1.0 / 9007199254740992.0);
    double u2 = ((train_rng = train_rng * 6364136223846793005ULL + 1442695040888963407ULL) >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2 * log(u1)) * cos(2 * 3.141592653589793 * u2);
}

int train_run_ex(const NCMConfig *c_, const char *weights_path,
                 const char *data_path, int max_steps, double lr,
                 const TrainOpts *o_) {
    TrainOpts o;
    memset(&o, 0, sizeof(o));
    if (o_) o = *o_;
    int ckpt_every = o.ckpt_every > 0 ? o.ckpt_every : 10;
    const char *tokid = o.tokenizer_id ? o.tokenizer_id : "";
    NCMConfig c = *c_;
    WeightStore ws = {NULL, 0};

    /* Resume decision first: exact (NCK1) vs fresh start from legacy
     * weights (NCM1) vs fresh start from --weights. */
    int want_exact = 0;
    if (o.resume_path) {
        int probe = ckpt_probe(o.resume_path);
        if (probe == 1) want_exact = 1;
        else if (probe == 0) {
            fprintf(stderr,
                "[train] LEGACY weight-only checkpoint '%s': optimizer state\n"
                "[train] RESET (fresh start, NOT an exact resume). Moments zeroed,\n"
                "[train] step counters restart at 0, batch RNG restarts at seed.\n",
                o.resume_path);
            weights_path = o.resume_path;
        } else {
            fprintf(stderr, "[train] cannot read resume file '%s'\n", o.resume_path);
            return 1;
        }
    }

    if (weights_load(&ws, weights_path)) return 1;
    arena_init_default();
    parena_init(1ULL << 28);
    VQVAEState *vqvae = vqvae_load(&c, &ws);
    (void)vqvae;
    NCPState *m = ncp_load(&c, &ws, vqvae);
    Tape *tp = tape_new();
    if (!(lr > 0)) lr = 3e-4;
    long target = max_steps > 0 ? max_steps : 1000;
    double sched_total = o.sched_total > 0 ? o.sched_total : (double)target;
    NCPTrainer *t = ncptrainer_new(m, &ws, &c, lr, 0.01, 100, sched_total);
    TrainData *d = train_data_load(data_path, 0);
    if (!d) return 1;
    int P = ncp_P(m);
    int total = vq_total_dummy(&c);
    if (d->row_len != P + total) {
        fprintf(stderr, "row_len %d != P+total %d\n", d->row_len, P + total);
        return 1;
    }
    uint64_t data_hash = ckpt_file_hash(data_path, NULL);
    uint64_t conf_hash = o.conf_path ? ckpt_file_hash(o.conf_path, NULL) : 0;

    long completed = 0;                 /* completed optimizer steps */
    unsigned long long rng = 0x12345;   /* batch sampler state */
    double noise_sd = 0;

    if (want_exact) {
        CkptExpect ex;
        ex.batch_B = 4; ex.n_rows = d->n_rows; ex.row_len = d->row_len;
        ex.data_hash = data_hash; ex.conf_hash = conf_hash;
        ex.tokenizer_id = tokid;
        ex.base_lr = t->lr; ex.warmup = t->warmup; ex.total = t->total;
        ex.b1 = t->opt->b1; ex.b2 = t->opt->b2;
        ex.eps = t->opt->eps; ex.wd = t->opt->wd;
        CkptRunState lrs;
        char err[512] = {0};
        struct timespec lt0, lt1;
        clock_gettime(CLOCK_MONOTONIC, &lt0);
        int lrc = ckpt_load_training(o.resume_path, &ws, t->opt, &ex, &lrs, err, sizeof(err));
        clock_gettime(CLOCK_MONOTONIC, &lt1);
        double ldt = (lt1.tv_sec - lt0.tv_sec) + (lt1.tv_nsec - lt0.tv_nsec) * 1e-9;
        if (lrc) {
            fprintf(stderr, "[train] resume REFUSED: %s\n"
                    "[train] active training state was not modified.\n", err);
            return 1;
        }
        fprintf(stderr, "[train] checkpoint loaded in %.2fs\n", ldt);
        completed = lrs.global_step;
        rng = lrs.batch_rng;
        noise_sd = lrs.noise_sd; /* inherited; CNCM_NOISE ignored on resume */
        const char *nz = getenv("CNCM_NOISE");
        if (nz && atof(nz) != noise_sd)
            fprintf(stderr, "[train] note: CNCM_NOISE=%s ignored on exact resume "
                    "(checkpoint noise_sd=%.3g)\n", nz, noise_sd);
        fprintf(stderr,
                "[train] exact resume from '%s': global_step=%ld opt_step=%ld "
                "batch_rng=0x%016llx\n",
                o.resume_path, completed, t->opt->step, rng);
        if (completed >= target) {
            printf("[train] already complete (%ld >= target %ld), nothing to do\n",
                   completed, target);
            return 0;
        }
    } else {
        const char *nz = getenv("CNCM_NOISE");
        if (nz && atof(nz) > 0) {
            noise_sd = atof(nz);
            fprintf(stderr, "[train] adding N(0,%.3f) noise to trainable params\n", noise_sd);
            for (int i = 0; i < ws.count; i++) {
                if (is_frozen_ncp(ws.items[i].name)) continue;
                float *h = (float *)malloc(ws.items[i].t->n * sizeof(float));
                mem_d2h(h, ws.items[i].t->data, ws.items[i].t->n * sizeof(float));
                for (int64_t k = 0; k < ws.items[i].t->n; k++) h[k] += (float)(train_randn() * noise_sd);
                mem_h2d(ws.items[i].t->data, h, ws.items[i].t->n * sizeof(float));
                free(h);
            }
        }
    }

    printf("[train] %d rows, P=%d total=%d target=%ld schedule_total=%.0f%s\n",
           d->n_rows, P, total, target, t->total, want_exact ? " (resumed)" : "");
    int B = 4;
    int *pbuf = (int *)malloc(B * P * sizeof(int));
    int *cbuf = (int *)malloc(B * total * sizeof(int));
    char cfg_json[2048];
    snprintf(cfg_json, sizeof(cfg_json),
             "{\"lr\":%.17g,\"warmup\":%.17g,\"total\":%.17g,\"B\":%d,"
             "\"noise_sd\":%.17g,\"tokenizer_id\":\"%s\",\"data\":\"%s\",\"conf\":\"%s\"}",
             t->lr, t->warmup, t->total, B, noise_sd, tokid,
             data_path, o.conf_path ? o.conf_path : "");
    int rc = 0;
    /* live viewer (explicit --view only; read-only snapshots, so numerics
     * and determinism are identical with or without it). 1 = SDL window,
     * 2 = console (synchronous ANSI redraw, works headless/piped). */
    int view_on = 0, view_mode = 0;
    struct timespec vt0;
    if (o.view == 2) {
        char vtitle[128];
        snprintf(vtitle, sizeof(vtitle), "CNCM train %s", c.task);
        con_viewer_open(vtitle);
        view_on = 1;
        view_mode = 2;
        clock_gettime(CLOCK_MONOTONIC, &vt0);
    } else if (o.view) {
        char vtitle[128];
        snprintf(vtitle, sizeof(vtitle), "CNCM train %s", c.task);
        if (viewer_open(vtitle) == 0) { view_on = 1; view_mode = 1; }
        else fprintf(stderr, "[train] viewer: no display/SDL2, continuing headless\n");
        clock_gettime(CLOCK_MONOTONIC, &vt0);
    }
    view_cap_on = view_on;
    int lvl_first = ncp_lvlfirst(m);
    int ntgt = ncp_ntgt(m);
    while (completed < target) {
        /* random batch (with replacement) */
        for (int b = 0; b < B; b++) {
            rng = rng_next(&rng);
            int r = (int)(rng % (unsigned long long)d->n_rows);
            memcpy(pbuf + b * P, d->rows + r * d->row_len, P * sizeof(int));
            memcpy(cbuf + b * total, d->rows + r * d->row_len + P, total * sizeof(int));
        }
        double loss = train_ncp_step(t, tp, pbuf, cbuf, B);
        completed++;
        if (view_on) {
            ViewSnap vs;
            memset(&vs, 0, sizeof(vs));
            vs.mode = 0;
            vs.step = (int)completed; vs.total = (int)target;
            vs.loss = loss;
            vs.lr = t->lr * sched_cosine_warmup(completed, t->warmup, t->total);
            struct timespec vtn;
            clock_gettime(CLOCK_MONOTONIC, &vtn);
            vs.elapsed = (vtn.tv_sec - vt0.tv_sec) + (vtn.tv_nsec - vt0.tv_nsec) * 1e-9;
            /* clues from board 0 of the batch (sudoku prefix framing) */
            {
                SudokuSize tsz;
                int okf = 0;
                if (sudoku_size_from_prefix(P, &tsz) == 0 &&
                    pbuf[0] == tsz.wrap_id && pbuf[P - 1] == tsz.wrap_id) {
                    okf = 1;
                    for (int r = 0; r < tsz.n && okf; r++) {
                        if (r < tsz.n - 1 && pbuf[1 + r * (tsz.n + 1) + tsz.n] != tsz.sep_id) { okf = 0; break; }
                        for (int cc = 0; cc < tsz.n; cc++) {
                            int id = pbuf[1 + r * (tsz.n + 1) + cc];
                            if (id < 0 || id > tsz.n) { okf = 0; break; }
                            vs.clues[r * tsz.n + cc] = id ? sudoku_val_char(id) : '.';
                        }
                    }
                    if (okf) vs.clues[tsz.ncells] = 0;
                    else vs.clues[0] = 0;
                } else vs.clues[0] = 0;
            }
            /* per-level loss/agreement + code-weighted accuracy */
            vs.nlv = ntgt < VIEW_MAXLV ? ntgt : VIEW_MAXLV;
            double hsum = 0, wsum = 0;
            for (int k = 0; k < vs.nlv; k++) {
                int lk = c.levels[k + lvl_first];
                vs.lvl_loss[k] = view_lvl_loss[k];
                vs.lvl_acc[k] = view_lvl_agree[k];
                hsum += view_lvl_agree[k] * lk;
                wsum += lk;
            }
            vs.acc = wsum > 0 ? hsum / wsum : 0;
            snprintf(vs.model, sizeof(vs.model), "%s", c.task);
#ifdef NCM_CUDA
            snprintf(vs.backend, sizeof(vs.backend), "cuda");
#else
            snprintf(vs.backend, sizeof(vs.backend), "cpu");
#endif
        snprintf(vs.status, sizeof(vs.status), "training B=%d", B);
        vs.ckpt = (completed % ckpt_every == 0 || completed == target) ? 1 : 0;
        if (view_mode == 2) con_viewer_push(&vs);
        else viewer_push(&vs);
        if (view_mode == 1 && viewer_stop_requested()) {
            printf("[train] stopped by viewer (step %ld)\n", completed);
            fflush(stdout);
            break;
        }
    }
        const char *save_path = NULL;
        char rot[256];
        if (completed % ckpt_every == 0 || completed == target) {
            if (o.ckpt_out) save_path = (char *)o.ckpt_out;
            else if (completed == target) save_path = "ckpt-final.nckpt";
            else { snprintf(rot, sizeof(rot), "ckpt-step%ld.nckpt", completed); save_path = rot; }
            CkptRunState rs;
            memset(&rs, 0, sizeof(rs));
            rs.global_step = completed; rs.batch_rng = rng;
            rs.base_lr = t->lr; rs.warmup = t->warmup; rs.total = t->total;
            rs.batch_B = B; rs.accum_pending = 0;
            rs.n_rows = d->n_rows; rs.row_len = d->row_len;
            rs.data_hash = data_hash; rs.data_path = data_path;
            rs.conf_hash = conf_hash; rs.tokenizer_id = tokid;
            rs.noise_sd = noise_sd; rs.train_cfg_json = cfg_json;
            char err[512] = {0};
            struct timespec st0, st1;
            clock_gettime(CLOCK_MONOTONIC, &st0);
            int src_ = ckpt_save_atomic(&ws, t->opt, &rs, save_path, err, sizeof(err));
            clock_gettime(CLOCK_MONOTONIC, &st1);
            double sdt = (st1.tv_sec - st0.tv_sec) + (st1.tv_nsec - st0.tv_nsec) * 1e-9;
            if (src_) {
                fprintf(stderr, "[train] CHECKPOINT SAVE FAILED: %s\n", err);
                rc = 1;
                break;
            }
            fprintf(stderr, "[train] checkpoint saved in %.2fs\n", sdt);
        }
        printf("[train] step %ld loss %.4f%s%s%s\n", completed - 1, loss,
               save_path ? " (saved " : "", save_path ? save_path : "",
               save_path ? ")" : "");
        fflush(stdout);
    }
    view_cap_on = 0;
    if (view_mode == 2) con_viewer_close();
    else { viewer_hold(); viewer_close(); }
    free(pbuf); free(cbuf);
    free(d->rows); free(d);
    return rc;
}

int train_run(const NCMConfig *c_, const char *weights_path,
              const char *data_path, int max_steps, double lr) {
    return train_run_ex(c_, weights_path, data_path, max_steps, lr, NULL);
}


/* ---------------- device/host backward parity ----------------
 * One deterministic forward, backward on host, snapshot grads, redo the
 * same forward, backward on device, compare every PMap grad tensor.
 * CUDA-only (the device path does not exist on CPU builds). */
int bwd_parity_run(const NCMConfig *c_, const char *weights_path,
                   const char *data_path, double tol) {
#ifdef NCM_CUDA
    NCMConfig c = *c_;
    WeightStore ws = {NULL, 0};
    if (weights_load(&ws, weights_path)) return 1;
    arena_init_default();
    parena_init(1ULL << 28);
    VQVAEState *vqvae = vqvae_load(&c, &ws);
    (void)vqvae;
    NCPState *m = ncp_load(&c, &ws, vqvae);
    Tape *tp = tape_new();
    NCPTrainer *t = ncptrainer_new(m, &ws, &c, 3e-4, 0.01, 100, 1000);
    TrainData *d = train_data_load(data_path, 0);
    if (!d) return 1;
    int P = ncp_P(m);
    int total = vq_total_dummy(&c);
    if (d->row_len != P + total) {
        fprintf(stderr, "row_len %d != P+total %d\n", d->row_len, P + total);
        return 1;
    }
    int B = 4;
    int *pbuf = (int *)malloc(B * P * sizeof(int));
    int *cbuf = (int *)malloc(B * total * sizeof(int));
    unsigned long long rng = 0x12345;
    for (int b = 0; b < B; b++) {
        rng = rng_next(&rng);
        int r = (int)(rng % (unsigned long long)d->n_rows);
        memcpy(pbuf + b * P, d->rows + r * d->row_len, P * sizeof(int));
        memcpy(cbuf + b * total, d->rows + r * d->row_len + P, total * sizeof(int));
    }
    /* host backward on first forward */
    double loss_h = train_ncp_forward(t, tp, pbuf, cbuf, B);
    tape_backward_force(tp, &t->pm, 0, 0);
    float **snap = (float **)malloc((size_t)t->pm.n * sizeof(float *));
    for (int i = 0; i < t->pm.n; i++) {
        snap[i] = (float *)malloc((size_t)t->pm.sizes[i] * sizeof(float));
        memcpy(snap[i], t->pm.grads[i], (size_t)t->pm.sizes[i] * sizeof(float));
    }
    adamw_zero_grad(t->opt);
    /* identical forward again, device backward */
    double loss_d = train_ncp_forward(t, tp, pbuf, cbuf, B);
    tape_backward_force(tp, &t->pm, 0, 1);
    printf("forward loss host-path=%.6f device-path=%.6f\n", loss_h, loss_d);
    double gmax = 0, worst_rel = 0;
    const char *worst_nm = "";
    for (int i = 0; i < t->pm.n; i++) {
        const char *nm = "?";
        for (int j = 0; j < ws.count; j++)
            if (ws.items[j].t->data == t->pm.params[i]) { nm = ws.items[j].name; break; }
        double mx = 0, hm = 0;
        int64_t n = t->pm.sizes[i];
        for (int64_t k = 0; k < n; k++) {
            double a = fabs((double)t->pm.grads[i][k] - snap[i][k]);
            if (a > mx) mx = a;
            double h = fabs((double)snap[i][k]);
            if (h > hm) hm = h;
        }
        double rel = mx / (hm + 1e-30);
        printf("  %-52s n=%-10lld maxabs=%.3g maxrel=%.3g\n",
               nm, (long long)n, mx, rel);
        if (mx > gmax) { gmax = mx; worst_nm = nm; worst_rel = rel; }
        free(snap[i]);
    }
    free(snap); free(pbuf); free(cbuf);
    printf("bwd-parity: global maxabs=%.3g at %s (rel=%.3g), tol=%.3g -> %s\n",
           gmax, worst_nm, worst_rel, tol, gmax <= tol ? "PASS" : "FAIL");
    return gmax <= tol ? 0 : 1;
#else
    (void)c_; (void)weights_path; (void)data_path; (void)tol;
    fprintf(stderr, "bwd-parity needs a CUDA build (no device backward on CPU)\n");
    return 2;
#endif
}

/* ---------------- synthetic per-op backward check ----------------
 * For every non-conv tape op: record it on random small tensors, seed
 * with an MSE terminal, run host backward, snapshot, redo device
 * backward, compare every grad. CUDA-only. Covers ops the NCP workload
 * never emits (TIED/SILU/MUL/...); conv-family is fallback-only. */
#ifdef NCM_CUDA
/* (helpers are CUDA-only; bwdops_run itself is defined on both builds) */
static unsigned long long bwo_rng = 0x243F6A8885A308D3ULL;
static float bwo_frand(float lo, float hi) {
    bwo_rng ^= bwo_rng << 13; bwo_rng ^= bwo_rng >> 7; bwo_rng ^= bwo_rng << 17;
    double u = (bwo_rng >> 11) * (1.0 / 9007199254740992.0);
    return (float)(lo + u * (hi - lo));
}
static Tensor *bwo_tensor(int ndim, const int64_t *shape, float lo, float hi) {
    Tensor *t = tensor_alloc(ndim, shape);
    float *h = (float *)malloc((size_t)t->n * sizeof(float));
    for (int64_t k = 0; k < t->n; k++) h[k] = bwo_frand(lo, hi);
    mem_h2d(t->data, h, (size_t)t->n * sizeof(float));
    free(h);
    return t;
}
static int bwo_nfail = 0;
static double bwo_gmax = 0;
static double bwo_tol = 1e-4;
static const char *bwo_gname = "";
/* compare PMap grads host-vs-device; snapshots host grads then zeroes */
static void bwo_compare(PMap *pm, float **snap, const char *op) {
    for (int i = 0; i < pm->n; i++) {
        double mx = 0, hm = 0;
        for (int64_t k = 0; k < pm->sizes[i]; k++) {
            double a = fabs((double)pm->grads[i][k] - snap[i][k]);
            if (a > mx) mx = a;
            double h = fabs((double)snap[i][k]);
            if (h > hm) hm = h;
        }
        double rel = mx / (hm + 1e-30);
        printf("  bwdop %-8s p%d n=%-7lld maxabs=%.3g maxrel=%.3g\n",
               op, i, (long long)pm->sizes[i], mx, rel);
        if (mx > bwo_tol) { bwo_nfail++; printf("  FAIL %s p%d\n", op, i); }
        if (mx > bwo_gmax) { bwo_gmax = mx; bwo_gname = op; }
    }
}
/* run one op: record + MSE seed, host bwd, snapshot, device bwd, compare.
 * build() emits records into tp and fills pm with the grad-carrying bufs. */
typedef void (*bwo_build_fn)(Tape *tp, PMap *pm);
static void bwo_run_one(bwo_build_fn build, const char *op) {
    PMap pm;
    memset(&pm, 0, sizeof(pm));
    pm.params = (float **)malloc(32 * sizeof(float *));
    pm.grads = (float **)malloc(32 * sizeof(float *));
    pm.sizes = (int64_t *)malloc(32 * sizeof(int64_t));
    pm.n = 0;
    Tape *tp = tape_new();
    arena_reset();
    build(tp, &pm);
    tape_backward_force(tp, &pm, 0, 0);
    float **snap = (float **)malloc((size_t)pm.n * sizeof(float *));
    for (int i = 0; i < pm.n; i++) {
        snap[i] = (float *)malloc((size_t)pm.sizes[i] * sizeof(float));
        memcpy(snap[i], pm.grads[i], (size_t)pm.sizes[i] * sizeof(float));
        memset(pm.grads[i], 0, (size_t)pm.sizes[i] * sizeof(float));
    }
    tape_backward_force(tp, &pm, 0, 1);
    bwo_compare(&pm, snap, op);
    for (int i = 0; i < pm.n; i++) { free(snap[i]); free(pm.grads[i]); }
    free(snap); free(pm.params); free(pm.grads); free(pm.sizes);
    tape_free(tp);
}
static void bwo_reg(PMap *pm, Tensor *t) {
    int i = pm->n++;
    pm->params[i] = t->data;
    pm->sizes[i] = t->n;
    pm->grads[i] = (float *)calloc((size_t)t->n, sizeof(float));
}
/* each builder records one op + MSE seed; registers grad buffers.
 * Leaks the tiny loss-side tensors per op (test-only; freed at exit). */
static void bwo_mse_seed(Tape *tp, float *y, int64_t n) {
    int64_t one = 1;
    Tensor *z = tensor_alloc(1, &n);
    float *zz = (float *)calloc((size_t)n, sizeof(float));
    mem_h2d(z->data, zz, (size_t)n * sizeof(float));
    free(zz);
    Tensor *l = tensor_alloc(1, &one);
    t_mse(tp, y, z->data, l->data, n, 1.0);
}
static void bwo_linear(Tape *tp, PMap *pm) {
    int64_t b = 8, M = 12, K = 16;
    int64_t sx[2] = {b, K}, sw[2] = {M, K}, sv[2] = {b, M}, s1[1] = {M};
    Tensor *x = bwo_tensor(2, sx, -1, 1), *W = bwo_tensor(2, sw, -1, 1);
    Tensor *bb = bwo_tensor(1, s1, -1, 1), *y = tensor_alloc(2, sv);
    bwo_reg(pm, x); bwo_reg(pm, W); bwo_reg(pm, bb);
    t_linear(tp, x->data, W->data, bb->data, y->data, b, (int)M, (int)K);
    bwo_mse_seed(tp, y->data, b * M);
}
/* Qwen-scale linear backward (down_proj dims: M=896 K=4864 batch=16).
 * Guards cuBLAS-vs-serial agreement at the shapes the Qwen loop uses. */
static void bwo_linear_big(Tape *tp, PMap *pm) {
    int64_t b = 16, M = 896, K = 4864;
    int64_t sx[2] = {b, K}, sw[2] = {M, K}, sv[2] = {b, M}, s1[1] = {M};
    Tensor *x = bwo_tensor(2, sx, -3, 3), *W = bwo_tensor(2, sw, -0.1, 0.1);
    Tensor *bb = bwo_tensor(1, s1, -1, 1), *y = tensor_alloc(2, sv);
    bwo_reg(pm, x); bwo_reg(pm, W); bwo_reg(pm, bb);
    t_linear(tp, x->data, W->data, bb->data, y->data, b, (int)M, (int)K);
    bwo_mse_seed(tp, y->data, b * M);
}
static void bwo_ln(Tape *tp, PMap *pm) {
    int64_t rows = 32, C = 16, s[2] = {rows, C}, s1[1] = {C};
    Tensor *x = bwo_tensor(2, s, -2, 2), *w = bwo_tensor(1, s1, 0.5, 1.5);
    Tensor *b = bwo_tensor(1, s1, -1, 1), *y = tensor_alloc(2, s);
    bwo_reg(pm, x); bwo_reg(pm, w); bwo_reg(pm, b);
    t_layernorm(tp, x->data, w->data, b->data, y->data, rows, (int)C);
    bwo_mse_seed(tp, y->data, rows * C);
}
static void bwo_rmsn(Tape *tp, PMap *pm) {
    int64_t rows = 32, C = 16, s[2] = {rows, C}, s1[1] = {C};
    Tensor *x = bwo_tensor(2, s, -2, 2), *w = bwo_tensor(1, s1, 0.5, 1.5);
    Tensor *y = tensor_alloc(2, s);
    bwo_reg(pm, x); bwo_reg(pm, w);
    t_rmsnorm(tp, x->data, w->data, y->data, rows, (int)C);
    bwo_mse_seed(tp, y->data, rows * C);
}
static void bwo_gn(Tape *tp, PMap *pm) {
    int N = 2, G = 2, C = 8, TT = 8;
    int64_t flat[1] = {(int64_t)N * C * TT};
    Tensor *x = bwo_tensor(1, flat, -2, 2), *y = tensor_alloc(1, flat);
    bwo_reg(pm, x);
    t_groupnorm(tp, x->data, y->data, N, G, C, TT);
    bwo_mse_seed(tp, y->data, (int64_t)N * C * TT);
}
static void bwo_act(Tape *tp, PMap *pm, int which) {
    int64_t n = 256, s[1] = {n};
    Tensor *x = bwo_tensor(1, s, -2, 2), *y = tensor_alloc(1, s);
    bwo_reg(pm, x);
    if (which == 0) t_gelu(tp, x->data, y->data, n);
    else if (which == 1) t_silu(tp, x->data, y->data, n);
    else t_relu2(tp, x->data, y->data, n);
    bwo_mse_seed(tp, y->data, n);
}
static void bwo_gelu(Tape *tp, PMap *pm) { bwo_act(tp, pm, 0); }
static void bwo_silu(Tape *tp, PMap *pm) { bwo_act(tp, pm, 1); }
static void bwo_relu2(Tape *tp, PMap *pm) { bwo_act(tp, pm, 2); }
static void bwo_softmax(Tape *tp, PMap *pm) {
    int64_t rows = 8, C = 16, s[2] = {rows, C};
    Tensor *x = bwo_tensor(2, s, -2, 2), *y = tensor_alloc(2, s);
    bwo_reg(pm, x);
    t_softmax(tp, x->data, y->data, rows, (int)C);
    bwo_mse_seed(tp, y->data, rows * C);
}
static void bwo_rope(Tape *tp, PMap *pm) {
    int B = 2, H = 4, T = 8, hd = 8;
    int64_t n = (int64_t)B * H * T * hd, s[1] = {n}, st[1] = {(int64_t)T * hd};
    Tensor *xq = bwo_tensor(1, s, -1, 1), *xk = bwo_tensor(1, s, -1, 1);
    Tensor *yq = tensor_alloc(1, s), *yk = tensor_alloc(1, s);
    Tensor *co = bwo_tensor(1, st, -1, 1), *si = bwo_tensor(1, st, -1, 1);
    bwo_reg(pm, xq); bwo_reg(pm, xk);
    t_rope(tp, xq->data, xk->data, yq->data, yk->data, co->data, si->data, B, H, H, T, hd);
    bwo_mse_seed(tp, yq->data, n);
    bwo_mse_seed(tp, yk->data, n);
}
static void bwo_sdpa(Tape *tp, PMap *pm) {
    int B = 2, H = 4, Tq = 8, Tkv = 8, hd = 8;
    int64_t sq[1] = {(int64_t)B * H * Tq * hd}, sk[1] = {(int64_t)B * H * Tkv * hd};
    Tensor *q = bwo_tensor(1, sq, -1, 1), *k = bwo_tensor(1, sk, -1, 1);
    Tensor *v = bwo_tensor(1, sk, -1, 1), *o = tensor_alloc(1, sq);
    bwo_reg(pm, q); bwo_reg(pm, k); bwo_reg(pm, v);
    t_sdpa(tp, q->data, k->data, v->data, NULL, o->data, B, H, H, Tq, Tkv, hd);
    bwo_mse_seed(tp, o->data, (int64_t)B * H * Tq * hd);
}
static void bwo_add(Tape *tp, PMap *pm) {
    int64_t n = 256, s[1] = {n};
    Tensor *a = bwo_tensor(1, s, -1, 1), *b = bwo_tensor(1, s, -1, 1);
    Tensor *y = tensor_alloc(1, s);
    bwo_reg(pm, a); bwo_reg(pm, b);
    t_add(tp, a->data, b->data, y->data, n);
    bwo_mse_seed(tp, y->data, n);
}
static void bwo_scale(Tape *tp, PMap *pm) {
    int64_t n = 256, s[1] = {n};
    Tensor *x = bwo_tensor(1, s, -1, 1), *y = tensor_alloc(1, s);
    bwo_reg(pm, x);
    t_scale(tp, x->data, 0.37f, y->data, n);
    bwo_mse_seed(tp, y->data, n);
}
static void bwo_mul(Tape *tp, PMap *pm) {
    int64_t n = 256, s[1] = {n};
    Tensor *a = bwo_tensor(1, s, -1, 1), *b = bwo_tensor(1, s, -1, 1);
    Tensor *y = tensor_alloc(1, s);
    bwo_reg(pm, a); bwo_reg(pm, b);
    t_mul(tp, a->data, b->data, y->data, n);
    bwo_mse_seed(tp, y->data, n);
}
static void bwo_concat(Tape *tp, PMap *pm) {
    int64_t na = 100, nb = 156, sa[1] = {na}, sb[1] = {nb}, sy[1] = {na + nb};
    Tensor *a = bwo_tensor(1, sa, -1, 1), *b = bwo_tensor(1, sb, -1, 1);
    Tensor *y = tensor_alloc(1, sy);
    bwo_reg(pm, a); bwo_reg(pm, b);
    t_concat(tp, a->data, b->data, y->data, na, nb);
    bwo_mse_seed(tp, y->data, na + nb);
}
static void bwo_join(Tape *tp, PMap *pm) {
    int B = 2, P = 6, T = 10, C = 8;
    int64_t sa[1] = {(int64_t)B * P * C}, sb[1] = {(int64_t)B * T * C};
    int64_t sy[1] = {(int64_t)B * (P + T) * C};
    Tensor *a = bwo_tensor(1, sa, -1, 1), *b = bwo_tensor(1, sb, -1, 1);
    Tensor *y = tensor_alloc(1, sy);
    bwo_reg(pm, a); bwo_reg(pm, b);
    t_join(tp, a->data, b->data, y->data, B, P, T, C);
    bwo_mse_seed(tp, y->data, (int64_t)B * (P + T) * C);
}
static void bwo_split(Tape *tp, PMap *pm) {
    /* Fused-qkv split keeps the replication contract: k/v come out at H
     * heads (rep>1 replicates). Un-replicated GQA k/v flow through
     * headsplit+rope(H,Hkv)+sdpa(H,Hkv); see bwo_rope_gqa/bwo_sdpa_gqa. */
    int B = 2, T = 8, C = 64, H = 4, hd = 16, K = 4, rep = 1;
    int64_t row = (int64_t)C + 2 * K * hd;
    int64_t si[1] = {(int64_t)B * T * row};
    int64_t sq[1] = {(int64_t)B * H * T * hd}, sk[1] = {(int64_t)B * K * T * hd};
    Tensor *in = bwo_tensor(1, si, -1, 1);
    Tensor *qq = tensor_alloc(1, sq), *kk = tensor_alloc(1, sk), *vv = tensor_alloc(1, sk);
    bwo_reg(pm, in);
    t_qkvsplit(tp, in->data, qq->data, kk->data, vv->data, B, T, C, H, hd, K, rep);
    bwo_mse_seed(tp, qq->data, (int64_t)B * H * T * hd);
    bwo_mse_seed(tp, kk->data, (int64_t)B * K * T * hd);
    bwo_mse_seed(tp, vv->data, (int64_t)B * K * T * hd);
}
static void bwo_merge(Tape *tp, PMap *pm) {
    int B = 2, T = 8, C = 64, H = 4, hd = 16;
    int64_t sc[1] = {(int64_t)B * H * T * hd}, so[1] = {(int64_t)B * T * C};
    Tensor *ctx = bwo_tensor(1, sc, -1, 1), *out = tensor_alloc(1, so);
    bwo_reg(pm, ctx);
    t_headmerge(tp, ctx->data, out->data, B, T, C, H, hd);
    bwo_mse_seed(tp, out->data, (int64_t)B * T * C);
}
static void bwo_headsplit(Tape *tp, PMap *pm) {
    int B = 2, T = 8, C = 64, H = 4, hd = 16;
    int64_t si[1] = {(int64_t)B * T * C}, so[1] = {(int64_t)B * H * T * hd};
    Tensor *in = bwo_tensor(1, si, -1, 1), *out = tensor_alloc(1, so);
    bwo_reg(pm, in);
    t_headsplit(tp, in->data, out->data, B, T, C, H, hd);
    bwo_mse_seed(tp, out->data, (int64_t)B * H * T * hd);
}
static void bwo_rope_gqa(Tape *tp, PMap *pm) {
    int B = 2, H = 4, K = 2, T = 8, hd = 8;
    int64_t sq[1] = {(int64_t)B * H * T * hd}, sk[1] = {(int64_t)B * K * T * hd};
    int64_t st[1] = {(int64_t)T * hd};
    Tensor *xq = bwo_tensor(1, sq, -1, 1), *xk = bwo_tensor(1, sk, -1, 1);
    Tensor *yq = tensor_alloc(1, sq), *yk = tensor_alloc(1, sk);
    Tensor *co = bwo_tensor(1, st, -1, 1), *si = bwo_tensor(1, st, -1, 1);
    bwo_reg(pm, xq); bwo_reg(pm, xk);
    t_rope(tp, xq->data, xk->data, yq->data, yk->data, co->data, si->data,
           B, H, K, T, hd);
    bwo_mse_seed(tp, yq->data, (int64_t)B * H * T * hd);
    bwo_mse_seed(tp, yk->data, (int64_t)B * K * T * hd);
}
static void bwo_sdpa_gqa(Tape *tp, PMap *pm) {
    int B = 2, H = 4, K = 2, Tq = 8, Tkv = 8, hd = 8;
    int64_t sq[1] = {(int64_t)B * H * Tq * hd}, sk[1] = {(int64_t)B * K * Tkv * hd};
    Tensor *q = bwo_tensor(1, sq, -1, 1), *k = bwo_tensor(1, sk, -1, 1);
    Tensor *v = bwo_tensor(1, sk, -1, 1), *o = tensor_alloc(1, sq);
    bwo_reg(pm, q); bwo_reg(pm, k); bwo_reg(pm, v);
    t_sdpa(tp, q->data, k->data, v->data, NULL, o->data, B, H, K, Tq, Tkv, hd);
    bwo_mse_seed(tp, o->data, (int64_t)B * H * Tq * hd);
}
/* Mini Qwen model (1 layer, synthetic) through the REAL train_qwen_forward
 * + CE: composed GQA path (embed, rmsnorm, q/k/v+bias, headsplit,
 * rope_gqa, sdpa_gqa, merge, swiglu, tied head) compared host-vs-device.
 * Catches composition bugs unit ops miss. B=1 fixed (board-0 path). */
static NamedTensor bwo_qw_items[14];
static WeightStore bwo_qw_ws;
static void bwo_qwenblock(Tape *tp, PMap *pm) {
    int B = 1, T = 4, C = 32, H = 4, K = 2, hd = 8, I = 64, V = 256;
    NCMConfig qc;
    memset(&qc, 0, sizeof(qc));
    qc.n_layer = 1; qc.n_head = H; qc.n_kv_heads = K; qc.n_embd = C;
    qc.context_length = 64; qc.rope_theta = 10000.0;
    struct { const char *nm; int rows, cols; } spec[] = {
        {"model.embed_tokens.weight", V, C},
        {"model.norm.weight", 1, C},
        {"model.layers.0.input_layernorm.weight", 1, C},
        {"model.layers.0.self_attn.q_proj.weight", C, C},
        {"model.layers.0.self_attn.q_proj.bias", 1, C},
        {"model.layers.0.self_attn.k_proj.weight", K * hd, C},
        {"model.layers.0.self_attn.k_proj.bias", 1, K * hd},
        {"model.layers.0.self_attn.v_proj.weight", K * hd, C},
        {"model.layers.0.self_attn.v_proj.bias", 1, K * hd},
        {"model.layers.0.self_attn.o_proj.weight", C, C},
        {"model.layers.0.post_attention_layernorm.weight", 1, C},
        {"model.layers.0.mlp.gate_proj.weight", I, C},
        {"model.layers.0.mlp.up_proj.weight", I, C},
        {"model.layers.0.mlp.down_proj.weight", C, I},
    };
    int NW = (int)(sizeof(spec) / sizeof(spec[0]));
    int hids[4] = {7, 42, 100, 5};
    int htg[4] = {3, 200, 17, 99};
    bwo_qw_ws.items = bwo_qw_items;
    bwo_qw_ws.count = NW;
    for (int i = 0; i < NW; i++) {
        int n = spec[i].rows * spec[i].cols;
        Tensor *t = tensor_alloc(2, (int64_t[]){spec[i].rows, spec[i].cols});
        float *h = (float *)malloc(n * sizeof(float));
        for (int k = 0; k < n; k++) {
            bwo_rng ^= bwo_rng << 13; bwo_rng ^= bwo_rng >> 7; bwo_rng ^= bwo_rng << 17;
            h[k] = (float)((bwo_rng >> 11) * (1.0 / 9007199254740992.0) - 0.5);
        }
        mem_h2d(t->data, h, n * sizeof(float));
        free(h);
        snprintf(bwo_qw_items[i].name, sizeof(bwo_qw_items[i].name), "%s", spec[i].nm);
        bwo_qw_items[i].t = t;
        bwo_reg(pm, t);
    }
    float *logits = train_qwen_forward(tp, &bwo_qw_ws, &qc, hids, B, T);
    float *loss = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
    /* t_ce takes HOST targets (copies internally) */
    t_ce(tp, logits, htg, NULL, loss, probs, (int64_t)B * T, V, 1.0);
}
/* Qwen-scale GQA (H=14 K=2 rep=7 T=16 hd=64): rep-dependent indexing only
 * shows at real reps, so pin the production shape here. */
static void bwo_sdpa_gqa_big(Tape *tp, PMap *pm) {
    int B = 1, H = 14, K = 2, Tq = 16, Tkv = 16, hd = 64;
    int64_t sq[1] = {(int64_t)B * H * Tq * hd}, sk[1] = {(int64_t)B * K * Tkv * hd};
    Tensor *q = bwo_tensor(1, sq, -1, 1), *k = bwo_tensor(1, sk, -1, 1);
    Tensor *v = bwo_tensor(1, sk, -1, 1), *o = tensor_alloc(1, sq);
    bwo_reg(pm, q); bwo_reg(pm, k); bwo_reg(pm, v);
    t_sdpa(tp, q->data, k->data, v->data, NULL, o->data, B, H, K, Tq, Tkv, hd);
    bwo_mse_seed(tp, o->data, (int64_t)B * H * Tq * hd);
}
static void bwo_gather(Tape *tp, PMap *pm) {
    int nrows = 16, C = 16, V = 32;
    int64_t sx[1] = {(int64_t)V * C}, sy[1] = {(int64_t)nrows * C};
    Tensor *x = bwo_tensor(1, sx, -1, 1), *y = tensor_alloc(1, sy);
    int *rows = (int *)malloc((size_t)nrows * sizeof(int));
    for (int i = 0; i < nrows; i++) rows[i] = (i * 7) % V;
    bwo_reg(pm, x);
    t_gather_rows(tp, x->data, rows, nrows, C, y->data);
    free(rows);
    bwo_mse_seed(tp, y->data, (int64_t)nrows * C);
}
static void bwo_embed(Tape *tp, PMap *pm) {
    int n = 16, D = 16, V = 32;
    int64_t sw[1] = {(int64_t)V * D}, sy[1] = {(int64_t)n * D};
    Tensor *W = bwo_tensor(1, sw, -1, 1), *y = tensor_alloc(1, sy);
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = (i * 5 + 1) % V;
    bwo_reg(pm, W);
    t_embed(tp, W->data, idx, V, y->data, n, D);
    free(idx);
    bwo_mse_seed(tp, y->data, (int64_t)n * D);
}
static void bwo_ce(Tape *tp, PMap *pm) {
    int64_t rows = 8, V = 32, s[2] = {rows, V};
    Tensor *lg = bwo_tensor(2, s, -3, 3);
    int64_t one = 1;
    Tensor *loss = tensor_alloc(1, &one), *probs = tensor_alloc(2, s);
    int *tgt = (int *)malloc((size_t)rows * sizeof(int));
    for (int i = 0; i < rows; i++) tgt[i] = (i * 3 + 1) % (int)V;
    bwo_reg(pm, lg);
    t_ce(tp, lg->data, tgt, NULL, loss->data, probs->data, rows, (int)V, 1.0);
    free(tgt);
}
static void bwo_tied(Tape *tp, PMap *pm) {
    int64_t rows = 8, Dd = 16, V = 32;
    int64_t sz[2] = {rows, Dd}, se[2] = {V, Dd}, so[2] = {rows, V};
    Tensor *z = bwo_tensor(2, sz, -1, 1), *E = bwo_tensor(2, se, -1, 1);
    Tensor *o = tensor_alloc(2, so);
    bwo_reg(pm, z); bwo_reg(pm, E);
    t_tied(tp, z->data, E->data, NULL, o->data, rows, (int)Dd, (int)V, 1.0, 1.0);
    bwo_mse_seed(tp, o->data, rows * V);
}
static void bwo_mse(Tape *tp, PMap *pm) {
    int64_t n = 256, s[1] = {n};
    Tensor *a = bwo_tensor(1, s, -1, 1), *b = bwo_tensor(1, s, -1, 1);
    int64_t one = 1;
    Tensor *l = tensor_alloc(1, &one);
    bwo_reg(pm, a);
    t_mse(tp, a->data, b->data, l->data, n, 0.5);
}
#endif /* NCM_CUDA (bwo helpers) */
int bwdops_run(double tol) {
#ifdef NCM_CUDA
    arena_init_default();
    static const struct { bwo_build_fn f; const char *nm; } ops[] = {
        {bwo_linear, "linear"}, {bwo_linear_big, "linear_qwen"}, {bwo_ln, "ln"}, {bwo_rmsn, "rmsn"},
        {bwo_gn, "gn"}, {bwo_gelu, "gelu"}, {bwo_silu, "silu"},
        {bwo_relu2, "relu2"}, {bwo_softmax, "softmax"}, {bwo_rope, "rope"},
        {bwo_sdpa, "sdpa"}, {bwo_add, "add"}, {bwo_scale, "scale"},
        {bwo_concat, "concat"}, {bwo_join, "join"}, {bwo_split, "split"},
        {bwo_merge, "merge"}, {bwo_headsplit, "headsplit"},
        {bwo_rope_gqa, "rope_gqa"}, {bwo_sdpa_gqa, "sdpa_gqa"},
        {bwo_sdpa_gqa_big, "sdpa_gqa_big"},
        {bwo_mul, "mul"}, {bwo_gather, "gather"},
        {bwo_embed, "embed"}, {bwo_ce, "ce"}, {bwo_tied, "tied"},
        {bwo_mse, "mse"}, {bwo_qwenblock, "qwenblock"},
    };
    bwo_nfail = 0; bwo_gmax = 0; bwo_gname = ""; bwo_tol = tol;
    for (unsigned i = 0; i < sizeof(ops) / sizeof(ops[0]); i++)
        bwo_run_one(ops[i].f, ops[i].nm);
    printf("bwdops: %s (global maxabs=%.3g at %s, tol=%.3g)\n",
           bwo_nfail ? "FAIL" : "PASS", bwo_gmax, bwo_gname, tol);
    return bwo_nfail ? 1 : 0;
#else
    (void)tol;
    fprintf(stderr, "bwdops needs a CUDA build (no device backward on CPU)\n");
    return 2;
#endif
}

/* ---------------- gradient check (finite differences) ----------------
 * Analytic grads from train_ncp_loss+tape vs numeric grads from the
 * VERIFIED ncp_forward + host CE. Validates recording forward + backward. */

static int gradcheck_run(NCPState *m, WeightStore *ws, NCMConfig *c,
                         const int *prefix, const int *codes, int B, int n_checks) {
    int P = ncp_P(m), T = ncp_T(m), S = P + T, D = c->vq_embed_dim;
    int lvl_first = ncp_lvlfirst(m);
    int total = vq_total_dummy(c);
    VQState *vq = ncp_vq(m);

    /* ---- build x once on host (frozen path, mirrors train_ncp_step) ---- */
    int *bids = (int *)malloc((size_t)B * P * sizeof(int));
    memcpy(bids, prefix, (size_t)B * P * sizeof(int));
    /* z_prefix via frozen encoder wte (host compute from staged weight) */
    Tensor *te = weights_get(ws, "tok.encoder.wte.weight");
    float *hW = (float*)backend_to_host(te->data, (size_t)te->n * sizeof(float));
    float *zpre_h = (float *)malloc((size_t)B * P * D * sizeof(float));
    for (int b = 0; b < B; b++) for (int t = 0; t < P; t++) for (int d = 0; d < D; d++)
        zpre_h[((size_t)b * P + t) * D + d] = hW[(size_t)bids[b * P + t] * D + d];
    free(bids); free(hW);
    /* xt from frozen VQ (host path mirroring idx_to_ncm_input is complex;
     * instead stage through backend once) */
    arena_reset();
    float *xt_b = arena_alloc((size_t)B * (total - (lvl_first ? 0 : c->levels[0])) * D);
    vq_idx_to_ncm_input(vq, codes, B, xt_b);
    float *xt_h = (float*)backend_to_host(xt_b, (size_t)B * (total - (lvl_first ? 0 : c->levels[0])) * D * sizeof(float));
    int T2 = total - (lvl_first ? 0 : c->levels[0]);
    /* l0q host */
    float *l0q_h = NULL;
    int L0 = (lvl_first == 0) ? c->levels[0] : 0;
    if (L0) {
        Tensor *tq = weights_get(ws, "gen.l0_queries");
        l0q_h = (float*)backend_to_host(tq->data, (size_t)L0 * D * sizeof(float));
    }
    float *xh = (float *)malloc((size_t)B * S * D * sizeof(float));
    for (int b = 0; b < B; b++) {
        memcpy(xh + (size_t)b * S * D, zpre_h + (size_t)b * P * D, (size_t)P * D * sizeof(float));
        if (L0) memcpy(xh + ((size_t)b * S + P) * D, l0q_h, (size_t)L0 * D * sizeof(float));
        memcpy(xh + ((size_t)b * S + P + L0) * D, xt_h + (size_t)b * T2 * D, (size_t)T2 * D * sizeof(float));
    }
    free(zpre_h); free(xt_h);

    /* ---- analytic grads via recorded path ---- */
    Tape *tp = tape_new();
    arena_reset();
    float *xfull = arena_alloc((size_t)B * S * D);
    mem_h2d(xfull, xh, (size_t)B * S * D * sizeof(float));
    double ppl = 0;
    double loss0 = train_ncp_loss(tp, m, ws, c, xfull, prefix, codes, NULL, B, &ppl);
    (void)ppl;
    (void)loss0;
    PMap pm;
    int cap = ws->count, n = 0;
    pm.params = (float **)malloc(cap * sizeof(float *));
    pm.grads = (float **)malloc(cap * sizeof(float *));
    pm.sizes = (int64_t *)malloc(cap * sizeof(int64_t));
    for (int i = 0; i < ws->count; i++) {
        if (is_frozen_ncp(ws->items[i].name)) continue;
        pm.params[n] = ws->items[i].t->data;
        pm.sizes[n] = ws->items[i].t->n;
        pm.grads[n] = (float *)calloc(ws->items[i].t->n, sizeof(float));
        n++;
    }
    pm.n = n;
    tape_backward(tp, &pm, 0);
    tape_free(tp);

    /* ---- numeric grads via verified ncp_forward + host CE ---- */
    int nlev = ncp_ntgt(m);
    double wlev = 1.0 / nlev;
    /* forward-match sanity: numeric loss at base params vs analytic loss */
    {
        arena_reset();
        float *xb = arena_alloc((size_t)B * S * D);
        mem_h2d(xb, xh, (size_t)B * S * D * sizeof(float));
        float *lg2 = arena_alloc((size_t)B * T * 8192);
        /* NOTE: ncp_forward pads to its own maxV; recompute it here */
        int maxV = 0;
        for (int q = 0; q < nlev; q++) if (c->vocab_size[q] > maxV) maxV = c->vocab_size[q];
        ncp_forward(m, xb, prefix, lg2, B);
        float *hl = (float*)backend_to_host(lg2, (size_t)B * T * maxV * sizeof(float));
        double tot = 0;
        int off = 0;
        for (int q = 0; q < nlev; q++) {
            int lk = c->levels[q + lvl_first], V = c->vocab_size[q];
            int bo = 0;
            for (int qq = 0; qq < q + lvl_first; qq++) bo += c->levels[qq];
            double lt = 0;
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++) {
                int tg = codes[bb * total + bo + t];
                const float *row = hl + ((size_t)bb * T + off + t) * maxV;
                double mx = row[0];
                for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
                double den = 0;
                for (int v = 0; v < V; v++) den += exp(row[v] - mx);
                double lv = wlev * (-log(exp(row[tg] - mx) / den + 1e-30)) / (B * lk);
                if (!(lv == lv)) fprintf(stderr, "[dbg] nan at q=%d bb=%d t=%d tg=%d mx=%.3f den=%.3f row0=%.3f\n", q, bb, t, tg, mx, den, row[0]);
                lt += lv;
                tot += lv;
            }
            off += lk;
            if (q < 3) fprintf(stderr, "[dbg] level %d contrib %.4f\n", q, lt);
        }
        free(hl);
        printf("[gradcheck] analytic loss=%.6f numeric loss=%.6f %s\n",
               loss0, tot, (fabs(loss0 - tot) < 1e-3 && tot == tot) ? "FORWARD-MATCH" : "FORWARD-DIFF");
        { /* debug first rows */
            float *hd = (float*)backend_to_host(lg2, (size_t)12 * sizeof(float));
            fprintf(stderr, "[dbg] lg2[0:3]=%.3f %.3f %.3f | code0=%d V0=%d\n", hd[0], hd[1], hd[2], codes[0], c->vocab_size[0]);
            free(hd);
        }
    }
    int fails = 0, checked = 0;
    srand(12345);
    for (int k = 0; k < n_checks; k++) {
        int pi = rand() % n;
        if (pm.sizes[pi] > 200000) continue; /* keep it fast */
        int64_t idx = rand() % pm.sizes[pi];
        double an = ((float *)pm.grads[pi])[idx];
        /* numeric: perturb backend param, forward, host CE */
        float *ph = (float *)malloc(pm.sizes[pi] * sizeof(float));
        mem_d2h(ph, pm.params[pi], pm.sizes[pi] * sizeof(float));
        double lp = 0, lm = 0;
        const double eps = getenv("CNCM_EPS") ? atof(getenv("CNCM_EPS")) : 1e-3;
        for (int s = -1; s <= 1; s += 2) {
            ph[idx] += (float)(s * eps * (s < 0 ? 1 : 2)); /* base-eps, then base+eps */
            mem_h2d(pm.params[pi], ph, pm.sizes[pi] * sizeof(float));
            arena_reset();
            float *xb = arena_alloc((size_t)B * S * D);
            mem_h2d(xb, xh, (size_t)B * S * D * sizeof(float));
            int maxV = 0;
            for (int q = 0; q < nlev; q++) if (c->vocab_size[q] > maxV) maxV = c->vocab_size[q];
            float *lg2 = arena_alloc((size_t)B * T * maxV);
            ncp_forward(m, xb, prefix, lg2, B);
            float *hl = (float*)backend_to_host(lg2, (size_t)B * T * maxV * sizeof(float));
            /* per-level CE with per-board level-major codes */
            double tot = 0;
            int off = 0;
            for (int q = 0; q < nlev; q++) {
                int lk = c->levels[q + lvl_first], V = c->vocab_size[q];
                int bo = 0;
                for (int qq = 0; qq < q + lvl_first; qq++) bo += c->levels[qq];
                for (int bb = 0; bb < B; bb++) {
                    int *tg = (int *)malloc(lk * sizeof(int));
                    for (int t = 0; t < lk; t++) tg[t] = codes[bb * total + bo + t];
                    /* rows of lg2 for this (b, level): offset off+t, width 8192, first V valid */
                    for (int t = 0; t < lk; t++) {
                        /* gather row into temp */
                        (void)t;
                    }
                    /* inline CE over these rows */
                    for (int t = 0; t < lk; t++) {
                        const float *row = hl + ((size_t)bb * T + off + t) * maxV;
                        double mx = row[0];
                        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
                        double den = 0;
                        for (int v = 0; v < V; v++) den += exp(row[v] - mx);
                        tot += wlev * (-log(exp(row[tg[t]] - mx) / den + 1e-30)) / (B * lk);
                    }
                    free(tg);
                }
                off += lk;
            }
            free(hl);
            if (s < 0) lm = tot; else lp = tot;
            if (s == 1) { /* revert to base- eps for next iter: ph already +eps; subtract 2eps */
            }
        }
        /* restore param (currently at base+eps) */
        ph[idx] -= (float)eps;
        mem_h2d(pm.params[pi], ph, pm.sizes[pi] * sizeof(float));
        free(ph);
        double num = (lp - lm) / (2 * eps);
        double denom = fabs(an) + fabs(num) + 1e-8;
        double rel = fabs(an - num) / denom;
        const char *nm = "?";
        /* find name */
        for (int i = 0; i < ws->count; i++)
            if (ws->items[i].t->data == pm.params[pi]) { nm = ws->items[i].name; break; }
        if (!(rel <= 5e-2)) {
            printf("FAIL grad %s[%lld]: analytic=%.6f numeric=%.6f rel=%.2e\n",
                   nm, (long long)idx, an, num, rel);
            fails++;
        } else {
            printf("ok   grad %s[%lld]: %.6f vs %.6f\n", nm, (long long)idx, an, num);
        }
        checked++;
    }
    printf("GCHECK %s (%d checked)\n", fails ? "FAIL" : "PASS", checked);
    for (int i = 0; i < n; i++) free(pm.grads[i]);
    free(pm.params); free(pm.grads); free(pm.sizes);
    free(xh);
    if (l0q_h) free(l0q_h);
    return fails;
}

int gradcheck_run_cfg(const NCMConfig *c_, const char *weights_path, const char *data_path, int n_checks) {
    NCMConfig c = *c_;
    WeightStore ws = {NULL, 0};
    if (weights_load(&ws, weights_path)) return 1;
    arena_init_default();
    parena_init(1ULL << 28);
    VQVAEState *vqvae = vqvae_load(&c, &ws);
    NCPState *m = ncp_load(&c, &ws, vqvae);
    TrainData *d = train_data_load(data_path, 4);
    if (!d) return 1;
    int P = ncp_P(m);
    int total = vq_total_dummy(&c);
    if (d->row_len != P + total) { fprintf(stderr, "row mismatch\n"); return 1; }
    int B = d->n_rows < 2 ? d->n_rows : 2;
    int *prefix = (int *)malloc((size_t)B * P * sizeof(int));
    int *codes = (int *)malloc((size_t)B * total * sizeof(int));
    for (int b = 0; b < B; b++) {
        memcpy(prefix + b * P, d->rows + b * d->row_len, P * sizeof(int));
        memcpy(codes + b * total, d->rows + b * d->row_len + P, total * sizeof(int));
    }
    int fails = gradcheck_run(m, &ws, &c, prefix, codes, B, n_checks);
    free(prefix); free(codes);
    free(d->rows); free(d);
    return fails != 0;
}
