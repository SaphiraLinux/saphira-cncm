/* vq.c — MultiscaleResidualVectorQuantizer inference path.
 * Mirrors models/quant.py (z_to_idx / idx_to_zhats / idx_to_ncm_input /
 * get_next_autoregressive_input, _down_for_quant, _up_to_full, Phi resi).
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NCM_CUDA
#include <cuda_runtime.h>
#endif

struct VQState {
    int n_levels, *levels, *vocabs, D, L;
    float **E;                       /* codebooks on backend */
    int resi_enabled;
    float resi_ratio;
    float **phiW, **phib;            /* per stored phi */
    int n_phi;
    double *ticks;                   /* phi tick positions */
    int has_l1;
    int n_down, n_up;                /* cascade stages */
    float **downWs, **downBs;        /* [D,D,K=s] each */
    int *downKs;
    float **upWs, **upBs;            /* convT [D,D,K=s] each */
    int *upKs;
    int l1_norm, l1_norm_final;
    int has_all;
    float **drefW, **drefB, **urefW, **urefB; /* per middle level or NULL */
    float **ema_cs, **ema_w;             /* host EMA buffers (training) */
};

/* backend->host staging: TWO rotating slots so that two consecutive b2h
 * results stay live simultaneously (accumulate / resi / refine sites). */
static float *stage[2] = {NULL, NULL};
static size_t stage_n = 0;
static int stage_flip = 0;

static float *b2h(const float *dev, size_t n) {
    if (n > stage_n) {
        for (int i = 0; i < 2; i++) {
            free(stage[i]);
            stage[i] = (float *)malloc(n * sizeof(float));
            if (!stage[i]) { fprintf(stderr, "oom stage\n"); exit(1); }
        }
        stage_n = n;
    }
    stage_flip ^= 1;
    float *st = stage[stage_flip];
#ifdef NCM_CUDA
    cudaMemcpy(st, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
    memcpy(st, dev, n * sizeof(float));
#endif
    return st;
}

static Tensor *req(WeightStore *ws, const char *prefix, const char *name) {
    char key[320];
    snprintf(key, sizeof(key), "%s%s", prefix, name);
    Tensor *t = weights_get(ws, key);
    if (!t) { fprintf(stderr, "missing weight %s\n", key); exit(1); }
    return t;
}

static int haskey(WeightStore *ws, const char *prefix, const char *name) {
    char key[320];
    snprintf(key, sizeof(key), "%s%s", prefix, name);
    return weights_get(ws, key) != NULL;
}

VQState *vq_load(const NCMConfig *c, WeightStore *ws, const char *prefix) {
    VQState *v = (VQState *)calloc(1, sizeof(VQState));
    v->n_levels = c->n_qlevels;
    v->levels = (int *)malloc(v->n_levels * sizeof(int));
    v->vocabs = (int *)malloc(v->n_levels * sizeof(int));
    memcpy(v->levels, c->levels, v->n_levels * sizeof(int));
    memcpy(v->vocabs, c->vq_vocab, v->n_levels * sizeof(int));
    v->D = c->vq_embed_dim;
    v->L = c->levels[c->n_qlevels - 1];

    v->E = (float **)malloc(v->n_levels * sizeof(float *));
    for (int i = 0; i < v->n_levels; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "embeddings.%d.weight", i);
        Tensor *t = req(ws, prefix, nm);
        v->E[i] = t->data;
    }

    v->resi_enabled = c->quant_resi_enabled && fabsf(c->quant_resi_ratio) > 1e-6f;
    v->resi_ratio = c->quant_resi_ratio;
    if (v->resi_enabled) {
        if (c->quant_resi_share_mode == 0) {
            v->n_phi = v->n_levels;
        } else if (c->quant_resi_share_mode == 1) {
            v->n_phi = 1;
        } else {
            v->n_phi = c->quant_resi_share_mode;
        }
        v->phiW = (float **)malloc(v->n_phi * sizeof(float *));
        v->phib = (float **)malloc(v->n_phi * sizeof(float *));
        for (int i = 0; i < v->n_phi; i++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "quant_resi.%d.weight", i);
            if (!haskey(ws, prefix, nm) && v->n_phi == 1) {
                snprintf(nm, sizeof(nm), "quant_resi.weight");
            }
            v->phiW[i] = req(ws, prefix, nm)->data;
            snprintf(nm, sizeof(nm), "quant_resi.%d.bias", i);
            if (!haskey(ws, prefix, nm) && v->n_phi == 1)
                snprintf(nm, sizeof(nm), "quant_resi.bias");
            v->phib[i] = req(ws, prefix, nm)->data;
        }
        /* ticks replicate PhiNonShared / PhiPartiallyShared */
        v->ticks = (double *)malloc(v->n_phi * sizeof(double));
        int K = v->n_phi;
        for (int i = 0; i < K; i++) {
            if (K == 4)
                v->ticks[i] = (i + 1.0 / 3.0) / K;   /* linspace(1/3K, 1-1/3K) */
            else
                v->ticks[i] = (i + 0.5) / K;
        }
    }

    if (c->learned_l1_sampling) {
        v->has_l1 = 1;
        v->l1_norm = c->learned_l1_norm;
        v->l1_norm_final = c->learned_l1_norm_final;
        /* cascaded Sequential: conv weights live at module indices; GroupNorm
         * (affine=False) contributes no keys, so probe each index and keep
         * the ones that have weights — convs arrive in application order. */
        v->downWs = (float **)calloc(16, sizeof(float *));
        v->downBs = (float **)calloc(16, sizeof(float *));
        v->downKs = (int *)calloc(16, sizeof(int));
        v->upWs = (float **)calloc(16, sizeof(float *));
        v->upBs = (float **)calloc(16, sizeof(float *));
        v->upKs = (int *)calloc(16, sizeof(int));
        v->n_down = v->n_up = 0;
        /* Norm layers (affine=False) contribute no keys, so module indices
         * may skip; scan a fixed window and keep weights in order. */
        for (int s = 0; s < 16; s++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "learned_down_l1.%d.weight", s);
            if (haskey(ws, prefix, nm)) {
                Tensor *t = req(ws, prefix, nm);
                v->downWs[v->n_down] = t->data;
                v->downKs[v->n_down] = (int)t->shape[2];
                snprintf(nm, sizeof(nm), "learned_down_l1.%d.bias", s);
                v->downBs[v->n_down] = req(ws, prefix, nm)->data;
                v->n_down++;
            }
            snprintf(nm, sizeof(nm), "learned_up_l1.%d.weight", s);
            if (haskey(ws, prefix, nm)) {
                Tensor *u = req(ws, prefix, nm);
                v->upWs[v->n_up] = u->data;
                v->upKs[v->n_up] = (int)u->shape[2];
                snprintf(nm, sizeof(nm), "learned_up_l1.%d.bias", s);
                v->upBs[v->n_up] = req(ws, prefix, nm)->data;
                v->n_up++;
            }
        }
        if (!v->n_down) { fprintf(stderr, "learned_l1 weights missing\n"); exit(1); }
    }

    if (c->learned_all_sampling) {
        v->has_all = 1;
        int n = v->n_levels;
        v->drefW = (float **)calloc(n - 1, sizeof(float *));
        v->drefB = (float **)calloc(n - 1, sizeof(float *));
        v->urefW = (float **)calloc(n - 1, sizeof(float *));
        v->urefB = (float **)calloc(n - 1, sizeof(float *));
        for (int i = 1; i <= n - 2; i++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "learned_down_refine.%d.weight", i);
            if (haskey(ws, prefix, nm)) {
                v->drefW[i] = req(ws, prefix, nm)->data;
                snprintf(nm, sizeof(nm), "learned_down_refine.%d.bias", i);
                v->drefB[i] = req(ws, prefix, nm)->data;
            }
            snprintf(nm, sizeof(nm), "learned_up_refine.%d.weight", i);
            if (haskey(ws, prefix, nm)) {
                v->urefW[i] = req(ws, prefix, nm)->data;
                snprintf(nm, sizeof(nm), "learned_up_refine.%d.bias", i);
                v->urefB[i] = req(ws, prefix, nm)->data;
            }
        }
    }
    return v;
}

void vq_free(VQState *vq) {
    if (!vq) return;
    free(vq->levels); free(vq->vocabs); free(vq->E);
    free(vq->phiW); free(vq->phib); free(vq->ticks);
    free(vq->drefW); free(vq->drefB); free(vq->urefW); free(vq->urefB);
    free(vq->downWs); free(vq->downBs); free(vq->downKs);
    free(vq->upWs); free(vq->upBs); free(vq->upKs);
    if (vq->ema_cs) { for (int i = 0; i < vq->n_levels; i++) free(vq->ema_cs[i]); free(vq->ema_cs); }
    if (vq->ema_w) { for (int i = 0; i < vq->n_levels; i++) free(vq->ema_w[i]); free(vq->ema_w); }
    free(vq);
}

int vq_total_codes(const VQState *vq) {
    int s = 0;
    for (int i = 0; i < vq->n_levels; i++) s += vq->levels[i];
    return s;
}

static int phi_index(const VQState *vq, int level_idx) {
    if (vq->n_phi == 1) return 0;
    double at = (double)level_idx / (vq->n_levels - 1);
    int best = 0; double bd = 1e300;
    for (int j = 0; j < vq->n_phi; j++) {
        double d = fabs(vq->ticks[j] - at);
        if (d < bd) { bd = d; best = j; }
    }
    return best;
}

/* BLC <-> BCL transpose, backend-resident */
#ifdef NCM_CUDA
__global__ static void k_tr_b12(float *y, const float *x, int B, int L, int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * L * D) return;
    int d = (int)(i % D); int l = (int)((i / D) % L); int b = (int)(i / (D * L));
    y[((int64_t)b * D + d) * L + l] = x[((int64_t)b * L + l) * D + d];
}
#endif

static void transpose_bld(float *y, const float *x, int B, int L, int D) {
#ifdef NCM_CUDA
    int64_t n = (int64_t)B * L * D;
    k_tr_b12<<<(n + 255) / 256, 256>>>(y, x, B, L, D);
#else
    for (int b = 0; b < B; b++) for (int l = 0; l < L; l++) for (int d = 0; d < D; d++)
        y[((int64_t)b * D + d) * L + l] = x[((int64_t)b * L + l) * D + d];
#endif
}

/* in-place residual conv on [B][L][D] */
static void apply_resi(VQState *vq, float *h, int B, int L, int level_idx) {
    if (!vq->resi_enabled) return;
    int j = phi_index(vq, level_idx);
    int D = vq->D;
    float *bcl = arena_alloc((size_t)B * L * D);
    float *cv = arena_alloc((size_t)B * L * D);
    transpose_bld(bcl, h, B, L, D);
    conv1d_fwd(bcl, vq->phiW[j], vq->phib[j], cv, B, D, D, L, 3, 1, 1);
    transpose_bld(bcl, cv, B, D, L); /* back to BLD into bcl */
    /* h = (1-r) h + r conv(h) — needs elementwise; do on host-staged? No:
     * add a tiny kernel path via existing ops is overkill; do it with a
     * dedicated loop on staged memory only if small, else inline below. */
    /* backend elementwise blend: stage through host is O(B*L*D) memcpy, fine */
    float *hh = b2h(h, (size_t)B * L * D);
    float *cc = b2h(bcl, (size_t)B * L * D);
    float r = vq->resi_ratio;
    for (size_t i = 0; i < (size_t)B * L * D; i++) hh[i] = (1 - r) * hh[i] + r * cc[i];
#ifdef NCM_CUDA
    cudaMemcpy(h, hh, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
    /* b2h memcpy'd; hh aliases stage, h already updated? No: h IS backend==host
     * on CPU so memcpy into stage didn't touch h. Copy back. */
    memcpy(h, hh, (size_t)B * L * D * sizeof(float));
#endif
}

/* downsample z_rest [B][L0][D] -> [B][tlen][D] for quantization at level */
static void down_for_quant(VQState *vq, const float *z, int B, int Lin,
                           int level, float *out) {
    int D = vq->D;
    int tlen = vq->levels[level];
    if (level == 0 && vq->has_l1) {
        /* cascaded learned downsampler: [GN?] conv_s per stage */
        float *bcl = arena_alloc((size_t)B * Lin * D);
        transpose_bld(bcl, z, B, Lin, D);
        int Tc = Lin;
        float *cur = bcl, *nxt = NULL;
        for (int i = 0; i < vq->n_down; i++) {
            int s = vq->downKs[i];
            int To = Tc / s;
            if (vq->l1_norm && i > 0) {
                float *gn = arena_alloc((size_t)B * Tc * D);
                groupnorm_fwd(cur, gn, B, 8, D, Tc);
                cur = gn;
            }
            nxt = arena_alloc((size_t)B * To * D);
            conv1d_fwd(cur, vq->downWs[i], vq->downBs[i], nxt, B, D, D, Tc, s, s, 0);
            cur = nxt;
            Tc = To;
        }
        if (vq->l1_norm_final) {
            float *gn = arena_alloc((size_t)B * Tc * D);
            groupnorm_fwd(cur, gn, B, 8, D, Tc);
            cur = gn;
        }
        transpose_bld(out, cur, B, D, Tc);
    } else {
        /* F.interpolate area on BCL */
        float *bcl = arena_alloc((size_t)B * Lin * D);
        float *cv = arena_alloc((size_t)B * tlen * D);
        transpose_bld(bcl, z, B, Lin, D);
        interpolate_area_fwd(bcl, cv, B, D, Lin, tlen);
        transpose_bld(out, cv, B, D, tlen);
    }
    if (vq->has_all && level >= 1 && level <= vq->n_levels - 2 && vq->drefW[level]) {
        float *bcl = arena_alloc((size_t)B * tlen * D);
        float *cv = arena_alloc((size_t)B * tlen * D);
        transpose_bld(bcl, out, B, tlen, D);
        conv1d_fwd(bcl, vq->drefW[level], vq->drefB[level], cv, B, D, D, tlen, 3, 1, 1);
        transpose_bld(bcl, cv, B, D, tlen);
        /* out += refine(out): need out + bcl */
        float *a = b2h(out, (size_t)B * tlen * D);
        float *b = b2h(bcl, (size_t)B * tlen * D);
        for (size_t i = 0; i < (size_t)B * tlen * D; i++) a[i] += b[i];
#ifdef NCM_CUDA
        cudaMemcpy(out, a, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(out, a, (size_t)B * tlen * D * sizeof(float));
#endif
    }
}

/* upsample q [B][tlen][D] -> [B][L][D] */
static void up_to_full(VQState *vq, const float *q, int B, int tlen,
                       int level, float *out) {
    int D = vq->D, L = vq->L;
    if (level == 0 && vq->has_l1) {
        float *bcl = arena_alloc((size_t)B * tlen * D);
        transpose_bld(bcl, q, B, tlen, D);
        int Tc = tlen;
        float *cur = bcl, *nxt = NULL;
        for (int i = 0; i < vq->n_up; i++) {
            int s = vq->upKs[i];
            int To = (Tc - 1) * s + s;
            if (vq->l1_norm && i > 0) {
                float *gn = arena_alloc((size_t)B * Tc * D);
                groupnorm_fwd(cur, gn, B, 8, D, Tc);
                cur = gn;
            }
            nxt = arena_alloc((size_t)B * To * D);
            convtranspose1d_fwd(cur, vq->upWs[i], vq->upBs[i], nxt, B, D, D, Tc, s, s);
            cur = nxt;
            Tc = To;
        }
        if (vq->l1_norm_final) {
            float *gn = arena_alloc((size_t)B * Tc * D);
            groupnorm_fwd(cur, gn, B, 8, D, Tc);
            cur = gn;
        }
        transpose_bld(out, cur, B, D, Tc);
    } else {
        float *bcl = arena_alloc((size_t)B * tlen * D);
        float *cv = arena_alloc((size_t)B * L * D);
        transpose_bld(bcl, q, B, tlen, D);
        interpolate_linear_fwd(bcl, cv, B, D, tlen, L);
        transpose_bld(out, cv, B, D, L);
    }
    if (vq->has_all && level >= 1 && level <= vq->n_levels - 2 && vq->urefW[level]) {
        float *bcl = arena_alloc((size_t)B * L * D);
        float *cv = arena_alloc((size_t)B * L * D);
        transpose_bld(bcl, out, B, L, D);
        conv1d_fwd(bcl, vq->urefW[level], vq->urefB[level], cv, B, D, D, L, 3, 1, 1);
        transpose_bld(bcl, cv, B, D, L);
        float *a = b2h(out, (size_t)B * L * D);
        float *b = b2h(bcl, (size_t)B * L * D);
        for (size_t i = 0; i < (size_t)B * L * D; i++) a[i] += b[i];
#ifdef NCM_CUDA
        cudaMemcpy(out, a, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(out, a, (size_t)B * L * D * sizeof(float));
#endif
    }
}

void vq_z_to_idx(VQState *vq, const float *z, int B, int *idx) {
    int D = vq->D, L = vq->L;
    float *z_rest = arena_alloc((size_t)B * L * D);
    float *z_hat = arena_alloc((size_t)B * L * D);
    float *down = arena_alloc((size_t)B * L * D);   /* max tlen = L */
    float *d2 = arena_alloc((size_t)B * L * 16);    /* max V rows reused below */
    (void)d2;
    /* copy z -> z_rest (backend copy) */
#ifdef NCM_CUDA
    cudaMemcpy(z_rest, z, (size_t)B * L * D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#else
    memcpy(z_rest, z, (size_t)B * L * D * sizeof(float));
    memset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#endif
    float *up = arena_alloc((size_t)B * L * D);
    int off = 0;
    for (int lv = 0; lv < vq->n_levels; lv++) {
        int tlen = vq->levels[lv], V = vq->vocabs[lv];
        float *dd = arena_alloc((size_t)B * tlen * V);
        down_for_quant(vq, z_rest, B, L, lv, down);
        rowwise_l2sq(down, vq->E[lv], dd, (int64_t)B * tlen, V, D);
        float *hd = b2h(dd, (size_t)B * tlen * V);
        int total = vq_total_codes(vq);
        for (int r = 0; r < B * tlen; r++) {
            int bi = 0; float bv = hd[r * V];
            for (int j = 1; j < V; j++) if (hd[r * V + j] < bv) { bv = hd[r * V + j]; bi = j; }
            int b = r / tlen, t = r % tlen;
            idx[b * total + off + t] = bi;
        }
        off += tlen;
        /* q = E[idx] (host gather, codebooks are small), then upsample. */
        float *qh = b2h(up, 1); /* ensure stage big enough later; no-op */
        (void)qh;
        {
            float *q = arena_alloc((size_t)B * tlen * D);
            /* host-side gather from codebook (codebooks small: V*D) */
            float *Eh = b2h(vq->E[lv], (size_t)V * D);
            /* NOTE: b2h reuses one stage buffer; Eh invalid after next b2h.
             * Copy codebook to a private host buffer. */
            float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
            memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
            /* level-major idx base for this level */
            int base = 0;
            for (int k = 0; k < lv; k++) base += vq->levels[k];
            float *qq = (float *)malloc((size_t)B * tlen * D * sizeof(float));
            for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++) {
                int code = idx[b * vq_total_codes(vq) + base + t];
                memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
            }
            free(Ec);
#ifdef NCM_CUDA
            cudaMemcpy(q, qq, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(q, qq, (size_t)B * tlen * D * sizeof(float));
#endif
            free(qq);
            up_to_full(vq, q, B, tlen, lv, up);
        }
        apply_resi(vq, up, B, L, lv);
        /* z_hat += up; z_rest -= up */
        {
            float *zh = b2h(z_hat, (size_t)B * L * D);
            float *uu = b2h(up, (size_t)B * L * D);
            float *uc = (float *)malloc((size_t)B * L * D * sizeof(float));
            memcpy(uc, uu, (size_t)B * L * D * sizeof(float));
            for (size_t i = 0; i < (size_t)B * L * D; i++) zh[i] += uc[i];
#ifdef NCM_CUDA
            cudaMemcpy(z_hat, zh, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(z_hat, zh, (size_t)B * L * D * sizeof(float));
#endif
            float *zr = b2h(z_rest, (size_t)B * L * D);
            float *zrc = (float *)malloc((size_t)B * L * D * sizeof(float));
            memcpy(zrc, zr, (size_t)B * L * D * sizeof(float));
            for (size_t i = 0; i < (size_t)B * L * D; i++) zrc[i] -= uc[i];
            free(uc);
#ifdef NCM_CUDA
            cudaMemcpy(z_rest, zrc, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(z_rest, zrc, (size_t)B * L * D * sizeof(float));
#endif
            free(zrc);
        }
    }
}

void vq_idx_to_zhats(VQState *vq, const int *idx, int B, float **zhats) {
    int D = vq->D, L = vq->L;
    float *z_hat = arena_alloc((size_t)B * L * D);
#ifdef NCM_CUDA
    cudaMemset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#else
    memset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#endif
    float *up = arena_alloc((size_t)B * L * D);
    int base = 0;
    for (int lv = 0; lv < vq->n_levels; lv++) {
        int tlen = vq->levels[lv], V = vq->vocabs[lv];
        float *Eh = b2h(vq->E[lv], (size_t)V * D);
        float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
        memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
        float *qq = (float *)malloc((size_t)B * tlen * D * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++) {
            int code = idx[b * vq_total_codes(vq) + base + t];
            memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
        }
        free(Ec);
        float *q = arena_alloc((size_t)B * tlen * D);
#ifdef NCM_CUDA
        cudaMemcpy(q, qq, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(q, qq, (size_t)B * tlen * D * sizeof(float));
#endif
        free(qq);
        up_to_full(vq, q, B, tlen, lv, up);
        apply_resi(vq, up, B, L, lv);
        {
            float *zh = b2h(z_hat, (size_t)B * L * D);
            float *uu = b2h(up, (size_t)B * L * D);
            float *uc = (float *)malloc((size_t)B * L * D * sizeof(float));
            memcpy(uc, uu, (size_t)B * L * D * sizeof(float));
            for (size_t i = 0; i < (size_t)B * L * D; i++) zh[i] += uc[i];
#ifdef NCM_CUDA
            cudaMemcpy(z_hat, zh, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(z_hat, zh, (size_t)B * L * D * sizeof(float));
#endif
            free(uc);
        }
        /* snapshot */
#ifdef NCM_CUDA
        cudaMemcpy(zhats[lv], z_hat, (size_t)B * L * D * sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(zhats[lv], z_hat, (size_t)B * L * D * sizeof(float));
#endif
        base += tlen;
    }
}

void vq_next_input(VQState *vq, const int *codes, int level_idx,
                   float *z_hat, float *x_out, int B) {
    int D = vq->D, L = vq->L;
    int tlen = vq->levels[level_idx], V = vq->vocabs[level_idx];
    float *Eh = b2h(vq->E[level_idx], (size_t)V * D);
    float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
    memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
    float *qq = (float *)malloc((size_t)B * tlen * D * sizeof(float));
    for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++) {
        int code = codes[b * tlen + t];
        memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
    }
    free(Ec);
    float *q = arena_alloc((size_t)B * tlen * D);
#ifdef NCM_CUDA
    cudaMemcpy(q, qq, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
    memcpy(q, qq, (size_t)B * tlen * D * sizeof(float));
#endif
    free(qq);
    float *up = arena_alloc((size_t)B * L * D);
    up_to_full(vq, q, B, tlen, level_idx, up);
    apply_resi(vq, up, B, L, level_idx);
    {
        float *zh = b2h(z_hat, (size_t)B * L * D);
        float *uu = b2h(up, (size_t)B * L * D);
        float *uc = (float *)malloc((size_t)B * L * D * sizeof(float));
        memcpy(uc, uu, (size_t)B * L * D * sizeof(float));
        for (size_t i = 0; i < (size_t)B * L * D; i++) zh[i] += uc[i];
#ifdef NCM_CUDA
        cudaMemcpy(z_hat, zh, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(z_hat, zh, (size_t)B * L * D * sizeof(float));
#endif
        /* x_out = area(z_hat -> levels[level_idx+1]) in BLD */
        float *bcl = arena_alloc((size_t)B * L * D);
        float *cv = arena_alloc((size_t)B * vq->levels[level_idx + 1] * D);
        transpose_bld(bcl, z_hat, B, L, D);
        interpolate_area_fwd(bcl, cv, B, D, L, vq->levels[level_idx + 1]);
        transpose_bld(x_out, cv, B, D, vq->levels[level_idx + 1]);
        free(uc);
    }
}

/* idx [B][total] level-major -> NCM target input [B][T][D] (idx_to_ncm_input).
 * x_out lives on the backend; assembled via host staging. */
void vq_idx_to_ncm_input(VQState *vq, const int *idx, int B, float *x_out) {
    int D = vq->D, L = vq->L;
    int T = vq_total_codes(vq) - vq->levels[0];
    float *z_hat = arena_alloc((size_t)B * L * D);
#ifdef NCM_CUDA
    cudaMemset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#else
    memset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#endif
    float *up = arena_alloc((size_t)B * L * D);
    float *full = (float *)malloc((size_t)B * T * D * sizeof(float));
    int base = 0, off = 0;
    for (int li = 0; li < vq->n_levels - 1; li++) {
        int tlen = vq->levels[li], V = vq->vocabs[li];
        int nlen = vq->levels[li + 1];
        float *Eh = b2h(vq->E[li], (size_t)V * D);
        float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
        memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
        float *qq = (float *)malloc((size_t)B * tlen * D * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++) {
            int code = idx[b * vq_total_codes(vq) + base + t];
            memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
        }
        free(Ec);
        float *q = arena_alloc((size_t)B * tlen * D);
#ifdef NCM_CUDA
        cudaMemcpy(q, qq, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(q, qq, (size_t)B * tlen * D * sizeof(float));
#endif
        free(qq);
        up_to_full(vq, q, B, tlen, li, up);
        apply_resi(vq, up, B, L, li);
        {
            float *zh = b2h(z_hat, (size_t)B * L * D);
            float *uu = b2h(up, (size_t)B * L * D);
            float *uc = (float *)malloc((size_t)B * L * D * sizeof(float));
            memcpy(uc, uu, (size_t)B * L * D * sizeof(float));
            for (size_t i = 0; i < (size_t)B * L * D; i++) zh[i] += uc[i];
            free(uc);
#ifdef NCM_CUDA
            cudaMemcpy(z_hat, zh, (size_t)B * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(z_hat, zh, (size_t)B * L * D * sizeof(float));
#endif
        }
        /* x_out chunk = area(z_hat -> nlen), BLD */
        float *bcl = arena_alloc((size_t)B * L * D);
        float *cv = arena_alloc((size_t)B * nlen * D);
        transpose_bld(bcl, z_hat, B, L, D);
        interpolate_area_fwd(bcl, cv, B, D, L, nlen);
        float *hc = (float *)malloc((size_t)B * nlen * D * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hc, cv, (size_t)B * nlen * D * sizeof(float), cudaMemcpyDeviceToHost);
        /* cv is BCL; transpose to BLD on host */
#else
        memcpy(hc, cv, (size_t)B * nlen * D * sizeof(float));
#endif
        {
            /* hc is BCL [B][D][nlen] -> BLD chunk */
            for (int b = 0; b < B; b++) for (int t = 0; t < nlen; t++) for (int d = 0; d < D; d++)
                full[((size_t)b * T + off + t) * D + d] = hc[((size_t)b * D + d) * nlen + t];
        }
        free(hc);
        base += tlen;
        off += nlen;
    }
#ifdef NCM_CUDA
    cudaMemcpy(x_out, full, (size_t)B * T * D * sizeof(float), cudaMemcpyHostToDevice);
#else
    memcpy(x_out, full, (size_t)B * T * D * sizeof(float));
#endif
    free(full);
}

/* ---- EMA codebook maintenance (training) ---- */
void vq_ema_init(VQState *vq) {
    /* host-side EMA buffers, cold-started from current codebooks */
    if (vq->ema_cs) return;
    vq->ema_cs = (float **)calloc(vq->n_levels, sizeof(float *));
    vq->ema_w = (float **)calloc(vq->n_levels, sizeof(float *));
    for (int i = 0; i < vq->n_levels; i++) {
        int V = vq->vocabs[i];
        /* NOTE: Python inits cluster counts to ZERO; we init to ONE to avoid
         * cold-start blowup at small batch sizes (zero init makes the
         * smoothed denominator ~0.0025 -> ~400x codebook explosion on step
         * 0). Steady state is identical (real counts dominate after a few
         * steps). Set to 0.0 for bit-exact paper parity at large batches. */
        vq->ema_cs[i] = (float *)malloc(V * sizeof(float));
        for (int v = 0; v < V; v++) vq->ema_cs[i][v] = 1.0f;
        vq->ema_w[i] = (float *)malloc((size_t)V * vq->D * sizeof(float));
        float *hE = b2h(vq->E[i], (size_t)V * vq->D);
        memcpy(vq->ema_w[i], hE, (size_t)V * vq->D * sizeof(float));
    }
}

void vq_ema_update(VQState *vq, const float *z, const int *codes, int B, double decay, double eps) {
    vq_ema_init(vq);
    int D = vq->D, L = vq->L;
    float *z_rest = arena_alloc((size_t)B * L * D);
    float *z_hat = arena_alloc((size_t)B * L * D);
#ifdef NCM_CUDA
    cudaMemcpy(z_rest, z, (size_t)B * L * D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#else
    memcpy(z_rest, z, (size_t)B * L * D * sizeof(float));
    memset(z_hat, 0, (size_t)B * L * D * sizeof(float));
#endif
    float *up = arena_alloc((size_t)B * L * D);
    float *down = arena_alloc((size_t)B * L * D);
    int off = 0;
    for (int lv = 0; lv < vq->n_levels; lv++) {
        int tlen = vq->levels[lv], V = vq->vocabs[lv];
        down_for_quant(vq, z_rest, B, L, lv, down);
        /* stage flat inputs + gather assigned rows */
        float *hf = b2h(down, (size_t)B * tlen * D);
        float *hfc = (float *)malloc((size_t)B * tlen * D * sizeof(float));
        memcpy(hfc, hf, (size_t)B * tlen * D * sizeof(float));
        /* EMA stats on host */
        double *enc = (double *)calloc(V, sizeof(double));
        for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++)
            enc[codes[b * vq_total_codes(vq) + off + t]] += 1.0;
        for (int v = 0; v < V; v++) {
            vq->ema_cs[lv][v] = (float)(decay * vq->ema_cs[lv][v] + (1 - decay) * enc[v]);
            for (int d = 0; d < D; d++) {
                double dw = 0;
                for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++)
                    if (codes[b * vq_total_codes(vq) + off + t] == v)
                        dw += hfc[((size_t)b * tlen + t) * D + d];
                vq->ema_w[lv][v * D + d] = (float)(decay * vq->ema_w[lv][v * D + d] + (1 - decay) * dw);
            }
        }
        free(enc);
        /* write back smoothed codebook */
        double n = 0;
        for (int v = 0; v < V; v++) n += vq->ema_cs[lv][v];
        float *hE = (float *)malloc((size_t)V * D * sizeof(float));
        for (int v = 0; v < V; v++) {
            double sm = (vq->ema_cs[lv][v] + eps) / (n + V * eps) * n;
            for (int d = 0; d < D; d++)
                hE[v * D + d] = (float)(vq->ema_w[lv][v * D + d] / sm);
        }
#ifdef NCM_CUDA
        cudaMemcpy((void *)vq->E[lv], hE, (size_t)V * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy((void *)vq->E[lv], hE, (size_t)V * D * sizeof(float));
#endif
        free(hE); free(hfc);
        /* advance residual (host-gathered q, backend upsample) */
        float *Eh = b2h(vq->E[lv], (size_t)V * D);
        float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
        memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
        float *qq = (float *)malloc((size_t)B * tlen * D * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < tlen; t++) {
            int code = codes[b * vq_total_codes(vq) + off + t];
            memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
        }
        free(Ec);
        float *q = arena_alloc((size_t)B * tlen * D);
#ifdef NCM_CUDA
        cudaMemcpy(q, qq, (size_t)B * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(q, qq, (size_t)B * tlen * D * sizeof(float));
#endif
        free(qq);
        up_to_full(vq, q, B, tlen, lv, up);
        apply_resi(vq, up, B, L, lv);
        {
            float *zh = b2h(z_hat, (size_t)B * L * D);
            float *uu = b2h(up, (size_t)B * L * D);
            float *zrc;
            {
                /* z_hat += up (host), write back */
                float *uc = (float *)malloc((size_t)B * L * D * sizeof(float));
                memcpy(uc, uu, (size_t)B * L * D * sizeof(float));
                for (size_t i = 0; i < (size_t)B * L * D; i++) zh[i] += uc[i];
                free(uc);
                mem_h2d(z_hat, zh, (size_t)B * L * D * sizeof(float));
                /* z_rest -= up */
                float *zr = b2h(z_rest, (size_t)B * L * D);
                zrc = (float *)malloc((size_t)B * L * D * sizeof(float));
                memcpy(zrc, zr, (size_t)B * L * D * sizeof(float));
                for (size_t i = 0; i < (size_t)B * L * D; i++) zrc[i] -= uu[i];
                mem_h2d(z_rest, zrc, (size_t)B * L * D * sizeof(float));
                free(zrc);
            }
        }
        off += tlen;
    }
}

int vq_nlevels(const VQState *vq) { return vq->n_levels; }
int vq_level_len(const VQState *vq, int k) { return vq->levels[k]; }
int vq_level_V(const VQState *vq, int k) { return vq->vocabs[k]; }
int vq_Ltotal(const VQState *vq) { return vq->L; }
int vq_Ddim(const VQState *vq) { return vq->D; }

/* ---- dead-code revival (training aid; no architecture change) ----
 * Detects codes with zero uses over a fixed observation window of real
 * encoder outputs, reseeds them from actual encoder-output frames at
 * the matching level (never random vectors), and resets their EMA
 * counts/accumulators consistently (ema_cs=1.0 like cold start,
 * ema_w=reseeded vector scaled so the smoothing write-back reproduces
 * it exactly; stale means therefore cannot snap it straight back).
 *
 * Traversal mirrors vq_z_to_idx (same downsample/residual/resi path),
 * but records per-level usage and keeps a bounded reservoir of input
 * frames per level instead of argmin codes. Deterministic given seed.
 * Returns the number of codes revived. Prints per-level used/total
 * before and after. */
static unsigned long long revive_rng_next(unsigned long long *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return (*s >> 11);
}

#define REVIVE_RESERVOIR 2048

int vq_revive_dead(VQState *vq, VQVAEState *vqvae, const int *ids, int nrows,
                   int T, int D, unsigned long long seed, double eps) {
    int nl = vq->n_levels, L = vq->L;
    int total = vq_total_codes(vq);
    if (nl <= 0 || L <= 0 || T <= 0 || D <= 0 || nrows <= 0 || total <= 0) return 0;
    vq_ema_init(vq);
    int stride = 0;
    for (int l = 0; l < nl; l++)
        if (vq->vocabs[l] > stride) stride = vq->vocabs[l];
    /* per-level usage over the window + frame reservoirs (host) */
    long *use = (long *)calloc((size_t)nl * stride, sizeof(long));
    float **pool = (float **)malloc((size_t)nl * sizeof(float *));
    int *poolfill = (int *)calloc((size_t)nl, sizeof(int));
    long *poolseen = (long *)calloc((size_t)nl, sizeof(long));
    for (int l = 0; l < nl; l++)
        pool[l] = (float *)malloc((size_t)REVIVE_RESERVOIR * D * sizeof(float));
    int *off = (int *)malloc((size_t)(nl + 1) * sizeof(int));
    off[0] = 0;
    for (int l = 0; l < nl; l++) off[l + 1] = off[l] + vq->levels[l];
    unsigned long long rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int B = 32;
    int *sbuf = (int *)malloc((size_t)B * T * sizeof(int));
    int *codes = (int *)malloc((size_t)B * total * sizeof(int));
    for (int s = 0; s < nrows; s += B) {
        int bb = nrows - s < B ? nrows - s : B;
        for (int b = 0; b < bb; b++)
            memcpy(sbuf + b * T, ids + (s + b) * T, T * sizeof(int));
        float *z = arena_alloc((size_t)bb * L * D);
        vqvae_encode_z(vqvae, sbuf, z, bb);
        vq_z_to_idx(vq, z, bb, codes);
        /* residual traversal (mirrors vq_z_to_idx): per-level inputs */
        float *z_rest = arena_alloc((size_t)bb * L * D);
        float *z_hat = arena_alloc((size_t)bb * L * D);
#ifdef NCM_CUDA
        cudaMemcpy(z_rest, z, (size_t)bb * L * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(z_hat, 0, (size_t)bb * L * D * sizeof(float));
#else
        memcpy(z_rest, z, (size_t)bb * L * D * sizeof(float));
        memset(z_hat, 0, (size_t)bb * L * D * sizeof(float));
#endif
        float *up = arena_alloc((size_t)bb * L * D);
        float *down = arena_alloc((size_t)bb * L * D);
        for (int lv = 0; lv < nl; lv++) {
            int tlen = vq->levels[lv], V = vq->vocabs[lv];
            down_for_quant(vq, z_rest, bb, L, lv, down);
            float *hf = b2h(down, (size_t)bb * tlen * D);
            float *hfc = (float *)malloc((size_t)bb * tlen * D * sizeof(float));
            memcpy(hfc, hf, (size_t)bb * tlen * D * sizeof(float));
            for (int b = 0; b < bb; b++) for (int t = 0; t < tlen; t++) {
                int cd = codes[(size_t)b * total + off[lv] + t];
                if (cd >= 0 && cd < V) use[(size_t)lv * stride + cd]++;
                /* reservoir-sample an input frame for later reseeding */
                long k = (long)poolseen[lv]++;
                float *dst;
                if (k < REVIVE_RESERVOIR) dst = pool[lv] + (size_t)k * D;
                else {
                    unsigned long long r = revive_rng_next(&rng);
                    if ((long)(r % (unsigned long long)(k + 1)) >= REVIVE_RESERVOIR) continue;
                    dst = pool[lv] + (size_t)(r % REVIVE_RESERVOIR) * D;
                }
                memcpy(dst, hfc + ((size_t)b * tlen + t) * D, D * sizeof(float));
                if (k < REVIVE_RESERVOIR) poolfill[lv]++;
            }
            free(hfc);
            /* advance residual exactly like vq_z_to_idx (host gather) */
            float *Eh = b2h(vq->E[lv], (size_t)V * D);
            float *Ec = (float *)malloc((size_t)V * D * sizeof(float));
            memcpy(Ec, Eh, (size_t)V * D * sizeof(float));
            int base = off[lv];
            float *qq = (float *)malloc((size_t)bb * tlen * D * sizeof(float));
            for (int b = 0; b < bb; b++) for (int t = 0; t < tlen; t++) {
                int code = codes[(size_t)b * total + base + t];
                if (code < 0 || code >= V) code = 0;
                memcpy(qq + ((size_t)b * tlen + t) * D, Ec + (size_t)code * D, D * sizeof(float));
            }
            free(Ec);
            float *q = arena_alloc((size_t)bb * tlen * D);
#ifdef NCM_CUDA
            cudaMemcpy(q, qq, (size_t)bb * tlen * D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy(q, qq, (size_t)bb * tlen * D * sizeof(float));
#endif
            free(qq);
            up_to_full(vq, q, bb, tlen, lv, up);
            apply_resi(vq, up, bb, L, lv);
            {
                /* z_hat += up; z_rest -= up (all bb-wide here) */
                float *zh = b2h(z_hat, (size_t)bb * L * D);
                float *uu = b2h(up, (size_t)bb * L * D);
                float *uc = (float *)malloc((size_t)bb * L * D * sizeof(float));
                memcpy(uc, uu, (size_t)bb * L * D * sizeof(float));
                for (size_t i = 0; i < (size_t)bb * L * D; i++) zh[i] += uc[i];
                free(uc);
#ifdef NCM_CUDA
                cudaMemcpy(z_hat, zh, (size_t)bb * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
                memcpy(z_hat, zh, (size_t)bb * L * D * sizeof(float));
#endif
                float *zr = b2h(z_rest, (size_t)bb * L * D);
                float *zrc = (float *)malloc((size_t)bb * L * D * sizeof(float));
                memcpy(zrc, zr, (size_t)bb * L * D * sizeof(float));
                for (size_t i = 0; i < (size_t)bb * L * D; i++) zrc[i] -= uu[i];
#ifdef NCM_CUDA
                cudaMemcpy(z_rest, zrc, (size_t)bb * L * D * sizeof(float), cudaMemcpyHostToDevice);
#else
                memcpy(z_rest, zrc, (size_t)bb * L * D * sizeof(float));
#endif
                free(zrc);
            }
        }
        arena_reset();
    }
    free(sbuf);
    /* usage report (before) */
    printf("[vq-revive] window=%d rows\n", nrows);
    for (int l = 0; l < nl; l++) {
        int u = 0;
        for (int v = 0; v < vq->vocabs[l]; v++) if (use[(size_t)l * stride + v]) u++;
        printf("[vq-revive] L%02d before used=%d/%d\n", l, u, vq->vocabs[l]);
    }
    fflush(stdout);
    /* reseed dead codes from distinct reservoir frames + exact EMA reset.
     * Pass 1: choose frames, set dead counts to cold-start 1.0.
     * Pass 2 (same level): recompute the level normalizer, then scale
     * each accumulator so the smoothing write-back reproduces its
     * reseeded vector exactly. A continued run therefore inherits an
     * EMA state consistent with the revised codebooks. */
    int revived = 0;
    int *revived_per = (int *)calloc((size_t)nl, sizeof(int));
    int *taken = (int *)calloc((size_t)REVIVE_RESERVOIR, sizeof(int));
    int *selfi = (int *)malloc((size_t)stride * sizeof(int));
    for (int l = 0; l < nl; l++) {
        int V = vq->vocabs[l];
        if (poolfill[l] <= 0) continue; /* no frames: cannot reseed */
        for (int v = 0; v < V; v++) selfi[v] = -1;
        for (int v = 0; v < V; v++) {
            if (use[(size_t)l * stride + v] > 0) continue;
            int fi = -1;
            for (int k = 0; k < poolfill[l]; k++) {
                unsigned long long r = revive_rng_next(&rng);
                int cand = (int)(r % (unsigned long long)poolfill[l]);
                if (!taken[cand]) { fi = cand; taken[cand] = 1; break; }
            }
            if (fi < 0) fi = (int)(revive_rng_next(&rng) % (unsigned long long)poolfill[l]);
            selfi[v] = fi;
            vq->ema_cs[l][v] = 1.0f;
            revived_per[l]++;
            revived++;
        }
        memset(taken, 0, (size_t)REVIVE_RESERVOIR * sizeof(int));
        if (!revived_per[l]) continue;
        double n = 0;
        for (int v = 0; v < V; v++) n += vq->ema_cs[l][v];
        float *hE = (float *)malloc((size_t)D * sizeof(float));
        for (int v = 0; v < V; v++) {
            if (selfi[v] < 0) continue;
            float *vec = pool[l] + (size_t)selfi[v] * D;
            double sm = (1.0 + eps) / (n + V * eps) * n;
            for (int d = 0; d < D; d++) {
                vq->ema_w[l][(size_t)v * D + d] = (float)(vec[d] * sm);
                hE[d] = vec[d];
            }
#ifdef NCM_CUDA
            cudaMemcpy((void *)(vq->E[l] + (size_t)v * D), hE, (size_t)D * sizeof(float), cudaMemcpyHostToDevice);
#else
            memcpy((void *)(vq->E[l] + (size_t)v * D), hE, (size_t)D * sizeof(float));
#endif
        }
        free(hE);
    }
    free(taken);
    free(selfi);
    /* usage report (after): previously-dead codes now point at data */
    for (int l = 0; l < nl; l++) {
        int used = 0;
        for (int v = 0; v < vq->vocabs[l]; v++) if (use[(size_t)l * stride + v]) used++;
        printf("[vq-revive] L%02d after used=%d/%d (+%d revived)\n",
               l, used + revived_per[l], vq->vocabs[l], revived_per[l]);
    }
    printf("[vq-revive] revived %d codes\n", revived);
    fflush(stdout);
    for (int l = 0; l < nl; l++) free(pool[l]);
    free(pool); free(poolfill); free(poolseen); free(use); free(off); free(codes);
    return revived;
}
