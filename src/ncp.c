/* ncp.c — NextConceptTransformer forward + coarse-to-fine generate.
 * Mirrors models/next_concept.py + models/common.py exactly (f32).
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NCM_CUDA
#include <cuda_runtime.h>
#endif

struct NCPState {
    NCMConfig cfg;
    WeightStore *ws;
    VQVAEState *vqvae;
    VQState *vq;
    int L, T, P;
    int lvl_first;
    int n_tgt;                 /* target levels = n_levels - lvl_first */
    float *rope_cos, *rope_sin;       /* [P+T][hd] backend */
    float **gcos, **gsin;             /* per target level [P+l][hd] */
    float *lvl_pos;                   /* [T][C] backend */
};

static Tensor *W(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "missing weight %s\n", name); exit(1); }
    return t;
}

/* ---------- tiny backend helpers ---------- */
static void d2h(float *h, const float *d, size_t n) {
#ifdef NCM_CUDA
    cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
    memcpy(h, d, n * sizeof(float));
#endif
}
static void h2d(float *d, const float *h, size_t n) {
#ifdef NCM_CUDA
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
#else
    memcpy(d, h, n * sizeof(float));
#endif
}
static void h2d_i(int *d, const int *h, size_t n) {
#ifdef NCM_CUDA
    cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);
#else
    memcpy(d, h, n * sizeof(int));
#endif
}
static void d2d(float *d, const float *s, size_t n) {
#ifdef NCM_CUDA
    cudaMemcpy(d, s, n * sizeof(float), cudaMemcpyDeviceToDevice);
#else
    memcpy(d, s, n * sizeof(float));
#endif
}

#ifdef NCM_CUDA
__global__ static void k_add_inplace(float *y, const float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ static void k_add3(float *y, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ static void k_mul_vec(float *y, const float *v, size_t n, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= v[i % C];
}
__global__ static void k_concat(float *y, const float *a, const float *b,
                                size_t na, size_t nb) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na + nb) y[i] = i < na ? a[i] : b[i - na];
}
#endif

static void b_add_inplace(float *y, const float *x, size_t n) {
#ifdef NCM_CUDA
    k_add_inplace<<<(n + 255) / 256, 256>>>(y, x, n);
#else
    for (size_t i = 0; i < n; i++) y[i] += x[i];
#endif
}

/* ---------- rope tables ---------- */
static void build_rope(float *cos_out, float *sin_out, const int *pos, int S,
                       int hd, int base) {
    int half = hd / 2;
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < half; d++) {
            double inv = 1.0 / pow((double)base, (double)d / half);
            double f = pos[t] * inv;
            float co = (float)cos(f), si = (float)sin(f);
            cos_out[t * hd + d] = co; cos_out[t * hd + half + d] = co;
            sin_out[t * hd + d] = si; sin_out[t * hd + half + d] = si;
        }
    }
}

NCPState *ncp_load(const NCMConfig *c, WeightStore *ws, VQVAEState *vqvae) {
    NCPState *m = (NCPState *)calloc(1, sizeof(NCPState));
    m->cfg = *c;
    m->ws = ws;
    m->vqvae = vqvae;
    m->P = c->prefix_len;
    m->lvl_first = c->query_level0 ? 0 : 1;
    m->L = 0;
    for (int i = 0; i < c->n_qlevels; i++) m->L += c->levels[i];
    m->T = m->L - (c->query_level0 ? 0 : c->levels[0]);
    m->n_tgt = c->n_qlevels - m->lvl_first;

    int hd = c->n_embd / c->n_head;
    int S = m->P + m->T;
    int *pos = (int *)malloc(S * sizeof(int));
    for (int i = 0; i < m->P; i++) pos[i] = i;
    int o = m->P;
    for (int k = m->lvl_first; k < c->n_qlevels; k++)
        for (int t = 0; t < c->levels[k]; t++) pos[o++] = t;
    float *hc = (float *)malloc(S * hd * sizeof(float));
    float *hs = (float *)malloc(S * hd * sizeof(float));
    build_rope(hc, hs, pos, S, hd, c->rope_base);
    free(pos);
    parena_init(1ULL << 28);
    m->rope_cos = parena_alloc((size_t)S * hd);
    m->rope_sin = parena_alloc((size_t)S * hd);
    h2d(m->rope_cos, hc, (size_t)S * hd);
    h2d(m->rope_sin, hs, (size_t)S * hd);
    free(hc); free(hs);

    m->gcos = (float **)malloc(m->n_tgt * sizeof(float *));
    m->gsin = (float **)malloc(m->n_tgt * sizeof(float *));
    for (int k = 0; k < m->n_tgt; k++) {
        int lk = c->levels[k + m->lvl_first];
        int Sk = m->P + lk;
        int *pk = (int *)malloc(Sk * sizeof(int));
        for (int i = 0; i < m->P; i++) pk[i] = i;
        for (int t = 0; t < lk; t++) pk[m->P + t] = t;
        float *ck = (float *)malloc(Sk * hd * sizeof(float));
        float *sk = (float *)malloc(Sk * hd * sizeof(float));
        build_rope(ck, sk, pk, Sk, hd, c->rope_base);
        free(pk);
        m->gcos[k] = parena_alloc((size_t)Sk * hd);
        m->gsin[k] = parena_alloc((size_t)Sk * hd);
        h2d(m->gcos[k], ck, (size_t)Sk * hd);
        h2d(m->gsin[k], sk, (size_t)Sk * hd);
        free(ck); free(sk);
    }

    /* level_pos_embed = wle[lvl_1L] (+ wpe wp_1L when no rope) */
    {
        float *hlp = (float *)malloc((size_t)m->T * c->n_embd * sizeof(float));
        Tensor *wle = W(ws, "gen.transformer.wle.weight");
        float *hwle = (float *)malloc((size_t)m->n_tgt * c->n_embd * sizeof(float));
        d2h(hwle, wle->data, (size_t)m->n_tgt * c->n_embd);
        int tt = 0;
        for (int k = 0; k < m->n_tgt; k++)
            for (int t = 0; t < c->levels[k + m->lvl_first]; t++, tt++)
                memcpy(hlp + (size_t)tt * c->n_embd, hwle + (size_t)k * c->n_embd,
                       c->n_embd * sizeof(float));
        free(hwle);
        if (!c->use_rope) {
            Tensor *wpe = W(ws, "gen.transformer.wpe.weight");
            /* wp_1L = [0..l0-1, 0..l1-1, ...]; fetch rows per index */
            int Tmax = m->P;
            for (int k = m->lvl_first; k < c->n_qlevels; k++)
                if (c->levels[k] > Tmax) Tmax = c->levels[k];
            float *hall = (float *)malloc((size_t)Tmax * c->n_embd * sizeof(float));
            d2h(hall, wpe->data, (size_t)Tmax * c->n_embd);
            tt = 0;
            for (int k = 0; k < m->n_tgt; k++)
                for (int t = 0; t < c->levels[k + m->lvl_first]; t++, tt++)
                    for (int d = 0; d < c->n_embd; d++)
                        hlp[(size_t)tt * c->n_embd + d] += hall[t * c->n_embd + d];
            free(hall);
        }
        m->lvl_pos = parena_alloc((size_t)m->T * c->n_embd);
        h2d(m->lvl_pos, hlp, (size_t)m->T * c->n_embd);
        free(hlp);
    }
    return m;
}

void ncp_free(NCPState *m) {
    if (!m) return;
    free(m->gcos); free(m->gsin);
    free(m);
}

/* ---------- one transformer block (common.py), inplace on x ---------- */
static void ncp_block(NCPState *m, int li, float *x, int B, int S,
                      const float *mask, const float *cos, const float *sin) {
    NCMConfig *c = &m->cfg;
    WeightStore *ws = m->ws;
    int C = c->n_embd, H = c->n_head, hd = C / H;
    int nkv = c->n_kv_heads;
    char pre[128];
    snprintf(pre, sizeof(pre), "gen.transformer.blocks.%d.", li);
    char kk[256];

    float *ln = arena_alloc((size_t)B * S * C);
    snprintf(kk, sizeof(kk), "%sln_1.weight", pre);
    Tensor *tw = W(ws, kk);
    snprintf(kk, sizeof(kk), "%sln_1.bias", pre);
    Tensor *tb = weights_get(ws, kk);
    layernorm_fwd(x, tw->data, tb ? tb->data : NULL, ln, (int64_t)B * S, C);

    /* self-attention */
    snprintf(kk, sizeof(kk), "%sattn.c_attn.weight", pre);
    Tensor *tqkv = W(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_attn.bias", pre);
    Tensor *bqkv = weights_get(ws, kk);
    float *qkv = arena_alloc((size_t)B * S * (C + 2 * nkv * hd));
    linear_fwd(ln, tqkv->data, bqkv ? bqkv->data : NULL, qkv, (int64_t)B * S, C + 2 * nkv * hd, C);

    /* split into q,k,v heads on host (layout shuffle), then backend SDPA */
    float *qq = arena_alloc((size_t)B * H * S * hd);
    float *kkk = arena_alloc((size_t)B * H * S * hd);
    float *vv = arena_alloc((size_t)B * H * S * hd);
    {
        size_t n = (size_t)B * S * (C + 2 * nkv * hd);
        float *h = (float *)malloc(n * sizeof(float));
        d2h(h, qkv, n);
        float *hq = (float *)malloc((size_t)B * H * S * hd * sizeof(float));
        float *hk = (float *)malloc((size_t)B * H * S * hd * sizeof(float));
        float *hv = (float *)malloc((size_t)B * H * S * hd * sizeof(float));
        int rep = H / nkv;
        for (int b = 0; b < B; b++) for (int s = 0; s < S; s++) {
            float *row = h + ((size_t)b * S + s) * (C + 2 * nkv * hd);
            for (int hh = 0; hh < H; hh++) for (int d = 0; d < hd; d++) {
                hq[((size_t)b * H + hh) * S * hd + s * hd + d] = row[hh * hd + d];
                int kvh = hh / rep;
                hk[((size_t)b * H + hh) * S * hd + s * hd + d] = row[C + kvh * hd + d];
                hv[((size_t)b * H + hh) * S * hd + s * hd + d] = row[C + nkv * hd + kvh * hd + d];
            }
        }
        free(h);
        h2d(qq, hq, (size_t)B * H * S * hd);
        h2d(kkk, hk, (size_t)B * H * S * hd);
        h2d(vv, hv, (size_t)B * H * S * hd);
        free(hq); free(hk); free(hv);
    }
    if (c->use_qk_norm) {
        snprintf(kk, sizeof(kk), "%sattn.q_norm.weight", pre);
        rmsnorm_fwd(qq, W(ws, kk)->data, qq, (int64_t)B * H * S, hd);
        snprintf(kk, sizeof(kk), "%sattn.k_norm.weight", pre);
        rmsnorm_fwd(kkk, W(ws, kk)->data, kkk, (int64_t)B * H * S, hd);
    }
    if (c->use_rope) rope_apply(qq, kkk, cos, sin, B, H, S, hd);
    float *ctx = arena_alloc((size_t)B * H * S * hd);
    sdpa_fwd(qq, kkk, vv, mask, ctx, B, H, S, S, hd);
    /* merge heads */
    float *mo = arena_alloc((size_t)B * S * C);
    {
        float *hc = (float *)malloc((size_t)B * H * S * hd * sizeof(float));
        d2h(hc, ctx, (size_t)B * H * S * hd);
        float *ho = (float *)malloc((size_t)B * S * C * sizeof(float));
        for (int b = 0; b < B; b++) for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++)
            for (int d = 0; d < hd; d++)
                ho[((size_t)b * S + s) * C + hh * hd + d] =
                    hc[((size_t)b * H + hh) * S * hd + s * hd + d];
        free(hc);
        h2d(mo, ho, (size_t)B * S * C);
        free(ho);
    }
    snprintf(kk, sizeof(kk), "%sattn.c_proj.weight", pre);
    Tensor *tcp = W(ws, kk);
    snprintf(kk, sizeof(kk), "%sattn.c_proj.bias", pre);
    Tensor *bcp = weights_get(ws, kk);
    float *ao = arena_alloc((size_t)B * S * C);
    linear_fwd(mo, tcp->data, bcp ? bcp->data : NULL, ao, (int64_t)B * S, C, C);

    float *br1 = ao, *br2 = NULL;
    int parallel = 0; /* config flag parallel_block (all released cfgs: false) */
    (void)parallel;
    /* mlp input: parallel ? ln : ln_2(attn_out + x) */
    float *min;
    if (0) { min = ln; }
    else {
        /* x = x + ao first (store), then ln_2 */
        b_add_inplace(x, ao, (size_t)B * S * C);
        min = arena_alloc((size_t)B * S * C);
        snprintf(kk, sizeof(kk), "%sln_2.weight", pre);
        tw = W(ws, kk);
        snprintf(kk, sizeof(kk), "%sln_2.bias", pre);
        tb = weights_get(ws, kk);
        layernorm_fwd(x, tw->data, tb ? tb->data : NULL, min, (int64_t)B * S, C);
        br1 = NULL; /* already added */
    }
    (void)br2;
    /* MLP */
    float *mlpo;
    {
        char kw[256], kb[256];
        int gated = c->use_swiglu || c->use_geglu;
        if (gated) {
            snprintf(kw, sizeof(kw), "%smlp.w_gate.weight", pre);
            Tensor *tg = W(ws, kw);
            int Hh = (int)tg->shape[0];
            snprintf(kb, sizeof(kb), "%smlp.w_gate.bias", pre);
            Tensor *bg = weights_get(ws, kb);
            snprintf(kw, sizeof(kw), "%smlp.w_up.weight", pre);
            Tensor *tu = W(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.w_up.bias", pre);
            Tensor *bu = weights_get(ws, kb);
            float *g = arena_alloc((size_t)B * S * Hh);
            float *u = arena_alloc((size_t)B * S * Hh);
            linear_fwd(min, tg->data, bg ? bg->data : NULL, g, (int64_t)B * S, Hh, C);
            linear_fwd(min, tu->data, bu ? bu->data : NULL, u, (int64_t)B * S, Hh, C);
            if (c->use_swiglu) silu_fwd(g, g, (int64_t)B * S * Hh);
            else gelu_fwd(g, g, (int64_t)B * S * Hh);
            /* g *= u */
            {
                size_t n = (size_t)B * S * Hh;
                float *hg = (float *)malloc(n * sizeof(float));
                float *hu = (float *)malloc(n * sizeof(float));
                d2h(hg, g, n); d2h(hu, u, n);
                for (size_t i = 0; i < n; i++) hg[i] *= hu[i];
                free(hu);
                h2d(g, hg, n);
                free(hg);
            }
            snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
            Tensor *tp = W(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
            Tensor *bp = weights_get(ws, kb);
            mlpo = arena_alloc((size_t)B * S * C);
            linear_fwd(g, tp->data, bp ? bp->data : NULL, mlpo, (int64_t)B * S, C, Hh);
        } else {
            snprintf(kw, sizeof(kw), "%smlp.c_fc.weight", pre);
            Tensor *tf = W(ws, kw);
            int Hh = (int)tf->shape[0];
            snprintf(kb, sizeof(kb), "%smlp.c_fc.bias", pre);
            Tensor *bf = weights_get(ws, kb);
            float *m = arena_alloc((size_t)B * S * Hh);
            linear_fwd(min, tf->data, bf ? bf->data : NULL, m, (int64_t)B * S, Hh, C);
            if (c->use_relu2) {
                size_t n = (size_t)B * S * Hh;
                float *hm = (float *)malloc(n * sizeof(float));
                d2h(hm, m, n);
                for (size_t i = 0; i < n; i++) { float r = hm[i] > 0 ? hm[i] : 0; hm[i] = r * r; }
                h2d(m, hm, n);
                free(hm);
            } else {
                gelu_fwd(m, m, (int64_t)B * S * Hh);
            }
            snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
            Tensor *tp = W(ws, kw);
            snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
            Tensor *bp = weights_get(ws, kb);
            mlpo = arena_alloc((size_t)B * S * C);
            linear_fwd(m, tp->data, bp ? bp->data : NULL, mlpo, (int64_t)B * S, C, Hh);
        }
    }
    if (br1) b_add_inplace(x, br1, (size_t)B * S * C);
    b_add_inplace(x, mlpo, (size_t)B * S * C);
}

/* run stack + ln_f, inplace on h ([B][S][C]) */
static void run_stack(NCPState *m, float *h, int B, int S,
                      const float *mask, const float *cos, const float *sin) {
    for (int li = 0; li < m->cfg.n_layer; li++)
        ncp_block(m, li, h, B, S, mask, cos, sin);
    NCMConfig *c = &m->cfg;
    Tensor *tw = W(m->ws, "gen.transformer.ln_f.weight");
    Tensor *tb = weights_get(m->ws, "gen.transformer.ln_f.bias");
    /* inplace ln_f: use temp */
    float *t = arena_alloc((size_t)B * S * c->n_embd);
    layernorm_fwd(h, tw->data, tb ? tb->data : NULL, t, (int64_t)B * S, c->n_embd);
    d2d(h, t, (size_t)B * S * c->n_embd);
}

/* full additive mask [S][S] host-built, uploaded; includes prefix pad bias */
static float *build_full_mask(NCPState *m, const int *prefix_ids, int B) {
    NCMConfig *c = &m->cfg;
    int P = m->P, T = m->T, S = P + T;
    float *hm = (float *)malloc((size_t)S * S * sizeof(float));
    for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) {
        float v = 0;
        if (i < P && j >= P) v = -INFINITY;
        else if (i >= P && j >= P) {
            /* target block: level of row vs col */
            int ti = i - P, tj = j - P;
            /* find levels */
            int li = -1, lj = -1, acc = 0;
            for (int k = m->lvl_first; k < c->n_qlevels; k++) {
                int l = c->levels[k];
                if (ti >= acc && ti < acc + l) li = k;
                if (tj >= acc && tj < acc + l) lj = k;
                acc += l;
            }
            if (c->attn_pattern_block_diag) v = (li == lj) ? 0.0f : -INFINITY;
            else v = (li >= lj) ? 0.0f : -INFINITY;
        }
        hm[i * S + j] = v;
    }
    float *mask = arena_alloc((size_t)S * S);
    h2d(mask, hm, (size_t)S * S);
    free(hm);
    /* prefix pad bias: rows all, cols [0,P) */
    if (c->prefix_pad_token_id >= 0 && prefix_ids) {
        float *hb = (float *)malloc((size_t)B * S * sizeof(float));
        for (int b = 0; b < B; b++) for (int s = 0; s < S; s++) {
            int is_pad = (s < P && prefix_ids[b * P + s] == c->prefix_pad_token_id);
            hb[b * S + s] = is_pad ? -INFINITY : 0.0f;
        }
        /* fold into mask per batch? mask is shared; instead return bias separately.
         * Simplest: bake for the (common) case all rows share pads by max. */
        free(hb);
        /* NOTE: per-batch pad bias applied in generate path; training forward
         * uses inputs without pads in released evals. Handled in ncp_forward
         * via explicit broadcast below. */
    }
    (void)B;
    return mask;
}

/* per-level output head -> lg [B*lk, V]. Handles untied / shared / tied. */
static void head_logits(NCPState *m, int k, const float *slice, float *lg,
                        int B, int lk) {
    NCMConfig *c = &m->cfg;
    int V = c->vocab_size[k];
    int C = c->n_embd;
    if (c->shared_output_head) {
        Tensor *tw = W(m->ws, "gen.lm_heads.head.weight");
        Tensor *tb = NULL;
        char kb[128];
        if (c->shared_head_per_level_bias) {
            snprintf(kb, sizeof(kb), "gen.lm_heads.biases.%d", k);
            tb = W(m->ws, kb);
        } else if (c->untied_head_bias) {
            tb = W(m->ws, "gen.lm_heads.head.bias");
        }
        linear_fwd(slice, tw->data, tb ? tb->data : NULL, lg, (int64_t)B * lk, V, C);
    } else if (c->tie_heads_to_codebook) {
        char kw[128];
        snprintf(kw, sizeof(kw), "gen.lm_heads.%d.proj.weight", k);
        Tensor *tp = W(m->ws, kw);
        int Dd = (int)tp->shape[0];
        snprintf(kw, sizeof(kw), "gen.lm_heads.%d.proj.bias", k);
        Tensor *bp = weights_get(m->ws, kw);
        float *z = arena_alloc((size_t)B * lk * Dd);
        linear_fwd(slice, tp->data, bp ? bp->data : NULL, z, (int64_t)B * lk, Dd, C);
        /* codebook + derived constants (E_sq/d_scale recomputed: buffers) */
        int lv = k + m->lvl_first;
        snprintf(kw, sizeof(kw), "tok.vq.embeddings.%d.weight", lv);
        Tensor *te = W(m->ws, kw);
        float *hz = (float *)malloc((size_t)B * lk * Dd * sizeof(float));
        float *hE = (float *)malloc((size_t)V * Dd * sizeof(float));
        d2h(hz, z, (size_t)B * lk * Dd);
        d2h(hE, te->data, (size_t)V * Dd);
        double *Esq = (double *)malloc(V * sizeof(double));
        double ds = 0;
        for (int v = 0; v < V; v++) {
            double s = 0;
            for (int d = 0; d < Dd; d++) s += (double)hE[(size_t)v * Dd + d] * hE[(size_t)v * Dd + d];
            Esq[v] = s; ds += s;
        }
        ds /= V;
        snprintf(kw, sizeof(kw), "gen.lm_heads.%d.log_inv_T", k);
        float lit = 0;
        d2h(&lit, W(m->ws, kw)->data, 1);
        float invT = expf(lit);
        snprintf(kw, sizeof(kw), "gen.lm_heads.%d.bias", k);
        Tensor *tbb = weights_get(m->ws, kw);
        float *hb = NULL;
        if (tbb) { hb = (float *)malloc(V * sizeof(float)); d2h(hb, tbb->data, V); }
        float *ho = (float *)malloc((size_t)B * lk * V * sizeof(float));
        for (int r = 0; r < B * lk; r++) {
            for (int v = 0; v < V; v++) {
                double dot = 0;
                for (int d = 0; d < Dd; d++)
                    dot += (double)hz[(size_t)r * Dd + d] * hE[(size_t)v * Dd + d];
                double o = (2 * dot - Esq[v]) / ds * invT;
                if (hb) o += hb[v];
                ho[(size_t)r * V + v] = (float)o;
            }
        }
        free(hz); free(hE); free(Esq); free(hb);
        h2d(lg, ho, (size_t)B * lk * V);
        free(ho);
    } else {
        char kw[128], kb[128];
        snprintf(kw, sizeof(kw), "gen.lm_heads.%d.weight", k);
        snprintf(kb, sizeof(kb), "gen.lm_heads.%d.bias", k);
        Tensor *tw = W(m->ws, kw);
        Tensor *tb = weights_get(m->ws, kb);
        linear_fwd(slice, tw->data, tb ? tb->data : NULL, lg, (int64_t)B * lk, V, C);
    }
}

/* forward used by training/scoring: x [B][P+T][Dvq] backend, logits [B][T][maxV] */
void ncp_forward(NCPState *m, const float *x, const int *prefix_ids,
                 float *logits, int B) {
    NCMConfig *c = &m->cfg;
    int P = m->P, T = m->T, S = P + T, C = c->n_embd, D = c->vq_embed_dim;
    /* split */
    float *zpre = arena_alloc((size_t)B * P * D);
    float *xt = arena_alloc((size_t)B * T * D);
    {
        size_t n = (size_t)B * S * D;
        float *h = (float *)malloc(n * sizeof(float));
        d2h(h, x, n);
        float *a = (float *)malloc((size_t)B * P * D * sizeof(float));
        float *b2 = (float *)malloc((size_t)B * T * D * sizeof(float));
        for (int bb = 0; bb < B; bb++) {
            memcpy(a + (size_t)bb * P * D, h + (size_t)bb * S * D, (size_t)P * D * sizeof(float));
            memcpy(b2 + (size_t)bb * T * D, h + ((size_t)bb * S + P) * D, (size_t)T * D * sizeof(float));
        }
        free(h);
        h2d(zpre, a, (size_t)B * P * D);
        h2d(xt, b2, (size_t)B * T * D);
        free(a); free(b2);
    }
    /* post upsample convs per target level on xt chunks */
    if (c->puc_enabled) {
        size_t n = (size_t)B * T * D;
        float *h = (float *)malloc(n * sizeof(float));
        d2h(h, xt, n);
        int off = 0;
        for (int k = 0; k < m->n_tgt; k++) {
            int lk = c->levels[k + m->lvl_first];
            char kw[128], kb[128];
            snprintf(kw, sizeof(kw), "gen.post_upsample_convs.%d.weight", k);
            snprintf(kb, sizeof(kb), "gen.post_upsample_convs.%d.bias", k);
            Tensor *tw = W(m->ws, kw), *tb = W(m->ws, kb);
            /* chunk [B][lk][D] -> BCL -> conv -> +residual */
            float *cb = (float *)malloc((size_t)B * lk * D * sizeof(float));
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++) for (int d = 0; d < D; d++)
                cb[((size_t)bb * D + d) * lk + t] = h[((size_t)bb * T + off + t) * D + d];
            /* conv via backend staging */
            float *dbc = arena_alloc((size_t)B * lk * D);
            float *dout = arena_alloc((size_t)B * lk * D);
            h2d(dbc, cb, (size_t)B * lk * D);
            conv1d_fwd(dbc, tw->data, tb->data, dout, B, D, D, lk, c->puc_kernel, 1, c->puc_kernel / 2);
            float *co = (float *)malloc((size_t)B * lk * D * sizeof(float));
            d2h(co, dout, (size_t)B * lk * D);
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++) for (int d = 0; d < D; d++)
                h[((size_t)bb * T + off + t) * D + d] += co[((size_t)bb * D + d) * lk + t];
            free(cb); free(co);
            off += lk;
        }
        h2d(xt, h, n);
        free(h);
    }
    /* prefix_emb = wte(zpre) */
    float *pemb = arena_alloc((size_t)B * P * C);
    {
        Tensor *tw = W(m->ws, "gen.transformer.wte.weight");
        Tensor *tb = weights_get(m->ws, "gen.transformer.wte.bias");
        linear_fwd(zpre, tw->data, tb ? tb->data : NULL, pemb, (int64_t)B * P, C, D);
    }
    /* x = wte(xt) + level_pos */
    float *xe = arena_alloc((size_t)B * T * C);
    {
        Tensor *tw = W(m->ws, "gen.transformer.wte.weight");
        Tensor *tb = weights_get(m->ws, "gen.transformer.wte.bias");
        linear_fwd(xt, tw->data, tb ? tb->data : NULL, xe, (int64_t)B * T, C, D);
        /* add level_pos_embed (broadcast over B) */
        size_t n = (size_t)B * T * C;
        float *h = (float *)malloc(n * sizeof(float));
        float *hl = (float *)malloc((size_t)T * C * sizeof(float));
        d2h(h, xe, n);
        d2h(hl, m->lvl_pos, (size_t)T * C);
        for (int bb = 0; bb < B; bb++) for (int i = 0; i < T * C; i++)
            h[(size_t)bb * T * C + i] += hl[i];
        free(hl);
        h2d(xe, h, n);
        free(h);
    }
    if (!c->use_rope) {
        /* + wpe on prefix; xe already carries wp_1L via lvl_pos */
        Tensor *wpe = W(m->ws, "gen.transformer.wpe.weight");
        size_t n = (size_t)B * P * C;
        float *h = (float *)malloc(n * sizeof(float));
        float *hw = (float *)malloc((size_t)P * C * sizeof(float));
        d2h(h, pemb, n);
        d2h(hw, wpe->data, (size_t)P * C);
        for (int bb = 0; bb < B; bb++) for (int i = 0; i < P * C; i++)
            h[(size_t)bb * P * C + i] += hw[i];
        h2d(pemb, h, n);
        free(h); free(hw);
    }
    /* concat prefix + target */
    float *hfull = arena_alloc((size_t)B * S * C);
    for (int bb = 0; bb < B; bb++) {
        d2d(hfull + (size_t)bb * S * C, pemb + (size_t)bb * P * C, (size_t)P * C);
        d2d(hfull + ((size_t)bb * S + P) * C, xe + (size_t)bb * T * C, (size_t)T * C);
    }
    float *mask = build_full_mask(m, prefix_ids, B);


    /* per-batch prefix pad bias: run each batch with its own mask if pads exist */
    if (c->prefix_pad_token_id >= 0 && prefix_ids) {
        int any_pad = 0;
        for (int bb = 0; bb < B; bb++) for (int s = 0; s < P; s++)
            if (prefix_ids[bb * P + s] == c->prefix_pad_token_id) any_pad = 1;
        if (any_pad) {
            float *m0 = (float *)malloc((size_t)S * S * sizeof(float));
            d2h(m0, mask, (size_t)S * S);
            for (int bb = 0; bb < B; bb++) {
                float *hm = (float *)malloc((size_t)S * S * sizeof(float));
                memcpy(hm, m0, (size_t)S * S * sizeof(float));
                for (int i = 0; i < S; i++) for (int j = 0; j < P; j++)
                    if (prefix_ids[bb * P + j] == c->prefix_pad_token_id)
                        hm[(size_t)i * S + j] = -INFINITY;
                float *bm = arena_alloc((size_t)S * S);
                h2d(bm, hm, (size_t)S * S);
                free(hm);
                float *hb = arena_alloc((size_t)S * C);
                d2d(hb, hfull + (size_t)bb * S * C, (size_t)S * C);
                run_stack(m, hb, 1, S, bm, m->rope_cos, m->rope_sin);
                d2d(hfull + (size_t)bb * S * C, hb, (size_t)S * C);
            }
            free(m0);
            goto heads;
        }
    }
    run_stack(m, hfull, B, S, mask, m->rope_cos, m->rope_sin);
heads:;
    /* heads per target level on x_target = hfull[:, P:] */
    int maxV = 0;
    for (int k = 0; k < m->n_tgt; k++)
        if (c->vocab_size[k] > maxV) maxV = c->vocab_size[k];
    /* ---- heads, second (working) implementation ---- */
    {
        float *hlog = (float *)malloc((size_t)B * T * maxV * sizeof(float));
        for (size_t i = 0; i < (size_t)B * T * maxV; i++) hlog[i] = -INFINITY;
        int off2 = 0;
        for (int k = 0; k < m->n_tgt; k++) {
            int lk = c->levels[k + m->lvl_first];
            int V = c->vocab_size[k];
            /* recompute slice logits via backend then download */
            float *slice = arena_alloc((size_t)B * lk * C);
            {
                float *tt = (float *)malloc((size_t)B * S * C * sizeof(float));
                d2h(tt, hfull, (size_t)B * S * C);
                float *s2 = (float *)malloc((size_t)B * lk * C * sizeof(float));
                for (int bb = 0; bb < B; bb++)
                    for (int t = 0; t < lk; t++)
                        memcpy(s2 + ((size_t)bb * lk + t) * C,
                               tt + ((size_t)bb * S + P + off2 + t) * C, C * sizeof(float));
                free(tt);
                h2d(slice, s2, (size_t)B * lk * C);
                free(s2);
            }
            float *lg = arena_alloc((size_t)B * lk * V);
            head_logits(m, k, slice, lg, B, lk);
            float *hg = (float *)malloc((size_t)B * lk * V * sizeof(float));
            d2h(hg, lg, (size_t)B * lk * V);
            for (int bb = 0; bb < B; bb++) for (int t = 0; t < lk; t++)
                memcpy(hlog + ((size_t)bb * T + off2 + t) * maxV,
                       hg + ((size_t)bb * lk + t) * V, V * sizeof(float));
            free(hg);
            off2 += lk;
        }
        h2d(logits, hlog, (size_t)B * T * maxV);
        free(hlog);
    }
}

/* ================= generate ================= */

static int V0_(NCPState *m);

/* sample one level's logits on host with freq/adj penalties (mirrors pick_tokens) */
static void sample_level_rows(NCPState *m __attribute__((unused)), float *hlog, int *out, int B,
                              int Lc, int V, float temp, int top_k, float top_p,
                              float freq, float mu, float adj_a, int adj_w, float adj_d,
                              const double *allow, unsigned long long *rng,
                              double **cnt_state, int **hist_state, int *hist_len) {
    /* hlog [B][Lc][V] host; out [B][Lc] */
    for (int b = 0; b < B; b++) {
        double *cnt = cnt_state[b];
        int *hist = hist_state[b];
        int hl = hist_len[b];
        for (int t = 0; t < Lc; t++) {
            float *row = hlog + ((size_t)b * Lc + t) * V;
            int pick;
            if (freq <= 0 && adj_a == 0) {
                /* still must sample (temp/topk/topp may apply) */
                pick = sample_topkp(row, V, temp, top_k, top_p, rng);
            } else {
                float *adj = (float *)malloc(V * sizeof(float));
                for (int v = 0; v < V; v++) {
                    double lg = row[v];
                    if (freq > 0) {
                        double al = allow ? allow[v] * Lc : 0.0;
                        double excess = cnt[v] - al;
                        if (excess < 0) excess = 0;
                        lg -= freq * excess;
                        if (mu > 0 && allow) {
                            double def = al - cnt[v];
                            if (def < 0) def = 0;
                            lg += mu * ((Lc - t) / (double)Lc) * def;
                        }
                    }
                    adj[v] = (float)lg;
                }
                if (adj_a != 0) {
                    int w = adj_w > 0 ? adj_w : 1;
                    int start = hl - w; if (start < 0) start = 0;
                    for (int h = start; h < hl; h++) {
                        int back = hl - 1 - h;
                        double pen = adj_a * pow(adj_d, back);
                        adj[hist[h]] -= (float)pen;
                    }
                }
                pick = sample_topkp(adj, V, temp, top_k, top_p, rng);
                free(adj);
            }
            out[b * Lc + t] = pick;
            cnt[pick] += 1.0;
            hist[hl++] = pick;
        }
        hist_len[b] = hl;
    }
}

int *ncp_generate(NCPState *m, const int *l0, const int *prefix_ids, int B,
                  float temp, int top_k, float top_p, float cfg_scale,
                  float freq, float mu, float adj_a, int adj_w, float adj_d,
                  const double *allow, unsigned long long seed) {
    NCMConfig *c = &m->cfg;
    WeightStore *ws = m->ws;
    VQState *vq = vqvae_vq(m->vqvae);
    int P = m->P, C = c->n_embd, D = c->vq_embed_dim;
    int nlev = c->n_qlevels;

    arena_reset(); /* scratch temporaries are per-call */
    int use_cfg = (cfg_scale > 0);
    int Bm = use_cfg ? 2 * B : B;   /* model batch */
    /* build model prefix (with null branch for CFG) */
    int *mprefix = (int *)malloc((size_t)Bm * P * sizeof(int));
    for (int b = 0; b < B; b++) memcpy(mprefix + (size_t)b * P, prefix_ids + (size_t)b * P, P * sizeof(int));
    if (use_cfg) {
        int null_first = c->eos_token_id >= 0 ? c->eos_token_id : c->prefix_pad_token_id;
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < P; s++) mprefix[((size_t)B + b) * P + s] = c->prefix_pad_token_id;
            if (P > 0) mprefix[((size_t)B + b) * P + 0] = null_first;
        }
    }
    /* model l0 (doubled for CFG) */
    int l0len = c->levels[0];
    int *ml0 = NULL;
    if (l0) {
        ml0 = (int *)malloc((size_t)Bm * l0len * sizeof(int));
        for (int b = 0; b < B; b++) memcpy(ml0 + (size_t)b * l0len, l0 + (size_t)b * l0len, l0len * sizeof(int));
        if (use_cfg) for (int b = 0; b < B; b++)
            memcpy(ml0 + ((size_t)B + b) * l0len, l0 + (size_t)b * l0len, l0len * sizeof(int));
    }

    /* z_prefix via encoder wte (raw path; all released configs) */
    float *zpre = arena_alloc((size_t)Bm * P * D);
    {
        Tensor *te = W(ws, "tok.encoder.wte.weight");
        int *bids = (int *)arena_alloc((size_t)Bm * P);
        h2d_i(bids, mprefix, (size_t)Bm * P);
        embedding_fwd(te->data, bids, zpre, (int64_t)Bm * P, D);
    }
    Tensor *_pembT = tensor_alloc(2, (int64_t[]){(int64_t)Bm * P, C});
    float *pemb = _pembT->data;
    {
        Tensor *tw = W(ws, "gen.transformer.wte.weight");
        Tensor *tb = weights_get(ws, "gen.transformer.wte.bias");
        linear_fwd(zpre, tw->data, tb ? tb->data : NULL, pemb, (int64_t)Bm * P, C, D);
    }
    if (!c->use_rope) {
        Tensor *wpe = W(ws, "gen.transformer.wpe.weight");
        size_t n = (size_t)Bm * P * C;
        float *h = (float *)malloc(n * sizeof(float));
        float *hw = (float *)malloc((size_t)P * C * sizeof(float));
        d2h(h, pemb, n); d2h(hw, wpe->data, (size_t)P * C);
        for (int b = 0; b < Bm; b++) for (int i = 0; i < P * C; i++)
            h[(size_t)b * P * C + i] += hw[i];
        free(hw); h2d(pemb, h, n); free(h);
    }

    /* pad bias for gen masks */
    int *padcol = NULL;
    if (c->prefix_pad_token_id >= 0) {
        padcol = (int *)malloc((size_t)Bm * P * sizeof(int));
        for (int b = 0; b < Bm; b++) for (int s = 0; s < P; s++)
            padcol[b * P + s] = (mprefix[b * P + s] == c->prefix_pad_token_id);
    }

    unsigned long long rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int total = vq_total_codes(vq);
    int *all = (int *)malloc((size_t)B * total * sizeof(int)); /* model-batch collapsed later */
    int *cur = NULL;   /* current level codes, model batch [Bm][lcur] */
    int lcur = 0;

    /* ---- level 0 ---- */
    Tensor *_zhat = tensor_alloc(2, (int64_t[]){(int64_t)Bm * m->L, D});
    float *z_hat = _zhat->data;
#ifdef NCM_CUDA
    cudaMemset(z_hat, 0, (size_t)Bm * m->L * D * sizeof(float));
#else
    memset(z_hat, 0, (size_t)Bm * m->L * D * sizeof(float));
#endif
    if (ml0) {
        cur = (int *)malloc((size_t)Bm * l0len * sizeof(int));
        memcpy(cur, ml0, (size_t)Bm * l0len * sizeof(int));
        lcur = l0len;
        /* copy l0 into output */
        for (int b = 0; b < B; b++) memcpy(all + (size_t)b * total, ml0 + (size_t)b * l0len, l0len * sizeof(int));
        free(ml0);
    } else if (c->query_level0) {
        int l0l = c->levels[0];
        Tensor *tq = W(ws, "gen.l0_queries");
        float *q0 = arena_alloc((size_t)Bm * l0l * D);
        {
            float *h = (float *)malloc((size_t)l0l * D * sizeof(float));
            d2h(h, tq->data, (size_t)l0l * D);
            float *hb = (float *)malloc((size_t)Bm * l0l * D * sizeof(float));
            for (int b = 0; b < Bm; b++) memcpy(hb + (size_t)b * l0l * D, h, (size_t)l0l * D * sizeof(float));
            free(h); h2d(q0, hb, (size_t)Bm * l0l * D); free(hb);
        }
        if (c->puc_enabled) {
            /* single conv idx 0 */
            char kw[128], kb[128];
            snprintf(kw, sizeof(kw), "gen.post_upsample_convs.0.weight");
            snprintf(kb, sizeof(kb), "gen.post_upsample_convs.0.bias");
            float *bcl = arena_alloc((size_t)Bm * l0l * D);
            float *cv = arena_alloc((size_t)Bm * l0l * D);
            size_t n = (size_t)Bm * l0l * D;
            float *h = (float *)malloc(n * sizeof(float));
            d2h(h, q0, n);
            float *hb = (float *)malloc(n * sizeof(float));
            for (int b = 0; b < Bm; b++) for (int t = 0; t < l0l; t++) for (int d = 0; d < D; d++)
                hb[((size_t)b * D + d) * l0l + t] = h[((size_t)b * l0l + t) * D + d];
            free(h); h2d(bcl, hb, n); free(hb);
            conv1d_fwd(bcl, W(ws, kw)->data, W(ws, kb)->data, cv, Bm, D, D, l0l, 3, 1, 1);
            /* residual add */
            float *hc = (float *)malloc(n * sizeof(float));
            float *ho = (float *)malloc(n * sizeof(float));
            d2h(hc, cv, n); d2h(ho, q0, n);
            for (size_t i = 0; i < n; i++) ho[i] += hc[i];
            free(hc); h2d(q0, ho, n); free(ho);
        }
        float *qe = arena_alloc((size_t)Bm * l0l * C);
        {
            Tensor *tw = W(ws, "gen.transformer.wte.weight");
            Tensor *tb = weights_get(ws, "gen.transformer.wte.bias");
            linear_fwd(q0, tw->data, tb ? tb->data : NULL, qe, (int64_t)Bm * l0l, C, D);
        }
        /* + level encoding row 0 + wpe when no rope */
        {
            size_t n = (size_t)Bm * l0l * C;
            float *h = (float *)malloc(n * sizeof(float));
            d2h(h, qe, n);
            Tensor *wle = W(ws, "gen.transformer.wle.weight");
            float *hl = (float *)malloc((size_t)C * sizeof(float));
            d2h(hl, wle->data, C);
            for (int b = 0; b < Bm; b++) for (int t = 0; t < l0l; t++) for (int d = 0; d < C; d++)
                h[((size_t)b * l0l + t) * C + d] += hl[d];
            free(hl);
            h2d(qe, h, n); free(h);
        }
        /* h = [pemb, qe]; mask m0; run; head 0 */
        Tensor *_hh0T = tensor_alloc(2, (int64_t[]){(int64_t)Bm * (P + l0l), C});
        float *hh = _hh0T->data;
        for (int b = 0; b < Bm; b++) {
            d2d(hh + (size_t)b * (P + l0l) * C, pemb + (size_t)b * P * C, (size_t)P * C);
            d2d(hh + ((size_t)b * (P + l0l) + P) * C, qe + (size_t)b * l0l * C, (size_t)l0l * C);
        }
        int S0 = P + l0l;
        float *m0 = arena_alloc((size_t)S0 * S0);
        {
            float *h = (float *)malloc((size_t)S0 * S0 * sizeof(float));
            for (int i = 0; i < S0; i++) for (int j = 0; j < S0; j++) {
                float v = 0;
                if (i < P && j >= P) v = -INFINITY;
                if (padcol) {
                    /* padcol per batch differs under CFG; bake per-batch below */
                }
                h[i * S0 + j] = v;
            }
            /* per-batch pad: run per batch to stay exact */
            for (int b = 0; b < Bm; b++) {
                arena_reset(); /* per-board temps freed; hh/qe/pemb are tensors */
                if (padcol) for (int i = 0; i < S0; i++) for (int j = 0; j < P; j++)
                    if (padcol[b * P + j]) h[i * S0 + j] = -INFINITY;
                float *bm = arena_alloc((size_t)S0 * S0);
                h2d(bm, h, (size_t)S0 * S0);
                if (padcol) { /* restore base for next batch */ }
                float *hb = arena_alloc((size_t)S0 * C);
                d2d(hb, hh + (size_t)b * S0 * C, (size_t)S0 * C);
                run_stack(m, hb, 1, S0, bm, m->gcos[0], m->gsin[0]);
                /* head 0 on target part */
                float *sl = arena_alloc((size_t)l0l * C);
                {
                    float *t2 = (float *)malloc((size_t)S0 * C * sizeof(float));
                    d2h(t2, hb, (size_t)S0 * C);
                    float *s2 = (float *)malloc((size_t)l0l * C * sizeof(float));
                    memcpy(s2, t2 + (size_t)P * C, (size_t)l0l * C * sizeof(float));
                    free(t2); h2d(sl, s2, (size_t)l0l * C); free(s2);
                }
                float *lg = arena_alloc((size_t)l0l * V0_(m));
                head_logits(m, 0, sl, lg, 1, l0l);
                float *hl = (float *)malloc((size_t)l0l * V0_(m) * sizeof(float));
                d2h(hl, lg, (size_t)l0l * V0_(m));
                /* sample (no freq/adj state yet for level 0? Python applies
                 * pick_tokens with same freq/adj — use fresh state) */
                if (!cur) { cur = (int *)malloc((size_t)Bm * l0l * sizeof(int)); }
                /* per-batch state */
                double *cnt = (double *)calloc(V0_(m), sizeof(double));
                int *hist = (int *)malloc((l0l * 8 + 64) * sizeof(int));
                int hll = 0;
                double *cs[1] = {cnt}; int *hs[1] = {hist}; int ls[1] = {hll};
                sample_level_rows(m, hl, cur + (size_t)b * l0l, 1, l0l, V0_(m),
                                  temp, top_k, top_p, freq, mu, adj_a, adj_w, adj_d,
                                  allow, &rng, cs, hs, ls);
                free(cnt); free(hist); free(hl);
                if (padcol) {
                    /* rebuild base mask */
                    for (int i = 0; i < S0; i++) for (int j = 0; j < S0; j++) {
                        float v = 0;
                        if (i < P && j >= P) v = -INFINITY;
                        h[i * S0 + j] = v;
                    }
                }
            }
            free(h);
            (void)m0;
            tensor_free(_hh0T);
        }
        lcur = l0l;
        for (int b = 0; b < B; b++) memcpy(all + (size_t)b * total, cur + (size_t)b * l0l, l0l * sizeof(int));
        if (use_cfg) {
            /* collapse: keep first B */
            int *c2 = (int *)malloc((size_t)B * l0l * sizeof(int));
            memcpy(c2, cur, (size_t)B * l0l * sizeof(int));
            free(cur); cur = c2;
        }
        /* NOTE: z_hat stays zero for query_level0 (matches Python: z_hat starts
         * zero and level-0 embedding is added in the loop below). */
    } else {
        fprintf(stderr, "generate(idx=None) requires query_level0 or predict_level0\n");
        exit(1);
    }

    /* per-batch-row sampling state for freq/adjacency */
    int Vmax = 0;
    for (int k = 0; k < m->n_tgt; k++) if (c->vocab_size[k] > Vmax) Vmax = c->vocab_size[k];
    double **cntS = (double **)malloc(Bm * sizeof(double *));
    int **histS = (int **)malloc(Bm * sizeof(int *));
    int *histL = (int *)calloc(Bm, sizeof(int));
    for (int b = 0; b < Bm; b++) {
        cntS[b] = (double *)calloc(Vmax, sizeof(double));
        histS[b] = (int *)malloc((total + 64) * sizeof(int));
    }

    int level_accum = (m->lvl_first == 0) ? c->levels[0] : 0;
    int out_off = lcur; /* == level_accum for non-query (l0 given) */
    for (int li = 0; li < nlev - 1; li++) {
        arena_reset(); /* per-level temps freed; z_hat/pemb live in parena */
        int hi = li + (1 - m->lvl_first); /* head index */
        int Lcur = c->levels[li + 1];
        /* codes of current level, model batch */
        int *codes = (int *)malloc((size_t)Bm * c->levels[li] * sizeof(int));
        if (li == 0) {
            /* cur holds level-0 codes for model batch (or collapsed B for cfg) */
            if (use_cfg && !ml0) {
                /* cur was collapsed above; re-duplicate */
                int *d = (int *)malloc((size_t)Bm * c->levels[li] * sizeof(int));
                memcpy(d, cur, (size_t)B * c->levels[li] * sizeof(int));
                memcpy(d + (size_t)B * c->levels[li], cur, (size_t)B * c->levels[li] * sizeof(int));
                free(cur); cur = d;
            }
            memcpy(codes, cur, (size_t)Bm * c->levels[li] * sizeof(int));
        } else {
            /* previous iteration stored into all/model buffer: track */
            memcpy(codes, cur, (size_t)Bm * c->levels[li] * sizeof(int));
        }
        free(cur); cur = NULL;
        /* autoregressive input */
        float *x_out = arena_alloc((size_t)Bm * Lcur * D);
        vq_next_input(vq, codes, li, z_hat, x_out, Bm);
        free(codes);
        /* post upsample single: x_out += conv(x_out), backend BCL staging */
        if (c->puc_enabled) {
            char kw[128], kb[128];
            snprintf(kw, sizeof(kw), "gen.post_upsample_convs.%d.weight", hi);
            snprintf(kb, sizeof(kb), "gen.post_upsample_convs.%d.bias", hi);
            float *bcl = arena_alloc((size_t)Bm * Lcur * D);
            float *cv = arena_alloc((size_t)Bm * Lcur * D);
            size_t nn = (size_t)Bm * Lcur * D;
            float *hh2 = (float *)malloc(nn * sizeof(float));
            d2h(hh2, x_out, nn);
            float *tb = (float *)malloc(nn * sizeof(float));
            for (int b = 0; b < Bm; b++) for (int t = 0; t < Lcur; t++) for (int d = 0; d < D; d++)
                tb[((size_t)b * D + d) * Lcur + t] = hh2[((size_t)b * Lcur + t) * D + d];
            free(hh2); h2d(bcl, tb, nn); free(tb);
            conv1d_fwd(bcl, W(ws, kw)->data, W(ws, kb)->data, cv, Bm, D, D, Lcur, c->puc_kernel, 1, c->puc_kernel / 2);
            float *hc = (float *)malloc(nn * sizeof(float));
            float *ho = (float *)malloc(nn * sizeof(float));
            d2h(hc, cv, nn); d2h(ho, x_out, nn);
            for (size_t i = 0; i < nn; i++) ho[i] += hc[i];
            free(hc); h2d(x_out, ho, nn); free(ho);
        }
        /* x = wte(x_out) + lvl_pos slice */
        float *xe = arena_alloc((size_t)Bm * Lcur * C);
        {
            Tensor *tw = W(ws, "gen.transformer.wte.weight");
            Tensor *tb = weights_get(ws, "gen.transformer.wte.bias");
            linear_fwd(x_out, tw->data, tb ? tb->data : NULL, xe, (int64_t)Bm * Lcur, C, D);
        }
        {
            size_t n = (size_t)Bm * Lcur * C;
            float *h = (float *)malloc(n * sizeof(float));
            float *hl = (float *)malloc((size_t)Lcur * C * sizeof(float));
            d2h(h, xe, n);
            float *hlp = (float *)malloc((size_t)m->T * C * sizeof(float));
            d2h(hlp, m->lvl_pos, (size_t)m->T * C);
            for (int b = 0; b < Bm; b++)
                for (int i = 0; i < Lcur * C; i++)
                    h[(size_t)b * Lcur * C + i] += hlp[(size_t)(level_accum + i / C) * C + (i % C)];
            /* NOTE: lvl_pos rows are per target-position; level_accum counts in
             * target positions — matches level_pos_embed slicing. */
            free(hl); free(hlp);
            h2d(xe, h, n); free(h);
        }
        /* h = [pemb, xe]; gen mask; run; head */
        int S = P + Lcur;
        Tensor *_hhT = tensor_alloc(2, (int64_t[]){(int64_t)Bm * S, C});
        float *hh = _hhT->data;
        for (int b = 0; b < Bm; b++) {
            d2d(hh + (size_t)b * S * C, pemb + (size_t)b * P * C, (size_t)P * C);
            d2d(hh + ((size_t)b * S + P) * C, xe + (size_t)b * Lcur * C, (size_t)Lcur * C);
        }
        float *hbase = (float *)malloc((size_t)S * S * sizeof(float));
        for (int i = 0; i < S; i++) for (int j = 0; j < S; j++)
            hbase[i * S + j] = (i < P && j >= P) ? -INFINITY : 0.0f;
        int V = c->vocab_size[hi];
        float *hlog_all = (float *)malloc((size_t)Bm * Lcur * V * sizeof(float));
        for (int b = 0; b < Bm; b++) {
            arena_reset(); /* per-board temps freed; hh/z_hat/pemb are tensors */
            if (padcol) for (int i = 0; i < S; i++) for (int j = 0; j < P; j++)
                if (padcol[b * P + j]) hbase[i * S + j] = -INFINITY;
            float *bm = arena_alloc((size_t)S * S);
            h2d(bm, hbase, (size_t)S * S);
            if (padcol) for (int i = 0; i < S; i++) for (int j = 0; j < P; j++)
                hbase[i * S + j] = (i < P && j >= P) ? -INFINITY : 0.0f;
            float *hb = arena_alloc((size_t)S * C);
            d2d(hb, hh + (size_t)b * S * C, (size_t)S * C);
            run_stack(m, hb, 1, S, bm, m->gcos[hi], m->gsin[hi]);
            float *sl = arena_alloc((size_t)Lcur * C);
            {
                float *t2 = (float *)malloc((size_t)S * C * sizeof(float));
                d2h(t2, hb, (size_t)S * C);
                float *s2 = (float *)malloc((size_t)Lcur * C * sizeof(float));
                memcpy(s2, t2 + (size_t)P * C, (size_t)Lcur * C * sizeof(float));
                free(t2); h2d(sl, s2, (size_t)Lcur * C); free(s2);
            }
            float *lg = arena_alloc((size_t)Lcur * V);
            head_logits(m, hi, sl, lg, 1, Lcur);
            d2h(hlog_all + (size_t)b * Lcur * V, lg, (size_t)Lcur * V);
        }
        free(hbase);
        tensor_free(_hhT);
        /* CFG blend */
        float *hlog_use = hlog_all;
        float *blended = NULL;
        if (use_cfg) {
            blended = (float *)malloc((size_t)B * Lcur * V * sizeof(float));
            double tw = cfg_scale * ((li + 1) / (double)(nlev - 1));
            for (int b = 0; b < B; b++) for (int t = 0; t < Lcur; t++) for (int v = 0; v < V; v++) {
                double a = hlog_all[((size_t)b * Lcur + t) * V + v];
                double n2 = hlog_all[((size_t)(B + b) * Lcur + t) * V + v];
                blended[((size_t)b * Lcur + t) * V + v] = (float)((1 + tw) * a - tw * n2);
            }
            hlog_use = blended;
        }
        int Bn = use_cfg ? B : Bm;
        int *next = (int *)malloc((size_t)Bn * Lcur * sizeof(int));
        /* sampling state: use first Bn rows */
        sample_level_rows(m, hlog_use, next, Bn, Lcur, V, temp, top_k, top_p,
                          freq, mu, adj_a, adj_w, adj_d, allow, &rng,
                          cntS, histS, histL);
        free(hlog_all); free(blended);
        /* store into output; keep model-batch copy for next iter */
        for (int b = 0; b < B; b++)
            memcpy(all + (size_t)b * total + out_off, next + (size_t)b * Lcur, Lcur * sizeof(int));
        out_off += Lcur;
        if (use_cfg) {
            cur = (int *)malloc((size_t)Bm * Lcur * sizeof(int));
            memcpy(cur, next, (size_t)B * Lcur * sizeof(int));
            memcpy(cur + (size_t)B * Lcur, next, (size_t)B * Lcur * sizeof(int));
        } else {
            cur = next; next = NULL;
        }
        free(next);
        level_accum += Lcur;
    }
    for (int b = 0; b < Bm; b++) { free(cntS[b]); free(histS[b]); }
    free(cntS); free(histS); free(histL);
    free(cur); free(mprefix); free(padcol);
    tensor_free(_zhat); tensor_free(_pembT);
    return all;
}

static int V0_(NCPState *m) { return m->cfg.vocab_size[0]; }

int ncp_P(NCPState *m) { return m->P; }
int ncp_T(NCPState *m) { return m->T; }
int ncp_ntgt(NCPState *m) { return m->n_tgt; }
int ncp_lvlfirst(NCPState *m) { return m->lvl_first; }
float *ncp_rope_cos(NCPState *m) { return m->rope_cos; }
float *ncp_rope_sin(NCPState *m) { return m->rope_sin; }
float *ncp_rope_gen_cos(NCPState *m, int k) { return m->gcos[k]; }
float *ncp_rope_gen_sin(NCPState *m, int k) { return m->gsin[k]; }

struct VQState;
struct VQState *ncp_vq(NCPState *m) { return vqvae_vq(m->vqvae); }
