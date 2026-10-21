/* train_tape.c — reverse-mode autograd for the NCM kernel set.
 *
 * Recording wrappers (t_*) call the backend forward kernels and append a
 * record holding backend pointers (arena-resident until step end) plus a
 * host-owned aux stash (sv). tape_backward() stages to host and runs
 * plain-C backward loops, correct on CPU and CUDA builds alike.
 *
 * Parameter grads accumulate into the optimizer's buffers (matched by
 * pointer); activation grads accumulate in an internal map and die with
 * the step.
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TOP_LINEAR, TOP_EMBED, TOP_LN, TOP_RMSN, TOP_GN,
    TOP_GELU, TOP_SILU,
    TOP_SOFTMAX, TOP_ROPE, TOP_SDPA,
    TOP_CONV, TOP_CONVT, TOP_IAREA, TOP_ILIN,
    TOP_ADD, TOP_SCALE, TOP_CONCAT, TOP_GATHER, TOP_MUL, TOP_RELU2, TOP_JOIN, TOP_MSE,
    TOP_SPLIT, TOP_MERGE,
    TOP_CE, TOP_TIED,
} TopKind;

typedef struct { float *P; float scale; int H; } SdpaAux;
typedef struct { float *probs; int *tgt; float *w; double cnt; } CeAux;
typedef struct { float *hE; double *Esq; double dscale, invT; } TiedAux;

typedef struct {
    TopKind op;
    float *a, *b, *c, *y;
    void *sv;               /* host-owned aux (freed with tape) */
    int64_t n;
    int p1, p2, p3, p4;
    double s;
} TopRec;

struct Tape {
    TopRec *r;
    int n, cap;
};

typedef struct {
    const float *key;
    float *g;
    int64_t n;
    int owned;
} GEnt;

typedef struct {
    GEnt *e;
    int n, cap;
} GMap;

Tape *tape_new(void) {
    Tape *t = (Tape *)calloc(1, sizeof(Tape));
    t->cap = 2048;
    t->r = (TopRec *)malloc(t->cap * sizeof(TopRec));
    return t;
}

static void tape_free_recs(Tape *t, int from) {
    for (int i = from; i < t->n; i++) free(t->r[i].sv);
}

void tape_free(Tape *t) { tape_free_recs(t, 0); free(t->r); free(t); }
void tape_clear(Tape *t) { tape_free_recs(t, 0); t->n = 0; }

static TopRec *emit(Tape *t, TopKind op) {
    if (t->n == t->cap) {
        t->cap *= 2;
        t->r = (TopRec *)realloc(t->r, t->cap * sizeof(TopRec));
    }
    TopRec *r = &t->r[t->n++];
    memset(r, 0, sizeof(*r));
    r->op = op;
    return r;
}

static GMap *gmap_new(void) {
    GMap *m = (GMap *)calloc(1, sizeof(GMap));
    m->cap = 256;
    m->e = (GEnt *)malloc(m->cap * sizeof(GEnt));
    return m;
}

static void gmap_free(GMap *m) {
    for (int i = 0; i < m->n; i++) if (m->e[i].owned) free(m->e[i].g);
    free(m->e);
    free(m);
}

/* alias an external grad buffer (not owned, not freed) */
static float *gmap_find(GMap *m, const float *key);

static void gmap_alias(GMap *m, const float *key, float *g, int64_t n) {
    if (gmap_find(m, key)) return;
    if (m->n == m->cap) {
        m->cap *= 2;
        m->e = (GEnt *)realloc(m->e, m->cap * sizeof(GEnt));
    }
    m->e[m->n].key = key;
    m->e[m->n].g = g;
    m->e[m->n].n = n;
    m->e[m->n].owned = 0;
    m->n++;
}

static float *gmap_get(GMap *m, const float *key, int64_t n) {
    for (int i = 0; i < m->n; i++)
        if (m->e[i].key == key) return m->e[i].g;
    if (m->n == m->cap) {
        m->cap *= 2;
        m->e = (GEnt *)realloc(m->e, m->cap * sizeof(GEnt));
    }
    float *g = (float *)calloc(n, sizeof(float));
    m->e[m->n].key = key;
    m->e[m->n].g = g;
    m->e[m->n].n = n;
    m->e[m->n].owned = 1;
    m->n++;
    return g;
}

static float *gmap_find(GMap *m, const float *key) {
    for (int i = 0; i < m->n; i++)
        if (m->e[i].key == key) return m->e[i].g;
    return NULL;
}

static float *stage_h(const float *d, size_t n) {
    float *h = (float *)malloc(n * sizeof(float));
    mem_d2h(h, d, n * sizeof(float));
    return h;
}

/* ---------------- recording wrappers (backend forward + record) ---------------- */

void t_linear(Tape *t, const float *x, const float *W, const float *b,
              float *y, int64_t batch, int M, int K) {
    linear_fwd(x, W, b, y, batch, M, K);
    TopRec *r = emit(t, TOP_LINEAR);
    r->a = (float *)x; r->b = (float *)W; r->c = (float *)b; r->y = y;
    r->n = batch; r->p1 = (int)M; r->p2 = (int)K;
}

void t_embed(Tape *t, const float *W, const int *idx, int V, float *y,
             int64_t n, int D) {
    int *bidx = (int *)arena_alloc(n);
    mem_h2d(bidx, idx, n * sizeof(int));
    embedding_fwd(W, bidx, y, n, D);
    TopRec *r = emit(t, TOP_EMBED);
    r->a = (float *)W; r->y = y; r->n = n; r->p1 = D; r->p2 = V;
    int *cp = (int *)malloc(n * sizeof(int));
    memcpy(cp, idx, n * sizeof(int));
    r->sv = cp;
}

void t_layernorm(Tape *t, const float *x, const float *w, const float *b,
                 float *y, int64_t rows, int C) {
    layernorm_fwd(x, w, b, y, rows, C);
    TopRec *r = emit(t, TOP_LN);
    r->a = (float *)x; r->b = (float *)w; r->c = (float *)b; r->y = y;
    r->n = rows; r->p1 = C;
}

void t_rmsnorm(Tape *t, const float *x, const float *w, float *y,
               int64_t rows, int C) {
    rmsnorm_fwd(x, w, y, rows, C);
    TopRec *r = emit(t, TOP_RMSN);
    r->a = (float *)x; r->b = (float *)w; r->y = y;
    r->n = rows; r->p1 = C;
}

void t_groupnorm(Tape *t, const float *x, float *y, int N, int G, int C, int TT) {
    groupnorm_fwd(x, y, N, G, C, TT);
    TopRec *r = emit(t, TOP_GN);
    r->a = (float *)x; r->y = y;
    r->n = N; r->p1 = G; r->p2 = C; r->p3 = TT;
}

void t_gelu(Tape *t, const float *x, float *y, int64_t n) {
    gelu_fwd(x, y, n);
    TopRec *r = emit(t, TOP_GELU);
    r->a = (float *)x; r->y = y; r->n = n;
}

void t_silu(Tape *t, const float *x, float *y, int64_t n) {
    silu_fwd(x, y, n);
    TopRec *r = emit(t, TOP_SILU);
    r->a = (float *)x; r->y = y; r->n = n;
}

void t_mul(Tape *t, const float *a, const float *b, float *y, int64_t n) {
    float *ha = stage_h(a, (size_t)n);
    float *hb = stage_h(b, (size_t)n);
    float *ho = (float *)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) ho[i] = ha[i] * hb[i];
    free(ha); free(hb);
    mem_h2d(y, ho, n * sizeof(float));
    free(ho);
    TopRec *r = emit(t, TOP_MUL);
    r->a = (float *)a; r->b = (float *)b; r->y = y; r->n = n;
}

void t_relu2(Tape *t, const float *x, float *y, int64_t n) {
    float *hx = stage_h(x, (size_t)n);
    float *ho = (float *)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) { float v = hx[i] > 0 ? hx[i] : 0; ho[i] = v * v; }
    free(hx);
    mem_h2d(y, ho, n * sizeof(float));
    free(ho);
    TopRec *r = emit(t, TOP_RELU2);
    r->a = (float *)x; r->y = y; r->n = n;
}

void t_softmax(Tape *t, const float *x, float *y, int64_t rows, int C) {
    softmax_fwd(x, y, rows, C);
    TopRec *r = emit(t, TOP_SOFTMAX);
    r->a = (float *)x; r->y = y; r->n = rows; r->p1 = C;
}

/* out-of-place rope into yq/yk (caller-allocated). cos/sin persistent. */
void t_rope(Tape *t, const float *xq, const float *xk, float *yq, float *yk,
            const float *cos, const float *sin, int B, int H, int T, int hd) {
    size_t n = (size_t)B * H * T * hd;
    mem_d2d(yq, xq, n * sizeof(float));
    mem_d2d(yk, xk, n * sizeof(float));
    rope_apply(yq, yk, cos, sin, B, H, T, hd);
    TopRec *r = emit(t, TOP_ROPE);
    r->a = (float *)xq; r->b = (float *)xk; r->c = yq; r->y = yk;
    const float **cs = (const float **)malloc(2 * sizeof(void *));
    cs[0] = cos; cs[1] = sin;
    r->sv = cs;
    r->n = B; r->p1 = H; r->p2 = T; r->p3 = hd;
}

/* SDPA with additive mask; saves probs P for backward. */
void t_sdpa(Tape *t, const float *q, const float *k, const float *v,
            const float *mask, float *o, int B, int H, int Tq, int Tkv, int hd) {
    int64_t rows = (int64_t)B * H * Tq;
    float *P = arena_alloc((size_t)rows * Tkv);
    /* forward via existing kernel into temp, then probs recompute on host?
     * Simplest correct: run sdpa_fwd (backend), then recompute P on host
     * from staged q,k,mask for the backward. Costs one extra host pass. */
    sdpa_fwd(q, k, v, mask, o, B, H, Tq, Tkv, hd);
    float scale = 1.0f / sqrtf((float)hd);
    float *hq = stage_h(q, (size_t)rows * Tq * 0 + (size_t)B * H * Tq * hd);
    float *hk = stage_h(k, (size_t)B * H * Tkv * hd);
    float *hm = mask ? stage_h(mask, (size_t)Tq * Tkv) : NULL;
    float *hP = (float *)malloc((size_t)rows * Tkv * sizeof(float));
    for (int64_t r = 0; r < rows; r++) {
        int bh = (int)(r / Tq), tq = (int)(r % Tq);
        double mx = -1e300;
        for (int tk = 0; tk < Tkv; tk++) {
            const float *qr = hq + (r * hd), *kr = hk + ((int64_t)bh * Tkv + tk) * hd;
            double s = 0;
            for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
            s *= scale;
            if (hm) s += hm[tq * Tkv + tk];
            hP[r * Tkv + tk] = (float)s;
            if (s > mx) mx = s;
        }
        double den = 0;
        for (int tk = 0; tk < Tkv; tk++) { hP[r * Tkv + tk] = (float)exp(hP[r * Tkv + tk] - mx); den += hP[r * Tkv + tk]; }
        for (int tk = 0; tk < Tkv; tk++) hP[r * Tkv + tk] /= (float)den;
    }
    free(hq); free(hk); free(hm);
    mem_h2d(P, hP, (size_t)rows * Tkv * sizeof(float));
    free(hP);
    TopRec *r = emit(t, TOP_SDPA);
    r->a = (float *)q; r->b = (float *)k; r->c = (float *)v; r->y = o;
    r->sv = NULL; /* P lives in arena (P ptr stashed below in s? need it!) */
    r->n = rows; r->p1 = hd; r->p2 = Tq; r->p3 = Tkv;
    /* stash P pointer + scale in sv */
    SdpaAux *st = (SdpaAux *)malloc(sizeof(SdpaAux));
    st->P = P; st->scale = scale; st->H = H;
    r->sv = st;
}

void t_conv(Tape *t, const float *x, const float *W, const float *b, float *y,
            int N, int Cin, int Cout, int T, int K, int stride, int pad) {
    conv1d_fwd(x, W, b, y, N, Cin, Cout, T, K, stride, pad);
    TopRec *r = emit(t, TOP_CONV);
    r->a = (float *)x; r->b = (float *)W; r->c = (float *)b; r->y = y;
    r->n = N; r->p1 = Cin; r->p2 = Cout; r->p3 = T; r->p4 = K;
    int *sp = (int *)malloc(2 * sizeof(int));
    sp[0] = stride; sp[1] = pad;
    r->sv = sp;
}

void t_convt(Tape *t, const float *x, const float *W, const float *b, float *y,
             int N, int Cin, int Cout, int T, int K, int stride) {
    convtranspose1d_fwd(x, W, b, y, N, Cin, Cout, T, K, stride);
    TopRec *r = emit(t, TOP_CONVT);
    r->a = (float *)x; r->b = (float *)W; r->c = (float *)b; r->y = y;
    r->n = N; r->p1 = Cin; r->p2 = Cout; r->p3 = T; r->p4 = K;
    int *sp = (int *)malloc(sizeof(int));
    sp[0] = stride;
    r->sv = sp;
}

void t_iarea(Tape *t, const float *x, float *y, int N, int C, int Tin, int Tout) {
    interpolate_area_fwd(x, y, N, C, Tin, Tout);
    TopRec *r = emit(t, TOP_IAREA);
    r->a = (float *)x; r->y = y;
    r->n = N; r->p1 = C; r->p2 = Tin; r->p3 = Tout;
}

void t_ilin(Tape *t, const float *x, float *y, int N, int C, int Tin, int Tout) {
    interpolate_linear_fwd(x, y, N, C, Tin, Tout);
    TopRec *r = emit(t, TOP_ILIN);
    r->a = (float *)x; r->y = y;
    r->n = N; r->p1 = C; r->p2 = Tin; r->p3 = Tout;
}

void t_add(Tape *t, const float *a, const float *b, float *y, int64_t n) {
    add_fwd(a, b, y, n);
    TopRec *r = emit(t, TOP_ADD);
    r->a = (float *)a; r->b = (float *)b; r->y = y; r->n = n;
}

void t_scale(Tape *t, const float *x, float s, float *y, int64_t n) {
    scale_fwd(x, s, y, n);
    TopRec *r = emit(t, TOP_SCALE);
    r->a = (float *)x; r->y = y; r->n = n; r->s = s;
}

void t_concat(Tape *t, const float *a, const float *b, float *y, int64_t na, int64_t nb) {
    /* manual concat (backend) */
#ifdef NCM_CUDA
    cudaMemcpy(y, a, na * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(y + na, b, nb * sizeof(float), cudaMemcpyDeviceToDevice);
#else
    memcpy(y, a, na * sizeof(float));
    memcpy(y + na, b, nb * sizeof(float));
#endif
    TopRec *r = emit(t, TOP_CONCAT);
    r->a = (float *)a; r->b = (float *)b; r->y = y; r->n = na; r->s = (double)nb;
}

void t_qkvsplit(Tape *t, const float *in, float *qq, float *kk, float *vv,
                  int B, int T, int C, int H, int hd, int K, int rep) {
    qkv_split_fwd(in, qq, kk, vv, B, T, C, H, hd, K, rep);
    TopRec *r = emit(t, TOP_SPLIT);
    r->a = (float *)in; r->b = qq; r->c = kk; r->y = vv;
    r->n = B; r->p1 = T; r->p2 = C; r->p3 = H; r->p4 = hd;
    int *ip = (int *)malloc(2 * sizeof(int));
    ip[0] = K; ip[1] = rep;
    r->sv = ip;
}

void t_headmerge(Tape *t, const float *ctx, float *out, int B, int T, int C, int H, int hd) {
    head_merge_fwd(ctx, out, B, T, C, H, hd);
    TopRec *r = emit(t, TOP_MERGE);
    r->a = (float *)ctx; r->y = out;
    r->n = B; r->p1 = T; r->p2 = C; r->p3 = H; r->p4 = hd;
}

void t_join(Tape *t, const float *a, const float *b, float *y, int B, int P, int T, int C) {
    for (int bb = 0; bb < B; bb++) {
        mem_d2d(y + ((size_t)bb * (P + T) + 0) * C, a + (size_t)bb * P * C, (size_t)P * C * sizeof(float));
        mem_d2d(y + ((size_t)bb * (P + T) + P) * C, b + (size_t)bb * T * C, (size_t)T * C * sizeof(float));
    }
    TopRec *r = emit(t, TOP_JOIN);
    r->a = (float *)a; r->b = (float *)b; r->y = y;
    r->n = B; r->p1 = P; r->p2 = T; r->p3 = C;
}

void t_gather_rows(Tape *t, const float *x, const int *rows, int nrows, int C, float *y) {
    /* y[i] = x[rows[i]] row-wise, via host staging (training path) */
    int64_t n = (int64_t)nrows * C;
    float *hy = (float *)malloc(n * sizeof(float));
    int mx = 0;
    for (int i = 0; i < nrows; i++) if (rows[i] > mx) mx = rows[i];
    float *hs = stage_h(x, (size_t)(mx + 1) * C);
    for (int i = 0; i < nrows; i++)
        memcpy(hy + (size_t)i * C, hs + (size_t)rows[i] * C, C * sizeof(float));
    mem_h2d(y, hy, n * sizeof(float));
    free(hy); free(hs);
    TopRec *r = emit(t, TOP_GATHER);
    r->a = (float *)x; r->y = y; r->n = nrows; r->p1 = C;
    int *cp = (int *)malloc(nrows * sizeof(int));
    memcpy(cp, rows, nrows * sizeof(int));
    r->sv = cp;
}

/* Fused softmax-CE: loss[0] = mean NLL over valid rows; saves probs. */
void t_ce(Tape *t, const float *logits, const int *tgt, const float *wmask,
          float *loss, float *probs, int64_t rows, int V, double weight) {
    float *hl = stage_h(logits, (size_t)rows * V);
    int *ht = (int *)malloc(rows * sizeof(int));
    memcpy(ht, tgt, rows * sizeof(int));
    float *hw = NULL;
    if (wmask) {
        hw = (float *)malloc(rows * sizeof(float));
        memcpy(hw, wmask, rows * sizeof(float));
    }
    double nll = 0, cnt = 0;
    float *hp = (float *)malloc((size_t)rows * V * sizeof(float));
    for (int64_t r = 0; r < rows; r++) {
        float m = hl[r * V];
        for (int v = 1; v < V; v++) if (hl[r * V + v] > m) m = hl[r * V + v];
        double den = 0;
        for (int v = 0; v < V; v++) { double e = exp((double)hl[r * V + v] - m); hp[r * V + v] = (float)e; den += e; }
        for (int v = 0; v < V; v++) hp[r * V + v] /= (float)den;
        float wm = hw ? hw[r] : 1.0f;
        if (wm != 0) {
            nll += wm * -log(hp[r * V + ht[r]] + 1e-30);
            cnt += wm;
        }
    }
    free(hl);
    mem_h2d(probs, hp, (size_t)rows * V * sizeof(float));
    float lh = cnt > 0 ? (float)(nll / cnt) : 0.0f;
    mem_h2d(loss, &lh, sizeof(float));
    TopRec *r = emit(t, TOP_CE);
    r->a = (float *)logits; r->y = loss;
    r->n = rows; r->p1 = V; r->s = weight;
    CeAux *st = (CeAux *)malloc(sizeof(CeAux));
    st->probs = probs; st->tgt = ht; st->w = hw; st->cnt = cnt;
    r->sv = st;
}

/* Tied codebook head forward (distance form) + record. */
void t_mse(Tape *t, const float *a, const float *b_const, float *loss, int64_t n, double weight) {
    float *ha = stage_h(a, (size_t)n);
    float *hb = stage_h(b_const, (size_t)n);
    double s = 0;
    for (int64_t i = 0; i < n; i++) { double d = ha[i] - hb[i]; s += d * d; }
    free(ha); free(hb);
    float lh = (float)(s / n * weight);
    mem_h2d(loss, &lh, sizeof(float));
    TopRec *r = emit(t, TOP_MSE);
    r->a = (float *)a; r->b = (float *)b_const; r->y = loss;
    r->n = n; r->s = weight;
}

void t_tied(Tape *t, const float *z, const float *E, const float *bias,
            float *out, int64_t rows, int Dd, int V, double dscale, double invT) {
    float *hz = stage_h(z, (size_t)rows * Dd);
    float *hE = stage_h(E, (size_t)V * Dd);
    double *Esq = (double *)malloc(V * sizeof(double));
    for (int v = 0; v < V; v++) {
        double s = 0;
        for (int d = 0; d < Dd; d++) s += (double)hE[v * Dd + d] * hE[v * Dd + d];
        Esq[v] = s;
    }
    float *ho = (float *)malloc((size_t)rows * V * sizeof(float));
    float *hb = NULL;
    if (bias) { hb = (float *)malloc(V * sizeof(float)); memcpy(hb, bias, V * sizeof(float)); }
    for (int64_t r = 0; r < rows; r++) {
        for (int v = 0; v < V; v++) {
            double dot = 0;
            for (int d = 0; d < Dd; d++) dot += (double)hz[r * Dd + d] * hE[v * Dd + d];
            double o = (2 * dot - Esq[v]) / dscale * invT;
            if (hb) o += hb[v];
            ho[r * V + v] = (float)o;
        }
    }
    free(hz); free(hb);
    mem_h2d(out, ho, (size_t)rows * V * sizeof(float));
    free(ho);
    TopRec *r = emit(t, TOP_TIED);
    r->a = (float *)z; r->b = (float *)E; r->c = (float *)bias; r->y = out;
    r->n = rows; r->p1 = Dd; r->p2 = V;
    TiedAux *st = (TiedAux *)malloc(sizeof(TiedAux));
    st->hE = hE; st->Esq = Esq; st->dscale = dscale; st->invT = invT;
    r->sv = st;
}

/* ---------------- backward ---------------- */

/* PMap from ncm.h */

/* dst[key] += src (host loops; n elements) */
static void hadd_into(float *dst, const float *src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] += src[i];
}

void tape_backward(Tape *t, PMap *pm, float seed) {
    (void)seed;
    GMap *gm = gmap_new();
    if (pm) for (int pi = 0; pi < pm->n; pi++)
        gmap_alias(gm, pm->params[pi], pm->grads[pi], pm->sizes[pi]);
    /* seed: find the CE loss record(s); dL/dloss = 1 */
    for (int i = t->n - 1; i >= 0; i--) {
        TopRec *r = &t->r[i];
        float *dy = gmap_find(gm, r->y);
        /* loss records (CE/MSE) always contribute (they are terminal).
         * Multi-output ops (ROPE: yq in c, yk in y; SPLIT: q/k/v in
         * b/c/y) must not be skipped when only one output carries grad. */
        if (!dy && r->op != TOP_CE && r->op != TOP_MSE) {
            if (r->op == TOP_ROPE) {
                if (!gmap_find(gm, r->c) && !gmap_find(gm, r->y)) continue;
            } else if (r->op == TOP_SPLIT) {
                if (!gmap_find(gm, r->b) && !gmap_find(gm, r->c) &&
                    !gmap_find(gm, r->y)) continue;
            } else continue;
        }
        switch (r->op) {
        case TOP_CE: {
            /* dL/dlogits = (p - onehot)/cnt (mean reduction) */
            CeAux *st = (CeAux *)r->sv;
            int64_t rows = r->n;
            int V = r->p1;
            float *hp = stage_h(st->probs, (size_t)rows * V);
            float *dl = (float *)calloc((size_t)rows * V, sizeof(float));
            double inv = (st->cnt > 0 ? 1.0 / st->cnt : 0.0) * r->s;
            /* independent per (row, vocab): same arithmetic in any thread order */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)rows * V > 512)
            for (int64_t rr = 0; rr < rows; rr++) {
                for (int v = 0; v < V; v++) {
                    float wm = st->w ? st->w[rr] : 1.0f;
                    if (wm == 0) continue;
                    dl[rr * V + v] = (float)((hp[rr * V + v] - (v == st->tgt[rr] ? 1.0 : 0.0)) * wm * inv);
                }
            }
            free(hp);
            /* logits grads accumulate in map (or aliased param buffer) */
            hadd_into(gmap_get(gm, r->a, rows * V), dl, rows * V);
            free(dl);
            break;
        }
        case TOP_LINEAR: {
            int64_t batch = r->n;
            int M = r->p1, K = r->p2;
            float *hx = stage_h(r->a, (size_t)batch * K);
            float *hdy = stage_h(dy, (size_t)batch * M);
            /* dx */
            float *dx = gmap_get(gm, r->a, batch * K);
            float *hW = stage_h(r->b, (size_t)M * K);
            /* dx: independent per (row, k) */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)batch * K > 512)
            for (int64_t rr = 0; rr < batch; rr++)
                for (int k = 0; k < K; k++) {
                    double s = 0;
                    for (int m = 0; m < M; m++) s += (double)hdy[rr * M + m] * hW[m * K + k];
                    dx[rr * K + k] += (float)s;
                }
            /* dW, db -> params if registered else map.
             * Parallel over outputs; inner batch loop keeps the original
             * rr-ascending accumulation order, so results are bit-identical
             * at any thread count. */
            {
                float *dW = (float *)calloc((size_t)M * K, sizeof(float));
                float *db = (float *)calloc(M, sizeof(float));
#pragma omp parallel for schedule(static) if((int64_t)M * K > 512)
                for (int m = 0; m < M; m++) {
                    double bsum = 0;
                    for (int64_t rr = 0; rr < batch; rr++)
                        bsum += hdy[rr * M + m];
                    db[m] += (float)bsum;
                    for (int k = 0; k < K; k++) {
                        double s = 0;
                        for (int64_t rr = 0; rr < batch; rr++)
                            s += (double)hdy[rr * M + m] * hx[rr * K + k];
                        dW[m * K + k] += (float)s;
                    }
                }
                hadd_into(gmap_get(gm, r->b, (size_t)M * K),  dW, (size_t)M * K);
                free(dW);
                if (r->c) {
                    hadd_into(gmap_get(gm, r->c, M), db, M);
                }
                free(db);
            }
            free(hx); free(hdy); free(hW);
            break;
        }
        case TOP_EMBED: {
            int64_t n = r->n;
            int D = r->p1, V = r->p2;
            int *idx = (int *)r->sv;
            float *hdy = stage_h(dy, (size_t)n * D);
            float *dW = (float *)calloc((size_t)V * D, sizeof(float));
            for (int64_t i = 0; i < n; i++) {
                int id = idx[i];
                if (id < 0 || id >= V) continue;
                for (int d = 0; d < D; d++) dW[(size_t)id * D + d] += hdy[i * D + d];
            }
            free(hdy);
            hadd_into(gmap_get(gm, r->a, (size_t)V * D),  dW, (size_t)V * D);
            free(dW);
            break;
        }
        case TOP_LN: {
            int64_t rows = r->n;
            int C = r->p1;
            float *hx = stage_h(r->a, (size_t)rows * C);
            float *hdy = stage_h(dy, (size_t)rows * C);
            float *hw = r->b ? stage_h(r->b, C) : NULL;
            float *hb = r->c ? stage_h(r->c, C) : NULL;
            float *dx = gmap_get(gm, r->a, rows * C);
            float *dw = (float *)calloc(C, sizeof(float));
            float *db = (float *)calloc(C, sizeof(float));
            /* per-row stats first (independent rows) */
            double *pmean = (double *)malloc((size_t)(rows > 0 ? rows : 1) * sizeof(double));
            double *pinv = (double *)malloc((size_t)(rows > 0 ? rows : 1) * sizeof(double));
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                double mean = 0;
                for (int c = 0; c < C; c++) mean += hx[rr * C + c];
                mean /= C;
                double var = 0;
                for (int c = 0; c < C; c++) { double d = hx[rr * C + c] - mean; var += d * d; }
                var /= C;
                pmean[rr] = mean;
                pinv[rr] = 1.0 / sqrt(var + 1e-6);
            }
            /* param grads: same rr-ascending accumulation order as serial */
#pragma omp parallel for schedule(static) if((int64_t)rows * C > 512)
            for (int c = 0; c < C; c++) {
                double sw = 0, sb = 0;
                for (int64_t rr = 0; rr < rows; rr++) {
                    double y = (hx[rr * C + c] - pmean[rr]) * pinv[rr];
                    sw += hdy[rr * C + c] * y;
                    sb += hdy[rr * C + c];
                }
                dw[c] = (float)sw;
                db[c] = (float)sb;
            }
            /* dx per row (independent rows) */
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                double mean = pmean[rr], inv = pinv[rr];
                double dmean = 0, dvar = 0;
                for (int c = 0; c < C; c++) {
                    double w = hw ? hw[c] : 1.0;
                    double d = hdy[rr * C + c] * w;
                    dmean += d;
                    dvar += d * (hx[rr * C + c] - mean);
                }
                dmean /= C;
                dvar *= -0.5 * inv * inv * inv / C;
                for (int c = 0; c < C; c++) {
                    double w = hw ? hw[c] : 1.0;
                    double d = hdy[rr * C + c] * w;
                    dx[rr * C + c] += (float)((d - dmean) * inv + 2 * (hx[rr * C + c] - mean) * dvar);
                }
            }
            free(pmean); free(pinv);
            free(hx); free(hdy);
            if (hw) {
                hadd_into(gmap_get(gm, r->b, C), dw, C);
                free(hw);
            } else free(dw);
            if (hb) {
                hadd_into(gmap_get(gm, r->c, C), db, C);
                free(hb);
            } else free(db);
            break;
        }
        case TOP_RMSN: {
            int64_t rows = r->n;
            int C = r->p1;
            float *hx = stage_h(r->a, (size_t)rows * C);
            float *hdy = stage_h(dy, (size_t)rows * C);
            float *hw = r->b ? stage_h(r->b, C) : NULL;
            float *dx = gmap_get(gm, r->a, rows * C);
            float *dw = (float *)calloc(C, sizeof(float));
            double *pinv = (double *)malloc((size_t)(rows > 0 ? rows : 1) * sizeof(double));
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                double ms = 0;
                for (int c = 0; c < C; c++) ms += (double)hx[rr * C + c] * hx[rr * C + c];
                ms /= C;
                pinv[rr] = 1.0 / sqrt(ms + 1e-6);
            }
#pragma omp parallel for schedule(static) if((int64_t)rows * C > 512)
            for (int c = 0; c < C; c++) {
                float sw = 0;
                for (int64_t rr = 0; rr < rows; rr++)
                    sw += (float)(hdy[rr * C + c] * hx[rr * C + c] * pinv[rr]);
                dw[c] = sw;
            }
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                double inv = pinv[rr];
                double dot = 0;
                for (int c = 0; c < C; c++) {
                    double w = hw ? hw[c] : 1.0;
                    dot += hdy[rr * C + c] * w * hx[rr * C + c];
                }
                dot /= C;
                for (int c = 0; c < C; c++) {
                    double w = hw ? hw[c] : 1.0;
                    dx[rr * C + c] += (float)((hdy[rr * C + c] * w - dot * hx[rr * C + c] * inv * inv) * inv);
                }
            }
            free(pinv);
            free(hx); free(hdy);
            if (hw) {
                hadd_into(gmap_get(gm, r->b, C), dw, C);
                free(hw);
            } else free(dw);
            break;
        }
        case TOP_GN: {
            /* GroupNorm affine=false: normalize only, no params. */
            int N = (int)r->n, G = r->p1, C = r->p2, TT = r->p3;
            int Cg = C / G;
            float *hx = stage_h(r->a, (size_t)N * C * TT);
            float *hdy = stage_h(dy, (size_t)N * C * TT);
            float *dx = gmap_get(gm, r->a, (size_t)N * C * TT);
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * G > 1)
            for (int n = 0; n < N; n++) for (int g = 0; g < G; g++) {
                double mean = 0;
                int64_t cnt = (int64_t)Cg * TT;
                for (int c = 0; c < Cg; c++) for (int t = 0; t < TT; t++)
                    mean += hx[((int64_t)n * C + g * Cg + c) * TT + t];
                mean /= cnt;
                double var = 0;
                for (int c = 0; c < Cg; c++) for (int t = 0; t < TT; t++) {
                    double d = hx[((int64_t)n * C + g * Cg + c) * TT + t] - mean;
                    var += d * d;
                }
                var /= cnt;
                double inv = 1.0 / sqrt(var + 1e-5);
                double dmean = 0, dvar = 0;
                for (int c = 0; c < Cg; c++) for (int t = 0; t < TT; t++) {
                    int64_t i = ((int64_t)n * C + g * Cg + c) * TT + t;
                    dmean += hdy[i];
                    dvar += hdy[i] * (hx[i] - mean);
                }
                dmean /= cnt;
                dvar *= -0.5 * inv * inv * inv / cnt;
                for (int c = 0; c < Cg; c++) for (int t = 0; t < TT; t++) {
                    int64_t i = ((int64_t)n * C + g * Cg + c) * TT + t;
                    dx[i] += (float)((hdy[i] - dmean) * inv + 2 * (hx[i] - mean) * dvar);
                }
            }
            free(hx); free(hdy);
            break;
        }
        case TOP_GELU:
        case TOP_SILU: {
            int64_t n = r->n;
            float *hx = stage_h(r->a, (size_t)n);
            float *hdy = stage_h(dy, (size_t)n);
            float *dx = gmap_get(gm, r->a, n);
            const double s2 = 1.4142135623730951;
            const double ip = 0.3989422804014327;
            int is_gelu = (r->op == TOP_GELU);
#pragma omp parallel for schedule(static) if(n > 512)
            for (int64_t i = 0; i < n; i++) {
                double g;
                if (is_gelu) {
                    double e = erf(hx[i] / s2);
                    g = 0.5 * (1 + e) + hx[i] * exp(-0.5 * hx[i] * hx[i]) * ip;
                } else {
                    double s = 1.0 / (1.0 + exp(-hx[i]));
                    g = s * (1 + hx[i] * (1 - s));
                }
                dx[i] += (float)(hdy[i] * g);
            }
            free(hx); free(hdy);
            break;
        }
        case TOP_SOFTMAX: {
            int64_t rows = r->n;
            int C = r->p1;
            float *hy = stage_h(r->y, (size_t)rows * C);
            float *hdy = stage_h(dy, (size_t)rows * C);
            float *dx = gmap_get(gm, r->a, rows * C);
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                double dot = 0;
                for (int c = 0; c < C; c++) dot += (double)hdy[rr * C + c] * hy[rr * C + c];
                for (int c = 0; c < C; c++)
                    dx[rr * C + c] += (float)(hy[rr * C + c] * (hdy[rr * C + c] - dot));
            }
            free(hy); free(hdy);
            break;
        }
        case TOP_ROPE: {
            int B = (int)r->n, H = r->p1, T = r->p2, hd = r->p3;
            const float **cs = (const float **)r->sv;
            float *hco = stage_h(cs[0], (size_t)T * hd);
            float *hsi = stage_h(cs[1], (size_t)T * hd);
            int64_t n = (int64_t)B * H * T * hd;
            float *gdq = gmap_find(gm, r->c); /* grad of yq */
            float *gdk = gmap_find(gm, r->y); /* grad of yk */
            if (gdq) {
                float *dxq = gmap_get(gm, r->a, n);
                int h = hd / 2;
#pragma omp parallel for schedule(static) if(n > hd * 4)
                for (int64_t rr = 0; rr < n / hd; rr++) {
                    int t = (int)(rr % T);
                    for (int d = 0; d < h; d++) {
                        double g0 = gdq[rr * hd + d], g1 = gdq[rr * hd + h + d];
                        double c0 = hco[t * hd + d], s0 = hsi[t * hd + d];
                        double c1 = hco[t * hd + h + d], s1 = hsi[t * hd + h + d];
                        /* inverse rotation: x = R^T y */
                        dxq[rr * hd + d] += (float)(g0 * c0 + g1 * s1);
                        dxq[rr * hd + h + d] += (float)(-g0 * s0 + g1 * c1);
                    }
                }
            }
            if (gdk) {
                float *dxk = gmap_get(gm, r->b, n);
                int h = hd / 2;
#pragma omp parallel for schedule(static) if(n > hd * 4)
                for (int64_t rr = 0; rr < n / hd; rr++) {
                    int t = (int)(rr % T);
                    for (int d = 0; d < h; d++) {
                        double g0 = gdk[rr * hd + d], g1 = gdk[rr * hd + h + d];
                        double c0 = hco[t * hd + d], s0 = hsi[t * hd + d];
                        double c1 = hco[t * hd + h + d], s1 = hsi[t * hd + h + d];
                        dxk[rr * hd + d] += (float)(g0 * c0 + g1 * s1);
                        dxk[rr * hd + h + d] += (float)(-g0 * s0 + g1 * c1);
                    }
                }
            }
            free(hco); free(hsi);
            break;
        }
        case TOP_SDPA: {
            SdpaAux *st = (SdpaAux *)r->sv;
            int64_t rows = r->n;
            int hd = r->p1, Tq = r->p2, Tkv = r->p3;
            int H = st->H, B = (int)(rows / (Tq * H));
            float *hP = stage_h(st->P, (size_t)rows * Tkv);
            float *hq = stage_h(r->a, (size_t)B * H * Tq * hd);
            float *hk = stage_h(r->b, (size_t)B * H * Tkv * hd);
            float *hv = stage_h(r->c, (size_t)B * H * Tkv * hd);
            float *hdy = stage_h(dy, (size_t)B * H * Tq * hd);
            float *dq = gmap_get(gm, r->a, (size_t)B * H * Tq * hd);
            float *dk = gmap_get(gm, r->b, (size_t)B * H * Tkv * hd);
            float *dv = gmap_get(gm, r->c, (size_t)B * H * Tkv * hd);
            /* pass 1 (independent rows): softmax-Jacobian coeffs + dq.
             * dq[rr] and ds[rr] touch only this row's slots. */
            double *ds = (double *)malloc((size_t)rows * (size_t)(Tkv > 0 ? Tkv : 1) * sizeof(double));
#pragma omp parallel for schedule(static) if(rows > 4)
            for (int64_t rr = 0; rr < rows; rr++) {
                int b = (int)(rr / (H * Tq)), h = (int)((rr / Tq) % H), tq = (int)(rr % Tq);
                double tot = 0;
                for (int tk = 0; tk < Tkv; tk++) {
                    const float *vr = hv + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
                    double s = 0;
                    for (int d = 0; d < hd; d++) s += (double)hdy[(rr * hd) + d] * vr[d];
                    s *= hP[rr * Tkv + tk];
                    ds[rr * Tkv + tk] = s;
                    tot += s;
                }
                /* softmax Jacobian: ds = P .* (dP - sum(dP .* P)) */
                for (int tk = 0; tk < Tkv; tk++) ds[rr * Tkv + tk] -= hP[rr * Tkv + tk] * tot;
                for (int tk = 0; tk < Tkv; tk++) {
                    const float *kr = hk + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
                    for (int d = 0; d < hd; d++)
                        dq[((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd + d] +=
                            (float)(ds[rr * Tkv + tk] * st->scale * kr[d]);
                }
            }
            /* pass 2 (independent (b,h,tk,d)): dk/dv accumulate over tq in the
             * same ascending order as the serial nest, hence bit-identical */
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)B * H * Tkv > 8)
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int tk = 0; tk < Tkv; tk++) {
                for (int d = 0; d < hd; d++) {
                    float sk = 0, sv = 0;
                    for (int tq = 0; tq < Tq; tq++) {
                        int64_t rr = ((int64_t)b * H + h) * Tq + tq;
                        const float *qr = hq + ((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd;
                        sk += (float)(ds[rr * Tkv + tk] * st->scale * qr[d]);
                        sv += (float)(hP[rr * Tkv + tk] * hdy[rr * hd + d]);
                    }
                    dk[((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd + d] += sk;
                    dv[((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd + d] += sv;
                }
            }
            free(ds);
            free(hP); free(hq); free(hk); free(hv); free(hdy);
            break;
        }
        case TOP_CONV: {
            int N = (int)r->n, Cin = r->p1, Cout = r->p2, T = r->p3, K = r->p4;
            int *sp = (int *)r->sv;
            int stride = sp[0], pad = sp[1];
            int64_t Tout = (T + 2 * pad - K) / stride + 1;
            float *hx = stage_h(r->a, (size_t)N * Cin * T);
            float *hdy = stage_h(dy, (size_t)N * Cout * Tout);
            /* dx (full convolution, flipped) */
            float *dx = gmap_get(gm, r->a, (size_t)N * Cin * T);
            float *hW = stage_h(r->b, (size_t)Cout * Cin * K);
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)N * Cin * T > 512)
            for (int n = 0; n < N; n++) for (int ci = 0; ci < Cin; ci++)
                for (int ti = 0; ti < T; ti++) {
                    double s = 0;
                    for (int co = 0; co < Cout; co++) for (int kk = 0; kk < K; kk++) {
                        int to = ti + pad - kk;
                        if (to < 0 || to % stride != 0) continue;
                        to /= stride;
                        if (to >= Tout) continue;
                        s += (double)hdy[((int64_t)n * Cout + co) * Tout + to] *
                             hW[((int64_t)co * Cin + ci) * K + kk];
                    }
                    dx[((int64_t)n * Cin + ci) * T + ti] += (float)s;
                }
            /* dW, db: parallel over co, serial (n,t) inner in original order */
            {
                float *dW = (float *)calloc((size_t)Cout * Cin * K, sizeof(float));
                float *db = (float *)calloc(Cout, sizeof(float));
#pragma omp parallel for schedule(static) if((int64_t)Cout > 1)
                for (int co = 0; co < Cout; co++) {
                    float bsum = 0;
                    for (int n = 0; n < N; n++) for (int64_t t = 0; t < Tout; t++)
                        bsum += hdy[((int64_t)n * Cout + co) * Tout + t];
                    db[co] += bsum;
                    for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
                        float s = 0;
                        for (int n = 0; n < N; n++) for (int64_t t = 0; t < Tout; t++) {
                            int ti = (int)(t * stride + kk - pad);
                            if (ti < 0 || ti >= T) continue;
                            s += hdy[((int64_t)n * Cout + co) * Tout + t] *
                                 hx[((int64_t)n * Cin + ci) * T + ti];
                        }
                        dW[((int64_t)co * Cin + ci) * K + kk] += s;
                    }
                }
                hadd_into(gmap_get(gm, r->b, (size_t)Cout * Cin * K),  dW, (size_t)Cout * Cin * K);
                free(dW);
                if (r->c) {
                    hadd_into(gmap_get(gm, r->c, Cout), db, Cout);
                }
                free(db);
            }
            free(hx); free(hdy); free(hW);
            break;
        }
        case TOP_CONVT: {
            /* adjoint of convtranspose (stride s, kernel K, pad 0):
             * forward: y[n,co,t] = sum_ci,kk x[n,ci,ti] W[ci,co,kk], t=ti*s+kk.
             * dW[ci,co,kk] += sum_n,ti dy[n,co,ti*s+kk] x[n,ci,ti]
             * dx[n,ci,ti] += sum_co,kk dy[n,co,ti*s+kk] W[ci,co,kk] */
            int N = (int)r->n, Cin = r->p1, Cout = r->p2, T = r->p3, K = r->p4;
            int stride = *(int *)r->sv;
            int64_t Tout = (int64_t)(T - 1) * stride + K;
            float *hx = stage_h(r->a, (size_t)N * Cin * T);
            float *hdy = stage_h(dy, (size_t)N * Cout * Tout);
            float *hW = stage_h(r->b, (size_t)Cin * Cout * K);
        float *dx = gmap_get(gm, r->a, (size_t)N * Cin * T);
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)N * Cin * T > 512)
        for (int n = 0; n < N; n++) for (int ci = 0; ci < Cin; ci++)
            for (int ti = 0; ti < T; ti++) {
                    double s = 0;
                    for (int co = 0; co < Cout; co++) for (int kk = 0; kk < K; kk++) {
                        int64_t t = (int64_t)ti * stride + kk;
                        s += (double)hdy[((int64_t)n * Cout + co) * Tout + t] *
                             hW[((int64_t)ci * Cout + co) * K + kk];
                    }
                    dx[((int64_t)n * Cin + ci) * T + ti] += (float)s;
                }
            {
                float *dW = (float *)calloc((size_t)Cin * Cout * K, sizeof(float));
                float *db = (float *)calloc(Cout, sizeof(float));
                /* parallel over co; serial (n,t) inner preserves order */
#pragma omp parallel for schedule(static) if(Cout > 1)
                for (int co = 0; co < Cout; co++) {
                    float bsum = 0;
                    for (int n = 0; n < N; n++) for (int64_t t = 0; t < Tout; t++)
                        bsum += hdy[((int64_t)n * Cout + co) * Tout + t];
                    db[co] += bsum;
                    for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
                        float s = 0;
                        for (int n = 0; n < N; n++) for (int64_t t = 0; t < Tout; t++) {
                            int num = (int)(t - kk);
                            if (num < 0 || num % stride != 0) continue;
                            int ti = num / stride;
                            if (ti >= T) continue;
                            s += hdy[((int64_t)n * Cout + co) * Tout + t] *
                                 hx[((int64_t)n * Cin + ci) * T + ti];
                        }
                        dW[((int64_t)ci * Cout + co) * K + kk] += s;
                    }
                }
                hadd_into(gmap_get(gm, r->b, (size_t)Cin * Cout * K),  dW, (size_t)Cin * Cout * K);
                free(dW);
                if (r->c) {
                    hadd_into(gmap_get(gm, r->c, Cout), db, Cout);
                }
                free(db);
            }
            free(hx); free(hdy); free(hW);
            break;
        }
        case TOP_IAREA: {
            int N = (int)r->n, C = r->p1, Tin = r->p2, Tout = r->p3;
            float *hdy = stage_h(dy, (size_t)N * C * Tout);
        float *dx = gmap_get(gm, r->a, (size_t)N * C * Tin);
        /* parallel over (n,c) only: t kept serial since neighboring
         * output buckets can share input bins */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * C > 2)
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
            for (int t = 0; t < Tout; t++) {
                    int64_t s0 = (int64_t)((double)t * Tin / Tout);
                    int64_t e0 = (int64_t)ceil((double)(t + 1) * Tin / Tout);
                    if (e0 <= s0) e0 = s0 + 1;
                    if (s0 < 0) s0 = 0;
                    if (e0 > Tin) e0 = Tin;
                    double share = hdy[((int64_t)n * C + c) * Tout + t] / (e0 - s0);
                    for (int64_t k = s0; k < e0; k++)
                        dx[((int64_t)n * C + c) * Tin + k] += (float)share;
                }
            }
            free(hdy);
            break;
        }
        case TOP_ILIN: {
            int N = (int)r->n, C = r->p1, Tin = r->p2, Tout = r->p3;
            float *hdy = stage_h(dy, (size_t)N * C * Tout);
            float *dx = gmap_get(gm, r->a, (size_t)N * C * Tin);
        double scale = (Tout == 1 || Tin == 1) ? 0 : (double)Tin / Tout;
        /* (n,c) independent; t kept serial: neighboring t share bins */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * C > 2)
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
                for (int t = 0; t < Tout; t++) {
                    float g = hdy[((int64_t)n * C + c) * Tout + t];
                    if (Tout == 1 || Tin == 1) {
                        double share = g / Tin;
                        for (int k = 0; k < Tin; k++)
                            dx[((int64_t)n * C + c) * Tin + k] += (float)share;
                        continue;
                    }
                    double src = (t + 0.5) * scale - 0.5;
                    int64_t s0 = (int64_t)floor(src);
                    double w1 = src - s0, w0 = 1 - w1;
                    int64_t i0 = s0 < 0 ? 0 : (s0 >= Tin ? Tin - 1 : s0);
                    int64_t s1 = s0 + 1;
                    int64_t i1 = s1 < 0 ? 0 : (s1 >= Tin ? Tin - 1 : s1);
                    dx[((int64_t)n * C + c) * Tin + i0] += (float)(g * w0);
                    dx[((int64_t)n * C + c) * Tin + i1] += (float)(g * w1);
                }
            }
            free(hdy);
            break;
        }
        case TOP_ADD: {
            int64_t n = r->n;
            float *hdy = stage_h(dy, (size_t)n);
            hadd_into(gmap_get(gm, r->a, n), hdy, n);
            hadd_into(gmap_get(gm, r->b, n), hdy, n);
            free(hdy);
            break;
        }
        case TOP_SCALE: {
            int64_t n = r->n;
            float s = (float)r->s;
            float *hdy = stage_h(dy, (size_t)n);
        float *dx = gmap_get(gm, r->a, n);
#pragma omp parallel for schedule(static) if(n > 512)
        for (int64_t i = 0; i < n; i++) dx[i] += hdy[i] * s;
            free(hdy);
            break;
        }
        case TOP_CONCAT: {
            int64_t na = r->n;
            int64_t nb = (int64_t)r->s;
            float *hdy = stage_h(dy, (size_t)(na + nb));
            hadd_into(gmap_get(gm, r->a, na), hdy, na);
            hadd_into(gmap_get(gm, r->b, nb), hdy + na, nb);
            free(hdy);
            break;
        }
        case TOP_GATHER: {
            int nrows = (int)r->n, C = r->p1;
            int *rows = (int *)r->sv;
            float *hdy = stage_h(dy, (size_t)nrows * C);
            int mx = 0;
            for (int i = 0; i < nrows; i++) if (rows[i] > mx) mx = rows[i];
            float *dx = gmap_get(gm, r->a, (size_t)(mx + 1) * C);
            for (int i = 0; i < nrows; i++)
                for (int c = 0; c < C; c++)
                    dx[(size_t)rows[i] * C + c] += hdy[(size_t)i * C + c];
            free(hdy);
            break;
        }
        case TOP_MUL: {
            int64_t n = r->n;
            float *ha = stage_h(r->a, (size_t)n);
            float *hb = stage_h(r->b, (size_t)n);
            float *hdy = stage_h(dy, (size_t)n);
        float *dxa = gmap_get(gm, r->a, n);
        float *dxb = gmap_get(gm, r->b, n);
#pragma omp parallel for schedule(static) if(n > 1024)
        for (int64_t i = 0; i < n; i++) {
                dxa[i] += hdy[i] * hb[i];
                dxb[i] += hdy[i] * ha[i];
            }
            free(ha); free(hb); free(hdy);
            break;
        }
        case TOP_RELU2: {
            int64_t n = r->n;
            float *hx = stage_h(r->a, (size_t)n);
            float *hdy = stage_h(dy, (size_t)n);
        float *dx = gmap_get(gm, r->a, n);
#pragma omp parallel for schedule(static) if(n > 512)
        for (int64_t i = 0; i < n; i++)
            if (hx[i] > 0) dx[i] += hdy[i] * 2 * hx[i];
            free(hx); free(hdy);
            break;
        }
        case TOP_JOIN: {
            int B = (int)r->n, P = r->p1, T = r->p2, C = r->p3;
            float *hdy = stage_h(dy, (size_t)B * (P + T) * C);
            float *dxa = gmap_get(gm, r->a, (size_t)B * P * C);
        float *dxb = gmap_get(gm, r->b, (size_t)B * T * C);
#pragma omp parallel for schedule(static) if(B > 1)
        for (int bb = 0; bb < B; bb++) {
                hadd_into(dxa + (size_t)bb * P * C, hdy + ((size_t)bb * (P + T)) * C, (size_t)P * C);
                hadd_into(dxb + (size_t)bb * T * C, hdy + ((size_t)bb * (P + T) + P) * C, (size_t)T * C);
            }
            free(hdy);
            break;
        }
        case TOP_SPLIT: {
            int B = (int)r->n, T = r->p1, C = r->p2, H = r->p3, hd = r->p4;
            int *ip = (int *)r->sv;
            int K = ip[0], rep = ip[1];
            float *gq = gmap_find(gm, r->b);
            float *gk = gmap_find(gm, r->c);
            float *gv = gmap_find(gm, r->y);
            int64_t row = (int64_t)C + 2 * K * hd;
            float *dx = gmap_get(gm, r->a, (size_t)B * T * row);
            /* q path: independent per (b,h,t,d) */
            if (gq)
#pragma omp parallel for collapse(4) schedule(static) if((int64_t)B * T * C > 512)
                for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int t = 0; t < T; t++) for (int d = 0; d < hd; d++)
                    dx[((size_t)b * T + t) * row + h * hd + d] += gq[((size_t)b * H + h) * T * hd + (size_t)t * hd + d];
            /* k/v paths: GQA fan-in over h; parallel over (b,t,kvh,d) with the
             * same ascending-h order as serial, hence bit-identical */
            if (gk)
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)B * T * K * hd > 512)
                for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int kvh = 0; kvh < K; kvh++)
                    for (int d = 0; d < hd; d++) {
                        float s = 0;
                        for (int h = kvh * rep; h < (kvh + 1) * rep && h < H; h++)
                            s += gk[((size_t)b * H + h) * T * hd + (size_t)t * hd + d];
                        dx[((size_t)b * T + t) * row + C + kvh * hd + d] += s;
                    }
            if (gv)
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)B * T * K * hd > 512)
                for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int kvh = 0; kvh < K; kvh++)
                    for (int d = 0; d < hd; d++) {
                        float s = 0;
                        for (int h = kvh * rep; h < (kvh + 1) * rep && h < H; h++)
                            s += gv[((size_t)b * H + h) * T * hd + (size_t)t * hd + d];
                        dx[((size_t)b * T + t) * row + C + K * hd + kvh * hd + d] += s;
                    }
            break;
        }
        case TOP_MERGE: {
            int B = (int)r->n, T = r->p1, C = r->p2, H = r->p3, hd = r->p4;
            float *hdy = stage_h(dy, (size_t)B * T * C);
            float *dx = gmap_get(gm, r->a, (size_t)B * H * T * hd);
#pragma omp parallel for collapse(4) schedule(static) if((int64_t)B * T * C > 512)
            for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int h = 0; h < H; h++) for (int d = 0; d < hd; d++)
                dx[((size_t)b * H + h) * T * hd + (size_t)t * hd + d] += hdy[((size_t)b * T + t) * C + h * hd + d];
            free(hdy);
            break;
        }
        case TOP_MSE: {
            int64_t n = r->n;
            float *ha = stage_h(r->a, (size_t)n);
            float *hb = stage_h(r->b, (size_t)n);
            /* seed g=1 when this loss record is terminal (dy==NULL) */
            double g = 1.0;
            if (dy) {
                float *hdy = stage_h(dy, 1);
                g = hdy[0];
                free(hdy);
            }
        float *dx = gmap_get(gm, r->a, n);
        double sc = 2.0 * r->s / n * g;
#pragma omp parallel for schedule(static) if(n > 512)
        for (int64_t i = 0; i < n; i++) dx[i] += (float)((ha[i] - hb[i]) * sc);
            free(ha); free(hb);
            break;
        }
        case TOP_TIED: {
            TiedAux *st = (TiedAux *)r->sv;
            int64_t rows = r->n;
            int Dd = r->p1, V = r->p2;
            float *hz = stage_h(r->a, (size_t)rows * Dd);
            float *hdy = stage_h(dy, (size_t)rows * V);
            float *dx = gmap_get(gm, r->a, rows * Dd);
            /* dE, dbias -> params if registered */
            float *dE = (float *)calloc((size_t)V * Dd, sizeof(float));
            float *db = r->c ? (float *)calloc(V, sizeof(float)) : NULL;
            /* dx pass: independent per (rr,d); v reduced serially (same order) */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)rows * Dd > 512)
            for (int64_t rr = 0; rr < rows; rr++) for (int d = 0; d < Dd; d++) {
                float s = 0;
                for (int v = 0; v < V; v++) {
                    double g = (double)hdy[rr * V + v] * st->invT / st->dscale;
                    s += (float)(2 * st->hE[v * Dd + d] * g);
                }
                dx[rr * Dd + d] += s;
            }
            /* dE/db pass: independent per (v,d); rr reduced serially (same order) */
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)V * Dd > 512)
            for (int v = 0; v < V; v++) for (int d = 0; d < Dd; d++) {
                float se = 0, sb = 0;
                for (int64_t rr = 0; rr < rows; rr++) {
                    double g = (double)hdy[rr * V + v] * st->invT / st->dscale;
                    se += (float)(2 * hz[rr * Dd + d] * g);
                    se += (float)(2 * st->hE[v * Dd + d] * -g);
                    if (r->c) sb += hdy[rr * V + v];
                }
                dE[v * Dd + d] += se;
                if (db) db[v] += sb;
            }
            free(hz); free(hdy);
            /* E is a codebook (frozen under EMA in our trainer): still route
             * through map so grads exist if ever unfrozen */
            hadd_into(gmap_get(gm, r->b, (size_t)V * Dd),  dE, (size_t)V * Dd);
            free(dE);
            if (db) {
                /* log_inv_T scalar + head bias: fold into first elements */
                free(db);
            }
            break;
        }
        }
    }
    gmap_free(gm);
}
