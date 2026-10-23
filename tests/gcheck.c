/* tests/gcheck.c — finite-difference grad check for tape backward kernels.
 * For each op: analytic grads via tape vs numeric grads via host recompute.
 * Usage: ./gcheck (CPU only). Exits nonzero on mismatch. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/resource.h>
#include "ncm.h"

static unsigned long long rng = 0x12345;
static double frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((rng >> 11) * (1.0 / 9007199254740992.0) - 0.5);
}

static int fails = 0;
static void check(const char *nm, double analytic, double numeric) {
    double denom = fabs(analytic) + fabs(numeric) + 1e-8;
    double rel = fabs(analytic - numeric) / denom;
    if (!(rel < 5e-3)) {
        printf("FAIL %s: analytic=%.6f numeric=%.6f rel=%.2e\n", nm, analytic, numeric, rel);
        fails++;
    } else {
        printf("ok   %s: %.6f vs %.6f\n", nm, analytic, numeric);
    }
}

/* host linear + softmaxCE loss, returns loss; grads via tape compared outside */
static double host_linear_ce(const float *x, const float *W, const float *b,
                             int B, int M, int K) {
    double nll = 0;
    for (int r = 0; r < B; r++) {
        double mx = -1e300;
        static double lg[1024];
        for (int m = 0; m < M; m++) {
            double s = b ? b[m] : 0;
            for (int k = 0; k < K; k++) s += (double)x[r * K + k] * W[m * K + k];
            lg[m] = s;
            if (s > mx) mx = s;
        }
        double den = 0;
        for (int m = 0; m < M; m++) { lg[m] = exp(lg[m] - mx); den += lg[m]; }
        nll += -log(lg[0] / den + 1e-30);
    }
    return nll / B;
}

int main(void) {
    arena_init(1ULL << 30);
    const double eps = 1e-2; /* float32-friendly step */

    /* ---- linear + CE ---- */
    {
        int B = 3, M = 4, K = 5;
        float *hx = malloc(B * K * 4), *hW = malloc(M * K * 4), *hb = malloc(M * 4);
        for (int i = 0; i < B * K; i++) hx[i] = (float)frand();
        for (int i = 0; i < M * K; i++) hW[i] = (float)frand();
        for (int i = 0; i < M; i++) hb[i] = (float)frand();
        float *x = arena_alloc(B * K), *W = arena_alloc(M * K), *b = arena_alloc(M);
        mem_h2d(x, hx, B * K * 4); mem_h2d(W, hW, M * K * 4); mem_h2d(b, hb, M * 4);
        float *y = arena_alloc(B * M);
        int *tgt = (int *)arena_alloc(B);
        mem_h2d(tgt, (int[]){0, 0, 0}, B * 4);
        Tape *tp = tape_new();
        t_linear(tp, x, W, b, y, B, M, K);
        float *loss = arena_alloc(1), *probs = arena_alloc(B * M * M);
        /* CE over B rows of width M (targets all 0) */
        t_ce(tp, y, (int *)tgt, NULL, loss, probs, B, M, 1.0);
        float *gW = calloc(M * K, 4), *gb = calloc(M, 4), *gx = calloc(B * K, 4);
        PMap pm;
        float *pp[3] = {W, b, x};
        float *pg[3] = {gW, gb, gx};
        int64_t sz[3] = {M * K, M, B * K};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 3;
        tape_backward(tp, &pm, 0);
        /* numeric */
        double f0 = host_linear_ce(hx, hW, hb, B, M, K);
        (void)f0;
        /* dW[7] */
        {
            float hW2[20];
            memcpy(hW2, hW, sizeof(hW2));
            hW2[7] += eps;
            double fp = host_linear_ce(hx, hW2, hb, B, M, K);
            hW2[7] -= 2 * eps;
            double fm = host_linear_ce(hx, hW2, hb, B, M, K);
            check("linear dW[7]", gW[7], (fp - fm) / (2 * eps));
        }
        /* db[1] */
        {
            float hb2[4];
            memcpy(hb2, hb, sizeof(hb2));
            hb2[1] += eps;
            double fp = host_linear_ce(hx, hW, hb2, B, M, K);
            hb2[1] -= 2 * eps;
            double fm = host_linear_ce(hx, hW, hb2, B, M, K);
            check("linear db[1]", gb[1], (fp - fm) / (2 * eps));
        }
        /* dx[4] */
        {
            float hx2[15];
            memcpy(hx2, hx, sizeof(hx2));
            hx2[4] += eps;
            double fp = host_linear_ce(hx2, hW, hb, B, M, K);
            hx2[4] -= 2 * eps;
            double fm = host_linear_ce(hx2, hW, hb, B, M, K);
            check("linear dx[4]", gx[4], (fp - fm) / (2 * eps));
        }
        free(hx); free(hW); free(hb);
        free(gW); free(gb); free(gx);
        tape_free(tp);
    }

    /* ---- layernorm + sum loss ---- */
    {
        int R = 4, C = 6;
        float *hx = malloc(R * C * 4), *hw = malloc(C * 4);
        for (int i = 0; i < R * C; i++) hx[i] = (float)frand();
        for (int i = 0; i < C; i++) hw[i] = 0.5f + (float)frand() * 0.2f;
        float *x = arena_alloc(R * C), *w = arena_alloc(C), *y = arena_alloc(R * C);
        mem_h2d(x, hx, R * C * 4); mem_h2d(w, hw, C * 4);
        Tape *tp = tape_new();
        t_layernorm(tp, x, w, NULL, y, R, C);
        /* scalar loss = sum(y): emulate via CE? NO — tape_backward only seeds
         * CE records. Workaround: t_ce over flattened y as single row with
         * large V? CE needs probs... Instead seed manually: our tape_backward
         * skips non-CE records without grads. HACK for test: append a CE
         * with V = R*C over y viewed as [1][R*C]? CE normalizes — changes
         * grads (chain rule still valid: dL/dy from CE backward is exact).
         * So: loss = CE(y_flat); analytic dy flows correctly; numeric
         * perturbs x and recomputes full forward+loss on host. */
        int V = R * C;
        int *tgt = (int *)arena_alloc(1);
        int zero = 0;
        mem_h2d(tgt, &zero, 4);
        float *loss = arena_alloc(1), *probs = arena_alloc(V);
        t_ce(tp, y, tgt, NULL, loss, probs, 1, V, 1.0);
        float *gx = calloc(R * C, 4), *gw = calloc(C, 4);
        PMap pm;
        float *pp[2] = {x, w};
        float *pg[2] = {gx, gw};
        int64_t sz[2] = {R * C, C};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 2;
        tape_backward(tp, &pm, 0);
        /* numeric: full host forward + same CE */
        double full_loss(const float *xx, const float *ww) {
            /* layernorm then CE over flattened */
            static double lg[1024 * 8];
            for (int r = 0; r < R; r++) {
                double mean = 0;
                for (int c = 0; c < C; c++) mean += xx[r * C + c];
                mean /= C;
                double var = 0;
                for (int c = 0; c < C; c++) { double d = xx[r * C + c] - mean; var += d * d; }
                var /= C;
                double inv = 1.0 / sqrt(var + 1e-6);
                for (int c = 0; c < C; c++) lg[r * C + c] = (xx[r * C + c] - mean) * inv * ww[c];
            }
            double mx = lg[0];
            for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
            double den = 0;
            for (int i = 0; i < V; i++) { lg[i] = exp(lg[i] - mx); den += lg[i]; }
            return -log(lg[0] / den + 1e-30);
        }
        {
            float hx2[24];
            memcpy(hx2, hx, sizeof(hx2));
            hx2[7] += eps;
            double fp = full_loss(hx2, hw);
            hx2[7] -= 2 * eps;
            double fm = full_loss(hx2, hw);
            check("layernorm dx[7]", gx[7], (fp - fm) / (2 * eps));
        }
        {
            float hw2[6];
            memcpy(hw2, hw, sizeof(hw2));
            hw2[2] += eps;
            double fp = full_loss(hx, hw2);
            hw2[2] -= 2 * eps;
            double fm = full_loss(hx, hw2);
            check("layernorm dw[2]", gw[2], (fp - fm) / (2 * eps));
        }
        free(hx); free(hw); free(gx); free(gw);
        tape_free(tp);
    }

    /* ---- qkv-split + per-head MSE (row-stride regression) ----
     * Catches TOP_SPLIT backward indexing dx with row stride C instead
     * of W=C+2*K*hd (writes land in wrong rows for t>0 and collide with
     * other regions; t=0 q-slots pass by luck since 0*C==0*W). Two
     * configs: rep=1 and GQA rep=2 (kvh fan-in). Indices span q/k/v and
     * rows 0..3, including t>=1. */
    for (int cfg = 0; cfg < 2; cfg++) {
        int B = 1, S = 4, C = 8, H = 2, hd = 4;
        int K = cfg ? 1 : 2, rep = H / K;
        int W = C + 2 * K * hd;
        int nin = B * S * W, nqk = B * H * S * hd;
        float *hx = malloc(nin * 4);
        for (int i = 0; i < nin; i++) hx[i] = (float)frand();
        float *in = arena_alloc(nin);
        float *qq = arena_alloc(nqk), *kk = arena_alloc(nqk), *vv = arena_alloc(nqk);
        mem_h2d(in, hx, nin * 4);
        Tape *tp = tape_new();
        t_qkvsplit(tp, in, qq, kk, vv, B, S, C, H, hd, K, rep);
        float *h0 = malloc(nqk * 4);
        float *tq = arena_alloc(nqk), *tk = arena_alloc(nqk), *tv = arena_alloc(nqk);
        for (int i = 0; i < nqk; i++) h0[i] = 1.0f;
        mem_h2d(tq, h0, nqk * 4);
        for (int i = 0; i < nqk; i++) h0[i] = 2.0f;
        mem_h2d(tk, h0, nqk * 4);
        for (int i = 0; i < nqk; i++) h0[i] = 3.0f;
        mem_h2d(tv, h0, nqk * 4);
        free(h0);
        float *lq = arena_alloc(1), *lk = arena_alloc(1), *lv = arena_alloc(1);
        t_mse(tp, qq, tq, lq, nqk, 1.0);
        t_mse(tp, kk, tk, lk, nqk, 1.0);
        t_mse(tp, vv, tv, lv, nqk, 1.0);
        float *gin = calloc(nin, 4);
        PMap pm;
        float *pp[1] = {in};
        float *pg[1] = {gin};
        int64_t sz[1] = {nin};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 1;
        tape_backward(tp, &pm, 0);
        int idxs[4];
        idxs[0] = 3;                            /* row0 q */
        idxs[1] = 1 * W + C + 2;                 /* row1 k (kvh 0) */
        idxs[2] = 2 * W + C + K * hd + 1;       /* row2 v */
        idxs[3] = 3 * W + (H - 1) * hd + 1;     /* row3 q, last head */
        for (int ti = 0; ti < 4; ti++) {
            int idx = idxs[ti];
            double ana = gin[idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nin * 4);
                memcpy(tx, hx, nin * 4);
                tx[idx] += (float)(sgn * eps);
                double s = 0;
                for (int b = 0; b < B; b++) for (int h = 0; h < H; h++)
                    for (int tt = 0; tt < S; tt++) for (int d = 0; d < hd; d++) {
                        int kvh = h / rep;
                        double qv = tx[((b * S + tt) * W) + h * hd + d];
                        double kv = tx[((b * S + tt) * W) + C + kvh * hd + d];
                        double vv2 = tx[((b * S + tt) * W) + C + K * hd + kvh * hd + d];
                        s += (qv - 1.0) * (qv - 1.0) + (kv - 2.0) * (kv - 2.0) +
                             (vv2 - 3.0) * (vv2 - 3.0);
                    }
                v[sgn > 0 ? 1 : 0] = s / nqk;
                free(tx);
            }
            char nm[64];
            snprintf(nm, sizeof(nm), "qkv-split cfg%d din[%d]", cfg, idx);
            check(nm, ana, (v[1] - v[0]) / (2 * eps));
        }
        free(hx); free(gin);
        tape_free(tp);
    }

    /* ---- headsplit (B-aware heads transpose + inverse-permute bwd) ---- */
    {
        int B = 2, T = 4, H = 2, hd = 4, C = H * hd;
        int nin = B * T * C, nout = B * H * T * hd;
        float *hx = malloc(nin * 4);
        for (int i = 0; i < nin; i++) hx[i] = (float)frand();
        float *in = arena_alloc(nin), *out = arena_alloc(nout);
        mem_h2d(in, hx, nin * 4);
        Tape *tp = tape_new();
        t_headsplit(tp, in, out, B, T, C, H, hd);
        float *h0 = malloc(nout * 4);
        float *tc = arena_alloc(nout);
        for (int i = 0; i < nout; i++) h0[i] = 1.0f;
        mem_h2d(tc, h0, nout * 4);
        free(h0);
        float *ll = arena_alloc(1);
        t_mse(tp, out, tc, ll, nout, 1.0);
        float *gin = calloc(nin, 4);
        PMap pm;
        float *pp[1] = {in};
        float *pg[1] = {gin};
        int64_t sz[1] = {nin};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 1;
        tape_backward(tp, &pm, 0);
        int idxs[3] = {0, nin / 2 + 3, nin - 1};
        for (int ti = 0; ti < 3; ti++) {
            int idx = idxs[ti];
            double v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nin * 4);
                memcpy(tx, hx, nin * 4);
                tx[idx] += (float)(sgn * eps);
                double s = 0;
                for (int b = 0; b < B; b++) for (int t = 0; t < T; t++)
                    for (int h = 0; h < H; h++) for (int d = 0; d < hd; d++) {
                        double o = tx[((b * T + t) * C) + h * hd + d];
                        s += (o - 1.0) * (o - 1.0);
                    }
                v[sgn > 0 ? 1 : 0] = s / nout;
                free(tx);
            }
            char nm[64];
            snprintf(nm, sizeof(nm), "headsplit din[%d]", idx);
            check(nm, gin[idx], (v[1] - v[0]) / (2 * eps));
        }
        free(hx); free(gin);
        tape_free(tp);
    }

    /* ---- rope GQA (H=4 query heads, K=2 key heads) ---- */
    {
        int B = 1, H = 4, K = 2, T = 4, hd = 8, h = hd / 2;
        int nq = B * H * T * hd, nk = B * K * T * hd, nt = T * hd;
        float *hxq = malloc(nq * 4), *hxk = malloc(nk * 4);
        float *hco = malloc(nt * 4), *hsi = malloc(nt * 4);
        for (int i = 0; i < nq; i++) hxq[i] = (float)frand();
        for (int i = 0; i < nk; i++) hxk[i] = (float)frand();
        for (int i = 0; i < nt; i++) { hco[i] = (float)frand(); hsi[i] = (float)frand(); }
        float *xq = arena_alloc(nq), *xk = arena_alloc(nk);
        float *co = arena_alloc(nt), *si = arena_alloc(nt);
        float *yq = arena_alloc(nq), *yk = arena_alloc(nk);
        mem_h2d(xq, hxq, nq * 4); mem_h2d(xk, hxk, nk * 4);
        mem_h2d(co, hco, nt * 4); mem_h2d(si, hsi, nt * 4);
        Tape *tp = tape_new();
        t_rope(tp, xq, xk, yq, yk, co, si, B, H, K, T, hd);
        float *h0 = malloc((nq > nk ? nq : nk) * 4);
        float *tq = arena_alloc(nq), *tk = arena_alloc(nk);
        for (int i = 0; i < nq; i++) h0[i] = 1.0f;
        mem_h2d(tq, h0, nq * 4);
        for (int i = 0; i < nk; i++) h0[i] = 2.0f;
        mem_h2d(tk, h0, nk * 4);
        free(h0);
        float *lq = arena_alloc(1), *lk = arena_alloc(1);
        t_mse(tp, yq, tq, lq, nq, 1.0);
        t_mse(tp, yk, tk, lk, nk, 1.0);
        float *gxq = calloc(nq, 4), *gxk = calloc(nk, 4);
        PMap pm;
        float *pp[2] = {xq, xk};
        float *pg[2] = {gxq, gxk};
        int64_t sz[2] = {nq, nk};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 2;
        tape_backward(tp, &pm, 0);
        {
            int idx = 3;
            double v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nq * 4);
                memcpy(tx, hxq, nq * 4);
                tx[idx] += (float)(sgn * eps);
                double s = 0;
                for (int rr = 0; rr < B * H * T; rr++) {
                    int t = rr % T;
                    for (int d = 0; d < h; d++) {
                        double a = tx[rr * hd + d], b = tx[rr * hd + h + d];
                        double y0 = a * hco[t * hd + d] - b * hsi[t * hd + d];
                        double y1 = b * hco[t * hd + h + d] + a * hsi[t * hd + h + d];
                        s += (y0 - 1.0) * (y0 - 1.0) + (y1 - 1.0) * (y1 - 1.0);
                    }
                }
                for (int rr = 0; rr < B * K * T; rr++) {
                    int t = rr % T;
                    for (int d = 0; d < h; d++) {
                        double a = hxk[rr * hd + d], b = hxk[rr * hd + h + d];
                        double y0 = a * hco[t * hd + d] - b * hsi[t * hd + d];
                        double y1 = b * hco[t * hd + h + d] + a * hsi[t * hd + h + d];
                        s += (y0 - 2.0) * (y0 - 2.0) + (y1 - 2.0) * (y1 - 2.0);
                    }
                }
                v[sgn > 0 ? 1 : 0] = s / nq;
                free(tx);
            }
            /* lq = sum/nq over yq only; the yk part is constant in xq
             * and drops out of the finite difference. */
            check("rope_gqa dxq[3]", gxq[idx], (v[1] - v[0]) / (2 * eps));
        }
        {
            int idx = 5;
            double v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nk * 4);
                memcpy(tx, hxk, nk * 4);
                tx[idx] += (float)(sgn * eps);
                double s = 0;
                for (int rr = 0; rr < B * K * T; rr++) {
                    int t = rr % T;
                    for (int d = 0; d < h; d++) {
                        double a = tx[rr * hd + d], b = tx[rr * hd + h + d];
                        double y0 = a * hco[t * hd + d] - b * hsi[t * hd + d];
                        double y1 = b * hco[t * hd + h + d] + a * hsi[t * hd + h + d];
                        s += (y0 - 2.0) * (y0 - 2.0) + (y1 - 2.0) * (y1 - 2.0);
                    }
                }
                v[sgn > 0 ? 1 : 0] = s / nk;
                free(tx);
            }
            check("rope_gqa dxk[5]", gxk[idx], (v[1] - v[0]) / (2 * eps));
        }
        free(hxq); free(hxk); free(hco); free(hsi);
        free(gxq); free(gxk);
        tape_free(tp);
    }

    /* ---- sdpa GQA (H=4 query heads, K=2 kv heads, no mask) ---- */
    {
        int B = 1, H = 4, K = 2, Tq = 4, Tkv = 4, hd = 8;
        int rep = H / K;
        int nq = B * H * Tq * hd, nkv = B * K * Tkv * hd;
        float scale = 1.0f / sqrtf((float)hd);
        float *hq0 = malloc(nq * 4), *hk0 = malloc(nkv * 4), *hv0 = malloc(nkv * 4);
        for (int i = 0; i < nq; i++) hq0[i] = (float)frand();
        for (int i = 0; i < nkv; i++) { hk0[i] = (float)frand(); hv0[i] = (float)frand(); }
        float *q = arena_alloc(nq), *k = arena_alloc(nkv), *v = arena_alloc(nkv);
        float *o = arena_alloc(nq);
        mem_h2d(q, hq0, nq * 4); mem_h2d(k, hk0, nkv * 4); mem_h2d(v, hv0, nkv * 4);
        Tape *tp = tape_new();
        t_sdpa(tp, q, k, v, NULL, o, B, H, K, Tq, Tkv, hd);
        float *h1 = malloc(nq * 4);
        float *tc = arena_alloc(nq);
        for (int i = 0; i < nq; i++) h1[i] = 1.0f;
        mem_h2d(tc, h1, nq * 4);
        free(h1);
        float *ll = arena_alloc(1);
        t_mse(tp, o, tc, ll, nq, 1.0);
        float *gq = calloc(nq, 4), *gk = calloc(nkv, 4), *gv = calloc(nkv, 4);
        PMap pm;
        float *pp[3] = {q, k, v};
        float *pg[3] = {gq, gk, gv};
        int64_t sz[3] = {nq, nkv, nkv};
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = 3;
        tape_backward(tp, &pm, 0);
        /* numeric: GQA sdpa forward + MSE vs 1.0, finite differences */
        double sdpa_loss(const float *qq, const float *kk, const float *vv) {
            double s = 0;
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
                int kvh = h / rep;
                for (int tq = 0; tq < Tq; tq++) {
                    double mx = -1e300;
                    static double sc[16];
                    for (int tk = 0; tk < Tkv; tk++) {
                        const float *qr = qq + ((b * H + h) * Tq + tq) * hd;
                        const float *kr = kk + ((b * K + kvh) * Tkv + tk) * hd;
                        double d = 0;
                        for (int d2 = 0; d2 < hd; d2++) d += (double)qr[d2] * kr[d2];
                        d *= scale;
                        sc[tk] = d;
                        if (d > mx) mx = d;
                    }
                    double den = 0;
                    for (int tk = 0; tk < Tkv; tk++) { sc[tk] = exp(sc[tk] - mx); den += sc[tk]; }
                    for (int d = 0; d < hd; d++) {
                        double ov = 0;
                        for (int tk = 0; tk < Tkv; tk++) {
                            const float *vr = vv + ((b * K + kvh) * Tkv + tk) * hd;
                            ov += (sc[tk] / den) * vr[d];
                        }
                        s += (ov - 1.0) * (ov - 1.0);
                    }
                }
            }
            return s / nq;
        }
        {
            int idx = 7;
            double ana = gq[idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nq * 4);
                memcpy(tx, hq0, nq * 4);
                tx[idx] += (float)(sgn * eps);
                v[sgn > 0 ? 1 : 0] = sdpa_loss(tx, hk0, hv0);
                free(tx);
            }
            check("sdpa_gqa dq[7]", ana, (v[1] - v[0]) / (2 * eps));
        }
        {
            int idx = 3;
            double ana = gk[idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nkv * 4);
                memcpy(tx, hk0, nkv * 4);
                tx[idx] += (float)(sgn * eps);
                v[sgn > 0 ? 1 : 0] = sdpa_loss(hq0, tx, hv0);
                free(tx);
            }
            check("sdpa_gqa dk[3]", ana, (v[1] - v[0]) / (2 * eps));
        }
        {
            int idx = 11;
            double ana = gv[idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                float *tx = malloc(nkv * 4);
                memcpy(tx, hv0, nkv * 4);
                tx[idx] += (float)(sgn * eps);
                v[sgn > 0 ? 1 : 0] = sdpa_loss(hq0, hk0, tx);
                free(tx);
            }
            check("sdpa_gqa dv[11]", ana, (v[1] - v[0]) / (2 * eps));
        }
        free(hq0); free(hk0); free(hv0);
        free(gq); free(gk); free(gv);
        tape_free(tp);
    }

    /* ---- taped Qwen block (1 layer, synthetic) + CE, analytic vs numeric ----
     * Exercises the real train_qwen_forward end to end: embed, rmsnorm,
     * q/k/v+bias, headsplit, rope_gqa, sdpa_gqa, merge, swiglu, tied head.
     * Numeric side re-runs the same taped forward with perturbed weights
     * (validates the backward only — that is the point of a gradcheck). */
    {
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
        WeightStore qws;
        qws.items = (NamedTensor *)calloc(NW, sizeof(NamedTensor));
        qws.count = NW;
        float **hostw = (float **)malloc(NW * sizeof(float *));
        for (int i = 0; i < NW; i++) {
            int n = spec[i].rows * spec[i].cols;
            hostw[i] = (float *)malloc(n * sizeof(float));
            for (int k = 0; k < n; k++) hostw[i][k] = (float)frand();
            Tensor *t = tensor_alloc(2, (int64_t[]){spec[i].rows, spec[i].cols});
            mem_h2d(t->data, hostw[i], n * sizeof(float));
            snprintf(qws.items[i].name, sizeof(qws.items[i].name), "%s", spec[i].nm);
            qws.items[i].t = t;
        }
        int hids[4] = {7, 42, 100, 5};
        int htg[4] = {3, 200, 17, 99};
        Tape *tp = tape_new();
        float *logits = train_qwen_forward(tp, &qws, &qc, hids, B, T);
        float *loss = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
        /* t_ce takes HOST targets (copies internally) */
        t_ce(tp, logits, htg, NULL, loss, probs, (int64_t)B * T, V, 1.0);
        float **gr = (float **)malloc(NW * sizeof(float *));
        PMap pm;
        float **pp = (float **)malloc(NW * sizeof(float *));
        float **pg = (float **)malloc(NW * sizeof(float *));
        int64_t *sz = (int64_t *)malloc(NW * sizeof(int64_t));
        for (int i = 0; i < NW; i++) {
            pp[i] = qws.items[i].t->data;
            gr[i] = (float *)calloc(qws.items[i].t->n, sizeof(float));
            pg[i] = gr[i];
            sz[i] = qws.items[i].t->n;
        }
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = NW;
        tape_backward(tp, &pm, 0);
        tape_free(tp);
        /* numeric: perturb host copy, push to backend, rerun taped forward */
        double qwen_loss_at(float **hw) {
            for (int i = 0; i < NW; i++) {
                int n = spec[i].rows * spec[i].cols;
                mem_h2d(qws.items[i].t->data, hw[i], n * sizeof(float));
            }
            Tape *t2 = tape_new();
            float *lg2 = train_qwen_forward(t2, &qws, &qc, hids, B, T);
            float *lo2 = arena_alloc(1), *pr2 = arena_alloc((size_t)B * T * V);
            t_ce(t2, lg2, htg, NULL, lo2, pr2, (int64_t)B * T, V, 1.0);
            float lh = 0;
            mem_d2h(&lh, lo2, sizeof(float));
            tape_free(t2);
            return lh;
        }
        /* restore backend = unperturbed before numeric loop */
        for (int i = 0; i < NW; i++) {
            int n = spec[i].rows * spec[i].cols;
            mem_h2d(qws.items[i].t->data, hostw[i], n * sizeof(float));
        }
        struct { int wi; const char *nm; } checks[] = {
            {3, "qwen q_proj.weight"}, {4, "qwen q_proj.bias"},
            {8, "qwen v_proj.weight"}, {11, "qwen gate_proj.weight"},
            {13, "qwen down_proj.weight"}, {1, "qwen lnf.weight"},
        };
        for (unsigned ci = 0; ci < sizeof(checks) / sizeof(checks[0]); ci++) {
            int wi = checks[ci].wi;
            /* check the largest-magnitude entry: finite differences on
             * ~1e-4 grads drown in float32 cancellation, so probe the entry
             * that actually drives the update (deterministic under the
             * fixed test RNG). */
            int idx = 0;
            double amx = 0;
            for (int64_t k = 0; k < sz[wi]; k++) {
                double a = fabs(gr[wi][k]);
                if (a > amx) { amx = a; idx = (int)k; }
            }
            double ana = gr[wi][idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                hostw[wi][idx] += (float)(sgn * eps);
                v[sgn > 0 ? 1 : 0] = qwen_loss_at(hostw);
                hostw[wi][idx] -= (float)(sgn * eps);
            }
            char nm[64];
            snprintf(nm, sizeof(nm), "%s[%d]", checks[ci].nm, idx);
            check(nm, ana, (v[1] - v[0]) / (2 * eps));
        }
        for (int i = 0; i < NW; i++) { free(hostw[i]); free(gr[i]); }
        free(hostw); free(gr); free(pp); free(pg); free(sz);
        free(qws.items);
    }

    /* ---- taped GPT-2 block (1 layer, synthetic) + CE, analytic vs numeric --
     * Exercises the real train_gpt2_forward end to end: wte+wpe embed,
     * layernorm(eps=1e-5), c_attn(+bias), qkv split, causal sdpa, merge,
     * c_proj, gelu_new(tanh), tied head. Numeric side re-runs the same
     * taped forward with perturbed weights (validates the backward only). */
    {
        int B = 1, T = 4, C = 32, H = 4, I = 128, V = 256;
        NCMConfig gc;
        memset(&gc, 0, sizeof(gc));
        gc.n_layer = 1; gc.n_head = H; gc.n_embd = C;
        gc.context_length = 64;
        struct { const char *nm; int rows, cols; } spec[] = {
            {"wte.weight", V, C},
            {"wpe.weight", 64, C},
            {"ln_f.weight", 1, C},
            {"ln_f.bias", 1, C},
            {"h.0.ln_1.weight", 1, C},
            {"h.0.ln_1.bias", 1, C},
            {"h.0.attn.c_attn.weight", 3 * C, C},
            {"h.0.attn.c_attn.bias", 1, 3 * C},
            {"h.0.attn.c_proj.weight", C, C},
            {"h.0.attn.c_proj.bias", 1, C},
            {"h.0.ln_2.weight", 1, C},
            {"h.0.ln_2.bias", 1, C},
            {"h.0.mlp.c_fc.weight", I, C},
            {"h.0.mlp.c_fc.bias", 1, I},
            {"h.0.mlp.c_proj.weight", C, I},
            {"h.0.mlp.c_proj.bias", 1, C},
        };
        int NW = (int)(sizeof(spec) / sizeof(spec[0]));
        WeightStore gws;
        gws.items = (NamedTensor *)calloc(NW, sizeof(NamedTensor));
        gws.count = NW;
        float **hostw = (float **)malloc(NW * sizeof(float *));
        for (int i = 0; i < NW; i++) {
            int n = spec[i].rows * spec[i].cols;
            hostw[i] = (float *)malloc(n * sizeof(float));
            for (int k = 0; k < n; k++) hostw[i][k] = (float)frand();
            Tensor *t = tensor_alloc(2, (int64_t[]){spec[i].rows, spec[i].cols});
            mem_h2d(t->data, hostw[i], n * sizeof(float));
            snprintf(gws.items[i].name, sizeof(gws.items[i].name), "%s", spec[i].nm);
            gws.items[i].t = t;
        }
        int hids[4] = {7, 42, 100, 5};
        int htg[4] = {3, 200, 17, 99};
        Tape *tp = tape_new();
        float *logits = train_gpt2_forward(tp, &gws, &gc, hids, B, T);
        float *loss = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
        /* t_ce takes HOST targets (copies internally) */
        t_ce(tp, logits, htg, NULL, loss, probs, (int64_t)B * T, V, 1.0);
        float **gr = (float **)malloc(NW * sizeof(float *));
        PMap pm;
        float **pp = (float **)malloc(NW * sizeof(float *));
        float **pg = (float **)malloc(NW * sizeof(float *));
        int64_t *sz = (int64_t *)malloc(NW * sizeof(int64_t));
        for (int i = 0; i < NW; i++) {
            pp[i] = gws.items[i].t->data;
            gr[i] = (float *)calloc(gws.items[i].t->n, sizeof(float));
            pg[i] = gr[i];
            sz[i] = gws.items[i].t->n;
        }
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = NW;
        tape_backward(tp, &pm, 0);
        tape_free(tp);
        /* numeric: perturb host copy, push to backend, rerun taped forward */
        double gpt2_loss_at(float **hw) {
            for (int i = 0; i < NW; i++) {
                int n = spec[i].rows * spec[i].cols;
                mem_h2d(gws.items[i].t->data, hw[i], n * sizeof(float));
            }
            Tape *t2 = tape_new();
            float *lg2 = train_gpt2_forward(t2, &gws, &gc, hids, B, T);
            float *lo2 = arena_alloc(1), *pr2 = arena_alloc((size_t)B * T * V);
            t_ce(t2, lg2, htg, NULL, lo2, pr2, (int64_t)B * T, V, 1.0);
            float lh = 0;
            mem_d2h(&lh, lo2, sizeof(float));
            tape_free(t2);
            return lh;
        }
        /* restore backend = unperturbed before numeric loop */
        for (int i = 0; i < NW; i++) {
            int n = spec[i].rows * spec[i].cols;
            mem_h2d(gws.items[i].t->data, hostw[i], n * sizeof(float));
        }
        struct { int wi; const char *nm; } checks[] = {
            {6, "gpt2 c_attn.weight"}, {7, "gpt2 c_attn.bias"},
            {8, "gpt2 c_proj.weight"}, {12, "gpt2 c_fc.weight"},
            {14, "gpt2 mlp proj.weight"}, {2, "gpt2 ln_f.weight"},
            {4, "gpt2 ln_1.weight"},
        };
        for (unsigned ci = 0; ci < sizeof(checks) / sizeof(checks[0]); ci++) {
            int wi = checks[ci].wi;
            /* check the largest-magnitude entry: finite differences on
             * ~1e-4 grads drown in float32 cancellation, so probe the entry
             * that actually drives the update (deterministic under the
             * fixed test RNG). */
            int idx = 0;
            double amx = 0;
            for (int64_t k = 0; k < sz[wi]; k++) {
                double a = fabs(gr[wi][k]);
                if (a > amx) { amx = a; idx = (int)k; }
            }
            double ana = gr[wi][idx], v[2];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                hostw[wi][idx] += (float)(sgn * eps);
                v[sgn > 0 ? 1 : 0] = gpt2_loss_at(hostw);
                hostw[wi][idx] -= (float)(sgn * eps);
            }
            char nm[64];
            snprintf(nm, sizeof(nm), "%s[%d]", checks[ci].nm, idx);
            check(nm, ana, (v[1] - v[0]) / (2 * eps));
        }
        for (int i = 0; i < NW; i++) { free(hostw[i]); free(gr[i]); }
        free(hostw); free(gr); free(pp); free(pg); free(sz);
        free(gws.items);
    }

    /* ---- tape host-memory regression gate (MISTAKES.md #3) ----
     * t_ce once orphaned its rows*V softmax buffer (~25 MiB at GPT-2
     * T=128) on EVERY training step, and aux structs leaked members —
     * ~260 GiB cumulative before the kernel OOM-killed a 12k-step run.
     * Loop full forward+CE+backward+clear K times at the real V size and
     * bound ru_maxrss growth: a leak adds K*hp (250 MiB here); the fixed
     * tape adds ~0. Weight-free, CPU-only, seconds. */
    {
        /* ru_maxrss is a high-water mark: one-time OpenMP pool creation
         * (~tens of MiB) must not count as growth. The warmup runs below,
         * after setup, before ru0 is read. */
        struct rusage ru0, ru1;
        int B = 1, T = 128, C = 16, H = 2, I = 64, V = 50257;
        NCMConfig lc;
        memset(&lc, 0, sizeof(lc));
        lc.n_layer = 1; lc.n_head = H; lc.n_embd = C;
        lc.context_length = 256;
        struct { const char *nm; int rows, cols; } spec[] = {
            {"wte.weight", V, C},
            {"wpe.weight", 256, C},
            {"ln_f.weight", 1, C},
            {"ln_f.bias", 1, C},
            {"h.0.ln_1.weight", 1, C},
            {"h.0.ln_1.bias", 1, C},
            {"h.0.attn.c_attn.weight", 3 * C, C},
            {"h.0.attn.c_attn.bias", 1, 3 * C},
            {"h.0.attn.c_proj.weight", C, C},
            {"h.0.attn.c_proj.bias", 1, C},
            {"h.0.ln_2.weight", 1, C},
            {"h.0.ln_2.bias", 1, C},
            {"h.0.mlp.c_fc.weight", I, C},
            {"h.0.mlp.c_fc.bias", 1, I},
            {"h.0.mlp.c_proj.weight", C, I},
            {"h.0.mlp.c_proj.bias", 1, C},
        };
        int NW = (int)(sizeof(spec) / sizeof(spec[0]));
        WeightStore lws;
        lws.items = (NamedTensor *)calloc(NW, sizeof(NamedTensor));
        lws.count = NW;
        for (int i = 0; i < NW; i++) {
            int n = spec[i].rows * spec[i].cols;
            float *hw = (float *)malloc(n * sizeof(float));
            for (int k = 0; k < n; k++) hw[k] = (float)frand() * 0.1f;
            Tensor *t = tensor_alloc(2, (int64_t[]){spec[i].rows, spec[i].cols});
            mem_h2d(t->data, hw, n * sizeof(float));
            free(hw);
            snprintf(lws.items[i].name, sizeof(lws.items[i].name), "%s", spec[i].nm);
            lws.items[i].t = t;
        }
        int *hids = (int *)malloc(T * sizeof(int));
        int *htg = (int *)malloc(T * sizeof(int));
        for (int t = 0; t < T; t++) { hids[t] = t % V; htg[t] = (t * 7) % V; }
        PMap pm;
        float **pp = (float **)malloc(NW * sizeof(float *));
        float **pg = (float **)malloc(NW * sizeof(float *));
        int64_t *sz = (int64_t *)malloc(NW * sizeof(int64_t));
        for (int i = 0; i < NW; i++) {
            pp[i] = lws.items[i].t->data;
            pg[i] = (float *)calloc(lws.items[i].t->n, sizeof(float));
            sz[i] = lws.items[i].t->n;
        }
        pm.params = pp; pm.grads = pg; pm.sizes = sz; pm.n = NW;
        for (int k = 0; k < 2; k++) {
            arena_reset();
            Tape *tp = tape_new();
            float *logits = train_gpt2_forward(tp, &lws, &lc, hids, B, T);
            float *loss = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
            t_ce(tp, logits, htg, NULL, loss, probs, (int64_t)B * T, V, 1.0);
            tape_backward(tp, &pm, 0);
            tape_free(tp);
        }
        getrusage(RUSAGE_SELF, &ru0);
        for (int k = 0; k < 10; k++) {
            /* mirror train_gpt2_step exactly: clear tape AND reset arena.
             * Without the reset, bump-allocated forward buffers accumulate
             * by design and the gate would false-positive. */
            arena_reset();
            Tape *tp = tape_new();
            float *logits = train_gpt2_forward(tp, &lws, &lc, hids, B, T);
            float *loss = arena_alloc(1), *probs = arena_alloc((size_t)B * T * V);
            t_ce(tp, logits, htg, NULL, loss, probs, (int64_t)B * T, V, 1.0);
            tape_backward(tp, &pm, 0);
            tape_free(tp); /* recs + owned members; grads reuse is fine */
        }
        getrusage(RUSAGE_SELF, &ru1);
        /* ru_maxrss is KiB on Linux: 10 iters x 25 MiB leak = +250 MiB
         * if the bug returns; bound at +128 MiB. */
        long dkb = (long)ru1.ru_maxrss - (long)ru0.ru_maxrss;
        if (dkb < 128 * 1024)
            printf("ok   tape-rss flat across 10 steps (maxrss +%ld KiB)\n", dkb);
        else {
            printf("FAIL tape-rss grew +%ld KiB over 10 steps (leak)\n", dkb);
            fails++;
        }
        free(hids); free(htg);
        for (int i = 0; i < NW; i++) free(pg[i]);
        free(pp); free(pg); free(sz);
        free(lws.items);
    }

    printf(fails ? "GCHECK-FAIL (%d)\n" : "GCHECK-PASS\n", fails);
    return fails != 0;
}

/* appended by builder: more op checks live here; see train validation below */
