/* tests/gcheck.c — finite-difference grad check for tape backward kernels.
 * For each op: analytic grads via tape vs numeric grads via host recompute.
 * Usage: ./gcheck (CPU only). Exits nonzero on mismatch. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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

    printf(fails ? "GCHECK-FAIL (%d)\n" : "GCHECK-PASS\n", fails);
    return fails != 0;
}

/* appended by builder: more op checks live here; see train validation below */
