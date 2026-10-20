/* ktest: exercise every kernel on fixed pseudo-random data, print checksums.
 * Compile CPU + CUDA, diff outputs. Uses backend allocators via tensor.c. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ncm.h"

static unsigned long long rng = 0x12345678ULL;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) * (1.0 / 9007199254740992.0) * 4.0 - 2.0);
}

static float *mk(int n) {
    Tensor *t = tensor_alloc(1, (int64_t[]){n});
    float *h = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = frand();
    mem_h2d(t->data, h, n * sizeof(float));
    free(h);
    return t->data; /* leaked intentionally; test only */
}

static void chk(const char *nm, const float *d, int n) {
    float *h = (float *)malloc(n * sizeof(float));
    mem_d2h(h, d, n * sizeof(float));
    double s = 0, mx = 0;
    for (int i = 0; i < n; i++) { s += h[i]; double a = fabs(h[i]); if (a > mx) mx = a; }
    printf("%s sum=%.6f maxabs=%.6f first=%.6f %.6f %.6f %.6f\n",
           nm, s, mx, h[0], h[1], h[2], h[3]);
    free(h);
}

int main(void) {
    arena_init(1ULL << 30);
    /* layernorm */
    {
        float *x = mk(4 * 64), *w = mk(64), *b = mk(64), *y = mk(4 * 64);
        layernorm_fwd(x, w, b, y, 4, 64);
        chk("layernorm", y, 4 * 64);
    }
    /* rmsnorm */
    {
        float *x = mk(4 * 64), *w = mk(64), *y = mk(4 * 64);
        rmsnorm_fwd(x, w, y, 4, 64);
        chk("rmsnorm", y, 4 * 64);
    }
    /* gelu / silu / softmax */
    {
        float *x = mk(512), *y = mk(512);
        gelu_fwd(x, y, 512); chk("gelu", y, 512);
        silu_fwd(x, y, 512); chk("silu", y, 512);
        softmax_fwd(x, y, 8, 64); chk("softmax", y, 512);
    }
    /* rope */
    {
        int B = 1, H = 2, T = 16, hd = 32;
        float *q = mk(B * H * T * hd), *k = mk(B * H * T * hd);
        float *co = mk(T * hd), *si = mk(T * hd);
        rope_apply(q, k, co, si, B, H, T, hd);
        chk("rope_q", q, B * H * T * hd);
        chk("rope_k", k, B * H * T * hd);
    }
    /* sdpa with additive mask */
    {
        int B = 1, H = 2, Tq = 8, Tkv = 12, hd = 16;
        float *q = mk(B * H * Tq * hd), *k = mk(B * H * Tkv * hd);
        float *v = mk(B * H * Tkv * hd), *m = mk(Tq * Tkv), *o = mk(B * H * Tq * hd);
        /* sparsify mask to 0/-inf */
        float *hm = (float *)malloc(Tq * Tkv * sizeof(float));
        mem_d2h(hm, m, Tq * Tkv * sizeof(float));
        for (int i = 0; i < Tq * Tkv; i++) hm[i] = hm[i] > 0 ? 0.0f : -INFINITY;
        mem_h2d(m, hm, Tq * Tkv * sizeof(float));
        free(hm);
        sdpa_fwd(q, k, v, m, o, B, H, Tq, Tkv, hd);
        chk("sdpa", o, B * H * Tq * hd);
    }
    /* conv / groupnorm / interp / l2 */
    {
        float *x = mk(2 * 4 * 32), *W = mk(4 * 4 * 3), *b = mk(4), *y = mk(2 * 4 * 32);
        conv1d_fwd(x, W, b, y, 2, 4, 4, 32, 3, 1, 1);
        chk("conv1d", y, 2 * 4 * 32);
        groupnorm_fwd(x, y, 2, 4, 4, 32);
        chk("groupnorm", y, 2 * 4 * 32);
        interpolate_area_fwd(x, y, 2, 4, 32, 9);
        chk("area", y, 2 * 4 * 9);
        interpolate_linear_fwd(x, y, 2, 4, 9, 32);
        chk("linear_interp", y, 2 * 4 * 32);
        float *a = mk(16 * 32), *bb = mk(8 * 32), *d2 = mk(16 * 8);
        rowwise_l2sq(a, bb, d2, 16, 8, 32);
        chk("l2sq", d2, 16 * 8);
    }
    /* linear + embedding */
    {
        float *x = mk(8 * 64), *W = mk(128 * 64), *b = mk(128), *y = mk(8 * 128);
        linear_fwd(x, W, b, y, 8, 128, 64);
        chk("linear", y, 8 * 128);
    }
    printf("KTEST-DONE\n");
    return 0;
}
