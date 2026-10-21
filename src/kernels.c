/* kernels.c — forward ops. CPU loops; CUDA kernels under NCM_CUDA.
 * All pointers refer to backend memory. Semantics mirror torch exactly:
 * LayerNorm eps=1e-6 (gpt2 variant takes explicit eps=1e-5),
 * GELU exact (erf), gelu_tanh for GPT-2, softmax stable, interpolate
 * align_corners=False, GroupNorm affine=false.
 */
#include "ncm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef NCM_CUDA
#include <cuda_runtime.h>
#define NCM_THREADS 256
#define NCM_LAUNCH(n, fn, ...) do { \
    int n_ = (int)(n); \
    fn<<<(n_ + NCM_THREADS - 1) / NCM_THREADS, NCM_THREADS>>>(__VA_ARGS__); \
} while (0)
#endif

/* ================= linear ================= */

#ifdef NCM_CUDA
__global__ static void k_linear(const float *x, const float *W, const float *b,
                                float *y, int64_t batch, int M, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * M) return;
    int64_t r = i / M; int m = (int)(i % M);
    double acc = b ? b[m] : 0.0;
    const float *xr = x + r * K;
    const float *Wr = W + (int64_t)m * K;
    for (int k = 0; k < K; k++) acc += (double)xr[k] * Wr[k];
    y[i] = (float)acc;
}
#endif

void linear_fwd(const float *x, const float *W, const float *b,
                float *y, int64_t batch, int M, int K) {
#ifdef NCM_CUDA
    NCM_LAUNCH(batch * M, k_linear, x, W, b, y, batch, M, K);
#else
#pragma omp parallel for schedule(static) if(batch * M > 512)
    for (int64_t r = 0; r < batch; r++) {
        for (int m = 0; m < M; m++) {
            double acc = b ? b[m] : 0.0;
            const float *xr = x + r * K;
            const float *Wr = W + (int64_t)m * K;
            for (int k = 0; k < K; k++) acc += (double)xr[k] * Wr[k];
            y[r * M + m] = (float)acc;
        }
    }
#endif
}

/* ================= embedding ================= */

#ifdef NCM_CUDA
__global__ static void k_embed(const float *W, const int *idx, float *y,
                               int64_t n, int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    y[i] = W[(int64_t)idx[i / D] * D + (i % D)];
}
#endif

void embedding_fwd(const float *W, const int *idx, float *y, int64_t n, int D) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n * D, k_embed, W, idx, y, n, D);
#else
    for (int64_t i = 0; i < n; i++)
        memcpy(y + i * D, W + (int64_t)idx[i] * D, D * sizeof(float));
#endif
}

/* ================= layernorm / rmsnorm ================= */

#ifdef NCM_CUDA
__global__ static void k_layernorm(const float *x, const float *w, const float *b,
                                   float *y, int64_t rows, int C) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * C; float *yr = y + r * C;
    double mean = 0; for (int c = 0; c < C; c++) mean += xr[c]; mean /= C;
    double var = 0; for (int c = 0; c < C; c++) { double d = xr[c] - mean; var += d * d; } var /= C;
    double inv = 1.0 / sqrt(var + 1e-6);
    for (int c = 0; c < C; c++)
        yr[c] = (float)((xr[c] - mean) * inv) * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
}
__global__ static void k_rmsnorm(const float *x, const float *w, float *y,
                                 int64_t rows, int C) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * C; float *yr = y + r * C;
    double ms = 0; for (int c = 0; c < C; c++) ms += (double)xr[c] * xr[c]; ms /= C;
    float s = (float)(1.0 / sqrt(ms + 1e-6));
    for (int c = 0; c < C; c++) yr[c] = xr[c] * s * (w ? w[c] : 1.0f);
}
#endif

void layernorm_fwd(const float *x, const float *w, const float *b,
                   float *y, int64_t rows, int C) {
#ifdef NCM_CUDA
    NCM_LAUNCH(rows, k_layernorm, x, w, b, y, rows, C);
#else
#pragma omp parallel for schedule(static) if(rows > 4)
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * C; float *yr = y + r * C;
        double mean = 0; for (int c = 0; c < C; c++) mean += xr[c]; mean /= C;
        double var = 0; for (int c = 0; c < C; c++) { double d = xr[c] - mean; var += d * d; } var /= C;
        double inv = 1.0 / sqrt(var + 1e-6);
        for (int c = 0; c < C; c++)
            yr[c] = (float)((xr[c] - mean) * inv) * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
    }
#endif
}

#ifdef NCM_CUDA
__global__ static void k_layernorm_eps(const float *x, const float *w, const float *b,
                                       float *y, int64_t rows, int C, float eps) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * C; float *yr = y + r * C;
    double mean = 0; for (int c = 0; c < C; c++) mean += xr[c]; mean /= C;
    double var = 0; for (int c = 0; c < C; c++) { double d = xr[c] - mean; var += d * d; } var /= C;
    double inv = 1.0 / sqrt(var + (double)eps);
    for (int c = 0; c < C; c++)
        yr[c] = (float)((xr[c] - mean) * inv) * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
}
#endif

/* LayerNorm with explicit epsilon (GPT-2 uses 1e-5; NCP default stays 1e-6). */
void layernorm_eps_fwd(const float *x, const float *w, const float *b,
                       float *y, int64_t rows, int C, float eps) {
#ifdef NCM_CUDA
    NCM_LAUNCH(rows, k_layernorm_eps, x, w, b, y, rows, C, eps);
#else
#pragma omp parallel for schedule(static) if(rows > 4)
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * C; float *yr = y + r * C;
        double mean = 0; for (int c = 0; c < C; c++) mean += xr[c]; mean /= C;
        double var = 0; for (int c = 0; c < C; c++) { double d = xr[c] - mean; var += d * d; } var /= C;
        double inv = 1.0 / sqrt(var + (double)eps);
        for (int c = 0; c < C; c++)
            yr[c] = (float)((xr[c] - mean) * inv) * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
    }
#endif
}

void rmsnorm_fwd(const float *x, const float *w, float *y, int64_t rows, int C) {
#ifdef NCM_CUDA
    NCM_LAUNCH(rows, k_rmsnorm, x, w, y, rows, C);
#else
#pragma omp parallel for schedule(static) if(rows > 4)
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * C; float *yr = y + r * C;
        double ms = 0; for (int c = 0; c < C; c++) ms += (double)xr[c] * xr[c]; ms /= C;
        float s = (float)(1.0 / sqrt(ms + 1e-6));
        for (int c = 0; c < C; c++) yr[c] = xr[c] * s * (w ? w[c] : 1.0f);
    }
#endif
}

/* ================= activations / softmax ================= */

#ifdef NCM_CUDA
__global__ static void k_gelu(const float *x, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f)); }
}
__global__ static void k_silu(const float *x, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); }
}
__global__ static void k_softmax(const float *x, float *y, int64_t rows, int C) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * C; float *yr = y + r * C;
    float mx = xr[0]; for (int c = 1; c < C; c++) if (xr[c] > mx) mx = xr[c];
    double s = 0; for (int c = 0; c < C; c++) s += exp((double)xr[c] - mx);
    for (int c = 0; c < C; c++) yr[c] = (float)(exp((double)xr[c] - mx) / s);
}
#endif

void gelu_fwd(const float *x, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_gelu, x, y, n);
#else
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
#endif
}

#ifdef NCM_CUDA
__global__ static void k_gelu_tanh(const float *x, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}
#endif

/* GPT-2 "gelu_new" (tanh approx). Matches HF activation_function=gelu_new. */
void gelu_tanh_fwd(const float *x, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_gelu_tanh, x, y, n);
#else
#pragma omp parallel for schedule(static) if(n > 512)
    for (int64_t i = 0; i < n; i++) {
        double v = x[i];
        double t = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        y[i] = (float)(0.5 * v * (1.0 + tanh(t)));
    }
#endif
}

void silu_fwd(const float *x, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_silu, x, y, n);
#else
    for (int64_t i = 0; i < n; i++) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); }
#endif
}

void softmax_fwd(const float *x, float *y, int64_t rows, int C) {
#ifdef NCM_CUDA
    NCM_LAUNCH(rows, k_softmax, x, y, rows, C);
#else
#pragma omp parallel for schedule(static) if(rows > 4)
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * C; float *yr = y + r * C;
        float mx = xr[0]; for (int c = 1; c < C; c++) if (xr[c] > mx) mx = xr[c];
        double s = 0; for (int c = 0; c < C; c++) s += exp((double)xr[c] - mx);
        for (int c = 0; c < C; c++) yr[c] = (float)(exp((double)xr[c] - mx) / s);
    }
#endif
}

/* ================= RoPE ================= */

/* CPU RoPE (exact mirror of apply_rotary_emb). With the repo's duplicated
 * cos/sin tables (cat(c,c)/cat(s,s)): y[d]=x[d]*c[d]+(-x[d+h])*s[d] (d<h),
 * y[d]=x[d]*c[d]+x[d-h]*s[d] (d>=h). Computed out-of-place per row.
 * CPU path only (CUDA uses k_rope_exact below). */
#ifndef NCM_CUDA
static void rope_one_exact(float *x, const float *cos, const float *sin,
                           int64_t rows, int T, int hd) {
    int h = hd / 2;
    /* copy to avoid in-place clobber within row */
    for (int64_t r = 0; r < rows; r++) {
        int t = (int)(r % T);
        float *xr = x + r * hd;
        const float *co = cos + (int64_t)t * hd;
        const float *si = sin + (int64_t)t * hd;
        for (int d = 0; d < h; d++) {
            float a = xr[d], b = xr[d + h];
            xr[d]     = a * co[d] + (-b) * si[d];
            xr[d + h] = b * co[d + h] + a * si[d + h];
        }
    }
}
#endif /* !NCM_CUDA */

#ifdef NCM_CUDA
__global__ static void k_rope_exact(float *x, const float *cos, const float *sin,
                                    int64_t rows, int T, int hd) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int h = hd / 2;
    int t = (int)(r % T);
    float *xr = x + r * hd;
    const float *co = cos + (int64_t)t * hd;
    const float *si = sin + (int64_t)t * hd;
    for (int d = 0; d < h; d++) {
        float a = xr[d], b = xr[d + h];
        xr[d]     = a * co[d] + (-b) * si[d];
        xr[d + h] = b * co[d + h] + a * si[d + h];
    }
}
#endif

void rope_apply(float *q, float *k, const float *cos, const float *sin,
                int B, int H, int T, int hd) {
#ifdef NCM_CUDA
    int64_t rows = (int64_t)B * H * T;
    if (q) NCM_LAUNCH(rows, k_rope_exact, q, cos, sin, rows, T, hd);
    if (k) NCM_LAUNCH(rows, k_rope_exact, k, cos, sin, rows, T, hd);
#else
    int64_t rows = (int64_t)B * H * T;
    if (q) rope_one_exact(q, cos, sin, rows, T, hd);
    if (k) rope_one_exact(k, cos, sin, rows, T, hd);
#endif
}

/* GQA RoPE: q carries H heads, k carries K heads (K<H). Same tables. */
void rope_apply_gqa(float *q, float *k, const float *cos, const float *sin,
                    int B, int H, int K, int T, int hd) {
#ifdef NCM_CUDA
    if (q) { int64_t rows = (int64_t)B * H * T; NCM_LAUNCH(rows, k_rope_exact, q, cos, sin, rows, T, hd); }
    if (k) { int64_t rows = (int64_t)B * K * T; NCM_LAUNCH(rows, k_rope_exact, k, cos, sin, rows, T, hd); }
#else
    if (q) rope_one_exact(q, cos, sin, (int64_t)B * H * T, T, hd);
    if (k) rope_one_exact(k, cos, sin, (int64_t)B * K * T, T, hd);
#endif
}

#ifdef NCM_CUDA
__global__ static void k_heads_transpose(const float *in, float *out,
                                         int64_t T, int H, int hd) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H * hd) return;
    int64_t t = i / (H * hd), r = i % (H * hd);
    int h = (int)(r / hd), d = (int)(r % hd);
    out[((int64_t)h * T + t) * hd + d] = in[((int64_t)t * H + h) * hd + d];
}
#endif

/* Rearrange [T][H*hd] row-major into heads-major [H*T][hd]. */
void heads_transpose_fwd(const float *in, float *out, int64_t T, int H, int hd) {
#ifdef NCM_CUDA
    NCM_LAUNCH(T * H * hd, k_heads_transpose, in, out, T, H, hd);
#else
#pragma omp parallel for schedule(static) if(T * H * hd > 4096)
    for (int64_t t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            for (int d = 0; d < hd; d++)
                out[((int64_t)h * T + t) * hd + d] = in[((int64_t)t * H + h) * hd + d];
#endif
}

/* B-aware variant: [B][T][H*hd] -> [B][H][T][hd]. */
#ifdef NCM_CUDA
__global__ static void k_heads_transpose_b(const float *in, float *out,
                                           int64_t B, int64_t T, int H, int hd) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * H * hd) return;
    int64_t t = (i / hd / H) % T, r = i % (H * hd);
    int64_t h = r / hd, d = r % hd, b = i / ((int64_t)T * H * hd);
    out[((b * H + h) * T + t) * hd + d] = in[((b * T + t) * H + h) * hd + d];
}
#endif
void heads_transpose_fwd_b(const float *in, float *out, int B, int64_t T, int H, int hd) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)B * T * H * hd, k_heads_transpose_b, in, out, B, T, H, hd);
#else
#pragma omp parallel for collapse(3) schedule(static) if((int64_t)B * T * H * hd > 4096)
    for (int b = 0; b < B; b++)
        for (int64_t t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                for (int d = 0; d < hd; d++)
                    out[((int64_t)(b * H + h) * T + t) * hd + d] =
                        in[((int64_t)(b * T + t) * H + h) * hd + d];
#endif
}

/* ================= SDPA ================= */

#ifdef NCM_CUDA
/* scores live in per-row GLOBAL scratch (shared memory is block-wide and
 * would race between rows). */
__global__ static void k_sdpa(const float *q, const float *k, const float *v,
                              const float *mask, float *out, float *scores,
                              int B, int H, int Tq, int Tkv, int hd, float scale) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int64_t)B * H * Tq) return;
    int b = (int)(row / (H * Tq)); int h = (int)((row / Tq) % H); int tq = (int)(row % Tq);
    const float *qr = q + ((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd;
    float *sh = scores + row * Tkv;
    float mx = -1e30f;
    for (int tk = 0; tk < Tkv; tk++) {
        const float *kr = k + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
        double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
        float sc = (float)(s * scale);
        if (mask) sc += mask[(int64_t)tq * Tkv + tk];
        sh[tk] = sc;
        if (sc > mx) mx = sc;
    }
    double den = 0; for (int tk = 0; tk < Tkv; tk++) den += exp((double)sh[tk] - mx);
    float *or_ = out + ((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd;
    for (int d = 0; d < hd; d++) or_[d] = 0;
    for (int tk = 0; tk < Tkv; tk++) {
        float p = (float)(exp((double)sh[tk] - mx) / den);
        const float *vr = v + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
        for (int d = 0; d < hd; d++) or_[d] += p * vr[d];
    }
}
#endif

#ifdef NCM_CUDA
__global__ static void k_sdpa_gqa(const float *q, const float *k, const float *v,
                                  const float *mask, float *out, float *scores,
                                  int B, int Hq, int Hkv, int Tq, int Tkv, int hd, float scale) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int64_t)B * Hq * Tq) return;
    int b = (int)(row / (Hq * Tq)); int h = (int)((row / Tq) % Hq); int tq = (int)(row % Tq);
    int rep = Hq / Hkv, kvh = h / rep;
    const float *qr = q + ((int64_t)b * Hq + h) * Tq * hd + (int64_t)tq * hd;
    float *sh = scores + row * Tkv;
    float mx = -1e30f;
    for (int tk = 0; tk < Tkv; tk++) {
        const float *kr = k + ((int64_t)b * Hkv + kvh) * Tkv * hd + (int64_t)tk * hd;
        double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
        float sc = (float)(s * scale);
        if (mask) sc += mask[(int64_t)tq * Tkv + tk];
        sh[tk] = sc;
        if (sc > mx) mx = sc;
    }
    double den = 0; for (int tk = 0; tk < Tkv; tk++) den += exp((double)sh[tk] - mx);
    float *or_ = out + ((int64_t)b * Hq + h) * Tq * hd + (int64_t)tq * hd;
    for (int d = 0; d < hd; d++) or_[d] = 0;
    for (int tk = 0; tk < Tkv; tk++) {
        float p = (float)(exp((double)sh[tk] - mx) / den);
        const float *vr = v + ((int64_t)b * Hkv + kvh) * Tkv * hd + (int64_t)tk * hd;
        for (int d = 0; d < hd; d++) or_[d] += p * vr[d];
    }
}
#endif

/* GQA SDPA: q has Hq heads, k/v have Hkv heads (kvh = h / (Hq/Hkv)). */
void sdpa_fwd_gqa(const float *q, const float *k, const float *v, const float *mask,
                  float *out, int B, int Hq, int Hkv, int Tq, int Tkv, int hd) {
    float scale = 1.0f / sqrtf((float)hd);
#ifdef NCM_CUDA
    {
        int64_t rows = (int64_t)B * Hq * Tq;
        float *dscores = arena_alloc((size_t)rows * Tkv);
        int blocks = (int)((rows + NCM_THREADS - 1) / NCM_THREADS);
        k_sdpa_gqa<<<blocks, NCM_THREADS>>>(
            q, k, v, mask, out, dscores, B, Hq, Hkv, Tq, Tkv, hd, scale);
    }
#else
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)B * Hq * Tq > 4)
    for (int b = 0; b < B; b++) for (int h = 0; h < Hq; h++) {
        int kvh = h / (Hq / Hkv);
        for (int tq = 0; tq < Tq; tq++) {
            const float *qr = q + ((int64_t)b * Hq + h) * Tq * hd + (int64_t)tq * hd;
            double mx = -1e300, den = 0;
            for (int tk = 0; tk < Tkv; tk++) {
                const float *kr = k + ((int64_t)b * Hkv + kvh) * Tkv * hd + (int64_t)tk * hd;
                double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
                s *= scale;
                if (mask) s += mask[(int64_t)tq * Tkv + tk];
                if (s > mx) mx = s;
            }
            float *or_ = out + ((int64_t)b * Hq + h) * Tq * hd + (int64_t)tq * hd;
            for (int d = 0; d < hd; d++) or_[d] = 0;
                for (int tk = 0; tk < Tkv; tk++) {
                    const float *kr = k + ((int64_t)b * Hkv + kvh) * Tkv * hd + (int64_t)tk * hd;
                    double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
                    s *= scale;
                    if (mask) s += mask[(int64_t)tq * Tkv + tk];
                    double p = exp(s - mx); den += p;
                    const float *vr = v + ((int64_t)b * Hkv + kvh) * Tkv * hd + (int64_t)tk * hd;
                    for (int d = 0; d < hd; d++) or_[d] += (float)p * vr[d];
                }
                double inv = 1.0 / den;
                for (int d = 0; d < hd; d++) or_[d] = (float)(or_[d] * inv);
        }
    }
#endif
}

void sdpa_fwd(const float *q, const float *k, const float *v, const float *mask,
              float *out, int B, int H, int Tq, int Tkv, int hd) {
    float scale = 1.0f / sqrtf((float)hd);
#ifdef NCM_CUDA
    {
        int64_t rows = (int64_t)B * H * Tq;
        float *dscores = arena_alloc((size_t)rows * Tkv);
        int blocks = (int)((rows + NCM_THREADS - 1) / NCM_THREADS);
        k_sdpa<<<blocks, NCM_THREADS>>>(
            q, k, v, mask, out, dscores, B, H, Tq, Tkv, hd, scale);
    }
#else
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)B * H * Tq > 4)
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
        for (int tq = 0; tq < Tq; tq++) {
            const float *qr = q + ((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd;
            double mx = -1e300, den = 0;
            /* two passes: max then accumulate (store scores on stack via arena-free local) */
            for (int tk = 0; tk < Tkv; tk++) {
                const float *kr = k + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
                double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
                s *= scale;
                if (mask) s += mask[(int64_t)tq * Tkv + tk];
                if (s > mx) mx = s;
            }
            float *or_ = out + ((int64_t)b * H + h) * Tq * hd + (int64_t)tq * hd;
            for (int d = 0; d < hd; d++) or_[d] = 0;
            for (int tk = 0; tk < Tkv; tk++) {
                const float *kr = k + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
                double s = 0; for (int d = 0; d < hd; d++) s += (double)qr[d] * kr[d];
                s *= scale;
                if (mask) s += mask[(int64_t)tq * Tkv + tk];
                double p = exp(s - mx); den += p;
                const float *vr = v + ((int64_t)b * H + h) * Tkv * hd + (int64_t)tk * hd;
                for (int d = 0; d < hd; d++) or_[d] += (float)p * vr[d];
            }
            double inv = 1.0 / den;
            for (int d = 0; d < hd; d++) or_[d] = (float)(or_[d] * inv);
        }
    }
#endif
}

/* ================= conv1d / convtranspose1d ================= */

#ifdef NCM_CUDA
__global__ static void k_conv1d(const float *x, const float *W, const float *b, float *y,
                               int N, int Cin, int Cout, int T, int K, int stride, int pad) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t Tout = (T + 2 * pad - K) / stride + 1;
    if (i >= (int64_t)N * Cout * Tout) return;
    int t = (int)(i % Tout); int co = (int)((i / Tout) % Cout); int n = (int)(i / (Tout * Cout));
    double acc = b ? b[co] : 0.0;
    for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
        int ti = t * stride + kk - pad;
        if (ti < 0 || ti >= T) continue;
        acc += (double)x[((int64_t)n * Cin + ci) * T + ti] *
               W[((int64_t)co * Cin + ci) * K + kk];
    }
    y[i] = (float)acc;
}
__global__ static void k_convT(const float *x, const float *W, const float *b, float *y,
                               int N, int Cin, int Cout, int T, int K, int stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t Tout = (T - 1) * stride + K;
    if (i >= (int64_t)N * Cout * Tout) return;
    int t = (int)(i % Tout); int co = (int)((i / Tout) % Cout); int n = (int)(i / (Tout * Cout));
    double acc = b ? b[co] : 0.0;
    for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
        int num = t - kk;
        if (num < 0 || num % stride != 0) continue;
        int ti = num / stride;
        if (ti >= T) continue;
        acc += (double)x[((int64_t)n * Cin + ci) * T + ti] *
               W[((int64_t)ci * Cout + co) * K + kk];
    }
    y[i] = (float)acc;
}
#endif

void conv1d_fwd(const float *x, const float *W, const float *b, float *y,
                int N, int Cin, int Cout, int T, int K, int stride, int pad) {
#ifdef NCM_CUDA
    int64_t Tout = (T + 2 * pad - K) / stride + 1;
    NCM_LAUNCH((int64_t)N * Cout * Tout, k_conv1d, x, W, b, y, N, Cin, Cout, T, K, stride, pad);
#else
    int64_t Tout = (T + 2 * pad - K) / stride + 1;
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * Cout > 2)
    for (int n = 0; n < N; n++) for (int co = 0; co < Cout; co++)
        for (int64_t t = 0; t < Tout; t++) {
            double acc = b ? b[co] : 0.0;
            for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
                int ti = (int)(t * stride + kk - pad);
                if (ti < 0 || ti >= T) continue;
                acc += (double)x[((int64_t)n * Cin + ci) * T + ti] *
                       W[((int64_t)co * Cin + ci) * K + kk];
            }
            y[((int64_t)n * Cout + co) * Tout + t] = (float)acc;
        }
#endif
}

void convtranspose1d_fwd(const float *x, const float *W, const float *b, float *y,
                         int N, int Cin, int Cout, int T, int K, int stride) {
    /* torch ConvTranspose1d weight layout: [Cin, Cout/groups, K] */
#ifdef NCM_CUDA
    int64_t Tout = (int64_t)(T - 1) * stride + K;
    NCM_LAUNCH((int64_t)N * Cout * Tout, k_convT, x, W, b, y, N, Cin, Cout, T, K, stride);
#else
    int64_t Tout = (int64_t)(T - 1) * stride + K;
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * Cout > 2)
    for (int n = 0; n < N; n++) for (int co = 0; co < Cout; co++)
        for (int64_t t = 0; t < Tout; t++) {
            double acc = b ? b[co] : 0.0;
            for (int ci = 0; ci < Cin; ci++) for (int kk = 0; kk < K; kk++) {
                int num = (int)(t - kk);
                if (num < 0 || num % stride != 0) continue;
                int ti = num / stride;
                if (ti >= T) continue;
                acc += (double)x[((int64_t)n * Cin + ci) * T + ti] *
                       W[((int64_t)ci * Cout + co) * K + kk];
            }
            y[((int64_t)n * Cout + co) * Tout + t] = (float)acc;
        }
#endif
}

/* ================= groupnorm (affine=false) ================= */

#ifdef NCM_CUDA
__global__ static void k_groupnorm(const float *x, float *y, int N, int G, int C, int T) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (int64_t)N * G) return;
    int n = (int)(r / G), g = (int)(r % G);
    int Cg = C / G;
    double mean = 0; int64_t cnt = (int64_t)Cg * T;
    for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++)
        mean += x[((int64_t)n * C + g * Cg + c) * T + t];
    mean /= cnt;
    double var = 0;
    for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++) {
        double d = x[((int64_t)n * C + g * Cg + c) * T + t] - mean; var += d * d;
    }
    var /= cnt;
    double inv = 1.0 / sqrt(var + 1e-5);
    for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++) {
        int64_t i = ((int64_t)n * C + g * Cg + c) * T + t;
        y[i] = (float)((x[i] - mean) * inv);
    }
}
#endif

void groupnorm_fwd(const float *x, float *y, int N, int G, int C, int T) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)N * G, k_groupnorm, x, y, N, G, C, T);
#else
    int Cg = C / G;
#pragma omp parallel for collapse(2) schedule(static) if(N * G > 2)
    for (int n = 0; n < N; n++) for (int g = 0; g < G; g++) {
        double mean = 0; int64_t cnt = (int64_t)Cg * T;
        for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++)
            mean += x[((int64_t)n * C + g * Cg + c) * T + t];
        mean /= cnt;
        double var = 0;
        for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++) {
            double d = x[((int64_t)n * C + g * Cg + c) * T + t] - mean; var += d * d;
        }
        var /= cnt;
        double inv = 1.0 / sqrt(var + 1e-5);
        for (int c = 0; c < Cg; c++) for (int t = 0; t < T; t++) {
            int64_t i = ((int64_t)n * C + g * Cg + c) * T + t;
            y[i] = (float)((x[i] - mean) * inv);
        }
    }
#endif
}

/* ================= interpolate ================= */

#ifdef NCM_CUDA
/* ATen area convention (verified cpu==cuda): equal-weight mean over
 * [floor(t*Tin/Tout), ceil((t+1)*Tin/Tout)). */
__global__ static void k_interp_area(const float *x, float *y, int N, int C, int Tin, int Tout) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * C * Tout) return;
    int t = (int)(i % Tout); int c = (int)((i / Tout) % C); int n = (int)(i / (Tout * C));
    int64_t s = (int64_t)((double)t * Tin / Tout);
    int64_t e = (int64_t)ceil((double)(t + 1) * Tin / Tout);
    if (e <= s) e = s + 1;
    if (s < 0) { s = 0; }
    if (e > Tin) { e = Tin; }
    double acc = 0; for (int64_t k = s; k < e; k++) acc += x[((int64_t)n * C + c) * Tin + k];
    y[i] = (float)(acc / (e - s));
}
__global__ static void k_interp_lin(const float *x, float *y, int N, int C, int Tin, int Tout) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * C * Tout) return;
    int t = (int)(i % Tout); int c = (int)((i / Tout) % C); int n = (int)(i / (Tout * C));
    const float *xr = x + ((int64_t)n * C + c) * Tin;
    if (Tout == 1 || Tin == 1) {
        /* area fallback for degenerate */
        double acc = 0; for (int k = 0; k < Tin; k++) acc += xr[k];
        y[i] = (float)(acc / Tin); return;
    }
    double scale = (double)Tin / Tout;
    double src = (t + 0.5) * scale - 0.5;
    int64_t s0 = (int64_t)floor(src);
    double w1 = src - s0; double w0 = 1.0 - w1;
    double v0 = xr[s0 < 0 ? 0 : (s0 >= Tin ? Tin - 1 : s0)];
    int64_t s1 = s0 + 1;
    double v1 = xr[s1 < 0 ? 0 : (s1 >= Tin ? Tin - 1 : s1)];
    y[i] = (float)(w0 * v0 + w1 * v1);
}
#endif

void interpolate_area_fwd(const float *x, float *y, int N, int C, int Tin, int Tout) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)N * C * Tout, k_interp_area, x, y, N, C, Tin, Tout);
#else
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * C > 4)
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
        const float *xr = x + ((int64_t)n * C + c) * Tin;
        float *yr = y + ((int64_t)n * C + c) * Tout;
        for (int t = 0; t < Tout; t++) {
            int64_t s = (int64_t)((double)t * Tin / Tout);
            int64_t e = (int64_t)ceil((double)(t + 1) * Tin / Tout);
            if (e <= s) e = s + 1;
            if (s < 0) { s = 0; }
            if (e > Tin) { e = Tin; }
            double acc = 0; for (int64_t k = s; k < e; k++) acc += xr[k];
            yr[t] = (float)(acc / (e - s));
        }
    }
#endif
}

void interpolate_linear_fwd(const float *x, float *y, int N, int C, int Tin, int Tout) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)N * C * Tout, k_interp_lin, x, y, N, C, Tin, Tout);
#else
#pragma omp parallel for collapse(2) schedule(static) if((int64_t)N * C > 4)
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
        const float *xr = x + ((int64_t)n * C + c) * Tin;
        float *yr = y + ((int64_t)n * C + c) * Tout;
        if (Tout == 1 || Tin == 1) {
            double acc = 0; for (int k = 0; k < Tin; k++) acc += xr[k];
            for (int t = 0; t < Tout; t++) yr[t] = (float)(acc / Tin);
            continue;
        }
        double scale = (double)Tin / Tout;
        for (int t = 0; t < Tout; t++) {
            double src = (t + 0.5) * scale - 0.5;
            int64_t s0 = (int64_t)floor(src);
            double w1 = src - s0, w0 = 1.0 - w1;
            double v0 = xr[s0 < 0 ? 0 : (s0 >= Tin ? Tin - 1 : s0)];
            int64_t s1 = s0 + 1;
            double v1 = xr[s1 < 0 ? 0 : (s1 >= Tin ? Tin - 1 : s1)];
            yr[t] = (float)(w0 * v0 + w1 * v1);
        }
    }
#endif
}

/* ================= misc ================= */

int argmax_row(const float *x, int n) {
    int bi = 0; float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; bi = i; }
    return bi;
}

/* d2[i*colsA+j] = ||a_i - b_j||^2, a:[rows,C], b:[colsA,C] */
#ifdef NCM_CUDA
__global__ static void k_l2sq(const float *a, const float *b, float *d2,
                              int64_t rows, int64_t colsA, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * colsA) return;
    int64_t r = i / colsA, j = i % colsA;
    double s = 0;
    for (int c = 0; c < C; c++) {
        double d = (double)a[r * C + c] - b[j * C + c];
        s += d * d;
    }
    d2[i] = (float)s;
}
#endif

void rowwise_l2sq(const float *a, const float *b, float *d2,
                  int64_t rows, int64_t colsA, int C) {
#ifdef NCM_CUDA
    NCM_LAUNCH(rows * colsA, k_l2sq, a, b, d2, rows, colsA, C);
#else
#pragma omp parallel for schedule(static) if(rows * colsA > 4096)
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < colsA; j++) {
            double s = 0;
            for (int c = 0; c < C; c++) {
                double d = (double)a[i * C + c] - b[j * C + c];
                s += d * d;
            }
            d2[i * colsA + j] = (float)s;
        }
    }
#endif
}

/* ================= RNG + sampling ================= */

unsigned long long rng_next(unsigned long long *state) {
    /* xorshift64* */
    unsigned long long x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(unsigned long long *rng) {
    return (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static int cmp_desc(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    extern float *srt_vals_tmp; /* set below */
    float va = srt_vals_tmp[ia], vb = srt_vals_tmp[ib];
    return (va < vb) - (va > vb);
}
float *srt_vals_tmp = NULL;

int sample_topkp(const float *logits, int V, float temp, int top_k, float top_p,
                 unsigned long long *rng) {
    if (!(temp > 0)) return argmax_row(logits, V);
    float *work = (float *)malloc(V * sizeof(float));
    int *ord = (int *)malloc(V * sizeof(int));
    for (int i = 0; i < V; i++) { work[i] = logits[i] / temp; ord[i] = i; }
    srt_vals_tmp = work;
    qsort(ord, V, sizeof(int), cmp_desc);
    srt_vals_tmp = NULL;
    /* top_k mask */
    int keep = (top_k > 0) ? (top_k < V ? top_k : V) : V;
    for (int r = keep; r < V; r++) work[ord[r]] = -INFINITY;
    /* nucleus mask on sorted order */
    double *probs = (double *)malloc(V * sizeof(double));
    double mx = work[ord[0]];
    double den = 0;
    for (int r = 0; r < keep; r++) { probs[r] = exp((double)work[ord[r]] - mx); den += probs[r]; }
    for (int r = 0; r < keep; r++) probs[r] /= den;
    double cum = 0;
    int nkeep = keep;
    for (int r = 0; r < keep; r++) {
        if (cum - probs[r] > top_p) { nkeep = r; break; }
        cum += probs[r];
    }
    /* renormalize over first nkeep */
    den = 0; for (int r = 0; r < nkeep; r++) den += probs[r];
    double u = rng_uniform(rng) * den, acc = 0;
    int pick = ord[nkeep - 1];
    for (int r = 0; r < nkeep; r++) { acc += probs[r]; if (u <= acc) { pick = ord[r]; break; } }
    free(work); free(ord); free(probs);
    return pick;
}

/* ================= elementwise add/scale (training) ================= */
#ifdef NCM_CUDA
__global__ static void k_add(const float *a, const float *b, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ static void k_scale(const float *x, float s, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}
__global__ static void k_add_inplace2(float *y, const float *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
#endif

void add_fwd(const float *a, const float *b, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_add, a, b, y, n);
#else
    for (int64_t i = 0; i < n; i++) y[i] = a[i] + b[i];
#endif
}

#ifdef NCM_CUDA
__global__ static void k_mul(const float *a, const float *b, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
#endif

/* Elementwise multiply (SwiGLU gating). */
void mul_fwd(const float *a, const float *b, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_mul, a, b, y, n);
#else
#pragma omp parallel for schedule(static) if(n > 512)
    for (int64_t i = 0; i < n; i++) y[i] = a[i] * b[i];
#endif
}

void scale_fwd(const float *x, float s, float *y, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_scale, x, s, y, n);
#else
    for (int64_t i = 0; i < n; i++) y[i] = x[i] * s;
#endif
}

void add_inplace_fwd(float *y, const float *x, int64_t n) {
#ifdef NCM_CUDA
    NCM_LAUNCH(n, k_add_inplace2, y, x, n);
#else
    for (int64_t i = 0; i < n; i++) y[i] += x[i];
#endif
}

/* ================= qkv split / head merge (training, recorded) ================= */
/* split: qkv[B,T,3C] (or [B,T,C+2*Kv*hd]) -> qq,kk,vv [B,H,T,hd].
 * head h rows: q from [h*hd, h*hd+hd), k from [C+h*hd/K... ] — general GQA:
 *   q[b,h,t,d] = in[b,t,h*hd+d]
 *   k[b,h,t,d] = in[b,t,C+(h/rep)*hd+d]
 *   v[b,h,t,d] = in[b,t,C+K*hd+(h/rep)*hd+d], K=n_kv_heads */
#ifdef NCM_CUDA
__global__ static void k_qkvsplit(const float *in, float *qq, float *kk, float *vv,
                                  int B, int T, int C, int H, int hd, int K, int rep) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * H * T * hd) return;
    int d = (int)(i % hd); int t = (int)((i / hd) % T);
    int h = (int)((i / (hd * T)) % H); int b = (int)(i / ((int64_t)H * T * hd));
    int kvh = h / rep;
    int64_t W = (int64_t)C + 2 * K * hd;
    qq[i] = in[((int64_t)b * T + t) * W + h * hd + d];
    kk[i] = in[((int64_t)b * T + t) * W + C + kvh * hd + d];
    vv[i] = in[((int64_t)b * T + t) * W + C + K * hd + kvh * hd + d];
}
__global__ static void k_headmerge(const float *ctx, float *out, int B, int T, int C, int H, int hd) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * T * C) return;
    int d = (int)(i % C);
    int hh = d / hd, dd = d % hd;
    int t = (int)((i / C) % T); int b = (int)(i / ((int64_t)T * C));
    out[i] = ctx[((int64_t)b * H + hh) * T * hd + (int64_t)t * hd + dd];
}
#endif

void qkv_split_fwd(const float *in, float *qq, float *kk, float *vv,
                   int B, int T, int C, int H, int hd, int K, int rep) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)B * H * T * hd, k_qkvsplit, in, qq, kk, vv, B, T, C, H, hd, K, rep);
#else
    int64_t W = (int64_t)C + 2 * K * hd;
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int t = 0; t < T; t++) for (int d = 0; d < hd; d++) {
        int kvh = h / rep;
        qq[((size_t)b * H + h) * T * hd + (size_t)t * hd + d] = in[((size_t)b * T + t) * W + h * hd + d];
        kk[((size_t)b * H + h) * T * hd + (size_t)t * hd + d] = in[((size_t)b * T + t) * W + C + kvh * hd + d];
        vv[((size_t)b * H + h) * T * hd + (size_t)t * hd + d] = in[((size_t)b * T + t) * W + C + K * hd + kvh * hd + d];
    }
#endif
}

void head_merge_fwd(const float *ctx, float *out, int B, int T, int C, int H, int hd) {
#ifdef NCM_CUDA
    NCM_LAUNCH((int64_t)B * T * C, k_headmerge, ctx, out, B, T, C, H, hd);
#else
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int hh = 0; hh < H; hh++) for (int d = 0; d < hd; d++)
        out[((size_t)b * T + t) * C + hh * hd + d] =
            ctx[((size_t)b * H + hh) * T * hd + (size_t)t * hd + d];
#endif
}
