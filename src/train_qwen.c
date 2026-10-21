/* train_qwen.c — taped Qwen forward (recorded, out-of-place).
 *
 * Mirrors qwen.c op-for-op (RMSNorm, q/k/v linears + bias, headsplit,
 * RoPE pair-duplicated tables, GQA SDPA with causal mask, merge, o_proj,
 * residual, RMSNorm, SwiGLU MLP, final RMSNorm, tied head) but records
 * every op into a Tape so the same path trains and verifies.
 *
 * GQA flows un-replicated: k/v stay at Hkv heads from headsplit through
 * rope(H,Hkv) and sdpa(H,Hkv). The NCP fused path instead replicates via
 * t_qkvsplit(rep); both are mathematically identical attention.
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
