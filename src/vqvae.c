/* vqvae.c — VQVAE encoder/decoder inference. Mirrors models/vae.py +
 * models/vqvae.py. Weight prefix "tok." (encoder./vq./decoder.).
 */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct VQVAEState {
    NCMConfig cfg;
    WeightStore *ws;
    VQState *vq;
    /* rope-free; vae blocks use plain attention, no mask (full) or window */
    float *rope_dummy;
};

static Tensor *W(WeightStore *ws, const char *a, const char *b) {
    char key[320];
    snprintf(key, sizeof(key), "%s%s", a, b);
    Tensor *t = weights_get(ws, key);
    if (!t) { fprintf(stderr, "missing weight %s\n", key); exit(1); }
    return t;
}

VQVAEState *vqvae_load(const NCMConfig *c, WeightStore *ws) {
    VQVAEState *v = (VQVAEState *)calloc(1, sizeof(VQVAEState));
    v->cfg = *c;
    v->ws = ws;
    v->vq = vq_load(c, ws, "tok.vq.");
    return v;
}

void vqvae_free(VQVAEState *v) {
    if (!v) return;
    vq_free(v->vq);
    free(v);
}

VQState *vqvae_vq(VQVAEState *v) { return v->vq; }

/* ---- one VAE Block (models/vae.py), x:[B][T][C] backend ---- */
static void vae_block(WeightStore *ws, const char *pre, float *x,
                      int B, int T, int C, int n_head, int use_qk_norm,
                      int attn_window) {
    float *ln = arena_alloc((size_t)B * T * C);
    float *qkv = arena_alloc((size_t)B * T * 3 * C);
    float *o = arena_alloc((size_t)B * T * C);
    /* --- ln_1 --- */


    {
        char kw[320], kb[320];
        snprintf(kw, sizeof(kw), "%sln_1.weight", pre);
        snprintf(kb, sizeof(kb), "%sln_1.bias", pre);
        Tensor *tw = W(ws, "", kw);
        Tensor *tb = weights_get(ws, kb);
        layernorm_fwd(x, tw->data, tb ? tb->data : NULL, ln, (int64_t)B * T, C);
    }
    /* qkv */
    {
        char kw[320], kb[320];
        snprintf(kw, sizeof(kw), "%sattn.c_attn.weight", pre);
        snprintf(kb, sizeof(kb), "%sattn.c_attn.bias", pre);
        Tensor *tw = W(ws, "", kw);
        Tensor *tb = weights_get(ws, kb);
        linear_fwd(ln, tw->data, tb ? tb->data : NULL, qkv, (int64_t)B * T, 3 * C, C);
    }
    /* split + heads */
    int hd = C / n_head;
    float *qq = arena_alloc((size_t)B * n_head * T * hd);
    float *kk = arena_alloc((size_t)B * n_head * T * hd);
    float *vv = arena_alloc((size_t)B * n_head * T * hd);
    /* qkv is [B,T,3C] B-major; produce [B,H,T,hd] */
    {
        /* stage to host for the interleave (small models) / or backend loop.
         * Use backend-agnostic element copy via host staging. */
        size_t n = (size_t)B * T * 3 * C;
        float *hq = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
#include <cuda_runtime.h>
        cudaMemcpy(hq, qkv, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hq, qkv, n * sizeof(float));
#endif
        float *hqq = (float *)malloc((size_t)B * n_head * T * hd * sizeof(float));
        float *hkk = (float *)malloc((size_t)B * n_head * T * hd * sizeof(float));
        float *hvv = (float *)malloc((size_t)B * n_head * T * hd * sizeof(float));
        #pragma omp parallel for collapse(2) schedule(static) if((int64_t)B * T * 2 > 4)
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
            float *row = hq + ((size_t)b * T + t) * 3 * C;
            for (int hh = 0; hh < n_head; hh++) for (int d = 0; d < hd; d++) {
                hqq[((size_t)b * n_head + hh) * T * hd + t * hd + d] = row[hh * hd + d];
                hkk[((size_t)b * n_head + hh) * T * hd + t * hd + d] = row[C + hh * hd + d];
                hvv[((size_t)b * n_head + hh) * T * hd + t * hd + d] = row[2 * C + hh * hd + d];
            }
        }
        free(hq);
#ifdef NCM_CUDA
        cudaMemcpy(qq, hqq, (size_t)B * n_head * T * hd * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(kk, hkk, (size_t)B * n_head * T * hd * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(vv, hvv, (size_t)B * n_head * T * hd * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(qq, hqq, (size_t)B * n_head * T * hd * sizeof(float));
        memcpy(kk, hkk, (size_t)B * n_head * T * hd * sizeof(float));
        memcpy(vv, hvv, (size_t)B * n_head * T * hd * sizeof(float));
#endif
        free(hqq); free(hkk); free(hvv);
    }

    if (use_qk_norm) {
        char kq[320], kk2[320];
        snprintf(kq, sizeof(kq), "%sattn.q_norm.weight", pre);
        snprintf(kk2, sizeof(kk2), "%sattn.k_norm.weight", pre);
        rmsnorm_fwd(qq, W(ws, "", kq)->data, qq, (int64_t)B * n_head * T, hd);
        rmsnorm_fwd(kk, W(ws, "", kk2)->data, kk, (int64_t)B * n_head * T, hd);
    }

    float *mask = NULL;
    float *hmask = NULL;
    if (attn_window > 0) {
        hmask = (float *)malloc((size_t)T * T * sizeof(float));
        for (int i = 0; i < T; i++) for (int j = 0; j < T; j++)
            hmask[i * T + j] = (abs(i - j) <= attn_window) ? 0.0f : -INFINITY;
        mask = arena_alloc((size_t)T * T);
#ifdef NCM_CUDA
        cudaMemcpy(mask, hmask, (size_t)T * T * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(mask, hmask, (size_t)T * T * sizeof(float));
#endif
    }
    float *ctx = arena_alloc((size_t)B * n_head * T * hd);
    sdpa_fwd(qq, kk, vv, mask, ctx, B, n_head, T, T, hd);
    free(hmask);
    /* merge heads back to [B,T,C] */
    {
        float *hctx = (float *)malloc((size_t)B * n_head * T * hd * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hctx, ctx, (size_t)B * n_head * T * hd * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hctx, ctx, (size_t)B * n_head * T * hd * sizeof(float));
#endif
        float *ho = (float *)malloc((size_t)B * T * C * sizeof(float));
        #pragma omp parallel for collapse(2) schedule(static) if((int64_t)B * T * n_head > 8)
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int hh = 0; hh < n_head; hh++)
            for (int d = 0; d < hd; d++)
                ho[((size_t)b * T + t) * C + hh * hd + d] =
                    hctx[((size_t)b * n_head + hh) * T * hd + t * hd + d];
        free(hctx);
#ifdef NCM_CUDA
        cudaMemcpy(o, ho, (size_t)B * T * C * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(o, ho, (size_t)B * T * C * sizeof(float));
#endif
        free(ho);
        /* c_proj + residual: x += proj(o) */
        char kw[320], kb[320];
        snprintf(kw, sizeof(kw), "%sattn.c_proj.weight", pre);
        snprintf(kb, sizeof(kb), "%sattn.c_proj.bias", pre);
        Tensor *tw = W(ws, "", kw);
        Tensor *tb = weights_get(ws, kb);
        float *po = arena_alloc((size_t)B * T * C);
        linear_fwd(o, tw->data, tb ? tb->data : NULL, po, (int64_t)B * T, C, C);
        /* x += po (backend axpy via stage) */
        size_t n2 = (size_t)B * T * C;
        float *hx = (float *)malloc(n2 * sizeof(float));
        float *hp = (float *)malloc(n2 * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hx, x, n2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp, po, n2 * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hx, x, n2 * sizeof(float));
        memcpy(hp, po, n2 * sizeof(float));
#endif
        for (size_t i = 0; i < n2; i++) hx[i] += hp[i];
#ifdef NCM_CUDA
        cudaMemcpy(x, hx, n2 * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(x, hx, n2 * sizeof(float));
#endif
        free(hx); free(hp);
    }
    /* MLP: ln2 + c_fc/gelu + c_proj + residual */
    {
        char kw[320], kb[320];
        snprintf(kw, sizeof(kw), "%sln_2.weight", pre);
        snprintf(kb, sizeof(kb), "%sln_2.bias", pre);
        layernorm_fwd(x, W(ws, "", kw)->data,
                      weights_get(ws, kb) ? weights_get(ws, kb)->data : NULL,
                      ln, (int64_t)B * T, C);
        int H4 = 4 * C;
        /* c_fc out dim from weight shape */
        snprintf(kw, sizeof(kw), "%smlp.c_fc.weight", pre);
        Tensor *tw = W(ws, "", kw);
        H4 = (int)tw->shape[0];
        snprintf(kb, sizeof(kb), "%smlp.c_fc.bias", pre);
        Tensor *tb = weights_get(ws, kb);
        float *m = arena_alloc((size_t)B * T * H4);
        linear_fwd(ln, tw->data, tb ? tb->data : NULL, m, (int64_t)B * T, H4, C);
        gelu_fwd(m, m, (int64_t)B * T * H4);
        snprintf(kw, sizeof(kw), "%smlp.c_proj.weight", pre);
        snprintf(kb, sizeof(kb), "%smlp.c_proj.bias", pre);
        tw = W(ws, "", kw); tb = weights_get(ws, kb);
        float *po = arena_alloc((size_t)B * T * C);
        linear_fwd(m, tw->data, tb ? tb->data : NULL, po, (int64_t)B * T, C, H4);
        size_t n2 = (size_t)B * T * C;
        float *hx = (float *)malloc(n2 * sizeof(float));
        float *hp = (float *)malloc(n2 * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hx, x, n2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp, po, n2 * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(hx, x, n2 * sizeof(float));
        memcpy(hp, po, n2 * sizeof(float));
#endif
        for (size_t i = 0; i < n2; i++) hx[i] += hp[i];
#ifdef NCM_CUDA
        cudaMemcpy(x, hx, n2 * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(x, hx, n2 * sizeof(float));
#endif
        free(hx); free(hp);
    }
}

/* encoder forward_nopos flag mirrors raw_token_prefix w/o contextual prefix */
void vqvae_encode_z(VQVAEState *v, const int *ids, float *z, int B) {
    NCMConfig *c = &v->cfg;
    WeightStore *ws = v->ws;
    int T = c->vq_context_length, D = c->vq_embed_dim;
    /* upload ids */
    int *bids = (int *)arena_alloc((size_t)B * T);
#ifdef NCM_CUDA
    cudaMemcpy(bids, ids, (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice);
#else
    memcpy(bids, ids, (size_t)B * T * sizeof(int));
#endif
    float *x = arena_alloc((size_t)B * T * D);
    embedding_fwd(W(ws, "tok.encoder.", "wte.weight")->data, bids, x, (int64_t)B * T, D);
    if (c->enc_positional) {
        Tensor *wpe = W(ws, "tok.encoder.", "wpe.weight");
        /* x[b,t,:] += wpe[t] */
        size_t n = (size_t)B * T * D;
        float *hx = (float *)malloc(n * sizeof(float));
        float *hw = (float *)malloc((size_t)T * D * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hx, x, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hw, wpe->data, (size_t)T * D * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hx, x, n * sizeof(float));
        memcpy(hw, wpe->data, (size_t)T * D * sizeof(float));
#endif
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hx[((size_t)b * T + t) * D + d] += hw[t * D + d];
        free(hw);
#ifdef NCM_CUDA
        cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(x, hx, n * sizeof(float));
#endif
        free(hx);
    }
    /* conv_blocks */
    for (int i = 0; i < c->enc_conv_layers; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.encoder.conv_blocks.%d.", i);
        char kw[320], kb[320];
        snprintf(kw, sizeof(kw), "%sln.weight", pre);
        float *ln = arena_alloc((size_t)B * T * D);
        layernorm_fwd(x, W(ws, "", kw)->data, NULL, ln, (int64_t)B * T, D);
        float *bcl = arena_alloc((size_t)B * T * D);
        float *c1 = arena_alloc((size_t)B * T * D);
        float *c2 = arena_alloc((size_t)B * T * D);
        /* transpose helpers are static in vq.c; do inline host transpose here */
        size_t n = (size_t)B * T * D;
        float *hln = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hln, ln, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hln, ln, n * sizeof(float));
#endif
        float *hb = (float *)malloc(n * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hb[((size_t)b * D + d) * T + t] = hln[((size_t)b * T + t) * D + d];
        free(hln);
#ifdef NCM_CUDA
        cudaMemcpy(bcl, hb, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(bcl, hb, n * sizeof(float));
#endif
        snprintf(kw, sizeof(kw), "%sc1.weight", pre);
        snprintf(kb, sizeof(kb), "%sc1.bias", pre);
        conv1d_fwd(bcl, W(ws, "", kw)->data, W(ws, "", kb)->data, c1, B, D, D, T, c->enc_conv_kernel, 1, c->enc_conv_kernel / 2);
        gelu_fwd(c1, c1, (int64_t)B * T * D);
        snprintf(kw, sizeof(kw), "%sc2.weight", pre);
        snprintf(kb, sizeof(kb), "%sc2.bias", pre);
        conv1d_fwd(c1, W(ws, "", kw)->data, W(ws, "", kb)->data, c2, B, D, D, T, c->enc_conv_kernel, 1, c->enc_conv_kernel / 2);
        /* x += transpose(c2) */
#ifdef NCM_CUDA
        cudaMemcpy(hb, c2, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hb, c2, n * sizeof(float));
#endif
        float *hx = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hx, x, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hx, x, n * sizeof(float));
#endif
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hx[((size_t)b * T + t) * D + d] += hb[((size_t)b * D + d) * T + t];
        free(hb);
#ifdef NCM_CUDA
        cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(x, hx, n * sizeof(float));
#endif
        free(hx);
    }
    int n_head_enc = c->vq_n_layers; /* n_layers is head count in vae blocks */
    for (int i = 0; i < c->enc_layers; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.encoder.blocks.%d.", i);
        vae_block(ws, pre, x, B, T, D, n_head_enc, c->vq_qk_norm, c->enc_attn_window);
    }
    if (c->enc_layers > 0 || c->enc_conv_layers > 0) {
        float *ln = arena_alloc((size_t)B * T * D);
        layernorm_fwd(x, W(ws, "tok.encoder.", "ln_e.weight")->data, NULL, ln, (int64_t)B * T, D);
        x = ln;
    }
    if (c->vq_compression > 1) {
        /* down conv stride f */
        float *bcl = arena_alloc((size_t)B * T * D);
        int T2 = T / c->vq_compression;
        float *cv = arena_alloc((size_t)B * T2 * D);
        size_t n = (size_t)B * T * D;
        float *h = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(h, x, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(h, x, n * sizeof(float));
#endif
        float *hb = (float *)malloc(n * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hb[((size_t)b * D + d) * T + t] = h[((size_t)b * T + t) * D + d];
        free(h);
#ifdef NCM_CUDA
        cudaMemcpy(bcl, hb, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(bcl, hb, n * sizeof(float));
#endif
        free(hb);
        conv1d_fwd(bcl, W(ws, "tok.encoder.", "down.weight")->data,
                   W(ws, "tok.encoder.", "down.bias")->data, cv, B, D, D, T, c->vq_compression, c->vq_compression, 0);
        /* back to BLD into z (T2 == context; for released cfgs compression==1) */
        float *hc = (float *)malloc((size_t)B * T2 * D * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hc, cv, (size_t)B * T2 * D * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hc, cv, (size_t)B * T2 * D * sizeof(float));
#endif
        float *hz = (float *)malloc((size_t)B * T2 * D * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < T2; t++) for (int d = 0; d < D; d++)
            hz[((size_t)b * T2 + t) * D + d] = hc[((size_t)b * D + d) * T2 + t];
        free(hc);
#ifdef NCM_CUDA
        cudaMemcpy(z, hz, (size_t)B * T2 * D * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(z, hz, (size_t)B * T2 * D * sizeof(float));
#endif
        free(hz);
        return;
    }
    /* pre-quant norm + copy to z */
    if (c->pre_quant_groupnorm > 0) {
        float *bcl = arena_alloc((size_t)B * T * D);
        float *gn = arena_alloc((size_t)B * T * D);
        size_t n = (size_t)B * T * D;
        float *h = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(h, x, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(h, x, n * sizeof(float));
#endif
        float *hb = (float *)malloc(n * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hb[((size_t)b * D + d) * T + t] = h[((size_t)b * T + t) * D + d];
        free(h);
#ifdef NCM_CUDA
        cudaMemcpy(bcl, hb, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(bcl, hb, n * sizeof(float));
#endif
        free(hb);
        groupnorm_fwd(bcl, gn, B, c->pre_quant_groupnorm, D, T);
        float *hg = (float *)malloc(n * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hg, gn, n * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hg, gn, n * sizeof(float));
#endif
        float *hz = (float *)malloc(n * sizeof(float));
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hz[((size_t)b * T + t) * D + d] = hg[((size_t)b * D + d) * T + t];
        free(hg);
#ifdef NCM_CUDA
        cudaMemcpy(z, hz, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(z, hz, n * sizeof(float));
#endif
        free(hz);
    } else {
#ifdef NCM_CUDA
        cudaMemcpy(z, x, (size_t)B * T * D * sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(z, x, (size_t)B * T * D * sizeof(float));
#endif
    }
}

void vqvae_decode(VQVAEState *v, const float *zhat, float *logits, int B) {
    NCMConfig *c = &v->cfg;
    WeightStore *ws = v->ws;
    int T = c->vq_context_length, D = c->vq_embed_dim, V = c->vq_in_vocab;
    float *x = arena_alloc((size_t)B * T * D);
#ifdef NCM_CUDA
    cudaMemcpy(x, zhat, (size_t)B * T * D * sizeof(float), cudaMemcpyDeviceToDevice);
#else
    memcpy(x, zhat, (size_t)B * T * D * sizeof(float));
#endif
    if (c->vq_compression > 1) {
        fprintf(stderr, "decoder upsampling (compression>1) not wired\n");
        exit(1);
    }
    if (c->dec_positional) {
        Tensor *wpe = W(ws, "tok.decoder.", "wpe.weight");
        size_t n = (size_t)B * T * D;
        float *hx = (float *)malloc(n * sizeof(float));
        float *hw = (float *)malloc((size_t)T * D * sizeof(float));
#ifdef NCM_CUDA
        cudaMemcpy(hx, x, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hw, wpe->data, (size_t)T * D * sizeof(float), cudaMemcpyDeviceToHost);
#else
        memcpy(hx, x, n * sizeof(float));
        memcpy(hw, wpe->data, (size_t)T * D * sizeof(float));
#endif
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++)
            hx[((size_t)b * T + t) * D + d] += hw[t * D + d];
        free(hw);
#ifdef NCM_CUDA
        cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice);
#else
        memcpy(x, hx, n * sizeof(float));
#endif
        free(hx);
    }
    vae_block(ws, "tok.decoder.middle_block.", x, B, T, D, c->vq_n_layers, c->vq_qk_norm, 0);
    for (int i = 0; i < c->dec_layers - 1; i++) {
        char pre[160];
        snprintf(pre, sizeof(pre), "tok.decoder.extra_blocks.%d.", i);
        vae_block(ws, pre, x, B, T, D, c->vq_n_layers, c->vq_qk_norm, 0);
    }
    {
        float *ln = arena_alloc((size_t)B * T * D);
        layernorm_fwd(x, W(ws, "tok.decoder.", "ln_f.weight")->data, NULL, ln, (int64_t)B * T, D);
        linear_fwd(ln, W(ws, "tok.decoder.", "output_layer.weight")->data, NULL,
                   logits, (int64_t)B * T, V, D);
    }
}
