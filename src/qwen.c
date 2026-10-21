/* qwen.c — native C Qwen2 decoder (verified on Qwen2.5-0.5B):
 * embeds, n_layer blocks (RMSNorm, GQA with q/k/v bias, RoPE theta from
 * conf, causal SDPA, o_proj, residual, RMSNorm, SwiGLU MLP), final
 * RMSNorm, tied lm_head. No QK-norm (Qwen2 has none — verified absent
 * from the checkpoint). CPU + CUDA via backend kernels.
 *
 * Weights: .ncm converted by tools/qwenimport (nn.Linear [out,in] needs
 * no transpose). Tokenizer: cncm kind=qwen (exact Qwen2 pretokenizer).
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define QWEN_MAX_LAYERS 64

typedef struct {
    float *ln1w;
    float *qw, *qb, *kw, *kb, *vw, *vb, *ow;
    float *ln2w, *gatew, *upw, *downw;
} QwenBlock;

typedef struct {
    WeightStore ws;
    TokState *tok;
    int n_layer, n_head, n_kv, n_embd, n_inner, n_ctx, vocab_size;
    double rope_theta;
    float *wte, *lnfw, *cosb, *sinb;
    QwenBlock blk[QWEN_MAX_LAYERS];
} QWEN;

static float *qw_get(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "qwen: missing tensor %s\n", name); exit(1); }
    return t->data;
}

static int qwen_open(QWEN *m, const NCMConfig *c, const char *weights, const char *adir) {
    memset(m, 0, sizeof(*m));
    arena_init_default();
    if (weights_load(&m->ws, weights)) return 1;
    m->tok = tok_load(c, adir);
    m->n_layer = c->n_layer;
    m->n_head = c->n_head;
    m->n_kv = c->n_kv_heads > 0 ? c->n_kv_heads : c->n_head;
    m->n_embd = c->n_embd;
    m->n_ctx = c->context_length > 0 ? c->context_length : 4096;
    m->rope_theta = c->rope_theta > 0 ? c->rope_theta : 1000000.0;
    Tensor *te = weights_get(&m->ws, "model.embed_tokens.weight");
    if (!te) return 1;
    m->vocab_size = (int)te->shape[0];
    m->wte = te->data;
    m->lnfw = qw_get(&m->ws, "model.norm.weight");
    if (m->n_layer > QWEN_MAX_LAYERS) { fprintf(stderr, "qwen: too many layers\n"); return 1; }
    char pre[128], kk[192];
    for (int li = 0; li < m->n_layer; li++) {
        snprintf(pre, sizeof(pre), "model.layers.%d.", li);
        QwenBlock *b = &m->blk[li];
#define QW(nm, dst) snprintf(kk, sizeof(kk), "%s" nm, pre); dst = qw_get(&m->ws, kk)
        QW("input_layernorm.weight", b->ln1w);
        QW("self_attn.q_proj.weight", b->qw); QW("self_attn.q_proj.bias", b->qb);
        QW("self_attn.k_proj.weight", b->kw); QW("self_attn.k_proj.bias", b->kb);
        QW("self_attn.v_proj.weight", b->vw); QW("self_attn.v_proj.bias", b->vb);
        QW("self_attn.o_proj.weight", b->ow);
        QW("post_attention_layernorm.weight", b->ln2w);
        QW("mlp.gate_proj.weight", b->gatew);
        QW("mlp.up_proj.weight", b->upw);
        QW("mlp.down_proj.weight", b->downw);
#undef QW
    }
    /* inner dim from gate weight shape [I][C] */
    {
        char kk2[192];
        snprintf(kk2, sizeof(kk2), "model.layers.0.mlp.gate_proj.weight");
        Tensor *t = weights_get(&m->ws, kk2);
        if (!t) return 1;
        m->n_inner = (int)t->shape[0];
    }
    /* RoPE tables [n_ctx][hd], pair-duplicated angles (NeoX convention) */
    {
        int hd = m->n_embd / m->n_head, T = m->n_ctx;
        int h = hd / 2;
        float *hc = (float *)malloc((size_t)T * hd * sizeof(float));
        float *hs = (float *)malloc((size_t)T * hd * sizeof(float));
        if (!hc || !hs) return 1;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < h; d++) {
                double a = (double)t * pow(m->rope_theta, -(double)(2 * d) / (double)hd);
                float co = (float)cos(a), si = (float)sin(a);
                hc[(size_t)t * hd + d] = co;
                hc[(size_t)t * hd + d + h] = co;
                hs[(size_t)t * hd + d] = si;
                hs[(size_t)t * hd + d + h] = si;
            }
        }
    /* RoPE tables: persistent (parena), never reset by forward. */
    parena_init((size_t)(2 * T * hd * sizeof(float)) + (1 << 20));
    m->cosb = parena_alloc((size_t)T * hd);
    m->sinb = parena_alloc((size_t)T * hd);
        mem_h2d(m->cosb, hc, (size_t)T * hd * sizeof(float));
        mem_h2d(m->sinb, hs, (size_t)T * hd * sizeof(float));
        free(hc);
        free(hs);
    }
    return 0;
}

static void qwen_close(QWEN *m) {
    tok_free(m->tok);
    (void)m;
}

/* Forward: ids[T] -> logits. last_only: emit only final position (gen).
 * Caller owns arena discipline. */
static float *qwen_forward(QWEN *m, const int *ids, int T, int last_only) {
    /* owns the scratch arena for the call: reset first. Callers must
     * backend_to_host anything they keep (cos/sin tables live in parena). */
    arena_reset();
    int C = m->n_embd, H = m->n_head, K = m->n_kv, hd = C / H, I = m->n_inner, V = m->vocab_size;
    int *bids = (int *)arena_alloc((size_t)T);
    mem_h2d(bids, ids, (size_t)T * sizeof(int));
    float *h = arena_alloc((size_t)T * C);
    embedding_fwd(m->wte, bids, h, T, C);
    float *mask = arena_alloc((size_t)T * T);
    {
        float *hm = (float *)malloc((size_t)T * T * sizeof(float));
        for (int tq = 0; tq < T; tq++)
            for (int tk = 0; tk < T; tk++)
                hm[(size_t)tq * T + tk] = (tk <= tq) ? 0.0f : -1e9f;
        mem_h2d(mask, hm, (size_t)T * T * sizeof(float));
        free(hm);
    }
    for (int li = 0; li < m->n_layer; li++) {
        QwenBlock *b = &m->blk[li];
        float *h1 = arena_alloc((size_t)T * C);
        rmsnorm_fwd(h, b->ln1w, h1, T, C);
        float *ql = arena_alloc((size_t)T * C), *kl = arena_alloc((size_t)T * K * hd),
                *vl = arena_alloc((size_t)T * K * hd);
        linear_fwd(h1, b->qw, b->qb, ql, T, C, C);
        linear_fwd(h1, b->kw, b->kb, kl, T, K * hd, C);
        linear_fwd(h1, b->vw, b->vb, vl, T, K * hd, C);
        float *qq = arena_alloc((size_t)H * T * hd), *kk = arena_alloc((size_t)K * T * hd),
                *vv = arena_alloc((size_t)K * T * hd);
        heads_transpose_fwd(ql, qq, T, H, hd);
        heads_transpose_fwd(kl, kk, T, K, hd);
        heads_transpose_fwd(vl, vv, T, K, hd);
        rope_apply_gqa(qq, kk, m->cosb, m->sinb, 1, H, K, T, hd);
        float *ctx = arena_alloc((size_t)H * T * hd);
        sdpa_fwd_gqa(qq, kk, vv, mask, ctx, 1, H, K, T, T, hd);
        float *mo = arena_alloc((size_t)T * C);
        head_merge_fwd(ctx, mo, 1, T, C, H, hd);
        float *ao = arena_alloc((size_t)T * C);
        linear_fwd(mo, b->ow, NULL, ao, T, C, C);
        float *x1 = arena_alloc((size_t)T * C);
        add_fwd(h, ao, x1, (size_t)T * C);
        float *h2 = arena_alloc((size_t)T * C);
        rmsnorm_fwd(x1, b->ln2w, h2, T, C);
        float *gt = arena_alloc((size_t)T * I);
        linear_fwd(h2, b->gatew, NULL, gt, T, I, C);
        float *up = arena_alloc((size_t)T * I);
        linear_fwd(h2, b->upw, NULL, up, T, I, C);
        silu_fwd(gt, gt, (int64_t)T * I);
        float *gu = arena_alloc((size_t)T * I);
        mul_fwd(gt, up, gu, (int64_t)T * I);
        float *mlpo = arena_alloc((size_t)T * C);
        linear_fwd(gu, b->downw, NULL, mlpo, T, C, I);
        float *out = arena_alloc((size_t)T * C);
        add_fwd(x1, mlpo, out, (size_t)T * C);
        h = out;
    }
    float *lnf = arena_alloc((size_t)T * C);
    rmsnorm_fwd(h, m->lnfw, lnf, T, C);
    float *logits;
    if (last_only) {
        float *last = arena_alloc((size_t)C);
        mem_d2d(last, lnf + (size_t)(T - 1) * C, (size_t)C * sizeof(float));
        logits = arena_alloc((size_t)V);
        linear_fwd(last, m->wte, NULL, logits, 1, V, C);
    } else {
        logits = arena_alloc((size_t)T * (size_t)V);
        linear_fwd(lnf, m->wte, NULL, logits, T, V, C);
    }
    return logits;
}

static double qwen_nll(QWEN *m, const int *ids, int T) {
    float *lg = qwen_forward(m, ids, T, 0);
    float *hl = (float *)backend_to_host(lg, (size_t)T * (size_t)m->vocab_size * sizeof(float));
    int V = m->vocab_size;
    double tot = 0;
    for (int t = 0; t + 1 < T; t++) {
        const float *row = hl + (size_t)t * V;
        double mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        double den = 0;
        for (int v = 0; v < V; v++) den += exp(row[v] - mx);
        tot += -log(exp(row[ids[t + 1]] - mx) / den + 1e-30);
    }
    free(hl);
    return tot / (T - 1);
}

static int qwen_generate(QWEN *m, const int *prompt, int np, int n_new,
                         float temp, int top_k, float top_p, unsigned long long *rng,
                         int *out) {
    int cap = np + n_new;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    memcpy(ids, prompt, (size_t)np * sizeof(int));
    int n = np;
    int eos = tok_eos(m->tok);
    for (int s = 0; s < n_new; s++) {
        if (n > m->n_ctx) { fprintf(stderr, "qwen: context overflow\n"); break; }
        float *lg = qwen_forward(m, ids, n, 1);
        float *row = (float *)backend_to_host(lg, (size_t)m->vocab_size * sizeof(float));
        int nx;
        if (!(temp > 0)) nx = argmax_row(row, m->vocab_size);
        else nx = sample_topkp(row, m->vocab_size, temp, top_k, top_p, rng);
        free(row);
        ids[n++] = nx;
        if (nx == eos) break;
    }
    memcpy(out, ids + np, (size_t)(n - np) * sizeof(int));
    free(ids);
    return n - np;
}

int qwen_cmd_gen(const NCMConfig *c, const char *weights, const char *adir,
                 const char *prompt, const char *ids_csv, int n_new,
                 float temp, int top_k, float top_p, unsigned long long seed) {
    QWEN m;
    if (qwen_open(&m, c, weights, adir)) return 1;
    int ids_buf[8192], n = 0;
    if (ids_csv) {
        const char *p = ids_csv;
        while (*p && n < 8192) { ids_buf[n++] = atoi(p); while (*p && *p != ',') p++; if (*p) p++; }
    } else if (prompt) {
        n = tok_encode(m.tok, prompt, ids_buf, 8192);
    } else {
        fprintf(stderr, "qwen-gen needs --prompt TEXT | --ids CSV\n");
        return 2;
    }
    if (n == 0) { fprintf(stderr, "qwen-gen: empty prompt\n"); return 2; }
    unsigned long long rng = seed ? seed : 0x12345678ULL;
    int *gen = (int *)malloc((size_t)n_new * sizeof(int));
    int ngen = qwen_generate(&m, ids_buf, n, n_new, temp, top_k, top_p, &rng, gen);
    printf("ids:");
    for (int i = 0; i < ngen; i++) printf(" %d", gen[i]);
    printf("\n");
    {
        char *text = (char *)malloc(65536);
        tok_decode(m.tok, gen, ngen, text, 65536, 1);
        printf("%s\n", text);
        free(text);
    }
    free(gen);
    qwen_close(&m);
    return 0;
}

int qwen_cmd_ppl(const NCMConfig *c, const char *weights, const char *adir, const char *texts_path) {
    QWEN m;
    if (qwen_open(&m, c, weights, adir)) return 1;
    FILE *f = fopen(texts_path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", texts_path); return 1; }
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    double tot = 0;
    long ntok = 0;
    int ndoc = 0;
    int *ids = (int *)malloc(8192 * sizeof(int));
    while ((len = getline(&line, &cap, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len) continue;
        int n = tok_encode(m.tok, line, ids, 8192);
        if (n < 2) continue;
        for (int o = 0; o + 1 < n; o += 512) {
            int T = n - o > 512 ? 512 : n - o;
            if (T < 2) break;
            tot += qwen_nll(&m, ids + o, T) * (T - 1);
            ntok += T - 1;
        }
        ndoc++;
    }
    free(ids);
    free(line);
    fclose(f);
    double nll = tot / ntok;
    printf("docs=%d tokens=%ld nll=%.6f ppl=%.6f\n", ndoc, ntok, nll, exp(nll));
    qwen_close(&m);
    return 0;
}

/* ---------- verification against transformers fixtures ---------- */

static int qhexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int qunhex(const unsigned char *h, size_t n, unsigned char *out, size_t cap) {
    if (n % 2) return -1;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 2) {
        int a = qhexval((char)h[i]), b = qhexval((char)h[i + 1]);
        if (a < 0 || b < 0 || o >= cap) return -1;
        out[o++] = (unsigned char)((a << 4) | b);
    }
    return (int)o;
}

static int qread_ids(const char *s, int *out, int cap) {
    int n = 0;
    while (*s && n < cap) {
        while (*s == ' ') s++;
        if (!*s || *s == '\n' || *s == '\r') break;
        out[n++] = atoi(s);
        while (*s && *s != ' ' && *s != '\n' && *s != '\r') s++;
    }
    return n;
}

static int qwen_check_ids(QWEN *m, const char *name, const unsigned char *text,
                          const int *exp, int nexp) {
    int got[8192];
    int n = tok_encode(m->tok, (const char *)text, got, 8192);
    if (n != nexp) {
        printf("FAIL tok %s: len %d vs %d\n", name, n, nexp);
        return 1;
    }
    for (int i = 0; i < n; i++)
        if (got[i] != exp[i]) {
            printf("FAIL tok %s: pos %d got %d want %d\n", name, i, got[i], exp[i]);
            return 1;
        }
    printf("ok   tok %s (%d ids exact)\n", name, n);
    return 0;
}

static int qwen_check_logits(QWEN *m, const char *dir, const char *name,
                             const int *ids, int T) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/qwen_logits_%s.f32", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL logits %s: missing %s\n", name, path); return 1; }
    int V = m->vocab_size;
    float *ref = (float *)malloc((size_t)T * V * sizeof(float));
    if (fread(ref, sizeof(float), (size_t)T * V, f) != (size_t)T * V) {
        printf("FAIL logits %s: short read\n", name);
        fclose(f);
        free(ref);
        return 1;
    }
    fclose(f);
    float *lg = qwen_forward(m, ids, T, 0);
    float *got = (float *)backend_to_host(lg, (size_t)T * V * sizeof(float));
    double maxabs = 0, maxrel = 0;
    for (int64_t i = 0; i < (int64_t)T * V; i++) {
        double a = got[i], b = ref[i], d = fabs(a - b);
        if (d > maxabs) maxabs = d;
        double r = d / (fabs(b) + 1.0);
        if (r > maxrel) maxrel = r;
    }
    free(ref);
    free(got);
    /* Threshold rationale (measured 2026-10): torch-fp32 vs torch-fp64 on the
     * same input self-disagrees at maxrel=1.14e-4 (BF16-sourced 24-layer net),
     * so 1e-4 sits AT the fp32 noise floor, not above it. 5e-4 keeps headroom
     * while any real defect seen so far (missing /den: rel~10; stride bugs:
     * rel 0.2-1.0; denom bug: 20x) exceeds it by orders of magnitude. */
    if (maxrel < 5e-4 && maxabs < 1e-3) {
        printf("ok   logits %s: maxabs=%.3g maxrel=%.3g\n", name, maxabs, maxrel);
        return 0;
    }
    printf("FAIL logits %s: maxabs=%.3g maxrel=%.3g\n", name, maxabs, maxrel);
    return 1;
}

int qwen_cmd_verify(const NCMConfig *c, const char *weights, const char *adir,
                    const char *dir) {
    QWEN m;
    if (qwen_open(&m, c, weights, adir)) return 1;
    int fails = 0, checked = 0;
    char path[1024], line[65536];
    FILE *f;
    snprintf(path, sizeof(path), "%s/qwen_tok.txt", dir);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
    while (fgets(line, sizeof(line), f)) {
        char *tab1 = strchr(line, '\t');
        if (!tab1) continue;
        *tab1 = 0;
        char *tab2 = strchr(tab1 + 1, '\t');
        if (!tab2) continue;
        *tab2 = 0;
        unsigned char text[65536];
        int nt = qunhex((unsigned char *)(tab1 + 1), strlen(tab1 + 1), text, sizeof(text) - 1);
        if (nt < 0) { printf("FAIL tok %s: bad hex\n", line); fails++; checked++; continue; }
        text[nt] = 0;
        int exp[8192];
        int ne = qread_ids(tab2 + 1, exp, 8192);
        fails += qwen_check_ids(&m, line, text, exp, ne);
        checked++;
        fails += qwen_check_logits(&m, dir, line, exp, ne);
        checked++;
    }
    fclose(f);
    snprintf(path, sizeof(path), "%s/qwen_gen.txt", dir);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
    while (fgets(line, sizeof(line), f)) {
        char *tab1 = strchr(line, '\t');
        if (!tab1) continue;
        *tab1 = 0;
        char *tab2 = strchr(tab1 + 1, '\t');
        if (!tab2) continue;
        *tab2 = 0;
        unsigned char text[65536];
        int nt = qunhex((unsigned char *)(tab1 + 1), strlen(tab1 + 1), text, sizeof(text) - 1);
        if (nt < 0) continue;
        text[nt] = 0;
        int exp[256];
        int ne = qread_ids(tab2 + 1, exp, 256);
        int prompt[8192];
        int np = tok_encode(m.tok, (const char *)text, prompt, 8192);
        int *gen = (int *)malloc((size_t)(ne > 0 ? ne : 1) * sizeof(int));
        unsigned long long rng = 0;
        int ng = qwen_generate(&m, prompt, np, ne, 0.0f, 0, 1.0f, &rng, gen);
        int ok = (ng == ne);
        for (int i = 0; ok && i < ne; i++) if (gen[i] != exp[i]) ok = 0;
        if (ok) printf("ok   gen %s (%d tokens exact)\n", line, ne);
        else {
            printf("FAIL gen %s: got %d tokens, want %d\n", line, ng, ne);
            fails++;
        }
        free(gen);
        checked++;
    }
    fclose(f);
    snprintf(path, sizeof(path), "%s/qwen_ppl.txt", dir);
    f = fopen(path, "r");
    if (f && fgets(line, sizeof(line), f)) {
        char *tab1 = strchr(line, '\t');
        char *tab2 = tab1 ? strchr(tab1 + 1, '\t') : NULL;
        if (tab1 && tab2) {
            *tab1 = 0;
            *tab2 = 0;
            unsigned char text[65536];
            int nt = qunhex((unsigned char *)line, strlen(line), text, sizeof(text) - 1);
            double want_nll = atof(tab1 + 1), want_ppl = atof(tab2 + 1);
            if (nt >= 0) {
                text[nt] = 0;
                int ids[8192];
                int n = tok_encode(m.tok, (const char *)text, ids, 8192);
                double nll = qwen_nll(&m, ids, n);
                double ppl = exp(nll);
                double d = fabs(nll - want_nll);
                if (d < 1e-4) printf("ok   ppl nll=%.6f (want %.6f) ppl=%.4f (want %.4f)\n",
                                     nll, want_nll, ppl, want_ppl);
                else { printf("FAIL ppl nll=%.6f want %.6f\n", nll, want_nll); fails++; }
                checked++;
            }
        }
    }
    if (f) fclose(f);
    printf("QWEN-VERIFY %s (%d checked)\n", fails ? "FAIL" : "PASS", checked);
    qwen_close(&m);
    return fails != 0;
}
