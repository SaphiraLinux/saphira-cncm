/* gpt2.c — native C GPT-2 decoder (e.g. 124M): wte+wpe, n_layer blocks,
 * final LayerNorm (eps=1e-5), tied lm_head. CPU + CUDA via backend kernels.
 *
 * Weights: .ncm converted from HF safetensors by tools/gpt2import (2-D
 * [in,out] Conv1D weights stored transposed to [out][in]; .attn.bias
 * mask buffers skipped, mask rebuilt here).
 * Tokenizer: cncm byte-BPE (bpe_vocab.txt/bpe_merges.txt) with the exact
 * GPT-2 splitter; ids must match HF tokenization (verified in tests).
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

#define GPT2_LN_EPS 1e-5f
#define GPT2_MAX_LAYERS 96

typedef struct {
    float *ln1w, *ln1b, *attnw, *attnb, *cprojw, *cprojb;
    float *ln2w, *ln2b, *cfcw, *cfcb, *p2w, *p2b;
} GPT2Block;

typedef struct {
    WeightStore ws;
    TokState *tok;
    int n_layer, n_head, n_embd, n_ctx, vocab_size, n_inner;
    float *wte, *wpe, *lnfw, *lnfb;
    GPT2Block blk[GPT2_MAX_LAYERS];
} GPT2;

static float *g2w(WeightStore *ws, const char *name) {
    Tensor *t = weights_get(ws, name);
    if (!t) { fprintf(stderr, "gpt2: missing tensor %s\n", name); exit(1); }
    return t->data;
}

static int gpt2_open(GPT2 *g, const NCMConfig *c, const char *weights, const char *adir) {
    memset(g, 0, sizeof(*g));
    arena_init_default(); /* 1 GiB scratch; logits alone are T*V floats */
    if (weights_load(&g->ws, weights)) return 1;
    g->tok = tok_load(c, adir);
    g->n_layer = c->n_layer;
    g->n_head = c->n_head;
    g->n_embd = c->n_embd;
    g->n_ctx = c->context_length > 0 ? c->context_length : 1024;
    g->vocab_size = (int)g->ws.items[0].t->n; /* replaced below */
    /* vocab size = wte rows */
    Tensor *wte_t = weights_get(&g->ws, "wte.weight");
    if (!wte_t) return 1;
    g->vocab_size = (int)wte_t->shape[0];
    g->n_inner = 4 * g->n_embd;
    if (g->n_layer > GPT2_MAX_LAYERS) { fprintf(stderr, "gpt2: too many layers\n"); return 1; }
    g->wte = wte_t->data;
    g->wpe = g2w(&g->ws, "wpe.weight");
    g->lnfw = g2w(&g->ws, "ln_f.weight");
    g->lnfb = g2w(&g->ws, "ln_f.bias");
    char pre[64], kk[160];
    for (int li = 0; li < g->n_layer; li++) {
        snprintf(pre, sizeof(pre), "h.%d.", li);
        GPT2Block *b = &g->blk[li];
#define GW(f, n) snprintf(kk, sizeof(kk), "%s" n, pre); b->f = g2w(&g->ws, kk)
#define GWO(f, n) snprintf(kk, sizeof(kk), "%s" n, pre); \
    do { Tensor *t = weights_get(&g->ws, kk); b->f = t ? t->data : NULL; } while (0)
        GW(ln1w, "ln_1.weight"); GW(ln1b, "ln_1.bias");
        GW(attnw, "attn.c_attn.weight"); GWO(attnb, "attn.c_attn.bias");
        GW(cprojw, "attn.c_proj.weight"); GWO(cprojb, "attn.c_proj.bias");
        GW(ln2w, "ln_2.weight"); GW(ln2b, "ln_2.bias");
        GW(cfcw, "mlp.c_fc.weight"); GWO(cfcb, "mlp.c_fc.bias");
        GW(p2w, "mlp.c_proj.weight"); GWO(p2b, "mlp.c_proj.bias");
#undef GW
#undef GWO
    }
    return 0;
}

static void gpt2_close(GPT2 *g) {
    tok_free(g->tok);
    for (int i = 0; i < g->ws.count; i++) {
        /* free backend tensors via tensor_free if available; else leak-on-exit is fine for CLI */
    }
    (void)g;
}

/* Forward: ids[T] -> backend logits [T][V]. Caller owns arena discipline:
 * resets before calling; stage out what it needs before the next reset. */
static float *gpt2_forward(GPT2 *g, const int *ids, int T) {
    /* owns the scratch arena for the call: reset first. Callers must
     * backend_to_host anything they keep (GPT2 holds no arena state). */
    arena_reset();
    int C = g->n_embd, H = g->n_head, hd = C / H, V = g->vocab_size;
    /* input + positions */
    int *bids = (int *)arena_alloc((size_t)T);
    mem_h2d(bids, ids, (size_t)T * sizeof(int));
    int *pids = (int *)arena_alloc((size_t)T);
    {
        int *ph = (int *)malloc((size_t)T * sizeof(int));
        for (int t = 0; t < T; t++) ph[t] = t;
        mem_h2d(pids, ph, (size_t)T * sizeof(int));
        free(ph);
    }
    float *xe = arena_alloc((size_t)T * C), *pe = arena_alloc((size_t)T * C);
    embedding_fwd(g->wte, bids, xe, T, C);
    embedding_fwd(g->wpe, pids, pe, T, C);
    float *h = arena_alloc((size_t)T * C);
    add_fwd(xe, pe, h, (size_t)T * C);
    /* causal mask [T][T] */
    float *mask = arena_alloc((size_t)T * T);
    {
        float *hm = (float *)malloc((size_t)T * T * sizeof(float));
        for (int tq = 0; tq < T; tq++)
            for (int tk = 0; tk < T; tk++)
                hm[(size_t)tq * T + tk] = (tk <= tq) ? 0.0f : -1e9f;
        mem_h2d(mask, hm, (size_t)T * T * sizeof(float));
        free(hm);
    }
    int64_t W3 = (int64_t)3 * C;
    for (int li = 0; li < g->n_layer; li++) {
        GPT2Block *b = &g->blk[li];
        float *ln = arena_alloc((size_t)T * C);
        layernorm_eps_fwd(h, b->ln1w, b->ln1b, ln, T, C, GPT2_LN_EPS);
        float *qkv = arena_alloc((size_t)T * (size_t)W3);
        linear_fwd(ln, b->attnw, b->attnb, qkv, T, (int)W3, C);
        float *qq = arena_alloc((size_t)H * T * hd), *kk = arena_alloc((size_t)H * T * hd),
                *vv = arena_alloc((size_t)H * T * hd);
        qkv_split_fwd(qkv, qq, kk, vv, 1, T, C, H, hd, H, 1); /* K=H, rep=1 */
        float *ctx = arena_alloc((size_t)H * T * hd);
        sdpa_fwd(qq, kk, vv, mask, ctx, 1, H, T, T, hd);
        float *mo = arena_alloc((size_t)T * C);
        head_merge_fwd(ctx, mo, 1, T, C, H, hd);
        float *ao = arena_alloc((size_t)T * C);
        linear_fwd(mo, b->cprojw, b->cprojb, ao, T, C, C);
        float *x1 = arena_alloc((size_t)T * C);
        add_fwd(h, ao, x1, (size_t)T * C);
        float *mn = arena_alloc((size_t)T * C);
        layernorm_eps_fwd(x1, b->ln2w, b->ln2b, mn, T, C, GPT2_LN_EPS);
        float *mm = arena_alloc((size_t)T * (size_t)g->n_inner);
        linear_fwd(mn, b->cfcw, b->cfcb, mm, T, g->n_inner, C);
        float *m2 = arena_alloc((size_t)T * (size_t)g->n_inner);
        gelu_tanh_fwd(mm, m2, (int64_t)T * g->n_inner);
        float *mlpo = arena_alloc((size_t)T * C);
        linear_fwd(m2, b->p2w, b->p2b, mlpo, T, C, g->n_inner);
        float *out = arena_alloc((size_t)T * C);
        add_fwd(x1, mlpo, out, (size_t)T * C);
        h = out;
    }
    float *lnf = arena_alloc((size_t)T * C);
    layernorm_eps_fwd(h, g->lnfw, g->lnfb, lnf, T, C, GPT2_LN_EPS);
    float *logits = arena_alloc((size_t)T * (size_t)V);
    linear_fwd(lnf, g->wte, NULL, logits, T, V, C); /* tied head */
    return logits;
}

/* Cross-entropy NLL of `ids[T]` (mean over shifted positions). */
static double gpt2_nll(GPT2 *g, const int *ids, int T) {
    float *lg = gpt2_forward(g, ids, T);
    float *hl = (float *)backend_to_host(lg, (size_t)T * (size_t)g->vocab_size * sizeof(float));
    int V = g->vocab_size;
    double tot = 0;
    for (int t = 0; t + 1 < T; t++) {
        const float *row = hl + (size_t)t * V;
        double mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        double den = 0;
        for (int v = 0; v < V; v++) den += exp(row[v] - mx);
        int tg = ids[t + 1];
        tot += -log(exp(row[tg] - mx) / den + 1e-30);
    }
    free(hl);
    return tot / (T - 1);
}

/* Greedy/sampling generation. Appends up to n_new ids; returns new count. */
static int gpt2_generate(GPT2 *g, const int *prompt, int np, int n_new,
                         float temp, int top_k, float top_p, unsigned long long *rng,
                         int *out) {
    int cap = np + n_new;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    memcpy(ids, prompt, (size_t)np * sizeof(int));
    int n = np;
    int eos = tok_eos(g->tok);
    for (int s = 0; s < n_new; s++) {
        if (n > g->n_ctx) { fprintf(stderr, "gpt2: context overflow\n"); break; }
        float *lg = gpt2_forward(g, ids, n);
        float *row = (float *)backend_to_host(lg + (size_t)(n - 1) * g->vocab_size,
                                     (size_t)g->vocab_size * sizeof(float));
        int nx;
        if (!(temp > 0)) nx = argmax_row(row, g->vocab_size);
        else nx = sample_topkp(row, g->vocab_size, temp, top_k, top_p, rng);
        free(row);
        ids[n++] = nx;
        if (nx == eos) break;
    }
    memcpy(out, ids + np, (size_t)(n - np) * sizeof(int));
    free(ids);
    return n - np;
}

int gpt2_cmd_gen(const NCMConfig *c, const char *weights, const char *adir,
                 const char *prompt, const char *ids_csv, int n_new,
                 float temp, int top_k, float top_p, unsigned long long seed) {
    GPT2 g;
    if (gpt2_open(&g, c, weights, adir)) return 1;
    int ids_buf[4096], n = 0;
    if (ids_csv) {
        const char *p = ids_csv;
        while (*p && n < 4096) { ids_buf[n++] = atoi(p); while (*p && *p != ',') p++; if (*p) p++; }
    } else if (prompt) {
        n = tok_encode(g.tok, prompt, ids_buf, 4096);
    } else {
        fprintf(stderr, "gpt2-gen needs --prompt TEXT or --ids CSV\n");
        return 2;
    }
    if (n == 0) { fprintf(stderr, "gpt2-gen: empty prompt\n"); return 2; }
    unsigned long long rng = seed ? seed : 0x12345678ULL;
    int *gen = (int *)malloc((size_t)n_new * sizeof(int));
    int ngen = gpt2_generate(&g, ids_buf, n, n_new, temp, top_k, top_p, &rng, gen);
    printf("ids:");
    for (int i = 0; i < ngen; i++) printf(" %d", gen[i]);
    printf("\n");
    {
        char *text = (char *)malloc(65536);
        tok_decode(g.tok, gen, ngen, text, 65536, 1);
        printf("%s\n", text);
        free(text);
    }
    free(gen);
    gpt2_close(&g);
    return 0;
}

int gpt2_cmd_ppl(const NCMConfig *c, const char *weights, const char *adir, const char *texts_path) {
    GPT2 g;
    if (gpt2_open(&g, c, weights, adir)) return 1;
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
        int n = tok_encode(g.tok, line, ids, 8192);
        if (n < 2) continue;
        /* non-overlapping chunks of n_ctx */
        for (int o = 0; o + 1 < n; o += g.n_ctx) {
            int T = n - o > g.n_ctx ? g.n_ctx : n - o;
            if (T < 2) break;
            tot += gpt2_nll(&g, ids + o, T) * (T - 1);
            ntok += T - 1;
        }
        ndoc++;
    }
    free(ids);
    free(line);
    fclose(f);
    double nll = tot / ntok;
    printf("docs=%d tokens=%ld nll=%.6f ppl=%.6f\n", ndoc, ntok, nll, exp(nll));
    gpt2_close(&g);
    return 0;
}

/* Taped-vs-reference parity: train_gpt2_forward must reproduce
 * gpt2_forward bit-comparably on the same weights (both use eps=1e-5
 * LayerNorm and gelu_new; any arch mismatch shows up here first). */
int gpt2_cmd_taped(const NCMConfig *c, const char *weights, const char *adir,
                   const char *prompt, int T) {
    GPT2 g;
    if (gpt2_open(&g, c, weights, adir)) return 1;
    int ids_buf[8192];
    int n = 0;
    if (prompt) n = tok_encode(g.tok, prompt, ids_buf, 8192);
    if (n < 2) { fprintf(stderr, "gpt2-taped needs --prompt TEXT (>=2 tokens)\n"); return 2; }
    if (T > 0 && n > T) n = T;
    int V = g.vocab_size;
    /* reference */
    float *lg = gpt2_forward(&g, ids_buf, n);
    float *href = (float *)backend_to_host(lg, (size_t)n * V * sizeof(float));
    /* taped */
    arena_reset();
    Tape *tp = tape_new();
    float *tl = train_gpt2_forward(tp, &g.ws, c, ids_buf, 1, n);
    float *htap = (float *)backend_to_host(tl, (size_t)n * V * sizeof(float));
    tape_free(tp);
    double mx = 0;
    int64_t mi = -1;
    for (int64_t i = 0; i < (int64_t)n * V; i++) {
        double d = fabs((double)href[i] - htap[i]);
        if (d > mx) { mx = d; mi = i; }
    }
    printf("gpt2-taped T=%d V=%d maxabs=%.3g at %lld (ref=%.6g tap=%.6g)\n",
           n, V, mx, (long long)mi,
           mi >= 0 ? href[mi] : 0.0, mi >= 0 ? htap[mi] : 0.0);
    int rc = (mx == 0.0) ? 0 : (mx < 1e-5 ? 0 : 1);
    printf("GPT2-TAPED %s\n", rc ? "FAIL" : "PASS");
    free(href); free(htap);
    gpt2_close(&g);
    return rc;
}

/* ---------- verification against transformers fixtures ---------- */

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* hex -> bytes. Returns byte count or -1. */
static int unhex2(const unsigned char *h, size_t n, unsigned char *out, size_t cap) {
    if (n % 2) return -1;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 2) {
        int a = hexval((char)h[i]), b = hexval((char)h[i + 1]);
        if (a < 0 || b < 0 || o >= cap) return -1;
        out[o++] = (unsigned char)((a << 4) | b);
    }
    return (int)o;
}

static int read_ids(const char *s, int *out, int cap) {
    int n = 0;
    while (*s && n < cap) {
        while (*s == ' ') s++;
        if (!*s || *s == '\n' || *s == '\r') break;
        out[n++] = atoi(s);
        while (*s && *s != ' ' && *s != '\n' && *s != '\r') s++;
    }
    return n;
}

static int gpt2_check_ids(GPT2 *g, const char *name, const unsigned char *text,
                          const int *exp, int nexp) {
    int got[8192];
    int n = tok_encode(g->tok, (const char *)text, got, 8192);
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

static int gpt2_check_logits(GPT2 *g, const char *dir, const char *name,
                             const int *ids, int T) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/gpt2_logits_%s.f32", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL logits %s: missing %s\n", name, path); return 1; }
    int V = g->vocab_size;
    float *ref = (float *)malloc((size_t)T * V * sizeof(float));
    if (fread(ref, sizeof(float), (size_t)T * V, f) != (size_t)T * V) {
        printf("FAIL logits %s: short read\n", name);
        fclose(f);
        free(ref);
        return 1;
    }
    fclose(f);
    float *lg = gpt2_forward(g, ids, T);
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
    if (maxrel < 1e-4 && maxabs < 1e-2) {
        printf("ok   logits %s: maxabs=%.3g maxrel=%.3g\n", name, maxabs, maxrel);
        return 0;
    }
    printf("FAIL logits %s: maxabs=%.3g maxrel=%.3g\n", name, maxabs, maxrel);
    return 1;
}

int gpt2_cmd_verify(const NCMConfig *c, const char *weights, const char *adir,
                    const char *dir) {
    GPT2 g;
    if (gpt2_open(&g, c, weights, adir)) return 1;
    int fails = 0, checked = 0;
    char path[1024], line[65536];
    /* 1+2. tokenization + logits */
    snprintf(path, sizeof(path), "%s/gpt2_tok.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "missing %s\n", path); return 1; }
    while (fgets(line, sizeof(line), f)) {
        char *tab1 = strchr(line, '\t');
        if (!tab1) continue;
        *tab1 = 0;
        char *tab2 = strchr(tab1 + 1, '\t');
        if (!tab2) continue;
        *tab2 = 0;
        unsigned char text[65536];
        int nt = unhex2((unsigned char *)(tab1 + 1), strlen(tab1 + 1), text, sizeof(text) - 1);
        if (nt < 0) { printf("FAIL tok %s: bad hex\n", line); fails++; checked++; continue; }
        text[nt] = 0;
        int exp[8192];
        int ne = read_ids(tab2 + 1, exp, 8192);
        fails += gpt2_check_ids(&g, line, text, exp, ne);
        checked++;
        fails += gpt2_check_logits(&g, dir, line, exp, ne);
        checked++;
    }
    fclose(f);
    /* 3. greedy generations */
    snprintf(path, sizeof(path), "%s/gpt2_gen.txt", dir);
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
        int nt = unhex2((unsigned char *)(tab1 + 1), strlen(tab1 + 1), text, sizeof(text) - 1);
        if (nt < 0) continue;
        text[nt] = 0;
        int exp[256];
        int ne = read_ids(tab2 + 1, exp, 256);
        int prompt[8192];
        int np = tok_encode(g.tok, (const char *)text, prompt, 8192);
        int *gen = (int *)malloc((size_t)(ne > 0 ? ne : 1) * sizeof(int));
        unsigned long long rng = 0;
        int ng = gpt2_generate(&g, prompt, np, ne, 0.0f, 0, 1.0f, &rng, gen);
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
    /* 4. perplexity fixture */
    snprintf(path, sizeof(path), "%s/gpt2_ppl.txt", dir);
    f = fopen(path, "r");
    if (f && fgets(line, sizeof(line), f)) {
        char *tab1 = strchr(line, '\t');
        char *tab2 = tab1 ? strchr(tab1 + 1, '\t') : NULL;
        if (tab1 && tab2) {
            *tab1 = 0;
            *tab2 = 0;
            unsigned char text[65536];
            int nt = unhex2((unsigned char *)line, strlen(line), text, sizeof(text) - 1);
            double want_nll = atof(tab1 + 1), want_ppl = atof(tab2 + 1);
            if (nt >= 0) {
                text[nt] = 0;
                int ids[8192];
                int n = tok_encode(g.tok, (const char *)text, ids, 8192);
                double nll = gpt2_nll(&g, ids, n);
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
    printf("GPT2-VERIFY %s (%d checked)\n", fails ? "FAIL" : "PASS", checked);
    gpt2_close(&g);
    return fails != 0;
}
