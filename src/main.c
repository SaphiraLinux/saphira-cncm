/* main.c — cncm CLI: infer (parity dumps), generate, eval-sudoku. */
#include "ncm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *a) {
    fprintf(stderr,
        "usage:\n"
        "  %s infer   --conf M.conf --weights M.ncm --asset-dir D --ids 1,2,3..\n"
        "  %s gen     --conf M.conf --weights M.ncm --asset-dir D [--prefix IDS|TEXT]\n"
        "             [--temp f] [--top-k n] [--top-p f] [--cfg f] [--freq f] [--mu f]\n"
        "             [--adj a,w,d] [--seed n] [--n N] [--l0 unigram|FILE]\n"
        "  %s eval-sudoku --conf M.conf --weights M.ncm --asset-dir D\n"
        "             --inputs inputs.txt --refs refs.txt [--n N]\n"
        "  %s gpt2-gen --conf M.conf --weights M.ncm --asset-dir D\n"
        "             (--prompt TEXT | --ids CSV) [--n N] [--temp f]\n"
        "             [--top-k n] [--top-p f] [--seed n]\n"
        "  %s gpt2-ppl --conf M.conf --weights M.ncm --asset-dir D\n"
        "             --texts FILE (one document per line)\n"
        "  %s gpt2-verify --conf M.conf --weights M.ncm --asset-dir D\n"
        "             --fixtures DIR (gpt2_tok/gen/ppl.txt + gpt2_logits_*.f32)\n"
        "  %s qwen-gen --conf M.conf --weights M.ncm --asset-dir D\n"
        "             (--prompt TEXT | --ids CSV) [--n N] [--temp f]\n"
        "             [--top-k n] [--top-p f] [--seed n]\n"
        "  %s qwen-ppl --conf M.conf --weights M.ncm --asset-dir D\n"
        "             --texts FILE (one document per line)\n"
        "  %s qwen-verify --conf M.conf --weights M.ncm --asset-dir D\n"
        "             --fixtures DIR (qwen_tok/gen/ppl.txt + qwen_logits_*.f32)\n", a, a, a, a, a, a, a, a, a);
    exit(2);
}

static const char *opt(int *argc, char ***argv, const char *flag) {
    for (int i = 1; i < *argc - 1; i++)
        if (strcmp((*argv)[i], flag) == 0) return (*argv)[i + 1];
    return NULL;
}

typedef struct {
    NCMConfig cfg;
    WeightStore ws;
    VQVAEState *vqvae;
    NCPState *ncp;
    TokState *tok;
} App;

static void app_load(App *a, const char *conf, const char *weights, const char *adir) {
    if (config_load(&a->cfg, conf)) exit(1);
    if (weights_load(&a->ws, weights)) exit(1);
    arena_init(1ULL << 30); /* 1 GiB scratch */
    a->vqvae = vqvae_load(&a->cfg, &a->ws);
    a->ncp = ncp_load(&a->cfg, &a->ws, a->vqvae);
    a->tok = tok_load(&a->cfg, adir);
    fprintf(stderr, "[cncm] task=%s levels=", a->cfg.task);
    for (int i = 0; i < a->cfg.n_qlevels; i++) fprintf(stderr, "%s%d", i ? "," : "", a->cfg.levels[i]);
    fprintf(stderr, "\n");
}

/* read entire file into malloc'd buffer */
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc(sz + 1);
    if (fread(b, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read err\n"); exit(1); }
    b[sz] = 0;
    fclose(f);
    if (n) *n = sz;
    return b;
}

static int parse_ids(const char *s, int *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p && n < cap) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p || *p == '\n') break;
        out[n++] = atoi(p);
        while (*p && *p != ',' && *p != '\n') p++;
    }
    return n;
}

/* checksum helper for parity dumps */
static void dump_sum(const char *name, const float *d, size_t n) {
    /* stage to host */
    float *h = (float *)malloc(n * sizeof(float));
    mem_d2h(h, d, n * sizeof(float));
    double s = 0, m = 0;
    for (size_t i = 0; i < n; i++) { s += h[i]; if (fabs(h[i]) > m) m = fabs(h[i]); }
    printf("%s sum=%.6f maxabs=%.6f first=%.6f %.6f %.6f %.6f\n",
           name, s, m, h[0], n > 1 ? h[1] : 0, n > 2 ? h[2] : 0, n > 3 ? h[3] : 0);
    free(h);
}

static int cmd_gen(App *a, int argc, char **argv);
static int cmd_eval_sudoku(App *a, int argc, char **argv);
static int cmd_infer(App *a, const char *ids_csv) {
    int P = a->cfg.prefix_len;
    int T = a->cfg.vq_context_length;
    int B = 1;
    /* ids csv holds prefix (P) followed by chunk (T) */
    int *ids = (int *)malloc((size_t)(P + T) * sizeof(int));
    int n = parse_ids(ids_csv, ids, P + T);
    if (n == 0) { fprintf(stderr, "no ids\n"); return 1; }
    int *inp = (int *)calloc(B * T, sizeof(int));
    /* chunk = ids[P:P+T]; fall back to ids[0:T] when only P ids given */
    for (int i = 0; i < T; i++)
        inp[i] = (P + i < n) ? ids[P + i] : (i < n ? ids[i] : 0);

    float *z = arena_alloc((size_t)B * T * a->cfg.vq_embed_dim);
    vqvae_encode_z(a->vqvae, inp, z, B);
    dump_sum("encode_z", z, (size_t)B * T * a->cfg.vq_embed_dim);

    VQState *vq = vqvae_vq(a->vqvae);
    int total = vq_total_codes(vq);
    int *idx = (int *)malloc(total * sizeof(int));
    vq_z_to_idx(vq, z, B, idx);
    printf("idx:");
    for (int i = 0; i < total; i++) printf(" %d", idx[i]);
    printf("\n");

    /* decode final level and argmax */
    float **zhats = (float **)malloc(a->cfg.n_qlevels * sizeof(float *));
    for (int i = 0; i < a->cfg.n_qlevels; i++)
        zhats[i] = arena_alloc((size_t)B * a->cfg.levels[a->cfg.n_qlevels - 1] * a->cfg.vq_embed_dim);
    vq_idx_to_zhats(vq, idx, B, zhats);
    for (int i = 0; i < a->cfg.n_qlevels; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "zhat_lvl%d", i);
        dump_sum(nm, zhats[i], (size_t)B * a->cfg.levels[a->cfg.n_qlevels - 1] * a->cfg.vq_embed_dim);
    }
    int L = a->cfg.levels[a->cfg.n_qlevels - 1];
    float *logits = arena_alloc((size_t)B * L * a->cfg.vq_in_vocab);
    vqvae_decode(a->vqvae, zhats[a->cfg.n_qlevels - 1], logits, B);
    dump_sum("decode_logits", logits, (size_t)B * L * a->cfg.vq_in_vocab);

    /* NCP forward over full input: x = [z_prefix(B,P) ; ncm_input(B,T2)] */
    int T2 = total - (a->cfg.query_level0 ? 0 : a->cfg.levels[0]);
    int S = P + T2;
    /* z_prefix = encoder.wte(prefix) — raw path, exactly like generate.
     * (All released checkpoints use raw_token_prefix.) */
    if (!a->cfg.raw_token_prefix) { fprintf(stderr, "non-raw prefix not wired\n"); return 1; }
    float *zpre = arena_alloc((size_t)B * P * a->cfg.vq_embed_dim);
    {
        int *pp = (int *)malloc((size_t)B * P * sizeof(int));
        for (int b = 0; b < B; b++) for (int t = 0; t < P; t++)
            pp[b * P + t] = (t < n) ? ids[t] : 0;
        int *bp = (int *)arena_alloc((size_t)B * P);
        mem_h2d(bp, pp, (size_t)B * P * sizeof(int));
        free(pp);
        Tensor *te = weights_get(&a->ws, "tok.encoder.wte.weight");
        embedding_fwd(te->data, bp, zpre, (int64_t)B * P, a->cfg.vq_embed_dim);
    }
    float *xt = arena_alloc((size_t)B * T2 * a->cfg.vq_embed_dim);
    vq_idx_to_ncm_input(vq, idx, B, xt);
    /* query_level0: prepend learned l0_queries (training input convention) */
    int L0 = a->cfg.query_level0 ? a->cfg.levels[0] : 0;
    S = P + L0 + T2;
    float *l0q = NULL;
    if (L0) {
        Tensor *tq = weights_get(&a->ws, "gen.l0_queries");
        if (!tq) { fprintf(stderr, "missing gen.l0_queries\n"); return 1; }
        l0q = arena_alloc((size_t)B * L0 * a->cfg.vq_embed_dim);
        float *h = (float *)malloc((size_t)L0 * a->cfg.vq_embed_dim * sizeof(float));
        mem_d2h(h, tq->data, (size_t)L0 * a->cfg.vq_embed_dim * sizeof(float));
        float *hb = (float *)malloc((size_t)B * L0 * a->cfg.vq_embed_dim * sizeof(float));
        for (int b = 0; b < B; b++) memcpy(hb + (size_t)b * L0 * a->cfg.vq_embed_dim, h,
               (size_t)L0 * a->cfg.vq_embed_dim * sizeof(float));
        free(h);
        mem_h2d(l0q, hb, (size_t)B * L0 * a->cfg.vq_embed_dim * sizeof(float));
        free(hb);
    }
    float *xfull = arena_alloc((size_t)B * S * a->cfg.vq_embed_dim);
    {
        float *a1 = (float *)malloc((size_t)B * P * a->cfg.vq_embed_dim * sizeof(float));
        float *a2 = (float *)malloc((size_t)B * T2 * a->cfg.vq_embed_dim * sizeof(float));
        mem_d2h(a1, zpre, (size_t)B * P * a->cfg.vq_embed_dim * sizeof(float));
        mem_d2h(a2, xt, (size_t)B * T2 * a->cfg.vq_embed_dim * sizeof(float));
        float *af = (float *)malloc((size_t)B * S * a->cfg.vq_embed_dim * sizeof(float));
        float *a0 = NULL;
        if (L0) { a0 = (float *)malloc((size_t)B * L0 * a->cfg.vq_embed_dim * sizeof(float)); mem_d2h(a0, l0q, (size_t)B * L0 * a->cfg.vq_embed_dim * sizeof(float)); }
        for (int b = 0; b < B; b++) {
            memcpy(af + (size_t)b * S * a->cfg.vq_embed_dim, a1 + (size_t)b * P * a->cfg.vq_embed_dim,
                   (size_t)P * a->cfg.vq_embed_dim * sizeof(float));
            if (L0) memcpy(af + ((size_t)b * S + P) * a->cfg.vq_embed_dim, a0 + (size_t)b * L0 * a->cfg.vq_embed_dim,
                   (size_t)L0 * a->cfg.vq_embed_dim * sizeof(float));
            memcpy(af + ((size_t)b * S + P + L0) * a->cfg.vq_embed_dim, a2 + (size_t)b * T2 * a->cfg.vq_embed_dim,
                   (size_t)T2 * a->cfg.vq_embed_dim * sizeof(float));
        }
        free(a0);
        free(a1); free(a2);
        mem_h2d(xfull, af, (size_t)B * S * a->cfg.vq_embed_dim * sizeof(float));
        free(af);
    }
    float *nlog = arena_alloc((size_t)B * T2 * 8192);
    int *pids = (int *)malloc(B * P * sizeof(int));
    for (int i = 0; i < B * P; i++) pids[i] = (i < n) ? ids[i] : 0;
    ncp_forward(a->ncp, xfull, pids, nlog, B);
    dump_sum("ncp_logits", nlog, (size_t)B * T2 * 8192);
    free(pids); free(idx); free(zhats); free(inp); free(ids);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);
    const char *conf = opt(&argc, &argv, "--conf");
    const char *weights = opt(&argc, &argv, "--weights");
    const char *adir = opt(&argc, &argv, "--asset-dir");
    if (!conf || !weights) usage(argv[0]);
    if (strcmp(argv[1], "train") != 0 && strcmp(argv[1], "gradcheck") != 0 && strcmp(argv[1], "train-vq") != 0 && !adir) usage(argv[0]);

    App a;
    memset(&a, 0, sizeof(a));
    if (strcmp(argv[1], "infer") == 0) {
        const char *ids = opt(&argc, &argv, "--ids");
        if (!ids) usage(argv[0]);
        app_load(&a, conf, weights, adir);
        return cmd_infer(&a, ids);
    }
    if (strcmp(argv[1], "gen") == 0) {
        app_load(&a, conf, weights, adir);
        return cmd_gen(&a, argc, argv);
    }
    if (strcmp(argv[1], "eval-sudoku") == 0) {
        app_load(&a, conf, weights, adir);
        return cmd_eval_sudoku(&a, argc, argv);
    }
    if (strcmp(argv[1], "gpt2-gen") == 0) {
        const char *prompt = opt(&argc, &argv, "--prompt");
        const char *ids = opt(&argc, &argv, "--ids");
        const char *nn = opt(&argc, &argv, "--n");
        const char *tp = opt(&argc, &argv, "--temp");
        const char *tk = opt(&argc, &argv, "--top-k");
        const char *tpp = opt(&argc, &argv, "--top-p");
        const char *ss = opt(&argc, &argv, "--seed");
        if (!conf || !weights) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return gpt2_cmd_gen(&cfg, weights, adir, prompt, ids, nn ? atoi(nn) : 20,
                            tp ? (float)atof(tp) : 0.0f, tk ? atoi(tk) : 0,
                            tpp ? (float)atof(tpp) : 1.0f,
                            ss ? strtoull(ss, NULL, 10) : 0ULL);
    }
    if (strcmp(argv[1], "gpt2-ppl") == 0) {
        const char *texts = opt(&argc, &argv, "--texts");
        if (!conf || !weights || !texts) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return gpt2_cmd_ppl(&cfg, weights, adir, texts);
    }
    if (strcmp(argv[1], "gpt2-verify") == 0) {
        const char *fx = opt(&argc, &argv, "--fixtures");
        if (!conf || !weights || !fx) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return gpt2_cmd_verify(&cfg, weights, adir, fx);
    }
    if (strcmp(argv[1], "qwen-gen") == 0) {
        const char *prompt = opt(&argc, &argv, "--prompt");
        const char *ids = opt(&argc, &argv, "--ids");
        const char *nn = opt(&argc, &argv, "--n");
        const char *tp = opt(&argc, &argv, "--temp");
        const char *tk = opt(&argc, &argv, "--top-k");
        const char *tpp = opt(&argc, &argv, "--top-p");
        const char *ss = opt(&argc, &argv, "--seed");
        if (!conf || !weights) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return qwen_cmd_gen(&cfg, weights, adir, prompt, ids, nn ? atoi(nn) : 20,
                            tp ? (float)atof(tp) : 0.0f, tk ? atoi(tk) : 0,
                            tpp ? (float)atof(tpp) : 1.0f,
                            ss ? strtoull(ss, NULL, 10) : 0ULL);
    }
    if (strcmp(argv[1], "qwen-ppl") == 0) {
        const char *texts = opt(&argc, &argv, "--texts");
        if (!conf || !weights || !texts) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return qwen_cmd_ppl(&cfg, weights, adir, texts);
    }
    if (strcmp(argv[1], "qwen-verify") == 0) {
        const char *fx = opt(&argc, &argv, "--fixtures");
        if (!conf || !weights || !fx) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        return qwen_cmd_verify(&cfg, weights, adir, fx);
    }
    if (strcmp(argv[1], "gradcheck") == 0) {
        const char *data = opt(&argc, &argv, "--data");
        const char *nn = opt(&argc, &argv, "--n");
        if (!data || !conf || !weights) usage(argv[0]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        extern int gradcheck_run_cfg(const NCMConfig*, const char*, const char*, int);
        return gradcheck_run_cfg(&cfg, weights, data, nn ? atoi(nn) : 6);
    }
    if (strcmp(argv[1], "train-vq") == 0) {
        const char *data = opt(&argc, &argv, "--data");
        const char *steps = opt(&argc, &argv, "--steps");
        if (!data || !conf || !weights) { fprintf(stderr, "train-vq needs --data <ids.nctrain> [--steps N]\n"); return 2; }
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        extern int train_vq_run(const NCMConfig *c, const char *weights, const char *data, int steps);
        return train_vq_run(&cfg, weights, data, steps ? atoi(steps) : 200);
    }
    if (strcmp(argv[1], "train") == 0) {
        const char *data = opt(&argc, &argv, "--data");
        const char *steps = opt(&argc, &argv, "--steps");
        const char *lr = opt(&argc, &argv, "--lr");
        if (!data) { fprintf(stderr, "train needs --data <file.nctrain> [--steps N] [--lr f]\n"); return 2; }
        if (!conf || !weights) usage(argv[1]);
        NCMConfig cfg;
        if (config_load(&cfg, conf)) return 1;
        (void)adir;
        return train_run(&cfg, weights, data, steps ? atoi(steps) : 200, lr ? atof(lr) : 0.0);
    }
    fprintf(stderr, "unknown subcommand %s\n", argv[1]);
    return 2;
}

/* ---------------- generate + eval-sudoku ---------------- */

typedef struct {
    float temp; int top_k; float top_p, cfg;
    float freq, mu, adj_a; int adj_w; float adj_d;
    unsigned long long seed;
} GenOpts;

static void genopts_defaults(GenOpts *g, NCMConfig *c) {
    g->temp = c->gen_temperature; g->top_k = c->gen_top_k; g->top_p = c->gen_top_p;
    g->cfg = c->gen_cfg_scale; g->freq = c->gen_freq; g->mu = c->gen_mu;
    g->adj_a = c->gen_adj_a; g->adj_w = c->gen_adj_w; g->adj_d = c->gen_adj_d;
    g->seed = 0;
}

/* load inference float vector as doubles (or NULL if absent) */
static double *load_allow(WeightStore *ws, const char *name, int *n_out) {
    Tensor *t = weights_get(ws, name);
    if (!t) { *n_out = 0; return NULL; }
    float *h = (float *)malloc(t->n * sizeof(float));
    mem_d2h(h, t->data, t->n * sizeof(float));
    double *d = (double *)malloc(t->n * sizeof(double));
    for (long long i = 0; i < t->n; i++) d[i] = h[i];
    free(h);
    *n_out = (int)t->n;
    return d;
}

/* sample l0 [B][l0] from unigram (host) */
static void sample_l0(const double *p, int V, int B, int l0len,
                      unsigned long long *rng, int *out) {
    double *cum = (double *)malloc(V * sizeof(double));
    double s = 0;
    for (int i = 0; i < V; i++) { s += p[i]; cum[i] = s; }
    for (int b = 0; b < B; b++) for (int t = 0; t < l0len; t++) {
        double u = ((rng_next(rng) >> 11) * (1.0 / 9007199254740992.0)) * s;
        int lo = 0, hi = V - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (cum[mid] < u) lo = mid + 1; else hi = mid; }
        out[b * l0len + t] = lo;
    }
    free(cum);
}

/* decode one board's codes -> text. codes:[total] level-major. */
static char *decode_board(App *a, const int *codes) {
    arena_reset(); /* per-board scratch */
    VQState *vq = vqvae_vq(a->vqvae);
    int L = a->cfg.levels[a->cfg.n_qlevels - 1];
    int Vp = a->cfg.vq_in_vocab;
    float **zhats = (float **)malloc(a->cfg.n_qlevels * sizeof(float *));
    for (int i = 0; i < a->cfg.n_qlevels; i++)
        zhats[i] = arena_alloc((size_t)L * a->cfg.vq_embed_dim);
    vq_idx_to_zhats(vq, codes, 1, zhats);
    float *logits = arena_alloc((size_t)L * Vp);
    vqvae_decode(a->vqvae, zhats[a->cfg.n_qlevels - 1], logits, 1);
    float *hl = (float *)malloc((size_t)L * Vp * sizeof(float));
    mem_d2h(hl, logits, (size_t)L * Vp * sizeof(float));
    int *ids = (int *)malloc(L * sizeof(int));
    for (int t = 0; t < L; t++) ids[t] = argmax_row(hl + (size_t)t * Vp, Vp);
    free(hl);
    char *txt = (char *)malloc(8192);
    tok_decode(a->tok, ids, L, txt, 8192, 0);
    free(ids); free(zhats);
    return txt;
}

/* decode a whole batch at once. codes:[B][total] level-major -> txts[b] (malloc'd). */
static void decode_boards(App *a, const int *codes, int B, char **txts) {
    arena_reset();
    VQState *vq = vqvae_vq(a->vqvae);
    int L = a->cfg.levels[a->cfg.n_qlevels - 1];
    int Vp = a->cfg.vq_in_vocab;
    float **zhats = (float **)malloc(a->cfg.n_qlevels * sizeof(float *));
    for (int i = 0; i < a->cfg.n_qlevels; i++)
        zhats[i] = arena_alloc((size_t)B * L * a->cfg.vq_embed_dim);
    vq_idx_to_zhats(vq, codes, B, zhats);
    float *logits = arena_alloc((size_t)B * L * Vp);
    vqvae_decode(a->vqvae, zhats[a->cfg.n_qlevels - 1], logits, B);
    float *hl = (float *)malloc((size_t)B * L * Vp * sizeof(float));
    mem_d2h(hl, logits, (size_t)B * L * Vp * sizeof(float));
    int *ids = (int *)malloc((size_t)B * L * sizeof(int));
    for (int b = 0; b < B; b++) for (int t = 0; t < L; t++)
        ids[b * L + t] = argmax_row(hl + ((size_t)b * L + t) * Vp, Vp);
    free(hl);
    for (int b = 0; b < B; b++) {
        txts[b] = (char *)malloc(8192);
        tok_decode(a->tok, ids + (size_t)b * L, L, txts[b], 8192, 0);
    }
    free(ids); free(zhats);
}

static int cmd_gen(App *a, int argc, char **argv) {
    const char *prefix_s = opt(&argc, &argv, "--prefix");
    const char *l0_s = opt(&argc, &argv, "--l0");
    GenOpts g;
    genopts_defaults(&g, &a->cfg);
    const char *s;
    if ((s = opt(&argc, &argv, "--temp"))) g.temp = (float)atof(s);
    if ((s = opt(&argc, &argv, "--top-k"))) g.top_k = atoi(s);
    if ((s = opt(&argc, &argv, "--top-p"))) g.top_p = (float)atof(s);
    if ((s = opt(&argc, &argv, "--cfg"))) g.cfg = (float)atof(s);
    if ((s = opt(&argc, &argv, "--freq"))) g.freq = (float)atof(s);
    if ((s = opt(&argc, &argv, "--mu"))) g.mu = (float)atof(s);
    if ((s = opt(&argc, &argv, "--seed"))) g.seed = strtoull(s, NULL, 10);
    int B = 1;
    if ((s = opt(&argc, &argv, "--n"))) B = atoi(s);
    if (!prefix_s) { fprintf(stderr, "gen needs --prefix\n"); return 2; }
    int P = a->cfg.prefix_len;
    int *prefix = (int *)malloc((size_t)B * P * sizeof(int));
    /* prefix may be ids csv or raw text */
    int np = parse_ids(prefix_s, prefix, P);
    if (np == 0) {
        int nt = tok_encode(a->tok, prefix_s, prefix, P);
        for (int i = nt; i < P; i++) prefix[i] = a->cfg.pad_token_id;
        np = P;
    } else {
        for (int i = np; i < P; i++) prefix[i] = prefix[0];
        for (int b = 1; b < B; b++) memcpy(prefix + b * P, prefix, P * sizeof(int));
    }
    int l0len = a->cfg.levels[0];
    int *l0 = (int *)malloc((size_t)B * l0len * sizeof(int));
    unsigned long long rng = g.seed ? g.seed : 0x9E3779B97F4A7C15ULL;
    int have_l0 = 1;
    if (l0_s && strcmp(l0_s, "unigram") != 0) {
        int nl = parse_ids(l0_s, l0, B * l0len);
        (void)nl;
    } else if (!l0_s && !a->cfg.query_level0 && !a->cfg.predict_level0) {
        fprintf(stderr, "model needs --l0 (no query/predict path)\n");
        return 1;
    } else if (l0_s) {
        int nv = 0;
        double *lp = load_allow(&a->ws, "inference.level0_unigram", &nv);
        if (!lp) { fprintf(stderr, "no level0 unigram saved; omit --l0 to use model queries\n"); return 1; }
        sample_l0(lp, nv, B, l0len, &rng, l0);
        free(lp);
    } else {
        have_l0 = 0; /* model generates l0 itself */
    }
    int na = 0;
    double *allow = load_allow(&a->ws, "inference.token_unigram", &na);
    int *codes = ncp_generate(a->ncp, have_l0 ? l0 : NULL, prefix, B, g.temp, g.top_k, g.top_p,
                              g.cfg, g.freq, g.mu, g.adj_a, g.adj_w, g.adj_d,
                              allow, g.seed);
    free(allow);
    int total = 0;
    for (int i = 0; i < a->cfg.n_qlevels; i++) total += a->cfg.levels[i];
    for (int b = 0; b < B; b++) {
        char *txt = decode_board(a, codes + (size_t)b * total);
        printf("%s\n", txt);
        free(txt);
    }
    free(codes); free(l0); free(prefix);
    return 0;
}

static void sudoku_cells(const char *s, char *out) {
    while (*s) { if (*s != '|') *out++ = *s; s++; }
    *out = 0;
}

static int sudoku_valid(const char *cells) {
    if (strlen(cells) != 81) return 0;
    for (int i = 0; i < 81; i++) if (cells[i] < '1' || cells[i] > '9') return 0;
    for (int r = 0; r < 9; r++) {
        int seen[10] = {0};
        for (int c = 0; c < 9; c++) { int d = cells[r * 9 + c] - '0'; if (seen[d]) return 0; else seen[d] = 1; }
    }
    for (int c = 0; c < 9; c++) {
        int seen[10] = {0};
        for (int r = 0; r < 9; r++) { int d = cells[r * 9 + c] - '0'; if (seen[d]) return 0; else seen[d] = 1; }
    }
    for (int br = 0; br < 3; br++) for (int bc = 0; bc < 3; bc++) {
        int seen[10] = {0};
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            int d = cells[(br * 3 + r) * 9 + bc * 3 + c] - '0';
            if (seen[d]) return 0;
            seen[d] = 1;
        }
    }
    return 1;
}

static int cmd_eval_sudoku(App *a, int argc, char **argv) {
    const char *inp = opt(&argc, &argv, "--inputs");
    const char *refs = opt(&argc, &argv, "--refs");
    const char *sn = opt(&argc, &argv, "--n");
    const char *ss = opt(&argc, &argv, "--seed");
    if (!inp || !refs) { fprintf(stderr, "need --inputs and --refs\n"); return 2; }
    int nmax = sn ? atoi(sn) : 2000;
    unsigned long long seed = ss ? strtoull(ss, NULL, 10) : 0;
    size_t ilen, rlen;
    char *ibuf = slurp(inp, &ilen);
    char *rbuf = slurp(refs, &rlen);
    /* split lines */
    int n = 0;
    for (size_t i = 0; i < ilen; i++) if (ibuf[i] == '\n') n++;
    if (n > nmax) n = nmax;
    char **ilines = (char **)malloc(n * sizeof(char *));
    char **rlines = (char **)malloc(n * sizeof(char *));
    ilines[0] = ibuf; rlines[0] = rbuf;
    int ni = 1, nr = 1;
    for (size_t i = 0; i < ilen && ni < n; i++) if (ibuf[i] == '\n') { ibuf[i] = 0; if (ni < n) ilines[ni++] = ibuf + i + 1; }
    for (size_t i = 0; i < rlen && nr < n; i++) if (rbuf[i] == '\n') { rbuf[i] = 0; if (nr < n) rlines[nr++] = rbuf + i + 1; }

    int P = a->cfg.prefix_len;
    int l0len = a->cfg.levels[0];
    int nl0 = 0;
    double *l0p = load_allow(&a->ws, "inference.level0_unigram", &nl0);
    int na = 0;
    double *allow = load_allow(&a->ws, "inference.token_unigram", &na);
    unsigned long long rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int B = 32, exact = 0, validn = 0;
    double cell = 0;
    int *prefix = (int *)malloc((size_t)B * P * sizeof(int));
    int *l0 = (int *)malloc((size_t)B * l0len * sizeof(int));
    for (int s = 0; s < n; s += B) {
        int bb = n - s < B ? n - s : B;
        for (int b = 0; b < bb; b++) {
            int got = parse_ids(ilines[s + b], prefix + b * P, P);
            for (int i = got; i < P; i++) prefix[b * P + i] = 0;
        }
        if (l0p) sample_l0(l0p, nl0, bb, l0len, &rng, l0);
        int *codes = ncp_generate(a->ncp, l0p ? l0 : NULL, prefix, bb, 0.0, 0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 0, 0.0, allow, seed + s);
        char **hyps = (char **)malloc(bb * sizeof(char *));
        decode_boards(a, codes, bb, hyps);
        for (int b = 0; b < bb; b++) {
            char *hyp = hyps[b];
            int *rid = (int *)malloc(1024 * sizeof(int));
            int nrid = parse_ids(rlines[s + b], rid, 1024);
            char *ref = (char *)malloc(8192);
            tok_decode(a->tok, rid, nrid, ref, 8192, 0);
            free(rid);
            if (strcmp(hyp, ref) == 0) exact++;
            char hc[128], rc[128];
            sudoku_cells(hyp, hc);
            sudoku_cells(ref, rc);
            int same = 0;
            for (int i = 0; i < 81 && hc[i] && rc[i]; i++) if (hc[i] == rc[i]) same++;
            cell += same / 81.0;
            if (sudoku_valid(hc)) validn++;
            if (s == 0 && b < 2) printf("eg hyp: %s\nref: %s\n", hyp, ref);
            free(hyp); free(ref);
        }
        free(hyps);
        free(codes);
        if ((s / B) % 10 == 0) { printf("  %d/%d\n", s + bb, n); fflush(stdout); }
    }
    printf("solved %d/%d (%.2f%%) cell %.4f valid %d/%d (%.2f%%)\n",
           exact, n, 100.0 * exact / n, cell / n, validn, n, 100.0 * validn / n);
    free(prefix); free(l0); free(ilines); free(rlines); free(ibuf); free(rbuf);
    free(l0p); free(allow);
    return 0;
}
