/* tensor.c — arena-free tensors, .ncm loader, .conf parser, backend glue. */
#include "ncm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NCM_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { fprintf(stderr, "cuda: %s:%d %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#endif

int64_t tensor_numel(int ndim, const int64_t *shape) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

Tensor *tensor_alloc(int ndim, const int64_t *shape) {
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    if (!t) { fprintf(stderr, "oom tensor\n"); exit(1); }
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->n = tensor_numel(ndim, shape);
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMalloc((void **)&t->data, t->n * sizeof(float)));
#else
    t->data = (float *)malloc(t->n * sizeof(float));
    if (!t->data) { fprintf(stderr, "oom tensor data\n"); exit(1); }
#endif
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
#ifdef NCM_CUDA
    cudaFree(t->data);
#else
    free(t->data);
#endif
    free(t);
}

void tensor_to_host(const Tensor *t, float *out) {
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMemcpy(out, t->data, t->n * sizeof(float),
                          cudaMemcpyDeviceToHost));
#else
    memcpy(out, t->data, t->n * sizeof(float));
#endif
}

void tensor_to_backend(Tensor *t, const float *in) {
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMemcpy(t->data, in, t->n * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    memcpy(t->data, in, t->n * sizeof(float));
#endif
}

/* ---------------- .ncm loader ---------------- */

static int read_u32(FILE *f, uint32_t *out) {
    return fread(out, 4, 1, f) == 1 ? 0 : -1;
}

int weights_load(WeightStore *ws, const char *path) {
    /* NCK1 training checkpoints load anywhere NCM1 weights load
     * (weights section only; optimizer/run state ignored). */
    if (ckpt_probe(path) == 1) {
        char err[256] = {0};
        if (ckpt_load_weights_only(path, ws, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            return -1;
        }
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NCM1", 4) != 0) {
        fprintf(stderr, "%s: bad magic\n", path); fclose(f); return -1;
    }
    uint32_t count = 0;
    if (read_u32(f, &count)) { fclose(f); return -1; }
    ws->items = (NamedTensor *)calloc(count, sizeof(NamedTensor));
    ws->count = 0;
    for (uint32_t i = 0; i < count; i++) {
        unsigned int nl = 0;
        uint32_t nd = 0;
        unsigned char nlb = 0;
        if (fread(&nlb, 1, 1, f) != 1) goto fail;
        nl = nlb;
        if (nl >= sizeof(ws->items[i].name)) goto fail;
        if (fread(ws->items[i].name, 1, nl, f) != nl) goto fail;
        ws->items[i].name[nl] = 0;
        unsigned char ndim = 0;
        if (fread(&ndim, 1, 1, f) != 1) goto fail;
        int64_t shape[NCM_MAX_DIMS] = {0};
        for (int d = 0; d < ndim; d++) {
            if (read_u32(f, &nd)) goto fail;
            shape[d] = nd;
        }
        uint64_t nbytes = 0;
        if (fread(&nbytes, 8, 1, f) != 1) goto fail;
        Tensor *t = tensor_alloc(ndim, shape);
        if ((uint64_t)t->n * 4 != nbytes) {
            fprintf(stderr, "%s: size mismatch\n", ws->items[i].name);
            tensor_free(t); goto fail;
        }
#ifdef NCM_CUDA
        {
            float *tmp = (float *)malloc(nbytes);
            if (fread(tmp, 1, nbytes, f) != nbytes) { free(tmp); tensor_free(t); goto fail; }
            tensor_to_backend(t, tmp);
            free(tmp);
        }
#else
        if (fread(t->data, 1, nbytes, f) != nbytes) { tensor_free(t); goto fail; }
#endif
        ws->items[i].t = t;
        ws->count++;
    }
    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

void weights_free(WeightStore *ws) {    for (int i = 0; i < ws->count; i++) tensor_free(ws->items[i].t);
    free(ws->items);
    ws->items = NULL; ws->count = 0;
}

Tensor *weights_get(WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++)
        if (strcmp(ws->items[i].name, name) == 0) return ws->items[i].t;
    return NULL;
}

/* ---------------- .conf parser ---------------- */

static void set_str(char *dst, size_t cap, const char *v) {
    strncpy(dst, v, cap - 1); dst[cap - 1] = 0;
}

static int parse_int(const char *v) {
    while (*v == ' ') v++;
    if (*v == 'T' || *v == 't') return 1;   /* True */
    if (*v == 'F' || *v == 'f') return 0;   /* False */
    return atoi(v);
}

static void parse_int_list(const char *v, int *out, int cap, int *n) {
    int k = 0;
    const char *p = v;
    while (*p && k < cap) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        out[k++] = atoi(p);
        while (*p && *p != ',') p++;
    }
    *n = k;
}

int config_load(NCMConfig *c, const char *path) {
    memset(c, 0, sizeof(*c));
    /* defaults matching the python `getattr(..., default)` fallbacks */
    c->bias = 1; c->rope_base = 10000; c->rope_theta = 10000.0; c->n_kv_heads = -1;
    c->prefix_pad_token_id = -1;
    c->eos_token_id = -1; c->pad_token_id = -1;
    c->attn_pattern_block_diag = 1;
    c->gen_top_k = 1; c->gen_top_p = 1.0f;
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        char *k = line, *v = eq + 1;
        v[strcspn(v, "\r\n")] = 0;
#define KEY(s) (strcmp(k, s) == 0)
#define INT(field) else if KEY("model." #field) { c->field = parse_int(v); }
#define FLT(field) else if KEY("model." #field) { c->field = (float)atof(v); }
        if KEY("task") { set_str(c->task, sizeof(c->task), v); }
        INT(n_layer) INT(n_head) INT(n_embd) INT(context_length) INT(prefix_len)
        INT(raw_token_prefix) INT(contextual_prefix) INT(prefix_pad_token_id)
        INT(predict_level0) INT(query_level0) INT(l0_autoregressive)
        INT(use_positional_encoding) INT(use_level_encoding)
        INT(use_rope) INT(rope_base) INT(use_qk_norm) INT(use_flex_attention)
        else if KEY("model.rope_theta") c->rope_theta = atof(v);
        INT(n_kv_heads)
        FLT(dropout)
        else if KEY("model.bias") c->bias = atoi(v);
        INT(ctx_enabled) INT(ctx_n_layer) INT(ctx_vocab)
        INT(puc_enabled) INT(puc_kernel)
        INT(shared_output_head) INT(shared_head_per_level_bias) INT(untied_head_bias)
        INT(tie_heads_to_codebook) INT(tie_heads_learn_scale)
        INT(tie_heads_trainable) INT(tie_heads_bias)
        INT(use_swiglu) INT(use_geglu) INT(use_relu2)
        INT(parallel_block) INT(normformer)
        INT(use_layerscale) INT(use_rezero)
        FLT(layerscale_init)
        else if KEY("model.attn_pattern") c->attn_pattern_block_diag = (strcmp(v, "block_diagonal") == 0);
        else if KEY("model.vocab_size") { int n = 0; parse_int_list(v, c->vocab_size, NCM_MAX_VOCAB_LIST, &n); c->n_levels_out = n; }
        else if KEY("vq.embed_dim") c->vq_embed_dim = atoi(v);
        else if KEY("vq.in_vocab") c->vq_in_vocab = atoi(v);
        else if KEY("vq.context_length") c->vq_context_length = atoi(v);
        else if KEY("vq.n_layers") c->vq_n_layers = atoi(v);
        else if KEY("vq.compression") c->vq_compression = atoi(v);
        else if KEY("vq.levels") parse_int_list(v, c->levels, NCM_MAX_LEVELS, &c->n_qlevels);
        else if KEY("vq.vocab") { int n = 0, tmp[NCM_MAX_LEVELS]; parse_int_list(v, tmp, NCM_MAX_LEVELS, &n); for (int i = 0; i < n; i++) c->vq_vocab[i] = tmp[i]; }
        else if KEY("vq.qr_enabled") c->quant_resi_enabled = atoi(v);
        else if KEY("vq.qr_ratio") c->quant_resi_ratio = (float)atof(v);
        else if KEY("vq.qr_share") c->quant_resi_share_mode = atoi(v);
        else if KEY("vq.l1_sampling") c->learned_l1_sampling = atoi(v);
        else if KEY("vq.l1_cascade") c->learned_l1_cascade = atoi(v);
        else if KEY("vq.l1_base") c->learned_l1_cascade_base = atoi(v);
        else if KEY("vq.l1_norm") c->learned_l1_norm = parse_int(v);
        else if KEY("vq.l1_norm_final") c->learned_l1_norm_final = parse_int(v);
        else if KEY("vq.all_sampling") c->learned_all_sampling = atoi(v);
        else if KEY("vq.pre_quant_gn") c->pre_quant_groupnorm = atoi(v);
        else if KEY("vq.qk_norm") c->vq_qk_norm = parse_int(v);
        else if KEY("vq.enc_layers") c->enc_layers = atoi(v);
        else if KEY("vq.enc_window") c->enc_attn_window = atoi(v);
        else if KEY("vq.enc_convs") c->enc_conv_layers = atoi(v);
        else if KEY("vq.enc_conv_kernel") c->enc_conv_kernel = atoi(v);
        else if KEY("vq.enc_positional") c->enc_positional = atoi(v);
        else if KEY("vq.dec_layers") c->dec_layers = atoi(v);
        else if KEY("vq.dec_positional") c->dec_positional = atoi(v);
        else if KEY("tok.kind") set_str(c->pre_tok, sizeof(c->pre_tok), v);
        else if KEY("tok.model_id") set_str(c->hf_model_id, sizeof(c->hf_model_id), v);
        else if KEY("tok.eos_id") c->eos_token_id = parse_int(v);
        else if KEY("tok.pad_id") c->pad_token_id = parse_int(v);
        else if KEY("gen.temperature") c->gen_temperature = (float)atof(v);
        else if KEY("gen.top_k") c->gen_top_k = atoi(v);
        else if KEY("gen.top_p") c->gen_top_p = (float)atof(v);
        else if KEY("gen.cfg_scale") c->gen_cfg_scale = (float)atof(v);
        else if KEY("gen.freq") c->gen_freq = (float)atof(v);
        else if KEY("gen.mu") c->gen_mu = (float)atof(v);
        else if KEY("gen.adj") {
            if (v[0]) {
                c->gen_adj_a = (float)atof(v);
                char *q = strchr(v, ',');
                c->gen_adj_w = q ? atoi(q + 1) : 1;
                q = q ? strchr(q + 1, ',') : NULL;
                c->gen_adj_d = q ? (float)atof(q + 1) : 1.0f;
            }
        }
#undef KEY
#undef INT
#undef FLT
    }
    fclose(f);
    if (c->n_kv_heads < 0) c->n_kv_heads = c->n_head; /* full MHA default */
    c->quant_resi_ratio = c->quant_resi_ratio; /* 0 => Identity phi */
    return 0;
}

/* ---------------- bump arenas: persistent (load-time) + scratch (per-call) --- */
static unsigned char *arena_base = NULL;
static size_t arena_size = 0, arena_off = 0;
static unsigned char *parena_base = NULL;
static size_t parena_size = 0, parena_off = 0;

void arena_init(size_t bytes) {
    arena_free();
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMalloc((void **)&arena_base, bytes));
#else
    arena_base = (unsigned char *)malloc(bytes);
    if (!arena_base) { fprintf(stderr, "oom arena\n"); exit(1); }
#endif
    arena_size = bytes;
    arena_off = 0;
}

/* Default scratch size: NCM_ARENA_MB env, else 2048 MB. CUDA training needs
 * >1 GiB (per-layer sdpa dscores accumulate in-arena); 2 GiB is modest for
 * the sm_86 12 GiB target and harmless as host malloc. */
void arena_init_default(void) {
    const char *e = getenv("NCM_ARENA_MB");
    size_t mb = e ? (size_t)atol(e) : 2048;
    if (mb < 64) mb = 64;
    arena_init(mb * (1ULL << 20));
}

float *arena_alloc(size_t n_floats) {
    size_t need = n_floats * sizeof(float);
    /* 16-byte align */
    arena_off = (arena_off + 15) & ~(size_t)15;
    if (arena_off + need > arena_size) {
        fprintf(stderr, "arena overflow (%zu > %zu)\n", arena_off + need, arena_size);
        exit(1);
    }
    float *p = (float *)(arena_base + arena_off);
    arena_off += need;
    return p;
}

void arena_reset(void) { arena_off = 0; }

void arena_free(void) {
    if (!arena_base) return;
#ifdef NCM_CUDA
    cudaFree(arena_base);
#else
    free(arena_base);
#endif
    arena_base = NULL; arena_size = 0; arena_off = 0;
}

/* raw backend memcpy helpers */
void mem_d2h(void *h, const void *d, size_t n) {
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
#else
    memcpy(h, d, n);
#endif
}
void mem_h2d(void *d, const void *h, size_t n) {
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
#else
    memcpy(d, h, n);
#endif
}
void mem_d2d(void *d, const void *s, size_t n) {
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice));
#else
    memcpy(d, s, n);
#endif
}

void parena_init(size_t bytes) {
    if (parena_base) return;
#ifdef NCM_CUDA
    CUDA_CHECK(cudaMalloc((void **)&parena_base, bytes));
#else
    parena_base = (unsigned char *)malloc(bytes);
    if (!parena_base) { fprintf(stderr, "oom parena\n"); exit(1); }
#endif
    parena_size = bytes;
    parena_off = 0;
}
float *parena_alloc(size_t n_floats) {
    size_t need = n_floats * sizeof(float);
    parena_off = (parena_off + 15) & ~(size_t)15;
    if (parena_off + need > parena_size) {
        fprintf(stderr, "parena overflow (%zu > %zu)\n", parena_off + need, parena_size);
        exit(1);
    }
    float *p = (float *)(parena_base + parena_off);
    parena_off += need;
    return p;
}

void *backend_to_host(const float *d, size_t nbytes) {
    void *h = malloc(nbytes ? nbytes : 1);
    mem_d2h(h, d, nbytes);
    return h;
}

TrainData *train_data_load(const char *path, int max_rows) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NCMT", 4) != 0) {
        fprintf(stderr, "%s: bad magic\n", path); fclose(f); return NULL;
    }
    uint64_t nr = 0, rl = 0;
    if (fread(&nr, 8, 1, f) != 1 || fread(&rl, 8, 1, f) != 1) { fclose(f); return NULL; }
    if (max_rows > 0 && (int)nr > max_rows) nr = (uint64_t)max_rows;
    TrainData *d = (TrainData *)calloc(1, sizeof(TrainData));
    d->n_rows = (int)nr; d->row_len = (int)rl;
    d->rows = (int *)malloc(nr * rl * sizeof(int));
    if (fread(d->rows, sizeof(int), nr * rl, f) != nr * rl) {
        fprintf(stderr, "%s: short read\n", path);
        free(d->rows); free(d); fclose(f); return NULL;
    }
    fclose(f);
    return d;
}
void train_data_free(TrainData *d) { if (d) { free(d->rows); free(d); } }
