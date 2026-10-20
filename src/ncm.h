/* cncm — Next-Concept Modeling in dependency-free C.
 *
 * Single binary, two backends selected at compile time:
 *   make          -> ./cncm      (CPU)
 *   make cuda     -> ./cncm-cuda (NVIDIA GPU, compute capability 8.6+)
 *
 * Subcommands:
 *   cncm infer    --conf <model.conf> --weights <model.ncm> --input <ids|text> ...
 *   cncm generate --conf ... --weights ... [--prefix ...] [--temp f] [--top-k n]
 *                 [--top-p f] [--cfg f] [--freq f] [--mu f] [--adj a,w,d] [--seed n]
 *   cncm eval     --conf ... --weights ... --data-dir ... [--n N] [--task sudoku|xsum|wmt14]
 *   cncm train    --conf ... [--resume ...]   (see TRAINING.md status)
 *   cncm ppl      --gpt2 <gpt2.ncm> --texts <file>   (generative-perplexity scorer)
 *
 * Precision: float32 everywhere, matching the Python reference.
 */
#ifndef NCM_H
#define NCM_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* tensor (tensor.h)                                                   */
/* ------------------------------------------------------------------ */

#define NCM_MAX_DIMS 4

typedef struct {
    float  *data;              /* host ptr (CPU) or device ptr (CUDA build) */
    int64_t shape[NCM_MAX_DIMS];
    int     ndim;
    int64_t n;                 /* total elements */
} Tensor;

/* Allocate / free. Memory lives on the active backend. */
Tensor *tensor_alloc(int ndim, const int64_t *shape);
void    tensor_free(Tensor *t);
int64_t tensor_numel(int ndim, const int64_t *shape);

/* Host <-> backend transfer helpers (no-ops on CPU builds). */
void tensor_to_host(const Tensor *t, float *out);
void tensor_to_backend(Tensor *t, const float *in);
void mem_d2h(void *h, const void *d, size_t n);
void mem_h2d(void *d, const void *h, size_t n);
void mem_d2d(void *d, const void *s, size_t n);

/* Bump arena for temporaries: one big backend allocation, reset per pass. */
void  arena_init(size_t bytes);
float *arena_alloc(size_t n_floats);
void  arena_reset(void);
void  arena_free(void);
void  parena_init(size_t bytes);
float *parena_alloc(size_t n_floats);

/* .ncm weight file: magic "NCM1", then raw float32 tensors. */
typedef struct {
    char    name[256];
    Tensor *t;
} NamedTensor;

typedef struct {
    NamedTensor *items;
    int          count;
} WeightStore;

int  weights_load(WeightStore *ws, const char *path);
void weights_free(WeightStore *ws);
Tensor *weights_get(WeightStore *ws, const char *name); /* NULL if missing */

/* ------------------------------------------------------------------ */
/* config (.conf key=value, written by tools/export.py)                */
/* ------------------------------------------------------------------ */

#define NCM_MAX_LEVELS 64
#define NCM_MAX_VOCAB_LIST 64

typedef struct {
    char task[64];
    int  n_layer, n_head, n_embd, context_length, prefix_len;
    int  bias;                 /* linear bias on/off */
    int  raw_token_prefix, contextual_prefix;
    int  prefix_pad_token_id;
    int  vocab_size[NCM_MAX_VOCAB_LIST];
    int  n_levels_out;         /* heads = levels - (query_level0 ? 0 : 1) */
    int  predict_level0, query_level0, l0_autoregressive;
    int  attn_pattern_block_diag; /* 1 = block_diagonal, 0 = block_causal */
    int  use_positional_encoding, use_level_encoding;
    int  use_rope, rope_base, use_qk_norm, use_flex_attention;
    int  n_kv_heads;
    /* ctx encoder */
    int  ctx_enabled, ctx_n_layer, ctx_vocab;
    /* post upsample conv */
    int  puc_enabled, puc_kernel;
    /* output heads */
    int  shared_output_head, shared_head_per_level_bias, untied_head_bias;
    int  tie_heads_to_codebook, tie_heads_learn_scale, tie_heads_trainable;
    int  tie_heads_bias;
    int  use_swiglu, use_geglu, use_relu2, parallel_block, normformer;
    int  use_layerscale, use_rezero;
    float layerscale_init;
    float dropout;             /* informational at inference */
    /* quantizer / vqvae */
    int  vq_embed_dim, vq_in_vocab, vq_context_length, vq_n_layers;
    int  vq_compression;
    int  levels[NCM_MAX_LEVELS];
    int  n_qlevels;
    int  vq_vocab[NCM_MAX_LEVELS];
    float quant_resi_ratio;
    int   quant_resi_enabled;
    int   quant_resi_share_mode;
    int   learned_l1_sampling, learned_l1_cascade;
    int   learned_l1_cascade_base;
    int   learned_l1_norm, learned_l1_norm_final;
    int   learned_all_sampling;
    int   pre_quant_groupnorm;
    int   vq_qk_norm;
    int   enc_layers, enc_attn_window, enc_conv_layers, enc_conv_kernel;
    int   enc_positional;
    int   dec_layers, dec_positional;
    /* pre-tokenizer */
    /* pre-tokenizer */
    char pre_tok[16];          /* sudoku | bpe | hf */
    char hf_model_id[128];
    int  eos_token_id, pad_token_id;
    /* generation defaults */
    float gen_temperature;
    int   gen_top_k;
    float gen_top_p, gen_cfg_scale;
    float gen_freq, gen_mu;
    float gen_adj_a; int gen_adj_w; float gen_adj_d;
} NCMConfig;

int config_load(NCMConfig *c, const char *path);

/* ------------------------------------------------------------------ */
/* kernels (kernels.h) — out-params last, all batched over leading dim */
/* ------------------------------------------------------------------ */

/* y = x @ W^T (+ b). x:[...,K], W:[M,K], b:[M] or NULL, y:[...,M] */
void linear_fwd(const float *x, const float *W, const float *b,
                float *y, int64_t batch, int M, int K);
/* embedding lookup: y[i,:] = W[idx[i],:] */
void embedding_fwd(const float *W, const int *idx, float *y, int64_t n, int D);
void layernorm_fwd(const float *x, const float *w, const float *b,
                   float *y, int64_t rows, int C);
void layernorm_eps_fwd(const float *x, const float *w, const float *b,
                       float *y, int64_t rows, int C, float eps);
void gelu_tanh_fwd(const float *x, float *y, int64_t n); /* HF gelu_new */
void rmsnorm_fwd(const float *x, const float *w, float *y, int64_t rows, int C);
void gelu_fwd(const float *x, float *y, int64_t n);
void silu_fwd(const float *x, float *y, int64_t n);
void softmax_fwd(const float *x, float *y, int64_t rows, int C); /* row softmax */
void rope_apply(float *q, float *k, const float *cos, const float *sin,
                int B, int H, int T, int hd);
/* SDPA with additive mask: out = softmax(QK'/s + mask) V. mask:[Tq,Tkv] or NULL */
void sdpa_fwd(const float *q, const float *k, const float *v, const float *mask,
              float *out, int B, int H, int Tq, int Tkv, int hd);
/* conv1d (NCL): y[n,c,t] = bias + sum_{kc,kk} W[c,kc,kk] x[n,kc,t+kk-pad] */
void conv1d_fwd(const float *x, const float *W, const float *b, float *y,
                int N, int Cin, int Cout, int T, int K, int stride, int pad);
/* conv-transpose1d, stride s, kernel K, pad 0 (matches repo usage) */
void convtranspose1d_fwd(const float *x, const float *W, const float *b, float *y,
                         int N, int Cin, int Cout, int T, int K, int stride);
void groupnorm_fwd(const float *x, float *y, int N, int G, int C, int T); /* NCL, affine=false */
void interpolate_area_fwd(const float *x, float *y, int N, int C, int Tin, int Tout);
void interpolate_linear_fwd(const float *x, float *y, int N, int C, int Tin, int Tout);
int  argmax_row(const float *x, int n);
void rowwise_l2sq(const float *a, const float *b, float *d2,
                  int64_t rows, int64_t colsA, int C); /* ||a_i - b_j||^2 */
void add_fwd(const float *a, const float *b, float *y, int64_t n);
void scale_fwd(const float *x, float s, float *y, int64_t n);
void add_inplace_fwd(float *y, const float *x, int64_t n);
void qkv_split_fwd(const float *in, float *qq, float *kk, float *vv,
                   int B, int T, int C, int H, int hd, int K, int rep);
void head_merge_fwd(const float *ctx, float *out, int B, int T, int C, int H, int hd);
/* training tape (defined in train_tape.c) */
typedef struct Tape Tape;
void t_qkvsplit(Tape *t, const float *in, float *qq, float *kk, float *vv,
                int B, int T, int C, int H, int hd, int K, int rep);
void t_headmerge(Tape *t, const float *ctx, float *out, int B, int T, int C, int H, int hd);
/* sampling */
int  sample_topkp(const float *logits, int V, float temp, int top_k, float top_p,
                  unsigned long long *rng);
unsigned long long rng_next(unsigned long long *state);

/* ------------------------------------------------------------------ */
/* model modules                                                       */
/* ------------------------------------------------------------------ */

typedef struct VQState VQState;
VQState *vq_load(const NCMConfig *c, WeightStore *ws, const char *prefix);
void     vq_free(VQState *vq);
int      vq_total_codes(const VQState *vq);
/* z:[B,L,D] -> idx:[B,sum(levels)] (argmin path) */
void vq_z_to_idx(VQState *vq, const float *z, int B, int *idx);
/* idx -> cumulative z_hats; zhats[lvl] is [B][L][D] at full length L */
void vq_idx_to_zhats(VQState *vq, const int *idx, int B, float **zhats);
/* autoregressive step input; mirrors get_next_autoregressive_input */
void vq_next_input(VQState *vq, const int *codes, int level_idx,
                   float *z_hat, float *x_out, int B);
/* NCM transformer target input from codes (idx_to_ncm_input) */
void vq_idx_to_ncm_input(VQState *vq, const int *idx, int B, float *x_out);

typedef struct VQVAEState VQVAEState;
VQVAEState *vqvae_load(const NCMConfig *c, WeightStore *ws);
void        vqvae_free(VQVAEState *v);
VQState    *vqvae_vq(VQVAEState *v);
void vq_ema_init(VQState *vq);
void vq_ema_update(VQState *vq, const float *z, const int *codes, int B, double decay, double eps);
int vq_nlevels(const VQState *vq);
int vq_level_len(const VQState *vq, int k);
int vq_level_V(const VQState *vq, int k);
int vq_Ltotal(const VQState *vq);
int vq_Ddim(const VQState *vq);
typedef struct VQTrainer VQTrainer;
VQTrainer *vqtrainer_new(VQVAEState *vqvae, WeightStore *ws, NCMConfig *c,
                         double lr, double wd, double warmup, double total);
double train_vq_step(VQTrainer *t, Tape *tp, const int *ids, int B);
void vqtrainer_free(VQTrainer *t);
void vqtrainer_free(VQTrainer *t);
void vqvae_encode_z(VQVAEState *v, const int *ids, float *z, int B); /* [B][T][D] */
void vqvae_decode(VQVAEState *v, const float *zhat, float *logits, int B);

typedef struct NCPState NCPState;
NCPState *ncp_load(const NCMConfig *c, WeightStore *ws, VQVAEState *vqvae);
void      ncp_free(NCPState *m);
int   ncp_P(NCPState *m);
int   ncp_T(NCPState *m);
int   ncp_ntgt(NCPState *m);
int   ncp_lvlfirst(NCPState *m);
float *ncp_rope_cos(NCPState *m);
float *ncp_rope_sin(NCPState *m);
float *ncp_rope_gen_cos(NCPState *m, int k);
struct VQState *ncp_vq(NCPState *m);
float *ncp_rope_gen_sin(NCPState *m, int k);
void *backend_to_host(const float *d, size_t nbytes);
typedef struct { int *rows; int n_rows, row_len; } TrainData;
TrainData *train_data_load(const char *path, int max_rows);
void train_data_free(TrainData *d);
int weights_save(WeightStore *ws, const char *path);
/* Full forward used in training/scoring: x:[B][P+T][D] in vqvae space. */
void ncp_forward(NCPState *m, const float *x, const int *prefix_ids,
                 float *logits, int B);
/* Coarse-to-fine generation. Returns malloc'd [B][sum levels] codes. */
int *ncp_generate(NCPState *m, const int *l0, const int *prefix_ids, int B,
                  float temp, int top_k, float top_p, float cfg_scale,
                  float freq, float mu, float adj_a, int adj_w, float adj_d,
                  const double *allow, unsigned long long seed);

/* ------------------------------------------------------------------ */
/* tokenizers (tok.h)                                                  */
/* ------------------------------------------------------------------ */

typedef struct TokState TokState;
TokState *tok_load(const NCMConfig *c, const char *asset_dir); /* bpe vocab/merges, wordpiece vocab */
void tok_free(TokState *t);
int uni_is_letter(int cp);
int uni_is_number(int cp);
int uni_is_space(int cp);
int  tok_encode(TokState *t, const char *text, int *out, int cap);
int  tok_decode(TokState *t, const int *ids, int n, char *out, int cap,
                int skip_special);
int  tok_eos(const TokState *t);
int  tok_pad(const TokState *t);

/* ------------------------------------------------------------------ */
/* metrics + gpt2 scorer                                               */
/* ------------------------------------------------------------------ */

double metric_bleu_corpus(char **hyps, char **refs, int n);      /* sacrebleu-compatible */
double metric_chrf(char **hyps, char **refs, int n);
void   metric_rouge(char **hyps, char **refs, int n,
                    double *r1, double *r2, double *rl);
double gpt2_genppl(const char *gpt2_weights, const char *gpt2_conf,
                   char **hyps, int n, int max_len);
int gpt2_cmd_gen(const NCMConfig *c, const char *weights, const char *adir,
                 const char *prompt, const char *ids_csv, int n_new,
                 float temp, int top_k, float top_p, unsigned long long seed);
int gpt2_cmd_ppl(const NCMConfig *c, const char *weights, const char *adir,
                 const char *texts_path);
int gpt2_cmd_verify(const NCMConfig *c, const char *weights, const char *adir,
                    const char *fixtures_dir);

/* ------------------------------------------------------------------ */
/* training (train.h) — present, CPU reference; see TRAINING.md        */
/* ------------------------------------------------------------------ */

int train_run(const NCMConfig *c, const char *weights_path,
              const char *data_path, int max_steps, double lr);

/* ---- training (train_tape.c, train.c) ---- */
Tape *tape_new(void);
void  tape_free(Tape *t);
void  tape_clear(Tape *t);
void t_linear(Tape *t, const float *x, const float *W, const float *b,
              float *y, int64_t batch, int M, int K);
void t_embed(Tape *t, const float *W, const int *idx, int V, float *y,
             int64_t n, int D);
void t_layernorm(Tape *t, const float *x, const float *w, const float *b,
                 float *y, int64_t rows, int C);
void t_rmsnorm(Tape *t, const float *x, const float *w, float *y,
               int64_t rows, int C);
void t_groupnorm(Tape *t, const float *x, float *y, int N, int G, int C, int T);
void t_gelu(Tape *t, const float *x, float *y, int64_t n);
void t_mul(Tape *t, const float *a, const float *b, float *y, int64_t n);
void t_relu2(Tape *t, const float *x, float *y, int64_t n);
void t_silu(Tape *t, const float *x, float *y, int64_t n);
void t_softmax(Tape *t, const float *x, float *y, int64_t rows, int C);
void t_rope(Tape *t, const float *xq, const float *xk, float *yq, float *yk,
            const float *cos, const float *sin, int B, int H, int T, int hd);
void t_sdpa(Tape *t, const float *q, const float *k, const float *v,
            const float *mask, float *o, int B, int H, int Tq, int Tkv, int hd);
void t_conv(Tape *t, const float *x, const float *W, const float *b, float *y,
            int N, int Cin, int Cout, int T, int K, int stride, int pad);
void t_convt(Tape *t, const float *x, const float *W, const float *b, float *y,
             int N, int Cin, int Cout, int T, int K, int stride);
void t_iarea(Tape *t, const float *x, float *y, int N, int C, int Tin, int Tout);
void t_ilin(Tape *t, const float *x, float *y, int N, int C, int Tin, int Tout);
void t_add(Tape *t, const float *a, const float *b, float *y, int64_t n);
void t_concat(Tape *t, const float *a, const float *b, float *y, int64_t na, int64_t nb);
void t_join(Tape *t, const float *a, const float *b, float *y, int B, int P, int T, int C);
void t_gather_rows(Tape *t, const float *x, const int *rows, int nrows, int C, float *y);
void t_scale(Tape *t, const float *x, float s, float *y, int64_t n);
void t_ce(Tape *t, const float *logits, const int *tgt, const float *wmask,
          float *loss, float *probs, int64_t rows, int V, double weight);
void t_mse(Tape *t, const float *a, const float *b_const, float *loss, int64_t n, double weight);
void t_tied(Tape *t, const float *z, const float *E, const float *bias,
            float *out, int64_t rows, int Dd, int V, double dscale, double invT);
typedef struct {
    float **params;
    float **grads;
    int64_t *sizes;
    int n;
} PMap;
void tape_backward(Tape *t, PMap *pm, float seed);
typedef struct AdamW {
    float **params;   /* backend pointers (WeightStore tensors) */
    float **m, **v;   /* host first/second moments */
    float **g;        /* host grad accumulators */
    int64_t *sizes;
    double *lr;       /* per-param lr multiplier */
    int n;
    double b1, b2, eps, wd;
    long step;
} AdamW;
AdamW *adamw_new(WeightStore *ws, const int *frozen);
void adamw_zero_grad(AdamW *o);
void adamw_step(AdamW *o, double base_lr, double warmup, double total);
void adamw_free(AdamW *o);
/* PMap view over an AdamW (shares grad buffers) */
PMap *pmap_from_adam(AdamW *o);
void pmap_free(PMap *pm);

#endif /* NCM_H */
