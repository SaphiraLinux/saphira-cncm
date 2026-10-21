# CNCM architecture

Native C port of Next-Concept Modeling: VQ-VAE tokenizer, NCP coarse-to-fine
generator, plus verified GPT-2 (124M) and Qwen2.5-0.5B decoders. One codebase
builds a CPU binary (`make` → `cncm`) and a CUDA binary (`make cuda` →
`cncm-cuda`); backend selection is compile-time (`NCM_CUDA`), same sources.

## Components

| Area | Files | Notes |
|---|---|---|
| Tensors, `.ncm`/`.conf` I/O, arenas, memcpy shims | `src/tensor.c`, `src/ncm.h` | bump arenas (1 GiB scratch, persistent parena); `mem_h2d`/`mem_d2h` abstract host/device |
| Forward kernels (CPU loops + CUDA kernels, one file) | `src/kernels.c` | linear, layernorm (+eps variant), RMSNorm, GELU exact/tanh, SiLU, softmax, SDPA (+GQA), rope/GQA rope, concat/gather/add, conv family |
| VQ-VAE (inference) | `src/vq.c`, `src/vqvae.c` | encoder/quantize/decode, tied codebook heads |
| NCP transformer (inference) | `src/ncp.c` | blocks, GQA, QK-norm, gated/ungated MLP, per-level heads, coarse-to-fine `gen` |
| Tokenizers | `src/tok.c`, `src/tok_tables.c` | sudoku / exact GPT-2 BPE (generated Unicode tables) / Qwen2 pretokenizer; kinds `sudoku\|bpe\|qwen\|hf` |
| GPT-2 decoder | `src/gpt2.c` | learned pos-embed, causal MHA, gelu_new, tied head; `gpt2-gen/ppl/verify` |
| Qwen2 decoder | `src/qwen.c` | RMSNorm, GQA (no QK-norm — matches Qwen2.5 checkpoints), SwiGLU, rope theta 1e6, tied head |
| Metrics | `src/metrics.c`, `src/gpt2.c` (`gpt2_genppl`) | BLEU/chrF/ROUGE, perplexity |
| Autograd tape + backward kernels | `src/train_tape.c` | reverse-mode, host-side; all ops grad-checked (`tests/gcheck.c`) |
| Trainers | `src/train.c` (NCP/AdamW), `src/train_vq.c` (VQ-VAE/EMA) | `train`, `train-vq`, `gradcheck` CLIs |
| Offline converters (pure C, no torch) | `tools/gpt2import.c`, `tools/qwenimport.c` | safetensors + HF vocab/merges → `.ncm` + cncm BPE text |
| Corpus exporter | `tools/export_text.c` | verse-tagged text → tokenized `.nctrain` + JSON span manifest |
| One-time Python helpers (never on a runtime path) | `tools/export*.py`, `tools/gen_*_ref.py`, `tools/gen_unitables.py` | torch/HF/unidata used offline only |

## Data flow

- **Train (NCP)**: `.nctrain` rows `[P prefix | level-major codes]` → record
  forward (`train_ncp_loss`) → tape backward (host) → AdamW → `.ncm`.
- **Train (VQ-VAE)**: `.nctrain` rows of token ids (length = `vq_context_length`).
- **Corpus export**: `bible_corpus.txt` (`TRANS|book|ch|v<TAB>text`) →
  `export_text` → `.nctrain` + manifest (row → verse spans for byte-exact
  round-trip, verified by `tests/roundtrip_text.py`).
- **Decode**: GPT-2/Qwen forward → tied-head logits → greedy or top-k/top-p
  sampling (`sample_topkp`); perplexity via chunked NLL.

## Threading (CPU)

Thread count: `NCM_THREADS` env var, default **8 workers**
(`omp_set_num_threads` at startup; `if(...)` guards keep small shapes
serial). No other configuration needed.

| Path | Status |
|---|---|
| Forward kernels (`kernels.c`, `ncp.c`) | OpenMP `parallel for` (pre-existing) |
| Training backward (`train_tape.c`) | OpenMP on all major ops (linear/CE/SDPA/LN/RMSN/GN/conv/split/merge/elementwise/adam); scatter-adds (embed/gather) intentionally serial |
| `adamw_step` | OpenMP over params |
| Stay serial by design | generation loop (autoregressive), tokenizer (single pass is fast), `hadd_into` (called from parallel regions; nesting stays serial) |

Determinism contract: every parallel loop partitions *outputs* and keeps
the serial summation order inside, so results are **bit-identical at any
thread count**. Proof (sudoku NCP, one train step): sha256 of the saved
checkpoint identical for NCM_THREADS=1,2,4,8. Tokenizer and greedy
generation likewise byte-identical across counts.

Measured (32-core box, sudoku train step, interleaved A/B at 32 threads):
serial-backward 74–77s vs parallel 18–19s (**~4.1x**); scaling 1→32 threads:
260s → 112s → 57s → 35s → 18s. GPT-2 generation (5 tokens): 2.24s at 1
thread → 0.67s at 8, flat after (memory-bound). Full 118MB Bible
tokenization: 6.1s single-threaded.

## CUDA status: training (native, incremental)

Forward kernels run on device (verified vs transformers: GPT-2 maxrel ~1e-6,
Qwen within fp32 noise floor). What runs where today:

| Piece | CPU build | CUDA build |
|---|---|---|
| Forward | OpenMP host | device kernels (`NCM_CUDA`) |
| Backward (tape) | OpenMP host | **same host code** (stages via `mem_d2h`) |
| AdamW | OpenMP over params (host) | **device kernel** by default (`NCM_ADAMW=cpu` forces host path) |
| Checkpointing | `.ncm` save/load, bit-exact | same |

Device AdamW keeps moments on-device (`Tensor` mirrors allocated once,
grads staged per step); it is **bit-identical** to the host path
(verified: identical checkpoint hashes, `NCM_ADAMW=cpu` vs default).
Training on CUDA works end-to-end (same loss as CPU: 14.2440 both;
gradcheck passes on both; CPU-vs-CUDA checkpoints agree to 3e-8).

Memory: the tape holds all intermediates, plus per-layer CUDA SDPA
scratch (`dscores`). Default arena is 2 GiB (`NCM_ARENA_MB` overrides);
a sudoku NCP step needs ~1.1 GiB on CUDA. Qwen-scale fine-tuning does
*not* fit this model yet — see below.

## Qwen2.5-0.5B fine-tuning: memory budget (measured, fp32)

Model: 494M params = **1.98 GiB** fp32 (embeddings alone 544 MB).
Full-AdamW training needs weights 2.0 + grads 2.0 + moments 4.0 =
**8.0 GiB** before a single activation. Activations at B=1, T=512 add
~2 GiB (logits [512×151936] alone are 311 MB; 24 layers × ~70 MB;
SDPA probs 14×512²×4 B/layer resident in the tape). Total ≈ **10 GiB**,
fitting a 12 GiB card at B=1/T≤512 with gradient accumulation for
larger effective batches.

Not yet built (explicit follow-ups, in order): a taped Qwen forward
(the tape has no GQA-SDPA; verify each op), gradcheck-vs-numeric on a
small config, then memory-conscious execution: micro-batch accumulation
(no new machinery), frozen-backbone / PMap-subset updates (the tape
already supports partial params), and only then fp16 masters +
activation checkpointing if needed. 1.5B stays a later step; 2B-class
models are the stated ceiling.

## Verification (all in-repo, no network at test time)

- `tests/gcheck.c`: per-op analytic-vs-numeric grads incl. a qkv-split
  row-stride regression test.
- `cncm gpt2-verify` / `qwen-verify`: token ids exact (incl. unicode/digits),
  full-logits maxrel vs transformers fp32 fixtures, greedy generation exact,
  perplexity NLL match (`tests/fixtures/`).
- `tests/roundtrip_text.py`: export round-trip byte-exact over corpora.
- `make test` runs the parity harness (currently broken upstream:
  references a `tests/test_parity.py` that was never committed; tracked).

## Build requirements

- CPU: `gcc` (C11), OpenMP (`libgomp`), `make`. No other dependencies.
- CUDA: `nvcc` (sm_86 target baked in `Makefile`; adjust `gencode` for other
  GPUs), CUDA runtime + `libgomp`.
- Offline-only Python (converters/reference fixtures, never linked):
  torch + transformers + safetensors + a HF hub snapshot.
- Test data that is *not* in git: model weights (`weights/`, git-ignored),
  corpora, `.nctrain` shards. Fixtures under `tests/fixtures/` *are*
  committed (small float32 blobs the verify commands need).
