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

## CUDA status

CUDA covers inference forwards (kernels in the same file, verified vs torch:
GPT-2 maxrel ~1e-6, Qwen within fp32 noise floor). Backward passes stage to
host and run as CPU code by explicit design (correctness first); native CUDA
training (device backward + optimizer) is tracked as future work.

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
