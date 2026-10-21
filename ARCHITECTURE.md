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
| Backward (tape) | OpenMP host | **device kernels** by default (`NCM_BWD=cpu` forces host path) |
| AdamW | OpenMP over params (host) | **device kernel** by default (`NCM_ADAMW=cpu` forces host path) |
| Checkpointing | NCK1, exact resume | same (moments synced to host on save) |

Device AdamW keeps moments on-device (`Tensor` mirrors allocated once,
grads staged per step); it is **bit-identical** to the host path
(verified: identical checkpoint hashes, `NCM_ADAMW=cpu` vs default).
Training on CUDA works end-to-end (same loss as CPU step-by-step;
gradcheck passes on both; 4-step CPU-vs-CUDA drift is 9.7e-8 weights,
~1e-9 moments — well inside the 1e-3 `test-resume-cuda` gate).

Native device backward (`src/train_tape.c`, `NCM_CUDA` section): the tape
dispatches to per-op device kernels with a device-side grad map; LINEAR
goes through cuBLAS (`-lcublas`), everything else reproduces the host
summation order exactly; scatter-adds (EMBED/GATHER) stay host-staged to
preserve run-to-run determinism. VQ-only conv-family tapes fall back to
the host backward wholesale (no mixed mode). Verified: `cncm bwd-parity`
(host vs device grads on the sudoku workload, maxabs 8.6e-7),
`cncm bwdops` (all 25 non-conv ops on synthetic tensors, maxabs 1.5e-8),
device `gradcheck` passes, and resumed training with device backward is
byte-identical run-to-run. Sudoku step: ~5s (device) vs ~39s (host
backward), same losses. `NCM_BWD=cpu` restores the legacy path.

Taped GQA (`src/train_tape.c`, `src/train_qwen.c`): `t_rope`/`t_sdpa`
take an explicit `Hkv` (k/v heads); GQA backward fans dk/dv in over the
rep query heads on host and device with the same serial order
(bit-identical); new `t_headsplit` records the [B][T][C]->[B][H][T][hd]
transpose for Qwen's separate q/k/v linears. The fused-qkv NCP path
keeps its replication contract (`t_qkvsplit` emits H-head k/v).
Verified: `tests/gcheck.c` numeric checks for headsplit/rope_gqa/sdpa_gqa,
`cncm bwdops` gains headsplit + rope_gqa + sdpa_gqa (host-vs-device,
PASS), and `cncm qwen-taped` proves the full recorded Qwen forward
bit-matches `qwen.c` (maxabs 0.0, T=5 and T=20, CPU and CUDA).
`t_embed` takes a HOST idx (it stages internally) — the one host/device
contract in the tape; passing a staged buffer crashes (caught by review,
not by the compiler).

Memory: the tape holds all intermediates, plus per-layer CUDA SDPA
scratch (`dscores`). Default arena is 2 GiB (`NCM_ARENA_MB` overrides);
a sudoku NCP step needs ~1.1 GiB on CUDA. Qwen-scale fine-tuning does
*not* fit this model yet — see below.

## Checkpoints: NCK1, exact resume (`src/ckpt.c`)

Format `NCK1` v1, little-endian: tensor names/shapes/weights, AdamW
moments + step + hyperparameters, global step, batch-RNG state, scheduler
+ training config, dataset/config fingerprints, tokenizer identity,
trailing FNV-1a-64 checksum. Deliberately absent: wall-clock, pid,
hostname, argv — identical trajectories yield **byte-identical files**.

- Save is atomic: write `F.tmp.<pid>` → re-read + validate → fsync →
  `rename(2)` → fsync directory. The previous file is never touched
  until the replacement validates. Only completed-optimizer-step
  boundaries are saved (`accum_pending` must be 0).
- Load verifies checksum first, then parses into an image, runs every
  compatibility check (shapes, dataset hash/geometry, conf hash, schedule,
  tokenizer, optimizer hypers), and only then applies — corrupt or
  incompatible files are refused before live state is touched.
- `weights_load` accepts NCK1 (weights section only), so every
  inference/verify command reads `.nckpt` directly.
- Legacy `NCM1` via `--resume` loads as a loud **fresh-optimizer start**,
  never an exact resume. VQ snapshots stay weight-only (documented
  non-resumable): resuming them would also need EMA + xorshift-RNG state.
- CLI: `--resume R.nckpt` (exact), `--ckpt-out F`, `--ckpt-every N`,
  `--total T` (scheduler horizon, inherited on resume), `--tokenizer-id S`;
  `cncm ckpt-info --weights F` prints state + validates checksum.

Measured (sudoku-easy NCP, 200 tensors / 142 trainable):
`ckpt-final.nckpt` = **348.2 MiB** (153.9 weights + 194.2 moments);
save ~0.8s, load ~0.5s on local SSD. Exact-resume proof
(`tests/test_ckpt.sh`, N=4): loss lines continuous vs save/kill/resume
agree (14.2440 / 13.3810 / 13.0410 / 13.3739); mid and final checkpoints
byte-identical at NCM_THREADS=1 and 8, and finals match across thread
counts; CUDA leg resumes with 0.0 numeric diff (gate 1e-3). Refusal matrix:
truncation, bit-flips, wrong data/lr/tokenizer/dims, missing file —
all refused with no state touched and no stray files.

## Qwen2.5-0.5B fine-tuning: memory budget (measured, fp32)

Model: 494M params = **1.98 GiB** fp32 (embeddings alone 544 MB).
Full-AdamW training needs weights 2.0 + grads 2.0 + moments 4.0 =
**8.0 GiB** before a single activation. Activations at B=1, T=512 add
~2 GiB (logits [512×151936] alone are 311 MB; 24 layers × ~70 MB;
SDPA probs 14×512²×4 B/layer resident in the tape). Total ≈ **10 GiB**,
fitting a 12 GiB card at B=1/T≤512 with gradient accumulation for
larger effective batches.

Built since: the taped Qwen forward exists (`src/train_qwen.c`,
`qwen-taped` parity maxabs 0.0) with GQA-verified ops (gcheck numeric +
bwdops host-vs-device). Not yet built (explicit follow-ups, in order):
gradcheck-vs-numeric on a small config, then memory-conscious execution:
micro-batch accumulation
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
- `make test`: `tests/test_units.sh` (gcheck + ckpt_unit, weight-free,
  clean-checkout safe) + `tests/test_ckpt.sh --quick` (format/refusals +
  N=2 exact-resume proof + bwd-parity/bwdops where the binary is CUDA;
  skips cleanly without weights or on CPU).
  `make test-resume` runs the full N=4 CPU proof (1 and 8 threads verified);
  `make test-resume-cuda` the CUDA leg (numeric tolerance gate 1e-3).
- `make test-viewer[-cuda]`: `tests/test_viewer.sh` — golden boards
  (stepper == verified monolithic gen), streamed == `gen` output,
  ms-normalized determinism, `sudoku-check` valid/invalid + bad-unit
  flags, prefix validation, font-strip md5 (pixel-exact bitmap font),
  train `--view` loss/checkpoint identity vs no-view, `gen --view`
  same-board. Skips cleanly without weights.

## Native live viewer (SDL2, `src/viewer.c`)

`cncm train --view` / `cncm gen --view` open a native window on the live
process: training shows the batch-0 input puzzle, scrolling loss curve,
per-level loss + exact-code agreement, step/target, lr, accuracy,
elapsed and checkpoint flags as each step completes; solving shows the
puzzle, per-level progress (each event follows a real model forward via
the `NCPGen` stepper) and the final board with validity flags when
available. Levels are presented as code counts per refinement step,
never as logical deductions; no simulated results, no replay mode.

Separation: a render thread owns the window; computation pushes small
plain-data snapshots through a trylock mutex and never waits. Training
capture is read-only staging + host argmax inside `train_ncp_loss`
(one branch when off); with `--view` off the trajectory is untouched.
Proven: same-seed training with the viewer on vs off yields identical
losses and byte-identical checkpoints (`make test-viewer`). Closing the
window stops cleanly after the current step (no stray checkpoint);
without a display (or `SDL_VIDEODRIVER=dummy`) the viewer is a silent
no-op and computation runs headless. `train*.c` math, `ckpt.c` and the
CUDA paths are unchanged — only additive hooks.

`cncm sudoku-stream` (NDJSON level events) and `cncm sudoku-check`
(validity query) remain as scriptable machine interfaces over the same
stepper; they are covered by `test_viewer.sh`, not by the window.

## Build requirements

- CPU: `gcc` (C11), OpenMP (`libgomp`), `make`. No other dependencies.
- CUDA: `nvcc` (sm_86 target baked in `Makefile`; adjust `gencode` for other
  GPUs), CUDA runtime + `libgomp`.
- Offline-only Python (converters/reference fixtures, never linked):
  torch + transformers + safetensors + a HF hub snapshot.
- Test data that is *not* in git: model weights (`weights/`, git-ignored),
  corpora, `.nctrain` shards. Fixtures under `tests/fixtures/` *are*
  committed (small float32 blobs the verify commands need).
