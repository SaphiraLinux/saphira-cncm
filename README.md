# cncm — Next-Concept Modeling in dependency-free C

A complete C port of [NCM](https://github.com/HazyResearch/ncm)
(`Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation`),
with **zero runtime dependencies** (C standard library + optional CUDA only).
A single `cncm` binary runs inference, generates text, evaluates, and trains.

## Status

| Area | Status |
|---|---|
| CPU backend (`make`) | ✅ verified vs torch (see Parity) |
| CUDA backend (`make cuda`) | ✅ verified vs torch |
| Weight format (`.ncm` + `.conf`) | ✅ via `tools/export.py` (one-time, needs torch) |
| VQ-VAE forward (encode / quantize / decode) | ✅ exact idx match, logits < 1e-3 |
| NCP forward (all head types, RoPE, GQA, CFG shapes) | ✅ < 1e-3 |
| Learned L1 cascade (multi-stage + GroupNorm) | ✅ verified (tinystories) |
| Tokenizers: Sudoku, byte-BPE (custom/GPT-2 incl. bytes_to_unicode) | ✅ exact match vs Python (custom + HF GPT-2) |
| Tokenizers: BERT WordPiece / Qwen patterns | ⚠️ implemented, lightly tested (pattern select present) |
| Generate (greedy + temperature/top-k/top-p, CFG, freq/adj penalties) | ✅ |
| `eval-sudoku` | ✅ 1988/2000 (99.40%), identical to Python |
| Autograd tape + all backward kernels | ✅ grad-checked (linear/layernorm/rmsnorm/SDPA/split/merge) |
| NCP training (AdamW, cosine schedule) | ✅ noise-recovery demonstrated |
| VQ-VAE training (commitment + EMA) | ✅ loss decreases; EMA ones-init (see TRAINING.md) |
| Metrics (BLEU/chrF/ROUGE) | ✅ BLEU exact; chrF/R-1 within 0.3 of sacrebleu/rouge (see Metrics) |
| genPPL scorer, XSum/WMT eval CLIs | 🚧 scorer + full conditional evals are follow-ups |

## Quick start

```bash
# 1. one-time export (needs torch + the ncm checkout + HF checkpoints)
python3 tools/export.py --config /scratch/ncm/config/eval/sudoku-easy.yaml \
    --checkpoint-dir /scratch/ncm/artifacts/sudoku/easy --out weights/sudoku-easy
python3 tools/export_data.py sudoku --in /scratch/ncm/data/sudoku/easy \
    --out weights/sudoku-easy

# 2. build (CPU) or CUDA
make            # -> ./cncm
make cuda       # -> ./cncm-cuda (sm_86, no other deps)

# 3. run
./cncm infer --conf weights/sudoku-easy/model.conf \
    --weights weights/sudoku-easy/model.ncm --asset-dir weights/sudoku-easy \
    --ids "$(head -1 weights/sudoku-easy/inputs.txt | cut -c1-400)"
./cncm gen --conf ... --weights ... --asset-dir ... --prefix "11,9,2,..." [--temp 0]
./cncm eval-sudoku --conf ... --weights ... --asset-dir ... \
    --inputs weights/sudoku-easy/inputs.txt --refs weights/sudoku-easy/refs.txt [--n 2000]
./cncm gradcheck --conf ... --weights ... --data weights/sudoku-easy/train64.nctrain [--n 4]
./cncm train --conf ... --weights ... --data train64.nctrain [--steps N]
./cncm train-vq --conf ... --weights ... --data train_ids.nctrain [--steps N]
./cncm gpt2-gen --conf weights/gpt2/model.conf --weights weights/gpt2/model.ncm \
    --asset-dir weights/gpt2 --prompt TEXT [--n N]
./cncm gpt2-verify --conf weights/gpt2/model.conf --weights weights/gpt2/model.ncm \
    --asset-dir weights/gpt2 --fixtures tests/fixtures
```

Training data: `tools/export_train.py` writes `.nctrain` shards
(`[P prefix | total codes]` for NCP, `[T token ids]` for VQ-VAE).

## Layout

```
src/
  ncm.h          public API (tensors, config, model + training API)
  tensor.c       tensors, .ncm loader, .conf parser, arenas, memcpy helpers
  kernels.c      forward kernels (CPU loops + CUDA kernels, same file)
  vq.c           multiscale residual quantizer + EMA codebook maintenance
  vqvae.c        VQ-VAE encoder/decoder (inference)
  ncp.c          Next-Concept transformer forward + coarse-to-fine generate
  tok.c          pre-tokenizers (sudoku / byte-BPE / WordPiece)
  train_tape.c   reverse-mode autograd tape + all backward kernels
  train.c        AdamW, recording NCP forward, NCP training step/loop
  train_vq.c     recording VAE forward, VQ-VAE training step/loop + EMA
   metrics.c      eval metrics (BLEU/chrF/ROUGE — in progress)
   gpt2.c         native GPT-2 decoder (124M verified vs transformers:
                  exact ids, logits maxrel~1e-6, greedy gen, ppl) +
                  gpt2-gen / gpt2-ppl / gpt2-verify
   main.c         `cncm` CLI
tools/
   export.py        .pt -> .ncm + .conf (one-time, needs torch)
   export_data.py   eval pickles -> plain text
   export_train.py  training rows -> .nctrain
   gpt2import.c     HF safetensors/vocab/merges -> .ncm + cncm BPE text (pure C)
   gpt2-124m.conf   template config for a converted GPT-2 124M dir
   gen_gpt2_ref.py  one-time transformers fixtures (offline only)
   gen_unitables.py one-time Unicode table generator for tok_tables.c
tests/
   fixtures/      GPT-2 transformers references (ids, logits, generations, ppl)
   ktest_kernels.c  kernel parity harness (CPU vs CUDA)
   gcheck.c         host-side grad-check pieces
   train_smoke.py   (see TRAINING.md)
```

## Design notes

- **No dependencies**: the binary links only libc/libm (+libcudart for `-cuda`).
  `.pt` files are converted once by `tools/export.py` into a trivial flat
  format (magic `NCM1`, name/shape/raw-f32 records) plus a `key=value`
  config file. Tokenizer assets (BPE vocab/merges, WordPiece lists) are
  plain text.
- **Two arenas**: a persistent arena (rope tables, level embeddings) and a
  per-step/per-level bump arena for temporaries. Training resets the scratch
  arena per level/board; cross-iteration state lives in explicit tensors.
- **Backend model**: the same `.c` files compile for CPU (`gcc`) and CUDA
  (`nvcc -x cu`). Device memory is abstracted (`mem_d2h`/`mem_h2d`/arenas);
  backward passes stage to host and run plain C loops (correct everywhere,
  CPU-adequate speed on purpose).
- **PyTorch parity**: `cncm infer` prints checksums of every stage; the
  `tests/` scripts compare against torch. Sudoku: idx 100% identical,
  logits < 4e-4. TinyStories incl. tied heads + cascades: < 2e-6 relative.

## Known limitations (honest list)

- Training is a **reference implementation**: correct, not fast. Full-model
  backward on CPU is ~10-25 s/step (B=4, sudoku-25M). CUDA forward helps
  inference; training backward is host-side by design.
- `train-vq` EMA uses ones-init for cluster counts (see TRAINING.md);
  set to zeros for bit-exact paper parity at large batch sizes.
- Level-0 codes are unpredictable (uniform-ish) on Sudoku-like tasks; the
  trainer supervises all levels equally. See TRAINING.md for the analysis.
- Metrics (BLEU/chrF/ROUGE) and the GPT-2 scorer are stubbed; XSum/WMT
  end-to-end eval CLIs are not wired yet.
- The Qwen/HF pre-tokenizer paths and non-RoPE models are implemented but
  lightly exercised (only sudoku + tinystories + BPE are fully verified).

## Metrics

`src/metrics.c` implements sacrebleu-compatible corpus BLEU (13a tokenization,
exp smoothing) and chrF (order 6, β=2, corpus micro-average), plus ROUGE-1/2/L
F-measure with Porter stemming. Validated on released NCM outputs:

| Metric | C | Python reference | Δ |
|---|---|---|---|
| BLEU (WMT14, n=50) | 15.5992 | 15.5992 (sacrebleu) | exact |
| chrF (WMT14, n=50) | 45.34 | 45.07 (sacrebleu) | 0.27 |
| ROUGE-1/2/L (XSum, n=50) | 32.53 / 12.16 / 26.47 | 32.76 / 12.16 / 26.47 (rouge_score) | ≤0.23 |

Remaining deltas are tokenizer/stemmer edge cases (1/52 Porter words differ
from NLTK; chrF whitespace/n-gram boundary conventions).
