# TRAINING.md — how `cncm train` / `cncm train-vq` / `cncm train-qwen` work

The upstream NCM repo ships **checkpoints and eval code only** — there is no
training script to port. What follows documents the training path built here:
what it implements, what was assumed (and verified), and where it differs.

## Architecture

```
.nctrain shard ([P prefix | total codes] or [T token ids])
        |
  train_ncp_forward (recorded)  /  train_encode/decode (recorded)
        | tape records every differentiable op
  t_ce / t_mse (per-level, weighted)
        |
  tape_backward  (host-side reverse pass, exact per-op grads)
        |
  AdamW (decoupled weight decay, cosine warmup schedule)
        |
  versioned .nckpt checkpoint (weights + optimizer + RNG + identity;
  exact resume via --resume; inference also reads .nckpt directly)
```

Key files: `src/train_tape.c` (tape + all backward kernels),
`src/train.c` (AdamW, recording NCP forward, NCP step/loop),
`src/train_vq.c` (recording VAE forward, VQ step/loop, EMA).

`cncm train` trains the generator (NCP). `cncm train-vq` trains the
tokenizer (VQ-VAE encoder/decoder + residual convs). Codebooks are frozen
under autograd and updated by EMA (same as the paper's scheme).

## Verified claims

- **Forward parity**: the recording forward produces bit-identical outputs
  to the verified inference forward (checked per level on sudoku +
  tinystories, including tied heads and learned-L1 cascades).
- **Backward unit grads**: linear, LayerNorm, RMSNorm, SDPA (with additive
  mask), qkv-split/head-merge, BPE-independent gather/concat/add/scale —
  each checked against central finite differences (`tests/gcheck.c`,
  `tests/ktest_kernels.c`, ad-hoc harnesses; rel. error < 5e-4 typical).
- **End-to-end learning**:
  - NCP: Gaussian noise (σ=0.02) on a trained sudoku generator, then
    `train` — loss falls monotonically (6.64 → 4.39 over 10 steps, B=4).
  - VQ-VAE: `train-vq` from the released tokenizer — total loss
    2.96 → 2.28 over 5 steps, no divergence.
- **Full-model analytic-vs-numeric grad check** (`cncm gradcheck`) matches
  on the forward pass exactly; per-parameter central differences use
  denominator `2*eps` and agree with the tape backward within tolerance
  for gradients above the float32-resolution floor (see below).

## Assumptions (no upstream training code to check against)

1. **Teacher forcing layout**: training input is
   `[z_prefix | l0_queries | idx_to_ncm_input(codes)]`, i.e. the same
   construction the inference path validates. Level-0 targets are the
   ground-truth level-0 codes; level-0 inputs are the learned queries.
2. **Loss = mean over target levels** of per-level cross-entropy
   (each level weighted 1/n_levels). VQ-VAE loss = recon CE +
   mean over levels of 0.25 · MSE(z_hat.detach, z).
3. **Level-0 is unpredictable** on Sudoku-like tasks (verified: teacher-forced
   argmax accuracy ≈ 0% on coarse levels while autoregressive generation
   solves 99.4% — the model leans on its own coarse predictions, not the
   ground truth). It is still supervised (weight 1/n_levels like the rest);
   down-weighting or masking level 0 is a one-line change if desired.
4. **Straight-through**: `z_hat = z + (z_hat - z).detach()` implemented as
   `t_add(z, delta)` with `delta` computed outside the tape; commitment
   gradient flows only into the encoder side, codebook side handled by EMA.
5. **EMA**: exact paper formulas (decay 0.99, ε=1e-5, EMA-cluster smoothing,
   restart logic omitted as in the released configs which set threshold 0).
   **Deviation**: cluster counts init to **ones**, not zeros. Zeros init
   makes the first update divide by ~0.0025 (≈400× codebook explosion) at
   small batch sizes; ones init is stable and converges to the same steady
   state. Use zeros for bit-exact paper parity at paper-scale batches.

## Practical notes

- Reference speed, not production speed. Expect ~10-25 s/step (B=4) for the
  25M-param sudoku model on CPU; larger models need patience (or GPU forward
  + host backward, which works but is still host-bound in backward).
- Data: `tools/export_train.py` builds `.nctrain` shards
  (`[P prefix | total codes]`). For VQ-VAE training, provide raw token-id
  rows of length `vq.context_length` in the same container.
- Checkpoints are versioned `.nckpt` files (format NCK1, see
  ARCHITECTURE.md): rotating `ckpt-stepN.nckpt` every `--ckpt-every`
  completed steps (default 10), `ckpt-final.nckpt` at the end, or a rolling
  `--ckpt-out F`. They hold weights + AdamW moments + counters + RNG +
  run identity, so `--resume R.nckpt` continues the exact trajectory
  (same `--steps` target, same data/conf/tokenizer; `--total` is inherited
  from the checkpoint). `--weights` on a resume run is scaffold only.
  Passing a legacy `.ncm` to `--resume` is a loud fresh-optimizer start,
  never exact. `cncm ckpt-info --weights F` inspects a checkpoint.
  Proven: stopping CNCM does not change the training trajectory under the
  same deterministic conditions (see `tests/test_ckpt.sh`).
- `cncm gradcheck` replays one analytic step and finite-differences a few
  params. Treat large *relative* errors on tiny grads with suspicion (see
  above); the forward-match line (`FORWARD-MATCH`) is the strong signal,
  plus the end-to-end loss curves.
- Non-RoPE models and the per-batch pad path in training forward are wired
  but untested (all released configs use RoPE; sudoku has no pads).

## GPT-2 training (`cncm train-gpt2`)

`src/train_gpt2.c` implements a complete fine-tuning loop on the taped
GPT-2 forward (`train_gpt2_forward`, verified bit-exact vs `gpt2.c` and
grad-vs-numeric in `tests/gcheck.c`): B=1 random-row batches of (T+1)
token ids, causal CE, device backward, global-norm clip 1.0, AdamW with
the same cosine schedule, and NCK1 checkpoints with the same exact-resume
contract as the Qwen loop (`--resume`, same identity rules).

Faithfulness needs two tape ops the NCP/Qwen paths never used, both with
host and device backward and both covered by `bwdops` (bit-identical)
and `gcheck` (analytic-vs-numeric):
- `t_layernorm_eps` — GPT-2 epsilon is 1e-5, not the 1e-6 baked into
  `layernorm_fwd`/`t_layernorm`. Eps travels in the tape record.
- `t_gelu_tanh` — HF gelu_new (tanh approx), not exact-erf `t_gelu`.
  A different function with a different derivative; training with the
  wrong one silently biases every MLP gradient.

`gpt2-taped --prompt TEXT` proves forward parity on real 124M weights
(maxabs=0, CPU and CUDA). `tests/test_train_gpt2.sh` is the training
gate: resume loss-lines identical, resume checkpoint byte-identical,
checkpoint loads for inference (`gpt2-ppl` on the `.nckpt`).

Memory (12 GB card, fp32 124M): weights 0.5 + device grad map ~0.5 +
activations. Same recipe as Qwen: `NCM_ADAMW=cpu` (host moments) and a
sized arena (`NCM_ARENA_MB=1536` at T=16 test, 3072 at T=128).

Query path fix (shared tokenizer): BPE `skip_special` used to blank ids
`[0, n_special)`, which for GPT-2/Qwen vocabs (specials at the END:
50256, 151643+) skipped the entire vocabulary — `gpt2-gen`/`qwen-gen`
printed ids but empty text. Now only the actual eos/pad ids are skipped.

## Math reasoning corpus v1 (`tools/mathdata.py`, `tools/eval_math.py`)

`mathdata.py` generates single-line docs (`mkqwenrows` takes one document
per file line, EOS-joined) with a worked step and a final answer:

    Solve: 47 + 58 = ? / Step: 47 + 58 = 105 / Answer: 105

Ops: add, sub (non-negative), mul, div (exact), two-step chain-add and
mul-add. Splits are disjoint by full expression, audited in-process, with
a manifest (counts + sha256). `results/math-v1/`: 20000/1000/2000 docs,
packed rows for GPT-2 BPE (`train.nctrain`, 3734 rows T=128) and Qwen
(`train_qwen.nctrain`, 6196 rows). `test_prompts.txt`/`test_answers.txt`
feed the query eval.

`eval_math.py` queries held-out prompts via `gpt2-gen` (greedy) and
verifies with a deterministic engine (exact integer arithmetic over the
same expression grammar, plus an expression-identity check so
problem-switching cannot score). A response counts as SOLVED only if the
step is valid AND the answer matches AND the two agree. Reports per-op
accuracy. Status: 200-step smoke (5% of an epoch) learns the format but
solves nothing — the capability run (12000 steps, ~3 epochs) is in
flight; checkpoints + eval numbers land in `results/math-v1/` and
`weights/math-gpt2/`.

## Qwen2.5-0.5B fine-tuning (`cncm train-qwen`)

`src/train_qwen.c` implements a complete fine-tuning loop on the taped
Qwen forward (`train_qwen_forward`, verified bit-exact vs `qwen.c` and
grad-vs-numeric in `tests/gcheck.c`): B=1 random-row batches of (T+1)
token ids, causal CE, device backward, global-norm clip 1.0, AdamW with
the same cosine schedule as NCP training, and NCK1 checkpoints with the
same exact-resume contract (`--resume`, same identity rules).

```
rows.nctrain ([T+1 token ids] per row: input first T, target last T)
        |
  train_qwen_forward (recorded, GQA, tied dot head) + t_ce
        |
  tape_backward (device; NCM_BWD=cpu forces the host path)
        |
  AdamW (decoupled wd, cosine warmup; NCM_ADAMW=cpu keeps moments host-side)
        |
  versioned .nckpt (weights + optimizer + RNG + identity; qwen-ppl and
  qwen-gen load .nckpt files directly via weights_load)
```

Corpus: `tools/mkqwenrows` packs raw text (`--in docs.txt`, one document
per line, Qwen-tokenized, EOS-joined) or id lists (`--ids`, e.g. recycled
tokenizer output — valid as long as every id fits the Qwen vocab) into
`(T+1)`-id rows. `--strip ID` drops a pad id everywhere (fixed-length
pickles pad interior blocks, not just tails); `--min-unique N` drops
low-diversity docs. Lesson learned the hard way: an XSum id dump was
29% pad token with whole single-token rows, which train to loss 0.0 and
tell you nothing — clean the stream, then trust the curve.

Memory (12 GB card, fp32 0.5B): weights 2.0 + device grad map ~2.0 (lives
in the arena) + activations. The device AdamW mirrors (+4 GiB) do not
fit alongside a training-size arena, so Qwen training uses
`NCM_ADAMW=cpu` (host moments; verified bit-identical update math) and a
sized arena: `NCM_ARENA_MB=3072` at T=128, 8192 at T=512 (measured:
forward activations plus per-intermediate grad buffers run ~2.5 GB at
T=128 and ~6.5 GB at T=512, on top of the 2 GB device grad map).

Verified: per-op host-vs-device incl. Qwen-scale LINEAR/SDPA-GQA and a
composed 1-layer Qwen block (`bwdops`, global maxabs ~1e-7); exact resume
(`--steps 2` vs 1+resume: identical losses, byte-identical checkpoints);
checkpoint loads for inference (`qwen-ppl` on the `.nckpt`).
Measured (xsum ids, B=1, T=128, lr 1e-5, warmup 100, 150 steps): mean loss
per 10-step window 10.16, 9.47, 8.95, 8.61, 8.41, 8.36, 8.22, 7.83, 7.72,
7.89, 7.62, 7.70, 7.46, 7.51, 7.28 — steady decrease, no plateau yet.
Resume from the step-150 checkpoint continues in the same band
(7.15, 7.92, 7.42, 8.04, 7.78).

Device-LINEAR dx uses beta=1 (accumulate), not beta=0: the output grad
buffer may already hold contributions from other consumers (residual
adds, split q/k/v heads reading one hidden state), and overwriting them
silently drops real gradient paths on Qwen (NCP's fused qkv never
shares an input this way, which is why only Qwen showed it). Caught by
the composed-block bwdops case: host-vs-device went from maxrel 0.95 to
1e-7 with the one-word fix.

## 16x16 Sudoku (and 25x25-ready) support

Parse, validate, render and evaluate generalize over the grid size
(`src/sudoku_size.h`); 9x9 is the frozen n=9 case and existing
checkpoints/weights are untouched. Size parameters:

| n | box | ncells | prefix_len (P) | text_len | symbols |
|---|-----|--------|----------------|----------|---------|
| 9 | 3 | 81 | 91 | 89 | 1-9 |
| 16 | 4 | 256 | 273 | 271 | 1-9,A-G |
| 25 | 5 | 625 | 651 | 649 | 1-9,A-P |

Token ids generalize too: 0=empty, 1..n=values, n+1=`|` separator,
n+2=wrap (`B`) — for n=9 this is exactly the historical 0..9,10,11
mapping, so 9x9 encode/decode bytes are unchanged. Grid size resolves
as: `--size` flag > `model.sudoku_size` conf key > guess from the
prefix length > 9. Existing 9x9 confs carry no `model.sudoku_size`
and resolve to 9 as before. **Existing 9x9 weights cannot solve
16x16** — new sizes need new model confs and new weights.

New CLI surface: `sudoku-check --board DIGITS [--size 9|16|25]
[--clues CELLS]` (adds optional `clue_ok`; without `--clues` the JSON
is unchanged) and `eval-sudoku --size N --jsonl out.jsonl` (per-puzzle
`{idx,size,exact,valid,clue_ok,cell_frac,ms,board}`; stdout aggregate
unchanged). Tests: `tests/test_sudoku_size.sh` (20 checks, CPU-only
except a bounded n=2 CUDA/CPU parity smoke) plus fixtures under
`tests/fixtures/sudoku{16,25}_*.txt`.

### What remains for 16x16 training (not run yet)

1. Train a 16x16 VQ-VAE tokenizer (context 271, in_vocab 19) and
   freeze a checkpoint.
2. Generate rows: `python3 tools/export_train.py --task sudoku-16
   --n ROWS --seed S --holes 96 --out - > rows16_prefix.txt`
   (stdlib-only dry-run of prefix rows), then VQ-encode to
   `train16.nctrain` (`[273 prefix | codes]` per row).
3. Write `weights/sudoku-16/model.conf` (copy sudoku-easy, set
   `model.sudoku_size=16`, `model.prefix_len=273`,
   `model.context_length=272`, `vq.context_length=271`,
   `vq.in_vocab=19`, fresh `vq.levels` sized for 271).
4. Train on CUDA (heavy CPU training is out of scope for this box):
   `NCM_ADAMW=cpu ./cncm-cuda train --conf weights/sudoku-16/model.conf
   --weights <init16.ncm> --data train16.nctrain --steps S --lr LR`,
   then `nvidia-smi` for the GPU-memory report. Start with a short
   smoke (`--steps 2` + resume-parity), scale up separately.
5. Evaluate bounded: `./cncm-cuda eval-sudoku --conf ... --inputs ...
   --refs ... --n 32 --seed 7 --jsonl eval16.jsonl` — CUDA only.
6. 25x25 needs only data + conf + weights following the same table;
   no code redesign (max buffers already cover 625 cells).
