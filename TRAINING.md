# TRAINING.md — how `cncm train` / `cncm train-vq` work

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
  .ncm checkpoint (same format as export; resumes anywhere)
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
  on the forward pass exactly; per-parameter finite differences are
  reported for inspection but are *not* a reliable oracle at this scale
  (float32 rounding dominates ε-perturbations; see below).

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
- Checkpoints are plain `.ncm` files (`ckpt-stepN.ncm`, `ckpt-*.ncm`);
  resume training from any of them with `--weights`.
- `cncm gradcheck` replays one analytic step and finite-differences a few
  params. Treat large *relative* errors on tiny grads with suspicion (see
  above); the forward-match line (`FORWARD-MATCH`) is the strong signal,
  plus the end-to-end loss curves.
- Non-RoPE models and the per-batch pad path in training forward are wired
  but untested (all released configs use RoPE; sudoku has no pads).
