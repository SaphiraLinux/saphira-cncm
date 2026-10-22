# results/ — CNCM learning measurements

All files here are regenerated from commands, never hand-edited.
Checkpoints live under `weights/` (git-ignored); these are the
human-readable traces.

## Magic 3×3 (CPU, deterministic, seed 7)

- `magic_held256.jsonl` — per-puzzle eval of the trained magic-3 solver
  on 256 held-out hole patterns (same 8 Lo Shu solutions, unseen givens):
  `./cncm eval-magic --conf weights/magic-3/model.conf \
    --weights weights/magic-3/ckpts/ckpt-final.nckpt \
    --asset-dir weights/magic-3 \
    --inputs weights/magic-3/data/held/prefix.txt \
    --refs weights/magic-3/data/held/solutions.txt \
    --n 256 --seed 7 --size 3 --jsonl results/magic_held256.jsonl`
- `magic_held_eval.svg` — exact 36.72%, valid 36.72%, clues 57.0%,
  mean cell 91.28%. Python cross-check agrees on all 256 validity flags.
- `logs/` — training logs for runs that start after the homer reboot.

## 16×16 Sudoku solver (CUDA)

- `logs/train800_resume.log` — live log of the extended NCP run,
  exact-resumed from `ckpt-step600.nckpt` after the reboot interrupted
  the original 800-step schedule (same flags, `--resume`).
- `ncp16_resume_curve.svg` — loss curve parsed from that log
  (regenerate after completion with `tools/plot_results.py`).
- `ncp16_held512.jsonl` — 512 held-out 16x16 puzzles vs the 800-step
  checkpoint (seed 7, command in same form as the magic eval above
  with the sudoku-16 conf/weights/inputs/refs and `--size 16`).
  Result: **0/512 exact, 0/512 valid, cell 0.074, givens 8.3%**
  (chance 6.25%) — mode collapse persists after 4x training
  (loss 0.55 -> 0.44). `ncp16_held_eval.svg` plots it.

## Controlled replacement: soft sampling at L0-L1 (vq.soft_levels=2, vq.soft_tau=0.5)

Same warm start (vq-final), 300 steps, argmin everywhere else, checkpoints
compatible (no new tensors). Config in `results/vq_soft300.conf`.
`vq_z_to_idx_soft` samples proportionally to softmax(-d/tau) with per-row
adaptive tau; tau<=0 or level>=soft_levels reproduces exact argmin.

- Train-probe L0: 1 -> 15/16 used (entropy 0.00 -> 2.09, dom 1.00 -> 0.65).
  L1 stayed collapsed (2/16) in-argmin — soft sampling alone did not
  populate it within 300 steps.
- Held-out 512: L0-L4 used = 14,2,1,4,11 (was 1,2,1,2,6). L0 winner is
  still code 8 (354/512) but 31% of inputs now route to 13 other codes
  (was 0%). Recon 96.0% -> 96.6% cell match, first valid board (1/64).
- No-surgery 600-step control: coarse usage byte-identical to the
  200-step baseline, isolating the mechanism change as causal.
- `vq16_soft_stages.svg` plots L0-L4 across stages; `vq16_recon_soft*.txt`
  hold the 64-board reconstructions.
- Verdict: PARTIAL revival. Soft assignment genuinely repopulates L0 and
  holds it through continued training, but L1/L2 resist. Next levers:
  longer run, per-level tau, or soft+surgery combined. NCP untouched.

## VQ revival experiment (+400 steps, no surgery)

Question: can CNCM recover its 16x16 representation with more VQ
training, or is the architecture/hyperparameters the problem?

- Control: `weights/sudoku-16/vq-final-step200-control.ncm` (byte copy
  of the VQ before revival). NCP checkpoints untouched.
- Run: warm start from vq-final (fresh optimizer, same lr/schedule
  shape), +400 steps on CUDA (`results/logs/vq_revival400.log`).
  Final weights of that run: `weights/sudoku-16/vq-cont600-nosurgery.ncm`
  (git-ignored; renamed from vq-revived-600.ncm for clarity).
- Telemetry: `train_vq` now prints `[vq-usage]` per-level occupancy,
  entropy (bits) and dominant-code share every 50 steps over a fixed
  512-row probe subset (read-only; no tape/grad/RNG impact).
- Result: loss 0.467 -> 0.425; coarse occupancy **frozen at every
  probe** (L0-L4 used 1,2,1,2,6 of 16, entropy 0.00-2.05, dominant
  shares 0.50-1.00) vs the mature 9x9 baseline of 14-16/16.
  Held-out recon: 96.04% -> 96.58% id match, valid 0/64 -> 0/64.
  Held code usage identical before/after.
  `vq16_occupancy.svg` plots 9x9 vs revived occupancy per level;
  `vq16_recon_baseline64.txt` / `vq16_recon_revived64.txt` are the
  64-board recon outputs.
- Verdict: more steps do not revive dead coarse codebooks under the
  current EMA/commitment setup. The fix must be hyperparameter or
  architectural (EMA schedule, commitment weight, dead-code reset,
  different ladder), not more steps.

## Dead-code surgery + retention experiment

`vq-revive` (new `cncm` command) + `vq_revive_dead` (new `vq.c` API):
encode a 1024-row window, count per-level usage, reseed zero-use codes
with actual encoder outputs at the matching level (distinct frames),
reset EMA counts/accumulators with an exact smoothing write-back so
continued training inherits a consistent state. Deterministic given
seed; writes a NEW weights file, inputs untouched
(`vq-revived.ncm`; control copy verified byte-identical).

- Surgery on the 200-step VQ revived **74 codes** (L0-L4: 15,14,15,14,10).
- Immediately after, held-out usage (512 unseen puzzles):
  L0-L4 = 1,14,7,15,16 of 16 (was 1,2,1,2,6). Revived codes ARE used.
- After +200 continued steps (fresh optimizer, same lr):
  held-out L0-L4 = 1,2,4,10,11. L1 eroded 14->2; L0 never informative
  (same winning code before/after). L2-L4 retain gains.
- Recon (64 held boards): baseline 95.81% cell match, post-surgery
  95.68%, post-continued 96.46%; valid boards 0/64 throughout.
- `vq16_revival_stages.svg` compares L0-L4 across stages;
  `vq16_recon_{baseline64,revived64,cont200}.txt` are the boards.
- Verdict: revival is mechanically possible and revived codes get used,
  but current training dynamics re-collapse L0/L1. Combined with the
  no-surgery control (600 steps: coarse usage byte-identical to 200),
  the evidence now favors quantiser-mechanism work over more steps.
  NCP was deliberately not retrained.

## Provenance note (2026-10-22)

The homer reboot wiped `/tmp`, taking the detailed pre-reboot logs
(VQ/NCP step-by-step curves, first eval JSONLs). What survives:
every checkpoint, every dataset, and the final numbers, all of which
are re-derivable: magic eval above reproduces 94/256 bit-for-bit,
and the 16×16 held-out eval will be re-run once training completes.
Future runs log straight into `results/logs/` (see the resume command).
