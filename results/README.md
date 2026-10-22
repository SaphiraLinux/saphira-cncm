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

## Provenance note (2026-10-22)

The homer reboot wiped `/tmp`, taking the detailed pre-reboot logs
(VQ/NCP step-by-step curves, first eval JSONLs). What survives:
every checkpoint, every dataset, and the final numbers, all of which
are re-derivable: magic eval above reproduces 94/256 bit-for-bit,
and the 16×16 held-out eval will be re-run once training completes.
Future runs log straight into `results/logs/` (see the resume command).
