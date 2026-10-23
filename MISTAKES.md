# Mistakes and what they taught us

Session record: VQ revival endgame, GPT-2 trainer build, math pipeline v1,
and the OOM incident of 2026-10-23. Written so the next session does not
repay this tuition.

## 1. No kill criterion on the VQ revival work

**Mistake.** Ran surgery, no-surgery control, soft sampling, then extended
soft sampling to 1000 steps — each run justified by marginal movement
(L0 1→16/16) while L1/L2 never moved. No pre-declared bar for "this
approach is dead", so every partial result bought another run.

**Lesson.** Any rescue line gets a written kill criterion before the run
starts: metric, threshold, step budget. The no-surgery 600-step control
(byte-identical collapse) was the most informative run of the whole line
— controls first, not after.

## 2. Ignored the 28-thread ceiling; overlapped uncapped jobs

**Mistake.** Launched two 100-prompt CPU evals plus probe loops, each
process using all 32 threads via OpenMP, on a shared box that also hosts
VMs and a distro build. Load hit 164 against a stated ceiling of 28.

**Lesson.** The cap is global across all jobs, not per process. One
workload at a time; caps set in a wrapper (`tools/run_capped.sh`), not
in good intentions; actual thread counts verified through `/proc`
(we measured 5, not "whatever OMP says"). Never trust the env var —
sample the process.

## 3. Blamed the OOM on CPU load without evidence

**Mistake.** First diagnosis was "load killed the trainer". Wrong axis:
CPU oversubscription does not explain 61 GiB of resident anonymous
memory, and saying so delayed the real audit.

**Lesson.** Size the anomaly first (61 GiB anon ÷ ~10867 steps ≈
5.6 MiB/step), then go read code with that number in hand. The audit
found `t_ce` orphaning a 25 MiB host buffer per step plus aux-member
leaks — arithmetic that fits RAM+swap exhaustion and the freeze
reports. Diagnose from measurements, never from the most recent
embarrassment.

## 4. The eval harness lied three times before it measured anything

- `test_prompts.txt` was built with `doc.split(" / ")[0]`, truncating
  every division/chain prompt at the first slash. The model then
  "solved" self-invented problems — garbage scores, confidently reported.
- `gpt2-gen` prints an `ids:` diagnostics line; the digit characters
  polluted the answer parser (`.` doesn't cross `\n`, ids do).
- Generation output excludes the prompt, so the `Step:` anchor never
  matched until prompt+generation were concatenated.

**Lesson.** A harness that reports 0% can be as broken as one that
reports 100%. Validate the harness on a known-good case first (we now
keep the base-model 0% + format checks), and eyeball raw generations
before trusting any aggregate. Every one of these was caught by
looking at actual outputs, none by staring at the score.

## 5. The dataset generator silently biased every split

Round-robin op assignment with a shared counter locked each split to 2
of 6 generators (test had no add/sub at all); expression keys that
omitted varying args starved muladd. Both passed silently because the
counts looked right at a glance (20k/1k/2k) while the mix was wrong.

**Lesson.** Always print the per-split × per-class contingency table
after generation and assert balance in code, not by eye. Counts lie;
cross-tabs don't.

## 6. The query path was silently broken for months

`tok_decode`'s `skip_special` blanked ids `[0, n_special)` — correct
for BERT, catastrophic for GPT-2/Qwen (specials at end of vocab), so
every generation printed ids with empty text. Nothing ever checked the
text leg; the id-level verify tests all passed.

**Lesson.** End-to-end smoke tests over the exact user-facing path
(prompt → text out), not just layer-level checks. IDs-exact is not
working software.

## 7. Moved checkpoint files out from under a running eval

Renamed `ckpt-gpt2-step7000.nckpt` mid-eval; every subsequent generation
failed to load weights and the "full" eval reported 2% parsed. Reran
from the stable path.

**Lesson.** Artifacts under measurement are read-only until the
measurement finishes. Move files only when no job holds the path.

## 8. Ran 12k steps on a hunch without interim gates

The capability run had no planned decision points; step-1000/2000 evals
were improvised after the fact, and the loss plateau (~0.9 from step
700 on) was watched rather than acted on. Final: 8/100 — real, but a
planned gate at 2k ("solved > 0% or stop/rethink LR") would have forced
the LR/schedule question 8k steps earlier.

**Lesson.** Long runs get gates at fixed fractions with pre-declared
actions (continue / adjust / kill). Watching a curve is not managing it.

## 9. What worked and should be kept

- **Exact-resume discipline (NCK1 identity).** The incident recovery —
  verify checksum, resume 10000→12000 bit-identically — worked first
  try because identity rules were enforced from day one. This is the
  practice that saved the run.
- **Controls that isolate causation.** No-surgery VQ control,
  base-vs-trained evals on identical prompts, loss-line resume parity.
- **Engine-verified scoring.** SOLVED = valid step + right answer +
  consistent, checked by deterministic code. Survived every harness bug
  because the definition is mechanical.
- **Committing failure evidence.** Dead-end checkpoints, full logs,
  negative results — all in-repo. The OOM post-mortem was possible
  because the logs existed.
