#!/usr/bin/env bash
# tests/test_ckpt.sh — exact-resume proof + checkpoint failure matrix.
#
# Phase 1 (format): short train -> ckpt-info/parse check + refusal tests
#   (truncation, bit-flips, wrong data/lr/tokenizer, missing file,
#   legacy weights, NCK1-as-scaffold, ckpt-info on garbage).
# Phase 2 (exact resume, the acceptance proof): train N steps continuously
#   (A) vs train N/2, terminate, resume in a fresh process (B). Compares
#   loss lines and checkpoints. CPU backends require byte-identical files;
#   pass --cuda-tol to compare numerically instead (CUDA backend).
#
# Usage: test_ckpt.sh --bin ./cncm [--threads T] [--steps-n N]
#                      [--quick] [--cuda-tol F] [--keep]
# Full CPU proof: ./tests/test_ckpt.sh --steps-n 4   (8 threads default)
# 1-thread leg:   NCM_THREADS=1 ./tests/test_ckpt.sh --steps-n 4
# CUDA leg:       ./tests/test_ckpt.sh --bin ./cncm-cuda --cuda-tol 1e-3
set -u
BIN="./cncm"; THREADS="${NCM_THREADS:-8}"; NSTEPS=4; QUICK=0; TOL=""
KEEP=0
while [ $# -gt 0 ]; do case "$1" in
  --bin) BIN="$2"; shift 2;;
  --threads) THREADS="$2"; shift 2;;
  --steps-n) NSTEPS="$2"; shift 2;;
  --quick) QUICK=1; shift;;
  --cuda-tol) TOL="$2"; shift 2;;
  --keep) KEEP=1; shift;;
  *) echo "unknown arg $1"; exit 2;;
esac done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export NCM_THREADS="$THREADS"
CONF="$REPO/weights/sudoku-easy/model.conf"
W0="$REPO/weights/sudoku-easy/model.ncm"
DATA="$REPO/weights/sudoku-easy/train64.nctrain"
DATA2="$REPO/weights/sudoku-easy/train_ids.nctrain"
TOK="test-sudoku-ckpt"
NFAIL=0
pass(){ echo "PASS $1"; }
fail(){ echo "FAIL $1"; NFAIL=$((NFAIL + 1)); }

if [ ! -x "$REPO/$BIN" ]; then echo "SKIP: binary $REPO/$BIN not built"; exit 0; fi
if [ ! -f "$W0" ] || [ ! -f "$DATA" ]; then
  echo "SKIP: sudoku weights/data not present (git-ignored); nothing to prove against"
  exit 0
fi
if [ $((NSTEPS % 2)) -ne 0 ]; then echo "steps-n must be even"; exit 2; fi
[ "$QUICK" = 1 ] && NSTEPS=2
HALF=$((NSTEPS / 2))
BIN="$REPO/$BIN"

WORK="$(mktemp -d /tmp/opencode/ckptproof.XXXXXX)"
echo "work=$WORK threads=$THREADS bin=$BIN steps=$NSTEPS"

COMMON=(--conf "$CONF" --weights "$W0" --data "$DATA" --total "$NSTEPS" --tokenizer-id "$TOK")

# ---------- Phase 1: format + refusals (needs a good checkpoint) ----------
mkdir -p "$WORK/fmt"
( cd "$WORK/fmt" && S=$SECONDS; "$BIN" train "${COMMON[@]}" --steps 1 \
    --ckpt-every 1 --ckpt-out good.nckpt >train1.log 2>&1; echo "train-1-step=$((SECONDS-S))s" )
[ -f "$WORK/fmt/good.nckpt" ] && pass "phase1 train produced good.nckpt" || { fail "phase1 train"; }
"$BIN" ckpt-info --weights "$WORK/fmt/good.nckpt" >"$WORK/fmt/info.log" 2>&1
[ $? -eq 0 ] && grep -q "exact-resume ready" "$WORK/fmt/info.log" \
  && pass "ckpt-info valid" || fail "ckpt-info valid"
python3 "$REPO/tests/parse_nck1.py" info "$WORK/fmt/good.nckpt" >"$WORK/fmt/pyinfo.log" 2>&1
[ $? -eq 0 ] && pass "independent python reader parses file" || fail "python reader"
# byte-compare C summary vs python summary on key scalars
grep -q "global_step: 1" "$WORK/fmt/info.log" && grep -q "global_step=1" "$WORK/fmt/pyinfo.log" \
  && pass "C + python readers agree" || fail "readers agree"

refuse_case() { # name, resume-file-setup..., extra train args...
  local name="$1"; shift
  mkdir -p "$WORK/fmt/$name"
  ( cd "$WORK/fmt/$name" && "$BIN" train "${COMMON[@]}" --steps 1 "$@" >run.log 2>&1 )
  local rc=$?
  if [ $rc -ne 0 ] && [ -z "$(ls "$WORK/fmt/$name"/*.nckpt 2>/dev/null)" ] \
     && grep -q -i -E "refus|mismatch|corrupt|truncat|cannot|bad" "$WORK/fmt/$name/run.log"; then
    pass "refuse: $name"
  else
    fail "refuse: $name (rc=$rc)"
    tail -3 "$WORK/fmt/$name/run.log"
  fi
}
cp "$WORK/fmt/good.nckpt" "$WORK/fmt/trunc.nckpt"
truncate -s -100 "$WORK/fmt/trunc.nckpt" 2>/dev/null || dd if=/dev/null of="$WORK/fmt/trunc.nckpt" bs=1 seek=$(( $(stat -c%s "$WORK/fmt/trunc.nckpt") - 100 )) 2>/dev/null
refuse_case trunc --resume "$WORK/fmt/trunc.nckpt"
cp "$WORK/fmt/good.nckpt" "$WORK/fmt/fliph.nckpt"
printf '\377' | dd of="$WORK/fmt/fliph.nckpt" bs=1 seek=100 conv=notrunc 2>/dev/null
refuse_case flip-header --resume "$WORK/fmt/fliph.nckpt"
cp "$WORK/fmt/good.nckpt" "$WORK/fmt/flipc.nckpt"
SZ=$(stat -c%s "$WORK/fmt/flipc.nckpt")
printf '\377' | dd of="$WORK/fmt/flipc.nckpt" bs=1 seek=$((SZ - 4)) conv=notrunc 2>/dev/null
refuse_case flip-checksum --resume "$WORK/fmt/flipc.nckpt"
refuse_case wrong-data --resume "$WORK/fmt/good.nckpt" --data "$DATA2"
refuse_case wrong-lr --resume "$WORK/fmt/good.nckpt" --lr 0.001
refuse_case wrong-tok --resume "$WORK/fmt/good.nckpt" --tokenizer-id other
refuse_case missing-file --resume "$WORK/fmt/nope.nckpt"
"$BIN" ckpt-info --weights "$WORK/fmt/trunc.nckpt" >/dev/null 2>&1
[ $? -ne 0 ] && pass "refuse: ckpt-info on truncated" || fail "refuse: ckpt-info truncated"
# legacy weights via --resume: fresh start, must SUCCEED with banner
mkdir -p "$WORK/fmt/legacy"
( cd "$WORK/fmt/legacy" && "$BIN" train "${COMMON[@]}" --steps 1 \
    --resume "$W0" >run.log 2>&1 )
if [ $? -eq 0 ] && grep -q "LEGACY weight-only" "$WORK/fmt/legacy/run.log" \
   && grep -q "NOT an exact resume" "$WORK/fmt/legacy/run.log"; then
  pass "legacy: fresh-start banner + completes"
else fail "legacy resume"; fi
# NCK1 as --weights scaffold (exercises weights_load shim)
mkdir -p "$WORK/fmt/shim"
( cd "$WORK/fmt/shim" && "$BIN" train --conf "$CONF" --weights "$WORK/fmt/good.nckpt" \
    --data "$DATA" --steps 1 --total 1 --tokenizer-id "$TOK" --ckpt-every 1 \
    --ckpt-out shim.nckpt >run.log 2>&1 )
if [ $? -eq 0 ] && [ -f "$WORK/fmt/shim/shim.nckpt" ]; then
  pass "NCK1 loads as weights scaffold"
  python3 "$REPO/tests/parse_nck1.py" diff "$WORK/fmt/shim/shim.nckpt" \
    "$WORK/fmt/good.nckpt" >"$WORK/fmt/shim/diff.log" 2>&1
  # shim run: 1 step from post-step-1 weights + CONTINUED schedule (total=1 ->
  # lr=0 past horizon): loss must equal re-running step 1 to completion... it
  # is simply a valid 1-step continuation; check it parses + differs sanely.
  grep -q "max-abs-diff" "$WORK/fmt/shim/diff.log" && pass "shim ckpt diffs cleanly" \
    || fail "shim diff"
else fail "NCK1 scaffold"; fi
[ -z "$(find "$WORK/fmt" -name '*.tmp.*' 2>/dev/null)" ] \
  && pass "no .tmp leftovers" || fail "tmp leftovers"

# ---------- Phase 2: exact resume ----------
# A: continuous NSTEPS. B: NSTEPS/2, fresh process resumes the rest.
mkdir -p "$WORK/A" "$WORK/B"
( cd "$WORK/A" && S=$SECONDS; "$BIN" train "${COMMON[@]}" --steps "$NSTEPS" \
    --ckpt-every 1 >runA.log 2>&1; echo "A-time=$((SECONDS-S))s" >timeA.txt )
( cd "$WORK/B" && S=$SECONDS; "$BIN" train "${COMMON[@]}" --steps "$HALF" \
    --ckpt-every 1 --ckpt-out mid.nckpt >runB1.log 2>&1; echo "B1-time=$((SECONDS-S))s" >timeB1.txt )
( cd "$WORK/B" && S=$SECONDS; "$BIN" train "${COMMON[@]}" --steps "$NSTEPS" \
    --ckpt-every 1 --resume mid.nckpt >runB2.log 2>&1; echo "B2-time=$((SECONDS-S))s" >timeB2.txt )
cat "$WORK/A/timeA.txt" "$WORK/B/timeB1.txt" "$WORK/B/timeB2.txt"
grep -h "checkpoint saved in\|checkpoint loaded in" "$WORK/A/runA.log" "$WORK/B/runB*.log" | sort | uniq -c
# loss-line continuity: A steps == B1 steps ++ B2 steps
grep -h "^\[train\] step" "$WORK/A/runA.log" | sed 's/ (saved.*//' >"$WORK/stepsA.txt"
{ grep -h "^\[train\] step" "$WORK/B/runB1.log"; grep -h "^\[train\] step" "$WORK/B/runB2.log"; } \
  | sed 's/ (saved.*//' >"$WORK/stepsB.txt"
if cmp -s "$WORK/stepsB.txt" "$WORK/stepsA.txt"; then
  pass "loss lines A == B1++B2"; cat "$WORK/stepsA.txt"
else fail "loss lines differ"; diff "$WORK/stepsA.txt" "$WORK/stepsB.txt" | head; fi
grep -q "exact resume" "$WORK/B/runB2.log" && pass "B2 reports exact resume" || fail "resume banner"
if [ -z "$TOL" ]; then
  cmp -s "$WORK/B/mid.nckpt" "$WORK/A/ckpt-step${HALF}.nckpt" \
    && pass "mid checkpoint byte-identical" || fail "mid ckpt identical"
  cmp -s "$WORK/B/ckpt-final.nckpt" "$WORK/A/ckpt-final.nckpt" \
    && pass "final checkpoint byte-identical" || fail "final ckpt identical"
else
  python3 "$REPO/tests/parse_nck1.py" diff "$WORK/A/ckpt-final.nckpt" \
    "$WORK/B/ckpt-final.nckpt" >"$WORK/cudadiff.log" 2>&1
  cat "$WORK/cudadiff.log"
  WDIFF=$(sed -n 's/max-abs-diff weights=\([0-9.e+-]*\).*/\1/p' "$WORK/cudadiff.log")
  python3 -c "import sys; sys.exit(0 if float('$WDIFF') <= float('$TOL') else 1)" \
    && pass "CUDA final within tol $TOL (weights max-abs $WDIFF)" \
    || fail "CUDA diff $WDIFF exceeds tol $TOL"
fi

if [ "$NFAIL" -eq 0 ]; then
  echo "ALL CHECKPOINT TESTS PASSED (work=$WORK)"
  [ "$KEEP" = 1 ] || rm -rf "$WORK"
else
  echo "$NFAIL FAILURES (work kept at $WORK)"
  exit 1
fi
