#!/usr/bin/env bash
# tests/test_units.sh — weight-free unit tests: gradcheck kernels + NCK1 format.
# Always runnable from a clean checkout (synthetic data only, no weights).
set -u
# no weights needed; any args are accepted and ignored
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
# Scratch space honors $TMPDIR (persistent) with /tmp fallback.
TMPD="${TMPDIR:-/tmp}"
mkdir -p "$TMPD/opencode"
NFAIL=0
pass(){ echo "PASS $1"; }
fail(){ echo "FAIL $1"; NFAIL=$((NFAIL + 1)); }

make cncm >/dev/null 2>&1 || { echo "FAIL build cncm"; exit 1; }
make tests/gcheck tests/ckpt_unit >/dev/null 2>&1 || { echo "FAIL build test binaries"; exit 1; }

./tests/gcheck > $TMPD/opencode/gcheck.log 2>&1
[ $? -eq 0 ] && pass "gcheck (tape grads vs numeric)" || { fail "gcheck"; tail -5 $TMPD/opencode/gcheck.log; }
./tests/ckpt_unit > $TMPD/opencode/ckpt_unit.log 2>&1
[ $? -eq 0 ] && pass "ckpt_unit (NCK1 round-trip + refusals)" || { fail "ckpt_unit"; tail -8 $TMPD/opencode/ckpt_unit.log; }

# ---------- MISTAKES.md enforcement gates (all weight-free, CPU-only) --
# #4 harness parsers: every historical mis-score must stay mis-scored-never
python3 "$REPO/tools/eval_math.py" --self-test > $TMPD/opencode/self_math.log 2>&1
[ $? -eq 0 ] && pass "eval_math --self-test (harness regression gate)" \
  || { fail "eval_math self-test"; tail -5 $TMPD/opencode/self_math.log; }
python3 "$REPO/tools/eval_sudoku_text.py" --self-test > $TMPD/opencode/self_sud.log 2>&1
[ $? -eq 0 ] && pass "eval_sudoku --self-test (board-extraction gate)" \
  || { fail "eval_sudoku self-test"; tail -5 $TMPD/opencode/self_sud.log; }
# #5 dataset balance: tiny generation must pass the fail-closed audit
python3 "$REPO/tools/mathdata.py" --out $TMPD/opencode/mathtiny \
  --train 600 --val 60 --test 120 --seed 99 > $TMPD/opencode/mathdata.log 2>&1
[ $? -eq 0 ] && pass "mathdata tiny (disjoint + balance audit)" \
  || { fail "mathdata tiny"; tail -5 $TMPD/opencode/mathdata.log; }
# #1/#8 gates: pass / fail-below / fail-above / bad-input exit codes
"$REPO/tools/gate_eval.sh" --name t --actual 8 --min 6 --max 100 >/dev/null 2>&1
[ $? -eq 0 ] || fail "gate_eval pass-case"
"$REPO/tools/gate_eval.sh" --name t --actual 5 --min 6 >/dev/null 2>&1
[ $? -eq 1 ] || fail "gate_eval fail-min exit code"
"$REPO/tools/gate_eval.sh" --name t --actual 8 --min 6 --max 7 >/dev/null 2>&1
[ $? -eq 1 ] || fail "gate_eval fail-max exit code"
"$REPO/tools/gate_eval.sh" --name t --actual x --min 6 >/dev/null 2>&1
[ $? -eq 2 ] || fail "gate_eval bad-input exit code"
pass "gate_eval exit-code contract (1 fail-closed gate)"
# #1/#8 orchestration: all-pass runs all chunks; mid-fail stops early
rm -rf $TMPD/opencode/gt $TMPD/opencode/gt2; mkdir -p $TMPD/opencode/gt $TMPD/opencode/gt2
"$REPO/tools/run_gated.sh" --chunks 3 \
  --work-cmd "echo did >> $TMPD/opencode/gt/work.log" \
  --eval-cmd "echo METRIC=8" --gate-name m --gate-min 6 \
  --log $TMPD/opencode/gt >/dev/null 2>&1
RC1=$?
[ "$RC1" -eq 0 ] && [ "$(wc -l < $TMPD/opencode/gt/work.log)" -eq 3 ] \
  && pass "run_gated all-pass (3 chunks)" || fail "run_gated all-pass rc=$RC1"
"$REPO/tools/run_gated.sh" --chunks 3 \
  --work-cmd "echo did >> $TMPD/opencode/gt2/work.log" \
  --eval-cmd "N=\$(wc -l < $TMPD/opencode/gt2/work.log); if [ \$N -ge 2 ]; then echo METRIC=2; else echo METRIC=8; fi" \
  --gate-name m --gate-min 6 --log $TMPD/opencode/gt2 >/dev/null 2>&1
RC2=$?
[ "$RC2" -ne 0 ] && [ "$(wc -l < $TMPD/opencode/gt2/work.log)" -eq 2 ] \
  && pass "run_gated mid-fail aborts (chunk 3 never runs)" || fail "run_gated mid-fail rc=$RC2"
# #2 thread ceiling: the wrapper must refuse >28
"$REPO/tools/run_capped.sh" --threads 29 -- /bin/true >/dev/null 2>&1
[ $? -eq 2 ] && pass "run_capped refuses --threads 29 (global ceiling)" \
  || fail "run_capped accepted 29 threads"

[ "$NFAIL" -eq 0 ] && echo "ALL UNIT TESTS PASSED" || { echo "$NFAIL FAILURES"; exit 1; }
