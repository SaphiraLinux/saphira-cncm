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

[ "$NFAIL" -eq 0 ] && echo "ALL UNIT TESTS PASSED" || { echo "$NFAIL FAILURES"; exit 1; }
