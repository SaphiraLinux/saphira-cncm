#!/bin/bash
# tests/test_arith.sh — deterministic number-theory checks (CPU-only, no weights).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$REPO/cncm}"
PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

# 1729: Hardy-Ramanujan, 1^3+12^3 == 9^3+10^3
"$BIN" arith-check --op taxicab --a 9 --b 10 --c 1 --d 12 2>/dev/null \
  | grep -q '"sum":1729.*"distinct":true,"valid":true' \
  && pass "1729 taxicab both ways" || fail "1729"
# same pair twice is not a discovery
"$BIN" arith-check --op taxicab --a 9 --b 10 --c 10 --d 9 2>/dev/null \
  | grep -q '"distinct":false,"valid":false' \
  && pass "taxicab rejects swapped pair" || fail "taxicab distinct"
# cubesum component
"$BIN" arith-check --op cubesum --a 1 --b 12 --n 1729 2>/dev/null \
  | grep -q '"sum":1729.*"valid":true' \
  && pass "1^3+12^3=1729" || fail "cubesum"
"$BIN" arith-check --op cubesum --a 9 --b 10 --n 1728 2>/dev/null \
  | grep -q '"valid":false' \
  && pass "cubesum rejects 1728" || fail "cubesum false"
# 2520 divisible by 1..10; multiples too (2520 not unique); 2521 fails
"$BIN" arith-check --op divisible --n 2520 --by 1,2,3,4,5,6,7,8,9,10 2>/dev/null \
  | grep -q '"valid":true,"failures":\[\]' \
  && pass "2520 divisible 1..10" || fail "2520"
"$BIN" arith-check --op divisible --n 5040 --by 1,2,3,4,5,6,7,8,9,10 2>/dev/null \
  | grep -q '"valid":true' \
  && pass "5040 also divisible (not unique)" || fail "5040"
"$BIN" arith-check --op divisible --n 2521 --by 1,2,3,4,5,6,7,8,9,10 2>/dev/null \
  | grep -q '"valid":false.*"failures":\[2' \
  && pass "2521 fails" || fail "2521"
# bad usage
"$BIN" arith-check --op nope >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad op exits 2" || fail "bad op"
# python discovery data selftest
python3 "$REPO/tools/arith.py" --selftest >/dev/null 2>&1 \
  && pass "arith discovery selftest" || fail "arith selftest"

echo "ARITH-TESTS done pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
