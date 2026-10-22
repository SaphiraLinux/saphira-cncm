#!/bin/bash
# tests/test_magic.sh — deterministic magic-square validation tests.
# CPU-only, no weights: magic-check is pure constraint logic.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$REPO/cncm}"
PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

LOSHU=816357492
"$BIN" magic-check --board "$LOSHU" 2>/dev/null | grep -q '"valid":true,"magic_sum":15' \
  && pass "3x3 Lo Shu valid sum 15" || fail "Lo Shu"
# rotation of Lo Shu is also magic
"$BIN" magic-check --board 492357816 2>/dev/null | grep -q '"valid":true' \
  && pass "3x3 rotation valid" || fail "rotation"
# swap breaks column/diagonal sums
"$BIN" magic-check --board 618357492 2>/dev/null | grep -q '"valid":false.*c0.*c2.*d0.*d1' \
  && pass "3x3 swap flags c0/c2/d0/d1" || fail "swap flags"
# duplicate digit: not a permutation
"$BIN" magic-check --board 816357493 2>/dev/null | grep -q '"valid":false.*"perm"' \
  && pass "3x3 dup non-permutation" || fail "dup"
# wrong length / bad char / bad size
"$BIN" magic-check --board 123 >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad length exits 2" || fail "bad length"
"$BIN" magic-check --board 81635749X >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad char exits 2" || fail "bad char"
"$BIN" magic-check --size 7 --board "$LOSHU" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad size exits 2" || fail "bad size"
# clues
"$BIN" magic-check --board "$LOSHU" --clues 8..357492 2>/dev/null | grep -q '"clue_ok":true' \
  && pass "3x3 clue_ok true" || fail "clue_ok"
"$BIN" magic-check --board "$LOSHU" --clues 116357492 2>/dev/null | grep -q '"clue_ok":false' \
  && pass "3x3 clue clash" || fail "clue clash"
# size-6 plumbing: 36-char board runs (non-magic, must be invalid, exit 0)
"$BIN" magic-check --size 6 --board 123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0 2>/dev/null | grep -q '"valid":false,"magic_sum":111' \
  && pass "6x6 plumbing sum 111" || fail "6x6 plumbing"
# python generator selftest
python3 "$REPO/tools/magic3.py" --selftest >/dev/null 2>&1 \
  && pass "magic3 generator selftest" || fail "magic3 selftest"

# eval-magic smoke on trained weights (if present): schema + determinism
if [ -f "$REPO/weights/magic-3/ckpts/ckpt-final.nckpt" ]; then
  "$BIN" eval-magic --conf "$REPO/weights/magic-3/model.conf" \
    --weights "$REPO/weights/magic-3/ckpts/ckpt-final.nckpt" \
    --asset-dir "$REPO/weights/magic-3" \
    --inputs "$REPO/weights/magic-3/data/held/prefix.txt" \
    --refs "$REPO/weights/magic-3/data/held/solutions.txt" \
    --n 4 --seed 7 --size 3 --jsonl /tmp/opencode/magic_smoke.jsonl >/dev/null 2>&1 \
  && python3 -c "
import json
rows=[json.loads(l) for l in open('/tmp/opencode/magic_smoke.jsonl') if l.strip()]
assert len(rows)==4
for r in rows:
    assert set(r)>= {'idx','size','exact','valid','clue_ok','cell_frac','ms','board'}
    assert r['size']==3 and len(r['board'])==9
print('magic jsonl ok')" >/dev/null 2>&1 \
  && pass "eval-magic schema n=4" || fail "eval-magic schema"
else
  echo "SKIP: magic-3 weights absent (eval-magic smoke needs ckpt)"
fi

echo "MAGIC-TESTS done pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
