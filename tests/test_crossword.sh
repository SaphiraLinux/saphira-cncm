#!/bin/bash
# tests/test_crossword.sh — deterministic crossword validation tests.
# CPU-only, no weights: crossword-check is pure constraint logic.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$REPO/cncm}"
PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

# mini crossword: CAT / ARE crossing at A
"$BIN" crossword-check --board 'CAT|#R#|#E#' 2>/dev/null | grep -q '"valid":true.*across":\["CAT"\].*"down":\["ARE"\]' \
  && pass "mini crossword valid + slots" || fail "mini valid"
"$BIN" crossword-check --board 'CAT|#R#|#E#' 2>/dev/null | grep -q '"unchecked":\[\[0,0\],\[0,2\],\[1,1\],\[2,1\]\]' \
  && pass "unchecked single-direction cells" || fail "unchecked"
"$BIN" crossword-check --board 'CAT|#R#|#E#' --strict >/dev/null 2>&1
"$BIN" crossword-check --board 'CAT|#R#|#E#' --strict 2>/dev/null | grep -q '"valid":false' \
  && pass "strict rejects unchecked" || fail "strict"
# clues
"$BIN" crossword-check --board 'CAT|#R#|#E#' --clues 'C..|#R#|#E.' 2>/dev/null | grep -q '"clue_ok":true' \
  && pass "clue_ok true" || fail "clue_ok"
"$BIN" crossword-check --board 'CAT|#R#|#E#' --clues 'X..|#R#|#E.' 2>/dev/null | grep -q '"clue_ok":false' \
  && pass "clue clash" || fail "clue clash"
# degenerate: no slots at all
"$BIN" crossword-check --board 'A#B|#C#' 2>/dev/null | grep -q '"valid":false.*"stray":\[\[0,0\],\[0,2\],\[1,1\]\]' \
  && pass "all-singleton invalid + stray" || fail "singletons"
# minlen enforcement with slots still reported
"$BIN" crossword-check --board 'AB|CD' --minlen 3 2>/dev/null | grep -q '"valid":false.*"across":\["AB","CD"\]' \
  && pass "minlen rejects 2-letter slots" || fail "minlen"
"$BIN" crossword-check --board 'AB|CD' 2>/dev/null | grep -q '"valid":true' \
  && pass "2-letter slots valid by default" || fail "default minlen"
# malformed input
"$BIN" crossword-check --board 'ABC|DE' >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "ragged rows exit 2" || fail "ragged rows"
"$BIN" crossword-check --board 'AB|C?' >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad char exits 2" || fail "bad char"
"$BIN" crossword-check --board "$(python3 -c "print('|'.join(['A'*16]*16))")" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "oversize exits 2" || fail "oversize"

echo "CROSSWORD-TESTS done pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
