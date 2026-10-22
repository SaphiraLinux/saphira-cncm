#!/bin/bash
# tests/test_sudoku_size.sh — deterministic Sudoku size tests (9/16/25).
# CPU-only except the final bounded CUDA/CPU parity smoke (n<=4, skipped
# gracefully when no GPU is present). No training, no benchmarks.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$REPO/cncm}"
CBIN="${CBIN:-$REPO/cncm-cuda}"
FIX="$REPO/tests/fixtures"
PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

# ---------- 1. python 16x16 generator selftest (stdlib only) ----------
python3 "$REPO/tools/sudoku16.py" --selftest >/dev/null 2>&1 \
  && pass "sudoku16 selftest" || fail "sudoku16 selftest"

# ---------- 2. export dry-run: 2 rows, P=273, wrap framing ----------
python3 "$REPO/tools/export_train.py" --task sudoku-16 --n 2 --seed 16 \
  --out - > /tmp/opencode/size_rows.txt 2>/dev/null \
  && python3 -c "
rows=[l.strip() for l in open('/tmp/opencode/size_rows.txt') if l.strip()]
assert len(rows)==2, rows
for r in rows:
    ids=list(map(int,r.split(',')))
    assert len(ids)==273 and ids[0]==18 and ids[-1]==18, len(ids)
print('rows ok')" >/dev/null 2>&1 \
  && pass "export dry-run P=273" || fail "export dry-run P=273"

# ---------- 3. 9x9 sudoku-check behavior lock ----------
VALID9="921647385764835291583912764452369178179458623836721549217584936698173452345296817"
"$BIN" sudoku-check --board "$VALID9" 2>/dev/null | grep -q '"valid":true' \
  && pass "9x9 valid" || fail "9x9 valid"
BAD9="121647385764835291583912764452369178179458623836721549217584936698173452345296817"
"$BIN" sudoku-check --board "$BAD9" 2>/dev/null | grep -q '"valid":false,"bad_rows":\[0\]' \
  && pass "9x9 invalid flags row 0" || fail "9x9 invalid: $($BIN sudoku-check --board "$BAD9" 2>/dev/null)"
"$BIN" sudoku-check --board "123" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "9x9 bad length exits 2" || fail "9x9 bad length"
"$BIN" sudoku-check --board "${VALID9%?}X" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "9x9 bad char exits 2" || fail "9x9 bad char"
CLUE9=".................................................................................3......8......21.17......7...........18.......7..9..5..6......4..2..1..69..5.."
# (81 dots + a real clue set is built below from the valid board: first row given)
CLUE9="$(echo "$VALID9" | cut -c1-9)........................................................................"
"$BIN" sudoku-check --board "$VALID9" --clues "$CLUE9" 2>/dev/null | grep -q '"clue_ok":true' \
  && pass "9x9 clue_ok true" || fail "9x9 clue_ok"
CLASH9="2$(echo "$VALID9" | cut -c2-81)"
"$BIN" sudoku-check --board "$VALID9" --clues "$CLASH9" 2>/dev/null | grep -q '"clue_ok":false' \
  && pass "9x9 clue clash detected" || fail "9x9 clue clash"

# ---------- 4. 16x16 sudoku-check ----------
B16="$(cat "$FIX/sudoku16_valid.txt")"
"$BIN" sudoku-check --size 16 --board "$B16" 2>/dev/null | grep -q '"valid":true' \
  && pass "16x16 valid" || fail "16x16 valid"
I16="$(cat "$FIX/sudoku16_invalid_row.txt")"
"$BIN" sudoku-check --size 16 --board "$I16" 2>/dev/null | grep -q '"valid":false,"bad_rows":\[0\]' \
  && pass "16x16 invalid flags row 0" || fail "16x16 invalid: $($BIN sudoku-check --size 16 --board "$I16" 2>/dev/null)"
"$BIN" sudoku-check --size 16 --board "${B16%?}H" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "16x16 bad char H exits 2" || fail "16x16 bad char"
"$BIN" sudoku-check --board "$B16" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "16 board under --size 9 rejected" || fail "16 under size 9"
C16="$(cat "$FIX/sudoku16_clues.txt")"
"$BIN" sudoku-check --size 16 --board "$B16" --clues "$C16" 2>/dev/null | grep -q '"clue_ok":true' \
  && pass "16x16 clue_ok true" || fail "16x16 clue_ok"
# clash: flip first given clue to a different valid symbol
CLASH16="$(python3 -c "
b=open('$FIX/sudoku16_valid.txt').read().strip(); c=open('$FIX/sudoku16_clues.txt').read().strip()
i=c.index(next(ch for ch in c if ch!='.'))
alt='2' if b[i]!='2' else '3'
print(c[:i]+alt+c[i+1:])")"
"$BIN" sudoku-check --size 16 --board "$B16" --clues "$CLASH16" 2>/dev/null | grep -q '"clue_ok":false' \
  && pass "16x16 clue clash detected" || fail "16x16 clue clash"
"$BIN" sudoku-check --size 7 --board "$VALID9" >/dev/null 2>&1; [ $? -eq 2 ] \
  && pass "bad --size exits 2" || fail "bad --size"

# ---------- 5. 25x25 sudoku-check ----------
B25="$(cat "$FIX/sudoku25_valid.txt")"
"$BIN" sudoku-check --size 25 --board "$B25" 2>/dev/null | grep -q '"valid":true' \
  && pass "25x25 valid" || fail "25x25 valid"
I25="$(cat "$FIX/sudoku25_invalid_row.txt")"
"$BIN" sudoku-check --size 25 --board "$I25" 2>/dev/null | grep -q '"valid":false,"bad_rows":\[0\]' \
  && pass "25x25 invalid flags row 0" || fail "25x25 invalid"

# ---------- 6. 9x9 end-to-end regression (existing weights, short) ----------
if [ -f "$REPO/weights/sudoku-easy/model.ncm" ]; then
  PRE="$(head -1 "$REPO/weights/sudoku-easy/inputs.txt")"
  GOLDEN9="921647385764835291583912764452369178179458623836721549217584936698173452345296817"
  OUT="$("$BIN" gen --conf "$REPO/weights/sudoku-easy/model.conf" \
    --weights "$REPO/weights/sudoku-easy/model.ncm" \
    --asset-dir "$REPO/weights/sudoku-easy" --prefix "$PRE" --seed 7 2>/dev/null)"
  echo "$OUT" | tr -d '|\n' | grep -q "$GOLDEN9" \
    && pass "9x9 gen still solves ref" || fail "9x9 gen solves ref"
  # JSONL smoke on CPU (n=2), schema check
  "$BIN" eval-sudoku --conf "$REPO/weights/sudoku-easy/model.conf" \
    --weights "$REPO/weights/sudoku-easy/model.ncm" \
    --asset-dir "$REPO/weights/sudoku-easy" \
    --inputs "$REPO/weights/sudoku-easy/inputs.txt" \
    --refs "$REPO/weights/sudoku-easy/refs.txt" --n 2 --seed 7 \
    --jsonl /tmp/opencode/size_cpu.jsonl >/dev/null 2>&1 \
    && python3 -c "
import json
rows=[json.loads(l) for l in open('/tmp/opencode/size_cpu.jsonl') if l.strip()]
assert len(rows)==2, len(rows)
for r in rows:
    assert set(r)>= {'idx','size','exact','valid','clue_ok','cell_frac','ms','board'}, r.keys()
    assert r['size']==9 and len(r['board'])==81
print('jsonl ok')" >/dev/null 2>&1 \
    && pass "9x9 eval JSONL schema (cpu)" || fail "9x9 eval JSONL"
else
  echo "SKIP: 9x9 weights absent (sections 6-7 need model.ncm)"
fi

# ---------- 7. bounded CUDA/CPU parity smoke (n<=4) ----------
if [ -x "$CBIN" ] && nvidia-smi -L >/dev/null 2>&1 && [ -f "$REPO/weights/sudoku-easy/model.ncm" ]; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null | head -3 || true
  "$CBIN" eval-sudoku --conf "$REPO/weights/sudoku-easy/model.conf" \
    --weights "$REPO/weights/sudoku-easy/model.ncm" \
    --asset-dir "$REPO/weights/sudoku-easy" \
    --inputs "$REPO/weights/sudoku-easy/inputs.txt" \
    --refs "$REPO/weights/sudoku-easy/refs.txt" --n 2 --seed 7 \
    --jsonl /tmp/opencode/size_cuda.jsonl >/dev/null 2>&1 \
    && python3 -c "
import json
c=[json.loads(l)['board'] for l in open('/tmp/opencode/size_cpu.jsonl') if l.strip()]
g=[json.loads(l)['board'] for l in open('/tmp/opencode/size_cuda.jsonl') if l.strip()]
assert c==g, (c,g)
print('parity ok')" >/dev/null 2>&1 \
    && pass "cuda/cpu parity n=2" || fail "cuda/cpu parity"
else
  echo "SKIP: cuda parity (no GPU or no cuda binary)"
fi

echo "SIZE-TESTS done pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
