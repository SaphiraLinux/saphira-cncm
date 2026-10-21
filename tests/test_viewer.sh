#!/usr/bin/env bash
# tests/test_viewer.sh — deterministic Sudoku viewer tests.
#
# Covers: golden boards (stepper refactor guard), streamed == gen
# (the stream IS the model), run-to-run determinism, sudoku-check
# valid/invalid + bad-unit flags, prefix validation, and the relay
# (page markers + begin/step-through to done).
#
# Usage: test_viewer.sh --bin ./cncm
set -u
BIN="./cncm"
while [ $# -gt 0 ]; do case "$1" in
  --bin) BIN="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;;
esac done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/$BIN"
NFAIL=0
pass(){ echo "PASS $1"; }
fail(){ echo "FAIL $1"; NFAIL=$((NFAIL + 1)); }

if [ ! -x "$BIN" ]; then echo "SKIP: binary $BIN not built"; exit 0; fi
CONF="$REPO/weights/sudoku-easy/model.conf"
W0="$REPO/weights/sudoku-easy/model.ncm"
ADIR="$REPO/weights/sudoku-easy"
FIX="$REPO/tests/fixtures/sudoku_viewer_golden.txt"
if [ ! -f "$W0" ] || [ ! -f "$CONF" ]; then
  echo "SKIP: sudoku weights not present (git-ignored); nothing to prove against"
  exit 0
fi
if ! command -v python3 >/dev/null 2>&1; then echo "SKIP: no python3 (relay is stdlib python)"; exit 0; fi
if ! command -v curl >/dev/null 2>&1; then echo "SKIP: no curl"; exit 0; fi

WORK="$(mktemp -d /tmp/opencode/viewertest.XXXXXX)"
echo "work=$WORK bin=$BIN"

norm_ms(){ sed -E 's/"ms":[0-9.]+/"ms":0/; s/"elapsed_ms":[0-9.]+/"elapsed_ms":0/' "$1"; }

# ---------- 1. golden boards (stepper == pre-refactor monolithic gen) ----------
GOLD_OK=1
while read -r seed prefix board; do
  [ -z "$seed" ] && continue
  case "$seed" in \#*) continue;; esac
  got="$("$BIN" sudoku-stream --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
    --prefix "$prefix" --seed "$seed" 2>"$WORK/stream.err" | tail -1 | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['board'])")"
  if [ "$got" = "$board" ]; then pass "golden seed=$seed board match"; else
    fail "golden seed=$seed board: got ${got:0:18}... want ${board:0:18}..."; GOLD_OK=0
  fi
done < "$FIX"
[ "$GOLD_OK" = 1 ] || { echo "--- stream stderr:"; tail -3 "$WORK/stream.err"; }

# ---------- 2. streamed board == gen board (genuineness) ----------
seed=7; prefix="$(sed -n '2p' "$FIX" | awk '{print $2}')"
sboard="$("$BIN" sudoku-stream --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "$prefix" --seed "$seed" 2>/dev/null | tail -1 | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['board'])")"
gboard="$("$BIN" gen --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "$prefix" --seed "$seed" 2>/dev/null | tail -1 | tr -d '|')"
if [ "$sboard" = "$gboard" ]; then pass "streamed == gen"; else
  fail "streamed == gen: ${sboard:0:18} vs ${gboard:0:18}"; fi

# ---------- 3. determinism (ms-normalized byte-identical) ----------
"$BIN" sudoku-stream --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "$prefix" --seed "$seed" --out "$WORK/t1.ndjson" >/dev/null 2>&1
"$BIN" sudoku-stream --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "$prefix" --seed "$seed" --out "$WORK/t2.ndjson" >/dev/null 2>&1
norm_ms "$WORK/t1.ndjson" > "$WORK/t1n"; norm_ms "$WORK/t2.ndjson" > "$WORK/t2n"
if cmp -s "$WORK/t1n" "$WORK/t2n"; then pass "determinism (ms-normalized)"; else
  fail "determinism"; diff "$WORK/t1n" "$WORK/t2n" | head -5; fi
nl1="$(wc -l < "$WORK/t1.ndjson")"
[ "$nl1" = 14 ] && pass "trace has header+12 levels+done" || fail "trace line count ($nl1 != 14)"

# ---------- 4. sudoku-check valid / invalid ----------
ok="$("$BIN" sudoku-check --board "$gboard")"
echo "$ok" | grep -q '"valid":true' && pass "check valid board" || fail "check valid: $ok"
bad="${gboard:0:80}8"
badout="$("$BIN" sudoku-check --board "$bad")"
echo "$badout" | grep -q '"valid":false,"bad_rows":\[8\],"bad_cols":\[8\],"bad_boxes":\[8\]' \
  && pass "check invalid flags [8]/[8]/[8]" || fail "check invalid: $badout"

# ---------- 5. prefix validation ----------
"$BIN" sudoku-stream --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "1,2,3" --seed 7 >/dev/null 2>&1
[ $? -eq 2 ] && pass "reject short prefix" || fail "short prefix accepted"
"$BIN" sudoku-check --board "123" >/dev/null 2>&1
[ $? -eq 2 ] && pass "reject short board" || fail "short board accepted"

# ---------- 6. relay: page + full live walk ----------
PORT=8909
python3 "$REPO/tools/viewer/server.py" --cncm "$BIN" --conf "$CONF" \
  --weights "$W0" --asset-dir "$ADIR" --port "$PORT" >"$WORK/relay.log" 2>&1 &
RPID=$!
trap 'kill $RPID 2>/dev/null' EXIT
sleep 1
curl -s -m 5 "http://127.0.0.1:$PORT/" | grep -q 'id="grid"' \
  && pass "relay serves page with grid" || fail "relay page"
curl -s -m 5 "http://127.0.0.1:$PORT/" | grep -q 'LIVE' \
  && pass "page carries LIVE badge" || fail "page LIVE marker"
curl -s -m 5 "http://127.0.0.1:$PORT/" | grep -q 'REPLAY' \
  && pass "page carries REPLAY badge" || fail "page REPLAY marker"
BSID="$(curl -s -m 10 -X POST "http://127.0.0.1:$PORT/api/begin" \
  -d "{\"prefix\": \"$prefix\", \"seed\": $seed}" | \
  python3 -c "import json,sys; print(json.load(sys.stdin).get('sid',''))")"
[ -n "$BSID" ] && pass "relay begin" || fail "relay begin"
WBOARD=""; WVALID=""
for _ in $(seq 1 15); do
  EV="$(curl -s -m 60 "http://127.0.0.1:$PORT/api/step?sid=$BSID")"
  ET="$(echo "$EV" | python3 -c "import json,sys; print(json.load(sys.stdin).get('type','?'))")"
  if [ "$ET" = "done" ]; then
    WBOARD="$(echo "$EV" | python3 -c "import json,sys; print(json.load(sys.stdin)['board'])")"
    WVALID="$(echo "$EV" | python3 -c "import json,sys; print(json.load(sys.stdin)['valid'])")"
    break
  fi
  [ "$ET" = "level" ] || [ "$ET" = "header" ] || { fail "relay walk event ($ET)"; break; }
done
[ "$WBOARD" = "$gboard" ] && [ "$WVALID" = "True" ] \
  && pass "relay walk reaches done (valid board)" \
  || fail "relay walk: board=${WBOARD:0:18} valid=$WVALID"
curl -s -m 10 -X POST "http://127.0.0.1:$PORT/api/end" -d "{\"sid\": \"$BSID\"}" \
  | grep -q '"ok": true' && pass "relay end" || fail "relay end"
kill $RPID 2>/dev/null; trap - EXIT

if [ "$NFAIL" = 0 ]; then echo "ALL VIEWER TESTS PASSED (work=$WORK)"; else
  echo "VIEWER-FAIL ($NFAIL)"; fi
exit $NFAIL
