#!/usr/bin/env bash
# tests/test_viewer.sh — deterministic tests for the native SDL viewer and
# the machine interfaces it builds on (sudoku-stream, sudoku-check).
#
# Covers: golden boards (stepper == verified monolithic gen), streamed ==
# gen output (the stream IS the model), run-to-run determinism,
# sudoku-check valid/invalid + bad-unit flags, prefix validation, font
# strip md5 (pixel-exact bitmap font), and headless SDL smoke runs:
# train --view loss/checkpoint identity vs no-view (the viewer must not
# change numerics), gen --view completing with the same board.
#
# Usage: test_viewer.sh --bin ./cncm
set -u
BIN="./cncm"
while [ $# -gt 0 ]; do case "$1" in
  --bin) BIN="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;;
esac
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/$BIN"
NFAIL=0
pass(){ echo "PASS $1"; }
fail(){ echo "FAIL $1"; NFAIL=$((NFAIL + 1)); }

if [ ! -x "$BIN" ]; then echo "SKIP: binary $BIN not built"; exit 0; fi
CONF="$REPO/weights/sudoku-easy/model.conf"
W0="$REPO/weights/sudoku-easy/model.ncm"
ADIR="$REPO/weights/sudoku-easy"
DATA="$REPO/weights/sudoku-easy/train64.nctrain"
FIX="$REPO/tests/fixtures/sudoku_stream_golden.txt"
if [ ! -f "$W0" ] || [ ! -f "$DATA" ]; then
  echo "SKIP: sudoku weights/data not present (git-ignored); nothing to prove against"
  exit 0
fi

WORK="$(mktemp -d /tmp/opencode/viewertest.XXXXXX)"
echo "work=$WORK bin=$BIN"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

norm_ms(){ sed -E 's/"ms":[0-9.]+/"ms":0/; s/"elapsed_ms":[0-9.]+/"elapsed_ms":0/' "$1"; }

# ---------- 1. golden boards ----------
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

# ---------- 6. font strip md5 (pixel-exact bitmap font) ----------
if [ ! -x "$REPO/tests/fontcheck" ]; then
  fail "fontcheck binary missing (run make tests/fontcheck)"
else
  "$REPO/tests/fontcheck" "$WORK/fontstrip.bmp" || fail "fontcheck run"
  FM="$(md5sum "$WORK/fontstrip.bmp" | awk '{print $1}')"
  FX="$(cat "$REPO/tests/fixtures/viewer_font.md5")"
  [ "$FM" = "$FX" ] && pass "font strip md5 $FM" || fail "font md5: got $FM want $FX"
  # full viewer frames through the live render path (headless dummy video):
  # must show content (white bg + grid + text), never all-black/blank.
  for m in 0 1; do
    SDL_VIDEODRIVER=dummy "$REPO/tests/fontcheck" frame "$WORK/frame$m.bmp" "$m" \
      || fail "frame render mode $m"
  done
  FR="$(python3 - "$WORK/frame0.bmp" "$WORK/frame1.bmp" <<'EOF'
import struct, sys
ok = True
for path in sys.argv[1:]:
    d = open(path, 'rb').read()
    if d[:2] != b'BM':
        print('not-bmp', path); ok = False; continue
    off = struct.unpack('<I', d[10:14])[0]
    w, h = struct.unpack('<ii', d[18:26])
    px = d[off:off + w * h * 4]
    n = w * h
    if len(px) < n * 4 or w != 820 or h != 620:
        print('bad-size', path); ok = False; continue
    dark = sum(1 for i in range(0, len(px), 4)
               if (px[i] + px[i+1] + px[i+2]) // 3 < 40)
    f = dark / n
    print('%s dark=%.3f' % (path.split('/')[-1], f))
    if not (0.01 < f < 0.60):
        print('bad-content', path); ok = False
print('FRAMES-OK' if ok else 'FRAMES-BAD')
EOF
)"
  echo "$FR" | grep -q FRAMES-OK \
    && pass "viewer frames render content (not black/blank)" \
    || { fail "viewer frame content"; echo "$FR" | tail -4; }
fi

# ---------- 7. train --view identity (viewer must not change numerics) ----------
"$BIN" train --conf "$CONF" --weights "$W0" --data "$DATA" \
  --steps 2 --ckpt-every 2 --ckpt-out "$WORK/off.nckpt" >"$WORK/off.log" 2>&1
off_loss="$(grep -o 'loss [0-9.]*' "$WORK/off.log" | tr '\n' ' ')"
"$BIN" train --conf "$CONF" --weights "$W0" --data "$DATA" \
  --steps 2 --ckpt-every 2 --ckpt-out "$WORK/on.nckpt" --view >"$WORK/on.log" 2>&1
[ $? -eq 0 ] && pass "train --view completes headless" || fail "train --view exit"
on_loss="$(grep -o 'loss [0-9.]*' "$WORK/on.log" | tr '\n' ' ')"
[ "$off_loss" = "$on_loss" ] && [ -n "$off_loss" ] \
  && pass "viewer on/off loss identical ($off_loss)" \
  || fail "loss drift: off='$off_loss' on='$on_loss'"
if cmp -s "$WORK/off.nckpt" "$WORK/on.nckpt"; then pass "viewer on/off ckpt identical"; else
  fail "ckpt drift with viewer"; fi

# ---------- 8. gen --view completes with the same board ----------
vboard="$("$BIN" gen --conf "$CONF" --weights "$W0" --asset-dir "$ADIR" \
  --prefix "$prefix" --seed "$seed" --view 2>/dev/null | tail -1 | tr -d '|')"
[ "$vboard" = "$gboard" ] && pass "gen --view same board" \
  || fail "gen --view: ${vboard:0:18} vs ${gboard:0:18}"

if [ "$NFAIL" = 0 ]; then echo "ALL VIEWER TESTS PASSED (work=$WORK)"; else
  echo "VIEWER-FAIL ($NFAIL)"; fi
exit $NFAIL
