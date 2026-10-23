#!/usr/bin/env bash
# tests/test_train_gpt2.sh — functional GPT-2 training gate.
#
# Builds a tiny (T=16) corpus with mkqwenrows (tokenizer-generic: driven by
# the GPT-2 conf + BPE assets), trains 1 step, resumes for a second step,
# and trains 2 steps continuously; asserts the resumed trajectory is
# identical (loss lines + byte-identical checkpoints — the NCK1
# exact-resume acceptance for the GPT-2 loop). Also proves a training
# checkpoint loads for inference (gpt2-ppl on the .nckpt).
#
# Usage: test_train_gpt2.sh --bin ./cncm
set -u
BIN="./cncm"
# Scratch space honors $TMPDIR (persistent) with /tmp fallback.
TMPD="${TMPDIR:-/tmp}"
mkdir -p "$TMPD/opencode"
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
CONF="$REPO/tools/gpt2-124m.conf"
W0="$REPO/weights/gpt2/model.ncm"
ADIR="$REPO/weights/gpt2"
if [ ! -f "$W0" ]; then
  echo "SKIP: GPT-2 weights not present (git-ignored); nothing to prove against"
  exit 0
fi
if [ ! -x "$REPO/tools/mkqwenrows" ]; then
  echo "SKIP: tools/mkqwenrows not built (make tools/mkqwenrows)"; exit 0
fi

WORK="$(mktemp -d $TMPD/opencode/gpt2test.XXXXXX)"
echo "work=$WORK bin=$BIN"
# 124M fp32: weights 0.5GB + device grad map 0.5GB + activations even at T=16.
# GPT-2 training always runs host-side AdamW (mirrors the Qwen loop: device
# mirrors do not fit a training-size arena on 12GB); the device backward
# is still fully exercised.
export NCM_ARENA_MB=1536 NCM_ADAMW=cpu
# 124M checkpoints are ~1.5GB each; need room for two plus scratch.
FREEKB="$(df -k "$WORK" | tail -1 | awk '{print $4}')"
if [ "$FREEKB" -lt 4194304 ]; then
  echo "SKIP: need ~4GB free in $TMPD for GPT-2 checkpoints (have $((FREEKB/1048576))GB)"
  exit 0
fi
cleanup() { [ "$NFAIL" = 0 ] && rm -rf "$WORK" || echo "kept $WORK for inspection"; }
trap cleanup EXIT

# ---------- tiny corpus (text mode exercises the real tokenizer) ----------
cat > "$WORK/docs.txt" <<'EOF'
The quick brown fox jumps over the lazy dog near the river bank.
Pack my box with five dozen liquor jugs for the field trip tomorrow.
Bright vixens jump while dozy fowl quack under the hazy morning sun.
The five boxing wizards jumped quickly over the sleeping stones.
EOF
"$REPO/tools/mkqwenrows" --conf "$CONF" --asset-dir "$ADIR" \
  --in "$WORK/docs.txt" --out "$WORK/tiny.nctrain" --tokens 17 \
  >"$WORK/mk.log" 2>&1 || { fail "mk rows (GPT-2 BPE)"; tail -3 "$WORK/mk.log"; }
TROWS="$(python3 -c "import struct; d=open('$WORK/tiny.nctrain','rb').read(); print(struct.unpack('<Q',d[4:12])[0])")"
[ "$TROWS" -ge 2 ] && pass "tiny corpus ($TROWS rows, T=16)" || fail "tiny corpus rows=$TROWS"

# ---------- 1 step, resume, vs continuous 2 steps ----------
"$BIN" train-gpt2 --conf "$CONF" --weights "$W0" --data "$WORK/tiny.nctrain" \
  --steps 1 --lr 1e-5 --total 100 --ckpt-every 1 --ckpt-out "$WORK/a.nckpt" \
  >"$WORK/a.log" 2>&1 || { fail "train step 0"; tail -3 "$WORK/a.log"; }
L0A="$(grep -o 'step 0 loss [0-9.]*' "$WORK/a.log" | awk '{print $4}')"
"$BIN" train-gpt2 --conf "$CONF" --weights "$W0" --data "$WORK/tiny.nctrain" \
  --steps 2 --lr 1e-5 --total 100 --ckpt-every 1 --ckpt-out "$WORK/z.nckpt" \
  --resume "$WORK/a.nckpt" \
  >"$WORK/b.log" 2>&1 || { fail "train resume"; tail -3 "$WORK/b.log"; }
L1B="$(grep -o 'step 1 loss [0-9.]*' "$WORK/b.log" | awk '{print $4}')"
rm -f "$WORK/a.nckpt"
"$BIN" train-gpt2 --conf "$CONF" --weights "$W0" --data "$WORK/tiny.nctrain" \
  --steps 2 --lr 1e-5 --total 100 --ckpt-every 2 --ckpt-out "$WORK/c.nckpt" \
  >"$WORK/c.log" 2>&1 || { fail "train continuous"; tail -3 "$WORK/c.log"; }
L0C="$(grep -o 'step 0 loss [0-9.]*' "$WORK/c.log" | awk '{print $4}')"
L1C="$(grep -o 'step 1 loss [0-9.]*' "$WORK/c.log" | awk '{print $4}')"
[ -n "$L0A" ] && [ "$L0A" = "$L0C" ] && [ -n "$L1B" ] && [ "$L1B" = "$L1C" ] \
  && pass "resume loss lines identical ($L0A / $L1B)" \
  || fail "resume drift: A=$L0A B=$L1B C=$L0C/$L1C"
if cmp -s "$WORK/z.nckpt" "$WORK/c.nckpt"; then pass "resume ckpt byte-identical"
else fail "resume ckpt differs"; fi

# ---------- checkpoint loads for inference ----------
PPL="$("$BIN" gpt2-ppl --conf "$CONF" --weights "$WORK/c.nckpt" \
  --asset-dir "$ADIR" --texts "$WORK/docs.txt" 2>/dev/null | tail -1)"
echo "$PPL" | grep -q 'ppl=' && pass "ckpt inference ($PPL)" || fail "ckpt ppl: $PPL"
PB="$("$BIN" gpt2-ppl --conf "$CONF" --weights "$W0" \
  --asset-dir "$ADIR" --texts "$WORK/docs.txt" 2>/dev/null | tail -1)"
[ "$PPL" != "$PB" ] && pass "trained ppl differs from base ($PB -> $PPL)" \
  || fail "ppl identical to base (optimizer idle?)"
rm -f "$WORK/z.nckpt" "$WORK/c.nckpt"

if [ "$NFAIL" = 0 ]; then echo "ALL TRAIN-GPT2 TESTS PASSED (work=$WORK)"; else
  echo "TRAIN-GPT2-FAIL ($NFAIL)"; fi
exit $NFAIL
