#!/usr/bin/env bash
# tools/gate_eval.sh — fail-closed metric gate for staged experiments.
# MISTAKES.md #1 ("no kill criterion") and #8 ("12k steps without a
# decision gate"): a threshold that lives in a script that exits nonzero
# is a stopping rule; a threshold in a notes file is a wish.
#
# Usage:
#   tools/gate_eval.sh --name NAME --actual F --min F [--max F]
# Exits 0 when min <= actual <= max (max optional), 1 otherwise, 2 on
# bad input. Call it at planned fractions of any long run (see
# tools/run_gated.sh); a failing gate must stop the run, never be noted.
set -u
NAME=""; ACTUAL=""; MIN=""; MAX=""
while [ $# -gt 0 ]; do case "$1" in
  --name) NAME="$2"; shift 2;;
  --actual) ACTUAL="$2"; shift 2;;
  --min) MIN="$2"; shift 2;;
  --max) MAX="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac
done
[ -n "$NAME" ] && [ -n "$ACTUAL" ] && [ -n "$MIN" ] || { echo "need --name --actual --min [--max]" >&2; exit 2; }
for v in "$ACTUAL" "$MIN" "$MAX"; do
  [ -z "$v" ] && continue
  case "$v" in *[!0-9.eE+-]*) echo "non-numeric metric: '$v'" >&2; exit 2;; esac
done
OK="$(python3 -c "
a, mn = float('$ACTUAL'), float('$MIN')
mx = float('$MAX') if '$MAX' else None
print('yes' if (a >= mn and (mx is None or a <= mx)) else 'no')")"
if [ "$OK" = "yes" ]; then
  echo "GATE-PASS $NAME: actual=$ACTUAL within [$MIN${MAX:+, $MAX}]"
  exit 0
fi
echo "GATE-FAIL $NAME: actual=$ACTUAL outside [$MIN${MAX:+, $MAX}] — STOP, do not continue" >&2
exit 1
