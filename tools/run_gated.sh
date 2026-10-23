#!/usr/bin/env bash
# tools/run_gated.sh — chunked work with mandatory evaluation gates.
# MISTAKES.md #1 + #8, engineered out: long work runs in chunks; after
# every chunk an eval command produces METRIC=<float> and gate_eval.sh
# decides. A failing gate (or a failing chunk) aborts with nonzero exit
# BEFORE the next chunk starts. There is no "keep watching the curve".
#
# Usage:
#   tools/run_gated.sh --chunks N --work-cmd CMD --eval-cmd CMD \
#       --gate-name NAME --gate-min F [--gate-max F] [--log DIR]
# CMD strings run via bash -c (quote accordingly). eval-cmd must print
# a METRIC=<float> line on stdout. Work stops at the first failure.
#
# Example (exact-resume training, 4 gated chunks of 1000 steps):
#   tools/run_gated.sh --chunks 4 \
#     --work-cmd "tools/run_capped.sh --threads 16 -- ./cncm-cuda train-gpt2 \
#        --conf G.conf --weights G.ncm --data T.nctrain --steps \$NEXT \
#        --resume \$LAST --ckpt-out run/ckpt-next.nckpt" \
#     --eval-cmd "./cncm-cuda gpt2-ppl --conf G.conf \
#        --weights run/ckpt-next.nckpt --texts val.txt | sed 's/.*nll=\\([0-9.]*\\).*/METRIC=\\1/'" \
#     --gate-name val-nll --gate-max 2.0 --log run/gates
# ($NEXT/$LAST are exported per chunk: target step and resume checkpoint.)
set -u
CHUNKS=""; WORK=""; EVALC=""; GNAME=""; GMIN=""; GMAX=""; LOG=""
while [ $# -gt 0 ]; do case "$1" in
  --chunks) CHUNKS="$2"; shift 2;;
  --work-cmd) WORK="$2"; shift 2;;
  --eval-cmd) EVALC="$2"; shift 2;;
  --gate-name) GNAME="$2"; shift 2;;
  --gate-min) GMIN="$2"; shift 2;;
  --gate-max) GMAX="$2"; shift 2;;
  --log) LOG="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac
done
[ -n "$CHUNKS" ] && [ -n "$WORK" ] && [ -n "$EVALC" ] && [ -n "$GNAME" ] && [ -n "$GMIN" ] \
  || { echo "need --chunks --work-cmd --eval-cmd --gate-name --gate-min" >&2; exit 2; }
[ -n "$LOG" ] && mkdir -p "$LOG"
HERE="$(cd "$(dirname "$0")" && pwd)"
GATE="$HERE/gate_eval.sh"
i=0
while [ "$i" -lt "$CHUNKS" ]; do
  i=$((i + 1))
  echo "=== gated chunk $i/$CHUNKS ==="
  export NEXT="$i" LAST="chunk$((i - 1))"
  if ! bash -c "$WORK" ${LOG:+">>$LOG/work.log"} 2>&1; then
    echo "GATED-ABORT chunk $i: work command failed" >&2; exit 1
  fi
  METRIC="$(bash -c "$EVALC" 2>/dev/null | grep -o 'METRIC=[0-9.eE+-]*' | tail -1 | cut -d= -f2)"
  if [ -z "$METRIC" ]; then
    echo "GATED-ABORT chunk $i: eval produced no METRIC=" >&2; exit 1
  fi
  echo "chunk $i metric=$METRIC"
  ARGS=(--name "$GNAME" --actual "$METRIC" --min "$GMIN")
  [ -n "$GMAX" ] && ARGS+=(--max "$GMAX")
  if ! "$GATE" "${ARGS[@]}"; then
    echo "GATED-ABORT chunk $i: gate failed" >&2; exit 1
  fi
done
echo "GATED-DONE all $CHUNKS chunks passed"
exit 0
