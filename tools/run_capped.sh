#!/usr/bin/env bash
# tools/run_capped.sh — run ONE CNCM job under global caps.
#
# Incident rule (2026-10-23 OOM): at most 28 runnable CPU threads across
# ALL CNCM jobs combined; one workload at a time; hard memory ceiling so
# a runaway can never OOM homer again (the t_ce host leak that ate
# 61 GiB RSS + ~200 GiB swap before the kernel killed train-gpt2).
#
# Usage:
#   tools/run_capped.sh --threads 16 --rss-gb 12 --vsize-gb 24 \
#       --monitor DIR -- <command...>
#
#   --threads N   OMP/NCM thread cap for the job (<=28, default 16).
#   --rss-gb G    SIGTERM the job if its VmRSS exceeds G (default 12).
#                 Legit GPT-2-124M host footprint is <6 GiB; Qwen-0.5B
#                 host moments ~3 GiB + arena. Anything near the cap is
#                 a leak, not work.
#   --vsize-gb G  ulimit -v backstop (default 24). Counts all process VM.
#   --monitor DIR append RSS/threads/load samples to DIR/monitor.log.
#   --                everything after is the job (runs in foreground).
set -u
THREADS=16; RSSGB=12; VSIZEGB=24; MON=""
while [ $# -gt 0 ]; do case "$1" in
  --threads) THREADS="$2"; shift 2;;
  --rss-gb) RSSGB="$2"; shift 2;;
  --vsize-gb) VSIZEGB="$2"; shift 2;;
  --monitor) MON="$2"; shift 2;;
  --) shift; break;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac
done
[ $# -ge 1 ] || { echo "no command" >&2; exit 2; }
[ "$THREADS" -le 28 ] || { echo "threads $THREADS exceeds global 28 cap" >&2; exit 2; }
if [ -n "$MON" ]; then mkdir -p "$MON"; fi

export NCM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export OMP_THREAD_LIMIT="$THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_PROC_BIND=spread OMP_PLACES=threads

ulimit -v $((VSIZEGB * 1024 * 1024)) || exit 2

"$@" &
JOB=$!
poll() {
  while kill -0 "$JOB" 2>/dev/null; do
    if [ -r "/proc/$JOB/status" ]; then
      RSSKB="$(awk '/VmRSS:/{print $2}' /proc/$JOB/status)"
      NTHR="$(awk '/Threads:/{print $2}' /proc/$JOB/status)"
      LOAD="$(cut -d' ' -f1 /proc/loadavg)"
      echo "$(date -u +%H:%M:%S) rss_kb=${RSSKB:-?} threads=${NTHR:-?} load1=${LOAD} pid=$JOB"
      if [ -n "${RSSKB:-}" ] && [ "$RSSKB" -gt $((RSSGB * 1024 * 1024)) ]; then
        echo "RSS CAP EXCEEDED (${RSSKB} kB > ${RSSGB} GiB): SIGTERM $JOB" >&2
        kill -TERM "$JOB" 2>/dev/null
        return 0
      fi
    fi
    sleep 5
  done
}
if [ -n "$MON" ]; then poll >>"$MON/monitor.log" 2>&1 & POLL=$!; else poll >/dev/null 2>&1 & POLL=$!; fi
wait "$JOB"; RC=$?
kill "$POLL" 2>/dev/null
wait 2>/dev/null
echo "run_capped: exit=$RC"
exit $RC
