#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# io_queue_depth sweep on the non-empty (300-query) workload to tell whether
# io_wait is disk-BANDWIDTH saturated (more in-flight IO won't help -> qps flat)
# or LATENCY bound (deeper queue hides latency -> qps rises with depth).
# O_DIRECT + io_uring; depth is per-IO-ring concurrency. Restores config on exit.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores DJ_NONEMPTY_ONLY=1
median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "io_queue_depth  qps_median  search_time_median | all_qps"
for d in 8 16 24 32 48 64 96; do
  TMP=$(mktemp /tmp/iod_XXXX.config)
  sed -e 's/^approx_alpha .*/approx_alpha 1.0/' -e "s/^io_queue_depth .*/io_queue_depth ${d}/" "$CONFIG" > "$TMP"
  qs=(); sts=(); ok=1
  for r in $(seq 1 5); do
    if out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null); then
      qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
      sts+=("$(echo "$out" | sed -n 's/^search_time = //p')")
    else ok=0; break; fi
  done
  if [ "$ok" = 1 ] && [ "${#qs[@]}" -gt 0 ]; then
    printf '%-15s %-11s %-19s | %s\n' "$d" "$(median "${qs[@]}")" "$(median "${sts[@]}")" "${qs[*]}"
  else
    printf '%-15s FAIL (budget/IO error)\n' "$d"
  fi
  rm -f "$TMP"
done
