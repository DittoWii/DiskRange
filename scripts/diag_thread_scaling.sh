#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Thread-scaling diagnostic for the non-empty (300-query) workload.
# If fewer threads give similar/better wall time, the 24-thread run is
# parallel-starved (too few work-units for the slow-path loop).

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1 DJ_NONEMPTY_ONLY=1

TMP=$(mktemp /tmp/thr_XXXX.config); trap 'rm -f "$TMP"' EXIT
sed "s/^approx_alpha .*/approx_alpha 1.0/" "$CONFIG" > "$TMP"

THREADS=(1 2 4 6 8 12 16 24)
ROUNDS=5
median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "threads qps_median search_time_median"
for t in "${THREADS[@]}"; do
  export OMP_NUM_THREADS="$t"
  last=$((t-1))
  qs=(); sts=()
  for r in $(seq 1 "$ROUNDS"); do
    out=$(numactl --physcpubind=0-"$last" --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
    sts+=("$(echo "$out" | sed -n 's/^search_time = //p')")
  done
  echo "$t $(median "${qs[@]}") $(median "${sts[@]}")"
done