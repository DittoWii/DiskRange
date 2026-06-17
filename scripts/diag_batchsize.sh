#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Sweep kQdotBatchSize to fix slow-path parallel starvation on the non-empty
# (300-query) workload. Smaller batch -> more work-units (better load balance)
# but smaller centroid GEMM (lower per-unit efficiency). Find the sweet spot.
# Restores the original value on exit.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1 DJ_NONEMPTY_ONLY=1

HDR="lib/DiskRange.h"
ORIG=$(grep -oE "kQdotBatchSize = [0-9]+;" "$HDR" | grep -oE "[0-9]+")
restore() { sed -i "s/kQdotBatchSize = [0-9]\+;/kQdotBatchSize = ${ORIG};/" "$HDR"; }
trap restore EXIT

TMP=$(mktemp /tmp/bs_XXXX.config); sed "s/^approx_alpha .*/approx_alpha 1.0/" "$CONFIG" > "$TMP"
median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "kQdotBatchSize qps_median search_time_median (24 threads, non-empty)"
for bs in 1 2 4 8 16; do
  sed -i "s/kQdotBatchSize = [0-9]\+;/kQdotBatchSize = ${bs};/" "$HDR"
  cmake --build build --target search -j >/dev/null 2>&1
  qs=(); sts=()
  for r in $(seq 1 5); do
    out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
    sts+=("$(echo "$out" | sed -n 's/^search_time = //p')")
  done
  echo "$bs $(median "${qs[@]}") $(median "${sts[@]}")"
done
rm -f "$TMP"