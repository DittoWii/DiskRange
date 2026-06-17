#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1

TMP=$(mktemp /tmp/smoke_nprobe_XXXX.config)
trap 'rm -f "$TMP"' EXIT

NPROBES=(1 2 3 4 6 8 12 16)

echo "nprobe_N recall qps skip_rate false_empty dist_comp"
for n in "${NPROBES[@]}"; do
  sed "s/^fast_path_nprobe_N .*/fast_path_nprobe_N ${n}/" "$CONFIG" > "$TMP"
  out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
  rc=$(echo "$out"  | sed -n 's/^recall = //p')
  qps=$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
  sk=$(echo "$out"  | sed -n 's/^fast_path_skip_rate = //p')
  fe=$(echo "$out"  | sed -n 's/^false_empty_query_rate = //p')
  dc=$(echo "$out"  | sed -n 's/^dist_comp = //p')
  echo "$n $rc $qps $sk $fe $dc"
done