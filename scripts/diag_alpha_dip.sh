#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1

TMP=$(mktemp /tmp/diag_XXXX.config)
trap 'rm -f "$TMP"' EXIT

ALPHAS=(1.3 1.4 1.5 1.6 1.7)
ROUNDS=7

echo "alpha round qps dist_comp search_time recall fast_path_skip_rate per_query_zero"
for a in "${ALPHAS[@]}"; do
  sed "s/^approx_alpha .*/approx_alpha ${a}/" "$CONFIG" > "$TMP"
  for r in $(seq 1 "$ROUNDS"); do
    out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qps=$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
    dc=$(echo "$out"  | sed -n 's/^dist_comp = //p')
    st=$(echo "$out"  | sed -n 's/^search_time = //p')
    rc=$(echo "$out"  | sed -n 's/^recall = //p')
    fps=$(echo "$out" | sed -n 's/^fast_path_skip_rate = //p')
    pqz=$(echo "$out" | sed -n 's/^per_query_zero = //p')
    echo "$a $r $qps $dc $st $rc $fps $pqz"
  done
done