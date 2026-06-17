#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 2-D grid sweep (approx_alpha x sqg_ef_search) for a recall-QPS Pareto front.
# alpha drives recall, ef drives QPS (ef does not move recall at this radius).
# Output: long CSV with per-round qps + median, for Pareto post-processing.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1

TMP=$(mktemp /tmp/grid_XXXX.config)
trap 'rm -f "$TMP"' EXIT

ALPHAS=(1.0 1.2 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
EFS=(20 30 50 80 150 300)
ROUNDS=3

mkdir -p results
OUT="results/deep10m_alpha_ef_grid.csv"
echo "approx_alpha,sqg_ef_search,recall,qps_r1,qps_r2,qps_r3,qps_median,dist_comp" > "$OUT"

median3() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[2]}'; }

echo "grid: ${#ALPHAS[@]} alphas x ${#EFS[@]} efs x ${ROUNDS} rounds"
printf '%-6s %-6s %-10s %-12s\n' alpha ef recall qps_median
for a in "${ALPHAS[@]}"; do
  for ef in "${EFS[@]}"; do
    sed -e "s/^approx_alpha .*/approx_alpha ${a}/" \
        -e "s/^sqg_ef_search .*/sqg_ef_search ${ef}/" "$CONFIG" > "$TMP"
    qs=()
    rc=""; dc=""
    for r in $(seq 1 "$ROUNDS"); do
      out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
      qps=$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
      qs+=("$qps")
      rc=$(echo "$out" | sed -n 's/^recall = //p')
      dc=$(echo "$out" | sed -n 's/^dist_comp = //p')
    done
    med=$(median3 "${qs[@]}")
    echo "${a},${ef},${rc},${qs[0]},${qs[1]},${qs[2]},${med},${dc}" >> "$OUT"
    printf '%-6s %-6s %-10s %-12s\n' "$a" "$ef" "$rc" "$med"
  done
done
echo "done -> $OUT"