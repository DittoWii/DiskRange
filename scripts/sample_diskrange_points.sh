#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Sample 10 clean (recall, QPS) points for the DiskRange curve using the
# optimal recipe: sqg_ef_search=20, fast_path_nprobe_N=1, 24 threads.
# alpha drives recall across the [~93.8%, ~99.6%] range. 7 rounds -> median QPS.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1

TMP=$(mktemp /tmp/sample_dr_XXXX.config)
trap 'rm -f "$TMP"' EXIT

EF=20
ALPHAS=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9)
ROUNDS=7

mkdir -p results
OUT="results/deep10m_diskrange_points.csv"
echo "algorithm,recall_percent,qps,approx_alpha,sqg_ef_search,threads" > "$OUT"

median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "recipe: ef=${EF} nprobe_N=1 threads=24 rounds=${ROUNDS}"
printf '%-6s %-12s %-12s\n' alpha recall_pct qps_median
for a in "${ALPHAS[@]}"; do
  sed -e "s/^approx_alpha .*/approx_alpha ${a}/" \
      -e "s/^sqg_ef_search .*/sqg_ef_search ${EF}/" "$CONFIG" > "$TMP"
  qs=(); rc=""
  for r in $(seq 1 "$ROUNDS"); do
    out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qps=$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
    qs+=("$qps")
    rc=$(echo "$out" | sed -n 's/^recall = //p')
  done
  med=$(median "${qs[@]}")
  rpct=$(awk -v x="$rc" 'BEGIN{printf "%.4f", x*100}')
  qfmt=$(awk -v x="$med" 'BEGIN{printf "%.1f", x}')
  echo "DiskRange,${rpct},${qfmt},${a},${EF},24" >> "$OUT"
  printf '%-6s %-12s %-12s\n' "$a" "$rpct" "$qfmt"
done
echo "done -> $OUT"