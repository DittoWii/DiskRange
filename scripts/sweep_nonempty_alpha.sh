#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# "Full non-empty workload" sweep: production config (fast_path + slow_path_rbq
# all on), but DJ_NONEMPTY_ONLY=1 restricts the workload to the ~300 queries
# that actually have results. QPS denominator becomes the non-empty count, so
# this reports the slow-path throughput on a fully non-empty workload.
# recall is invariant to this filter (empty queries contribute 0 to recall).

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1
export DJ_NONEMPTY_ONLY=1   # <-- the experiment knob

TMP=$(mktemp /tmp/sweep_ne_XXXX.config); trap 'rm -f "$TMP"' EXIT
ALPHAS=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9)
ROUNDS=7

mkdir -p results
OUT="results/deep10m_nonempty_alpha.csv"
echo "approx_alpha,recall_percent,qps_nonempty_median,search_time,n_nonempty" > "$OUT"

median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "recipe: production config + DJ_NONEMPTY_ONLY=1, 24 threads, ${ROUNDS} rounds"
printf '%-6s %-12s %-14s %s\n' alpha recall_pct qps_nonempty search_t
for a in "${ALPHAS[@]}"; do
  sed "s/^approx_alpha .*/approx_alpha ${a}/" "$CONFIG" > "$TMP"
  qs=(); rc=""; st=""
  for r in $(seq 1 "$ROUNDS"); do
    out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
    rc=$(echo "$out" | sed -n 's/^recall = //p')
    st=$(echo "$out" | sed -n 's/^search_time = //p')
  done
  med=$(median "${qs[@]}")
  rpct=$(awk -v x="$rc" 'BEGIN{printf "%.4f", x*100}')
  echo "${a},${rpct},${med},${st},300" >> "$OUT"
  printf '%-6s %-12s %-14s %s\n' "$a" "$rpct" "$med" "$st"
done
echo "done -> $OUT"