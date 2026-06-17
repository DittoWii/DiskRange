#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A2 sparse-gather vs dense-GEMM centroid q_dot, on the non-empty (300-query)
# workload. A2 (env DJ_A2_SPARSE=1) computes q_dot only for each candidate's
# self + slab neighbours (~7644/query) instead of the full 50000 GEMM.
# recall/dist_comp must stay identical (A2 is full-precision, lossless);
# the question is whether 6.6x fewer dot-products beats gather/dedup overhead.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_NONEMPTY_ONLY=1
ROUNDS=7

TMP=$(mktemp /tmp/a2_bench_XXXX.config); trap 'rm -f "$TMP"' EXIT
sed 's/^approx_alpha .*/approx_alpha 1.0/' "$CONFIG" > "$TMP"
median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

echo "mode qps_median search_time_median recall dist_comp  | all_qps"
for mode in baseline a2; do
  if [ "$mode" = a2 ]; then export DJ_A2_SPARSE=1; else unset DJ_A2_SPARSE; fi
  qs=(); sts=(); rc=""; dc=""
  for r in $(seq 1 "$ROUNDS"); do
    out=$(numactl --physcpubind=0-23 --membind=0 ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
    sts+=("$(echo "$out" | sed -n 's/^search_time = //p')")
    rc="$(echo "$out" | sed -n 's/^recall = //p')"
    dc="$(echo "$out" | sed -n 's/^dist_comp = //p')"
  done
  printf '%-9s %-12s %-20s %-9s %-9s | %s\n' \
    "$mode" "$(median "${qs[@]}")" "$(median "${sts[@]}")" "$rc" "$dc" "${qs[*]}"
done
