#!/usr/bin/env bash
# Query_Classifier_1bit_vs_8bit.md §5 ablation on Deep10M (d=96).
# baseline (1-bit + 8-bit refine) vs 1-bit-only (DJ_ORACLE_1BIT_ONLY=1).
# Same index/GT/nprobe/K; measures skip_rate / recall / qps / false_empty.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores

run() {
  local label="$1"; shift
  local out
  out=$(env "$@" numactl --physcpubind=0-23 --membind=0 \
        ./build/search --config "$CONFIG" --gt "$GT" 2>/dev/null)
  local rc qps sk fe dc
  rc=$(echo "$out"  | sed -n 's/^recall = //p')
  qps=$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
  sk=$(echo "$out"  | sed -n 's/^fast_path_skip_rate = //p')
  fe=$(echo "$out"  | sed -n 's/^false_empty_query_rate = //p')
  dc=$(echo "$out"  | sed -n 's/^dist_comp = //p')
  printf "%-18s | recall=%-10s | qps=%-10s | skip=%-10s | false_empty=%-12s | dist_comp=%s\n" \
         "$label" "$rc" "$qps" "$sk" "$fe" "$dc"
}

echo "Deep10M  d=96  nprobe=6  K=160  radius^2=0.04006  (nq=10000, in_range=73369)"
echo "-------------------------------------------------------------------------------"
run "baseline(1b+8b)"
run "1bit_only"        DJ_ORACLE_1BIT_ONLY=1
echo "DONE"
