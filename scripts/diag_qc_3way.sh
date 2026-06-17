#!/usr/bin/env bash
# Query Classifier 3-way ablation on Deep10M (§4 Soundness table).
#   baseline   : fast_path=false + slow_path_rbq_prefilter=false  (pure slow path, no RBQ at all)
#   1bit_only  : fast_path=true (DJ_ORACLE_1BIT_ONLY) + prefilter=true
#   1bit+8bit  : fast_path=true (full) + prefilter=true
# NOTE: slow_path_rbq_prefilter REQUIRES fast_path=true, so baseline cannot keep it.
# Cache OFF (stats mode): bytes_read is the exact logical=physical fetch volume.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1

run() {
  local label="$1" fp="$2" pf="$3"; shift 3
  local TMP; TMP=$(mktemp /tmp/qc3_XXXX.config)
  sed -e "s/^fast_path_enabled .*/fast_path_enabled ${fp}/" \
      -e "s/^slow_path_rbq_prefilter_enabled .*/slow_path_rbq_prefilter_enabled ${pf}/" \
      -e "s/^slow_path_rbq_phase_b_enabled .*/slow_path_rbq_phase_b_enabled ${pf}/" \
      "$BASE_CONFIG" > "$TMP"
  local out
  out=$(env "$@" numactl --physcpubind=0-23 --membind=0 \
        ./build/search_stats --config "$TMP" --gt "$GT" 2>/dev/null || true)
  local recall qps skip fe io zc nc dc cells
  recall=$(echo "$out" | sed -n 's/^recall = //p')
  qps=$(echo "$out"    | sed -n 's/^qps = \(.*\) query\/sec/\1/p')
  skip=$(echo "$out"   | sed -n 's/^fast_path_skip_rate = //p')
  fe=$(echo "$out"     | sed -n 's/^false_empty_query_rate = //p')
  zb=$(echo "$out"     | sed -n 's/^zero_result_bytes_read = //p')
  nb=$(echo "$out"     | sed -n 's/^nonzero_result_bytes_read = //p')
  zc=$(echo "$out"     | sed -n 's/^zero_result_cells_read = //p')
  nc=$(echo "$out"     | sed -n 's/^nonzero_result_cells_read = //p')
  dc=$(echo "$out"     | sed -n 's/^dist_comp = //p')
  cells=$(( ${zc:-0} + ${nc:-0} ))
  io=$(( ${zb:-0} + ${nb:-0} ))
  printf "%-12s | recall=%-9s | qps=%-9s | skip=%-8s | cells_read=%-11s | bytes_read=%-13s | dist_comp=%-12s | false_empty=%s\n" \
         "$label" "${recall:-FAIL}" "${qps:-FAIL}" "${skip:-0}" "$cells" "${io:-FAIL}" "${dc:-FAIL}" "${fe:-NA}"
  echo "    (cells: zero=${zc:-NA} + nonzero=${nc:-NA}   bytes: zero=${zb:-NA} + nonzero=${nb:-NA})"
  rm -f "$TMP"
}

echo "Deep10M  d=96  radius^2=0.04006  K=160  nprobe=6  cache=OFF (clean fetch counts)"
echo "================================================================================"
run "baseline"   false false
run "1bit_only"  true  true  DJ_ORACLE_1BIT_ONLY=1
run "1bit+8bit"  true  true
echo "DONE"
