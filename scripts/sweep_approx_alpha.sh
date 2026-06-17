#!/usr/bin/env bash
set -euo pipefail

# Sweep approx_alpha for the dataset selected in scripts/_dataset.sh.
# For each alpha we patch only the `approx_alpha` line of the base config,
# run ./build/search, and collect recall / qps. QPS is measured over ROUNDS
# repeats (outer loop = round, inner loop = alpha, so repeats are sampled at
# different times) and reduced to a median. recall is deterministic, so it is
# recorded once.
#
# Usage:
#   bash scripts/sweep_approx_alpha.sh [alpha1 alpha2 ...]
#   ROUNDS=3 bash scripts/sweep_approx_alpha.sh
#
# Outputs under results/:
#   sweep_approx_alpha_<dataset>.csv          # wide: per-round qps + median
#   sweep_approx_alpha_<dataset>.round<r>.csv # raw per-round dumps

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source "$(dirname "${BASH_SOURCE[0]}")/_dataset.sh"
CONFIG="${DEFAULT_CONFIG}"
GT="${DEFAULT_GT}"
ROUNDS="${ROUNDS:-3}"

ALPHAS=("$@")
if [[ ${#ALPHAS[@]} -eq 0 ]]; then
  # Dense grid: fine steps through the 1.0-1.5 sweet spot, coarser past it.
  ALPHAS=(1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.6 1.8 2.0)
fi

# Same NUMA / OMP pinning as a_run_search.sh for comparable QPS numbers.
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1
export DJ_CELL_CACHE_DISABLE=1

mkdir -p results
DATASET_TAG="$(basename "${CONFIG}" .config)"
OUT_CSV="results/sweep_approx_alpha_${DATASET_TAG}.csv"
TMP_CONFIG="$(mktemp /tmp/sweep_alpha_XXXX.config)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

# Accumulators keyed by alpha.
declare -A RECALL QPS_LIST STIME RLOSS

median() {  # median of space-separated numbers on $1
  local sorted n
  sorted=$(printf '%s\n' $1 | sort -n)
  n=$(printf '%s\n' $1 | wc -w)
  printf '%s\n' "${sorted}" | awk -v n="${n}" 'NR==int((n+1)/2){print; exit}'
}

run_one() {  # $1=alpha  -> echoes "recall qps stime rloss"
  local alpha="$1"
  sed "s/^approx_alpha .*/approx_alpha ${alpha}/" "${CONFIG}" > "${TMP_CONFIG}"
  grep -q '^approx_alpha ' "${TMP_CONFIG}" || echo "approx_alpha ${alpha}" >> "${TMP_CONFIG}"
  local out
  out="$(numactl --physcpubind=0-23 --membind=0 \
        ./build/search --config "${TMP_CONFIG}" --gt "${GT}" 2>/dev/null)"
  echo "$(echo "${out}" | sed -n 's/^recall = //p') \
$(echo "${out}" | sed -n 's/^qps = \(.*\) query\/sec/\1/p') \
$(echo "${out}" | sed -n 's/^search_time = //p') \
$(echo "${out}" | sed -n 's/^range_recall_loss_hit_weighted = //p')"
}

echo "config=${CONFIG}  gt=${GT}  rounds=${ROUNDS}"
for ((r=1; r<=ROUNDS; r++)); do
  echo "=== round ${r}/${ROUNDS} ==="
  ROUND_CSV="results/sweep_approx_alpha_${DATASET_TAG}.round${r}.csv"
  echo "approx_alpha,recall,qps,search_time,range_recall_loss_hit_weighted" > "${ROUND_CSV}"
  for alpha in "${ALPHAS[@]}"; do
    read -r recall qps stime rloss <<< "$(run_one "${alpha}")"
    RECALL[$alpha]="${recall}"
    STIME[$alpha]="${stime}"
    RLOSS[$alpha]="${rloss}"
    QPS_LIST[$alpha]="${QPS_LIST[$alpha]:-} ${qps}"
    echo "${alpha},${recall},${qps},${stime},${rloss}" >> "${ROUND_CSV}"
    printf '  alpha=%-6s recall=%-10s qps=%s\n' "${alpha}" "${recall}" "${qps}"
  done
done

# Wide summary with per-round qps columns + median.
header="approx_alpha,recall"
for ((r=1; r<=ROUNDS; r++)); do header+=",qps_r${r}"; done
header+=",qps_median,search_time,range_recall_loss_hit_weighted"
echo "${header}" > "${OUT_CSV}"

echo ""
echo "=== summary (qps median over ${ROUNDS} rounds) ==="
printf '%-8s %-10s %-12s %s\n' alpha recall qps_median rloss
for alpha in "${ALPHAS[@]}"; do
  med="$(median "${QPS_LIST[$alpha]}")"
  row="${alpha},${RECALL[$alpha]}"
  for q in ${QPS_LIST[$alpha]}; do row+=",${q}"; done
  row+=",${med},${STIME[$alpha]},${RLOSS[$alpha]}"
  echo "${row}" >> "${OUT_CSV}"
  printf '%-8s %-10s %-12s %s\n' "${alpha}" "${RECALL[$alpha]}" "${med}" "${RLOSS[$alpha]}"
done
echo "done -> ${OUT_CSV}"
