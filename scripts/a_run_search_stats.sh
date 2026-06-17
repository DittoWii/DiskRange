#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Default CONFIG / GT come from scripts/_dataset.sh (single source of truth
# for all a_run_*.sh scripts). Edit that file to switch dataset.
source "$(dirname "${BASH_SOURCE[0]}")/_dataset.sh"
CONFIG="${1:-$DEFAULT_CONFIG}"
GT="${2:-$DEFAULT_GT}"

# Historical default (kept as memo; no longer active — edit _dataset.sh instead):
# CONFIG="${1:-configs/deep100m_disk_fast_path.config}"
# GT="${2:-experiments/range_gt/output/deep100m_0.1459_gt.json}"

# Pin to NUMA node 0's 24 physical cores (avoid SMT siblings 48-71 and cross-socket).
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export DJ_CLUSTER_CACHE_DISABLE=1
export DJ_CELL_CACHE_DISABLE=1

# Use the ISOLATED breakdown build (PRUNE_BREAKDOWN_STATS=ON) so the slab/pca/cell
# prune-breakdown counters are populated. This is a different binary from ./build,
# which a_run_search.sh uses for production-faithful timing. Build it with:
#   bash scripts/a_build_search_stats.sh
# Override with SEARCH_STATS_BIN=/path/to/search_stats.
BIN="${SEARCH_STATS_BIN:-${REPO_ROOT}/build_stats/search_stats}"
if [[ ! -x "${BIN}" ]]; then
  echo "error: ${BIN} not found. Build the breakdown binary first:" >&2
  echo "       bash scripts/a_build_search_stats.sh" >&2
  exit 1
fi

exec numactl --physcpubind=0-23 --membind=0 \
  "${BIN}" --config "${CONFIG}" --gt "${GT}"
