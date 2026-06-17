#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Default CONFIG / GT come from scripts/_dataset.sh (single source of truth
# for all a_run_*.sh scripts). Edit that file to switch dataset.
source "$(dirname "${BASH_SOURCE[0]}")/_dataset.sh"
CONFIG="${1:-$DEFAULT_CONFIG}"
GT="${2:-$DEFAULT_GT}"

# Historical defaults (kept as memo; no longer active — edit _dataset.sh instead):
# CONFIG="${1:-configs/spacev1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep10m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# CONFIG="${1:-configs/deep1m_disk_fast_path.config}"
# GT="${2:-experiments/range_gt/output/spacev1m_63.7338_gt.json}"
# GT="${2:-experiments/range_gt/output/deep10m_0.2002_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"
# GT="${2:-experiments/range_gt/output/deep1m_0.2632_gt.json}"

# Pin to NUMA node 0's 24 physical cores (avoid SMT siblings 48-71 and cross-socket).
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export DJ_CLUSTER_CACHE_DISABLE=1
export DJ_CELL_CACHE_DISABLE=1

exec numactl --physcpubind=0-23 --membind=0 \
  ./build/search --config "${CONFIG}" --gt "${GT}"
