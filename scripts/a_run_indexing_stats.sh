#!/usr/bin/env bash
#
# Build the on-disk index and print detailed build-stage stats.
#
# Usage:
#   bash scripts/a_run_indexing_stats.sh [config_path] [gt_json_path]
#
# Defaults (when no args given):
#   config = configs/deep100m_disk_fast_path.config
#   gt     = experiments/range_gt/output/deep100m_0.1459_gt.json
#
# Examples:
#   # default deep100m stats build
#   bash scripts/a_run_indexing_stats.sh
#
#   # smaller dataset for a quick stats dry run
#   bash scripts/a_run_indexing_stats.sh \
#     configs/deep1m_disk_fast_path.config \
#     experiments/range_gt/output/deep1m_0.2632_gt.json
#
# Prerequisites:
#   1. bash scripts/a_build_indexing.sh   (compile ./build/build_stats)
#   2. The GT JSON's provenance.base_path / query_path must exist on disk.
#
# Use this when you want per-phase breakdowns (KMeans / PCA fit / PC2 reorder /
# slab fit / RBQ / SQG) instead of just total build time. Otherwise prefer
# a_run_indexing.sh.

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

exec numactl --physcpubind=0-23 --membind=0 \
  ./build/build_stats --config "${CONFIG}" --gt "${GT}"
