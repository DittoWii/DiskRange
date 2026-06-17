#!/usr/bin/env bash
#
# Build the on-disk index (clusters + metadata + RBQ + SQG) for a dataset.
#
# Usage:
#   bash scripts/a_run_indexing.sh [config_path] [gt_json_path]
#
# Defaults (when no args given):
#   config = configs/deep1m_disk_fast_path.config
#   gt     = experiments/range_gt/output/deep1m_0.2632_gt.json
#
# Examples:
#   # default deep1m fast-path build
#   bash scripts/a_run_indexing.sh
#
#   # custom dataset (e.g. int8)
#   bash scripts/a_run_indexing.sh \
#     configs/sift1m_int8.config \
#     experiments/range_gt/output/sift1m_int8_gt.json
#
# Prerequisites:
#   1. bash scripts/a_build_indexing.sh   (compile ./build/build)
#   2. The GT JSON's provenance.base_path / query_path must exist on disk.
#
# Notes:
#   - Pinned to NUMA node 0 (cores 0-23) to keep memory local.
#   - If a metadata file already exists with an older schema, the binary will
#     reject it and ask you to delete the stale index before rebuilding.

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
  ./build/build --config "${CONFIG}" --gt "${GT}"
