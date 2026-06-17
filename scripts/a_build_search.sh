#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

if [[ ! -f "${BUILD_DIR}/build.ninja" && ! -f "${BUILD_DIR}/Makefile" ]]; then
  cmake -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}"
fi

# Build ONLY the performance binary. search_stats (prune-breakdown) lives in its
# own isolated build via scripts/a_build_search_stats.sh, so ./build stays a pure
# production build with PRUNE_BREAKDOWN_STATS=OFF and faithful t_cluster_bound.
cmake --build "${BUILD_DIR}" --target search -j"$(nproc)"
