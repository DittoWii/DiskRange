#!/usr/bin/env bash
set -euo pipefail

# Build ONLY the search_stats binary, with PRUNE_BREAKDOWN_STATS=ON, into a
# SEPARATE build directory (default: build_stats). This keeps the breakdown
# build fully isolated from ./build, which a_build_search.sh / a_run_search.sh
# use for production-faithful timing (PRUNE_BREAKDOWN_STATS=OFF).
#
#   search       (./build)        -> optimal performance, clean t_cluster_bound
#   search_stats (./build_stats)  -> slab/pca/cell prune breakdown counters
#
# Toggling one never recompiles the other: PRUNE_BREAKDOWN_STATS is a CMake
# cache variable, so each value needs its own build directory.
#
# Override the directory with STATS_BUILD_DIR=/some/path bash scripts/a_build_search_stats.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${STATS_BUILD_DIR:-${REPO_ROOT}/build_stats}"

# Always (re)configure with the flag ON so the dir can never silently carry an
# OFF value left over from an earlier configure. CMake no-ops if nothing changed.
cmake -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DPRUNE_BREAKDOWN_STATS=ON

cmake --build "${BUILD_DIR}" --target search_stats -j"$(nproc)"

echo "built: ${BUILD_DIR}/search_stats (PRUNE_BREAKDOWN_STATS=ON)"
