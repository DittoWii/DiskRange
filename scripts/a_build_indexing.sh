#!/usr/bin/env bash
#
# Build the indexing binaries.
#
# Usage:
#   bash scripts/a_build_indexing.sh
#
# Targets built:
#   - build        ./build/build         (used by a_run_indexing.sh)
#   - build_stats  ./build/build_stats   (used by a_run_indexing_stats.sh)
#
# Overrides:
#   BUILD_DIR=/custom/dir bash scripts/a_build_indexing.sh
#
# Run this first; a_run_indexing.sh / a_run_indexing_stats.sh assume the
# binaries exist at ./build/build and ./build/build_stats.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

if [[ ! -f "${BUILD_DIR}/build.ninja" && ! -f "${BUILD_DIR}/Makefile" ]]; then
  cmake -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}"
fi

cmake --build "${BUILD_DIR}" --target build --target build_stats -j"$(nproc)"
