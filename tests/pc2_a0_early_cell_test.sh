#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# A0 early-cell behavior is now subsumed by schema 5 P-C2 KD-tree cells. Keep
# this archived-change test entry point as a compatibility wrapper.
exec tests/pc2_kdtree_cell_cpp_deployment_test.sh "$@"
