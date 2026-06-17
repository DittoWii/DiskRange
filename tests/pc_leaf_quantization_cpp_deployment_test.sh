#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# P-C schema 4 leaf-quantization was retired by P-C2 schema 5 KD-tree cells.
# Keep this legacy test entry point as a compatibility wrapper for older local
# scripts that still invoke the P-C deploy test name.
exec tests/pc2_kdtree_cell_cpp_deployment_test.sh "$@"
