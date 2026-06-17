#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
from pathlib import Path

cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
if 'option(PRUNE_BREAKDOWN_STATS "Compute standalone slab/PCA prune breakdown stats" OFF)' not in cmake:
    raise SystemExit("missing PRUNE_BREAKDOWN_STATS CMake option")
if "add_definitions(-DDJ_ENABLE_PRUNE_BREAKDOWN_STATS=" in cmake:
    raise SystemExit("global DJ_ENABLE_PRUNE_BREAKDOWN_STATS add_definitions must be removed")
for marker in [
    "target_compile_definitions(main PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=${PRUNE_BREAKDOWN_DEF})",
    "target_compile_definitions(build_prune_stats PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=1)",
]:
    if marker not in cmake:
        raise SystemExit(f"missing per-target prune compile definition: {marker}")

run_result = Path("lib/RunResult.h").read_text(encoding="utf-8")
for field in [
    "size_t slab_only_pruned_clusters = 0;",
    "size_t pca_only_pruned_clusters = 0;",
    "size_t combined_pruned_clusters = 0;",
    "double slab_only_prune_rate = 0.0;",
    "double pca_only_prune_rate = 0.0;",
    "double combined_prune_rate = 0.0;",
    "size_t cell_candidate_vectors = 0;",
    "size_t cell_pruned_vectors = 0;",
    "size_t cell_kept_vectors = 0;",
    "double cell_prune_rate = 0.0;",
]:
    if field not in run_result:
        raise SystemExit(f"missing RunResult field: {field}")

disk_range = Path("lib/DiskRange.h").read_text(encoding="utf-8")
required_diskrange = [
    "if constexpr (kDiskRangePruneBreakdownStatsEnabled)",
    "if constexpr (!kDiskRangePruneBreakdownStatsEnabled)",
    "radius_prescreened_clusters",
    "slab_threshold = std::numeric_limits<float>::infinity();",
    "slab_only_pruned_clusters",
    "pca_only_pruned_clusters",
    "combined_pruned_clusters",
    "result.slab_only_prune_rate",
    "result.pca_only_prune_rate",
    "result.combined_prune_rate",
    "cell_candidate_vectors",
    "cell_pruned_vectors",
    "cell_kept_vectors",
    "result.cell_prune_rate",
]
for needle in required_diskrange:
    if needle not in disk_range:
        raise SystemExit(f"missing DiskRange debug stats marker: {needle}")

pragma_start = disk_range.find("#pragma omp parallel for")
pragma_end = disk_range.find("for (size_t batch_start = 0;", pragma_start)
if pragma_start == -1 or pragma_end == -1:
    raise SystemExit("could not locate DiskRange OpenMP search loop")
reduction_clause = disk_range[pragma_start:pragma_end]
for field in [
    "cell_candidate_vectors",
    "cell_pruned_vectors",
    "cell_kept_vectors",
]:
    if field not in reduction_clause:
        raise SystemExit(f"missing OpenMP reduction variable: {field}")

main = Path("expr/main.cpp").read_text(encoding="utf-8")
for call in [
    'msginfo_s("slab_only_prune_rate = {}", result.slab_only_prune_rate);',
    'msginfo_s("pca_only_prune_rate = {}", result.pca_only_prune_rate);',
    'msginfo_s("combined_prune_rate = {}", result.combined_prune_rate);',
    'msginfo_s("slab_only_pruned_clusters = {}", result.slab_only_pruned_clusters);',
    'msginfo_s("pca_only_pruned_clusters = {}", result.pca_only_pruned_clusters);',
    'msginfo_s("combined_pruned_clusters = {}", result.combined_pruned_clusters);',
    'msginfo_s("cell_candidate_vectors = {}", result.cell_candidate_vectors);',
    'msginfo_s("cell_pruned_vectors = {}", result.cell_pruned_vectors);',
    'msginfo_s("cell_kept_vectors = {}", result.cell_kept_vectors);',
    'msginfo_s("cell_prune_rate = {}", result.cell_prune_rate);',
]:
    if call not in main:
        raise SystemExit(f"missing debug print call: {call}")

if "--no-cell-filter" not in main:
    raise SystemExit("missing --no-cell-filter in expr/main.cpp")

if "add_executable(build_prune_stats expr/build_prune_stats/main.cpp)" not in cmake:
    raise SystemExit("missing CMake build_prune_stats executable marker")
link_line = next(
    (
        line.strip()
        for line in cmake.splitlines()
        if line.strip().startswith("target_link_libraries(build_prune_stats ")
    ),
    "",
)
if not link_line or "liburing_vendored" not in link_line:
    raise SystemExit("missing CMake build_prune_stats liburing link marker")
PY
