#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:---static-only}"
source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
n_queries=8
cluster_num="$(awk '$1 == "cluster_num" { print $2; exit }' "$config")"

run_static_checks() {
  python3 - <<'PY'
from pathlib import Path

cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
expected_targets = {
    "build_stats": "expr/build_stats/main.cpp",
    "search_stats": "expr/search_stats/main.cpp",
    "build": "expr/build/main.cpp",
    "search": "expr/search/main.cpp",
}
entrypoint_sources = {
    "main": "expr/main.cpp",
    "search": "expr/search/main.cpp",
    "search_stats": "expr/search_stats/main.cpp",
    "build": "expr/build/main.cpp",
    "build_stats": "expr/build_stats/main.cpp",
    "build_prune_stats": "expr/build_prune_stats/main.cpp",
}
for target, source in expected_targets.items():
    marker = f"add_executable({target} {source})"
    if marker not in cmake:
        raise SystemExit(f"missing CMake target marker: {marker}")
    link_line = next(
        (
            line.strip()
            for line in cmake.splitlines()
            if line.strip().startswith(f"target_link_libraries({target} ")
        ),
        "",
    )
    if not link_line or "liburing_vendored" not in link_line:
        raise SystemExit(f"missing CMake liburing link marker for {target}")

for marker in (
    "target_compile_definitions(search_stats PRIVATE DJ_ENABLE_SEARCH_PHASE_TIMING=1)",
    "target_compile_definitions(search PRIVATE DJ_ENABLE_SEARCH_PHASE_TIMING=0)",
):
    if marker not in cmake:
        raise SystemExit(f"missing CMake profiling marker: {marker}")

for marker in (
    "target_compile_definitions(main PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=${PRUNE_BREAKDOWN_DEF})",
    "target_compile_definitions(build_prune_stats PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=1)",
    "target_compile_definitions(search_stats PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=${PRUNE_BREAKDOWN_DEF})",
    "target_compile_definitions(search PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=0)",
    "target_compile_definitions(build PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=0)",
    "target_compile_definitions(build_stats PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=0)",
):
    if marker not in cmake:
        raise SystemExit(f"missing CMake prune marker: {marker}")
if "add_definitions(-DDJ_ENABLE_PRUNE_BREAKDOWN_STATS=" in cmake:
    raise SystemExit("global DJ_ENABLE_PRUNE_BREAKDOWN_STATS add_definitions must be removed")
for marker in (
    'option(DJ_USE_OPENBLAS "Route Eigen GEMM through OpenBLAS via EIGEN_USE_BLAS" ON)',
    "if(DJ_USE_OPENBLAS)",
    "add_definitions(-DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS)",
    "target_link_libraries(search_stats ${OPENBLAS_LIBRARIES})",
):
    if marker not in cmake:
        raise SystemExit(f"missing OpenBLAS CMake marker: {marker}")

gitignore = Path(".gitignore").read_text(encoding="utf-8")
for path in (
    "!expr/build/",
    "!expr/build/main.cpp",
    "!expr/build_stats/",
    "!expr/build_stats/main.cpp",
    "!tests/build_search_entrypoints_test.sh",
):
    if path not in gitignore:
        raise SystemExit(f"missing .gitignore exception: {path}")

sources = {name: Path(path).read_text(encoding="utf-8") for name, path in expected_targets.items()}
entrypoints = {
    name: Path(path).read_text(encoding="utf-8")
    for name, path in entrypoint_sources.items()
}
search_main = Path("expr/search/main.cpp").read_text(encoding="utf-8")

for name, source in entrypoints.items():
    if source.count("Eigen::setNbThreads(1);") != 1:
        raise SystemExit(f"{name} must call Eigen::setNbThreads(1) exactly once")
    if "openblas_set_num_threads(1);" not in source:
        raise SystemExit(f"{name} missing guarded OpenBLAS thread pin")
    if "#ifdef DJ_USE_OPENBLAS" not in source and "#if defined(EIGEN_USE_BLAS)" not in source:
        raise SystemExit(f"{name} missing source-level OpenBLAS guard")

for name in ("build_stats", "build"):
    if "index.build(config);" not in sources[name]:
        raise SystemExit(f"{name} must call DiskRange::build")
    if "index.search(config)" in sources[name]:
        raise SystemExit(f"{name} must not call DiskRange::search")

for name in ("search_stats", "search"):
    if "index.search(config)" not in sources[name]:
        raise SystemExit(f"{name} must call DiskRange::search")
    if "index.build(config);" in sources[name]:
        raise SystemExit(f"{name} must not call DiskRange::build")

for marker in (
    "--dump-range-hits",
    "query_id,original_base_id,distance",
    "result.verified_hits",
):
    if marker not in search_main:
        raise SystemExit(f"search missing verified hit dump marker: {marker}")

if "[Layer 1 cluster]" not in sources["search_stats"]:
    raise SystemExit("search_stats must print the detailed search stats section")
if "[Search phase timing]" not in sources["search_stats"]:
    raise SystemExit("search_stats must print the search phase timing section")
for marker in (
    "[Zero-query layer breakdown]",
    "print_zero_query_layer_breakdown",
    "_queries_with_candidates",
    "_queries_after_prescreen",
    "_queries_after_cluster_bound",
    "_queries_after_cell",
    "_mean_cells_read",
    "_mean_bytes_read",
    "zero_share_thread_sum",
    'std::cout << prefix << "_candidates = "',
    'std::cout << prefix << "_radius_prescreened = "',
    'std::cout << prefix << "_distinct_qdot_used = "',
    'std::cout << prefix << "_surviving = "',
    'std::cout << prefix << "_cell_kept = "',
    'std::cout << prefix << "_dist_comp = "',
    'std::cout << prefix << "_mean_distinct_qdot_used = "',
    'std::cout << "radius_prescreen_rate = "',
    'std::cout << "radius_prescreened_clusters = "',
    "PRUNE_BREAKDOWN_STATS=OFF",
    "t_cluster_bound",
):
    if marker not in sources["search_stats"]:
        raise SystemExit(f"missing search_stats output marker: {marker}")
if "build_time = " not in sources["build_stats"]:
    raise SystemExit("build_stats must print build_time")

minimal_markers = {
    'std::cout << "recall = " << result.recall',
    'std::cout << "search_time = " << result.search_time',
    'std::cout << "dist_comp = " << result.dist_comp',
    'std::cout << "per_query_zero = " << result.per_query_zero',
}
missing = [marker for marker in minimal_markers if marker not in sources["search"]]
if missing:
    raise SystemExit(f"search minimal output missing markers: {missing}")
if "[Search phase timing]" in sources["search"]:
    raise SystemExit("search minimal output must not print profiling details")
if "[Layer 1 cluster]" in sources["search"]:
    raise SystemExit("search minimal output must not print layer details")

run_result = Path("lib/RunResult.h").read_text(encoding="utf-8")
for marker in (
    "DJ_ENABLE_SEARCH_PHASE_TIMING",
    "kDiskRangeSearchPhaseTimingEnabled",
    "DataHNSWTraceRecord",
    "data_hnsw_skip_count",
    "data_hnsw_oracle_total_us",
    "data_hnsw_skip_rate",
    "data_hnsw_trace",
    "struct QueryClassStats",
    "queries_with_candidates",
    "queries_after_prescreen",
    "queries_after_cluster_bound",
    "queries_after_cell",
    "cells_read",
    "bytes_read",
    "radius_prescreened",
    "distinct_qdot_used",
    "radius_prescreened_clusters",
    "radius_prescreen_rate",
    "QueryClassStats zero_result",
    "QueryClassStats nonzero_result",
    "size_t n_threads",
):
    if marker not in run_result:
        raise SystemExit(f"missing RunResult profiling marker: {marker}")
for marker in (
    "queries_with_candidates += other.queries_with_candidates",
    "queries_after_prescreen += other.queries_after_prescreen",
    "queries_after_cluster_bound += other.queries_after_cluster_bound",
    "queries_after_cell += other.queries_after_cell",
    "cells_read += other.cells_read",
    "bytes_read += other.bytes_read",
):
    if marker not in run_result:
        raise SystemExit(f"missing QueryClassStats add marker: {marker}")

disk_range = Path("lib/DiskRange.h").read_text(encoding="utf-8")
for marker in (
    "kDiskRangeSearchPhaseTimingEnabled",
    "build_and_save_data_hnsw",
    "if (config.data_hnsw_enabled)",
    "load_data_hnsw(config.data_hnsw_path, config.dim, config.data_hnsw_ef)",
    "oracle.evaluate",
    "data_hnsw_skip_count",
    "data_hnsw_oracle_total_us",
    "skipped_by_data_hnsw",
    "Approximate decision: HNSW can miss in-radius neighbours.",
    "per_thread_stats",
    "t_centroid_dot",
    "t_candidate_gen",
    "t_cluster_bound",
    "t_cell_filter",
    "t_io_wait",
    "t_exact_l2",
    "qdot_seen",
    "qdot_epoch",
    "slab_neighbour_indices",
    "radius_prescreen_thr_sq",
    "cluster_reader.radii",
    "radius_prescreened_clusters",
    "surviving_total",
    "q_cells_read",
    "q_bytes_read",
    "state.q_cells_read += kept_cells.size()",
    "vec_count * config.dim * cluster_reader.bytes_per_element",
    "bucket_size * config.dim * cluster_reader.bytes_per_element",
    "state.q_stats.queries_with_candidates",
    "state.q_stats.queries_after_prescreen",
    "state.q_stats.queries_after_cluster_bound",
    "state.q_stats.queries_after_cell",
    "state.q_stats.cells_read = state.q_cells_read",
    "state.q_stats.bytes_read = state.q_bytes_read",
    "candidate.first > radius_prescreen_thr_sq",
    "query_hits[state.q].empty() ? 0 : 1",
):
    if marker not in disk_range:
        raise SystemExit(f"missing DiskRange profiling marker: {marker}")
for marker in (
    "#ifndef DUMP_LB_SLAB",
    "kDiskRangePruneBreakdownStatsEnabled",
    "config.K < cluster_reader.cluster_num",
):
    radius_idx = disk_range.find("candidate.first > radius_prescreen_thr_sq")
    if radius_idx == -1:
        raise SystemExit("missing radius pre-screen comparison marker")
    radius_window = disk_range[max(0, radius_idx - 900) : radius_idx + 3500]
    if marker not in radius_window:
        raise SystemExit(f"radius pre-screen missing gate marker: {marker}")
if not all(
    marker in disk_range
    for marker in (
        "candidate_partition_ok",
        "triangle_candidate_clusters ==",
        "radius_prescreened_clusters +",
        "triangle_pruned_clusters +",
        "surviving_total",
    )
):
    raise SystemExit("missing four-way candidate partition assertion")
cluster_io = Path("lib/ClusterIO.h").read_text(encoding="utf-8")
for marker in (
    '#include "DataHNSWOracle.h"',
):
    if marker not in cluster_io:
        raise SystemExit(f"missing ClusterIO data HNSW marker: {marker}")
for marker in (
    "float pca_slab_lower_bound(size_t cluster_id,",
    "const std::vector<float>& q_dot_centroids",
    "const float* q_dot_centroids",
    "pca_slab_lower_bound_impl",
    "float prune_threshold",
    "max_slab > prune_threshold",
    "std::numeric_limits<float>::infinity()",
    "reorder_slab_slots_by_width",
    "slab_width_key",
    "std::stable_sort",
):
    if marker not in cluster_io:
        raise SystemExit(f"missing ClusterIO slab early-exit/reorder marker: {marker}")
for marker in (
    "#include <Eigen/Dense>",
    "constexpr size_t kQdotBatchSize",
    "using QdotMatrixMap",
    "using QdotMatrixConstMap",
    "for (size_t batch_start = 0;",
    "batch_start += kQdotBatchSize",
    "q_batch_buf",
    "q_dot_block",
    "q_batch_compact_buf",
    "q_dot_compact_block",
    "q_dot_compact_rows",
    "Eigen::Map",
    "R.noalias() = Q * M.transpose()",
    "q_dot_block.data() + i * cluster_reader.cluster_num",
):
    if marker not in disk_range:
        raise SystemExit(f"missing GEMM qdot marker: {marker}")
if "record_phase(state, state.q_stats.t_candidate_gen)" not in disk_range:
    raise SystemExit("missing candidate_gen phase marker")
if "t_centroid_dot" not in disk_range or "per_query_qdot" not in disk_range:
    raise SystemExit("missing centroid_dot phase marker")
if disk_range.find("record_phase(state, state.q_stats.t_candidate_gen)") > disk_range.find(
    "compute_batched_qdot"
):
    raise SystemExit("candidate_gen phase must be recorded before centroid_dot")
timing_idx = disk_range.find("per_thread_stats.resize")
if timing_idx == -1:
    raise SystemExit("missing per_thread_stats resize marker")
timing_window = disk_range[max(0, timing_idx - 160) : timing_idx + 200]
if "kDiskRangeSearchPhaseTimingEnabled" not in timing_window:
    raise SystemExit("per_thread_stats resize must be guarded by search timing flag")
if "kDiskRangePruneBreakdownStatsEnabled" in timing_window:
    raise SystemExit("search phase timing block must not be guarded by prune breakdown flag")
PY
}

run_build_checks() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDUMP_LB_SLAB=OFF \
    -DPRUNE_BREAKDOWN_STATS=OFF \
    >/tmp/build_search_entrypoints_cmake.log
  cmake --build build --target main build_prune_stats build_stats search_stats \
    build search data_hnsw_recall_bench data_hnsw_range_recall -j \
    >/tmp/build_search_entrypoints_build.log
}

prepare_subset() {
  local run_dir="$1"
  local index_prefix="$run_dir/deep1m"
  local gt_json="$run_dir/deep1m_0.2632_gt.json"
  local build_gt_json="$run_dir/deep1m_0.2632_build_gt.json"
  local query_file="$run_dir/deep1m_query_subset.fbin"

  python3 - "$source_gt_json" "$gt_json" "$build_gt_json" "$index_prefix" "$query_file" "$n_queries" <<'PY'
import copy
import json
import struct
import sys

source_gt, out_gt, build_gt, index_prefix, query_file, requested_raw = sys.argv[1:7]
requested = int(requested_raw)
if requested % 2 != 0:
    raise SystemExit("n_queries must be even for mixed zero/nonzero subset")

with open(source_gt, "r", encoding="utf-8") as f:
    data = json.load(f)

counts_all = [int(x) for x in data["per_query_counts"]]
half = requested // 2
zero_ids = [idx for idx, count in enumerate(counts_all) if count == 0][:half]
nonzero_ids = [idx for idx, count in enumerate(counts_all) if count > 0][:half]
if len(zero_ids) != half or len(nonzero_ids) != half:
    raise SystemExit(
        f"needed {half} zero and {half} nonzero queries, found "
        f"{len(zero_ids)} zero and {len(nonzero_ids)} nonzero"
    )
selected = zero_ids + nonzero_ids

query_path = data["provenance"]["query_path"]
dim = int(data["dim"])
row_bytes = dim * 4
with open(query_path, "rb") as src, open(query_file, "wb") as dst:
    n_file, dim_file = struct.unpack("<II", src.read(8))
    if dim_file != dim:
        raise SystemExit(f"query dim mismatch: json={dim}, file={dim_file}")
    dst.write(struct.pack("<II", len(selected), dim))
    for idx in selected:
        if idx >= n_file:
            raise SystemExit(f"selected query {idx} outside file query count {n_file}")
        src.seek(8 + idx * row_bytes)
        row = src.read(row_bytes)
        if len(row) != row_bytes:
            raise SystemExit(f"short query row at index {idx}")
        dst.write(row)

counts = [counts_all[idx] for idx in selected]
sorted_counts = sorted(counts)
data["n_queries"] = len(selected)
data["total_in_range"] = sum(counts)
data["per_query_counts"] = counts
data["per_query_count_stats"] = {
    "min": min(counts) if counts else 0,
    "median": (
        sorted_counts[(len(sorted_counts) - 1) // 2]
        + sorted_counts[len(sorted_counts) // 2]
    )
    / 2.0
    if sorted_counts
    else 0.0,
    "max": max(counts) if counts else 0,
    "zero_hit_queries": sum(1 for count in counts if count == 0),
}
data["total_in_range"] = sum(counts)
data["provenance"]["query_path"] = query_file
data["index"]["prefix"] = index_prefix

with open(out_gt, "w", encoding="utf-8") as f:
    json.dump(data, f)

build_data = copy.deepcopy(data)
with open(build_gt, "w", encoding="utf-8") as f:
    json.dump(build_data, f)
PY
}

run_runtime_checks() {
  test -f "$source_gt_json"
  test -f "$config"

  local run_dir
  run_dir="$(mktemp -d /tmp/build_search_entrypoints.XXXXXX)"
  trap 'rm -rf "$run_dir" "${run_dir}/deep1m_diskrange"' RETURN
  prepare_subset "$run_dir"

  local gt_json="$run_dir/deep1m_0.2632_gt.json"
  local build_gt_json="$run_dir/deep1m_0.2632_build_gt.json"
  local search_out="$run_dir/search.out"
  local search_stats_out="$run_dir/search_stats.out"
  local radius_positive_gt_json="$run_dir/deep1m_radius_positive_gt.json"
  local radius_positive_out="$run_dir/search_stats_radius_positive.out"
  local search_stats_off_out="$run_dir/search_stats_off.out"
  local search_stats_on_out="$run_dir/search_stats_on.out"

  ./build/build --config "$config" --gt "$build_gt_json" >"$run_dir/build.out"
  test ! -e "$run_dir/deep1m_diskrange/_data_hnsw_file.bin"
  ./build/search --config "$config" --gt "$gt_json" >"$search_out"
  ./build/search_stats --config "$config" --gt "$gt_json" >"$search_stats_out"

  python3 - "$search_out" "$search_stats_out" "$n_queries" "$cluster_num" <<'PY'
import math
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")

def parse(path):
    values = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = line_re.search(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values, text

search, _ = parse(sys.argv[1])
stats, stats_text = parse(sys.argv[2])
n_queries = int(sys.argv[3])
cluster_num = int(sys.argv[4])
if "[Search phase timing]" not in stats_text:
    raise SystemExit("search_stats missing [Search phase timing] at runtime")
for key in ("recall", "dist_comp", "per_query_zero"):
    if key not in search or key not in stats:
        raise SystemExit(f"missing parity key: {key}")
if not math.isclose(float(search["recall"]), float(stats["recall"]), rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(f"recall mismatch: search={search['recall']} search_stats={stats['recall']}")
for key in ("dist_comp", "per_query_zero"):
    if int(float(search[key])) != int(float(stats[key])):
        raise SystemExit(f"{key} mismatch: search={search[key]} search_stats={stats[key]}")
required = (
    "recall",
    "dist_comp",
    "per_query_zero",
    "per_query_min",
    "per_query_median",
    "per_query_max",
    "mean_keep_after_filter",
    "triangle_candidate_clusters",
    "triangle_pruned_clusters",
    "radius_prescreened_clusters",
    "radius_prescreen_rate",
    "zero_result_n_queries",
    "nonzero_result_n_queries",
    "zero_result_candidates",
    "nonzero_result_candidates",
    "zero_result_radius_prescreened",
    "nonzero_result_radius_prescreened",
    "zero_result_surviving",
    "nonzero_result_surviving",
    "zero_result_mean_surviving",
    "nonzero_result_mean_surviving",
    "zero_result_cell_kept",
    "nonzero_result_cell_kept",
    "zero_result_mean_cell_kept",
    "nonzero_result_mean_cell_kept",
    "zero_result_distinct_qdot_used",
    "nonzero_result_distinct_qdot_used",
    "zero_result_mean_distinct_qdot_used",
    "nonzero_result_mean_distinct_qdot_used",
    "zero_result_dist_comp",
    "nonzero_result_dist_comp",
)
for key in required:
    if key not in stats:
        raise SystemExit(f"missing search_stats raw key: {key}")
new_required = (
    "zero_result_queries_with_candidates",
    "zero_result_queries_after_prescreen",
    "zero_result_queries_after_cluster_bound",
    "zero_result_queries_after_cell",
    "zero_result_cells_read",
    "zero_result_bytes_read",
    "zero_result_mean_cells_read",
    "zero_result_mean_bytes_read",
    "nonzero_result_queries_with_candidates",
    "nonzero_result_queries_after_prescreen",
    "nonzero_result_queries_after_cluster_bound",
    "nonzero_result_queries_after_cell",
    "nonzero_result_cells_read",
    "nonzero_result_bytes_read",
    "nonzero_result_mean_cells_read",
    "nonzero_result_mean_bytes_read",
    "zero_n_queries",
    "zero_queries_with_candidates",
    "zero_queries_after_prescreen",
    "zero_queries_after_cluster_bound",
    "zero_queries_after_cell",
    "zero_mean_active_candidates",
    "zero_mean_cluster_pruned",
    "zero_mean_cells_read",
    "zero_mean_bytes_read",
    "zero_share_thread_sum",
)
for key in new_required:
    if key not in stats:
        raise SystemExit(f"missing zero-query layer key: {key}")
if "[Zero-query layer breakdown]" not in stats_text:
    raise SystemExit("search_stats missing [Zero-query layer breakdown]")

# Documented post-gemm-batched-qdot pins for this test's deterministic deep1m subset:
# queries [0, 1, 2, 3, 83, 148, 167, 215], counts [0, 0, 0, 0, 1, 34, 5, 2].
# all-mode runs search_stats with SQG centroid pruning && use_slab, so these assertions cover
# the early-exit slab bound, HNSW-before-qdot reorder, and radius pre-screen path.
EXPECTED_DIST_COMP = 18440
EXPECTED_ZERO_SURVIVING = 187
EXPECTED_NONZERO_SURVIVING = 89
EXPECTED_ZERO_CELL_KEPT = 10325
EXPECTED_NONZERO_CELL_KEPT = 8115
EXPECTED_ZERO_DISTINCT_QDOT_USED = 20742
EXPECTED_NONZERO_DISTINCT_QDOT_USED = 15123
EXPECTED_TOTAL_DISTINCT_QDOT_USED = (
    EXPECTED_ZERO_DISTINCT_QDOT_USED + EXPECTED_NONZERO_DISTINCT_QDOT_USED
)
if int(float(stats["per_query_zero"])) != 4:
    raise SystemExit(f"per_query_zero changed: expected=4 actual={stats['per_query_zero']}")
if int(float(stats["zero_result_n_queries"])) != 4 or int(
    float(stats["nonzero_result_n_queries"])
) != 4:
    raise SystemExit(
        "expected deterministic subset to contain 4 zero and 4 nonzero queries"
    )
if int(float(stats["triangle_candidate_clusters"])) != n_queries * 245:
    raise SystemExit(
        f"triangle_candidate_clusters changed: expected={n_queries * 245} "
        f"actual={stats['triangle_candidate_clusters']}"
    )
if not math.isclose(float(stats["recall"]), 1.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(f"recall changed: expected=1.0 actual={stats['recall']}")
zero_counts = [
    int(float(stats["zero_queries_after_cell"])),
    int(float(stats["zero_queries_after_cluster_bound"])),
    int(float(stats["zero_queries_after_prescreen"])),
    int(float(stats["zero_queries_with_candidates"])),
    int(float(stats["zero_n_queries"])),
]
if zero_counts != sorted(zero_counts):
    raise SystemExit(f"zero-query layer counts are not monotonic: {zero_counts}")
if int(float(stats["zero_n_queries"])) != int(float(stats["per_query_zero"])):
    raise SystemExit("zero_n_queries must equal per_query_zero")
if not (0.0 <= float(stats["zero_share_thread_sum"]) <= 1.0):
    raise SystemExit("zero_share_thread_sum out of range")
active = float(stats["zero_mean_candidates"]) - float(
    stats["zero_mean_radius_prescreened"]
)
if not math.isclose(
    float(stats["zero_mean_active_candidates"]), active, rel_tol=0.0, abs_tol=1e-6
):
    raise SystemExit("zero_mean_active_candidates identity failed")
cluster_pruned = float(stats["zero_mean_active_candidates"]) - float(
    stats["zero_mean_surviving"]
)
if not math.isclose(
    float(stats["zero_mean_cluster_pruned"]),
    cluster_pruned,
    rel_tol=0.0,
    abs_tol=1e-6,
):
    raise SystemExit("zero_mean_cluster_pruned identity failed")
if int(float(stats["zero_result_queries_with_candidates"])) != int(
    float(stats["zero_n_queries"])
):
    raise SystemExit("zero_result_queries_with_candidates should equal zero_n_queries")
if int(float(stats["zero_result_queries_after_cell"])) > int(
    float(stats["zero_result_queries_after_cluster_bound"])
):
    raise SystemExit("zero_result_queries_after_cell exceeds cluster-bound survivors")
if int(float(stats["zero_result_n_queries"])) + int(float(stats["nonzero_result_n_queries"])) != n_queries:
    raise SystemExit("query class counts do not partition the subset")
class_dist_comp = int(float(stats["zero_result_dist_comp"])) + int(
    float(stats["nonzero_result_dist_comp"])
)
if class_dist_comp != int(float(stats["dist_comp"])):
    raise SystemExit(
        f"dist_comp partition mismatch: classes={class_dist_comp} dist_comp={stats['dist_comp']}"
    )
total_candidates = int(float(stats["zero_result_candidates"])) + int(
    float(stats["nonzero_result_candidates"])
)
class_radius_prescreened = int(float(stats["zero_result_radius_prescreened"])) + int(
    float(stats["nonzero_result_radius_prescreened"])
)
if class_radius_prescreened != int(float(stats["radius_prescreened_clusters"])):
    raise SystemExit(
        f"radius_prescreened partition mismatch: classes={class_radius_prescreened} "
        f"total={stats['radius_prescreened_clusters']}"
    )
if class_radius_prescreened < 0 or class_radius_prescreened > total_candidates:
    raise SystemExit(
        f"radius_prescreened out of range: radius={class_radius_prescreened} "
        f"candidates={total_candidates}"
    )
surviving_total = int(float(stats["zero_result_surviving"])) + int(
    float(stats["nonzero_result_surviving"])
)
partition_total = (
    int(float(stats["radius_prescreened_clusters"]))
    + int(float(stats["triangle_pruned_clusters"]))
    + surviving_total
)
if partition_total != int(float(stats["triangle_candidate_clusters"])):
    raise SystemExit(
        f"candidate partition mismatch: partition={partition_total} "
        f"candidate={stats['triangle_candidate_clusters']}"
    )
total_distinct_qdot_used = int(float(stats["zero_result_distinct_qdot_used"])) + int(
    float(stats["nonzero_result_distinct_qdot_used"])
)
active_candidates = total_candidates - class_radius_prescreened
if total_distinct_qdot_used < active_candidates:
    raise SystemExit(
        f"distinct_qdot_used total must cover post-prescreen candidates: "
        f"distinct={total_distinct_qdot_used} active_candidates={active_candidates}"
    )
if total_distinct_qdot_used > cluster_num * n_queries:
    raise SystemExit(
        f"distinct_qdot_used total exceeds query cluster universe: "
        f"distinct={total_distinct_qdot_used} cluster_num={cluster_num} "
        f"queries={n_queries}"
    )
for prefix in ("zero_result", "nonzero_result"):
    class_queries = int(float(stats[f"{prefix}_n_queries"]))
    candidates = int(float(stats[f"{prefix}_candidates"]))
    radius_prescreened = int(float(stats[f"{prefix}_radius_prescreened"]))
    distinct_qdot_used = int(float(stats[f"{prefix}_distinct_qdot_used"]))
    active = candidates - radius_prescreened
    if radius_prescreened < 0 or radius_prescreened > candidates:
        raise SystemExit(
            f"{prefix}_radius_prescreened out of range: "
            f"radius={radius_prescreened} candidates={candidates}"
        )
    if distinct_qdot_used < active:
        raise SystemExit(
            f"{prefix}_distinct_qdot_used must cover post-prescreen candidates: "
            f"distinct={distinct_qdot_used} active_candidates={active}"
        )
    if distinct_qdot_used > cluster_num * class_queries:
        raise SystemExit(
            f"{prefix}_distinct_qdot_used exceeds per-query cluster universe: "
            f"distinct={distinct_qdot_used} cluster_num={cluster_num} "
            f"queries={class_queries}"
        )
PY

  local data_hnsw_config="$run_dir/data_hnsw_on.config"
  local search_after_unused_artifact_out="$run_dir/search_after_unused_artifact.out"
  local data_hnsw_on_out="$run_dir/search_stats_data_hnsw_on.out"
  local data_hnsw_missing_out="$run_dir/search_stats_data_hnsw_missing.out"
  local data_hnsw_missing_err="$run_dir/search_stats_data_hnsw_missing.err"
  local data_hnsw_range_recall_json="$run_dir/data_hnsw_range_recall.json"
  local data_hnsw_artifact="$run_dir/deep1m_diskrange/_data_hnsw_file.bin"
  cp "$config" "$data_hnsw_config"
  printf '\ndata_hnsw_enabled true\ndata_hnsw_ef 100\n' >>"$data_hnsw_config"

  printf 'not a valid data HNSW artifact\n' >"$data_hnsw_artifact"
  ./build/search --config "$config" --gt "$gt_json" \
    >"$search_after_unused_artifact_out"
  python3 - "$search_out" "$search_after_unused_artifact_out" <<'PY'
import math
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")

def parse(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = line_re.search(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values

baseline = parse(sys.argv[1])
after = parse(sys.argv[2])
for key in ("recall", "dist_comp", "per_query_zero"):
    if key not in baseline or key not in after:
        raise SystemExit(f"missing unused-artifact parity key: {key}")
    if not math.isclose(float(baseline[key]), float(after[key]), rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"unused data HNSW artifact changed {key}: "
            f"baseline={baseline[key]} after={after[key]}"
        )
PY

  ./build/data_hnsw_recall_bench \
    --data /md1/dongjiang_data/deep/deep1m/deep1m_base.fbin \
    --queries /md1/dongjiang_data/deep/deep1m/deep1m_query.fbin \
    --d-q experiments/PCA_enhancement/PF_pivot_witness_sim/results/deep1m_d_q.bin \
    --M 16 --ef-construction 100 --ef 100 --threads 24 \
    --csv "$run_dir/data_hnsw_recall_bench.csv" \
    --save-prefix "$run_dir/deep1m_diskrange/_data_hnsw_file" \
    >"$run_dir/data_hnsw_recall_bench.out"
  mv "$run_dir/deep1m_diskrange/_data_hnsw_file_M16_efc100.bin" \
    "$data_hnsw_artifact"
  test -s "$data_hnsw_artifact"

  ./build/search_stats --config "$data_hnsw_config" --gt "$gt_json" \
    >"$data_hnsw_on_out"
  python3 - "$search_stats_out" "$data_hnsw_on_out" "$n_queries" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")

def parse(path):
    values = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = line_re.search(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values

baseline = parse(sys.argv[1])
enabled = parse(sys.argv[2])
n_queries = int(sys.argv[3])
skip_count = int(float(enabled.get("data_hnsw_skip_count", "0")))
if skip_count <= 0:
    raise SystemExit("data HNSW opt-in expected data_hnsw_skip_count > 0")
if float(enabled.get("data_hnsw_skip_rate", "0")) <= 0.0:
    raise SystemExit("data HNSW opt-in expected positive skip rate")
if int(float(enabled["nonzero_result_dist_comp"])) != int(
    float(baseline["nonzero_result_dist_comp"])
):
    raise SystemExit("non-skipped nonzero pipeline dist_comp changed")
trace_path = Path(enabled["data_hnsw_trace_path"])
if not trace_path.is_file():
    raise SystemExit(f"missing data HNSW trace file: {trace_path}")
with trace_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
if len(rows) != n_queries + 1:
    raise SystemExit(
        f"trace row count mismatch: expected {n_queries + 1}, got {len(rows)}"
    )
if rows[0] != ["query_id", "d_hat_l2", "oracle_decision"]:
    raise SystemExit(f"unexpected trace header: {rows[0]}")
PY

  ./build/data_hnsw_range_recall --config "$data_hnsw_config" --gt "$gt_json" \
    --output "$data_hnsw_range_recall_json"
  python3 - "$data_hnsw_range_recall_json" "$n_queries" <<'PY'
import json
import math
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    result = json.load(f)
n_queries = int(sys.argv[2])
required = (
    "dataset",
    "n_queries",
    "data_hnsw_path",
    "data_hnsw_ef",
    "data_hnsw_margin_delta",
    "false_empty_count",
    "false_empty_hit_count",
    "total_GT_hits",
    "range_recall_loss",
    "skip_count",
    "skip_rate",
    "per_query",
)
for key in required:
    if key not in result:
        raise SystemExit(f"missing data_hnsw_range_recall key: {key}")
if result["n_queries"] != n_queries:
    raise SystemExit(
        f"range recall n_queries mismatch: expected={n_queries} "
        f"actual={result['n_queries']}"
    )
per_query = result["per_query"]
if len(per_query) != n_queries:
    raise SystemExit(
        f"range recall per_query length mismatch: expected={n_queries} "
        f"actual={len(per_query)}"
    )
skip_count = sum(1 for row in per_query if row["oracle_decision"])
false_empty = [row for row in per_query if row["false_empty"]]
false_empty_hits = sum(int(row["gt_count"]) for row in false_empty)
total_hits = sum(int(row["gt_count"]) for row in per_query)
if result["skip_count"] != skip_count:
    raise SystemExit("range recall skip_count does not match per_query decisions")
if result["false_empty_count"] != len(false_empty):
    raise SystemExit("range recall false_empty_count does not match per_query")
if result["false_empty_hit_count"] != false_empty_hits:
    raise SystemExit("range recall false_empty_hit_count does not match per_query")
if result["total_GT_hits"] != total_hits:
    raise SystemExit("range recall total_GT_hits does not match per_query")
expected_loss = 0.0 if total_hits == 0 else false_empty_hits / float(total_hits)
if not math.isclose(
    result["range_recall_loss"], expected_loss, rel_tol=0.0, abs_tol=1e-12
):
    raise SystemExit("range recall loss formula mismatch")
expected_skip_rate = 0.0 if n_queries == 0 else skip_count / float(n_queries)
if not math.isclose(result["skip_rate"], expected_skip_rate, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("range recall skip_rate formula mismatch")
PY

  rm -f "$data_hnsw_artifact"
  ./build/search_stats --config "$data_hnsw_config" --gt "$gt_json" \
    >"$data_hnsw_missing_out" 2>"$data_hnsw_missing_err"
  python3 - "$search_stats_out" "$data_hnsw_missing_out" "$data_hnsw_missing_err" <<'PY'
import math
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")

def parse(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = line_re.search(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values

baseline = parse(sys.argv[1])
missing = parse(sys.argv[2])
stderr_lines = [
    line for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(stderr_lines) != 1 or "data HNSW artifact unavailable" not in stderr_lines[0]:
    raise SystemExit(f"expected one data HNSW fallback warning, got {stderr_lines}")
for key in ("recall", "dist_comp", "per_query_zero"):
    if not math.isclose(float(baseline[key]), float(missing[key]), rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"missing-artifact fallback changed {key}: "
            f"baseline={baseline[key]} missing={missing[key]}"
        )
if int(float(missing.get("data_hnsw_skip_count", "-1"))) != 0:
    raise SystemExit("missing-artifact fallback should have zero data_hnsw skips")
PY

  cmake -S . -B build_prune_on -DCMAKE_BUILD_TYPE=Release \
    -DPRUNE_BREAKDOWN_STATS=ON -DDUMP_LB_SLAB=OFF \
    >/tmp/build_search_entrypoints_cmake_on.log
  cmake --build build_prune_on --target search_stats -j \
    >/tmp/build_search_entrypoints_build_on.log
  ./build_prune_on/search_stats --config "$config" --gt "$gt_json" \
    >"$search_stats_on_out"

  python3 - "$search_stats_on_out" <<'PY'
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")
values = {}
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
for line in text.splitlines():
    match = line_re.search(line)
    if match:
        values[match.group(1)] = match.group(2)
for key in (
    "cell_bytes_read",
    "zero_result_bytes_read",
    "nonzero_result_bytes_read",
):
    if key not in values:
        raise SystemExit(f"missing PRUNE ON byte-accounting key: {key}")
cell_bytes_read = int(float(values["cell_bytes_read"]))
class_bytes_read = int(float(values["zero_result_bytes_read"])) + int(
    float(values["nonzero_result_bytes_read"])
)
if cell_bytes_read <= 0:
    raise SystemExit(
        f"PRUNE ON cell_bytes_read should be populated, got {cell_bytes_read}"
    )
if class_bytes_read != cell_bytes_read:
    raise SystemExit(
        f"PRUNE ON byte partition mismatch: classes={class_bytes_read} "
        f"cell_bytes_read={cell_bytes_read}"
    )
PY

  python3 - "$gt_json" "$radius_positive_gt_json" "$run_dir/deep1m_radius_positive" <<'PY'
import json
import os
import pathlib
import shutil
import struct
import sys

source_gt, out_gt, new_prefix = sys.argv[1:4]
with open(source_gt, "r", encoding="utf-8") as f:
    data = json.load(f)

source_prefix = data["index"]["prefix"]
source_dir = pathlib.Path(source_prefix + "_diskrange")
target_dir = pathlib.Path(new_prefix + "_diskrange")
target_dir.mkdir(parents=True, exist_ok=False)

for name in ("_cluster_file.bin", "_hnsw_file.bin"):
    os.symlink(source_dir / name, target_dir / name)
shutil.copyfile(source_dir / "_metadata_file.bin", target_dir / "_metadata_file.bin")

metadata_path = target_dir / "_metadata_file.bin"
with open(metadata_path, "r+b") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("bad metadata magic in radius-positive fixture")
    version = struct.unpack("<Q", f.read(8))[0]
    if version != 6:
        raise SystemExit(f"bad metadata schema version: {version}")
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs, vec_dtype = struct.unpack(
        "<QQQQQQQ", f.read(56)
    )
    if vec_dtype != 0:
        raise SystemExit(f"radius-positive fixture expects float metadata, got vec_dtype={vec_dtype}")
    radii_offset = 8 + 8 + 56 + 8 * cluster_num + 4 * cluster_num * dim
    f.seek(radii_offset)
    f.write(struct.pack(f"<{cluster_num}f", *([0.0] * cluster_num)))

data["index"]["prefix"] = new_prefix
data["radius_squared"] = 0.0
with open(out_gt, "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

  ./build/search_stats --config "$config" --gt "$radius_positive_gt_json" \
    >"$radius_positive_out"
  python3 - "$radius_positive_out" "$n_queries" <<'PY'
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")
values = {}
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
n_queries = int(sys.argv[2])
for line in text.splitlines():
    match = line_re.search(line)
    if match:
        values[match.group(1)] = match.group(2)
for key in (
    "triangle_candidate_clusters",
    "triangle_pruned_clusters",
    "radius_prescreened_clusters",
    "zero_result_radius_prescreened",
    "nonzero_result_radius_prescreened",
    "zero_result_distinct_qdot_used",
    "nonzero_result_distinct_qdot_used",
    "zero_result_surviving",
    "nonzero_result_surviving",
    "dist_comp",
    "per_query_zero",
    "mean_keep_after_filter",
):
    if key not in values:
        raise SystemExit(f"missing radius-positive key: {key}")

candidates = int(float(values["triangle_candidate_clusters"]))
pruned = int(float(values["triangle_pruned_clusters"]))
radius = int(float(values["radius_prescreened_clusters"]))
class_radius = int(float(values["zero_result_radius_prescreened"])) + int(
    float(values["nonzero_result_radius_prescreened"])
)
distinct_qdot = int(float(values["zero_result_distinct_qdot_used"])) + int(
    float(values["nonzero_result_distinct_qdot_used"])
)
surviving = int(float(values["zero_result_surviving"])) + int(
    float(values["nonzero_result_surviving"])
)
if candidates != n_queries * 245:
    raise SystemExit(f"radius-positive candidate count changed: {candidates}")
if radius != candidates or class_radius != radius:
    raise SystemExit(
        f"radius-positive pre-screen did not cover every candidate: "
        f"candidates={candidates} radius={radius} class_radius={class_radius}"
    )
if pruned != 0 or surviving != 0:
    raise SystemExit(
        f"radius-positive partition expected only pre-screened candidates: "
        f"pruned={pruned} surviving={surviving}"
    )
if distinct_qdot != 0:
    raise SystemExit(
        f"radius-positive all-pre-screened query should skip q-dot, got {distinct_qdot}"
    )
if int(float(values["dist_comp"])) != 0:
    raise SystemExit(f"radius-positive dist_comp expected 0, got {values['dist_comp']}")
if int(float(values["per_query_zero"])) != n_queries:
    raise SystemExit(
        f"radius-positive all queries should be zero-result, got {values['per_query_zero']}"
    )
if float(values["mean_keep_after_filter"]) != 0.0:
    raise SystemExit(
        f"radius-positive mean_keep_after_filter expected 0, got "
        f"{values['mean_keep_after_filter']}"
    )
PY

  cmake -S . -B build_prune_off -DCMAKE_BUILD_TYPE=Release \
    -DPRUNE_BREAKDOWN_STATS=OFF -DDUMP_LB_SLAB=OFF \
    >/tmp/build_search_entrypoints_cmake_off.log
  cmake --build build_prune_off --target search_stats search -j \
    >/tmp/build_search_entrypoints_build_off.log
  ./build_prune_off/search_stats --config "$config" --gt "$gt_json" \
    >"$search_stats_off_out"

  python3 - "$search_stats_off_out" "$cluster_num" <<'PY'
import re
import sys
from pathlib import Path

line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
cluster_num = int(sys.argv[2])
if "[Search phase timing]" not in text:
    raise SystemExit("PRUNE_BREAKDOWN_STATS=OFF search_stats missing timing section")
values = {}
for line in text.splitlines():
    match = line_re.search(line)
    if match:
        values[match.group(1)] = match.group(2)
for key in (
    "triangle_candidate_clusters",
    "triangle_pruned_clusters",
    "radius_prescreened_clusters",
    "radius_prescreen_rate",
    "zero_result_radius_prescreened",
    "nonzero_result_radius_prescreened",
    "zero_result_surviving",
    "nonzero_result_surviving",
    "zero_result_distinct_qdot_used",
    "nonzero_result_distinct_qdot_used",
    "zero_result_mean_distinct_qdot_used",
    "nonzero_result_mean_distinct_qdot_used",
    "zero_result_dist_comp",
    "nonzero_result_dist_comp",
    "dist_comp",
):
    if key not in values:
        raise SystemExit(f"missing OFF dist_comp key: {key}")
class_dist_comp = int(float(values["zero_result_dist_comp"])) + int(
    float(values["nonzero_result_dist_comp"])
)
if class_dist_comp != int(float(values["dist_comp"])):
    raise SystemExit(
        f"OFF dist_comp partition mismatch: classes={class_dist_comp} dist_comp={values['dist_comp']}"
    )
n_queries = int(float(values["zero_result_n_queries"])) + int(
    float(values["nonzero_result_n_queries"])
)
total_candidates = int(float(values["zero_result_candidates"])) + int(
    float(values["nonzero_result_candidates"])
)
class_radius_prescreened = int(float(values["zero_result_radius_prescreened"])) + int(
    float(values["nonzero_result_radius_prescreened"])
)
if class_radius_prescreened != int(float(values["radius_prescreened_clusters"])):
    raise SystemExit(
        f"OFF radius_prescreened partition mismatch: classes={class_radius_prescreened} "
        f"total={values['radius_prescreened_clusters']}"
    )
if class_radius_prescreened < 0 or class_radius_prescreened > total_candidates:
    raise SystemExit(
        f"OFF radius_prescreened out of range: radius={class_radius_prescreened} "
        f"candidates={total_candidates}"
    )
surviving_total = int(float(values["zero_result_surviving"])) + int(
    float(values["nonzero_result_surviving"])
)
partition_total = (
    int(float(values["radius_prescreened_clusters"]))
    + int(float(values["triangle_pruned_clusters"]))
    + surviving_total
)
if partition_total != int(float(values["triangle_candidate_clusters"])):
    raise SystemExit(
        f"OFF candidate partition mismatch: partition={partition_total} "
        f"candidate={values['triangle_candidate_clusters']}"
    )
total_distinct_qdot_used = int(float(values["zero_result_distinct_qdot_used"])) + int(
    float(values["nonzero_result_distinct_qdot_used"])
)
active_candidates = total_candidates - class_radius_prescreened
if total_distinct_qdot_used < active_candidates:
    raise SystemExit(
        f"OFF distinct_qdot_used total must cover post-prescreen candidates: "
        f"distinct={total_distinct_qdot_used} active_candidates={active_candidates}"
    )
if total_distinct_qdot_used > cluster_num * n_queries:
    raise SystemExit(
        f"OFF distinct_qdot_used total exceeds query cluster universe: "
        f"distinct={total_distinct_qdot_used} cluster_num={cluster_num} "
        f"queries={n_queries}"
    )
for prefix in ("zero_result", "nonzero_result"):
    class_queries = int(float(values[f"{prefix}_n_queries"]))
    candidates = int(float(values[f"{prefix}_candidates"]))
    radius_prescreened = int(float(values[f"{prefix}_radius_prescreened"]))
    distinct_qdot_used = int(float(values[f"{prefix}_distinct_qdot_used"]))
    active = candidates - radius_prescreened
    if radius_prescreened < 0 or radius_prescreened > candidates:
        raise SystemExit(
            f"OFF {prefix}_radius_prescreened out of range: "
            f"radius={radius_prescreened} candidates={candidates}"
        )
    if distinct_qdot_used < active:
        raise SystemExit(
            f"OFF {prefix}_distinct_qdot_used must cover post-prescreen candidates: "
            f"distinct={distinct_qdot_used} active_candidates={active}"
        )
    if distinct_qdot_used > cluster_num * class_queries:
        raise SystemExit(
            f"OFF {prefix}_distinct_qdot_used exceeds per-query cluster universe: "
            f"distinct={distinct_qdot_used} cluster_num={cluster_num} "
            f"queries={class_queries}"
        )
for key in (
    "slab_only_prune_rate",
    "pca_only_prune_rate",
    "combined_prune_rate",
    "slab_only_pruned_clusters",
    "pca_only_pruned_clusters",
    "combined_pruned_clusters",
    "cell_candidate_vectors",
    "cell_pruned_vectors",
    "cell_kept_vectors",
    "zero_kept_cells",
    "zero_kept_surviving_clusters",
    "io_bytes_reduction",
):
    if key not in values:
        raise SystemExit(f"missing OFF zeroed layer key: {key}")
    if float(values[key]) != 0.0:
        raise SystemExit(f"expected {key}=0 under PRUNE_BREAKDOWN_STATS=OFF, got {values[key]}")
PY
}

case "$mode" in
  --static-only)
    run_static_checks
    ;;
  --build-only)
    run_build_checks
    ;;
  all)
    run_static_checks
    run_build_checks
    run_runtime_checks
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
