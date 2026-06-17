#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/deep1m_disk.config"
gt="experiments/range_gt/output/deep1m_0.2632_gt.json"

python3 - <<'PY'
from pathlib import Path

checks = {
    "lib/ResolvedConfig.h": [
        "fast_path_enabled",
        "fast_path_mode",
        "fast_path_nprobe_N",
        "sqg_path",
        "rabitq_1bit_codes_path",
        "rabitq_8bit_codes_path",
    ],
    "lib/ConfigReader.h": [
        "fast_path_enabled",
        "fast_path_mode",
        "sqg_ef_search",
        "fast_path_nprobe_N",
    ],
    "lib/IndexPaths.h": [
        "rabitq_1bit_codes",
        "_rabitq_8bit_codes_file.bin",
    ],
    "lib/SQGCentroidIndex.h": [
        "class SQGCentroidIndex",
        "const symqg::FHTRotator& rotator",
    ],
    "lib/RBQCodeStorage.h": [
        "class RBQCodeStorage",
        "RBQCodeFileHeader",
        "no vec-id array",
    ],
    "lib/FastPathPhase3AsyncFetch.h": [
        "async_fetch_8bit_codes",
        "BorderlineRef",
    ],
    "lib/FastPathEmptinessOracle.h": [
        "class FastPathEmptinessOracle",
        "FastPathOracleResult evaluate",
        "evaluate_exact",
        "ClusterReader* cluster_reader_",
        "submit_and_drain(omp_get_thread_num()",
    ],
    "lib/RunResult.h": [
        "FastPathTraceRecord",
        "false_empty_query_rate",
        "range_recall_loss_hit_weighted",
    ],
    "lib/DiskRange.h": [
        "fast_path_oracle",
        "fast_path_skip_count",
        "build_fast_path_artifacts",
        "[fast-path] enabled mode=",
    ],
    "expr/search_stats/main.cpp": [
        "write_fast_path_trace",
        "fast_path_trace_path",
        "range_recall_loss_hit_weighted",
    ],
}

for path, markers in checks.items():
    text = Path(path).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise SystemExit(f"missing marker {marker!r} in {path}")

cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
for marker in (
    "add_subdirectory(third/symqglib)",
    "add_subdirectory(third/RaBitQ-Library)",
    "rabitq_headers",
    "add_executable(fast_path_emptiness_oracle_unit_test tests/fast_path_emptiness_oracle_unit_test.cpp)",
):
    if marker not in cmake:
        raise SystemExit(f"missing CMake marker: {marker}")

sqg = Path("third/symqglib/qg/qg.hpp").read_text(encoding="utf-8")
for marker in (
    "const FHTRotator& rotator() const",
    "padded_dim() const",
    "QuantizedGraph::search(\n    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results\n) const",
):
    if marker not in sqg:
        raise SystemExit(f"missing SQG accessor marker: {marker}")

oracle = Path("lib/FastPathEmptinessOracle.h").read_text(encoding="utf-8")
for marker in (
    "rabitqlib::SplitBatchQuery<float>",
    "rabitqlib::split_batch_estdist",
    "rabitqlib::split_distance_boosting",
    "std::isfinite(low_distance[i])",
    "std::isfinite(est_dist)",
    "fast_path_mode",
    "evaluate_exact",
    "ClusterReader* cluster_reader_",
    "submit_and_drain(omp_get_thread_num()",
):
    if marker not in oracle:
        raise SystemExit(f"missing FastScan marker: {marker}")
exact_body = oracle[
    oracle.index("FastPathOracleResult evaluate_exact"):
    oracle.index("FastPathOracleResult evaluate_rbq")
]
for forbidden in (
    ".rotator()",
    ".rotate(",
    "SplitBatchQuery",
    "split_batch_estdist",
    "split_distance_boosting",
):
    if forbidden in exact_body:
        raise SystemExit(f"exact mode raw-L2 path still uses rotated/RBQ logic: {forbidden}")
for forbidden in (
    "split_single_estdist",
    "split_single_fulldist",
    "sqg_mutex_",
    "std::lock_guard<std::mutex>",
):
    if forbidden in oracle:
        raise SystemExit(f"forbidden serialized/single-vector marker remains: {forbidden}")

storage = Path("lib/RBQCodeStorage.h").read_text(encoding="utf-8")
for marker in (
    "quantize_split_batch",
    "BatchDataMap<float>",
    "batch_code_ptr",
):
    if marker not in storage:
        raise SystemExit(f"missing RBQ batch-storage marker: {marker}")
if "quantize_split_single" in storage:
    raise SystemExit("RBQ storage still uses single-vector quantization")
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDUMP_LB_SLAB=OFF \
  >/tmp/fast_path_cmake.log
cmake --build build --target fast_path_emptiness_oracle_unit_test -j \
  >/tmp/fast_path_build.log
./build/fast_path_emptiness_oracle_unit_test "$config" "$gt"
