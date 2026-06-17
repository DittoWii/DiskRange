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
        "data_hnsw_enabled",
        "data_hnsw_margin_delta",
        "data_hnsw_path",
        "data_hnsw_trace_path",
    ],
    "lib/ConfigReader.h": [
        "data_hnsw_enabled",
        "data_hnsw_margin_delta must be >= 0",
    ],
    "lib/DataHNSWOracle.h": [
        "struct DataHNSWOracleResult",
        "struct DataHNSWOracle",
        "std::unique_ptr<hnswlib::L2Space> space",
        "std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph",
        "DataHNSWOracleResult evaluate",
        "graph->setEf",
    ],
    "lib/data_hnsw_search.h": [
        "data_hnsw_1nn_squared",
        "searchBaseLayerST<false>",
    ],
    "lib/DiskRange.h": [
        "data_hnsw_oracle",
        "oracle.evaluate",
        "data_hnsw_skip_count",
        "data_hnsw_oracle_total_us",
        "data_hnsw_trace",
    ],
    "expr/search_stats/main.cpp": [
        "data_hnsw_skip_count",
        "data_hnsw_trace_path",
        "query_id,d_hat_l2,oracle_decision",
    ],
    "expr/data_hnsw_recall_bench/main.cpp": [
        "data_hnsw_1nn_squared",
    ],
}

for path, markers in checks.items():
    text = Path(path).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise SystemExit(f"missing marker {marker!r} in {path}")

cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
for marker in (
    "add_executable(data_hnsw_oracle_unit_test tests/data_hnsw_oracle_unit_test.cpp)",
    "add_executable(data_hnsw_range_recall expr/data_hnsw_range_recall/main.cpp)",
):
    if marker not in cmake:
        raise SystemExit(f"missing CMake marker: {marker}")
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDUMP_LB_SLAB=OFF \
  >/tmp/data_hnsw_cmake.log
cmake --build build --target data_hnsw_oracle_unit_test data_hnsw_recall_bench data_hnsw_range_recall -j \
  >/tmp/data_hnsw_build.log
./build/data_hnsw_oracle_unit_test "$config" "$gt"
