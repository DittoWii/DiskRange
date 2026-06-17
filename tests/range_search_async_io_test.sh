#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/range_search_async_io.XXXXXX)"
index_prefix="$run_dir/deep1m"
index_dir="${index_prefix}_diskrange"
gt_json="$run_dir/deep1m_0.2632_gt.json"
query_file="$run_dir/deep1m_query_subset.fbin"
n_queries=8

if pkg-config --exists openblas; then
  openblas_cflags="$(pkg-config --cflags openblas)"
  openblas_libs="$(pkg-config --libs openblas)"
else
  openblas_cflags="-I/usr/include/x86_64-linux-gnu/openblas-pthread"
  openblas_libs="-L/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas"
fi

cleanup() {
  rm -rf "$run_dir" "$index_dir"
}
trap cleanup EXIT

test -f "$source_gt_json"
test -f "$config"

python3 - "$source_gt_json" "$gt_json" "$index_prefix" "$query_file" "$n_queries" <<'PY'
import json, struct, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
query_path = data["provenance"]["query_path"]
requested = int(sys.argv[5])
selected = [
    idx for idx, count in enumerate(data["per_query_counts"]) if int(count) > 0
][:requested]
if len(selected) != requested:
    raise SystemExit(f"needed {requested} nonzero queries, found {len(selected)}")

dim = int(data["dim"])
row_bytes = dim * 4
with open(query_path, "rb") as src, open(sys.argv[4], "wb") as dst:
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

counts = [int(data["per_query_counts"][idx]) for idx in selected]
sorted_counts = sorted(counts)
data["n_queries"] = len(selected)
data["total_in_range"] = sum(counts)
data["per_query_counts"] = counts
data["per_query_count_stats"] = {
    "min": min(counts) if counts else 0,
    "median": (
        sorted_counts[(len(sorted_counts) - 1) // 2]
        + sorted_counts[len(sorted_counts) // 2]
    ) / 2.0 if sorted_counts else 0.0,
    "max": max(counts) if counts else 0,
    "zero_hit_queries": sum(1 for count in counts if count == 0),
}
data["index"]["prefix"] = sys.argv[3]
data["provenance"]["query_path"] = sys.argv[4]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/range_search_async_io_cmake.log
cmake --build build >/tmp/range_search_async_io_build.log

if grep -rn 'omp critical' lib/DiskRange.h >/tmp/range_search_async_io_static.log; then
  cat /tmp/range_search_async_io_static.log >&2
  exit 1
fi
if grep -rn 'posixDirectReadCluster\|readCluster(' lib/ >/tmp/range_search_async_io_static.log; then
  cat /tmp/range_search_async_io_static.log >&2
  exit 1
fi
if grep -rn 'IORING_OP_READ\b' lib/ | grep -v 'IORING_OP_READ_FIXED' >/tmp/range_search_async_io_static.log; then
  cat /tmp/range_search_async_io_static.log >&2
  exit 1
fi
grep -rn 'io_uring_submit' lib/ >/dev/null
if grep -F 'io_uring_submit expected 1 submitted SQE' lib/ClusterIO.h >/tmp/range_search_async_io_static.log; then
  cat /tmp/range_search_async_io_static.log >&2
  exit 1
fi
nm build/main | grep io_uring >/dev/null

check_uring_cpp="$run_dir/check_io_uring.cpp"
check_uring_bin="$run_dir/check_io_uring"
cat >"$check_uring_cpp" <<'CPP'
#include <cerrno>
#include <cstring>
#include <iostream>
#include <liburing.h>

int main()
{
  io_uring ring;
  int ret = io_uring_queue_init(1, &ring, 0);
  if (ret < 0)
  {
    std::cerr << "io_uring_queue_init failed: " << ret << " ("
              << std::strerror(-ret) << ")\n";
    return 1;
  }
  io_uring_queue_exit(&ring);
  return 0;
}
CPP
g++ -std=c++17 -Ithird/liburing/src/include "$check_uring_cpp" \
  build/third/liburing/libliburing_vendored.a -o "$check_uring_bin"
if ! "$check_uring_bin" >/tmp/range_search_async_io_preflight.out 2>/tmp/range_search_async_io_preflight.err; then
  cat /tmp/range_search_async_io_preflight.err >&2
  if [[ "${DISKRANGE_ALLOW_IO_URING_SKIP:-0}" == "1" ]]; then
    echo "range_search_async_io_test: SKIP runtime E2E because io_uring is unavailable in this environment" >&2
    exit 0
  fi
  echo "range_search_async_io_test: FAIL because io_uring is unavailable; set DISKRANGE_ALLOW_IO_URING_SKIP=1 only for sandboxed local static checks" >&2
  exit 1
fi

normal_stdout="$(mktemp)"
normal_stderr="$(mktemp)"
./build/main --config "$config" --gt "$gt_json" >"$normal_stdout" 2>"$normal_stderr"
normal_results="$(mktemp)"
python3 tests/extract_run_result.py "$normal_stdout" "$normal_stderr" >"$normal_results"

test -s "$index_dir/_cluster_file.bin"
test -s "$index_dir/_metadata_file.bin"
test -s "$index_dir/_hnsw_file.bin"

python3 - "$normal_results" "$n_queries" <<'PY'
import sys
values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
required = {
    "recall", "build_time", "search_time", "dist_comp",
    "triangle_prune_rate",
    "triangle_candidate_clusters",
    "triangle_pruned_clusters",
    "slab_pruned_clusters",
    "box_residual_pruned_clusters",
    "mean_keep_after_filter",
    "per_query_min", "per_query_median", "per_query_max", "per_query_zero",
}
missing = required - values.keys()
if missing:
    raise SystemExit(f"missing RunResult keys: {sorted(missing)}")
recall = float(values["recall"])
search_time = float(values["search_time"])
dist_comp = int(values["dist_comp"])
triangle_candidate_clusters = int(values["triangle_candidate_clusters"])
triangle_pruned_clusters = int(values["triangle_pruned_clusters"])
slab_pruned_clusters = int(values["slab_pruned_clusters"])
box_residual_pruned_clusters = int(values["box_residual_pruned_clusters"])
mean_keep_after_filter = float(values["mean_keep_after_filter"])
if not (0.0 < recall <= 1.0001):
    raise SystemExit(f"recall out of bounds: {recall}")
if search_time <= 0.0:
    raise SystemExit(f"search_time must be positive: {search_time}")
if dist_comp <= 0:
    raise SystemExit(f"dist_comp must be positive: {dist_comp}")
if triangle_candidate_clusters <= 0:
    raise SystemExit(
        f"triangle_candidate_clusters must be positive: {triangle_candidate_clusters}"
    )
if slab_pruned_clusters + box_residual_pruned_clusters != triangle_pruned_clusters:
    raise SystemExit(
        "prune partition mismatch: "
        f"{slab_pruned_clusters} + {box_residual_pruned_clusters} != {triangle_pruned_clusters}"
    )
if mean_keep_after_filter < 0.0:
    raise SystemExit(
        f"mean_keep_after_filter must be non-negative: {mean_keep_after_filter}"
    )
print(f"normal recall={recall:.6f} search_time={search_time:.6f} dist_comp={dist_comp}")
PY

search_only_cpp="$run_dir/search_only.cpp"
search_only_bin="$run_dir/search_only"
cat >"$search_only_cpp" <<'CPP'
#include <exception>
#include <iomanip>
#include <iostream>

#include "lib/ConfigLoader.h"
#include "lib/DiskRange.h"

int r_server_id = 0;

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: search_only <config> <gt>" << std::endl;
    return 2;
  }

  try
  {
    ResolvedConfig config = load_resolved_config(argv[1], argv[2]);
    DiskRange index;
    RunResult result = index.search(config);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "recall = " << result.recall << '\n';
    std::cout << "search_time = " << result.search_time << '\n';
    std::cout << "dist_comp = " << result.dist_comp << '\n';
    std::cout << "per_query_min = " << result.per_query_min << '\n';
    std::cout << "per_query_median = " << result.per_query_median << '\n';
    std::cout << "per_query_max = " << result.per_query_max << '\n';
    std::cout << "per_query_zero = " << result.per_query_zero << '\n';
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
CPP
g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
  -Ithird/liburing/src/include -Ithird/eigen -Ithird/symqglib \
  -Ithird/RaBitQ-Library/include -DNDEBUG -mtune=native -march=native -mavx2 -pthread \
  -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc \
  -fno-builtin-realloc -fno-builtin-free -fopenmp-simd -funroll-loops \
  -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS $openblas_cflags \
  -DUSE_AVX2 "$search_only_cpp" build/third/liburing/libliburing_vendored.a \
  $openblas_libs \
  -o "$search_only_bin"

full_config="$(mktemp)"
python3 - "$config" "$full_config" <<'PY'
import sys
replacements = {"K": "10000"}
seen = set()
with open(sys.argv[1], "r", encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as dst:
    for line in src:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            dst.write(line)
            continue
        key, *rest = stripped.split()
        if key in replacements:
            dst.write(f"{key} {replacements[key]}\n")
            seen.add(key)
        else:
            dst.write(line)
missing = set(replacements) - seen
if missing:
    raise SystemExit(f"missing keys in config: {sorted(missing)}")
PY

full_stdout="$(mktemp)"
full_stderr="$(mktemp)"
mv "$index_dir/_hnsw_file.bin" "$index_dir/_hnsw_file.bin.bak"
if ! "$search_only_bin" "$full_config" "$gt_json" >"$full_stdout" 2>"$full_stderr"; then
  echo "full-candidate search unexpectedly failed" >&2
  cat "$full_stderr" >&2
  exit 1
fi
mv "$index_dir/_hnsw_file.bin.bak" "$index_dir/_hnsw_file.bin"
python3 - "$full_stdout" <<'PY'
import sys
values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
recall = float(values["recall"])
if recall < 0.999:
    raise SystemExit(f"full-candidate recall below 0.999: {recall}")
print(f"full_candidate recall={recall:.6f}")
PY

bad_config="$(mktemp)"
python3 - "$config" "$bad_config" <<'PY'
import sys
with open(sys.argv[1], "r", encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as dst:
    wrote = False
    for line in src:
        if line.strip().startswith("io_queue_depth "):
            dst.write("io_queue_depth 0\n")
            wrote = True
        else:
            dst.write(line)
    if not wrote:
        dst.write("io_queue_depth 0\n")
PY
if ./build/main --config "$bad_config" --gt "$gt_json" >/tmp/range_search_async_io_bad.out 2>/tmp/range_search_async_io_bad.err; then
  echo "io_queue_depth 0 unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'io_queue_depth' /tmp/range_search_async_io_bad.err >/dev/null

cluster_file="$index_dir/_cluster_file.bin"
cp "$cluster_file" "$cluster_file.bak"
truncate -s 0 "$cluster_file"
if "$search_only_bin" "$full_config" "$gt_json" >/tmp/range_search_async_io_trunc.out 2>/tmp/range_search_async_io_trunc.err; then
  echo "truncated cluster file unexpectedly succeeded" >&2
  exit 1
fi
grep -E 'cluster|cqe|res' /tmp/range_search_async_io_trunc.err >/dev/null
mv "$cluster_file.bak" "$cluster_file"

normal_recall="$(awk -F' = ' '$1=="recall"{print $2}' "$normal_results")"
normal_search_time="$(awk -F' = ' '$1=="search_time"{print $2}' "$normal_results")"
full_recall="$(awk -F' = ' '$1=="recall"{print $2}' "$full_stdout")"
echo "range_search_async_io_test: PASS recall=${normal_recall} full_candidate_recall=${full_recall} async_search_time=${normal_search_time}"
