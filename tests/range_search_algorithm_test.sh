#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/range_search_algorithm.XXXXXX)"
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
    )
    / 2.0
    if sorted_counts
    else 0.0,
    "max": max(counts) if counts else 0,
    "zero_hit_queries": sum(1 for count in counts if count == 0),
}
data["index"]["prefix"] = sys.argv[3]
data["provenance"]["query_path"] = sys.argv[4]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/range_search_algorithm_cmake.log
cmake --build build >/tmp/range_search_algorithm_build.log

normal_stdout="$(mktemp)"
normal_stderr="$(mktemp)"
./build/main --config "$config" --gt "$gt_json" >"$normal_stdout" 2>"$normal_stderr"
normal_results="$(mktemp)"
python3 tests/extract_run_result.py "$normal_stdout" "$normal_stderr" >"$normal_results"

test -s "$index_dir/_cluster_file.bin"
test -s "$index_dir/_metadata_file.bin"
test -s "$index_dir/_hnsw_file.bin"
test "$(wc -l < "$normal_results")" -ge 14
if grep -Fq '[STUB]' "$normal_stderr"; then
  echo "normal algorithm run still emitted STUB marker" >&2
  exit 1
fi

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
triangle_prune_rate = float(values["triangle_prune_rate"])
triangle_candidate_clusters = int(values["triangle_candidate_clusters"])
triangle_pruned_clusters = int(values["triangle_pruned_clusters"])
slab_pruned_clusters = int(values["slab_pruned_clusters"])
box_residual_pruned_clusters = int(values["box_residual_pruned_clusters"])
mean_keep_after_filter = float(values["mean_keep_after_filter"])
per_query_min = int(values["per_query_min"])
per_query_median = float(values["per_query_median"])
per_query_max = int(values["per_query_max"])
per_query_zero = int(values["per_query_zero"])
n_queries = int(sys.argv[2])
if not (0.0 < recall <= 1.0001):
    raise SystemExit(f"normal recall out of bounds: {recall}")
if search_time <= 0.0:
    raise SystemExit(f"search_time must be positive: {search_time}")
if dist_comp <= 0:
    raise SystemExit(f"dist_comp must be positive: {dist_comp}")
if not (0.0 <= triangle_prune_rate <= 1.0):
    raise SystemExit(f"triangle_prune_rate out of bounds: {triangle_prune_rate}")
if triangle_candidate_clusters <= 0:
    raise SystemExit(
        f"triangle_candidate_clusters must be positive: {triangle_candidate_clusters}"
    )
if triangle_pruned_clusters < 0:
    raise SystemExit(f"triangle_pruned_clusters must be non-negative: {triangle_pruned_clusters}")
if slab_pruned_clusters + box_residual_pruned_clusters != triangle_pruned_clusters:
    raise SystemExit(
        "prune partition mismatch: "
        f"{slab_pruned_clusters} + {box_residual_pruned_clusters} != {triangle_pruned_clusters}"
    )
if mean_keep_after_filter < 0.0:
    raise SystemExit(
        f"mean_keep_after_filter must be non-negative: {mean_keep_after_filter}"
    )
if per_query_min > per_query_max:
    raise SystemExit(f"bad min/max stats: {per_query_min}, {per_query_max}")
if per_query_zero > n_queries:
    raise SystemExit(f"too many zero-hit queries: {per_query_zero}")
if not (per_query_min <= per_query_median <= per_query_max):
    raise SystemExit(
        f"median outside range: {per_query_min}, {per_query_median}, {per_query_max}"
    )
print(
    f"normal recall={recall:.6f} search_time={search_time:.6f} dist_comp={dist_comp}"
)
PY

search_only_cpp="$run_dir/search_only.cpp"
search_only_bin="$run_dir/search_only"
python3 - "$search_only_cpp" <<'PY'
import sys
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

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
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
''')
PY
g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
  -Ithird/liburing/src/include -Ithird/eigen -Ithird/symqglib \
  -Ithird/RaBitQ-Library/include -DNDEBUG -mtune=native -march=native -mavx2 -pthread -mfma -msse2 \
  -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc \
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
with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
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
if grep -Fq '[STUB]' "$full_stderr"; then
  echo "full-candidate run still emitted STUB marker" >&2
  exit 1
fi

python3 - "$full_stdout" <<'PY'
import sys
values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
recall = float(values["recall"])
search_time = float(values["search_time"])
dist_comp = int(values["dist_comp"])
if recall < 0.999:
    raise SystemExit(f"full-candidate recall too low: {recall}")
if recall > 1.0001:
    raise SystemExit(f"full-candidate recall too high: {recall}")
if search_time <= 0.0 or dist_comp <= 0:
    raise SystemExit(
        f"full-candidate metrics invalid: search_time={search_time}, dist_comp={dist_comp}"
    )
print(
    f"full recall={recall:.6f} search_time={search_time:.6f} dist_comp={dist_comp}"
)
PY

mv "$index_dir/_cluster_file.bin" "$index_dir/_cluster_file.bin.bak"
missing_cluster_stdout="$(mktemp)"
missing_cluster_stderr="$(mktemp)"
if "$search_only_bin" "$full_config" "$gt_json" \
  >"$missing_cluster_stdout" 2>"$missing_cluster_stderr"; then
  echo "missing cluster file unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "$index_dir/_cluster_file.bin" "$missing_cluster_stderr"
mv "$index_dir/_cluster_file.bin.bak" "$index_dir/_cluster_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.bak"
missing_metadata_stdout="$(mktemp)"
missing_metadata_stderr="$(mktemp)"
if "$search_only_bin" "$full_config" "$gt_json" \
  >"$missing_metadata_stdout" 2>"$missing_metadata_stderr"; then
  echo "missing metadata file unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "$index_dir/_metadata_file.bin" "$missing_metadata_stderr"
mv "$index_dir/_metadata_file.bin.bak" "$index_dir/_metadata_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
if data[:8] != b"DJMETA\0\0":
    raise SystemExit("metadata does not have schema magic")
data[32:40] = struct.pack("<Q", 10001)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
bad_metadata_stdout="$(mktemp)"
bad_metadata_stderr="$(mktemp)"
if "$search_only_bin" "$full_config" "$gt_json" \
  >"$bad_metadata_stdout" 2>"$bad_metadata_stderr"; then
  echo "corrupt metadata header unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "metadata cluster count mismatch" "$bad_metadata_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
n, d, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
    "<QQQQQQ", data[16:64]
)
if cluster_num == 0:
    raise SystemExit("metadata has no clusters")
data[64:72] = struct.pack("<Q", max_points + 1)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
bad_bucket_stdout="$(mktemp)"
bad_bucket_stderr="$(mktemp)"
if "$search_only_bin" "$full_config" "$gt_json" \
  >"$bad_bucket_stdout" 2>"$bad_bucket_stderr"; then
  echo "corrupt metadata bucket size unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "metadata bucket size out of range" "$bad_bucket_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
n, d, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
    "<QQQQQQ", data[16:64]
)
data[40:48] = struct.pack("<Q", max_points + 1)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
bad_max_points_stdout="$(mktemp)"
bad_max_points_stderr="$(mktemp)"
if "$search_only_bin" "$full_config" "$gt_json" \
  >"$bad_max_points_stdout" 2>"$bad_max_points_stderr"; then
  echo "corrupt metadata max_points unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "metadata max_points mismatch" "$bad_max_points_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

if grep -n '\[STUB\]' lib/DiskRange.h expr/main.cpp; then
  echo "STUB marker remains in source" >&2
  exit 1
fi
if grep -n 'id % 100000' lib/DiskRange.h; then
  echo "legacy sampling denominator remains in DiskRange.h" >&2
  exit 1
fi
if grep -n 'config\.gt\b\|config_reader\.gt\b' \
  lib/DiskRange.h expr/main.cpp lib/Kmeans.h lib/ConfigLoader.h lib/ConfigReader.h; then
  echo "legacy gt field remains in source" >&2
  exit 1
fi
if grep -n 'searchKnn(' lib/DiskRange.h; then
  echo "DiskRange.h must use searchBaseLayerST for external query vectors" >&2
  exit 1
fi
if grep -n 'InMemoryDiskRange' lib/DiskRange.h; then
  echo "DiskRange.h must not depend on InMemoryDiskRange" >&2
  exit 1
fi
one_level_count="$(grep -Ec 'one_level_kmeans\(config\);' lib/DiskRange.h || true)"
if [[ "$one_level_count" != "1" ]]; then
  echo "DiskRange.h should call one_level_kmeans(config) exactly once" >&2
  exit 1
fi

main_size="$(stat -c '%s' build/main)"
test "$main_size" -gt 300000
