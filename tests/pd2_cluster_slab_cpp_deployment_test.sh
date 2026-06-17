#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/pd2_cluster_slab_cpp.XXXXXX)"
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
import json
import struct
import sys

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

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/pd2_cluster_slab_cpp_cmake.log
cmake --build build >/tmp/pd2_cluster_slab_cpp_build.log

config_probe_cpp="$run_dir/config_probe.cpp"
config_probe_bin="$run_dir/config_probe"
python3 - "$config_probe_cpp" <<'PY'
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <exception>
#include <iostream>

#include "lib/ConfigLoader.h"

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: config_probe <config> <gt>" << std::endl;
    return 2;
  }
  try
  {
    const ResolvedConfig config = load_resolved_config(argv[1], argv[2]);
    std::cout << "pca_rank = " << config.pca_rank << '\n';
    std::cout << "slab_count = " << config.slab_count << '\n';
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
  -Ithird/liburing/src/include -Ithird/eigen "$config_probe_cpp" \
  -o "$config_probe_bin"

default_slab_config="$run_dir/default_slab.config"
python3 - "$config" "$default_slab_config" <<'PY'
import sys

with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        if line.strip().startswith("slab_count "):
            continue
        dst.write(line)
PY
"$config_probe_bin" "$default_slab_config" "$gt_json" | grep -Fq "slab_count = 128"

too_large_slab_config="$run_dir/too_large_slab.config"
python3 - "$config" "$too_large_slab_config" <<'PY'
import sys

replaced = False
with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        stripped = line.strip()
        if stripped.startswith("slab_count "):
            dst.write("slab_count 10000\n")
            replaced = True
        else:
            dst.write(line)
    if not replaced:
        dst.write("slab_count 10000\n")
PY
too_large_stdout="$(mktemp)"
too_large_stderr="$(mktemp)"
if ./build/main --config "$too_large_slab_config" --gt "$gt_json" \
  >"$too_large_stdout" 2>"$too_large_stderr"; then
  echo "too-large slab_count config unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "slab_count 10000 must be < cluster_num 10000" "$too_large_stderr"
grep -Fq "need at least slab_count + 1 distinct clusters" "$too_large_stderr"

python3 - "lib/DiskRange.h" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
required_snippets = [
    "const bool use_slab = config.slab_cluster_prune_enabled &&\n                          cluster_reader.slab_count > 0;",
    "q_dot_centroids.size() != cluster_reader.cluster_num",
    "float slab_threshold = epsilon;",
    "slab_threshold = std::numeric_limits<float>::infinity();",
    "cluster_reader.pca_slab_lower_bound(",
    "cluster_id, q_dot_centroids, slab_threshold",
    "#ifdef DUMP_LB_SLAB",
    "radius_prescreened_clusters",
]
for snippet in required_snippets:
    if snippet not in source:
        raise SystemExit(f"missing expected DiskRange slab-guard snippet: {snippet!r}")
for forbidden in [
    "cluster_reader.pca_slab_lower_bound(cluster_id, q_dot_centroids);",
    "q_dot_blocks.resize",
]:
    if forbidden in source:
        raise SystemExit(f"found stale DiskRange snippet: {forbidden!r}")
PY

build_only_cpp="$run_dir/build_only.cpp"
build_only_bin="$run_dir/build_only"
python3 - "$build_only_cpp" <<'PY'
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <exception>
#include <iostream>

#include "lib/ConfigLoader.h"
#include "lib/DiskRange.h"

int r_server_id = 0;

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: build_only <config> <gt>" << std::endl;
    return 2;
  }

  try
  {
    ResolvedConfig config = load_resolved_config(argv[1], argv[2]);
    DiskRange index;
    index.build(config);
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
  -Ithird/RaBitQ-Library/include -DNDEBUG -mtune=native -march=native -mavx2 \
  -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc \
  -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free \
  -fopenmp-simd -funroll-loops -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS \
  $openblas_cflags -DUSE_AVX2 "$build_only_cpp" \
  build/third/liburing/libliburing_vendored.a $openblas_libs \
  -o "$build_only_bin"
"$build_only_bin" "$config" "$gt_json"

test -s "$index_dir/_cluster_file.bin"
test -s "$index_dir/_metadata_file.bin"
test -s "$index_dir/_hnsw_file.bin"

python3 - "$index_dir/_metadata_file.bin" "$gt_json" <<'PY'
import json
import struct
import sys

with open(sys.argv[1], "rb") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("bad metadata magic")
    version = struct.unpack("<Q", f.read(8))[0]
    if version != 5:
        raise SystemExit(f"bad metadata schema version: {version}")
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
    bucket_sizes = struct.unpack(f"<{cluster_num}Q", f.read(8 * cluster_num))
    f.seek(4 * cluster_num * dim, 1)
    f.seek(4 * cluster_num, 1)
    f.seek(4 * cluster_num * pca_rank * dim, 1)
    f.seek(4 * cluster_num * pca_rank, 1)
    f.seek(4 * cluster_num * pca_rank, 1)
    f.seek(4 * cluster_num, 1)
    f.seek(4 * cluster_num, 1)
    f.seek(8 * sum(bucket_sizes), 1)
    slab_count = struct.unpack("<Q", f.read(8))[0]
with open(sys.argv[2], "r", encoding="utf-8") as f:
    gt = json.load(f)
if n != int(gt["n_base"]) or dim != int(gt["dim"]) or cluster_num != 10000:
    raise SystemExit(
        f"bad metadata header: n={n}, dim={dim}, cluster_num={cluster_num}, max_points={max_points}"
    )
if pca_rank != 32:
    raise SystemExit(f"expected pca_rank 32, got {pca_rank}")
if slab_count != 512:
    raise SystemExit(f"expected slab_count 512, got {slab_count}")
PY

python3 - "$index_dir/_metadata_file.bin" <<'PY'
import sys

from experiments.range_aware_profiling._io import read_metadata

metadata = read_metadata(sys.argv[1])
if metadata.slab_count != 512:
    raise SystemExit(f"expected slab_count=512, got {metadata.slab_count}")
if metadata.slab_neighbour_indices.shape != (metadata.cluster_num, metadata.slab_count):
    raise SystemExit(
        f"bad slab_neighbour_indices shape: {metadata.slab_neighbour_indices.shape}"
    )
if metadata.slab_l.shape != (metadata.cluster_num, metadata.slab_count):
    raise SystemExit(f"bad slab_l shape: {metadata.slab_l.shape}")
if metadata.slab_h.shape != (metadata.cluster_num, metadata.slab_count):
    raise SystemExit(f"bad slab_h shape: {metadata.slab_h.shape}")
PY

coincident_cpp="$run_dir/coincident_centroids.cpp"
coincident_bin="$run_dir/coincident_centroids"
python3 - "$coincident_cpp" <<'PY'
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lib/ClusterIO.h"

namespace
{
void write_fbin(const std::string& path)
{
  constexpr uint32_t n = 2;
  constexpr uint32_t d = 2;
  const float data[4] = {0.0f, 0.0f, 1.0f, 1.0f};
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(data), sizeof(data));
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: coincident_centroids <run_dir>\n";
    return 2;
  }

  try
  {
    const std::string run_dir = argv[1];
    std::filesystem::create_directories(run_dir);
    const std::string data_path = run_dir + "/toy.fbin";
    const std::string cluster_path = run_dir + "/_cluster_file.bin";
    const std::string metadata_path = run_dir + "/_metadata_file.bin";
    write_fbin(data_path);

    std::vector<std::vector<size_t>> assignment = {{0}, {1}};
    std::vector<float> centroids = {0.0f, 0.0f, 0.0f, 0.0f};
    ClusterWriter writer(data_path, cluster_path, metadata_path, 0, 1);
    writer.writeClusters(assignment, centroids.data(), 0.001f);
    writer.writeMetadata(assignment);
    std::cerr << "coincident-centroid build unexpectedly succeeded\n";
    return 1;
  }
  catch (const std::exception& e)
  {
    const std::string msg = e.what();
    if (msg.find("coincident centroids") == std::string::npos)
    {
      std::cerr << "unexpected error: " << msg << '\n';
      return 1;
    }
  }

  return 0;
}
''')
PY
g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
  -Ithird/liburing/src/include -Ithird/eigen -Ithird/symqglib \
  -Ithird/RaBitQ-Library/include -DNDEBUG -mtune=native -march=native -mavx2 \
  -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc \
  -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free \
  -fopenmp-simd -funroll-loops -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS \
  $openblas_cflags -DUSE_AVX2 "$coincident_cpp" \
  $openblas_libs \
  -o "$coincident_bin"
"$coincident_bin" "$run_dir/coincident"

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
if ! "$check_uring_bin" >/tmp/pd2_cluster_slab_cpp_preflight.out 2>/tmp/pd2_cluster_slab_cpp_preflight.err; then
  cat /tmp/pd2_cluster_slab_cpp_preflight.err >&2
  if [[ "${DISKRANGE_ALLOW_IO_URING_SKIP:-0}" == "1" ]]; then
    echo "pd2_cluster_slab_cpp_deployment_test: SKIP runtime search/dump checks because io_uring is unavailable in this environment" >&2
    exit 0
  fi
  echo "pd2_cluster_slab_cpp_deployment_test: FAIL because io_uring is unavailable; set DISKRANGE_ALLOW_IO_URING_SKIP=1 only for sandboxed local build/metadata checks" >&2
  exit 1
fi

normal_stdout="$(mktemp)"
normal_stderr="$(mktemp)"
if ! ./build/main --config "$config" --gt "$gt_json" \
  >"$normal_stdout" 2>"$normal_stderr"; then
  cat "$normal_stderr" >&2
  exit 1
fi
normal_results="$(mktemp)"
python3 tests/extract_run_result.py "$normal_stdout" "$normal_stderr" >"$normal_results"

python3 - "$normal_results" "$n_queries" <<'PY'
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
required = {
    "recall",
    "triangle_prune_rate",
    "triangle_candidate_clusters",
    "triangle_pruned_clusters",
    "slab_pruned_clusters",
    "box_residual_pruned_clusters",
    "mean_keep_after_filter",
    "build_time",
    "search_time",
    "dist_comp",
    "per_query_min",
    "per_query_median",
    "per_query_max",
    "per_query_zero",
}
missing = required - values.keys()
if missing:
    raise SystemExit(f"missing RunResult keys: {sorted(missing)}")
triangle_candidate_clusters = int(values["triangle_candidate_clusters"])
triangle_pruned_clusters = int(values["triangle_pruned_clusters"])
slab_pruned_clusters = int(values["slab_pruned_clusters"])
box_residual_pruned_clusters = int(values["box_residual_pruned_clusters"])
if triangle_candidate_clusters <= 0:
    raise SystemExit("triangle_candidate_clusters must be positive")
if slab_pruned_clusters + box_residual_pruned_clusters != triangle_pruned_clusters:
    raise SystemExit(
        "prune partition mismatch: "
        f"{slab_pruned_clusters} + {box_residual_pruned_clusters} != {triangle_pruned_clusters}"
    )
mean_keep = float(values["mean_keep_after_filter"])
if mean_keep < 0.0:
    raise SystemExit(f"mean_keep_after_filter must be non-negative: {mean_keep}")
PY

slab_zero_config="$run_dir/slab_zero.config"
python3 - "$config" "$slab_zero_config" <<'PY'
import sys

replaced = False
with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        stripped = line.strip()
        if stripped.startswith("slab_count "):
            dst.write("slab_count 0\n")
            replaced = True
        else:
            dst.write(line)
    if not replaced:
        dst.write("slab_count 0\n")
PY
slab_zero_gt_json="$run_dir/deep1m_0.2632_slab_zero_gt.json"
python3 - "$gt_json" "$slab_zero_gt_json" "$run_dir/deep1m_slab_zero" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
data["index"]["prefix"] = sys.argv[3]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY
slab_zero_stdout="$(mktemp)"
slab_zero_stderr="$(mktemp)"
./build/main --config "$slab_zero_config" --gt "$slab_zero_gt_json" \
  >"$slab_zero_stdout" 2>"$slab_zero_stderr"
python3 - "$slab_zero_stdout" "$slab_zero_stderr" <<'PY'
import sys

values = {}
for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if " = " not in line:
                continue
            key, raw = line.strip().split(" = ", 1)
            key = key.split(":")[-1].strip()
            values[key] = raw.split()[0]
slab_pruned = int(values["slab_pruned_clusters"])
box_pruned = int(values["box_residual_pruned_clusters"])
triangle_pruned = int(values["triangle_pruned_clusters"])
if slab_pruned != 0:
    raise SystemExit(f"expected slab_pruned_clusters=0 when slab_count=0, got {slab_pruned}")
if box_pruned != triangle_pruned:
    raise SystemExit(
        f"expected box_residual_pruned_clusters == triangle_pruned_clusters, got {box_pruned} != {triangle_pruned}"
    )
PY

search_only_cpp="$run_dir/search_only.cpp"
search_only_bin="$run_dir/search_only"
python3 - "$search_only_cpp" <<'PY'
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <exception>
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
    std::cout << "recall = " << result.recall << '\n';
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
  -Ithird/RaBitQ-Library/include -DNDEBUG -mtune=native -march=native -mavx2 \
  -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc \
  -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free \
  -fopenmp-simd -funroll-loops -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS \
  $openblas_cflags -DUSE_AVX2 "$search_only_cpp" \
  build/third/liburing/libliburing_vendored.a $openblas_libs \
  -o "$search_only_bin"

v2_metadata="$run_dir/v2_metadata.bin"
python3 - "$index_dir/_metadata_file.bin" "$v2_metadata" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
data[8:16] = struct.pack("<Q", 2)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
mv "$v2_metadata" "$index_dir/_metadata_file.bin"
schema_stdout="$(mktemp)"
schema_stderr="$(mktemp)"
if "$search_only_bin" "$config" "$gt_json" >"$schema_stdout" 2>"$schema_stderr"; then
  echo "schema-v2 metadata unexpectedly loaded under v3 binary" >&2
  exit 1
fi
grep -Fq "Delete the index file at" "$schema_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDUMP_LB_SLAB=ON >/tmp/pd2_cluster_slab_cpp_dump_cmake.log
cmake --build build >/tmp/pd2_cluster_slab_cpp_dump_build.log
dump_stdout="$(mktemp)"
dump_stderr="$(mktemp)"
./build/main --config "$config" --gt "$gt_json" >"$dump_stdout" 2>"$dump_stderr"
test -s "$index_dir/cpp_lb_dump.csv"
PYTHONPATH="$repo_root" python3 experiments/PCA_enhancement/verify_cpp_lb_dump.py \
  --cpp-dump "$index_dir/cpp_lb_dump.csv" \
  --config "$config" \
  --gt "$gt_json" \
  --atol 3e-4
