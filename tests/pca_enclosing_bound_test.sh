#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/pca_enclosing_bound.XXXXXX)"
index_prefix="$run_dir/deep1m"
index_dir="${index_prefix}_diskrange"
gt_json="$run_dir/deep1m_0.2632_gt.json"
query_file="$run_dir/deep1m_query_subset.fbin"
n_queries=64

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

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/pca_enclosing_bound_cmake.log
cmake --build build >/tmp/pca_enclosing_bound_build.log

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

default_config="$run_dir/default_pca.config"
python3 - "$config" "$default_config" <<'PY'
import sys

with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        if line.strip().startswith("pca_rank "):
            continue
        dst.write(line)
PY
"$config_probe_bin" "$default_config" "$gt_json" | grep -Fq "pca_rank = 32"

bad_config="$run_dir/bad_negative_pca.config"
python3 - "$config" "$bad_config" <<'PY'
import sys

replaced = False
with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        stripped = line.strip()
        if stripped.startswith("pca_rank "):
            dst.write("pca_rank -1\n")
            replaced = True
        else:
            dst.write(line)
    if not replaced:
        dst.write("pca_rank -1\n")
PY

bad_config_stdout="$(mktemp)"
bad_config_stderr="$(mktemp)"
if ./build/main --config "$bad_config" --gt "$gt_json" \
  >"$bad_config_stdout" 2>"$bad_config_stderr"; then
  echo "negative pca_rank config unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "pca_rank must be non-negative" "$bad_config_stderr"

pca_rank0_config="$run_dir/pca_rank0.config"
pca_rank0_gt_json="$run_dir/deep1m_0.2632_rank0_gt.json"
pca_rank0_prefix="$run_dir/deep1m_rank0"
python3 - "$config" "$pca_rank0_config" <<'PY'
import sys

replaced = False
with open(sys.argv[1], "r", encoding="utf-8") as src, open(
    sys.argv[2], "w", encoding="utf-8"
) as dst:
    for line in src:
        stripped = line.strip()
        if stripped.startswith("pca_rank "):
            dst.write("pca_rank 0\n")
            replaced = True
        else:
            dst.write(line)
    if not replaced:
        dst.write("pca_rank 0\n")
PY
python3 - "$gt_json" "$pca_rank0_gt_json" "$pca_rank0_prefix" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
data["index"]["prefix"] = sys.argv[3]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

pca_rank0_stdout="$(mktemp)"
pca_rank0_stderr="$(mktemp)"
./build/main --config "$pca_rank0_config" --gt "$pca_rank0_gt_json" \
  >"$pca_rank0_stdout" 2>"$pca_rank0_stderr"
pca_rank0_results="$(mktemp)"
python3 tests/extract_run_result.py "$pca_rank0_stdout" "$pca_rank0_stderr" >"$pca_rank0_results"
python3 - "$pca_rank0_results" "${pca_rank0_prefix}_diskrange/_metadata_file.bin" <<'PY'
import struct
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
with open(sys.argv[2], "rb") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("rank0 metadata missing magic")
    version = struct.unpack("<Q", f.read(8))[0]
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
if version != 5 or pca_rank != 0:
    raise SystemExit(f"bad rank0 metadata header: version={version}, pca_rank={pca_rank}")
pca_only_prune_rate = float(values.get("pca_only_prune_rate", values["triangle_prune_rate"]))
if not (0.50 <= pca_only_prune_rate <= 1.0):
    raise SystemExit(
        f"rank0 fallback PCA prune rate outside sphere range: {pca_only_prune_rate}"
    )
recall = float(values["recall"])
if not (0.0 < recall <= 1.0001):
    raise SystemExit(f"rank0 recall out of bounds: {recall}")
print(f"rank0 pca-only prune rate={pca_only_prune_rate:.6f} recall={recall:.6f}")
PY

normal_stdout="$(mktemp)"
normal_stderr="$(mktemp)"
./build/main --config "$config" --gt "$gt_json" >"$normal_stdout" 2>"$normal_stderr"
normal_results="$(mktemp)"
python3 tests/extract_run_result.py "$normal_stdout" "$normal_stderr" >"$normal_results"

test -s "$index_dir/_cluster_file.bin"
test -s "$index_dir/_metadata_file.bin"
test -s "$index_dir/_hnsw_file.bin"

python3 - "$normal_results" "$index_dir/_metadata_file.bin" "$gt_json" <<'PY'
import json
import struct
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        key, raw = line.strip().split(" = ", 1)
        values[key] = raw
with open(sys.argv[2], "rb") as f:
    magic = f.read(8)
    version = struct.unpack("<Q", f.read(8))[0]
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
with open(sys.argv[3], "r", encoding="utf-8") as f:
    gt = json.load(f)
if magic != b"DJMETA\0\0":
    raise SystemExit(f"bad metadata magic: {magic!r}")
if version != 5:
    raise SystemExit(f"bad metadata schema version: {version}")
if pca_rank != 32:
    raise SystemExit(f"expected pca_rank 32, got {pca_rank}")
if n != int(gt["n_base"]) or dim != int(gt["dim"]) or cluster_num != 10000:
    raise SystemExit(
        f"bad metadata header: n={n}, dim={dim}, cluster_num={cluster_num}, max_points={max_points}"
    )
triangle_prune_rate = float(values["triangle_prune_rate"])
if triangle_prune_rate < 0.30:
    raise SystemExit(f"PCA prune rate too low: {triangle_prune_rate}")
recall = float(values["recall"])
if not (0.0 < recall <= 1.0001):
    raise SystemExit(f"recall out of bounds: {recall}")
print(f"pca prune rate={triangle_prune_rate:.6f} recall={recall:.6f}")
PY

python3 - "$index_dir/_metadata_file.bin" "$gt_json" <<'PY'
import json
import sys
from experiments.range_aware_profiling._io import read_metadata

metadata = read_metadata(sys.argv[1])
with open(sys.argv[2], "r", encoding="utf-8") as f:
    gt = json.load(f)
if metadata.n != int(gt["n_base"]):
    raise SystemExit(f"metadata.n mismatch: {metadata.n}")
if metadata.dim != int(gt["dim"]):
    raise SystemExit(f"metadata.dim mismatch: {metadata.dim}")
if metadata.cluster_num != 10000:
    raise SystemExit(f"metadata.cluster_num mismatch: {metadata.cluster_num}")
if int(metadata.bucket_sizes.sum()) != int(gt["n_base"]):
    raise SystemExit("metadata bucket sum mismatch")
PY

python3 - "$index_dir/_metadata_file.bin" "$query_file" "$gt_json" <<'PY'
import json
import math
import struct
import sys

import numpy as np

from experiments.range_aware_profiling._io import (
    blocked_l2_matrix,
    exact_topk_indices,
    read_fbin,
)
from experiments.range_aware_profiling import p0_oracle_prune_ceiling as p0

metadata_path, query_path, gt_path = sys.argv[1:4]
with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)
with open(metadata_path, "rb") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("metadata missing schema magic")
    (version,) = struct.unpack("<Q", f.read(8))
    if version != 5:
        raise SystemExit(f"unexpected schema version: {version}")
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
    bucket_sizes = np.frombuffer(f.read(8 * cluster_num), dtype=np.uint64).copy()
    centroids = np.frombuffer(f.read(4 * cluster_num * dim), dtype=np.float32).reshape(
        cluster_num, dim
    ).copy()
    radii = np.frombuffer(f.read(4 * cluster_num), dtype=np.float32).copy()
    pca_basis = np.frombuffer(
        f.read(4 * cluster_num * pca_rank * dim), dtype=np.float32
    ).reshape(cluster_num, pca_rank, dim).copy()
    pca_min = np.frombuffer(
        f.read(4 * cluster_num * pca_rank), dtype=np.float32
    ).reshape(cluster_num, pca_rank).copy()
    pca_max = np.frombuffer(
        f.read(4 * cluster_num * pca_rank), dtype=np.float32
    ).reshape(cluster_num, pca_rank).copy()
    pca_residual = np.frombuffer(f.read(4 * cluster_num), dtype=np.float32).copy()
    pca_effective = np.frombuffer(f.read(4 * cluster_num), dtype=np.uint32).copy()

queries = read_fbin(query_path, expected_dim=dim)
distances = blocked_l2_matrix(queries, centroids)
topk = exact_topk_indices(distances, int(gt["index"].get("top_k", 245)) if "top_k" in gt.get("index", {}) else 245)
radius = float(gt.get("radius_l2", math.sqrt(float(gt["radius_squared"]))))

bounds = []
for cluster_id in range(cluster_num):
    rank = int(pca_effective[cluster_id])
    if rank > pca_rank:
        raise SystemExit(f"effective rank out of range at cluster {cluster_id}")
    bounds.append(
        p0.ClusterBound(
            mins=np.empty((0,), dtype=np.float32),
            maxs=np.empty((0,), dtype=np.float32),
            dist_p95=0.0,
            pca_center=centroids[cluster_id],
            pca_components=pca_basis[cluster_id, :rank],
            pca_min=pca_min[cluster_id, :rank],
            pca_max=pca_max[cluster_id, :rank],
            pca_residual_radius=float(pca_residual[cluster_id]),
        )
    )


def lower_value(query: np.ndarray, cluster_id: int) -> float:
    rank = int(pca_effective[cluster_id])
    centered = query - centroids[cluster_id]
    if rank == 0:
        return max(0.0, float(np.linalg.norm(centered)) - float(pca_residual[cluster_id]))
    components = pca_basis[cluster_id, :rank]
    coords = components @ centered
    below = np.maximum(pca_min[cluster_id, :rank] - coords, 0.0)
    above = np.maximum(coords - pca_max[cluster_id, :rank], 0.0)
    box_delta = np.maximum(below, above)
    projected_sq = float(np.sum(coords * coords))
    total_sq = float(np.sum(centered * centered))
    residual_q = math.sqrt(max(0.0, total_sq - projected_sq))
    residual_delta = max(0.0, residual_q - float(pca_residual[cluster_id]))
    return math.sqrt(float(np.sum(box_delta * box_delta)) + residual_delta * residual_delta)


checked = 0
kept = 0
for query, clusters in zip(queries, topk):
    oracle_keep = p0.pca_keep(query, clusters, bounds, radius)
    local_keep = np.asarray([lower_value(query, int(cid)) <= radius for cid in clusters])
    if not np.array_equal(oracle_keep, local_keep):
        raise SystemExit("p0.pca_keep decisions differ from persisted lower-bound formula")
    checked += int(clusters.size)
    kept += int(np.count_nonzero(oracle_keep))
print(f"python_pca_oracle checked={checked} kept={kept}")
PY

no_false_prune_cpp="$run_dir/pca_no_false_prune.cpp"
no_false_prune_bin="$run_dir/pca_no_false_prune"
python3 - "$no_false_prune_cpp" <<'PY'
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "lib/ClusterIO.h"
#include "lib/ConfigLoader.h"

int r_server_id = 0;

namespace
{
std::vector<float> read_queries(const ResolvedConfig& config)
{
  std::ifstream in(config.query_file, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open query file");
  }
  uint32_t n = 0;
  uint32_t d = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
  if (n != config.n_queries || d != config.dim)
  {
    throw std::runtime_error("query header mismatch");
  }
  std::vector<float> queries(config.n_queries * config.dim);
  in.read(reinterpret_cast<char*>(queries.data()),
          static_cast<std::streamsize>(queries.size() * sizeof(float)));
  if (!in)
  {
    throw std::runtime_error("short query file");
  }
  return queries;
}

float l2_sqr(const float* left, const float* right, size_t d)
{
  float sum = 0.0f;
  for (size_t j = 0; j < d; ++j)
  {
    const float diff = left[j] - right[j];
    sum += diff * diff;
  }
  return sum;
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: pca_no_false_prune <config> <gt>" << std::endl;
    return 2;
  }

  try
  {
    const ResolvedConfig config = load_resolved_config(argv[1], argv[2]);
    const std::vector<float> queries = read_queries(config);
    ClusterReader reader(config.cluster_path, config.metadata_path,
                         config.io_queue_depth, 1);
    reader.readMetaData();

    std::ifstream cluster_file(config.cluster_path, std::ios::binary);
    if (!cluster_file)
    {
      throw std::runtime_error("failed to open cluster file");
    }

    const float epsilon = std::sqrt(static_cast<float>(config.radius_squared));
    const float radius_sq = static_cast<float>(config.radius_squared);
    size_t checked_candidates = 0;
    size_t pruned_candidates = 0;
    size_t false_pruned = 0;

    for (size_t q = 0; q < config.n_queries; ++q)
    {
      const float* query = queries.data() + q * config.dim;
      std::vector<std::pair<float, size_t>> candidates;
      candidates.reserve(reader.cluster_num);
      for (size_t cluster_id = 0; cluster_id < reader.cluster_num; ++cluster_id)
      {
        const float* centroid = reader.centroids.data() + cluster_id * config.dim;
        candidates.emplace_back(l2_sqr(query, centroid, config.dim), cluster_id);
      }
      if (config.K < candidates.size())
      {
        std::nth_element(candidates.begin(), candidates.begin() + config.K,
                         candidates.end());
        candidates.resize(config.K);
      }

      for (const auto& candidate : candidates)
      {
        const size_t cluster_id = candidate.second;
        ++checked_candidates;
        if (reader.pca_lower_bound(cluster_id, query) <= epsilon)
        {
          continue;
        }
        ++pruned_candidates;

        const size_t bucket_size = reader.bucket_sizes[cluster_id];
        std::vector<float> points(bucket_size * config.dim);
        cluster_file.clear();
        cluster_file.seekg(static_cast<std::streamoff>(reader.file_pos[cluster_id]),
                           std::ios::beg);
        cluster_file.read(
            reinterpret_cast<char*>(points.data()),
            static_cast<std::streamsize>(points.size() * sizeof(float)));
        if (!cluster_file)
        {
          throw std::runtime_error("short cluster payload");
        }
        for (size_t i = 0; i < bucket_size; ++i)
        {
          if (l2_sqr(query, points.data() + i * config.dim, config.dim) <=
              radius_sq)
          {
            std::cerr << "false PCA prune: query=" << q
                      << " cluster=" << cluster_id << std::endl;
            ++false_pruned;
            return 1;
          }
        }
      }
    }

    std::cout << "diagnostic checked_candidates=" << checked_candidates
              << " pruned_candidates=" << pruned_candidates
              << " false_pruned=" << false_pruned << std::endl;
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
  $openblas_cflags -DUSE_AVX2 "$no_false_prune_cpp" \
  build/third/liburing/libliburing_vendored.a $openblas_libs \
  -o "$no_false_prune_bin"
"$no_false_prune_bin" "$config" "$gt_json"

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
    (void)index.search(config);
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

mismatch_stdout="$(mktemp)"
mismatch_stderr="$(mktemp)"
if ! "$search_only_bin" "$config" "$pca_rank0_gt_json" \
  >"$mismatch_stdout" 2>"$mismatch_stderr"; then
  echo "metadata/config pca_rank mismatch unexpectedly failed" >&2
  cat "$mismatch_stderr" >&2
  exit 1
fi
grep -Fq "warning: metadata pca_rank 0 differs from config pca_rank 32" "$mismatch_stderr"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
    "<QQQQQQ", data[16:64]
)
data[48:56] = struct.pack("<Q", dim + 1)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
bad_rank_stdout="$(mktemp)"
bad_rank_stderr="$(mktemp)"
if "$search_only_bin" "$config" "$gt_json" \
  >"$bad_rank_stdout" 2>"$bad_rank_stderr"; then
  echo "metadata pca_rank > dim unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "metadata pca_rank out of range" "$bad_rank_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
    "<QQQQQQ", data[16:64]
)
rank_block = (
    64
    + cluster_num * 8
    + cluster_num * dim * 4
    + cluster_num * 4
    + cluster_num * pca_rank * dim * 4
    + cluster_num * pca_rank * 4
    + cluster_num * pca_rank * 4
    + cluster_num * 4
)
data[rank_block : rank_block + 4] = struct.pack("<I", pca_rank + 1)
with open(sys.argv[2], "wb") as f:
    f.write(data)
PY
bad_effective_stdout="$(mktemp)"
bad_effective_stderr="$(mktemp)"
if "$search_only_bin" "$config" "$gt_json" \
  >"$bad_effective_stdout" 2>"$bad_effective_stderr"; then
  echo "metadata effective rank corruption unexpectedly succeeded" >&2
  exit 1
fi
grep -Eq "metadata effective PCA rank out of range|metadata cell_vec_offsets|metadata cluster_cell_offsets|metadata file size mismatch" "$bad_effective_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"

mv "$index_dir/_metadata_file.bin" "$index_dir/_metadata_file.bin.good"
python3 - "$gt_json" "$index_dir/_metadata_file.bin" <<'PY'
import json
import struct
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
with open(sys.argv[2], "wb") as f:
    f.write(
        struct.pack(
            "<QQQQ",
            int(data["n_base"]),
            int(data["dim"]),
            10000,
            1,
        )
    )
PY
old_schema_stdout="$(mktemp)"
old_schema_stderr="$(mktemp)"
if "$search_only_bin" "$config" "$gt_json" \
  >"$old_schema_stdout" 2>"$old_schema_stderr"; then
  echo "old schema metadata unexpectedly succeeded" >&2
  exit 1
fi
grep -Fq "incompatible metadata schema" "$old_schema_stderr"
grep -Fq "rebuild with the current binary" "$old_schema_stderr"
mv "$index_dir/_metadata_file.bin.good" "$index_dir/_metadata_file.bin"
