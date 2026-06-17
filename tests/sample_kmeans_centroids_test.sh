#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="$(mktemp -d /tmp/sample_kmeans_centroids.XXXXXX)"

cleanup() {
  rm -rf "$run_dir"
}
trap cleanup EXIT

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/sample_kmeans_centroids_cmake.log
cmake --build build >/tmp/sample_kmeans_centroids_build.log

harness_cpp="$run_dir/sample_kmeans_centroids_harness.cpp"
harness_bin="$run_dir/sample_kmeans_centroids_harness"

cat >"$harness_cpp" <<'CPP'
#include <cmath>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lib/Kmeans.h"

int r_server_id = 0;

namespace
{
std::vector<float> make_base(size_t n, size_t d)
{
  std::vector<float> base(n * d);
  for (size_t i = 0; i < n; ++i)
  {
    const float x =
        i < n / 2 ? static_cast<float>(i) * 0.01f
                  : 100.0f + static_cast<float>(i - n / 2) * 0.01f;
    base[i * d] = x;
    base[i * d + 1] = x * x + 0.123f * x;
  }
  return base;
}

void write_fbin(const std::string& path, const std::vector<float>& base, size_t n,
                size_t d)
{
  std::ofstream out(path, std::ios::binary);
  const uint32_t n32 = static_cast<uint32_t>(n);
  const uint32_t d32 = static_cast<uint32_t>(d);
  out.write(reinterpret_cast<const char*>(&n32), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&d32), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(base.data()),
            static_cast<std::streamsize>(base.size() * sizeof(float)));
}

std::vector<float> read_metadata_centroids(const std::string& path, size_t n,
                                           size_t d, size_t cluster_num,
                                           size_t expected_pca_rank)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open metadata file: " + path);
  }

  std::array<char, 8> magic{};
  uint64_t schema_version = 0;
  size_t meta_n = 0;
  size_t meta_d = 0;
  size_t meta_cluster_num = 0;
  size_t max_points = 0;
  uint64_t pca_rank = 0;
  uint64_t target_cell_vecs = 0;
  uint64_t vec_dtype = 0;
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  in.read(reinterpret_cast<char*>(&schema_version), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&meta_n), sizeof(size_t));
  in.read(reinterpret_cast<char*>(&meta_d), sizeof(size_t));
  in.read(reinterpret_cast<char*>(&meta_cluster_num), sizeof(size_t));
  in.read(reinterpret_cast<char*>(&max_points), sizeof(size_t));
  in.read(reinterpret_cast<char*>(&pca_rank), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&target_cell_vecs), sizeof(uint64_t));
  in.read(reinterpret_cast<char*>(&vec_dtype), sizeof(uint64_t));
  const std::array<char, 8> expected_magic = {
      'D', 'J', 'M', 'E', 'T', 'A', '\0', '\0'};
  if (magic != expected_magic || schema_version != 6)
  {
    throw std::runtime_error("metadata schema header mismatch");
  }
  if (meta_n != n || meta_d != d || meta_cluster_num != cluster_num ||
      max_points == 0 || pca_rank != expected_pca_rank ||
      target_cell_vecs < 2 || vec_dtype != 0)
  {
    throw std::runtime_error("metadata header mismatch");
  }

  std::vector<size_t> bucket_sizes(cluster_num);
  in.read(reinterpret_cast<char*>(bucket_sizes.data()),
          static_cast<std::streamsize>(bucket_sizes.size() * sizeof(size_t)));

  std::vector<float> centroids(cluster_num * d);
  in.read(reinterpret_cast<char*>(centroids.data()),
          static_cast<std::streamsize>(centroids.size() * sizeof(float)));
  if (!in)
  {
    throw std::runtime_error("failed to read metadata centroids");
  }
  return centroids;
}

bool matches_raw_vector(const std::vector<float>& base, size_t n, size_t d,
                        const float* centroid)
{
  constexpr float kTolerance = 1e-6f;
  for (size_t i = 0; i < n; ++i)
  {
    bool match = true;
    for (size_t j = 0; j < d; ++j)
    {
      if (std::fabs(base[i * d + j] - centroid[j]) > kTolerance)
      {
        match = false;
        break;
      }
    }
    if (match)
    {
      return true;
    }
  }
  return false;
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: sample_kmeans_centroids_harness <run_dir>\n";
    return 2;
  }

  try
  {
    const std::string run_dir = argv[1];
    const std::string index_dir = run_dir + "/toy_diskrange";
    std::filesystem::create_directories(index_dir);

    constexpr size_t n = 2000;
    constexpr size_t d = 64;
    constexpr size_t cluster_num = 8;
    const std::vector<float> base = make_base(n, d);
    const std::string data_path = run_dir + "/toy.fbin";
    write_fbin(data_path, base, n, d);

    ResolvedConfig config;
    config.cluster_num = cluster_num;
    config.mem_budget = 0.001f;
    config.slab_count = 1;
    config.partition_objective = "anisotropic";
    config.unit_normalize_prebuild = true;
    config.data_file = data_path;
    config.cluster_path = index_dir + "/_cluster_file.bin";
    config.metadata_path = index_dir + "/_metadata_file.bin";
    config.hnsw_path = index_dir + "/_hnsw_file.bin";
    config.sqg_path = index_dir + "/_sqg_file.bin";
    config.sqg_m = 4;

    one_level_kmeans(config);

    const std::vector<float> centroids =
        read_metadata_centroids(config.metadata_path, n, d, cluster_num,
                                config.pca_rank);
    size_t raw_matches = 0;
    for (size_t i = 0; i < cluster_num; ++i)
    {
      if (matches_raw_vector(base, n, d, centroids.data() + i * d))
      {
        ++raw_matches;
      }
    }

    if (raw_matches != 0)
    {
      std::cerr << "expected trained centroids, but " << raw_matches
                << " metadata centroid(s) exactly match raw base vectors\n";
      return 1;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}
CPP

if pkg-config --exists openblas; then
  openblas_cflags="$(pkg-config --cflags openblas)"
  openblas_libs="$(pkg-config --libs openblas)"
else
  openblas_cflags="-I/usr/include/x86_64-linux-gnu/openblas-pthread"
  openblas_libs="-L/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas"
fi

g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
  -Ithird/symqglib \
  -Ithird/liburing/src/include -Ithird/eigen -DNDEBUG -march=native -mtune=native -mavx2 -pthread \
  -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc \
  -fno-builtin-realloc -fno-builtin-free -fopenmp-simd -funroll-loops \
  -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS $openblas_cflags \
  -DUSE_AVX2 "$harness_cpp" build/third/liburing/libliburing_vendored.a \
  $openblas_libs \
  -o "$harness_bin"

OMP_NUM_THREADS=1 "$harness_bin" "$run_dir"

python3 - <<'PY'
from pathlib import Path

source = Path("lib/Kmeans.h").read_text(encoding="utf-8")
start = source.index("void one_level_kmeans")
body = source[start:]

train_pos = body.find("kmeans.train(")
writer_pos = body.find("ClusterWriter cluster_writer")
if train_pos == -1:
    raise SystemExit("one_level_kmeans() does not call kmeans.train()")
if writer_pos == -1:
    raise SystemExit("one_level_kmeans() does not write clusters")
if train_pos > writer_pos:
    raise SystemExit("kmeans.train() must run before cluster writing")
if "kmeans.add_anisotropic_exact(num, data_buffer.data(), ids)" not in body:
    raise SystemExit("anisotropic full-data assignment path is missing")
if "kmeans.add_standard_prebuild_batch(num, data_buffer.data(), ids," not in body:
    raise SystemExit("standard full-data assignment path is missing")
if "add2choice(n, data, ids, sqg)" not in source:
    raise SystemExit("add_standard_prebuild_batch must still delegate to add2choice")
PY
