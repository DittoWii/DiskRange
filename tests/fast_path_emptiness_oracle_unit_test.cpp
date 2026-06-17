#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ClusterIO.h"
#include "../lib/ConfigLoader.h"
#include "../lib/FastPathEmptinessOracle.h"
#include "../lib/FastPathPhase3AsyncFetch.h"
#include "../lib/IndexPaths.h"
#include "../lib/RBQCodeStorage.h"
#include "../lib/SQGCentroidIndex.h"

int r_server_id = 0;

namespace fs = std::filesystem;

namespace
{
void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

uint32_t float_bits(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "fast_path_emptiness_oracle_unit";
  fs::remove_all(base);
  fs::create_directories(base);
  return base.string();
}

fs::path copy_config_with_suffix(const std::string& config_path,
                                 const std::string& temp_dir,
                                 const std::string& file_name,
                                 const std::string& suffix)
{
  const fs::path out_path = fs::path(temp_dir) / file_name;
  std::ifstream in(config_path);
  std::ofstream out(out_path);
  out << in.rdbuf();
  out << suffix;
  return out_path;
}

std::vector<float> make_centroids(size_t cluster_count, size_t dim)
{
  std::vector<float> centroids(cluster_count * dim, 0.0f);
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    centroids[cid * dim] = static_cast<float>(cid);
    centroids[cid * dim + 1] = static_cast<float>(cid % 7) * 0.25f;
    centroids[cid * dim + 2] = static_cast<float>(cid % 11) * 0.125f;
  }
  return centroids;
}

void write_cluster_file(const fs::path& path, const std::vector<float>& vectors)
{
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out)
  {
    throw std::runtime_error("failed to create cluster file: " + path.string());
  }
  out.write(reinterpret_cast<const char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!out)
  {
    throw std::runtime_error("failed to write cluster file: " + path.string());
  }
}

void write_test_metadata_file(
    const fs::path& path, size_t dim, const std::vector<float>& centroids,
    const std::vector<std::vector<size_t>>& assignment)
{
  const size_t cluster_count = assignment.size();
  std::vector<size_t> bucket_sizes(cluster_count, 0);
  size_t n = 0;
  size_t max_points = 0;
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    bucket_sizes[cid] = assignment[cid].size();
    n += bucket_sizes[cid];
    max_points = std::max(max_points, bucket_sizes[cid]);
  }

  std::vector<float> radii(cluster_count, 0.0f);
  std::vector<float> pca_residual_radius(cluster_count, 0.0f);
  std::vector<uint32_t> pca_effective_rank(cluster_count, 0);
  std::vector<uint64_t> cluster_cell_offsets(cluster_count + 1, 0);
  std::vector<uint64_t> cell_vec_offsets;
  cell_vec_offsets.push_back(0);
  uint64_t total_cells = 0;
  uint64_t total_vecs = 0;
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    cluster_cell_offsets[cid] = total_cells;
    if (bucket_sizes[cid] != 0)
    {
      ++total_cells;
      total_vecs += static_cast<uint64_t>(bucket_sizes[cid]);
      cell_vec_offsets.push_back(total_vecs);
    }
  }
  cluster_cell_offsets[cluster_count] = total_cells;
  std::vector<float> cell_bounds(static_cast<size_t>(total_cells) * 2, 0.0f);

  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out)
  {
    throw std::runtime_error("failed to create metadata file: " +
                             path.string());
  }
  out.write(kMetadataMagic.data(),
            static_cast<std::streamsize>(kMetadataMagic.size()));
  const uint64_t schema_version = kMetadataSchemaVersion;
  const size_t pca_rank = 0;
  const uint64_t metadata_pca_rank = 0;
  const uint64_t metadata_target_cell_vecs = 2;
  const uint64_t metadata_vec_dtype = range_search_config::kVecDTypeFloat32;
  const uint64_t metadata_slab_count = 0;
  out.write(reinterpret_cast<const char*>(&schema_version),
            sizeof(schema_version));
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char*>(&cluster_count),
            sizeof(cluster_count));
  out.write(reinterpret_cast<const char*>(&max_points), sizeof(max_points));
  out.write(reinterpret_cast<const char*>(&metadata_pca_rank),
            sizeof(metadata_pca_rank));
  out.write(reinterpret_cast<const char*>(&metadata_target_cell_vecs),
            sizeof(metadata_target_cell_vecs));
  out.write(reinterpret_cast<const char*>(&metadata_vec_dtype),
            sizeof(metadata_vec_dtype));
  out.write(reinterpret_cast<const char*>(bucket_sizes.data()),
            static_cast<std::streamsize>(bucket_sizes.size() * sizeof(size_t)));
  out.write(reinterpret_cast<const char*>(centroids.data()),
            static_cast<std::streamsize>(centroids.size() * sizeof(float)));
  out.write(reinterpret_cast<const char*>(radii.data()),
            static_cast<std::streamsize>(radii.size() * sizeof(float)));
  out.write(
      reinterpret_cast<const char*>(pca_residual_radius.data()),
      static_cast<std::streamsize>(pca_residual_radius.size() * sizeof(float)));
  out.write(reinterpret_cast<const char*>(pca_effective_rank.data()),
            static_cast<std::streamsize>(pca_effective_rank.size() *
                                         sizeof(uint32_t)));
  for (const auto& cluster_assignment : assignment)
  {
    if (!cluster_assignment.empty())
    {
      out.write(reinterpret_cast<const char*>(cluster_assignment.data()),
                static_cast<std::streamsize>(cluster_assignment.size() *
                                             sizeof(size_t)));
    }
  }
  out.write(reinterpret_cast<const char*>(&metadata_slab_count),
            sizeof(metadata_slab_count));
  write_aligned_padding(out, "cluster_cell_offsets");
  out.write(reinterpret_cast<const char*>(cluster_cell_offsets.data()),
            static_cast<std::streamsize>(cluster_cell_offsets.size() *
                                         sizeof(uint64_t)));
  write_aligned_padding(out, "cell_vec_offsets");
  out.write(
      reinterpret_cast<const char*>(cell_vec_offsets.data()),
      static_cast<std::streamsize>(cell_vec_offsets.size() * sizeof(uint64_t)));
  write_aligned_padding(out, "cell_bounds");
  if (!cell_bounds.empty())
  {
    out.write(reinterpret_cast<const char*>(cell_bounds.data()),
              static_cast<std::streamsize>(cell_bounds.size() * sizeof(float)));
  }
  if (!out)
  {
    throw std::runtime_error("failed to write metadata file: " + path.string());
  }
  (void)pca_rank;
}

FastPathOracleResult evaluate_in_omp(FastPathEmptinessOracle& oracle,
                                     const float* query, double radius_sq)
{
  FastPathOracleResult result;
#pragma omp parallel num_threads(1)
  {
    result = oracle.evaluate(query, radius_sq);
  }
  return result;
}

struct RBQTestLayout
{
  RBQCodeFileHeader header;
  std::vector<RBQClusterDirectoryEntry> directory;
  size_t payload_offset = 0;
};

RBQTestLayout read_rbq_layout(const fs::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open RBQ test file: " + path.string());
  }
  RBQTestLayout layout;
  in.read(reinterpret_cast<char*>(&layout.header), sizeof(layout.header));
  if (!in)
  {
    throw std::runtime_error("failed to read RBQ test header: " +
                             path.string());
  }
  layout.directory.assign(static_cast<size_t>(layout.header.cluster_count), {});
  if (!layout.directory.empty())
  {
    in.read(reinterpret_cast<char*>(layout.directory.data()),
            static_cast<std::streamsize>(layout.directory.size() *
                                         sizeof(layout.directory.front())));
  }
  if (!in)
  {
    throw std::runtime_error("failed to read RBQ test directory: " +
                             path.string());
  }
  layout.payload_offset =
      sizeof(RBQCodeFileHeader) +
      layout.directory.size() * sizeof(RBQClusterDirectoryEntry);
  return layout;
}

float read_float_at(const fs::path& path, size_t offset)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open RBQ test file for reading: " +
                             path.string());
  }
  float value = 0.0f;
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in)
  {
    throw std::runtime_error("failed to read RBQ test float: " + path.string());
  }
  return value;
}

struct OneBitFactors
{
  float f_add = 0.0f;
  float f_rescale = 0.0f;
  float f_error = 0.0f;
};

struct EightBitFactors
{
  float f_add_ex = 0.0f;
  float f_rescale_ex = 0.0f;
};

OneBitFactors read_one_bit_factors(const fs::path& path, size_t cluster_id,
                                   size_t local_index)
{
  const RBQTestLayout layout = read_rbq_layout(path);
  require(layout.header.ex_bits == 0, "expected 1-bit RBQ file");
  require(cluster_id < layout.directory.size(), "1-bit factor cluster id");
  const auto& entry = layout.directory[cluster_id];
  require(local_index < entry.vec_count, "1-bit factor local index");

  const size_t padded_dim = static_cast<size_t>(layout.header.padded_dim);
  const size_t block_bytes =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  const size_t batch_bin_bytes =
      padded_dim * rabitqlib::fastscan::kBatchSize / 8;
  const size_t block = local_index / rabitqlib::fastscan::kBatchSize;
  const size_t lane = local_index % rabitqlib::fastscan::kBatchSize;
  const size_t block_base = layout.payload_offset +
                            static_cast<size_t>(entry.code_offset_bytes) +
                            block * block_bytes;
  const size_t f_add_offset =
      block_base + batch_bin_bytes + lane * sizeof(float);
  const size_t f_rescale_offset =
      block_base + batch_bin_bytes +
      (rabitqlib::fastscan::kBatchSize + lane) * sizeof(float);
  const size_t f_error_offset =
      block_base + batch_bin_bytes +
      (2 * rabitqlib::fastscan::kBatchSize + lane) * sizeof(float);

  return {read_float_at(path, f_add_offset),
          read_float_at(path, f_rescale_offset),
          read_float_at(path, f_error_offset)};
}

EightBitFactors read_eight_bit_factors(const fs::path& path, size_t cluster_id,
                                       size_t local_index)
{
  const RBQTestLayout layout = read_rbq_layout(path);
  require(layout.header.ex_bits == 7, "expected 8-bit RBQ file");
  require(cluster_id < layout.directory.size(), "8-bit factor cluster id");
  const auto& entry = layout.directory[cluster_id];
  require(local_index < entry.vec_count, "8-bit factor local index");

  const size_t padded_dim = static_cast<size_t>(layout.header.padded_dim);
  const size_t ex_bits = static_cast<size_t>(layout.header.ex_bits);
  const size_t ex_code_bytes = padded_dim * ex_bits / 8;
  const size_t code_bytes =
      static_cast<size_t>(layout.header.code_bytes_per_vec);
  const size_t base = layout.payload_offset +
                      static_cast<size_t>(entry.code_offset_bytes) +
                      local_index * code_bytes + ex_code_bytes;

  return {read_float_at(path, base), read_float_at(path, base + sizeof(float))};
}

void write_float_at(const fs::path& path, size_t offset, float value)
{
  std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!io)
  {
    throw std::runtime_error("failed to open RBQ test file for mutation: " +
                             path.string());
  }
  io.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  io.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!io)
  {
    throw std::runtime_error("failed to mutate RBQ test file: " +
                             path.string());
  }
}

void poison_one_bit_f_add_with_nan(const fs::path& path,
                                   const std::vector<size_t>& cluster_ids)
{
  const RBQTestLayout layout = read_rbq_layout(path);
  require(layout.header.ex_bits == 0, "expected 1-bit RBQ file to poison");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const size_t padded_dim = static_cast<size_t>(layout.header.padded_dim);
  const size_t block_bytes =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  const size_t batch_bin_bytes =
      padded_dim * rabitqlib::fastscan::kBatchSize / 8;
  for (const size_t cid : cluster_ids)
  {
    require(cid < layout.directory.size(), "poison cluster id out of range");
    const auto& entry = layout.directory[cid];
    const size_t blocks = (static_cast<size_t>(entry.vec_count) +
                           rabitqlib::fastscan::kBatchSize - 1) /
                          rabitqlib::fastscan::kBatchSize;
    for (size_t block = 0; block < blocks; ++block)
    {
      const size_t f_add_offset = layout.payload_offset +
                                  static_cast<size_t>(entry.code_offset_bytes) +
                                  block * block_bytes + batch_bin_bytes;
      for (size_t lane = 0; lane < rabitqlib::fastscan::kBatchSize; ++lane)
      {
        write_float_at(path, f_add_offset + lane * sizeof(float), nan);
      }
    }
  }
}

void poison_eight_bit_f_add_with_nan(const fs::path& path,
                                     const std::vector<size_t>& cluster_ids)
{
  const RBQTestLayout layout = read_rbq_layout(path);
  require(layout.header.ex_bits == 7, "expected 8-bit RBQ file to poison");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const size_t padded_dim = static_cast<size_t>(layout.header.padded_dim);
  const size_t ex_code_bytes =
      padded_dim * static_cast<size_t>(layout.header.ex_bits) / 8;
  const size_t code_bytes =
      static_cast<size_t>(layout.header.code_bytes_per_vec);
  for (const size_t cid : cluster_ids)
  {
    require(cid < layout.directory.size(), "poison cluster id out of range");
    const auto& entry = layout.directory[cid];
    for (size_t local = 0; local < entry.vec_count; ++local)
    {
      const size_t f_add_offset = layout.payload_offset +
                                  static_cast<size_t>(entry.code_offset_bytes) +
                                  local * code_bytes + ex_code_bytes;
      write_float_at(path, f_add_offset, nan);
    }
  }
}

std::vector<size_t> exact_hits(const std::vector<float>& vectors, size_t dim,
                               const float* query, float radius_sq)
{
  std::vector<size_t> hits;
  size_t distance_dim = dim;
  const size_t n_vectors = vectors.size() / dim;
  for (size_t vec_id = 0; vec_id < n_vectors; ++vec_id)
  {
    const float dist =
        utils::L2Sqr(query, vectors.data() + vec_id * dim, &distance_dim);
    if (std::isfinite(dist) && dist <= radius_sq)
    {
      hits.push_back(vec_id);
    }
  }
  return hits;
}

std::vector<float> find_empty_candidate_query(
    const SQGCentroidIndex& sqg, const std::vector<float>& centroids,
    const std::vector<size_t>& bucket_sizes, size_t dim, size_t nprobe)
{
  for (size_t cid = 0; cid < bucket_sizes.size(); ++cid)
  {
    if (bucket_sizes[cid] != 0)
    {
      continue;
    }
    std::vector<float> query(
        centroids.begin() + static_cast<std::ptrdiff_t>(cid * dim),
        centroids.begin() + static_cast<std::ptrdiff_t>((cid + 1) * dim));
    const std::vector<uint32_t> candidate_ids =
        sqg.search(query.data(), nprobe);
    bool all_empty = candidate_ids.size() == nprobe;
    for (const uint32_t candidate_id : candidate_ids)
    {
      const size_t candidate = static_cast<size_t>(candidate_id);
      all_empty = all_empty && candidate < bucket_sizes.size() &&
                  bucket_sizes[candidate] == 0;
    }
    if (all_empty)
    {
      return query;
    }
  }
  throw std::runtime_error("failed to find all-empty SQG candidate query");
}

std::vector<float> find_populated_candidate_query(
    const SQGCentroidIndex& sqg, const std::vector<float>& centroids,
    const std::vector<size_t>& bucket_sizes, size_t dim, size_t nprobe)
{
  for (size_t cid = 0; cid < bucket_sizes.size(); ++cid)
  {
    if (bucket_sizes[cid] == 0)
    {
      continue;
    }
    std::vector<float> query(
        centroids.begin() + static_cast<std::ptrdiff_t>(cid * dim),
        centroids.begin() + static_cast<std::ptrdiff_t>((cid + 1) * dim));
    const std::vector<uint32_t> candidate_ids =
        sqg.search(query.data(), nprobe);
    for (const uint32_t candidate_id : candidate_ids)
    {
      const size_t candidate = static_cast<size_t>(candidate_id);
      if (candidate < bucket_sizes.size() && bucket_sizes[candidate] > 0)
      {
        return query;
      }
    }
  }
  throw std::runtime_error("failed to find populated SQG candidate query");
}

std::vector<float> make_cluster_vectors(size_t dim)
{
  std::vector<float> vectors(6 * dim, 0.0f);
  const float coords[6] = {0.0f, 0.2f, 0.4f, 10.0f, 10.2f, 10.4f};
  for (size_t i = 0; i < 6; ++i)
  {
    vectors[i * dim] = coords[i];
    vectors[i * dim + 1] = coords[i] * 0.5f;
  }
  return vectors;
}

void test_fast_path_config_defaults_and_paths(const std::string& config_path,
                                              const std::string& gt_path)
{
  const ResolvedConfig config = load_resolved_config(config_path, gt_path);
  require(!config.fast_path_enabled, "fast_path_enabled must default false");
  require(config.fast_path_mode == "rbq", "default fast_path_mode");
  require(config.fast_path_nprobe_N == 128, "default fast_path_nprobe_N");
  require(config.sqg_m == 64, "default sqg_m");
  require(config.sqg_num_iter == 3, "default sqg_num_iter");
  require(config.sqg_ef_build == 100, "default sqg_ef_build");
  require(config.sqg_ef_search == 300, "default sqg_ef_search");
  require(config.sqg_path.find("_sqg_file.bin") != std::string::npos,
          "derived sqg_path");
  require(config.rabitq_1bit_codes_path.find("_rabitq_1bit_codes_file.bin") !=
              std::string::npos,
          "derived rabitq_1bit_codes_path");
  require(config.rabitq_8bit_codes_path.find("_rabitq_8bit_codes_file.bin") !=
              std::string::npos,
          "derived rabitq_8bit_codes_path");
  require(config.per_query_counts.size() == config.n_queries,
          "per-query GT counts must be retained for fast-path metrics");

  const IndexPaths paths = make_diskrange_paths(config.index_prefix);
  require(paths.sqg == config.sqg_path, "IndexPaths sqg must match config");
  require(paths.rabitq_1bit_codes == config.rabitq_1bit_codes_path,
          "IndexPaths 1-bit codes must match config");
  require(paths.rabitq_8bit_codes == config.rabitq_8bit_codes_path,
          "IndexPaths 8-bit codes must match config");
}

void test_fast_path_validation(const std::string& config_path,
                               const std::string& gt_path,
                               const std::string& temp_dir)
{
  auto expect_rejected = [&](const fs::path& path,
                             const std::string& required_a,
                             const std::string& required_b)
  {
    bool rejected = false;
    try
    {
      (void)load_resolved_config(path.string(), gt_path);
    }
    catch (const std::exception& e)
    {
      const std::string message = e.what();
      rejected = message.find(required_a) != std::string::npos &&
                 message.find(required_b) != std::string::npos;
    }
    require(rejected, "expected config rejection containing " + required_a +
                          " and " + required_b);
  };

  const fs::path both_oracles =
      copy_config_with_suffix(config_path, temp_dir, "both_oracles.config",
                              "fast_path_enabled true\n"
                              "data_hnsw_enabled true\n");
  expect_rejected(both_oracles, "fast_path_enabled", "data_hnsw_enabled");

  const fs::path anisotropic =
      copy_config_with_suffix(config_path, temp_dir, "anisotropic.config",
                              "fast_path_enabled true\n"
                              "partition_objective anisotropic\n");
  expect_rejected(anisotropic, "fast_path_enabled", "partition_objective");

  const fs::path too_many =
      copy_config_with_suffix(config_path, temp_dir, "too_many_nprobe.config",
                              "fast_path_enabled true\n"
                              "fast_path_nprobe_N 10001\n");
  expect_rejected(too_many, "fast_path_nprobe_N", "cluster_num");

  const fs::path default_mode = copy_config_with_suffix(
      config_path, temp_dir, "default_fast_path_mode.config",
      "fast_path_enabled true\n");
  const ResolvedConfig default_mode_config =
      load_resolved_config(default_mode.string(), gt_path);
  require(default_mode_config.fast_path_mode == "rbq",
          "fast_path_enabled without fast_path_mode must default to rbq");

  const fs::path exact_mode =
      copy_config_with_suffix(config_path, temp_dir, "exact_mode.config",
                              "fast_path_enabled true\n"
                              "fast_path_mode exact\n");
  const ResolvedConfig exact_config =
      load_resolved_config(exact_mode.string(), gt_path);
  require(exact_config.fast_path_mode == "exact",
          "fast_path_mode exact must be accepted");

  const fs::path invalid_mode = copy_config_with_suffix(
      config_path, temp_dir, "invalid_fast_path_mode.config",
      "fast_path_enabled true\n"
      "fast_path_mode foo\n");
  expect_rejected(invalid_mode, "fast_path_mode", "foo");

  const fs::path path_override =
      copy_config_with_suffix(config_path, temp_dir, "path_override.config",
                              "fast_path_sqg_path /tmp/not_allowed.bin\n");
  expect_rejected(path_override, "unrecognized config key",
                  "fast_path_sqg_path");
}

void test_sqg_roundtrip(const std::string& temp_dir)
{
  const size_t cluster_count = 128;
  const size_t dim = 64;
  const std::vector<float> centroids = make_centroids(cluster_count, dim);

  SQGCentroidIndex built;
  built.build(centroids, cluster_count, dim, 32, 3, 40, 50);
  const fs::path path = fs::path(temp_dir) / "tiny_sqg.bin";
  built.save(path.string());

  SQGCentroidIndex loaded;
  loaded.load(path.string(), cluster_count, dim, 32, 50);
  const std::vector<uint32_t> ids =
      loaded.search(centroids.data() + 7 * dim, 8);
  require(ids.size() == 8, "SQG search must return exactly requested N");
  for (const uint32_t id : ids)
  {
    require(id < cluster_count, "SQG result id out of range");
  }
  require(loaded.padded_dim() == 64, "SQG padded dim for dim=64");
}

void test_zero_residual_rbq_build_sentinel(const std::string& temp_dir)
{
  const size_t dim = 64;
  const size_t cluster_count = 64;
  const size_t sqg_m = 32;
  const size_t ef_search = 50;
  const size_t nprobe = cluster_count;

  const std::vector<float> centroids = make_centroids(cluster_count, dim);

  std::vector<std::vector<size_t>> assignment(cluster_count);
  assignment[0] = {0};
  std::vector<float> vectors(dim, 0.0f);
  std::copy(centroids.begin(), centroids.begin() + dim, vectors.begin());
  const fs::path cluster_path =
      fs::path(temp_dir) / "zero_residual_cluster.bin";
  write_cluster_file(cluster_path, vectors);

  SQGCentroidIndex sqg;
  sqg.build(centroids, cluster_count, dim, sqg_m, 3, 40, ef_search);
  const fs::path sqg_path = fs::path(temp_dir) / "zero_residual_sqg.bin";
  sqg.save(sqg_path.string());

  RBQCodeStorage one_bit;
  one_bit.build_from_cluster_file(cluster_path.string(), assignment, centroids,
                                  dim, sqg.rotator(), 0);
  const fs::path one_path = fs::path(temp_dir) / "zero_residual_one_bit.rbq";
  one_bit.save(one_path.string());

  RBQCodeStorage eight_bit;
  eight_bit.build_from_cluster_file(cluster_path.string(), assignment,
                                    centroids, dim, sqg.rotator(), 7);
  const fs::path eight_path =
      fs::path(temp_dir) / "zero_residual_eight_bit.rbq";
  eight_bit.save(eight_path.string());

  const OneBitFactors one = read_one_bit_factors(one_path, 0, 0);
  require(one.f_add == 0.0f, "1-bit zero-residual f_add sentinel");
  require(one.f_rescale == 0.0f, "1-bit zero-residual f_rescale sentinel");
  require(one.f_error == 0.0f, "1-bit zero-residual f_error sentinel");
  require(float_bits(one.f_add) == 0u,
          "1-bit zero-residual f_add sentinel must be +0.0f");
  require(float_bits(one.f_rescale) == 0u,
          "1-bit zero-residual f_rescale sentinel must be +0.0f");
  require(float_bits(one.f_error) == 0u,
          "1-bit zero-residual f_error sentinel must be +0.0f");
  require(std::isfinite(one.f_add) && std::isfinite(one.f_rescale) &&
              std::isfinite(one.f_error),
          "1-bit zero-residual sentinels must be finite");

  const EightBitFactors eight = read_eight_bit_factors(eight_path, 0, 0);
  require(eight.f_add_ex == 0.0f, "8-bit zero-residual f_add_ex sentinel");
  require(eight.f_rescale_ex == 0.0f,
          "8-bit zero-residual f_rescale_ex sentinel");
  require(float_bits(eight.f_add_ex) == 0u,
          "8-bit zero-residual f_add_ex sentinel must be +0.0f");
  require(float_bits(eight.f_rescale_ex) == 0u,
          "8-bit zero-residual f_rescale_ex sentinel must be +0.0f");
  require(std::isfinite(eight.f_add_ex) && std::isfinite(eight.f_rescale_ex),
          "8-bit zero-residual sentinels must be finite");

  std::vector<size_t> bucket_sizes(cluster_count, 0);
  bucket_sizes[0] = 1;
  FastPathEmptinessOracle oracle(sqg_path.string(), one_path.string(),
                                 eight_path.string(), cluster_count, dim, sqg_m,
                                 ef_search, nprobe, centroids, bucket_sizes);
  std::vector<float> query = vectors;
  query[0] += 0.125f;
  const float expected = 0.125f * 0.125f;
  const FastPathOracleResult result = oracle.evaluate(query.data(), 1.0e9);
  require(result.first_nan_field_mask == 0,
          "zero residual must not NaN early-exit");
  require(std::fabs(result.min_lower_bound - expected) < 1e-5f,
          "zero residual Phase 2 lower bound must equal ||q-c||^2");

  const FastPathOracleResult hit_result =
      oracle.evaluate(query.data(), static_cast<double>(expected + 1e-6f));
  require(!hit_result.is_empty,
          "zero residual true hit must not be false-empty");
}

void test_rbq_storage_and_async_fetch(const std::string& temp_dir)
{
  const size_t dim = 64;
  const size_t cluster_count = 64;
  std::vector<float> centroids = make_centroids(cluster_count, dim);
  centroids[dim] = 10.0f;
  centroids[dim + 1] = 5.0f;
  std::vector<std::vector<size_t>> assignment(cluster_count);
  assignment[0] = {0, 2, 4};
  assignment[1] = {1, 3, 5};
  const std::vector<float> vectors = make_cluster_vectors(dim);
  const fs::path cluster_path = fs::path(temp_dir) / "cluster_payload.bin";
  write_cluster_file(cluster_path, vectors);

  SQGCentroidIndex sqg;
  sqg.build(centroids, cluster_count, dim, 32, 3, 40, 50);

  RBQCodeStorage one_bit;
  one_bit.build_from_cluster_file(cluster_path.string(), assignment, centroids,
                                  dim, sqg.rotator(), 0);
  const fs::path one_path = fs::path(temp_dir) / "one_bit.rbq";
  one_bit.save(one_path.string());

  RBQCodeStorage loaded_one;
  loaded_one.load_1bit(one_path.string());
  require(loaded_one.n_data() == 6, "1-bit n_data");
  require(loaded_one.cluster_count() == cluster_count, "1-bit cluster count");
  require(loaded_one.cluster_vec_count(0) == 3, "cluster 0 vec count");
  require(loaded_one.cluster_vec_count(1) == 3, "cluster 1 vec count");
  require(loaded_one.total_code_entries() == 6, "no replication");
  require(loaded_one.cluster_block_count(0) == 1, "1-bit batch block count");
  require(loaded_one.batch_code_ptr(0, 0) != nullptr,
          "loaded 1-bit FastScan batch bytes");
  require(std::isfinite(loaded_one.batch_f_error(0, 1)),
          "loaded 1-bit batch f_error");

  RBQCodeStorage eight_bit;
  eight_bit.build_from_cluster_file(cluster_path.string(), assignment,
                                    centroids, dim, sqg.rotator(), 7);
  const fs::path eight_path = fs::path(temp_dir) / "eight_bit.rbq";
  eight_bit.save(eight_path.string());

  RBQCodeStorage loaded_eight;
  loaded_eight.open_8bit_for_pread(eight_path.string());
  std::vector<BorderlineRef> refs;
  for (size_t i = 0; i < 24; ++i)
  {
    refs.push_back({i % 2, i % 3});
  }
  const std::vector<BorderlineCode> async_codes =
      async_fetch_8bit_codes(refs, loaded_eight).get();
  require(async_codes.size() == refs.size(), "async fetch count");
  for (size_t i = 0; i < refs.size(); ++i)
  {
    const std::vector<char> direct =
        loaded_eight.read_code(refs[i].cluster_id, refs[i].local_index);
    require(async_codes[i].code_bytes == direct,
            "async pread bytes must match direct pread");
  }
}

void test_fast_path_oracle_evaluate(const std::string& temp_dir)
{
  const size_t dim = 64;
  const size_t cluster_count = 64;
  const size_t sqg_m = 32;
  const size_t ef_search = 50;
  const size_t nprobe = 2;
  std::vector<float> centroids = make_centroids(cluster_count, dim);
  centroids[dim] = 10.0f;
  centroids[dim + 1] = 5.0f;
  std::vector<std::vector<size_t>> assignment(cluster_count);
  assignment[0] = {0, 2, 4};
  assignment[1] = {1, 3, 5};
  const std::vector<float> vectors = make_cluster_vectors(dim);
  const fs::path cluster_path =
      fs::path(temp_dir) / "oracle_cluster_payload.bin";
  write_cluster_file(cluster_path, vectors);

  SQGCentroidIndex sqg;
  sqg.build(centroids, cluster_count, dim, sqg_m, 3, 40, ef_search);
  const fs::path sqg_path = fs::path(temp_dir) / "oracle_sqg.bin";
  sqg.save(sqg_path.string());

  RBQCodeStorage one_bit;
  one_bit.build_from_cluster_file(cluster_path.string(), assignment, centroids,
                                  dim, sqg.rotator(), 0);
  const fs::path one_path = fs::path(temp_dir) / "oracle_one_bit.rbq";
  one_bit.save(one_path.string());

  RBQCodeStorage eight_bit;
  eight_bit.build_from_cluster_file(cluster_path.string(), assignment,
                                    centroids, dim, sqg.rotator(), 7);
  const fs::path eight_path = fs::path(temp_dir) / "oracle_eight_bit.rbq";
  eight_bit.save(eight_path.string());

  std::vector<size_t> bucket_sizes(cluster_count, 0);
  bucket_sizes[0] = 3;
  bucket_sizes[1] = 3;
  FastPathEmptinessOracle oracle(sqg_path.string(), one_path.string(),
                                 eight_path.string(), cluster_count, dim, sqg_m,
                                 ef_search, nprobe, centroids, bucket_sizes);
  std::vector<size_t> bad_bucket_sizes = bucket_sizes;
  bad_bucket_sizes[0] = 2;
  bool rejected_stale_rbq = false;
  try
  {
    FastPathEmptinessOracle bad_oracle(sqg_path.string(), one_path.string(),
                                       eight_path.string(), cluster_count, dim,
                                       sqg_m, ef_search, nprobe, centroids,
                                       bad_bucket_sizes);
  }
  catch (const std::exception& e)
  {
    rejected_stale_rbq =
        std::string(e.what()).find("vec_count mismatch") != std::string::npos;
  }
  require(
      rejected_stale_rbq,
      "oracle must reject RBQ artifacts that mismatch metadata bucket sizes");
  std::vector<float> near_query(dim, 0.0f);
  near_query[0] = 0.01f;
  const FastPathOracleResult near = oracle.evaluate(near_query.data(), 0.25f);
  require(!near.is_empty, "near query must fall through");
  require(near.phase1_candidate_count == nprobe,
          "phase 1 must return exactly N candidates");

  const std::vector<float> rbq_empty_candidate_query =
      find_empty_candidate_query(sqg, centroids, bucket_sizes, dim, nprobe);
  const FastPathOracleResult rbq_empty_candidates =
      oracle.evaluate(rbq_empty_candidate_query.data(), 0.25f);
  require(rbq_empty_candidates.phase2_scanned_vectors == 0,
          "RBQ all-empty candidate fixture must scan zero vectors");
  require(!rbq_empty_candidates.is_empty,
          "RBQ mode must fall through when candidate scan sees zero vectors");

  const size_t exhaustive_nprobe = cluster_count;
  const size_t exhaustive_ef_search = 128;
  FastPathEmptinessOracle exhaustive_oracle(
      sqg_path.string(), one_path.string(), eight_path.string(), cluster_count,
      dim, sqg_m, exhaustive_ef_search, exhaustive_nprobe, centroids,
      bucket_sizes);
  std::vector<float> far_query(dim, 1000.0f);
  const FastPathOracleResult far =
      exhaustive_oracle.evaluate(far_query.data(), 0.0001f);
  require(far.phase1_candidate_count == exhaustive_nprobe,
          "far query phase 1 must return exactly N candidates");
  require(far.phase2_scanned_vectors > 0,
          "exhaustive RBQ query must scan populated clusters");

  std::vector<std::future<bool>> futures;
  for (size_t i = 0; i < 32; ++i)
  {
    futures.push_back(std::async(std::launch::async,
                                 [&exhaustive_oracle, &far_query]
                                 {
                                   const FastPathOracleResult result =
                                       exhaustive_oracle.evaluate(
                                           far_query.data(), 0.0001f);
                                   return result.phase1_candidate_count > 0;
                                 }));
  }
  for (auto& future : futures)
  {
    require(future.get(), "concurrent SQG-backed evaluate must stay valid");
  }

  const std::vector<size_t> populated_clusters = {0, 1};

  const fs::path phase2_nan_one_path =
      fs::path(temp_dir) / "oracle_one_bit_phase2_nan.rbq";
  fs::copy_file(one_path, phase2_nan_one_path,
                fs::copy_options::overwrite_existing);
  poison_one_bit_f_add_with_nan(phase2_nan_one_path, populated_clusters);
  FastPathEmptinessOracle phase2_nan_oracle(
      sqg_path.string(), phase2_nan_one_path.string(), eight_path.string(),
      cluster_count, dim, sqg_m, exhaustive_ef_search, exhaustive_nprobe,
      centroids, bucket_sizes);
  const FastPathOracleResult phase2_nan =
      phase2_nan_oracle.evaluate(near_query.data(), 0.0001f);
  require(!phase2_nan.is_empty,
          "non-finite Phase 2 lower bound must fall through");

  const fs::path phase3_nan_eight_path =
      fs::path(temp_dir) / "oracle_eight_bit_phase3_nan.rbq";
  fs::copy_file(eight_path, phase3_nan_eight_path,
                fs::copy_options::overwrite_existing);
  poison_eight_bit_f_add_with_nan(phase3_nan_eight_path, populated_clusters);
  FastPathEmptinessOracle phase3_nan_oracle(
      sqg_path.string(), one_path.string(), phase3_nan_eight_path.string(),
      cluster_count, dim, sqg_m, exhaustive_ef_search, exhaustive_nprobe,
      centroids, bucket_sizes);
  const FastPathOracleResult phase3_nan =
      phase3_nan_oracle.evaluate(near_query.data(), 1.0e9);
  require(!phase3_nan.is_empty,
          "non-finite Phase 3 lower bound must fall through");
}

void test_fast_path_oracle_exact_mode(const std::string& temp_dir)
{
  const size_t dim = 64;
  const size_t cluster_count = 64;
  const size_t sqg_m = 32;
  const size_t ef_search = 128;
  const size_t nprobe = cluster_count;
  std::vector<float> centroids = make_centroids(cluster_count, dim);
  centroids[dim] = 10.0f;
  centroids[dim + 1] = 5.0f;
  std::vector<std::vector<size_t>> assignment(cluster_count);
  assignment[0] = {0, 1, 2};
  assignment[1] = {3, 4, 5};
  const std::vector<float> vectors = make_cluster_vectors(dim);
  const fs::path cluster_path =
      fs::path(temp_dir) / "exact_cluster_payload.bin";
  const fs::path metadata_path = fs::path(temp_dir) / "exact_metadata.bin";
  write_cluster_file(cluster_path, vectors);
  write_test_metadata_file(metadata_path, dim, centroids, assignment);

  SQGCentroidIndex sqg;
  sqg.build(centroids, cluster_count, dim, sqg_m, 3, 40, ef_search);
  const fs::path sqg_path = fs::path(temp_dir) / "exact_sqg.bin";
  sqg.save(sqg_path.string());

  ClusterReader cluster_reader(cluster_path.string(), metadata_path.string(), 8,
                               1);
  cluster_reader.readMetaData();

  FastPathEmptinessOracle exact_oracle(
      "exact", sqg_path.string(), "", "", cluster_count, dim, sqg_m, ef_search,
      nprobe, centroids, cluster_reader.bucket_sizes, &cluster_reader);

  bool rejected_missing_rbq = false;
  try
  {
    FastPathEmptinessOracle rbq_missing_oracle(
        "rbq", sqg_path.string(),
        (fs::path(temp_dir) / "missing_one_bit.rbq").string(),
        (fs::path(temp_dir) / "missing_eight_bit.rbq").string(), cluster_count,
        dim, sqg_m, ef_search, nprobe, centroids, cluster_reader.bucket_sizes,
        &cluster_reader);
  }
  catch (const std::exception& e)
  {
    const std::string message = e.what();
    rejected_missing_rbq =
        message.find("missing RBQ artifacts") != std::string::npos &&
        message.find("rebuild with matching mode") != std::string::npos;
  }
  require(rejected_missing_rbq,
          "rbq mode must fail fast when RBQ artifacts are missing");

  const size_t empty_nprobe = 2;
  const std::vector<float> exact_empty_candidate_query =
      find_empty_candidate_query(sqg, centroids, cluster_reader.bucket_sizes,
                                 dim, empty_nprobe);
  FastPathEmptinessOracle exact_empty_candidate_oracle(
      "exact", sqg_path.string(), "", "", cluster_count, dim, sqg_m, ef_search,
      empty_nprobe, centroids, cluster_reader.bucket_sizes, &cluster_reader);
  const FastPathOracleResult exact_empty_candidates = evaluate_in_omp(
      exact_empty_candidate_oracle, exact_empty_candidate_query.data(), 0.25f);
  require(exact_empty_candidates.phase2_scanned_vectors == 0,
          "exact all-empty candidate fixture must scan zero vectors");
  require(!exact_empty_candidates.is_empty,
          "exact mode must fall through when candidate scan sees zero vectors");

  const std::vector<float> near_query = find_populated_candidate_query(
      sqg, centroids, cluster_reader.bucket_sizes, dim, nprobe);
  const FastPathOracleResult exact_near =
      evaluate_in_omp(exact_oracle, near_query.data(), 0.25f);
  require(!exact_near.is_empty, "exact near query must fall through");
  const std::vector<size_t> baseline_near_hits =
      exact_hits(vectors, dim, near_query.data(), 0.25f);
  const std::vector<size_t> fast_path_near_hits =
      exact_near.is_empty ? std::vector<size_t>{} : baseline_near_hits;
  require(fast_path_near_hits == baseline_near_hits,
          "exact fall-through hit list must match baseline");
  require(exact_near.phase1_candidate_count == nprobe,
          "exact phase 1 must return exactly N candidates");
  require(exact_near.phase2_scanned_vectors > 0,
          "exact mode must scan raw vectors in candidate clusters");
  require(exact_near.phase2_borderline_count == 0,
          "exact mode must not report RBQ borderline vectors");
  require(exact_near.phase3_refined_count == 0,
          "exact mode must not run RBQ Phase 3");
  require(std::isfinite(exact_near.min_lower_bound),
          "exact trace min distance must be finite");

  std::vector<float> far_query = near_query;
  far_query[dim - 1] += 0.5f;
  const FastPathOracleResult exact_far =
      evaluate_in_omp(exact_oracle, far_query.data(), 0.0001f);
  require(exact_far.is_empty, "exact far query must be proven empty");
  require(exact_far.phase2_scanned_vectors > 0,
          "exact empty-proof query must scan populated clusters");
  require(exact_hits(vectors, dim, far_query.data(), 0.0001f).empty(),
          "exact empty-proof query baseline must be empty");

  ClusterReader concurrent_reader(cluster_path.string(), metadata_path.string(),
                                  8, 4);
  concurrent_reader.readMetaData();
  FastPathEmptinessOracle concurrent_exact_oracle(
      "exact", sqg_path.string(), "", "", cluster_count, dim, sqg_m, ef_search,
      nprobe, centroids, concurrent_reader.bucket_sizes, &concurrent_reader);
  std::vector<int> concurrent_empty(16, 0);
#pragma omp parallel for num_threads(4)
  for (size_t i = 0; i < concurrent_empty.size(); ++i)
  {
    concurrent_empty[i] =
        concurrent_exact_oracle.evaluate(far_query.data(), 0.0001f).is_empty
            ? 1
            : 0;
  }
  for (const int empty : concurrent_empty)
  {
    require(empty == 1,
            "concurrent exact evaluate must use per-thread IO slots");
  }

  std::vector<float> nan_vectors = vectors;
  for (size_t row = 0; row < nan_vectors.size() / dim; ++row)
  {
    nan_vectors[row * dim] = std::numeric_limits<float>::quiet_NaN();
  }
  const fs::path nan_cluster_path =
      fs::path(temp_dir) / "exact_nan_cluster_payload.bin";
  write_cluster_file(nan_cluster_path, nan_vectors);
  ClusterReader nan_cluster_reader(nan_cluster_path.string(),
                                   metadata_path.string(), 8, 1);
  nan_cluster_reader.readMetaData();
  FastPathEmptinessOracle exact_nan_oracle(
      "exact", sqg_path.string(), "", "", cluster_count, dim, sqg_m, ef_search,
      nprobe, centroids, nan_cluster_reader.bucket_sizes, &nan_cluster_reader);
  const FastPathOracleResult exact_nan =
      evaluate_in_omp(exact_nan_oracle, near_query.data(), 0.0001f);
  require(!exact_nan.is_empty,
          "non-finite exact L2 distance must conservatively fall through");
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: fast_path_emptiness_oracle_unit_test <config> <gt>\n";
    return 2;
  }

  try
  {
    const std::string temp_dir = make_temp_dir();
    test_fast_path_config_defaults_and_paths(argv[1], argv[2]);
    test_fast_path_validation(argv[1], argv[2], temp_dir);
    test_sqg_roundtrip(temp_dir);
    test_zero_residual_rbq_build_sentinel(temp_dir);
    test_rbq_storage_and_async_fetch(temp_dir);
    test_fast_path_oracle_evaluate(temp_dir);
    test_fast_path_oracle_exact_mode(temp_dir);
    fs::remove_all(temp_dir);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
