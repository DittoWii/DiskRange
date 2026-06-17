#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../lib/ClusterIO.h"
#include "../lib/ConfigLoader.h"
#include "../lib/ConfigReader.h"
#include "../lib/DiskRange.h"
#include "../lib/IoQueueDepth.h"
#include "../lib/RunResult.h"

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

void require_contains(const std::string& text, const std::string& needle)
{
  require(text.find(needle) != std::string::npos,
          "expected message to contain '" + needle + "': " + text);
}

void require_mem_breakdown_keys(const std::string& message)
{
  for (const char* key : {"metadata=", "rbq_1bit=", "data_hnsw=",
                          "query=", "per_thread=", "sqg=",
                          "rotated_centroids=", "trace=",
                          "hit_bookkeeping=", "cluster_cache=",
                          "cell_cache=", "slack=", "budget="})
  {
    require_contains(message, key);
  }
}

std::string make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "cluster_reader_mem_budget_test";
  fs::remove_all(base);
  fs::create_directories(base);
  return base.string();
}

void write_cluster_file(const fs::path& path, const std::vector<float>& vectors)
{
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create cluster file");
  out.write(reinterpret_cast<const char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  require(static_cast<bool>(out), "failed to write cluster file");
}

void write_sparse_file(const fs::path& path, size_t bytes)
{
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create sparse file");
  if (bytes != 0)
  {
    out.seekp(static_cast<std::streamoff>(bytes - 1));
    out.put('\0');
  }
  require(static_cast<bool>(out), "failed to size sparse file");
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
  require(static_cast<bool>(out), "failed to create metadata file");
  out.write(kMetadataMagic.data(),
            static_cast<std::streamsize>(kMetadataMagic.size()));
  const uint64_t schema_version = kMetadataSchemaVersion;
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
  out.write(reinterpret_cast<const char*>(pca_residual_radius.data()),
            static_cast<std::streamsize>(pca_residual_radius.size() *
                                         sizeof(float)));
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
  out.write(reinterpret_cast<const char*>(cell_vec_offsets.data()),
            static_cast<std::streamsize>(cell_vec_offsets.size() *
                                         sizeof(uint64_t)));
  write_aligned_padding(out, "cell_bounds");
  if (!cell_bounds.empty())
  {
    out.write(reinterpret_cast<const char*>(cell_bounds.data()),
              static_cast<std::streamsize>(cell_bounds.size() * sizeof(float)));
  }
  require(static_cast<bool>(out), "failed to write metadata file");
}

struct Fixture
{
  std::string temp_dir;
  fs::path cluster_path;
  fs::path metadata_path;
};

Fixture make_fixture(size_t dim = 8, size_t cluster_count = 4,
                     size_t points_per_cluster = 4)
{
  Fixture fixture;
  fixture.temp_dir = make_temp_dir();
  fixture.cluster_path = fs::path(fixture.temp_dir) / "cluster.bin";
  fixture.metadata_path = fs::path(fixture.temp_dir) / "metadata.bin";

  std::vector<std::vector<size_t>> assignment(cluster_count);
  std::vector<float> vectors;
  vectors.reserve(cluster_count * points_per_cluster * dim);
  size_t row = 0;
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    for (size_t i = 0; i < points_per_cluster; ++i)
    {
      assignment[cid].push_back(row++);
      for (size_t col = 0; col < dim; ++col)
      {
        vectors.push_back(static_cast<float>(cid * 10 + i + col) * 0.01f);
      }
    }
  }
  std::vector<float> centroids(cluster_count * dim, 0.0f);
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    centroids[cid * dim] = static_cast<float>(cid);
  }
  write_cluster_file(fixture.cluster_path, vectors);
  write_test_metadata_file(fixture.metadata_path, dim, centroids, assignment);
  return fixture;
}

size_t fixture_metadata_resident_bytes(size_t dim = 8, size_t cluster_count = 4,
                                       size_t points_per_cluster = 4)
{
  const size_t n = cluster_count * points_per_cluster;
  const size_t total_cells = cluster_count;
  return cluster_count * sizeof(size_t)                  // bucket_sizes
         + cluster_count * dim * sizeof(float)           // centroids
         + cluster_count * sizeof(float)                 // radii
         + cluster_count * sizeof(float)                 // pca_residual_radius
         + cluster_count * sizeof(uint32_t)              // pca_effective_rank
         + (cluster_count + 1) * sizeof(uint64_t)        // cluster_cell_offsets
         + (total_cells + 1) * sizeof(uint64_t)          // cell_vec_offsets
         + (cluster_count + 1) * sizeof(size_t)          // cluster_bound_offsets
         + total_cells * 2 * sizeof(float)               // cell_bounds
         + cluster_count * sizeof(size_t)                // file_pos
         + n * sizeof(size_t);                           // assignment payload
}

ClusterReader::MemBudgetInputs budget_for_io(size_t io_budget_bytes)
{
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes() + kReservedSlack + io_budget_bytes;
  return budget;
}

void read_or_throw(ClusterReader& reader)
{
  reader.readMetaData();
}

void test_happy_path_keeps_requested_depth_and_slots()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 8,
                       ClusterReader::MemBudgetInputs{}, 2);
  read_or_throw(reader);
  require(reader.effective_io_queue_depth() == 8, "happy depth");
  require(reader.effective_n_io_slots() == 2, "happy slots");
  require(reader.io_buffer_size() >= PAGE_SIZE, "buffer size");
}

void test_compress_depth_before_slots()
{
  const Fixture fixture = make_fixture();
  const size_t buf_size = 8192;
  ClusterReader reader(
      fixture.cluster_path.string(), fixture.metadata_path.string(), 16,
      budget_for_io(2 * 4 * buf_size), 2);
  read_or_throw(reader);
  require(reader.effective_io_queue_depth() == 4, "depth compressed to 4");
  require(reader.effective_n_io_slots() == 2, "slots unchanged");
}

void test_compress_slots_after_depth_floor()
{
  const Fixture fixture = make_fixture();
  const size_t buf_size = 8192;
  ClusterReader reader(
      fixture.cluster_path.string(), fixture.metadata_path.string(), 16,
      budget_for_io(2 * 4 * buf_size), 4);
  read_or_throw(reader);
  require(reader.effective_io_queue_depth() == 4, "depth at floor");
  require(reader.effective_n_io_slots() == 2, "slots compressed to 2");
}

void test_depth_floored_to_power_of_two()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 50,
                       ClusterReader::MemBudgetInputs{}, 1);
  read_or_throw(reader);
  require(reader.effective_io_queue_depth() == 32, "50 floors to 32");
}

void test_requested_depth_below_effective_floor_preserved()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 2,
                       ClusterReader::MemBudgetInputs{}, 1);
  read_or_throw(reader);
  require(reader.effective_io_queue_depth() == 2, "requested depth 2 kept");
}

void test_fail_below_fixed_reports_all_fields()
{
  const Fixture fixture = make_fixture();
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = 1;
  budget.rbq_1bit_resident_bytes = 2;
  budget.data_hnsw_resident_bytes = 9;
  budget.query_resident_bytes = 3;
  budget.per_thread_resident_bytes = 4;
  budget.sqg_resident_bytes = 5;
  budget.rotated_centroids_resident_bytes = 6;
  budget.trace_resident_bytes = 7;
  budget.hit_bookkeeping_resident_bytes = 8;

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 8, budget, 1);
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    const std::string message = e.what();
    require_contains(message, "mem_budget too small");
    require_mem_breakdown_keys(message);
  }
  require(failed, "fixed-resident failure expected");
}

void test_fail_below_fixed_fast_path_reports_inactive_data_hnsw()
{
  const Fixture fixture = make_fixture();
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = 1;
  budget.rbq_1bit_resident_bytes = 2;

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 8, budget, 1);
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "mem_budget too small");
    require_contains(e.what(), "rbq_1bit=");
    require_contains(e.what(), "data_hnsw=0");
  }
  require(failed, "fast-path fixed-resident failure expected");
}

void test_fail_below_fixed_data_hnsw_reports_inactive_rbq()
{
  const Fixture fixture = make_fixture();
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = 1;
  budget.data_hnsw_resident_bytes = 2;

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 8, budget, 1);
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "mem_budget too small");
    require_contains(e.what(), "rbq_1bit=0");
    require_contains(e.what(), "data_hnsw=");
  }
  require(failed, "data-hnsw fixed-resident failure expected");
}

void test_fail_io_budget_reports_io_fields()
{
  const Fixture fixture = make_fixture();
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes() + kReservedSlack + PAGE_SIZE - 1;

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 8, budget, 1);
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    const std::string message = e.what();
    require_contains(message, "cannot fit IO buffers in mem_budget");
    require_contains(message, "slots=");
    require_contains(message, "depth=");
    require_contains(message, "buf_size=");
    require_contains(message, "io_budget=");
    require_mem_breakdown_keys(message);
  }
  require(failed, "io-budget failure expected");
}

void test_ring_entries_guard_reports_budget()
{
  bool failed = false;
  try
  {
    ClusterReader::verify_io_buffer_budget(2, 8, PAGE_SIZE, PAGE_SIZE);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "io_uring registered more buffers than budgeted");
    require_contains(e.what(), "slots=");
    require_contains(e.what(), "ring_entries=");
    require_contains(e.what(), "buf_size=");
    require_contains(e.what(), "io_budget=");
  }
  require(failed, "ring-entry budget guard expected");
}

void test_fixed_resident_counts_hit_bookkeeping()
{
  const Fixture fixture = make_fixture();
  const size_t n_queries = 10000;
  const size_t total_in_range = 45861;
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = 1;
  budget.hit_bookkeeping_resident_bytes =
      n_queries * 64 * sizeof(HitRef) +
      total_in_range *
          (sizeof(HitRef) + sizeof(uint8_t) +
           sizeof(std::tuple<size_t, uint32_t, size_t>) + kHitLookupOverhead);

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 8, budget, 1);
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "hit_bookkeeping=");
  }
  require(failed, "hit bookkeeping must contribute to fixed resident");
}

void test_legacy_brace_init_keeps_all_four_cache_fields_zero()
{
  ClusterReader::MemBudgetInputs budget{};
  require(budget.cluster_cache_requested_bytes == 0,
          "default cluster_cache_requested_bytes");
  require(budget.cluster_cache_resident_bytes == 0,
          "default cluster_cache_resident_bytes");
  require(budget.cell_cache_requested_bytes == 0,
          "default cell_cache_requested_bytes");
  require(budget.cell_cache_resident_bytes == 0,
          "default cell_cache_resident_bytes");
}

void test_config_reader_accepts_cell_cache_share_pct()
{
  const fs::path dir = fs::temp_directory_path() / "cell_cache_config_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path config = dir / "diskrange.config";
  {
    std::ofstream out(config);
    out << "cluster_num 4\n";
    out << "K 2\n";
    out << "mem_budget 1\n";
    out << "mode disk\n";
    out << "gorder_window 0\n";
    out << "cell_cache_share_pct 50\n";
  }

  ConfigReader reader(config.string());
  require(reader.cell_cache_share_pct == 50.0f,
          "ConfigReader parses cell_cache_share_pct");
}

void test_checked_mul_saturates()
{
  require(checked_mul(size_t{0}, size_t{123}) == 0, "checked_mul zero lhs");
  require(checked_mul(size_t{7}, size_t{9}) == 63, "checked_mul normal");
  require(checked_mul(std::numeric_limits<size_t>::max(), size_t{2}) ==
              std::numeric_limits<size_t>::max(),
          "checked_mul overflow saturates");
}

void test_reconcile_two_phase_allocates_cache()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = kClusterCacheShardCount * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cluster_cache_requested_bytes;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_requested_bytes() ==
              budget.cluster_cache_requested_bytes,
          "requested cache budget preserved");
  require(reader.cluster_cache_resident_bytes() > 0,
          "cache resident allocated");
  require(reader.cluster_cache_resident_bytes() <=
              budget.cluster_cache_requested_bytes,
          "resident cache budget <= requested");
}

void test_requested_preserves_through_disabled_mode()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = kClusterCacheShardCount * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cluster_cache_requested_bytes;

  setenv("DJ_CLUSTER_CACHE_DISABLE", "1", 1);
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  unsetenv("DJ_CLUSTER_CACHE_DISABLE");
  require(reader.cluster_cache_requested_bytes() ==
              budget.cluster_cache_requested_bytes,
          "disabled mode preserves requested cache budget");
  require(reader.cluster_cache_resident_bytes() == 0,
          "disabled mode clears resident cache budget");
}

void test_cell_cache_takes_half_when_share_50()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  const size_t cache_pool = 8 * kClusterCacheShardCount * buf_size;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = cache_pool;
  budget.cell_cache_requested_bytes = cache_pool;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) + cache_pool;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2, 50.0f);
  read_or_throw(reader);
  require(reader.cell_cache_resident_bytes() > 0,
          "cell cache gets share");
  require(reader.cluster_cache_resident_bytes() > 0,
          "cluster cache gets share");
  require(reader.cluster_cache_resident_bytes() +
              reader.cell_cache_resident_bytes() <=
          cache_pool,
          "cache sum bounded by pool");
}

void test_cell_env_disable_redirects_full_pool_to_cluster()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  const size_t cache_pool = 8 * kClusterCacheShardCount * buf_size;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = cache_pool;
  budget.cell_cache_requested_bytes = cache_pool;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) + cache_pool;

  ClusterReader baseline(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 4, budget, 2, 0.0f);
  read_or_throw(baseline);
  const size_t baseline_cluster = baseline.cluster_cache_resident_bytes();

  setenv("DJ_CELL_CACHE_DISABLE", "1", 1);
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2, 50.0f);
  read_or_throw(reader);
  unsetenv("DJ_CELL_CACHE_DISABLE");
  require(reader.cluster_cache_resident_bytes() == baseline_cluster,
          "cell env disable redirects pool to cluster");
  require(reader.cell_cache_resident_bytes() == 0,
          "cell env disable clears cell resident");
}

void test_cluster_env_disable_with_share_redirects_to_cell()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  const size_t cache_pool = 8 * kClusterCacheShardCount * buf_size;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = cache_pool;
  budget.cell_cache_requested_bytes = cache_pool;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) + cache_pool;

  setenv("DJ_CLUSTER_CACHE_DISABLE", "1", 1);
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2, 50.0f);
  read_or_throw(reader);
  unsetenv("DJ_CLUSTER_CACHE_DISABLE");
  require(reader.cluster_cache_resident_bytes() == 0,
          "cluster env disable clears cluster resident");
  require(reader.cell_cache_resident_bytes() == cache_pool,
          "cluster env disable redirects pool to cell");
}

void test_cluster_env_disable_with_zero_share_goes_to_iopool()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  const size_t cache_pool = 8 * kClusterCacheShardCount * buf_size;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = cache_pool;
  budget.cell_cache_requested_bytes = cache_pool;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) + cache_pool;

  setenv("DJ_CLUSTER_CACHE_DISABLE", "1", 1);
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2, 0.0f);
  read_or_throw(reader);
  unsetenv("DJ_CLUSTER_CACHE_DISABLE");
  require(reader.cluster_cache_resident_bytes() == 0,
          "cluster env disable clears cluster resident");
  require(reader.cell_cache_resident_bytes() == 0,
          "zero share does not covertly enable cell cache");
}

void test_tiny_cell_budget_returned_to_iopool()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = 0;
  budget.cell_cache_requested_bytes = PAGE_SIZE;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cell_cache_requested_bytes;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2, 50.0f);
  read_or_throw(reader);
  require(reader.cell_cache_resident_bytes() == 0,
          "tiny cell budget returned to io_pool");
  require(!reader.cell_cache_for_test()->active(),
          "tiny cell cache inactive");
}

void test_reconcile_rejects_invalid_cell_cache_share_pct()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = std::numeric_limits<size_t>::max();

  bool failed = false;
  try
  {
    ClusterReader reader(fixture.cluster_path.string(),
                         fixture.metadata_path.string(), 4, budget, 2,
                         std::numeric_limits<float>::quiet_NaN());
    read_or_throw(reader);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "cell_cache_share_pct");
  }
  require(failed, "invalid cell_cache_share_pct rejected");
}

void test_refined_cap_reserves_io_preferred()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = size_t{1000} * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      (kClusterCacheShardCount * buf_size);

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.effective_n_io_slots() == 2,
          "refined cache cap preserves preferred slots");
  require(reader.cluster_cache_resident_bytes() <=
              kClusterCacheShardCount * buf_size,
          "resident cache budget capped by available_for_cache");
}

void test_tight_budget_cache_yields_to_io()
{
  const Fixture fixture = make_fixture();
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = size_t{1000} * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes() + kReservedSlack +
      (size_t{1} * kMinEffectiveIoQueueDepth * buf_size);

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 16, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_resident_bytes() == 0,
          "tight budget gives cache zero bytes");
  require(reader.effective_n_io_slots() == 1, "tight budget compresses slots");
  require(reader.effective_io_queue_depth() == kMinEffectiveIoQueueDepth,
          "tight budget keeps depth at floor");
}

void test_effective_capacity_reflects_shard_truncation()
{
  constexpr size_t kClusterCount = 1000;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = size_t{1000} * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cluster_cache_requested_bytes;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_resident_bytes() ==
              size_t{960} * buf_size,
          "cache resident reflects 64-shard truncation");
  require(reader.cluster_cache_for_test()->stats().entries_max == 960,
          "cache entries reflect 64-shard truncation");
}

void test_reconcile_cache_not_capped_by_file_size()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 1);
  const size_t buf_size = 4096 * 2;
  require(fs::file_size(fixture.cluster_path) <
              kClusterCount * buf_size,
          "fixture file smaller than cluster_num * buf_size");
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = kClusterCount * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 1) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cluster_cache_requested_bytes;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_resident_bytes() == 64 * buf_size,
          "cache resident capped by cluster_num*buf_size not file size");
  require(reader.cluster_cache_resident_bytes() > fs::file_size(fixture.cluster_path),
          "cache resident may exceed compact cluster file size");
}

void test_reconcile_cache_capped_by_kClusterCacheMaxBytes()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 1);
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = std::numeric_limits<size_t>::max();
  budget.mem_budget_bytes = std::numeric_limits<size_t>::max();

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_resident_bytes() <= kClusterCacheMaxBytes,
          "cache resident capped by kClusterCacheMaxBytes");
}

void test_sub_shard_count_budget_yields_zero_effective()
{
  constexpr size_t kClusterCount = kClusterCacheShardCount + 1;
  const Fixture fixture = make_fixture(8, kClusterCount, 4);
  const size_t buf_size = 8192;
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = size_t{30} * buf_size;
  budget.mem_budget_bytes =
      fixture_metadata_resident_bytes(8, kClusterCount, 4) + kReservedSlack +
      (size_t{2} * kMinEffectiveIoQueueDepth * buf_size) +
      budget.cluster_cache_requested_bytes;

  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 2);
  read_or_throw(reader);
  require(reader.cluster_cache_resident_bytes() == 0,
          "sub-shard budget yields zero resident cache");
  require(!reader.cluster_cache_for_test()->active(),
          "sub-shard budget inactive");
}

void test_slot_id_maps_by_modulo_when_slots_are_compressed()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 8,
                       ClusterReader::MemBudgetInputs{}, 1);
  read_or_throw(reader);
  std::vector<size_t> clusters = {0};
  size_t callbacks = 0;
  reader.submit_and_drain(omp_get_max_threads() + 3, clusters,
                          [&](size_t, const void*, size_t, size_t) {
                            ++callbacks;
                          });
  require(callbacks == 1, "thread id should map modulo effective slots");
}

ResolvedConfig deep1m_like_config(const std::string& gt_path,
                                  const fs::path& temp_dir)
{
  std::ifstream in(gt_path);
  require(static_cast<bool>(in), "failed to open deep1m gt json");
  nlohmann::json tree;
  in >> tree;

  ResolvedConfig config;
  config.cluster_num = 10000;
  config.K = 245;
  config.mem_budget = 1.0f;
  config.mode = "disk";
  config.io_queue_depth = 256;
  config.pca_rank = 32;
  config.slab_count = 128;
  config.n_queries = tree.at("n_queries").get<size_t>();
  config.n_base = tree.at("n_base").get<size_t>();
  config.dim = tree.at("dim").get<size_t>();
  config.radius_squared = tree.at("radius_squared").get<double>();
  config.total_in_range = tree.at("total_in_range").get<size_t>();
  config.cluster_path = (temp_dir / "cluster.bin").string();
  config.metadata_path = (temp_dir / "metadata.bin").string();
  config.query_file = (temp_dir / "query.fbin").string();
  config.data_file = (temp_dir / "base.fbin").string();
  config.sqg_path = (temp_dir / "sqg.bin").string();
  config.rabitq_1bit_codes_path = (temp_dir / "rbq1.bin").string();
  config.rabitq_8bit_codes_path = (temp_dir / "rbq8.bin").string();
  config.data_hnsw_path = (temp_dir / "data_hnsw.bin").string();
  return config;
}

void test_preflight_counts_deep1m_gt_hit_bookkeeping(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.query_file = (fs::path(fixture.temp_dir) / "missing_query.fbin").string();

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  const size_t expected =
      config.n_queries * 64 * sizeof(HitRef) +
      config.total_in_range *
          (sizeof(HitRef) + sizeof(uint8_t) +
           sizeof(std::tuple<size_t, uint32_t, size_t>) + kHitLookupOverhead);
  require(preflight.hit_bookkeeping_bytes == expected,
          "deep1m gt hit bookkeeping formula");
  require(preflight.budget_inputs.hit_bookkeeping_resident_bytes == expected,
          "budget inputs hit bookkeeping");
}

void test_preflight_both_caches_full_envelope_when_cell_share_set(
    const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.query_file = (fs::path(fixture.temp_dir) / "missing_query.fbin").string();
  config.cell_cache_share_pct = 50.0f;

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  require(preflight.cluster_cache_requested_bytes > 0,
          "cluster cache requested nonzero");
  require(preflight.cell_cache_requested_bytes ==
              preflight.cluster_cache_requested_bytes,
          "cell cache requested full envelope");
  require(preflight.budget_inputs.cell_cache_requested_bytes ==
              preflight.cell_cache_requested_bytes,
          "budget inputs carries cell requested");
}

void test_preflight_cell_zero_when_share_zero(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.query_file = (fs::path(fixture.temp_dir) / "missing_query.fbin").string();
  config.cell_cache_share_pct = 0.0f;

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  require(preflight.cluster_cache_requested_bytes > 0,
          "cluster cache requested nonzero");
  require(preflight.cell_cache_requested_bytes == 0,
          "cell cache not requested by default");
  require(preflight.budget_inputs.cell_cache_requested_bytes == 0,
          "budget inputs cell requested default zero");
}

void test_preflight_oversized_query_fails_before_reading(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.query_file = (fs::path(fixture.temp_dir) / "missing_query.fbin").string();
  config.mem_budget = 1.0f;
  config.n_queries = (size_t{1} << 30) / (config.dim * sizeof(float)) + 1;

  bool failed = false;
  try
  {
    (void)DiskRange::compute_search_mem_budget_preflight(config);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "mem_budget too small");
    require_contains(e.what(), "query=");
  }
  require(failed, "oversized query preflight must fail without reading query file");
}

void test_legacy_deep1m_budget_breaks_loudly(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.mem_budget = 0.24f;
  config.fast_path_enabled = true;
  config.fast_path_mode = "rbq";
  write_sparse_file(config.rabitq_1bit_codes_path, size_t{160} << 20);

  bool failed = false;
  try
  {
    (void)DiskRange::compute_search_mem_budget_preflight(config);
  }
  catch (const std::exception& e)
  {
    failed = true;
    require_contains(e.what(), "mem_budget too small");
    require_contains(e.what(), "rbq_1bit=");
    require_contains(e.what(), "budget=");
  }
  require(failed, "legacy deep1m mem_budget 0.24 must fail loudly");
}

void test_preflight_missing_data_hnsw_counts_zero(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.data_hnsw_enabled = true;
  config.data_hnsw_path =
      (fs::path(fixture.temp_dir) / "missing_data_hnsw.bin").string();

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  require(preflight.data_hnsw_bytes == 0, "missing data_hnsw counts zero");
}

void test_preflight_invalid_data_hnsw_counts_zero(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.data_hnsw_enabled = true;
  config.data_hnsw_path =
      (fs::path(fixture.temp_dir) / "invalid_data_hnsw.bin").string();
  {
    std::ofstream out(config.data_hnsw_path, std::ios::binary);
    out << "not a valid hnsw header";
  }

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  require(preflight.data_hnsw_bytes == 0, "invalid data_hnsw counts zero");
}

void test_preflight_sqg_rotated_trace_accounting(const std::string& gt_path)
{
  const Fixture fixture = make_fixture();
  ResolvedConfig config = deep1m_like_config(gt_path, fixture.temp_dir);
  config.metadata_path = fixture.metadata_path.string();
  config.fast_path_enabled = true;
  config.fast_path_mode = "rbq";
  config.K = 1;
  {
    std::ofstream sqg(config.sqg_path, std::ios::binary);
    sqg << std::string(1024, 's');
  }
  {
    std::ofstream rbq(config.rabitq_1bit_codes_path, std::ios::binary);
    rbq << std::string(2048, 'r');
  }

  const auto preflight = DiskRange::compute_search_mem_budget_preflight(config);
  require(preflight.sqg_bytes == 2048, "sqg counted twice for fast path + slow path");
  require(preflight.rbq_1bit_bytes == 2048, "rbq file counted");
  require(preflight.rotated_centroids_bytes ==
              config.cluster_num * (size_t{1} << symqg::ceil_log2(config.dim)) *
                  sizeof(float),
          "rotated centroids counted");
  require(preflight.trace_bytes >= config.n_queries * sizeof(FastPathTraceRecord),
          "fast path trace counted");
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: cluster_reader_mem_budget_test <deep1m_gt_json>\n";
    return 2;
  }

  try
  {
    test_happy_path_keeps_requested_depth_and_slots();
    test_compress_depth_before_slots();
    test_compress_slots_after_depth_floor();
    test_depth_floored_to_power_of_two();
    test_requested_depth_below_effective_floor_preserved();
    test_fail_below_fixed_reports_all_fields();
    test_fail_below_fixed_fast_path_reports_inactive_data_hnsw();
    test_fail_below_fixed_data_hnsw_reports_inactive_rbq();
    test_fail_io_budget_reports_io_fields();
    test_ring_entries_guard_reports_budget();
    test_fixed_resident_counts_hit_bookkeeping();
    test_legacy_brace_init_keeps_all_four_cache_fields_zero();
    test_config_reader_accepts_cell_cache_share_pct();
    test_checked_mul_saturates();
    test_reconcile_two_phase_allocates_cache();
    test_requested_preserves_through_disabled_mode();
    test_cell_cache_takes_half_when_share_50();
    test_cell_env_disable_redirects_full_pool_to_cluster();
    test_cluster_env_disable_with_share_redirects_to_cell();
    test_cluster_env_disable_with_zero_share_goes_to_iopool();
    test_tiny_cell_budget_returned_to_iopool();
    test_reconcile_rejects_invalid_cell_cache_share_pct();
    test_refined_cap_reserves_io_preferred();
    test_tight_budget_cache_yields_to_io();
    test_effective_capacity_reflects_shard_truncation();
    test_reconcile_cache_not_capped_by_file_size();
    test_reconcile_cache_capped_by_kClusterCacheMaxBytes();
    test_sub_shard_count_budget_yields_zero_effective();
    test_slot_id_maps_by_modulo_when_slots_are_compressed();
    test_preflight_counts_deep1m_gt_hit_bookkeeping(argv[1]);
    test_preflight_both_caches_full_envelope_when_cell_share_set(argv[1]);
    test_preflight_cell_zero_when_share_zero(argv[1]);
    test_preflight_oversized_query_fails_before_reading(argv[1]);
    test_legacy_deep1m_budget_breaks_loudly(argv[1]);
    test_preflight_missing_data_hnsw_counts_zero(argv[1]);
    test_preflight_invalid_data_hnsw_counts_zero(argv[1]);
    test_preflight_sqg_rotated_trace_accounting(argv[1]);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
