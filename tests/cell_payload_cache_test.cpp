#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../lib/ClusterIO.h"
#include "../lib/IoQueueDepth.h"

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

std::string make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "cell_payload_cache_test";
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
  const size_t bytes = vectors.size() * sizeof(float);
  const size_t padded_bytes = div_round_up(bytes, PAGE_SIZE) * PAGE_SIZE;
  if (padded_bytes > bytes)
  {
    out.seekp(static_cast<std::streamoff>(padded_bytes - 1));
    out.put('\0');
  }
  require(static_cast<bool>(out), "failed to write cluster file");
}

void write_test_metadata_file(
    const fs::path& path, size_t dim, size_t target_cell_vecs,
    const std::vector<float>& centroids,
    const std::vector<std::vector<size_t>>& assignment,
    const std::vector<std::vector<size_t>>& cell_sizes)
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
    size_t cluster_total = 0;
    for (const size_t count : cell_sizes[cid])
    {
      cluster_total += count;
      ++total_cells;
      total_vecs += static_cast<uint64_t>(count);
      cell_vec_offsets.push_back(total_vecs);
    }
    require(cluster_total == bucket_sizes[cid],
            "cell sizes must sum to bucket size");
  }
  cluster_cell_offsets[cluster_count] = total_cells;
  std::vector<float> cell_bounds(static_cast<size_t>(total_cells) * 2, 0.0f);

  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create metadata file");
  out.write(kMetadataMagic.data(),
            static_cast<std::streamsize>(kMetadataMagic.size()));
  const uint64_t schema_version = kMetadataSchemaVersion;
  const uint64_t metadata_pca_rank = 0;
  const uint64_t metadata_target_cell_vecs =
      static_cast<uint64_t>(target_cell_vecs);
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
    out.write(reinterpret_cast<const char*>(cluster_assignment.data()),
              static_cast<std::streamsize>(cluster_assignment.size() *
                                           sizeof(size_t)));
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
  out.write(reinterpret_cast<const char*>(cell_bounds.data()),
            static_cast<std::streamsize>(cell_bounds.size() * sizeof(float)));
  require(static_cast<bool>(out), "failed to write metadata file");
}

struct Fixture
{
  std::string temp_dir;
  fs::path cluster_path;
  fs::path metadata_path;
  size_t dim = 0;
  std::vector<float> vectors;
};

Fixture make_fixture()
{
  Fixture fixture;
  fixture.temp_dir = make_temp_dir();
  fixture.cluster_path = fs::path(fixture.temp_dir) / "cluster.bin";
  fixture.metadata_path = fs::path(fixture.temp_dir) / "metadata.bin";
  fixture.dim = 3;
  std::vector<std::vector<size_t>> cell_sizes = {
      {1, 2},
      {2, 3},
      {1, 1}};
  while (cell_sizes.size() < kClusterCacheShardCount + 1)
  {
    cell_sizes.push_back({1});
  }
  std::vector<std::vector<size_t>> assignment(cell_sizes.size());
  size_t row = 0;
  for (size_t cid = 0; cid < cell_sizes.size(); ++cid)
  {
    for (const size_t count : cell_sizes[cid])
    {
      for (size_t i = 0; i < count; ++i)
      {
        assignment[cid].push_back(row++);
        for (size_t col = 0; col < fixture.dim; ++col)
        {
          fixture.vectors.push_back(
              static_cast<float>(cid * 100 + assignment[cid].size() * 10 + col));
        }
      }
    }
  }
  std::vector<float> centroids(cell_sizes.size() * fixture.dim, 0.0f);
  write_cluster_file(fixture.cluster_path, fixture.vectors);
  write_test_metadata_file(fixture.metadata_path, fixture.dim, 2, centroids,
                           assignment, cell_sizes);
  return fixture;
}

size_t cell_budget_for_per_shard_pages(size_t pages)
{
  return static_cast<size_t>(
             static_cast<double>(kCellCacheShardCount * pages * PAGE_SIZE) /
             (1.0 - kCellCacheIndexOverheadRatio)) +
         PAGE_SIZE;
}

ClusterReader::MemBudgetInputs active_cell_cache_budget()
{
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = kClusterCacheShardCount * PAGE_SIZE * 4;
  budget.cell_cache_requested_bytes = cell_budget_for_per_shard_pages(16);
  budget.mem_budget_bytes = SIZE_MAX;
  return budget;
}

void test_disabled_cell_cache_passthrough()
{
  ClusterReader::CellPayloadCache cache(0, false);
  std::vector<char> source(PAGE_SIZE, 'a');
  for (int i = 0; i < 1000; ++i)
  {
    require(!cache.lookup(static_cast<uint64_t>(i)), "disabled lookup miss");
  }
  require(!cache.try_insert(1, source.data(), 0, source.size(), 1,
                            source.size()),
          "disabled insert no-op");
  const auto stats = cache.stats();
  require(!stats.enabled, "disabled stats enabled=N");
  require(stats.hits == 0 && stats.misses == 0,
          "disabled cache does not count pass-through misses");
}

void test_basic_hit_after_insert()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), false);
  std::vector<char> source(PAGE_SIZE, 'x');
  source[123] = 'z';
  require(cache.try_insert(42, source.data(), 17, source.size(), 7,
                           source.size()),
          "insert");
  auto ref = cache.lookup(42);
  require(static_cast<bool>(ref), "lookup hit");
  require(ref.payload_offset() == 17, "payload offset stored");
  require(ref.valid_bytes() == source.size(), "valid bytes stored");
  require(ref.vec_count() == 7, "vec count stored");
  require(std::memcmp(ref.buffer(), source.data(), source.size()) == 0,
          "buffer copied");
}

void test_stats_line_contains_17_keys()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), false);
  std::vector<char> source(PAGE_SIZE, 's');
  cache.try_insert(1, source.data(), 0, source.size(), 1, source.size());
  (void)cache.lookup(1);
  const std::string line = cache.stats_line();
  for (const char* key : {"enabled=", "env_disabled=", "budget=",
                          "per_shard_budget=", "current_resident_sum=",
                          "shard_peak_sum=", "entries=", "hits=",
                          "misses=", "hit_rate=", "evictions=",
                          "eviction_failed=", "alloc_failures=",
                          "oversize_rejects=", "duplicate_inserts=",
                          "shards_used=", "shard_max_resident="})
  {
    require(line.find(key) != std::string::npos,
            std::string("stats line missing ") + key);
  }
}

void test_lru_eviction_evicts_when_per_shard_quota_exceeded()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), false);
  const size_t target = 5;
  std::vector<uint64_t> ids;
  for (uint64_t cell_id = 0; ids.size() < 5; ++cell_id)
  {
    if (ClusterReader::CellPayloadCache::shard_index_for(cell_id) == target)
    {
      ids.push_back(cell_id);
    }
  }
  std::vector<char> source(PAGE_SIZE, 'e');
  for (const uint64_t id : ids)
  {
    require(cache.try_insert(id, source.data(), 0, source.size(), 1,
                             source.size()),
            "same-shard insert succeeds");
  }
  require(!cache.lookup(ids.front()), "oldest entry evicted");
  require(static_cast<bool>(cache.lookup(ids.back())), "newest entry remains");
  require(cache.stats().current_resident_sum <=
              cache.stats().per_shard_budget_bytes,
          "single-shard resident within quota");
}

void test_per_shard_isolation_no_cross_shard_strand()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(2), false);
  uint64_t shard_a_id = 0;
  uint64_t shard_a_id_2 = 0;
  uint64_t shard_b_id = 0;
  for (uint64_t cell_id = 0; cell_id < 100000; ++cell_id)
  {
    if (ClusterReader::CellPayloadCache::shard_index_for(cell_id) == 1)
    {
      if (shard_a_id == 0)
      {
        shard_a_id = cell_id;
      }
      else if (shard_a_id_2 == 0)
      {
        shard_a_id_2 = cell_id;
      }
    }
    if (ClusterReader::CellPayloadCache::shard_index_for(cell_id) == 2 &&
        shard_b_id == 0)
    {
      shard_b_id = cell_id;
    }
  }
  require(shard_a_id != 0 && shard_a_id_2 != 0 && shard_b_id != 0,
          "found ids in two shards");
  std::vector<char> one_page(PAGE_SIZE, 'a');
  require(cache.try_insert(shard_a_id, one_page.data(), 0, one_page.size(),
                           1, one_page.size()),
          "fill shard A first page");
  require(cache.try_insert(shard_a_id_2, one_page.data(), 0, one_page.size(),
                           1, one_page.size()),
          "fill shard A second page");
  require(cache.try_insert(shard_b_id, one_page.data(), 0, one_page.size(),
                           1, one_page.size()),
          "empty shard B succeeds despite shard A full");
}

void test_eviction_skips_in_use()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(1), false);
  const size_t target = 9;
  std::vector<uint64_t> ids;
  for (uint64_t cell_id = 0; ids.size() < 2; ++cell_id)
  {
    if (ClusterReader::CellPayloadCache::shard_index_for(cell_id) == target)
    {
      ids.push_back(cell_id);
    }
  }
  std::vector<char> one_page(PAGE_SIZE, 'u');
  require(cache.try_insert(ids[0], one_page.data(), 0, one_page.size(), 1,
                           one_page.size()),
          "insert held cell");
  auto held = cache.lookup(ids[0]);
  require(static_cast<bool>(held), "hold CacheRef");
  require(!cache.try_insert(ids[1], one_page.data(), 0, one_page.size(), 1,
                            one_page.size()),
          "held entry is not evicted");
  require(cache.stats().eviction_failed == 1, "eviction failed count");
  require(static_cast<bool>(cache.lookup(ids[0])), "held cell remains");
}

void test_oversize_cell_rejects_distinctly()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(2), false);
  std::vector<char> source(4 * PAGE_SIZE, 'o');
  require(!cache.try_insert(1, source.data(), 0, source.size(), 1,
                            source.size()),
          "oversize insert rejected");
  const auto stats = cache.stats();
  require(stats.oversize_rejects == 1, "oversize counter");
  require(stats.eviction_failed == 0, "not eviction failure");
  require(stats.alloc_failures == 0, "not alloc failure");
  require(stats.current_resident_sum == 0, "state unchanged");
}

void test_buffer_page_aligned()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), false);
  std::vector<char> source(PAGE_SIZE, 'p');
  require(cache.try_insert(7, source.data(), 0, source.size(), 1,
                           source.size()),
          "insert page-aligned cell");
  auto ref = cache.lookup(7);
  require(reinterpret_cast<uintptr_t>(ref.buffer()) % PAGE_SIZE == 0,
          "cache buffer page aligned");
}

void test_concurrent_same_cell_miss_idempotent()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), false);
  std::vector<char> source(PAGE_SIZE, 'c');
  std::vector<std::thread> threads;
  std::vector<bool> results(16, false);
  for (size_t i = 0; i < results.size(); ++i)
  {
    threads.emplace_back([&, i]() {
      results[i] = cache.try_insert(77, source.data(), 0, source.size(), 1,
                                    source.size());
    });
  }
  for (auto& thread : threads)
  {
    thread.join();
  }
  for (const bool ok : results)
  {
    require(ok, "duplicate concurrent insert succeeds");
  }
  require(cache.resident_entries_for_test() == 1,
          "concurrent inserts create one entry");
  require(cache.stats().duplicate_inserts == results.size() - 1,
          "duplicate insert count");
}

void test_shard_hash_avalanche_for_sequential_cell_id()
{
  std::set<size_t> shards;
  std::vector<size_t> counts(kCellCacheShardCount, 0);
  for (uint64_t cell_id = 0; cell_id < 100000; ++cell_id)
  {
    const size_t shard =
        ClusterReader::CellPayloadCache::shard_index_for(cell_id);
    shards.insert(shard);
    ++counts[shard];
  }
  require(shards.size() == kCellCacheShardCount,
          "sequential cells span all shards");
  require(*std::max_element(counts.begin(), counts.end()) <= 1562,
          "sequential cells not overly skewed");
}

void test_env_disable_overrides_budget()
{
  ClusterReader::CellPayloadCache cache(
      cell_budget_for_per_shard_pages(4), true);
  std::vector<char> source(PAGE_SIZE, 'd');
  require(!cache.try_insert(1, source.data(), 0, source.size(), 1,
                            source.size()),
          "env disabled insert no-op");
  require(!cache.lookup(1), "env disabled lookup miss");
  const auto stats = cache.stats();
  require(!stats.enabled, "env disabled stats enabled=N");
  require(stats.env_disabled, "env disabled stats flag");
  require(stats.hits == 0 && stats.misses == 0, "env disabled counters zero");
}

void test_cell_filter_miss_uses_legacy_io_not_full_cluster()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cell_cache_budget(), 1, 50.0f);
  reader.readMetaData();
  reader.enable_read_request_capture_for_test(true);

  std::vector<std::pair<size_t, ClusterReader::IOPlan>> plans;
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  plans.push_back({1, ClusterReader::IOPlan{first_cell}});

  size_t cell_callbacks = 0;
  reader.submit_and_drain(0, plans,
                          [&](size_t cluster_id, uint64_t cell_id,
                              const void*, size_t vec_count, size_t)
                          {
                            require(cluster_id == 1, "miss cluster id");
                            require(cell_id == first_cell, "miss cell id");
                            require(vec_count == 2, "miss vec count");
                            ++cell_callbacks;
                          });
  require(cell_callbacks == 1, "miss callback count");
  const auto& reads = reader.read_requests_for_test();
  require(reads.size() == 1, "cell miss read count");
  const size_t vec_start =
      static_cast<size_t>(reader.cell_vec_offsets[first_cell]);
  const size_t vec_end =
      static_cast<size_t>(reader.cell_vec_offsets[first_cell + 1]);
  const size_t abs_byte_start = vec_start * fixture.dim * sizeof(float);
  const size_t abs_byte_end = vec_end * fixture.dim * sizeof(float);
  const size_t expected_offset = (abs_byte_start / PAGE_SIZE) * PAGE_SIZE;
  const size_t expected_length =
      div_round_up(abs_byte_start - expected_offset +
                       (abs_byte_end - abs_byte_start),
                   PAGE_SIZE) *
      PAGE_SIZE;
  require(reads[0].offset == expected_offset,
          "active cell miss legacy offset");
  require(reads[0].length == expected_length,
          "active cell miss legacy length");
  require(reader.cluster_cache_for_test()->stats().misses == 0,
          "Path A does not consult cluster cache");
  require(reader.cell_cache_for_test()->stats().misses == 1,
          "Path A records cell cache miss");
}

void test_cell_filter_hit_byte_equivalent_to_io()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cell_cache_budget(), 1, 50.0f);
  reader.readMetaData();

  std::vector<std::pair<size_t, ClusterReader::IOPlan>> plans;
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  plans.push_back({1, ClusterReader::IOPlan{first_cell}});

  std::vector<float> miss_payload;
  reader.submit_and_drain(0, plans,
                          [&](size_t, uint64_t, const void* payload,
                              size_t vec_count, size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            miss_payload.assign(
                                data, data + vec_count * fixture.dim);
                          });
  std::vector<float> hit_payload;
  reader.submit_and_drain(0, plans,
                          [&](size_t, uint64_t, const void* payload,
                              size_t vec_count, size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            hit_payload.assign(
                                data, data + vec_count * fixture.dim);
                          });
  require(miss_payload == hit_payload, "cell hit matches miss payload");
  require(reader.cell_cache_for_test()->stats().hits == 1,
          "second read hits cell cache");
}

void test_path_a_malformed_plan_throws_before_cache_lookup()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cell_cache_budget(), 1, 50.0f);
  reader.readMetaData();
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  std::vector<std::pair<size_t, ClusterReader::IOPlan>> good_plan = {
      {1, ClusterReader::IOPlan{first_cell}}};
  reader.submit_and_drain(0, good_plan,
                          [](size_t, uint64_t, const void*, size_t, size_t) {});
  const size_t hits_before = reader.cell_cache_for_test()->stats().hits;

  bool failed = false;
  try
  {
    std::vector<std::pair<size_t, ClusterReader::IOPlan>> bad_plan = {
        {2, ClusterReader::IOPlan{first_cell}}};
    reader.submit_and_drain(0, bad_plan,
                            [](size_t, uint64_t, const void*, size_t, size_t) {});
  }
  catch (const std::exception& e)
  {
    failed = true;
    require(std::string(e.what()).find("does not belong to cluster") !=
                std::string::npos,
            "malformed plan throws membership error");
  }
  require(failed, "malformed plan must throw");
  require(reader.cell_cache_for_test()->stats().hits == hits_before,
          "validate happens before lookup");
}

void test_path_a_out_of_range_cell_id_throws_before_cache_lookup()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cell_cache_budget(), 1, 50.0f);
  reader.readMetaData();
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  std::vector<std::pair<size_t, ClusterReader::IOPlan>> good_plan = {
      {1, ClusterReader::IOPlan{first_cell}}};
  reader.submit_and_drain(0, good_plan,
                          [](size_t, uint64_t, const void*, size_t, size_t) {});
  const size_t hits_before = reader.cell_cache_for_test()->stats().hits;

  bool failed = false;
  try
  {
    std::vector<std::pair<size_t, ClusterReader::IOPlan>> bad_plan = {
        {1, ClusterReader::IOPlan{std::numeric_limits<uint64_t>::max()}}};
    reader.submit_and_drain(0, bad_plan,
                            [](size_t, uint64_t, const void*, size_t, size_t) {});
  }
  catch (const std::exception& e)
  {
    failed = true;
    require(std::string(e.what()).find("cell id out of range") !=
                std::string::npos,
            "out-of-range plan throws range error");
  }
  require(failed, "out-of-range plan must throw");
  require(reader.cell_cache_for_test()->stats().hits == hits_before,
          "out-of-range validate happens before lookup");
}

void test_cluster_wide_path_with_cell_share_set_does_not_touch_cell_cache()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cell_cache_budget(), 1, 50.0f);
  reader.readMetaData();

  size_t callbacks = 0;
  reader.submit_and_drain(0, std::vector<size_t>{1},
                          [&](size_t cluster_id, const void*, size_t, size_t)
                          {
                            require(cluster_id == 1, "cluster-wide id");
                            ++callbacks;
                          });
  require(callbacks == 1, "cluster-wide callback");
  const auto stats = reader.cell_cache_for_test()->stats();
  require(stats.hits == 0 && stats.misses == 0,
          "cluster-wide path does not touch cell cache");
}
}  // namespace

int main()
{
  try
  {
    test_disabled_cell_cache_passthrough();
    test_basic_hit_after_insert();
    test_stats_line_contains_17_keys();
    test_lru_eviction_evicts_when_per_shard_quota_exceeded();
    test_per_shard_isolation_no_cross_shard_strand();
    test_eviction_skips_in_use();
    test_oversize_cell_rejects_distinctly();
    test_buffer_page_aligned();
    test_concurrent_same_cell_miss_idempotent();
    test_shard_hash_avalanche_for_sequential_cell_id();
    test_env_disable_overrides_budget();
    test_cell_filter_miss_uses_legacy_io_not_full_cluster();
    test_cell_filter_hit_byte_equivalent_to_io();
    test_path_a_malformed_plan_throws_before_cache_lookup();
    test_path_a_out_of_range_cell_id_throws_before_cache_lookup();
    test_cluster_wide_path_with_cell_share_set_does_not_touch_cell_cache();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
