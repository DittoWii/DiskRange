#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  fs::path base = fs::temp_directory_path() / "cluster_payload_cache_test";
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

ClusterReader::MemBudgetInputs active_cache_budget()
{
  ClusterReader::MemBudgetInputs budget;
  budget.cluster_cache_requested_bytes = kClusterCacheShardCount * PAGE_SIZE * 4;
  budget.mem_budget_bytes = SIZE_MAX;
  return budget;
}

void test_disabled_cache_passthrough()
{
  ClusterReader::ClusterPayloadCache cache(0, PAGE_SIZE, false);
  std::vector<char> source(PAGE_SIZE, 'a');
  for (int i = 0; i < 1000; ++i)
  {
    require(!cache.lookup(static_cast<uint32_t>(i)), "disabled lookup miss");
  }
  require(!cache.try_insert(1, source.data(), 0, source.size()),
          "disabled insert no-op");
  require(cache.stats().hits == 0, "disabled hits");
  require(cache.stats().misses == 1000, "disabled misses tracked");
}

void test_basic_hit_after_insert()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  std::vector<char> source(PAGE_SIZE, 'x');
  source[123] = 'z';
  require(cache.try_insert(42, source.data(), 17, source.size()), "insert");
  auto ref = cache.lookup(42);
  require(static_cast<bool>(ref), "lookup hit");
  require(ref.payload_offset() == 17, "payload offset stored");
  require(ref.valid_bytes() == source.size(), "valid bytes stored");
  require(std::memcmp(ref.buffer(), source.data(), source.size()) == 0,
          "buffer copied");
}

void test_lru_eviction_order()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  const size_t target = 3;
  std::vector<uint32_t> ids;
  for (uint32_t cid = 0; ids.size() < 2; ++cid)
  {
    if (ClusterReader::ClusterPayloadCache::shard_index_for(cid) == target)
    {
      ids.push_back(cid);
    }
  }
  std::vector<char> source(PAGE_SIZE, 'q');
  require(cache.try_insert(ids[0], source.data(), 0, source.size()), "insert 0");
  require(cache.try_insert(ids[1], source.data(), 0, source.size()), "insert 1");
  require(!cache.lookup(ids[0]), "oldest evicted");
  require(static_cast<bool>(cache.lookup(ids[1])), "newest present");
}

void test_per_shard_cap_strict()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount * 2,
                                           PAGE_SIZE, false);
  const size_t target = 17;
  std::vector<uint32_t> ids;
  for (uint32_t cid = 0; ids.size() < 100; ++cid)
  {
    if (ClusterReader::ClusterPayloadCache::shard_index_for(cid) == target)
    {
      ids.push_back(cid);
    }
  }
  std::vector<char> source(PAGE_SIZE, 'p');
  for (const uint32_t id : ids)
  {
    require(cache.try_insert(id, source.data(), 0, source.size()),
            "same shard insert succeeds");
    require(cache.resident_entries_for_test() <= 2,
            "same shard resident entries respect cap");
  }
  require(cache.resident_entries_for_test() == 2,
          "same shard cap leaves two entries");
  require(!cache.lookup(ids.front()), "old same-shard entry evicted");
  require(static_cast<bool>(cache.lookup(ids[ids.size() - 1])),
          "newest same-shard entry present");
  require(static_cast<bool>(cache.lookup(ids[ids.size() - 2])),
          "second newest same-shard entry present");
}

void test_eviction_skips_in_use()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  const size_t target = 11;
  std::vector<uint32_t> ids;
  for (uint32_t cid = 0; ids.size() < 2; ++cid)
  {
    if (ClusterReader::ClusterPayloadCache::shard_index_for(cid) == target)
    {
      ids.push_back(cid);
    }
  }
  std::vector<char> source(PAGE_SIZE, 'u');
  require(cache.try_insert(ids[0], source.data(), 0, source.size()),
          "insert held entry");
  auto held_ref = cache.lookup(ids[0]);
  require(static_cast<bool>(held_ref), "held entry lookup");
  require(!cache.try_insert(ids[1], source.data(), 0, source.size()),
          "all in-use shard insert fails");
  require(cache.stats().eviction_failed == 1, "eviction failed count");
  require(static_cast<bool>(cache.lookup(ids[0])), "held entry not evicted");
}

void test_concurrent_same_cluster_miss_idempotent()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  std::vector<char> source(PAGE_SIZE, 'c');
  std::vector<std::thread> threads;
  std::vector<bool> results(16, false);
  threads.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i)
  {
    threads.emplace_back([&, i]() {
      results[i] = cache.try_insert(77, source.data(), 0, source.size());
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
  require(static_cast<bool>(cache.lookup(77)), "concurrent insert present");
  require(cache.resident_entries_for_test() == 1,
          "concurrent insert creates one resident entry");
  require(cache.stats().duplicate_inserts == results.size() - 1,
          "duplicate insert count");
}

void test_shard_hash_resilient_to_low_bits()
{
  std::set<size_t> shards;
  std::vector<size_t> counts(kClusterCacheShardCount, 0);
  for (uint32_t k = 0; k < 10000; ++k)
  {
    const size_t shard =
        ClusterReader::ClusterPayloadCache::shard_index_for(k * 64);
    shards.insert(shard);
    ++counts[shard];
  }
  require(shards.size() >= kClusterCacheShardCount / 2,
          "low-bit clustered ids span shards");
  require(*std::max_element(counts.begin(), counts.end()) <= 625,
          "low-bit clustered ids not overly skewed");
}

void test_env_disable_overrides_budget()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           true);
  std::vector<char> source(PAGE_SIZE, 'd');
  require(!cache.try_insert(1, source.data(), 0, source.size()),
          "env disabled insert no-op");
  require(!cache.lookup(1), "env disabled lookup miss");
  const auto stats = cache.stats();
  require(!stats.enabled, "env disabled stats enabled");
  require(stats.env_disabled, "env disabled stats flag");
  require(stats.hits == 0 && stats.misses == 0, "env disabled counters zero");
}

void test_sub_shard_count_budget_yields_zero_effective()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount - 1,
                                           PAGE_SIZE, false);
  require(!cache.active(), "sub-shard budget inactive");
  require(cache.stats().entries_max == 0, "sub-shard entries zero");
}

void test_cell_filter_hit_for_cluster_not_zero()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();

  std::vector<size_t> prime = {1};
  size_t cluster_callbacks = 0;
  reader.submit_and_drain(0, prime,
                          [&](size_t cluster_id, const void* payload,
                              size_t bucket_size, size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            require(cluster_id == 1, "primed cluster id");
                            require(bucket_size == 5, "primed bucket size");
                            require(data[0] == fixture.vectors[3 * fixture.dim],
                                    "primed data");
                            ++cluster_callbacks;
                          });
  require(cluster_callbacks == 1, "prime callback count");

  std::vector<std::pair<size_t, ClusterReader::IOPlan>> plans;
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  const uint64_t second_cell = first_cell + 1;
  plans.push_back({1, ClusterReader::IOPlan{second_cell}});

  size_t cell_callbacks = 0;
  reader.submit_and_drain(0, plans,
                          [&](size_t cluster_id, uint64_t cell_id,
                              const void* payload, size_t vec_count, size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            require(cluster_id == 1, "cell cluster id");
                            require(cell_id == second_cell, "cell id");
                            require(vec_count == 3, "vec count");
                            const size_t expected_global_vec = 3 + 2;
                            require(data[0] ==
                                        fixture.vectors[expected_global_vec *
                                                        fixture.dim],
                                    "cluster-local cell slice");
                            ++cell_callbacks;
                          });
  require(cell_callbacks == 1, "cell callback count");
}

void test_raii_decrement_on_exception()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();
  reader.submit_and_drain(0, std::vector<size_t>{1},
                          [](size_t, const void*, size_t, size_t) {});
  bool failed = false;
  try
  {
    reader.submit_and_drain(0, std::vector<size_t>{1},
                            [](size_t, const void*, size_t, size_t) {
                              throw std::runtime_error("callback failure");
                            });
  }
  catch (const std::runtime_error& e)
  {
    failed = true;
    require(std::string(e.what()) == "callback failure",
            "callback exception preserved");
  }
  require(failed, "cache hit callback should throw");

  const size_t target_shard =
      ClusterReader::ClusterPayloadCache::shard_index_for(1);
  uint32_t replacement = 0;
  for (uint32_t cid = 3; cid < 100000; ++cid)
  {
    if (ClusterReader::ClusterPayloadCache::shard_index_for(cid) ==
        target_shard)
    {
      replacement = cid;
      break;
    }
  }
  require(replacement != 0, "replacement same shard");
  std::vector<char> source(reader.io_buffer_size(), 'r');
  require(reader.cluster_cache_for_test()->try_insert(
              replacement, source.data(), 0, PAGE_SIZE),
          "ref released after throwing callback");
}

void test_payload_offset_stored_correctly()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();
  reader.submit_and_drain(0, std::vector<size_t>{1},
                          [](size_t, const void*, size_t, size_t) {});
  auto ref = reader.cluster_cache_for_test()->lookup(1);
  require(static_cast<bool>(ref), "payload offset lookup");
  require(ref.payload_offset() == reader.file_pos[1] % PAGE_SIZE,
          "payload offset equals page offset");
  const size_t expected_valid =
      div_round_up(ref.payload_offset() + 5 * fixture.dim * sizeof(float),
                   PAGE_SIZE) *
      PAGE_SIZE;
  require(ref.valid_bytes() == expected_valid,
          "valid bytes equals aligned read size");
}

void test_cell_filter_miss_inserts_cluster_payload()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
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
  require(cell_callbacks == 1, "miss cell callback count");
  const auto& reads = reader.read_requests_for_test();
  require(reads.size() == 1, "active cell miss read count");
  const size_t expected_offset =
      reader.file_pos[1] - (reader.file_pos[1] % PAGE_SIZE);
  const size_t expected_length =
      div_round_up(reader.file_pos[1] % PAGE_SIZE +
                       reader.bucket_sizes[1] * fixture.dim * sizeof(float),
                   PAGE_SIZE) *
      PAGE_SIZE;
  require(reads[0].offset == expected_offset,
          "active cell miss full-cluster offset");
  require(reads[0].length == expected_length,
          "active cell miss full-cluster length");
  auto cached = reader.cluster_cache_for_test()->lookup(1);
  require(static_cast<bool>(cached), "cell miss inserted cluster");
  require(cached.valid_bytes() ==
              div_round_up(cached.payload_offset() + 5 * fixture.dim *
                                                   sizeof(float),
                           PAGE_SIZE) *
                  PAGE_SIZE,
          "cell miss inserted full-cluster aligned read");
  const size_t hits_before_cluster_wide =
      reader.cluster_cache_for_test()->stats().hits;

  std::vector<float> hit_payload;
  reader.submit_and_drain(0, std::vector<size_t>{1},
                          [&](size_t, const void* payload, size_t bucket_size,
                              size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            hit_payload.assign(data,
                                               data + bucket_size * fixture.dim);
                          });
  require(hit_payload.size() == 5 * fixture.dim,
          "cell miss populated cluster cache");
  require(reader.cluster_cache_for_test()->stats().hits ==
              hits_before_cluster_wide + 1,
          "cluster-wide hit after cell miss");
}

void test_last_cluster_near_eof_no_overread()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();
  const size_t last_cluster = reader.cluster_num - 1;
  reader.submit_and_drain(0, std::vector<size_t>{last_cluster},
                          [](size_t, const void*, size_t, size_t) {});
  auto ref =
      reader.cluster_cache_for_test()->lookup(static_cast<uint32_t>(last_cluster));
  require(static_cast<bool>(ref), "last cluster cached");
  const size_t expected_valid =
      div_round_up(ref.payload_offset() +
                       reader.bucket_sizes[last_cluster] * fixture.dim *
                           sizeof(float),
                   PAGE_SIZE) *
      PAGE_SIZE;
  require(ref.valid_bytes() == expected_valid,
          "last cluster valid bytes equals aligned read size");
  require(ref.valid_bytes() < reader.io_buffer_size(),
          "last cluster does not read full io buffer");
}

void test_cell_filter_miss_keeps_legacy_when_cache_inactive()
{
  const Fixture fixture = make_fixture();
  ClusterReader::MemBudgetInputs budget;
  budget.mem_budget_bytes = SIZE_MAX;
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4, budget, 1);
  reader.readMetaData();
  reader.enable_read_request_capture_for_test(true);

  std::vector<std::pair<size_t, ClusterReader::IOPlan>> plans;
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  plans.push_back({1, ClusterReader::IOPlan{first_cell}});

  size_t cell_callbacks = 0;
  reader.submit_and_drain(0, plans,
                          [&](size_t, uint64_t, const void*, size_t, size_t)
                          {
                            ++cell_callbacks;
                          });
  require(cell_callbacks == 1, "inactive cell callback count");
  const auto& reads = reader.read_requests_for_test();
  require(reads.size() == 1, "inactive cell miss read count");
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
          "inactive cell miss legacy offset");
  require(reads[0].length == expected_length,
          "inactive cell miss legacy length");
  const auto stats = reader.cluster_cache_for_test()->stats();
  require(!stats.enabled, "inactive cache disabled");
  require(stats.misses == 1, "inactive lookup miss tracked");
  require(stats.peak_resident == 0, "inactive miss does not insert");
}

void test_cell_filter_try_insert_failure_still_uses_source_buffer()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();
  const size_t target_shard =
      ClusterReader::ClusterPayloadCache::shard_index_for(1);
  uint32_t held_id = 0;
  for (uint32_t cid = 3; cid < 100000; ++cid)
  {
    if (ClusterReader::ClusterPayloadCache::shard_index_for(cid) ==
        target_shard)
    {
      held_id = cid;
      break;
    }
  }
  require(held_id != 0, "found same-shard held id");
  std::vector<char> source(reader.io_buffer_size(), 'h');
  require(reader.cluster_cache_for_test()->try_insert(
              held_id, source.data(), 0, PAGE_SIZE),
          "prime held entry");
  auto held_ref = reader.cluster_cache_for_test()->lookup(held_id);
  require(static_cast<bool>(held_ref), "hold ref");

  std::vector<std::pair<size_t, ClusterReader::IOPlan>> plans;
  const uint64_t first_cell = reader.cluster_cell_offsets[1];
  plans.push_back({1, ClusterReader::IOPlan{first_cell}});
  size_t callbacks = 0;
  reader.submit_and_drain(0, plans,
                          [&](size_t cluster_id, uint64_t cell_id,
                              const void* payload, size_t vec_count, size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            require(cluster_id == 1, "failed insert cluster id");
                            require(cell_id == first_cell, "failed insert cell id");
                            require(vec_count == 2, "failed insert vec count");
                            require(data[0] == fixture.vectors[3 * fixture.dim],
                                    "failed insert source buffer data");
                            ++callbacks;
                          });
  require(callbacks == 1, "failed insert callback count");
  require(reader.cluster_cache_for_test()->stats().eviction_failed == 1,
          "failed insert increments eviction_failed");
}

void test_cluster_wide_hit_byte_equivalent()
{
  const Fixture fixture = make_fixture();
  ClusterReader reader(fixture.cluster_path.string(),
                       fixture.metadata_path.string(), 4,
                       active_cache_budget(), 1);
  reader.readMetaData();

  std::vector<float> miss_payload;
  reader.submit_and_drain(0, std::vector<size_t>{2},
                          [&](size_t, const void* payload, size_t bucket_size,
                              size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            miss_payload.assign(data,
                                                data + bucket_size * fixture.dim);
                          });
  std::vector<float> hit_payload;
  reader.submit_and_drain(0, std::vector<size_t>{2},
                          [&](size_t, const void* payload, size_t bucket_size,
                              size_t)
                          {
                            const float* data =
                                static_cast<const float*>(payload);
                            hit_payload.assign(data,
                                               data + bucket_size * fixture.dim);
                          });
  require(miss_payload == hit_payload, "cache hit matches miss payload");
}

void test_stats_line_contains_required_keys()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  std::vector<char> source(PAGE_SIZE, 's');
  cache.try_insert(1, source.data(), 0, source.size());
  (void)cache.lookup(1);
  const std::string line = cache.stats_line();
  for (const char* key : {"enabled=", "env_disabled=", "capacity=",
                          "entries_max=", "per_shard_capacity=",
                          "hits=", "misses=", "hit_rate=",
                          "evictions=", "eviction_failed=",
                          "duplicate_inserts=", "peak_resident=",
                          "shards_used=", "shard_max_resident="})
  {
    require(line.find(key) != std::string::npos,
            std::string("stats line missing ") + key);
  }
}

void test_buffer_page_aligned()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount, PAGE_SIZE,
                                           false);
  std::vector<char> source(PAGE_SIZE, 'a');
  cache.try_insert(7, source.data(), 0, source.size());
  auto ref = cache.lookup(7);
  require(reinterpret_cast<uintptr_t>(ref.buffer()) % PAGE_SIZE == 0,
          "cache buffer page aligned");
}

void test_effective_capacity_reflects_shard_truncation()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount * 15 + 40,
                                           PAGE_SIZE, false);
  const auto stats = cache.stats();
  require(stats.entries_max == kClusterCacheShardCount * 15,
          "effective capacity shard truncation");
  require(stats.per_shard_capacity == 15, "per-shard capacity");
}

void test_total_resident_never_exceeds_budget()
{
  ClusterReader::ClusterPayloadCache cache(kClusterCacheShardCount * 3,
                                           PAGE_SIZE, false);
  std::vector<char> source(PAGE_SIZE, 'b');
  for (uint32_t id = 0; id < 10000; ++id)
  {
    require(cache.try_insert(id, source.data(), 0, source.size()),
            "budget bound insert");
    require(cache.resident_entries_for_test() * PAGE_SIZE <=
                cache.stats().capacity_bytes,
            "resident bytes bounded by effective capacity");
  }
}
}  // namespace

int main()
{
  try
  {
    test_disabled_cache_passthrough();
    test_basic_hit_after_insert();
    test_lru_eviction_order();
    test_per_shard_cap_strict();
    test_eviction_skips_in_use();
    test_concurrent_same_cluster_miss_idempotent();
    test_shard_hash_resilient_to_low_bits();
    test_env_disable_overrides_budget();
    test_sub_shard_count_budget_yields_zero_effective();
    test_cell_filter_hit_for_cluster_not_zero();
    test_raii_decrement_on_exception();
    test_payload_offset_stored_correctly();
    test_cell_filter_miss_inserts_cluster_payload();
    test_last_cluster_near_eof_no_overread();
    test_cell_filter_miss_keeps_legacy_when_cache_inactive();
    test_cell_filter_try_insert_failure_still_uses_source_buffer();
    test_cluster_wide_hit_byte_equivalent();
    test_stats_line_contains_required_keys();
    test_buffer_page_aligned();
    test_effective_capacity_reflects_shard_truncation();
    test_total_resident_never_exceeds_budget();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
