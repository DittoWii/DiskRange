#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <immintrin.h>

#include <Eigen/Dense>

#include "../utils/dist_func.h"
#include "../utils/vector_cast.h"
#include "BuildPhaseTimer.h"
#include "ClusterIO.h"
#include "DataHNSWOracle.h"
#include "FastPathEmptinessOracle.h"
#include "Kmeans.h"
#include "RBQCodeStorage.h"
#include "ResolvedConfig.h"
#include "RunResult.h"
#include "SQGCentroidIndex.h"

struct DiskRange
{
  // Slow-path batch size for the q.centroid GEMM. After the survivor pre-pass
  // (Pass 1) the slow path only iterates fast-path survivors, so this also sets
  // the parallel work-unit count: n_survivors / kQdotBatchSize must stay >=
  // #threads or the I/O-bound slow path loses concurrency. 8 balances GEMM
  // amortization (centroid matrix reused across 8 query rows) against having
  // enough batches for ~24 threads. Tuned on deep10m (~265 survivors -> ~33
  // batches); +7.9% wall QPS vs the old 32 in a cache-fair A/B.
  static constexpr size_t kQdotBatchSize = 8;
  using QdotMatrixMap =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>;
  using QdotMatrixConstMap =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>;

  struct MemBudgetPreflight
  {
    size_t mem_budget_bytes = 0;
    size_t metadata_upper_bound_bytes = 0;
    size_t rbq_1bit_bytes = 0;
    size_t data_hnsw_bytes = 0;
    size_t query_bytes = 0;
    size_t per_thread_bytes = 0;
    size_t sqg_bytes = 0;
    size_t rotated_centroids_bytes = 0;
    size_t trace_bytes = 0;
    size_t hit_bookkeeping_bytes = 0;
    size_t cluster_cache_requested_bytes = 0;
    size_t cluster_cache_resident_bytes = 0;
    size_t cell_cache_requested_bytes = 0;
    size_t cell_cache_resident_bytes = 0;
    size_t fixed_pre_flight_bytes = 0;
    ClusterReader::MemBudgetInputs budget_inputs;
  };

  static std::string format_mib(size_t bytes)
  {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << static_cast<double>(bytes) / static_cast<double>(size_t{1} << 20);
    return out.str();
  }

  static size_t padded_dim_for_mem_budget(size_t dim)
  {
    size_t padded = 1;
    while (padded < dim)
    {
      padded <<= 1;
    }
    return padded;
  }

  static std::string preflight_error_message(const MemBudgetPreflight& p)
  {
    std::ostringstream out;
    out << "mem_budget too small (pre-flight): metadata="
        << format_mib(p.metadata_upper_bound_bytes)
        << ", rbq_1bit=" << format_mib(p.rbq_1bit_bytes)
        << ", data_hnsw=" << format_mib(p.data_hnsw_bytes)
        << ", query=" << format_mib(p.query_bytes)
        << ", per_thread=" << format_mib(p.per_thread_bytes)
        << ", sqg=" << format_mib(p.sqg_bytes)
        << ", rotated_centroids=" << format_mib(p.rotated_centroids_bytes)
        << ", trace=" << format_mib(p.trace_bytes)
        << ", hit_bookkeeping=" << format_mib(p.hit_bookkeeping_bytes)
        << ", cluster_cache=" << format_mib(p.cluster_cache_resident_bytes)
        << ", cell_cache=" << format_mib(p.cell_cache_resident_bytes)
        << ", cluster_cache_requested="
        << format_mib(p.cluster_cache_requested_bytes)
        << ", cell_cache_requested="
        << format_mib(p.cell_cache_requested_bytes)
        << ", slack=" << format_mib(kReservedSlack)
        << ", budget=" << format_mib(p.mem_budget_bytes);
    return out.str();
  }

  static MemBudgetPreflight compute_search_mem_budget_preflight(
      const ResolvedConfig& config)
  {
    if (config.mem_budget <= 0.0f)
    {
      throw std::runtime_error("mem_budget must be positive in search config");
    }

    MemBudgetPreflight p;
    p.mem_budget_bytes =
        static_cast<size_t>(config.mem_budget * double(1ULL << 30));
    p.metadata_upper_bound_bytes =
        static_cast<size_t>(std::filesystem::file_size(config.metadata_path));
    if (config.fast_path_enabled && config.fast_path_mode == "rbq")
    {
      p.rbq_1bit_bytes =
          static_cast<size_t>(std::filesystem::file_size(
              config.rabitq_1bit_codes_path));
    }
    if (config.data_hnsw_enabled &&
        std::filesystem::exists(config.data_hnsw_path))
    {
      std::string reason;
      if (validate_data_hnsw_header(config.data_hnsw_path, config.dim, reason))
      {
        p.data_hnsw_bytes =
            static_cast<size_t>(std::filesystem::file_size(
                config.data_hnsw_path)) *
                3 / 2 +
            kHnswOverheadFixed;
      }
    }
    if (range_search_config::is_byte_dtype(config.vec_dtype))
    {
      p.query_bytes = config.n_queries * config.dim *
                          (sizeof(uint8_t) + sizeof(float)) +
                      config.n_queries * sizeof(int32_t);
    }
    else
    {
      p.query_bytes = config.n_queries * config.dim * sizeof(float);
    }
    p.per_thread_bytes =
        static_cast<size_t>(omp_get_max_threads()) *
        (2 * kQdotBatchSize * config.cluster_num * sizeof(float) +
         kQdotBatchSize * config.dim * sizeof(float) +
         kPerThreadConstOverhead);
    if (std::filesystem::exists(config.sqg_path))
    {
      const size_t sqg_file_bytes =
          static_cast<size_t>(std::filesystem::file_size(config.sqg_path));
      if (config.K < config.cluster_num)
      {
        p.sqg_bytes += sqg_file_bytes;
      }
      if (config.fast_path_enabled)
      {
        p.sqg_bytes += sqg_file_bytes;
      }
    }
    if (config.fast_path_enabled && config.fast_path_mode == "rbq")
    {
      p.rotated_centroids_bytes =
          config.cluster_num * padded_dim_for_mem_budget(config.dim) *
          sizeof(float);
    }
    p.trace_bytes =
        config.n_queries *
        ((config.fast_path_enabled ? sizeof(FastPathTraceRecord) : 0) +
         (p.data_hnsw_bytes != 0 ? sizeof(DataHNSWTraceRecord) : 0) +
         (kDiskRangeSearchPhaseTimingEnabled ? sizeof(SlowPathTraceRecord)
                                            : 0));
    p.hit_bookkeeping_bytes =
        config.n_queries * 64 * sizeof(HitRef) +
        config.total_in_range *
            (sizeof(HitRef) + sizeof(uint8_t) +
             sizeof(std::tuple<size_t, uint32_t, size_t>) +
             kHitLookupOverhead);
    p.fixed_pre_flight_bytes =
        p.metadata_upper_bound_bytes + p.rbq_1bit_bytes + p.data_hnsw_bytes +
        p.query_bytes + p.per_thread_bytes + p.sqg_bytes +
        p.rotated_centroids_bytes + p.trace_bytes + p.hit_bookkeeping_bytes +
        kReservedSlack;

    if (p.mem_budget_bytes <= p.fixed_pre_flight_bytes)
    {
      throw std::runtime_error(preflight_error_message(p));
    }

    size_t depth = bit_floor_compat(config.io_queue_depth);
    if (depth == 0)
    {
      depth = 1;
    }
    const size_t effective_min_depth =
        std::min(depth, kMinEffectiveIoQueueDepth);
    const size_t io_uring_min_lower_bound =
        checked_mul(effective_min_depth, static_cast<size_t>(PAGE_SIZE));
    const size_t remaining_after_fixed =
        p.mem_budget_bytes - p.fixed_pre_flight_bytes;
    if (remaining_after_fixed > io_uring_min_lower_bound)
    {
      const size_t envelope =
          remaining_after_fixed - io_uring_min_lower_bound;
      p.cluster_cache_requested_bytes =
          std::min(envelope, kClusterCacheMaxBytes);
      if (config.cell_cache_share_pct > 0.0f)
      {
        p.cell_cache_requested_bytes =
            std::min(envelope, kCellCacheMaxBytes);
      }
    }

    p.budget_inputs.mem_budget_bytes = p.mem_budget_bytes;
    p.budget_inputs.rbq_1bit_resident_bytes = p.rbq_1bit_bytes;
    p.budget_inputs.data_hnsw_resident_bytes = p.data_hnsw_bytes;
    p.budget_inputs.query_resident_bytes = p.query_bytes;
    p.budget_inputs.per_thread_resident_bytes = p.per_thread_bytes;
    p.budget_inputs.sqg_resident_bytes = p.sqg_bytes;
    p.budget_inputs.rotated_centroids_resident_bytes =
        p.rotated_centroids_bytes;
    p.budget_inputs.trace_resident_bytes = p.trace_bytes;
    p.budget_inputs.hit_bookkeeping_resident_bytes =
        p.hit_bookkeeping_bytes;
    p.budget_inputs.cluster_cache_requested_bytes =
        p.cluster_cache_requested_bytes;
    p.budget_inputs.cluster_cache_resident_bytes =
        p.cluster_cache_resident_bytes;
    p.budget_inputs.cell_cache_requested_bytes =
        p.cell_cache_requested_bytes;
    p.budget_inputs.cell_cache_resident_bytes =
        p.cell_cache_resident_bytes;
    return p;
  }

  void build(const ResolvedConfig& config)
  {
    const auto start = std::chrono::steady_clock::now();
    if (config.slab_count > 0 && config.slab_count >= config.cluster_num)
    {
      throw std::runtime_error("slab_count " +
                               std::to_string(config.slab_count) +
                               " must be < cluster_num " +
                               std::to_string(config.cluster_num));
    }
    const std::filesystem::path cluster_path(config.cluster_path);
    const std::filesystem::path parent = cluster_path.parent_path();
    if (!parent.empty())
    {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec)
      {
        throw std::runtime_error("failed to create index directory " +
                                 parent.string() + ": " + ec.message());
      }
    }

    {
      BuildPhaseTimer _t("one_level_kmeans");
      one_level_kmeans(config);
    }
    if (config.fast_path_enabled)
    {
      BuildPhaseTimer _t("build_fast_path_artifacts");
      build_fast_path_artifacts(config);
    }
    if (config.data_hnsw_enabled)
    {
      BuildPhaseTimer _t("build_and_save_data_hnsw");
      build_and_save_data_hnsw(config);
    }

    const auto end = std::chrono::steady_clock::now();
    build_time_ = std::chrono::duration<double>(end - start).count();
    msginfo_s("build time = {} seconds", build_time_);
  }

  RunResult search(const ResolvedConfig& config)
  {
    // const auto start = std::chrono::steady_clock::now();
    require_file_size(config.cluster_path, "cluster file");
    const uintmax_t metadata_file_size =
        require_file_size(config.metadata_path, "metadata file");
    MetadataHeader metadata_header = read_metadata_header(config);
    validate_metadata_header(config, metadata_header);
    validate_query_header(config, metadata_header.vec_dtype);

    if (config.n_queries == 0)
    {
      RunResult result;
      result.build_time = build_time_;
      result.search_time = 0;
      return result;
    }

    const MemBudgetPreflight mem_budget_preflight =
        compute_search_mem_budget_preflight(config);
    const QueryData query_data = read_query_data(config, metadata_header.vec_dtype);
    const std::vector<float>& queries = query_data.as_float;
    const std::vector<size_t> metadata_bucket_sizes =
        read_metadata_bucket_sizes(config, metadata_header);
    metadata_header.slab_count =
        read_metadata_slab_count(config, metadata_header, metadata_bucket_sizes);
    validate_metadata_layout(config, metadata_header, metadata_bucket_sizes,
                             metadata_file_size);

    ClusterReader cluster_reader(config.cluster_path, config.metadata_path,
                                 config.io_queue_depth,
                                 mem_budget_preflight.budget_inputs,
                                 omp_get_max_threads(),
                                 config.cell_cache_share_pct);
    cluster_reader.readMetaData();
    if (!cluster_reader.fmeta)
    {
      throw std::runtime_error("failed to read metadata file: " +
                               config.metadata_path);
    }
    if (cluster_reader.d != config.dim)
    {
      throw std::runtime_error("metadata dim mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(config.dim) + ", actual " +
                               std::to_string(cluster_reader.d));
    }
    if (cluster_reader.cluster_num != config.cluster_num)
    {
      throw std::runtime_error(
          "metadata cluster count mismatch for " + config.metadata_path +
          ": expected " + std::to_string(config.cluster_num) + ", actual " +
          std::to_string(cluster_reader.cluster_num));
    }

    const size_t metadata_points =
        std::accumulate(cluster_reader.bucket_sizes.begin(),
                        cluster_reader.bucket_sizes.end(), size_t{0});
    if (metadata_points != config.n_base)
    {
      throw std::runtime_error("metadata base count mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(config.n_base) + ", actual " +
                               std::to_string(metadata_points));
    }
    // Per-query hit list with (cluster_id, cell_id, row) refs. Verified by
    // exact L2 recomputation after timer stop (post_verify_recall). cell_id =
    // kHitRefClusterWide indicates the no-cell-filter path where row indexes
    // into the cluster bucket directly.
    constexpr uint32_t kHitRefClusterWide = std::numeric_limits<uint32_t>::max();
    std::vector<std::vector<HitRef>> query_hits(config.n_queries);
    for (auto& v : query_hits)
    {
      v.reserve(64);  // ~7.5 MB total for 10k queries, amortizes realloc
    }
    size_t dist_comp = 0;
    size_t triangle_candidate_clusters = 0;
    size_t triangle_pruned_clusters = 0;
    size_t slab_only_pruned_clusters = 0;
    size_t pca_only_pruned_clusters = 0;
    size_t combined_pruned_clusters = 0;
    size_t cell_candidate_vectors = 0;
    size_t cell_pruned_vectors = 0;
    size_t cell_kept_vectors = 0;
    size_t zero_kept_cells = 0;
    size_t zero_kept_surviving_clusters = 0;
    size_t cell_bytes_read = 0;
    size_t cluster_bytes_baseline = 0;
    size_t radius_prescreened_clusters = 0;
    size_t surviving_total = 0;
    const float epsilon =
        std::sqrt(static_cast<float>(config.radius_squared)) / config.approx_alpha;
    const float radius_l2_f = std::sqrt(static_cast<float>(config.radius_squared));
    const float radius_sq_f = static_cast<float>(config.radius_squared);
    const size_t dim = config.dim;
    const bool use_sqg =
        config.K < cluster_reader.cluster_num &&
        std::filesystem::exists(config.sqg_path);
    if (config.K < cluster_reader.cluster_num && !use_sqg)
    {
      std::cerr << "[index-graph-sqg] missing SQG artifact at "
                << config.sqg_path
                << "; falling back to brute-force centroid scan\n";
    }
    const bool use_slab = config.slab_cluster_prune_enabled &&
                          cluster_reader.slab_count > 0;
    const bool use_pca_cluster_prune = config.pca_cluster_prune_enabled;
    // A2 sparse-gather prototype (env DJ_A2_SPARSE=1): instead of the full
    // batch x 50000 centroid GEMM, compute q_dot only for the centroids the
    // slab lower bound will actually read (each candidate's self + its
    // slab_count neighbours, deduped ~7644/query). Cuts centroid dot-products
    // ~6.6x at the cost of losing batch-level reuse and doing scattered gathers.
    // Off by default; the dense GEMM path is unchanged.
    const bool a2_sparse = std::getenv("DJ_A2_SPARSE") != nullptr;
    // int8-VNNI dense centroid GEMM (DEFAULT ON; escape hatch DJ_NO_INT8=1):
    // mirror the dense centroid GEMM in int8 with AVX-512 VNNI (vpdpbusd),
    // keeping the sequential scan + batch reuse, cutting centroid-matrix
    // bandwidth ~4x (float32 -> int8). Per-centroid absmax int8 quant +
    // sign-correction sum computed once at startup. Verified 2026-06-01:
    // recall bit-identical (0.997628, dist_comp +1/145968 -- int8 quant error
    // stays well under the slab bound's ~2% tolerance, see non_zero_query doc
    // 5.6), GEMM 0.250->0.186s (1.34x) -> net ~+5%; ASan clean + UBSan clean
    // (only pre-existing third-party RaBitQ space.hpp shifts, identical to the
    // float path). Requires AVX-512 VNNI (build already targets it). Explicit
    // DJ_A2_SPARSE still wins if set (keeps the sparse-gather probe usable).
    const bool int8_vnni =
        std::getenv("DJ_NO_INT8") == nullptr && !a2_sparse;
    // centroid_dot critical-path probe (env DJ_QDOT_REPEAT=N, default 1):
    // recompute the dense centroid GEMM N times, discarding the redundant
    // results (final write to R is identical -> recall unchanged). Linearly
    // scales ONLY centroid_dot's cost, so a qps sweep tells whether centroid_dot
    // sits on the wall critical path (qps drops) or is shadow time (qps flat) --
    // the same perturb-and-watch test that falsified io_wait (README §18).
    const int qdot_repeat = [] {
      const char* e = std::getenv("DJ_QDOT_REPEAT");
      const int v = e ? std::atoi(e) : 1;
      return v > 1 ? v : 1;
    }();
    // Stage-0 perturb-and-watch probe (env DJ_QDOT_NOISE=sigma, default 0):
    // inject deterministic Gaussian noise (std = sigma * per-query RMS(q_dot))
    // into q_dot_block, which is consumed ONLY by the slab lower bound. Measures
    // how much q.centroid error the slab bound tolerates before recall drops --
    // a feasibility filter for replacing the float GEMM with a coarse RaBitQ
    // 1-bit q.centroid estimate (relative error ~0.1-0.3 at this dim). Off by
    // default; q_dot_block is byte-identical at sigma=0.
    const float dj_qdot_noise = [] {
      const char* e = std::getenv("DJ_QDOT_NOISE");
      const double v = e ? std::atof(e) : 0.0;
      return v > 0.0 ? static_cast<float>(v) : 0.0f;
    }();
    const size_t vnni_pad = ((config.dim + 63) / 64) * 64;
    std::vector<int8_t> c_i8;
    std::vector<float> c_i8_scale;
    std::vector<int32_t> c_i8_s8sum;
    if (int8_vnni)
    {
      const size_t cn = cluster_reader.cluster_num;
      c_i8.assign(cn * vnni_pad, 0);
      c_i8_scale.resize(cn);
      c_i8_s8sum.resize(cn);
      for (size_t c = 0; c < cn; ++c)
      {
        const float* cp = cluster_reader.centroids.data() + c * config.dim;
        float amax = 0.0f;
        for (size_t j = 0; j < config.dim; ++j)
          amax = std::max(amax, std::abs(cp[j]));
        const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        c_i8_scale[c] = sc;
        const float inv = 1.0f / sc;
        int8_t* dst = c_i8.data() + c * vnni_pad;
        int32_t s8sum = 0;
        for (size_t j = 0; j < config.dim; ++j)
        {
          const int v = static_cast<int>(std::lround(cp[j] * inv));
          dst[j] = static_cast<int8_t>(v);
          s8sum += v;
        }
        c_i8_s8sum[c] = s8sum;
      }
    }
    std::unique_ptr<FastPathEmptinessOracle> fast_path_oracle;
    if (config.fast_path_enabled)
    {
      std::cerr << "[fast-path] enabled mode=" << config.fast_path_mode
                << " nprobe_N=" << config.fast_path_nprobe_N << '\n';
      fast_path_oracle = std::make_unique<FastPathEmptinessOracle>(
          config.fast_path_mode, config.sqg_path, config.rabitq_1bit_codes_path,
          config.rabitq_8bit_codes_path, config.cluster_num, config.dim,
          config.sqg_m, config.sqg_ef_search, config.fast_path_nprobe_N,
          cluster_reader.centroids, cluster_reader.bucket_sizes, &cluster_reader);
    }
    const bool use_fast_path_oracle = static_cast<bool>(fast_path_oracle);
    std::unique_ptr<DataHNSWOracle> data_hnsw_oracle;
    if (config.data_hnsw_enabled)
    {
      data_hnsw_oracle =
          load_data_hnsw(config.data_hnsw_path, config.dim, config.data_hnsw_ef);
    }
    const bool use_data_hnsw_oracle = static_cast<bool>(data_hnsw_oracle);
    std::vector<DataHNSWTraceRecord> data_hnsw_trace;
    if (use_data_hnsw_oracle)
    {
      data_hnsw_trace.resize(config.n_queries);
    }
    std::vector<FastPathTraceRecord> fast_path_trace;
    if (use_fast_path_oracle)
    {
      fast_path_trace.resize(config.n_queries);
    }
    std::vector<SlowPathTraceRecord> slow_path_trace;
    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
    {
      slow_path_trace.resize(config.n_queries);
    }
    size_t data_hnsw_skip_count = 0;
    double data_hnsw_oracle_total_us = 0.0;
    size_t fast_path_skip_count = 0;
    double fast_path_oracle_total_us = 0.0;
    // Fused single-pass dispatch (default; set DJ_NO_FUSE=1 to force the legacy
    // two-pass path). Runs the emptiness oracle and the slow path in one
    // parallel region (no barrier between them); each thread accumulates
    // survivors into a thread-local batch of kQdotBatchSize so the GEMM still
    // amortizes the centroid matrix over a full batch. Survivors are counted
    // here for the QPS denominator (two-pass uses survivor_qids.size()).
    // Verified recall- and dist_comp-identical to two-pass (a batch only shares
    // the GEMM matrix; all pruning/exact work is per-query) and ASan/UBSan
    // clean; deep10m non-empty qps +9.8%, full-workload neutral.
    const bool dj_fuse = std::getenv("DJ_NO_FUSE") == nullptr;
    // Fused-loop dynamic chunk (env DJ_FUSE_CHUNK, default 16): how many queries
    // a thread grabs per work-steal. Smaller spreads scattered survivors more
    // evenly (helps the non-empty workload); larger cuts oracle scheduling
    // overhead on the full workload. Tunable while we characterize the trade-off.
    const int dj_fuse_chunk = [] {
      const char* e = std::getenv("DJ_FUSE_CHUNK");
      const int v = e ? std::atoi(e) : 4;
      return v > 0 ? v : 4;
    }();
    // decide-then-fetch overlap probe (env DJ_OVERLAP=1): measures the foldable
    // redundancy between the oracle's 1-bit scan (SQG top-nprobe_N clusters) and
    // slow-path phaseA's 1-bit classify (tl_io_plans clusters). For every
    // surviving (non-empty) query: oracle_1bit_vecs = vecs the oracle scanned;
    // phaseA_vecs = vecs phaseA scanned; intersect_vecs = vecs in clusters BOTH
    // scanned (re-runs the oracle's top-nprobe_N nav inside phaseA against the
    // same const SQG instance -> exact top-set, deterministic). The ceiling of
    // decide-then-fetch is bounded by (intersect/phaseA) * (phaseA time share).
    // Off by default; the env bool short-circuits all probe work (zero hot-path
    // cost when unset).
    const bool dj_overlap = std::getenv("DJ_OVERLAP") != nullptr;
    std::atomic<size_t> dj_ov_oracle_vecs{0};
    std::atomic<size_t> dj_ov_phasea_vecs{0};
    std::atomic<size_t> dj_ov_intersect_vecs{0};
    size_t fused_survivor_count = 0;

    const auto start = std::chrono::steady_clock::now();
    std::vector<float> radius_prescreen_thr_sq;
#ifndef DUMP_LB_SLAB
    if constexpr (!kDiskRangePruneBreakdownStatsEnabled)
    {
      if (config.K < cluster_reader.cluster_num)
      {
        radius_prescreen_thr_sq.resize(cluster_reader.cluster_num);
        for (size_t cluster_id = 0; cluster_id < cluster_reader.cluster_num;
             ++cluster_id)
        {
          const float threshold = epsilon + cluster_reader.radii[cluster_id];
          radius_prescreen_thr_sq[cluster_id] = threshold * threshold;
        }
      }
    }
#endif

    const bool dj_setuplog = std::getenv("DJ_BATCHLOG") != nullptr;
    const auto dj_t_prescreen = std::chrono::steady_clock::now();
    // Slow-path candidate-generation graph. When the fast-path oracle is active
    // it has already loaded the same SQG artifact, so the slow path reuses that
    // instance instead of loading a second copy of the 121MB graph inside the
    // search timer. Measured recall-identical (same graph, same ef) and ~54ms
    // faster per search() call (deep10m: full-workload qps +46%, non-empty
    // +60%). The two uses -- the oracle pre-pass and the slow loop -- are
    // separated by the parallel-for barrier, so sharing one instance introduces
    // no concurrency beyond what each segment already runs on its own copy.
    // Falls back to an owned load when no fast-path oracle is present.
    SQGCentroidIndex sqg_slow_owned;
    const SQGCentroidIndex* sqg_slow_path = &sqg_slow_owned;
    if (use_sqg && use_fast_path_oracle)
    {
      sqg_slow_path = &fast_path_oracle->centroid_index();
    }
    else if (use_sqg)
    {
      sqg_slow_owned.load(config.sqg_path, cluster_reader.cluster_num, dim,
                          config.sqg_m, config.sqg_ef_search);
    }
    if (dj_setuplog)
    {
      const auto now = std::chrono::steady_clock::now();
      std::cerr << "[setup] prescreen_thr="
                << std::chrono::duration<double>(dj_t_prescreen - start).count() *
                       1000
                << "ms  sqg_load="
                << std::chrono::duration<double>(now - dj_t_prescreen).count() *
                       1000
                << "ms\n";
    }
    std::atomic<bool> invalid_candidate(false);
    std::atomic<size_t> invalid_cluster_id(0);
    std::vector<std::array<QueryClassStats, 2>> per_thread_stats;
    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
    {
      per_thread_stats.resize(static_cast<size_t>(omp_get_max_threads()));
    }

    // Slow-path RBQ prefilter helpers (Phase A pre-compute + Phase B filter).
    // Generic lambdas so they can take the loop-local BatchQueryState by ref.
    // Caller is responsible for the config-flag guard;these only check the
    // necessary preconditions (plans non-empty, RBQ bounds present).
    // See slow_path_rbq_prefilter.md §六 (Phase A) and §七 (Phase B).
    auto precompute_rbq_bounds_phase_a =
        [&](auto& state,
            const std::vector<std::pair<size_t, ClusterReader::IOPlan>>& plans)
    {
      using AlignedFloatVector =
          std::vector<float, symqg::memory::AlignedAllocator<float>>;
      AlignedFloatVector rotated_query(fast_path_oracle->padded_dim(), 0.0f);
      fast_path_oracle->rotate_query(state.query_ptr, rotated_query.data());
      state.rbq_bounds_per_cluster.reserve(plans.size());
      // decide-then-fetch overlap probe: reproduce the oracle's top-nprobe_N nav
      // (same const SQG instance the oracle used) so we can tag which plan
      // clusters the oracle already 1-bit-scanned. Measurement-only (gated).
      std::vector<uint32_t> dj_ov_top;
      if (dj_overlap)
      {
        dj_ov_top = fast_path_oracle->centroid_index().search(
            state.query_ptr, config.fast_path_nprobe_N);
        std::sort(dj_ov_top.begin(), dj_ov_top.end());
      }
      for (const auto& plan : plans)
      {
        const size_t cid = plan.first;
        if (dj_overlap)
        {
          const size_t vc = fast_path_oracle->cluster_vec_count(cid);
          dj_ov_phasea_vecs.fetch_add(vc, std::memory_order_relaxed);
          if (std::binary_search(dj_ov_top.begin(), dj_ov_top.end(),
                                 static_cast<uint32_t>(cid)))
          {
            dj_ov_intersect_vecs.fetch_add(vc, std::memory_order_relaxed);
          }
        }
        std::vector<float> lb;
        std::vector<float> ub;
        fast_path_oracle->classify_cluster_rbq_1bit(rotated_query.data(), cid,
                                                    lb, ub);
        state.rbq_bounds_per_cluster.emplace_back(cid, std::move(lb),
                                                  std::move(ub));
      }
      std::sort(state.rbq_bounds_per_cluster.begin(),
                state.rbq_bounds_per_cluster.end(),
                [](const auto& a, const auto& b) {
                  return std::get<0>(a) < std::get<0>(b);
                });
    };

    auto filter_io_plans_phase_b =
        [&](auto& state,
            std::vector<std::pair<size_t, ClusterReader::IOPlan>>& plans)
    {
      std::vector<std::pair<size_t, ClusterReader::IOPlan>> filtered;
      filtered.reserve(plans.size());
      for (auto& plan : plans)
      {
        const size_t cluster_id = plan.first;
        auto bounds_it = std::lower_bound(
            state.rbq_bounds_per_cluster.begin(),
            state.rbq_bounds_per_cluster.end(), cluster_id,
            [](const auto& a, size_t b) { return std::get<0>(a) < b; });
        if (bounds_it == state.rbq_bounds_per_cluster.end() ||
            std::get<0>(*bounds_it) != cluster_id)
        {
          // No RBQ bounds for this cluster (defensive: shouldn't happen since
          // Phase A pre-computed for every cluster in plans);pass-through.
          filtered.emplace_back(cluster_id, std::move(plan.second));
          continue;
        }
        const std::vector<float>& lb_arr = std::get<1>(*bounds_it);
        const size_t first_cell_global =
            cluster_reader.cluster_cell_offsets[cluster_id];
        const size_t cluster_first_vec_global =
            cluster_reader.cell_vec_offsets[first_cell_global];
        ClusterReader::IOPlan new_kept_cells;
        new_kept_cells.reserve(plan.second.size());
        for (uint64_t cell_id_global : plan.second)
        {
          const size_t cell_idx_local =
              static_cast<size_t>(cell_id_global) - first_cell_global;
          const auto vec_range =
              cluster_reader.cell_global_vec_range(cluster_id, cell_idx_local);
          const size_t cluster_local_base =
              vec_range.first - cluster_first_vec_global;
          const size_t cell_vec_count = vec_range.second - vec_range.first;
          bool any_borderline = false;
          for (size_t i = 0; i < cell_vec_count; ++i)
          {
            if (lb_arr[cluster_local_base + i] <= radius_sq_f)
            {
              any_borderline = true;
              break;
            }
          }
          if (any_borderline)
          {
            new_kept_cells.push_back(cell_id_global);
          }
          else if constexpr (kDiskRangeSearchPhaseTimingEnabled)
          {
            ++state.q_cells_skipped_phase_b;
          }
        }
        if (!new_kept_cells.empty())
        {
          filtered.emplace_back(cluster_id, std::move(new_kept_cells));
        }
      }
      plans = std::move(filtered);
    };

    // ---- Pass 1: emptiness-oracle pre-pass + survivor compaction ----
    // The fast-path / data-HNSW oracle culls the large majority of queries
    // (~97% on deep10m). If the slow path below batched queries by contiguous
    // id, each kQdotBatchSize batch would carry <1 surviving query and the
    // q.centroid GEMM would degenerate into a per-query stream of the entire
    // centroid matrix (the centroid_dot bottleneck). Instead, evaluate every
    // oracle here and pack the survivors so each slow-path batch is densely
    // filled and the GEMM amortizes the centroid matrix over ~kQdotBatchSize
    // rows. Exact: identical decisions, ascending query order preserved.
    // Experiment knob (env DJ_NONEMPTY_ONLY=1): restrict the workload to
    // queries that actually have results (per_query_counts[q] > 0), to measure
    // slow-path throughput on a fully non-empty workload. Off by default, so the
    // normal search path is unaffected. recall is invariant (empty queries
    // contribute 0 to both total_hits and total_in_range); only the QPS
    // denominator changes (see qps computation below).
    const bool nonempty_only = std::getenv("DJ_NONEMPTY_ONLY") != nullptr;
    // Makespan probe (env DJ_MAKESPAN=1): the search is two SERIAL parallel-for
    // segments -- (A) fast-path oracle loop + survivor build, then (B) slow-path
    // loop. Time each segment's wall + segment B's per-thread busy distribution
    // to expose the real 24-thread critical path (overlap vs serial vs imbalance),
    // instead of inferring from thread-summed phase shares.
    const bool dj_mk = std::getenv("DJ_MAKESPAN") != nullptr;
    const int dj_mk_N = omp_get_max_threads();
    std::vector<double> dj_mk_sb(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<double> dj_mk_bw(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<double> dj_mk_ta(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<double> dj_sec1(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<double> dj_sec2(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<double> dj_mk_first(static_cast<size_t>(dj_mk_N), 1e18);
    std::vector<double> dj_mk_last(static_cast<size_t>(dj_mk_N), 0.0);
    std::vector<long> dj_mk_sn(static_cast<size_t>(dj_mk_N), 0);
    // Fused-mode per-thread span (DJ_MAKESPAN print): each thread records its
    // wall span over the fused omp-for (entry -> last pending flush). max span =
    // the fused loop's makespan; max/mean = load imbalance. The two-pass
    // makespan probe above does NOT populate per-thread busy in fused mode, so
    // this is the only valid imbalance read for the default path. 2 wtime/thread.
    std::vector<double> dj_fuse_span(static_cast<size_t>(dj_mk_N), 0.0);
    const double dj_mk_t0 = dj_mk ? omp_get_wtime() : 0.0;
    std::vector<size_t> survivor_qids;
    // Two-pass survivor pre-pass + compaction. Skipped when DJ_FUSE is set --
    // the fused dispatch runs the oracle inline and batches survivors per
    // thread, so survivor_qids stays empty in that mode.
    if (!dj_fuse)
    {
      std::vector<uint8_t> survives;
      if (use_fast_path_oracle || use_data_hnsw_oracle)
      {
        survives.assign(config.n_queries, 1);
#pragma omp parallel for schedule(dynamic, 256) reduction( \
        + : fast_path_skip_count, fast_path_oracle_total_us, \
            data_hnsw_skip_count, data_hnsw_oracle_total_us)
        for (size_t q = 0; q < config.n_queries; ++q)
        {
          // Non-empty-only experiment: skip the oracle on empty queries too, so
          // their evaluation time stays out of search_time (survives[q] keeps
          // its initial 1 but the survivor build below drops it anyway).
          if (nonempty_only && config.per_query_counts[q] == 0)
          {
            continue;
          }
          const float* qptr = queries.data() + q * config.dim;
          if (use_fast_path_oracle)
          {
            const auto oracle_start = std::chrono::steady_clock::now();
            const FastPathOracleResult oracle_result =
                fast_path_oracle->evaluate(qptr, config.radius_squared);
            const auto oracle_end = std::chrono::steady_clock::now();
            fast_path_oracle_total_us +=
                std::chrono::duration<double, std::micro>(oracle_end -
                                                          oracle_start)
                    .count();
            fast_path_trace[q] = {
                q,
                oracle_result.is_empty,
                oracle_result.phase1_candidate_count,
                oracle_result.phase2_scanned_vectors,
                oracle_result.phase2_borderline_count,
                oracle_result.phase3_refined_count,
                oracle_result.min_lower_bound,
                oracle_result.first_nan_field_mask,
                oracle_result.first_nan_cluster_id,
                oracle_result.first_nan_local_index};
            if (oracle_result.is_empty)
            {
              survives[q] = 0;
              ++fast_path_skip_count;
            }
          }
          else  // use_data_hnsw_oracle
          {
            const auto oracle_start = std::chrono::steady_clock::now();
            const DataHNSWOracleResult oracle_result =
                data_hnsw_oracle->evaluate(qptr, radius_l2_f,
                                           config.data_hnsw_margin_delta);
            const auto oracle_end = std::chrono::steady_clock::now();
            data_hnsw_oracle_total_us +=
                std::chrono::duration<double, std::micro>(oracle_end -
                                                          oracle_start)
                    .count();
            data_hnsw_trace[q] = {q, oracle_result.d_hat_l2,
                                  oracle_result.is_empty};
            if (oracle_result.is_empty)
            {
              survives[q] = 0;
              ++data_hnsw_skip_count;
            }
          }
        }
      }

      survivor_qids.reserve(config.n_queries);
      for (size_t q = 0; q < config.n_queries; ++q)
      {
        // Non-empty-only experiment: drop queries with no in-range results.
        if (nonempty_only && config.per_query_counts[q] == 0)
        {
          continue;
        }
        if (survives.empty() || survives[q])
        {
          survivor_qids.push_back(q);
          continue;
        }
        // Culled query: forms the zero-result class with zero phase time.
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          QueryClassStats culled{};
          culled.n_queries = 1;
          per_thread_stats[0][0].add(culled);
          SlowPathTraceRecord rec{};
          rec.query_id = q;
          rec.skipped_by_fast_path = true;
          slow_path_trace[q] = rec;
        }
      }
    }

    const double dj_mk_t1 = dj_mk ? omp_get_wtime() : 0.0;
    // Per-batch wall instrumentation (env DJ_BATCHLOG=1): record each batch's
    // start/end wall, owning thread, query count, cells read and io_wait to
    // expose the true batch-duration distribution and segB timeline. Each batch
    // is written once by its owning thread (unique ordinal), so no sync needed.
    const bool dj_blog = std::getenv("DJ_BATCHLOG") != nullptr;
    struct DjBatchRec
    {
      double t0 = 0, t1 = 0;
      int tid = -1;
      size_t nq = 0, cells = 0;
      double io = 0;
    };
    const size_t dj_blog_n =
        (survivor_qids.size() + kQdotBatchSize - 1) / kQdotBatchSize;
    std::vector<DjBatchRec> dj_blog_recs(dj_blog ? dj_blog_n : 0);
    const double dj_blog_origin = dj_blog ? omp_get_wtime() : 0.0;
    // Pass-2 accumulators are combined manually (thread-local shadows + one
    // critical merge) instead of an omp reduction clause. This is the
    // prerequisite for the fused single-pass loop (DJ_FUSE) below, where a
    // thread also flushes a partial survivor batch *after* the omp-for barrier
    // -- a point where omp-reduction privates are no longer writable. Behavior
    // is identical to a reduction: each thread sums into same-named shadows and
    // adds them to the shared totals once. The oracle counters (fast_path_*,
    // data_hnsw_*) are not written in this region, so they stay shared.
    size_t* const p_dist_comp = &dist_comp;
    size_t* const p_tcc = &triangle_candidate_clusters;
    size_t* const p_tpc = &triangle_pruned_clusters;
    size_t* const p_sopc = &slab_only_pruned_clusters;
    size_t* const p_popc = &pca_only_pruned_clusters;
    size_t* const p_cpc = &combined_pruned_clusters;
    size_t* const p_ccv = &cell_candidate_vectors;
    size_t* const p_cpv = &cell_pruned_vectors;
    size_t* const p_ckv = &cell_kept_vectors;
    size_t* const p_zkc = &zero_kept_cells;
    size_t* const p_zksc = &zero_kept_surviving_clusters;
    size_t* const p_cbr = &cell_bytes_read;
    size_t* const p_cbb = &cluster_bytes_baseline;
    size_t* const p_rpc = &radius_prescreened_clusters;
    size_t* const p_st = &surviving_total;
    // Oracle / fused-survivor accumulators (only written by the fused dispatch;
    // in two-pass mode Pass 1 already populated the shared totals and these
    // shadows stay 0, so the merge is a no-op).
    size_t* const p_fpsc = &fast_path_skip_count;
    double* const p_fpot = &fast_path_oracle_total_us;
    size_t* const p_dhsc = &data_hnsw_skip_count;
    double* const p_dhot = &data_hnsw_oracle_total_us;
    size_t* const p_fsc = &fused_survivor_count;
#pragma omp parallel
    {
      // Thread-local shadows (same names -> batch body unchanged).
      size_t dist_comp = 0, triangle_candidate_clusters = 0,
             triangle_pruned_clusters = 0, slab_only_pruned_clusters = 0,
             pca_only_pruned_clusters = 0, combined_pruned_clusters = 0,
             cell_candidate_vectors = 0, cell_pruned_vectors = 0,
             cell_kept_vectors = 0, zero_kept_cells = 0,
             zero_kept_surviving_clusters = 0, cell_bytes_read = 0,
             cluster_bytes_baseline = 0, radius_prescreened_clusters = 0,
             surviving_total = 0;
      size_t fast_path_skip_count = 0, data_hnsw_skip_count = 0,
             fused_survivor_count = 0;
      double fast_path_oracle_total_us = 0.0, data_hnsw_oracle_total_us = 0.0;
    // Core batch processing extracted into a per-thread lambda so both the
    // default two-pass dispatch and the fused single-pass dispatch (DJ_FUSE)
    // can drive it. Captures the thread-local shadows + thread_local buffers by
    // reference; qids/batch_size replace the old survivor_qids[batch_start+i].
    auto process_batch = [&](const size_t* qids, const size_t batch_size,
                             const size_t batch_ord, const double dj_blog_t0) {
      struct BatchQueryState
      {
        size_t q = 0;
        const float* query_ptr = nullptr;
        const uint8_t* query_raw_byte_ptr = nullptr;
        int32_t query_norm_sq_byte = 0;
        QueryClassStats q_stats;
        size_t q_dist_comp = 0;
        size_t q_cell_kept = 0;
        size_t q_cells_read = 0;
        size_t q_bytes_read = 0;
        double l2_secs_this_query = 0.0;
        double q_hit_dist_sum = 0.0;
        size_t q_s_nh = 0;
        size_t q_s_h = 0;
        size_t q_s_borderline = 0;
        size_t q_cells_all_rejected = 0;
        size_t q_cells_total = 0;
        size_t q_cells_skipped_phase_b = 0;
        std::chrono::steady_clock::time_point phase_start;
        bool skipped_by_fast_path = false;
        bool skipped_by_data_hnsw = false;
        std::vector<std::pair<float, size_t>> candidates;
        std::vector<std::pair<float, size_t>> active_candidates;
        std::vector<size_t> surviving;
        // Slow-path RBQ prefilter cache: per cluster id, vec-level lb/ub arrays
        // sized to that cluster's vec_count. Sorted by cluster_id so the cell
        // callback can binary-search.
        std::vector<std::tuple<size_t, std::vector<float>, std::vector<float>>>
            rbq_bounds_per_cluster;

        bool skipped() const
        {
          return skipped_by_fast_path || skipped_by_data_hnsw;
        }
      };

      std::array<BatchQueryState, kQdotBatchSize> batch_state;
      thread_local std::vector<float> q_batch_buf;
      thread_local std::vector<float> q_dot_block;
      thread_local std::vector<float> q_batch_compact_buf;
      thread_local std::vector<float> q_dot_compact_block;
      thread_local std::array<size_t, kQdotBatchSize> q_dot_compact_rows;
      thread_local std::array<std::vector<float>, kQdotBatchSize>
          q_dot_centroids_legacy;
      if (q_batch_buf.size() < kQdotBatchSize * dim)
      {
        q_batch_buf.resize(kQdotBatchSize * dim);
      }
      if constexpr (!kDiskRangePruneBreakdownStatsEnabled)
      {
#ifndef DUMP_LB_SLAB
        if (q_dot_block.size() < kQdotBatchSize * cluster_reader.cluster_num)
        {
          q_dot_block.resize(kQdotBatchSize * cluster_reader.cluster_num);
        }
#endif
      }

      auto record_phase = [&](BatchQueryState& state, double& target) {
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          const auto phase_end = std::chrono::steady_clock::now();
          target +=
              std::chrono::duration<double>(phase_end - state.phase_start).count();
          state.phase_start = phase_end;
        }
      };

      for (size_t i = 0; i < batch_size; ++i)
      {
        BatchQueryState& state = batch_state[i];
        state.q = qids[i];
        state.query_ptr = queries.data() + state.q * config.dim;
        if (range_search_config::is_byte_dtype(config.vec_dtype))
        {
          state.query_raw_byte_ptr =
              query_data.raw_bytes.data() + state.q * config.dim;
          state.query_norm_sq_byte = query_data.norm_sq_byte[state.q];
        }
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          state.phase_start = std::chrono::steady_clock::now();
        }
        // Phase 1 (emptiness oracle) ran in the survivor pre-pass above; every
        // query in this batch is a survivor, so there is no per-query skip
        // check here and the slow-path GEMM batch is densely packed.
        state.candidates.reserve(std::min(config.K, cluster_reader.cluster_num));
        size_t distance_dim = dim;
        if (!use_sqg)
        {
          state.candidates.reserve(cluster_reader.cluster_num);
          for (size_t cluster_id = 0; cluster_id < cluster_reader.cluster_num;
               ++cluster_id)
          {
            const float* centroid_ptr =
                cluster_reader.centroids.data() + cluster_id * config.dim;
            const float centroid_dist_sq =
                utils::L2Sqr(state.query_ptr, centroid_ptr, &distance_dim);
            state.candidates.emplace_back(centroid_dist_sq, cluster_id);
          }
        }
        else
        {
          const std::vector<uint32_t> ids =
              sqg_slow_path->search(state.query_ptr, config.K);
          state.candidates.reserve(ids.size());
          for (uint32_t id : ids)
          {
            const size_t cluster_id = static_cast<size_t>(id);
            if (cluster_id >= cluster_reader.cluster_num)
            {
              invalid_cluster_id.store(cluster_id, std::memory_order_relaxed);
              invalid_candidate.store(true, std::memory_order_relaxed);
              continue;
            }
            const float* centroid_ptr =
                cluster_reader.centroids.data() + cluster_id * config.dim;
            const float centroid_dist_sq =
                utils::L2Sqr(state.query_ptr, centroid_ptr, &distance_dim);
            state.candidates.emplace_back(centroid_dist_sq, cluster_id);
          }
        }
        record_phase(state, state.q_stats.t_candidate_gen);

        state.active_candidates.reserve(state.candidates.size());
#ifndef DUMP_LB_SLAB
        if constexpr (!kDiskRangePruneBreakdownStatsEnabled)
        {
          if (config.K < cluster_reader.cluster_num)
          {
            for (const auto& candidate : state.candidates)
            {
              if (candidate.first > radius_prescreen_thr_sq[candidate.second])
              {
                ++radius_prescreened_clusters;
                if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                {
                  ++state.q_stats.radius_prescreened;
                }
                continue;
              }
              state.active_candidates.push_back(candidate);
            }
          }
          else
          {
            state.active_candidates = state.candidates;
          }
        }
        else
        {
          state.active_candidates = state.candidates;
        }
#else
        state.active_candidates = state.candidates;
#endif
        std::copy(state.query_ptr, state.query_ptr + dim,
                  q_batch_buf.data() + i * dim);
        record_phase(state, state.q_stats.t_centroid_dot);
      }

      auto compute_naive_qdot_block = [&]() {
        for (size_t i = 0; i < batch_size; ++i)
        {
          BatchQueryState& state = batch_state[i];
          if (state.skipped())
          {
            continue;
          }
          const auto qdot_start = std::chrono::steady_clock::now();
          if (use_slab && !state.active_candidates.empty())
          {
            std::vector<float>& q_dot_centroids = q_dot_centroids_legacy[i];
            if (q_dot_centroids.size() != cluster_reader.cluster_num)
            {
              q_dot_centroids.resize(cluster_reader.cluster_num);
            }
            for (size_t cluster_id = 0; cluster_id < cluster_reader.cluster_num;
                 ++cluster_id)
            {
              const float* centroid_ptr =
                  cluster_reader.centroids.data() + cluster_id * config.dim;
              float dot = 0.0f;
              for (size_t j = 0; j < config.dim; ++j)
              {
                dot += state.query_ptr[j] * centroid_ptr[j];
              }
              q_dot_centroids[cluster_id] = dot;
            }
          }
          if constexpr (kDiskRangeSearchPhaseTimingEnabled)
          {
            const auto qdot_end = std::chrono::steady_clock::now();
            state.q_stats.t_centroid_dot +=
                std::chrono::duration<double>(qdot_end - qdot_start).count();
            state.phase_start = qdot_end;
          }
        }
      };

      auto compute_batched_qdot = [&]() {
        size_t compact_size = 0;
        size_t timing_rows = 0;
        for (size_t i = 0; i < batch_size; ++i)
        {
          if (batch_state[i].skipped())
          {
            continue;
          }
          ++timing_rows;
          if (use_slab && !batch_state[i].active_candidates.empty())
          {
            q_dot_compact_rows[compact_size++] = i;
          }
        }
        const auto qdot_start = std::chrono::steady_clock::now();
        if (compact_size > 0)
        {
          QdotMatrixConstMap M(cluster_reader.centroids.data(),
                               cluster_reader.cluster_num, dim);
          if (use_data_hnsw_oracle || use_fast_path_oracle)
          {
            const size_t compact_query_floats = compact_size * dim;
            const size_t compact_qdot_floats =
                compact_size * cluster_reader.cluster_num;
            if (q_batch_compact_buf.size() < compact_query_floats)
            {
              q_batch_compact_buf.resize(compact_query_floats);
            }
            if (q_dot_compact_block.size() < compact_qdot_floats)
            {
              q_dot_compact_block.resize(compact_qdot_floats);
            }
            if (int8_vnni)
            {
              // int8-VNNI dense GEMM: outer over centroids (sequential scan,
              // bandwidth-friendly), inner over the batch's queries reusing the
              // centroid's int8 row held in zmm registers (GEMM-style reuse).
              const size_t cn = cluster_reader.cluster_num;
              const size_t nblk = vnni_pad / 64;
              thread_local std::vector<uint8_t> q_u8;
              thread_local std::vector<float> q_sc;
              q_u8.assign(compact_size * vnni_pad, 128);  // pad lanes: q_s8=0
              q_sc.resize(compact_size);
              for (size_t r = 0; r < compact_size; ++r)
              {
                const float* qp = batch_state[q_dot_compact_rows[r]].query_ptr;
                float amax = 0.0f;
                for (size_t j = 0; j < dim; ++j)
                  amax = std::max(amax, std::abs(qp[j]));
                const float sq = amax > 0.0f ? amax / 127.0f : 1.0f;
                q_sc[r] = sq;
                const float inv = 1.0f / sq;
                uint8_t* dst = q_u8.data() + r * vnni_pad;
                for (size_t j = 0; j < dim; ++j)
                {
                  const int v = static_cast<int>(std::lround(qp[j] * inv)) + 128;
                  dst[j] = static_cast<uint8_t>(v);
                }
              }
              for (size_t c = 0; c < cn; ++c)
              {
                const int8_t* cptr = c_i8.data() + c * vnni_pad;
                __m512i cv[8];
                for (size_t b = 0; b < nblk; ++b)
                  cv[b] = _mm512_loadu_si512(cptr + b * 64);
                const float scc = c_i8_scale[c];
                const int32_t corr = 128 * c_i8_s8sum[c];
                for (size_t r = 0; r < compact_size; ++r)
                {
                  const uint8_t* qptr = q_u8.data() + r * vnni_pad;
                  __m512i acc = _mm512_setzero_si512();
                  for (size_t b = 0; b < nblk; ++b)
                    acc = _mm512_dpbusd_epi32(
                        acc, _mm512_loadu_si512(qptr + b * 64), cv[b]);
                  const int32_t dot = _mm512_reduce_add_epi32(acc) - corr;
                  q_dot_block[q_dot_compact_rows[r] * cn + c] =
                      q_sc[r] * scc * static_cast<float>(dot);
                }
              }
            }
            else if (a2_sparse)
            {
              // Sparse path: for each query, compute q_dot only for the
              // centroids the slab bound will read (each active candidate's
              // self + its slab_count neighbours), deduped via an epoch-seen
              // set. The dense q_dot_block keeps full width so the consumer
              // (pca_slab_lower_bound) still indexes by cluster_id unchanged;
              // untouched rows are never read for this query.
              const size_t cn = cluster_reader.cluster_num;
              const size_t sc = cluster_reader.slab_count;
              thread_local std::vector<uint32_t> need_seen;
              thread_local uint32_t need_epoch = 0;
              thread_local std::vector<uint32_t> need_ids;
              if (need_seen.size() != cn)
              {
                need_seen.assign(cn, 0);
                need_epoch = 0;
              }
              for (size_t row = 0; row < compact_size; ++row)
              {
                const size_t bi = q_dot_compact_rows[row];
                const BatchQueryState& state = batch_state[bi];
                const float* qp = state.query_ptr;
                float* out = q_dot_block.data() + bi * cn;
                ++need_epoch;
                if (need_epoch == 0)
                {
                  std::fill(need_seen.begin(), need_seen.end(), 0);
                  need_epoch = 1;
                }
                need_ids.clear();
                for (const auto& candidate : state.active_candidates)
                {
                  const size_t cid = candidate.second;
                  if (need_seen[cid] != need_epoch)
                  {
                    need_seen[cid] = need_epoch;
                    need_ids.push_back(static_cast<uint32_t>(cid));
                  }
                  const size_t so = cid * sc;
                  for (size_t j = 0; j < sc; ++j)
                  {
                    const uint32_t nb =
                        cluster_reader.slab_neighbour_indices[so + j];
                    if (need_seen[nb] != need_epoch)
                    {
                      need_seen[nb] = need_epoch;
                      need_ids.push_back(nb);
                    }
                  }
                }
                for (const uint32_t cid : need_ids)
                {
                  const float* cp =
                      cluster_reader.centroids.data() +
                      static_cast<size_t>(cid) * dim;
                  float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
                  for (size_t j = 0; j < dim; ++j)
                  {
                    acc += qp[j] * cp[j];
                  }
                  out[cid] = acc;
                }
              }
            }
            else
            {
              for (size_t row = 0; row < compact_size; ++row)
              {
                const BatchQueryState& state =
                    batch_state[q_dot_compact_rows[row]];
                std::copy(state.query_ptr, state.query_ptr + dim,
                          q_batch_compact_buf.data() + row * dim);
              }
              QdotMatrixConstMap Q(q_batch_compact_buf.data(), compact_size,
                                   dim);
              QdotMatrixMap R(q_dot_compact_block.data(), compact_size,
                              cluster_reader.cluster_num);
              for (int rep = 0; rep < qdot_repeat; ++rep)
                R.noalias() = Q * M.transpose();
              for (size_t row = 0; row < compact_size; ++row)
              {
                std::copy(q_dot_compact_block.data() +
                              row * cluster_reader.cluster_num,
                          q_dot_compact_block.data() +
                              (row + 1) * cluster_reader.cluster_num,
                          q_dot_block.data() +
                              q_dot_compact_rows[row] *
                                  cluster_reader.cluster_num);
              }
            }
          }
          else
          {
            QdotMatrixConstMap Q(q_batch_buf.data(), batch_size, dim);
            QdotMatrixMap R(q_dot_block.data(), batch_size,
                            cluster_reader.cluster_num);
            for (int rep = 0; rep < qdot_repeat; ++rep)
              R.noalias() = Q * M.transpose();
          }
        }
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          const auto qdot_end = std::chrono::steady_clock::now();
          const double per_query_qdot =
              timing_rows == 0
                  ? 0.0
                  : std::chrono::duration<double>(qdot_end - qdot_start)
                            .count() /
                        static_cast<double>(timing_rows);
          for (size_t i = 0; i < batch_size; ++i)
          {
            if (batch_state[i].skipped())
            {
              continue;
            }
            batch_state[i].q_stats.t_centroid_dot += per_query_qdot;
            batch_state[i].phase_start = qdot_end;
          }
        }
      };
      (void)compute_naive_qdot_block;
      (void)compute_batched_qdot;

      if (dj_mk)
        dj_mk_ta[static_cast<size_t>(omp_get_thread_num())] = omp_get_wtime();
      if constexpr (kDiskRangePruneBreakdownStatsEnabled)
      {
        compute_naive_qdot_block();
      }
      else
      {
#ifdef DUMP_LB_SLAB
        compute_naive_qdot_block();
#else
        compute_batched_qdot();
#endif
      }
      if (dj_mk)
      {
        const size_t dj_tid2 = static_cast<size_t>(omp_get_thread_num());
        const double dj_now2 = omp_get_wtime();
        dj_sec1[dj_tid2] += dj_mk_ta[dj_tid2] - dj_mk_bw[dj_tid2];
        dj_sec2[dj_tid2] += dj_now2 - dj_mk_ta[dj_tid2];
      }

      for (size_t i = 0; i < batch_size; ++i)
      {
        BatchQueryState& state = batch_state[i];
        if (state.skipped())
        {
          continue;
        }
        if (dj_qdot_noise > 0.0f)
        {
          // Stage-0 perturb-and-watch: deterministic Gaussian noise into
          // q_dot_block row i (std = sigma * RMS(q_dot)). Seeded by (query,
          // centroid) via splitmix64 -> recall reproducible across thread
          // schedules. Perturbs ONLY the slab bound's input; the PCA box bound
          // (state.query_ptr) stays exact, mirroring "RaBitQ the slab, keep the
          // box bound as backstop".
          float* const qrow =
              q_dot_block.data() + i * cluster_reader.cluster_num;
          const size_t cn = cluster_reader.cluster_num;
          double sumsq = 0.0;
          for (size_t c = 0; c < cn; ++c)
            sumsq += static_cast<double>(qrow[c]) * qrow[c];
          const float nscale =
              dj_qdot_noise *
              static_cast<float>(std::sqrt(sumsq / static_cast<double>(cn)));
          const auto sm64 = [](uint64_t x) {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
          };
          const uint64_t qbase =
              static_cast<uint64_t>(state.q) * 0xD1B54A32D192ED03ULL;
          for (size_t c = 0; c < cn; ++c)
          {
            const uint64_t h1 = sm64(qbase + c);
            const uint64_t h2 = sm64(h1);
            const double u1 =
                (static_cast<double>(h1 >> 11) + 1.0) / 9007199254740993.0;
            const double u2 =
                static_cast<double>(h2 >> 11) / 9007199254740992.0;
            const double z = std::sqrt(-2.0 * std::log(u1)) *
                             std::cos(6.283185307179586 * u2);
            qrow[c] += nscale * static_cast<float>(z);
          }
        }
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          if (use_slab)
          {
            thread_local std::vector<uint32_t> qdot_seen;
            thread_local uint32_t qdot_epoch = 0;
            if (qdot_seen.size() != cluster_reader.cluster_num)
            {
              qdot_seen.assign(cluster_reader.cluster_num, 0);
              qdot_epoch = 0;
            }
            ++qdot_epoch;
            if (qdot_epoch == 0)
            {
              std::fill(qdot_seen.begin(), qdot_seen.end(), 0);
              qdot_epoch = 1;
            }

            size_t q_distinct_qdot_used = 0;
            const auto mark_qdot_used = [&](size_t cluster_id) {
              if (qdot_seen[cluster_id] == qdot_epoch)
              {
                return;
              }
              qdot_seen[cluster_id] = qdot_epoch;
              ++q_distinct_qdot_used;
            };

            for (const auto& candidate : state.active_candidates)
            {
              const size_t cluster_id = candidate.second;
              mark_qdot_used(cluster_id);
              const size_t slab_offset = cluster_id * cluster_reader.slab_count;
              for (size_t j = 0; j < cluster_reader.slab_count; ++j)
              {
                mark_qdot_used(static_cast<size_t>(
                    cluster_reader.slab_neighbour_indices[slab_offset + j]));
              }
            }
            state.q_stats.distinct_qdot_used = q_distinct_qdot_used;
          }
          state.phase_start = std::chrono::steady_clock::now();
        }

        state.surviving.reserve(state.active_candidates.size());
        triangle_candidate_clusters += state.candidates.size();
#ifdef DUMP_LB_SLAB
        auto dump_lb_row = [&](size_t cluster_id, float lb_slab, float lb_box) {
          constexpr size_t kDumpQueryLimit = 100;
          if (state.q >= kDumpQueryLimit)
          {
            return;
          }
          static std::mutex dump_mutex;
          static std::ofstream dump_file;
          std::lock_guard<std::mutex> guard(dump_mutex);
          if (!dump_file.is_open())
          {
            const std::filesystem::path dump_path =
                std::filesystem::path(cluster_reader.metadatafile()).parent_path() /
                "cpp_lb_dump.csv";
            dump_file.open(dump_path);
            if (!dump_file)
            {
              throw std::runtime_error(
                  "failed to open cpp_lb_dump.csv for write at " +
                  dump_path.string());
            }
            dump_file << "query_id,cluster_id,lb_slab,lb_box_residual,lb_final\n";
          }
          dump_file << std::fixed << std::setprecision(7) << state.q << ','
                    << cluster_id << ',' << lb_slab << ',' << lb_box << ','
                    << std::max(lb_slab, lb_box) << '\n';
        };
#endif
        for (const auto& candidate : state.active_candidates)
        {
          const size_t cluster_id = candidate.second;
          float lb_slab = 0.0f;
          if (use_slab)
          {
            float slab_threshold = epsilon;
            if constexpr (kDiskRangePruneBreakdownStatsEnabled)
            {
              slab_threshold = std::numeric_limits<float>::infinity();
            }
#ifdef DUMP_LB_SLAB
            slab_threshold = std::numeric_limits<float>::infinity();
#endif
            const std::vector<float>& q_dot_centroids =
                q_dot_centroids_legacy[i];
            if constexpr (kDiskRangePruneBreakdownStatsEnabled)
            {
              lb_slab = cluster_reader.pca_slab_lower_bound(
                  cluster_id, q_dot_centroids, slab_threshold);
            }
            else
            {
#ifdef DUMP_LB_SLAB
              lb_slab = cluster_reader.pca_slab_lower_bound(
                  cluster_id, q_dot_centroids, slab_threshold);
#else
              lb_slab = cluster_reader.pca_slab_lower_bound(
                  cluster_id,
                  q_dot_block.data() + i * cluster_reader.cluster_num,
                  cluster_reader.cluster_num, slab_threshold);
#endif
            }
          }
          if constexpr (kDiskRangePruneBreakdownStatsEnabled)
          {
            const float lb_box =
                use_pca_cluster_prune
                    ? cluster_reader.pca_lower_bound(cluster_id,
                                                     state.query_ptr)
                    : 0.0f;
            const bool slab_pruned = lb_slab > epsilon;
            const bool pca_pruned = lb_box > epsilon;
            if (slab_pruned)
            {
              ++slab_only_pruned_clusters;
            }
            if (pca_pruned)
            {
              ++pca_only_pruned_clusters;
            }
            if (slab_pruned || pca_pruned)
            {
              ++combined_pruned_clusters;
            }
#ifdef DUMP_LB_SLAB
            dump_lb_row(cluster_id, lb_slab, lb_box);
#endif
            if (slab_pruned)
            {
              ++triangle_pruned_clusters;
              cluster_reader.slab_pruned_clusters.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            if (pca_pruned)
            {
              ++triangle_pruned_clusters;
              cluster_reader.box_residual_pruned_clusters.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
            state.surviving.push_back(cluster_id);
            continue;
          }
          if (lb_slab > epsilon)
          {
#ifdef DUMP_LB_SLAB
            const float lb_box_dump =
                use_pca_cluster_prune
                    ? cluster_reader.pca_lower_bound(cluster_id,
                                                     state.query_ptr)
                    : 0.0f;
            dump_lb_row(cluster_id, lb_slab, lb_box_dump);
#endif
            ++triangle_pruned_clusters;
            cluster_reader.slab_pruned_clusters.fetch_add(
                1, std::memory_order_relaxed);
            continue;
          }
          if (use_pca_cluster_prune)
          {
            const float lb_box =
                cluster_reader.pca_lower_bound(cluster_id, state.query_ptr);
#ifdef DUMP_LB_SLAB
            dump_lb_row(cluster_id, lb_slab, lb_box);
#endif
            if (lb_box > epsilon)
            {
              ++triangle_pruned_clusters;
              cluster_reader.box_residual_pruned_clusters.fetch_add(
                  1, std::memory_order_relaxed);
              continue;
            }
          }
#ifdef DUMP_LB_SLAB
          else
          {
            dump_lb_row(cluster_id, lb_slab, 0.0f);
          }
#endif
          state.surviving.push_back(cluster_id);
        }
        surviving_total += state.surviving.size();
        record_phase(state, state.q_stats.t_cluster_bound);

        thread_local std::vector<size_t> tl_io_clusters;
        thread_local std::vector<std::pair<size_t, ClusterReader::IOPlan>>
            tl_io_plans;
        thread_local std::vector<float> tl_q_centered;
        thread_local std::vector<float> tl_q_p;
        tl_io_clusters.clear();
        tl_io_plans.clear();

        if (!config.cell_filter_enabled)
        {
          tl_io_clusters = state.surviving;
          if constexpr (kDiskRangePruneBreakdownStatsEnabled ||
                        kDiskRangeSearchPhaseTimingEnabled)
          {
            for (const size_t cluster_id : state.surviving)
            {
              const size_t bucket_size = cluster_reader.bucket_sizes[cluster_id];
              if (bucket_size == 0)
              {
                continue;
              }
              const size_t bytes =
                  bucket_size * dim * cluster_reader.bytes_per_element;
              if constexpr (kDiskRangePruneBreakdownStatsEnabled)
              {
                cluster_bytes_baseline += bytes;
                cell_bytes_read += bytes;
              }
              // bytes_read 累加移到 no-cell-filter callback 内,反映实际读的 bytes
            }
          }
        }
        else
        {
          for (const size_t cluster_id : state.surviving)
          {
            const size_t bucket_size = cluster_reader.bucket_sizes[cluster_id];
            if (bucket_size == 0)
            {
              continue;
            }

            tl_q_centered.assign(config.dim, 0.0f);
            const float* centroid_ptr =
                cluster_reader.centroids.data() + cluster_id * config.dim;
            float total_sq = 0.0f;
            for (size_t col = 0; col < config.dim; ++col)
            {
              const float diff = state.query_ptr[col] - centroid_ptr[col];
              tl_q_centered[col] = diff;
              total_sq += diff * diff;
            }
            const uint32_t effective_rank =
                cluster_reader.pca_effective_rank[cluster_id];
            tl_q_p.assign(effective_rank, 0.0f);
            const size_t basis_offset =
                cluster_id * cluster_reader.pca_rank * config.dim;
            float projected_sq = 0.0f;
            for (size_t axis = 0; axis < effective_rank; ++axis)
            {
              const float* basis_row =
                  cluster_reader.pca_basis.data() + basis_offset +
                  axis * config.dim;
              float coord = 0.0f;
              for (size_t col = 0; col < config.dim; ++col)
              {
                coord += basis_row[col] * tl_q_centered[col];
              }
              tl_q_p[axis] = coord;
              projected_sq += tl_q_p[axis] * tl_q_p[axis];
            }
            const float q_r_norm =
                std::sqrt(std::max(0.0f, total_sq - projected_sq));

            const size_t cell_count = cluster_reader.cell_count_of(cluster_id);
            ClusterReader::IOPlan kept_cells;
            kept_cells.reserve(cell_count);
            size_t kept_vec_count = 0;
            size_t pruned_vec_count = 0;
            size_t zero_kept_cell_count = 0;
            const uint64_t first_cell =
                cluster_reader.cluster_cell_offsets[cluster_id];
            for (size_t cell_idx = 0; cell_idx < cell_count; ++cell_idx)
            {
              const auto [vec_start, vec_end] =
                  cluster_reader.cell_global_vec_range(cluster_id, cell_idx);
              const size_t cell_vec_count = vec_end - vec_start;
              const float cell_lb = utils::cell_lb_batch(
                  cluster_reader.cell_lo_ptr(cluster_id, cell_idx),
                  cluster_reader.cell_hi_ptr(cluster_id, cell_idx),
                  cluster_reader.cell_res_lo(cluster_id, cell_idx),
                  cluster_reader.cell_res_hi(cluster_id, cell_idx),
                  tl_q_p.data(), q_r_norm, effective_rank);
              if (cell_lb <= epsilon)
              {
                kept_cells.push_back(first_cell + cell_idx);
                kept_vec_count += cell_vec_count;
              }
              else
              {
                pruned_vec_count += cell_vec_count;
                ++zero_kept_cell_count;
              }
            }
            if constexpr (kDiskRangePruneBreakdownStatsEnabled)
            {
              cell_candidate_vectors += bucket_size;
              cell_pruned_vectors += pruned_vec_count;
              cell_kept_vectors += kept_vec_count;
              zero_kept_cells += zero_kept_cell_count;
              cluster_bytes_baseline +=
                  bucket_size * dim * cluster_reader.bytes_per_element;
              cell_bytes_read +=
                  kept_vec_count * dim * cluster_reader.bytes_per_element;
            }
            if constexpr (kDiskRangeSearchPhaseTimingEnabled)
            {
              state.q_cell_kept += kept_vec_count;
              // bytes_read 累加移到 callback 内,反映实际读的 bytes
              // (Phase B 跳过的 cell 不会进 callback,自然不累加)
            }
            if (kept_cells.empty())
            {
              if constexpr (kDiskRangePruneBreakdownStatsEnabled)
              {
                ++zero_kept_surviving_clusters;
              }
              continue;
            }
            if constexpr (kDiskRangeSearchPhaseTimingEnabled)
            {
              state.q_cells_read += kept_cells.size();
            }
            tl_io_plans.emplace_back(cluster_id, std::move(kept_cells));
          }
        }
        record_phase(state, state.q_stats.t_cell_filter);

        // Slow-path RBQ prefilter dispatch (helpers defined above search()
        // entrypoint loop). Phase A: pre-compute per-vec lb/ub for each
        // candidate cluster. Phase B: drop cells with all vec lb-rejected
        // from the IO plan.
        if (config.slow_path_rbq_prefilter_enabled && use_fast_path_oracle &&
            !tl_io_plans.empty())
        {
          precompute_rbq_bounds_phase_a(state, tl_io_plans);
        }
        if (config.slow_path_rbq_phase_b_enabled &&
            !state.rbq_bounds_per_cluster.empty())
        {
          filter_io_plans_phase_b(state, tl_io_plans);
        }
        record_phase(state, state.q_stats.t_rbq_prefilter);

        size_t distance_dim = dim;
        auto exact_l2 = [&](const BatchQueryState& state, size_t cluster_id,
                            const void* payload, size_t row,
                            size_t cluster_local_row_base) -> double {
          if (cluster_reader.vec_dtype == range_search_config::kVecDTypeInt8)
          {
            const size_t global_row =
                cluster_reader.cluster_global_vec_offset(cluster_id) +
                cluster_local_row_base + row;
            if (global_row >= cluster_reader.sum_for_offset.size() ||
                global_row >= cluster_reader.norm_sq_for_l2.size())
            {
              throw std::runtime_error(
                  "int8 metadata row out of range during exact L2");
            }
            const int8_t* base_ptr =
                static_cast<const int8_t*>(payload) + row * config.dim;
            return utils::L2SqrInt8SIMD(
                reinterpret_cast<const int8_t*>(state.query_raw_byte_ptr),
                base_ptr, config.dim, state.query_norm_sq_byte,
                cluster_reader.sum_for_offset[global_row],
                cluster_reader.norm_sq_for_l2[global_row]);
          }
          if (cluster_reader.vec_dtype ==
              range_search_config::kVecDTypeUint8Reserved)
          {
            const size_t global_row =
                cluster_reader.cluster_global_vec_offset(cluster_id) +
                cluster_local_row_base + row;
            if (global_row >= cluster_reader.sum_for_offset.size() ||
                global_row >= cluster_reader.norm_sq_for_l2.size())
            {
              throw std::runtime_error(
                  "uint8 metadata row out of range during exact L2");
            }
            const uint8_t* base_ptr =
                static_cast<const uint8_t*>(payload) + row * config.dim;
            return static_cast<double>(utils::L2SqrUint8SIMDInt64(
                state.query_raw_byte_ptr, base_ptr, config.dim,
                state.query_norm_sq_byte,
                cluster_reader.sum_for_offset[global_row],
                cluster_reader.norm_sq_for_l2[global_row]));
          }
          const float* base_ptr =
              static_cast<const float*>(payload) + row * config.dim;
          return utils::L2Sqr(state.query_ptr, base_ptr, &distance_dim);
        };
        auto exact_hit_within_radius = [&](double dist) -> bool {
          if (cluster_reader.vec_dtype ==
              range_search_config::kVecDTypeUint8Reserved)
          {
            return dist <= config.radius_squared;
          }
          return dist <= radius_sq_f;
        };
        std::chrono::steady_clock::time_point drain_start;
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          drain_start = std::chrono::steady_clock::now();
        }
        if (config.cell_filter_enabled)
        {
          cluster_reader.submit_and_drain(
              omp_get_thread_num(), tl_io_plans,
              [&](size_t cluster_id, uint64_t cell_id, const void* cell_data,
                  size_t vec_count, size_t cluster_local_row_base)
              {
                (void)cell_id;
                std::chrono::steady_clock::time_point l2_start;
                if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                {
                  l2_start = std::chrono::steady_clock::now();
                  ++state.q_cells_total;
                  state.q_bytes_read +=
                      vec_count * config.dim * cluster_reader.bytes_per_element;
                }
                const size_t base_s_nh = state.q_s_nh;
                // Look up RBQ bounds for this cluster (Phase A prefilter).
                const float* lb_ptr = nullptr;
                const float* ub_ptr = nullptr;
                if (config.slow_path_rbq_prefilter_enabled &&
                    !state.rbq_bounds_per_cluster.empty())
                {
                  auto it = std::lower_bound(
                      state.rbq_bounds_per_cluster.begin(),
                      state.rbq_bounds_per_cluster.end(), cluster_id,
                      [](const auto& a, size_t b) {
                        return std::get<0>(a) < b;
                      });
                  if (it != state.rbq_bounds_per_cluster.end() &&
                      std::get<0>(*it) == cluster_id)
                  {
                    const size_t first_cell_global =
                        cluster_reader.cluster_cell_offsets[cluster_id];
                    const size_t cluster_first_vec_global =
                        cluster_reader.cell_vec_offsets[first_cell_global];
                    const size_t cell_idx_local =
                        static_cast<size_t>(cell_id) - first_cell_global;
                    const auto vec_range =
                        cluster_reader.cell_global_vec_range(cluster_id,
                                                              cell_idx_local);
                    const size_t cluster_local_base =
                        vec_range.first - cluster_first_vec_global;
                    lb_ptr = std::get<1>(*it).data() + cluster_local_base;
                    ub_ptr = std::get<2>(*it).data() + cluster_local_base;
                  }
                }
                for (size_t row = 0; row < vec_count; ++row)
                {
                  if (lb_ptr != nullptr)
                  {
                    // NOTE (2026-05-20): RaBitQ 1-bit lb is a *high-probability*
                    // sound bound, not strictly deterministic. PoC measured
                    // ~0.012% lb violation rate per vec on slow-path candidates
                    // (vecs near hit boundary, where RaBitQ noise is tightest).
                    // We short-circuit anyway and pay the small recall loss
                    // (~0.2% on deep1m) for the disk-IO savings. ub-acceptance
                    // is even more probabilistic (~0.05% violation in test);
                    // tracked as a counter only, NOT short-circuited.
                    const float lb = lb_ptr[row];
                    if (lb > radius_sq_f)
                    {
                      if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                      {
                        ++state.q_s_nh;
                      }
                      continue;  // approx NOT-hit (high prob), skip L2
                    }
                    const float ub = ub_ptr[row];
                    if (ub <= radius_sq_f)
                    {
                      if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                      {
                        ++state.q_s_h;  // diagnostic only
                      }
                    }
                    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                    {
                      ++state.q_s_borderline;
                    }
                  }
                  const double dist =
                      exact_l2(state, cluster_id, cell_data, row,
                               cluster_local_row_base);
                  if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                  {
                    ++state.q_dist_comp;
                  }
                  else
                  {
                    ++dist_comp;
                  }
                  if (exact_hit_within_radius(dist))
                  {
                    query_hits[state.q].push_back(
                        HitRef{static_cast<uint32_t>(cluster_id),
                               static_cast<uint32_t>(cell_id),
                               static_cast<uint32_t>(row)});
                    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                    {
                      state.q_hit_dist_sum += dist;
                    }
                  }
                }
                if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                {
                  const auto l2_end = std::chrono::steady_clock::now();
                  state.l2_secs_this_query +=
                      std::chrono::duration<double>(l2_end - l2_start).count();
                  // Per-cell skip detection: if every vec in this cell was
                  // lb-rejected, the cell could be skipped by Phase B/C.
                  if (vec_count > 0 &&
                      state.q_s_nh - base_s_nh == vec_count)
                  {
                    ++state.q_cells_all_rejected;
                  }
                }
              });
        }
        else
        {
          cluster_reader.submit_and_drain(
              omp_get_thread_num(), tl_io_clusters,
              [&](size_t cluster_id, const void* cluster_data,
                  size_t bucket_size, size_t cluster_local_row_base)
            {
              (void)cluster_id;
              std::chrono::steady_clock::time_point l2_start;
              if constexpr (kDiskRangeSearchPhaseTimingEnabled)
              {
                l2_start = std::chrono::steady_clock::now();
                state.q_bytes_read +=
                    bucket_size * config.dim * cluster_reader.bytes_per_element;
              }
              for (size_t row = 0; row < bucket_size; ++row)
              {
                const double dist = exact_l2(state, cluster_id, cluster_data,
                                             row, cluster_local_row_base);
                if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                {
                  ++state.q_dist_comp;
                }
                else
                {
                  ++dist_comp;
                }
                if (exact_hit_within_radius(dist))
                {
                  query_hits[state.q].push_back(
                      HitRef{static_cast<uint32_t>(cluster_id),
                             kHitRefClusterWide,
                             static_cast<uint32_t>(row)});
                  if constexpr (kDiskRangeSearchPhaseTimingEnabled)
                  {
                    state.q_hit_dist_sum += dist;
                  }
                }
              }
              if constexpr (kDiskRangeSearchPhaseTimingEnabled)
              {
                const auto l2_end = std::chrono::steady_clock::now();
                state.l2_secs_this_query +=
                    std::chrono::duration<double>(l2_end - l2_start).count();
              }
            });
        }
        if constexpr (kDiskRangeSearchPhaseTimingEnabled)
        {
          const auto drain_end = std::chrono::steady_clock::now();
          const double drain_wall =
              std::chrono::duration<double>(drain_end - drain_start).count();
          state.q_stats.t_exact_l2 = state.l2_secs_this_query;
          state.q_stats.t_io_wait =
              std::max(0.0, drain_wall - state.l2_secs_this_query);
          dist_comp += state.q_dist_comp;
          state.q_stats.n_queries = 1;
          state.q_stats.candidates = state.candidates.size();
          state.q_stats.queries_with_candidates =
              state.candidates.empty() ? 0 : 1;
          state.q_stats.queries_after_prescreen =
              state.active_candidates.empty() ? 0 : 1;
          state.q_stats.queries_after_cluster_bound =
              state.surviving.empty() ? 0 : 1;
          state.q_stats.queries_after_cell = state.q_dist_comp == 0 ? 0 : 1;
          state.q_stats.surviving = state.surviving.size();
          state.q_stats.cell_kept = state.q_cell_kept;
          state.q_stats.cells_read = state.q_cells_read;
          state.q_stats.bytes_read = state.q_bytes_read;
          state.q_stats.dist_comp = state.q_dist_comp;
          const size_t bucket = query_hits[state.q].empty() ? 0 : 1;
          per_thread_stats[static_cast<size_t>(omp_get_thread_num())][bucket].add(
              state.q_stats);
          SlowPathTraceRecord rec{};
          rec.query_id = state.q;
          rec.skipped_by_fast_path = false;
          rec.t_cell_filter = state.q_stats.t_cell_filter;
          rec.t_io_wait = state.q_stats.t_io_wait;
          rec.t_exact_l2 = state.q_stats.t_exact_l2;
          rec.cell_kept = state.q_cell_kept;
          rec.cells_read = state.q_cells_read;
          rec.bytes_read = state.q_bytes_read;
          rec.dist_comp = state.q_dist_comp;
          rec.candidates = state.candidates.size();
          rec.hit_count = query_hits[state.q].size();
          rec.hit_dist_sum = state.q_hit_dist_sum;
          rec.s_nh = state.q_s_nh;
          rec.s_h = state.q_s_h;
          rec.s_borderline = state.q_s_borderline;
          rec.cells_all_rejected = state.q_cells_all_rejected;
          rec.cells_total = state.q_cells_total;
          rec.cells_skipped_phase_b = state.q_cells_skipped_phase_b;
          slow_path_trace[state.q] = rec;
        }
      }
      if (dj_blog && batch_ord < dj_blog_recs.size())
      {
        DjBatchRec& br = dj_blog_recs[batch_ord];
        br.t0 = dj_blog_t0 - dj_blog_origin;
        br.t1 = omp_get_wtime() - dj_blog_origin;
        br.tid = omp_get_thread_num();
        br.nq = batch_size;
        for (size_t i = 0; i < batch_size; ++i)
        {
          br.cells += batch_state[i].q_cells_read;
          br.io += batch_state[i].q_stats.t_io_wait;
        }
      }
    };  // end process_batch lambda

    if (!dj_fuse)
    {
      // Default dispatch: two-pass -- iterate the precomputed survivor batches.
#pragma omp for schedule(dynamic)
      for (size_t batch_start = 0; batch_start < survivor_qids.size();
           batch_start += kQdotBatchSize)
      {
        if (dj_mk)
        {
          const size_t dj_t = static_cast<size_t>(omp_get_thread_num());
          const double dj_now = omp_get_wtime();
          if (dj_mk_sn[dj_t] == 0)
            dj_mk_first[dj_t] = dj_now;
          else
            dj_mk_sb[dj_t] += dj_now - dj_mk_bw[dj_t];
          dj_mk_bw[dj_t] = dj_now;
          dj_mk_last[dj_t] = dj_now;
          ++dj_mk_sn[dj_t];
        }
        const double t0 = dj_blog ? omp_get_wtime() : 0.0;
        const size_t bs =
            std::min(kQdotBatchSize, survivor_qids.size() - batch_start);
        process_batch(&survivor_qids[batch_start], bs,
                      batch_start / kQdotBatchSize, t0);
      }
    }
    else
    {
      // Fused dispatch: run the emptiness oracle and the slow path in one pass
      // (no barrier between them). Each thread accumulates survivors into a
      // local batch and flushes at kQdotBatchSize (and once more at the end), so
      // the GEMM still amortizes the centroid matrix over a full batch. The
      // per-query results are identical to the two-pass path (a batch only
      // shares the GEMM matrix; all pruning/exact work is per-query), so recall
      // and dist_comp are unchanged -- only batch composition differs.
      std::vector<size_t> pending;
      pending.reserve(kQdotBatchSize);
      const double dj_fuse_t0 = omp_get_wtime();
#pragma omp for schedule(dynamic, dj_fuse_chunk) nowait
      for (size_t q = 0; q < config.n_queries; ++q)
      {
        if (nonempty_only && config.per_query_counts[q] == 0) continue;
        const float* qptr = queries.data() + q * config.dim;
        bool survives = true;
        if (use_fast_path_oracle)
        {
          const auto os = std::chrono::steady_clock::now();
          const FastPathOracleResult r =
              fast_path_oracle->evaluate(qptr, config.radius_squared);
          const auto oe = std::chrono::steady_clock::now();
          fast_path_oracle_total_us +=
              std::chrono::duration<double, std::micro>(oe - os).count();
          fast_path_trace[q] = {q,
                                r.is_empty,
                                r.phase1_candidate_count,
                                r.phase2_scanned_vectors,
                                r.phase2_borderline_count,
                                r.phase3_refined_count,
                                r.min_lower_bound,
                                r.first_nan_field_mask,
                                r.first_nan_cluster_id,
                                r.first_nan_local_index};
          if (r.is_empty)
          {
            survives = false;
            ++fast_path_skip_count;
          }
          else if (dj_overlap)
          {
            dj_ov_oracle_vecs.fetch_add(r.phase2_scanned_vectors,
                                        std::memory_order_relaxed);
          }
        }
        else if (use_data_hnsw_oracle)
        {
          const auto os = std::chrono::steady_clock::now();
          const DataHNSWOracleResult r = data_hnsw_oracle->evaluate(
              qptr, radius_l2_f, config.data_hnsw_margin_delta);
          const auto oe = std::chrono::steady_clock::now();
          data_hnsw_oracle_total_us +=
              std::chrono::duration<double, std::micro>(oe - os).count();
          data_hnsw_trace[q] = {q, r.d_hat_l2, r.is_empty};
          if (r.is_empty)
          {
            survives = false;
            ++data_hnsw_skip_count;
          }
        }
        if (!survives)
        {
          if constexpr (kDiskRangeSearchPhaseTimingEnabled)
          {
            QueryClassStats culled{};
            culled.n_queries = 1;
            per_thread_stats[static_cast<size_t>(omp_get_thread_num())][0].add(
                culled);
            SlowPathTraceRecord rec{};
            rec.query_id = q;
            rec.skipped_by_fast_path = true;
            slow_path_trace[q] = rec;
          }
          continue;
        }
        ++fused_survivor_count;
        pending.push_back(q);
        if (pending.size() == kQdotBatchSize)
        {
          process_batch(pending.data(), pending.size(), SIZE_MAX, 0.0);
          pending.clear();
        }
      }
      if (!pending.empty())
        process_batch(pending.data(), pending.size(), SIZE_MAX, 0.0);
      dj_fuse_span[static_cast<size_t>(omp_get_thread_num())] =
          omp_get_wtime() - dj_fuse_t0;
    }
#pragma omp critical
    {
      *p_dist_comp += dist_comp;
      *p_tcc += triangle_candidate_clusters;
      *p_tpc += triangle_pruned_clusters;
      *p_sopc += slab_only_pruned_clusters;
      *p_popc += pca_only_pruned_clusters;
      *p_cpc += combined_pruned_clusters;
      *p_ccv += cell_candidate_vectors;
      *p_cpv += cell_pruned_vectors;
      *p_ckv += cell_kept_vectors;
      *p_zkc += zero_kept_cells;
      *p_zksc += zero_kept_surviving_clusters;
      *p_cbr += cell_bytes_read;
      *p_cbb += cluster_bytes_baseline;
      *p_rpc += radius_prescreened_clusters;
      *p_st += surviving_total;
      *p_fpsc += fast_path_skip_count;
      *p_fpot += fast_path_oracle_total_us;
      *p_dhsc += data_hnsw_skip_count;
      *p_dhot += data_hnsw_oracle_total_us;
      *p_fsc += fused_survivor_count;
    }
    }  // end #pragma omp parallel (Pass-2)
    if (dj_mk && dj_fuse)
    {
      double mx = 0.0, mn = 1e18, sm = 0.0;
      for (int t = 0; t < dj_mk_N; ++t)
      {
        mx = std::max(mx, dj_fuse_span[static_cast<size_t>(t)]);
        mn = std::min(mn, dj_fuse_span[static_cast<size_t>(t)]);
        sm += dj_fuse_span[static_cast<size_t>(t)];
      }
      const double mean = sm / static_cast<double>(dj_mk_N);
      std::cerr << "[fuse-busy] threads=" << dj_mk_N
                << " makespan(max_span)=" << mx << "s mean=" << mean
                << "s min=" << mn << "s imbalance(max/mean)="
                << (mean > 0.0 ? mx / mean : 0.0)
                << "x  (max_span is the fused-loop wall lower bound; "
                << "gap vs sum/N=" << (mx > 0.0 ? sm / static_cast<double>(dj_mk_N) / mx : 0.0)
                << " util)\n";
    }
    if (dj_overlap)
    {
      const size_t o = dj_ov_oracle_vecs.load();
      const size_t a = dj_ov_phasea_vecs.load();
      const size_t ov = dj_ov_intersect_vecs.load();
      const double fold_phasea =
          a > 0 ? static_cast<double>(ov) / static_cast<double>(a) : 0.0;
      const double reuse_oracle =
          o > 0 ? static_cast<double>(ov) / static_cast<double>(o) : 0.0;
      std::cerr << "[overlap] oracle_1bit_vecs=" << o
                << " phaseA_vecs=" << a << " intersect_vecs=" << ov
                << "  fold_fraction_of_phaseA=" << fold_phasea
                << "  reuse_fraction_of_oracle=" << reuse_oracle
                << "  (ceiling = 1/(1 - fold_fraction_of_phaseA * "
                   "t_rbq_prefilter/total))\n";
    }

    if (dj_blog)
    {
      double seg_end = 0.0, sum_dur = 0.0;
      for (const auto& r : dj_blog_recs)
      {
        seg_end = std::max(seg_end, r.t1);
        sum_dur += (r.t1 - r.t0);
      }
      std::vector<size_t> idx(dj_blog_recs.size());
      for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return (dj_blog_recs[a].t1 - dj_blog_recs[a].t0) >
               (dj_blog_recs[b].t1 - dj_blog_recs[b].t0);
      });
      const double oracle_summed = fast_path_oracle_total_us / 1e6;
      std::cerr << "[batchlog] n_batches=" << dj_blog_recs.size()
                << " segB_end=" << seg_end * 1000 << "ms sum_batch_dur="
                << sum_dur * 1000 << "ms ideal(/24)=" << sum_dur / 24 * 1000
                << "ms\n";
      std::cerr << "[batchlog] oracle_summed(segA work)=" << oracle_summed * 1000
                << "ms ideal(/24)=" << oracle_summed / 24 * 1000
                << "ms  => fused_ideal((oracle+slow)/24)="
                << (oracle_summed + sum_dur) / 24 * 1000 << "ms\n";
      for (size_t k = 0; k < idx.size(); ++k)
      {
        const auto& r = dj_blog_recs[idx[k]];
        std::cerr << "[batchlog] b=" << idx[k] << " tid=" << r.tid
                  << " dur=" << (r.t1 - r.t0) * 1000 << "ms start=" << r.t0 * 1000
                  << " end=" << r.t1 * 1000 << " nq=" << r.nq
                  << " cells=" << r.cells << " io=" << r.io * 1000 << "ms\n";
      }
    }

    if (invalid_candidate.load(std::memory_order_relaxed))
    {
      throw std::runtime_error(
          "SQG candidate cluster id out of range: " +
          std::to_string(invalid_cluster_id.load(std::memory_order_relaxed)) +
          " >= " + std::to_string(cluster_reader.cluster_num));
    }

    const auto end = std::chrono::steady_clock::now();
    cluster_reader.report_cluster_cache_stats();
    cluster_reader.report_cell_cache_stats();

    std::vector<size_t> query_hit_counts(config.n_queries);
    for (size_t q = 0; q < config.n_queries; ++q)
    {
      query_hit_counts[q] = query_hits[q].size();
    }
    const size_t total_hits = std::accumulate(
        query_hit_counts.begin(), query_hit_counts.end(), size_t{0});

    // Post-search L2 verify (outside search timer). Re-reads each hit's cell
    // via cluster_reader and recomputes exact L2 to drop false positives. With
    // the current callback (which already runs exact L2 before accepting a
    // hit) this is a no-op (recall_verified == recall_raw); it provides a
    // safety net for future sound-bound accept paths (e.g. ub-acceptance).
    std::vector<std::vector<uint8_t>> hits_verified(config.n_queries);
    for (size_t q = 0; q < config.n_queries; ++q)
    {
      hits_verified[q].assign(query_hits[q].size(), uint8_t{0});
    }
    std::vector<VerifiedHit> verified_hits;
    verified_hits.reserve(total_hits);
    {
      using CellKey = std::pair<uint32_t, uint32_t>;
      // (q, row, hit_idx)
      using HitLookup = std::tuple<size_t, uint32_t, size_t>;
      std::map<CellKey, std::vector<HitLookup>> cell_lookups;
      std::map<uint32_t, std::vector<HitLookup>> cluster_lookups;
      auto record_verified_hit = [&](size_t q, size_t cluster_id,
                                     size_t cluster_local_row,
                                     float distance) {
        if (cluster_id >= cluster_reader.assignment.size() ||
            cluster_local_row >= cluster_reader.assignment[cluster_id].size())
        {
          throw std::runtime_error(
              "verified hit row out of cluster assignment range");
        }
        verified_hits.push_back(
            VerifiedHit{q,
                        cluster_reader.assignment[cluster_id]
                                                 [cluster_local_row],
                        distance});
      };
      auto verify_l2 = [&](size_t q, size_t cluster_id, const void* payload,
                           size_t row, size_t cluster_local_row_base,
                           size_t* dim_ptr) -> double {
        if (cluster_reader.vec_dtype == range_search_config::kVecDTypeInt8)
        {
          const size_t global_row =
              cluster_reader.cluster_global_vec_offset(cluster_id) +
              cluster_local_row_base + row;
          if (global_row >= cluster_reader.sum_for_offset.size() ||
              global_row >= cluster_reader.norm_sq_for_l2.size())
          {
            throw std::runtime_error(
                "int8 metadata row out of range during hit verification");
          }
          const int8_t* query_ptr =
              reinterpret_cast<const int8_t*>(query_data.raw_bytes.data()) +
              q * config.dim;
          const int8_t* base_ptr =
              static_cast<const int8_t*>(payload) + row * config.dim;
          return utils::L2SqrInt8SIMD(
              query_ptr, base_ptr, config.dim, query_data.norm_sq_byte[q],
              cluster_reader.sum_for_offset[global_row],
              cluster_reader.norm_sq_for_l2[global_row]);
        }
        if (cluster_reader.vec_dtype ==
            range_search_config::kVecDTypeUint8Reserved)
        {
          const size_t global_row =
              cluster_reader.cluster_global_vec_offset(cluster_id) +
              cluster_local_row_base + row;
          if (global_row >= cluster_reader.sum_for_offset.size() ||
              global_row >= cluster_reader.norm_sq_for_l2.size())
          {
            throw std::runtime_error(
                "uint8 metadata row out of range during hit verification");
          }
          const uint8_t* query_ptr =
              query_data.raw_bytes.data() + q * config.dim;
          const uint8_t* base_ptr =
              static_cast<const uint8_t*>(payload) + row * config.dim;
          return static_cast<double>(utils::L2SqrUint8SIMDInt64(
              query_ptr, base_ptr, config.dim, query_data.norm_sq_byte[q],
              cluster_reader.sum_for_offset[global_row],
              cluster_reader.norm_sq_for_l2[global_row]));
        }
        const float* base_ptr =
            static_cast<const float*>(payload) + row * config.dim;
        const float* query_ptr = queries.data() + q * config.dim;
        return utils::L2Sqr(query_ptr, base_ptr, dim_ptr);
      };
      auto verified_hit_within_radius = [&](double dist) -> bool {
        if (cluster_reader.vec_dtype ==
            range_search_config::kVecDTypeUint8Reserved)
        {
          return dist <= config.radius_squared;
        }
        return dist <= radius_sq_f;
      };
      for (size_t q = 0; q < config.n_queries; ++q)
      {
        const auto& hits = query_hits[q];
        for (size_t i = 0; i < hits.size(); ++i)
        {
          const HitRef& h = hits[i];
          if (h.cell_id == kHitRefClusterWide)
          {
            cluster_lookups[h.cluster_id].emplace_back(q, h.row_in_cell, i);
          }
          else
          {
            cell_lookups[{h.cluster_id, h.cell_id}].emplace_back(
                q, h.row_in_cell, i);
          }
        }
      }

      if (!cell_lookups.empty())
      {
        std::map<uint32_t, ClusterReader::IOPlan> per_cluster;
        for (const auto& kv : cell_lookups)
        {
          per_cluster[kv.first.first].push_back(kv.first.second);
        }
        std::vector<std::pair<size_t, ClusterReader::IOPlan>> verify_plans;
        verify_plans.reserve(per_cluster.size());
        for (auto& kv : per_cluster)
        {
          verify_plans.emplace_back(static_cast<size_t>(kv.first),
                                    std::move(kv.second));
        }
        size_t verify_distance_dim = config.dim;
        cluster_reader.submit_and_drain(
            0, verify_plans,
            [&](size_t cluster_id, uint64_t cell_id, const void* cell_data,
                size_t vec_count, size_t cluster_local_row_base)
            {
              auto it = cell_lookups.find(
                  {static_cast<uint32_t>(cluster_id),
                   static_cast<uint32_t>(cell_id)});
              if (it == cell_lookups.end())
              {
                return;
              }
              for (const auto& lookup : it->second)
              {
                const size_t q = std::get<0>(lookup);
                const uint32_t row = std::get<1>(lookup);
                const size_t hit_idx = std::get<2>(lookup);
                if (static_cast<size_t>(row) >= vec_count)
                {
                  continue;
                }
                const double dist =
                    verify_l2(q, cluster_id, cell_data, row,
                              cluster_local_row_base, &verify_distance_dim);
                if (verified_hit_within_radius(dist))
                {
                  hits_verified[q][hit_idx] = 1;
                  record_verified_hit(q, cluster_id,
                                      cluster_local_row_base + row,
                                      static_cast<float>(dist));
                }
              }
            });
      }
      if (!cluster_lookups.empty())
      {
        std::vector<size_t> verify_cluster_ids;
        verify_cluster_ids.reserve(cluster_lookups.size());
        for (const auto& kv : cluster_lookups)
        {
          verify_cluster_ids.push_back(static_cast<size_t>(kv.first));
        }
        size_t verify_distance_dim = config.dim;
        cluster_reader.submit_and_drain(
            0, verify_cluster_ids,
            [&](size_t cluster_id, const void* cluster_data,
                size_t bucket_size, size_t cluster_local_row_base)
            {
              auto it =
                  cluster_lookups.find(static_cast<uint32_t>(cluster_id));
              if (it == cluster_lookups.end())
              {
                return;
              }
              for (const auto& lookup : it->second)
              {
                const size_t q = std::get<0>(lookup);
                const uint32_t row = std::get<1>(lookup);
                const size_t hit_idx = std::get<2>(lookup);
                if (static_cast<size_t>(row) >= bucket_size)
                {
                  continue;
                }
                const double dist =
                    verify_l2(q, cluster_id, cluster_data, row,
                              cluster_local_row_base, &verify_distance_dim);
                if (verified_hit_within_radius(dist))
                {
                  hits_verified[q][hit_idx] = 1;
                  record_verified_hit(q, cluster_id,
                                      cluster_local_row_base + row,
                                      static_cast<float>(dist));
                }
              }
            });
      }
    }
    if (dj_mk)
    {
      const double dj_mk_t2 = omp_get_wtime();
      double mb = 0.0, sb = 0.0, minf = 1e18, maxl = 0.0, maxspan = 0.0, maxlast = 0.0;
      long mn = 0, tn = 0;
      for (int t = 0; t < dj_mk_N; ++t)
      {
        const size_t tt = static_cast<size_t>(t);
        mb = std::max(mb, dj_mk_sb[tt]);
        sb += dj_mk_sb[tt];
        mn = std::max(mn, dj_mk_sn[tt]);
        tn += dj_mk_sn[tt];
        if (dj_mk_sn[tt] > 0)
        {
          minf = std::min(minf, dj_mk_first[tt]);
          maxl = std::max(maxl, dj_mk_last[tt]);
          maxspan = std::max(maxspan, dj_mk_last[tt] - dj_mk_first[tt]);
          maxlast = std::max(maxlast, dj_mk_t2 - dj_mk_bw[tt]);
        }
      }
      std::cerr << "[makespan] segA_fast+build=" << (dj_mk_t1 - dj_mk_t0)
                << "s  segB_slow=" << (dj_mk_t2 - dj_mk_t1) << "s  (wall_total="
                << (dj_mk_t2 - dj_mk_t0) << "s)\n";
      std::cerr << "[makespan] segB busy~sum=" << sb << " (misses each thread's"
                << " last batch)  ideal=" << (sb / dj_mk_N) << "  max_busy~="
                << mb << "  busy_imbalance=" << (mb / (sb / dj_mk_N + 1e-12))
                << "x\n";
      std::cerr << "[makespan] segB: schedule_spread(maxL-minF)=" << (maxl - minf)
                << "  tail_batch(t2-maxL)=" << (dj_mk_t2 - maxl)
                << " (=makespan lower bound)  total_batches=" << tn << "\n";
      double ssec1 = 0.0, ssec2 = 0.0;
      for (int t = 0; t < dj_mk_N; ++t)
      {
        ssec1 += dj_sec1[static_cast<size_t>(t)];
        ssec2 += dj_sec2[static_cast<size_t>(t)];
      }
      std::cerr << "[makespan] thread-summed wall sections: sec1_candidate_loop="
                << ssec1 << "  sec2_compute_qdot=" << ssec2
                << "  (vs phase t_candidate_gen, t_centroid_dot below)\n";
      for (int t = 0; t < dj_mk_N; ++t)
      {
        const size_t tt = static_cast<size_t>(t);
        QueryClassStats ps = per_thread_stats[tt][0];
        ps.add(per_thread_stats[tt][1]);
        const double tw =
            dj_mk_sn[tt] > 0 ? dj_mk_sb[tt] + (dj_mk_t2 - dj_mk_bw[tt]) : 0.0;
        const double psum = ps.t_io_wait + ps.t_centroid_dot +
                            ps.t_cluster_bound + ps.t_cell_filter +
                            ps.t_exact_l2 + ps.t_candidate_gen +
                            ps.t_rbq_prefilter;
        std::cerr << "[makespan]   tid=" << t << " b=" << dj_mk_sn[tt]
                  << " WALL=" << tw << " PHASE_SUM=" << psum
                  << " GAP=" << (tw - psum) << " | io=" << ps.t_io_wait
                  << " cdot=" << ps.t_centroid_dot << " clb=" << ps.t_cluster_bound
                  << " celf=" << ps.t_cell_filter << " l2=" << ps.t_exact_l2
                  << " cand=" << ps.t_candidate_gen
                  << " RBQ_PREFILT=" << ps.t_rbq_prefilter << "\n";
      }
    }
    size_t total_hits_verified = 0;
    for (size_t q = 0; q < config.n_queries; ++q)
    {
      for (uint8_t v : hits_verified[q])
      {
        if (v != 0)
        {
          ++total_hits_verified;
        }
      }
    }

    std::vector<size_t> sorted_counts = query_hit_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());

    RunResult result;
    result.recall_raw =
        total_hits / static_cast<double>(config.total_in_range);
    result.recall_verified =
        total_hits_verified / static_cast<double>(config.total_in_range);
    result.recall = result.recall_raw;  // backward compat
    const auto prune_rate = [triangle_candidate_clusters](size_t pruned) {
      return triangle_candidate_clusters == 0
                 ? 0.0
                 : pruned / static_cast<double>(triangle_candidate_clusters);
    };
    result.triangle_prune_rate = prune_rate(triangle_pruned_clusters);
    result.triangle_candidate_clusters = triangle_candidate_clusters;
    result.triangle_pruned_clusters = triangle_pruned_clusters;
    result.radius_prescreened_clusters = radius_prescreened_clusters;
    result.radius_prescreen_rate = prune_rate(radius_prescreened_clusters);
    result.slab_pruned_clusters =
        cluster_reader.slab_pruned_clusters.load(std::memory_order_relaxed);
    result.box_residual_pruned_clusters =
        cluster_reader.box_residual_pruned_clusters.load(
            std::memory_order_relaxed);
    result.slab_only_pruned_clusters = slab_only_pruned_clusters;
    result.pca_only_pruned_clusters = pca_only_pruned_clusters;
    result.combined_pruned_clusters = combined_pruned_clusters;
    result.slab_only_prune_rate = prune_rate(result.slab_only_pruned_clusters);
    result.pca_only_prune_rate = prune_rate(result.pca_only_pruned_clusters);
    result.combined_prune_rate = prune_rate(result.combined_pruned_clusters);
    result.cell_candidate_vectors = cell_candidate_vectors;
    result.cell_pruned_vectors = cell_pruned_vectors;
    result.cell_kept_vectors = cell_kept_vectors;
    result.zero_kept_cells = zero_kept_cells;
    result.zero_kept_surviving_clusters = zero_kept_surviving_clusters;
    result.cell_bytes_read = cell_bytes_read;
    result.cluster_bytes_baseline = cluster_bytes_baseline;
    result.cell_metadata_bytes =
        cluster_reader.cluster_cell_offsets.size() * sizeof(uint64_t) +
        cluster_reader.cell_vec_offsets.size() * sizeof(uint64_t) +
        cluster_reader.cell_bounds.size() * sizeof(float);
    result.cell_prune_rate =
        cell_candidate_vectors == 0
            ? 0.0
            : cell_pruned_vectors / static_cast<double>(cell_candidate_vectors);
    result.io_bytes_reduction =
        cluster_bytes_baseline == 0
            ? 0.0
            : 1.0 -
                  cell_bytes_read / static_cast<double>(cluster_bytes_baseline);
    if (result.slab_pruned_clusters +
            result.box_residual_pruned_clusters !=
        result.triangle_pruned_clusters)
    {
      throw std::runtime_error(
          "prune counter partition mismatch: slab_pruned_clusters=" +
          std::to_string(result.slab_pruned_clusters) +
          ", box_residual_pruned_clusters=" +
          std::to_string(result.box_residual_pruned_clusters) +
          ", triangle_pruned_clusters=" +
          std::to_string(result.triangle_pruned_clusters));
    }
    const bool candidate_partition_ok =
        triangle_candidate_clusters == radius_prescreened_clusters +
                                           triangle_pruned_clusters +
                                           surviving_total;
    if (!candidate_partition_ok)
    {
      throw std::runtime_error(
          "candidate partition mismatch: triangle_candidate_clusters=" +
          std::to_string(triangle_candidate_clusters) +
          ", radius_prescreened_clusters=" +
          std::to_string(radius_prescreened_clusters) +
          ", triangle_pruned_clusters=" +
          std::to_string(triangle_pruned_clusters) +
          ", surviving_total=" + std::to_string(surviving_total));
    }
    result.mean_keep_after_filter =
        config.n_queries == 0
            ? 0.0
            : static_cast<double>(surviving_total) /
                  static_cast<double>(config.n_queries);
    result.build_time = build_time_;
    result.search_time = std::chrono::duration<double>(end - start).count();
    // QPS denominator: under DJ_NONEMPTY_ONLY, only the non-empty survivors
    // were actually searched, so divide by that count instead of n_queries.
    const size_t qps_denom =
        nonempty_only ? (dj_fuse ? fused_survivor_count : survivor_qids.size())
                      : config.n_queries;
    result.qps = result.search_time > 0.0
                     ? static_cast<double>(qps_denom) / result.search_time
                     : 0.0;
    result.dist_comp = dist_comp;
    result.data_hnsw_skip_count = data_hnsw_skip_count;
    result.data_hnsw_oracle_total_us = data_hnsw_oracle_total_us;
    result.data_hnsw_skip_rate =
        config.n_queries == 0
            ? 0.0
            : data_hnsw_skip_count / static_cast<double>(config.n_queries);
    result.data_hnsw_trace = std::move(data_hnsw_trace);
    result.data_hnsw_trace_path = config.data_hnsw_trace_path;
    result.verified_hits = std::move(verified_hits);
    result.fast_path_skip_count = fast_path_skip_count;
    result.fast_path_oracle_total_us = fast_path_oracle_total_us;
    result.fast_path_skip_rate =
        config.n_queries == 0
            ? 0.0
            : fast_path_skip_count / static_cast<double>(config.n_queries);
    if (use_fast_path_oracle)
    {
      size_t zero_result_queries = 0;
      size_t zero_result_short_circuits = 0;
      for (size_t q = 0; q < config.per_query_counts.size(); ++q)
      {
        const size_t baseline_hits = config.per_query_counts[q];
        const bool oracle_empty = fast_path_trace[q].oracle_decision;
        if (baseline_hits > 0)
        {
          ++result.fast_path_ground_truth_nonempty_queries;
          if (oracle_empty)
          {
            ++result.fast_path_false_empty_queries;
            result.fast_path_suppressed_baseline_hits += baseline_hits;
          }
        }
        else
        {
          ++zero_result_queries;
          if (oracle_empty)
          {
            ++zero_result_short_circuits;
          }
        }
      }
      result.false_empty_query_rate =
          result.fast_path_ground_truth_nonempty_queries == 0
              ? 0.0
              : result.fast_path_false_empty_queries /
                    static_cast<double>(
                        result.fast_path_ground_truth_nonempty_queries);
      result.range_recall_loss_hit_weighted =
          config.total_in_range == 0
              ? 0.0
              : result.fast_path_suppressed_baseline_hits /
                    static_cast<double>(config.total_in_range);
      result.zero_result_short_circuit_rate =
          zero_result_queries == 0
              ? 0.0
              : zero_result_short_circuits /
                    static_cast<double>(zero_result_queries);
      result.fast_path_trace = std::move(fast_path_trace);
      const std::filesystem::path code_path(config.sqg_path);
      result.fast_path_trace_path =
          (code_path.parent_path() / "_fast_path_trace.csv").string();
    }
    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
    {
      result.slow_path_trace = std::move(slow_path_trace);
      const std::filesystem::path cluster_dir(config.cluster_path);
      result.slow_path_trace_path =
          (cluster_dir.parent_path() / "_slow_path_trace.csv").string();
    }
    result.per_query_min = sorted_counts.empty() ? 0 : sorted_counts.front();
    result.per_query_max = sorted_counts.empty() ? 0 : sorted_counts.back();
    result.per_query_zero = static_cast<size_t>(std::count(
        query_hit_counts.begin(), query_hit_counts.end(), size_t{0}));
    result.per_query_median =
        sorted_counts.empty() ? 0.0
                              : (sorted_counts[(sorted_counts.size() - 1) / 2] +
                                 sorted_counts[sorted_counts.size() / 2]) /
                                    2.0;
    if constexpr (kDiskRangeSearchPhaseTimingEnabled)
    {
      result.n_threads = per_thread_stats.size();
      for (const auto& thread_stats : per_thread_stats)
      {
        result.zero_result.add(thread_stats[0]);
        result.nonzero_result.add(thread_stats[1]);
      }
    }
    return result;
  }

 private:
  struct MetadataHeader
  {
    size_t n = 0;
    size_t d = 0;
    size_t cluster_num = 0;
    size_t max_points = 0;
    size_t pca_rank = 0;
    size_t target_cell_vecs = 24;
    uint32_t vec_dtype = range_search_config::kVecDTypeFloat32;
    size_t slab_count = 0;
  };

  struct QueryData
  {
    std::vector<float> as_float;
    std::vector<uint8_t> raw_bytes;
    std::vector<int32_t> norm_sq_byte;
  };

  double build_time_ = 0.0;

  static void build_fast_path_artifacts(const ResolvedConfig& config)
  {
    if (config.fast_path_nprobe_N > config.cluster_num)
    {
      throw std::runtime_error(
          "fast_path_nprobe_N must be <= cluster_num when fast_path_enabled=true");
    }

    // Build-time artifact generation keeps the historical build mem_budget
    // semantics; search-time resident memory enforcement is not applied here.
    ClusterReader cluster_reader(config.cluster_path, config.metadata_path,
                                 config.io_queue_depth,
                                 ClusterReader::MemBudgetInputs{}, 1);
    {
      BuildPhaseTimer _t("fp.readMetaData");
      cluster_reader.readMetaData();
    }
    if (cluster_reader.cluster_num != config.cluster_num ||
        cluster_reader.d != config.dim)
    {
      throw std::runtime_error("metadata mismatch while building fast-path artifacts");
    }

    SQGCentroidIndex sqg;
    {
      BuildPhaseTimer _t("fp.sqg_load+retune");
      sqg.load(config.sqg_path, config.cluster_num, config.dim, config.sqg_m,
               config.sqg_ef_search);
    }
    if (config.fast_path_mode == "exact")
    {
      return;
    }

    RBQCodeStorage one_bit;
    {
      BuildPhaseTimer _t("fp.rbq_1bit");
      one_bit.build_from_cluster_file(config.cluster_path,
                                      cluster_reader.assignment,
                                      cluster_reader.centroids, config.dim,
                                      sqg.rotator(), 0, config.vec_dtype);
      one_bit.save(config.rabitq_1bit_codes_path);
    }

    RBQCodeStorage eight_bit;
    {
      BuildPhaseTimer _t("fp.rbq_8bit");
      eight_bit.build_from_cluster_file(
          config.cluster_path, cluster_reader.assignment,
          cluster_reader.centroids, config.dim, sqg.rotator(), 7,
          config.vec_dtype);
      eight_bit.save(config.rabitq_8bit_codes_path);
    }
  }

  static void cast_raw_byte_vector_to_float(uint32_t vec_dtype,
                                            const uint8_t* raw,
                                            size_t count, float* out)
  {
    if (vec_dtype == range_search_config::kVecDTypeInt8)
    {
      utils::cast_int8_to_float(reinterpret_cast<const int8_t*>(raw), count,
                                out);
      return;
    }
    if (vec_dtype == range_search_config::kVecDTypeUint8Reserved)
    {
      utils::cast_uint8_to_float(raw, count, out);
      return;
    }
    throw std::runtime_error("cast_raw_byte_vector_to_float requires byte dtype");
  }

  static void build_and_save_data_hnsw(const ResolvedConfig& config)
  {
    DataReader data_reader(config.data_file, config.vec_dtype);
    if (data_reader.n != config.n_base || data_reader.d != config.dim)
    {
      throw std::runtime_error("data .fbin header mismatch for " +
                               config.data_file);
    }
    const uintmax_t value_count_u =
        checked_multiply(config.n_base, config.dim, config.data_file);
    if (value_count_u > std::numeric_limits<size_t>::max())
    {
      throw std::runtime_error("data_hnsw vector payload is too large: " +
                               config.data_file);
    }
    const size_t value_count = static_cast<size_t>(value_count_u);
    if (range_search_config::is_byte_dtype(data_reader.dtype))
    {
      const uintmax_t float_scratch_bytes =
          checked_multiply(value_count_u, sizeof(float), config.data_file);
      const long double mem_budget_bytes =
          static_cast<long double>(config.mem_budget) *
          static_cast<long double>(1ULL << 30);
      if (!std::isfinite(mem_budget_bytes) || mem_budget_bytes <= 0.0L)
      {
        throw std::runtime_error("invalid mem_budget for data_hnsw byte-dtype "
                                 "float scratch preflight: " +
                                 std::to_string(config.mem_budget));
      }
      if (static_cast<long double>(float_scratch_bytes) >
          mem_budget_bytes * 0.8L)
      {
        throw std::runtime_error(
            "data_hnsw byte-dtype float scratch exceeds memory budget: float_scratch_bytes=" +
            std::to_string(float_scratch_bytes) +
            ", mem_budget_gb=" + std::to_string(config.mem_budget) +
            "; disable data_hnsw_enabled, increase mem_budget, or use fast_path_mode=rbq");
      }
      std::vector<float> data(value_count);
      std::vector<uint8_t> raw(value_count);
      data_reader.get_batch(reinterpret_cast<char*>(raw.data()), config.n_base);
      cast_raw_byte_vector_to_float(data_reader.dtype, raw.data(), raw.size(),
                                    data.data());
      if (!data_reader.in)
      {
        throw std::runtime_error("data .fbin file is shorter than expected: " +
                                 config.data_file);
      }
      auto oracle = build_data_hnsw(data.data(), config.n_base, config.dim,
                                    config.data_hnsw_M,
                                    config.data_hnsw_ef_construction);
      save_data_hnsw(*oracle->graph, config.data_hnsw_path);
      return;
    }
    std::vector<float> data(value_count);
    data_reader.get_batch(reinterpret_cast<char*>(data.data()), config.n_base);
    if (!data_reader.in)
    {
      throw std::runtime_error("data .fbin file is shorter than expected: " +
                               config.data_file);
    }
    auto oracle = build_data_hnsw(data.data(), config.n_base, config.dim,
                                  config.data_hnsw_M,
                                  config.data_hnsw_ef_construction);
    save_data_hnsw(*oracle->graph, config.data_hnsw_path);
  }

  static void validate_query_header(const ResolvedConfig& config,
                                    uint32_t metadata_vec_dtype)
  {
    const range_search_config::FbinHeader header =
        range_search_config::read_fbin_header(config.query_file,
                                              metadata_vec_dtype);

    if (header.n != config.n_queries || header.dim != config.dim)
    {
      throw std::runtime_error(
          "query .fbin header mismatch for " + config.query_file +
          ": expected (" + std::to_string(config.n_queries) + ", " +
          std::to_string(config.dim) + "), actual (" +
          std::to_string(header.n) + ", " + std::to_string(header.dim) + ")");
    }
    if (header.dtype != metadata_vec_dtype)
    {
      throw std::runtime_error(
          "query dtype != metadata.vec_dtype for " + config.query_file +
          ": query dtype=" + std::to_string(header.dtype) +
          ", metadata.vec_dtype=" + std::to_string(metadata_vec_dtype));
    }
  }

  static QueryData read_query_data(const ResolvedConfig& config,
                                   uint32_t metadata_vec_dtype)
  {
    std::ifstream in(config.query_file, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open query .fbin file: " +
                               config.query_file);
    }

    const range_search_config::FbinHeader header =
        range_search_config::read_fbin_header(config.query_file,
                                              metadata_vec_dtype);
    if (header.dtype != metadata_vec_dtype)
    {
      throw std::runtime_error(
          "query dtype != metadata.vec_dtype while reading query data");
    }
    in.seekg(static_cast<std::streamoff>(header.payload_offset), std::ios::beg);
    const size_t expected_values = config.n_queries * config.dim;
    QueryData query_data;
    query_data.as_float.resize(expected_values);
    if (range_search_config::is_byte_dtype(metadata_vec_dtype))
    {
      query_data.raw_bytes.resize(expected_values);
      query_data.norm_sq_byte.assign(config.n_queries, 0);
      in.read(reinterpret_cast<char*>(query_data.raw_bytes.data()),
              static_cast<std::streamsize>(expected_values));
      if (in.gcount() !=
          static_cast<std::streamsize>(expected_values))
      {
        throw std::runtime_error("query .fbin file is shorter than expected: " +
                                 config.query_file);
      }
      cast_raw_byte_vector_to_float(metadata_vec_dtype,
                                    query_data.raw_bytes.data(),
                                    expected_values,
                                    query_data.as_float.data());
      for (size_t q = 0; q < config.n_queries; ++q)
      {
        int64_t norm = 0;
        const uint8_t* query = query_data.raw_bytes.data() + q * config.dim;
        for (size_t col = 0; col < config.dim; ++col)
        {
          const int64_t value =
              metadata_vec_dtype == range_search_config::kVecDTypeInt8
                  ? static_cast<int64_t>(
                        reinterpret_cast<const int8_t*>(query)[col])
                  : static_cast<int64_t>(query[col]);
          norm += value * value;
        }
        query_data.norm_sq_byte[q] = static_cast<int32_t>(norm);
      }
      return query_data;
    }
    in.read(reinterpret_cast<char*>(query_data.as_float.data()),
            static_cast<std::streamsize>(expected_values * sizeof(float)));
    if (in.gcount() !=
        static_cast<std::streamsize>(expected_values * sizeof(float)))
    {
      throw std::runtime_error("query .fbin file is shorter than expected: " +
                               config.query_file);
    }
    return query_data;
  }

  static uintmax_t require_file_size(const std::string& path,
                                     const std::string& description)
  {
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec)
    {
      throw std::runtime_error("failed to open " + description + ": " + path +
                               ": " + ec.message());
    }
    return size;
  }

  static MetadataHeader read_metadata_header(const ResolvedConfig& config)
  {
    std::ifstream in(config.metadata_path, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open metadata file: " +
                               config.metadata_path);
    }

    MetadataHeader header;
    std::array<char, 8> observed_magic{};
    in.read(observed_magic.data(),
            static_cast<std::streamsize>(observed_magic.size()));
    if (!in || observed_magic != kMetadataMagic)
    {
      throw std::runtime_error(
          "incompatible metadata schema for " + config.metadata_path +
          ": observed magic " + format_metadata_magic(observed_magic) +
          ", expected DJMETA\\0\\0 schema version " +
          std::to_string(kMetadataSchemaVersion) + ". Delete the index file at " +
          config.metadata_path + " and rebuild with the current binary.");
    }

    uint64_t schema_version = 0;
    in.read(reinterpret_cast<char*>(&schema_version), sizeof(uint64_t));
    if (!in || schema_version != kMetadataSchemaVersion)
    {
      throw std::runtime_error(
          "unsupported metadata schema version for " + config.metadata_path +
          ": observed " + std::to_string(schema_version) + ", expected " +
          std::to_string(kMetadataSchemaVersion) + ". Delete the index file at " +
          config.metadata_path + " and rebuild with the current binary.");
    }

    in.read(reinterpret_cast<char*>(&header.n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&header.d), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&header.cluster_num), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&header.max_points), sizeof(size_t));
    uint64_t metadata_pca_rank = 0;
    in.read(reinterpret_cast<char*>(&metadata_pca_rank), sizeof(uint64_t));
    header.pca_rank = static_cast<size_t>(metadata_pca_rank);
    uint64_t metadata_target_cell_vecs = 0;
    in.read(reinterpret_cast<char*>(&metadata_target_cell_vecs),
            sizeof(uint64_t));
    header.target_cell_vecs =
        static_cast<size_t>(metadata_target_cell_vecs);
    uint64_t metadata_vec_dtype = 0;
    in.read(reinterpret_cast<char*>(&metadata_vec_dtype), sizeof(uint64_t));
    header.vec_dtype = static_cast<uint32_t>(metadata_vec_dtype);
    if (!in)
    {
      throw std::runtime_error("failed to read metadata header: " +
                               config.metadata_path);
    }
    return header;
  }

  static void validate_metadata_header(const ResolvedConfig& config,
                                       const MetadataHeader& header)
  {
    if (header.n != config.n_base)
    {
      throw std::runtime_error("metadata base count mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(config.n_base) + ", actual " +
                               std::to_string(header.n));
    }
    if (header.d != config.dim)
    {
      throw std::runtime_error(
          "metadata dim mismatch for " + config.metadata_path + ": expected " +
          std::to_string(config.dim) + ", actual " + std::to_string(header.d));
    }
    if (header.cluster_num != config.cluster_num)
    {
      throw std::runtime_error(
          "metadata cluster count mismatch for " + config.metadata_path +
          ": expected " + std::to_string(config.cluster_num) + ", actual " +
          std::to_string(header.cluster_num));
    }
    if (header.vec_dtype != config.vec_dtype)
    {
      throw std::runtime_error(
          "metadata.vec_dtype (" + std::to_string(header.vec_dtype) +
          ") != config.vec_dtype (" + std::to_string(config.vec_dtype) +
          ") for " + config.metadata_path +
          "; index and config are out of sync; rebuild the index or restore config");
    }
    if (!range_search_config::is_supported_vec_dtype(header.vec_dtype))
    {
      throw std::runtime_error("metadata vec_dtype unsupported for " +
                               config.metadata_path + ": " +
                               std::to_string(header.vec_dtype));
    }
    if (header.max_points == 0 || header.max_points > config.n_base)
    {
      throw std::runtime_error("metadata max_points out of range for " +
                               config.metadata_path + ": " +
                               std::to_string(header.max_points));
    }
    if (header.pca_rank > header.d)
    {
      throw std::runtime_error("metadata pca_rank out of range for " +
                               config.metadata_path + ": " +
                               std::to_string(header.pca_rank) + " > dim " +
                               std::to_string(header.d));
    }
    if (header.pca_rank != config.pca_rank)
    {
      std::cerr << "warning: metadata pca_rank " << header.pca_rank
                << " differs from config pca_rank " << config.pca_rank
                << "; using metadata rank" << std::endl;
    }
    if (header.target_cell_vecs < 2)
    {
      throw std::runtime_error("metadata target_cell_vecs out of range for " +
                               config.metadata_path + ": " +
                               std::to_string(header.target_cell_vecs));
    }
    if (header.target_cell_vecs != config.target_cell_vecs)
    {
      throw std::runtime_error(
          "metadata target_cell_vecs mismatch for " + config.metadata_path +
          ": metadata target_cell_vecs=" +
          std::to_string(header.target_cell_vecs) +
          "; config target_cell_vecs=" +
          std::to_string(config.target_cell_vecs));
    }
  }

  static std::vector<size_t> read_metadata_bucket_sizes(
      const ResolvedConfig& config, const MetadataHeader& header)
  {
    std::ifstream in(config.metadata_path, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open metadata file: " +
                               config.metadata_path);
    }

    in.seekg(static_cast<std::streamoff>(metadata_header_bytes()),
             std::ios::beg);
    std::vector<size_t> bucket_sizes(header.cluster_num);
    in.read(reinterpret_cast<char*>(bucket_sizes.data()),
            static_cast<std::streamsize>(bucket_sizes.size() * sizeof(size_t)));
    if (!in)
    {
      throw std::runtime_error("failed to read metadata bucket sizes: " +
                               config.metadata_path);
    }
    return bucket_sizes;
  }

  static size_t read_metadata_slab_count(
      const ResolvedConfig& config, const MetadataHeader& header,
      const std::vector<size_t>& bucket_sizes)
  {
    std::ifstream in(config.metadata_path, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open metadata file: " +
                               config.metadata_path);
    }

    uintmax_t offset =
        metadata_header_bytes();
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(size_t),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num, header.d,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(
        offset,
        checked_multiply(checked_multiply(header.cluster_num, header.pca_rank,
                                          config.metadata_path),
                         checked_multiply(header.d, sizeof(float),
                                          config.metadata_path),
                         config.metadata_path),
        config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(uint32_t),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(
        offset,
        checked_multiply(
            std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), uintmax_t{0}),
            sizeof(size_t), config.metadata_path),
        config.metadata_path);

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    uint64_t slab_count = 0;
    in.read(reinterpret_cast<char*>(&slab_count), sizeof(uint64_t));
    if (!in)
    {
      throw std::runtime_error("failed to read metadata slab_count: " +
                               config.metadata_path);
    }
    return static_cast<size_t>(slab_count);
  }

  static void validate_metadata_layout(const ResolvedConfig& config,
                                       const MetadataHeader& header,
                                       const std::vector<size_t>& bucket_sizes,
                                       uintmax_t metadata_file_size)
  {
    uintmax_t bucket_sum = 0;
    size_t max_bucket_size = 0;
    for (size_t i = 0; i < bucket_sizes.size(); ++i)
    {
      if (bucket_sizes[i] > header.max_points)
      {
        throw std::runtime_error("metadata bucket size out of range for " +
                                 config.metadata_path + " at cluster " +
                                 std::to_string(i) + ": " +
                                 std::to_string(bucket_sizes[i]) + " > " +
                                 std::to_string(header.max_points));
      }
      max_bucket_size = std::max(max_bucket_size, bucket_sizes[i]);
      checked_add(bucket_sum, bucket_sizes[i], config.metadata_path);
    }

    if (header.max_points != max_bucket_size)
    {
      throw std::runtime_error("metadata max_points mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(max_bucket_size) + ", actual " +
                               std::to_string(header.max_points));
    }

    if (bucket_sum != config.n_base)
    {
      throw std::runtime_error("metadata bucket size sum mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(config.n_base) + ", actual " +
                               std::to_string(bucket_sum));
    }

    uintmax_t expected_size =
        metadata_header_bytes();
    checked_add(expected_size,
                checked_multiply(header.cluster_num, sizeof(size_t),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(expected_size,
                checked_multiply(checked_multiply(header.cluster_num, header.d,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(expected_size,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(
        expected_size,
        checked_multiply(checked_multiply(header.cluster_num, header.pca_rank,
                                          config.metadata_path),
                         checked_multiply(header.d, sizeof(float),
                                          config.metadata_path),
                         config.metadata_path),
        config.metadata_path);
    checked_add(expected_size,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(expected_size,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(expected_size,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(expected_size,
                checked_multiply(header.cluster_num, sizeof(uint32_t),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(
        expected_size,
        checked_multiply(bucket_sum, sizeof(size_t), config.metadata_path),
        config.metadata_path);
    checked_add(expected_size, sizeof(uint64_t), config.metadata_path);
    if (header.slab_count > 0)
    {
      checked_add(
          expected_size,
          checked_multiply(checked_multiply(header.cluster_num, header.slab_count,
                                            config.metadata_path),
                           sizeof(uint32_t), config.metadata_path),
          config.metadata_path);
      for (size_t i = 0; i < 3; ++i)
      {
        checked_add(
            expected_size,
            checked_multiply(
                checked_multiply(header.cluster_num, header.slab_count,
                                 config.metadata_path),
                sizeof(float), config.metadata_path),
            config.metadata_path);
      }
    }

    align_metadata_offset(expected_size, config.metadata_path);
    const std::vector<uint32_t> effective_ranks =
        read_metadata_effective_ranks(config, header, bucket_sum);
    const std::vector<uint64_t> cluster_cell_offsets = read_metadata_u64_array_at(
        config.metadata_path, expected_size, header.cluster_num + 1,
        "cluster_cell_offsets");
    validate_metadata_cluster_cell_offsets(config, header, bucket_sizes,
                                           cluster_cell_offsets);
    checked_add(expected_size,
                checked_multiply(header.cluster_num + 1, sizeof(uint64_t),
                                 config.metadata_path),
                config.metadata_path);
    const uint64_t total_cells = cluster_cell_offsets.back();
    align_metadata_offset(expected_size, config.metadata_path);
    const std::vector<uint64_t> cell_vec_offsets = read_metadata_u64_array_at(
        config.metadata_path, expected_size,
        static_cast<size_t>(total_cells) + 1, "cell_vec_offsets");
    validate_metadata_cell_vec_offsets(config, header, bucket_sizes,
                                       cluster_cell_offsets, cell_vec_offsets);
    checked_add(expected_size,
                checked_multiply(total_cells + 1, sizeof(uint64_t),
                                 config.metadata_path),
                config.metadata_path);
    align_metadata_offset(expected_size, config.metadata_path);
    uintmax_t bound_floats = 0;
    for (size_t cluster_id = 0; cluster_id < header.cluster_num; ++cluster_id)
    {
      const uint64_t cell_count =
          cluster_cell_offsets[cluster_id + 1] -
          cluster_cell_offsets[cluster_id];
      const uintmax_t cell_bound_floats =
          checked_multiply(cell_count,
                           2 * static_cast<uintmax_t>(
                                   effective_ranks[cluster_id]) +
                               2,
                           config.metadata_path);
      checked_add(bound_floats, cell_bound_floats, config.metadata_path);
    }
    checked_add(expected_size,
                checked_multiply(bound_floats, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    if (range_search_config::is_byte_dtype(header.vec_dtype))
    {
      checked_add(expected_size,
                  checked_multiply(bucket_sum, sizeof(int32_t),
                                   config.metadata_path),
                  config.metadata_path);
      checked_add(expected_size,
                  checked_multiply(bucket_sum, sizeof(int32_t),
                                   config.metadata_path),
                  config.metadata_path);
    }

    if (metadata_file_size != expected_size)
    {
      throw std::runtime_error("metadata file size mismatch for " +
                               config.metadata_path + ": expected " +
                               std::to_string(expected_size) + ", actual " +
                               std::to_string(metadata_file_size));
    }
  }

  static uintmax_t checked_multiply(uintmax_t left, uintmax_t right,
                                    const std::string& path)
  {
    if (right != 0 && left > std::numeric_limits<uintmax_t>::max() / right)
    {
      throw std::runtime_error("size values are too large: " + path);
    }
    return left * right;
  }

  static void checked_add(uintmax_t& target, uintmax_t value,
                          const std::string& path)
  {
    if (target > std::numeric_limits<uintmax_t>::max() - value)
    {
      throw std::runtime_error("size values are too large: " + path);
    }
    target += value;
  }

  static void align_metadata_offset(uintmax_t& target, const std::string& path)
  {
    const uintmax_t remainder = target % 64;
    if (remainder == 0)
    {
      return;
    }
    checked_add(target, 64 - remainder, path);
  }

  static constexpr uintmax_t metadata_header_bytes()
  {
    return kMetadataMagic.size() + sizeof(uint64_t) + 4 * sizeof(size_t) +
           3 * sizeof(uint64_t);
  }

  static std::vector<uint64_t> read_metadata_u64_array_at(
      const std::string& path, uintmax_t offset, size_t count,
      const std::string& label)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open metadata file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint64_t> values(count);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(uint64_t)));
    if (!in)
    {
      throw std::runtime_error("failed to read metadata " + label + ": " +
                               path);
    }
    return values;
  }

  static std::vector<uint32_t> read_metadata_effective_ranks(
      const ResolvedConfig& config, const MetadataHeader& header,
      uintmax_t bucket_sum)
  {
    uintmax_t offset =
        metadata_header_bytes();
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(size_t),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num, header.d,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);
    checked_add(
        offset,
        checked_multiply(checked_multiply(header.cluster_num, header.pca_rank,
                                          config.metadata_path),
                         checked_multiply(header.d, sizeof(float),
                                          config.metadata_path),
                         config.metadata_path),
        config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(checked_multiply(header.cluster_num,
                                                  header.pca_rank,
                                                  config.metadata_path),
                                 sizeof(float), config.metadata_path),
                config.metadata_path);
    checked_add(offset,
                checked_multiply(header.cluster_num, sizeof(float),
                                 config.metadata_path),
                config.metadata_path);

    std::ifstream in(config.metadata_path, std::ios::binary);
    if (!in)
    {
      throw std::runtime_error("failed to open metadata file: " +
                               config.metadata_path);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint32_t> ranks(header.cluster_num);
    in.read(reinterpret_cast<char*>(ranks.data()),
            static_cast<std::streamsize>(ranks.size() * sizeof(uint32_t)));
    if (!in)
    {
      throw std::runtime_error("failed to read metadata pca_effective_rank: " +
                               config.metadata_path);
    }
    (void)bucket_sum;
    return ranks;
  }

  static void validate_metadata_cluster_cell_offsets(
      const ResolvedConfig& config, const MetadataHeader& header,
      const std::vector<size_t>& bucket_sizes,
      const std::vector<uint64_t>& offsets)
  {
    if (offsets.size() != header.cluster_num + 1)
    {
      throw std::runtime_error(
          "metadata cluster_cell_offsets size mismatch for " +
          config.metadata_path);
    }
    if (offsets.empty() || offsets[0] != 0)
    {
      throw std::runtime_error("metadata cluster_cell_offsets must start at zero for " +
                               config.metadata_path);
    }
    for (size_t cluster_id = 0; cluster_id < header.cluster_num; ++cluster_id)
    {
      if (offsets[cluster_id + 1] < offsets[cluster_id])
      {
        throw std::runtime_error(
            "metadata cluster_cell_offsets not monotonic for " +
            config.metadata_path + " at cluster " +
            std::to_string(cluster_id));
      }
      if (bucket_sizes[cluster_id] > 0 &&
          offsets[cluster_id + 1] == offsets[cluster_id])
      {
        throw std::runtime_error(
            "metadata cluster_cell_offsets gives zero cells for non-empty cluster " +
            std::to_string(cluster_id) + " in " + config.metadata_path);
      }
    }
  }

  static void validate_metadata_cell_vec_offsets(
      const ResolvedConfig& config, const MetadataHeader& header,
      const std::vector<size_t>& bucket_sizes,
      const std::vector<uint64_t>& cluster_cell_offsets,
      const std::vector<uint64_t>& cell_vec_offsets)
  {
    if (cell_vec_offsets.empty() || cell_vec_offsets[0] != 0)
    {
      throw std::runtime_error("metadata cell_vec_offsets must start at zero for " +
                               config.metadata_path);
    }
    for (size_t i = 0; i + 1 < cell_vec_offsets.size(); ++i)
    {
      if (cell_vec_offsets[i + 1] < cell_vec_offsets[i])
      {
        throw std::runtime_error(
            "metadata cell_vec_offsets not monotonic for " +
            config.metadata_path + " at cell " + std::to_string(i));
      }
      if (cell_vec_offsets[i + 1] == cell_vec_offsets[i])
      {
        throw std::runtime_error("metadata cell_vec_offsets has empty cell for " +
                                 config.metadata_path + " at cell " +
                                 std::to_string(i));
      }
    }
    if (cell_vec_offsets.back() != header.n)
    {
      throw std::runtime_error(
          "metadata cell_vec_offsets final value mismatch for " +
          config.metadata_path + ": expected " + std::to_string(header.n) +
          ", observed " + std::to_string(cell_vec_offsets.back()));
    }

    uintmax_t cluster_base = 0;
    for (size_t cluster_id = 0; cluster_id < header.cluster_num; ++cluster_id)
    {
      const size_t first_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id]);
      const size_t next_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id + 1]);
      if (first_cell >= cell_vec_offsets.size() ||
          next_cell >= cell_vec_offsets.size())
      {
        throw std::runtime_error(
            "metadata cluster_cell_offsets references missing cell_vec_offsets for " +
            config.metadata_path + " at cluster " +
            std::to_string(cluster_id));
      }
      if (cell_vec_offsets[first_cell] != cluster_base)
      {
        throw std::runtime_error(
            "metadata cell_vec_offsets cluster base mismatch for " +
            config.metadata_path + " at cluster " +
            std::to_string(cluster_id) + ": expected " +
            std::to_string(cluster_base) + ", observed " +
            std::to_string(cell_vec_offsets[first_cell]));
      }
      checked_add(cluster_base, bucket_sizes[cluster_id],
                  config.metadata_path);
      if (cell_vec_offsets[next_cell] != cluster_base)
      {
        throw std::runtime_error(
            "metadata cell_vec_offsets cluster end mismatch for " +
            config.metadata_path + " at cluster " +
            std::to_string(cluster_id) + ": expected " +
            std::to_string(cluster_base) + ", observed " +
            std::to_string(cell_vec_offsets[next_cell]));
      }
    }
  }
};
