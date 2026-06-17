#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "ClusterIO.h"
#include "FastPathPhase3AsyncFetch.h"
#include "RBQCodeStorage.h"
#include "SQGCentroidIndex.h"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/space.hpp"
#include "utils/memory.hpp"

struct FastPathOracleResult
{
  bool is_empty = false;
  size_t phase1_candidate_count = 0;
  size_t phase2_scanned_vectors = 0;
  size_t phase2_borderline_count = 0;
  size_t phase3_refined_count = 0;
  float min_lower_bound = std::numeric_limits<float>::infinity();
  // NaN instrumentation (set on first NaN encountered, then evaluation exits).
  // field_mask bits: 1=est, 2=low_dist, 4=ip_x0_qr, 8=bin_f_error (Phase 2),
  //                  16=phase3_est_dist, 32=phase3_low_dist.
  uint32_t first_nan_field_mask = 0;
  size_t first_nan_cluster_id = 0;
  size_t first_nan_local_index = 0;
};

// FastPathEmptinessOracle is an opt-in, pre-slow-path emptiness oracle. SQG
// receives raw queries. In rbq mode, all RaBitQ query/centroid/data buffers are
// rotated by the SQG rotator exposed through SQGCentroidIndex. In exact mode,
// evaluate() reuses the caller-owned ClusterReader and its per-OpenMP-thread IO
// slots; callers should invoke it from the DiskRange per-query loop or another
// context with the same one-thread-per-slot discipline.
class FastPathEmptinessOracle
{
 public:
  enum class Mode
  {
    kRbq,
    kExact
  };

  FastPathEmptinessOracle(const std::string& sqg_path,
                          const std::string& rabitq_1bit_codes_path,
                          const std::string& rabitq_8bit_codes_path,
                          size_t cluster_count, size_t dim, size_t sqg_m,
                          size_t sqg_ef_search, size_t nprobe,
                          const std::vector<float>& centroids,
                          const std::vector<size_t>& bucket_sizes)
      : FastPathEmptinessOracle(
            "rbq", sqg_path, rabitq_1bit_codes_path, rabitq_8bit_codes_path,
            cluster_count, dim, sqg_m, sqg_ef_search, nprobe, centroids,
            bucket_sizes, nullptr)
  {
  }

  FastPathEmptinessOracle(const std::string& fast_path_mode,
                          const std::string& sqg_path,
                          const std::string& rabitq_1bit_codes_path,
                          const std::string& rabitq_8bit_codes_path,
                          size_t cluster_count, size_t dim, size_t sqg_m,
                          size_t sqg_ef_search, size_t nprobe,
                          const std::vector<float>& centroids,
                          const std::vector<size_t>& bucket_sizes,
                          ClusterReader* cluster_reader)
      : nprobe_(nprobe), dim_(dim)
  {
    mode_ = parse_mode(fast_path_mode);
    cluster_count_ = cluster_count;
    cluster_reader_ = cluster_reader;
    if (nprobe == 0 || nprobe > cluster_count)
    {
      throw std::runtime_error(
          "fast_path_nprobe_N must be <= cluster_num and positive");
    }
    if (centroids.size() != cluster_count * dim)
    {
      throw std::runtime_error("fast-path centroid payload size mismatch");
    }
    if (bucket_sizes.size() != cluster_count)
    {
      throw std::runtime_error("fast-path metadata bucket size mismatch");
    }
    sqg_.load(sqg_path, cluster_count, dim, sqg_m, sqg_ef_search);
    padded_dim_ = sqg_.padded_dim();
    if (mode_ == Mode::kExact)
    {
      if (cluster_reader_ == nullptr)
      {
        throw std::runtime_error(
            "fast_path_mode=exact requires a ClusterReader instance");
      }
      if (cluster_reader_->cluster_num != cluster_count ||
          cluster_reader_->d != dim || cluster_reader_->bucket_sizes != bucket_sizes)
      {
        throw std::runtime_error(
            "fast-path exact ClusterReader metadata mismatch");
      }
      return;
    }

    require_existing_rbq_artifact(rabitq_1bit_codes_path, "1-bit");
    require_existing_rbq_artifact(rabitq_8bit_codes_path, "8-bit");
    one_bit_codes_.load_1bit(rabitq_1bit_codes_path);
    if (std::getenv("DJ_PHASE3_INMEM") != nullptr)
    {
      // option C upper-bound probe: 8-bit codes fully in RAM -> read_code
      // memcpy (no pread, no page-cache contention). Ceiling for any phase3
      // io_uring/async optimization. See 8bit_async.md §6.
      eight_bit_codes_.load_8bit(rabitq_8bit_codes_path);
    }
    else
    {
      eight_bit_codes_.open_8bit_for_pread(rabitq_8bit_codes_path);
    }
    if (one_bit_codes_.cluster_count() != cluster_count ||
        eight_bit_codes_.cluster_count() != cluster_count ||
        one_bit_codes_.dim() != dim || eight_bit_codes_.dim() != dim ||
        one_bit_codes_.padded_dim() != padded_dim_ ||
        eight_bit_codes_.padded_dim() != padded_dim_)
    {
      throw std::runtime_error("fast-path RBQ artifact metadata mismatch");
    }
    one_bit_codes_.validate_bucket_sizes(bucket_sizes, "1-bit");
    eight_bit_codes_.validate_bucket_sizes(bucket_sizes, "8-bit");
    build_rotated_centroids(centroids, cluster_count);
  }

  // Slow-path RBQ prefilter helpers. Expose 1-bit lb/ub classification on a
  // single cluster's vecs without short-circuit semantics. Reuses the same
  // SplitBatchQuery + FastScan machinery as evaluate_rbq. Caller is responsible
  // for rotating the query once via rotate_query() and reusing it across
  // multiple classify_cluster_rbq_1bit() calls (one per candidate cluster).
  // Soundness: for every vec i, `lb_out[i] <= ||q-y_i||^2 <= ub_out[i]`.
  // On NaN (any field non-finite for any vec in any batch), the function sets
  // all entries for the offending cluster to lb=-inf, ub=+inf and returns false
  // so the caller treats every vec as borderline (i.e. falls back to exact L2).
  bool classify_cluster_rbq_1bit(const float* rotated_query, size_t cluster_id,
                                 std::vector<float>& lb_out,
                                 std::vector<float>& ub_out) const
  {
    if (mode_ != Mode::kRbq)
    {
      throw std::runtime_error(
          "classify_cluster_rbq_1bit requires fast_path_mode=rbq");
    }
    if (cluster_id >= one_bit_codes_.cluster_count())
    {
      throw std::runtime_error("classify_cluster_rbq_1bit: cluster id out of range");
    }
    const size_t vec_count = one_bit_codes_.cluster_vec_count(cluster_id);
    lb_out.assign(vec_count, -std::numeric_limits<float>::infinity());
    ub_out.assign(vec_count, std::numeric_limits<float>::infinity());
    if (vec_count == 0)
    {
      return true;
    }
    rabitqlib::SplitBatchQuery<float> q_obj(rotated_query, padded_dim_,
                                            kPhase3ExBits, rabitqlib::METRIC_L2,
                                            true);
    set_query_cluster_distance(q_obj, rotated_query, cluster_id);
    std::array<float, rabitqlib::fastscan::kBatchSize> est_distance{};
    std::array<float, rabitqlib::fastscan::kBatchSize> low_distance{};
    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr{};
    for (size_t local = 0; local < vec_count;
         local += rabitqlib::fastscan::kBatchSize)
    {
      const size_t n =
          std::min(rabitqlib::fastscan::kBatchSize, vec_count - local);
      rabitqlib::split_batch_estdist(
          one_bit_codes_.batch_code_ptr(
              cluster_id, local / rabitqlib::fastscan::kBatchSize),
          q_obj, padded_dim_, est_distance.data(), low_distance.data(),
          ip_x0_qr.data(), true);
      for (size_t i = 0; i < n; ++i)
      {
        if (!std::isfinite(est_distance[i]) ||
            !std::isfinite(low_distance[i]))
        {
          // Conservative fallback: mark whole cluster as borderline.
          std::fill(lb_out.begin(), lb_out.end(),
                    -std::numeric_limits<float>::infinity());
          std::fill(ub_out.begin(), ub_out.end(),
                    std::numeric_limits<float>::infinity());
          return false;
        }
        lb_out[local + i] = low_distance[i];
        // ub = est + err = 2 * est - lb (since lb = est - err).
        ub_out[local + i] = 2.0f * est_distance[i] - low_distance[i];
      }
    }
    return true;
  }

  void rotate_query(const float* query, float* rotated_out) const
  {
    sqg_.rotator().rotate(query, rotated_out);
  }

  size_t padded_dim() const { return padded_dim_; }

  // Expose the loaded SQG centroid graph so the slow path can reuse it for
  // candidate generation instead of loading a second copy of the same artifact.
  const SQGCentroidIndex& centroid_index() const { return sqg_; }

  size_t cluster_vec_count(size_t cluster_id) const
  {
    return one_bit_codes_.cluster_vec_count(cluster_id);
  }

  bool is_rbq_mode() const { return mode_ == Mode::kRbq; }

  FastPathOracleResult evaluate(const float* query, double radius_sq)
  {
    FastPathOracleResult result;
    if (!std::isfinite(radius_sq) || radius_sq < 0.0)
    {
      throw std::runtime_error(
          "fast-path radius_sq must be finite and non-negative");
    }
    for (size_t i = 0; i < dim_; ++i)
    {
      if (!std::isfinite(query[i]))
      {
        return result;
      }
    }
    // Ablation (env DJ_ORACLE_NEVER_SKIP=1): the oracle runs as a no-op that
    // never declares any query empty, so every query flows to the slow path
    // (whose RBQ prefilter stays intact). Isolates the fast-path oracle's skip
    // contribution: (never-skip row) -> 1bit_only diff = pure fast-path skip;
    // baseline -> (never-skip row) diff = slow-path RBQ prefilter. is_empty
    // defaults to false. Off by default; composes with DJ_ORACLE_1BIT_ONLY.
    static const bool never_skip =
        std::getenv("DJ_ORACLE_NEVER_SKIP") != nullptr;
    if (never_skip)
    {
      return result;
    }
    const std::vector<uint32_t> candidate_ids = sqg_.search(query, nprobe_);
    result.phase1_candidate_count = candidate_ids.size();
    if (mode_ == Mode::kExact)
    {
      return evaluate_exact(query, radius_sq, candidate_ids, result);
    }
    return evaluate_rbq(query, radius_sq, candidate_ids, result);
  }

 private:
  static constexpr size_t kPhase3ExBits = 7;

  static Mode parse_mode(const std::string& fast_path_mode)
  {
    if (fast_path_mode == "rbq")
    {
      return Mode::kRbq;
    }
    if (fast_path_mode == "exact")
    {
      return Mode::kExact;
    }
    throw std::runtime_error("fast_path_mode must be one of rbq/exact: " +
                             fast_path_mode);
  }

  static void require_existing_rbq_artifact(const std::string& path,
                                            const std::string& label)
  {
    if (path.empty() || !std::filesystem::exists(path))
    {
      throw std::runtime_error("missing RBQ artifacts for fast_path_mode=rbq: " +
                               label + " artifact missing at " + path +
                               "; rebuild with matching mode");
    }
  }

  FastPathOracleResult evaluate_exact(
      const float* query, double radius_sq,
      const std::vector<uint32_t>& candidate_ids, FastPathOracleResult result)
  {
    if (cluster_reader_ == nullptr)
    {
      throw std::runtime_error("fast-path exact mode has no ClusterReader");
    }
    std::vector<size_t> cluster_ids;
    cluster_ids.reserve(candidate_ids.size());
    for (const uint32_t cid32 : candidate_ids)
    {
      const size_t cid = static_cast<size_t>(cid32);
      if (cid >= cluster_count_)
      {
        throw std::runtime_error("fast-path SQG candidate id out of range");
      }
      cluster_ids.push_back(cid);
    }

    bool non_finite = false;
    size_t distance_dim = dim_;
    assert(omp_in_parallel());
    cluster_reader_->submit_and_drain(omp_get_thread_num(), cluster_ids,
                                      [&](size_t cluster_id, const void* data,
                                          size_t bucket_size,
                                          size_t cluster_local_row_base) {
                                        (void)cluster_id;
                                        (void)cluster_local_row_base;
                                        if (non_finite)
                                        {
                                          return;
                                        }
                                        const float* float_data =
                                            static_cast<const float*>(data);
                                        for (size_t row = 0; row < bucket_size;
                                             ++row)
                                        {
                                          const float dist = utils::L2Sqr(
                                              query,
                                              float_data + row * dim_,
                                              &distance_dim);
                                          if (!std::isfinite(dist))
                                          {
                                            non_finite = true;
                                            return;
                                          }
                                          result.min_lower_bound = std::min(
                                              result.min_lower_bound, dist);
                                          ++result.phase2_scanned_vectors;
                                        }
                                      });
    if (non_finite)
    {
      result.is_empty = false;
      return result;
    }
    if (result.phase2_scanned_vectors == 0)
    {
      return result;
    }
    result.phase2_borderline_count = 0;
    result.phase3_refined_count = 0;
    result.is_empty =
        result.min_lower_bound > static_cast<float>(radius_sq);
    return result;
  }

  FastPathOracleResult evaluate_rbq(
      const float* query, double radius_sq,
      const std::vector<uint32_t>& candidate_ids, FastPathOracleResult result)
  {
    using AlignedFloatVector =
        std::vector<float, symqg::memory::AlignedAllocator<float>>;
    AlignedFloatVector rotated_query(padded_dim_);
    sqg_.rotator().rotate(query, rotated_query.data());
    rabitqlib::SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, kPhase3ExBits, rabitqlib::METRIC_L2,
        true);

    struct Phase2Borderline
    {
      BorderlineRef ref;
      float ip_x0_qr = 0.0f;
      float bin_f_error = 0.0f;
    };
    std::array<float, rabitqlib::fastscan::kBatchSize> est_distance{};
    std::array<float, rabitqlib::fastscan::kBatchSize> low_distance{};
    std::array<float, rabitqlib::fastscan::kBatchSize> ip_x0_qr{};
    std::vector<Phase2Borderline> borderline_infos;
    std::vector<BorderlineRef> borderline_refs;
    for (const uint32_t cid32 : candidate_ids)
    {
      const size_t cid = static_cast<size_t>(cid32);
      if (cid >= one_bit_codes_.cluster_count())
      {
        throw std::runtime_error("fast-path SQG candidate id out of range");
      }
      const size_t vec_count = one_bit_codes_.cluster_vec_count(cid);
      if (vec_count == 0)
      {
        continue;
      }
      set_query_cluster_distance(q_obj, rotated_query.data(), cid);
      for (size_t local = 0; local < vec_count;
           local += rabitqlib::fastscan::kBatchSize)
      {
        const size_t n =
            std::min(rabitqlib::fastscan::kBatchSize, vec_count - local);
        rabitqlib::split_batch_estdist(
            one_bit_codes_.batch_code_ptr(
                cid, local / rabitqlib::fastscan::kBatchSize),
            q_obj, padded_dim_, est_distance.data(), low_distance.data(),
            ip_x0_qr.data(), true);
        result.phase2_scanned_vectors += n;
        for (size_t i = 0; i < n; ++i)
        {
          const size_t local_index = local + i;
          const float bin_f_error =
              one_bit_codes_.batch_f_error(cid, local_index);
          if (!std::isfinite(est_distance[i]) ||
              !std::isfinite(low_distance[i]) ||
              !std::isfinite(ip_x0_qr[i]) || !std::isfinite(bin_f_error))
          {
            result.is_empty = false;
            result.phase2_borderline_count = borderline_refs.size();
            uint32_t mask = 0;
            if (!std::isfinite(est_distance[i])) mask |= 1u;
            if (!std::isfinite(low_distance[i])) mask |= 2u;
            if (!std::isfinite(ip_x0_qr[i])) mask |= 4u;
            if (!std::isfinite(bin_f_error)) mask |= 8u;
            result.first_nan_field_mask = mask;
            result.first_nan_cluster_id = cid;
            result.first_nan_local_index = local_index;
            return result;
          }
          const float low_dist = low_distance[i];
          result.min_lower_bound =
              std::min(result.min_lower_bound, low_dist);
          if (low_dist <= static_cast<float>(radius_sq))
          {
            const BorderlineRef ref{cid, local_index};
            borderline_refs.push_back(ref);
            borderline_infos.push_back({ref, ip_x0_qr[i], bin_f_error});
          }
        }
      }
    }

    if (result.phase2_scanned_vectors == 0)
    {
      return result;
    }
    result.phase2_borderline_count = borderline_refs.size();
    if (borderline_refs.empty())
    {
      result.is_empty = true;
      return result;
    }

    // Ablation (env DJ_ORACLE_1BIT_ONLY=1): drop the 8-bit refine entirely.
    // Any query with a 1-bit borderline is declared NON-empty and flows to the
    // slow path (Cluster Pruner), which makes the final exact decision. Sound:
    // the Phase-2 1-bit skip (min lb > r^2, lines above) is unchanged; we only
    // give up the 8-bit reclamation of borderline-but-actually-empty queries.
    // Expected: recall unchanged (slow path is exact), skip_rate down, qps down.
    // Off by default -> original two-stage (1-bit + 8-bit) path. Measures the
    // Query_Classifier_1bit_vs_8bit.md §5 ablation.
    static const bool one_bit_only =
        std::getenv("DJ_ORACLE_1BIT_ONLY") != nullptr;
    if (one_bit_only)
    {
      result.is_empty = false;
      return result;
    }

    // Step-0 probe (env DJ_PHASE3_SYNC=1): fetch borderline 8-bit codes on the
    // calling thread via plain serial pread, bypassing async_fetch_8bit_codes'
    // std::async(launch::async) + immediate .get(). Isolates "remove the
    // per-query helper thread" from "batch the IO" (see 8bit_async.md Step 0).
    // Off by default -> original std::async path, byte-identical results.
    static const bool phase3_sync = std::getenv("DJ_PHASE3_SYNC") != nullptr;
    std::vector<BorderlineCode> fetched;
    if (phase3_sync)
    {
      fetched.reserve(borderline_refs.size());
      for (const BorderlineRef& ref : borderline_refs)
      {
        fetched.push_back({ref.cluster_id, ref.local_index,
                           eight_bit_codes_.read_code(ref.cluster_id,
                                                      ref.local_index)});
      }
    }
    else
    {
      auto fetched_future =
          async_fetch_8bit_codes(std::move(borderline_refs), eight_bit_codes_);
      fetched = fetched_future.get();
    }
    const auto ip_func = rabitqlib::select_excode_ipfunc(kPhase3ExBits);
    if (fetched.size() != borderline_infos.size())
    {
      throw std::runtime_error("fast-path Phase 3 fetch count mismatch");
    }
    for (size_t i = 0; i < fetched.size(); ++i)
    {
      const BorderlineCode& code = fetched[i];
      const Phase2Borderline& info = borderline_infos[i];
      if (code.cluster_id != info.ref.cluster_id ||
          code.local_index != info.ref.local_index)
      {
        throw std::runtime_error("fast-path Phase 3 fetch order mismatch");
      }
      ++result.phase3_refined_count;
      set_query_cluster_distance(q_obj, rotated_query.data(), code.cluster_id);
      float est_dist = 0.0f;
      est_dist = rabitqlib::split_distance_boosting(
          code.code_bytes.data(), ip_func, q_obj, padded_dim_, kPhase3ExBits,
          info.ip_x0_qr);
      const float low_dist =
          est_dist - (info.bin_f_error * q_obj.g_error() /
                      static_cast<float>(1u << kPhase3ExBits));
      if (!std::isfinite(est_dist) || !std::isfinite(low_dist))
      {
        result.is_empty = false;
        uint32_t mask = 0;
        if (!std::isfinite(est_dist)) mask |= 16u;
        if (!std::isfinite(low_dist)) mask |= 32u;
        result.first_nan_field_mask = mask;
        result.first_nan_cluster_id = code.cluster_id;
        result.first_nan_local_index = code.local_index;
        return result;
      }
      result.min_lower_bound = std::min(result.min_lower_bound, low_dist);
      if (low_dist <= static_cast<float>(radius_sq))
      {
        result.is_empty = false;
        return result;
      }
    }
    result.is_empty = true;
    return result;
  }

  void build_rotated_centroids(const std::vector<float>& centroids,
                               size_t cluster_count)
  {
    using AlignedFloatVector =
        std::vector<float, symqg::memory::AlignedAllocator<float>>;
    rotated_centroids_.assign(cluster_count * padded_dim_, 0.0f);
    AlignedFloatVector tmp(padded_dim_);
    for (size_t cid = 0; cid < cluster_count; ++cid)
    {
      sqg_.rotator().rotate(centroids.data() + cid * dim_, tmp.data());
      std::copy(tmp.begin(), tmp.end(),
                rotated_centroids_.begin() +
                    static_cast<std::ptrdiff_t>(cid * padded_dim_));
    }
  }

  void set_query_cluster_distance(rabitqlib::SplitBatchQuery<float>& q_obj,
                                  const float* rotated_query,
                                  size_t cluster_id) const
  {
    const float* centroid =
        rotated_centroids_.data() + cluster_id * padded_dim_;
    float dist_sq = 0.0f;
    for (size_t i = 0; i < padded_dim_; ++i)
    {
      const float diff = rotated_query[i] - centroid[i];
      dist_sq += diff * diff;
    }
    q_obj.set_g_add(std::sqrt(std::max(0.0f, dist_sq)));
  }

  SQGCentroidIndex sqg_;
  RBQCodeStorage one_bit_codes_;
  RBQCodeStorage eight_bit_codes_;
  Mode mode_ = Mode::kRbq;
  size_t nprobe_ = 0;
  size_t dim_ = 0;
  size_t cluster_count_ = 0;
  size_t padded_dim_ = 0;
  ClusterReader* cluster_reader_ = nullptr;
  std::vector<float, symqg::memory::AlignedAllocator<float>> rotated_centroids_;
};
