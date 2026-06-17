#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef DJ_ENABLE_PRUNE_BREAKDOWN_STATS
#define DJ_ENABLE_PRUNE_BREAKDOWN_STATS 0
#endif

#ifndef DJ_ENABLE_SEARCH_PHASE_TIMING
#define DJ_ENABLE_SEARCH_PHASE_TIMING 0
#endif

inline constexpr bool kDiskRangePruneBreakdownStatsEnabled =
    DJ_ENABLE_PRUNE_BREAKDOWN_STATS != 0;

inline constexpr bool kDiskRangeSearchPhaseTimingEnabled =
    DJ_ENABLE_SEARCH_PHASE_TIMING != 0;

struct QueryClassStats
{
  size_t n_queries = 0;
  double t_centroid_dot = 0.0;
  double t_candidate_gen = 0.0;
  double t_cluster_bound = 0.0;
  double t_cell_filter = 0.0;
  double t_io_wait = 0.0;
  double t_exact_l2 = 0.0;
  double t_rbq_prefilter = 0.0;  // RBQ phase_a/phase_b, previously uncounted
  size_t candidates = 0;
  size_t radius_prescreened = 0;
  size_t queries_with_candidates = 0;
  size_t queries_after_prescreen = 0;
  size_t queries_after_cluster_bound = 0;
  size_t queries_after_cell = 0;
  size_t distinct_qdot_used = 0;
  size_t surviving = 0;
  size_t cell_kept = 0;
  size_t cells_read = 0;
  size_t bytes_read = 0;
  size_t dist_comp = 0;

  void add(const QueryClassStats& other)
  {
    n_queries += other.n_queries;
    t_centroid_dot += other.t_centroid_dot;
    t_candidate_gen += other.t_candidate_gen;
    t_cluster_bound += other.t_cluster_bound;
    t_cell_filter += other.t_cell_filter;
    t_io_wait += other.t_io_wait;
    t_exact_l2 += other.t_exact_l2;
    t_rbq_prefilter += other.t_rbq_prefilter;
    candidates += other.candidates;
    radius_prescreened += other.radius_prescreened;
    queries_with_candidates += other.queries_with_candidates;
    queries_after_prescreen += other.queries_after_prescreen;
    queries_after_cluster_bound += other.queries_after_cluster_bound;
    queries_after_cell += other.queries_after_cell;
    distinct_qdot_used += other.distinct_qdot_used;
    surviving += other.surviving;
    cell_kept += other.cell_kept;
    cells_read += other.cells_read;
    bytes_read += other.bytes_read;
    dist_comp += other.dist_comp;
  }
};

struct DataHNSWTraceRecord
{
  size_t query_id = 0;
  float d_hat_l2 = 0.0f;
  bool oracle_decision = false;
};

struct FastPathTraceRecord
{
  size_t query_id = 0;
  bool oracle_decision = false;
  size_t phase1_candidate_count = 0;
  size_t phase2_scanned_vectors = 0;
  size_t phase2_borderline_count = 0;
  size_t phase3_refined_count = 0;
  float min_lower_bound = 0.0f;
  // NaN instrumentation: when a NaN early-exit fires in Phase 2 or Phase 3,
  // these record the first occurrence. field_mask bits:
  //   1 = est_distance, 2 = low_distance, 4 = ip_x0_qr, 8 = bin_f_error (P2),
  //   16 = phase3_est_dist, 32 = phase3_low_dist.
  // 0 means no NaN was hit on this query.
  uint32_t first_nan_field_mask = 0;
  size_t first_nan_cluster_id = 0;
  size_t first_nan_local_index = 0;
};

// Per-query slow-path breakdown. Populated when DJ_ENABLE_SEARCH_PHASE_TIMING=1.
// Records every query, including those skipped by fast-path (with the flag set
// and timings zero) so the analyzer can filter slow-path-only subset.
// When slow_path_rbq_prefilter_enabled is true, s_nh/s_h/s_borderline split the
// cell_kept population into RBQ-rejected / RBQ-accepted / fell-back-to-exact.
struct SlowPathTraceRecord
{
  size_t query_id = 0;
  bool skipped_by_fast_path = false;
  double t_cell_filter = 0.0;
  double t_io_wait = 0.0;
  double t_exact_l2 = 0.0;
  size_t cell_kept = 0;
  size_t cells_read = 0;
  size_t bytes_read = 0;
  size_t dist_comp = 0;
  size_t candidates = 0;
  size_t hit_count = 0;
  double hit_dist_sum = 0.0;
  size_t s_nh = 0;          // vecs proven NOT-hit by RBQ lb > R²
  size_t s_h = 0;           // vecs proven HIT by RBQ ub <= R² (diagnostic)
  size_t s_borderline = 0;  // vecs that fell back to exact L2
  size_t cells_all_rejected = 0;       // cells where every vec was lb-rejected
  size_t cells_total = 0;              // cells actually visited (== cells_read)
  size_t cells_skipped_phase_b = 0;    // cells skipped before submit (Phase B)
};

// Hit reference for post-search L2 verification.
// Stored per-query so verify can re-read the cell and recompute exact L2
// outside the search timer, filtering out false positives that may arise from
// future probabilistic-accept paths (e.g. ub-acceptance).
struct HitRef
{
  uint32_t cluster_id = 0;
  uint32_t cell_id = 0;
  uint32_t row_in_cell = 0;
};

struct VerifiedHit
{
  size_t query_id = 0;
  size_t original_base_id = 0;
  float distance = 0.0f;
};

struct RunResult
{
  double recall = 0.0;           // backward compat, aliases recall_raw
  double recall_raw = 0.0;       // hits accepted by callback / total_in_range
  double recall_verified = 0.0;  // post-L2-verify hits / total_in_range
  double qps = 0.0;              // n_queries / search_time
  double build_time = 0.0;
  double search_time = 0.0;
  // Geometric cluster prune ratio. Since OpenSpec change pca-enclosing-bound,
  // the numerator counts clusters pruned by the PCA enclosing lower bound.
  double triangle_prune_rate = 0.0;
  size_t triangle_candidate_clusters = 0;
  size_t triangle_pruned_clusters = 0;
  size_t radius_prescreened_clusters = 0;
  double radius_prescreen_rate = 0.0;
  size_t slab_pruned_clusters = 0;
  size_t box_residual_pruned_clusters = 0;
  size_t slab_only_pruned_clusters = 0;
  size_t pca_only_pruned_clusters = 0;
  size_t combined_pruned_clusters = 0;
  double slab_only_prune_rate = 0.0;
  double pca_only_prune_rate = 0.0;
  double combined_prune_rate = 0.0;
  size_t cell_candidate_vectors = 0;
  size_t cell_pruned_vectors = 0;
  size_t cell_kept_vectors = 0;
  size_t zero_kept_cells = 0;
  size_t zero_kept_surviving_clusters = 0;
  size_t cell_bytes_read = 0;
  size_t cluster_bytes_baseline = 0;
  size_t cell_metadata_bytes = 0;
  double cell_prune_rate = 0.0;
  double io_bytes_reduction = 0.0;
  double mean_keep_after_filter = 0.0;
  size_t dist_comp = 0;
  size_t per_query_min = 0;
  double per_query_median = 0.0;
  size_t per_query_max = 0;
  size_t per_query_zero = 0;
  QueryClassStats zero_result;
  QueryClassStats nonzero_result;
  size_t n_threads = 0;
  size_t data_hnsw_skip_count = 0;
  double data_hnsw_oracle_total_us = 0.0;
  double data_hnsw_skip_rate = 0.0;
  std::vector<DataHNSWTraceRecord> data_hnsw_trace;
  std::string data_hnsw_trace_path;
  size_t fast_path_skip_count = 0;
  double fast_path_oracle_total_us = 0.0;
  double fast_path_skip_rate = 0.0;
  size_t fast_path_false_empty_queries = 0;
  size_t fast_path_ground_truth_nonempty_queries = 0;
  size_t fast_path_suppressed_baseline_hits = 0;
  double false_empty_query_rate = 0.0;
  double range_recall_loss_hit_weighted = 0.0;
  double zero_result_short_circuit_rate = 0.0;
  std::vector<FastPathTraceRecord> fast_path_trace;
  std::string fast_path_trace_path;
  std::vector<SlowPathTraceRecord> slow_path_trace;
  std::string slow_path_trace_path;
  std::vector<VerifiedHit> verified_hits;

  RunResult() = default;
};
