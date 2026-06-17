#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#endif

#include "../../lib/ConfigLoader.h"
#include "../../lib/DiskRange.h"
#include "../../lib/RunResult.h"

int r_server_id = 0;

namespace
{
void print_usage()
{
  std::cerr << "Usage: ./search_stats --config <path> --gt <path> "
               "[--no-cell-filter]\n"
               "       ./search_stats --print-build-info"
            << std::endl;
}

void print_build_info()
{
  std::cout << "{\"binary_name\":\"search_stats\","
            << "\"DJ_ENABLE_PRUNE_BREAKDOWN_STATS\":"
            << DJ_ENABLE_PRUNE_BREAKDOWN_STATS << ","
            << "\"DJ_ENABLE_SEARCH_PHASE_TIMING\":"
            << DJ_ENABLE_SEARCH_PHASE_TIMING << "}\n";
}

bool parse_args(int argc, char** argv, std::string& config_path,
                std::string& gt_path, bool& no_cell_filter)
{
  bool saw_config = false;
  bool saw_gt = false;
  no_cell_filter = false;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--config")
    {
      if (saw_config || i + 1 >= argc ||
          std::string(argv[i + 1]).rfind("--", 0) == 0)
      {
        return false;
      }
      config_path = argv[++i];
      saw_config = true;
    }
    else if (arg == "--gt")
    {
      if (saw_gt || i + 1 >= argc ||
          std::string(argv[i + 1]).rfind("--", 0) == 0)
      {
        return false;
      }
      gt_path = argv[++i];
      saw_gt = true;
    }
    else if (arg == "--no-cell-filter")
    {
      if (no_cell_filter)
      {
        return false;
      }
      no_cell_filter = true;
    }
    else
    {
      return false;
    }
  }
  return saw_config && saw_gt;
}

double query_class_total_time(const QueryClassStats& stats)
{
  return stats.t_centroid_dot + stats.t_candidate_gen + stats.t_cluster_bound +
         stats.t_cell_filter + stats.t_rbq_prefilter + stats.t_io_wait +
         stats.t_exact_l2;
}

size_t clamped_subtract(size_t lhs, size_t rhs)
{
  return lhs >= rhs ? lhs - rhs : 0;
}

void print_phase_metric(const std::string& prefix, const std::string& phase,
                        double phase_time, double total_time)
{
  const double pct =
      total_time == 0.0 ? 0.0 : phase_time * 100.0 / total_time;
  std::cout << prefix << "_t_" << phase << " = " << phase_time << '\n';
  std::cout << prefix << "_pct_" << phase << " = " << pct << '\n';
}

void print_query_class_stats(const std::string& prefix,
                             const QueryClassStats& stats)
{
  const double total_time = query_class_total_time(stats);
  const double n_queries = static_cast<double>(stats.n_queries);
  const auto mean_or_zero = [n_queries](size_t value) {
    return n_queries == 0.0 ? 0.0 : value / n_queries;
  };
  std::cout << prefix << "_n_queries = " << stats.n_queries << '\n';
  std::cout << prefix << "_total_time = " << total_time << '\n';
  std::cout << prefix << "_mean_us_per_query = "
            << (n_queries == 0.0 ? 0.0 : total_time * 1000000.0 / n_queries)
            << '\n';
  print_phase_metric(prefix, "centroid_dot", stats.t_centroid_dot, total_time);
  print_phase_metric(prefix, "candidate_gen", stats.t_candidate_gen, total_time);
  print_phase_metric(prefix, "cluster_bound", stats.t_cluster_bound,
                     total_time);
  print_phase_metric(prefix, "cell_filter", stats.t_cell_filter, total_time);
  print_phase_metric(prefix, "rbq_prefilter", stats.t_rbq_prefilter, total_time);
  print_phase_metric(prefix, "io_wait", stats.t_io_wait, total_time);
  print_phase_metric(prefix, "exact_l2", stats.t_exact_l2, total_time);
  std::cout << prefix << "_candidates = " << stats.candidates << '\n';
  std::cout << prefix << "_radius_prescreened = " << stats.radius_prescreened
            << '\n';
  std::cout << prefix << "_queries_with_candidates = "
            << stats.queries_with_candidates << '\n';
  std::cout << prefix << "_queries_after_prescreen = "
            << stats.queries_after_prescreen << '\n';
  std::cout << prefix << "_queries_after_cluster_bound = "
            << stats.queries_after_cluster_bound << '\n';
  std::cout << prefix << "_queries_after_cell = " << stats.queries_after_cell
            << '\n';
  std::cout << prefix << "_distinct_qdot_used = " << stats.distinct_qdot_used
            << '\n';
  std::cout << prefix << "_surviving = " << stats.surviving << '\n';
  std::cout << prefix << "_cell_kept = " << stats.cell_kept << '\n';
  std::cout << prefix << "_cells_read = " << stats.cells_read << '\n';
  std::cout << prefix << "_bytes_read = " << stats.bytes_read << '\n';
  std::cout << prefix << "_dist_comp = " << stats.dist_comp << '\n';
  std::cout << prefix << "_mean_candidates = "
            << mean_or_zero(stats.candidates) << '\n';
  std::cout << prefix << "_mean_radius_prescreened = "
            << mean_or_zero(stats.radius_prescreened) << '\n';
  std::cout << prefix << "_mean_distinct_qdot_used = "
            << mean_or_zero(stats.distinct_qdot_used) << '\n';
  std::cout << prefix << "_mean_surviving = " << mean_or_zero(stats.surviving)
            << '\n';
  std::cout << prefix << "_mean_cell_kept = " << mean_or_zero(stats.cell_kept)
            << '\n';
  std::cout << prefix << "_mean_cells_read = " << mean_or_zero(stats.cells_read)
            << '\n';
  std::cout << prefix << "_mean_bytes_read = " << mean_or_zero(stats.bytes_read)
            << '\n';
  std::cout << prefix << "_mean_dist_comp = " << mean_or_zero(stats.dist_comp)
            << '\n';
}

void print_zero_query_layer_breakdown(const RunResult& result)
{
  if constexpr (kDiskRangeSearchPhaseTimingEnabled)
  {
    const QueryClassStats& zero = result.zero_result;
    const QueryClassStats& nonzero = result.nonzero_result;
    const double n_queries = static_cast<double>(zero.n_queries);
    const auto mean_or_zero = [n_queries](size_t value) {
      return n_queries == 0.0 ? 0.0 : value / n_queries;
    };
    const size_t active_candidates =
        clamped_subtract(zero.candidates, zero.radius_prescreened);
    const size_t cluster_pruned =
        clamped_subtract(active_candidates, zero.surviving);
    const double zero_total = query_class_total_time(zero);
    const double nonzero_total = query_class_total_time(nonzero);
    const double thread_sum_total = zero_total + nonzero_total;
    const double zero_share =
        thread_sum_total == 0.0 ? 0.0 : zero_total / thread_sum_total;

    std::cout << "[Zero-query layer breakdown]\n";
    std::cout << "zero_n_queries = " << zero.n_queries << '\n';
    std::cout << "zero_queries_with_candidates = "
              << zero.queries_with_candidates << '\n';
    std::cout << "zero_mean_candidates = " << mean_or_zero(zero.candidates)
              << '\n';
    std::cout << "zero_mean_radius_prescreened = "
              << mean_or_zero(zero.radius_prescreened) << '\n';
    std::cout << "zero_queries_after_prescreen = "
              << zero.queries_after_prescreen << '\n';
    std::cout << "zero_mean_active_candidates = "
              << mean_or_zero(active_candidates) << '\n';
    std::cout << "zero_mean_cluster_pruned = " << mean_or_zero(cluster_pruned)
              << '\n';
    std::cout << "zero_queries_after_cluster_bound = "
              << zero.queries_after_cluster_bound << '\n';
    std::cout << "zero_mean_surviving = " << mean_or_zero(zero.surviving)
              << '\n';
    std::cout << "zero_queries_after_cell = " << zero.queries_after_cell
              << '\n';
    std::cout << "zero_mean_cells_read = " << mean_or_zero(zero.cells_read)
              << '\n';
    std::cout << "zero_mean_cell_kept = " << mean_or_zero(zero.cell_kept)
              << '\n';
    std::cout << "zero_mean_bytes_read = " << mean_or_zero(zero.bytes_read)
              << '\n';
    std::cout << "zero_mean_dist_comp = " << mean_or_zero(zero.dist_comp)
              << '\n';
    std::cout << "zero_total_time = " << zero_total << '\n';
    std::cout << "zero_share_thread_sum = " << zero_share << '\n';
    std::cout
        << "zero_share_note = per-class total_time sums are "
           "OMP-thread-summed, not wall-clock; wall-clock zero-query share is "
           "recorded in PF from full-vs-nonzero-only search medians\n";
    std::cout
        << "cells_read_note = under --no-cell-filter, cells_read is 0 "
           "because no cell plans are materialized; use bytes_read / "
           "dist_comp for that mode\n";
    std::cout
        << "bytes_read_note = under --no-cell-filter, bytes_read is "
           "cluster-granular (every byte of every surviving cluster); under "
           "the default cell-filter path it is cell-granular (only cells kept "
           "by the filter); the two modes are not directly comparable\n";
  }
}

void print_search_phase_timing(const RunResult& result)
{
  if constexpr (kDiskRangeSearchPhaseTimingEnabled)
  {
    std::cout << "[Search phase timing]\n";
    std::cout << "n_threads = " << result.n_threads << '\n';
    print_query_class_stats("zero_result", result.zero_result);
    print_query_class_stats("nonzero_result", result.nonzero_result);
    std::cout
        << "timing_note = per-class totals are summed over OpenMP threads and "
           "do not equal search_time; PRUNE_BREAKDOWN_STATS=ON over-counts "
           "t_cluster_bound because prune-breakdown adds an extra "
           "pca_lower_bound per candidate; configure with "
           "-DPRUNE_BREAKDOWN_STATS=OFF for production-faithful "
           "t_cluster_bound; under OFF, slab_only/pca_only/combined/cell/"
           "zero_kept/io_bytes_reduction layer counters read 0\n";
  }
}

void print_layers(const RunResult& result)
{
  std::cout << "[Layer 1 cluster]\n";
  std::cout << "mean_keep_after_filter = " << result.mean_keep_after_filter
            << '\n';
  std::cout << "triangle_prune_rate = " << result.triangle_prune_rate << '\n';
  std::cout << "triangle_candidate_clusters = "
            << result.triangle_candidate_clusters << '\n';
  std::cout << "triangle_pruned_clusters = " << result.triangle_pruned_clusters
            << '\n';
  std::cout << "radius_prescreen_rate = " << result.radius_prescreen_rate
            << '\n';
  std::cout << "radius_prescreened_clusters = "
            << result.radius_prescreened_clusters << '\n';
  std::cout << "slab_only_prune_rate = " << result.slab_only_prune_rate
            << '\n';
  std::cout << "pca_only_prune_rate = " << result.pca_only_prune_rate << '\n';
  std::cout << "combined_prune_rate = " << result.combined_prune_rate << '\n';
  std::cout << "slab_only_pruned_clusters = "
            << result.slab_only_pruned_clusters << '\n';
  std::cout << "pca_only_pruned_clusters = "
            << result.pca_only_pruned_clusters << '\n';
  std::cout << "combined_pruned_clusters = "
            << result.combined_pruned_clusters << '\n';

  std::cout << "[Layer 1.5 cell]\n";
  std::cout << "cell_candidate_vectors = " << result.cell_candidate_vectors
            << '\n';
  std::cout << "cell_pruned_vectors = " << result.cell_pruned_vectors << '\n';
  std::cout << "cell_kept_vectors = " << result.cell_kept_vectors << '\n';
  std::cout << "zero_kept_cells = " << result.zero_kept_cells << '\n';
  const size_t pruned_before_cell =
      result.radius_prescreened_clusters + result.triangle_pruned_clusters;
  const size_t surviving_clusters =
      result.triangle_candidate_clusters >= pruned_before_cell
          ? result.triangle_candidate_clusters - pruned_before_cell
          : 0;
  const double zero_kept_surviving_rate =
      surviving_clusters == 0
          ? 0.0
          : result.zero_kept_surviving_clusters /
                static_cast<double>(surviving_clusters);
  std::cout << "zero_kept_surviving_clusters = "
            << result.zero_kept_surviving_clusters << '\n';
  std::cout << "zero_kept_surviving_rate = " << zero_kept_surviving_rate
            << '\n';
  std::cout << "cell_prune_rate = " << result.cell_prune_rate << '\n';
  std::cout << "cell_bytes_read = " << result.cell_bytes_read << '\n';
  std::cout << "cluster_bytes_baseline = " << result.cluster_bytes_baseline
            << '\n';
  std::cout << "io_bytes_reduction = " << result.io_bytes_reduction << '\n';
  std::cout << "cell_metadata_bytes = " << result.cell_metadata_bytes << '\n';

  std::cout << "[Layer 3 exact]\n";
  std::cout << "dist_comp = " << result.dist_comp << '\n';
  std::cout << "recall = " << result.recall << '\n';
  std::cout << "recall_raw = " << result.recall_raw << '\n';
  std::cout << "recall_verified = " << result.recall_verified << '\n';
  std::cout << "per_query_min = " << result.per_query_min << '\n';
  std::cout << "per_query_median = " << result.per_query_median << '\n';
  std::cout << "per_query_max = " << result.per_query_max << '\n';
  std::cout << "per_query_zero = " << result.per_query_zero << '\n';
  std::cout << "build_time = " << result.build_time << '\n';
  std::cout << "search_time = " << result.search_time << '\n';
  std::cout << "qps = " << result.qps << " query/sec\n";
  std::cout << "[Fast-path emptiness oracle]\n";
  std::cout << "fast_path_skip_count = " << result.fast_path_skip_count << '\n';
  std::cout << "fast_path_oracle_total_us = "
            << result.fast_path_oracle_total_us << '\n';
  std::cout << "fast_path_skip_rate = " << result.fast_path_skip_rate << '\n';
  std::cout << "fast_path_false_empty_queries = "
            << result.fast_path_false_empty_queries << '\n';
  std::cout << "fast_path_ground_truth_nonempty_queries = "
            << result.fast_path_ground_truth_nonempty_queries << '\n';
  std::cout << "fast_path_suppressed_baseline_hits = "
            << result.fast_path_suppressed_baseline_hits << '\n';
  std::cout << "false_empty_query_rate = "
            << result.false_empty_query_rate << '\n';
  std::cout << "range_recall_loss_hit_weighted = "
            << result.range_recall_loss_hit_weighted << '\n';
  std::cout << "zero_result_short_circuit_rate = "
            << result.zero_result_short_circuit_rate << '\n';
  std::cout << "fast_path_trace_path = " << result.fast_path_trace_path << '\n';
  std::cout << "slow_path_trace_path = " << result.slow_path_trace_path << '\n';
  std::cout << "[Data HNSW oracle]\n";
  std::cout << "data_hnsw_skip_count = " << result.data_hnsw_skip_count << '\n';
  std::cout << "data_hnsw_oracle_total_us = "
            << result.data_hnsw_oracle_total_us << '\n';
  std::cout << "data_hnsw_skip_rate = " << result.data_hnsw_skip_rate << '\n';
  std::cout << "data_hnsw_trace_path = " << result.data_hnsw_trace_path << '\n';
  print_search_phase_timing(result);
  print_zero_query_layer_breakdown(result);
}

void write_data_hnsw_trace(const RunResult& result)
{
  if (result.data_hnsw_trace.empty())
  {
    return;
  }
  const std::filesystem::path trace_path(result.data_hnsw_trace_path);
  const std::filesystem::path parent = trace_path.parent_path();
  if (!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(trace_path);
  if (!out)
  {
    throw std::runtime_error("failed to open data_hnsw_trace_path for write: " +
                             result.data_hnsw_trace_path);
  }
  out << "query_id,d_hat_l2,oracle_decision\n";
  for (const DataHNSWTraceRecord& record : result.data_hnsw_trace)
  {
    out << record.query_id << ',' << record.d_hat_l2 << ','
        << (record.oracle_decision ? 1 : 0) << '\n';
  }
}

void write_fast_path_trace(const RunResult& result)
{
  if (result.fast_path_trace.empty())
  {
    return;
  }
  const std::filesystem::path trace_path(result.fast_path_trace_path);
  const std::filesystem::path parent = trace_path.parent_path();
  if (!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(trace_path);
  if (!out)
  {
    throw std::runtime_error("failed to open fast_path_trace_path for write: " +
                             result.fast_path_trace_path);
  }
  out << "query_id,oracle_decision,phase1_candidate_count,"
         "phase2_scanned_vectors,phase2_borderline_count,"
         "phase3_refined_count,min_lower_bound,"
         "first_nan_field_mask,first_nan_cluster_id,first_nan_local_index\n";
  for (const FastPathTraceRecord& record : result.fast_path_trace)
  {
    out << record.query_id << ',' << (record.oracle_decision ? 1 : 0) << ','
        << record.phase1_candidate_count << ','
        << record.phase2_scanned_vectors << ','
        << record.phase2_borderline_count << ','
        << record.phase3_refined_count << ',' << record.min_lower_bound << ','
        << record.first_nan_field_mask << ',' << record.first_nan_cluster_id
        << ',' << record.first_nan_local_index << '\n';
  }
}

void write_slow_path_trace(const RunResult& result)
{
  if (result.slow_path_trace.empty())
  {
    return;
  }
  const std::filesystem::path trace_path(result.slow_path_trace_path);
  const std::filesystem::path parent = trace_path.parent_path();
  if (!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(trace_path);
  if (!out)
  {
    throw std::runtime_error("failed to open slow_path_trace_path for write: " +
                             result.slow_path_trace_path);
  }
  out << "query_id,skipped_by_fast_path,t_cell_filter,t_io_wait,t_exact_l2,"
         "cell_kept,cells_read,bytes_read,dist_comp,candidates,hit_count,"
         "hit_dist_sum,s_nh,s_h,s_borderline,cells_all_rejected,cells_total,"
         "cells_skipped_phase_b\n";
  out << std::setprecision(9);
  for (const SlowPathTraceRecord& record : result.slow_path_trace)
  {
    out << record.query_id << ','
        << (record.skipped_by_fast_path ? 1 : 0) << ','
        << record.t_cell_filter << ',' << record.t_io_wait << ','
        << record.t_exact_l2 << ',' << record.cell_kept << ','
        << record.cells_read << ',' << record.bytes_read << ','
        << record.dist_comp << ',' << record.candidates << ','
        << record.hit_count << ',' << record.hit_dist_sum << ','
        << record.s_nh << ',' << record.s_h << ',' << record.s_borderline
        << ',' << record.cells_all_rejected << ',' << record.cells_total
        << ',' << record.cells_skipped_phase_b << '\n';
  }
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc == 2 && std::string(argv[1]) == "--print-build-info")
  {
    print_build_info();
    return 0;
  }

  std::string config_path;
  std::string gt_path;
  bool no_cell_filter = false;
  if (!parse_args(argc, argv, config_path, gt_path, no_cell_filter))
  {
    print_usage();
    return 1;
  }

  Eigen::setNbThreads(1);
#ifdef DJ_USE_OPENBLAS
  openblas_set_num_threads(1);
#endif

  try
  {
    ResolvedConfig config = load_resolved_config(config_path, gt_path);
    if (no_cell_filter)
    {
      config.cell_filter_enabled = false;
    }
    DiskRange index;
    const RunResult result = index.search(config);
    write_data_hnsw_trace(result);
    write_fast_path_trace(result);
    write_slow_path_trace(result);
    print_layers(result);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
