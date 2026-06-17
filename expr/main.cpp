#include <exception>
#include <iostream>
#include <string>

#include <Eigen/Core>
#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#endif

#include "../lib/ConfigLoader.h"
#include "../lib/DiskRange.h"
#include "../lib/RunResult.h"
#include "Basic/Console/console_RDMA.hpp"

int r_server_id = 0;

namespace
{
void print_usage()
{
  std::cerr << "Usage: ./main --config <path> --gt <path> [--no-cell-filter]"
            << std::endl;
}

bool parse_args(int argc, char** argv, std::string& config_path,
                std::string& gt_path, bool& no_cell_filter)
{
  bool saw_config = false;
  bool saw_gt = false;
  no_cell_filter = false;

  for (int i = 1; i < argc; i++)
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

void print_run_result(const RunResult& result)
{
  msginfo_s("recall = {}", result.recall);
  msginfo_s("triangle_prune_rate = {}", result.triangle_prune_rate);
  msginfo_s("triangle_candidate_clusters = {}", result.triangle_candidate_clusters);
  msginfo_s("triangle_pruned_clusters = {}", result.triangle_pruned_clusters);
  msginfo_s("slab_pruned_clusters = {}", result.slab_pruned_clusters);
  msginfo_s("box_residual_pruned_clusters = {}", result.box_residual_pruned_clusters);
  if constexpr (kDiskRangePruneBreakdownStatsEnabled)
  {
    msginfo_s("slab_only_prune_rate = {}", result.slab_only_prune_rate);
    msginfo_s("pca_only_prune_rate = {}", result.pca_only_prune_rate);
    msginfo_s("combined_prune_rate = {}", result.combined_prune_rate);
    msginfo_s("slab_only_pruned_clusters = {}", result.slab_only_pruned_clusters);
    msginfo_s("pca_only_pruned_clusters = {}", result.pca_only_pruned_clusters);
    msginfo_s("combined_pruned_clusters = {}", result.combined_pruned_clusters);
    msginfo_s("cell_candidate_vectors = {}", result.cell_candidate_vectors);
    msginfo_s("cell_pruned_vectors = {}", result.cell_pruned_vectors);
    msginfo_s("cell_kept_vectors = {}", result.cell_kept_vectors);
    msginfo_s("zero_kept_cells = {}", result.zero_kept_cells);
    msginfo_s("zero_kept_surviving_clusters = {}", result.zero_kept_surviving_clusters);
    msginfo_s("cell_bytes_read = {}", result.cell_bytes_read);
    msginfo_s("cluster_bytes_baseline = {}", result.cluster_bytes_baseline);
    msginfo_s("io_bytes_reduction = {}", result.io_bytes_reduction);
    msginfo_s("cell_prune_rate = {}", result.cell_prune_rate);
  }
  msginfo_s("mean_keep_after_filter = {}", result.mean_keep_after_filter);
  msginfo_s("build_time = {}", result.build_time);
  msginfo_s("search_time = {}", result.search_time);
  msginfo_s("dist_comp = {}", result.dist_comp);
  msginfo_s("per_query_min = {}", result.per_query_min);
  msginfo_s("per_query_median = {}", result.per_query_median);
  msginfo_s("per_query_max = {}", result.per_query_max);
  msginfo_s("per_query_zero = {}", result.per_query_zero);
}
}  // namespace

int main(int argc, char** argv)
{
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
    index.build(config);
    RunResult result = index.search(config);
    print_run_result(result);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
