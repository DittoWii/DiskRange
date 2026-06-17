#include <exception>
#include <iostream>
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
  std::cerr << "Usage: ./build_prune_stats --config <path> --gt <path> "
               "[--no-cell-filter]"
            << std::endl;
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
  const size_t surviving_clusters =
      result.triangle_candidate_clusters >= result.triangle_pruned_clusters
          ? result.triangle_candidate_clusters - result.triangle_pruned_clusters
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
  std::cout << "per_query_min = " << result.per_query_min << '\n';
  std::cout << "per_query_median = " << result.per_query_median << '\n';
  std::cout << "per_query_max = " << result.per_query_max << '\n';
  std::cout << "per_query_zero = " << result.per_query_zero << '\n';
  std::cout << "build_time = " << result.build_time << '\n';
  std::cout << "search_time = " << result.search_time << '\n';
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
    const RunResult result = index.search(config);
    print_layers(result);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
