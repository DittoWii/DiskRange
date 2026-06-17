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
  std::cerr << "Usage: ./approx_alpha_search_stats_prod --config <path> "
               "--gt <path> [--no-cell-filter]\n"
               "       ./approx_alpha_search_stats_prod --print-build-info"
            << std::endl;
}

void print_build_info()
{
  std::cout << "{\"binary_name\":\"approx_alpha_search_stats_prod\","
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

void print_production_result(const RunResult& result)
{
  std::cout << "recall_raw = " << result.recall_raw << '\n';
  std::cout << "recall_verified = " << result.recall_verified << '\n';
  std::cout << "qps = " << result.qps << " query/sec\n";
  std::cout << "search_time = " << result.search_time << '\n';
  std::cout << "dist_comp = " << result.dist_comp << '\n';
  std::cout << "radius_prescreened_clusters = "
            << result.radius_prescreened_clusters << '\n';
  std::cout << "slab_pruned_clusters = " << result.slab_pruned_clusters << '\n';
  std::cout << "box_residual_pruned_clusters = "
            << result.box_residual_pruned_clusters << '\n';
  std::cout << "data_hnsw_skip_count = " << result.data_hnsw_skip_count << '\n';
  std::cout << "fast_path_skip_count = " << result.fast_path_skip_count << '\n';
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
    print_production_result(result);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
