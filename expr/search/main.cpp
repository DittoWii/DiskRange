#include <exception>
#include <fstream>
#include <iomanip>
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
  std::cerr << "Usage: ./search --config <path> --gt <path> "
               "[--no-cell-filter] [--dump-range-hits <csv>]"
            << std::endl;
}

bool parse_args(int argc, char** argv, std::string& config_path,
                std::string& gt_path, bool& no_cell_filter,
                std::string& dump_range_hits_path)
{
  bool saw_config = false;
  bool saw_gt = false;
  bool saw_dump_range_hits = false;
  no_cell_filter = false;
  dump_range_hits_path.clear();
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
    else if (arg == "--dump-range-hits")
    {
      if (saw_dump_range_hits || i + 1 >= argc ||
          std::string(argv[i + 1]).rfind("--", 0) == 0)
      {
        return false;
      }
      dump_range_hits_path = argv[++i];
      saw_dump_range_hits = true;
    }
    else
    {
      return false;
    }
  }
  return saw_config && saw_gt;
}

void print_minimal_result(const RunResult& result)
{
  std::cout << "recall = " << result.recall << '\n';
  std::cout << "recall_verified = " << result.recall_verified << '\n';
  std::cout << "search_time = " << result.search_time << '\n';
  std::cout << "qps = " << result.qps << " query/sec\n";
  std::cout << "dist_comp = " << result.dist_comp << '\n';
  std::cout << "per_query_zero = " << result.per_query_zero << '\n';
  std::cout << "fast_path_skip_count = " << result.fast_path_skip_count << '\n';
  std::cout << "fast_path_skip_rate = " << result.fast_path_skip_rate << '\n';
  std::cout << "false_empty_query_rate = "
            << result.false_empty_query_rate << '\n';
  std::cout << "range_recall_loss_hit_weighted = "
            << result.range_recall_loss_hit_weighted << '\n';
}

void write_verified_hit_dump(const RunResult& result, const std::string& path)
{
  if (path.empty())
  {
    return;
  }
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out)
  {
    throw std::runtime_error("cannot open range hit dump for writing: " + path);
  }
  out << "query_id,original_base_id,distance\n";
  out << std::setprecision(9);
  for (const VerifiedHit& hit : result.verified_hits)
  {
    out << hit.query_id << ',' << hit.original_base_id << ','
        << hit.distance << '\n';
  }
}
}  // namespace

int main(int argc, char** argv)
{
  std::string config_path;
  std::string gt_path;
  std::string dump_range_hits_path;
  bool no_cell_filter = false;
  if (!parse_args(argc, argv, config_path, gt_path, no_cell_filter,
                  dump_range_hits_path))
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
    print_minimal_result(result);
    write_verified_hit_dump(result, dump_range_hits_path);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
