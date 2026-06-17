#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#include <Eigen/Core>
#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#endif

#include "../../lib/ConfigLoader.h"
#include "../../lib/DiskRange.h"

int r_server_id = 0;

namespace
{
void print_usage()
{
  std::cerr << "Usage: ./build --config <path> --gt <path>" << std::endl;
}

bool parse_args(int argc, char** argv, std::string& config_path,
                std::string& gt_path)
{
  bool saw_config = false;
  bool saw_gt = false;
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
    else
    {
      return false;
    }
  }
  return saw_config && saw_gt;
}
}  // namespace

int main(int argc, char** argv)
{
  std::string config_path;
  std::string gt_path;
  if (!parse_args(argc, argv, config_path, gt_path))
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
    const ResolvedConfig config = load_resolved_config(config_path, gt_path);
    DiskRange index;
    const auto start = std::chrono::steady_clock::now();
    index.build(config);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "build_time = "
              << std::chrono::duration<double>(end - start).count() << '\n';
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
