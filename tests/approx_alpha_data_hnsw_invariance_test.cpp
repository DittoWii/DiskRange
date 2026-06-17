#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>

#include <Eigen/Core>
#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#endif

#include "../lib/ConfigLoader.h"
#include "../lib/DiskRange.h"

int r_server_id = 0;

namespace
{
void require(bool cond, const std::string& message)
{
  if (!cond)
  {
    throw std::runtime_error(message);
  }
}

RunResult run_with_alpha(const ResolvedConfig& base, float alpha)
{
  ResolvedConfig config = base;
  config.approx_alpha = alpha;
  DiskRange index;
  return index.search(config);
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: approx_alpha_data_hnsw_invariance_test <config> <gt>\n";
    return 1;
  }

  Eigen::setNbThreads(1);
#ifdef DJ_USE_OPENBLAS
  openblas_set_num_threads(1);
#endif

  try
  {
    ResolvedConfig base = load_resolved_config(argv[1], argv[2]);
    require(base.data_hnsw_enabled, "test config must enable data_hnsw");
    base.mem_budget = std::max(base.mem_budget, 3.0f);

    const RunResult alpha1 = run_with_alpha(base, 1.0f);
    const RunResult alpha2 = run_with_alpha(base, 2.0f);

    require(alpha1.data_hnsw_skip_count == alpha2.data_hnsw_skip_count,
            "data_hnsw_skip_count changed between alpha=1.0 and alpha=2.0");
    require(alpha1.data_hnsw_trace.size() == alpha2.data_hnsw_trace.size(),
            "data_hnsw_trace size changed between alpha=1.0 and alpha=2.0");
    require(!alpha1.data_hnsw_trace.empty(), "data_hnsw_trace was empty");

    for (size_t i = 0; i < alpha1.data_hnsw_trace.size(); ++i)
    {
      const DataHNSWTraceRecord& lhs = alpha1.data_hnsw_trace[i];
      const DataHNSWTraceRecord& rhs = alpha2.data_hnsw_trace[i];
      require(lhs.query_id == rhs.query_id,
              "data_hnsw_trace query_id changed at index " + std::to_string(i));
      require(lhs.oracle_decision == rhs.oracle_decision,
              "data_hnsw_trace oracle_decision changed at query " +
                  std::to_string(lhs.query_id));
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
