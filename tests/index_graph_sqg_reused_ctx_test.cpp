#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ConfigLoader.h"
#include "../lib/SQGCentroidIndex.h"

int r_server_id = 0;

namespace
{
void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

struct Args
{
  std::string config_path;
  std::string gt_path;
  size_t query_limit = 100;
  size_t nprobe = 0;
};

void print_usage()
{
  std::cerr << "Usage: index_graph_sqg_reused_ctx_test "
               "[--config <path> --gt <path> --queries N --nprobe K]\n";
}

bool parse_args(int argc, char** argv, Args& args)
{
  bool saw_config = false;
  bool saw_gt = false;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    const auto need_value = [&](std::string& out) -> bool {
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
      {
        return false;
      }
      out = argv[++i];
      return true;
    };
    const auto need_size = [&](size_t& out) -> bool {
      std::string value;
      if (!need_value(value))
      {
        return false;
      }
      out = static_cast<size_t>(std::stoull(value));
      return true;
    };
    if (arg == "--config")
    {
      if (saw_config || !need_value(args.config_path))
      {
        return false;
      }
      saw_config = true;
    }
    else if (arg == "--gt")
    {
      if (saw_gt || !need_value(args.gt_path))
      {
        return false;
      }
      saw_gt = true;
    }
    else if (arg == "--queries")
    {
      if (!need_size(args.query_limit))
      {
        return false;
      }
    }
    else if (arg == "--nprobe")
    {
      if (!need_size(args.nprobe))
      {
        return false;
      }
    }
    else
    {
      return false;
    }
  }
  return (saw_config == saw_gt) && args.query_limit > 0;
}

std::vector<float> make_centroids(size_t cluster_count, size_t dim)
{
  std::vector<float> centroids(cluster_count * dim, 0.0f);
  for (size_t cid = 0; cid < cluster_count; ++cid)
  {
    for (size_t col = 0; col < dim; ++col)
    {
      const float phase =
          static_cast<float>((cid + 1) * (col + 3)) * 0.013f;
      centroids[cid * dim + col] =
          std::sin(phase) + 0.25f * std::cos(phase * 0.7f) +
          static_cast<float>(cid % 17) * 0.01f;
    }
  }
  return centroids;
}

std::vector<float> make_query(const std::vector<float>& centroids,
                              size_t cluster_count, size_t dim, size_t q)
{
  const size_t base_id = (q * 37 + 11) % cluster_count;
  std::vector<float> query(dim, 0.0f);
  for (size_t col = 0; col < dim; ++col)
  {
    const float noise =
        static_cast<float>((static_cast<int>((q + 1) * (col + 5)) % 23) - 11) *
        0.0007f;
    query[col] = centroids[base_id * dim + col] + noise;
  }
  return query;
}

std::vector<float> read_queries(const ResolvedConfig& config,
                                size_t query_limit, size_t& query_count)
{
  std::ifstream in(config.query_file, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open query file: " + config.query_file);
  }
  uint32_t file_n = 0;
  uint32_t file_dim = 0;
  in.read(reinterpret_cast<char*>(&file_n), sizeof(file_n));
  in.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
  if (!in || file_dim != config.dim)
  {
    throw std::runtime_error("query fbin header mismatch for " +
                             config.query_file);
  }
  query_count =
      std::min({query_limit, static_cast<size_t>(file_n), config.n_queries});
  std::vector<float> queries(query_count * config.dim);
  in.read(reinterpret_cast<char*>(queries.data()),
          static_cast<std::streamsize>(queries.size() * sizeof(float)));
  if (!in)
  {
    throw std::runtime_error("short query fbin payload: " + config.query_file);
  }
  return queries;
}

void check_reused_ctx_equivalence(const SQGCentroidIndex& sqg,
                                  const std::vector<float>& queries,
                                  size_t query_count, size_t dim,
                                  size_t nprobe)
{
  auto reused_ctx = sqg.make_search_context();
  std::vector<uint32_t> reused(nprobe, 0);

  for (size_t q = 0; q < query_count; ++q)
  {
    const float* query = queries.data() + q * dim;
    const std::vector<uint32_t> fresh = sqg.search(query, nprobe);
    sqg.search_into(query, static_cast<uint32_t>(nprobe), reused.data(),
                    reused_ctx);

    require(fresh.size() == reused.size(), "fresh/reused result size mismatch");
    for (size_t pos = 0; pos < nprobe; ++pos)
    {
      if (fresh[pos] != reused[pos])
      {
        std::cerr << "reused SearchContext mismatch at query " << q
                  << ", position " << pos << ": fresh=" << fresh[pos]
                  << " reused=" << reused[pos] << '\n';
        std::exit(1);
      }
    }
  }
}

void run_synthetic()
{
  constexpr size_t kClusterCount = 128;
  constexpr size_t kDim = 64;
  constexpr size_t kSqgM = 32;
  constexpr size_t kNumIter = 3;
  constexpr size_t kEfBuild = 80;
  constexpr size_t kEfSearch = 64;
  constexpr size_t kNprobe = 12;
  constexpr size_t kQueryCount = 100;

  const std::vector<float> centroids = make_centroids(kClusterCount, kDim);
  std::vector<float> queries(kQueryCount * kDim, 0.0f);
  for (size_t q = 0; q < kQueryCount; ++q)
  {
    const std::vector<float> query =
        make_query(centroids, kClusterCount, kDim, q);
    std::copy(query.begin(), query.end(), queries.begin() + q * kDim);
  }

  SQGCentroidIndex sqg;
  sqg.build(centroids, kClusterCount, kDim, kSqgM, kNumIter, kEfBuild,
            kEfSearch);
  check_reused_ctx_equivalence(sqg, queries, kQueryCount, kDim, kNprobe);
}

void run_dataset(const Args& args)
{
  const ResolvedConfig config =
      load_resolved_config(args.config_path, args.gt_path);
  size_t query_count = 0;
  const std::vector<float> queries =
      read_queries(config, args.query_limit, query_count);
  const size_t nprobe =
      args.nprobe == 0 ? std::min(config.K, config.cluster_num) : args.nprobe;
  require(nprobe > 0 && nprobe <= config.cluster_num,
          "nprobe must be in [1, cluster_num]");

  SQGCentroidIndex sqg;
  sqg.load(config.sqg_path, config.cluster_num, config.dim, config.sqg_m,
           config.sqg_ef_search);
  check_reused_ctx_equivalence(sqg, queries, query_count, config.dim, nprobe);
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    omp_set_num_threads(std::min(4, omp_get_max_threads()));

    Args args;
    if (!parse_args(argc, argv, args))
    {
      print_usage();
      return 2;
    }
    if (args.config_path.empty())
    {
      run_synthetic();
    }
    else
    {
      run_dataset(args);
    }

    std::cout << "index_graph_sqg reused-context equivalence passed\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
