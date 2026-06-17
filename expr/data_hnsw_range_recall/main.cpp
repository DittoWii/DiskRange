#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../lib/ConfigLoader.h"
#include "../../lib/DataHNSWOracle.h"
#include "../../utils/vector_cast.h"

int r_server_id = 0;

namespace
{
struct Args
{
  std::string config_path;
  std::string gt_path;
  std::string output_path;
};

struct GTData
{
  std::string dataset;
  std::vector<size_t> counts;
};

void print_usage()
{
  std::cerr << "Usage: ./data_hnsw_range_recall --config <path> --gt <path> "
               "--output <path>\n";
}

bool parse_args(int argc, char** argv, Args& args)
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
      {
        throw std::runtime_error(std::string(name) + " needs a value");
      }
      return argv[++i];
    };
    if (arg == "--config")
    {
      args.config_path = need_value("--config");
    }
    else if (arg == "--gt")
    {
      args.gt_path = need_value("--gt");
    }
    else if (arg == "--output")
    {
      args.output_path = need_value("--output");
    }
    else
    {
      return false;
    }
  }
  return !args.config_path.empty() && !args.gt_path.empty() &&
         !args.output_path.empty();
}

std::vector<float> read_queries(const ResolvedConfig& config)
{
  std::ifstream in(config.query_file, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("failed to open query .fbin file: " +
                             config.query_file);
  }
  const range_search_config::FbinHeader header =
      range_search_config::read_fbin_header(config.query_file, config.vec_dtype);
  if (header.n != config.n_queries || header.dim != config.dim)
  {
    throw std::runtime_error("query .fbin header mismatch for " +
                             config.query_file);
  }
  in.seekg(static_cast<std::streamoff>(header.payload_offset), std::ios::beg);
  std::vector<float> queries(config.n_queries * config.dim);
  if (header.dtype == range_search_config::kVecDTypeInt8)
  {
    std::vector<int8_t> raw(queries.size());
    in.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(int8_t)));
    if (in.gcount() !=
        static_cast<std::streamsize>(raw.size() * sizeof(int8_t)))
    {
      throw std::runtime_error("query .fbin file is shorter than expected: " +
                               config.query_file);
    }
    utils::cast_int8_to_float(raw.data(), raw.size(), queries.data());
  }
  else
  {
    in.read(reinterpret_cast<char*>(queries.data()),
            static_cast<std::streamsize>(queries.size() * sizeof(float)));
    if (in.gcount() !=
        static_cast<std::streamsize>(queries.size() * sizeof(float)))
    {
      throw std::runtime_error("query .fbin file is shorter than expected: " +
                               config.query_file);
    }
  }
  return queries;
}

GTData read_gt_data(const std::string& gt_path)
{
  std::ifstream in(gt_path);
  if (!in)
  {
    throw std::runtime_error("failed to open GT JSON file: " + gt_path);
  }
  nlohmann::json tree;
  in >> tree;
  return {range_search_config::require_value<std::string>(tree, "dataset"),
          range_search_config::read_per_query_counts(tree)};
}
}  // namespace

int main(int argc, char** argv)
{
  Args args;
  try
  {
    if (!parse_args(argc, argv, args))
    {
      print_usage();
      return 2;
    }

    ResolvedConfig config = load_resolved_config(args.config_path, args.gt_path);
    config.data_hnsw_enabled = true;
    const GTData gt = read_gt_data(args.gt_path);
    const std::vector<size_t>& gt_counts = gt.counts;
    if (gt_counts.size() != config.n_queries)
    {
      throw std::runtime_error("GT per_query_counts length mismatch");
    }
    const std::vector<float> queries = read_queries(config);
    auto oracle = load_data_hnsw(config.data_hnsw_path, config.dim,
                                 config.data_hnsw_ef);
    if (!oracle)
    {
      throw std::runtime_error("data HNSW oracle artifact is required");
    }

    const float radius_l2 =
        std::sqrt(static_cast<float>(config.radius_squared));
    size_t false_empty_count = 0;
    size_t false_empty_hit_count = 0;
    size_t skip_count = 0;
    const size_t total_gt_hits =
        std::accumulate(gt_counts.begin(), gt_counts.end(), size_t{0});
    nlohmann::json per_query = nlohmann::json::array();

    for (size_t q = 0; q < config.n_queries; ++q)
    {
      const DataHNSWOracleResult result = oracle->evaluate(
          queries.data() + q * config.dim, radius_l2,
          config.data_hnsw_margin_delta);
      if (result.is_empty)
      {
        ++skip_count;
        if (gt_counts[q] > 0)
        {
          ++false_empty_count;
          false_empty_hit_count += gt_counts[q];
        }
      }
      per_query.push_back({
          {"query_id", q},
          {"d_hat_l2", result.d_hat_l2},
          {"oracle_decision", result.is_empty},
          {"gt_count", gt_counts[q]},
          {"false_empty", result.is_empty && gt_counts[q] > 0},
      });
    }

    nlohmann::json out;
    out["dataset"] = gt.dataset;
    out["n_queries"] = config.n_queries;
    out["data_hnsw_path"] = config.data_hnsw_path;
    out["data_hnsw_ef"] = config.data_hnsw_ef;
    out["data_hnsw_margin_delta"] = config.data_hnsw_margin_delta;
    out["false_empty_count"] = false_empty_count;
    out["false_empty_hit_count"] = false_empty_hit_count;
    out["total_GT_hits"] = total_gt_hits;
    out["range_recall_loss"] =
        total_gt_hits == 0
            ? 0.0
            : false_empty_hit_count / static_cast<double>(total_gt_hits);
    out["skip_count"] = skip_count;
    out["skip_rate"] =
        config.n_queries == 0
            ? 0.0
            : skip_count / static_cast<double>(config.n_queries);
    out["per_query"] = std::move(per_query);

    const std::filesystem::path output_path(args.output_path);
    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty())
    {
      std::filesystem::create_directories(parent);
    }
    std::ofstream output(output_path);
    if (!output)
    {
      throw std::runtime_error("failed to open output path: " + args.output_path);
    }
    output << out.dump(2) << '\n';
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
