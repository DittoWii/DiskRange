#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ConfigLoader.h"
#include "../lib/DataHNSWOracle.h"
#include "../lib/IndexPaths.h"
#include "../lib/data_hnsw_search.h"

int r_server_id = 0;

namespace fs = std::filesystem;

namespace
{
void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

std::string make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "data_hnsw_oracle_unit";
  fs::remove_all(base);
  fs::create_directories(base);
  return base.string();
}

std::unique_ptr<DataHNSWOracle> load_and_drop_local_scope(
    const std::string& path, size_t dim, size_t ef)
{
  return load_data_hnsw(path, dim, ef);
}

void test_config_defaults_and_paths(const std::string& config_path,
                                    const std::string& gt_path)
{
  const ResolvedConfig config = load_resolved_config(config_path, gt_path);
  require(!config.data_hnsw_enabled, "data_hnsw_enabled must default false");
  require(config.data_hnsw_M == 16, "default data_hnsw_M");
  require(config.data_hnsw_ef_construction == 100,
          "default data_hnsw_ef_construction");
  require(config.data_hnsw_ef == 100, "default data_hnsw_ef");
  require(config.data_hnsw_margin_delta == 0.0f,
          "default data_hnsw_margin_delta");
  require(config.data_hnsw_path.find("_data_hnsw_file.bin") != std::string::npos,
          "derived data_hnsw_path");
  require(config.data_hnsw_trace_path.find("_data_hnsw_trace.csv") !=
              std::string::npos,
          "derived data_hnsw_trace_path");

  const IndexPaths paths = make_diskrange_paths(config.index_prefix);
  require(paths.data_hnsw == config.data_hnsw_path,
          "IndexPaths data_hnsw must match resolved config");
}

void test_negative_margin_rejected(const std::string& config_path,
                                   const std::string& gt_path,
                                   const std::string& temp_dir)
{
  const fs::path bad_config = fs::path(temp_dir) / "bad_margin.config";
  {
    std::ifstream in(config_path);
    std::ofstream out(bad_config);
    out << in.rdbuf();
    out << "data_hnsw_margin_delta -0.001\n";
  }

  bool rejected = false;
  try
  {
    (void)load_resolved_config(bad_config.string(), gt_path);
  }
  catch (const std::exception& e)
  {
    rejected =
        std::string(e.what()).find("data_hnsw_margin_delta") != std::string::npos;
  }
  require(rejected, "negative data_hnsw_margin_delta must be rejected");
}

void test_data_hnsw_path_config_key_rejected(const std::string& config_path,
                                             const std::string& gt_path,
                                             const std::string& temp_dir)
{
  const fs::path bad_config = fs::path(temp_dir) / "bad_path.config";
  {
    std::ifstream in(config_path);
    std::ofstream out(bad_config);
    out << in.rdbuf();
    out << "data_hnsw_path /tmp/should_not_be_configured.bin\n";
  }

  bool rejected = false;
  try
  {
    (void)load_resolved_config(bad_config.string(), gt_path);
  }
  catch (const std::exception& e)
  {
    rejected =
        std::string(e.what()).find("unrecognized config key: data_hnsw_path") !=
        std::string::npos;
  }
  require(rejected, "data_hnsw_path must not be accepted as a config key");
}

void test_roundtrip_ownership_and_evaluate(const std::string& temp_dir)
{
  const size_t dim = 2;
  const std::vector<float> data = {
      0.0f, 0.0f,
      3.0f, 4.0f,
      10.0f, 0.0f,
      0.0f, 10.0f,
  };
  auto oracle = build_data_hnsw(data.data(), 4, dim, 8, 40);
  const fs::path path = fs::path(temp_dir) / "tiny_data_hnsw.bin";
  save_data_hnsw(*oracle->graph, path.string());

  auto loaded = load_and_drop_local_scope(path.string(), dim, 37);
  require(static_cast<size_t>(loaded->graph->ef_) == 37,
          "load_data_hnsw must set configured ef after loadIndex");

  const float q0[2] = {0.1f, 0.2f};
  const DataHNSWOracleResult near = loaded->evaluate(q0, 1.0f, 0.0f);
  require(!near.is_empty, "near query must not be skipped");
  require(near.d_hat_l2 < 1.0f, "near query d_hat_l2");

  const float q1[2] = {50.0f, 50.0f};
  const DataHNSWOracleResult far = loaded->evaluate(q1, 1.0f, 0.0f);
  require(far.is_empty, "far query must be skipped");
  require(far.d_hat_l2 > 1.0f, "far query d_hat_l2");

  const float raw_sq = data_hnsw_1nn_squared(*loaded->graph, q0);
  require(raw_sq >= 0.0f, "shared helper returns non-negative squared L2");

  for (size_t iter = 0; iter < 256; ++iter)
  {
    const float q[2] = {static_cast<float>(iter % 13) * 0.01f,
                        static_cast<float>(iter % 17) * 0.01f};
    const DataHNSWOracleResult loop_result = loaded->evaluate(q, 100.0f, 0.0f);
    require(!loop_result.is_empty,
            "loaded wrapper must own L2Space across repeated queries");
  }
}

void test_load_failures_warn_and_disable(const std::string& temp_dir)
{
  const fs::path missing = fs::path(temp_dir) / "missing_data_hnsw.bin";
  require(!load_data_hnsw(missing.string(), 2, 10),
          "missing artifact must disable oracle");

  const size_t dim = 2;
  const std::vector<float> data = {
      0.0f, 0.0f,
      1.0f, 1.0f,
  };
  auto oracle = build_data_hnsw(data.data(), 2, dim, 4, 20);
  const fs::path path = fs::path(temp_dir) / "dim_mismatch_data_hnsw.bin";
  save_data_hnsw(*oracle->graph, path.string());
  require(!load_data_hnsw(path.string(), 3, 10),
          "dimension mismatch must disable oracle");
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: data_hnsw_oracle_unit_test <config> <gt>\n";
    return 2;
  }

  try
  {
    const std::string temp_dir = make_temp_dir();
    test_config_defaults_and_paths(argv[1], argv[2]);
    test_negative_margin_rejected(argv[1], argv[2], temp_dir);
    test_data_hnsw_path_config_key_rejected(argv[1], argv[2], temp_dir);
    test_roundtrip_ownership_and_evaluate(temp_dir);
    test_load_failures_warn_and_disable(temp_dir);
    fs::remove_all(temp_dir);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
