#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ConfigLoader.h"
#include "../lib/ConfigReader.h"
#include "../lib/DataReader.h"
#include "../utils/dist_func.h"
#include "../utils/utils.h"

namespace fs = std::filesystem;

namespace
{
constexpr uint32_t kFbinMagic = 0x31424A44u;

void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

void write_u32(std::ofstream& out, uint32_t value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  require(static_cast<bool>(out), "failed to write u32");
}

template <typename T>
void write_payload(std::ofstream& out, const std::vector<T>& values)
{
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  require(static_cast<bool>(out), "failed to write payload");
}

fs::path make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "int8_vector_pipeline_test";
  fs::remove_all(base);
  fs::create_directories(base);
  return base;
}

fs::path write_v1_float_fbin(const fs::path& dir, const std::string& name,
                             uint32_t n, uint32_t dim,
                             const std::vector<float>& values)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create v1 float fbin");
  write_u32(out, n);
  write_u32(out, dim);
  write_payload(out, values);
  return path;
}

fs::path write_v2_float_fbin(const fs::path& dir, const std::string& name,
                             uint32_t n, uint32_t dim,
                             const std::vector<float>& values)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create v2 float fbin");
  write_u32(out, kFbinMagic);
  write_u32(out, 0);
  write_u32(out, n);
  write_u32(out, dim);
  write_payload(out, values);
  return path;
}

fs::path write_v2_int8_fbin(const fs::path& dir, const std::string& name,
                            uint32_t n, uint32_t dim,
                            const std::vector<int8_t>& values)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create v2 int8 fbin");
  write_u32(out, kFbinMagic);
  write_u32(out, 1);
  write_u32(out, n);
  write_u32(out, dim);
  write_payload(out, values);
  return path;
}

fs::path write_v1_uint8_fbin(const fs::path& dir, const std::string& name,
                             uint32_t n, uint32_t dim,
                             const std::vector<uint8_t>& values)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create v1 uint8 fbin");
  write_u32(out, n);
  write_u32(out, dim);
  write_payload(out, values);
  return path;
}

fs::path write_v2_uint8_fbin(const fs::path& dir, const std::string& name,
                             uint32_t n, uint32_t dim,
                             const std::vector<uint8_t>& values)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create v2 uint8 fbin");
  write_u32(out, kFbinMagic);
  write_u32(out, 2);
  write_u32(out, n);
  write_u32(out, dim);
  write_payload(out, values);
  return path;
}

fs::path write_v2_reserved_dtype_fbin(const fs::path& dir,
                                      const std::string& name, uint32_t dtype,
                                      uint32_t n, uint32_t dim,
                                      size_t payload_bytes)
{
  const fs::path path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create reserved dtype fbin");
  write_u32(out, kFbinMagic);
  write_u32(out, dtype);
  write_u32(out, n);
  write_u32(out, dim);
  std::vector<uint8_t> zeros(payload_bytes, 0);
  write_payload(out, zeros);
  return path;
}

fs::path write_minimal_config(const fs::path& dir, const std::string& name,
                              const std::vector<std::string>& extra)
{
  const fs::path path = dir / name;
  std::ofstream out(path);
  require(static_cast<bool>(out), "failed to create minimal config");
  out << "cluster_num 4\n";
  out << "K 2\n";
  out << "mem_budget 1\n";
  out << "mode disk\n";
  out << "gorder_window 0\n";
  out << "pca_rank 2\n";
  out << "slab_count 1\n";
  out << "target_cell_vecs 2\n";
  for (const std::string& line : extra)
  {
    out << line << '\n';
  }
  return path;
}

fs::path write_gt_json(const fs::path& dir, const std::string& name,
                       const fs::path& base, const fs::path& query,
                       size_t n_base, size_t n_queries, size_t dim,
                       const std::string& vec_dtype)
{
  const fs::path path = dir / name;
  nlohmann::json tree;
  tree["dataset"] = "tiny_uint8";
  tree["metric"] = "l2";
  tree["n_queries"] = n_queries;
  tree["n_base"] = n_base;
  tree["dim"] = dim;
  tree["vec_dtype"] = vec_dtype;
  tree["radius_squared"] = 0.0;
  tree["total_in_range"] = 0;
  tree["per_query_counts"] = std::vector<size_t>(n_queries, 0);
  tree["per_query_count_stats"] = {
      {"min", 0}, {"median", 0.0}, {"max", 0}, {"zero_hit_queries", n_queries}};
  tree["index"] = {{"prefix", (dir / "tiny_uint8_index").string()}};
  tree["provenance"] = {
      {"base_path", base.string()},
      {"query_path", query.string()},
  };
  std::ofstream out(path);
  require(static_cast<bool>(out), "failed to create GT JSON");
  out << tree.dump(2);
  return path;
}

fs::path write_truncated_magic_collision_v1_fbin(const fs::path& dir)
{
  const fs::path path = dir / "magic_collision_v1.fbin";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create magic collision fbin");
  write_u32(out, kFbinMagic);
  write_u32(out, 99);
  std::vector<float> one(1, 1.0f);
  write_payload(out, one);
  return path;
}

fs::path copy_config_with_tail(const fs::path& dir,
                               const std::string& base_config,
                               const std::string& name,
                               const std::vector<std::string>& tail)
{
  const fs::path path = dir / name;
  std::ifstream in(base_config);
  require(static_cast<bool>(in), "failed to open base config");
  std::ofstream out(path);
  require(static_cast<bool>(out), "failed to create temp config");
  out << in.rdbuf();
  if (!tail.empty())
  {
    out << '\n';
  }
  for (const std::string& line : tail)
  {
    out << line << '\n';
  }
  return path;
}

int64_t brute_l2_i8(const std::vector<int8_t>& q,
                    const std::vector<int8_t>& b)
{
  require(q.size() == b.size(), "brute_l2_i8 size mismatch");
  int64_t sum = 0;
  for (size_t i = 0; i < q.size(); ++i)
  {
    const int64_t diff = static_cast<int64_t>(q[i]) - static_cast<int64_t>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

int64_t brute_l2_u8(const std::vector<uint8_t>& q,
                    const std::vector<uint8_t>& b, size_t dim)
{
  require(q.size() >= dim && b.size() >= dim, "brute_l2_u8 size mismatch");
  int64_t sum = 0;
  for (size_t i = 0; i < dim; ++i)
  {
    const int64_t diff = static_cast<int64_t>(q[i]) - static_cast<int64_t>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

int32_t sum_i8(const std::vector<int8_t>& values)
{
  int64_t sum = 0;
  for (int8_t value : values)
  {
    sum += value;
  }
  return static_cast<int32_t>(sum);
}

int32_t sum_u8(const std::vector<uint8_t>& values, size_t dim)
{
  require(values.size() >= dim, "sum_u8 size mismatch");
  int64_t sum = 0;
  for (size_t i = 0; i < dim; ++i)
  {
    sum += values[i];
  }
  return static_cast<int32_t>(sum);
}

int32_t norm_sq_i8(const std::vector<int8_t>& values)
{
  int64_t sum = 0;
  for (int8_t value : values)
  {
    const int64_t v = value;
    sum += v * v;
  }
  return static_cast<int32_t>(sum);
}

int32_t norm_sq_u8(const std::vector<uint8_t>& values, size_t dim)
{
  require(values.size() >= dim, "norm_sq_u8 size mismatch");
  int64_t sum = 0;
  for (size_t i = 0; i < dim; ++i)
  {
    const int64_t value = values[i];
    sum += value * value;
  }
  return static_cast<int32_t>(sum);
}

std::string read_text(const fs::path& path)
{
  std::ifstream in(path);
  require(static_cast<bool>(in), "failed to open text file: " + path.string());
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

void test_fbin_headers(const fs::path& dir)
{
  using range_search_config::read_fbin_header;
  const auto v1 = write_v1_float_fbin(dir, "legacy_float.fbin", 2, 3,
                                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  const auto v2_float =
      write_v2_float_fbin(dir, "v2_float.fbin", 2, 3,
                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  const auto v2_int8 =
      write_v2_int8_fbin(dir, "v2_int8.fbin", 2, 3, {-3, -2, -1, 1, 2, 3});
  const auto v1_uint8 =
      write_v1_uint8_fbin(dir, "legacy_uint8.u8bin", 2, 3,
                          {0, 1, 255, 200, 127, 64});
  const auto v2_uint8 =
      write_v2_uint8_fbin(dir, "v2_uint8.fbin", 2, 3,
                          {0, 1, 255, 200, 127, 64});
  const auto collision = write_truncated_magic_collision_v1_fbin(dir);

  auto h = read_fbin_header(v1.string());
  require(h.n == 2 && h.dim == 3 && h.dtype == 0 && h.payload_offset == 8 &&
              h.bytes_per_element == 4,
          "legacy float header mismatch");

  h = read_fbin_header(v2_float.string());
  require(h.n == 2 && h.dim == 3 && h.dtype == 0 && h.payload_offset == 16 &&
              h.bytes_per_element == 4,
          "v2 float header mismatch");

  h = read_fbin_header(v2_int8.string());
  require(h.n == 2 && h.dim == 3 && h.dtype == 1 && h.payload_offset == 16 &&
              h.bytes_per_element == 1,
          "v2 int8 header mismatch");

  h = read_fbin_header(v1_uint8.string(), 2);
  require(h.n == 2 && h.dim == 3 && h.dtype == 2 && h.payload_offset == 8 &&
              h.bytes_per_element == 1,
          "legacy uint8 header mismatch");

  h = read_fbin_header(v2_uint8.string());
  require(h.n == 2 && h.dim == 3 && h.dtype == 2 && h.payload_offset == 16 &&
              h.bytes_per_element == 1,
          "v2 uint8 header mismatch");

  bool corrupted_collision_rejected = false;
  try
  {
    (void)read_fbin_header(collision.string());
  }
  catch (const std::exception& e)
  {
    corrupted_collision_rejected =
        std::string(e.what()).find("neither valid v2 nor valid v1") !=
        std::string::npos;
  }
  require(corrupted_collision_rejected,
          "truncated magic-collision v1 fbin must be rejected");

  bool rejected = false;
  try
  {
    const auto reserved =
        write_v2_reserved_dtype_fbin(dir, "reserved_fp16.fbin", 3, 4, 5, 40);
    (void)read_fbin_header(reserved.string());
  }
  catch (const std::exception& e)
  {
    const std::string message = e.what();
    rejected = message.find("dtype=3") != std::string::npos ||
               message.find("fp16") != std::string::npos;
  }
  require(rejected, "v2 fp16 dtype must be rejected");
}

void test_data_reader_and_read_fbin(const fs::path& dir)
{
  const auto v2_int8 =
      write_v2_int8_fbin(dir, "reader_int8.fbin", 3, 4,
                         {-8, -7, -6, -5, 1, 2, 3, 4, 120, 121, 122, 123});

  DataReader reader(v2_int8.string());
  require(reader.n == 3 && reader.d == 4, "DataReader n/d mismatch");
  require(reader.dtype == 1 && reader.bytes_per_element == 1 &&
              reader.payload_offset == 16,
          "DataReader int8 layout metadata mismatch");

  std::vector<int8_t> row(4, 0);
  reader.get_batch_at(reinterpret_cast<char*>(row.data()), 1, 2);
  require((row == std::vector<int8_t>{120, 121, 122, 123}),
          "DataReader int8 get_batch_at read wrong row");

  unsigned n = 0;
  unsigned d = 0;
  int8_t* values = read_fbin<int8_t>(v2_int8.string().c_str(), n, d);
  require(n == 3 && d == 4, "read_fbin int8 n/d mismatch");
  require(values[0] == -8 && values[11] == 123,
          "read_fbin int8 payload mismatch");
  delete[] values;

  bool rejected = false;
  try
  {
    unsigned nf = 0;
    unsigned df = 0;
    float* bad = read_fbin<float>(v2_int8.string().c_str(), nf, df);
    delete[] bad;
  }
  catch (const std::exception& e)
  {
    rejected = std::string(e.what()).find("dtype") != std::string::npos;
  }
  require(rejected, "read_fbin<float> must reject int8 fbin");

  const auto v1_uint8 =
      write_v1_uint8_fbin(dir, "reader_legacy_uint8.u8bin", 2, 4,
                          {0, 1, 2, 255, 200, 201, 202, 203});
  const auto v2_uint8 =
      write_v2_uint8_fbin(dir, "reader_uint8.fbin", 2, 4,
                          {0, 1, 2, 255, 200, 201, 202, 203});

  DataReader u8_reader(v1_uint8.string(), 2);
  require(u8_reader.n == 2 && u8_reader.d == 4,
          "DataReader uint8 n/d mismatch");
  require(u8_reader.dtype == 2 && u8_reader.bytes_per_element == 1 &&
              u8_reader.payload_offset == 8,
          "DataReader uint8 layout metadata mismatch");

  std::vector<uint8_t> u8_row(4, 0);
  u8_reader.get_batch_at(reinterpret_cast<char*>(u8_row.data()), 1, 1);
  require((u8_row == std::vector<uint8_t>{200, 201, 202, 203}),
          "DataReader uint8 get_batch_at read wrong row");

  unsigned nu = 0;
  unsigned du = 0;
  uint8_t* u8_values = read_fbin<uint8_t>(v2_uint8.string().c_str(), nu, du);
  require(nu == 2 && du == 4, "read_fbin uint8 n/d mismatch");
  require(u8_values[0] == 0 && u8_values[3] == 255 && u8_values[7] == 203,
          "read_fbin uint8 payload mismatch");
  delete[] u8_values;

  rejected = false;
  try
  {
    unsigned nf = 0;
    unsigned df = 0;
    float* bad = read_fbin<float>(v2_uint8.string().c_str(), nf, df);
    delete[] bad;
  }
  catch (const std::exception& e)
  {
    rejected = std::string(e.what()).find("dtype") != std::string::npos;
  }
  require(rejected, "read_fbin<float> must reject uint8 fbin");
}

void test_config_vec_dtype(const fs::path& dir, const std::string& gt_path)
{
  const fs::path gt_abs = fs::absolute(gt_path);
  const fs::path repo_root =
      gt_abs.parent_path().parent_path().parent_path().parent_path();
  const std::string base_config =
      (repo_root / "configs" / "deep1m_disk.config").string();
  const fs::path default_config =
      copy_config_with_tail(dir, base_config, "default_float.config", {});
  const ResolvedConfig default_resolved =
      load_resolved_config(default_config.string(), gt_path);
  require(default_resolved.vec_dtype == 0, "missing vec_dtype must default to 0");

  const fs::path explicit_float =
      copy_config_with_tail(dir, base_config, "explicit_float.config",
                            {"vec_dtype 0"});
  const ResolvedConfig float_resolved =
      load_resolved_config(explicit_float.string(), gt_path);
  require(float_resolved.vec_dtype == 0, "vec_dtype 0 not propagated");

  const fs::path explicit_uint8_num =
      write_minimal_config(dir, "explicit_uint8_num.config", {"vec_dtype 2"});
  ConfigReader uint8_num_reader(explicit_uint8_num.string());
  require(uint8_num_reader.vec_dtype == 2, "vec_dtype 2 not parsed as uint8");

  const fs::path explicit_uint8_name =
      write_minimal_config(dir, "explicit_uint8_name.config",
                           {"vec_dtype uint8"});
  ConfigReader uint8_name_reader(explicit_uint8_name.string());
  require(uint8_name_reader.vec_dtype == 2,
          "vec_dtype uint8 not parsed as uint8");

  auto expect_reader_rejects = [&](const std::vector<std::string>& tail,
                                   const std::string& name,
                                   const std::string& needle) {
    const fs::path path = write_minimal_config(dir, name, tail);
    bool rejected = false;
    try
    {
      ConfigReader reader(path.string());
      (void)reader;
    }
    catch (const std::exception& e)
    {
      rejected = std::string(e.what()).find(needle) != std::string::npos;
    }
    require(rejected, "ConfigReader must reject " + name);
  };

  expect_reader_rejects({"vec_dtype 3"}, "invalid_dtype_3.config",
                        "vec_dtype");
  expect_reader_rejects({"vec_dtype fp16"}, "invalid_dtype_fp16.config",
                        "vec_dtype");
  expect_reader_rejects({"vec_dtype foo"}, "invalid_dtype_foo.config",
                        "vec_dtype");
  expect_reader_rejects({"vec_dtype 2abc"}, "invalid_dtype_partial.config",
                        "vec_dtype");
  expect_reader_rejects({"vec_dtype -1"}, "invalid_dtype_negative.config",
                        "vec_dtype");

  bool rejected = false;
  const fs::path normalize =
      copy_config_with_tail(dir, base_config, "int8_normalize.config",
                            {"vec_dtype 1", "unit_normalize_prebuild true"});
  try
  {
    ConfigReader reader(normalize.string());
    (void)reader;
  }
  catch (const std::exception& e)
  {
    rejected = std::string(e.what()).find("unit_normalize_prebuild") !=
               std::string::npos;
  }
  require(rejected, "int8 + unit_normalize_prebuild must be rejected");

  expect_reader_rejects({"vec_dtype uint8", "unit_normalize_prebuild true"},
                        "uint8_normalize.config",
                        "unit_normalize_prebuild");
  expect_reader_rejects({"vec_dtype uint8", "partition_objective anisotropic"},
                        "uint8_anisotropic.config",
                        "partition_objective");
  expect_reader_rejects({"vec_dtype uint8", "fast_path_mode exact"},
                        "uint8_exact.config", "fast_path_mode");

  const auto base_u8 = write_v2_uint8_fbin(dir, "config_base_uint8.fbin", 1, 4,
                                          {0, 1, 2, 3});
  const auto query_u8 =
      write_v2_uint8_fbin(dir, "config_query_uint8.fbin", 1, 4,
                          {4, 5, 6, 7});
  const fs::path gt_u8 =
      write_gt_json(dir, "tiny_uint8_gt.json", base_u8, query_u8, 1, 1, 4,
                    "uint8");
  const fs::path config_no_dtype =
      write_minimal_config(dir, "uint8_no_dtype.config", {});
  const ResolvedConfig uint8_resolved =
      load_resolved_config(config_no_dtype.string(), gt_u8.string());
  require(uint8_resolved.vec_dtype == 2,
          "GT JSON vec_dtype uint8 not propagated");

  auto expect_load_rejects = [&](const fs::path& gt, const fs::path& config,
                                 const std::string& needle,
                                 const std::string& label) {
    bool load_rejected = false;
    try
    {
      (void)load_resolved_config(config.string(), gt.string());
    }
    catch (const std::exception& e)
    {
      load_rejected = std::string(e.what()).find(needle) != std::string::npos;
    }
    require(load_rejected, "load_resolved_config must reject " + label);
  };

  expect_load_rejects(
      gt_u8,
      write_minimal_config(dir, "uint8_gt_normalize.config",
                           {"unit_normalize_prebuild true"}),
      "unit_normalize_prebuild", "GT uint8 + unit_normalize_prebuild");
  expect_load_rejects(
      gt_u8,
      write_minimal_config(dir, "uint8_gt_anisotropic.config",
                           {"partition_objective anisotropic"}),
      "partition_objective", "GT uint8 + anisotropic");
  expect_load_rejects(
      gt_u8,
      write_minimal_config(dir, "uint8_gt_exact.config",
                           {"fast_path_mode exact"}),
      "fast_path_mode", "GT uint8 + fast_path_mode exact");

  const auto high_dim_base =
      write_v2_uint8_fbin(dir, "config_base_uint8_dim40000.fbin", 1, 40000,
                          std::vector<uint8_t>(40000, 1));
  const auto high_dim_query =
      write_v2_uint8_fbin(dir, "config_query_uint8_dim40000.fbin", 1, 40000,
                          std::vector<uint8_t>(40000, 2));
  const fs::path high_dim_gt =
      write_gt_json(dir, "tiny_uint8_dim40000_gt.json", high_dim_base,
                    high_dim_query, 1, 1, 40000, "uint8");
  expect_load_rejects(high_dim_gt, config_no_dtype, "uint8 requires dim <= 33025",
                      "uint8 dim > 33025");
}

void test_int8_distance_static_dispatch_markers(const fs::path& repo_root)
{
  const std::string source =
      read_text(repo_root / "utils" / "dist_func.h");
  for (const std::string marker :
       {"L2SqrInt8VNNIInt64", "L2SqrInt8AVX512VNNI",
        "L2SqrInt8AVX512FallbackInt64", "L2SqrInt8AVX512Fallback",
        "__AVX512VNNI__", "__AVX512BW__", "_mm512_dpbusd_epi32"})
  {
    require(source.find(marker) != std::string::npos,
            "missing int8 distance dispatch marker: " + marker);
  }
}

void test_int8_distance_at_d(size_t d, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  std::vector<int8_t> q(d);
  std::vector<int8_t> b(d);
  for (size_t i = 0; i < d; ++i)
  {
    q[i] = static_cast<int8_t>(dist(rng));
    b[i] = static_cast<int8_t>(dist(rng));
  }
  const int64_t expected = brute_l2_i8(q, b);
  const int32_t norm_q = norm_sq_i8(q);
  const int32_t sum_b = sum_i8(b);
  const int32_t norm_b = norm_sq_i8(b);

  require(utils::L2SqrInt8ScalarInt64(q.data(), b.data(), d) == expected,
          "scalar int8 int64 L2 mismatch at d=" + std::to_string(d));
  require(utils::L2SqrInt8Scalar(q.data(), b.data(), d) ==
              static_cast<float>(expected),
          "scalar int8 public L2 mismatch at d=" + std::to_string(d));
  require(utils::L2SqrInt8SIMD(q.data(), b.data(), d, norm_q, sum_b, norm_b) ==
              static_cast<float>(expected),
          "int8 dispatch L2 mismatch at d=" + std::to_string(d));
}

void test_int8_distance_extremes(size_t d)
{
  std::vector<int8_t> q(d, static_cast<int8_t>(-128));
  std::vector<int8_t> b(d, static_cast<int8_t>(127));
  const int64_t expected = brute_l2_i8(q, b);
  require(utils::L2SqrInt8SIMD(q.data(), b.data(), d, norm_sq_i8(q),
                               sum_i8(b), norm_sq_i8(b)) ==
              static_cast<float>(expected),
          "int8 dispatch L2 mismatch for -128/127 extremes");
}

void test_int8_distance(const fs::path& repo_root)
{
  test_int8_distance_static_dispatch_markers(repo_root);
  for (size_t d : {size_t{1}, size_t{31}, size_t{33}, size_t{63}, size_t{64},
                   size_t{65}, size_t{127}, size_t{128}, size_t{129},
                   size_t{256}, size_t{1023}, size_t{1025}, size_t{4096},
                   size_t{65537}, size_t{65804}, size_t{131071}})
  {
    test_int8_distance_at_d(d, static_cast<uint32_t>(42 + d));
  }
  test_int8_distance_extremes(128);
}

void test_uint8_distance_static_dispatch_markers(const fs::path& repo_root)
{
  const std::string source =
      read_text(repo_root / "utils" / "dist_func.h");
  for (const std::string marker :
       {"L2SqrUint8ScalarInt64", "L2SqrUint8VNNIInt64",
        "L2SqrUint8AVX512VNNI", "L2SqrUint8AVX512FallbackInt64",
        "L2SqrUint8AVX512Fallback", "L2SqrUint8SIMDInt64",
        "L2SqrUint8SIMD",
        "_mm512_dpbusd_epi32", "_mm512_cvtepi32_epi64",
        "_mm512_reduce_add_epi64", "_mm512_maskz_loadu_epi8"})
  {
    require(source.find(marker) != std::string::npos,
            "missing uint8 distance dispatch marker: " + marker);
  }
}

void assert_uint8_distance_matches(const std::vector<uint8_t>& q,
                                   const std::vector<uint8_t>& b, size_t dim,
                                   const std::string& label)
{
  const int64_t expected = brute_l2_u8(q, b, dim);
  const int32_t norm_q = norm_sq_u8(q, dim);
  const int32_t sum_b = sum_u8(b, dim);
  const int32_t norm_b = norm_sq_u8(b, dim);

  require(utils::L2SqrUint8ScalarInt64(q.data(), b.data(), dim) == expected,
          "scalar uint8 int64 L2 mismatch: " + label);
  require(utils::L2SqrUint8Scalar(q.data(), b.data(), dim) ==
              static_cast<float>(expected),
          "scalar uint8 public L2 mismatch: " + label);
  require(utils::L2SqrUint8SIMDInt64(q.data(), b.data(), dim, norm_q, sum_b,
                                     norm_b) == expected,
          "uint8 dispatch int64 L2 mismatch: " + label);
  require(utils::L2SqrUint8SIMD(q.data(), b.data(), dim, norm_q, sum_b,
                                norm_b) == static_cast<float>(expected),
          "uint8 dispatch L2 mismatch: " + label);
#if defined(__AVX512VNNI__)
  require(utils::L2SqrUint8VNNIInt64(q.data(), b.data(), dim, norm_q, sum_b,
                                     norm_b) == expected,
          "uint8 VNNI int64 L2 mismatch: " + label);
  require(utils::L2SqrUint8AVX512VNNI(q.data(), b.data(), dim, norm_q, sum_b,
                                      norm_b) ==
              static_cast<float>(expected),
          "uint8 VNNI public L2 mismatch: " + label);
#endif
#if defined(__AVX512BW__)
  require(utils::L2SqrUint8AVX512FallbackInt64(q.data(), b.data(), dim) ==
              expected,
          "uint8 AVX512BW fallback int64 L2 mismatch: " + label);
  require(utils::L2SqrUint8AVX512Fallback(q.data(), b.data(), dim) ==
              static_cast<float>(expected),
          "uint8 AVX512BW fallback public L2 mismatch: " + label);
#endif
}

void test_uint8_distance_at_d(size_t d, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> q(d);
  std::vector<uint8_t> b(d);
  for (size_t i = 0; i < d; ++i)
  {
    q[i] = static_cast<uint8_t>(dist(rng));
    b[i] = static_cast<uint8_t>(dist(rng));
  }
  assert_uint8_distance_matches(q, b, d,
                                "random d=" + std::to_string(d));

  std::fill(q.begin(), q.end(), uint8_t{0});
  std::fill(b.begin(), b.end(), uint8_t{255});
  assert_uint8_distance_matches(q, b, d,
                                "all-zero/all-255 d=" + std::to_string(d));

  for (size_t i = 0; i < d; ++i)
  {
    q[i] = static_cast<uint8_t>((i % 2) ? 255 : 0);
    b[i] = static_cast<uint8_t>((i % 2) ? 0 : 255);
  }
  assert_uint8_distance_matches(q, b, d,
                                "alternating extremes d=" + std::to_string(d));
}

void test_uint8_distance_tail_garbage(size_t d)
{
  const size_t padded = 64;
  require(d < padded, "tail garbage test expects d < 64");
  std::vector<uint8_t> q(padded, 173);
  std::vector<uint8_t> b(padded, 91);
  for (size_t i = 0; i < d; ++i)
  {
    q[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
    b[i] = static_cast<uint8_t>((i * 29 + 7) & 0xFF);
  }
  assert_uint8_distance_matches(q, b, d,
                                "tail garbage d=" + std::to_string(d));
}

void test_uint8_distance_widening_oracle()
{
#if defined(__AVX512VNNI__)
  const size_t d = 132105;
  std::vector<uint8_t> q(d, 0);
  std::vector<uint8_t> b(d, 127);
  const int64_t expected = static_cast<int64_t>(127) * 127 * d;
  const int32_t norm_q = 0;
  const int32_t sum_b = static_cast<int32_t>(127 * d);
  const int32_t norm_b = static_cast<int32_t>(127 * 127 * d);
  require(utils::L2SqrUint8VNNIInt64(q.data(), b.data(), d, norm_q, sum_b,
                                     norm_b) == expected,
          "uint8 VNNI horizontal reduction must widen to int64");
#endif
}

void test_uint8_distance_threshold_precision()
{
  const size_t d = 33025;
  std::vector<uint8_t> q(d, 0);
  std::vector<uint8_t> b(d, 255);
  const int64_t expected = static_cast<int64_t>(255) * 255 * d;
  const int32_t norm_q = 0;
  const int32_t sum_b = static_cast<int32_t>(255 * d);
  const int32_t norm_b = static_cast<int32_t>(expected);
  const int64_t exact = utils::L2SqrUint8SIMDInt64(q.data(), b.data(), d,
                                                   norm_q, sum_b, norm_b);
  const float rounded = utils::L2SqrUint8SIMD(q.data(), b.data(), d, norm_q,
                                              sum_b, norm_b);
  const double radius_between_float_and_exact =
      static_cast<double>(rounded) + 0.5;
  require(exact == expected, "uint8 exact L2 threshold fixture mismatch");
  require(static_cast<double>(rounded) < static_cast<double>(exact),
          "uint8 threshold fixture must force float L2 to round down");
  require(!(static_cast<double>(exact) <= radius_between_float_and_exact),
          "exact uint8 integer L2 must reject radius below true distance");
  require(static_cast<double>(rounded) <= radius_between_float_and_exact,
          "float-rounded uint8 L2 would incorrectly accept this radius");
}

void test_uint8_distance(const fs::path& repo_root)
{
  test_uint8_distance_static_dispatch_markers(repo_root);
  for (size_t d : {size_t{1}, size_t{31}, size_t{65}, size_t{129},
                   size_t{255}, size_t{257}, size_t{33025}})
  {
    test_uint8_distance_at_d(d, static_cast<uint32_t>(777 + d));
  }
  for (size_t d : {size_t{1}, size_t{31}})
  {
    test_uint8_distance_tail_garbage(d);
  }
  test_uint8_distance_widening_oracle();
  test_uint8_distance_threshold_precision();
}

void test_uint8_diskrange_static_dispatch_markers(const fs::path& repo_root)
{
  const std::string source = read_text(repo_root / "lib" / "DiskRange.h");
  for (const std::string marker :
       {"is_byte_dtype(config.vec_dtype)",
        "is_byte_dtype(data_reader.dtype)",
        "kVecDTypeUint8Reserved",
        "L2SqrUint8SIMDInt64",
        "L2SqrUint8SIMD",
        "cast_uint8_to_float",
        "raw_byte",
        "norm_sq_byte",
        "config.radius_squared"})
  {
    require(source.find(marker) != std::string::npos,
            "missing uint8 DiskRange dispatch marker: " + marker);
  }
  require(source.find("data_hnsw byte-dtype float scratch exceeds memory budget") !=
              std::string::npos,
          "data_hnsw uint8 must share byte-dtype scratch preflight");
  require(source.find("dist <= config.radius_squared") != std::string::npos,
          "uint8 exact hit threshold must compare against double radius");
  require(source.find("static_cast<double>(utils::L2SqrUint8SIMDInt64") !=
              std::string::npos,
          "uint8 DiskRange exact/verify must compare int64 L2 against radius");
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: int8_vector_pipeline_test <deep1m_gt_json>\n";
    return 2;
  }

  try
  {
    const std::string gt_path = argv[1];
    require(fs::exists(gt_path), "GT JSON does not exist: " + gt_path);
    const fs::path gt_abs = fs::absolute(gt_path);
    const fs::path repo_root =
        gt_abs.parent_path().parent_path().parent_path().parent_path();
    const fs::path dir = make_temp_dir();
    test_fbin_headers(dir);
    test_data_reader_and_read_fbin(dir);
    test_config_vec_dtype(dir, gt_path);
    test_int8_distance(repo_root);
    test_uint8_distance(repo_root);
    test_uint8_diskrange_static_dispatch_markers(repo_root);
    fs::remove_all(dir);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
