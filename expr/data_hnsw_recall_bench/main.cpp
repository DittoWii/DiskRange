// Micro-bench: build a data-level HNSW with configurable hyperparams,
// measure (build time, storage, per-query latency, recall@1 vs cached d_q).
//
// Goal: identify the (M, ef, ef_construction) sweet spot for an
// "emptiness-oriented" HNSW where we only need recall@1 >= ~95% and want to
// minimize build time + storage + query latency.
//
// I/O contract:
//   --data <path.fbin>         data vectors (DiskRange fbin v1/v2)
//   --queries <path.fbin>      query vectors (same fbin)
//   --d-q <path.bin>           cached true 1-NN distance, raw f32 (n_queries floats)
//   --M <list>                 comma-separated M values (default: 8,16)
//   --ef-construction <list>   comma-separated ef_construction (default: 50,100,200)
//   --ef <list>                comma-separated ef-at-query (default: 5,10,20,50,100)
//   --threads <n>              OpenMP threads for build/search
//   --csv <path>               output CSV
//   --save-prefix <path>       optional, save each built index to <prefix>_M<M>_efc<E>.bin
//
// Recall@1 definition:
//   for each query q with cached true 1-NN distance d_q,
//   let d_hat = HNSW's returned best distance (top-1).
//   "correct" iff d_hat <= d_q * (1 + tol) where tol = 1e-4.
//   recall@1 = correct / n_queries.
//
// NOTE: Because hnswlib's searchKnn(size_t cluster_id, ...) takes a label
// (intended for re-querying an indexed point), this bench replicates the
// multi-level descent locally so it can query arbitrary external vectors.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include "../../lib/data_hnsw_search.h"
#include "../../lib/ConfigLoader.h"
#include "../../lib/hnswlib/hnswlib.h"
#include "../../utils/vector_cast.h"

namespace fs = std::filesystem;

namespace
{
struct Args
{
  std::string data_path;
  std::string queries_path;
  std::string d_q_path;
  std::string csv_path;
  std::string save_prefix;
  std::vector<size_t> M_grid;
  std::vector<size_t> efc_grid;
  std::vector<size_t> ef_grid;
  int threads = 0;
};

std::vector<size_t> parse_uint_list(const std::string& s)
{
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    if (!tok.empty()) out.push_back(static_cast<size_t>(std::stoul(tok)));
  }
  return out;
}

void print_usage()
{
  std::cerr
      << "Usage: data_hnsw_recall_bench --data X.fbin --queries Q.fbin "
         "--d-q D.bin [--M 8,16] [--ef-construction 50,100,200] "
         "[--ef 5,10,20,50,100] [--threads N] [--csv OUT.csv] "
         "[--save-prefix PFX]"
      << std::endl;
}

bool parse_args(int argc, char** argv, Args& a)
{
  a.M_grid = {8, 16};
  a.efc_grid = {50, 100, 200};
  a.ef_grid = {5, 10, 20, 50, 100};
  for (int i = 1; i < argc; ++i)
  {
    std::string k = argv[i];
    auto need_arg = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
      return std::string(argv[++i]);
    };
    if (k == "--data")
      a.data_path = need_arg("--data");
    else if (k == "--queries")
      a.queries_path = need_arg("--queries");
    else if (k == "--d-q")
      a.d_q_path = need_arg("--d-q");
    else if (k == "--csv")
      a.csv_path = need_arg("--csv");
    else if (k == "--save-prefix")
      a.save_prefix = need_arg("--save-prefix");
    else if (k == "--M")
      a.M_grid = parse_uint_list(need_arg("--M"));
    else if (k == "--ef-construction")
      a.efc_grid = parse_uint_list(need_arg("--ef-construction"));
    else if (k == "--ef")
      a.ef_grid = parse_uint_list(need_arg("--ef"));
    else if (k == "--threads")
      a.threads = std::stoi(need_arg("--threads"));
    else
    {
      std::cerr << "unknown arg: " << k << std::endl;
      return false;
    }
  }
  return !a.data_path.empty() && !a.queries_path.empty() && !a.d_q_path.empty();
}

// Read a DiskRange-format fbin and return a float view. Int8 payloads are cast
// exactly to float for the HNSW benchmark. `expected_v1_dtype` is the v1-
// fallback hint for legacy fbin files without v2 magic; default float32
// preserves the historical behavior.
std::vector<float> read_fbin(
    const std::string& path, size_t& n, size_t& dim,
    uint32_t expected_v1_dtype = range_search_config::kVecDTypeFloat32)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  const range_search_config::FbinHeader header =
      range_search_config::read_fbin_header(path, expected_v1_dtype);
  n = header.n;
  dim = header.dim;
  std::vector<float> buf(static_cast<size_t>(n) * dim);
  f.seekg(static_cast<std::streamoff>(header.payload_offset), std::ios::beg);
  if (header.dtype == range_search_config::kVecDTypeInt8)
  {
    std::vector<int8_t> raw(buf.size());
    f.read(reinterpret_cast<char*>(raw.data()),
           static_cast<std::streamsize>(raw.size() * sizeof(int8_t)));
    if (!f) throw std::runtime_error("short int8 fbin read: " + path);
    utils::cast_int8_to_float(raw.data(), raw.size(), buf.data());
  }
  else
  {
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(float)));
    if (!f) throw std::runtime_error("short fbin read: " + path);
  }
  return buf;
}

// Read raw f32 array.
std::vector<float> read_f32_bin(const std::string& path, size_t expected_n = 0)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  size_t nbytes = f.tellg();
  f.seekg(0, std::ios::beg);
  size_t n = nbytes / sizeof(float);
  if (expected_n != 0 && n != expected_n)
  {
    throw std::runtime_error("d_q file size mismatch: expected " +
                             std::to_string(expected_n) + " floats, got " +
                             std::to_string(n));
  }
  std::vector<float> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()),
         static_cast<std::streamsize>(nbytes));
  return buf;
}

double now_sec()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char** argv)
{
  Args a;
  if (!parse_args(argc, argv, a))
  {
    print_usage();
    return 1;
  }

  if (a.threads > 0) omp_set_num_threads(a.threads);

  std::cout << "[load] data=" << a.data_path << " queries=" << a.queries_path
            << " d_q=" << a.d_q_path << std::endl;
  double t0 = now_sec();
  size_t n_data, dim_data;
  std::vector<float> data = read_fbin(a.data_path, n_data, dim_data);
  std::cout << "  data shape: " << n_data << " x " << dim_data
            << "  (loaded in " << (now_sec() - t0) << "s)" << std::endl;

  t0 = now_sec();
  size_t n_q, dim_q;
  std::vector<float> queries = read_fbin(a.queries_path, n_q, dim_q);
  std::cout << "  queries shape: " << n_q << " x " << dim_q
            << "  (loaded in " << (now_sec() - t0) << "s)" << std::endl;
  if (dim_q != dim_data)
  {
    std::cerr << "dim mismatch: data " << dim_data << " vs queries " << dim_q
              << std::endl;
    return 1;
  }
  const size_t dim = dim_data;

  std::vector<float> d_q = read_f32_bin(a.d_q_path, n_q);

  std::ofstream csv;
  if (!a.csv_path.empty())
  {
    csv.open(a.csv_path);
    csv << "n_data,dim,M,ef_construction,ef,build_sec,build_threads,"
           "storage_bytes,query_total_sec,query_us_per_q,recall_at_1\n";
  }

  for (size_t M : a.M_grid)
  {
    for (size_t efc : a.efc_grid)
    {
      std::cout << "\n[build] M=" << M << " ef_construction=" << efc
                << " threads=" << omp_get_max_threads() << std::endl;
      hnswlib::L2Space space(dim);
      auto graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          &space, n_data, M, efc, /*random_seed=*/100);

      double tb0 = now_sec();
      // First point — must be sequential to set enterpoint.
      graph->addPoint(static_cast<const void*>(data.data() + 0), 0);
      // Remaining points — parallel.
#pragma omp parallel for schedule(static)
      for (size_t i = 1; i < n_data; ++i)
      {
        graph->addPoint(
            static_cast<const void*>(data.data() + i * dim),
            static_cast<hnswlib::labeltype>(i));
      }
      double build_sec = now_sec() - tb0;
      std::cout << "  built in " << build_sec << "s" << std::endl;

      // Storage estimate: write to a temp file (or save_prefix), measure.
      std::string idx_path;
      if (!a.save_prefix.empty())
      {
        idx_path = a.save_prefix + "_M" + std::to_string(M) + "_efc" +
                   std::to_string(efc) + ".bin";
      }
      else
      {
        idx_path = "/tmp/data_hnsw_bench_M" + std::to_string(M) + "_efc" +
                   std::to_string(efc) + ".bin";
      }
      graph->saveIndex(idx_path);
      size_t storage = static_cast<size_t>(fs::file_size(idx_path));
      std::cout << "  storage: " << (storage / (1024.0 * 1024.0)) << " MiB"
                << std::endl;

      // For each ef-at-query, sweep recall.
      for (size_t ef : a.ef_grid)
      {
        graph->setEf(ef);
        size_t correct = 0;
        double qt0 = now_sec();
#pragma omp parallel for reduction(+ : correct) schedule(dynamic, 64)
        for (size_t q = 0; q < n_q; ++q)
        {
          float d_hat_sq =
              data_hnsw_1nn_squared(*graph, queries.data() + q * dim);
          float d_hat = std::sqrt(std::max(d_hat_sq, 0.0f));
          float d_true = d_q[q];
          // Tolerance for fp accumulation.
          if (d_hat <= d_true * 1.0001f + 1e-6f) correct++;
        }
        double qt = now_sec() - qt0;
        double recall = static_cast<double>(correct) / n_q;
        double us_per_q = qt * 1e6 / static_cast<double>(n_q);
        std::cout << "  ef=" << ef << "  q_total=" << qt << "s  "
                  << us_per_q << " us/q  recall@1=" << recall << std::endl;
        if (csv.is_open())
        {
          csv << n_data << "," << dim << "," << M << "," << efc << "," << ef
              << "," << build_sec << "," << omp_get_max_threads() << ","
              << storage << "," << qt << "," << us_per_q << "," << recall
              << "\n";
          csv.flush();
        }
      }

      // Cleanup temp index file (only if we wrote to /tmp).
      if (a.save_prefix.empty())
      {
        std::error_code ec;
        fs::remove(idx_path, ec);
      }
    }
  }

  std::cout << "\n[done] csv=" << a.csv_path << std::endl;
  return 0;
}
