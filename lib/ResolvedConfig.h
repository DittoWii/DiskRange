#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ResolvedConfig is constructed exclusively by ConfigLoader and must not be
// mutated afterward.
struct ResolvedConfig
{
  size_t cluster_num = 0;
  size_t K = 0;
  float mem_budget = 0.0f;
  std::string mode;
  size_t gorder_window = 0;
  size_t io_queue_depth = 256;
  float cell_cache_share_pct = 0.0f;
  // Approximate geometric pruning radius divisor; invariant: finite and >= 1.0.
  float approx_alpha = 1.0f;
  size_t pca_rank = 32;
  size_t slab_count = 128;
  bool cell_filter_enabled = true;
  bool pca_cluster_prune_enabled = true;
  bool slab_cluster_prune_enabled = true;
  size_t target_cell_vecs = 24;
  std::string split_axis_policy = "fixed";
  std::string partition_objective = "standard";
  float eta = 1.0f;
  bool unit_normalize_prebuild = false;
  uint32_t vec_dtype = 0;

  std::string data_file;
  std::string query_file;
  std::string index_prefix;
  size_t n_base = 0;
  size_t n_queries = 0;
  size_t dim = 0;
  double radius_squared = 0.0;
  size_t total_in_range = 0;
  std::vector<size_t> per_query_counts;

  std::string cluster_path;
  std::string metadata_path;
  // Deprecated after index-graph-sqg: kept as a derived legacy path for old
  // index directories, but the Standard build/search path no longer writes or
  // consumes centroid HNSW.
  std::string hnsw_path;
  std::string data_hnsw_path;
  std::string data_hnsw_trace_path;
  std::string sqg_path;
  std::string rabitq_1bit_codes_path;
  std::string rabitq_8bit_codes_path;

  bool data_hnsw_enabled = false;
  size_t data_hnsw_M = 16;
  size_t data_hnsw_ef_construction = 100;
  size_t data_hnsw_ef = 100;
  float data_hnsw_margin_delta = 0.0f;

  bool fast_path_enabled = false;
  std::string fast_path_mode = "rbq";
  size_t fast_path_nprobe_N = 128;
  bool slow_path_rbq_prefilter_enabled = false;
  bool slow_path_rbq_phase_b_enabled = false;
  size_t sqg_m = 64;
  size_t sqg_num_iter = 3;
  size_t sqg_ef_build = 100;
  size_t sqg_ef_search = 300;
  // Build-time centroid assignment ef. Default 20 is the index-graph-sqg
  // measured-tuned value at d=96, K=50000, top-K=2 (search recall stays
  // >= 0.999 on deep10m; higher ef inflates km.assign_full_data wall
  // without recall benefit). Search-time/fast-path keeps sqg_ef_search.
  size_t sqg_centroid_ef_search = 20;

  ResolvedConfig() = default;

  std::string partition_suffix() const
  {
    if (partition_objective == "standard")
    {
      return unit_normalize_prebuild ? "std_norm" : "";
    }
    if (partition_objective == "anisotropic")
    {
      if (!unit_normalize_prebuild)
      {
        throw std::runtime_error(
            "anisotropic partition_objective requires unit_normalize_prebuild=true");
      }
      std::ostringstream tag;
      tag << "aniso_eta" << std::fixed << std::setprecision(3) << eta
          << "_norm";
      return tag.str();
    }
    throw std::runtime_error("unrecognized partition_objective '" +
                             partition_objective +
                             "'; must be 'standard' or 'anisotropic'");
  }

  static std::string suffixed_index_path(const std::string& path,
                                         const std::string& suffix)
  {
    if (suffix.empty())
    {
      return path;
    }
    const size_t slash = path.find_last_of("/\\");
    const std::string dir =
        slash == std::string::npos ? "" : path.substr(0, slash + 1);
    const std::string name =
        slash == std::string::npos ? path : path.substr(slash + 1);
    const std::string prefix = "_" + suffix + "_";
    if (name == "_cluster_file.bin") return dir + prefix + "cluster_file.bin";
    if (name == "_metadata_file.bin") return dir + prefix + "metadata_file.bin";
    if (name == "_hnsw_file.bin") return dir + prefix + "hnsw_file.bin";
    if (name == "_data_hnsw_file.bin")
      return dir + prefix + "data_hnsw_file.bin";
    if (name == "_data_hnsw_trace.csv")
      return dir + prefix + "data_hnsw_trace.csv";
    if (name == "_sqg_file.bin") return dir + prefix + "sqg_file.bin";
    if (name == "_rabitq_1bit_codes_file.bin")
      return dir + prefix + "rabitq_1bit_codes_file.bin";
    if (name == "_rabitq_8bit_codes_file.bin")
      return dir + prefix + "rabitq_8bit_codes_file.bin";
    return dir + prefix + name;
  }

  void apply_partition_suffix_to_paths()
  {
    const std::string suffix = partition_suffix();
    cluster_path = suffixed_index_path(cluster_path, suffix);
    metadata_path = suffixed_index_path(metadata_path, suffix);
    hnsw_path = suffixed_index_path(hnsw_path, suffix);
    data_hnsw_path = suffixed_index_path(data_hnsw_path, suffix);
    data_hnsw_trace_path = suffixed_index_path(data_hnsw_trace_path, suffix);
    sqg_path = suffixed_index_path(sqg_path, suffix);
    rabitq_1bit_codes_path =
        suffixed_index_path(rabitq_1bit_codes_path, suffix);
    rabitq_8bit_codes_path =
        suffixed_index_path(rabitq_8bit_codes_path, suffix);
  }
};
