#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ConfigReader.h"
#include "IndexPaths.h"
#include "IoQueueDepth.h"
#include "ResolvedConfig.h"
#include "VecDType.h"

namespace range_search_config
{
using json = nlohmann::json;

struct FbinHeader
{
  size_t n = 0;
  size_t dim = 0;
  uint32_t dtype = 0;
  size_t payload_offset = 8;
  size_t bytes_per_element = sizeof(float);
};

constexpr uint32_t kFbinMagicDjb1 = 0x31424A44u;
constexpr size_t kFbinV1HeaderBytes = 8;
constexpr size_t kFbinV2HeaderBytes = 16;

inline std::string format_pair(size_t n, size_t dim)
{
  return "(" + std::to_string(n) + ", " + std::to_string(dim) + ")";
}

inline bool checked_payload_file_size(size_t header_bytes, uint32_t n,
                                      uint32_t dim, size_t bytes_per_element,
                                      size_t file_size)
{
  if (n == 0 || dim == 0)
  {
    return false;
  }
  constexpr uint32_t kMaxReasonableU32Dim = 1u << 31;
  if (n > kMaxReasonableU32Dim || dim > kMaxReasonableU32Dim)
  {
    return false;
  }
  const size_t max = std::numeric_limits<size_t>::max();
  const size_t n_size = static_cast<size_t>(n);
  const size_t dim_size = static_cast<size_t>(dim);
  if (n_size > max / dim_size)
  {
    return false;
  }
  const size_t elements = n_size * dim_size;
  if (elements > (max - header_bytes) / bytes_per_element)
  {
    return false;
  }
  return header_bytes + elements * bytes_per_element == file_size;
}

// `expected_v1_dtype` is the v1-fallback dtype hint. It ONLY affects v1
// headers (no DJB1 magic), where legacy .i8bin-style files
// ([u32 n][u32 dim][int8 payload]) need an external hint to compute the
// correct payload size.
//   - kVecDTypeFloat32 (default): preserves the historical spec contract that
//     v1 fbin is implicitly float32.
//   - kVecDTypeInt8: v1 payload is interpreted as 1 byte/element.
// v2-magic files are ALWAYS parsed by their self-described dtype byte; the
// hint is ignored for v2. If callers also want a strict dtype check on the
// returned header, use validate_fbin_header(path, n, dim, dtype) which
// performs that check after parsing.
inline FbinHeader read_fbin_header(
    const std::string& path,
    uint32_t expected_v1_dtype = kVecDTypeFloat32)
{
  if (!is_supported_vec_dtype(expected_v1_dtype))
  {
    throw std::runtime_error(
        "read_fbin_header: expected_v1_dtype must be one of 0 (float32), "
        "1 (int8), or 2 (uint8); "
        "got " + std::to_string(expected_v1_dtype));
  }

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
  {
    throw std::runtime_error("failed to open .fbin file: " + path);
  }
  const std::streamoff end_pos = in.tellg();
  if (end_pos < static_cast<std::streamoff>(kFbinV1HeaderBytes))
  {
    throw std::runtime_error("fbin header corrupted: file too small: " + path);
  }
  const size_t file_size = static_cast<size_t>(end_pos);
  in.seekg(0, std::ios::beg);

  uint32_t word0 = 0;
  uint32_t word1 = 0;
  in.read(reinterpret_cast<char*>(&word0), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&word1), sizeof(uint32_t));
  if (!in)
  {
    throw std::runtime_error("failed to read .fbin header: " + path);
  }

  if (word0 == kFbinMagicDjb1 && is_fbin_dtype_enum_value(word1) &&
      file_size >= kFbinV2HeaderBytes)
  {
    uint32_t n = 0;
    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    if (!in)
    {
      throw std::runtime_error("failed to read v2 .fbin header: " + path);
    }
    const size_t bytes_per_element = bytes_per_element_for_fbin_dtype(word1);
    const bool v2_self_consistent = checked_payload_file_size(
        kFbinV2HeaderBytes, n, dim, bytes_per_element, file_size);
    if (v2_self_consistent)
    {
      if (!is_supported_vec_dtype(word1))
      {
        throw std::runtime_error(
            "v2 fbin dtype=" + std::to_string(word1) + " (" +
            vec_dtype_to_string(word1) +
            ") is reserved but not implemented; supported dtypes are {0, 1, 2}: " +
            path);
      }
      return {static_cast<size_t>(n), static_cast<size_t>(dim), word1,
              kFbinV2HeaderBytes, bytes_per_element};
    }
  }

  const size_t v1_bytes_per_element =
      bytes_per_element_for_fbin_dtype(expected_v1_dtype);
  if (!checked_payload_file_size(kFbinV1HeaderBytes, word0, word1,
                                 v1_bytes_per_element, file_size))
  {
    throw std::runtime_error(
        "fbin header corrupted: neither valid v2 nor valid v1 .fbin layout "
        "for " +
        path + ": v1 n=" + std::to_string(word0) +
        ", dim=" + std::to_string(word1) +
        ", file_size=" + std::to_string(file_size) +
        ", expected_v1_dtype=" + vec_dtype_to_string(expected_v1_dtype) +
        " (if this is a legacy v1 fbin with byte payload, set GT JSON "
        "vec_dtype=int8/uint8 or .config vec_dtype 1/2)");
  }

  return {static_cast<size_t>(word0), static_cast<size_t>(word1),
          expected_v1_dtype, kFbinV1HeaderBytes, v1_bytes_per_element};
}

// `expected_dtype` is the resolved authoritative dtype (from GT JSON
// vec_dtype). It is BOTH passed to read_fbin_header as the v1 fallback
// hint AND checked strictly against the parsed header dtype. Callers must
// always know the authoritative dtype before calling this function.
inline void validate_fbin_header(const std::string& path, size_t expected_n,
                                 size_t expected_dim,
                                 uint32_t expected_dtype)
{
  const FbinHeader actual = read_fbin_header(path, expected_dtype);
  if (actual.n != expected_n || actual.dim != expected_dim)
  {
    throw std::runtime_error("fbin header mismatch for " + path +
                             ": expected " +
                             format_pair(expected_n, expected_dim) +
                             ", actual " + format_pair(actual.n, actual.dim));
  }
  if (actual.dtype != expected_dtype)
  {
    throw std::runtime_error(
        "fbin dtype mismatch for " + path + ": expected " +
        std::to_string(expected_dtype) + " (" +
        vec_dtype_to_string(expected_dtype) + "), actual " +
        std::to_string(actual.dtype) + " (" +
        vec_dtype_to_string(actual.dtype) + ")");
  }
}

inline const json& require_child(const json& tree, const std::string& path)
{
  const json* current = &tree;
  size_t start = 0;
  while (start <= path.size())
  {
    const size_t end = path.find('.', start);
    const std::string key = path.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    if (!current->is_object() || !current->contains(key))
    {
      throw std::runtime_error("missing required JSON field: " + path);
    }
    current = &current->at(key);
    if (end == std::string::npos)
    {
      break;
    }
    start = end + 1;
  }
  return *current;
}

template <typename T>
inline T require_value(const json& tree, const std::string& path)
{
  const json& value = require_child(tree, path);
  try
  {
    return value.get<T>();
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error("invalid JSON field " + path + ": " + e.what());
  }
}

inline std::vector<size_t> read_per_query_counts(const json& tree)
{
  const auto& counts_json = require_child(tree, "per_query_counts");
  if (!counts_json.is_array())
  {
    throw std::runtime_error(
        "invalid JSON field per_query_counts: expected array");
  }

  std::vector<size_t> counts;
  counts.reserve(counts_json.size());
  for (const auto& item : counts_json)
  {
    counts.push_back(item.get<size_t>());
  }
  return counts;
}

inline void validate_per_query_counts(const std::vector<size_t>& counts,
                                      size_t n_queries, size_t total_in_range,
                                      const json& tree)
{
  if (counts.size() != n_queries)
  {
    throw std::runtime_error("per_query_counts length mismatch: expected " +
                             std::to_string(n_queries) + ", actual " +
                             std::to_string(counts.size()));
  }

  const size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  if (sum != total_in_range)
  {
    throw std::runtime_error("per_query_counts sum " + std::to_string(sum) +
                             " does not equal total_in_range " +
                             std::to_string(total_in_range));
  }

  const size_t expected_min =
      require_value<size_t>(tree, "per_query_count_stats.min");
  const double expected_median =
      require_value<double>(tree, "per_query_count_stats.median");
  const size_t expected_max =
      require_value<size_t>(tree, "per_query_count_stats.max");
  const size_t expected_zero =
      require_value<size_t>(tree, "per_query_count_stats.zero_hit_queries");

  std::vector<size_t> sorted = counts;
  std::sort(sorted.begin(), sorted.end());
  const size_t actual_min = sorted.empty() ? 0 : sorted.front();
  const size_t actual_max = sorted.empty() ? 0 : sorted.back();
  const size_t actual_zero =
      static_cast<size_t>(std::count(counts.begin(), counts.end(), size_t{0}));
  const double actual_median =
      sorted.empty()
          ? 0.0
          : (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2.0;

  if (actual_min != expected_min)
  {
    throw std::runtime_error("per_query_count_stats.min mismatch: expected " +
                             std::to_string(expected_min) + ", actual " +
                             std::to_string(actual_min));
  }
  if (std::fabs(actual_median - expected_median) > 1e-9)
  {
    throw std::runtime_error(
        "per_query_count_stats.median mismatch: expected " +
        std::to_string(expected_median) + ", actual " +
        std::to_string(actual_median));
  }
  if (actual_max != expected_max)
  {
    throw std::runtime_error("per_query_count_stats.max mismatch: expected " +
                             std::to_string(expected_max) + ", actual " +
                             std::to_string(actual_max));
  }
  if (actual_zero != expected_zero)
  {
    throw std::runtime_error(
        "per_query_count_stats.zero_hit_queries mismatch: expected " +
        std::to_string(expected_zero) + ", actual " +
        std::to_string(actual_zero));
  }
}
}  // namespace range_search_config

inline ResolvedConfig load_resolved_config(const std::string& config_path,
                                           const std::string& gt_json_path)
{
  ConfigReader config_reader(config_path);

  std::ifstream json_input(gt_json_path);
  if (!json_input)
  {
    throw std::runtime_error("failed to open GT JSON file: " + gt_json_path);
  }

  nlohmann::json tree;
  try
  {
    json_input >> tree;
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error("failed to parse GT JSON file " + gt_json_path +
                             ": " + e.what());
  }

  ResolvedConfig resolved;
  resolved.cluster_num = config_reader.cluster_num;
  resolved.K = config_reader.K;
  resolved.mem_budget = config_reader.mem_budget;
  resolved.mode = config_reader.mode;
  resolved.gorder_window = config_reader.gorder_window;
  resolved.io_queue_depth = config_reader.io_queue_depth;
  resolved.cell_cache_share_pct = config_reader.cell_cache_share_pct;
  resolved.approx_alpha = config_reader.approx_alpha;
  resolved.pca_rank = config_reader.pca_rank;
  resolved.slab_count = config_reader.slab_count;
  resolved.cell_filter_enabled = config_reader.cell_filter_enabled;
  resolved.pca_cluster_prune_enabled =
      config_reader.pca_cluster_prune_enabled;
  resolved.slab_cluster_prune_enabled =
      config_reader.slab_cluster_prune_enabled;
  resolved.target_cell_vecs = config_reader.target_cell_vecs;
  resolved.split_axis_policy = config_reader.split_axis_policy;
  resolved.partition_objective = config_reader.partition_objective;
  resolved.eta = config_reader.eta;
  resolved.unit_normalize_prebuild = config_reader.unit_normalize_prebuild;
  resolved.vec_dtype = config_reader.vec_dtype;
  resolved.data_hnsw_enabled = config_reader.data_hnsw_enabled;
  resolved.data_hnsw_M = config_reader.data_hnsw_M;
  resolved.data_hnsw_ef_construction = config_reader.data_hnsw_ef_construction;
  resolved.data_hnsw_ef = config_reader.data_hnsw_ef;
  resolved.data_hnsw_margin_delta = config_reader.data_hnsw_margin_delta;
  resolved.fast_path_enabled = config_reader.fast_path_enabled;
  resolved.fast_path_mode = config_reader.fast_path_mode;
  resolved.fast_path_nprobe_N = config_reader.fast_path_nprobe_N;
  resolved.slow_path_rbq_prefilter_enabled =
      config_reader.slow_path_rbq_prefilter_enabled;
  resolved.slow_path_rbq_phase_b_enabled =
      config_reader.slow_path_rbq_phase_b_enabled;
  resolved.sqg_m = config_reader.sqg_m;
  resolved.sqg_num_iter = config_reader.sqg_num_iter;
  resolved.sqg_ef_build = config_reader.sqg_ef_build;
  resolved.sqg_ef_search = config_reader.sqg_ef_search;
  resolved.sqg_centroid_ef_search = config_reader.sqg_centroid_ef_search;

  validate_io_queue_depth(resolved.io_queue_depth);
  if (!std::isfinite(resolved.cell_cache_share_pct) ||
      resolved.cell_cache_share_pct < 0.0f ||
      resolved.cell_cache_share_pct > 100.0f)
  {
    throw std::runtime_error(
        "cell_cache_share_pct must be a finite value in [0, 100]; got " +
        std::to_string(resolved.cell_cache_share_pct));
  }
  if (resolved.fast_path_enabled && resolved.data_hnsw_enabled)
  {
    throw std::runtime_error(
        "fast_path_enabled and data_hnsw_enabled cannot both be true");
  }
  if (!std::isfinite(resolved.approx_alpha) || resolved.approx_alpha < 1.0f)
  {
    throw std::runtime_error("approx_alpha must be finite and >= 1.0; got " +
                             std::to_string(resolved.approx_alpha));
  }
  if (resolved.fast_path_enabled && resolved.partition_objective != "standard")
  {
    throw std::runtime_error(
        "fast_path_enabled requires partition_objective=standard; current "
        "partition_objective=" +
        resolved.partition_objective);
  }
  if (resolved.fast_path_mode != "rbq" && resolved.fast_path_mode != "exact")
  {
    throw std::runtime_error("fast_path_mode must be one of rbq/exact: " +
                             resolved.fast_path_mode);
  }
  if (resolved.fast_path_enabled &&
      resolved.fast_path_nprobe_N > resolved.cluster_num)
  {
    throw std::runtime_error(
        "fast_path_nprobe_N must be <= cluster_num when fast_path_enabled=true: " +
        std::to_string(resolved.fast_path_nprobe_N) + " > " +
        std::to_string(resolved.cluster_num));
  }
  if (resolved.slow_path_rbq_prefilter_enabled)
  {
    if (!resolved.fast_path_enabled || resolved.fast_path_mode != "rbq")
    {
      throw std::runtime_error(
          "slow_path_rbq_prefilter_enabled requires fast_path_enabled=true and "
          "fast_path_mode=rbq (RBQ codes must be present)");
    }
    if (!resolved.cell_filter_enabled)
    {
      throw std::runtime_error(
          "slow_path_rbq_prefilter_enabled currently requires cell_filter_enabled=true");
    }
  }
  if (resolved.slow_path_rbq_phase_b_enabled &&
      !resolved.slow_path_rbq_prefilter_enabled)
  {
    throw std::runtime_error(
        "slow_path_rbq_phase_b_enabled requires slow_path_rbq_prefilter_enabled=true "
        "(Phase B reuses the RBQ classify pre-compute done by Phase A)");
  }

  if (resolved.mode == "memory")
  {
    throw std::runtime_error(
        "memory mode not supported in the range-search-skeleton phase");
  }
  if (resolved.mode != "disk")
  {
    throw std::runtime_error("unsupported mode '" + resolved.mode +
                             "' in range-search-skeleton phase");
  }

  const std::string metric =
      range_search_config::require_value<std::string>(tree, "metric");
  if (metric != "l2")
  {
    throw std::runtime_error("only L2 metric is supported; JSON metric is '" +
                             metric + "'");
  }

  resolved.n_queries =
      range_search_config::require_value<size_t>(tree, "n_queries");
  resolved.n_base = range_search_config::require_value<size_t>(tree, "n_base");
  resolved.dim = range_search_config::require_value<size_t>(tree, "dim");

  // === vec_dtype resolution ===
  // GT JSON `vec_dtype` (string "float32"/"int8"/"uint8" or integer 0/1/2) is the
  // authoritative dataset type. The `.config` vec_dtype is optional cross-
  // check; if both are present they must agree. If neither is present, fall
  // back to float32 for backward compatibility with old GT JSON / configs.
  bool gt_has_vec_dtype = false;
  uint32_t gt_vec_dtype = range_search_config::kVecDTypeFloat32;
  if (tree.contains("vec_dtype"))
  {
    const auto& v = tree.at("vec_dtype");
    if (v.is_string())
    {
      gt_vec_dtype =
          range_search_config::parse_vec_dtype_string(v.get<std::string>());
    }
    else if (v.is_number_integer() || v.is_number_unsigned())
    {
      const long long raw = v.get<long long>();
      if (raw < 0 || raw > static_cast<long long>(
                                range_search_config::kVecDTypeFp16Reserved))
      {
        throw std::runtime_error(
            "GT JSON vec_dtype out of range: " + std::to_string(raw));
      }
      gt_vec_dtype = static_cast<uint32_t>(raw);
    }
    else
    {
      throw std::runtime_error(
          "GT JSON vec_dtype must be a string ('float32'/'int8'/'uint8') or "
          "an integer (0/1/2)");
    }
    if (!range_search_config::is_supported_vec_dtype(gt_vec_dtype))
    {
      throw std::runtime_error(
          "GT JSON vec_dtype=" + std::to_string(gt_vec_dtype) + " (" +
          range_search_config::vec_dtype_to_string(gt_vec_dtype) +
          ") is reserved but not implemented; supported are {0, 1, 2}");
    }
    gt_has_vec_dtype = true;
  }

  const bool config_explicit_vec_dtype =
      (config_reader.vec_dtype != ConfigReader::kVecDTypeUnspecified);
  if (gt_has_vec_dtype && config_explicit_vec_dtype &&
      config_reader.vec_dtype != gt_vec_dtype)
  {
    throw std::runtime_error(
        "vec_dtype mismatch: .config vec_dtype=" +
        std::to_string(config_reader.vec_dtype) + " (" +
        range_search_config::vec_dtype_to_string(config_reader.vec_dtype) +
        "), GT JSON vec_dtype=" + std::to_string(gt_vec_dtype) + " (" +
        range_search_config::vec_dtype_to_string(gt_vec_dtype) + ")");
  }
  if (gt_has_vec_dtype)
  {
    resolved.vec_dtype = gt_vec_dtype;
  }
  else if (config_explicit_vec_dtype)
  {
    resolved.vec_dtype = config_reader.vec_dtype;
  }
  else
  {
    resolved.vec_dtype = range_search_config::kVecDTypeFloat32;
  }

  // Re-run the byte-dtype reject combinations on the resolved (authoritative)
  // vec_dtype. ConfigReader's internal reject only fires when the user wrote
  // `vec_dtype` explicitly; if the user omitted it in .config but the GT JSON
  // says int8/uint8, we still need to reject the bad combinations here.
  if (range_search_config::is_byte_dtype(resolved.vec_dtype))
  {
    const std::string dtype_name =
        range_search_config::vec_dtype_to_string(resolved.vec_dtype);
    const std::string dtype_num = std::to_string(resolved.vec_dtype);
    if (resolved.partition_objective == "anisotropic")
    {
      throw std::runtime_error(
          "vec_dtype=" + dtype_num + " (" + dtype_name +
          ") does not support partition_objective=anisotropic");
    }
    if (resolved.unit_normalize_prebuild)
    {
      throw std::runtime_error(
          "vec_dtype=" + dtype_num + " (" + dtype_name +
          ") does not support unit_normalize_prebuild=true");
    }
    if (resolved.fast_path_mode == "exact")
    {
      throw std::runtime_error(
          "vec_dtype=" + dtype_num + " (" + dtype_name +
          ") does not support fast_path_mode=exact");
    }
  }

  if (resolved.vec_dtype == range_search_config::kVecDTypeInt8 &&
      resolved.dim > range_search_config::kInt8DimMetadataLimit)
  {
    throw std::runtime_error(
        "vec_dtype=1 (int8) requires dim <= " +
        std::to_string(range_search_config::kInt8DimMetadataLimit) +
        " because sum/norm metadata is stored as int32; dim=" +
        std::to_string(resolved.dim));
  }
  if (resolved.vec_dtype == range_search_config::kVecDTypeUint8Reserved &&
      resolved.dim > range_search_config::kUint8DimMetadataLimit)
  {
    throw std::runtime_error(
        "uint8 requires dim <= " +
        std::to_string(range_search_config::kUint8DimMetadataLimit) +
        " (vec_dtype=2) because sum/norm metadata is stored as int32; dim=" +
        std::to_string(resolved.dim));
  }
  if (resolved.pca_rank > resolved.dim)
  {
    throw std::runtime_error("pca_rank must be <= dim: pca_rank=" +
                             std::to_string(resolved.pca_rank) +
                             ", dim=" + std::to_string(resolved.dim));
  }
  if (resolved.slab_count > 0 && resolved.slab_count >= resolved.cluster_num)
  {
    throw std::runtime_error("slab_count " +
                             std::to_string(resolved.slab_count) +
                             " must be < cluster_num " +
                             std::to_string(resolved.cluster_num) +
                             " (need at least slab_count + 1 distinct clusters)");
  }
  resolved.radius_squared =
      range_search_config::require_value<double>(tree, "radius_squared");
  resolved.total_in_range =
      range_search_config::require_value<size_t>(tree, "total_in_range");
  resolved.data_file = range_search_config::require_value<std::string>(
      tree, "provenance.base_path");
  resolved.query_file = range_search_config::require_value<std::string>(
      tree, "provenance.query_path");
  const std::string prefix =
      range_search_config::require_value<std::string>(tree, "index.prefix");
  resolved.index_prefix = prefix;

  const IndexPaths paths = make_diskrange_paths(prefix);
  resolved.cluster_path = paths.cluster;
  resolved.metadata_path = paths.metadata;
  resolved.hnsw_path = paths.hnsw;
  resolved.data_hnsw_path = paths.data_hnsw;
  resolved.data_hnsw_trace_path = paths.data_hnsw_trace;
  resolved.sqg_path = paths.sqg;
  resolved.rabitq_1bit_codes_path = paths.rabitq_1bit_codes;
  resolved.rabitq_8bit_codes_path = paths.rabitq_8bit_codes;
  resolved.apply_partition_suffix_to_paths();

  range_search_config::validate_fbin_header(
      resolved.data_file, resolved.n_base, resolved.dim, resolved.vec_dtype);
  range_search_config::validate_fbin_header(
      resolved.query_file, resolved.n_queries, resolved.dim, resolved.vec_dtype);

  std::vector<size_t> counts = range_search_config::read_per_query_counts(tree);
  range_search_config::validate_per_query_counts(counts, resolved.n_queries,
                                                 resolved.total_in_range, tree);
  resolved.per_query_counts = std::move(counts);

  return resolved;
}
