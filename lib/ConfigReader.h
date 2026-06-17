#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "VecDType.h"

using namespace std;

struct ConfigReader
{
  size_t cluster_num = 0;
  size_t K = 0;
  float mem_budget = 0.0f;
  string mode;
  size_t gorder_window = 0;
  size_t io_queue_depth = 256;
  float cell_cache_share_pct = 0.0f;
  float approx_alpha = 1.0f;
  size_t pca_rank = 32;
  size_t slab_count = 128;
  bool cell_filter_enabled = true;
  bool pca_cluster_prune_enabled = true;
  bool slab_cluster_prune_enabled = true;
  size_t target_cell_vecs = 24;
  string split_axis_policy = "fixed";
  bool data_hnsw_enabled = false;
  size_t data_hnsw_M = 16;
  size_t data_hnsw_ef_construction = 100;
  size_t data_hnsw_ef = 100;
  float data_hnsw_margin_delta = 0.0f;
  bool fast_path_enabled = false;
  string fast_path_mode = "rbq";
  size_t fast_path_nprobe_N = 128;
  bool slow_path_rbq_prefilter_enabled = false;
  bool slow_path_rbq_phase_b_enabled = false;
  size_t sqg_m = 64;
  size_t sqg_num_iter = 3;
  size_t sqg_ef_build = 100;
  size_t sqg_ef_search = 300;
  // Build-time centroid assignment ef. Measured-tuned (index-graph-sqg
  // post-implementation profile at d=96, K=50000, top-K=2): ef_search=20
  // keeps deep10m search recall >= 0.999196 with km.assign_full_data median
  // ~8s; ef_search=100 inflates assign to ~21s with no recall benefit
  // because top-K=2 retrieval saturates well below the higher ef. Separate
  // from search-time sqg_ef_search.
  size_t sqg_centroid_ef_search = 20;
  string partition_objective = "standard";
  float eta = 1.0f;
  bool unit_normalize_prebuild = false;
  // Sentinel value 0xFFFFFFFFu means "not specified in the .config file"; in
  // that case load_resolved_config takes vec_dtype from the GT JSON instead.
  // If the user did write `vec_dtype 0|1|2` in the .config, it is cross-checked
  // against the GT JSON and rejected on mismatch.
  static constexpr uint32_t kVecDTypeUnspecified = 0xFFFFFFFFu;
  uint32_t vec_dtype = kVecDTypeUnspecified;

  ConfigReader() = default;

  ConfigReader(string config)
  {
    ifstream in(config);
    if (!in)
    {
      throw runtime_error("failed to open config file: " + config);
    }

    unordered_set<string> seen;
    string line;
    size_t line_no = 0;
    while (getline(in, line))
    {
      line_no++;
      const auto first = line.find_first_not_of(" \t\r\n");
      if (first == string::npos || line[first] == '#')
      {
        continue;
      }

      istringstream iss(line.substr(first));
      string key;
      string value;
      iss >> key >> value;
      if (key.empty() || value.empty())
      {
        throw runtime_error("invalid config line " + to_string(line_no) +
                            " in " + config + ": expected <key> <value>");
      }
      if (key == "approx_alpha")
      {
        string extra;
        if (iss >> extra)
        {
          throw runtime_error("invalid config line " + to_string(line_no) +
                              " in " + config +
                              ": approx_alpha has extra token after value: " +
                              extra);
        }
      }
      if (!seen.insert(key).second)
      {
        throw runtime_error("duplicate config key: " + key);
      }

      try
      {
        if (key == "cluster_num")
        {
          cluster_num = stoull(value);
        }
        else if (key == "K")
        {
          K = stoull(value);
        }
        else if (key == "mem_budget")
        {
          mem_budget = stof(value);
        }
        else if (key == "mode")
        {
          mode = value;
        }
        else if (key == "gorder_window")
        {
          gorder_window = stoull(value);
        }
        else if (key == "io_queue_depth")
        {
          io_queue_depth = stoull(value);
        }
        else if (key == "cell_cache_share_pct")
        {
          cell_cache_share_pct = stof(value);
        }
        else if (key == "approx_alpha")
        {
          size_t idx = 0;
          approx_alpha = stof(value, &idx);
          if (idx != value.size())
          {
            throw runtime_error("invalid value for config key approx_alpha: " +
                                value);
          }
          if (!std::isfinite(approx_alpha) || approx_alpha < 1.0f)
          {
            throw runtime_error(
                "approx_alpha must be finite and >= 1.0: " + value);
          }
        }
        else if (key == "pca_rank")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("pca_rank must be non-negative: " + value);
          }
          pca_rank = stoull(value);
        }
        else if (key == "slab_count")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("slab_count must be non-negative: " + value);
          }
          slab_count = stoull(value);
        }
        else if (key == "cell_filter_enabled")
        {
          if (value == "true" || value == "1")
          {
            cell_filter_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            cell_filter_enabled = false;
          }
          else
          {
            throw runtime_error(
                "cell_filter_enabled must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "pca_cluster_prune_enabled")
        {
          if (value == "true" || value == "1")
          {
            pca_cluster_prune_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            pca_cluster_prune_enabled = false;
          }
          else
          {
            throw runtime_error(
                "pca_cluster_prune_enabled must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "slab_cluster_prune_enabled")
        {
          if (value == "true" || value == "1")
          {
            slab_cluster_prune_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            slab_cluster_prune_enabled = false;
          }
          else
          {
            throw runtime_error(
                "slab_cluster_prune_enabled must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "target_cell_vecs")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("target_cell_vecs must be >= 2: " + value);
          }
          target_cell_vecs = stoull(value);
          if (target_cell_vecs < 2)
          {
            throw runtime_error("target_cell_vecs must be >= 2: " + value);
          }
        }
        else if (key == "split_axis_policy")
        {
          if (value == "fixed")
          {
            split_axis_policy = value;
          }
          else if (value == "dynamic")
          {
            throw runtime_error(
                "split_axis_policy=dynamic is a P-C2 v2 placeholder; current implementation supports fixed only");
          }
          else
          {
            throw runtime_error(
                "split_axis_policy must be one of fixed/dynamic: " + value);
          }
        }
        else if (key == "data_hnsw_enabled")
        {
          if (value == "true" || value == "1")
          {
            data_hnsw_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            data_hnsw_enabled = false;
          }
          else
          {
            throw runtime_error(
                "data_hnsw_enabled must be one of true/false/1/0: " + value);
          }
        }
        else if (key == "data_hnsw_M")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("data_hnsw_M must be positive: " + value);
          }
          data_hnsw_M = stoull(value);
          if (data_hnsw_M == 0)
          {
            throw runtime_error("data_hnsw_M must be positive: " + value);
          }
        }
        else if (key == "data_hnsw_ef_construction")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error(
                "data_hnsw_ef_construction must be positive: " + value);
          }
          data_hnsw_ef_construction = stoull(value);
          if (data_hnsw_ef_construction == 0)
          {
            throw runtime_error(
                "data_hnsw_ef_construction must be positive: " + value);
          }
        }
        else if (key == "data_hnsw_ef")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("data_hnsw_ef must be positive: " + value);
          }
          data_hnsw_ef = stoull(value);
          if (data_hnsw_ef == 0)
          {
            throw runtime_error("data_hnsw_ef must be positive: " + value);
          }
        }
        else if (key == "data_hnsw_margin_delta")
        {
          data_hnsw_margin_delta = stof(value);
          if (data_hnsw_margin_delta < 0.0f)
          {
            throw runtime_error("data_hnsw_margin_delta must be >= 0: " + value);
          }
        }
        else if (key == "fast_path_enabled")
        {
          if (value == "true" || value == "1")
          {
            fast_path_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            fast_path_enabled = false;
          }
          else
          {
            throw runtime_error(
                "fast_path_enabled must be one of true/false/1/0: " + value);
          }
        }
        else if (key == "fast_path_mode")
        {
          if (value != "rbq" && value != "exact")
          {
            throw runtime_error("fast_path_mode must be one of rbq/exact: " +
                                value);
          }
          fast_path_mode = value;
        }
        else if (key == "slow_path_rbq_prefilter_enabled")
        {
          if (value == "true" || value == "1")
          {
            slow_path_rbq_prefilter_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            slow_path_rbq_prefilter_enabled = false;
          }
          else
          {
            throw runtime_error(
                "slow_path_rbq_prefilter_enabled must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "slow_path_rbq_phase_b_enabled")
        {
          if (value == "true" || value == "1")
          {
            slow_path_rbq_phase_b_enabled = true;
          }
          else if (value == "false" || value == "0")
          {
            slow_path_rbq_phase_b_enabled = false;
          }
          else
          {
            throw runtime_error(
                "slow_path_rbq_phase_b_enabled must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "fast_path_nprobe_N")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("fast_path_nprobe_N must be positive: " +
                                value);
          }
          fast_path_nprobe_N = stoull(value);
          if (fast_path_nprobe_N == 0)
          {
            throw runtime_error("fast_path_nprobe_N must be positive: " +
                                value);
          }
        }
        else if (key == "sqg_m")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("sqg_m must be positive: " + value);
          }
          sqg_m = stoull(value);
          if (sqg_m == 0)
          {
            throw runtime_error("sqg_m must be positive: " + value);
          }
        }
        else if (key == "sqg_num_iter")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("sqg_num_iter must be >= 3: " + value);
          }
          sqg_num_iter = stoull(value);
          if (sqg_num_iter < 3)
          {
            throw runtime_error("sqg_num_iter must be >= 3: " + value);
          }
        }
        else if (key == "sqg_ef_build")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("sqg_ef_build must be positive: " + value);
          }
          sqg_ef_build = stoull(value);
          if (sqg_ef_build == 0)
          {
            throw runtime_error("sqg_ef_build must be positive: " + value);
          }
        }
        else if (key == "sqg_ef_search")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("sqg_ef_search must be positive: " + value);
          }
          sqg_ef_search = stoull(value);
          if (sqg_ef_search == 0)
          {
            throw runtime_error("sqg_ef_search must be positive: " + value);
          }
        }
        else if (key == "sqg_centroid_ef_search")
        {
          if (!value.empty() && value[0] == '-')
          {
            throw runtime_error("sqg_centroid_ef_search must be positive: " +
                                value);
          }
          sqg_centroid_ef_search = stoull(value);
          if (sqg_centroid_ef_search == 0)
          {
            throw runtime_error("sqg_centroid_ef_search must be positive: " +
                                value);
          }
        }
        else if (key == "partition_objective")
        {
          if (value != "standard" && value != "anisotropic")
          {
            throw runtime_error("unrecognized partition_objective '" + value +
                                "'; must be 'standard' or 'anisotropic'");
          }
          partition_objective = value;
        }
        else if (key == "eta")
        {
          eta = stof(value);
          if (!(eta > 0.0f))
          {
            throw runtime_error("eta must be positive: " + value);
          }
        }
        else if (key == "unit_normalize_prebuild")
        {
          if (value == "true" || value == "1")
          {
            unit_normalize_prebuild = true;
          }
          else if (value == "false" || value == "0")
          {
            unit_normalize_prebuild = false;
          }
          else
          {
            throw runtime_error(
                "unit_normalize_prebuild must be one of true/false/1/0: " +
                value);
          }
        }
        else if (key == "vec_dtype")
        {
          vec_dtype = range_search_config::parse_vec_dtype_token_strict(value);
        }
        else if (key.rfind("cell_", 0) == 0 && key.size() == 6 &&
                 (key[5] == 'b' || key[5] == 'q'))
        {
          throw runtime_error(
              "P-C2 schema 5 uses target_cell_vecs for cell sizing");
        }
        else
        {
          throw runtime_error("unrecognized config key: " + key);
        }
      }
      catch (const invalid_argument&)
      {
        throw runtime_error("invalid value for config key " + key + ": " +
                            value);
      }
      catch (const out_of_range&)
      {
        throw runtime_error("out-of-range value for config key " + key + ": " +
                            value);
      }
    }

    const string required[] = {"cluster_num", "K", "mem_budget", "mode",
                               "gorder_window"};
    for (const auto& key : required)
    {
      if (seen.find(key) == seen.end())
      {
        throw runtime_error("missing required config key: " + key);
      }
    }
    if (range_search_config::is_byte_dtype(vec_dtype) &&
        unit_normalize_prebuild)
    {
      throw runtime_error(
          "vec_dtype=" + to_string(vec_dtype) + " (" +
          range_search_config::vec_dtype_to_string(vec_dtype) +
          ") does not support unit_normalize_prebuild=true");
    }
    if (range_search_config::is_byte_dtype(vec_dtype) &&
        partition_objective == "anisotropic")
    {
      throw runtime_error(
          "vec_dtype=" + to_string(vec_dtype) + " (" +
          range_search_config::vec_dtype_to_string(vec_dtype) +
          ") does not support partition_objective=anisotropic");
    }
    if (range_search_config::is_byte_dtype(vec_dtype) &&
        fast_path_mode == "exact")
    {
      throw runtime_error(
          "vec_dtype=" + to_string(vec_dtype) + " (" +
          range_search_config::vec_dtype_to_string(vec_dtype) +
          ") does not support fast_path_mode=exact");
    }
    if (partition_objective == "anisotropic" &&
        seen.find("unit_normalize_prebuild") == seen.end())
    {
      unit_normalize_prebuild = true;
    }
    if (partition_objective == "anisotropic" && !unit_normalize_prebuild)
    {
      throw runtime_error(
          "anisotropic partition_objective requires unit_normalize_prebuild=true");
    }
  }
};
