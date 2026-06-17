#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../lib/ClusterIO.h"
#include "../lib/ConfigLoader.h"
#include "../lib/SQGCentroidIndex.h"
#include "../utils/dist_func.h"

int r_server_id = 0;

namespace
{
struct Args
{
  std::string config_path;
  std::string gt_path;
  size_t query_limit = 50;
  size_t candidate_limit = 0;
};

void print_usage()
{
  std::cerr << "Usage: pca_lb_soundness_probe --config <path> --gt <path> "
               "[--queries N] [--candidates K (default: config K)]"
            << std::endl;
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
      std::string value;
      if (!need_value(value))
      {
        return false;
      }
      args.query_limit = static_cast<size_t>(std::stoull(value));
    }
    else if (arg == "--candidates")
    {
      std::string value;
      if (!need_value(value))
      {
        return false;
      }
      args.candidate_limit = static_cast<size_t>(std::stoull(value));
    }
    else
    {
      return false;
    }
  }
  return saw_config && saw_gt && args.query_limit > 0;
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
  in.read(reinterpret_cast<char*>(&file_n), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&file_dim), sizeof(uint32_t));
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

std::vector<size_t> cluster_file_offsets(const ClusterReader& reader)
{
  std::vector<size_t> offsets(reader.cluster_num + 1, 0);
  for (size_t c = 0; c < reader.cluster_num; ++c)
  {
    offsets[c + 1] =
        offsets[c] + reader.bucket_sizes[c] * reader.d * sizeof(float);
  }
  return offsets;
}

std::vector<float> read_bucket(std::ifstream& cluster_file,
                               const std::vector<size_t>& offsets,
                               const ClusterReader& reader,
                               size_t cluster_id)
{
  const size_t value_count = reader.bucket_sizes[cluster_id] * reader.d;
  std::vector<float> bucket(value_count);
  if (value_count == 0)
  {
    return bucket;
  }
  cluster_file.clear();
  cluster_file.seekg(static_cast<std::streamoff>(offsets[cluster_id]),
                     std::ios::beg);
  cluster_file.read(reinterpret_cast<char*>(bucket.data()),
                    static_cast<std::streamsize>(value_count * sizeof(float)));
  if (cluster_file.gcount() !=
      static_cast<std::streamsize>(value_count * sizeof(float)))
  {
    throw std::runtime_error("short cluster payload in probe at cluster " +
                             std::to_string(cluster_id));
  }
  return bucket;
}

float l2_distance(const float* left, const float* right, size_t dim)
{
  size_t distance_dim = dim;
  return std::sqrt(utils::L2Sqr(left, right, &distance_dim));
}

std::vector<size_t> candidate_clusters(const SQGCentroidIndex& sqg,
                                       const float* query, size_t limit)
{
  const std::vector<uint32_t> ids = sqg.search(query, limit);
  std::vector<size_t> clusters;
  clusters.reserve(ids.size());
  for (uint32_t id : ids)
  {
    clusters.push_back(static_cast<size_t>(id));
  }
  return clusters;
}

void prepare_query_projection(const ClusterReader& reader, size_t cluster_id,
                              const float* query,
                              std::vector<float>& q_projected,
                              float& q_residual_norm)
{
  std::vector<float> centered(reader.d, 0.0f);
  float total_sq = 0.0f;
  const float* centroid = reader.centroids.data() + cluster_id * reader.d;
  for (size_t col = 0; col < reader.d; ++col)
  {
    const float diff = query[col] - centroid[col];
    centered[col] = diff;
    total_sq += diff * diff;
  }

  const uint32_t effective_rank = reader.pca_effective_rank[cluster_id];
  q_projected.assign(effective_rank, 0.0f);
  const size_t basis_offset = cluster_id * reader.pca_rank * reader.d;
  float projected_sq = 0.0f;
  for (size_t axis = 0; axis < effective_rank; ++axis)
  {
    const float* basis_row =
        reader.pca_basis.data() + basis_offset + axis * reader.d;
    float coord = 0.0f;
    for (size_t col = 0; col < reader.d; ++col)
    {
      coord += basis_row[col] * centered[col];
    }
    q_projected[axis] = coord;
    projected_sq += coord * coord;
  }
  q_residual_norm = std::sqrt(std::max(0.0f, total_sq - projected_sq));
}
}  // namespace

int main(int argc, char** argv)
{
  Args args;
  if (!parse_args(argc, argv, args))
  {
    print_usage();
    return 2;
  }

  try
  {
    const ResolvedConfig config =
        load_resolved_config(args.config_path, args.gt_path);
    ClusterReader reader(config.cluster_path, config.metadata_path,
                         config.io_queue_depth, 1);
    reader.readMetaData();

    size_t query_count = 0;
    const std::vector<float> queries =
        read_queries(config, args.query_limit, query_count);
    SQGCentroidIndex sqg_probe;
    sqg_probe.load(config.sqg_path, config.cluster_num, config.dim,
                   config.sqg_m, config.sqg_ef_search);
    const size_t candidate_limit =
        args.candidate_limit == 0 ? config.K : args.candidate_limit;
    if (candidate_limit == 0)
    {
      throw std::runtime_error("candidate limit must be positive");
    }

    std::ifstream cluster_file(config.cluster_path, std::ios::binary);
    if (!cluster_file)
    {
      throw std::runtime_error("failed to open cluster file: " +
                               config.cluster_path);
    }
    const std::vector<size_t> offsets = cluster_file_offsets(reader);

    std::vector<unsigned char> degenerate_seen(reader.cluster_num, 0);
    size_t checked_vectors = 0;
    size_t checked_cells = 0;
    size_t degenerate_count = 0;
    float worst_cluster_margin = std::numeric_limits<float>::infinity();
    float worst_cell_margin = std::numeric_limits<float>::infinity();

    for (size_t q = 0; q < query_count; ++q)
    {
      const float* query = queries.data() + q * config.dim;
      const std::vector<size_t> clusters =
          candidate_clusters(sqg_probe, query, candidate_limit);
      for (const size_t cluster_id : clusters)
      {
        if (cluster_id >= reader.cluster_num)
        {
          throw std::runtime_error("candidate cluster id out of range: " +
                                   std::to_string(cluster_id));
        }
        if (reader.bucket_sizes[cluster_id] == 0)
        {
          continue;
        }
        if (reader.pca_effective_rank[cluster_id] == 0 &&
            degenerate_seen[cluster_id] == 0)
        {
          degenerate_seen[cluster_id] = 1;
          ++degenerate_count;
        }

        const std::vector<float> bucket =
            read_bucket(cluster_file, offsets, reader, cluster_id);
        const float cluster_lb = reader.pca_lower_bound(cluster_id, query);
        std::vector<float> q_projected;
        float q_residual_norm = 0.0f;
        prepare_query_projection(reader, cluster_id, query, q_projected,
                                 q_residual_norm);

        const size_t cell_count = reader.cell_count_of(cluster_id);
        if (cell_count == 0)
        {
          throw std::runtime_error("non-empty cluster has no cells: " +
                                   std::to_string(cluster_id));
        }
        const uint64_t first_cell = reader.cluster_cell_offsets[cluster_id];
        const uint64_t cluster_global_begin = reader.cell_vec_offsets[first_cell];
        for (size_t cell_idx = 0; cell_idx < cell_count; ++cell_idx)
        {
          const auto [g_begin, g_end] =
              reader.cell_global_vec_range(cluster_id, cell_idx);
          if (g_begin < cluster_global_begin || g_end < g_begin)
          {
            throw std::runtime_error("invalid cell range at cluster " +
                                     std::to_string(cluster_id));
          }
          const size_t local_begin =
              static_cast<size_t>(g_begin - cluster_global_begin);
          const size_t local_end =
              static_cast<size_t>(g_end - cluster_global_begin);
          if (local_end > reader.bucket_sizes[cluster_id])
          {
            throw std::runtime_error("cell local range exceeds bucket at cluster " +
                                     std::to_string(cluster_id));
          }
          const uint32_t effective_rank = reader.pca_effective_rank[cluster_id];
          const float cell_lb = utils::cell_lb_batch(
              reader.cell_lo_ptr(cluster_id, cell_idx),
              reader.cell_hi_ptr(cluster_id, cell_idx),
              reader.cell_res_lo(cluster_id, cell_idx),
              reader.cell_res_hi(cluster_id, cell_idx), q_projected.data(),
              q_residual_norm, effective_rank);
          ++checked_cells;

          for (size_t local = local_begin; local < local_end; ++local)
          {
            const float* y = bucket.data() + local * reader.d;
            const float true_dist = l2_distance(query, y, reader.d);
            const float cluster_margin = true_dist - cluster_lb;
            const float cell_margin = true_dist - cell_lb;
            worst_cluster_margin =
                std::min(worst_cluster_margin, cluster_margin);
            worst_cell_margin = std::min(worst_cell_margin, cell_margin);
            ++checked_vectors;
            if (cluster_lb > true_dist + 1e-5f)
            {
              std::cerr << "PCA cluster LB violation query=" << q
                        << " cluster=" << cluster_id << " cell=" << cell_idx
                        << " y_local=" << local << " lb=" << cluster_lb
                        << " true_dist=" << true_dist << std::endl;
              return 1;
            }
            if (cell_lb > true_dist + 1e-5f)
            {
              std::cerr << "PCA cell LB violation query=" << q
                        << " cluster=" << cluster_id << " cell=" << cell_idx
                        << " y_local=" << local << " lb=" << cell_lb
                        << " true_dist=" << true_dist << std::endl;
              return 1;
            }
          }
        }
      }
    }

    if (worst_cluster_margin < -1e-5f || worst_cell_margin < -1e-5f)
    {
      throw std::runtime_error("worst PCA LB margin below tolerance");
    }
    std::cout << "pca_lb_soundness_probe checked_vectors=" << checked_vectors
              << " checked_cells=" << checked_cells
              << " worst_cluster_margin=" << worst_cluster_margin
              << " worst_cell_margin=" << worst_cell_margin
              << " degenerate_clusters=" << degenerate_count << std::endl;
    if (degenerate_count == 0)
    {
      std::cout << "WARN: no degenerate clusters observed; "
                   "emit_residual_only path covered by "
                   "synthetic_residual_only_unit_test instead"
                << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
