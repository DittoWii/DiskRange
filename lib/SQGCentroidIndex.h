#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"

// SQGCentroidIndex owns the SQG-on-centroids graph used by the fast-path
// emptiness oracle. RaBitQ buffers must be rotated with rotator(), which is the
// sole rotation source for this OpenSpec change.
class SQGCentroidIndex
{
 public:
  SQGCentroidIndex() = default;

  void build(const std::vector<float>& centroids, size_t cluster_count,
             size_t dim, size_t m, size_t num_iter, size_t ef_build,
             size_t ef_search)
  {
    validate_params(cluster_count, dim, m, num_iter, ef_build, ef_search);
    if (centroids.size() != cluster_count * dim)
    {
      throw std::runtime_error("SQG centroid payload size mismatch");
    }
    qg_ = std::make_unique<symqg::QuantizedGraph>(cluster_count, m, dim);
    symqg::QGBuilder builder(*qg_, static_cast<uint32_t>(ef_build),
                             centroids.data(),
                             static_cast<size_t>(omp_get_max_threads()));
    builder.build(num_iter);
    qg_->set_ef(ef_search);
    cluster_count_ = cluster_count;
    dim_ = dim;
    m_ = m;
    ef_search_ = ef_search;
  }

  void save(const std::string& path) const
  {
    require_loaded();
    const std::filesystem::path out_path(path);
    const std::filesystem::path parent = out_path.parent_path();
    if (!parent.empty())
    {
      std::filesystem::create_directories(parent);
    }
    qg_->save_index(path.c_str());
  }

  void load(const std::string& path, size_t cluster_count, size_t dim, size_t m,
            size_t ef_search)
  {
    validate_params(cluster_count, dim, m, 3, 1, ef_search);
    if (!std::filesystem::exists(path))
    {
      throw std::runtime_error("missing SQG artifact: " + path);
    }
    qg_ = std::make_unique<symqg::QuantizedGraph>(cluster_count, m, dim);
    qg_->load_index(path.c_str());
    qg_->set_ef(ef_search);
    cluster_count_ = cluster_count;
    dim_ = dim;
    m_ = m;
    ef_search_ = ef_search;
  }

  std::vector<uint32_t> search(const float* query, size_t nprobe) const
  {
    require_loaded();
    if (nprobe == 0 || nprobe > cluster_count_)
    {
      throw std::runtime_error("SQG search nprobe must be in [1, cluster_count]");
    }
    std::vector<uint32_t> results(nprobe);
    qg_->search(query, static_cast<uint32_t>(nprobe), results.data());
    return results;
  }

  symqg::SearchContext make_search_context() const
  {
    require_loaded();
    return qg_->make_search_context();
  }

  void search_into(const float* query, uint32_t nprobe, uint32_t* out_ids,
                   symqg::SearchContext& ctx) const
  {
    require_loaded();
    if (nprobe == 0 || nprobe > cluster_count_)
    {
      throw std::runtime_error("SQG search_into nprobe must be in [1, cluster_count]");
    }
    if (out_ids == nullptr)
    {
      throw std::runtime_error("SQG search_into output buffer must be non-null");
    }
    ctx.search_pool.clear();
    ctx.visited.clear();
    qg_->search_qg(query, nprobe, out_ids, ctx);
  }

  const symqg::FHTRotator& rotator() const
  {
    require_loaded();
    return qg_->rotator();
  }

  size_t padded_dim() const
  {
    require_loaded();
    return qg_->padded_dim();
  }

  size_t cluster_count() const { return cluster_count_; }
  size_t dim() const { return dim_; }

 private:
  static void validate_params(size_t cluster_count, size_t dim, size_t m,
                              size_t num_iter, size_t ef_build,
                              size_t ef_search)
  {
    if (cluster_count == 0 || dim == 0)
    {
      throw std::runtime_error("SQG cluster_count and dim must be positive");
    }
    if (m == 0 || m >= cluster_count)
    {
      throw std::runtime_error("SQG degree sqg_m must be in [1, cluster_count)");
    }
    if (num_iter < 3)
    {
      throw std::runtime_error("SQG sqg_num_iter must be >= 3");
    }
    if (ef_build == 0 || ef_search == 0)
    {
      throw std::runtime_error("SQG ef_build and ef_search must be positive");
    }
  }

  void require_loaded() const
  {
    if (!qg_)
    {
      throw std::runtime_error("SQGCentroidIndex is not loaded");
    }
  }

  std::unique_ptr<symqg::QuantizedGraph> qg_;
  size_t cluster_count_ = 0;
  size_t dim_ = 0;
  size_t m_ = 0;
  size_t ef_search_ = 0;
};
