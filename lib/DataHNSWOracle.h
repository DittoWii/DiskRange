#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <omp.h>

#include "data_hnsw_search.h"
#include "hnswlib/hnswlib.h"

struct DataHNSWOracleResult
{
  float d_hat_l2 = std::numeric_limits<float>::infinity();
  bool is_empty = false;
};

struct DataHNSWOracle
{
  std::unique_ptr<hnswlib::L2Space> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;

  explicit DataHNSWOracle(size_t dim)
      : space(std::make_unique<hnswlib::L2Space>(dim))
  {
  }

  DataHNSWOracleResult evaluate(const float* q, float radius_l2,
                                float delta) const
  {
    if (delta < 0.0f)
    {
      throw std::runtime_error("data_hnsw_margin_delta must be >= 0");
    }
    if (!graph)
    {
      throw std::runtime_error("DataHNSWOracle graph is not loaded");
    }

    const float d_hat_squared =
        data_hnsw_1nn_squared(*graph, static_cast<const void*>(q));
    const float d_hat_l2 = std::sqrt(std::max(0.0f, d_hat_squared));
    return {d_hat_l2, d_hat_l2 > radius_l2 + delta};
  }
};

inline std::unique_ptr<DataHNSWOracle> build_data_hnsw(
    const float* data, size_t n, size_t dim, size_t M,
    size_t ef_construction)
{
  if (n == 0)
  {
    throw std::runtime_error("cannot build data HNSW for empty dataset");
  }
  auto oracle = std::make_unique<DataHNSWOracle>(dim);
  oracle->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      oracle->space.get(), n, M, ef_construction, /*random_seed=*/100);
  oracle->graph->addPoint(static_cast<const void*>(data), 0);
#pragma omp parallel for schedule(static)
  for (size_t i = 1; i < n; ++i)
  {
    oracle->graph->addPoint(static_cast<const void*>(data + i * dim),
                            static_cast<hnswlib::labeltype>(i));
  }
  return oracle;
}

inline void save_data_hnsw(const hnswlib::HierarchicalNSW<float>& graph,
                           const std::string& path)
{
  const std::filesystem::path out_path(path);
  const std::filesystem::path parent = out_path.parent_path();
  if (!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
  const_cast<hnswlib::HierarchicalNSW<float>&>(graph).saveIndex(path);
}

inline bool validate_data_hnsw_header(const std::string& path,
                                      size_t expected_dim,
                                      std::string& reason)
{
  std::ifstream input(path, std::ios::binary);
  if (!input)
  {
    reason = "missing";
    return false;
  }

  size_t offset_level0 = 0;
  size_t max_elements = 0;
  size_t cur_element_count = 0;
  size_t size_data_per_element = 0;
  size_t label_offset = 0;
  size_t offset_data = 0;
  input.read(reinterpret_cast<char*>(&offset_level0), sizeof(offset_level0));
  input.read(reinterpret_cast<char*>(&max_elements), sizeof(max_elements));
  input.read(reinterpret_cast<char*>(&cur_element_count),
             sizeof(cur_element_count));
  input.read(reinterpret_cast<char*>(&size_data_per_element),
             sizeof(size_data_per_element));
  input.read(reinterpret_cast<char*>(&label_offset), sizeof(label_offset));
  input.read(reinterpret_cast<char*>(&offset_data), sizeof(offset_data));
  if (!input)
  {
    reason = "corrupt header";
    return false;
  }
  if (cur_element_count == 0 || cur_element_count > max_elements)
  {
    reason = "empty or invalid element count";
    return false;
  }
  if (offset_data > label_offset || label_offset > size_data_per_element)
  {
    reason = "invalid level-0 layout";
    return false;
  }
  const size_t data_size_bytes = label_offset - offset_data;
  const size_t expected_data_size = expected_dim * sizeof(float);
  if (data_size_bytes != expected_data_size)
  {
    reason = "dimensionality mismatch";
    return false;
  }
  return true;
}

inline std::unique_ptr<DataHNSWOracle> load_data_hnsw(const std::string& path,
                                                       size_t expected_dim,
                                                       size_t ef)
{
  try
  {
    if (!std::filesystem::exists(path))
    {
      std::cerr << "warning: data HNSW artifact unavailable: " << path
                << " (missing); oracle disabled\n";
      return nullptr;
    }
    std::string reason;
    if (!validate_data_hnsw_header(path, expected_dim, reason))
    {
      std::cerr << "warning: data HNSW artifact unavailable: " << path << " ("
                << reason << "); oracle disabled\n";
      return nullptr;
    }
    auto oracle = std::make_unique<DataHNSWOracle>(expected_dim);
    oracle->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        oracle->space.get(), path);
    if (oracle->graph->cur_element_count == 0)
    {
      std::cerr << "warning: data HNSW artifact unavailable: " << path
                << " (empty index); oracle disabled\n";
      return nullptr;
    }
    oracle->graph->setEf(ef);
    return oracle;
  }
  catch (const std::exception& e)
  {
    std::cerr << "warning: data HNSW artifact unavailable: " << path << " ("
              << e.what() << "); oracle disabled\n";
    return nullptr;
  }
}
