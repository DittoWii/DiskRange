#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../lib/PcaFit.h"
#include "../utils/dist_func.h"

int r_server_id = 0;

namespace
{
float l2_distance(const float* left, const float* right, size_t dim)
{
  size_t distance_dim = dim;
  return std::sqrt(utils::L2Sqr(left, right, &distance_dim));
}

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void verify_residual_only_bounds(const std::vector<float>& points,
                                 const std::vector<float>& centroid,
                                 size_t n_pts, size_t dim, size_t pca_rank,
                                 const std::string& label)
{
  std::vector<float> basis(pca_rank * dim, -1.0f);
  std::vector<float> pca_min(pca_rank, -1.0f);
  std::vector<float> pca_max(pca_rank, -1.0f);
  float residual = -1.0f;
  uint32_t effective_rank = 99;
  std::vector<PcaCell> cells;

  fit_pca_bucket(points.data(), centroid.data(), n_pts, dim, pca_rank,
                 basis.data(), pca_min.data(), pca_max.data(), &residual,
                 &effective_rank, 24, &cells);

  require(effective_rank == 0, label + ": expected effective rank 0");
  require(residual >= 0.0f, label + ": expected nonnegative residual");
  require(!cells.empty(), label + ": expected a residual-only cell");
  require(cells.size() == 1, label + ": expected one residual-only cell");
  require(cells[0].cell_lo.empty(), label + ": expected no cell_lo axes");
  require(cells[0].cell_hi.empty(), label + ": expected no cell_hi axes");
  require(cells[0].member_idx.size() == n_pts,
          label + ": expected all points in the residual-only cell");

  const std::vector<float> queries = {
      0.0f, 0.0f, 0.0f,
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.5f, 2.0f,
  };
  require(queries.size() % dim == 0, "query fixture dim mismatch");

  for (size_t q = 0; q < queries.size() / dim; ++q)
  {
    const float* query = queries.data() + q * dim;
    float total_sq = 0.0f;
    for (size_t col = 0; col < dim; ++col)
    {
      const float diff = query[col] - centroid[col];
      total_sq += diff * diff;
    }
    const float q_r_norm = std::sqrt(total_sq);
    const float cluster_lb = std::max(0.0f, q_r_norm - residual);
    const float cell_lb =
        utils::cell_lb_batch(cells[0].cell_lo.data(), cells[0].cell_hi.data(),
                             cells[0].res_lo, cells[0].res_hi, nullptr,
                             q_r_norm, 0);

    for (size_t i = 0; i < n_pts; ++i)
    {
      const float true_dist =
          l2_distance(query, points.data() + i * dim, dim);
      if (cluster_lb > true_dist + 1e-5f || cell_lb > true_dist + 1e-5f)
      {
        std::cerr << label << " q=" << q << " point=" << i
                  << " cluster_lb=" << cluster_lb << " cell_lb=" << cell_lb
                  << " true_dist=" << true_dist << std::endl;
        throw std::runtime_error(label + ": residual-only LB violation");
      }
    }
  }
}
}  // namespace

int main()
{
  try
  {
    verify_residual_only_bounds({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f}, 1,
                                3, 2, "single-point");
    verify_residual_only_bounds(
        {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f},
        {1.0f, 2.0f, 3.0f}, 3, 3, 2, "all-equal");
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
