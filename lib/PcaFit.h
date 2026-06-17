#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

struct PcaCell
{
  std::vector<size_t> member_idx;
  std::vector<float> cell_lo;
  std::vector<float> cell_hi;
  float res_lo = 0.0f;
  float res_hi = 0.0f;
};

inline void zero_pca_outputs(size_t d, size_t pca_rank, float* basis_out,
                             float* pca_min_out, float* pca_max_out)
{
  if (pca_rank == 0)
  {
    return;
  }
  std::fill(basis_out, basis_out + pca_rank * d, 0.0f);
  std::fill(pca_min_out, pca_min_out + pca_rank, 0.0f);
  std::fill(pca_max_out, pca_max_out + pca_rank, 0.0f);
}

inline float point_centroid_distance(const float* point, const float* centroid,
                                     size_t d)
{
  float sum = 0.0f;
  for (size_t j = 0; j < d; ++j)
  {
    const float diff = point[j] - centroid[j];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

inline size_t pc2_kdtree_max_depth(size_t n_pts, size_t target_cell_vecs)
{
  if (target_cell_vecs == 0)
  {
    throw std::runtime_error("target_cell_vecs must be positive");
  }
  const double ratio =
      std::max(1.0, static_cast<double>(n_pts) /
                        static_cast<double>(target_cell_vecs));
  return static_cast<size_t>(std::ceil(std::log2(ratio)));
}

inline float pc2_median_threshold(const std::vector<size_t>& members,
                                  const std::vector<float>& coords,
                                  size_t effective_rank, size_t axis)
{
  std::vector<float> values;
  values.reserve(members.size());
  for (const size_t member : members)
  {
    values.push_back(coords[member * effective_rank + axis]);
  }
  const size_t n = values.size();
  const size_t lower_index = (n - 1) / 2;
  auto lower_it = values.begin() + static_cast<std::ptrdiff_t>(lower_index);
  std::nth_element(values.begin(), lower_it, values.end());
  const float lower = *lower_it;
  if (n % 2 != 0)
  {
    return lower;
  }
  const float upper = *std::min_element(lower_it + 1, values.end());
  return (lower + upper) / 2.0f;
}

inline PcaCell make_pca_cell(const std::vector<size_t>& members,
                             const std::vector<float>& coords,
                             const std::vector<float>& residual_norms,
                             size_t effective_rank)
{
  if (members.empty())
  {
    throw std::runtime_error("cannot create an empty PCA cell");
  }

  PcaCell cell;
  cell.member_idx = members;
  cell.cell_lo.assign(effective_rank, std::numeric_limits<float>::infinity());
  cell.cell_hi.assign(effective_rank, -std::numeric_limits<float>::infinity());
  cell.res_lo = std::numeric_limits<float>::infinity();
  cell.res_hi = -std::numeric_limits<float>::infinity();

  for (const size_t member : members)
  {
    for (size_t axis = 0; axis < effective_rank; ++axis)
    {
      const float coord = coords[member * effective_rank + axis];
      cell.cell_lo[axis] = std::min(cell.cell_lo[axis], coord);
      cell.cell_hi[axis] = std::max(cell.cell_hi[axis], coord);
    }
    const float residual = residual_norms[member];
    cell.res_lo = std::min(cell.res_lo, residual);
    cell.res_hi = std::max(cell.res_hi, residual);
  }
  return cell;
}

inline void build_kdtree_cells_recursive(
    const std::vector<size_t>& members, const std::vector<float>& coords,
    const std::vector<float>& residual_norms, size_t effective_rank,
    size_t target_cell_vecs, size_t depth, size_t max_depth,
    std::vector<PcaCell>& out)
{
  if (members.empty())
  {
    return;
  }
  if (effective_rank == 0 || members.size() <= target_cell_vecs ||
      depth >= max_depth)
  {
    out.push_back(make_pca_cell(members, coords, residual_norms,
                                effective_rank));
    return;
  }

  const size_t axis = depth % effective_rank;
  const float threshold =
      pc2_median_threshold(members, coords, effective_rank, axis);
  std::vector<size_t> left;
  std::vector<size_t> right;
  left.reserve((members.size() + 1) / 2);
  right.reserve(members.size() / 2);
  for (const size_t member : members)
  {
    const float coord = coords[member * effective_rank + axis];
    if (coord <= threshold)
    {
      left.push_back(member);
    }
    else
    {
      right.push_back(member);
    }
  }
  if (left.empty() || right.empty())
  {
    out.push_back(make_pca_cell(members, coords, residual_norms,
                                effective_rank));
    return;
  }

  build_kdtree_cells_recursive(left, coords, residual_norms, effective_rank,
                               target_cell_vecs, depth + 1, max_depth, out);
  build_kdtree_cells_recursive(right, coords, residual_norms, effective_rank,
                               target_cell_vecs, depth + 1, max_depth, out);
}

inline std::vector<PcaCell> build_kdtree_cells(
    const std::vector<float>& coords, const std::vector<float>& residual_norms,
    size_t n_pts, size_t effective_rank, size_t target_cell_vecs)
{
  std::vector<PcaCell> cells;
  if (n_pts == 0)
  {
    return cells;
  }
  std::vector<size_t> members(n_pts);
  std::iota(members.begin(), members.end(), size_t{0});
  const size_t max_depth = pc2_kdtree_max_depth(n_pts, target_cell_vecs);
  build_kdtree_cells_recursive(members, coords, residual_norms, effective_rank,
                               target_cell_vecs, 0, max_depth, cells);
  return cells;
}

inline void fit_pca_bucket(const float* points, const float* centroid,
                           size_t n_pts, size_t d, size_t pca_rank,
                           float* basis_out, float* pca_min_out,
                           float* pca_max_out, float* residual_out,
                           uint32_t* effective_rank_out,
                           size_t target_cell_vecs,
                           std::vector<PcaCell>* cells_out)
{
  if (target_cell_vecs < 2)
  {
    throw std::runtime_error("target_cell_vecs must be >= 2");
  }
  if (cells_out == nullptr)
  {
    throw std::runtime_error("fit_pca_bucket requires a cell output vector");
  }

  zero_pca_outputs(d, pca_rank, basis_out, pca_min_out, pca_max_out);
  *residual_out = 0.0f;
  *effective_rank_out = 0;
  cells_out->clear();
  if (n_pts == 0 || d == 0)
  {
    return;
  }

  auto emit_residual_only = [&]() {
    std::vector<float> residual_norms(n_pts, 0.0f);
    float min_dist = std::numeric_limits<float>::infinity();
    float max_dist = 0.0f;
    for (size_t i = 0; i < n_pts; ++i)
    {
      const float dist = point_centroid_distance(points + i * d, centroid, d);
      residual_norms[i] = dist;
      min_dist = std::min(min_dist, dist);
      max_dist = std::max(max_dist, dist);
    }
    *residual_out = max_dist;
    *effective_rank_out = 0;
    std::vector<float> empty_coords;
    *cells_out = build_kdtree_cells(empty_coords, residual_norms, n_pts, 0,
                                    target_cell_vecs);
    if (!cells_out->empty())
    {
      (*cells_out)[0].res_lo = min_dist;
      (*cells_out)[0].res_hi = max_dist;
    }
  };

  if (pca_rank == 0 || n_pts == 1)
  {
    emit_residual_only();
    return;
  }

  const size_t requested_rank = std::min({pca_rank, n_pts, d});
  Eigen::MatrixXf centered(static_cast<Eigen::Index>(n_pts),
                           static_cast<Eigen::Index>(d));
  for (size_t i = 0; i < n_pts; ++i)
  {
    for (size_t j = 0; j < d; ++j)
    {
      centered(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          points[i * d + j] - centroid[j];
    }
  }

  Eigen::MatrixXf cov = centered.adjoint() * centered;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(cov);
  if (es.info() != Eigen::Success)
  {
    emit_residual_only();
    return;
  }
  const Eigen::VectorXf eigvals = es.eigenvalues();
  const Eigen::MatrixXf eigvecs = es.eigenvectors();
  const Eigen::Index max_idx = eigvals.size() - 1;
  const float max_eig = eigvals(max_idx);
  if (max_eig <= 0.0f)
  {
    emit_residual_only();
    return;
  }

  const float tolerance = 1e-12f * max_eig;
  size_t effective_rank = 0;
  for (size_t j = 0; j < requested_rank; ++j)
  {
    const Eigen::Index eig_idx = max_idx - static_cast<Eigen::Index>(j);
    if (eigvals(eig_idx) > tolerance)
    {
      ++effective_rank;
    }
  }
  if (effective_rank == 0)
  {
    emit_residual_only();
    return;
  }

  for (size_t axis = 0; axis < effective_rank; ++axis)
  {
    const Eigen::Index col_idx = max_idx - static_cast<Eigen::Index>(axis);
    for (size_t col = 0; col < d; ++col)
    {
      basis_out[axis * d + col] =
          eigvecs(static_cast<Eigen::Index>(col), col_idx);
    }
  }

  std::vector<float> pca_min(effective_rank,
                             std::numeric_limits<float>::infinity());
  std::vector<float> pca_max(effective_rank,
                             -std::numeric_limits<float>::infinity());
  std::vector<float> coords_by_point(n_pts * effective_rank, 0.0f);
  std::vector<float> residual_norms(n_pts, 0.0f);
  float max_residual_sq = 0.0f;

  for (size_t i = 0; i < n_pts; ++i)
  {
    std::vector<float> coords(effective_rank, 0.0f);
    float centered_norm_sq = 0.0f;
    float projected_norm_sq = 0.0f;
    for (size_t col = 0; col < d; ++col)
    {
      const float value =
          centered(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(col));
      centered_norm_sq += value * value;
      for (size_t axis = 0; axis < effective_rank; ++axis)
      {
        coords[axis] += value * basis_out[axis * d + col];
      }
    }
    for (size_t axis = 0; axis < effective_rank; ++axis)
    {
      const float coord = coords[axis];
      pca_min[axis] = std::min(pca_min[axis], coord);
      pca_max[axis] = std::max(pca_max[axis], coord);
      coords_by_point[i * effective_rank + axis] = coord;
      projected_norm_sq += coord * coord;
    }
    const float residual_sq =
        std::max(0.0f, centered_norm_sq - projected_norm_sq);
    residual_norms[i] = std::sqrt(residual_sq);
    max_residual_sq = std::max(max_residual_sq, residual_sq);
  }

  for (size_t axis = 0; axis < effective_rank; ++axis)
  {
    pca_min_out[axis] = pca_min[axis];
    pca_max_out[axis] = pca_max[axis];
  }
  *residual_out = std::sqrt(max_residual_sq);
  *effective_rank_out = static_cast<uint32_t>(effective_rank);
  *cells_out = build_kdtree_cells(coords_by_point, residual_norms, n_pts,
                                  effective_rank, target_cell_vecs);
}
