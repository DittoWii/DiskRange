#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../utils/dist_func.h"
#include "../utils/vector_cast.h"
#include "BuildPhaseTimer.h"
#include "ClusterIO.h"
#include "OpenBlasThreadGuard.h"
#include "ResolvedConfig.h"
#include "SQGCentroidIndex.h"

#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#else
#error "kmeans-train-gemm requires DJ_USE_OPENBLAS; configure cmake with -DDJ_USE_OPENBLAS=ON"
#endif

#define EPS (1 / 1024.)

auto dist_ = utils::L2Sqr;

inline size_t bounded_kmeans_training_sample_size(size_t n, size_t cluster_num)
{
  constexpr size_t kSamplePerCentroid = 10;
  if (cluster_num == 0)
  {
    return 0;
  }
  if (cluster_num > std::numeric_limits<size_t>::max() / kSamplePerCentroid)
  {
    return n;
  }
  return std::min(n, cluster_num * kSamplePerCentroid);
}

template <typename T>
struct DefaultInitAllocator : std::allocator<T>
{
  using value_type = T;

  DefaultInitAllocator() noexcept = default;

  template <typename U>
  DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept
  {
  }

  template <typename U>
  struct rebind
  {
    using other = DefaultInitAllocator<U>;
  };

  template <typename U>
  void construct(U* p)
  {
    ::new (static_cast<void*>(p)) U;
  }

  template <typename U, typename... Args>
  void construct(U* p, Args&&... args)
  {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }
};

enum class KmeansObjective
{
  Standard,
  Anisotropic,
};

inline KmeansObjective kmeans_objective_from_string(const std::string& value)
{
  if (value == "standard") return KmeansObjective::Standard;
  if (value == "anisotropic") return KmeansObjective::Anisotropic;
  throw std::runtime_error("unrecognized partition_objective '" + value +
                           "'; must be 'standard' or 'anisotropic'");
}

inline float anisotropic_loss(const float* x, const float* c, float eta,
                              size_t d)
{
  float x_norm_sq = 0.0f;
  float dot = 0.0f;
  float residual_sq = 0.0f;
  for (size_t i = 0; i < d; ++i)
  {
    const float r = x[i] - c[i];
    residual_sq += r * r;
    x_norm_sq += x[i] * x[i];
    dot += r * x[i];
  }
  const float r_parallel_sq = (dot * dot) / (x_norm_sq + 1e-12f);
  const float r_orthogonal_sq = std::max(0.0f, residual_sq - r_parallel_sq);
  return eta * r_parallel_sq + r_orthogonal_sq;
}

inline void normalize_vectors_in_place(float* data, size_t n, size_t d)
{
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i)
  {
    float* x = data + i * d;
    float norm_sq = 0.0f;
    for (size_t k = 0; k < d; ++k)
    {
      norm_sq += x[k] * x[k];
    }
    const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
    for (size_t k = 0; k < d; ++k)
    {
      x[k] *= inv_norm;
    }
  }
}

inline void read_data_batch_as_float(DataReader& data_reader,
                                     std::vector<float>& data_buffer,
                                     std::vector<uint8_t>& byte_buffer,
                                     size_t num)
{
  const size_t value_count = num * data_reader.d;
  if (range_search_config::is_byte_dtype(data_reader.dtype))
  {
    if (byte_buffer.size() < value_count)
    {
      byte_buffer.resize(value_count);
    }
    data_reader.get_batch(reinterpret_cast<char*>(byte_buffer.data()), num);
    utils::cast_byte_payload_to_float(data_reader.dtype, byte_buffer.data(),
                                      value_count, data_buffer.data());
    return;
  }
  data_reader.get_batch(reinterpret_cast<char*>(data_buffer.data()), num);
}

inline void assert_anisotropic_unit_norm(size_t n, const float* data, size_t d)
{
  const size_t sample_n = std::min(n, static_cast<size_t>(1000));
  for (size_t i = 0; i < sample_n; ++i)
  {
    float norm_sq = 0.0f;
    for (size_t k = 0; k < d; ++k)
    {
      const float v = data[i * d + k];
      norm_sq += v * v;
    }
    const float norm = std::sqrt(norm_sq);
    if (norm < 0.95f || norm > 1.05f)
    {
      std::cerr << "FATAL: anisotropic K-means requires unit-normalized input, "
                << "but vec[" << i << "] has norm " << norm
                << " not in [0.95, 1.05]\n";
      std::abort();
    }
  }
}

inline size_t best_anisotropic_centroid(const float* x,
                                        const std::vector<float>& centroids,
                                        size_t cluster_num, size_t d,
                                        float eta)
{
  size_t best = 0;
  float min_loss = anisotropic_loss(x, centroids.data(), eta, d);
  for (size_t c = 1; c < cluster_num; ++c)
  {
    const float loss =
        anisotropic_loss(x, centroids.data() + c * d, eta, d);
    if (loss < min_loss)
    {
      min_loss = loss;
      best = c;
    }
  }
  return best;
}

struct Kmeans
{
  size_t d;
  size_t cluster_num;
  std::vector<float> centroids_;
  std::vector<std::vector<size_t>> inverted_list_;
  std::vector<std::vector<std::pair<float, unsigned>>> dists;
  KmeansObjective objective_ = KmeansObjective::Standard;
  float eta_ = 1.0f;
  bool trace_training_loss_ = false;
  std::vector<float> training_loss_trace_;

  Kmeans(size_t d, size_t cluster_num)
      : Kmeans(d, cluster_num, KmeansObjective::Standard, 1.0f)
  {
  }

  Kmeans(size_t d, size_t cluster_num, KmeansObjective objective, float eta)
      : d(d), cluster_num(cluster_num), objective_(objective), eta_(eta)
  {
    if (!(eta_ > 0.0f))
    {
      throw std::runtime_error("eta must be positive");
    }
    centroids_.resize(cluster_num * d);
    inverted_list_.resize(cluster_num);
  }

  float total_anisotropic_loss(size_t n, const float* data) const
  {
    float total = 0.0f;
#pragma omp parallel for reduction(+ : total)
    for (size_t i = 0; i < n; ++i)
    {
      size_t best = best_anisotropic_centroid(data + i * d, centroids_,
                                              cluster_num, d, eta_);
      total += anisotropic_loss(data + i * d, centroids_.data() + best * d,
                                eta_, d);
    }
    return total;
  }

  void train(size_t n, float* data, size_t kmeans_iter = 10)
  {
    if (objective_ == KmeansObjective::Anisotropic)
    {
      assert_anisotropic_unit_norm(n, data, d);
    }
    if (trace_training_loss_)
    {
      training_loss_trace_.clear();
      training_loss_trace_.reserve(kmeans_iter);
    }
    size_t bucket_size = n / cluster_num;
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i)
    {
      float* data_ptr = data + i * bucket_size * d;
      memcpy(centroids_.data() + i * d, data_ptr, d * sizeof(float));
    }
    std::vector<size_t> assign(n);
    std::vector<float> x_norm_sq;
    std::vector<float, DefaultInitAllocator<float>> chunk_dist;
    std::vector<float> c_norm_sq;
    size_t chunk_size = 0;
    if (objective_ == KmeansObjective::Standard)
    {
      x_norm_sq.resize(n);
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i)
      {
        float sum = 0.0f;
        const float* data_ptr = data + i * d;
        for (size_t k = 0; k < d; ++k)
        {
          sum += data_ptr[k] * data_ptr[k];
        }
        x_norm_sq[i] = sum;
      }
      constexpr size_t kTargetChunkBytes = size_t(2) << 30;
      chunk_size = std::max<size_t>(
          size_t(1024),
          std::min<size_t>(
              n, kTargetChunkBytes / (cluster_num * sizeof(float))));
      chunk_size = std::min(chunk_size, n);
      const size_t chunk_dist_values = chunk_size * cluster_num;
      chunk_dist.resize(chunk_dist_values);
      constexpr size_t kPageStrideFloats = 4096 / sizeof(float);
#pragma omp parallel for schedule(static)
      for (size_t offset = 0; offset < chunk_dist_values;
           offset += kPageStrideFloats)
      {
        chunk_dist[offset] = 0.0f;
      }
      c_norm_sq.resize(cluster_num);
    }
    std::optional<OpenBlasThreadGuard> blas_guard;
    if (objective_ == KmeansObjective::Standard)
    {
      blas_guard.emplace(omp_get_max_threads());
    }
    while (kmeans_iter--)
    {
      if (objective_ == KmeansObjective::Standard)
      {
#pragma omp parallel for
        for (size_t j = 0; j < cluster_num; ++j)
        {
          float sum = 0.0f;
          const float* centroid = centroids_.data() + j * d;
          for (size_t k = 0; k < d; ++k)
          {
            sum += centroid[k] * centroid[k];
          }
          c_norm_sq[j] = sum;
        }
        for (size_t base = 0; base < n; base += chunk_size)
        {
          const size_t this_chunk = std::min(chunk_size, n - base);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                      static_cast<int>(this_chunk),
                      static_cast<int>(cluster_num), static_cast<int>(d),
                      -2.0f, data + base * d, static_cast<int>(d),
                      centroids_.data(), static_cast<int>(d), 0.0f,
                      chunk_dist.data(), static_cast<int>(cluster_num));
#pragma omp parallel for
          for (size_t i = 0; i < this_chunk; ++i)
          {
            const float xnorm = x_norm_sq[base + i];
            float* row = chunk_dist.data() + i * cluster_num;
            float min_dist = row[0] + xnorm + c_norm_sq[0];
            size_t best_j = 0;
            for (size_t j = 1; j < cluster_num; ++j)
            {
              const float dist = row[j] + xnorm + c_norm_sq[j];
              if (dist < min_dist)
              {
                best_j = j;
                min_dist = dist;
              }
            }
            assign[base + i] = best_j;
          }
        }
      }
      else
      {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
          assign[i] = 0;
          float* data_ptr = data + i * d;
          float min_dist =
              anisotropic_loss(data_ptr, centroids_.data(), eta_, d);
          for (size_t j = 1; j < cluster_num; ++j)
          {
            const float dist =
                anisotropic_loss(data_ptr, centroids_.data() + j * d, eta_, d);
            if (dist < min_dist)
            {
              assign[i] = j;
              min_dist = dist;
            }
          }
        }
      }
      std::vector<std::vector<size_t>> ivf(cluster_num);
      for (size_t i = 0; i < n; ++i)
      {
        ivf[assign[i]].push_back(i);
      }
      if (objective_ == KmeansObjective::Standard)
      {
#pragma omp parallel for
        for (std::size_t i = 0; i < cluster_num; ++i)
        {
          if (ivf[i].size())
          {
            for (std::size_t j = 0; j < d; ++j)
            {
              centroids_[i * d + j] = data[ivf[i][0] * d + j];
            }
            for (std::size_t j = 1; j < ivf[i].size(); ++j)
            {
              for (std::size_t k = 0; k < d; ++k)
              {
                centroids_[i * d + k] += data[ivf[i][j] * d + k];
              }
            }
            for (std::size_t j = 0; j < d; ++j)
            {
              centroids_[i * d + j] /= ivf[i].size();
            }
          }
        }
      }
      else
      {
#pragma omp parallel for
        for (std::size_t i = 0; i < cluster_num; ++i)
        {
          if (ivf[i].empty())
          {
            continue;
          }
          Eigen::MatrixXf A =
              Eigen::MatrixXf::Identity(static_cast<Eigen::Index>(d),
                                        static_cast<Eigen::Index>(d)) *
              static_cast<float>(ivf[i].size());
          Eigen::VectorXf b =
              Eigen::VectorXf::Zero(static_cast<Eigen::Index>(d));
          for (const auto idx : ivf[i])
          {
            Eigen::Map<const Eigen::VectorXf> x(data + idx * d,
                                                static_cast<Eigen::Index>(d));
            const float x_norm_sq = x.squaredNorm();
            A.noalias() +=
                (eta_ - 1.0f) * (x * x.transpose()) / (x_norm_sq + 1e-12f);
            b.noalias() += eta_ * x;
          }
          const Eigen::VectorXf c_star = A.ldlt().solve(b);
          memcpy(centroids_.data() + i * d, c_star.data(),
                 d * sizeof(float));
        }
      }
      for (size_t ci = 0; ci < cluster_num; ci++)
      {
        if (ivf[ci].size() == 0)
        {
          size_t cj;
          for (cj = 0; 1; cj = (cj + 1) % cluster_num)
          {
            float p = (ivf[cj].size() - 1.0) / (float)(n - cluster_num);
            float r = (float)rand() / RAND_MAX;
            if (r < p)
            {
              break;
            }
          }
          memcpy(centroids_.data() + ci * d, centroids_.data() + cj * d,
                 sizeof(float) * d);
          for (size_t j = 0; j < d; j++)
          {
            if (j % 2 == 0)
            {
              centroids_[ci * d + j] *= 1 + EPS;
              centroids_[cj * d + j] *= 1 - EPS;
            }
            else
            {
              centroids_[ci * d + j] *= 1 - EPS;
              centroids_[cj * d + j] *= 1 + EPS;
            }
          }
        }
      }
      if (trace_training_loss_ && objective_ == KmeansObjective::Anisotropic)
      {
        training_loss_trace_.push_back(total_anisotropic_loss(n, data));
      }
    }
  }

  void add(size_t n, float* data, std::vector<size_t>& ids)
  {
    std::vector<size_t> assign(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
      size_t c = 0;
      float* data_ptr = data + i * d;
      float min_dist =
          objective_ == KmeansObjective::Standard
              ? dist_(data_ptr, centroids_.data(), &d)
              : anisotropic_loss(data_ptr, centroids_.data(), eta_, d);
      for (size_t j = 1; j < cluster_num; ++j)
      {
        const float dist =
            objective_ == KmeansObjective::Standard
                ? dist_(data_ptr, centroids_.data() + j * d, &d)
                : anisotropic_loss(data_ptr, centroids_.data() + j * d, eta_,
                                   d);
        if (dist < min_dist)
        {
          c = j;
          min_dist = dist;
        }
      }
      assign[i] = c;
    }
    for (size_t i = 0; i < n; ++i)
    {
      inverted_list_[assign[i]].push_back(ids[i]);
    }
  }

  void add_anisotropic_exact(size_t n, float* data, std::vector<size_t>& ids)
  {
    if (objective_ != KmeansObjective::Anisotropic)
    {
      throw std::runtime_error(
          "add_anisotropic_exact requires anisotropic objective");
    }
    std::vector<size_t> assign(n);
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i)
    {
      assign[i] = best_anisotropic_centroid(data + i * d, centroids_,
                                            cluster_num, d, eta_);
    }
    for (size_t i = 0; i < n; ++i)
    {
      inverted_list_[assign[i]].push_back(ids[i]);
    }
  }

  void add_standard_prebuild_batch(size_t n, float* data,
                                   std::vector<size_t>& ids,
                                   const SQGCentroidIndex& sqg,
                                   bool normalize_points)
  {
    if (objective_ != KmeansObjective::Standard)
    {
      throw std::runtime_error(
          "add_standard_prebuild_batch requires standard objective");
    }
    if (normalize_points)
    {
      normalize_vectors_in_place(data, n, d);
    }
    add2choice(n, data, ids, sqg);
  }

  void add2choice(size_t n, float* data, std::vector<size_t>& ids,
                  const SQGCentroidIndex& sqg)
  {
    struct TwoChoice
    {
      size_t best_id = 0;
      float best_dist = 0.0f;
      size_t alt_id = 0;
      float alt_dist = 0.0f;
    };

    std::vector<TwoChoice> assign(n);
#pragma omp parallel
    {
      auto ctx = sqg.make_search_context();
      std::array<uint32_t, 2> ids_buf{};
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < n; ++i)
      {
        const float* data_ptr = data + i * d;
        sqg.search_into(data_ptr, 2, ids_buf.data(), ctx);

        std::array<float, 2> dist_sq{{0.0f, 0.0f}};
        for (size_t k = 0; k < ids_buf.size(); ++k)
        {
          const float* centroid_ptr =
              centroids_.data() + static_cast<size_t>(ids_buf[k]) * d;
          for (size_t col = 0; col < d; ++col)
          {
            const float diff = data_ptr[col] - centroid_ptr[col];
            dist_sq[k] += diff * diff;
          }
        }

        const size_t best_idx = (dist_sq[0] <= dist_sq[1]) ? 0 : 1;
        const size_t alt_idx = 1 - best_idx;
        assign[i].best_id = ids_buf[best_idx];
        assign[i].best_dist = dist_sq[best_idx];
        assign[i].alt_id = ids_buf[alt_idx];
        assign[i].alt_dist = dist_sq[alt_idx];
      }
    }

    for (size_t i = 0; i < n; ++i)
    {
      const TwoChoice& choice = assign[i];
      const float alt_dist = choice.alt_dist;
      const float best_dist = choice.best_dist;
      if (alt_dist < best_dist * 1.2f &&
          inverted_list_[choice.alt_id].size() <
              inverted_list_[choice.best_id].size())
      {
        inverted_list_[choice.alt_id].push_back(ids[i]);
      }
      else
      {
        inverted_list_[choice.best_id].push_back(ids[i]);
      }
    }
  }

  void assign()
  {
    std::vector<std::pair<float, unsigned>> priority;
    for (size_t i = 0; i < dists.size(); i++)
    {
      priority.push_back(
          std::make_pair(dists[i][1].first - dists[i][0].first, i));
    }
    std::sort(priority.begin(), priority.end());
    for (size_t i = 0; i < priority.size(); i++)
    {
      auto id = priority[i].second;
      if (inverted_list_[dists[id][0].second].size() <
          inverted_list_[dists[id][1].second].size())
      {
        inverted_list_[dists[id][0].second].push_back(id);
      }
      else
      {
        inverted_list_[dists[id][1].second].push_back(id);
      }
    }
  }
};

inline std::vector<size_t> build_cluster_lookup(
    const std::vector<std::vector<size_t>>& inverted_list, size_t n)
{
  std::vector<size_t> lookup(n, std::numeric_limits<size_t>::max());
  for (size_t c = 0; c < inverted_list.size(); ++c)
  {
    for (const size_t id : inverted_list[c])
    {
      if (id < n)
      {
        lookup[id] = c;
      }
    }
  }
  return lookup;
}

inline void verify_anisotropic_assignments_memory(
    const Kmeans& kmeans, const float* data, size_t n, size_t d,
    const std::vector<std::vector<size_t>>& inverted_list)
{
  if (kmeans.d != d)
  {
    throw std::runtime_error("anisotropic verification dim mismatch");
  }
  const size_t sample_target = std::min(n, static_cast<size_t>(10000));
  if (sample_target == 0)
  {
    return;
  }
  const size_t stride = std::max<size_t>(1, n / sample_target);
  const std::vector<size_t> lookup = build_cluster_lookup(inverted_list, n);
  size_t checked = 0;
  size_t mismatches = 0;
  for (size_t id = 0; id < n && checked < sample_target; id += stride)
  {
    const size_t expected =
        best_anisotropic_centroid(data + id * d, kmeans.centroids_,
                                  kmeans.cluster_num, kmeans.d, kmeans.eta_);
    if (lookup[id] != expected)
    {
      ++mismatches;
    }
    ++checked;
  }
  if (checked == 0)
  {
    return;
  }
  const double mismatch_rate = static_cast<double>(mismatches) / checked;
  if (mismatch_rate > 0.005)
  {
    std::cerr << "FATAL: anisotropic build produced partition that disagrees "
                 "with argmin anisotropic_loss in "
              << (mismatch_rate * 100.0)
              << "% of samples; persisted inverted_list_ is incorrect\n";
    std::abort();
  }
}

inline void verify_anisotropic_assignments_streaming(
    const Kmeans& kmeans, const std::string& datafile, bool normalize_points,
    const std::vector<std::vector<size_t>>& inverted_list,
    uint32_t expected_v1_dtype = range_search_config::kVecDTypeFloat32)
{
  DataReader data_reader(datafile, expected_v1_dtype);
  const size_t n = data_reader.n;
  const size_t d = data_reader.d;
  const size_t sample_target = std::min(n, static_cast<size_t>(10000));
  if (sample_target == 0)
  {
    return;
  }
  const size_t stride = std::max<size_t>(1, n / sample_target);
  const std::vector<size_t> lookup = build_cluster_lookup(inverted_list, n);
  size_t checked = 0;
  size_t mismatches = 0;
  const size_t batch_size = std::max<size_t>(1, n / 1000);
  std::vector<float> data_buffer(batch_size * d);
  std::vector<uint8_t> byte_buffer;
  data_reader.reset();
  for (size_t i = 0; i < div_round_up(n, batch_size) && checked < sample_target;
       i++)
  {
    const size_t num =
        batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
    read_data_batch_as_float(data_reader, data_buffer, byte_buffer, num);
    if (normalize_points)
    {
      normalize_vectors_in_place(data_buffer.data(), num, d);
    }
    for (size_t j = 0; j < num && checked < sample_target; ++j)
    {
      const size_t id = i * batch_size + j;
      if (id % stride != 0)
      {
        continue;
      }
      const size_t expected =
          best_anisotropic_centroid(data_buffer.data() + j * d,
                                    kmeans.centroids_, kmeans.cluster_num,
                                    kmeans.d, kmeans.eta_);
      if (lookup[id] != expected)
      {
        ++mismatches;
      }
      ++checked;
    }
  }
  if (checked == 0)
  {
    return;
  }
  const double mismatch_rate = static_cast<double>(mismatches) / checked;
  if (mismatch_rate > 0.005)
  {
    std::cerr << "FATAL: anisotropic build produced partition that disagrees "
                 "with argmin anisotropic_loss in "
              << (mismatch_rate * 100.0)
              << "% of samples; persisted inverted_list_ is incorrect\n";
    std::abort();
  }
}

void one_level_kmeans(const ResolvedConfig& config)
{
  auto datafile = config.data_file;
  auto cluster_num = config.cluster_num;
  const KmeansObjective objective =
      kmeans_objective_from_string(config.partition_objective);
  DataReader data_reader(datafile, config.vec_dtype);
  size_t n = data_reader.n;
  size_t d = data_reader.d;
  const size_t train_sample_num =
      bounded_kmeans_training_sample_size(n, cluster_num);
  if (train_sample_num < cluster_num)
  {
    throw std::runtime_error("k-means training sample smaller than cluster_num");
  }
  std::vector<float> training_sample(train_sample_num * d);
  {
    BuildPhaseTimer _t("km.sample_training");
    std::vector<size_t> sample_id(n);
    for (size_t i = 0; i < n; i++) sample_id[i] = i;
    std::srand(283);
    std::random_shuffle(sample_id.begin(), sample_id.end());
    std::sort(sample_id.begin(), sample_id.begin() + train_sample_num);
    size_t batch_num = 1000;
    size_t batch_size = std::max<size_t>(1, n / batch_num);
    std::vector<float> data_buffer(batch_size * d);
    std::vector<uint8_t> byte_buffer;
    data_reader.reset();
    size_t ptr = 0;
    for (size_t i = 0; i < div_round_up(n, batch_size); i++)
    {
      size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
      read_data_batch_as_float(data_reader, data_buffer, byte_buffer, num);
      while (ptr < train_sample_num && sample_id[ptr] < i * batch_size + num)
      {
        memcpy(training_sample.data() + ptr * d,
               data_buffer.data() + (sample_id[ptr] - i * batch_size) * d,
               d * sizeof(float));
        ptr++;
      }
    }
    if (config.unit_normalize_prebuild)
    {
      normalize_vectors_in_place(training_sample.data(), train_sample_num, d);
    }
  }
  Kmeans kmeans(d, cluster_num, objective, config.eta);
  {
    BuildPhaseTimer _t("km.train");
    kmeans.train(train_sample_num, training_sample.data(), 5);
  }
  if (objective == KmeansObjective::Anisotropic)
  {
    BuildPhaseTimer _t("km.assign_full_data_anisotropic");
    size_t batch_num = 1000;
    size_t batch_size = std::max<size_t>(1, n / batch_num);
    std::vector<float> data_buffer(batch_size * d);
    std::vector<uint8_t> byte_buffer;
    std::vector<size_t> ids(batch_size);
    data_reader.reset();
    for (size_t i = 0; i < div_round_up(n, batch_size); i++)
    {
      size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
      for (size_t j = 0; j < num; j++)
      {
        ids[j] = i * batch_size + j;
      }
      read_data_batch_as_float(data_reader, data_buffer, byte_buffer, num);
      if (config.unit_normalize_prebuild)
      {
        normalize_vectors_in_place(data_buffer.data(), num, d);
      }
      kmeans.add_anisotropic_exact(num, data_buffer.data(), ids);
    }
    verify_anisotropic_assignments_streaming(
        kmeans, datafile, config.unit_normalize_prebuild,
        kmeans.inverted_list_, config.vec_dtype);
  }
  else
  {
    SQGCentroidIndex sqg_centroid;
    {
      BuildPhaseTimer _t("km.centroid_sqg_build+save");
      sqg_centroid.build(kmeans.centroids_, cluster_num, d, config.sqg_m,
                         config.sqg_num_iter, config.sqg_ef_build,
                         config.sqg_centroid_ef_search);
      sqg_centroid.save(config.sqg_path);
    }
    {
      BuildPhaseTimer _t("km.assign_full_data");
      size_t batch_num = 1000;
      size_t batch_size = std::max<size_t>(1, n / batch_num);
      std::vector<float> data_buffer(batch_size * d);
      std::vector<uint8_t> byte_buffer;
      std::vector<size_t> ids(batch_size);
      data_reader.reset();
      for (size_t i = 0; i < div_round_up(n, batch_size); i++)
      {
        size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
        for (size_t j = 0; j < num; j++)
        {
          ids[j] = i * batch_size + j;
        }
        read_data_batch_as_float(data_reader, data_buffer, byte_buffer, num);
        kmeans.add_standard_prebuild_batch(num, data_buffer.data(), ids,
                                           sqg_centroid,
                                           config.unit_normalize_prebuild);
      }
    }
  }
  ClusterWriter cluster_writer(datafile, config.cluster_path,
                               config.metadata_path, config.pca_rank,
                               config.slab_count, config.target_cell_vecs,
                               config.unit_normalize_prebuild,
                               config.vec_dtype);
  cluster_writer.writeClusters(kmeans.inverted_list_, kmeans.centroids_.data(),
                               config.mem_budget);
  cluster_writer.writeMetadata(kmeans.inverted_list_);
}
