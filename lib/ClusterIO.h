#pragma once

#include <fcntl.h>
#include <liburing.h>
#include <omp.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../utils/dist_func.h"
#include "../utils/utils.h"
#include "../utils/vector_cast.h"
#include "BuildPhaseTimer.h"
#include "DataReader.h"
#include "DataHNSWOracle.h"
#include "IoQueueDepth.h"
#include "PcaFit.h"
#include "OpenBlasThreadGuard.h"

#ifdef DJ_USE_OPENBLAS
#include <cblas.h>
#else
#error "slab-fit-gemm requires DJ_USE_OPENBLAS; configure cmake with -DDJ_USE_OPENBLAS=ON"
#endif

auto dist_l2 = utils::L2Sqr;

#define MAX_IO_SIZE 2147479552
#define PAGE_SIZE 4096

inline constexpr std::array<char, 8> kMetadataMagic = {
    'D', 'J', 'M', 'E', 'T', 'A', '\0', '\0'};
inline constexpr uint64_t kMetadataSchemaVersion = 6u;
inline constexpr float kSlabFitFloat32UnitRoundoff = 5.96e-8f;
inline constexpr float kSlabFitDotProductSafetyFactor = 4.0f;
inline constexpr float kSlabFitCentroidCandidateSafetyFactor = 64.0f;

inline std::string format_metadata_magic(const std::array<char, 8>& magic)
{
  std::ostringstream out;
  out << "0x";
  for (const char value : magic)
  {
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(
               static_cast<unsigned char>(value));
  }
  return out.str();
}

inline void write_aligned_padding(std::ofstream& out,
                                  const std::string& label)
{
  const std::streamoff pos = out.tellp();
  if (pos < 0)
  {
    throw std::runtime_error("failed to query metadata write offset before " +
                             label);
  }
  const size_t remainder = static_cast<size_t>(pos) % 64;
  if (remainder == 0)
  {
    return;
  }
  const size_t padding = 64 - remainder;
  const std::array<char, 64> zeros{};
  out.write(zeros.data(), static_cast<std::streamsize>(padding));
}

inline void skip_aligned_padding(std::ifstream& in, const std::string& path,
                                 const std::string& label)
{
  const std::streamoff pos = in.tellg();
  if (pos < 0)
  {
    throw std::runtime_error("failed to query metadata read offset before " +
                             label + ": " + path);
  }
  const size_t remainder = static_cast<size_t>(pos) % 64;
  if (remainder == 0)
  {
    return;
  }
  in.seekg(static_cast<std::streamoff>(64 - remainder), std::ios::cur);
  if (!in)
  {
    throw std::runtime_error("failed to skip metadata padding before " +
                             label + ": " + path);
  }
}

inline void normalize_l2_batch_in_place(float* data, size_t n, size_t d)
{
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i)
  {
    float norm_sq = 0.0f;
    float* x = data + i * d;
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

struct ClusterWriter
{
  DataReader data_reader;
  std::string clusterfile;
  std::ofstream fcluster;
  std::ofstream fmeta;

  size_t n;
  size_t d;
  uint32_t vec_dtype = range_search_config::kVecDTypeFloat32;
  size_t bytes_per_element = sizeof(float);
  size_t cluster_num;
  size_t max_points;
  size_t pca_rank;
  size_t slab_count;
  size_t target_cell_vecs;
  std::vector<size_t> bucket_sizes;
  float* centroids_;
  std::vector<float> radii;
  std::vector<float> pca_basis;
  std::vector<float> pca_min;
  std::vector<float> pca_max;
  std::vector<float> pca_residual_radius;
  std::vector<uint32_t> pca_effective_rank;
  std::vector<uint32_t> slab_neighbour_indices;
  std::vector<float> slab_norms;
  std::vector<float> slab_l;
  std::vector<float> slab_h;
  std::vector<std::vector<PcaCell>> cells_by_cluster;
  std::vector<uint64_t> cluster_cell_offsets;
  std::vector<uint64_t> cell_vec_offsets;
  std::vector<float> cell_bounds;
  std::vector<int32_t> sum_for_offset;
  std::vector<int32_t> norm_sq_for_l2;
  bool normalize_points = false;

  ClusterWriter(std::string datafile, std::string clusterfile,
                std::string metafile, size_t pca_rank = 32,
                size_t slab_count = 128, size_t target_cell_vecs = 24,
                bool normalize_points = false,
                uint32_t expected_v1_dtype =
                    range_search_config::kVecDTypeFloat32)
      : data_reader(datafile, expected_v1_dtype),
        clusterfile(clusterfile),
        fcluster(clusterfile, std::ios::binary | std::ios::out),
        fmeta(metafile, std::ios::binary | std::ios::out),
        pca_rank(pca_rank),
        slab_count(slab_count),
        target_cell_vecs(target_cell_vecs),
        normalize_points(normalize_points)
  {
    if (this->target_cell_vecs < 2)
    {
      throw std::runtime_error("target_cell_vecs must be >= 2: " +
                               std::to_string(this->target_cell_vecs));
    }
    n = data_reader.n;
    d = data_reader.d;
    vec_dtype = data_reader.dtype;
    bytes_per_element = data_reader.bytes_per_element;
    if (range_search_config::is_byte_dtype(vec_dtype) && normalize_points)
    {
      throw std::runtime_error(
          "byte vec_dtype does not support normalize_points: " +
          range_search_config::vec_dtype_to_string(vec_dtype));
    }
  }

  void writeClusters(std::vector<std::vector<size_t>>& assignment,
                     float* centroids, float budget)
  {
    BuildPhaseTimer _t_raw("wc.raw_cluster_write");
    centroids_ = centroids;
    std::vector<unsigned> id_cluster_map(n);
    max_points = 0;
    for (size_t i = 0; i < assignment.size(); i++)
    {
      auto& cluster = assignment[i];
      if (cluster.size() > max_points) max_points = cluster.size();
      for (auto id : cluster)
      {
        id_cluster_map[id] = i;
      }
      bucket_sizes.push_back(cluster.size());
      std::sort(cluster.begin(), cluster.end());
    }
    cluster_num = assignment.size();
    radii.resize(cluster_num);
    const size_t vector_bytes = d * bytes_per_element;
    size_t buffer_size =
        div_round_up(budget * (size_t)1024 * 1024 * 1024 / cluster_num,
                     vector_bytes) *
        vector_bytes;
    std::vector<char> buffer(cluster_num * buffer_size);
    std::vector<size_t> buffer_pos(cluster_num);
    std::vector<size_t> file_pos(cluster_num);
    size_t cumu_size = 0;
    for (size_t i = 0; i < cluster_num; i++)
    {
      buffer_pos[i] = buffer_size * i;
      file_pos[i] = cumu_size * vector_bytes;
      cumu_size += bucket_sizes[i];
    }
    size_t batch_size = std::max<size_t>(1, n / 1000);
    std::vector<float> data_buf(batch_size * d);
    std::vector<uint8_t> byte_buf;
    if (range_search_config::is_byte_dtype(vec_dtype))
    {
      byte_buf.resize(batch_size * d);
    }
    for (size_t i = 0; i < div_round_up(n, batch_size); i++)
    {
      size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
      if (range_search_config::is_byte_dtype(vec_dtype))
      {
        data_reader.get_batch(reinterpret_cast<char*>(byte_buf.data()), num);
        utils::cast_byte_payload_to_float(vec_dtype, byte_buf.data(), num * d,
                                          data_buf.data());
      }
      else
      {
        data_reader.get_batch(reinterpret_cast<char*>(data_buf.data()), num);
      }
      if (normalize_points)
      {
        normalize_l2_batch_in_place(data_buf.data(), num, d);
      }
      for (size_t j = 0; j < num; j++)
      {
        auto id = j + batch_size * i;
        auto cluster = id_cluster_map[id];
        float dist = dist_l2(data_buf.data() + j * d, centroids + cluster * d, &d);
        if (dist > radii[cluster]) radii[cluster] = dist;
        const char* payload =
            range_search_config::is_byte_dtype(vec_dtype)
                ? reinterpret_cast<const char*>(byte_buf.data() + j * d)
                : reinterpret_cast<const char*>(data_buf.data() + j * d);
        memcpy(buffer.data() + buffer_pos[cluster], payload, vector_bytes);
        buffer_pos[cluster] += vector_bytes;
        if (buffer_pos[cluster] == buffer_size * (cluster + 1))
        {
          buffer_pos[cluster] -= buffer_size;
          fcluster.seekp(file_pos[cluster], std::ios::beg);
          fcluster.write(buffer.data() + buffer_pos[cluster], buffer_size);
          file_pos[cluster] += buffer_size;
        }
      }
    }
    for (size_t i = 0; i < cluster_num; i++)
    {
      if (buffer_pos[i] % buffer_size != 0)
      {
        fcluster.seekp(file_pos[i], std::ios::beg);
        fcluster.write(buffer.data() + buffer_size * i,
                       buffer_pos[i] % buffer_size);
      }
    }
    const size_t payload_bytes = n * vector_bytes;
    const size_t padded_file_size =
        div_round_up(payload_bytes, PAGE_SIZE) * PAGE_SIZE;
    if (padded_file_size > payload_bytes)
    {
      fcluster.seekp(static_cast<std::streamoff>(padded_file_size - 1),
                     std::ios::beg);
      fcluster.put('\0');
    }
    for (size_t i = 0; i < cluster_num; i++)
    {
      radii[i] = sqrt(radii[i]);
    }
    fcluster.flush();
    if (!fcluster)
    {
      throw std::runtime_error("failed to write cluster file: " + clusterfile);
    }
    _t_raw.stop();
    {
      BuildPhaseTimer _t("wc.fitPcaFromClusterFile");
      fitPcaFromClusterFile();
    }
    {
      BuildPhaseTimer _t("wc.rewriteClusterFileByCellOrder");
      rewriteClusterFileByCellOrder(assignment);
    }
    {
      BuildPhaseTimer _t("wc.fitClusterSlabsFromClusterFile");
      fitClusterSlabsFromClusterFile();
    }
  }

  void writeMetadata(std::vector<std::vector<size_t>>& assignment)
  {
    BuildPhaseTimer _t("wc.writeMetadata");
    fmeta.write(kMetadataMagic.data(),
                static_cast<std::streamsize>(kMetadataMagic.size()));
    const uint64_t schema_version = kMetadataSchemaVersion;
    fmeta.write((char*)&schema_version, sizeof(uint64_t));
    fmeta.write((char*)&n, sizeof(size_t));
    fmeta.write((char*)&d, sizeof(size_t));
    fmeta.write((char*)&cluster_num, sizeof(size_t));
    fmeta.write((char*)&max_points, sizeof(size_t));
    const uint64_t metadata_pca_rank = static_cast<uint64_t>(pca_rank);
    fmeta.write((char*)&metadata_pca_rank, sizeof(uint64_t));
    const uint64_t metadata_target_cell_vecs =
        static_cast<uint64_t>(target_cell_vecs);
    fmeta.write((char*)&metadata_target_cell_vecs, sizeof(uint64_t));
    const uint64_t metadata_vec_dtype = static_cast<uint64_t>(vec_dtype);
    fmeta.write((char*)&metadata_vec_dtype, sizeof(uint64_t));
    fmeta.write((char*)bucket_sizes.data(), sizeof(size_t) * cluster_num);
    fmeta.write((char*)centroids_, sizeof(float) * cluster_num * d);
    fmeta.write((char*)radii.data(), sizeof(float) * cluster_num);
    if (!pca_basis.empty())
    {
      fmeta.write((char*)pca_basis.data(),
                  sizeof(float) * pca_basis.size());
    }
    if (!pca_min.empty())
    {
      fmeta.write((char*)pca_min.data(), sizeof(float) * pca_min.size());
    }
    if (!pca_max.empty())
    {
      fmeta.write((char*)pca_max.data(), sizeof(float) * pca_max.size());
    }
    fmeta.write((char*)pca_residual_radius.data(),
                sizeof(float) * pca_residual_radius.size());
    fmeta.write((char*)pca_effective_rank.data(),
                sizeof(uint32_t) * pca_effective_rank.size());
    for (size_t i = 0; i < cluster_num; i++)
    {
      fmeta.write((char*)assignment[i].data(),
                  assignment[i].size() * sizeof(size_t));
    }
    const uint64_t metadata_slab_count = static_cast<uint64_t>(slab_count);
    fmeta.write((char*)&metadata_slab_count, sizeof(uint64_t));
    if (slab_count > 0)
    {
      fmeta.write((char*)slab_neighbour_indices.data(),
                  sizeof(uint32_t) * slab_neighbour_indices.size());
      fmeta.write((char*)slab_norms.data(),
                  sizeof(float) * slab_norms.size());
      fmeta.write((char*)slab_l.data(), sizeof(float) * slab_l.size());
      fmeta.write((char*)slab_h.data(), sizeof(float) * slab_h.size());
    }
    write_aligned_padding(fmeta, "cluster_cell_offsets");
    fmeta.write((char*)cluster_cell_offsets.data(),
                sizeof(uint64_t) * cluster_cell_offsets.size());
    write_aligned_padding(fmeta, "cell_vec_offsets");
    fmeta.write((char*)cell_vec_offsets.data(),
                sizeof(uint64_t) * cell_vec_offsets.size());
    write_aligned_padding(fmeta, "cell_bounds");
    if (!cell_bounds.empty())
    {
      fmeta.write((char*)cell_bounds.data(),
                  sizeof(float) * cell_bounds.size());
    }
    if (range_search_config::is_byte_dtype(vec_dtype))
    {
      if (sum_for_offset.size() != n || norm_sq_for_l2.size() != n)
      {
        throw std::runtime_error(
            "byte-dtype metadata sum/norm arrays must be computed before writeMetadata");
      }
      fmeta.write(reinterpret_cast<const char*>(sum_for_offset.data()),
                  static_cast<std::streamsize>(sum_for_offset.size() *
                                               sizeof(int32_t)));
      fmeta.write(reinterpret_cast<const char*>(norm_sq_for_l2.data()),
                  static_cast<std::streamsize>(norm_sq_for_l2.size() *
                                               sizeof(int32_t)));
    }

    fmeta.seekp(0, std::ios::end);
    if (!fmeta)
    {
      throw std::runtime_error("failed to write metadata file");
    }
  }

  ~ClusterWriter()
  {
    fcluster.close();
    fmeta.close();
  }

 private:
  void fitPcaFromClusterFile()
  {
    pca_basis.assign(cluster_num * pca_rank * d, 0.0f);
    pca_min.assign(cluster_num * pca_rank, 0.0f);
    pca_max.assign(cluster_num * pca_rank, 0.0f);
    pca_residual_radius.assign(cluster_num, 0.0f);
    pca_effective_rank.assign(cluster_num, 0);
    cells_by_cluster.assign(cluster_num, {});
    cluster_cell_offsets.assign(cluster_num + 1, 0);
    cell_vec_offsets.clear();
    cell_bounds.clear();

    std::vector<size_t> cluster_file_offsets(cluster_num + 1, 0);
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      cluster_file_offsets[cluster_id + 1] =
          cluster_file_offsets[cluster_id] +
          bucket_sizes[cluster_id] * d * bytes_per_element;
    }

    std::atomic<bool> abort_requested(false);
    std::string first_error_message;
    auto record_error = [&](const std::string& message) {
      abort_requested.store(true, std::memory_order_relaxed);
#pragma omp critical
      {
        if (first_error_message.empty())
        {
          first_error_message = message;
        }
      }
    };

#pragma omp parallel
    {
      try
      {
        std::ifstream in(clusterfile, std::ios::binary | std::ios::in);
        if (!in)
        {
          record_error("failed to open cluster file for PCA pass: " +
                       clusterfile);
        }
        std::vector<float> points;
        std::vector<uint8_t> raw_points;

#pragma omp for schedule(dynamic)
        for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
        {
          if (abort_requested.load(std::memory_order_relaxed))
          {
            continue;
          }

          try
          {
            const size_t bucket_size = bucket_sizes[cluster_id];
            const size_t value_count = bucket_size * d;
            points.assign(value_count, 0.0f);
            if (value_count != 0)
            {
              in.clear();
              in.seekg(static_cast<std::streamoff>(
                           cluster_file_offsets[cluster_id]),
                       std::ios::beg);
              if (range_search_config::is_byte_dtype(vec_dtype))
              {
                raw_points.assign(value_count, 0);
                in.read(reinterpret_cast<char*>(raw_points.data()),
                        static_cast<std::streamsize>(value_count));
                if (in.gcount() !=
                    static_cast<std::streamsize>(value_count))
                {
                  record_error(
                      "short cluster payload during PCA pass at cluster " +
                      std::to_string(cluster_id));
                  continue;
                }
                utils::cast_byte_payload_to_float(vec_dtype, raw_points.data(),
                                                  value_count, points.data());
              }
              else
              {
                in.read(reinterpret_cast<char*>(points.data()),
                        static_cast<std::streamsize>(value_count * sizeof(float)));
                if (in.gcount() !=
                    static_cast<std::streamsize>(value_count * sizeof(float)))
                {
                  record_error(
                      "short cluster payload during PCA pass at cluster " +
                      std::to_string(cluster_id));
                  continue;
                }
              }
            }

            const size_t basis_offset = cluster_id * pca_rank * d;
            const size_t extent_offset = cluster_id * pca_rank;
            fit_pca_bucket(
                points.data(), centroids_ + cluster_id * d, bucket_size, d,
                pca_rank, pca_basis.data() + basis_offset,
                pca_min.data() + extent_offset,
                pca_max.data() + extent_offset,
                pca_residual_radius.data() + cluster_id,
                pca_effective_rank.data() + cluster_id, target_cell_vecs,
                &cells_by_cluster[cluster_id]);
          }
          catch (const std::exception& e)
          {
            record_error("PCA pass at cluster " + std::to_string(cluster_id) +
                         ": " + e.what());
            continue;
          }
          catch (...)
          {
            record_error("unexpected exception during PCA pass at cluster " +
                         std::to_string(cluster_id));
            continue;
          }
        }
      }
      catch (...)
      {
        record_error("unexpected exception during PCA pass parallel block");
      }
    }

    if (!first_error_message.empty())
    {
      throw std::runtime_error(first_error_message);
    }
  }

  void rewriteClusterFileByCellOrder(
      std::vector<std::vector<size_t>>& assignment)
  {
    fcluster.close();
    const std::filesystem::path original(clusterfile);
    const std::filesystem::path temp =
        original.parent_path() /
        (original.filename().string() + ".pc2_reorder_tmp");
    std::ifstream in(clusterfile, std::ios::binary | std::ios::in);
    if (!in)
    {
      throw std::runtime_error("failed to open cluster file for PC2 reorder: " +
                               clusterfile);
    }
    std::ofstream out(temp, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out)
    {
      throw std::runtime_error("failed to open temp cluster file for PC2 reorder: " +
                               temp.string());
    }

    cluster_cell_offsets.assign(cluster_num + 1, 0);
    cell_vec_offsets.clear();
    cell_bounds.clear();
    cell_vec_offsets.reserve(n + 1);
    cell_vec_offsets.push_back(0);

    size_t read_offset = 0;
    uint64_t global_vec_offset = 0;
    uint64_t total_cells = 0;
    sum_for_offset.clear();
    norm_sq_for_l2.clear();
    if (range_search_config::is_byte_dtype(vec_dtype))
    {
      sum_for_offset.reserve(n);
      norm_sq_for_l2.reserve(n);
    }
    const size_t vector_bytes = d * bytes_per_element;
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      cluster_cell_offsets[cluster_id] = total_cells;
      const size_t bucket_size = bucket_sizes[cluster_id];
      const size_t value_count = bucket_size * d;
      const size_t byte_count = bucket_size * vector_bytes;
      std::vector<char> points(byte_count);
      if (value_count != 0)
      {
        in.seekg(static_cast<std::streamoff>(read_offset), std::ios::beg);
        in.read(points.data(), static_cast<std::streamsize>(byte_count));
        if (in.gcount() !=
            static_cast<std::streamsize>(byte_count))
        {
          throw std::runtime_error("short cluster payload during PC2 reorder at cluster " +
                                   std::to_string(cluster_id));
        }
      }

      std::vector<char> reordered_points(byte_count);
      std::vector<size_t> reordered_assignment;
      reordered_assignment.reserve(bucket_size);
      size_t out_local = 0;
      for (const PcaCell& cell : cells_by_cluster[cluster_id])
      {
        for (const size_t local_idx : cell.member_idx)
        {
          if (local_idx >= bucket_size)
          {
            throw std::runtime_error("PC2 cell member index out of range at cluster " +
                                     std::to_string(cluster_id));
          }
          std::copy(points.begin() +
                        static_cast<std::ptrdiff_t>(local_idx * vector_bytes),
                    points.begin() +
                        static_cast<std::ptrdiff_t>((local_idx + 1) * vector_bytes),
                    reordered_points.begin() +
                        static_cast<std::ptrdiff_t>(out_local * vector_bytes));
          reordered_assignment.push_back(assignment[cluster_id][local_idx]);
          ++out_local;
        }

        ++total_cells;
        global_vec_offset += static_cast<uint64_t>(cell.member_idx.size());
        cell_vec_offsets.push_back(global_vec_offset);
        cell_bounds.insert(cell_bounds.end(), cell.cell_lo.begin(),
                           cell.cell_lo.end());
        cell_bounds.insert(cell_bounds.end(), cell.cell_hi.begin(),
                           cell.cell_hi.end());
        cell_bounds.push_back(cell.res_lo);
        cell_bounds.push_back(cell.res_hi);
      }
      if (out_local != bucket_size)
      {
        throw std::runtime_error("PC2 cell partition mismatch at cluster " +
                                 std::to_string(cluster_id));
      }
      if (bucket_size != 0)
      {
        out.write(reordered_points.data(),
                  static_cast<std::streamsize>(byte_count));
        if (range_search_config::is_byte_dtype(vec_dtype))
        {
          for (size_t row = 0; row < bucket_size; ++row)
          {
            int64_t sum = 0;
            int64_t norm = 0;
            for (size_t col = 0; col < d; ++col)
            {
              const uint8_t raw = static_cast<uint8_t>(
                  reordered_points[row * d + col]);
              const int64_t value =
                  vec_dtype == range_search_config::kVecDTypeInt8
                      ? static_cast<int64_t>(
                            static_cast<int8_t>(raw))
                      : static_cast<int64_t>(raw);
              sum += value;
              norm += value * value;
            }
            sum_for_offset.push_back(static_cast<int32_t>(sum));
            norm_sq_for_l2.push_back(static_cast<int32_t>(norm));
          }
        }
      }
      assignment[cluster_id].swap(reordered_assignment);
      read_offset += byte_count;
    }
    cluster_cell_offsets[cluster_num] = total_cells;
    if (global_vec_offset != static_cast<uint64_t>(n))
    {
      throw std::runtime_error("PC2 cell vec offsets do not cover all vectors");
    }
    out.flush();
    if (!out)
    {
      throw std::runtime_error("failed to write reordered cluster file: " +
                               temp.string());
    }
    const size_t payload_bytes = n * vector_bytes;
    const size_t padded_file_size =
        div_round_up(payload_bytes, PAGE_SIZE) * PAGE_SIZE;
    if (padded_file_size > payload_bytes)
    {
      out.seekp(static_cast<std::streamoff>(padded_file_size - 1),
                std::ios::beg);
      out.put('\0');
      out.flush();
      if (!out)
      {
        throw std::runtime_error("failed to pad reordered cluster file: " +
                                 temp.string());
      }
    }
    in.close();
    out.close();
    std::error_code ec;
    std::filesystem::rename(temp, original, ec);
    if (ec)
    {
      std::filesystem::remove(original, ec);
      ec.clear();
      std::filesystem::rename(temp, original, ec);
    }
    if (ec)
    {
      throw std::runtime_error("failed to replace cluster file after PC2 reorder: " +
                               ec.message());
    }
    if (range_search_config::is_byte_dtype(vec_dtype) &&
        (sum_for_offset.size() != n || norm_sq_for_l2.size() != n))
    {
      throw std::runtime_error(
          "byte-dtype sum/norm metadata count does not match base vector count");
    }
  }

  void fitClusterSlabsFromClusterFile()
  {
    OpenBlasThreadGuard fit_blas_guard(1);
    slab_neighbour_indices.clear();
    slab_norms.clear();
    slab_l.clear();
    slab_h.clear();
    if (slab_count == 0)
    {
      return;
    }
    if (slab_count >= cluster_num)
    {
      throw std::runtime_error("slab_count " + std::to_string(slab_count) +
                               " must be < cluster_num " +
                               std::to_string(cluster_num));
    }

    slab_neighbour_indices.resize(cluster_num * slab_count);
    slab_norms.resize(cluster_num * slab_count);
    slab_l.resize(cluster_num * slab_count);
    slab_h.resize(cluster_num * slab_count);

    std::vector<size_t> cluster_file_offsets(cluster_num, 0);
    size_t file_offset = 0;
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      cluster_file_offsets[cluster_id] = file_offset;
      file_offset += bucket_sizes[cluster_id] * d * bytes_per_element;
    }
    const float slab_pad_coeff =
        static_cast<float>(d) * kSlabFitFloat32UnitRoundoff *
        kSlabFitDotProductSafetyFactor;

    std::atomic<bool> abort_requested(false);
    std::string first_error_message;
    auto record_error = [&](const std::string& message) {
      abort_requested.store(true, std::memory_order_relaxed);
#pragma omp critical
      {
        if (first_error_message.empty())
        {
          first_error_message = message;
        }
      }
    };

    std::vector<float> centroid_norm_sq(cluster_num, 0.0f);
#pragma omp parallel for schedule(static)
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      const float* centroid = centroids_ + cluster_id * d;
      float norm_sq = 0.0f;
      for (size_t col = 0; col < d; ++col)
      {
        norm_sq += centroid[col] * centroid[col];
      }
      centroid_norm_sq[cluster_id] = norm_sq;
    }

    const size_t neighbour_candidate_count = slab_count;
    const float centroid_candidate_error_coeff =
        static_cast<float>(d) * kSlabFitFloat32UnitRoundoff *
        kSlabFitCentroidCandidateSafetyFactor;
    const size_t centroid_block_rows = 32;
#pragma omp parallel
    {
      try
      {
        std::vector<float> centroid_dist_block;
        centroid_dist_block.reserve(centroid_block_rows * cluster_num);
        std::vector<size_t> order(cluster_num);
        std::vector<std::pair<float, size_t>> exact_candidates;
        exact_candidates.reserve(neighbour_candidate_count);

#pragma omp for schedule(dynamic)
        for (size_t block_start = 0; block_start < cluster_num;
             block_start += centroid_block_rows)
        {
          if (abort_requested.load(std::memory_order_relaxed))
          {
            continue;
          }
          const size_t block_rows =
              std::min(centroid_block_rows, cluster_num - block_start);
          centroid_dist_block.resize(block_rows * cluster_num);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                      static_cast<int>(block_rows),
                      static_cast<int>(cluster_num), static_cast<int>(d),
                      -2.0f, centroids_ + block_start * d,
                      static_cast<int>(d), centroids_, static_cast<int>(d),
                      0.0f, centroid_dist_block.data(),
                      static_cast<int>(cluster_num));

          for (size_t local_row = 0; local_row < block_rows; ++local_row)
          {
            const size_t cluster_id = block_start + local_row;
            float* approx_dist =
                centroid_dist_block.data() + local_row * cluster_num;
            const float self_norm_sq = centroid_norm_sq[cluster_id];
            for (size_t other = 0; other < cluster_num; ++other)
            {
              approx_dist[other] += self_norm_sq + centroid_norm_sq[other];
            }
            approx_dist[cluster_id] = std::numeric_limits<float>::infinity();

            std::iota(order.begin(), order.end(), size_t{0});
            const auto approx_cmp = [&](size_t left, size_t right) {
              if (approx_dist[left] != approx_dist[right])
              {
                return approx_dist[left] < approx_dist[right];
              }
              return left < right;
            };
            std::nth_element(order.begin(),
                             order.begin() + neighbour_candidate_count - 1,
                             order.end(), approx_cmp);
            const float approx_cutoff =
                approx_dist[order[neighbour_candidate_count - 1]];
            const auto approx_error_bound = [&](size_t other) {
              const float dot_est =
                  0.5f * (self_norm_sq + centroid_norm_sq[other] -
                          approx_dist[other]);
              const float magnitude =
                  std::max(self_norm_sq, 0.0f) +
                  std::max(centroid_norm_sq[other], 0.0f) +
                  std::abs(2.0f * dot_est) + 1.0f;
              return centroid_candidate_error_coeff * magnitude;
            };
            float cutoff_error = 0.0f;
            for (size_t candidate_id = 0;
                 candidate_id < neighbour_candidate_count; ++candidate_id)
            {
              cutoff_error =
                  std::max(cutoff_error,
                           approx_error_bound(order[candidate_id]));
            }

            exact_candidates.clear();
            const float* self_centroid = centroids_ + cluster_id * d;
            for (size_t other = 0; other < cluster_num; ++other)
            {
              if (other == cluster_id)
              {
                continue;
              }
              const float candidate_threshold =
                  approx_cutoff + cutoff_error + approx_error_bound(other);
              if (approx_dist[other] > candidate_threshold)
              {
                continue;
              }
              const float* other_centroid = centroids_ + other * d;
              float dist_sq = 0.0f;
              for (size_t col = 0; col < d; ++col)
              {
                const float diff = other_centroid[col] - self_centroid[col];
                dist_sq += diff * diff;
              }
              exact_candidates.emplace_back(dist_sq, other);
            }
            if (exact_candidates.size() < slab_count)
            {
              record_error("slab build: centroid candidate refinement kept " +
                           std::to_string(exact_candidates.size()) +
                           " candidates for cluster " +
                           std::to_string(cluster_id) + ", below slab_count " +
                           std::to_string(slab_count));
              continue;
            }

            const auto exact_cmp =
                [](const std::pair<float, size_t>& left,
                   const std::pair<float, size_t>& right) {
                  if (left.first != right.first)
                  {
                    return left.first < right.first;
                  }
                  return left.second < right.second;
                };
            std::sort(exact_candidates.begin(), exact_candidates.end(),
                      exact_cmp);

            const size_t slab_offset = cluster_id * slab_count;
            for (size_t j = 0; j < slab_count; ++j)
            {
              const size_t neighbour = exact_candidates[j].second;
              slab_neighbour_indices[slab_offset + j] =
                  static_cast<uint32_t>(neighbour);
              const float norm = std::sqrt(exact_candidates[j].first);
              if (norm < 1e-9f)
              {
                record_error("slab build: clusters " +
                             std::to_string(cluster_id) + " and " +
                             std::to_string(neighbour) +
                             " have coincident centroids (||c_a - c_b|| = " +
                             std::to_string(norm) + " < 1e-9)");
                break;
              }
              slab_norms[slab_offset + j] = norm;
            }
          }
        }
      }
      catch (...)
      {
        record_error("unexpected exception during slab neighbour pass");
      }
    }
    if (!first_error_message.empty())
    {
      throw std::runtime_error(first_error_message);
    }

#pragma omp parallel
    {
      try
      {
        std::ifstream in(clusterfile, std::ios::binary | std::ios::in);
        if (!in)
        {
          record_error("failed to open cluster file for slab pass: " +
                       clusterfile);
        }
        std::vector<float> points;
        std::vector<uint8_t> raw_points;
        std::vector<float> normalized_directions(slab_count * d, 0.0f);
        std::vector<float> mins(slab_count, 0.0f);
        std::vector<float> maxs(slab_count, 0.0f);
        std::vector<float> proj_matrix;
        proj_matrix.reserve(4096 * slab_count);

#pragma omp for schedule(dynamic)
        for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
        {
          if (abort_requested.load(std::memory_order_relaxed))
          {
            continue;
          }
          const size_t slab_offset = cluster_id * slab_count;
          for (size_t j = 0; j < slab_count; ++j)
          {
            const size_t neighbour = slab_neighbour_indices[slab_offset + j];
            float* direction = normalized_directions.data() + j * d;
            const float* neighbour_centroid = centroids_ + neighbour * d;
            const float* self_centroid = centroids_ + cluster_id * d;
            for (size_t col = 0; col < d; ++col)
            {
              const float delta = neighbour_centroid[col] - self_centroid[col];
              direction[col] = delta;
            }
            const float inv_norm = 1.0f / slab_norms[slab_offset + j];
            for (size_t col = 0; col < d; ++col)
            {
              direction[col] *= inv_norm;
            }
          }
          if (abort_requested.load(std::memory_order_relaxed))
          {
            continue;
          }

          if (bucket_sizes[cluster_id] == 0)
          {
            for (size_t j = 0; j < slab_count; ++j)
            {
              slab_l[slab_offset + j] = std::numeric_limits<float>::infinity();
              slab_h[slab_offset + j] =
                  -std::numeric_limits<float>::infinity();
            }
            continue;
          }

          const size_t value_count = bucket_sizes[cluster_id] * d;
          points.resize(value_count);
          in.clear();
          in.seekg(static_cast<std::streamoff>(cluster_file_offsets[cluster_id]),
                   std::ios::beg);
          if (range_search_config::is_byte_dtype(vec_dtype))
          {
            raw_points.resize(value_count);
            in.read(reinterpret_cast<char*>(raw_points.data()),
                    static_cast<std::streamsize>(value_count));
            if (in.gcount() !=
                static_cast<std::streamsize>(value_count))
            {
              record_error("short cluster payload during slab pass at cluster " +
                           std::to_string(cluster_id));
              continue;
            }
            utils::cast_byte_payload_to_float(vec_dtype, raw_points.data(),
                                              value_count, points.data());
          }
          else
          {
            in.read(reinterpret_cast<char*>(points.data()),
                    static_cast<std::streamsize>(value_count * sizeof(float)));
            if (in.gcount() !=
                static_cast<std::streamsize>(value_count * sizeof(float)))
            {
              record_error("short cluster payload during slab pass at cluster " +
                           std::to_string(cluster_id));
              continue;
            }
          }

          std::fill(mins.begin(), mins.end(),
                    std::numeric_limits<float>::infinity());
          std::fill(maxs.begin(), maxs.end(),
                    -std::numeric_limits<float>::infinity());
          proj_matrix.resize(bucket_sizes[cluster_id] * slab_count);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                      static_cast<int>(bucket_sizes[cluster_id]),
                      static_cast<int>(slab_count), static_cast<int>(d), 1.0f,
                      points.data(), static_cast<int>(d),
                      normalized_directions.data(), static_cast<int>(d), 0.0f,
                      proj_matrix.data(), static_cast<int>(slab_count));
          for (size_t point_id = 0; point_id < bucket_sizes[cluster_id];
               ++point_id)
          {
            const float* row = proj_matrix.data() + point_id * slab_count;
            for (size_t j = 0; j < slab_count; ++j)
            {
              const float projection = row[j];
              mins[j] = std::min(mins[j], projection);
              maxs[j] = std::max(maxs[j], projection);
            }
          }

          float max_l2_sq = 0.0f;
          for (size_t point_id = 0; point_id < bucket_sizes[cluster_id];
               ++point_id)
          {
            const float* point = points.data() + point_id * d;
            float point_l2_sq = 0.0f;
            for (size_t col = 0; col < d; ++col)
            {
              point_l2_sq += point[col] * point[col];
            }
            max_l2_sq = std::max(max_l2_sq, point_l2_sq);
          }
          const float max_l2 = std::sqrt(max_l2_sq);
          const float pad_abs = slab_pad_coeff * std::max(max_l2, 1.0f);
          for (size_t j = 0; j < slab_count; ++j)
          {
            slab_l[slab_offset + j] = mins[j] - pad_abs;
            slab_h[slab_offset + j] = maxs[j] + pad_abs;
          }
        }
      }
      catch (...)
      {
        record_error("unexpected exception during slab pass");
      }
    }

    if (!first_error_message.empty())
    {
      throw std::runtime_error(first_error_message);
    }
  }
};

struct ClusterReader
{
  struct MemBudgetInputs
  {
    size_t mem_budget_bytes = SIZE_MAX;
    size_t rbq_1bit_resident_bytes = 0;
    size_t data_hnsw_resident_bytes = 0;
    size_t query_resident_bytes = 0;
    size_t per_thread_resident_bytes = 0;
    size_t sqg_resident_bytes = 0;
    size_t rotated_centroids_resident_bytes = 0;
    size_t trace_resident_bytes = 0;
    size_t hit_bookkeeping_resident_bytes = 0;
    size_t cluster_cache_requested_bytes = 0;
    size_t cluster_cache_resident_bytes = 0;
    size_t cell_cache_requested_bytes = 0;
    size_t cell_cache_resident_bytes = 0;
  };

  class ClusterPayloadCache
  {
   private:
    struct CacheEntry
    {
      uint32_t cluster_id = 0;
      void* buffer = nullptr;
      size_t payload_offset = 0;
      size_t valid_bytes = 0;
      std::atomic<int> ref_count{0};
      typename std::list<CacheEntry>::iterator self_it;

      CacheEntry(uint32_t id, void* data, size_t offset, size_t bytes)
          : cluster_id(id),
            buffer(data),
            payload_offset(offset),
            valid_bytes(bytes)
      {
      }

      CacheEntry(const CacheEntry&) = delete;
      CacheEntry& operator=(const CacheEntry&) = delete;

      ~CacheEntry()
      {
        if (ref_count.load(std::memory_order_acquire) != 0)
        {
          std::cerr << "[cluster-cache] ref_count non-zero during destruction "
                    << "for cluster_id=" << cluster_id << '\n';
          std::abort();
        }
        free(buffer);
      }
    };

   public:
    struct Stats
    {
      bool enabled = false;
      bool env_disabled = false;
      size_t capacity_bytes = 0;
      size_t entries_max = 0;
      size_t per_shard_capacity = 0;
      size_t hits = 0;
      size_t misses = 0;
      double hit_rate = 0.0;
      size_t evictions = 0;
      size_t eviction_failed = 0;
      size_t duplicate_inserts = 0;
      size_t peak_resident = 0;
      size_t shards_used = 0;
      size_t shard_max_resident = 0;
    };

    class CacheRef
    {
     public:
      CacheRef() = default;
      CacheRef(const CacheRef&) = delete;
      CacheRef& operator=(const CacheRef&) = delete;

      CacheRef(CacheRef&& other) noexcept : entry_(other.entry_)
      {
        other.entry_ = nullptr;
      }

      CacheRef& operator=(CacheRef&& other) noexcept
      {
        if (this != &other)
        {
          release();
          entry_ = other.entry_;
          other.entry_ = nullptr;
        }
        return *this;
      }

      ~CacheRef()
      {
        release();
      }

      explicit operator bool() const noexcept
      {
        return entry_ != nullptr;
      }

      const char* buffer() const noexcept
      {
        return entry_ == nullptr ? nullptr : static_cast<const char*>(entry_->buffer);
      }

      size_t payload_offset() const noexcept
      {
        return entry_ == nullptr ? 0 : entry_->payload_offset;
      }

      size_t valid_bytes() const noexcept
      {
        return entry_ == nullptr ? 0 : entry_->valid_bytes;
      }

     private:
      explicit CacheRef(CacheEntry* entry) : entry_(entry) {}

      void release() noexcept
      {
        if (entry_ != nullptr)
        {
          entry_->ref_count.fetch_sub(1, std::memory_order_acq_rel);
          entry_ = nullptr;
        }
      }

      CacheEntry* entry_ = nullptr;
      friend class ClusterPayloadCache;
    };

    ClusterPayloadCache(size_t global_capacity_entries, size_t buf_size,
                        bool env_disabled)
        : buf_size_(buf_size),
          env_disabled_(env_disabled),
          per_shard_capacity_(env_disabled
                                  ? 0
                                  : global_capacity_entries /
                                        kClusterCacheShardCount),
          effective_capacity_entries_(per_shard_capacity_ *
                                      kClusterCacheShardCount),
          cache_active_(effective_capacity_entries_ > 0 && !env_disabled)
    {
      if (!env_disabled_ && global_capacity_entries > 0 &&
          effective_capacity_entries_ == 0)
      {
        std::cerr << "[cluster-cache] WARNING: budget too small for sharded cache "
                  << "(need >= " << kClusterCacheShardCount
                  << " entries, got " << global_capacity_entries
                  << "); cache disabled\n";
      }
    }

    static size_t shard_index_for(uint32_t cluster_id) noexcept
    {
      uint64_t x = static_cast<uint64_t>(cluster_id);
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      x = x ^ (x >> 31);
      return static_cast<size_t>(x) & (kClusterCacheShardCount - 1);
    }

    bool active() const noexcept
    {
      return cache_active_;
    }

    CacheRef lookup(uint32_t cluster_id)
    {
      if (env_disabled_)
      {
        return CacheRef{};
      }
      if (!cache_active_)
      {
        inactive_misses_.fetch_add(1, std::memory_order_relaxed);
        return CacheRef{};
      }

      Shard& shard = shards_[shard_index_for(cluster_id)];
      std::lock_guard<std::mutex> guard(shard.mu);
      auto it = shard.idx.find(cluster_id);
      if (it == shard.idx.end())
      {
        ++shard.misses;
        return CacheRef{};
      }
      ++shard.hits;
      shard.lru.splice(shard.lru.begin(), shard.lru, it->second);
      it->second = shard.lru.begin();
      it->second->self_it = it->second;
      it->second->ref_count.fetch_add(1, std::memory_order_acq_rel);
      return CacheRef(&*it->second);
    }

    bool try_insert(uint32_t cluster_id, const void* source,
                    size_t payload_offset, size_t valid_bytes)
    {
      if (!cache_active_ || source == nullptr)
      {
        return false;
      }
      if (payload_offset > buf_size_)
      {
        return false;
      }
      if (valid_bytes > buf_size_)
      {
        return false;
      }

      Shard& shard = shards_[shard_index_for(cluster_id)];
      std::lock_guard<std::mutex> guard(shard.mu);
      auto existing = shard.idx.find(cluster_id);
      if (existing != shard.idx.end())
      {
        ++shard.duplicate_inserts;
        shard.lru.splice(shard.lru.begin(), shard.lru, existing->second);
        existing->second = shard.lru.begin();
        existing->second->self_it = existing->second;
        return true;
      }

      if (shard.lru.size() >= per_shard_capacity_ && !evict_one_locked(shard))
      {
        ++shard.eviction_failed;
        return false;
      }

      void* buffer = std::aligned_alloc(PAGE_SIZE, buf_size_);
      if (buffer == nullptr)
      {
        ++shard.eviction_failed;
        return false;
      }
      std::memcpy(buffer, source, valid_bytes);
      try
      {
        shard.lru.emplace_front(cluster_id, buffer, payload_offset, valid_bytes);
      }
      catch (...)
      {
        free(buffer);
        throw;
      }
      shard.idx[cluster_id] = shard.lru.begin();
      shard.lru.begin()->self_it = shard.lru.begin();
      shard.peak_resident =
          std::max(shard.peak_resident, shard.lru.size());
      return true;
    }

    Stats stats() const
    {
      Stats stats;
      stats.enabled = cache_active_;
      stats.env_disabled = env_disabled_;
      stats.entries_max = effective_capacity_entries_;
      stats.per_shard_capacity = per_shard_capacity_;
      stats.capacity_bytes = effective_capacity_entries_ * buf_size_;

      if (env_disabled_)
      {
        return stats;
      }
      if (!cache_active_)
      {
        stats.misses = inactive_misses_.load(std::memory_order_relaxed);
        return stats;
      }

      for (const Shard& shard : shards_)
      {
        std::lock_guard<std::mutex> guard(shard.mu);
        stats.hits += shard.hits;
        stats.misses += shard.misses;
        stats.evictions += shard.evictions;
        stats.eviction_failed += shard.eviction_failed;
        stats.duplicate_inserts += shard.duplicate_inserts;
        stats.peak_resident += shard.peak_resident * buf_size_;
        if (shard.peak_resident > 0)
        {
          ++stats.shards_used;
        }
        stats.shard_max_resident =
            std::max(stats.shard_max_resident, shard.peak_resident);
      }

      const size_t lookups = stats.hits + stats.misses;
      if (lookups > 0)
      {
        stats.hit_rate = static_cast<double>(stats.hits) /
                         static_cast<double>(lookups);
      }
      return stats;
    }

    size_t resident_entries_for_test() const
    {
      size_t total = 0;
      if (!cache_active_)
      {
        return total;
      }
      for (const Shard& shard : shards_)
      {
        std::lock_guard<std::mutex> guard(shard.mu);
        total += shard.lru.size();
      }
      return total;
    }

    std::string stats_line() const
    {
      const Stats s = stats();
      std::ostringstream out;
      out << "[cluster-cache] enabled=" << (s.enabled ? "Y" : "N")
          << " env_disabled=" << (s.env_disabled ? "Y" : "N")
          << " capacity=" << format_mib_local(s.capacity_bytes)
          << " entries_max=" << s.entries_max
          << " per_shard_capacity=" << s.per_shard_capacity
          << " hits=" << s.hits
          << " misses=" << s.misses
          << " hit_rate=" << std::fixed << std::setprecision(3)
          << s.hit_rate
          << " evictions=" << s.evictions
          << " eviction_failed=" << s.eviction_failed
          << " duplicate_inserts=" << s.duplicate_inserts
          << " peak_resident=" << format_mib_local(s.peak_resident)
          << " shards_used=" << s.shards_used
          << " shard_max_resident=" << s.shard_max_resident;
      return out.str();
    }

    void report_stats()
    {
      bool expected = false;
      if (reported_.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel))
      {
        std::cerr << stats_line() << '\n';
      }
    }

   private:
    struct Shard
    {
      mutable std::mutex mu;
      std::list<CacheEntry> lru;
      std::unordered_map<uint32_t,
                         typename std::list<CacheEntry>::iterator>
          idx;
      size_t hits = 0;
      size_t misses = 0;
      size_t evictions = 0;
      size_t eviction_failed = 0;
      size_t duplicate_inserts = 0;
      size_t peak_resident = 0;
    };

    static std::string format_mib_local(size_t bytes)
    {
      std::ostringstream out;
      out << std::fixed << std::setprecision(3)
          << static_cast<double>(bytes) / static_cast<double>(size_t{1} << 20);
      return out.str();
    }

    static bool evict_one_locked(Shard& shard)
    {
      for (auto rit = shard.lru.rbegin(); rit != shard.lru.rend(); ++rit)
      {
        if (rit->ref_count.load(std::memory_order_acquire) != 0)
        {
          continue;
        }
        auto it = rit.base();
        --it;
        shard.idx.erase(it->cluster_id);
        shard.lru.erase(it);
        ++shard.evictions;
        return true;
      }
      return false;
    }

    size_t buf_size_ = 0;
    bool env_disabled_ = false;
    size_t per_shard_capacity_ = 0;
    size_t effective_capacity_entries_ = 0;
    bool cache_active_ = false;
    std::array<Shard, kClusterCacheShardCount> shards_;
    std::atomic<size_t> inactive_misses_{0};
    std::atomic<bool> reported_{false};
  };

  class CellPayloadCache
  {
   private:
    struct CacheEntry
    {
      uint64_t cell_id = 0;
      void* buffer = nullptr;
      size_t alloc_bytes = 0;
      size_t payload_offset = 0;
      size_t valid_bytes = 0;
      size_t vec_count = 0;
      std::atomic<int> ref_count{0};
      typename std::list<CacheEntry>::iterator self_it;

      CacheEntry(uint64_t id, void* data, size_t allocated,
                 size_t offset, size_t bytes, size_t vectors)
          : cell_id(id),
            buffer(data),
            alloc_bytes(allocated),
            payload_offset(offset),
            valid_bytes(bytes),
            vec_count(vectors)
      {
      }

      CacheEntry(const CacheEntry&) = delete;
      CacheEntry& operator=(const CacheEntry&) = delete;

      ~CacheEntry()
      {
        if (ref_count.load(std::memory_order_acquire) != 0)
        {
          std::cerr << "[cell-cache] ref_count non-zero during destruction "
                    << "for cell_id=" << cell_id << '\n';
          std::abort();
        }
        free(buffer);
      }
    };

   public:
    struct Stats
    {
      bool enabled = false;
      bool env_disabled = false;
      size_t budget_bytes = 0;
      size_t per_shard_budget_bytes = 0;
      size_t current_resident_sum = 0;
      size_t shard_peak_sum = 0;
      size_t entries = 0;
      size_t hits = 0;
      size_t misses = 0;
      double hit_rate = 0.0;
      size_t evictions = 0;
      size_t eviction_failed = 0;
      size_t alloc_failures = 0;
      size_t oversize_rejects = 0;
      size_t duplicate_inserts = 0;
      size_t shards_used = 0;
      size_t shard_max_resident_bytes = 0;
    };

    class CacheRef
    {
     public:
      CacheRef() = default;
      CacheRef(const CacheRef&) = delete;
      CacheRef& operator=(const CacheRef&) = delete;

      CacheRef(CacheRef&& other) noexcept : entry_(other.entry_)
      {
        other.entry_ = nullptr;
      }

      CacheRef& operator=(CacheRef&& other) noexcept
      {
        if (this != &other)
        {
          release();
          entry_ = other.entry_;
          other.entry_ = nullptr;
        }
        return *this;
      }

      ~CacheRef()
      {
        release();
      }

      explicit operator bool() const noexcept
      {
        return entry_ != nullptr;
      }

      const char* buffer() const noexcept
      {
        return entry_ == nullptr ? nullptr : static_cast<const char*>(entry_->buffer);
      }

      size_t payload_offset() const noexcept
      {
        return entry_ == nullptr ? 0 : entry_->payload_offset;
      }

      size_t valid_bytes() const noexcept
      {
        return entry_ == nullptr ? 0 : entry_->valid_bytes;
      }

      size_t vec_count() const noexcept
      {
        return entry_ == nullptr ? 0 : entry_->vec_count;
      }

     private:
      explicit CacheRef(CacheEntry* entry) : entry_(entry) {}

      void release() noexcept
      {
        if (entry_ != nullptr)
        {
          entry_->ref_count.fetch_sub(1, std::memory_order_acq_rel);
          entry_ = nullptr;
        }
      }

      CacheEntry* entry_ = nullptr;
      friend class CellPayloadCache;
    };

    CellPayloadCache(size_t budget_bytes, bool env_disabled)
        : budget_bytes_(env_disabled ? 0 : budget_bytes),
          env_disabled_(env_disabled)
    {
      effective_payload_budget_bytes_ =
          static_cast<size_t>(static_cast<double>(budget_bytes_) *
                              (1.0 - kCellCacheIndexOverheadRatio));
      per_shard_budget_bytes_ =
          effective_payload_budget_bytes_ / kCellCacheShardCount;
      cache_active_ =
          effective_payload_budget_bytes_ >= kCellCacheShardCount * PAGE_SIZE &&
          !env_disabled_;
    }

    static size_t shard_index_for(uint64_t cell_id) noexcept
    {
      uint64_t x = cell_id;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      x = x ^ (x >> 31);
      return static_cast<size_t>(x) & (kCellCacheShardCount - 1);
    }

    bool active() const noexcept
    {
      return cache_active_;
    }

    CacheRef lookup(uint64_t cell_id)
    {
      if (!cache_active_)
      {
        return CacheRef{};
      }

      Shard& shard = shards_[shard_index_for(cell_id)];
      std::lock_guard<std::mutex> guard(shard.mu);
      auto it = shard.idx.find(cell_id);
      if (it == shard.idx.end())
      {
        ++shard.misses;
        return CacheRef{};
      }
      ++shard.hits;
      shard.lru.splice(shard.lru.begin(), shard.lru, it->second);
      it->second = shard.lru.begin();
      it->second->self_it = it->second;
      it->second->ref_count.fetch_add(1, std::memory_order_acq_rel);
      return CacheRef(&*it->second);
    }

    bool try_insert(uint64_t cell_id, const void* source,
                    size_t payload_offset, size_t valid_bytes,
                    size_t vec_count, size_t alloc_bytes)
    {
      if (!cache_active_ || source == nullptr)
      {
        return false;
      }
      if (valid_bytes > alloc_bytes || payload_offset > valid_bytes)
      {
        return false;
      }

      Shard& shard = shards_[shard_index_for(cell_id)];
      std::lock_guard<std::mutex> guard(shard.mu);
      if (alloc_bytes > per_shard_budget_bytes_)
      {
        ++shard.oversize_rejects;
        return false;
      }

      auto existing = shard.idx.find(cell_id);
      if (existing != shard.idx.end())
      {
        ++shard.duplicate_inserts;
        shard.lru.splice(shard.lru.begin(), shard.lru, existing->second);
        existing->second = shard.lru.begin();
        existing->second->self_it = existing->second;
        return true;
      }

      while (shard.current_resident_bytes + alloc_bytes >
             per_shard_budget_bytes_)
      {
        size_t freed = 0;
        if (!evict_one_locked(shard, &freed))
        {
          ++shard.eviction_failed;
          return false;
        }
      }

      void* buffer = std::aligned_alloc(PAGE_SIZE, alloc_bytes);
      if (buffer == nullptr)
      {
        ++shard.alloc_failures;
        return false;
      }
      std::memcpy(buffer, source, valid_bytes);
      try
      {
        shard.lru.emplace_front(cell_id, buffer, alloc_bytes, payload_offset,
                                valid_bytes, vec_count);
      }
      catch (...)
      {
        free(buffer);
        ++shard.alloc_failures;
        return false;
      }
      auto inserted = shard.lru.begin();
      try
      {
        shard.idx.emplace(cell_id, inserted);
      }
      catch (...)
      {
        shard.lru.erase(inserted);
        ++shard.alloc_failures;
        return false;
      }
      inserted->self_it = inserted;
      shard.current_resident_bytes += alloc_bytes;
      shard.inserted_once = true;
      shard.peak_resident_bytes =
          std::max(shard.peak_resident_bytes, shard.current_resident_bytes);
      return true;
    }

    Stats stats() const
    {
      Stats stats;
      stats.enabled = cache_active_;
      stats.env_disabled = env_disabled_;
      stats.budget_bytes = budget_bytes_;
      stats.per_shard_budget_bytes = per_shard_budget_bytes_;
      if (!cache_active_)
      {
        return stats;
      }

      for (const Shard& shard : shards_)
      {
        std::lock_guard<std::mutex> guard(shard.mu);
        stats.current_resident_sum += shard.current_resident_bytes;
        stats.shard_peak_sum += shard.peak_resident_bytes;
        stats.entries += shard.lru.size();
        stats.hits += shard.hits;
        stats.misses += shard.misses;
        stats.evictions += shard.evictions;
        stats.eviction_failed += shard.eviction_failed;
        stats.alloc_failures += shard.alloc_failures;
        stats.oversize_rejects += shard.oversize_rejects;
        stats.duplicate_inserts += shard.duplicate_inserts;
        if (shard.inserted_once)
        {
          ++stats.shards_used;
        }
        stats.shard_max_resident_bytes =
            std::max(stats.shard_max_resident_bytes,
                     shard.peak_resident_bytes);
      }
      const size_t lookups = stats.hits + stats.misses;
      if (lookups > 0)
      {
        stats.hit_rate = static_cast<double>(stats.hits) /
                         static_cast<double>(lookups);
      }
      return stats;
    }

    std::string stats_line() const
    {
      const Stats s = stats();
      std::ostringstream out;
      out << "[cell-cache] enabled=" << (s.enabled ? "Y" : "N")
          << " env_disabled=" << (s.env_disabled ? "Y" : "N")
          << " budget=" << format_mib_local(s.budget_bytes)
          << " per_shard_budget=" << format_mib_local(s.per_shard_budget_bytes)
          << " current_resident_sum="
          << format_mib_local(s.current_resident_sum)
          << " shard_peak_sum=" << format_mib_local(s.shard_peak_sum)
          << " entries=" << s.entries
          << " hits=" << s.hits
          << " misses=" << s.misses
          << " hit_rate=" << std::fixed << std::setprecision(3)
          << s.hit_rate
          << " evictions=" << s.evictions
          << " eviction_failed=" << s.eviction_failed
          << " alloc_failures=" << s.alloc_failures
          << " oversize_rejects=" << s.oversize_rejects
          << " duplicate_inserts=" << s.duplicate_inserts
          << " shards_used=" << s.shards_used
          << " shard_max_resident="
          << format_mib_local(s.shard_max_resident_bytes);
      return out.str();
    }

    void report_stats()
    {
      bool expected = false;
      if (reported_.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel))
      {
        std::cerr << stats_line() << '\n';
      }
    }

    size_t resident_entries_for_test() const
    {
      size_t total = 0;
      if (!cache_active_)
      {
        return total;
      }
      for (const Shard& shard : shards_)
      {
        std::lock_guard<std::mutex> guard(shard.mu);
        total += shard.lru.size();
      }
      return total;
    }

    size_t resident_bytes_for_test() const
    {
      return stats().current_resident_sum;
    }

   private:
    struct Shard
    {
      mutable std::mutex mu;
      std::list<CacheEntry> lru;
      std::unordered_map<uint64_t,
                         typename std::list<CacheEntry>::iterator>
          idx;
      size_t current_resident_bytes = 0;
      size_t peak_resident_bytes = 0;
      size_t hits = 0;
      size_t misses = 0;
      size_t evictions = 0;
      size_t eviction_failed = 0;
      size_t duplicate_inserts = 0;
      size_t alloc_failures = 0;
      size_t oversize_rejects = 0;
      bool inserted_once = false;
    };

    static std::string format_mib_local(size_t bytes)
    {
      std::ostringstream out;
      out << std::fixed << std::setprecision(3)
          << static_cast<double>(bytes) / static_cast<double>(size_t{1} << 20);
      return out.str();
    }

    static bool evict_one_locked(Shard& shard, size_t* freed_bytes_out)
    {
      for (auto rit = shard.lru.rbegin(); rit != shard.lru.rend(); ++rit)
      {
        if (rit->ref_count.load(std::memory_order_acquire) != 0)
        {
          continue;
        }
        auto it = rit.base();
        --it;
        const size_t freed = it->alloc_bytes;
        shard.idx.erase(it->cell_id);
        shard.current_resident_bytes -= freed;
        shard.lru.erase(it);
        ++shard.evictions;
        if (freed_bytes_out != nullptr)
        {
          *freed_bytes_out = freed;
        }
        return true;
      }
      return false;
    }

    size_t budget_bytes_ = 0;
    size_t effective_payload_budget_bytes_ = 0;
    size_t per_shard_budget_bytes_ = 0;
    bool env_disabled_ = false;
    bool cache_active_ = false;
    std::array<Shard, kCellCacheShardCount> shards_;
    std::atomic<bool> reported_{false};
  };

  std::ifstream fmeta;

  size_t n;
  size_t d;
  uint32_t vec_dtype = range_search_config::kVecDTypeFloat32;
  size_t bytes_per_element = sizeof(float);
  size_t cluster_num;
  size_t max_points;
  size_t pca_rank;
  size_t target_cell_vecs = 24;
  std::vector<size_t> bucket_sizes;
  std::vector<float> centroids;
  std::vector<float> radii;
  std::vector<float> pca_basis;
  std::vector<float> pca_min;
  std::vector<float> pca_max;
  std::vector<float> pca_residual_radius;
  std::vector<uint32_t> pca_effective_rank;
  size_t slab_count = 0;
  std::vector<uint32_t> slab_neighbour_indices;
  std::vector<float> slab_norms;
  std::vector<float> slab_l;
  std::vector<float> slab_h;
  std::vector<uint64_t> cluster_cell_offsets;
  std::vector<uint64_t> cell_vec_offsets;
  std::vector<float> cell_bounds;
  std::vector<int32_t> sum_for_offset;
  std::vector<int32_t> norm_sq_for_l2;
  std::vector<size_t> cluster_bound_offsets;
  std::vector<size_t> file_pos;
  std::vector<std::vector<size_t>> assignment;
  mutable std::atomic<size_t> slab_pruned_clusters{0};
  mutable std::atomic<size_t> box_residual_pruned_clusters{0};

  ClusterReader(std::string clusterfile, std::string metafile,
                size_t requested_io_queue_depth,
                const MemBudgetInputs& budget,
                size_t requested_n_io_slots = omp_get_max_threads(),
                float cell_cache_share_pct = 0.0f)
      : fmeta(metafile, std::ios::binary | std::ios::in),
        metadatafile_(std::move(metafile)),
        clusterfile_(std::move(clusterfile)),
        direct_fd_(-1),
        budget_inputs_(budget),
        requested_io_queue_depth_(requested_io_queue_depth),
        requested_n_io_slots_(requested_n_io_slots == 0 ? 1
                                                        : requested_n_io_slots),
        cell_cache_share_pct_(cell_cache_share_pct),
        n_io_slots_(1),
        io_queue_depth_(1),
        buf_size_(0),
        cluster_cache_env_disabled_(
            std::getenv("DJ_CLUSTER_CACHE_DISABLE") != nullptr &&
            std::getenv("DJ_CLUSTER_CACHE_DISABLE")[0] != '\0'),
        cell_cache_env_disabled_(
            std::getenv("DJ_CELL_CACHE_DISABLE") != nullptr &&
            std::getenv("DJ_CELL_CACHE_DISABLE")[0] != '\0')
  {
    validate_io_queue_depth(requested_io_queue_depth_);
    direct_fd_ = open(clusterfile_.c_str(), O_RDONLY | O_DIRECT);
    if (direct_fd_ < 0)
    {
      throw std::runtime_error("failed to open cluster file for direct read: " +
                               clusterfile_ + ": " + strerror(errno));
    }
  }

  ClusterReader(std::string clusterfile, std::string metafile,
                size_t requested_io_queue_depth)
      : ClusterReader(std::move(clusterfile), std::move(metafile),
                      requested_io_queue_depth, MemBudgetInputs{},
                      omp_get_max_threads())
  {
  }

  ClusterReader(std::string clusterfile, std::string metafile,
                size_t requested_io_queue_depth, size_t requested_n_io_slots)
      : ClusterReader(std::move(clusterfile), std::move(metafile),
                      requested_io_queue_depth, MemBudgetInputs{},
                      requested_n_io_slots)
  {
  }

  static float slab_width_key(float lo, float hi)
  {
    if (!std::isfinite(lo) || !std::isfinite(hi))
    {
      return std::numeric_limits<float>::infinity();
    }
    const float width = hi - lo;
    return std::isfinite(width) ? width : std::numeric_limits<float>::infinity();
  }

  void reorder_slab_slots_by_width()
  {
    if (slab_count == 0 || cluster_num == 0)
    {
      return;
    }

    std::vector<size_t> order(slab_count);
    std::vector<uint32_t> tmp_neighbours(slab_count);
    std::vector<float> tmp_norms(slab_count);
    std::vector<float> tmp_l(slab_count);
    std::vector<float> tmp_h(slab_count);

    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      const size_t slab_offset = cluster_id * slab_count;
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const float lhs_width =
            slab_width_key(slab_l[slab_offset + lhs], slab_h[slab_offset + lhs]);
        const float rhs_width =
            slab_width_key(slab_l[slab_offset + rhs], slab_h[slab_offset + rhs]);
        if (lhs_width == rhs_width)
        {
          return lhs < rhs;
        }
        return lhs_width < rhs_width;
      });

      for (size_t j = 0; j < slab_count; ++j)
      {
        const size_t src = slab_offset + order[j];
        tmp_neighbours[j] = slab_neighbour_indices[src];
        tmp_norms[j] = slab_norms[src];
        tmp_l[j] = slab_l[src];
        tmp_h[j] = slab_h[src];
      }
      for (size_t j = 0; j < slab_count; ++j)
      {
        const size_t dst = slab_offset + j;
        slab_neighbour_indices[dst] = tmp_neighbours[j];
        slab_norms[dst] = tmp_norms[j];
        slab_l[dst] = tmp_l[j];
        slab_h[dst] = tmp_h[j];
      }
    }
  }

  void readMetaData()
  {
    std::array<char, 8> observed_magic{};
    fmeta.read(observed_magic.data(),
               static_cast<std::streamsize>(observed_magic.size()));
    if (!fmeta || observed_magic != kMetadataMagic)
    {
      throw std::runtime_error(
          "incompatible metadata schema for " + metadatafile_ +
          ": observed magic " + format_metadata_magic(observed_magic) +
          ", expected DJMETA\\0\\0 schema version " +
          std::to_string(kMetadataSchemaVersion) + ". Delete the index file at " +
          metadatafile_ + " and rebuild with the current binary.");
    }

    uint64_t schema_version = 0;
    fmeta.read((char*)&schema_version, sizeof(uint64_t));
    if (!fmeta || schema_version != kMetadataSchemaVersion)
    {
      throw std::runtime_error(
          "unsupported metadata schema version for " + metadatafile_ +
          ": observed " + std::to_string(schema_version) + ", expected " +
          std::to_string(kMetadataSchemaVersion) + ". Delete the index file at " +
          metadatafile_ + " and rebuild with the current binary.");
    }

    fmeta.read((char*)&n, sizeof(size_t));
    fmeta.read((char*)&d, sizeof(size_t));
    fmeta.read((char*)&cluster_num, sizeof(size_t));
    fmeta.read((char*)&max_points, sizeof(size_t));
    uint64_t metadata_pca_rank = 0;
    fmeta.read((char*)&metadata_pca_rank, sizeof(uint64_t));
    pca_rank = static_cast<size_t>(metadata_pca_rank);
    uint64_t metadata_target_cell_vecs = 0;
    fmeta.read((char*)&metadata_target_cell_vecs, sizeof(uint64_t));
    target_cell_vecs = static_cast<size_t>(metadata_target_cell_vecs);
    uint64_t metadata_vec_dtype = 0;
    fmeta.read((char*)&metadata_vec_dtype, sizeof(uint64_t));
    if (!fmeta)
    {
      throw std::runtime_error("failed to read vec_dtype from metadata: " +
                               metadatafile_);
    }
    if (!range_search_config::is_supported_vec_dtype(
            static_cast<uint32_t>(metadata_vec_dtype)))
    {
      throw std::runtime_error("unsupported metadata vec_dtype for " +
                               metadatafile_ + ": " +
                               std::to_string(metadata_vec_dtype));
    }
    vec_dtype = static_cast<uint32_t>(metadata_vec_dtype);
    bytes_per_element =
        range_search_config::bytes_per_element_for_fbin_dtype(vec_dtype);
    if (target_cell_vecs < 2)
    {
      throw std::runtime_error("metadata target_cell_vecs out of range for " +
                               metadatafile_ + ": " +
                               std::to_string(target_cell_vecs));
    }
    if (pca_rank > d)
    {
      throw std::runtime_error("metadata pca_rank out of range for " +
                               metadatafile_ + ": " +
                               std::to_string(pca_rank) + " > dim " +
                               std::to_string(d));
    }
    bucket_sizes.resize(cluster_num);
    fmeta.read((char*)bucket_sizes.data(), sizeof(size_t) * cluster_num);
    centroids.resize(d * cluster_num);
    fmeta.read((char*)centroids.data(), sizeof(float) * d * cluster_num);
    radii.resize(cluster_num);
    fmeta.read((char*)radii.data(), sizeof(float) * cluster_num);
    pca_basis.resize(cluster_num * pca_rank * d);
    pca_min.resize(cluster_num * pca_rank);
    pca_max.resize(cluster_num * pca_rank);
    pca_residual_radius.resize(cluster_num);
    pca_effective_rank.resize(cluster_num);
    if (!pca_basis.empty())
    {
      fmeta.read((char*)pca_basis.data(), sizeof(float) * pca_basis.size());
    }
    if (!pca_min.empty())
    {
      fmeta.read((char*)pca_min.data(), sizeof(float) * pca_min.size());
    }
    if (!pca_max.empty())
    {
      fmeta.read((char*)pca_max.data(), sizeof(float) * pca_max.size());
    }
    fmeta.read((char*)pca_residual_radius.data(),
               sizeof(float) * pca_residual_radius.size());
    fmeta.read((char*)pca_effective_rank.data(),
               sizeof(uint32_t) * pca_effective_rank.size());
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      const size_t max_effective_rank =
          std::min({pca_rank, bucket_sizes[cluster_id], d});
      if (pca_effective_rank[cluster_id] > max_effective_rank)
      {
        throw std::runtime_error(
            "metadata effective PCA rank out of range for " + metadatafile_ +
            " at cluster " + std::to_string(cluster_id) + ": " +
            std::to_string(pca_effective_rank[cluster_id]) + " > " +
            std::to_string(max_effective_rank));
      }
    }
    assignment.resize(cluster_num);
    for (size_t i = 0; i < cluster_num; i++)
    {
      assignment[i].resize(bucket_sizes[i]);
      fmeta.read((char*)assignment[i].data(), bucket_sizes[i] * sizeof(size_t));
    }
    uint64_t metadata_slab_count = 0;
    fmeta.read((char*)&metadata_slab_count, sizeof(uint64_t));
    if (!fmeta)
    {
      throw std::runtime_error("failed to read slab_count from metadata: " +
                               metadatafile_);
    }
    slab_count = static_cast<size_t>(metadata_slab_count);
    if (slab_count > 0)
    {
      slab_neighbour_indices.resize(cluster_num * slab_count);
      fmeta.read((char*)slab_neighbour_indices.data(),
                 sizeof(uint32_t) * slab_neighbour_indices.size());
      slab_norms.resize(cluster_num * slab_count);
      fmeta.read((char*)slab_norms.data(), sizeof(float) * slab_norms.size());
      slab_l.resize(cluster_num * slab_count);
      fmeta.read((char*)slab_l.data(), sizeof(float) * slab_l.size());
      slab_h.resize(cluster_num * slab_count);
      fmeta.read((char*)slab_h.data(), sizeof(float) * slab_h.size());
      if (!fmeta)
      {
        throw std::runtime_error("failed to read slab arrays from metadata: " +
                                 metadatafile_);
      }

      for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
      {
        const size_t slab_offset = cluster_id * slab_count;
        for (size_t j = 0; j < slab_count; ++j)
        {
          const uint32_t neighbour = slab_neighbour_indices[slab_offset + j];
          if (neighbour >= cluster_num)
          {
            throw std::runtime_error(
                "metadata slab neighbour out of range for " + metadatafile_ +
                " at cluster " + std::to_string(cluster_id) + ", offset " +
                std::to_string(j) + ": " + std::to_string(neighbour) + " >= " +
                std::to_string(cluster_num));
          }
          if (neighbour == cluster_id)
          {
            throw std::runtime_error(
                "metadata slab neighbour self-edge for " + metadatafile_ +
                " at cluster " + std::to_string(cluster_id) + ", offset " +
                std::to_string(j));
          }
          if (slab_norms[slab_offset + j] < 1e-9f)
          {
            throw std::runtime_error(
                "metadata slab norm out of range for " + metadatafile_ +
                " at cluster " + std::to_string(cluster_id) + ", offset " +
                std::to_string(j) + ": " +
                std::to_string(slab_norms[slab_offset + j]));
          }
        }
      }
      reorder_slab_slots_by_width();
    }
    skip_aligned_padding(fmeta, metadatafile_, "cluster_cell_offsets");
    cluster_cell_offsets.resize(cluster_num + 1);
    fmeta.read((char*)cluster_cell_offsets.data(),
               sizeof(uint64_t) * cluster_cell_offsets.size());
    if (!fmeta)
    {
      throw std::runtime_error(
          "failed to read cluster_cell_offsets from metadata: " +
          metadatafile_);
    }
    validate_cluster_cell_offsets();

    const size_t total_cells =
        static_cast<size_t>(cluster_cell_offsets.back());
    skip_aligned_padding(fmeta, metadatafile_, "cell_vec_offsets");
    cell_vec_offsets.resize(total_cells + 1);
    fmeta.read((char*)cell_vec_offsets.data(),
               sizeof(uint64_t) * cell_vec_offsets.size());
    if (!fmeta)
    {
      throw std::runtime_error("failed to read cell_vec_offsets from metadata: " +
                               metadatafile_);
    }
    validate_cell_vec_offsets();

    cluster_bound_offsets.assign(cluster_num + 1, 0);
    size_t bound_floats = 0;
    for (size_t cid = 0; cid < cluster_num; ++cid)
    {
      cluster_bound_offsets[cid] = bound_floats;
      const size_t cell_count =
          static_cast<size_t>(cluster_cell_offsets[cid + 1] -
                              cluster_cell_offsets[cid]);
      bound_floats += cell_count *
                      (2 * static_cast<size_t>(pca_effective_rank[cid]) + 2);
    }
    cluster_bound_offsets[cluster_num] = bound_floats;
    skip_aligned_padding(fmeta, metadatafile_, "cell_bounds");
    cell_bounds.resize(bound_floats);
    if (!cell_bounds.empty())
    {
      fmeta.read((char*)cell_bounds.data(),
                 sizeof(float) * cell_bounds.size());
    }
    if (!fmeta)
    {
      throw std::runtime_error("failed to read cell_bounds from metadata: " +
                               metadatafile_);
    }
    if (!fmeta)
    {
      throw std::runtime_error("failed to read metadata file: " +
                               metadatafile_);
    }
    if (range_search_config::is_byte_dtype(vec_dtype))
    {
      sum_for_offset.resize(n);
      norm_sq_for_l2.resize(n);
      fmeta.read(reinterpret_cast<char*>(sum_for_offset.data()),
                 static_cast<std::streamsize>(sum_for_offset.size() *
                                              sizeof(int32_t)));
      fmeta.read(reinterpret_cast<char*>(norm_sq_for_l2.data()),
                 static_cast<std::streamsize>(norm_sq_for_l2.size() *
                                              sizeof(int32_t)));
      if (!fmeta)
      {
        throw std::runtime_error("failed to read byte-dtype sum/norm metadata tail: " +
                                 metadatafile_);
      }
    }
    const std::streamoff metadata_end_pos = fmeta.tellg();
    if (metadata_end_pos < 0)
    {
      throw std::runtime_error("failed to determine metadata read offset: " +
                               metadatafile_);
    }
    const uintmax_t metadata_file_size =
        std::filesystem::file_size(metadatafile_);
    if (static_cast<uintmax_t>(metadata_end_pos) != metadata_file_size)
    {
      throw std::runtime_error(
          "metadata file size mismatch for " + metadatafile_ + ": expected " +
          std::to_string(static_cast<uintmax_t>(metadata_end_pos)) +
          ", actual " + std::to_string(metadata_file_size));
    }

    file_pos.resize(cluster_num);
    size_t cumu_size = 0;
    for (size_t i = 0; i < cluster_num; i++)
    {
      file_pos[i] = cumu_size * bytes_per_element * d;
      cumu_size += bucket_sizes[i];
    }

    const size_t payload_size = max_points * d * bytes_per_element;
    buf_size_ = PAGE_SIZE * (div_round_up(payload_size, PAGE_SIZE) + 1);
    if (buf_size_ == 0)
    {
      buf_size_ = PAGE_SIZE;
    }
    reconcile_mem_budget();
    init_io_slots();
    const size_t capacity_entries =
        buf_size_ == 0 ? 0
                       : budget_inputs_.cluster_cache_resident_bytes /
                             buf_size_;
    cluster_cache_ = std::make_unique<ClusterPayloadCache>(
        capacity_entries, buf_size_, cluster_cache_env_disabled_);
    cell_cache_ = std::make_unique<CellPayloadCache>(
        budget_inputs_.cell_cache_resident_bytes, cell_cache_env_disabled_);
    verify_registered_io_budget();
    log_mem_budget();
  }

  float pca_lower_bound(size_t cluster_id, const float* query) const
  {
    if (cluster_id >= cluster_num)
    {
      throw std::runtime_error("cluster id out of range for PCA bound: " +
                               std::to_string(cluster_id));
    }

    const float* centroid = centroids.data() + cluster_id * d;
    float total_sq = 0.0f;
    for (size_t col = 0; col < d; ++col)
    {
      const float diff = query[col] - centroid[col];
      total_sq += diff * diff;
    }

    const uint32_t effective_rank = pca_effective_rank[cluster_id];
    if (effective_rank == 0)
    {
      const float residual_delta =
          std::max(0.0f, std::sqrt(total_sq) -
                             pca_residual_radius[cluster_id]);
      return residual_delta;
    }

    const size_t basis_offset = cluster_id * pca_rank * d;
    const size_t extent_offset = cluster_id * pca_rank;
    float projected_sq = 0.0f;
    float lower_sq = 0.0f;
    for (size_t axis = 0; axis < effective_rank; ++axis)
    {
      const float* basis_row = pca_basis.data() + basis_offset + axis * d;
      float coord = 0.0f;
      for (size_t col = 0; col < d; ++col)
      {
        coord += basis_row[col] * (query[col] - centroid[col]);
      }
      projected_sq += coord * coord;
      const float below = std::max(pca_min[extent_offset + axis] - coord, 0.0f);
      const float above = std::max(coord - pca_max[extent_offset + axis], 0.0f);
      const float box_delta = std::max(below, above);
      lower_sq += box_delta * box_delta;
    }

    const float residual_q =
        std::sqrt(std::max(0.0f, total_sq - projected_sq));
    const float residual_delta =
        std::max(0.0f, residual_q - pca_residual_radius[cluster_id]);
    return std::sqrt(lower_sq + residual_delta * residual_delta);
  }

  float pca_slab_lower_bound_impl(size_t cluster_id,
                                  const float* q_dot_centroids,
                                  size_t q_dot_centroids_size,
                                  float prune_threshold) const
  {
    if (slab_count == 0)
    {
      return 0.0f;
    }
    if (cluster_id >= cluster_num)
    {
      throw std::runtime_error("cluster id out of range for slab bound: " +
                               std::to_string(cluster_id));
    }
    if (q_dot_centroids_size < cluster_num)
    {
      throw std::runtime_error("q_dot_centroids too small for slab bound: " +
                               std::to_string(q_dot_centroids_size) + " < " +
                               std::to_string(cluster_num));
    }

    const float q_dot_self = q_dot_centroids[cluster_id];
    const size_t slab_offset = cluster_id * slab_count;
    float max_slab = 0.0f;
    for (size_t j = 0; j < slab_count; ++j)
    {
      const uint32_t neighbour = slab_neighbour_indices[slab_offset + j];
      const float proj =
          (q_dot_centroids[neighbour] - q_dot_self) / slab_norms[slab_offset + j];
      const float below = std::max(slab_l[slab_offset + j] - proj, 0.0f);
      const float above = std::max(proj - slab_h[slab_offset + j], 0.0f);
      max_slab = std::max(max_slab, std::max(below, above));
      if (max_slab > prune_threshold)
      {
        return max_slab;
      }
    }
    return max_slab;
  }

  float pca_slab_lower_bound(size_t cluster_id,
                             const float* q_dot_centroids,
                             size_t q_dot_centroids_size,
                             float prune_threshold) const
  {
    return pca_slab_lower_bound_impl(
        cluster_id, q_dot_centroids, q_dot_centroids_size, prune_threshold);
  }

  float pca_slab_lower_bound(size_t cluster_id,
                             const std::vector<float>& q_dot_centroids,
                             float prune_threshold) const
  {
    return pca_slab_lower_bound_impl(
        cluster_id, q_dot_centroids.data(), q_dot_centroids.size(),
        prune_threshold);
  }

  float pca_slab_lower_bound(size_t cluster_id,
                             const std::vector<float>& q_dot_centroids) const
  {
    return pca_slab_lower_bound(
        cluster_id, q_dot_centroids, std::numeric_limits<float>::infinity());
  }

  size_t cell_count_of(size_t cluster_id) const
  {
    if (cluster_id >= cluster_num)
    {
      throw std::runtime_error("cluster id out of range for cell count: " +
                               std::to_string(cluster_id));
    }
    return static_cast<size_t>(cluster_cell_offsets[cluster_id + 1] -
                               cluster_cell_offsets[cluster_id]);
  }

  std::pair<size_t, size_t> cell_global_vec_range(size_t cluster_id,
                                                  size_t cell_idx) const
  {
    if (cell_idx >= cell_count_of(cluster_id))
    {
      throw std::runtime_error("cell index out of range for cluster " +
                               std::to_string(cluster_id) + ": " +
                               std::to_string(cell_idx));
    }
    const size_t fid =
        static_cast<size_t>(cluster_cell_offsets[cluster_id]) + cell_idx;
    return {static_cast<size_t>(cell_vec_offsets[fid]),
            static_cast<size_t>(cell_vec_offsets[fid + 1])};
  }

  size_t cluster_global_vec_offset(size_t cluster_id) const
  {
    if (cluster_id >= cluster_num)
    {
      throw std::runtime_error("cluster id out of range for row offset: " +
                               std::to_string(cluster_id));
    }
    const size_t first_cell =
        static_cast<size_t>(cluster_cell_offsets[cluster_id]);
    if (first_cell >= cell_vec_offsets.size())
    {
      return 0;
    }
    return static_cast<size_t>(cell_vec_offsets[first_cell]);
  }

  const float* cell_lo_ptr(size_t cluster_id, size_t cell_idx) const
  {
    return cell_bound_ptr(cluster_id, cell_idx);
  }

  const float* cell_hi_ptr(size_t cluster_id, size_t cell_idx) const
  {
    return cell_bound_ptr(cluster_id, cell_idx) +
           pca_effective_rank[cluster_id];
  }

  float cell_res_lo(size_t cluster_id, size_t cell_idx) const
  {
    const size_t effective_rank = pca_effective_rank[cluster_id];
    return cell_bound_ptr(cluster_id, cell_idx)[2 * effective_rank];
  }

  float cell_res_hi(size_t cluster_id, size_t cell_idx) const
  {
    const size_t effective_rank = pca_effective_rank[cluster_id];
    return cell_bound_ptr(cluster_id, cell_idx)[2 * effective_rank + 1];
  }

  using IOPlan = std::vector<uint64_t>;

  void submit_and_drain(
      int slot_id, const std::vector<std::pair<size_t, IOPlan>>& plans,
      const std::function<void(size_t cluster_id, uint64_t cell_id,
                               const void* data, size_t vec_count,
                               size_t cluster_local_row_base)>&
          on_complete)
  {
    if (plans.empty())
    {
      return;
    }
    if (slot_id < 0)
    {
      throw std::runtime_error("invalid io_uring slot_id: " +
                               std::to_string(slot_id));
    }
    if (rings_.empty())
    {
      throw std::runtime_error(
          "ClusterReader metadata must be loaded before cell reads");
    }
    if (n_io_slots_ == 0 || slot_locks_.size() != n_io_slots_)
    {
      throw std::runtime_error("ClusterReader io slots are not initialized");
    }
    const size_t effective_slot = static_cast<size_t>(slot_id) % n_io_slots_;
    std::lock_guard<std::mutex> guard(slot_locks_[effective_slot]);
    if (poisoned_slots_[effective_slot])
    {
      throw std::runtime_error("io_uring slot is no longer usable: " +
                               std::to_string(effective_slot));
    }

    struct CellRead
    {
      size_t cluster_id = 0;
      std::vector<uint64_t> cell_ids;
      uint64_t cell_id = 0;
      size_t buffer_offset = 0;
      size_t required_read_size = 0;
      size_t aligned_read_size = 0;
      size_t read_offset = 0;
      size_t vec_count = 0;
      size_t cluster_local_row_base = 0;
      bool full_cluster = false;
    };

    auto validate_cell_for_cluster = [&](size_t cluster_id, size_t fid) {
      if (fid >= cell_vec_offsets.size() ||
          fid == cell_vec_offsets.size() - 1)
      {
        throw std::runtime_error("cell id out of range for IO plan: " +
                                 std::to_string(fid));
      }
      const size_t first_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id]);
      const size_t next_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id + 1]);
      if (fid < first_cell || fid >= next_cell)
      {
        throw std::runtime_error(
            "cell id " + std::to_string(fid) +
            " does not belong to cluster " + std::to_string(cluster_id));
      }
    };

    auto emit_cell_from_cluster_payload =
        [&](size_t cluster_id, uint64_t fid_u64, const char* payload_base,
            size_t payload_offset, size_t valid_bytes) {
          const size_t fid = static_cast<size_t>(fid_u64);
          validate_cell_for_cluster(cluster_id, fid);
          const size_t first_cell_idx =
              static_cast<size_t>(cluster_cell_offsets[cluster_id]);
          const size_t first_cell_vec_start =
              static_cast<size_t>(cell_vec_offsets[first_cell_idx]);
          const size_t this_cell_vec_start =
              static_cast<size_t>(cell_vec_offsets[fid]);
          const size_t this_cell_vec_end =
              static_cast<size_t>(cell_vec_offsets[fid + 1]);
          if (this_cell_vec_start < first_cell_vec_start)
          {
            throw std::runtime_error(
                "cell_vec_offsets underflow for cluster-local slice");
          }
          const size_t local_vec_offset =
              this_cell_vec_start - first_cell_vec_start;
          const size_t vec_count = this_cell_vec_end - this_cell_vec_start;
          const size_t byte_offset =
              payload_offset + local_vec_offset * d * bytes_per_element;
          const size_t byte_count = vec_count * d * bytes_per_element;
          if (byte_offset > valid_bytes || byte_count > valid_bytes - byte_offset)
          {
            throw std::runtime_error("cached cluster payload too small for cell " +
                                     std::to_string(fid));
          }
          const void* data_ptr = payload_base + byte_offset;
          on_complete(cluster_id, fid_u64, data_ptr, vec_count,
                      local_vec_offset);
        };

    std::vector<CellRead> reads;
    const bool use_cell_cache = cell_cache_ && cell_cache_->active();
    for (const auto& item : plans)
    {
      const size_t cluster_id = item.first;
      if (cluster_id >= cluster_num)
      {
        throw std::runtime_error("cluster id out of range: " +
                                 std::to_string(cluster_id));
      }

      if (use_cell_cache)
      {
        for (const uint64_t fid_u64 : item.second)
        {
          const size_t fid = static_cast<size_t>(fid_u64);
          validate_cell_for_cluster(cluster_id, fid);
          CellPayloadCache::CacheRef cached = cell_cache_->lookup(fid_u64);
          if (cached)
          {
            const size_t first_cell =
                static_cast<size_t>(cluster_cell_offsets[cluster_id]);
            const size_t cluster_first_vec =
                static_cast<size_t>(cell_vec_offsets[first_cell]);
            const size_t local_vec_offset =
                static_cast<size_t>(cell_vec_offsets[fid]) - cluster_first_vec;
            const void* data_ptr = cached.buffer() + cached.payload_offset();
            on_complete(cluster_id, fid_u64, data_ptr, cached.vec_count(),
                        local_vec_offset);
            continue;
          }

          const size_t vec_start = static_cast<size_t>(cell_vec_offsets[fid]);
          const size_t vec_end = static_cast<size_t>(cell_vec_offsets[fid + 1]);
          const size_t vec_count = vec_end - vec_start;
          if (vec_count == 0)
          {
            throw std::runtime_error("empty cell in IO plan: " +
                                     std::to_string(fid));
          }
          const size_t abs_byte_start = vec_start * d * bytes_per_element;
          const size_t abs_byte_end = vec_end * d * bytes_per_element;
          const size_t read_offset = (abs_byte_start / PAGE_SIZE) * PAGE_SIZE;
          const size_t buffer_offset = abs_byte_start - read_offset;
          const size_t required_read_size =
              buffer_offset + (abs_byte_end - abs_byte_start);
          const size_t aligned_read_size =
              div_round_up(required_read_size, PAGE_SIZE) * PAGE_SIZE;
          if (aligned_read_size > buf_size_)
          {
            throw std::runtime_error("cell read size exceeds registered buffer");
          }
          CellRead read;
          read.cluster_id = cluster_id;
          read.cell_id = fid_u64;
          read.buffer_offset = buffer_offset;
          read.required_read_size = required_read_size;
          read.aligned_read_size = aligned_read_size;
          read.read_offset = read_offset;
          read.vec_count = vec_count;
          {
            const size_t first_cell =
                static_cast<size_t>(cluster_cell_offsets[cluster_id]);
            const size_t cluster_first_vec =
                static_cast<size_t>(cell_vec_offsets[first_cell]);
            read.cluster_local_row_base = vec_start - cluster_first_vec;
          }
          read.full_cluster = false;
          reads.push_back(std::move(read));
        }
        continue;
      }

      ClusterPayloadCache::CacheRef cached =
          cluster_cache_ ? cluster_cache_->lookup(static_cast<uint32_t>(cluster_id))
                         : ClusterPayloadCache::CacheRef{};
      if (cached)
      {
        for (const uint64_t fid_u64 : item.second)
        {
          emit_cell_from_cluster_payload(
              cluster_id, fid_u64, cached.buffer(), cached.payload_offset(),
              cached.valid_bytes());
        }
        continue;
      }

      if (cluster_cache_ && cluster_cache_->active())
      {
        for (const uint64_t fid_u64 : item.second)
        {
          validate_cell_for_cluster(cluster_id, static_cast<size_t>(fid_u64));
        }
        const size_t buffer_offset = file_pos[cluster_id] % PAGE_SIZE;
        const size_t file_offset = file_pos[cluster_id] - buffer_offset;
        const size_t read_size =
            bucket_sizes[cluster_id] * d * bytes_per_element;
        const size_t required_read_size = buffer_offset + read_size;
        const size_t aligned_read_size =
            div_round_up(required_read_size, PAGE_SIZE) * PAGE_SIZE;
        if (aligned_read_size > buf_size_)
        {
          throw std::runtime_error("cluster read size exceeds registered buffer");
        }
        CellRead read;
        read.cluster_id = cluster_id;
        read.cell_ids = item.second;
        read.buffer_offset = buffer_offset;
        read.required_read_size = required_read_size;
        read.aligned_read_size = aligned_read_size;
        read.read_offset = file_offset;
        read.cluster_local_row_base = 0;
        read.full_cluster = true;
        reads.push_back(std::move(read));
        continue;
      }

      for (const uint64_t fid_u64 : item.second)
      {
        const size_t fid = static_cast<size_t>(fid_u64);
        validate_cell_for_cluster(cluster_id, fid);
        const size_t vec_start = static_cast<size_t>(cell_vec_offsets[fid]);
        const size_t vec_end = static_cast<size_t>(cell_vec_offsets[fid + 1]);
        const size_t vec_count = vec_end - vec_start;
        if (vec_count == 0)
        {
          throw std::runtime_error("empty cell in IO plan: " +
                                   std::to_string(fid));
        }
        const size_t abs_byte_start = vec_start * d * bytes_per_element;
        const size_t abs_byte_end = vec_end * d * bytes_per_element;
        const size_t read_offset = (abs_byte_start / PAGE_SIZE) * PAGE_SIZE;
        const size_t buffer_offset = abs_byte_start - read_offset;
        const size_t required_read_size = buffer_offset + (abs_byte_end - abs_byte_start);
        const size_t aligned_read_size =
            div_round_up(required_read_size, PAGE_SIZE) * PAGE_SIZE;
        if (aligned_read_size > buf_size_)
        {
          throw std::runtime_error("cell read size exceeds registered buffer");
        }
        CellRead read;
        read.cluster_id = cluster_id;
        read.cell_id = fid_u64;
        read.buffer_offset = buffer_offset;
        read.required_read_size = required_read_size;
        read.aligned_read_size = aligned_read_size;
        read.read_offset = read_offset;
        read.vec_count = vec_count;
        {
          const size_t first_cell =
              static_cast<size_t>(cluster_cell_offsets[cluster_id]);
          const size_t cluster_first_vec =
              static_cast<size_t>(cell_vec_offsets[first_cell]);
          read.cluster_local_row_base = vec_start - cluster_first_vec;
        }
        read.full_cluster = false;
        reads.push_back(std::move(read));
      }
    }
    if (reads.empty())
    {
      return;
    }

    IoSlot& slot = slots_[effective_slot];
    io_uring& ring = rings_[effective_slot];
    std::vector<InFlight> in_flight(slots_[effective_slot].buffers.size());
    std::vector<size_t> free_buffers;
    free_buffers.reserve(slots_[effective_slot].buffers.size());
    for (size_t i = 0; i < slots_[effective_slot].buffers.size(); ++i)
    {
      free_buffers.push_back(slots_[effective_slot].buffers.size() - 1 - i);
    }

    size_t next_to_prep = 0;
    size_t submitted = 0;
    size_t reaped = 0;
    size_t active = 0;
    bool io_failed = false;
    std::string io_error;
    std::exception_ptr callback_error;

    auto prep_one = [&](const CellRead& read, size_t buffer_index) {
      io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (sqe == nullptr)
      {
        throw std::runtime_error("io_uring_get_sqe returned null");
      }
      io_uring_prep_read_fixed(sqe, direct_fd_, slot.buffers[buffer_index].data,
                               read.aligned_read_size,
                               static_cast<off_t>(read.read_offset),
                               buffer_index);
      if (capture_read_requests_for_test_)
      {
        read_requests_for_test_.push_back(
            {read.read_offset, read.aligned_read_size});
      }
      io_uring_sqe_set_data64(sqe, static_cast<unsigned long long>(buffer_index));
      in_flight[buffer_index] = {read.cluster_id, read.buffer_offset,
                                 read.required_read_size,
                                 read.aligned_read_size, false,
                                 read.full_cluster, read.cell_id,
                                 read.vec_count, read.cell_ids,
                                 read.cluster_local_row_base};
    };

    auto wait_one_cqe = [&](io_uring_cqe** cqe) {
      int ret = 0;
      do
      {
        ret = io_uring_wait_cqe(&ring, cqe);
      } while (ret == -EINTR);
      return ret;
    };

    auto drain_active = [&]() {
      while (active > 0)
      {
        io_uring_cqe* cqe = nullptr;
        const int wait_ret = wait_one_cqe(&cqe);
        if (wait_ret < 0)
        {
          poisoned_slots_[effective_slot] = true;
          if (io_error.empty())
          {
            io_error = "io_uring_wait_cqe failed while draining: " +
                       std::to_string(wait_ret) + " (" +
                       strerror(-wait_ret) + ")";
          }
          break;
        }
        const size_t buffer_index =
            static_cast<size_t>(io_uring_cqe_get_data64(cqe));
        if (buffer_index < in_flight.size())
        {
          in_flight[buffer_index].active = false;
        }
        io_uring_cqe_seen(&ring, cqe);
        ++reaped;
        --active;
      }
    };

    auto submit_prepared = [&](const std::vector<size_t>& prepared_buffers) {
      if (prepared_buffers.empty())
      {
        return;
      }
      int ret = 0;
      do
      {
        ret = io_uring_submit(&ring);
      } while (ret == -EINTR);
      if (ret < 0)
      {
        poisoned_slots_[effective_slot] = true;
        for (const size_t buffer_index : prepared_buffers)
        {
          in_flight[buffer_index].active = false;
          free_buffers.push_back(buffer_index);
        }
        throw std::runtime_error("io_uring_submit failed: " +
                                 std::to_string(ret) + " (" +
                                 strerror(-ret) + ")");
      }
      const size_t accepted = static_cast<size_t>(ret);
      if (accepted != prepared_buffers.size())
      {
        poisoned_slots_[effective_slot] = true;
      }
      for (size_t i = 0; i < prepared_buffers.size(); ++i)
      {
        const size_t buffer_index = prepared_buffers[i];
        if (i < accepted)
        {
          in_flight[buffer_index].active = true;
          ++active;
          ++submitted;
        }
        else
        {
          in_flight[buffer_index].active = false;
          free_buffers.push_back(buffer_index);
        }
      }
      if (accepted != prepared_buffers.size())
      {
        drain_active();
        throw std::runtime_error("io_uring_submit submitted " +
                                 std::to_string(accepted) + " of " +
                                 std::to_string(prepared_buffers.size()) +
                                 " prepared SQEs");
      }
    };

    auto prep_next = [&]() {
      const size_t buffer_index = free_buffers.back();
      free_buffers.pop_back();
      try
      {
        prep_one(reads[next_to_prep], buffer_index);
        ++next_to_prep;
        return buffer_index;
      }
      catch (...)
      {
        free_buffers.push_back(buffer_index);
        throw;
      }
    };

    auto submit_next_batch = [&](size_t count) {
      std::vector<size_t> prepared_buffers;
      prepared_buffers.reserve(count);
      try
      {
        for (size_t i = 0; i < count; ++i)
        {
          prepared_buffers.push_back(prep_next());
        }
        submit_prepared(prepared_buffers);
      }
      catch (const std::exception& e)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = e.what();
        }
        drain_active();
      }
    };

    const size_t initial = std::min(reads.size(), io_queue_depth_);
    submit_next_batch(initial);
    if (io_failed)
    {
      throw std::runtime_error(io_error);
    }

    while (reaped < submitted)
    {
      io_uring_cqe* cqe = nullptr;
      const int wait_ret = wait_one_cqe(&cqe);
      if (wait_ret < 0)
      {
        poisoned_slots_[effective_slot] = true;
        io_failed = true;
        if (io_error.empty())
        {
          io_error = "io_uring_wait_cqe failed: " +
                     std::to_string(wait_ret) + " (" +
                     strerror(-wait_ret) + ")";
        }
        break;
      }

      const size_t buffer_index =
          static_cast<size_t>(io_uring_cqe_get_data64(cqe));
      if (buffer_index >= in_flight.size() || !in_flight[buffer_index].active)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = "unexpected cqe buffer index: " +
                     std::to_string(buffer_index);
        }
        io_uring_cqe_seen(&ring, cqe);
        ++reaped;
        --active;
        continue;
      }

      InFlight meta = in_flight[buffer_index];
      in_flight[buffer_index].active = false;
      const int cqe_res = cqe->res;
      io_uring_cqe_seen(&ring, cqe);
      ++reaped;
      --active;

      if (cqe_res < 0 ||
          static_cast<size_t>(cqe_res) < meta.required_read_size)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = meta.full_cluster
                         ? "cluster " + std::to_string(meta.cluster_id) +
                               " async read failed with cqe res " +
                               std::to_string(cqe_res)
                         : "cell " + std::to_string(meta.cell_id) +
                               " async read failed with cqe res " +
                               std::to_string(cqe_res);
        }
        free_buffers.push_back(buffer_index);
      }
      else if (!io_failed && !callback_error)
      {
        try
        {
          const char* payload_base =
              static_cast<const char*>(slot.buffers[buffer_index].data);
          if (meta.full_cluster)
          {
            if (cluster_cache_ && cluster_cache_->active())
            {
              (void)cluster_cache_->try_insert(
                  static_cast<uint32_t>(meta.cluster_id), payload_base,
                  meta.buffer_offset, meta.aligned_read_size);
            }
            for (const uint64_t fid_u64 : meta.cell_ids)
            {
              emit_cell_from_cluster_payload(
                  meta.cluster_id, fid_u64, payload_base, meta.buffer_offset,
                  meta.aligned_read_size);
            }
          }
          else
          {
            const void* data_ptr = payload_base + meta.buffer_offset;
            if (cell_cache_ && cell_cache_->active())
            {
              (void)cell_cache_->try_insert(
                  meta.cell_id, payload_base, meta.buffer_offset,
                  meta.aligned_read_size, meta.vec_count,
                  meta.aligned_read_size);
            }
            on_complete(meta.cluster_id, meta.cell_id, data_ptr,
                        meta.vec_count, meta.cluster_local_row_base);
          }
        }
        catch (...)
        {
          callback_error = std::current_exception();
        }
        free_buffers.push_back(buffer_index);
      }
      else
      {
        free_buffers.push_back(buffer_index);
      }

      if (!io_failed && !callback_error && next_to_prep < reads.size())
      {
        submit_next_batch(1);
        if (io_failed)
        {
          break;
        }
      }
    }

    drain_active();

    if (callback_error)
    {
      std::rethrow_exception(callback_error);
    }
    if (io_failed)
    {
      throw std::runtime_error(io_error);
    }
  }

  void submit_and_drain(
      int slot_id, const std::vector<size_t>& cluster_ids,
      const std::function<void(size_t cluster_id, const void* data,
                               size_t bucket_size,
                               size_t cluster_local_row_base)>& on_complete)
  {
    if (cluster_ids.empty())
    {
      return;
    }
    if (slot_id < 0)
    {
      throw std::runtime_error("invalid io_uring slot_id: " +
                               std::to_string(slot_id));
    }
    if (rings_.empty())
    {
      throw std::runtime_error(
          "ClusterReader metadata must be loaded before cluster reads");
    }
    if (n_io_slots_ == 0 || slot_locks_.size() != n_io_slots_)
    {
      throw std::runtime_error("ClusterReader io slots are not initialized");
    }
    const size_t effective_slot = static_cast<size_t>(slot_id) % n_io_slots_;
    std::lock_guard<std::mutex> guard(slot_locks_[effective_slot]);
    if (poisoned_slots_[effective_slot])
    {
      throw std::runtime_error("io_uring slot is no longer usable: " +
                               std::to_string(effective_slot));
    }

    std::vector<size_t> pending_cluster_ids;
    pending_cluster_ids.reserve(cluster_ids.size());
    for (const size_t cluster_id : cluster_ids)
    {
      if (cluster_id >= cluster_num)
      {
        throw std::runtime_error("cluster id out of range: " +
                                 std::to_string(cluster_id));
      }
      ClusterPayloadCache::CacheRef cached =
          cluster_cache_ ? cluster_cache_->lookup(static_cast<uint32_t>(cluster_id))
                         : ClusterPayloadCache::CacheRef{};
      if (cached)
      {
        if (cached.payload_offset() > cached.valid_bytes())
        {
          throw std::runtime_error("cached cluster payload has invalid offset");
        }
        const size_t read_size =
            bucket_sizes[cluster_id] * d * bytes_per_element;
        if (read_size > cached.valid_bytes() - cached.payload_offset())
        {
          throw std::runtime_error(
              "cached cluster payload too small for cluster " +
              std::to_string(cluster_id));
        }
        const void* data_ptr = cached.buffer() + cached.payload_offset();
        on_complete(cluster_id, data_ptr, bucket_sizes[cluster_id], 0);
      }
      else
      {
        pending_cluster_ids.push_back(cluster_id);
      }
    }
    if (pending_cluster_ids.empty())
    {
      return;
    }

    IoSlot& slot = slots_[effective_slot];
    io_uring& ring = rings_[effective_slot];
    std::vector<InFlight> in_flight(slots_[effective_slot].buffers.size());
    std::vector<size_t> free_buffers;
    free_buffers.reserve(slots_[effective_slot].buffers.size());
    for (size_t i = 0; i < slots_[effective_slot].buffers.size(); ++i)
    {
      free_buffers.push_back(slots_[effective_slot].buffers.size() - 1 - i);
    }

    size_t next_to_prep = 0;
    size_t submitted = 0;
    size_t reaped = 0;
    size_t active = 0;
    bool io_failed = false;
    std::string io_error;
    std::exception_ptr callback_error;

    auto prep_one = [&](size_t cluster_id, size_t buffer_index) {
      if (cluster_id >= cluster_num)
      {
        throw std::runtime_error("cluster id out of range: " +
                                 std::to_string(cluster_id));
      }
      const size_t buffer_offset = file_pos[cluster_id] % PAGE_SIZE;
      const size_t file_offset = file_pos[cluster_id] - buffer_offset;
      const size_t read_size = bucket_sizes[cluster_id] * d * bytes_per_element;
      const size_t required_read_size = buffer_offset + read_size;
      const size_t aligned_read_size =
          div_round_up(buffer_offset + read_size, PAGE_SIZE) * PAGE_SIZE;
      if (aligned_read_size > buf_size_)
      {
        throw std::runtime_error("cluster read size exceeds registered buffer");
      }

      io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (sqe == nullptr)
      {
        throw std::runtime_error("io_uring_get_sqe returned null");
      }
      io_uring_prep_read_fixed(sqe, direct_fd_, slot.buffers[buffer_index].data,
                               aligned_read_size,
                               static_cast<off_t>(file_offset), buffer_index);
      if (capture_read_requests_for_test_)
      {
        read_requests_for_test_.push_back({file_offset, aligned_read_size});
      }
      io_uring_sqe_set_data64(sqe, static_cast<unsigned long long>(buffer_index));
      in_flight[buffer_index] = {cluster_id, buffer_offset, required_read_size,
                                 aligned_read_size, false};
    };

    auto wait_one_cqe = [&](io_uring_cqe** cqe) {
      int ret = 0;
      do
      {
        ret = io_uring_wait_cqe(&ring, cqe);
      } while (ret == -EINTR);
      return ret;
    };

    auto drain_active = [&]() {
      while (active > 0)
      {
        io_uring_cqe* cqe = nullptr;
        const int wait_ret = wait_one_cqe(&cqe);
        if (wait_ret < 0)
        {
          poisoned_slots_[effective_slot] = true;
          if (io_error.empty())
          {
            io_error = "io_uring_wait_cqe failed while draining: " +
                       std::to_string(wait_ret) + " (" +
                       strerror(-wait_ret) + ")";
          }
          break;
        }
        const size_t buffer_index =
            static_cast<size_t>(io_uring_cqe_get_data64(cqe));
        if (buffer_index < in_flight.size())
        {
          in_flight[buffer_index].active = false;
        }
        io_uring_cqe_seen(&ring, cqe);
        ++reaped;
        --active;
      }
    };

    auto submit_prepared = [&](const std::vector<size_t>& prepared_buffers) {
      if (prepared_buffers.empty())
      {
        return;
      }
      int ret = 0;
      do
      {
        ret = io_uring_submit(&ring);
      } while (ret == -EINTR);
      if (ret < 0)
      {
        poisoned_slots_[effective_slot] = true;
        for (const size_t buffer_index : prepared_buffers)
        {
          in_flight[buffer_index].active = false;
          free_buffers.push_back(buffer_index);
        }
        throw std::runtime_error("io_uring_submit failed: " +
                                 std::to_string(ret) + " (" +
                                 strerror(-ret) + ")");
      }
      const size_t accepted = static_cast<size_t>(ret);
      if (accepted != prepared_buffers.size())
      {
        poisoned_slots_[effective_slot] = true;
      }
      for (size_t i = 0; i < prepared_buffers.size(); ++i)
      {
        const size_t buffer_index = prepared_buffers[i];
        if (i < accepted)
        {
          in_flight[buffer_index].active = true;
          ++active;
          ++submitted;
        }
        else
        {
          in_flight[buffer_index].active = false;
          free_buffers.push_back(buffer_index);
        }
      }
      if (accepted != prepared_buffers.size())
      {
        drain_active();
        throw std::runtime_error("io_uring_submit submitted " +
                                 std::to_string(accepted) + " of " +
                                 std::to_string(prepared_buffers.size()) +
                                 " prepared SQEs");
      }
    };

    auto prep_next = [&]() {
      const size_t buffer_index = free_buffers.back();
      free_buffers.pop_back();
      try
      {
        prep_one(pending_cluster_ids[next_to_prep], buffer_index);
        ++next_to_prep;
        return buffer_index;
      }
      catch (...)
      {
        free_buffers.push_back(buffer_index);
        throw;
      }
    };

    auto submit_next_batch = [&](size_t count) {
      std::vector<size_t> prepared_buffers;
      prepared_buffers.reserve(count);
      try
      {
        for (size_t i = 0; i < count; ++i)
        {
          prepared_buffers.push_back(prep_next());
        }
        submit_prepared(prepared_buffers);
      }
      catch (const std::exception& e)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = e.what();
        }
        drain_active();
      }
    };

    const size_t initial = std::min(pending_cluster_ids.size(), io_queue_depth_);
    submit_next_batch(initial);
    if (io_failed)
    {
      throw std::runtime_error(io_error);
    }

    while (reaped < submitted)
    {
      io_uring_cqe* cqe = nullptr;
      const int wait_ret = wait_one_cqe(&cqe);
      if (wait_ret < 0)
      {
        poisoned_slots_[effective_slot] = true;
        io_failed = true;
        if (io_error.empty())
        {
          io_error = "io_uring_wait_cqe failed: " +
                     std::to_string(wait_ret) + " (" +
                     strerror(-wait_ret) + ")";
        }
        break;
      }

      const size_t buffer_index =
          static_cast<size_t>(io_uring_cqe_get_data64(cqe));
      if (buffer_index >= in_flight.size() || !in_flight[buffer_index].active)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = "unexpected cqe buffer index: " +
                     std::to_string(buffer_index);
        }
        io_uring_cqe_seen(&ring, cqe);
        ++reaped;
        --active;
        continue;
      }

      InFlight meta = in_flight[buffer_index];
      in_flight[buffer_index].active = false;
      const int cqe_res = cqe->res;
      io_uring_cqe_seen(&ring, cqe);
      ++reaped;
      --active;

      if (cqe_res < 0 ||
          static_cast<size_t>(cqe_res) < meta.required_read_size)
      {
        io_failed = true;
        if (io_error.empty())
        {
          io_error = "cluster " + std::to_string(meta.cluster_id) +
                     " async read failed with cqe res " +
                     std::to_string(cqe_res);
        }
        free_buffers.push_back(buffer_index);
      }
      else if (!io_failed && !callback_error)
      {
        try
        {
          if (cluster_cache_ && cluster_cache_->active())
          {
            (void)cluster_cache_->try_insert(
                static_cast<uint32_t>(meta.cluster_id),
                slot.buffers[buffer_index].data, meta.buffer_offset,
                meta.aligned_read_size);
          }
          const void* data_ptr =
              static_cast<const char*>(slot.buffers[buffer_index].data) +
              meta.buffer_offset;
          on_complete(meta.cluster_id, data_ptr,
                      bucket_sizes[meta.cluster_id], 0);
        }
        catch (...)
        {
          callback_error = std::current_exception();
        }
        free_buffers.push_back(buffer_index);
      }
      else
      {
        free_buffers.push_back(buffer_index);
      }

      if (!io_failed && !callback_error &&
          next_to_prep < pending_cluster_ids.size())
      {
        submit_next_batch(1);
        if (io_failed)
        {
          break;
        }
      }
    }

    drain_active();

    if (callback_error)
    {
      std::rethrow_exception(callback_error);
    }
    if (io_failed)
    {
      throw std::runtime_error(io_error);
    }
  }

  ~ClusterReader()
  {
    fmeta.close();
    if (cluster_cache_)
    {
      cluster_cache_->report_stats();
    }
    if (cell_cache_)
    {
      cell_cache_->report_stats();
    }
    for (size_t i = 0; i < rings_.size(); ++i)
    {
      if (ring_initialized_[i])
      {
        io_uring_queue_exit(&rings_[i]);
      }
    }
    for (auto& slot : slots_)
    {
      for (auto& buffer : slot.buffers)
      {
        free(buffer.data);
      }
    }
    if (direct_fd_ >= 0)
    {
      close(direct_fd_);
    }
  }

  const std::string& metadatafile() const
  {
    return metadatafile_;
  }

  size_t effective_n_io_slots() const
  {
    return n_io_slots_;
  }

  size_t effective_io_queue_depth() const
  {
    return io_queue_depth_;
  }

  size_t io_buffer_size() const
  {
    return buf_size_;
  }

  size_t cluster_cache_requested_bytes() const
  {
    return budget_inputs_.cluster_cache_requested_bytes;
  }

  size_t cluster_cache_resident_bytes() const
  {
    return budget_inputs_.cluster_cache_resident_bytes;
  }

  size_t cell_cache_requested_bytes() const
  {
    return budget_inputs_.cell_cache_requested_bytes;
  }

  size_t cell_cache_resident_bytes() const
  {
    return budget_inputs_.cell_cache_resident_bytes;
  }

  void report_cluster_cache_stats()
  {
    if (cluster_cache_)
    {
      cluster_cache_->report_stats();
    }
  }

  void report_cell_cache_stats()
  {
    if (cell_cache_)
    {
      cell_cache_->report_stats();
    }
  }

  struct ReadRequestForTest
  {
    size_t offset = 0;
    size_t length = 0;
  };

  void enable_read_request_capture_for_test(bool enabled)
  {
    capture_read_requests_for_test_ = enabled;
    if (!enabled)
    {
      read_requests_for_test_.clear();
    }
  }

  void clear_read_requests_for_test()
  {
    read_requests_for_test_.clear();
  }

  const std::vector<ReadRequestForTest>& read_requests_for_test() const
  {
    return read_requests_for_test_;
  }

  const ClusterPayloadCache* cluster_cache_for_test() const
  {
    return cluster_cache_.get();
  }

  ClusterPayloadCache* cluster_cache_for_test()
  {
    return cluster_cache_.get();
  }

  const CellPayloadCache* cell_cache_for_test() const
  {
    return cell_cache_.get();
  }

  CellPayloadCache* cell_cache_for_test()
  {
    return cell_cache_.get();
  }

  static void verify_io_buffer_budget(size_t n_io_slots,
                                      size_t actual_ring_entries,
                                      size_t buf_size,
                                      size_t io_budget_bytes)
  {
    if (multiply_exceeds(n_io_slots, actual_ring_entries, buf_size,
                         io_budget_bytes))
    {
      std::ostringstream out;
      out << "io_uring registered more buffers than budgeted: slots="
          << n_io_slots << ", ring_entries=" << actual_ring_entries
          << ", buf_size=" << format_mib(buf_size)
          << ", io_budget=" << format_mib(io_budget_bytes);
      throw std::runtime_error(out.str());
    }
  }

 private:
  struct AlignedBuffer
  {
    void* data = nullptr;
  };

  struct IoSlot
  {
    std::vector<AlignedBuffer> buffers;
    std::vector<iovec> iovs;
  };

  struct InFlight
  {
    size_t cluster_id = 0;
    size_t buffer_offset = 0;
    size_t required_read_size = 0;
    size_t aligned_read_size = 0;
    bool active = false;
    bool full_cluster = false;
    uint64_t cell_id = 0;
    size_t vec_count = 0;
    std::vector<uint64_t> cell_ids;
    size_t cluster_local_row_base = 0;
  };

  std::string metadatafile_;
  std::string clusterfile_;
  int direct_fd_;
  MemBudgetInputs budget_inputs_;
  size_t requested_io_queue_depth_;
  size_t requested_n_io_slots_;
  float cell_cache_share_pct_ = 0.0f;
  size_t n_io_slots_;
  size_t io_queue_depth_;
  size_t buf_size_;
  size_t last_metadata_resident_bytes_ = 0;
  size_t last_fixed_resident_bytes_ = 0;
  size_t last_io_budget_bytes_ = 0;
  size_t last_actual_ring_entries_ = 0;
  size_t last_registered_buffer_count_ = 0;
  bool cluster_cache_env_disabled_ = false;
  bool cell_cache_env_disabled_ = false;
  std::unique_ptr<ClusterPayloadCache> cluster_cache_;
  std::unique_ptr<CellPayloadCache> cell_cache_;
  bool capture_read_requests_for_test_ = false;
  std::vector<ReadRequestForTest> read_requests_for_test_;
  std::vector<io_uring> rings_;
  std::vector<bool> ring_initialized_;
  std::vector<bool> poisoned_slots_;
  std::vector<IoSlot> slots_;
  std::vector<std::mutex> slot_locks_;

  static std::string format_mib(size_t bytes)
  {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << static_cast<double>(bytes) / static_cast<double>(size_t{1} << 20);
    return out.str();
  }

  static bool multiply_exceeds(size_t a, size_t b, size_t c, size_t limit)
  {
    if (a == 0 || b == 0 || c == 0)
    {
      return false;
    }
    if (a > std::numeric_limits<size_t>::max() / b)
    {
      return true;
    }
    const size_t ab = a * b;
    if (ab > std::numeric_limits<size_t>::max() / c)
    {
      return true;
    }
    return ab * c > limit;
  }

  template <typename Vector>
  static size_t vector_bytes(const Vector& values)
  {
    using Value = typename Vector::value_type;
    return values.size() * sizeof(Value);
  }

  size_t metadata_resident_estimate() const
  {
    size_t total = 0;
    total += vector_bytes(bucket_sizes);
    total += vector_bytes(centroids);
    total += vector_bytes(radii);
    total += vector_bytes(pca_basis);
    total += vector_bytes(pca_min);
    total += vector_bytes(pca_max);
    total += vector_bytes(pca_residual_radius);
    total += vector_bytes(pca_effective_rank);
    total += vector_bytes(slab_neighbour_indices);
    total += vector_bytes(slab_norms);
    total += vector_bytes(slab_l);
    total += vector_bytes(slab_h);
    total += vector_bytes(cluster_cell_offsets);
    total += vector_bytes(cell_vec_offsets);
    total += vector_bytes(cluster_bound_offsets);
    total += vector_bytes(cell_bounds);
    total += vector_bytes(sum_for_offset);
    total += vector_bytes(norm_sq_for_l2);
    total += vector_bytes(file_pos);
    total += n * sizeof(size_t);
    return total;
  }

  std::string fixed_breakdown_message(const std::string& prefix,
                                      size_t metadata_resident,
                                      size_t fixed_resident) const
  {
    std::ostringstream out;
    out << prefix << ": metadata=" << format_mib(metadata_resident)
        << ", rbq_1bit="
        << format_mib(budget_inputs_.rbq_1bit_resident_bytes)
        << ", data_hnsw="
        << format_mib(budget_inputs_.data_hnsw_resident_bytes)
        << ", query=" << format_mib(budget_inputs_.query_resident_bytes)
        << ", per_thread="
        << format_mib(budget_inputs_.per_thread_resident_bytes)
        << ", sqg=" << format_mib(budget_inputs_.sqg_resident_bytes)
        << ", rotated_centroids="
        << format_mib(budget_inputs_.rotated_centroids_resident_bytes)
        << ", trace=" << format_mib(budget_inputs_.trace_resident_bytes)
        << ", hit_bookkeeping="
        << format_mib(budget_inputs_.hit_bookkeeping_resident_bytes)
        << ", cluster_cache="
        << format_mib(budget_inputs_.cluster_cache_resident_bytes)
        << ", cell_cache="
        << format_mib(budget_inputs_.cell_cache_resident_bytes)
        << ", slack=" << format_mib(kReservedSlack)
        << ", budget=" << format_mib(budget_inputs_.mem_budget_bytes)
        << ", fixed=" << format_mib(fixed_resident);
    return out.str();
  }

  std::string io_budget_failure_message(const std::string& prefix,
                                        size_t slots, size_t depth,
                                        size_t io_budget) const
  {
    std::ostringstream out;
    out << prefix << ": slots=" << slots
        << ", depth=" << depth << ", buf_size=" << format_mib(buf_size_)
        << ", io_budget=" << format_mib(io_budget) << ", "
        << fixed_breakdown_message("fixed_breakdown",
                                   last_metadata_resident_bytes_,
                                   last_fixed_resident_bytes_);
    return out.str();
  }

  void reconcile_mem_budget()
  {
    if (!std::isfinite(cell_cache_share_pct_) ||
        cell_cache_share_pct_ < 0.0f || cell_cache_share_pct_ > 100.0f)
    {
      throw std::runtime_error(
          "cell_cache_share_pct out of range or non-finite: " +
          std::to_string(cell_cache_share_pct_));
    }
    last_metadata_resident_bytes_ = metadata_resident_estimate();
    last_fixed_resident_bytes_ =
        last_metadata_resident_bytes_ +
        budget_inputs_.rbq_1bit_resident_bytes +
        budget_inputs_.data_hnsw_resident_bytes +
        budget_inputs_.query_resident_bytes +
        budget_inputs_.per_thread_resident_bytes +
        budget_inputs_.sqg_resident_bytes +
        budget_inputs_.rotated_centroids_resident_bytes +
        budget_inputs_.trace_resident_bytes +
        budget_inputs_.hit_bookkeeping_resident_bytes + kReservedSlack;

    if (budget_inputs_.mem_budget_bytes <= last_fixed_resident_bytes_)
    {
      budget_inputs_.cluster_cache_resident_bytes = 0;
      budget_inputs_.cell_cache_resident_bytes = 0;
      throw std::runtime_error(fixed_breakdown_message(
          "mem_budget too small", last_metadata_resident_bytes_,
          last_fixed_resident_bytes_));
    }

    if (buf_size_ < PAGE_SIZE)
    {
      throw std::runtime_error("ClusterReader buf_size is below PAGE_SIZE");
    }

    size_t depth = bit_floor_compat(requested_io_queue_depth_);
    if (depth == 0)
    {
      depth = 1;
    }
    const size_t effective_min_depth =
        std::min(depth, kMinEffectiveIoQueueDepth);
    const size_t io_floor_bytes = checked_mul(effective_min_depth, buf_size_);
    const size_t remaining_after_fixed =
        budget_inputs_.mem_budget_bytes - last_fixed_resident_bytes_;
    if (remaining_after_fixed < io_floor_bytes)
    {
      budget_inputs_.cluster_cache_resident_bytes = 0;
      budget_inputs_.cell_cache_resident_bytes = 0;
      throw std::runtime_error(io_budget_failure_message(
          "cannot fit IO buffers in mem_budget", 1, effective_min_depth,
          remaining_after_fixed));
    }

    size_t slots = requested_n_io_slots_ == 0 ? 1 : requested_n_io_slots_;
    // io_preferred_bytes reserves enough budget for io_uring to honor the
    // requested (slots, depth) shape. Cache only consumes whatever remains
    // after that reservation. Using requested_depth (not kMinEffectiveIoQueueDepth)
    // here prevents cache from silently compressing io_queue_depth, which was
    // observed to regress search_time on deep100m by 3x.
    const size_t io_preferred_bytes =
        checked_mul(slots, checked_mul(depth, buf_size_));

    const size_t available_for_cache =
        remaining_after_fixed > io_preferred_bytes
            ? remaining_after_fixed - io_preferred_bytes
            : 0;
    const double raw_cell_share =
        static_cast<double>(cell_cache_share_pct_) / 100.0;
    double effective_cell_share = 0.0;
    if (cell_cache_env_disabled_ || raw_cell_share == 0.0)
    {
      effective_cell_share = 0.0;
    }
    else if (cluster_cache_env_disabled_)
    {
      effective_cell_share = 1.0;
    }
    else
    {
      effective_cell_share = raw_cell_share;
    }

    double effective_cluster_share = 0.0;
    if (cluster_cache_env_disabled_)
    {
      effective_cluster_share = 0.0;
    }
    else if (cell_cache_env_disabled_ || raw_cell_share == 0.0)
    {
      effective_cluster_share = 1.0;
    }
    else
    {
      effective_cluster_share = 1.0 - raw_cell_share;
    }

    const size_t cell_share_bytes =
        static_cast<size_t>(static_cast<double>(available_for_cache) *
                                effective_cell_share +
                            0.5);
    const size_t cluster_share_bytes =
        cell_share_bytes <= available_for_cache
            ? available_for_cache - cell_share_bytes
            : 0;

    size_t cluster_cache_resident_bytes = 0;
    if (cluster_num == 0)
    {
      std::cerr << "[cluster-cache] WARNING: cluster_num == 0, cache disabled\n";
    }
    else if (effective_cluster_share > 0.0)
    {
      size_t candidate_resident = std::min(
          {budget_inputs_.cluster_cache_requested_bytes,
           checked_mul(cluster_num, buf_size_),
           kClusterCacheMaxBytes,
           cluster_share_bytes});
      const size_t candidate_entries =
          buf_size_ == 0 ? 0 : candidate_resident / buf_size_;
      const size_t per_shard_entries =
          candidate_entries / kClusterCacheShardCount;
      const size_t effective_capacity =
          per_shard_entries * kClusterCacheShardCount;
      if (candidate_entries > 0 && effective_capacity == 0)
      {
        std::cerr << "[cluster-cache] WARNING: budget too small for sharded cache "
                  << "(need >= " << kClusterCacheShardCount
                  << " entries, got " << candidate_entries
                  << "); cache disabled\n";
      }
      cluster_cache_resident_bytes = checked_mul(effective_capacity, buf_size_);
    }

    size_t cell_cache_resident_bytes = 0;
    if (effective_cell_share > 0.0)
    {
      cell_cache_resident_bytes =
          std::min({budget_inputs_.cell_cache_requested_bytes,
                    kCellCacheMaxBytes,
                    cell_share_bytes});
      const size_t effective_payload_budget =
          static_cast<size_t>(static_cast<double>(cell_cache_resident_bytes) *
                              (1.0 - kCellCacheIndexOverheadRatio));
      if (cell_cache_resident_bytes > 0 &&
          effective_payload_budget < kCellCacheShardCount * PAGE_SIZE)
      {
        std::cerr << "[cell-cache] WARNING: cell_cache_resident_bytes ("
                  << cell_cache_resident_bytes
                  << " bytes) below minimum (need >= ~1.2 MiB); "
                     "returning to io_pool\n";
        cell_cache_resident_bytes = 0;
      }
    }
    budget_inputs_.cluster_cache_resident_bytes =
        cluster_cache_resident_bytes;
    budget_inputs_.cell_cache_resident_bytes =
        cell_cache_resident_bytes;
    last_io_budget_bytes_ =
        remaining_after_fixed - budget_inputs_.cluster_cache_resident_bytes -
        budget_inputs_.cell_cache_resident_bytes;

    while (multiply_exceeds(slots, depth, buf_size_, last_io_budget_bytes_))
    {
      if (depth / 2 < effective_min_depth)
      {
        break;
      }
      depth /= 2;
    }
    while (multiply_exceeds(slots, depth, buf_size_, last_io_budget_bytes_) &&
           slots > 1)
    {
      slots = (slots + 1) / 2;
    }
    if (multiply_exceeds(slots, depth, buf_size_, last_io_budget_bytes_))
    {
      throw std::runtime_error(io_budget_failure_message(
          "cannot fit IO buffers in mem_budget", slots, depth,
          last_io_budget_bytes_));
    }

    io_queue_depth_ = depth;
    n_io_slots_ = slots;
  }

  void verify_registered_io_budget()
  {
    size_t total_registered_buffers = 0;
    last_actual_ring_entries_ = 0;
    for (const io_uring& ring : rings_)
    {
      const size_t entries = static_cast<size_t>(ring.sq.ring_entries);
      last_actual_ring_entries_ = std::max(last_actual_ring_entries_, entries);
      if (total_registered_buffers >
          std::numeric_limits<size_t>::max() - entries)
      {
        throw std::runtime_error(
            "io_uring registered more buffers than budgeted");
      }
      total_registered_buffers += entries;
    }
    verify_io_buffer_budget(total_registered_buffers, 1, buf_size_,
                            last_io_budget_bytes_);
    last_registered_buffer_count_ = total_registered_buffers;
  }

  void log_mem_budget() const
  {
    std::cerr << "[mem-budget] fixed=" << format_mib(last_fixed_resident_bytes_)
              << " (metadata=" << format_mib(last_metadata_resident_bytes_)
              << " rbq_1bit="
              << format_mib(budget_inputs_.rbq_1bit_resident_bytes)
              << " data_hnsw="
              << format_mib(budget_inputs_.data_hnsw_resident_bytes)
              << " query=" << format_mib(budget_inputs_.query_resident_bytes)
              << " per_thread="
              << format_mib(budget_inputs_.per_thread_resident_bytes)
              << " sqg=" << format_mib(budget_inputs_.sqg_resident_bytes)
              << " rotated_centroids="
              << format_mib(budget_inputs_.rotated_centroids_resident_bytes)
              << " trace=" << format_mib(budget_inputs_.trace_resident_bytes)
              << " hit_bookkeeping="
              << format_mib(budget_inputs_.hit_bookkeeping_resident_bytes)
              << " cluster_cache="
              << format_mib(budget_inputs_.cluster_cache_resident_bytes)
              << " cell_cache="
              << format_mib(budget_inputs_.cell_cache_resident_bytes)
              << " cluster_cache_requested="
              << format_mib(budget_inputs_.cluster_cache_requested_bytes)
              << " cell_cache_requested="
              << format_mib(budget_inputs_.cell_cache_requested_bytes)
              << " slack=" << format_mib(kReservedSlack)
              << "), io_budget=" << format_mib(last_io_budget_bytes_)
              << ", slots=" << n_io_slots_ << ", depth=" << io_queue_depth_
              << ", buf=" << format_mib(buf_size_)
              << ", io_total="
              << format_mib(last_registered_buffer_count_ * buf_size_)
              << ", mem_total="
              << format_mib(last_fixed_resident_bytes_ +
                            budget_inputs_.cluster_cache_resident_bytes +
                            budget_inputs_.cell_cache_resident_bytes +
                            last_registered_buffer_count_ * buf_size_)
              << '\n';
  }

  const float* cell_bound_ptr(size_t cluster_id, size_t cell_idx) const
  {
    if (cluster_id >= cluster_num)
    {
      throw std::runtime_error("cluster id out of range for cell bound: " +
                               std::to_string(cluster_id));
    }
    if (cell_idx >= cell_count_of(cluster_id))
    {
      throw std::runtime_error("cell index out of range for cell bound: " +
                               std::to_string(cell_idx));
    }
    const size_t effective_rank = pca_effective_rank[cluster_id];
    const size_t stride = 2 * effective_rank + 2;
    const size_t offset = cluster_bound_offsets[cluster_id] + cell_idx * stride;
    return cell_bounds.data() + offset;
  }

  void validate_cluster_cell_offsets() const
  {
    if (cluster_cell_offsets.size() != cluster_num + 1)
    {
      throw std::runtime_error("metadata cluster_cell_offsets size mismatch for " +
                               metadatafile_);
    }
    if (cluster_cell_offsets[0] != 0)
    {
      throw std::runtime_error(
          "metadata cluster_cell_offsets must start at zero for " +
          metadatafile_);
    }
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      if (cluster_cell_offsets[cluster_id + 1] <
          cluster_cell_offsets[cluster_id])
      {
        throw std::runtime_error(
            "metadata cluster_cell_offsets not monotonic for " +
            metadatafile_ + " at cluster " + std::to_string(cluster_id));
      }
      if (bucket_sizes[cluster_id] > 0 &&
          cluster_cell_offsets[cluster_id + 1] ==
              cluster_cell_offsets[cluster_id])
      {
        throw std::runtime_error(
            "metadata cluster_cell_offsets gives zero cells for non-empty cluster " +
            std::to_string(cluster_id));
      }
    }
  }

  void validate_cell_vec_offsets() const
  {
    if (cell_vec_offsets.empty() || cell_vec_offsets[0] != 0)
    {
      throw std::runtime_error("metadata cell_vec_offsets must start at zero for " +
                               metadatafile_);
    }
    if (cell_vec_offsets.back() != n)
    {
      throw std::runtime_error("metadata cell_vec_offsets final value mismatch for " +
                               metadatafile_ + ": expected " +
                               std::to_string(n) + ", observed " +
                               std::to_string(cell_vec_offsets.back()));
    }
    for (size_t i = 0; i + 1 < cell_vec_offsets.size(); ++i)
    {
      if (cell_vec_offsets[i + 1] < cell_vec_offsets[i])
      {
        throw std::runtime_error(
            "metadata cell_vec_offsets not monotonic at cell " +
            std::to_string(i));
      }
    }
    uint64_t base = 0;
    for (size_t cluster_id = 0; cluster_id < cluster_num; ++cluster_id)
    {
      const size_t first_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id]);
      const size_t next_cell =
          static_cast<size_t>(cluster_cell_offsets[cluster_id + 1]);
      if (first_cell >= cell_vec_offsets.size() ||
          next_cell >= cell_vec_offsets.size())
      {
        throw std::runtime_error("metadata cluster_cell_offsets references missing cell_vec_offsets at cluster " +
                                 std::to_string(cluster_id));
      }
      if (cell_vec_offsets[first_cell] != base)
      {
        throw std::runtime_error("metadata cell_vec_offsets cluster base mismatch at cluster " +
                                 std::to_string(cluster_id));
      }
      base += static_cast<uint64_t>(bucket_sizes[cluster_id]);
      if (cell_vec_offsets[next_cell] != base)
      {
        throw std::runtime_error("metadata cell_vec_offsets cluster end mismatch at cluster " +
                                 std::to_string(cluster_id));
      }
    }
  }

  void init_io_slots()
  {
    rings_.resize(n_io_slots_);
    ring_initialized_.assign(n_io_slots_, false);
    poisoned_slots_.assign(n_io_slots_, false);
    slots_.resize(n_io_slots_);
    slot_locks_ = std::vector<std::mutex>(n_io_slots_);
    last_actual_ring_entries_ = 0;
    last_registered_buffer_count_ = 0;

    for (size_t slot_id = 0; slot_id < n_io_slots_; ++slot_id)
    {
      int ret = io_uring_queue_init(io_queue_depth_, &rings_[slot_id], 0);
      if (ret < 0)
      {
        throw std::runtime_error("io_uring_queue_init failed: " +
                                 std::to_string(ret) + " (" +
                                 strerror(-ret) + ")");
      }
      ring_initialized_[slot_id] = true;

      const size_t registered_buffer_count = rings_[slot_id].sq.ring_entries;
      last_actual_ring_entries_ =
          std::max(last_actual_ring_entries_, registered_buffer_count);
      verify_io_buffer_budget(n_io_slots_, registered_buffer_count, buf_size_,
                              last_io_budget_bytes_);
      if (last_registered_buffer_count_ >
          std::numeric_limits<size_t>::max() - registered_buffer_count)
      {
        throw std::runtime_error(
            "io_uring registered more buffers than budgeted");
      }
      const size_t projected_registered_buffers =
          last_registered_buffer_count_ + registered_buffer_count;
      verify_io_buffer_budget(projected_registered_buffers, 1, buf_size_,
                              last_io_budget_bytes_);

      IoSlot& slot = slots_[slot_id];
      slot.buffers.resize(registered_buffer_count);
      slot.iovs.resize(registered_buffer_count);
      for (size_t i = 0; i < registered_buffer_count; ++i)
      {
        slot.buffers[i].data = aligned_alloc(PAGE_SIZE, buf_size_);
        if (slot.buffers[i].data == nullptr)
        {
          throw std::runtime_error("aligned_alloc failed for io buffer");
        }
        slot.iovs[i].iov_base = slot.buffers[i].data;
        slot.iovs[i].iov_len = buf_size_;
      }

      ret = io_uring_register_buffers(&rings_[slot_id], slot.iovs.data(),
                                      static_cast<unsigned>(slot.iovs.size()));
      if (ret < 0)
      {
        throw std::runtime_error("io_uring_register_buffers failed: " +
                                 std::to_string(ret) + " (" +
                                 strerror(-ret) + ")");
      }
      last_registered_buffer_count_ = projected_registered_buffers;
    }
  }
};
