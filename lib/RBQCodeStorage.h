#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/space.hpp"
#include "utils/memory.hpp"
#include "utils/rotator.hpp"

#include "../utils/vector_cast.h"
#include "ConfigLoader.h"

inline constexpr std::array<char, 8> kRBQCodeMagic = {'D', 'J',  'R',  'B',
                                                      'Q', '\0', '\0', '\0'};
inline constexpr uint64_t kRBQCodeVersion = 2u;
static_assert(std::numeric_limits<float>::is_iec559,
              "IEEE-754 required for r^2==0 exact compare");

struct RBQCodeFileHeader
{
  std::array<char, 8> magic{};
  uint64_t version = kRBQCodeVersion;
  uint64_t cluster_count = 0;
  uint64_t n_data = 0;
  uint64_t dim = 0;
  uint64_t padded_dim = 0;
  uint64_t ex_bits = 0;
  uint64_t code_bytes_per_vec = 0;
};

struct RBQClusterDirectoryEntry
{
  uint64_t vec_count = 0;
  uint64_t code_offset_bytes = 0;
};

// RBQCodeStorage stores only per-cluster code bytes; there is deliberately no
// vec-id array. `(cluster_id, local_index)` maps to global vec_id through the
// existing metadata assignment table. Marker: no vec-id array.
class RBQCodeStorage
{
 public:
  RBQCodeStorage() = default;
  RBQCodeStorage(const RBQCodeStorage&) = delete;
  RBQCodeStorage& operator=(const RBQCodeStorage&) = delete;

  RBQCodeStorage(RBQCodeStorage&& other) noexcept
  {
    move_from(std::move(other));
  }

  RBQCodeStorage& operator=(RBQCodeStorage&& other) noexcept
  {
    if (this != &other)
    {
      close_fd();
      move_from(std::move(other));
    }
    return *this;
  }

  ~RBQCodeStorage()
  {
    close_fd();
  }

  void build_from_cluster_file(
      const std::string& cluster_path,
      const std::vector<std::vector<size_t>>& assignment,
      const std::vector<float>& centroids, size_t dim,
      const symqg::FHTRotator& rotator, size_t ex_bits,
      uint32_t vec_dtype = range_search_config::kVecDTypeFloat32)
  {
    if (dim == 0)
    {
      throw std::runtime_error("RBQ dim must be positive");
    }
    if (ex_bits != 0 && ex_bits != 7)
    {
      throw std::runtime_error("RBQ ex_bits must be 0 or 7");
    }
    const size_t cluster_count = assignment.size();
    if (centroids.size() != cluster_count * dim)
    {
      throw std::runtime_error("RBQ centroid payload size mismatch");
    }

    const size_t padded_dim = size_t{1} << symqg::ceil_log2(dim);
    const size_t code_bytes_per_vec =
        ex_bits == 0
            ? rabitqlib::BatchDataMap<float>::data_bytes(padded_dim) /
                  rabitqlib::fastscan::kBatchSize
            : rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);

    header_ = {};
    header_.magic = kRBQCodeMagic;
    header_.version = kRBQCodeVersion;
    header_.cluster_count = static_cast<uint64_t>(cluster_count);
    header_.dim = static_cast<uint64_t>(dim);
    header_.padded_dim = static_cast<uint64_t>(padded_dim);
    header_.ex_bits = static_cast<uint64_t>(ex_bits);
    header_.code_bytes_per_vec = static_cast<uint64_t>(code_bytes_per_vec);
    directory_.assign(cluster_count, {});

    uint64_t offset = 0;
    for (size_t cid = 0; cid < cluster_count; ++cid)
    {
      directory_[cid].vec_count = static_cast<uint64_t>(assignment[cid].size());
      directory_[cid].code_offset_bytes = offset;
      const size_t vec_count = assignment[cid].size();
      offset += static_cast<uint64_t>(
          ex_bits == 0 ? one_bit_cluster_bytes(vec_count, padded_dim)
                       : vec_count * code_bytes_per_vec);
      header_.n_data += static_cast<uint64_t>(assignment[cid].size());
    }
    code_bytes_.assign(static_cast<size_t>(offset), 0);

    std::ifstream in(cluster_path, std::ios::binary | std::ios::in);
    if (!in)
    {
      throw std::runtime_error("failed to open cluster file for RBQ build: " +
                               cluster_path);
    }

    using AlignedFloatVector =
        std::vector<float, symqg::memory::AlignedAllocator<float>>;
    AlignedFloatVector rotated_centroid(padded_dim);
    AlignedFloatVector rotated_data;
    std::vector<float> raw_cluster;
    std::vector<uint8_t> raw_cluster_bytes;
    std::vector<char> batch_tmp(
        rabitqlib::BatchDataMap<float>::data_bytes(padded_dim));
    std::vector<char> ex_tmp(
        rabitqlib::fastscan::kBatchSize *
        rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits));
    const rabitqlib::quant::RabitqConfig config =
        rabitqlib::quant::faster_config(padded_dim, ex_bits + 1);

    for (size_t cid = 0; cid < cluster_count; ++cid)
    {
      const size_t vec_count = assignment[cid].size();
      raw_cluster.assign(vec_count * dim, 0.0f);
      if (!raw_cluster.empty())
      {
        if (range_search_config::is_byte_dtype(vec_dtype))
        {
          raw_cluster_bytes.assign(vec_count * dim, 0);
          in.read(reinterpret_cast<char*>(raw_cluster_bytes.data()),
                  static_cast<std::streamsize>(raw_cluster_bytes.size()));
          if (!in)
          {
            throw std::runtime_error(
                "short byte-dtype cluster file during RBQ build at cluster " +
                std::to_string(cid));
          }
          utils::cast_byte_payload_to_float(
              vec_dtype, raw_cluster_bytes.data(), raw_cluster_bytes.size(),
              raw_cluster.data());
        }
        else
        {
          in.read(
              reinterpret_cast<char*>(raw_cluster.data()),
              static_cast<std::streamsize>(raw_cluster.size() * sizeof(float)));
          if (!in)
          {
            throw std::runtime_error(
                "short cluster file during RBQ build at cluster " +
                std::to_string(cid));
          }
        }
      }
      rotator.rotate(centroids.data() + cid * dim, rotated_centroid.data());
      rotated_data.assign(vec_count * padded_dim, 0.0f);
      for (size_t local = 0; local < vec_count; ++local)
      {
        rotator.rotate(raw_cluster.data() + local * dim,
                       rotated_data.data() + local * padded_dim);
      }
      for (size_t start = 0; start < vec_count;
           start += rabitqlib::fastscan::kBatchSize)
      {
        const size_t n =
            std::min(rabitqlib::fastscan::kBatchSize, vec_count - start);
        if (ex_bits == 0)
        {
          rabitqlib::quant::quantize_split_batch(
              rotated_data.data() + start * padded_dim, rotated_centroid.data(),
              n, padded_dim, ex_bits, mutable_batch_code_ptr(cid, start),
              nullptr, rabitqlib::METRIC_L2, config);
          // BatchDataMap exposes mutable float* factor accessors.
          rabitqlib::BatchDataMap<float> batch(
              mutable_batch_code_ptr(cid, start), padded_dim);
          const float* centroid = centroids.data() + cid * dim;
          for (size_t j = 0; j < n; ++j)
          {
            const size_t local = start + j;
            const float* vec = raw_cluster.data() + local * dim;
            if (compute_residual_sq(vec, centroid, dim) == 0.0f)
            {
              batch.f_add()[j] = 0.0f;
              batch.f_rescale()[j] = 0.0f;
              batch.f_error()[j] = 0.0f;
            }
          }
        }
        else
        {
          rabitqlib::quant::quantize_split_batch(
              rotated_data.data() + start * padded_dim, rotated_centroid.data(),
              n, padded_dim, ex_bits, batch_tmp.data(), ex_tmp.data(),
              rabitqlib::METRIC_L2, config);
          std::memcpy(mutable_code_ptr(cid, start), ex_tmp.data(),
                      n * code_bytes_per_vec);
          const float* centroid = centroids.data() + cid * dim;
          for (size_t j = 0; j < n; ++j)
          {
            const size_t local = start + j;
            const float* vec = raw_cluster.data() + local * dim;
            if (compute_residual_sq(vec, centroid, dim) == 0.0f)
            {
              // ExDataMap exposes mutable float& factor accessors.
              rabitqlib::ExDataMap<float> ex(mutable_code_ptr(cid, local),
                                             padded_dim, ex_bits);
              ex.f_add_ex() = 0.0f;
              ex.f_rescale_ex() = 0.0f;
            }
          }
        }
      }
    }
  }

  void save(const std::string& path) const
  {
    const std::filesystem::path out_path(path);
    const std::filesystem::path parent = out_path.parent_path();
    if (!parent.empty())
    {
      std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out)
    {
      throw std::runtime_error("failed to open RBQ code file for write: " +
                               path);
    }
    out.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
    if (!directory_.empty())
    {
      out.write(reinterpret_cast<const char*>(directory_.data()),
                static_cast<std::streamsize>(directory_.size() *
                                             sizeof(directory_.front())));
    }
    if (!code_bytes_.empty())
    {
      out.write(code_bytes_.data(),
                static_cast<std::streamsize>(code_bytes_.size()));
    }
    if (!out)
    {
      throw std::runtime_error("failed to write RBQ code file: " + path);
    }
  }

  void load_1bit(const std::string& path)
  {
    read_header_and_directory(path);
    if (header_.ex_bits != 0)
    {
      throw std::runtime_error("expected 1-bit RBQ code file: " + path);
    }
    std::ifstream in(path, std::ios::binary | std::ios::in);
    if (!in)
    {
      throw std::runtime_error("failed to open RBQ code file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(payload_offset()), std::ios::beg);
    code_bytes_.assign(total_code_bytes(), 0);
    if (!code_bytes_.empty())
    {
      in.read(code_bytes_.data(),
              static_cast<std::streamsize>(code_bytes_.size()));
      if (!in)
      {
        throw std::runtime_error("failed to read RBQ code payload: " + path);
      }
    }
  }

  // option C upper-bound probe: load the full 8-bit payload into RAM so
  // read_code() takes the memcpy branch (no pread, no page-cache contention).
  // Mirrors load_1bit but for ex_bits==7. Caller opts in via DJ_PHASE3_INMEM.
  void load_8bit(const std::string& path)
  {
    read_header_and_directory(path);
    if (header_.ex_bits != 7)
    {
      throw std::runtime_error("expected 8-bit RBQ code file: " + path);
    }
    std::ifstream in(path, std::ios::binary | std::ios::in);
    if (!in)
    {
      throw std::runtime_error("failed to open RBQ code file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(payload_offset()), std::ios::beg);
    code_bytes_.assign(total_code_bytes(), 0);
    if (!code_bytes_.empty())
    {
      in.read(code_bytes_.data(),
              static_cast<std::streamsize>(code_bytes_.size()));
      if (!in)
      {
        throw std::runtime_error("failed to read RBQ code payload: " + path);
      }
    }
  }

  void open_8bit_for_pread(const std::string& path)
  {
    read_header_and_directory(path);
    if (header_.ex_bits != 7)
    {
      throw std::runtime_error("expected 8-bit RBQ code file: " + path);
    }
    close_fd();
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
    {
      throw std::runtime_error("failed to open RBQ 8-bit code file: " + path +
                               ": " + std::strerror(errno));
    }
    path_ = path;
    code_bytes_.clear();
  }

  std::vector<char> read_code(size_t cluster_id, size_t local_index) const
  {
    if (header_.ex_bits == 0)
    {
      throw std::runtime_error(
          "read_code is not supported for 1-bit FastScan batch storage");
    }
    const size_t bytes = code_bytes_per_vec();
    std::vector<char> out(bytes);
    if (!code_bytes_.empty())
    {
      const char* src = code_ptr(cluster_id, local_index);
      std::memcpy(out.data(), src, bytes);
      return out;
    }
    if (fd_ < 0)
    {
      throw std::runtime_error("RBQ storage has no loaded payload or pread fd");
    }
    const off_t off =
        static_cast<off_t>(absolute_code_offset(cluster_id, local_index));
    const ssize_t got = pread(fd_, out.data(), bytes, off);
    if (got != static_cast<ssize_t>(bytes))
    {
      throw std::runtime_error("failed to pread RBQ code at " +
                               std::to_string(off) + " from " + path_);
    }
    return out;
  }

  const char* code_ptr(size_t cluster_id, size_t local_index) const
  {
    if (header_.ex_bits == 0)
    {
      throw std::runtime_error(
          "single-vector code_ptr is not available for 1-bit FastScan batch "
          "storage");
    }
    if (code_bytes_.empty())
    {
      throw std::runtime_error("RBQ code payload is not loaded into memory");
    }
    return code_bytes_.data() + relative_code_offset(cluster_id, local_index);
  }

  size_t cluster_count() const
  {
    return static_cast<size_t>(header_.cluster_count);
  }

  size_t n_data() const
  {
    return static_cast<size_t>(header_.n_data);
  }
  size_t dim() const
  {
    return static_cast<size_t>(header_.dim);
  }
  size_t padded_dim() const
  {
    return static_cast<size_t>(header_.padded_dim);
  }
  size_t ex_bits() const
  {
    return static_cast<size_t>(header_.ex_bits);
  }

  size_t code_bytes_per_vec() const
  {
    return static_cast<size_t>(header_.code_bytes_per_vec);
  }

  size_t cluster_vec_count(size_t cluster_id) const
  {
    check_cluster(cluster_id);
    return static_cast<size_t>(directory_[cluster_id].vec_count);
  }

  size_t cluster_block_count(size_t cluster_id) const
  {
    return div_round_up(cluster_vec_count(cluster_id),
                        rabitqlib::fastscan::kBatchSize);
  }

  const char* batch_code_ptr(size_t cluster_id, size_t block_index) const
  {
    if (header_.ex_bits != 0)
    {
      throw std::runtime_error("batch_code_ptr requires a 1-bit RBQ code file");
    }
    if (code_bytes_.empty())
    {
      throw std::runtime_error(
          "RBQ 1-bit batch payload is not loaded into memory");
    }
    if (block_index >= cluster_block_count(cluster_id))
    {
      throw std::runtime_error("RBQ 1-bit block_index out of range: " +
                               std::to_string(block_index));
    }
    return code_bytes_.data() +
           static_cast<size_t>(directory_[cluster_id].code_offset_bytes) +
           block_index * batch_code_bytes_per_block();
  }

  float batch_f_error(size_t cluster_id, size_t local_index) const
  {
    check_local_index(cluster_id, local_index);
    rabitqlib::ConstBatchDataMap<float> batch(
        batch_code_ptr(cluster_id,
                       local_index / rabitqlib::fastscan::kBatchSize),
        padded_dim());
    return batch.f_error()[local_index % rabitqlib::fastscan::kBatchSize];
  }

  size_t total_code_entries() const
  {
    size_t total = 0;
    for (const auto& entry : directory_)
    {
      total += static_cast<size_t>(entry.vec_count);
    }
    return total;
  }

  void validate_bucket_sizes(const std::vector<size_t>& expected,
                             const std::string& label) const
  {
    if (expected.size() != directory_.size())
    {
      throw std::runtime_error(label +
                               " RBQ cluster_count mismatch with metadata");
    }
    size_t expected_total = 0;
    for (size_t cid = 0; cid < expected.size(); ++cid)
    {
      expected_total += expected[cid];
      if (static_cast<uint64_t>(expected[cid]) != directory_[cid].vec_count)
      {
        throw std::runtime_error(
            label + " RBQ vec_count mismatch with metadata at cluster " +
            std::to_string(cid));
      }
    }
    if (expected_total != n_data())
    {
      throw std::runtime_error(label + " RBQ n_data mismatch with metadata");
    }
  }

 private:
  void read_header_and_directory(const std::string& path)
  {
    std::ifstream in(path, std::ios::binary | std::ios::in);
    if (!in)
    {
      throw std::runtime_error("failed to open RBQ code file: " + path);
    }
    in.read(reinterpret_cast<char*>(&header_), sizeof(header_));
    if (!in || header_.magic != kRBQCodeMagic ||
        header_.version != kRBQCodeVersion)
    {
      throw std::runtime_error("invalid RBQ code file header: " + path);
    }
    directory_.assign(static_cast<size_t>(header_.cluster_count), {});
    if (!directory_.empty())
    {
      in.read(reinterpret_cast<char*>(directory_.data()),
              static_cast<std::streamsize>(directory_.size() *
                                           sizeof(directory_.front())));
      if (!in)
      {
        throw std::runtime_error("failed to read RBQ code directory: " + path);
      }
    }
  }

  size_t payload_offset() const
  {
    return sizeof(RBQCodeFileHeader) +
           directory_.size() * sizeof(RBQClusterDirectoryEntry);
  }

  size_t total_code_bytes() const
  {
    size_t total = 0;
    for (const auto& entry : directory_)
    {
      total += cluster_payload_bytes(entry);
    }
    return total;
  }

  void check_cluster(size_t cluster_id) const
  {
    if (cluster_id >= directory_.size())
    {
      throw std::runtime_error("RBQ cluster_id out of range: " +
                               std::to_string(cluster_id));
    }
  }

  size_t relative_code_offset(size_t cluster_id, size_t local_index) const
  {
    check_cluster(cluster_id);
    if (header_.ex_bits == 0)
    {
      throw std::runtime_error(
          "single-vector offsets are not available for 1-bit FastScan batch "
          "storage");
    }
    check_local_index(cluster_id, local_index);
    return static_cast<size_t>(directory_[cluster_id].code_offset_bytes) +
           local_index * code_bytes_per_vec();
  }

  size_t absolute_code_offset(size_t cluster_id, size_t local_index) const
  {
    return payload_offset() + relative_code_offset(cluster_id, local_index);
  }

  char* mutable_code_ptr(size_t cluster_id, size_t local_index)
  {
    return code_bytes_.data() + relative_code_offset(cluster_id, local_index);
  }

  char* mutable_batch_code_ptr(size_t cluster_id, size_t local_start)
  {
    if (header_.ex_bits != 0)
    {
      throw std::runtime_error(
          "mutable_batch_code_ptr requires a 1-bit RBQ code file");
    }
    check_cluster(cluster_id);
    if (local_start >= directory_[cluster_id].vec_count)
    {
      throw std::runtime_error("RBQ local_start out of range: " +
                               std::to_string(local_start));
    }
    if (local_start % rabitqlib::fastscan::kBatchSize != 0)
    {
      throw std::runtime_error("RBQ local_start must be batch aligned");
    }
    return code_bytes_.data() +
           static_cast<size_t>(directory_[cluster_id].code_offset_bytes) +
           (local_start / rabitqlib::fastscan::kBatchSize) *
               batch_code_bytes_per_block();
  }

  void check_local_index(size_t cluster_id, size_t local_index) const
  {
    check_cluster(cluster_id);
    if (local_index >= directory_[cluster_id].vec_count)
    {
      throw std::runtime_error("RBQ local_index out of range: " +
                               std::to_string(local_index));
    }
  }

  static size_t div_round_up(size_t n, size_t denom)
  {
    return (n + denom - 1) / denom;
  }

  static float compute_residual_sq(const float* vec, const float* centroid,
                                   size_t dim)
  {
    float r_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
      const float diff = vec[i] - centroid[i];
      r_sq += diff * diff;
    }
    return r_sq;
  }

  static size_t one_bit_cluster_bytes(size_t vec_count, size_t padded_dim)
  {
    return div_round_up(vec_count, rabitqlib::fastscan::kBatchSize) *
           rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  }

  size_t batch_code_bytes_per_block() const
  {
    return rabitqlib::BatchDataMap<float>::data_bytes(padded_dim());
  }

  size_t cluster_payload_bytes(const RBQClusterDirectoryEntry& entry) const
  {
    return header_.ex_bits == 0
               ? one_bit_cluster_bytes(static_cast<size_t>(entry.vec_count),
                                       padded_dim())
               : static_cast<size_t>(entry.vec_count) * code_bytes_per_vec();
  }

  void close_fd()
  {
    if (fd_ >= 0)
    {
      close(fd_);
      fd_ = -1;
    }
  }

  void move_from(RBQCodeStorage&& other)
  {
    header_ = other.header_;
    directory_ = std::move(other.directory_);
    code_bytes_ = std::move(other.code_bytes_);
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    other.fd_ = -1;
  }

  RBQCodeFileHeader header_{};
  std::vector<RBQClusterDirectoryEntry> directory_;
  std::vector<char> code_bytes_;
  std::string path_;
  int fd_ = -1;
};
