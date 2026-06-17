#include <cstdint>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ClusterIO.h"
#include "../lib/ConfigLoader.h"

namespace fs = std::filesystem;

namespace
{
void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

void write_u32(std::ofstream& out, uint32_t value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  require(static_cast<bool>(out), "failed to write u32");
}

fs::path make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "int8_cluster_layout_test";
  fs::remove_all(base);
  fs::create_directories(base);
  return base;
}

fs::path write_v2_int8_fbin(const fs::path& dir,
                            const std::vector<int8_t>& values, uint32_t n,
                            uint32_t dim)
{
  const fs::path path = dir / "base_i8.fbin";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create int8 fbin");
  write_u32(out, range_search_config::kFbinMagicDjb1);
  write_u32(out, range_search_config::kVecDTypeInt8);
  write_u32(out, n);
  write_u32(out, dim);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int8_t)));
  require(static_cast<bool>(out), "failed to write int8 fbin payload");
  return path;
}

fs::path write_v2_uint8_fbin(const fs::path& dir,
                             const std::vector<uint8_t>& values, uint32_t n,
                             uint32_t dim)
{
  const fs::path path = dir / "base_u8.fbin";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(out), "failed to create uint8 fbin");
  write_u32(out, range_search_config::kFbinMagicDjb1);
  write_u32(out, range_search_config::kVecDTypeUint8Reserved);
  write_u32(out, n);
  write_u32(out, dim);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(uint8_t)));
  require(static_cast<bool>(out), "failed to write uint8 fbin payload");
  return path;
}

template <typename T>
std::vector<T> read_cluster_prefix(const fs::path& path, size_t count)
{
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "failed to open cluster file");
  std::vector<T> values(count);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(T)));
  require(in.gcount() == static_cast<std::streamsize>(values.size()),
          "cluster payload shorter than expected");
  return values;
}

template <typename T>
T read_value(std::ifstream& in, const std::string& label)
{
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  require(static_cast<bool>(in), "failed to read " + label);
  return value;
}

void verify_metadata_header_and_tail(const fs::path& path,
                                     const std::vector<int32_t>& expected_sum,
                                     const std::vector<int32_t>& expected_norm,
                                     uint32_t expected_dtype)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(in), "failed to open metadata");
  const std::streamoff end = in.tellg();
  require(end > 0, "metadata file is empty");
  in.seekg(0, std::ios::beg);

  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  require(magic == kMetadataMagic, "metadata magic mismatch");
  const uint64_t schema = read_value<uint64_t>(in, "schema");
  require(schema == kMetadataSchemaVersion, "metadata schema mismatch");
  require(read_value<size_t>(in, "n") == expected_sum.size(),
          "metadata n mismatch");
  require(read_value<size_t>(in, "d") == 4, "metadata d mismatch");
  require(read_value<size_t>(in, "cluster_num") == 2,
          "metadata cluster_num mismatch");
  (void)read_value<size_t>(in, "max_points");
  (void)read_value<uint64_t>(in, "pca_rank");
  (void)read_value<uint64_t>(in, "target_cell_vecs");
  const uint64_t vec_dtype = read_value<uint64_t>(in, "vec_dtype");
  require(vec_dtype == expected_dtype, "metadata vec_dtype mismatch");
  require(static_cast<size_t>(in.tellg()) == 72,
          "metadata header must be 72 bytes");

  const size_t n = expected_sum.size();
  const std::streamoff tail_bytes =
      static_cast<std::streamoff>(2 * n * sizeof(int32_t));
  require(end >= tail_bytes, "metadata too small for int8 tail");
  in.seekg(end - tail_bytes, std::ios::beg);
  std::vector<int32_t> sums(n);
  std::vector<int32_t> norms(n);
  in.read(reinterpret_cast<char*>(sums.data()),
          static_cast<std::streamsize>(n * sizeof(int32_t)));
  in.read(reinterpret_cast<char*>(norms.data()),
          static_cast<std::streamsize>(n * sizeof(int32_t)));
  require(static_cast<bool>(in), "failed to read int8 metadata tail");
  require(sums == expected_sum, "metadata sum_for_offset tail mismatch");
  require(norms == expected_norm, "metadata norm_sq_for_l2 tail mismatch");
}
}  // namespace

int main()
{
  try
  {
    const fs::path dir = make_temp_dir();
    {
    const std::vector<int8_t> rows = {
        -4, -3, -2, -1,  // row 0
        1,  2,  3,  4,   // row 1
        10, 11, 12, 13,  // row 2
        -8, 7,  -6, 5,   // row 3
    };
    const fs::path base = write_v2_int8_fbin(dir, rows, 4, 4);
    const fs::path cluster = dir / "cluster.bin";
    const fs::path metadata = dir / "metadata.bin";

    std::vector<std::vector<size_t>> assignment = {{0, 2}, {1, 3}};
    std::vector<float> centroids = {
        3.0f, 4.0f, 5.0f, 6.0f,
        -3.5f, 4.5f, -1.5f, 4.5f,
    };

    ClusterWriter writer(base.string(), cluster.string(), metadata.string(),
                         0, 0, 2, false);
    writer.writeClusters(assignment, centroids.data(), 0.001f);
    writer.writeMetadata(assignment);

    const std::vector<int8_t> expected_payload = {
        -4, -3, -2, -1, 10, 11, 12, 13,
        1,  2,  3,  4,  -8, 7,  -6, 5,
    };
    require(read_cluster_prefix<int8_t>(cluster, expected_payload.size()) ==
                expected_payload,
            "int8 cluster payload must remain raw int8 bytes");

    verify_metadata_header_and_tail(metadata, {-10, 46, 10, -2},
                                    {30, 534, 30, 174},
                                    range_search_config::kVecDTypeInt8);
    {
      ClusterReader reader(cluster.string(), metadata.string(), 1, 1);
      reader.readMetaData();
    }

    const fs::path metadata_extra = dir / "metadata_extra.bin";
    fs::copy_file(metadata, metadata_extra,
                  fs::copy_options::overwrite_existing);
    {
      std::ofstream out(metadata_extra, std::ios::binary | std::ios::app);
      require(static_cast<bool>(out), "failed to append metadata extra byte");
      const char extra = '\0';
      out.write(&extra, 1);
      require(static_cast<bool>(out), "failed to write metadata extra byte");
    }
    bool rejected_extra = false;
    try
    {
      ClusterReader reader(cluster.string(), metadata_extra.string(), 1, 1);
      reader.readMetaData();
    }
    catch (const std::exception& e)
    {
      rejected_extra =
          std::string(e.what()).find("metadata file size mismatch") !=
          std::string::npos;
    }
    require(rejected_extra, "metadata with trailing bytes must be rejected");
    }

    {
    const std::vector<uint8_t> rows = {
        0,   1,   2,   255,  // row 0
        200, 201, 3,   4,    // row 1
        10,  11,  12,  13,   // row 2
        128, 127, 126, 125,  // row 3
    };
    const fs::path base = write_v2_uint8_fbin(dir, rows, 4, 4);
    const fs::path cluster = dir / "cluster_u8.bin";
    const fs::path metadata = dir / "metadata_u8.bin";

    std::vector<std::vector<size_t>> assignment = {{0, 2}, {1, 3}};
    std::vector<float> centroids = {
        5.0f, 6.0f, 7.0f, 128.0f,
        164.0f, 164.0f, 64.5f, 64.5f,
    };

    ClusterWriter writer(base.string(), cluster.string(), metadata.string(),
                         0, 0, 2, false,
                         range_search_config::kVecDTypeUint8Reserved);
    writer.writeClusters(assignment, centroids.data(), 0.001f);
    writer.writeMetadata(assignment);

    const std::vector<uint8_t> expected_payload = {
        0,   1,   2,   255, 10,  11,  12,  13,
        200, 201, 3,   4,   128, 127, 126, 125,
    };
    require(read_cluster_prefix<uint8_t>(cluster, expected_payload.size()) ==
                expected_payload,
            "uint8 cluster payload must remain raw uint8 bytes");

    verify_metadata_header_and_tail(
        metadata, {258, 46, 408, 506},
        {65030, 534, 80426, 64014},
        range_search_config::kVecDTypeUint8Reserved);
    {
      ClusterReader reader(cluster.string(), metadata.string(), 1, 1);
      reader.readMetaData();
    }

    const fs::path metadata_truncated = dir / "metadata_u8_truncated.bin";
    fs::copy_file(metadata, metadata_truncated,
                  fs::copy_options::overwrite_existing);
    fs::resize_file(metadata_truncated, fs::file_size(metadata_truncated) - 1);
    bool rejected_truncated = false;
    try
    {
      ClusterReader reader(cluster.string(), metadata_truncated.string(), 1, 1);
      reader.readMetaData();
    }
    catch (const std::exception& e)
    {
      const std::string message = e.what();
      rejected_truncated =
          message.find("metadata file size mismatch") != std::string::npos ||
          message.find("sum/norm metadata tail") != std::string::npos;
    }
    require(rejected_truncated,
            "uint8 metadata with truncated tail must be rejected");
    }

    fs::remove_all(dir);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
