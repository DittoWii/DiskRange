#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr uint32_t kFbinMagicDjb1 = 0x31424A44u;
constexpr uint32_t kVecDTypeInt8 = 1u;
constexpr size_t kI8binHeaderBytes = 8;
constexpr size_t kCopyBufferBytes = 1 << 20;

void write_u32(std::ofstream& out, uint32_t value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out)
  {
    throw std::runtime_error("failed to write output header");
  }
}

uint32_t read_u32(std::ifstream& in, const std::string& label)
{
  uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in)
  {
    throw std::runtime_error("failed to read " + label);
  }
  return value;
}

void wrap_i8bin(const std::string& input_path, const std::string& output_path)
{
  std::ifstream in(input_path, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("input file not found: " + input_path);
  }
  const uint32_t n = read_u32(in, "n");
  const uint32_t dim = read_u32(in, "dim");
  if (n == 0 || dim == 0)
  {
    throw std::runtime_error("input i8bin n/dim must be positive");
  }

  const uintmax_t input_size = std::filesystem::file_size(input_path);
  const uintmax_t expected_payload = static_cast<uintmax_t>(n) * dim;
  const uintmax_t actual_payload =
      input_size >= kI8binHeaderBytes ? input_size - kI8binHeaderBytes : 0;
  if (actual_payload != expected_payload)
  {
    throw std::runtime_error("expected payload bytes: " +
                             std::to_string(expected_payload) + ", actual: " +
                             std::to_string(actual_payload));
  }

  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out)
  {
    throw std::runtime_error("cannot open output file for writing: " +
                             output_path);
  }
  write_u32(out, kFbinMagicDjb1);
  write_u32(out, kVecDTypeInt8);
  write_u32(out, n);
  write_u32(out, dim);

  std::vector<char> buffer(kCopyBufferBytes);
  while (in)
  {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = in.gcount();
    if (got > 0)
    {
      out.write(buffer.data(), got);
      if (!out)
      {
        throw std::runtime_error("failed while writing output payload");
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: fbin_dtype_wrap <input.i8bin> <output.fbin>\n";
    return 2;
  }
  try
  {
    wrap_i8bin(argv[1], argv[2]);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
