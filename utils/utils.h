#pragma once

#include <fcntl.h>
#include <linux/mman.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <type_traits>

#include "../lib/ConfigLoader.h"

typedef std::priority_queue<std::pair<float, unsigned>> candidate_pool;

struct Timer
{
  std::chrono::_V2::system_clock::time_point s;
  std::chrono::_V2::system_clock::time_point e;
  std::chrono::duration<double> diff;

  void tick()
  {
    s = std::chrono::high_resolution_clock::now();
  }

  void tuck(std::string message, bool print = true)
  {
    e = std::chrono::high_resolution_clock::now();
    diff = e - s;
    if (print)
    {
      std::cout << "[" << diff.count() << " s] " << message << std::endl;
    }
  }
};

template <class T>
T* read_fbin(const char* filename, unsigned& n, unsigned& d)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error(std::string("failed to open .fbin file: ") +
                             filename);
  }
  // Caller's template type T directly determines the expected dtype. We use it
  // as the v1-fallback hint so that read_fbin<int8_t> can load a legacy v1
  // fbin whose payload is int8.
  uint32_t expected_dtype = range_search_config::kVecDTypeFloat32;
  if constexpr (std::is_same<T, float>::value)
  {
    expected_dtype = range_search_config::kVecDTypeFloat32;
  }
  else if constexpr (std::is_same<T, int8_t>::value)
  {
    expected_dtype = range_search_config::kVecDTypeInt8;
  }
  else if constexpr (std::is_same<T, uint8_t>::value)
  {
    expected_dtype = range_search_config::kVecDTypeUint8Reserved;
  }
  else
  {
    throw std::runtime_error(
        "read_fbin<T> supports only float, int8_t, and uint8_t");
  }
  const range_search_config::FbinHeader header =
      range_search_config::read_fbin_header(filename, expected_dtype);
  if (header.dtype != expected_dtype)
  {
    throw std::runtime_error(
        "read_fbin dtype mismatch for " + std::string(filename) +
        ": requested " + range_search_config::vec_dtype_to_string(expected_dtype) +
        ", file is " + range_search_config::vec_dtype_to_string(header.dtype));
  }
  n = static_cast<unsigned>(header.n);
  d = static_cast<unsigned>(header.dim);
  auto data = new T[(size_t)n * (size_t)d];
  in.seekg(static_cast<std::streamoff>(header.payload_offset), std::ios::beg);
  in.read((char*)data, (size_t)n * (size_t)d * sizeof(T));
  if (!in)
  {
    delete[] data;
    throw std::runtime_error(std::string("failed to read .fbin payload: ") +
                             filename);
  }
  in.close();
  return data;
}

void read_fvecs(char* filename, float*& data, unsigned& n, unsigned& d)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open())
  {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  float num, dim;
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (float)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];
  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++)
  {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * (size_t)dim), dim * 4);
  }
  in.close();
  n = (unsigned)num;
  d = (unsigned)dim;
  std::cout << "data num: " << n << ", dimension:" << d << std::endl;
}

size_t div_round_up(size_t x, size_t y)
{
  return (x / y) + static_cast<size_t>((x % y) != 0);
}
