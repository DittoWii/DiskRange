#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ConfigLoader.h"

struct DataReader
{
  size_t n;
  size_t d;
  uint32_t dtype = 0;
  size_t payload_offset = 8;
  size_t bytes_per_element = sizeof(float);
  std::ifstream in;

  DataReader()
  {
  }

  // `expected_v1_dtype` is the caller-resolved authoritative dtype (usually
  // resolved.vec_dtype from GT JSON). It is passed straight to
  // read_fbin_header, which uses it as the v1-fallback hint when the file has
  // no v2 magic. v2-magic files still self-describe; the helper rejects when
  // expected_v1_dtype disagrees with a v2-self-described dtype.
  DataReader(std::string fn,
             uint32_t expected_v1_dtype = range_search_config::kVecDTypeFloat32)
      : in(fn, std::ios::binary)
  {
    const range_search_config::FbinHeader header =
        range_search_config::read_fbin_header(fn, expected_v1_dtype);
    n = header.n;
    d = header.dim;
    dtype = header.dtype;
    payload_offset = header.payload_offset;
    bytes_per_element = header.bytes_per_element;
    in.seekg(static_cast<std::streamoff>(payload_offset), std::ios::beg);
  }

  void get_batch(char* buf, size_t num)
  {
    in.read(buf,
            static_cast<std::streamsize>(d * num * bytes_per_element));
  }

  void reset()
  {
    in.seekg(static_cast<std::streamoff>(payload_offset), std::ios::beg);
  }

  void get_batch_at(char* buf, size_t num, size_t start)
  {
    in.seekg(static_cast<std::streamoff>(
                 start * d * bytes_per_element + payload_offset),
             std::ios::beg);
    in.read(buf,
            static_cast<std::streamsize>(d * num * bytes_per_element));
  }

  ~DataReader()
  {
    in.close();
  }
};
