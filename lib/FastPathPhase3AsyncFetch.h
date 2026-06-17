#pragma once

#include <future>
#include <vector>

#include "RBQCodeStorage.h"

struct BorderlineRef
{
  size_t cluster_id = 0;
  size_t local_index = 0;
};

struct BorderlineCode
{
  size_t cluster_id = 0;
  size_t local_index = 0;
  std::vector<char> code_bytes;
};

inline std::future<std::vector<BorderlineCode>> async_fetch_8bit_codes(
    std::vector<BorderlineRef> borderline_refs,
    const RBQCodeStorage& storage)
{
  return std::async(std::launch::async,
                    [borderline_refs = std::move(borderline_refs), &storage]() {
                      std::vector<BorderlineCode> out;
                      out.reserve(borderline_refs.size());
                      for (const BorderlineRef& ref : borderline_refs)
                      {
                        out.push_back({ref.cluster_id, ref.local_index,
                                       storage.read_code(ref.cluster_id,
                                                         ref.local_index)});
                      }
                      return out;
                    });
}
