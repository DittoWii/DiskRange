#pragma once

#include <string>

struct IndexPaths
{
  std::string cluster;
  std::string metadata;
  std::string hnsw;
  std::string data_hnsw;
  std::string data_hnsw_trace;
  std::string sqg;
  std::string rabitq_1bit_codes;
  std::string rabitq_8bit_codes;
};

inline IndexPaths make_diskrange_paths(const std::string& prefix)
{
  const std::string dir = prefix + "_diskrange";
  return {dir + "/_cluster_file.bin",
          dir + "/_metadata_file.bin",
          dir + "/_hnsw_file.bin",
          dir + "/_data_hnsw_file.bin",
          dir + "/_data_hnsw_trace.csv",
          dir + "/_sqg_file.bin",
          dir + "/_rabitq_1bit_codes_file.bin",
          dir + "/_rabitq_8bit_codes_file.bin"};
}
