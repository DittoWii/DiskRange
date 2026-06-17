#pragma once

#include <cassert>
#include <filesystem>
#include <string>

#include "../wheel/Basic/Console/console_RDMA.hpp"
#include "data_type.h"

enum class JoinType
{
  SELF_JOIN,
  CROSS_JOIN
};

inline std::string JoinTypeToToken(JoinType join_type)
{
  switch (join_type)
  {
    case JoinType::SELF_JOIN:
      return "sj";
    case JoinType::CROSS_JOIN:
      return "cj";
  }
  assert(false && "join_type is not supported");
  return "";
}

inline std::string AddSuffixBeforeExtension(const std::string& path,
                                            const std::string& suffix)
{
  std::filesystem::path file_path(path);
  auto stem = file_path.stem().string();
  auto ext = file_path.extension().string();
  file_path.replace_filename(stem + suffix + ext);
  return file_path.string();
}

struct VectorFileType
{
  std::string dataset = "/md1/dongjiang_data/";
  std::string base_path = "";
  std::string vec_file = "";
  std::string query_file = "";
  std::string gt_file = "";
  std::string index_path = "";

  std::string left_file = "";
  std::string right_file = "";
  std::string join_gt_file = "";

  idtype vec_num = 0;

  inline void ClearJoinFiles()
  {
    left_file.clear();
    right_file.clear();
    join_gt_file.clear();
  }

  inline void GetVecFile(std::string vec_name, std::string file_type,
                         std::string index_name)
  {
    ClearJoinFiles();
    std::string deep = "deep";
    if (vec_name.substr(0, deep.length()) == deep)
    {
      base_path = dataset + "deep/" + vec_name;
      index_path = base_path + "/" + index_name + "/";
    }
    else
    {
      base_path = dataset + vec_name;
      index_path = base_path + "/" + index_name + "/";
    }
    if (vec_name == "siftsmall")
    {
      vec_num = 10'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/siftsmall_base.fbin";
        query_file = base_path + "/siftsmall_query.fbin";
        gt_file = base_path + "/siftsmall_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/siftsmall_base.fvecs";
        query_file = base_path + "/siftsmall_query.fvecs";
        gt_file = base_path + "/siftsmall_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "sift")
    {
      vec_num = 1'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/sift_base.fbin";
        query_file = base_path + "/sift_query.fbin";
        gt_file = base_path + "/sift_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/sift_base.fvecs";
        query_file = base_path + "/sift_query.fvecs";
        gt_file = base_path + "/sift_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "gist")
    {
      vec_num = 1'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/gist_base.fbin";
        query_file = base_path + "/gist_query.fbin";
        gt_file = base_path + "/gist_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/gist_base.fvecs";
        query_file = base_path + "/gist_query.fvecs";
        gt_file = base_path + "/gist_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "deep1m")
    {
      vec_num = 1'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/deep1m_base.fbin";
        query_file = base_path + "/deep1m_query.fbin";
        gt_file = base_path + "/deep1m_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/deep1m_base.fvecs";
        query_file = base_path + "/deep1m_query.fvecs";
        gt_file = base_path + "/deep1m_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "deep10m")
    {
      vec_num = 10'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/deep10m_base.fbin";
        query_file = base_path + "/deep10m_query.fbin";
        gt_file = base_path + "/deep10m_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/deep10m_base.fvecs";
        query_file = base_path + "/deep10m_query.fvecs";
        gt_file = base_path + "/deep10m_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "deep100m")
    {
      vec_num = 100'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/deep100m_base.fbin";
        query_file = base_path + "/deep100m_query.fbin";
        gt_file = base_path + "/deep100m_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/deep100m_base.fvecs";
        query_file = base_path + "/deep100m_query.fvecs";
        gt_file = base_path + "/deep100m_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else if (vec_name == "deep1b")
    {
      vec_num = 1'000'000'000;
      if (file_type == "bin")
      {
        vec_file = base_path + "/deep1b_base.fbin";
        query_file = base_path + "/deep1b_query.fbin";
        gt_file = base_path + "/deep1b_gt100.fbin";
      }
      else if (file_type == "fvecs")
      {
        vec_file = base_path + "/deep1b_base.fvecs";
        query_file = base_path + "/deep1b_query.fvecs";
        gt_file = base_path + "/deep1b_gt100.fvecs";
      }
      else
      {
        msgerror_s("file_type {} is not supported", vec_name.c_str());
        assert(false);
      }
    }
    else
    {
      msgerror_s("vec_name {} is not supported", vec_name.c_str());
      assert(false);
    }
  }

  inline void GetJoinVecFile(std::string vec_name, std::string file_type,
                             std::string index_name, JoinType join_type,
                             const std::string& eps_token)
  {
    assert(!eps_token.empty());
    GetVecFile(vec_name, file_type, index_name);

    if (join_type == JoinType::SELF_JOIN)
    {
      left_file = vec_file;
      right_file = vec_file;
    }
    else if (join_type == JoinType::CROSS_JOIN)
    {
      left_file = AddSuffixBeforeExtension(vec_file, "_left");
      right_file = AddSuffixBeforeExtension(vec_file, "_right");
    }
    else
    {
      msgerror_s("join_type for dataset {} is not supported", vec_name.c_str());
      assert(false);
    }

    join_gt_file = base_path + "/" + vec_name + "_" +
                   JoinTypeToToken(join_type) + "_" + eps_token + "_gt.fbin";
  }

  void PrintVecFile()
  {
    msginfo_s("base_path = {}", base_path.c_str());
    msginfo_s("vec_file = {}", vec_file.c_str());
    msginfo_s("query_file = {}", query_file.c_str());
    msginfo_s("gt_file = {}", gt_file.c_str());
    msginfo_s("left_file = {}", left_file.c_str());
    msginfo_s("right_file = {}", right_file.c_str());
    msginfo_s("join_gt_file = {}", join_gt_file.c_str());
    msginfo_s("index_path = {}", index_path.c_str());
    msginfo_s("vec_num = {}", vec_num);
  }
};
