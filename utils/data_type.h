#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

using int64 = uint64_t;
using int32 = uint32_t;

using idtype = uint32_t;
using datatype = float;
using disttype = float;
using i8vec_t = int8_t;
using msgtype = char;
using tagtype = int;
using flagtype = uint8_t;
using readtype = uint32_t;

using sifttype = float;

using packtype = uint64_t;

struct VectorSet
{
  std::vector<idtype> global_id_;
  std::vector<float> vec_data_;
  idtype vec_dim_ = 0, vec_num_ = 0;

  datatype* GetPoint(idtype i)
  {
    return &vec_data_[i * vec_dim_];
  }
  void Drop()
  {
    vec_data_.clear();
    vec_data_.shrink_to_fit();
  }
  void Alloc()
  {
    vec_data_.resize(vec_num_ * vec_dim_, 0.f);
    global_id_.resize(vec_num_, 0);
  }
  void Resize(idtype vec_num)
  {
    vec_num_ = vec_num;
    global_id_.resize(vec_num_);
    vec_data_.resize(vec_num_ * vec_dim_);
  }
  bool Empty() const
  {
    return vec_num_ == 0;
  }
};

struct GTSet
{
  int topk_ = 0;
  int query_num_ = 0;
  std::vector<idtype> gt_ids_;
  std::vector<disttype> gt_dists_;

  GTSet()
  {
  }
  void Init(int topk, int query_num)
  {
    topk_ = topk;
    query_num_ = query_num;
    gt_ids_.resize(topk_ * query_num_);
    gt_dists_.resize(topk_ * query_num_);
  }

  GTSet(int topk, int query_num) : topk_(topk), query_num_(query_num)
  {
    gt_ids_.resize(topk_ * query_num_);
    gt_dists_.resize(topk_ * query_num_);
  }

  idtype* GetGT(int i)
  {
    return &gt_ids_[i * topk_];
  }
  const disttype* GetGTDist(int i) const
  {
    return &gt_dists_[i * topk_];
  }

  void Drop()
  {
    gt_ids_.clear();
    gt_ids_.shrink_to_fit();
    gt_dists_.clear();
    gt_dists_.shrink_to_fit();
  }
};

struct JoinGTSet
{
  int query_num_ = 0;
  std::vector<int64> offsets_;
  std::vector<idtype> gt_ids_;
  std::vector<disttype> gt_dists_;

  JoinGTSet()
  {
  }

  void Init(int query_num, int64 total_results = 0)
  {
    query_num_ = query_num;
    offsets_.assign((size_t)query_num_ + 1, 0);
    gt_ids_.resize((size_t)total_results);
    gt_dists_.resize((size_t)total_results);
  }

  int64 GetGTCount(int i) const
  {
    return offsets_[(size_t)i + 1] - offsets_[(size_t)i];
  }
  idtype* GetGT(int i)
  {
    return gt_ids_.empty() ? nullptr : gt_ids_.data() + offsets_[(size_t)i];
  }
  const idtype* GetGT(int i) const
  {
    return gt_ids_.empty() ? nullptr : gt_ids_.data() + offsets_[(size_t)i];
  }
  disttype* GetGTDist(int i)
  {
    return gt_dists_.empty() ? nullptr : gt_dists_.data() + offsets_[(size_t)i];
  }
  const disttype* GetGTDist(int i) const
  {
    return gt_dists_.empty() ? nullptr : gt_dists_.data() + offsets_[(size_t)i];
  }

  void Drop()
  {
    offsets_.clear();
    offsets_.shrink_to_fit();
    gt_ids_.clear();
    gt_ids_.shrink_to_fit();
    gt_dists_.clear();
    gt_dists_.shrink_to_fit();
  }
};

#define HIGH_16_MASK (0xFFFF000000000000ULL)
#define LOW_48_MASK (0x0000FFFFFFFFFFFFULL)

inline uint16_t GetGidFromPackId(packtype pack_id)
{
  return (pack_id & HIGH_16_MASK) >> 48;
}
inline uint64_t GetLocalIdFromPackId(packtype pack_id)
{
  return pack_id & LOW_48_MASK;
}
inline packtype PackIds(uint16_t gid, uint64_t local_id)
{
  return ((uint64_t)gid << 48) | (local_id & LOW_48_MASK);
}
