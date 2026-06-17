#pragma once

#include <limits>

#include "hnswlib/hnswlib.h"

template <typename DistT>
DistT data_hnsw_1nn_squared(hnswlib::HierarchicalNSW<DistT>& graph,
                            const void* query)
{
  using tableint = hnswlib::tableint;
  if (graph.cur_element_count == 0)
  {
    return std::numeric_limits<DistT>::infinity();
  }

  tableint cur = graph.enterpoint_node_;
  DistT curdist = graph.fstdistfunc_(query, graph.getDataByInternalId(cur),
                                     graph.dist_func_param_);

  for (int level = graph.maxlevel_; level > 0; --level)
  {
    bool changed = true;
    while (changed)
    {
      changed = false;
      unsigned int* data = (unsigned int*)graph.get_linklist(cur, level);
      int size = graph.getListCount(data);
      tableint* datal = (tableint*)(data + 1);
      for (int i = 0; i < size; ++i)
      {
        tableint cand = datal[i];
        DistT d = graph.fstdistfunc_(query, graph.getDataByInternalId(cand),
                                     graph.dist_func_param_);
        if (d < curdist)
        {
          curdist = d;
          cur = cand;
          changed = true;
        }
      }
    }
  }

  auto top = graph.template searchBaseLayerST<false>(cur, query, graph.ef_,
                                                     nullptr);
  if (top.empty())
  {
    return curdist;
  }
  while (top.size() > 1)
  {
    top.pop();
  }
  return top.top().first;
}
