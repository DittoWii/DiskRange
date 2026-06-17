#pragma once

#include <vector>

#include "Unitheap.h"

using namespace std;
typedef unsigned long ul;

void move_window(const vector<vector<size_t>> &out_neigh,
                 const vector<vector<size_t>> &in_neigh, UnitHeap &heap,
                 const ul &new_node, const ul &old_node)
{
  auto old_parent =
      out_neigh[old_node].begin();  // note that neighbours are sorted so we can
  auto new_parent =
      out_neigh[new_node].begin();  // compute their intersection in linear time

  if (old_node == new_node)
    old_parent =
        out_neigh[old_node].end();  // put directly at the END to ignore
  else
  {  // decrease children of old node
     // for (auto child : in_neigh[old_node]) heap.lazyIncrement(child, -1);
  }

  vector<ul> tmp_old_parents, tmp_new_parents;
  while (true)
  {  // find parents that are not in the intersection of old and new node
    int factor = -1;
    if (old_parent >= out_neigh[old_node].end())
    {
      if (new_parent >= out_neigh[new_node].end()) break;  // no parents remain
      factor = 1;
    }
    else if (new_parent < out_neigh[new_node].end())
    {
      if (*new_parent == *old_parent)
      {  // common parents should be ignored
        old_parent++;
        new_parent++;
        continue;
      }
      if (*new_parent < *old_parent) factor = 1;
    }

    if (factor == -1)
    {
      tmp_old_parents.push_back(*old_parent);
      old_parent++;
    }
    else
    {
      tmp_new_parents.push_back(*new_parent);
      new_parent++;
    }
  }

  for (auto &parent : tmp_old_parents)
  {  // decrease parents and siblings of old node
    // heap.lazyIncrement(parent, -1);
    for (auto sibling : in_neigh[parent])
    {
      if (sibling != old_node) heap.lazyIncrement(sibling, -1);
    }
  }

  // for (auto child : in_neigh[new_node]) heap.lazyIncrement(child, +1);

  for (auto &parent : tmp_new_parents)
  {  // increase parents and siblings of new node
    // heap.lazyIncrement(parent, +1);
    for (auto sibling : in_neigh[parent])
    {
      if (sibling != new_node) heap.lazyIncrement(sibling, +1);
    }
  }
}

vector<size_t> order_gorder(const vector<vector<size_t>> &out_neigh, ul window,
                            ul n2 = 0)
{
  auto n = out_neigh.size();
  if (n2 == 0) n2 = n;
  vector<vector<size_t>> in_neigh(n2);
  for (size_t u = 0; u < n; u++)
  {
    for (auto &v : out_neigh[u])
    {
      in_neigh[v].push_back(u);
    }
  }
  vector<ul> order;
  order.reserve(n);
  UnitHeap heap(n);

  for (ul u = 0; u < n; ++u)
  {
    heap.InsertElement(u, out_neigh[u].size());
  }

  heap.ReConstruct();  // heap indegree DESC sort

  ul hub = heap.top;
  order.push_back(hub);
  heap.DeleteElement(hub);

  move_window(out_neigh, in_neigh, heap, hub, hub);

  while (heap.heapsize > 0)
  {
    // if (order.size() % 10000 == 0)
    // cout << order.size() << "\n";
    ul new_node = heap.ExtractMax();
    order.push_back(new_node);
    ul old_node = new_node;
    if (order.size() > window)  // remove oldest node of window
      old_node = order[order.size() - window - 1];
    // else Debug("no window"<<endl)
    move_window(out_neigh, in_neigh, heap, new_node, old_node);
  }

  vector<size_t> rank(n);
  for (size_t i = 0; i < n; i++)
  {
    rank[order[i]] = i;
  }

  return rank;
}