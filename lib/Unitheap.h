#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>

typedef unsigned long ul;

class ListElement
{
 public:
  int key;
  ul prev;
  ul next;
};

class HeadEnd
{
 public:
  ul first;
  ul second;
};

using namespace std;

class UnitHeap
{
 public:
  std::vector<int> update;
  std::vector<ListElement> LinkedList;  // key=degree, prev=id of previous node
                                        // in degree DESC ordering, next=...
  std::vector<HeadEnd> Header;  // first=id of first node with this degree,
                                // second=id of last node...
  size_t heapsize = 0;          // updated at each instant
  ul top;                       // id of node with highest degree
  ul huge;                      // ignore huge nodes (deg > sqrt of graph size)
  ul none;                      // integer that cannot be a node index
  const int infty = INT_MAX / 2;

  // --------------------------------------------------
  // ---------- Initialisation and insertion ----------
  // --------------------------------------------------

  UnitHeap(ul size)
  {
    none = size + 2;
    huge = sqrt((double)size);  // should be in ReConstruct with actual heapsize
    LinkedList.resize(size, {.key = infty, .prev = none, .next = none});
    update.resize(size, infty);
  }
  void InsertElement(const ul index, const int key)
  {
    LinkedList[index].key = key;  //+1;
    update[index] = -key;
    heapsize++;
  }

  void ReConstruct()
  {
    vector<ul> g(heapsize);
    for (ul i = 0; i < heapsize; i++) g[i] = i;

    sort(g.begin(), g.end(),
         [&](const ul a, const ul b) -> bool
         {
           return LinkedList[a].key > LinkedList[b].key or
                  (LinkedList[a].key == LinkedList[b].key and a < b);
         });  // degree DESC
    // sorting update should be simpler as it's a straightforward array
    // sort(g.begin(), g.end(), [&](const int a, const int b) -> bool {
    //   return (update[a] < update[b]);
    // });

    top = g[0];
    int current_key = LinkedList[top].key;  // max degree
    Header.resize(10 * current_key + 1, {.first = none, .second = none});
    Header[current_key].first = top;
    for (size_t i = 0; i < g.size(); i++)
    {
      ul v = g[i];
      LinkedList[v].prev = (i > 0) ? g[i - 1] : none;
      LinkedList[v].next = (i < g.size() - 1) ? g[i + 1] : none;

      int key = LinkedList[g[i]].key;
      if (key != current_key)
      {
        Header[current_key].second = g[i - 1];
        Header[key].first = g[i];
        current_key = key;
      }
    }

    Header[current_key].second = g.back();
    // print_status();
  }

  // --------------------------------------------------
  // --------------- General heap tools ---------------
  // --------------------------------------------------

  void erase_key_element(const ul index, const ul next, const ul prev)
  {
    int key = LinkedList[index].key;
    if (Header[key].first == Header[key].second)  // == index
      Header[key].first = Header[key].second = none;
    else if (index == Header[key].first)
      Header[key].first = next;
    else if (index == Header[key].second)
      Header[key].second = prev;
  }

  // --------------------------------------------------
  // --------------- Optimal extraction ---------------
  // --------------------------------------------------

  ul ExtractMax()
  {
    // Debug("Extract from candidates "<<heapsize)
    ul tmptop;  // int loops=0;
    do
    {
      tmptop = top;
      if (update[top] < 0) DecreaseTop();
      // if(++loops%100000==0) Debug(loops <<" exctract loophole "<<top<<"
      // "<<tmptop)
    }
    while (top != tmptop);

    // Debug("extracted" << tmptop)

    DeleteElement(top);
    return tmptop;
  }

  void DecreaseTop()
  {
    // If the "top" has to be updated, try and update half what you should
    // and see if someone else comes on top
    // Recursion will take care of further updates

    const ul next = LinkedList[top].next;
    if (next == none)
    {
      return;
    }

    const int key = LinkedList[top].key;
    const int leftover = update[top] / 2;
    const int new_key = key + update[top] - leftover;  //*//
    // Debug("Decrease "<<top<<" = "<<key<<" "<<update[top])
    if (new_key >= LinkedList[next].key) return;
    update[top] = leftover;  //*//

    ul level_tail = Header[key].second;
    ul next_level = LinkedList[level_tail].next;
    ul loops = 0;
    while (next_level != none && LinkedList[next_level].key >= new_key)
    {
      level_tail = Header[LinkedList[next_level].key].second;
      next_level = LinkedList[level_tail].next;
      if (++loops % (huge * huge) == 0)
      {
        // safety_check();
      }
    }
    // Debug(top << " from key "<<key<<" to "<<new_key<<" between
    // "<<level_tail<<" and "<<next_level)

    LinkedList[next].prev = none;
    LinkedList[top].prev = level_tail;
    LinkedList[top].next = next_level;
    LinkedList[level_tail].next = top;
    if (next_level != none) LinkedList[next_level].prev = top;

    erase_key_element(top, next, none);

    LinkedList[top].key = new_key;

    Header[new_key].second = top;
    if (Header[new_key].first == none) Header[new_key].first = top;

    top = next;
  }

  void DeleteElement(const ul index)
  {
    // Debug("Delete "<< index)
    update[index] = infty;
    ul prev = LinkedList[index].prev;
    ul next = LinkedList[index].next;

    if (prev != none) LinkedList[prev].next = next;
    if (next != none) LinkedList[next].prev = prev;

    erase_key_element(index, next, prev);

    if (top == index) top = next;
    LinkedList[index].prev = LinkedList[index].next = none;
    // Debug("heapsize "<<heapsize)
    heapsize--;
    // if(heapsize % (LinkedList.size()/2) == 1) safety_check();
    // Debug("deleted")
  }

  // --------------------------------------------------
  // ------------------ Key updates -------------------
  // --------------------------------------------------

  void lazyIncrement(const ul index, const int up)
  {
    if (update[index] == infty) return;
    if (update[index] == 0 and up > 0)
      IncrementKey(index);  // unlikely
    else
    {
      update[index] += up;  // up can be negative
    }
  }

  void IncrementKey(const ul index)
  {
    const ul level_head = Header[LinkedList[index].key].first;
    const ul prev = LinkedList[index].prev;
    const ul next = LinkedList[index].next;
    // Debug("Increment "<<index<<" =
    // "<<LinkedList[index].key<<"+"<<update[index]<<" between "<<prev<<" and
    // "<<next)

    if (level_head != index)
    {  // if index is head of level, keep LinkedList
      LinkedList[prev].next = next;
      if (next != none) LinkedList[next].prev = prev;

      ul prev_level = LinkedList[level_head].prev;
      LinkedList[index].prev = prev_level;
      LinkedList[index].next = level_head;
      LinkedList[level_head].prev = index;
      if (prev_level != none) LinkedList[prev_level].next = index;
    }

    erase_key_element(index, next, prev);

    int key = ++LinkedList[index].key;
    Header[key].second = index;
    if (Header[key].first == none)
    {
      Header[key].first = index;
      if (key > LinkedList[top].key)
        top = index;  // was not in the if initially
    }

    if (key + 4 >= (int)Header.size())
      Header.resize(Header.size() * 1.5, {.first = none, .second = none});
  }

  // --------------------------------------------------
  // ----- Debug: vector status and safety check ------
  // --------------------------------------------------

  void print_status()
  {
#ifdef DEBUG
    safety_check();
    Debug("----------------- Heap status -----------------")
  Debug("| Top: " << top)
  for (size_t i = 0; i < 22; i++) {
    Debug("| "<<i << " key:" << LinkedList[i].key << " up:" << update[i]
        << "\t\tprev: "<<LinkedList[i].prev << " next:" << LinkedList[i].next)
  }
  Debug("----------------- ---- ------ -----------------")
  for(size_t i=0; i <Header.size(); i++)
    {
      auto h = Header[i];
      Debug("| Level " << i << ": " << h.first << " to " << h.second)
    }
    Debug("----------------- ---- ------ -----------------")
#endif
        return;
  }

  void safety_check()
  {
    int problem = 0;

    int key_out_header = 0, first_not_first = 0, last_not_last = 0,
        negative_key = 0;
    int beginners = 0, enders = 0;
    for (size_t i = 0; i < LinkedList.size(); i++)
    {
      int key = LinkedList[i].key;
      if (update[i] >= infty) continue;
      if (key < 0 and ++negative_key) continue;

      ul first = Header[key].first, second = Header[key].second;
      // keys are well in their header value
      if (key != LinkedList[first].key or key != LinkedList[second].key)
      {
        key_out_header += 1;
      }
      // header first is lower than prev
      if (first == i and LinkedList[i].prev != none)
        if (LinkedList[LinkedList[i].prev].key <= key) first_not_first++;
      // header last is higher than next
      if (second == i and LinkedList[i].next != none)
        if (LinkedList[LinkedList[i].next].key >= key) last_not_last++;
      // only one beginner and ender
      if (LinkedList[i].prev == none) beginners += 1;
      if (LinkedList[i].next == none) enders += 1;
    }

    // heapsize and LinkedList size are the same
    ul u = top;
    ul chain_size = 0;
    while (u != none && ++chain_size && chain_size < heapsize * 2)
      u = LinkedList[u].next;
  }
};
