#pragma once

#include "../utils/heap.h"
#include "../utils/utils.h"
#include "string.h"

struct Cache
{
  size_t budget;
  size_t size;
  size_t length;
  float* data;

  size_t filled;
  size_t hit;
  size_t total;
  updateable_heap<size_t, size_t, std::greater<size_t>> priority;
  std::vector<int> address_table;
  std::vector<size_t> ptr;
  std::vector<std::vector<size_t>> iters;

  Cache(size_t budget, size_t length) : budget(budget), length(length)
  {
    size = budget / 4 / length;
    data = new float[length * size];
    priority = updateable_heap<size_t, size_t, std::greater<size_t>>(size + 1);
    filled = 0;
    hit = 0;
    total = 0;
  }

  void init(std::vector<std::vector<size_t>> tasks, size_t n = 0)
  {
    if (n == 0) n = tasks.size();
    address_table = std::vector<int>(n, -1);
    ptr.resize(n);
    iters.resize(n);
    size_t num = 0;
    for (size_t i = 0; i < tasks.size(); i++)
    {
      for (size_t id : tasks[i])
      {
        iters[id].push_back(num++);
      }
    }
    for (size_t i = 0; i < n; i++)
      iters[i].push_back(std::numeric_limits<size_t>::max());
  }

  inline bool find(size_t id, float* buffer, unsigned bucket_length)
  {
    total++;
    ptr[id]++;
    if (address_table[id] == -1) return false;
    hit++;
    priority.update(std::make_pair(id, iters[id][ptr[id]]));
    memcpy(buffer, data + address_table[id] * length,
           bucket_length * sizeof(float));
    return true;
  }

  inline void push(size_t id, float* buffer, unsigned bucket_length)
  {
    if (filled < size)
    {
      address_table[id] = filled;
      memcpy(data + address_table[id] * length, buffer,
             bucket_length * sizeof(float));
      priority.add(std::make_pair(id, iters[id][ptr[id]]));
      filled++;
    }
    else if (priority.top().second > iters[id][ptr[id]])
    {
      memcpy(data + address_table[priority.top().first] * length, buffer,
             bucket_length * sizeof(float));
      address_table[id] = address_table[priority.top().first];
      address_table[priority.top().first] = -1;
      priority.pop();
      priority.add(std::make_pair(id, iters[id][ptr[id]]));
    }
  }

  ~Cache()
  {
    delete[] data;
  }
};

struct LRUCache
{
  size_t budget;
  size_t size;
  size_t length;
  float* data;

  size_t filled;
  size_t hit;
  size_t total;
  size_t cnt;
  updateable_heap<size_t, size_t, std::greater<size_t>> priority;
  std::vector<int> address_table;

  LRUCache(size_t budget, size_t length) : budget(budget), length(length)
  {
    size = budget / 4 / length;
    data = new float[length * size];
    priority = updateable_heap<size_t, size_t, std::greater<size_t>>(size + 1);
    filled = 0;
    hit = 0;
    total = 0;
    cnt = 0;
  }

  void init(std::vector<std::vector<size_t>> tasks)
  {
    auto n = tasks.size();
    address_table = std::vector<int>(n, -1);
    for (size_t i = 0; i < n; i++) total += tasks[i].size();
  }

  inline bool find(size_t id, float* buffer, unsigned bucket_length)
  {
    cnt++;
    if (address_table[id] == -1) return false;
    hit++;
    priority.update(std::make_pair(id, total - cnt));
    memcpy(buffer, data + address_table[id] * length,
           bucket_length * sizeof(float));
    return true;
  }

  inline void push(size_t id, float* buffer, unsigned bucket_length)
  {
    if (filled < size)
    {
      address_table[id] = filled;
      memcpy(data + address_table[id] * length, buffer,
             bucket_length * sizeof(float));
      priority.add(std::make_pair(id, total - cnt));
      filled++;
    }
    else
    {
      memcpy(data + address_table[priority.top().first] * length, buffer,
             bucket_length * sizeof(float));
      address_table[id] = address_table[priority.top().first];
      address_table[priority.top().first] = -1;
      priority.pop();
      priority.add(std::make_pair(id, total - cnt));
    }
  }

  ~LRUCache()
  {
    delete[] data;
  }
};

void simulate_cache(std::vector<std::vector<size_t>> tasks, size_t n,
                    size_t size)
{
  size_t CACHE_SIZE = size;
  std::vector<unsigned> cache(CACHE_SIZE);
  std::vector<int> address_table(n, -1);

  std::vector<unsigned> ptr(n);
  std::vector<std::vector<unsigned>> iters(n);
  unsigned num = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (unsigned id : tasks[i])
    {
      iters[id].push_back(num++);
    }
  }
  for (size_t i = 0; i < n; i++)
    iters[i].push_back(std::numeric_limits<unsigned>::max());

  updateable_heap<unsigned, unsigned, std::greater<unsigned>> priority(
      CACHE_SIZE + 1);

  unsigned cache_hit = 0, total = 0;
  unsigned filled = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (unsigned id : tasks[i])
    {
      total++;
      ptr[id]++;
      if (address_table[id] == -1)
      {
        if (filled < CACHE_SIZE)
        {
          cache[filled] = id;
          address_table[id] = filled;
          priority.add(std::make_pair(id, iters[id][ptr[id]]));
          filled++;
        }
        else if (priority.top().second > iters[id][ptr[id]])
        {
          cache[address_table[priority.top().first]] = id;
          address_table[id] = address_table[priority.top().first];
          address_table[priority.top().first] = -1;
          priority.pop();
          priority.add(std::make_pair(id, iters[id][ptr[id]]));
        }
      }
      else
      {
        cache_hit++;
        priority.update(std::make_pair(id, iters[id][ptr[id]]));
      }
    }
  }

  std::cout << "[stat] total: " << total << ", cache hit: " << cache_hit << " "
            << 1.0 * cache_hit / total << "\n";
}

void simulate_cache_LRU(std::vector<std::vector<size_t>> tasks, size_t n,
                        size_t size)
{
  size_t CACHE_SIZE = size;
  std::vector<unsigned> cache(CACHE_SIZE);
  std::vector<int> address_table(n, -1);

  updateable_heap<unsigned, unsigned, std::greater<unsigned>> priority(
      CACHE_SIZE + 1);

  unsigned cache_hit = 0, total = 0, cnt = 0;
  for (size_t i = 0; i < n; i++) total += tasks[i].size();
  unsigned filled = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (unsigned id : tasks[i])
    {
      cnt++;
      if (address_table[id] == -1)
      {
        if (filled < CACHE_SIZE)
        {
          cache[filled] = id;
          address_table[id] = filled;
          priority.add(std::make_pair(id, total - cnt));
          filled++;
        }
        else
        {
          cache[address_table[priority.top().first]] = id;
          address_table[id] = address_table[priority.top().first];
          address_table[priority.top().first] = -1;
          priority.pop();
          priority.add(std::make_pair(id, total - cnt));
        }
      }
      else
      {
        cache_hit++;
        priority.update(std::make_pair(id, total - cnt));
      }
    }
  }

  std::cout << "total: " << total << ", cache hit: " << cache_hit << " "
            << 1.0 * cache_hit / total << "\n";
}