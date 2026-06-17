#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

// Shared constants for io_uring mem-budget enforcement and payload cache sizing.
constexpr size_t kMinIoQueueDepth = 1;
constexpr size_t kMaxIoQueueDepth = 32768;
constexpr size_t kMinEffectiveIoQueueDepth = 4;
constexpr size_t kHnswOverheadFixed = size_t{64} << 20;
constexpr size_t kPerThreadConstOverhead = size_t{4} << 20;
constexpr size_t kHitLookupOverhead = size_t{64};
constexpr size_t kReservedSlack = size_t{128} << 20;
constexpr size_t kClusterCacheShardCount = 64;
constexpr size_t kClusterCacheMaxBytes = size_t{256} << 30;
constexpr size_t kCellCacheShardCount = 256;
constexpr size_t kCellCacheMaxBytes = size_t{256} << 30;
constexpr double kCellCacheIndexOverheadRatio = 0.15;
static_assert((kClusterCacheShardCount & (kClusterCacheShardCount - 1)) == 0,
              "cluster cache shard count must be a power of two");
static_assert((kCellCacheShardCount & (kCellCacheShardCount - 1)) == 0,
              "cell cache shard count must be a power of two");

constexpr size_t checked_mul(size_t a, size_t b) noexcept
{
  if (a == 0 || b == 0)
  {
    return 0;
  }
  if (a > std::numeric_limits<size_t>::max() / b)
  {
    return std::numeric_limits<size_t>::max();
  }
  return a * b;
}

constexpr size_t bit_floor_compat(size_t x) noexcept
{
  if (x == 0)
  {
    return 0;
  }
  size_t result = 1;
  while ((x >>= 1) != 0)
  {
    result <<= 1;
  }
  return result;
}

inline void validate_io_queue_depth(size_t io_queue_depth)
{
  if (io_queue_depth < kMinIoQueueDepth ||
      io_queue_depth > kMaxIoQueueDepth)
  {
    throw std::runtime_error("io_queue_depth out of range [" +
                             std::to_string(kMinIoQueueDepth) + ", " +
                             std::to_string(kMaxIoQueueDepth) + "]: " +
                             std::to_string(io_queue_depth));
  }
}
