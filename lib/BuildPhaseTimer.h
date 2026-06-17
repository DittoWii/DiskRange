#pragma once

#include <chrono>
#include <iostream>

#ifndef DJ_ENABLE_BUILD_PHASE_TIMING
#define DJ_ENABLE_BUILD_PHASE_TIMING 0
#endif

inline constexpr bool kDiskRangeBuildPhaseTimingEnabled =
    DJ_ENABLE_BUILD_PHASE_TIMING != 0;

class BuildPhaseTimer
{
 public:
  explicit BuildPhaseTimer(const char* name) : name_(name)
  {
    if constexpr (kDiskRangeBuildPhaseTimingEnabled)
    {
      t0_ = std::chrono::steady_clock::now();
    }
  }

  ~BuildPhaseTimer()
  {
    if constexpr (kDiskRangeBuildPhaseTimingEnabled)
    {
      if (!stopped_)
      {
        report();
      }
    }
  }

  void stop()
  {
    if constexpr (kDiskRangeBuildPhaseTimingEnabled)
    {
      if (!stopped_)
      {
        report();
        stopped_ = true;
      }
    }
  }

  BuildPhaseTimer(const BuildPhaseTimer&) = delete;
  BuildPhaseTimer& operator=(const BuildPhaseTimer&) = delete;
  BuildPhaseTimer(BuildPhaseTimer&&) = delete;
  BuildPhaseTimer& operator=(BuildPhaseTimer&&) = delete;

 private:
  void report() const
  {
    const auto dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0_)
                        .count();
    std::cout << "[build-timing] " << name_ << " = " << dt << " seconds\n";
  }

  const char* name_;
  std::chrono::steady_clock::time_point t0_;
  bool stopped_ = false;
};
