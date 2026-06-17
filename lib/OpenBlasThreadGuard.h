#pragma once

#ifdef DJ_USE_OPENBLAS
#ifdef __linux__
#include <cstdlib>
#include <sched.h>
#endif

#include <cblas.h>

class OpenBlasThreadGuard
{
 public:
  explicit OpenBlasThreadGuard(int desired)
      : prev_(openblas_get_num_threads())
#ifdef __linux__
        ,
        restore_affinity_(false)
#endif
  {
#ifdef __linux__
    expand_affinity_from_gomp();
#endif
    openblas_set_num_threads(desired);
  }

  ~OpenBlasThreadGuard()
  {
    openblas_set_num_threads(prev_);
#ifdef __linux__
    restore_affinity();
#endif
  }

  OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete;
  OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete;
  OpenBlasThreadGuard(OpenBlasThreadGuard&&) = delete;
  OpenBlasThreadGuard& operator=(OpenBlasThreadGuard&&) = delete;

 private:
#ifdef __linux__
  static bool parse_cpu_list(const char* text, cpu_set_t* mask)
  {
    if (text == nullptr || *text == '\0')
    {
      return false;
    }
    CPU_ZERO(mask);
    bool any = false;
    const char* p = text;
    while (*p != '\0')
    {
      while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')
      {
        ++p;
      }
      char* end = nullptr;
      long first = std::strtol(p, &end, 10);
      if (end == p)
      {
        while (*p != '\0' && *p != ',' && *p != ' ' && *p != '\t' &&
               *p != '\n')
        {
          ++p;
        }
        continue;
      }
      p = end;

      long last = first;
      if (*p == '-')
      {
        ++p;
        char* range_end = nullptr;
        const long parsed_last = std::strtol(p, &range_end, 10);
        if (range_end != p)
        {
          last = parsed_last;
          p = range_end;
        }
      }
      if (first > last)
      {
        const long tmp = first;
        first = last;
        last = tmp;
      }

      long step = 1;
      if (*p == ':')
      {
        ++p;
        char* step_end = nullptr;
        const long parsed_step = std::strtol(p, &step_end, 10);
        if (step_end != p && parsed_step > 0)
        {
          step = parsed_step;
          p = step_end;
        }
      }

      for (long cpu = first; cpu <= last && cpu < CPU_SETSIZE; cpu += step)
      {
        if (cpu >= 0)
        {
          CPU_SET(static_cast<int>(cpu), mask);
          any = true;
        }
      }
      while (*p != '\0' && *p != ',' && *p != ' ' && *p != '\t' &&
             *p != '\n')
      {
        ++p;
      }
    }
    return any;
  }

  static bool affinity_equal(const cpu_set_t& lhs, const cpu_set_t& rhs)
  {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    {
      if (CPU_ISSET(cpu, &lhs) != CPU_ISSET(cpu, &rhs))
      {
        return false;
      }
    }
    return true;
  }

  void expand_affinity_from_gomp()
  {
    cpu_set_t target;
    if (!parse_cpu_list(std::getenv("GOMP_CPU_AFFINITY"), &target))
    {
      return;
    }
    CPU_ZERO(&prev_affinity_);
    if (sched_getaffinity(0, sizeof(prev_affinity_), &prev_affinity_) != 0)
    {
      return;
    }
    if (affinity_equal(prev_affinity_, target))
    {
      return;
    }
    restore_affinity_ =
        sched_setaffinity(0, sizeof(target), &target) == 0;
  }

  void restore_affinity()
  {
    if (restore_affinity_)
    {
      sched_setaffinity(0, sizeof(prev_affinity_), &prev_affinity_);
    }
  }
#endif

  int prev_;
#ifdef __linux__
  cpu_set_t prev_affinity_;
  bool restore_affinity_;
#endif
};
#endif
