#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <cblas.h>

int main()
{
  constexpr int m = 10000;
  constexpr int n = 50000;
  constexpr int k = 96;

  std::vector<float> a(static_cast<size_t>(m) * static_cast<size_t>(k));
  std::vector<float> b(static_cast<size_t>(n) * static_cast<size_t>(k));
  std::vector<float> c(static_cast<size_t>(m) * static_cast<size_t>(n));

  std::mt19937 rng(283);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f);
  for (float& v : a) v = values(rng);
  for (float& v : b) v = values(rng);

  std::vector<double> measured_seconds;
  for (int run = 0; run < 4; ++run)
  {
    const auto start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, -2.0f,
                a.data(), k, b.data(), k, 0.0f, c.data(), n);
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - start).count();
    std::cerr << "run " << run << " wall_seconds=" << seconds << '\n';
    if (run > 0)
    {
      measured_seconds.push_back(seconds);
    }
  }

  std::sort(measured_seconds.begin(), measured_seconds.end());
  const double median = measured_seconds[measured_seconds.size() / 2];
  const double aggregate_gflops =
      (2.0 * static_cast<double>(m) * static_cast<double>(n) *
       static_cast<double>(k)) /
      median / 1e9;

  std::cout << "median_seconds=" << median << '\n';
  std::cout << "aggregate_gflops=" << aggregate_gflops << '\n';
  return 0;
}
