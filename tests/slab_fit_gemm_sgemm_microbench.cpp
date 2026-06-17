#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include <cblas.h>

int main()
{
  const int m = 200;
  const int n = 512;
  const int k = 96;
  std::vector<float> points(static_cast<size_t>(m) * k);
  std::vector<float> directions(static_cast<size_t>(n) * k);
  std::vector<float> projection(static_cast<size_t>(m) * n);

  for (size_t i = 0; i < points.size(); ++i)
  {
    points[i] = static_cast<float>((i % 251) - 125) * 0.001f;
  }
  for (size_t i = 0; i < directions.size(); ++i)
  {
    directions[i] = static_cast<float>((i % 257) - 128) * 0.001f;
  }

  openblas_set_num_threads(1);

  std::vector<double> walls;
  walls.reserve(4);
  for (int run = 0; run < 4; ++run)
  {
    const auto start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                points.data(), k, directions.data(), k, 0.0f,
                projection.data(), n);
    const auto end = std::chrono::steady_clock::now();
    walls.push_back(std::chrono::duration<double>(end - start).count());
  }

  std::vector<double> hot(walls.begin() + 1, walls.end());
  std::sort(hot.begin(), hot.end());
  const double median = hot[hot.size() / 2];
  const double aggregate_gflops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) *
      static_cast<double>(k) / median / 1e9;

  std::printf("slab_fit_gemm sgemm microbench:\n");
  std::printf("  shape M=%d N=%d K=%d (single-thread)\n", m, n, k);
  std::printf("  walls (s): %.6f %.6f %.6f %.6f (run 0 = warmup)\n",
              walls[0], walls[1], walls[2], walls[3]);
  std::printf("  median (runs 1-3): %.6f s\n", median);
  std::printf("  aggregate_gflops:  %.3f GFLOPS\n", aggregate_gflops);
  return 0;
}
