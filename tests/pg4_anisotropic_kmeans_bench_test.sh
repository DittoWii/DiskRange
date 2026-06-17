#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="$(mktemp -d /tmp/pg4_aniso_kmeans.XXXXXX)"

cleanup() {
  rm -rf "$run_dir"
}
trap cleanup EXIT

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/pg4_aniso_cmake.log
cmake --build build >/tmp/pg4_aniso_build.log

harness_cpp="$run_dir/pg4_aniso_harness.cpp"
harness_bin="$run_dir/pg4_aniso_harness"

cat >"$harness_cpp" <<'CPP'
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "lib/Kmeans.h"

int r_server_id = 0;

namespace
{
void require_close(float actual, float expected, float tolerance, const char* label)
{
  if (std::fabs(actual - expected) > tolerance)
  {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << '\n';
    std::exit(1);
  }
}

void require_true(bool value, const char* label)
{
  if (!value)
  {
    std::cerr << label << '\n';
    std::exit(1);
  }
}

void expect_unit_norm_abort()
{
  const pid_t pid = fork();
  if (pid < 0)
  {
    throw std::runtime_error("fork failed");
  }
  if (pid == 0)
  {
    std::vector<float> bad = {2.0f, 0.0f};
    Kmeans kmeans(2, 1, KmeansObjective::Anisotropic, 4.0f);
    kmeans.train(1, bad.data(), 1);
    _exit(0);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
  {
    throw std::runtime_error("waitpid failed");
  }
  require_true(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
               "anisotropic non-unit input must abort");
}
}  // namespace

int main()
{
  {
    const float x[] = {0.8f, 0.6f};
    const float c[] = {0.1f, 0.1f};
    require_close(anisotropic_loss(x, c, 4.0f, 2), 2.9588f, 1e-5f,
                  "anisotropic_loss hand check");
  }

  {
    Kmeans default_kmeans(2, 1);
    Kmeans explicit_standard(2, 1, KmeansObjective::Standard, 1.0f);
    require_true(default_kmeans.objective_ == KmeansObjective::Standard,
                 "default constructor must select standard objective");
    require_close(default_kmeans.eta_, 1.0f, 0.0f, "default eta");

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = data_a;
    default_kmeans.train(2, data_a.data(), 1);
    explicit_standard.train(2, data_b.data(), 1);
    require_true(default_kmeans.centroids_ == explicit_standard.centroids_,
                 "default and explicit standard centroids must match");
  }

  {
    std::vector<float> data = {1.0f, 0.0f, 0.0f, 1.0f};
    Kmeans kmeans(2, 1, KmeansObjective::Anisotropic, 4.0f);
    kmeans.train(2, data.data(), 1);
    require_close(kmeans.centroids_[0], 0.8f, 1e-5f,
                  "Theorem 4.2 centroid x");
    require_close(kmeans.centroids_[1], 0.8f, 1e-5f,
                  "Theorem 4.2 centroid y");
  }

  {
    Kmeans kmeans(2, 2, KmeansObjective::Anisotropic, 4.0f);
    kmeans.centroids_ = {0.0f, 0.0f, 0.9f, 0.4f};
    std::vector<float> data = {0.8f, 0.6f};
    std::vector<size_t> ids = {42};
    kmeans.add_anisotropic_exact(1, data.data(), ids);
    const float loss0 =
        anisotropic_loss(data.data(), kmeans.centroids_.data(), 4.0f, 2);
    const float loss1 =
        anisotropic_loss(data.data(), kmeans.centroids_.data() + 2, 4.0f, 2);
    const size_t expected = loss0 <= loss1 ? 0 : 1;
    require_true(kmeans.inverted_list_[expected].size() == 1,
                 "exact anisotropic assignment must pick argmin");
    require_true(kmeans.inverted_list_[expected][0] == 42,
                 "exact anisotropic assignment must preserve id");

    Kmeans verifier(2, 2, KmeansObjective::Anisotropic, 4.0f);
    verifier.centroids_ = kmeans.centroids_;
    std::vector<size_t> zero_based_ids = {0};
    verifier.add_anisotropic_exact(1, data.data(), zero_based_ids);
    verify_anisotropic_assignments_memory(verifier, data.data(), 1, 2,
                                          verifier.inverted_list_);
  }

  {
    Kmeans kmeans(2, 2, KmeansObjective::Standard, 1.0f);
    kmeans.centroids_ = {0.0f, 0.0f, 1.0f, 0.0f};
    hnswlib::L2Space space(2);
    hnswlib::HierarchicalNSW<float> graph(&space, 2, 20, 100);
    graph.addPoint(kmeans.centroids_.data(), 0);
    graph.addPoint(kmeans.centroids_.data() + 2, 1);

    std::vector<float> data = {0.01f, 0.0f};
    std::vector<size_t> ids = {7};
    kmeans.add_standard_prebuild_batch(1, data.data(), ids, graph, true);
    require_true(kmeans.inverted_list_[1].size() == 1,
                 "standard unit-normalized prebuild must assign normalized batch");
    require_true(kmeans.inverted_list_[1][0] == 7,
                 "standard unit-normalized assignment must preserve id");
  }

  {
    std::vector<float> standard_data = {
        1.0f, 0.0f,
        0.8f, 0.6f,
        0.0f, 1.0f,
        -0.6f, 0.8f,
    };
    std::vector<float> aniso_data = standard_data;
    Kmeans standard(2, 2, KmeansObjective::Standard, 1.0f);
    Kmeans aniso(2, 2, KmeansObjective::Anisotropic, 4.125f);
    standard.train(4, standard_data.data(), 5);
    aniso.train(4, aniso_data.data(), 5);

    float centroid_l1 = 0.0f;
    for (size_t i = 0; i < standard.centroids_.size(); ++i)
    {
      centroid_l1 += std::fabs(standard.centroids_[i] - aniso.centroids_[i]);
    }
    require_true(centroid_l1 > 1e-3f,
                 "anisotropic centroids must differ from standard centroids");
  }

  {
    std::vector<float> data = {
        1.0f, 0.0f,
        0.8f, 0.6f,
        0.0f, 1.0f,
        -0.6f, 0.8f,
    };
    Kmeans kmeans(2, 2, KmeansObjective::Anisotropic, 4.0f);
    kmeans.trace_training_loss_ = true;
    kmeans.train(4, data.data(), 10);
    require_true(kmeans.training_loss_trace_.size() == 10,
                 "anisotropic loss trace must record one value per iteration");
    for (size_t i = 1; i < kmeans.training_loss_trace_.size(); ++i)
    {
      require_true(kmeans.training_loss_trace_[i] <=
                       kmeans.training_loss_trace_[i - 1] + 1e-3f,
                   "anisotropic Lloyd loss must be monotone non-increasing");
    }
  }

  expect_unit_norm_abort();
  std::cout << "pg4 anisotropic kmeans C++ toy checks passed\n";
  return 0;
}
CPP

if pkg-config --exists openblas; then
  openblas_cflags="$(pkg-config --cflags openblas)"
  openblas_libs="$(pkg-config --libs openblas)"
else
  openblas_cflags="-I/usr/include/x86_64-linux-gnu/openblas-pthread"
  openblas_libs="-L/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas"
fi

g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
  -Ithird/liburing/src/include -Ithird/eigen -DNDEBUG -mtune=native -mavx2 -pthread \
  -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc \
  -fno-builtin-realloc -fno-builtin-free -fopenmp-simd -funroll-loops \
  -DEIGEN_USE_BLAS -DDJ_USE_OPENBLAS $openblas_cflags \
  -DUSE_AVX2 "$harness_cpp" build/third/liburing/libliburing_vendored.a \
  $openblas_libs \
  -o "$harness_bin"

OMP_NUM_THREADS=1 "$harness_bin"
python3 -m unittest experiments.PCA_enhancement.test_pg4_anisotropic_kmeans_bench
