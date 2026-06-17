# DiskRange

DiskRange is a disk-resident range-search engine for high-dimensional vectors.
It combines an IVF partitioning, RaBitQ quantization, a SymphonyQG-style
emptiness oracle, and `io_uring` asynchronous I/O to answer radius queries while
keeping most of the data on disk.

## Building

Requirements:

- A C++17 compiler with OpenMP support.
- CMake.
- An x86-64 CPU with AVX2. AVX-512 VNNI is enabled by default for the int8
  distance kernels; on hardware or toolchains without it, configure with
  `-DDJ_ENABLE_AVX512VNNI=OFF`.
- OpenBLAS is optional (used to route Eigen GEMM through BLAS when found;
  toggle with `-DDJ_USE_OPENBLAS`).

All other dependencies (liburing, Eigen, nlohmann/json, RaBitQ-Library,
symqglib) are vendored under `third/`.

```shell
mkdir build && cd build
cmake ..
make -j
```

## Ground truth

Range-search ground truth is computed by the scripts under
`experiments/range_gt/` (managed with [uv](https://docs.astral.sh/uv/)):

```shell
cd experiments/range_gt
uv run compute_range_gt_fixed_radius.py --help
```

Each run produces a `*_gt.json` file consumed by the search binaries. One
example fixture, `experiments/range_gt/output/deep1m_0.2632_gt.json`, ships
with the repository.

## Running

Index parameters live in `configs/`. The `scripts/a_run_*.sh` helpers wire a
config and a ground-truth file together; the dataset is selected once in
`scripts/_dataset.sh` (defaulting to the bundled deep1m example) and can be
overridden per call:

```shell
bash scripts/a_run_indexing.sh                       # build an index
bash scripts/a_run_search.sh                         # search with defaults
bash scripts/a_run_search.sh <config> <gt_json>      # explicit config / GT
```

The base and query vectors themselves are not distributed; point the config at
your own `.fbin` files.

## Contributions

Querying runs a three-layer in-memory cascade inside `DiskRange::search()`
(`lib/DiskRange.h`); the matching index structures are built in
`DiskRange::build()`. Each layer prunes more candidates before any disk read.

1. **Query Classifier (Section IV)** — drops empty-result queries in memory.
   Mainly `lib/FastPathEmptinessOracle.h` (the `fast_path_*` oracle), with the
   centroid graph in `lib/SQGCentroidIndex.h` and the 1-bit RaBitQ codes in
   `lib/RBQCodeStorage.h`.
2. **Cluster Pruner (Section V)** — prunes whole clusters with the PCA-box and
   slab bounds. The bounds (`pca_lower_bound` / `pca_slab_lower_bound`) and their
   index-time fitting live in `lib/ClusterIO.h`, the local PCA in `lib/PcaFit.h`.
3. **I/O Optimizer (Section VI)** — reads only the surviving cells via `io_uring`
   + `O_DIRECT`. Cell splitting in `lib/PcaFit.h`, the per-cell bound
   `cell_lb_batch` in `utils/dist_func.h`, and the streaming reader
   `submit_and_drain` in `lib/ClusterIO.h`.

