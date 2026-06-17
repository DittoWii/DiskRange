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

## Optional Data HNSW Emptiness Oracle

DiskRange supports an opt-in raw-data HNSW emptiness oracle for range search.
It is disabled by default; existing configs produce no `_data_hnsw_file.bin`
and search does not try to read it.

To enable it, add `data_hnsw_enabled true` to the build/search config.  The
defaults are `data_hnsw_M 16`, `data_hnsw_ef_construction 100`,
`data_hnsw_ef 100`, and `data_hnsw_margin_delta 0.0`.  The artifact path
is derived from the GT JSON `index.prefix` as
`<index_prefix>_diskrange/_data_hnsw_file.bin`, matching the existing
DiskRange artifact paths.

## License

DiskRange is distributed under the GNU Affero General Public License v3.0; see
`LICENSE`. It bundles third-party components under `third/` (including an
AGPL-3.0 component derived from Intel Scalable Vector Search, which is why the
combined work is AGPL-3.0). See `THIRD_PARTY_LICENSES.md` for the full
inventory and per-component licenses.
