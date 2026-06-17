"""Compute L2 range-search groundtruth for a *user-given* radius.

Same JSON output schema as compute_range_gt.py, except the radius is supplied
directly on the CLI instead of being derived from a k-NN groundtruth file.
Because there is no gt-derived radius, the fields that only existed to document
that derivation are omitted entirely: `k_reference`, `radius_percentile`, and
`provenance.gt100_path` (the `_schema` block drops them too).

Exactly one of --radius / --radius-squared is required (mutually exclusive):
  --radius R           plain L2 radius; radius_squared = R**2
  --radius-squared S   squared L2 threshold (a base vec hits iff L2² <= S);
                       radius_l2 = sqrt(S)
The script always compares in squared form; filenames use radius_l2 (the plain
L2 radius), matching compute_range_gt.py.

The heavy machinery (fbin parsing, BLAS brute-force range count, worker pool)
is reused verbatim from compute_range_gt.py so the counting logic has a single
source of truth; only the radius front-end differs.

Usage:
cd experiments/range_gt

# SSNPP range track: official threshold is the *squared* L2 value (e.g. 96237).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /md1/dongjiang_data/deep/deep100m_base.fbin \
--query /md1/dongjiang_data/deep/deep100m_query.fbin \
--radius-squared 0.02 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype uint8 \
--out-dir output \
--dataset-name deep100m

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /md1/dongjiang_data/deep/deep1m/deep1m_base.fbin \
--query /md1/dongjiang_data/deep/deep1m/deep1m_query.fbin \
--radius-squared 0.02 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 48 \
--input-dtype float32 \
--out-dir output \
--dataset-name deep1m

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /md1/dongjiang_data/deep/deep10m/deep10m_base.fbin \
--query /md1/dongjiang_data/deep/deep10m/deep10m_query.fbin \
--radius-squared 0.02 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 48 \
--input-dtype float32 \
--out-dir output \
--dataset-name deep10m


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /nvme0/dongjiang_data/deep/deep100m/deep100m_base.fbin \
--query /nvme0/dongjiang_data/deep/deep100m/deep100m_query.fbin \
--radius-squared 0.02 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype float32 \
--out-dir output \
--dataset-name deep100m


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_base.fbin \
--query /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_query.fbin \
--radius 22.5 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype int8 \
--out-dir output \
--dataset-name spacev100m

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_base.fbin \
--query /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_query.fbin \
--radius 30 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype int8 \
--out-dir output \
--dataset-name spacev100m


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /nvme0/dongjiang_data/ssnpp/ssnpp100m/ssnpp100m_base.fbin \
--query /nvme0/dongjiang_data/ssnpp/ssnpp100m/ssnpp100m_query.fbin \
--radius 206 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype uint8 \
--out-dir output \
--dataset-name ssnpp100m

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /nvme0/dongjiang_data/ssnpp/ssnpp100m/ssnpp100m_base.fbin \
--query /nvme0/dongjiang_data/ssnpp/ssnpp100m/ssnpp100m_query.fbin \
--radius 465 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype uint8 \
--out-dir output \
--dataset-name ssnpp100m



OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt_fixed_radius.py \
--base /data/raid0/dongjiang_data/sift/sift100m/sift100m_base.fbin \
--query /data/raid0/dongjiang_data/sift/sift100m/sift100m_query.fbin \
--radius-squared 10000 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype uint8 \
--out-dir output \
--dataset-name sift100m


# Or give the plain L2 radius directly:
#   --radius 310.2212
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

# Reuse the counting core + fbin loaders from the sibling script (single source
# of truth). `uv run <script>` puts the script's own directory on sys.path[0],
# so this import resolves regardless of the caller's cwd. If it ever fails, let
# the ImportError surface loudly rather than silently re-implementing.
from compute_range_gt import (
    active_worker_count,
    count_in_range,
    dtype_name,
    estimate_temp_gib,
    load_fbin,
    prepare_output_dir,
)


@dataclass(frozen=True)
class Args:
    base: pathlib.Path
    query: pathlib.Path
    out_dir: pathlib.Path
    radius_squared: float
    radius_l2: float
    batch_size: int
    query_batch_size: int
    workers: int
    dataset_name: str
    index_prefix: str
    input_dtype: str  # "auto" | "float32" | "int8" | "uint8"


def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        description="Brute-force L2 range-search groundtruth for a fixed, "
                    "user-given radius (total hit count + per-query counts)."
    )
    p.add_argument("--base", type=pathlib.Path, required=True,
                   help="Path to base .fbin (legacy float32 or v2 float32/int8/uint8).")
    p.add_argument("--query", type=pathlib.Path, required=True,
                   help="Path to query .fbin (legacy float32 or v2 float32/int8/uint8).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Output directory; file will be {dataset_name}_{radius_l2:.4f}_gt.json.")

    radius_group = p.add_mutually_exclusive_group(required=True)
    radius_group.add_argument("--radius", type=float, default=None,
                              help="Plain L2 radius; internally squared "
                                   "(radius_squared = radius**2). Mutually "
                                   "exclusive with --radius-squared.")
    radius_group.add_argument("--radius-squared", type=float, default=None,
                              dest="radius_squared",
                              help="Squared L2 threshold; a base vector hits "
                                   "iff L2² <= this value. radius_l2 = sqrt(it) "
                                   "is used for filenames. Mutually exclusive "
                                   "with --radius.")

    p.add_argument("--batch-size", type=int, default=200_000,
                   help="Base rows per batch for the brute-force L2 range count.")
    p.add_argument("--query-batch-size", type=int, default=256,
                   help="Query rows per GEMM block; lowers temporary matrix memory.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes for base-batch counting.")
    p.add_argument("--dataset-name", type=str, default="gist1m",
                   help="Label written into the JSON metadata.")
    p.add_argument("--index-prefix", type=str, default=None,
                   help="Full index path prefix written to index.prefix; defaults to <base-dir>/<dataset-name>.")
    p.add_argument("--input-dtype", type=str, default="auto",
                   choices=("auto", "float32", "int8", "uint8"),
                   help="Override fbin payload dtype for legacy v1 (no-magic) "
                        "headers, where the file does not self-describe its "
                        "dtype. 'auto' (default): v2 magic uses the v2 dtype "
                        "byte; v1 falls back to float32. Explicit value applies "
                        "to both base and query (they must share the same "
                        "dtype); for v2-magic files it must match the v2 dtype "
                        "byte or the script rejects the input.")
    ns = p.parse_args(argv)
    if ns.batch_size <= 0:
        p.error("--batch-size must be positive")
    if ns.query_batch_size <= 0:
        p.error("--query-batch-size must be positive")
    if ns.workers <= 0:
        p.error("--workers must be positive")

    if ns.radius is not None:
        if not np.isfinite(ns.radius) or ns.radius <= 0.0:
            p.error("--radius must be a positive, finite number")
        radius_l2 = float(ns.radius)
        radius_squared = radius_l2 * radius_l2
    else:
        if not np.isfinite(ns.radius_squared) or ns.radius_squared <= 0.0:
            p.error("--radius-squared must be a positive, finite number")
        radius_squared = float(ns.radius_squared)
        radius_l2 = float(np.sqrt(radius_squared))

    return Args(
        base=ns.base, query=ns.query, out_dir=ns.out_dir,
        radius_squared=radius_squared, radius_l2=radius_l2,
        batch_size=ns.batch_size, query_batch_size=ns.query_batch_size,
        workers=ns.workers, dataset_name=ns.dataset_name,
        index_prefix=ns.index_prefix if ns.index_prefix is not None
        else str(ns.base.resolve().parent / ns.dataset_name),
        input_dtype=ns.input_dtype,
    )


def build_schema() -> dict:
    return {
        "description": (
            "L2 range-search groundtruth for a user-given radius. For each query "
            "we computed its L2 distance to every base vector and counted how "
            "many fall within the L2 radius (radius_l2). The radius is supplied "
            "directly on the CLI (--radius or --radius-squared), NOT derived from "
            "a k-NN groundtruth file; therefore k_reference / radius_percentile / "
            "provenance.gt100_path are intentionally absent. Both the grand total "
            "and the per-query counts are persisted."
        ),
        "fields": {
            "dataset":           "Dataset label (CLI --dataset-name).",
            "metric":            "Distance metric (fixed: l2). Internally compared in squared form for performance.",
            "n_queries":         "Number of query vectors.",
            "n_base":            "Number of base vectors.",
            "dim":               "Vector dimensionality.",
            "vec_dtype":         "Authoritative payload dtype for both base and query .fbin (one of 'float32' / 'int8' / 'uint8'). Downstream DiskRange build/search reads this field to skip relying on per-config vec_dtype.",
            "radius_squared":    "Threshold actually compared against; a base vector hits iff L2² <= this value. From --radius-squared, or (--radius)².",
            "radius_l2":         "Plain L2 radius (= sqrt(radius_squared)); used for filenames and external presentation. From --radius, or sqrt(--radius-squared).",
            "total_in_range":    "STAR: sum over all queries of the number of base vectors with L2 distance <= radius_l2.",
            "per_query_counts":  "List of length n_queries; entry i = number of base vectors with L2 distance <= radius_l2 for query i (order matches the query .fbin).",
            "per_query_count_stats": {
                "min":    "Minimum per-query hit count.",
                "median": "Median per-query hit count (float; np.percentile 50).",
                "max":    "Maximum per-query hit count.",
                "zero_hit_queries": "Number of queries with zero hits.",
            },
            "index": {
                "prefix": "Full index path prefix. Scripts append config_local.sh build parameters to derive concrete index paths.",
            },
            "provenance": {
                "base_path":  "Absolute path of input base .fbin.",
                "query_path": "Absolute path of input query .fbin.",
                "base_dtype": "Vector dtype loaded from base .fbin (float32, int8, or uint8).",
                "query_dtype": "Vector dtype loaded from query .fbin (float32, int8, or uint8).",
                "batch_size": "GEMM batch size over base rows.",
                "query_batch_size": "GEMM batch size over query rows; bounds temporary matrix memory.",
                "workers": "Requested worker process count (legacy alias for workers_requested).",
                "workers_requested": "Requested worker process count from CLI --workers.",
                "workers_active": "Actual worker process count used after capping to the number of base batches.",
                "runtime_seconds": "Wall-clock seconds for the count step.",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prepare_output_dir(args.out_dir)

    print(f"[1/3] Loading base  : {args.base} (input_dtype={args.input_dtype})")
    t = time.perf_counter()
    base = load_fbin(args.base, args.input_dtype)
    print(f"      -> shape={base.shape} dtype={base.dtype}  ({time.perf_counter()-t:.2f}s)")

    print(f"[2/3] Loading query : {args.query} (input_dtype={args.input_dtype})")
    t = time.perf_counter()
    query = load_fbin(args.query, args.input_dtype)
    print(f"      -> shape={query.shape} dtype={query.dtype}  ({time.perf_counter()-t:.2f}s)")

    print(
        f"\nRadius (user-given): "
        f"radius_l2 = {args.radius_l2:.6f}  "
        f"(radius_squared = {args.radius_squared:.6f})"
    )

    estimated_temp = estimate_temp_gib(args.batch_size, args.query_batch_size)
    workers_active = active_worker_count(base.shape[0], args.batch_size, args.workers)
    print(
        f"\n[3/3] Brute-force range count "
        f"(batch_size={args.batch_size}, query_batch_size={args.query_batch_size}, "
        f"workers_requested={args.workers}, workers_active={workers_active}, "
        f"estimated_temp_per_worker>={estimated_temp:.3f} GiB)..."
    )
    t_count = time.perf_counter()
    total, per_query = count_in_range(
        query,
        base,
        args.radius_squared,
        args.batch_size,
        args.query_batch_size,
        args.workers,
        args.input_dtype,
    )
    runtime = time.perf_counter() - t_count

    pq = per_query
    stats = {
        "min":              int(pq.min()),
        "median":           float(np.median(pq)),
        "max":              int(pq.max()),
        "zero_hit_queries": int((pq == 0).sum()),
    }
    print(
        f"\nPer-query count distribution:\n"
        f"  min    = {stats['min']}\n"
        f"  median = {stats['median']}\n"
        f"  max    = {stats['max']}\n"
        f"  zero   = {stats['zero_hit_queries']}\n"
        f"  total_in_range = {total}\n"
        f"  count runtime  = {runtime:.2f}s"
    )

    base_dtype_name = dtype_name(base)
    query_dtype_name = dtype_name(query)
    if base_dtype_name != query_dtype_name:
        raise SystemExit(
            f"base/query dtype mismatch: base={base_dtype_name} "
            f"query={query_dtype_name}; --input-dtype applies to both"
        )

    payload = {
        "_schema": build_schema(),
        "dataset":           args.dataset_name,
        "metric":            "l2",
        "n_queries":         int(query.shape[0]),
        "n_base":            int(base.shape[0]),
        "dim":               int(base.shape[1]),
        "vec_dtype":         base_dtype_name,
        "radius_squared":    args.radius_squared,
        "radius_l2":         args.radius_l2,
        "total_in_range":    total,
        "per_query_counts":   per_query.tolist(),
        "per_query_count_stats": stats,
        "index": {
            "prefix": args.index_prefix,
        },
        "provenance": {
            "base_path":       str(args.base.resolve()),
            "query_path":      str(args.query.resolve()),
            "base_dtype":       base_dtype_name,
            "query_dtype":      query_dtype_name,
            "batch_size":      args.batch_size,
            "query_batch_size": args.query_batch_size,
            "workers":         args.workers,
            "workers_requested": args.workers,
            "workers_active":   workers_active,
            "runtime_seconds": runtime,
        },
    }

    out_path = args.out_dir / f"{args.dataset_name}_{args.radius_l2:.4f}_gt.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
