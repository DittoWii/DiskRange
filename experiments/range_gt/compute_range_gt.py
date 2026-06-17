"""Compute L2 range-search groundtruth total hit count for a DiskANN-format dataset.

Input:
    --base     .fbin  (legacy float32 or v2 float32/int8/uint8 base vectors)
    --query    .fbin  (legacy float32 or v2 float32/int8/uint8 query vectors)
    --gt100    .bin   (uint32 k-NN ids-only, DiskANN layout [npts][k][ids])
Output:
    <out-dir>/{dataset_name}_{radius_l2:.4f}_gt.json

    
cd experiments/range_gt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt.py \
--base /md1/dongjiang_data/sift/sift100m/sift100m_base.fbin \
--query /md1/dongjiang_data/sift/sift100m/sift100m_query.fbin \
--gt100 /md1/dongjiang_data/sift/sift100m/sift100m_gt100.fbin \
--k 1 --percentile 15 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 90 \
--input-dtype int8 \
--out-dir output \
--dataset-name sift100m && \


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt.py \
--base /md1/dongjiang_data/deep/deep1b/deep1b_base.fbin \
--query /md1/dongjiang_data/deep/deep1b/deep1b_query.fbin \
--gt100 /md1/dongjiang_data/deep/deep1b/deep1b_gt100.fbin \
--k 1 --percentile 5 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 96 \
--input-dtype float32 \
--out-dir output \
--dataset-name deep1b


OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run compute_range_gt.py \
--base /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_base.fbin \
--query /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_query.fbin \
--gt100 /nvme0/dongjiang_data/spacev/spacev100m/spacev100m_gt100.fbin \
--k 1 --percentile 13 \
--batch-size 200000 \
--query-batch-size 256 \
--workers 190 \
--input-dtype int8 \
--out-dir output \
--dataset-name spacev100m


"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np


type FloatArray = np.ndarray
type IdArray = np.ndarray

FBIN_MAGIC_DJB1 = 0x31424A44
FBIN_DTYPE_FLOAT32 = 0
FBIN_DTYPE_INT8 = 1
FBIN_DTYPE_UINT8 = 2


@dataclass(frozen=True)
class FbinLayout:
    npts: int
    dim: int
    dtype: int
    payload_offset: int
    numpy_dtype: np.dtype
    bytes_per_element: int


@dataclass(frozen=True)
class Args:
    base: pathlib.Path
    query: pathlib.Path
    gt100: pathlib.Path
    out_dir: pathlib.Path
    k: int
    percentile: float
    batch_size: int
    query_batch_size: int
    workers: int
    dataset_name: str
    index_prefix: str
    input_dtype: str  # "auto" | "float32" | "int8" | "uint8"


def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(
        description="Brute-force L2 range-search groundtruth (total hit count)."
    )
    p.add_argument("--base", type=pathlib.Path, required=True,
                   help="Path to base .fbin (legacy float32 or v2 float32/int8/uint8).")
    p.add_argument("--query", type=pathlib.Path, required=True,
                   help="Path to query .fbin (legacy float32 or v2 float32/int8/uint8).")
    p.add_argument("--gt100", type=pathlib.Path, required=True,
                   help="Path to k-NN groundtruth ids .bin ([npts][k][uint32 ids]).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Output directory; file will be {dataset_name}_{radius_l2:.4f}_gt.json.")
    p.add_argument("--k", type=int, default=50,
                   help="k-th nearest neighbor position used for radius (1-indexed).")
    p.add_argument("--percentile", type=float, default=50.0,
                   help="Percentile over queries of the k-NN L2 distance used as radius; 50 = median.")
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
                        "dtype. 'auto' (default) keeps the existing behavior: "
                        "v2 magic uses the v2 dtype byte; v1 falls back to "
                        "float32. Explicit 'float32' / 'int8' / 'uint8' "
                        "applies to both base and query files (they must share the same "
                        "dtype). For v2-magic files, the explicit value must "
                        "match the v2 dtype byte or the script rejects the "
                        "input.")
    ns = p.parse_args(argv)
    if ns.batch_size <= 0:
        p.error("--batch-size must be positive")
    if ns.query_batch_size <= 0:
        p.error("--query-batch-size must be positive")
    if ns.workers <= 0:
        p.error("--workers must be positive")
    return Args(
        base=ns.base, query=ns.query, gt100=ns.gt100, out_dir=ns.out_dir,
        k=ns.k, percentile=ns.percentile, batch_size=ns.batch_size,
        query_batch_size=ns.query_batch_size, workers=ns.workers,
        dataset_name=ns.dataset_name,
        index_prefix=ns.index_prefix if ns.index_prefix is not None else str(ns.base.resolve().parent / ns.dataset_name),
        input_dtype=ns.input_dtype,
    )


_INPUT_DTYPE_TO_FBIN: dict[str, int] = {
    "float32": FBIN_DTYPE_FLOAT32,
    "int8": FBIN_DTYPE_INT8,
    "uint8": FBIN_DTYPE_UINT8,
}


def _resolve_numpy_dtype(fbin_dtype: int, path: pathlib.Path) -> np.dtype:
    if fbin_dtype == FBIN_DTYPE_FLOAT32:
        return np.dtype(np.float32)
    if fbin_dtype == FBIN_DTYPE_INT8:
        return np.dtype(np.int8)
    if fbin_dtype == FBIN_DTYPE_UINT8:
        return np.dtype(np.uint8)
    raise ValueError(f"{path}: unsupported .fbin dtype {fbin_dtype}")


def read_fbin_layout(
    path: pathlib.Path, input_dtype: str = "auto"
) -> FbinLayout:
    """Parse the fbin header.

    `input_dtype`:
      - "auto" (default): v2 magic uses the v2 dtype byte; v1 falls back to
        float32 (legacy behavior, fully backward compatible).
      - "float32" / "int8" / "uint8": override applies to the v1 fallback path so the
        payload size is computed for that element width. For v2-magic files
        the explicit value MUST match the v2 dtype byte; otherwise this is a
        user-input mismatch and we reject.
    """
    if input_dtype not in ("auto", "float32", "int8", "uint8"):
        raise ValueError(f"invalid input_dtype: {input_dtype!r}")

    actual = path.stat().st_size
    with path.open("rb") as f:
        header_bytes = f.read(16)
    if len(header_bytes) < 8:
        raise ValueError(f"{path}: failed to read .fbin header")

    first, _ = [int(v) for v in np.frombuffer(header_bytes[:8], dtype=np.uint32)]
    if first == FBIN_MAGIC_DJB1:
        if len(header_bytes) < 16:
            raise ValueError(f"{path}: failed to read v2 .fbin header")
        _, dtype, npts, dim = [
            int(v) for v in np.frombuffer(header_bytes[:16], dtype=np.uint32)
        ]
        numpy_dtype = _resolve_numpy_dtype(dtype, path)
        payload_offset = 16
        if input_dtype != "auto":
            requested = _INPUT_DTYPE_TO_FBIN[input_dtype]
            if requested != dtype:
                raise ValueError(
                    f"{path}: --input-dtype={input_dtype} conflicts with v2 "
                    f"self-described dtype={dtype}; either drop --input-dtype "
                    f"to use the v2 self-described value, or pass an "
                    f"--input-dtype matching the file"
                )
    else:
        npts = int(np.frombuffer(header_bytes[:4], dtype=np.int32)[0])
        dim = int(np.frombuffer(header_bytes[4:8], dtype=np.int32)[0])
        if input_dtype == "auto":
            dtype = FBIN_DTYPE_FLOAT32
        else:
            dtype = _INPUT_DTYPE_TO_FBIN[input_dtype]
        numpy_dtype = _resolve_numpy_dtype(dtype, path)
        payload_offset = 8

    if npts <= 0 or dim <= 0:
        raise ValueError(f"{path}: non-positive .fbin shape ({npts},{dim})")
    expected = payload_offset + npts * dim * numpy_dtype.itemsize
    if actual != expected:
        raise ValueError(
            f"{path}: size {actual} != expected {expected} for "
            f"({npts},{dim}) with dtype={numpy_dtype.name} "
            f"(input_dtype={input_dtype}); if this is a v1 fbin with byte "
            f"payload, pass --input-dtype int8 or --input-dtype uint8"
        )
    return FbinLayout(
        npts=npts,
        dim=dim,
        dtype=dtype,
        payload_offset=payload_offset,
        numpy_dtype=numpy_dtype,
        bytes_per_element=numpy_dtype.itemsize,
    )


def read_fbin_header(
    path: pathlib.Path, input_dtype: str = "auto"
) -> tuple[int, int]:
    layout = read_fbin_layout(path, input_dtype)
    return layout.npts, layout.dim


def dtype_name(data: np.ndarray) -> str:
    if data.dtype == np.dtype(np.float32):
        return "float32"
    if data.dtype == np.dtype(np.int8):
        return "int8"
    if data.dtype == np.dtype(np.uint8):
        return "uint8"
    return str(data.dtype)


def row_norms(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        values = data.astype(np.int64, copy=False)
        return np.sum(values * values, axis=1, dtype=np.int64)
    return np.einsum("ij,ij->i", data, data, dtype=np.float64)


def load_fbin(
    path: pathlib.Path, input_dtype: str = "auto"
) -> FloatArray:
    layout = read_fbin_layout(path, input_dtype)
    return np.memmap(
        path,
        dtype=layout.numpy_dtype,
        mode="r",
        offset=layout.payload_offset,
        shape=(layout.npts, layout.dim),
    )


def prepare_output_dir(out_dir: pathlib.Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=out_dir, prefix=".compute_range_gt_write_test_", delete=True
        ):
            pass
    except OSError as exc:
        raise SystemExit(
            f"error: output directory is not writable or cannot be created: "
            f"{out_dir} ({exc})"
        )


def load_gt_ids(path: pathlib.Path) -> IdArray:
    actual = path.stat().st_size
    with path.open("rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        npts, k = int(header[0]), int(header[1])
        size_ids_only  = 8 + npts * k * 4
        size_ids_dists = 8 + npts * k * 4 * 2
        if actual == size_ids_only:
            layout = "ids-only"
        elif actual == size_ids_dists:
            layout = "ids+float32_dists"
        else:
            raise ValueError(
                f"{path}: size {actual} matches neither ids-only ({size_ids_only}) "
                f"nor ids+dists ({size_ids_dists}) for header ({npts},{k})"
            )
        ids = np.frombuffer(f.read(npts * k * 4), dtype=np.uint32)
    print(f"      -> gt layout: {layout}  (dists segment, if present, is ignored)")
    return ids.reshape(npts, k)


def pick_radius_squared(
    query: FloatArray, base: FloatArray, gt_ids: IdArray, k: int, percentile: float,
) -> tuple[float, FloatArray]:
    if not (1 <= k <= gt_ids.shape[1]):
        raise ValueError(f"k={k} must be in [1, {gt_ids.shape[1]}]")
    if query.shape[0] != gt_ids.shape[0]:
        raise ValueError(
            f"query npts ({query.shape[0]}) != gt npts ({gt_ids.shape[0]})"
        )
    neigh_ids = gt_ids[:, k - 1].astype(np.int64)
    if neigh_ids.max() >= base.shape[0]:
        raise ValueError(f"gt id {int(neigh_ids.max())} out of base range {base.shape[0]}")
    if query.dtype != base.dtype:
        raise ValueError(f"base/query dtype mismatch: {base.dtype} vs {query.dtype}")
    if np.issubdtype(query.dtype, np.integer):
        diff = query.astype(np.int64, copy=False) - base[neigh_ids].astype(np.int64, copy=False)
        dist_sq = np.sum(diff * diff, axis=1, dtype=np.int64).astype(np.float64)
    else:
        diff = query.astype(np.float64, copy=False) - base[neigh_ids].astype(np.float64, copy=False)
        dist_sq = np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
    radius_sq = float(np.percentile(dist_sq, percentile))
    return radius_sq, dist_sq


_WORKER_BASE: FloatArray | None = None
_WORKER_QUERY: FloatArray | None = None
_WORKER_Q_NORM: FloatArray | None = None
_WORKER_RADIUS_SQ: float = 0.0
_WORKER_QUERY_BATCH_SIZE: int = 0


def iter_base_batches(nb: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, nb, batch_size):
        yield start, min(start + batch_size, nb)


def base_batch_count(nb: int, batch_size: int) -> int:
    if nb <= 0:
        return 0
    return (nb + batch_size - 1) // batch_size


def active_worker_count(nb: int, batch_size: int, workers: int) -> int:
    return min(workers, base_batch_count(nb, batch_size))


def estimate_temp_gib(batch_size: int, query_batch_size: int) -> float:
    # One float32 distance block plus one bool comparison block. BLAS may use
    # additional internal scratch, so this is a lower-bound estimate.
    bytes_per_worker = batch_size * query_batch_size * (np.dtype(np.float32).itemsize + np.dtype(bool).itemsize)
    return bytes_per_worker / (1024.0 ** 3)


def count_base_batch(
    query: FloatArray,
    base: FloatArray,
    q_norm: FloatArray,
    radius_sq: float,
    query_batch_size: int,
    start: int,
    end: int,
) -> IdArray:
    nq = query.shape[0]
    partial = np.zeros(nq, dtype=np.int64)
    b_slice = base[start:end]
    b_norm = row_norms(b_slice)
    # The dot product MUST go through BLAS. NumPy *integer* matmul (int64 @
    # int64) does NOT dispatch to BLAS -- it falls back to a generic ~10x-slower
    # loop, which made int8/uint8 runs at 100M base look hung. Casting int8/uint8
    # to float for the matmul is BIT-EXACT: float32 represents integers exactly
    # up to 2**24, and the largest |dot| for uint8 is dim*255*255 (< 2**24 for
    # dim <= 257, covering sift/spacev/ssnpp), so dist_sq stays exact (2*dot is
    # even -> exact to 2**25). Larger dims fall back to float64 (exact to 2**53).
    # Float inputs keep float64 unchanged, preserving previously generated GT.
    # The cast is hoisted out of the query loop (was redundantly recomputed per
    # query block).
    is_integer = (
        np.issubdtype(query.dtype, np.integer) or
        np.issubdtype(b_slice.dtype, np.integer)
    )
    if is_integer:
        matmul_dtype = (
            np.float32 if b_slice.shape[1] * 255 * 255 < (1 << 24) else np.float64
        )
    else:
        matmul_dtype = np.float64
    b_block = b_slice.astype(matmul_dtype, copy=False)
    for q_start in range(0, nq, query_batch_size):
        q_end = min(q_start + query_batch_size, nq)
        q_block = query[q_start:q_end].astype(matmul_dtype, copy=False)
        dist_sq = q_block @ b_block.T
        dist_sq *= -2.0
        dist_sq += q_norm[q_start:q_end, None]
        dist_sq += b_norm[None, :]
        partial[q_start:q_end] += np.count_nonzero(dist_sq <= radius_sq, axis=1)
    return partial


def init_count_worker(
    base_path: str,
    query_path: str,
    radius_sq: float,
    query_batch_size: int,
    input_dtype: str = "auto",
) -> None:
    global _WORKER_BASE, _WORKER_QUERY, _WORKER_Q_NORM
    global _WORKER_RADIUS_SQ, _WORKER_QUERY_BATCH_SIZE
    _WORKER_BASE = load_fbin(pathlib.Path(base_path), input_dtype)
    _WORKER_QUERY = load_fbin(pathlib.Path(query_path), input_dtype)
    _WORKER_Q_NORM = row_norms(_WORKER_QUERY)
    _WORKER_RADIUS_SQ = radius_sq
    _WORKER_QUERY_BATCH_SIZE = query_batch_size


def count_base_batch_worker(start: int, end: int) -> tuple[int, int, IdArray]:
    if _WORKER_BASE is None or _WORKER_QUERY is None or _WORKER_Q_NORM is None:
        raise RuntimeError("count worker was not initialized")
    partial = count_base_batch(
        _WORKER_QUERY,
        _WORKER_BASE,
        _WORKER_Q_NORM,
        _WORKER_RADIUS_SQ,
        _WORKER_QUERY_BATCH_SIZE,
        start,
        end,
    )
    return start, end, partial


def count_in_range(
    query: FloatArray,
    base: FloatArray,
    radius_sq: float,
    batch_size: int,
    query_batch_size: int,
    workers: int,
    input_dtype: str = "auto",
) -> tuple[int, IdArray]:
    nq, dim = query.shape
    nb = base.shape[0]
    if base.shape[1] != dim:
        raise ValueError(f"dim mismatch: query {dim} vs base {base.shape[1]}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if query.dtype != base.dtype:
        raise ValueError(f"base/query dtype mismatch: {base.dtype} vs {query.dtype}")
    q_norm = row_norms(query)
    per_query = np.zeros(nq, dtype=np.int64)
    t0 = time.perf_counter()
    if workers == 1:
        for start, end in iter_base_batches(nb, batch_size):
            per_query += count_base_batch(
                query, base, q_norm, radius_sq, query_batch_size, start, end
            )
            elapsed = time.perf_counter() - t0
            pct = 100.0 * end / nb
            print(
                f"  [{elapsed:6.1f}s] base {start:>8d}..{end:<8d}  "
                f"({pct:5.1f}%)  cumulative_hits={int(per_query.sum()):>12d}"
            )
    else:
        base_path = getattr(base, "filename", None)
        query_path = getattr(query, "filename", None)
        if base_path is None or query_path is None:
            raise ValueError("--workers > 1 requires .fbin inputs loaded as memmap")
        active_workers = active_worker_count(nb, batch_size, workers)
        if active_workers == 0:
            return 0, per_query
        completed = 0
        batch_iter = iter_base_batches(nb, batch_size)
        futures = set()

        def submit_next(pool: ProcessPoolExecutor) -> bool:
            try:
                start, end = next(batch_iter)
            except StopIteration:
                return False
            futures.add(pool.submit(count_base_batch_worker, start, end))
            return True

        with ProcessPoolExecutor(
            max_workers=active_workers,
            initializer=init_count_worker,
            initargs=(str(base_path), str(query_path), radius_sq,
                      query_batch_size, input_dtype),
        ) as pool:
            for _ in range(active_workers * 2):
                if not submit_next(pool):
                    break
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    start, end, partial = future.result()
                    per_query += partial
                    completed += end - start
                    elapsed = time.perf_counter() - t0
                    pct = 100.0 * completed / nb
                    print(
                        f"  [{elapsed:6.1f}s] base {start:>8d}..{end:<8d} done "
                        f"({pct:5.1f}%)  cumulative_hits={int(per_query.sum()):>12d}"
                    )
                    submit_next(pool)
    return int(per_query.sum()), per_query


def build_schema() -> dict:
    return {
        "description": (
            "L2 range-search groundtruth. For each query we computed its L2 distance "
            "to every base vector and counted how many fall within the L2 radius (radius_l2). "
            "Both the grand total and the per-query counts are persisted; per-query counts are also printed to stdout."
        ),
        "fields": {
            "dataset":           "Dataset label (CLI --dataset-name).",
            "metric":            "Distance metric (fixed: l2). Internally compared in squared form for performance.",
            "n_queries":         "Number of query vectors.",
            "n_base":            "Number of base vectors.",
            "dim":               "Vector dimensionality.",
            "vec_dtype":         "Authoritative payload dtype for both base and query .fbin (one of 'float32' / 'int8' / 'uint8'). Downstream DiskRange build/search reads this field to skip relying on per-config vec_dtype.",
            "k_reference":       "k-th NN position whose distance drives radius selection (CLI --k).",
            "radius_percentile": "Percentile (over queries) of the k-NN L2 distances (CLI --percentile).",
            "radius_squared":    "Threshold actually compared against; a base vector hits iff L2² <= this value.",
            "radius_l2":         "Plain L2 radius (= sqrt(radius_squared)); used for filenames and external presentation.",
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
                "gt100_path": "Absolute path of k-NN ids file used to pick radius.",
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

    print(f"[1/4] Loading base  : {args.base} (input_dtype={args.input_dtype})")
    t = time.perf_counter()
    base = load_fbin(args.base, args.input_dtype)
    print(f"      -> shape={base.shape} dtype={base.dtype}  ({time.perf_counter()-t:.2f}s)")

    print(f"[2/4] Loading query : {args.query} (input_dtype={args.input_dtype})")
    t = time.perf_counter()
    query = load_fbin(args.query, args.input_dtype)
    print(f"      -> shape={query.shape} dtype={query.dtype}  ({time.perf_counter()-t:.2f}s)")

    print(f"[3/4] Loading gt100 : {args.gt100}")
    t = time.perf_counter()
    gt_ids = load_gt_ids(args.gt100)
    print(f"      -> shape={gt_ids.shape} dtype={gt_ids.dtype}  ({time.perf_counter()-t:.2f}s)")

    radius_sq, kth_dist_sq = pick_radius_squared(
        query, base, gt_ids, args.k, args.percentile
    )
    print(
        f"\nRadius selection (k={args.k}, percentile={args.percentile}):\n"
        f"  {len(kth_dist_sq)} k-th NN L2 values: "
        f"min={np.sqrt(kth_dist_sq.min()):.6f}  "
        f"median={np.sqrt(np.median(kth_dist_sq)):.6f}  "
        f"max={np.sqrt(kth_dist_sq.max()):.6f}\n"
        f"  -> radius (L2) = {np.sqrt(radius_sq):.6f}  (radius_squared = {radius_sq:.6f})"
    )

    estimated_temp = estimate_temp_gib(args.batch_size, args.query_batch_size)
    workers_active = active_worker_count(base.shape[0], args.batch_size, args.workers)
    print(
        f"\n[4/4] Brute-force range count "
        f"(batch_size={args.batch_size}, query_batch_size={args.query_batch_size}, "
        f"workers_requested={args.workers}, workers_active={workers_active}, "
        f"estimated_temp_per_worker>={estimated_temp:.3f} GiB)..."
    )
    t_count = time.perf_counter()
    total, per_query = count_in_range(
        query,
        base,
        radius_sq,
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
        "k_reference":       args.k,
        "radius_percentile": args.percentile,
        "radius_squared":    radius_sq,
        "radius_l2":         float(np.sqrt(radius_sq)),
        "total_in_range":    total,
        "per_query_counts":   per_query.tolist(),
        "per_query_count_stats": stats,
        "index": {
            "prefix": args.index_prefix,
        },
        "provenance": {
            "base_path":       str(args.base.resolve()),
            "query_path":      str(args.query.resolve()),
            "gt100_path":      str(args.gt100.resolve()),
            "base_dtype":       dtype_name(base),
            "query_dtype":      dtype_name(query),
            "batch_size":      args.batch_size,
            "query_batch_size": args.query_batch_size,
            "workers":         args.workers,
            "workers_requested": args.workers,
            "workers_active":   workers_active,
            "runtime_seconds": runtime,
        },
    }

    out_path = args.out_dir / f"{args.dataset_name}_{np.sqrt(radius_sq):.4f}_gt.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
