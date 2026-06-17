"""Smoke test compute_range_gt.py against a hand-crafted toy dataset.

Layout (1-D vectors for easy manual distance computation):
    base  = [(0,), (1,), (3,), (5,)]  ids 0..3
    query = [(0,), (4,)]
    gt2   = top-2 NN ids per query: Q0 -> [0,1]   Q1 -> [3,2]

With k=2 and percentile=50:
    Q0's 2nd-NN is id=1, L2² = (0-1)² = 1.0
    Q1's 2nd-NN is id=2, L2² = (4-3)² = 1.0
    radius² = median(1.0, 1.0) = 1.0

Expected hits (L2² <= 1.0):
    Q0: id=0 (0), id=1 (1)       -> 2
    Q1: id=3 (1), id=2 (1)       -> 2
    total = 4
"""
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import types

import numpy as np

import compute_range_gt


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True)


def write_fbin(path: pathlib.Path, data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype=np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", data.shape[0], data.shape[1]))
        f.write(data.tobytes())


def write_fbin_v2_int8(path: pathlib.Path, data: np.ndarray) -> None:
    data = np.ascontiguousarray(data, dtype=np.int8)
    with path.open("wb") as f:
        f.write(struct.pack("<IIII", 0x31424A44, 1, data.shape[0], data.shape[1]))
        f.write(data.tobytes())


def write_gt_ids(path: pathlib.Path, ids: np.ndarray) -> None:
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", ids.shape[0], ids.shape[1]))
        f.write(ids.tobytes())


def test_toy_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base = np.array([[0.0], [1.0], [3.0], [5.0]], dtype=np.float32)
        query = np.array([[0.0], [4.0]], dtype=np.float32)
        gt = np.array([[0, 1], [3, 2]], dtype=np.uint32)
        base_path = tmpdir / "base.fbin"
        query_path = tmpdir / "query.fbin"
        gt_path = tmpdir / "gt.bin"
        write_fbin(base_path, base)
        write_fbin(query_path, query)
        write_gt_ids(gt_path, gt)

        script = pathlib.Path(__file__).parent / "compute_range_gt.py"
        out_dir = tmpdir / "out"
        cmd = [
            sys.executable, str(script),
            "--base", str(base_path),
            "--query", str(query_path),
            "--gt100", str(gt_path),
            "--out-dir", str(out_dir),
            "--k", "2",
            "--percentile", "50",
            "--batch-size", "2",
            "--dataset-name", "toy",
        ]
        r = run_cmd(cmd)
        assert r.returncode == 0
        print(r.stdout)

        out_files = list(out_dir.glob("*.json"))
        assert len(out_files) == 1, f"expected 1 output, got {out_files}"
        with out_files[0].open() as f:
            result = json.load(f)

        expected = {
            "radius_squared": 1.0,
            "total_in_range": 4,
            "n_queries": 2,
            "n_base": 4,
            "dim": 1,
        }
        for key, val in expected.items():
            got = result[key]
            if isinstance(val, float):
                assert abs(got - val) < 1e-6, f"{key}: got {got}, expected {val}"
            else:
                assert got == val, f"{key}: got {got}, expected {val}"

        pq = result["per_query_count_stats"]
        assert pq["min"] == 2 and pq["max"] == 2, f"per_query stats wrong: {pq}"

        print("\n[PASS] smoke test")
        print("Output file:", out_files[0].name)
        print("Payload:")
        print(json.dumps({k: v for k, v in result.items() if k != "_schema"}, indent=2))


def test_load_fbin_uses_memmap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base_path = tmpdir / "base.fbin"
        write_fbin(base_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

        loaded = compute_range_gt.load_fbin(base_path)

        assert isinstance(loaded, np.memmap), type(loaded)
        assert loaded.shape == (2, 2)
        np.testing.assert_allclose(loaded, [[1.0, 2.0], [3.0, 4.0]])


def test_load_fbin_v2_int8_uses_memmap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base_path = tmpdir / "base_i8.fbin"
        write_fbin_v2_int8(base_path, np.array([[-3, 2], [4, -5]], dtype=np.int8))

        loaded = compute_range_gt.load_fbin(base_path)

        assert isinstance(loaded, np.memmap), type(loaded)
        assert loaded.dtype == np.int8
        assert loaded.shape == (2, 2)
        np.testing.assert_array_equal(loaded, np.array([[-3, 2], [4, -5]], dtype=np.int8))


def test_cli_accepts_v2_int8_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base = np.array([[0, 0], [1, 0], [3, 0], [5, 0]], dtype=np.int8)
        query = np.array([[0, 0], [4, 0]], dtype=np.int8)
        gt = np.array([[0, 1], [3, 2]], dtype=np.uint32)
        base_path = tmpdir / "base_i8.fbin"
        query_path = tmpdir / "query_i8.fbin"
        gt_path = tmpdir / "gt.bin"
        write_fbin_v2_int8(base_path, base)
        write_fbin_v2_int8(query_path, query)
        write_gt_ids(gt_path, gt)

        script = pathlib.Path(__file__).parent / "compute_range_gt.py"
        out_dir = tmpdir / "out"
        cmd = [
            sys.executable, str(script),
            "--base", str(base_path),
            "--query", str(query_path),
            "--gt100", str(gt_path),
            "--out-dir", str(out_dir),
            "--k", "2",
            "--percentile", "50",
            "--batch-size", "2",
            "--query-batch-size", "1",
            "--dataset-name", "toy_i8",
        ]
        r = run_cmd(cmd)
        assert r.returncode == 0, r.stderr

        out_files = list(out_dir.glob("*.json"))
        assert len(out_files) == 1, f"expected 1 output, got {out_files}"
        with out_files[0].open() as f:
            result = json.load(f)

        assert result["radius_squared"] == 1.0
        assert result["per_query_counts"] == [2, 2]
        assert result["total_in_range"] == 4
        assert result["provenance"]["base_dtype"] == "int8"
        assert result["provenance"]["query_dtype"] == "int8"


def test_parallel_query_batched_cli_matches_expected_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base = np.array([[0.0], [1.0], [2.0], [4.0], [7.0]], dtype=np.float32)
        query = np.array([[0.0], [3.0], [8.0]], dtype=np.float32)
        gt = np.array([[0], [3], [4]], dtype=np.uint32)
        base_path = tmpdir / "base.fbin"
        query_path = tmpdir / "query.fbin"
        gt_path = tmpdir / "gt.bin"
        write_fbin(base_path, base)
        write_fbin(query_path, query)
        write_gt_ids(gt_path, gt)

        script = pathlib.Path(__file__).parent / "compute_range_gt.py"
        out_dir = tmpdir / "out"
        cmd = [
            sys.executable, str(script),
            "--base", str(base_path),
            "--query", str(query_path),
            "--gt100", str(gt_path),
            "--out-dir", str(out_dir),
            "--k", "1",
            "--percentile", "50",
            "--batch-size", "2",
            "--query-batch-size", "2",
            "--workers", "2",
            "--dataset-name", "parallel_toy",
        ]
        r = run_cmd(cmd)
        assert r.returncode == 0, r.stderr

        out_files = list(out_dir.glob("*.json"))
        assert len(out_files) == 1, f"expected 1 output, got {out_files}"
        with out_files[0].open() as f:
            result = json.load(f)

        assert result["radius_squared"] == 1.0
        assert result["per_query_counts"] == [2, 2, 1]
        assert result["total_in_range"] == 5
        assert result["provenance"]["batch_size"] == 2
        assert result["provenance"]["query_batch_size"] == 2
        assert result["provenance"]["workers"] == 2


def test_parallel_count_matches_serial_for_2d_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base_path = tmpdir / "base.fbin"
        query_path = tmpdir / "query.fbin"
        base = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 3.0],
                [5.0, 5.0],
                [9.0, 9.0],
            ],
            dtype=np.float32,
        )
        query = np.array(
            [
                [0.0, 0.0],
                [3.0, 2.0],
                [8.0, 8.0],
            ],
            dtype=np.float32,
        )
        write_fbin(base_path, base)
        write_fbin(query_path, query)
        base_map = compute_range_gt.load_fbin(base_path)
        query_map = compute_range_gt.load_fbin(query_path)

        serial_total, serial_counts = compute_range_gt.count_in_range(
            query_map,
            base_map,
            radius_sq=5.0,
            batch_size=2,
            query_batch_size=2,
            workers=1,
        )
        parallel_total, parallel_counts = compute_range_gt.count_in_range(
            query_map,
            base_map,
            radius_sq=5.0,
            batch_size=2,
            query_batch_size=2,
            workers=2,
        )

        assert serial_total == parallel_total
        np.testing.assert_array_equal(serial_counts, parallel_counts)
        np.testing.assert_array_equal(serial_counts, np.array([3, 1, 1], dtype=np.int64))


def test_base_batches_are_generated_lazily() -> None:
    batches = compute_range_gt.iter_base_batches(10, 3)

    assert isinstance(batches, types.GeneratorType), type(batches)
    assert next(batches) == (0, 3)
    assert next(batches) == (3, 6)


def test_worker_provenance_records_requested_and_active_workers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        base = np.array([[0.0], [1.0]], dtype=np.float32)
        query = np.array([[0.0]], dtype=np.float32)
        gt = np.array([[0]], dtype=np.uint32)
        base_path = tmpdir / "base.fbin"
        query_path = tmpdir / "query.fbin"
        gt_path = tmpdir / "gt.bin"
        write_fbin(base_path, base)
        write_fbin(query_path, query)
        write_gt_ids(gt_path, gt)

        script = pathlib.Path(__file__).parent / "compute_range_gt.py"
        out_dir = tmpdir / "out"
        cmd = [
            sys.executable, str(script),
            "--base", str(base_path),
            "--query", str(query_path),
            "--gt100", str(gt_path),
            "--out-dir", str(out_dir),
            "--k", "1",
            "--percentile", "50",
            "--batch-size", "10",
            "--query-batch-size", "1",
            "--workers", "4",
            "--dataset-name", "worker_toy",
        ]
        r = run_cmd(cmd)
        assert r.returncode == 0, r.stderr

        out_files = list(out_dir.glob("*.json"))
        assert len(out_files) == 1, f"expected 1 output, got {out_files}"
        with out_files[0].open() as f:
            result = json.load(f)

        assert result["provenance"]["workers"] == 4
        assert result["provenance"]["workers_requested"] == 4
        assert result["provenance"]["workers_active"] == 1


def test_output_dir_is_checked_before_loading_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        file_parent = tmpdir / "not_a_dir"
        file_parent.write_text("blocks directory creation")

        script = pathlib.Path(__file__).parent / "compute_range_gt.py"
        cmd = [
            sys.executable, str(script),
            "--base", str(tmpdir / "missing_base.fbin"),
            "--query", str(tmpdir / "missing_query.fbin"),
            "--gt100", str(tmpdir / "missing_gt.bin"),
            "--out-dir", str(file_parent / "output"),
            "--dataset-name", "toy",
        ]
        r = run_cmd(cmd)
        assert r.returncode != 0
        assert "output directory" in r.stderr.lower(), r.stderr
        assert "[1/4] Loading base" not in r.stdout, r.stdout


def main() -> int:
    test_toy_dataset()
    test_load_fbin_uses_memmap()
    test_load_fbin_v2_int8_uses_memmap()
    test_cli_accepts_v2_int8_inputs()
    test_parallel_query_batched_cli_matches_expected_counts()
    test_parallel_count_matches_serial_for_2d_data()
    test_base_batches_are_generated_lazily()
    test_worker_provenance_records_requested_and_active_workers()
    test_output_dir_is_checked_before_loading_inputs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
