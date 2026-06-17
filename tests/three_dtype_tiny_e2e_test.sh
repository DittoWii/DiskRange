#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
build_dir="${1:-${repo_root}/build}"
build_bin="${build_dir}/build"
search_bin="${build_dir}/search"
if [[ ! -x "${build_bin}" || ! -x "${search_bin}" ]]; then
  echo "missing build/search binaries in ${build_dir}" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export TMP_THREE_DTYPE_E2E_DIR="${tmp_dir}"

uv run python - <<'PY'
import json
import os
import pathlib
import struct

tmp = pathlib.Path(os.environ["TMP_THREE_DTYPE_E2E_DIR"])

def write_v1(path: pathlib.Path, rows, fmt: str) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            if fmt == "f":
                f.write(struct.pack("<" + "f" * len(row), *row))
            elif fmt == "b":
                f.write(struct.pack("<" + "b" * len(row), *row))
            elif fmt == "B":
                f.write(bytes(row))
            else:
                raise AssertionError(fmt)

dim = 64

def pad(prefix, fill=0):
    return prefix + [fill] * (dim - len(prefix))

datasets = {
    "float32": {
        "fmt": "f",
        "config_dtype": "float32",
        "base": [
            pad([0.0, 0.0, 0.0, 0.0]),
            pad([1.0, 1.0, 1.0, 1.0]),
            pad([5.0, 5.0, 5.0, 5.0]),
        ] + [pad([20.0 + i, 21.0 + i, 22.0 + i, 23.0 + i]) for i in range(125)],
        "query": [pad([0.0, 0.0, 0.0, 0.0]), pad([5.0, 4.0, 5.0, 4.0])],
        "radius_sq": 4.0,
    },
    "int8": {
        "fmt": "b",
        "config_dtype": "int8",
        "base": [
            pad([0, 0, 0, 0]),
            pad([-2, -2, -2, -2]),
            pad([5, 5, 5, 5]),
        ] + [pad([30 + i % 40, -30 + i % 40, 10, -10]) for i in range(125)],
        "query": [pad([0, 0, 0, 0]), pad([5, 4, 5, 4])],
        "radius_sq": 4.0,
    },
    "uint8": {
        "fmt": "B",
        "config_dtype": "uint8",
        "base": [
            pad([0, 0, 0, 0]),
            pad([10, 10, 10, 10]),
            pad([200, 1, 2, 3]),
        ] + [pad([50 + i, 70 + i, 90 + i, 110 + i]) for i in range(125)],
        "query": [pad([0, 0, 0, 0]), pad([200, 0, 0, 0])],
        "radius_sq": 20.0,
    },
}

for name, ds in datasets.items():
    root = tmp / name
    root.mkdir()
    base = ds["base"]
    query = ds["query"]
    write_v1(root / "base.fbin", base, ds["fmt"])
    write_v1(root / "query.fbin", query, ds["fmt"])
    expected = []
    counts = []
    for qid, q in enumerate(query):
        count = 0
        for bid, b in enumerate(base):
            dist = sum((float(q[i]) - float(b[i])) ** 2 for i in range(dim))
            if dist <= ds["radius_sq"]:
                count += 1
                expected.append((qid, bid, float(dist)))
        counts.append(count)
    (root / "expected_hits.json").write_text(json.dumps(expected))
    gt = {
        "dataset": f"tiny_{name}_e2e",
        "metric": "l2",
        "n_queries": len(query),
        "n_base": len(base),
        "dim": dim,
        "vec_dtype": ds["config_dtype"],
        "radius_squared": ds["radius_sq"],
        "total_in_range": sum(counts),
        "per_query_counts": counts,
        "per_query_count_stats": {
            "min": min(counts),
            "median": sum(sorted(counts)) / len(counts),
            "max": max(counts),
            "zero_hit_queries": sum(1 for c in counts if c == 0),
        },
        "index": {"prefix": str(root / "index")},
        "provenance": {
            "base_path": str(root / "base.fbin"),
            "query_path": str(root / "query.fbin"),
        },
    }
    (root / "gt.json").write_text(json.dumps(gt, indent=2))
    config = f"""\
cluster_num 64
K 64
mem_budget 1
mode disk
gorder_window 0
io_queue_depth 1
pca_rank 0
slab_count 0
cell_filter_enabled false
target_cell_vecs 2
vec_dtype {ds["config_dtype"]}
fast_path_enabled false
data_hnsw_enabled false
sqg_m 32
sqg_num_iter 3
sqg_ef_build 64
sqg_ef_search 64
sqg_centroid_ef_search 64
"""
    (root / "config").write_text(config)
PY

for dtype in float32 int8 uint8; do
  dtype_dir="${tmp_dir}/${dtype}"
  "${build_bin}" --config "${dtype_dir}/config" --gt "${dtype_dir}/gt.json" \
    >"/tmp/three_dtype_${dtype}_build.out" \
    2>"/tmp/three_dtype_${dtype}_build.err"
  "${search_bin}" --config "${dtype_dir}/config" --gt "${dtype_dir}/gt.json" \
    --no-cell-filter --dump-range-hits "${dtype_dir}/hits.csv" \
    >"/tmp/three_dtype_${dtype}_search.out" \
    2>"/tmp/three_dtype_${dtype}_search.err"
done

uv run python - <<'PY'
import csv
import json
import math
import os
import pathlib
import re

tmp = pathlib.Path(os.environ["TMP_THREE_DTYPE_E2E_DIR"])
for dtype in ["float32", "int8", "uint8"]:
    root = tmp / dtype
    expected = {
        (int(q), int(b), float(d))
        for q, b, d in json.loads((root / "expected_hits.json").read_text())
    }
    actual = set()
    with (root / "hits.csv").open() as f:
        for row in csv.DictReader(f):
            actual.add(
                (
                    int(row["query_id"]),
                    int(row["original_base_id"]),
                    float(row["distance"]),
                )
            )
    if actual != expected:
        raise SystemExit(f"{dtype}: hit mismatch expected={expected} actual={actual}")
    stdout = pathlib.Path(f"/tmp/three_dtype_{dtype}_search.out").read_text()
    match = re.search(r"^recall_verified = ([0-9.]+)", stdout, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"{dtype}: missing recall_verified output")
    if not math.isclose(float(match.group(1)), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"{dtype}: recall_verified not 1.0: {match.group(1)}")
PY
