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
export TMP_UINT8_E2E_DIR="${tmp_dir}"

uv run python - <<'PY'
import json
import os
import pathlib
import struct

tmp = pathlib.Path(os.environ["TMP_UINT8_E2E_DIR"])

def write_u8bin(path: pathlib.Path, rows: list[list[int]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            f.write(bytes(row))

dim = 64
def pad(prefix: list[int]) -> list[int]:
    return prefix + [0] * (dim - len(prefix))

base = [
    pad([0, 0, 0, 0]),
    pad([10, 10, 10, 10]),
    pad([255, 255, 255, 255]),
    pad([200, 1, 2, 3]),
    pad([201, 1, 2, 3]),
    pad([30, 31, 32, 33]),
]
for i in range(122):
    base.append(pad([50 + i, 70 + i, 90 + i, 110 + i]))
query = [
    pad([0, 0, 0, 0]),
    pad([200, 0, 0, 0]),
]
radius_sq = 20.0
counts = []
expected = []
for qid, q in enumerate(query):
    count = 0
    for bid, b in enumerate(base):
        dist = sum((int(q[i]) - int(b[i])) ** 2 for i in range(len(q)))
        if dist <= radius_sq:
            count += 1
            expected.append((qid, bid, float(dist)))
    counts.append(count)

write_u8bin(tmp / "base.u8bin", base)
write_u8bin(tmp / "query.u8bin", query)
(tmp / "expected_hits.json").write_text(json.dumps(expected))
gt = {
    "dataset": "tiny_uint8_e2e",
    "metric": "l2",
    "n_queries": len(query),
    "n_base": len(base),
    "dim": len(base[0]),
    "vec_dtype": "uint8",
    "radius_squared": radius_sq,
    "total_in_range": sum(counts),
    "per_query_counts": counts,
    "per_query_count_stats": {
        "min": min(counts),
        "median": sum(sorted(counts)) / len(counts),
        "max": max(counts),
        "zero_hit_queries": sum(1 for c in counts if c == 0),
    },
    "index": {"prefix": str(tmp / "tiny_uint8_index")},
    "provenance": {
        "base_path": str(tmp / "base.u8bin"),
        "query_path": str(tmp / "query.u8bin"),
    },
}
(tmp / "gt.json").write_text(json.dumps(gt, indent=2))
config = """\
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
vec_dtype uint8
fast_path_enabled false
data_hnsw_enabled false
sqg_m 32
sqg_num_iter 3
sqg_ef_build 64
sqg_ef_search 64
sqg_centroid_ef_search 64
"""
(tmp / "tiny_uint8.config").write_text(config)
PY

"${build_bin}" --config "${tmp_dir}/tiny_uint8.config" --gt "${tmp_dir}/gt.json" \
  >/tmp/uint8_tiny_e2e_build.out 2>/tmp/uint8_tiny_e2e_build.err

"${search_bin}" --config "${tmp_dir}/tiny_uint8.config" --gt "${tmp_dir}/gt.json" \
  --no-cell-filter --dump-range-hits "${tmp_dir}/hits.csv" \
  >/tmp/uint8_tiny_e2e_search.out 2>/tmp/uint8_tiny_e2e_search.err

uv run python - <<'PY'
import csv
import json
import math
import os
import pathlib
import re

tmp = pathlib.Path(os.environ["TMP_UINT8_E2E_DIR"])
expected = {
    (int(q), int(b), float(d))
    for q, b, d in json.loads((tmp / "expected_hits.json").read_text())
}
actual = set()
with (tmp / "hits.csv").open() as f:
    for row in csv.DictReader(f):
        actual.add(
            (
                int(row["query_id"]),
                int(row["original_base_id"]),
                float(row["distance"]),
            )
        )
if actual != expected:
    raise SystemExit(f"hit mismatch expected={expected} actual={actual}")
stdout = pathlib.Path("/tmp/uint8_tiny_e2e_search.out").read_text()
match = re.search(r"^recall_verified = ([0-9.]+)", stdout, flags=re.MULTILINE)
if not match:
    raise SystemExit("missing recall_verified output")
if not math.isclose(float(match.group(1)), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(f"recall_verified not 1.0: {match.group(1)}")
PY
