#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

export TMP_UINT8_GT_DIR="${tmp_dir}"
uv run python - <<'PY'
import os
import pathlib
import struct

tmp = pathlib.Path(os.environ["TMP_UINT8_GT_DIR"])

def write_u8bin(path: pathlib.Path, rows: list[list[int]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            f.write(bytes(row))

base = [
    [0, 0, 0, 0],
    [10, 10, 10, 10],
    [255, 255, 255, 255],
    [200, 1, 2, 3],
]
query = [
    [0, 0, 0, 0],
    [200, 0, 0, 0],
]
write_u8bin(tmp / "base.u8bin", base)
write_u8bin(tmp / "query.u8bin", query)

# k=1 nearest ids: query0 -> row0 (distance 0), query1 -> row3
# (distance 1^2 + 2^2 + 3^2 = 14).
with (tmp / "gt100.bin").open("wb") as f:
    f.write(struct.pack("<II", 2, 1))
    f.write(struct.pack("<II", 0, 3))
PY

(
  cd "${repo_root}/experiments/range_gt"
  uv run compute_range_gt.py \
  --base "${tmp_dir}/base.u8bin" \
  --query "${tmp_dir}/query.u8bin" \
  --gt100 "${tmp_dir}/gt100.bin" \
  --k 1 \
  --percentile 100 \
  --batch-size 2 \
  --query-batch-size 1 \
  --workers 1 \
  --input-dtype uint8 \
  --out-dir "${tmp_dir}/out" \
  --dataset-name tiny_uint8 \
  --index-prefix "${tmp_dir}/tiny_uint8_index" >/dev/null
)

uv run python - <<'PY'
import json
import os
import pathlib

out_dir = pathlib.Path(os.environ["TMP_UINT8_GT_DIR"]) / "out"
paths = list(out_dir.glob("tiny_uint8_*_gt.json"))
if len(paths) != 1:
    raise SystemExit(f"expected one gt json, found {paths}")
payload = json.loads(paths[0].read_text())
if payload["vec_dtype"] != "uint8":
    raise SystemExit("top-level vec_dtype must be uint8")
if payload["provenance"]["base_dtype"] != "uint8":
    raise SystemExit("base_dtype must be uint8")
if payload["provenance"]["query_dtype"] != "uint8":
    raise SystemExit("query_dtype must be uint8")
if payload["per_query_counts"] != [1, 1]:
    raise SystemExit(f"unexpected counts: {payload['per_query_counts']}")
if payload["total_in_range"] != 2:
    raise SystemExit(f"unexpected total: {payload['total_in_range']}")
PY
