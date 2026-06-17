#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="$(mktemp -d /tmp/fbin_dtype_wrap.XXXXXX)"
cleanup() {
  rm -rf "$run_dir"
}
trap cleanup EXIT

cmake --build build --target fbin_dtype_wrap >/tmp/fbin_dtype_wrap_build.log

python3 - "$run_dir/in.i8bin" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
payload = bytes([255, 0, 1, 2, 3, 4])
with path.open("wb") as f:
    f.write(struct.pack("<II", 2, 3))
    f.write(payload)
PY

./build/fbin_dtype_wrap "$run_dir/in.i8bin" "$run_dir/out.fbin"

python3 - "$run_dir/out.fbin" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
magic, dtype, n, dim = struct.unpack("<IIII", data[:16])
if (magic, dtype, n, dim) != (0x31424A44, 1, 2, 3):
    raise SystemExit(f"bad v2 header: {(magic, dtype, n, dim)}")
if data[16:] != bytes([255, 0, 1, 2, 3, 4]):
    raise SystemExit("payload was not copied byte-for-byte")
PY

if ./build/fbin_dtype_wrap "$run_dir/missing.i8bin" "$run_dir/missing.fbin" 2>"$run_dir/missing.err"; then
  echo "missing input should fail" >&2
  exit 1
fi
grep -F "input file not found" "$run_dir/missing.err"
grep -F "$run_dir/missing.i8bin" "$run_dir/missing.err"

python3 - "$run_dir/bad.i8bin" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
with path.open("wb") as f:
    f.write(struct.pack("<II", 2, 3))
    f.write(b"\x00\x01")
PY
if ./build/fbin_dtype_wrap "$run_dir/bad.i8bin" "$run_dir/bad.fbin" 2>"$run_dir/bad.err"; then
  echo "bad payload size should fail" >&2
  exit 1
fi
grep -F "expected payload bytes: 6, actual: 2" "$run_dir/bad.err"

mkdir "$run_dir/out_dir"
if ./build/fbin_dtype_wrap "$run_dir/in.i8bin" "$run_dir/out_dir" 2>"$run_dir/out.err"; then
  echo "directory output should fail" >&2
  exit 1
fi
grep -F "cannot open output file for writing" "$run_dir/out.err"
grep -F "$run_dir/out_dir" "$run_dir/out.err"
