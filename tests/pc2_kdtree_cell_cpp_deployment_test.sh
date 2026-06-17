#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-all}"
source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
source_config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/pc2_kdtree_cell_cpp.XXXXXX)"
index_prefix="$run_dir/deep1m"
gt_json="$run_dir/deep1m_0.2632_gt.json"
query_file="$run_dir/deep1m_query_subset.fbin"
config="$run_dir/deep1m_pc2.config"
n_queries=8

cleanup() {
  rm -rf "$run_dir" "${index_prefix}_diskrange"
}
trap cleanup EXIT

extract_values() {
  python3 tests/extract_run_result.py "$@"
}

run_static_checks() {
  python3 - <<'PY'
import pathlib
import re

def read(path):
    return pathlib.Path(path).read_text(encoding="utf-8")

cluster_io = read("lib/ClusterIO.h")
disk_range = read("lib/DiskRange.h")
pca_fit = read("lib/PcaFit.h")
dist_func = read("utils/dist_func.h")
config_reader = read("lib/ConfigReader.h")
config_loader = read("lib/ConfigLoader.h")
resolved = read("lib/ResolvedConfig.h")
run_result = read("lib/RunResult.h")
main = read("expr/main.cpp")
stats = read("expr/build_prune_stats/main.cpp")
io_py = read("experiments/range_aware_profiling/_io.py")
deep1m_config = read("configs/deep1m_disk.config")
deep10m_config = read("configs/deep10m_disk.config")

schema_match = re.search(r"SUPPORTED_METADATA_SCHEMA_VERSIONS\s*=\s*\{([^}]+)\}", io_py)
supported_schema_text = schema_match.group(1) if schema_match else ""

checks = {
    "schema version 6": "kMetadataSchemaVersion = 6u" in cluster_io,
    "target_cell_vecs in reader/writer": "target_cell_vecs" in cluster_io and "target_cell_vecs" in disk_range,
    "schema 6 cluster_cell_offsets": "cluster_cell_offsets" in cluster_io and "cluster_cell_offsets" in disk_range,
    "schema 6 cell_vec_offsets": "cell_vec_offsets" in cluster_io and "cell_vec_offsets" in disk_range,
    "per-cell bound accessors": all(s in cluster_io for s in ("cell_lo_ptr", "cell_hi_ptr", "cell_res_lo", "cell_res_hi")),
    "kdtree cell type": "PcaCell" in pca_fit and "build_kdtree_cells" in pca_fit,
    "median split exact semantics": all(s in pca_fit for s in ("nth_element", "min_element", "2.0f", "<= threshold")),
    "no schema4 quantizer config parser": 'key == "cell_b"' not in config_reader and 'key == "cell_q"' not in config_reader,
    "target_cell_vecs config parser": "target_cell_vecs" in config_reader and "split_axis_policy" in config_reader,
    "resolved target_cell_vecs": "target_cell_vecs" in resolved and "cell_b" not in resolved and "cell_q" not in resolved,
    "config loader target": "target_cell_vecs" in config_loader and "cell_b" not in config_loader and "cell_q" not in config_loader,
    "cell lb scalar helper": "cell_lb_batch(" in dist_func and all(s in dist_func for s in ("cell_lo", "cell_hi", "res_lo", "res_hi", "q_r_norm")),
    "no fake cell lb SIMD wrappers": all(s not in dist_func for s in ("cell_lb_batch_naive", "cell_lb_batch_avx512", "cell_lb_batch_avx(", "cell_lb_batch_sse", "cell_lb_batch_simd")),
    "old cell code signature removed": "cell_box_codes" not in dist_func and "cell_res_codes" not in dist_func,
    "RunResult byte counters": all(s in run_result for s in ("zero_kept_cells", "cell_bytes_read", "cluster_bytes_baseline", "io_bytes_reduction")),
    "main prints byte counters": all(s in main for s in ("zero_kept_cells", "cell_bytes_read", "cluster_bytes_baseline", "io_bytes_reduction")),
    "stats layer 1.5": "[Layer 1.5 cell]" in stats and "cell_metadata_bytes" in stats,
    "python reader schema 6": "SUPPORTED_METADATA_SCHEMA_VERSIONS" in io_py and "6" in supported_schema_text,
    "python reader skips schema 6 dtype": "schema_version >= 6" in io_py and "vec_dtype_header = f.read(8)" in io_py,
    "deep1m config target": "target_cell_vecs 24" in deep1m_config and "cell_b" not in deep1m_config and "cell_q" not in deep1m_config,
    "deep10m config target": "target_cell_vecs 24" in deep10m_config and "cell_b" not in deep10m_config and "cell_q" not in deep10m_config,
}

missing = [name for name, ok in checks.items() if not ok]
if missing:
    raise SystemExit("missing PC2 schema-6 static requirements: " + ", ".join(missing))
PY
}

prepare_smoke_inputs() {
  test -f "$source_gt_json"
  test -f "$source_config"

  python3 - "$source_gt_json" "$gt_json" "$index_prefix" "$query_file" "$n_queries" <<'PY'
import json
import struct
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

query_path = data["provenance"]["query_path"]
requested = int(sys.argv[5])
selected = [
    idx for idx, count in enumerate(data["per_query_counts"]) if int(count) > 0
][:requested]
if len(selected) != requested:
    raise SystemExit(f"needed {requested} nonzero queries, found {len(selected)}")

dim = int(data["dim"])
row_bytes = dim * 4
with open(query_path, "rb") as src, open(sys.argv[4], "wb") as dst:
    n_file, dim_file = struct.unpack("<II", src.read(8))
    if dim_file != dim:
        raise SystemExit(f"query dim mismatch: json={dim}, file={dim_file}")
    dst.write(struct.pack("<II", len(selected), dim))
    for idx in selected:
        src.seek(8 + idx * row_bytes)
        row = src.read(row_bytes)
        if len(row) != row_bytes:
            raise SystemExit(f"short query row at index {idx}")
        dst.write(row)

counts = [int(data["per_query_counts"][idx]) for idx in selected]
sorted_counts = sorted(counts)
data["n_queries"] = len(selected)
data["total_in_range"] = sum(counts)
data["per_query_counts"] = counts
data["per_query_count_stats"] = {
    "min": min(counts) if counts else 0,
    "median": (
        sorted_counts[(len(sorted_counts) - 1) // 2]
        + sorted_counts[len(sorted_counts) // 2]
    )
    / 2.0
    if sorted_counts
    else 0.0,
    "max": max(counts) if counts else 0,
    "zero_hit_queries": sum(1 for count in counts if count == 0),
}
data["index"]["prefix"] = sys.argv[3]
data["provenance"]["query_path"] = sys.argv[4]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

  python3 - "$source_config" "$config" <<'PY'
import sys

with open(sys.argv[1], "r", encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as dst:
    saw_target_cell_vecs = False
    for raw in src:
        stripped = raw.strip()
        if stripped.startswith("cell_b ") or stripped.startswith("cell_q "):
            continue
        if stripped.startswith("target_cell_vecs "):
            dst.write("target_cell_vecs 24\n")
            saw_target_cell_vecs = True
        else:
            dst.write(raw)
    # Old configs did not have this key; append for explicit smoke stability.
    if not saw_target_cell_vecs:
        dst.write("target_cell_vecs 24\n")
PY
}

run_smoke_checks() {
  prepare_smoke_inputs
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPRUNE_BREAKDOWN_STATS=ON \
    >/tmp/pc2_kdtree_cell_cpp_cmake.log
  cmake --build build -j >/tmp/pc2_kdtree_cell_cpp_build.log

  config_probe_cpp="$run_dir/config_probe.cpp"
  config_probe_bin="$run_dir/config_probe"
  python3 - "$config_probe_cpp" <<'PY'
import sys
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(r'''
#include <exception>
#include <iostream>
#include "lib/ConfigLoader.h"

int main(int argc, char** argv)
{
  if (argc != 3) return 2;
  try
  {
    const ResolvedConfig config = load_resolved_config(argv[1], argv[2]);
    std::cout << "target_cell_vecs = " << config.target_cell_vecs << '\n';
    std::cout << "split_axis_policy = " << config.split_axis_policy << '\n';
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
''')
PY
  g++ -std=c++17 -O3 -fopenmp -I. -Iwheel -Ithird/json/single_include \
    -Ithird/liburing/src/include -Ithird/eigen "$config_probe_cpp" \
    -o "$config_probe_bin"

  "$config_probe_bin" "$config" "$gt_json" | grep -Fq "target_cell_vecs = 24"

  old_key_config="$run_dir/old_key.config"
  cp "$config" "$old_key_config"
  echo "cell_b 4" >>"$old_key_config"
  if "$config_probe_bin" "$old_key_config" "$gt_json" >/tmp/pc2_old_key.out 2>/tmp/pc2_old_key.err; then
    echo "old cell_b key unexpectedly accepted" >&2
    exit 1
  fi
  grep -Fq "target_cell_vecs" /tmp/pc2_old_key.err

  dynamic_config="$run_dir/dynamic.config"
  cp "$config" "$dynamic_config"
  echo "split_axis_policy dynamic" >>"$dynamic_config"
  if "$config_probe_bin" "$dynamic_config" "$gt_json" >/tmp/pc2_dynamic.out 2>/tmp/pc2_dynamic.err; then
    echo "dynamic split_axis_policy unexpectedly accepted" >&2
    exit 1
  fi
  grep -Fq "dynamic" /tmp/pc2_dynamic.err

  stdout_on="$run_dir/main_on.out"
  stderr_on="$run_dir/main_on.err"
  ./build/main --config "$config" --gt "$gt_json" >"$stdout_on" 2>"$stderr_on"
  extract_values "$stdout_on" "$stderr_on" >"$run_dir/on.values"

  stdout_off="$run_dir/stats_off.out"
  stderr_off="$run_dir/stats_off.err"
  ./build/build_prune_stats --config "$config" --gt "$gt_json" --no-cell-filter \
    >"$stdout_off" 2>"$stderr_off"
  extract_values "$stdout_off" "$stderr_off" >"$run_dir/off.values"

  python3 - "$run_dir/on.values" "$run_dir/off.values" <<'PY'
import math
import sys

def load(path):
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if " = " in line:
                k, v = line.strip().split(" = ", 1)
                values[k] = v
    return values

on = load(sys.argv[1])
off = load(sys.argv[2])
required = {
    "recall",
    "cell_candidate_vectors",
    "cell_pruned_vectors",
    "cell_kept_vectors",
    "zero_kept_cells",
    "zero_kept_surviving_clusters",
    "cell_bytes_read",
    "cluster_bytes_baseline",
    "io_bytes_reduction",
    "dist_comp",
    "per_query_zero",
}
missing = required - on.keys()
if missing:
    raise SystemExit(f"missing cell-filter RunResult keys: {sorted(missing)}")
missing = required - off.keys()
if missing:
    raise SystemExit(f"missing no-cell-filter RunResult keys: {sorted(missing)}")

if not math.isclose(float(on["recall"]), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(f"cell-filter recall is not 1.0: {on['recall']}")
if not math.isclose(float(off["recall"]), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(f"no-cell-filter recall is not 1.0: {off['recall']}")

candidate = int(on["cell_candidate_vectors"])
pruned = int(on["cell_pruned_vectors"])
kept = int(on["cell_kept_vectors"])
zero_cells = int(on["zero_kept_cells"])
cell_bytes = int(on["cell_bytes_read"])
cluster_bytes = int(on["cluster_bytes_baseline"])
if candidate <= 0 or pruned <= 0 or kept <= 0:
    raise SystemExit("cell filter counters must be positive")
if candidate != pruned + kept:
    raise SystemExit("cell candidate vectors must partition into pruned + kept")
if zero_cells <= 0:
    raise SystemExit("expected at least one zero-kept cell")
if cell_bytes <= 0 or cluster_bytes <= 0 or not cell_bytes < cluster_bytes:
    raise SystemExit(f"expected cell_bytes_read < cluster_bytes_baseline, got {cell_bytes}/{cluster_bytes}")
if not float(on["io_bytes_reduction"]) > 0.0:
    raise SystemExit("io_bytes_reduction must be positive with cell filter on")
if int(on["dist_comp"]) != kept:
    raise SystemExit("dist_comp must equal cell_kept_vectors with schema 6 cell filter")

for key in ("cell_candidate_vectors", "cell_pruned_vectors", "cell_kept_vectors", "zero_kept_cells", "zero_kept_surviving_clusters"):
    if int(off[key]) != 0:
        raise SystemExit(f"{key} must be 0 with --no-cell-filter")
if int(off["cell_bytes_read"]) != int(off["cluster_bytes_baseline"]):
    raise SystemExit("no-cell-filter bytes must match baseline")
if float(off["io_bytes_reduction"]) != 0.0:
    raise SystemExit("no-cell-filter io_bytes_reduction must be 0")
if int(off["dist_comp"]) <= int(on["dist_comp"]):
    raise SystemExit("no-cell-filter dist_comp should exceed cell-filter dist_comp")
PY
}

case "$mode" in
  --static-only)
    run_static_checks
    ;;
  --smoke-only)
    run_smoke_checks
    ;;
  all)
    run_static_checks
    run_smoke_checks
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
