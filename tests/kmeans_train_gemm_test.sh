#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:---static-only}"
config="configs/deep1m_disk.config"
gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
deep10m_config="configs/deep10m_disk_fast_path.config"
deep10m_gt_json="experiments/range_gt/output/deep10m_0.2002_gt.json"
stats_fixture="tests/fixtures/deep1m_disk_build_prune_stats_pre_kmeans_train_gemm.txt"
rank_fixture="tests/fixtures/deep1m_disk_pca_effective_rank_pre_kmeans_train_gemm.txt"
KM_TRAIN_GATE_SECONDS=80.0
TOTAL_WALL_GATE_SECONDS=160.0
deep1m_build_gate_seconds=9.0

run_static_checks() {
  python3 - <<'PY'
from pathlib import Path


def extract_braced_function(text: str, marker: str) -> str:
    start = text.find(marker)
    if start == -1:
        raise SystemExit(f"missing marker: {marker}")
    brace = text.find("{", start)
    if brace == -1:
        raise SystemExit(f"missing opening brace for {marker}")
    depth = 0
    for idx in range(brace, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise SystemExit(f"unterminated body for {marker}")


kmeans = Path("lib/Kmeans.h").read_text(encoding="utf-8")
guard_path = Path("lib/OpenBlasThreadGuard.h")
if not guard_path.exists():
    raise SystemExit("missing lib/OpenBlasThreadGuard.h")
guard = guard_path.read_text(encoding="utf-8")
for token in (
    "#ifdef DJ_USE_OPENBLAS",
    "#include <cblas.h>",
    "class OpenBlasThreadGuard",
    "openblas_get_num_threads",
    "openblas_set_num_threads",
    "OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete",
    "OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete",
	    "OpenBlasThreadGuard(OpenBlasThreadGuard&&) = delete",
	    "OpenBlasThreadGuard& operator=(OpenBlasThreadGuard&&) = delete",
	    "int prev_",
	    "GOMP_CPU_AFFINITY",
	    "sched_getaffinity",
	    "sched_setaffinity",
	    "restore_affinity_",
	):
	    if token not in guard:
	        raise SystemExit(f"missing OpenBlasThreadGuard token: {token}")

for token in (
    "#include <optional>",
    '#include "OpenBlasThreadGuard.h"',
    '#error "kmeans-train-gemm requires DJ_USE_OPENBLAS; configure cmake with -DDJ_USE_OPENBLAS=ON"',
):
    if token not in kmeans:
        raise SystemExit(f"missing Kmeans.h include/guard token: {token}")

train_body = extract_braced_function(kmeans, "void train(size_t n, float* data")
for token in (
    "std::optional<OpenBlasThreadGuard>",
    "blas_guard.emplace(omp_get_max_threads())",
    "DefaultInitAllocator",
    "kPageStrideFloats",
    "x_norm_sq",
    "c_norm_sq",
    "kTargetChunkBytes",
    "cblas_sgemm",
    "CblasRowMajor",
    "CblasNoTrans",
    "CblasTrans",
):
    if token not in train_body:
        raise SystemExit(f"missing train GEMM token: {token}")
if "openblas_set_num_threads" in train_body or "openblas_get_num_threads" in train_body:
    raise SystemExit("Kmeans::train must not call openblas_* directly")
if train_body.count("blas_guard.emplace(omp_get_max_threads())") != 1:
    raise SystemExit("Kmeans::train must construct exactly one OpenBlasThreadGuard")
if "dist_(data_ptr" in train_body:
    raise SystemExit("Kmeans::train still contains brute-force dist_ assignment marker")
if "anisotropic_loss(data_ptr, centroids_.data() + j * d, eta_, d)" not in train_body:
    raise SystemExit("Anisotropic branch lost brute-force anisotropic_loss marker")

microbench = Path("tests/kmeans_train_gemm_sgemm_microbench.cpp").read_text(
    encoding="utf-8"
)
for token in ("cblas_sgemm", "constexpr int m = 10000", "aggregate_gflops"):
    if token not in microbench:
        raise SystemExit(f"missing microbench token: {token}")

for script in (
    "tests/pg4_anisotropic_kmeans_bench_test.sh",
    "tests/sample_kmeans_centroids_test.sh",
    "tests/range_search_algorithm_test.sh",
    "tests/range_search_async_io_test.sh",
    "tests/pca_enclosing_bound_test.sh",
    "tests/pd2_cluster_slab_cpp_deployment_test.sh",
):
    text = Path(script).read_text(encoding="utf-8")
    for token in ("-DEIGEN_USE_BLAS", "-DDJ_USE_OPENBLAS", "openblas_libs"):
        if token not in text:
            raise SystemExit(f"{script} missing OpenBLAS direct-g++ token: {token}")

parallel = Path("tests/parallel_pca_cluster_fit_test.sh").read_text(encoding="utf-8")
for token in (
    "RUN_PARALLEL_PCA_CLUSTER_FIT_HISTORY",
    "archived after kmeans-train-gemm",
    "exit 77",
):
    if token not in parallel:
        raise SystemExit(f"parallel_pca_cluster_fit_test.sh missing archive token: {token}")
PY
}

run_build_only() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target build build_profile search search_stats build_stats \
    main build_prune_stats pca_lb_soundness_probe synthetic_residual_only_unit_test -j
}

parse_build_times() {
  python3 - "$@" <<'PY'
import re
import statistics
import sys
from pathlib import Path

values = []
for path in sys.argv[1:]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"build_time = ([0-9.]+)", text)
    if not matches:
        raise SystemExit(f"missing build_time in {path}")
    values.append(float(matches[-1]))
print(statistics.median(values))
PY
}

parse_profile_medians() {
  python3 - "$@" <<'PY'
import re
import statistics
import sys
from pathlib import Path

build_times = []
km_train_times = []
for path in sys.argv[1:]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    build_matches = re.findall(r"build_time = ([0-9.]+)", text)
    km_matches = re.findall(r"\[build-timing\] km\.train = ([0-9.]+)", text)
    if not build_matches:
        raise SystemExit(f"missing build_time in {path}")
    if not km_matches:
        raise SystemExit(f"missing km.train timing in {path}")
    build_times.append(float(build_matches[-1]))
    km_train_times.append(float(km_matches[-1]))
print(f"{statistics.median(km_train_times)} {statistics.median(build_times)}")
PY
}

parse_stats_output() {
  local source_file="$1"
  local out_file="$2"
  python3 - "$source_file" "$out_file" <<'PY'
import re
import sys
from pathlib import Path

wanted = {
    "recall",
    "dist_comp",
    "cell_candidate_vectors",
    "cell_pruned_vectors",
    "cell_kept_vectors",
    "triangle_pruned_clusters",
    "slab_only_pruned_clusters",
    "pca_only_pruned_clusters",
    "combined_pruned_clusters",
    "zero_kept_cells",
    "zero_kept_surviving_clusters",
}
values = {}
line_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*) = ([^ ]+)")
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = line_re.search(line)
    if match and match.group(1) in wanted:
        values[match.group(1)] = match.group(2)
missing = sorted(wanted - values.keys())
if missing:
    raise SystemExit(f"missing stats keys: {missing}")
Path(sys.argv[2]).write_text(
    "".join(f"{key}={values[key]}\n" for key in sorted(wanted)),
    encoding="utf-8",
)
PY
}

compare_stats() {
  local actual_file="$1"
  local baseline_file="$2"
  python3 - "$actual_file" "$baseline_file" <<'PY'
import sys
from pathlib import Path


def read_kv(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


actual = read_kv(sys.argv[1])
baseline = read_kv(sys.argv[2])
if actual.keys() != baseline.keys():
    raise SystemExit(
        f"stats key mismatch: actual_only={sorted(actual.keys() - baseline.keys())} "
        f"baseline_only={sorted(baseline.keys() - actual.keys())}"
    )
for key in sorted(baseline):
    a = float(actual[key])
    b = float(baseline[key])
    if key == "recall":
        if abs(a - b) > 0.005:
            raise SystemExit(f"recall drift too high: baseline={b} actual={a}")
        continue
    tolerance = 0.05 if key in {"zero_kept_surviving_clusters", "dist_comp"} else 0.02
    if b == 0.0:
        if a != 0.0:
            raise SystemExit(f"{key} baseline is 0 but actual is {a}")
    else:
        rel = abs(a - b) / b
        if rel > tolerance:
            raise SystemExit(
                f"{key} drift too high: baseline={b} actual={a} rel={rel} tolerance={tolerance}"
            )
PY
}

metadata_path_from_gt() {
  python3 - "$gt_json" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], "r", encoding="utf-8") as f:
    gt = json.load(f)
print(str(Path(str(gt["index"]["prefix"]) + "_diskrange") / "_metadata_file.bin"))
PY
}

extract_rank_fixture() {
  local metadata="$1"
  local out_file="$2"
  python3 - "$metadata" "$out_file" <<'PY'
import struct
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
with metadata.open("rb") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("metadata missing magic")
    version = struct.unpack("<Q", f.read(8))[0]
    if version != 5:
        raise SystemExit(f"unexpected metadata schema {version}")
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
    f.seek(cluster_num * 8, 1)
    f.seek(cluster_num * dim * 4, 1)
    f.seek(cluster_num * 4, 1)
    f.seek(cluster_num * pca_rank * dim * 4, 1)
    f.seek(cluster_num * pca_rank * 4, 1)
    f.seek(cluster_num * pca_rank * 4, 1)
    f.seek(cluster_num * 4, 1)
    raw = f.read(cluster_num * 4)
if len(raw) != cluster_num * 4:
    raise SystemExit("short pca_effective_rank block")
ranks = struct.unpack(f"<{cluster_num}I", raw)
Path(sys.argv[2]).write_text("".join(f"{rank}\n" for rank in ranks), encoding="utf-8")
PY
}

compare_rank_fixture() {
  local actual_file="$1"
  local baseline_file="$2"
  python3 - "$actual_file" "$baseline_file" <<'PY'
import sys
from collections import Counter
from pathlib import Path

actual = [int(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line]
baseline = [int(line) for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line]
if len(actual) != len(baseline):
    raise SystemExit(f"rank length mismatch: actual={len(actual)} baseline={len(baseline)}")
if not actual:
    raise SystemExit("empty rank fixture")
baseline_sum = sum(baseline)
actual_sum = sum(actual)
sum_rel = 0.0 if baseline_sum == 0 else abs(actual_sum - baseline_sum) / baseline_sum
actual_hist = Counter(actual)
baseline_hist = Counter(baseline)
hist_l1 = sum(
    abs(actual_hist.get(rank, 0) - baseline_hist.get(rank, 0))
    for rank in set(actual_hist) | set(baseline_hist)
)
hist_l1_rate = hist_l1 / len(actual)
rank0_delta = abs(actual_hist.get(0, 0) - baseline_hist.get(0, 0))
full_rank = max(max(actual), max(baseline))
full_rank_delta = abs(actual_hist.get(full_rank, 0) - baseline_hist.get(full_rank, 0))
if sum_rel > 0.01 or hist_l1_rate > 0.08 or rank0_delta > 100 or full_rank_delta > 100:
    raise SystemExit(
        "pca_effective_rank drift too high: "
        f"sum_rel={sum_rel} hist_l1_rate={hist_l1_rate} "
        f"rank0_delta={rank0_delta} full_rank_delta={full_rank_delta}"
    )
PY
}

extract_centroids() {
  local metadata="$1"
  local out_file="$2"
  python3 - "$metadata" "$out_file" <<'PY'
import struct
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
with metadata.open("rb") as f:
    if f.read(8) != b"DJMETA\0\0":
        raise SystemExit("metadata missing magic")
    version = struct.unpack("<Q", f.read(8))[0]
    if version != 5:
        raise SystemExit(f"unexpected metadata schema {version}")
    n, dim, cluster_num, max_points, pca_rank, target_cell_vecs = struct.unpack(
        "<QQQQQQ", f.read(48)
    )
    f.seek(cluster_num * 8, 1)
    raw = f.read(cluster_num * dim * 4)
if len(raw) != cluster_num * dim * 4:
    raise SystemExit("short centroid block")
Path(sys.argv[2]).write_bytes(raw)
PY
}

compare_centroids() {
  local first_file="$1"
  local second_file="$2"
  python3 - "$first_file" "$second_file" <<'PY'
import array
import sys
from pathlib import Path

a = array.array("f")
b = array.array("f")
a.frombytes(Path(sys.argv[1]).read_bytes())
b.frombytes(Path(sys.argv[2]).read_bytes())
if len(a) != len(b):
    raise SystemExit(f"centroid length mismatch: {len(a)} vs {len(b)}")
mse = sum((x - y) * (x - y) for x, y in zip(a, b)) / len(a)
print(f"centroid_mse={mse}")
if mse >= 1e-7:
    raise SystemExit(f"centroid drift too high: mse={mse}")
PY
}

run_deep1m_checks() {
  if [[ ! -f "$stats_fixture" || ! -f "$rank_fixture" ]]; then
    echo "missing pre-kmeans fixture(s): $stats_fixture / $rank_fixture" >&2
    exit 1
  fi

  local run_dir
  run_dir="$(mktemp -d /tmp/kmeans_train_gemm.XXXXXX)"
  local build_logs=()
  for i in 1 2 3; do
    local log="$run_dir/build_${i}.log"
    ./build/build --config "$config" --gt "$gt_json" >"$log" 2>&1
    build_logs+=("$log")
  done
  local median_build
  median_build="$(parse_build_times "${build_logs[@]}")"
  python3 - "$median_build" "$deep1m_build_gate_seconds" <<'PY'
import sys
value = float(sys.argv[1])
gate = float(sys.argv[2])
if value > gate:
    raise SystemExit(f"deep1m build median {value} exceeds gate {gate}")
print(f"deep1m_build_median={value}")
PY

  local stats_log="$run_dir/build_prune_stats.log"
  local actual_stats="$run_dir/actual_stats.txt"
  ./build/build_prune_stats --config "$config" --gt "$gt_json" >"$stats_log" 2>&1
  parse_stats_output "$stats_log" "$actual_stats"
  compare_stats "$actual_stats" "$stats_fixture"

  local metadata
  metadata="$(metadata_path_from_gt)"
  local actual_ranks="$run_dir/actual_ranks.txt"
  extract_rank_fixture "$metadata" "$actual_ranks"
  compare_rank_fixture "$actual_ranks" "$rank_fixture"

  ./build/pca_lb_soundness_probe --config "$config" --gt "$gt_json" --queries 50

  local centroid_a="$run_dir/centroids_a.bin"
  local centroid_b="$run_dir/centroids_b.bin"
  ./build/build --config "$config" --gt "$gt_json" >"$run_dir/determinism_a.log" 2>&1
  extract_centroids "$metadata" "$centroid_a"
  ./build/build --config "$config" --gt "$gt_json" >"$run_dir/determinism_b.log" 2>&1
  extract_centroids "$metadata" "$centroid_b"
  compare_centroids "$centroid_a" "$centroid_b"

  local recall_values=()
  for i in 1 2 3; do
    local recall_log="$run_dir/recall_${i}.log"
    local recall_stats="$run_dir/recall_${i}.txt"
    ./build/build_prune_stats --config "$config" --gt "$gt_json" >"$recall_log" 2>&1
    parse_stats_output "$recall_log" "$recall_stats"
    recall_values+=("$(awk -F= '$1 == "recall" { print $2 }' "$recall_stats")")
  done
  python3 - "${recall_values[@]}" <<'PY'
import sys
values = [float(v) for v in sys.argv[1:]]
spread = max(values) - min(values)
print(f"recall_values={values} spread={spread}")
if spread >= 1e-4:
    raise SystemExit(f"recall spread too high: {spread}")
PY
}

run_deep10m_timing_checks() {
  local run_dir
  run_dir="$(mktemp -d /tmp/kmeans_train_gemm_deep10m.XXXXXX)"
  local profile_logs=()
  for i in 1 2 3; do
    local log="$run_dir/build_profile_${i}.log"
    ./build/build_profile --config "$deep10m_config" --gt "$deep10m_gt_json" >"$log" 2>&1
    profile_logs+=("$log")
  done
  local medians
  medians="$(parse_profile_medians "${profile_logs[@]}")"
  python3 - "$medians" "$KM_TRAIN_GATE_SECONDS" "$TOTAL_WALL_GATE_SECONDS" <<'PY'
import sys

km_train, total = [float(x) for x in sys.argv[1].split()]
km_gate = float(sys.argv[2])
total_gate = float(sys.argv[3])
if km_train > km_gate:
    raise SystemExit(f"deep10m km.train median {km_train} exceeds gate {km_gate}")
if total > total_gate:
    raise SystemExit(f"deep10m build median {total} exceeds gate {total_gate}")
print(f"deep10m_km_train_median={km_train}")
print(f"deep10m_build_median={total}")
if km_train <= 25.0:
    print("INFO: deep10m km.train stretch goal met")
if total <= 100.0:
    print("INFO: deep10m total wall stretch goal met")
PY
}

case "$mode" in
  static|--static-only)
    run_static_checks
    ;;
  --build-only)
    run_static_checks
    run_build_only
    ;;
  --deep1m-only)
    run_static_checks
    run_deep1m_checks
    ;;
  --deep10m-only)
    run_static_checks
    run_deep10m_timing_checks
    ;;
  all)
    run_static_checks
    run_build_only
    run_deep1m_checks
    run_deep10m_timing_checks
    ./tests/pg4_anisotropic_kmeans_bench_test.sh
    ./tests/sample_kmeans_centroids_test.sh
    ;;
  *)
    echo "usage: $0 [static|--static-only|--build-only|--deep1m-only|--deep10m-only|all]" >&2
    exit 2
    ;;
esac
