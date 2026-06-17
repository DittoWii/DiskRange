#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_PARALLEL_PCA_CLUSTER_FIT_HISTORY:-}" ]]; then
  echo "[skip] tests/parallel_pca_cluster_fit_test.sh: archived after kmeans-train-gemm." >&2
  echo "[skip]   The byte-equal recall gate no longer matches the post-kmeans-train-gemm binary." >&2
  echo "[skip]   Re-enable for historical verification with:" >&2
  echo "[skip]   RUN_PARALLEL_PCA_CLUSTER_FIT_HISTORY=1 $0 $*" >&2
  exit 77
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:---static-only}"
config="configs/deep1m_disk.config"
gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
stats_fixture="tests/fixtures/deep1m_disk_build_prune_stats_baseline.txt"
rank_fixture="tests/fixtures/deep1m_disk_pca_effective_rank_baseline.txt"
baseline_build_time="65.8"
build_time_threshold="22.0"

run_static_checks() {
  python3 - <<'PY'
from pathlib import Path


def extract_braced_function(text: str, name: str) -> str:
    start = text.find(name)
    if start == -1:
        raise SystemExit(f"missing function marker: {name}")
    brace = text.find("{", start)
    if brace == -1:
        raise SystemExit(f"missing opening brace for {name}")
    depth = 0
    for idx in range(brace, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise SystemExit(f"unterminated function body for {name}")


pca = Path("lib/PcaFit.h").read_text(encoding="utf-8")
cluster = Path("lib/ClusterIO.h").read_text(encoding="utf-8")
cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")

if "SelfAdjointEigenSolver<Eigen::MatrixXf>" not in pca:
    raise SystemExit("missing SelfAdjointEigenSolver<Eigen::MatrixXf>")
if "Eigen::MatrixXf cov" not in pca:
    raise SystemExit("missing Eigen::MatrixXf cov")
for forbidden in (
    "JacobiSVD",
    "Eigen::MatrixXd",
    "MatrixXd",
    "Eigen::VectorXd",
    "VectorXd",
):
    if forbidden in pca:
        raise SystemExit(f"forbidden PcaFit token remains: {forbidden}")

fit_body = extract_braced_function(pca, "inline void fit_pca_bucket")
for forbidden in (
    "static_cast<double>",
    "double centered_norm_sq",
    "double projected_norm_sq",
    "double coords",
):
    if forbidden in fit_body:
        raise SystemExit(f"forbidden fit_pca_bucket token remains: {forbidden}")
if (
    "static_cast<double>(n_pts)" not in pca
    or "static_cast<double>(target_cell_vecs)" not in pca
):
    raise SystemExit("pc2_kdtree_max_depth double arithmetic was accidentally changed")

pca_pass_body = extract_braced_function(cluster, "void fitPcaFromClusterFile()")
for marker in (
    "#pragma omp parallel",
    "#pragma omp for schedule(dynamic)",
    "cluster_file_offsets",
    "abort_requested",
    "record_error",
    "catch (const std::exception&",
    "catch (...)",
):
    if marker not in pca_pass_body:
        raise SystemExit(f"missing fitPcaFromClusterFile marker: {marker}")
before_parallel = pca_pass_body.split("#pragma omp parallel", 1)[0]
if "std::ifstream in(clusterfile, std::ios::binary | std::ios::in);" in before_parallel:
    raise SystemExit("serial shared PCA ifstream still exists before parallel region")

for marker in (
    "add_executable(pca_lb_soundness_probe tests/pca_lb_soundness_probe.cpp)",
    "add_executable(synthetic_residual_only_unit_test tests/synthetic_residual_only_unit_test.cpp)",
    "target_link_libraries(pca_lb_soundness_probe liburing_vendored symqglib rabitq_headers)",
    "target_link_libraries(synthetic_residual_only_unit_test liburing_vendored symqglib rabitq_headers)",
):
    if marker not in cmake:
        raise SystemExit(f"missing CMake marker: {marker}")
PY
}

run_build_only() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target build search search_stats build_stats main \
    build_prune_stats pca_lb_soundness_probe synthetic_residual_only_unit_test -j
}

parse_stats_output() {
  local source_file="$1"
  python3 - "$source_file" <<'PY'
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
for key in sorted(wanted):
    print(f"{key}={values[key]}")
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
    if key == "recall":
        if actual[key] != baseline[key]:
            raise SystemExit(f"recall mismatch: baseline={baseline[key]} actual={actual[key]}")
        continue
    b = float(baseline[key])
    a = float(actual[key])
    tolerance = 0.01 if key == "zero_kept_surviving_clusters" else 0.005
    if b == 0.0:
        if a != 0.0:
            raise SystemExit(f"{key} baseline is 0 but actual is {actual[key]}")
    else:
        rel = abs(a - b) / b
        if rel > tolerance:
            raise SystemExit(
                f"{key} drift too high: baseline={baseline[key]} actual={actual[key]} rel={rel} tolerance={tolerance}"
            )
PY
}

metadata_path_from_config() {
  python3 - "$config" "$gt_json" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[2], "r", encoding="utf-8") as f:
    data = json.load(f)
prefix = data["index"]["prefix"]
print(str(Path(str(prefix) + "_diskrange") / "_metadata_file.bin"))
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
    raise SystemExit("empty rank fixtures")
if min(actual) < 0 or max(actual) > 32:
    raise SystemExit(f"actual rank out of expected range: min={min(actual)} max={max(actual)}")

baseline_sum = sum(baseline)
actual_sum = sum(actual)
sum_rel = 0.0 if baseline_sum == 0 else abs(actual_sum - baseline_sum) / baseline_sum
baseline_hist = Counter(baseline)
actual_hist = Counter(actual)
hist_l1 = sum(
    abs(actual_hist.get(rank, 0) - baseline_hist.get(rank, 0))
    for rank in set(actual_hist) | set(baseline_hist)
)
hist_l1_rate = hist_l1 / len(actual)
full_rank_delta = abs(actual_hist.get(32, 0) - baseline_hist.get(32, 0))
rank0_delta = abs(actual_hist.get(0, 0) - baseline_hist.get(0, 0))

if sum_rel > 0.005 or hist_l1_rate > 0.05 or full_rank_delta > 50 or rank0_delta > 50:
    print(
        "rank distribution drift too high: "
        f"baseline_sum={baseline_sum} actual_sum={actual_sum} "
        f"sum_rel={sum_rel:.6f} hist_l1_rate={hist_l1_rate:.6f} "
        f"full_rank_delta={full_rank_delta} rank0_delta={rank0_delta}"
    )
    for rank in sorted(set(actual_hist) | set(baseline_hist)):
        b = baseline_hist.get(rank, 0)
        a = actual_hist.get(rank, 0)
        if a != b:
            print(f"rank={rank} baseline_count={b} actual_count={a}")
    raise SystemExit("pca_effective_rank distribution check failed")
print(
    "rank distribution ok: "
    f"baseline_sum={baseline_sum} actual_sum={actual_sum} "
    f"sum_rel={sum_rel:.6f} hist_l1_rate={hist_l1_rate:.6f} "
    f"full_rank_delta={full_rank_delta} rank0_delta={rank0_delta}"
)
PY
}

run_timed_builds() {
  local run_dir="$1"
  for i in 1 2 3; do
    ./build/build --config "$config" --gt "$gt_json" >"$run_dir/build_${i}.out"
  done
  python3 - "$run_dir" "$build_time_threshold" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
threshold = float(sys.argv[2])
values = []
for i in range(1, 4):
    text = (run_dir / f"build_{i}.out").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bbuild_time = ([0-9.]+)", text)
    if not match:
        raise SystemExit(f"missing build_time in build_{i}.out")
    values.append(float(match.group(1)))
median = sorted(values)[1]
print(f"build_times={values} median={median:.6f} threshold={threshold:.6f}")
if median > threshold:
    raise SystemExit(f"build_time median exceeds threshold: {median} > {threshold}")
PY
}

run_all() {
  test -f "$stats_fixture"
  test -f "$rank_fixture"
  run_static_checks
  run_build_only
  local run_dir
  run_dir="$(mktemp -d /tmp/parallel_pca_cluster_fit.XXXXXX)"
  trap 'rm -rf "$run_dir"' RETURN

  run_timed_builds "$run_dir"
  ./build/build_prune_stats --config "$config" --gt "$gt_json" \
    >"$run_dir/build_prune_stats.out"
  parse_stats_output "$run_dir/build_prune_stats.out" >"$run_dir/actual_stats.txt"
  compare_stats "$run_dir/actual_stats.txt" "$stats_fixture"

  local metadata
  metadata="$(metadata_path_from_config)"
  extract_rank_fixture "$metadata" "$run_dir/actual_ranks.txt"
  compare_rank_fixture "$run_dir/actual_ranks.txt" "$rank_fixture"

  ./build/pca_lb_soundness_probe --config "$config" --gt "$gt_json" \
    --queries 50
  ./build/synthetic_residual_only_unit_test
}

case "$mode" in
  static|--static-only)
    run_static_checks
    ;;
  --build-only)
    run_static_checks
    run_build_only
    ;;
  all)
    run_all
    ;;
  *)
    echo "Usage: $0 [static|--static-only|--build-only|all]" >&2
    exit 2
    ;;
esac
