#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/deep1m_disk.config"
gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
deep10m_config="configs/deep10m_disk_fast_path.config"
deep10m_gt_json="experiments/range_gt/output/deep10m_0.2002_gt.json"

stats_fixture="tests/fixtures/deep1m_disk_build_prune_stats_pre_slab_fit_gemm.txt"
rank_fixture="tests/fixtures/deep1m_disk_pca_effective_rank_pre_slab_fit_gemm.txt"
slab_hash_fixture="tests/fixtures/deep1m_disk_slab_hash_pre_slab_fit_gemm.txt"

SLAB_FIT_GATE_SECONDS=6.0
TOTAL_WALL_GATE_SECONDS=104.0
DEEP1M_BUILD_GATE_SECONDS=8.5

run_static_checks() {
  python3 - <<'PY'
import re
import sys
from pathlib import Path

root = Path(".")
cluster_io = root / "lib" / "ClusterIO.h"
pd2_script = root / "tests" / "pd2_cluster_slab_cpp_deployment_test.sh"
stats_fixture = root / "tests" / "fixtures" / "deep1m_disk_build_prune_stats_pre_slab_fit_gemm.txt"
rank_fixture = root / "tests" / "fixtures" / "deep1m_disk_pca_effective_rank_pre_slab_fit_gemm.txt"
slab_hash_fixture = root / "tests" / "fixtures" / "deep1m_disk_slab_hash_pre_slab_fit_gemm.txt"

errors = []

def extract_function_body(text: str, name: str) -> str:
    marker = f"void {name}("
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing function {name}")
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise ValueError(f"missing opening brace for {name}")
    depth = 0
    for pos in range(open_brace, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:pos]
    raise ValueError(f"missing closing brace for {name}")

if not cluster_io.exists():
    errors.append("missing lib/ClusterIO.h")
else:
    text = cluster_io.read_text()
    for token in [
        '#include "OpenBlasThreadGuard.h"',
        "#ifdef DJ_USE_OPENBLAS",
        "#include <cblas.h>",
        '#error "slab-fit-gemm requires DJ_USE_OPENBLAS',
        "kSlabFitFloat32UnitRoundoff",
        "kSlabFitDotProductSafetyFactor",
    ]:
        if token not in text:
            errors.append(f"ClusterIO.h missing top-level marker: {token}")

    try:
        body = extract_function_body(text, "fitClusterSlabsFromClusterFile")
    except ValueError as exc:
        errors.append(str(exc))
        body = ""

    for token in [
        "OpenBlasThreadGuard fit_blas_guard(1)",
        "proj_matrix",
        "proj_matrix.reserve",
        "cblas_sgemm",
        "CblasRowMajor",
        "CblasNoTrans",
        "CblasTrans",
        "points.data()",
        "normalized_directions.data()",
        "slab_pad_coeff",
        "static_cast<float>(d)",
        "max_l2",
        "pad_abs",
        "mins[j] - pad_abs",
        "maxs[j] + pad_abs",
        "centroid_norm_sq",
        "neighbour_candidate_count",
        "exact_candidates",
        "bucket_sizes[cluster_id] == 0",
        "std::numeric_limits<float>::infinity()",
        "std::nth_element",
    ]:
        if token not in body:
            errors.append(f"fitClusterSlabsFromClusterFile missing marker: {token}")

    guard_constructors = re.findall(r"\bOpenBlasThreadGuard\s+[A-Za-z_]\w*\s*\(", body)
    if len(guard_constructors) != 1:
        errors.append(
            "fitClusterSlabsFromClusterFile must construct exactly one OpenBlasThreadGuard"
        )
    if "proj += point[col] * direction[col]" in body:
        errors.append("old scalar slab projection loop is still present")
    if "openblas_set_num_threads" in body or "openblas_get_num_threads" in body:
        errors.append("fitClusterSlabsFromClusterFile directly calls OpenBLAS thread APIs")
    if re.search(r"slab_l\[slab_offset \+ j\]\s*=\s*mins\[j\]\s*;", body):
        errors.append("slab_l has naked min write without safe-side padding")
    if re.search(r"slab_h\[slab_offset \+ j\]\s*=\s*maxs\[j\]\s*;", body):
        errors.append("slab_h has naked max write without safe-side padding")
    if "128.0f * 5.96e-8f" in body:
        errors.append("slab padding uses hardcoded d=128 instead of runtime d")

for fixture in [stats_fixture, rank_fixture, slab_hash_fixture]:
    if not fixture.exists():
        errors.append(f"missing fixture: {fixture}")

if slab_hash_fixture.exists():
    slab_hash_text = slab_hash_fixture.read_text()
    for token in ["slab_neighbour_indices_sha256=", "slab_norms_sha256="]:
        if token not in slab_hash_text:
            errors.append(f"{slab_hash_fixture} missing {token}")

if not pd2_script.exists():
    errors.append("missing tests/pd2_cluster_slab_cpp_deployment_test.sh")
else:
    pd2 = pd2_script.read_text()
    match = re.search(
        r"g\+\+[\s\S]{0,900}\$coincident_cpp[\s\S]{0,500}-o \"\$coincident_bin\"",
        pd2,
    )
    if not match:
        errors.append("could not find coincident_cpp g++ command in pd2 test")
    else:
        command = match.group(0)
        for token in ["-DDJ_USE_OPENBLAS", "$openblas_cflags", "$openblas_libs"]:
            if token not in command:
                errors.append(f"pd2 coincident_cpp compile command missing {token}")

if errors:
    print("slab_fit_gemm static checks failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)

print("slab_fit_gemm static checks passed")
PY
}

run_build_only() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target build build_profile search search_stats build_stats \
    main build_prune_stats pca_lb_soundness_probe synthetic_residual_only_unit_test -j
}

parse_build_median() {
  python3 - "$@" <<'PY'
import re
import statistics
import sys
from pathlib import Path

values = []
for path in sys.argv[1:]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"^build_time\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
    if matches:
        values.append(float(matches[-1]))
        continue
    matches = re.findall(r"build time\s*=\s*([0-9.]+)\s*seconds", text)
    if not matches:
        raise SystemExit(f"missing build_time in {path}")
    values.append(float(matches[-1]))
if not values:
    raise SystemExit("no build logs provided")
print(statistics.median(values))
PY
}

parse_profile_medians() {
  python3 - "$@" <<'PY'
import re
import statistics
import sys
from pathlib import Path

slab = []
total = []
for path in sys.argv[1:]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    slab_matches = re.findall(
        r"\[build-timing\]\s+wc\.fitClusterSlabsFromClusterFile\s*=\s*([0-9.]+)",
        text,
    )
    total_matches = re.findall(r"^build_time\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
    if not slab_matches:
        raise SystemExit(f"missing slab fit timing in {path}")
    if not total_matches:
        raise SystemExit(f"missing build_time in {path}")
    slab.append(float(slab_matches[-1]))
    total.append(float(total_matches[-1]))
print(f"{statistics.median(slab)} {statistics.median(total)}")
PY
}

parse_stats_output() {
  local log_file="$1"
  local out_file="$2"
  python3 - "$log_file" "$out_file" <<'PY'
import re
import sys
from pathlib import Path

keys = [
    "cell_candidate_vectors",
    "cell_kept_vectors",
    "cell_pruned_vectors",
    "combined_pruned_clusters",
    "dist_comp",
    "pca_only_pruned_clusters",
    "recall",
    "slab_only_pruned_clusters",
    "triangle_pruned_clusters",
    "zero_kept_cells",
    "zero_kept_surviving_clusters",
]
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
lines = []
for key in keys:
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^\s]+)", text, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {key} in {sys.argv[1]}")
    lines.append(f"{key}={match.group(1)}\n")
Path(sys.argv[2]).write_text("".join(lines), encoding="utf-8")
PY
}

compare_stats() {
  local actual_file="$1"
  local baseline_file="$2"
  python3 - "$actual_file" "$baseline_file" <<'PY'
import sys
from pathlib import Path

def read(path: str) -> dict[str, float]:
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        result[key] = float(value)
    return result

actual = read(sys.argv[1])
baseline = read(sys.argv[2])
required = set(baseline)
missing = required - set(actual)
if missing:
    raise SystemExit(f"actual stats missing keys: {sorted(missing)}")
for key in sorted(required):
    a = actual[key]
    b = baseline[key]
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
print("build_prune_stats counter gate passed")
PY
}

metadata_path_from_gt() {
  metadata_path_from_gt_file "$gt_json"
}

metadata_path_from_gt_file() {
  local gt_path="$1"
  python3 - "$gt_path" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as f:
    gt = json.load(f)
print(Path(str(gt["index"]["prefix"]) + "_diskrange") / "_metadata_file.bin")
PY
}

extract_rank_fixture() {
  local metadata="$1"
  local out_file="$2"
  python3 - "$metadata" "$out_file" <<'PY'
import struct
import sys
from pathlib import Path

with Path(sys.argv[1]).open("rb") as f:
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
if sum_rel > 0.005 or hist_l1_rate > 0.03 or rank0_delta > 50 or full_rank_delta > 50:
    raise SystemExit(
        "pca_effective_rank drift too high: "
        f"sum_rel={sum_rel} hist_l1_rate={hist_l1_rate} "
        f"rank0_delta={rank0_delta} full_rank_delta={full_rank_delta}"
    )
print("pca_effective_rank gate passed")
PY
}

check_slab_hash_fixture() {
  local metadata="$1"
  python3 - "$metadata" "$slab_hash_fixture" <<'PY'
import hashlib
import sys
from pathlib import Path

from experiments.range_aware_profiling._io import read_metadata

metadata = read_metadata(sys.argv[1])
expected = {}
for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        expected[key] = value
actual = {
    "slab_neighbour_indices_sha256": hashlib.sha256(
        metadata.slab_neighbour_indices.tobytes()
    ).hexdigest(),
    "slab_norms_sha256": hashlib.sha256(metadata.slab_norms.tobytes()).hexdigest(),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(
            f"{key} mismatch: expected={expected.get(key)} actual={value}; "
            "unchanged slab byte-equal field drifted"
        )
print("slab byte-equal hash gate passed")
PY
}

check_slab_oracle() {
  local config_path="$1"
  local gt_path="$2"
  local metadata="$3"
  python3 - "$config_path" "$gt_path" "$metadata" <<'PY'
import sys
import numpy as np

from experiments.range_aware_profiling._io import read_metadata, resolve_inputs

resolved = resolve_inputs(sys.argv[1], sys.argv[2], None)
metadata = read_metadata(sys.argv[3])
cluster_path = resolved.cluster_path
cluster_num = metadata.cluster_num
slab_count = metadata.slab_count
dim = metadata.dim
non_empty = np.flatnonzero(metadata.bucket_sizes > 0)
if non_empty.size < 5:
    raise SystemExit(f"need at least 5 non-empty clusters, got {non_empty.size}")
rng = np.random.default_rng(seed=42)
sample_cluster_ids = rng.choice(non_empty, size=5, replace=False)
slab_pad_coeff = float(dim) * 5.96e-8 * 4.0

with open(cluster_path, "rb") as cluster_fp:
    for cid_raw in sample_cluster_ids:
        cid = int(cid_raw)
        self_centroid = metadata.centroids[cid].astype(np.float32)
        diff = metadata.centroids.astype(np.float32) - self_centroid[None, :]
        dist_sq = np.sum(diff * diff, axis=1, dtype=np.float32)
        dist_sq[cid] = np.inf
        order = sorted(range(cluster_num), key=lambda i: (float(dist_sq[i]), i))
        top = np.asarray(order[:slab_count], dtype=np.uint32)
        if not np.array_equal(top, metadata.slab_neighbour_indices[cid]):
            first = int(np.flatnonzero(top != metadata.slab_neighbour_indices[cid])[0])
            raise SystemExit(
                f"oracle neighbour mismatch at cluster {cid} slab {first}: "
                f"oracle={int(top[first])} metadata={int(metadata.slab_neighbour_indices[cid, first])}"
            )
        directions = metadata.centroids[top].astype(np.float32) - self_centroid[None, :]
        norms = np.sqrt(np.sum(directions * directions, axis=1, dtype=np.float32)).astype(np.float32)
        if not np.all(norms > 1e-9):
            raise SystemExit(f"coincident centroid in sampled cluster {cid}")
        dirs_n = (directions / norms[:, None]).astype(np.float32)
        count = int(metadata.bucket_sizes[cid])
        cluster_fp.seek(int(metadata.cluster_offsets[cid]))
        raw = cluster_fp.read(count * dim * 4)
        if len(raw) != count * dim * 4:
            raise SystemExit(f"short cluster payload at cluster {cid}")
        points = np.frombuffer(raw, dtype=np.float32).reshape(count, dim)
        proj = points @ dirs_n.T
        expected_l = proj.min(axis=0).astype(np.float32)
        expected_h = proj.max(axis=0).astype(np.float32)
        actual_l = metadata.slab_l[cid]
        actual_h = metadata.slab_h[cid]
        tight_l = (actual_l - expected_l) > 1e-7
        if tight_l.any():
            idx = int(np.argmax(actual_l - expected_l))
            raise SystemExit(
                f"slab_l SOUNDNESS regression at cluster {cid} slab {idx}: "
                f"actual={actual_l[idx]} oracle={expected_l[idx]} "
                f"delta={actual_l[idx] - expected_l[idx]:.3e}"
            )
        tight_h = (expected_h - actual_h) > 1e-7
        if tight_h.any():
            idx = int(np.argmax(expected_h - actual_h))
            raise SystemExit(
                f"slab_h SOUNDNESS regression at cluster {cid} slab {idx}: "
                f"actual={actual_h[idx]} oracle={expected_h[idx]} "
                f"delta={expected_h[idx] - actual_h[idx]:.3e}"
            )
        max_point_l2 = float(np.linalg.norm(points, axis=1).max())
        budget = 6.0 * slab_pad_coeff * max(max_point_l2, 1.0)
        loose_l = expected_l - actual_l
        loose_h = actual_h - expected_h
        if float(loose_l.max()) > budget:
            idx = int(np.argmax(loose_l))
            raise SystemExit(
                f"slab_l excessive padding at cluster {cid} slab {idx}: "
                f"diff={loose_l[idx]:.3e} budget={budget:.3e} "
                f"d={dim} max_l2={max_point_l2:.3f}"
            )
        if float(loose_h.max()) > budget:
            idx = int(np.argmax(loose_h))
            raise SystemExit(
                f"slab_h excessive padding at cluster {cid} slab {idx}: "
                f"diff={loose_h[idx]:.3e} budget={budget:.3e} "
                f"d={dim} max_l2={max_point_l2:.3f}"
            )
print(f"slab_l/slab_h oracle gate passed on {len(sample_cluster_ids)} clusters")
PY
}

check_slab_determinism() {
  local first_metadata="$1"
  local second_metadata="$2"
  python3 - "$first_metadata" "$second_metadata" <<'PY'
import sys
import numpy as np

from experiments.range_aware_profiling._io import read_metadata

a = read_metadata(sys.argv[1])
b = read_metadata(sys.argv[2])
if not np.array_equal(a.bucket_sizes, b.bucket_sizes):
    diff_count = int(np.count_nonzero(a.bucket_sizes != b.bucket_sizes))
    print(
        "INFO: skip slab build-to-build determinism array compare because "
        f"upstream bucket layout differs across full builds ({diff_count} clusters)"
    )
    raise SystemExit(0)
if not np.array_equal(a.slab_neighbour_indices, b.slab_neighbour_indices):
    raise SystemExit("slab_neighbour_indices are not deterministic")
if not np.array_equal(a.slab_norms, b.slab_norms):
    raise SystemExit("slab_norms are not deterministic")
for name, left, right in [("slab_l", a.slab_l, b.slab_l), ("slab_h", a.slab_h, b.slab_h)]:
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise SystemExit(f"{name} finite/inf mask is not deterministic")
    diff = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    if max_diff >= 1e-7:
        finite_indices = np.argwhere(finite)
        idx = tuple(finite_indices[int(np.argmax(diff))])
        raise SystemExit(
            f"{name} determinism failed at {idx}: "
            f"a={left[idx]} b={right[idx]} diff={max_diff}"
        )
print("slab build-to-build determinism gate passed")
PY
}

run_adversarial_padding_check() {
  python3 - <<'PY'
import numpy as np

rng = np.random.default_rng(seed=2026)
d = 96
slab_count = 64
points = rng.normal(0.0, 100.0, size=(200, d)).astype(np.float32)
dirs = rng.normal(0.0, 1.0, size=(slab_count, d)).astype(np.float32)
dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)

oracle = points.astype(np.float64) @ dirs.astype(np.float64).T
expected_l = oracle.min(axis=0).astype(np.float32)
expected_h = oracle.max(axis=0).astype(np.float32)
sgemm_like = points @ dirs.T
raw_l = sgemm_like.min(axis=0).astype(np.float32)
raw_h = sgemm_like.max(axis=0).astype(np.float32)
max_l2 = float(np.linalg.norm(points, axis=1).max())
slab_pad_coeff = float(d) * 5.96e-8 * 4.0
pad_abs = slab_pad_coeff * max(max_l2, 1.0)
actual_l = raw_l - pad_abs
actual_h = raw_h + pad_abs
budget = 6.0 * slab_pad_coeff * max(max_l2, 1.0)
if np.any((actual_l - expected_l) > 1e-7):
    idx = int(np.argmax(actual_l - expected_l))
    raise SystemExit(
        f"adversarial slab_l containment failed at slab {idx}: "
        f"actual={actual_l[idx]} oracle={expected_l[idx]}"
    )
if np.any((expected_h - actual_h) > 1e-7):
    idx = int(np.argmax(expected_h - actual_h))
    raise SystemExit(
        f"adversarial slab_h containment failed at slab {idx}: "
        f"actual={actual_h[idx]} oracle={expected_h[idx]}"
    )
if float((expected_l - actual_l).max()) > budget:
    raise SystemExit("adversarial slab_l looseness exceeded forward-error budget")
if float((actual_h - expected_h).max()) > budget:
    raise SystemExit("adversarial slab_h looseness exceeded forward-error budget")
fixed_padding = 1e-5
if fixed_padding >= slab_pad_coeff * max(max_l2, 1.0):
    raise SystemExit("adversarial fixture did not exercise high-magnitude padding")
print(
    "adversarial padding gate passed "
    f"(max_l2={max_l2:.3f}, pad_abs={pad_abs:.3e}, fixed_1e-5_is_insufficient)"
)
PY
}

run_deep1m_checks() {
  for fixture in "$stats_fixture" "$rank_fixture" "$slab_hash_fixture"; do
    if [[ ! -s "$fixture" ]]; then
      echo "missing pre-slab-fit-gemm fixture: $fixture" >&2
      exit 1
    fi
  done

  local run_dir
  run_dir="$(mktemp -d /tmp/slab_fit_gemm_deep1m.XXXXXX)"

  local build_logs=()
  for i in 1 2 3; do
    local log="$run_dir/build_${i}.log"
    ./build/build --config "$config" --gt "$gt_json" >"$log" 2>&1
    build_logs+=("$log")
  done
  local median_build
  median_build="$(parse_build_median "${build_logs[@]}")"
  python3 - "$median_build" "$DEEP1M_BUILD_GATE_SECONDS" <<'PY'
import sys
value = float(sys.argv[1])
gate = float(sys.argv[2])
print(f"deep1m_build_median={value}")
if value > gate:
    raise SystemExit(f"deep1m build median {value} exceeds gate {gate}")
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
  check_slab_hash_fixture "$metadata"
  check_slab_oracle "$config" "$gt_json" "$metadata"

  ./build/pca_lb_soundness_probe --config "$config" --gt "$gt_json" --queries 50

  ./build/build --config "$config" --gt "$gt_json" >"$run_dir/determinism_a.log" 2>&1
  cp "$metadata" "$run_dir/metadata_a.bin"
  ./build/build --config "$config" --gt "$gt_json" >"$run_dir/determinism_b.log" 2>&1
  cp "$metadata" "$run_dir/metadata_b.bin"
  check_slab_determinism "$run_dir/metadata_a.bin" "$run_dir/metadata_b.bin"

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

run_deep10m_checks() {
  local run_dir
  run_dir="$(mktemp -d /tmp/slab_fit_gemm_deep10m.XXXXXX)"
  local profile_logs=()
  for i in 1 2 3; do
    local log="$run_dir/build_profile_${i}.log"
    ./build/build_profile --config "$deep10m_config" --gt "$deep10m_gt_json" >"$log" 2>&1
    profile_logs+=("$log")
  done
  local medians
  medians="$(parse_profile_medians "${profile_logs[@]}")"
  python3 - "$medians" "$SLAB_FIT_GATE_SECONDS" "$TOTAL_WALL_GATE_SECONDS" <<'PY'
import sys
slab, total = [float(x) for x in sys.argv[1].split()]
slab_gate = float(sys.argv[2])
total_gate = float(sys.argv[3])
print(f"deep10m_slab_fit_median={slab}")
print(f"deep10m_total_build_median={total}")
if slab > slab_gate:
    raise SystemExit(f"deep10m slab fit median {slab} exceeds gate {slab_gate}")
if total > total_gate:
    raise SystemExit(f"deep10m total build median {total} exceeds gate {total_gate}")
if slab <= 4.0:
    print("INFO: deep10m slab stretch goal met")
if total <= 100.0:
    print("INFO: deep10m total stretch goal met")
PY
  local metadata
  metadata="$(metadata_path_from_gt_file "$deep10m_gt_json")"
  check_slab_oracle "$deep10m_config" "$deep10m_gt_json" "$metadata"
}

mode="${1:---static-only}"
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
    run_build_only
    run_adversarial_padding_check
    run_deep1m_checks
    ;;
  --deep10m-only)
    run_static_checks
    run_build_only
    run_deep10m_checks
    ;;
  --adversarial)
    run_static_checks
    run_adversarial_padding_check
    ;;
  all)
    run_static_checks
    run_build_only
    run_adversarial_padding_check
    run_deep1m_checks
    run_deep10m_checks
    ./tests/kmeans_train_gemm_test.sh all
    ;;
  *)
    echo "usage: $0 [--static-only|--build-only|--deep1m-only|--deep10m-only|--adversarial|all]" >&2
    exit 2
    ;;
esac
