#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/deep1m_disk.config"
gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"

ASSIGN_GATE_SECONDS=8.5
TOTAL_WALL_GATE_SECONDS=99.5
DEEP1M_BUILD_GATE_SECONDS=8.5
ASSIGNMENT_RECALL_GATE=0.999
SEARCH_RECALL_DRIFT_GATE=1e-4

run_static_checks() {
  python3 - <<'PY'
import re
import sys
from pathlib import Path

root = Path(".")
errors = []

def read(path):
    p = root / path
    if not p.exists():
        errors.append(f"missing {path}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")

def extract_function_body(text, marker):
    start = text.find(marker)
    if start < 0:
        errors.append(f"missing function marker: {marker}")
        return ""
    brace = text.find("{", start)
    if brace < 0:
        errors.append(f"missing opening brace for: {marker}")
        return ""
    depth = 0
    for pos in range(brace, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:pos]
    errors.append(f"missing closing brace for: {marker}")
    return ""

config_reader = read("lib/ConfigReader.h")
resolved = read("lib/ResolvedConfig.h")
config_loader = read("lib/ConfigLoader.h")
sqg_index = read("lib/SQGCentroidIndex.h")
qg = read("third/symqglib/qg/qg.hpp")
kmeans = read("lib/Kmeans.h")
disk_range = read("lib/DiskRange.h")
probe = read("tests/pca_lb_soundness_probe.cpp")
cmake = read("CMakeLists.txt")

for path, text in [
    ("lib/ConfigReader.h", config_reader),
    ("lib/ResolvedConfig.h", resolved),
    ("lib/ConfigLoader.h", config_loader),
]:
    if "sqg_centroid_ef_search" not in text:
        errors.append(f"{path} missing sqg_centroid_ef_search")

if 'key == "sqg_centroid_ef_search"' not in config_reader:
    errors.append("ConfigReader.h does not parse sqg_centroid_ef_search")
if "resolved.sqg_centroid_ef_search = config_reader.sqg_centroid_ef_search" not in config_loader:
    errors.append("ConfigLoader.h does not copy sqg_centroid_ef_search")
if "Deprecated after index-graph-sqg" not in resolved:
    errors.append("ResolvedConfig.h hnsw_path missing deprecation comment")

if "make_search_context()" not in sqg_index:
    errors.append("SQGCentroidIndex.h missing make_search_context public wrapper")
if "search_into(" not in sqg_index:
    errors.append("SQGCentroidIndex.h missing non-allocating search_into")
if "std::vector<uint32_t> search(const float* query, size_t nprobe) const" not in sqg_index:
    errors.append("SQGCentroidIndex::search signature changed")
search_into_body = extract_function_body(sqg_index, "void search_into(")
if search_into_body:
    for token in ["ctx.search_pool.clear()", "ctx.visited.clear()", "search_qg"]:
        if token not in search_into_body:
            errors.append(f"SQGCentroidIndex::search_into missing {token}")
    first_clear = search_into_body.find("ctx.search_pool.clear()")
    second_clear = search_into_body.find("ctx.visited.clear()")
    search_call = search_into_body.find("search_qg")
    if not (0 <= first_clear < search_call and 0 <= second_clear < search_call):
        errors.append("search_into must clear ctx before search_qg")

for marker in ["void search_qg(", "SearchContext make_search_context() const"]:
    idx = qg.find(marker)
    if idx < 0:
        errors.append(f"qg.hpp missing marker {marker}")
        continue
    last_public = qg.rfind("public:", 0, idx)
    last_private = qg.rfind("private:", 0, idx)
    if last_public < last_private:
        errors.append(f"qg.hpp marker is not public: {marker}")

for token in ["HierarchicalNSW", "searchBaseLayerST", "config.hnsw_path", "centroid_hnsw"]:
    if token in kmeans:
        errors.append(f"lib/Kmeans.h still contains HNSW build/assignment marker: {token}")
for token in [
    '#include "SQGCentroidIndex.h"',
    "const SQGCentroidIndex& sqg",
    "sqg.make_search_context()",
    "sqg.search_into",
    "struct TwoChoice",
    "alt_dist < best_dist * 1.2f",
    "km.centroid_sqg_build+save",
    "config.sqg_centroid_ef_search",
]:
    if token not in kmeans:
        errors.append(f"lib/Kmeans.h missing SQG assignment marker: {token}")

search_body = extract_function_body(disk_range, "RunResult search(")
for token in ["use_hnsw", "searchBaseLayerST", "config.hnsw_path"]:
    if token in search_body:
        errors.append(f"DiskRange::search still contains HNSW slow-path marker: {token}")
for token in [
    "SQGCentroidIndex sqg_slow_path",
    "std::filesystem::exists(config.sqg_path)",
    "sqg_slow_path.search",
    "utils::L2Sqr(state.query_ptr",
    "index-graph-sqg",
]:
    if token not in search_body:
        errors.append(f"DiskRange::search missing SQG slow-path marker: {token}")

fp_body = extract_function_body(disk_range, "static void build_fast_path_artifacts(")
if "fp.sqg_load+retune" not in fp_body:
    errors.append("build_fast_path_artifacts missing fp.sqg_load+retune timer")
if ".build(" in fp_body and "sqg.build(" in fp_body:
    errors.append("build_fast_path_artifacts still rebuilds SQG")
if "sqg.save(config.sqg_path)" in fp_body:
    errors.append("build_fast_path_artifacts still saves SQG")
if "sqg.load(config.sqg_path" not in fp_body:
    errors.append("build_fast_path_artifacts does not load existing SQG")

for token in ["hnswlib", "HierarchicalNSW", "searchBaseLayerST", "config.hnsw_path"]:
    if token in probe:
        errors.append(f"pca_lb_soundness_probe.cpp still contains HNSW marker: {token}")
for token in ['#include "../lib/SQGCentroidIndex.h"', "SQGCentroidIndex sqg_probe", "sqg.search(query, limit)"]:
    if token not in probe:
        errors.append(f"pca_lb_soundness_probe.cpp missing SQG marker: {token}")

for token in ["index_graph_sqg_reused_ctx_test", "tests/index_graph_sqg_reused_ctx_test.cpp"]:
    if token not in cmake:
        errors.append(f"CMakeLists.txt missing {token}")

if ASSIGN_GATE_SECONDS := re.search(r"ASSIGN_GATE_SECONDS=([0-9.]+)", read("tests/index_graph_sqg_test.sh")):
    if ASSIGN_GATE_SECONDS.group(1) != "8.5":
        errors.append("ASSIGN_GATE_SECONDS must be 8.5")
else:
    errors.append("tests/index_graph_sqg_test.sh missing ASSIGN_GATE_SECONDS")

if errors:
    print("index_graph_sqg static checks failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)

print("index_graph_sqg static checks passed")
PY
}

run_build_only() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target build build_profile search search_stats build_stats \
    main build_prune_stats pca_lb_soundness_probe synthetic_residual_only_unit_test \
    index_graph_sqg_reused_ctx_test -j
}

run_reused_ctx_only() {
  if [[ ! -x build/index_graph_sqg_reused_ctx_test ]]; then
    run_build_only
  fi
  ./build/index_graph_sqg_reused_ctx_test
}

run_deep1m_only() {
  if [[ ! -f "$config" ]]; then
    echo "missing config: $config" >&2
    exit 1
  fi
  if [[ ! -f "$gt_json" ]]; then
    echo "missing gt json: $gt_json" >&2
    exit 1
  fi
  if [[ ! -x build/build_profile || ! -x build/search || ! -x build/pca_lb_soundness_probe || ! -x build/index_graph_sqg_reused_ctx_test ]]; then
    run_build_only
  fi

  local log_dir
  log_dir="$(mktemp -d)"
  local build_log="$log_dir/deep1m_build_profile.log"
  local search_log="$log_dir/deep1m_search.log"
  local probe_log="$log_dir/deep1m_pca_probe.log"

  ./build/build_profile --config "$config" --gt "$gt_json" >"$build_log" 2>&1
  python3 - "$build_log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
build_times = re.findall(r"^build_time\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
if not build_times:
    raise SystemExit("missing build_time in deep1m build_profile log")
build_time = float(build_times[-1])
if build_time > 8.5:
    raise SystemExit(f"deep1m build_time gate failed: {build_time} > 8.5")
for forbidden in ["km.centroid_hnsw_build+save", "_hnsw_file.bin"]:
    if forbidden in text:
        raise SystemExit(f"deep1m build log contains forbidden marker: {forbidden}")
if "km.centroid_sqg_build+save" not in text:
    raise SystemExit("deep1m build log missing km.centroid_sqg_build+save")
print(f"deep1m build_time={build_time}")
PY

  ./build/index_graph_sqg_reused_ctx_test --config "$config" --gt "$gt_json" \
    --queries 100

  ./build/search --config "$config" --gt "$gt_json" >"$search_log" 2>&1
  python3 - "$search_log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"^recall\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
if not match:
    raise SystemExit("missing recall in deep1m search log")
print(f"deep1m recall={match.group(1)}")
PY

  ./build/pca_lb_soundness_probe --config "$config" --gt "$gt_json" \
    --queries 50 >"$probe_log" 2>&1
  tail -n 5 "$probe_log"
}

run_deep10m_only() {
  local deep10m_config="configs/deep10m_disk_fast_path.config"
  local deep10m_gt="experiments/range_gt/output/deep10m_0.2002_gt.json"
  if [[ ! -f "$deep10m_config" ]]; then
    echo "missing config: $deep10m_config" >&2
    exit 1
  fi
  if [[ ! -f "$deep10m_gt" ]]; then
    echo "missing gt json: $deep10m_gt" >&2
    exit 1
  fi
  if [[ ! -x build/build_profile ]]; then
    run_build_only
  fi

  local log_dir
  log_dir="$(mktemp -d)"
  for i in 1 2 3; do
    local log="$log_dir/deep10m_run_${i}.log"
    echo "=== deep10m run $i start: $(date +%H:%M:%S) ==="
    ./build/build_profile --config "$deep10m_config" --gt "$deep10m_gt" \
      >"$log" 2>&1
    echo "=== deep10m run $i end:   $(date +%H:%M:%S) ==="
  done

  ASSIGN_GATE="$ASSIGN_GATE_SECONDS" \
  TOTAL_GATE="$TOTAL_WALL_GATE_SECONDS" \
  python3 - "$log_dir" <<'PY'
import os
import re
import statistics
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
assign_gate = float(os.environ["ASSIGN_GATE"])
total_gate = float(os.environ["TOTAL_GATE"])

assign_re = re.compile(
    r"\[build-timing\]\s+km\.assign_full_data\s*=\s*([0-9.]+)"
)
build_re = re.compile(r"^build_time\s*=\s*([0-9.]+)", re.MULTILINE)
sqg_phase_re = re.compile(r"km\.centroid_sqg_build\+save")
hnsw_phase_re = re.compile(r"km\.centroid_hnsw_build\+save")
fp_load_re = re.compile(r"fp\.sqg_load\+retune")
fp_build_re = re.compile(r"fp\.sqg_build\+save")

assigns = []
totals = []
for i in (1, 2, 3):
    log = log_dir / f"deep10m_run_{i}.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    m_assign = assign_re.search(text)
    m_total = build_re.search(text)
    if not m_assign:
        raise SystemExit(f"run {i}: missing km.assign_full_data in log")
    if not m_total:
        raise SystemExit(f"run {i}: missing build_time in log")
    if not sqg_phase_re.search(text):
        raise SystemExit(f"run {i}: missing km.centroid_sqg_build+save phase")
    if hnsw_phase_re.search(text):
        raise SystemExit(
            f"run {i}: forbidden km.centroid_hnsw_build+save phase present"
        )
    if not fp_load_re.search(text):
        raise SystemExit(f"run {i}: missing fp.sqg_load+retune phase")
    if fp_build_re.search(text):
        raise SystemExit(f"run {i}: forbidden fp.sqg_build+save phase present")
    assigns.append(float(m_assign.group(1)))
    totals.append(float(m_total.group(1)))

assign_median = statistics.median(assigns)
total_median = statistics.median(totals)

print(f"deep10m km.assign_full_data runs (s): {assigns}")
print(f"deep10m total build_time runs (s):    {totals}")
print(f"deep10m km.assign_full_data median:   {assign_median:.3f}")
print(f"deep10m total build_time median:      {total_median:.3f}")
print(f"gates: assign <= {assign_gate}, total <= {total_gate}")

failures = []
if assign_median > assign_gate:
    failures.append(
        f"km.assign_full_data median {assign_median:.3f} > {assign_gate}"
    )
if total_median > total_gate:
    failures.append(
        f"total build_time median {total_median:.3f} > {total_gate}"
    )

if failures:
    print("deep10m gate FAILED:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    sys.exit(1)

# Stretch goals (informational, not gated)
stretch_assign = 4.0
stretch_total = 92.0
notes = []
if assign_median <= stretch_assign:
    notes.append(f"stretch met: assign {assign_median:.3f} <= {stretch_assign}")
if total_median <= stretch_total:
    notes.append(f"stretch met: total {total_median:.3f} <= {stretch_total}")
if notes:
    print("INFO: " + "; ".join(notes))

print("deep10m gate PASSED")
PY
}

mode="${1:-all}"
case "$mode" in
  --static-only|static)
    run_static_checks
    ;;
  --build-only|build)
    run_build_only
    ;;
  --reused-ctx-only|reused-ctx)
    run_reused_ctx_only
    ;;
  --deep1m-only|deep1m)
    run_deep1m_only
    ;;
  --deep10m-only|deep10m)
    run_deep10m_only
    ;;
  all)
    run_static_checks
    run_build_only
    run_reused_ctx_only
    run_deep1m_only
    run_deep10m_only
    ;;
  *)
    echo "Usage: $0 [--static-only|--build-only|--reused-ctx-only|--deep1m-only|--deep10m-only|all]" >&2
    exit 2
    ;;
esac
