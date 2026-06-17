#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
from pathlib import Path

def require_contains(path: str, needle: str) -> None:
    text = Path(path).read_text(encoding="utf-8")
    if needle not in text:
        raise SystemExit(f"{path}: missing marker: {needle}")

cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
for marker in (
    "add_executable(approx_alpha_config_test tests/approx_alpha_config_test.cpp)",
    "add_executable(approx_alpha_data_hnsw_invariance_test tests/approx_alpha_data_hnsw_invariance_test.cpp)",
    "add_executable(approx_alpha_search_stats_prod expr/approx_alpha_search_stats_prod/main.cpp)",
    "target_compile_definitions(approx_alpha_search_stats_prod PRIVATE DJ_ENABLE_SEARCH_PHASE_TIMING=0)",
    "target_compile_definitions(approx_alpha_search_stats_prod PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=0)",
    "add_executable(approx_alpha_search_stats_breakdown expr/search_stats/main.cpp)",
    "target_compile_definitions(approx_alpha_search_stats_breakdown PRIVATE DJ_ENABLE_SEARCH_PHASE_TIMING=1)",
    "target_compile_definitions(approx_alpha_search_stats_breakdown PRIVATE DJ_ENABLE_PRUNE_BREAKDOWN_STATS=1)",
    "add_test(\n    NAME approx_alpha_config",
    "add_test(\n    NAME approx_alpha_static",
    "add_test(\n    NAME approx_alpha_sweep_harness",
    "add_test(\n    NAME approx_alpha_alpha1_regression",
    "add_test(\n    NAME approx_alpha_data_hnsw_invariance",
):
    if marker not in cmake:
        raise SystemExit(f"CMakeLists.txt: missing marker: {marker}")

disk_range = Path("lib/DiskRange.h").read_text(encoding="utf-8")
for marker in (
    "const float epsilon =\n        std::sqrt(static_cast<float>(config.radius_squared)) / config.approx_alpha;",
    "const float radius_l2_f = std::sqrt(static_cast<float>(config.radius_squared));",
    "const float radius_sq_f = static_cast<float>(config.radius_squared);",
    "oracle.evaluate(state.query_ptr, radius_l2_f,",
    "if (cell_lb <= epsilon)",
    "lb > radius_sq_f",
    "ub <= radius_sq_f",
    "dist <= radius_sq_f",
):
    if marker not in disk_range:
        raise SystemExit(f"lib/DiskRange.h: missing marker: {marker}")

prod_source = Path("expr/approx_alpha_search_stats_prod/main.cpp")
if not prod_source.exists():
    raise SystemExit("missing production report source")
prod_text = prod_source.read_text(encoding="utf-8")
for marker in (
    "--print-build-info",
    "recall_raw = ",
    "recall_verified = ",
    "qps = ",
    " query/sec",
    "search_time = ",
    "dist_comp = ",
    "radius_prescreened_clusters = ",
    "slab_pruned_clusters = ",
    "box_residual_pruned_clusters = ",
    "data_hnsw_skip_count = ",
    "fast_path_skip_count = ",
):
    if marker not in prod_text:
        raise SystemExit(f"production report source missing marker: {marker}")

search_stats = Path("expr/search_stats/main.cpp").read_text(encoding="utf-8")
for marker in (
    "--print-build-info",
    "DJ_ENABLE_SEARCH_PHASE_TIMING",
    "DJ_ENABLE_PRUNE_BREAKDOWN_STATS",
):
    if marker not in search_stats:
        raise SystemExit(f"search_stats source missing marker: {marker}")

proposal = Path("experiments/approx_alpha/proposal.md").read_text(encoding="utf-8")
for stale in (
    "radius_l2_f = epsilon",
    "实际只改 1 行",
    "prune_aggressiveness_alpha",
):
    if stale in proposal:
        raise SystemExit(f"experiments/approx_alpha/proposal.md: stale marker remains: {stale}")

require_contains(
    "experiments/PCA_enhancement/Formal Pipeline/pipeline_aligned.md",
    "user-tunable (sound when α=1)",
)
require_contains("experiments/approx_alpha/proposal.md", "openspec/changes/approx-alpha")

plot = Path("experiments/approx_alpha/plot_pareto.py")
if not plot.exists():
    raise SystemExit("missing plot_pareto.py")
plot_text = plot.read_text(encoding="utf-8")
for marker in (
    "_prod.csv",
    "_breakdown.csv",
    "diagnostic only - stats=ON disables Tier 2a",
    "baseline_mask",
    "marker=\"D\"",
    "knee candidate",
):
    if marker not in plot_text:
        raise SystemExit(f"plot_pareto.py missing marker: {marker}")

harness = Path("experiments/approx_alpha/alpha_sweep.py").read_text(encoding="utf-8")
for marker in (
    "git_dirty",
    "git_tracked_diff_sha256",
    "git_untracked_files_sha256",
    "compile_commands.json",
    "dataset_artifact_hashes",
    "alpha_is_baseline",
    "--print-build-info",
):
    if marker not in harness:
        raise SystemExit(f"alpha_sweep.py missing marker: {marker}")

pipeline = Path(
    "experiments/PCA_enhancement/Formal Pipeline/pipeline_aligned.md"
).read_text(encoding="utf-8")
for stale in (
    "Tier 2 — cluster_bound (SOUND, 已落地)",
    "Tier 3 — cell_bound (SOUND, 已落地 / P-C2)",
    "**Soundness**: YES,无 recall loss",
    "**Soundness**: YES,O(1) per cell,metadata-only(内存)",
    "两个独立 source",
):
    if stale in pipeline:
        raise SystemExit(f"pipeline_aligned.md stale soundness marker remains: {stale}")
PY
