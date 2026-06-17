#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
from pathlib import Path

text = Path("expr/main.cpp").read_text(encoding="utf-8")
start = text.index("void print_run_result(const RunResult& result)")
end = text.index("}\n}  // namespace", start)
body = text[start:end]

required_calls = [
    'msginfo_s("recall = {}", result.recall);',
    'msginfo_s("triangle_prune_rate = {}", result.triangle_prune_rate);',
    'msginfo_s("triangle_candidate_clusters = {}", result.triangle_candidate_clusters);',
    'msginfo_s("triangle_pruned_clusters = {}", result.triangle_pruned_clusters);',
    'msginfo_s("slab_pruned_clusters = {}", result.slab_pruned_clusters);',
    'msginfo_s("box_residual_pruned_clusters = {}", result.box_residual_pruned_clusters);',
    'msginfo_s("mean_keep_after_filter = {}", result.mean_keep_after_filter);',
    'msginfo_s("build_time = {}", result.build_time);',
    'msginfo_s("search_time = {}", result.search_time);',
    'msginfo_s("dist_comp = {}", result.dist_comp);',
    'msginfo_s("per_query_min = {}", result.per_query_min);',
    'msginfo_s("per_query_median = {}", result.per_query_median);',
    'msginfo_s("per_query_max = {}", result.per_query_max);',
    'msginfo_s("per_query_zero = {}", result.per_query_zero);',
]

missing = [call for call in required_calls if call not in body]
if missing:
    raise SystemExit(f"missing msginfo_s calls: {missing}")

if "std::cout <<" in body:
    raise SystemExit("print_run_result still contains std::cout output")

if "msginfo(" in body:
    raise SystemExit("print_run_result should not use msginfo() for variable output")
PY
