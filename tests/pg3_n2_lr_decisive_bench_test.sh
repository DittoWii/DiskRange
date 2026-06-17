#!/usr/bin/env bash
set -euo pipefail
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY
unset OMP_PROC_BIND
export OMP_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="$ROOT/experiments/PCA_enhancement/PG_rabitq_pipeline"
CONFIG="$ROOT/experiments/range_gt/output/deep1m_0.2632_gt.json"
CPP_BIN="$PIPELINE/closure_coverage_bench"

if [[ ! -f "$CONFIG" ]]; then
  echo "missing required deep1m config: $CONFIG" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

per_query_csv="$tmp_dir/deep1m_per_query_features.csv"
aggregate_csv="$tmp_dir/deep1m_coverage.csv"
out_json="$tmp_dir/n2_lr_decisive.json"
out_md="$tmp_dir/n2_lr_decisive.md"

make -C "$PIPELINE" closure_coverage_bench >/dev/null

dry_run="$(
  python3 "$PIPELINE/closure_coverage_bench.py" \
    --dataset deep1m \
    --config-json "$CONFIG" \
    --epsilon-list 0 \
    --max-replications-list 0 \
    --angle-prune-deg-list 0 \
    --query-radius-multiplier-list 0 \
    --rerank-by-list none \
    --adaptive-mode-list off \
    --adaptive-config-list off_baseline \
    --adaptive-ratio-ref-list r_top2 \
    --r6-stop-lb-list off \
    --partition-objective-list standard \
    --eta-list 1 \
    --unit-normalize-list false \
    --n-queries 16 \
    --threads 1 \
    --cpp-bin "$CPP_BIN" \
    --dump-per-query-features "$per_query_csv" \
    --dry-run
)"
grep -F -- "--dump-per-query-features" <<<"$dry_run" >/dev/null
grep -F -- "$per_query_csv" <<<"$dry_run" >/dev/null

"$CPP_BIN" --toy-self-check --epsilon 0 --dump-per-query-features "$per_query_csv" >"$aggregate_csv"

python3 - "$per_query_csv" "$aggregate_csv" <<'PY'
import csv
import sys

per_query_csv, aggregate_csv = sys.argv[1:3]
with open(per_query_csv, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with open(aggregate_csv, newline="", encoding="utf-8") as f:
    agg_rows = list(csv.DictReader(f))

required = [
    "qid",
    "k_1nn",
    "scan_budget_for_1nn",
    "miss_at_4",
    "scan_vecs_at_4",
    "scan_vecs_at_8",
    "scan_vecs_at_16",
    "scan_vecs_at_32",
]
required += [f"d_sqg_rank_{i}" for i in range(1, 33)]
missing = [column for column in required if column not in rows[0]]
assert not missing, missing
assert len(rows) >= 4
assert len(agg_rows) == 1
agg = agg_rows[0]
assert agg["per_query_features_path"] == per_query_csv
assert all(float(row[f"d_sqg_rank_{i}"]) > 0 for row in rows for i in range(1, 33))
assert any(
    float(row["d_sqg_rank_1"]) > float(row["d_sqg_rank_2"])
    for row in rows
), "d_sqg_rank_* must preserve SQG rank order, not sorted order"
assert all(int(row["qid"]) == i for i, row in enumerate(rows))
assert all(int(row["miss_at_4"]) in (0, 1) for row in rows)
assert all((int(row["miss_at_4"]) == 1) == (int(row["k_1nn"]) < 0 or int(row["k_1nn"]) >= 4) for row in rows)
assert all(int(row["scan_vecs_at_4"]) <= int(row["scan_vecs_at_8"]) <= int(row["scan_vecs_at_16"]) <= int(row["scan_vecs_at_32"]) for row in rows)

def mean(values):
    values = list(values)
    return sum(values) / len(values)

for n in (4, 8, 16, 32):
    scan_mean = mean(float(r[f"scan_vecs_at_{n}"]) for r in rows)
    cov_mean = mean(1.0 if 0 <= int(r["k_1nn"]) < n else 0.0 for r in rows)
    assert abs(scan_mean - float(agg[f"avg_scan_vecs_at_{n}"])) <= 1e-3
    assert abs(cov_mean - float(agg[f"sqg_coverage_at_{n}"])) <= 1e-3
PY

python3 "$PIPELINE/analyze_n2_lr_decisive.py" \
  --per-query-csv "$per_query_csv" \
  --aggregate-csv "$aggregate_csv" \
  --out-json "$out_json" \
  --out-md "$out_md" \
  --cv-folds 2 >/dev/null

python3 - "$out_json" "$out_md" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = Path(sys.argv[2]).read_text(encoding="utf-8")
assert payload["meta"]["per_query_rows"] >= 4
assert payload["verdict"]["label"].startswith("H")
assert "do_not_close_q11_q12" in payload["verdict"]
assert "rank_view" in payload["l1_lr"]
assert "sorted_view" in payload["l1_lr"]
assert "binary" in payload["scalar_ablation"]
assert "triclass" in payload["scalar_ablation"]
assert "N2 LR decisive verdict" in report
PY
