#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/experiments/PCA_enhancement/PG_rabitq_pipeline/run_pg4_anisotropic_sweep.sh"

require_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    printf 'expected output to contain:\n%s\n\nactual output:\n%s\n' "$needle" "$haystack" >&2
    exit 1
  fi
}

require_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" == *"$needle"* ]]; then
    printf 'expected output not to contain:\n%s\n\nactual output:\n%s\n' "$needle" "$haystack" >&2
    exit 1
  fi
}

output="$("$script" \
  --dry-run \
  --datasets deep1m,deep10m \
  --skip-binary-build \
  --out-dir /tmp/pg4_aniso_runner_test \
  --threads 7)"

require_contains "$output" "configs/deep1m_disk.config"
require_contains "$output" "experiments/range_gt/output/deep1m_0.2632_gt.json"
require_contains "$output" "configs/deep10m_disk.config"
require_contains "$output" "experiments/range_gt/output/deep10m_0.2002_gt.json"
require_contains "$output" "--partition-objective standard --eta 1.0 --unit-normalize"
require_contains "$output" "--partition-objective anisotropic --eta 4.125 --unit-normalize"
require_contains "$output" "--dataset deep1m"
require_contains "$output" "--dataset deep10m"
require_contains "$output" "--partition-objective-list standard,anisotropic"
require_contains "$output" "--eta-list 2.0,4.125,8.0"
require_contains "$output" "--unit-normalize-list false,true"
require_contains "$output" "--threads 7"
require_contains "$output" "pg4_anisotropic_kmeans_verdict.json"
require_not_contains "$output" " --no-unit-normalize"

anchor_output="$("$script" \
  --dry-run \
  --anchor-check-only \
  --datasets deep1m \
  --out-dir /tmp/pg4_aniso_runner_test)"

require_contains "$anchor_output" "anchor metrics compare"
require_contains "$anchor_output" "results/closure_coverage/deep1m_coverage.csv"
require_contains "$anchor_output" "/tmp/pg4_aniso_runner_test/deep1m_coverage.csv"

printf 'pg4 full sweep runner smoke test passed\n'
