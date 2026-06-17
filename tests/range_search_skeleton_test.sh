#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_gt_json="experiments/range_gt/output/deep1m_0.2632_gt.json"
config="configs/deep1m_disk.config"
run_dir="$(mktemp -d /tmp/range_search_skeleton.XXXXXX)"
index_prefix="$run_dir/deep1m"
index_dir="${index_prefix}_diskrange"
gt_json="$run_dir/deep1m_0.2632_gt.json"

cleanup() {
  rm -rf "$run_dir" "$index_dir"
}
trap cleanup EXIT

test -f "$source_gt_json"
test -f "$config"
grep -Fq '#include <nlohmann/json.hpp>' lib/ConfigLoader.h
if grep -Fq 'boost/property_tree' lib/ConfigLoader.h; then
  echo "ConfigLoader must use nlohmann/json instead of Boost.PropertyTree" >&2
  exit 1
fi

python3 - "$source_gt_json" "$gt_json" "$index_prefix" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
data["index"]["prefix"] = sys.argv[3]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/range_search_cmake.log
cmake --build build >/tmp/range_search_build.log

# Guard: keep deep1m search configs above the fixed-resident floor introduced
# by search-time mem_budget enforcement. Use numeric awk parsing to match
# ConfigReader's stof() behavior instead of a brittle grep.
if awk '
  $1 == "mem_budget" && ($2 + 0) < 1.0 {
    printf "[guard] %s:%d: deep1m mem_budget=%s < 1.0 GiB - migration regression\n", FILENAME, FNR, $2 > "/dev/stderr"
    bad = 1
  }
  END { exit (bad ? 1 : 0) }
' configs/deep1m_*.config; then
  :
else
  echo "[guard] deep1m migration regression detected; abort" >&2
  exit 1
fi

rm -f "$index_dir/_cluster_file.bin" \
      "$index_dir/_metadata_file.bin" \
      "$index_dir/_hnsw_file.bin"

stdout1="$(mktemp)"
stderr1="$(mktemp)"
./build/main --config "$config" --gt "$gt_json" >"$stdout1" 2>"$stderr1"
results1="$(mktemp)"
python3 tests/extract_run_result.py "$stdout1" "$stderr1" >"$results1"

test -s "$index_dir/_cluster_file.bin"
test -s "$index_dir/_metadata_file.bin"
test -s "$index_dir/_hnsw_file.bin"
grep -F '[STUB] DiskRange::search() not yet implemented (range-search-skeleton phase). Reporting recall=0.0.' "$stderr1"
grep -Fx 'recall = 0.000000' "$results1"
grep -Ex 'build_time = [0-9]+(\.[0-9]+)?' "$results1"
grep -Fx 'search_time = 0.000000' "$results1"
grep -Fx 'dist_comp = 0' "$results1"
grep -Fx 'per_query_min = 0' "$results1"
grep -Fx 'per_query_median = 0.000000' "$results1"
grep -Fx 'per_query_max = 0' "$results1"
grep -Fx 'per_query_zero = 0' "$results1"
test "$(wc -l < "$results1")" -ge 8

stdout2="$(mktemp)"
stderr2="$(mktemp)"
./build/main --gt "$gt_json" --config "$config" >"$stdout2" 2>"$stderr2"
grep -F '[STUB] DiskRange::search()' "$stderr2"

if ./build/main "$config" >/tmp/range_positional.out 2>/tmp/range_positional.err; then
  echo "positional invocation unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'Usage:' /tmp/range_positional.err

if ./build/main --config "$config" >/tmp/range_missing.out 2>/tmp/range_missing.err; then
  echo "missing --gt unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'Usage:' /tmp/range_missing.err

if ./build/main --config "$config" --gt "$gt_json" --unknown-flag x >/tmp/range_unknown_flag.out 2>/tmp/range_unknown_flag.err; then
  echo "unknown flag unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'Usage:' /tmp/range_unknown_flag.err

if ./build/main --config "$config" --gt /nonexistent.json >/tmp/range_bad_gt.out 2>/tmp/range_bad_gt.err; then
  echo "nonexistent gt unexpectedly succeeded" >&2
  exit 1
fi
grep -F '/nonexistent.json' /tmp/range_bad_gt.err

tmp_config="$(mktemp)"
cat >"$tmp_config" <<'CFG'
cluster_num 10000
K 245
mem_budget 0.24
mode memory
gorder_window 0
CFG
if ./build/main --config "$tmp_config" --gt "$gt_json" >/tmp/range_memory.out 2>/tmp/range_memory.err; then
  echo "memory mode unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'memory mode not supported' /tmp/range_memory.err

cat >"$tmp_config" <<'CFG'
cluster_num 10000
K 245
mem_budget 0.24
mode disk
gorder_window 0
radius 0.05
CFG
if ./build/main --config "$tmp_config" --gt "$gt_json" >/tmp/range_unknown.out 2>/tmp/range_unknown.err; then
  echo "unknown key unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'radius' /tmp/range_unknown.err

cat >"$tmp_config" <<'CFG'
cluster_num 10000
K 245
mem_budget 0.24
mode disk
gorder_window 0
error_bound 0.1
CFG
if ./build/main --config "$tmp_config" --gt "$gt_json" >/tmp/range_error_bound.out 2>/tmp/range_error_bound.err; then
  echo "legacy error_bound unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'error_bound' /tmp/range_error_bound.err

bad_sum_json="$(mktemp --suffix=.json)"
python3 - "$gt_json" "$bad_sum_json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
data["total_in_range"] += 1
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY
if ./build/main --config "$config" --gt "$bad_sum_json" >/tmp/range_bad_sum.out 2>/tmp/range_bad_sum.err; then
  echo "bad per_query total unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'per_query_counts' /tmp/range_bad_sum.err
grep -F 'total_in_range' /tmp/range_bad_sum.err

bad_query_json="$(mktemp --suffix=.json)"
python3 - "$gt_json" "$bad_query_json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
data["n_queries"] += 1
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data, f)
PY
if ./build/main --config "$config" --gt "$bad_query_json" >/tmp/range_bad_query.out 2>/tmp/range_bad_query.err; then
  echo "bad query header unexpectedly succeeded" >&2
  exit 1
fi
grep -F 'deep1m_query.fbin' /tmp/range_bad_query.err
grep -F 'expected' /tmp/range_bad_query.err
grep -F 'actual' /tmp/range_bad_query.err

test -x build/main
test ! -e build/compute_gt
