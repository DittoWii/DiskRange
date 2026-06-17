#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

run_guard() {
  awk '
    $1 == "mem_budget" && ($2 + 0) < 1.0 {
      printf "[guard] %s:%d: deep1m mem_budget=%s < 1.0 GiB - migration regression\n", FILENAME, FNR, $2 > "/dev/stderr"
      bad = 1
    }
    END { exit (bad ? 1 : 0) }
  ' "$@"
}

bad_values=("0.24  # legacy" "0.6" "0.99")
for i in "${!bad_values[@]}"; do
  path="$tmp_dir/deep1m_bad_${i}.config"
  {
    echo "cluster_num 10000"
    echo "mem_budget   ${bad_values[$i]}"
  } >"$path"
  if run_guard "$path" >/dev/null 2>"$tmp_dir/bad_${i}.err"; then
    echo "expected guard to reject ${bad_values[$i]}" >&2
    exit 1
  fi
done

good_values=("1.0" "1.0001")
for i in "${!good_values[@]}"; do
  path="$tmp_dir/deep1m_good_${i}.config"
  {
    echo "cluster_num 10000"
    echo "mem_budget ${good_values[$i]}"
  } >"$path"
  run_guard "$path"
done
