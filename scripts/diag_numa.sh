#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# NUMA / bandwidth-wall diagnostic for the non-empty (300-query) workload.
# Splits the 25% parallel-efficiency "bandwidth contention" (README §10) into
# memory-bandwidth wall vs disk-bandwidth wall by varying socket placement:
#   A 24-core single socket (baseline): node0 memory channels only
#   B 48-core dual socket: 2x cores + 2x memory channels (disk still shared)
#   C 24-core dual socket: same cores, 2x memory channels (isolates memory)
# memory wall  -> B >> A and C > A (extra channels help)
# disk wall    -> B ~ A or worse (disk shared; more threads just contend)

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_PROC_BIND=close OMP_PLACES=cores DJ_NONEMPTY_ONLY=1
TMP=$(mktemp /tmp/numa_XXXX.config); trap 'rm -f "$TMP"' EXIT
sed 's/^approx_alpha .*/approx_alpha 1.0/' "$CONFIG" > "$TMP"
median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

run_cfg() {
  local name="$1" threads="$2"; shift 2
  export OMP_NUM_THREADS="$threads"
  local qs=() sts=()
  for r in $(seq 1 5); do
    out=$(numactl "$@" ./build/search --config "$TMP" --gt "$GT" 2>/dev/null)
    qs+=("$(echo "$out" | sed -n 's/^qps = \(.*\) query\/sec/\1/p')")
    sts+=("$(echo "$out" | sed -n 's/^search_time = //p')")
  done
  printf '%-14s %-11s %-19s | %s\n' "$name" "$(median "${qs[@]}")" "$(median "${sts[@]}")" "${qs[*]}"
}

echo "config         qps_median  search_time_median  | all_qps"
run_cfg "A 24c-1sock"  24  --physcpubind=0-23 --membind=0
run_cfg "B 48c-2sock"  48  --physcpubind=0-47 --interleave=0,1
run_cfg "C 24c-2sock"  24  --physcpubind=0-11,24-35 --interleave=0,1
