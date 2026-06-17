#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Thread-count sweep tracking io_wait (NOT qps -- that is diag_thread_scaling.sh)
# to settle the open question in README §17.4: is io_wait's ~0.25s "fixed
# component" CONCURRENCY CONTENTION (scales with thread count -> kernel IO
# stack / NVMe device queueing) or an INTRINSIC per-query IO cost (flat across
# thread count -> read amplification)?
#
# t_io_wait = SUM over 300 queries of (drain_wall - l2). Each query reads the
# SAME cells regardless of thread count, so the only variable is how many
# threads hammer the same NVMe concurrently. Single-thread is the irreducible
# floor (no cross-thread contention); the 24/1 ratio IS the contention premium.
# Caches disabled so every query truly hits disk (fair 1-vs-24 comparison, no
# cache-hit skew). Cores pinned 0..N-1 on node0 (single socket) -> no NUMA.

CONFIG="configs/deep10m_disk_fast_path.config"
GT="experiments/range_gt/output/deep10m_0.2002_gt.json"
export OMP_PROC_BIND=close OMP_PLACES=cores
export DJ_CLUSTER_CACHE_DISABLE=1 DJ_CELL_CACHE_DISABLE=1 DJ_NONEMPTY_ONLY=1

TMP=$(mktemp /tmp/thriow_XXXX.config); trap 'rm -f "$TMP"' EXIT
sed 's/^approx_alpha .*/approx_alpha 1.0/' "$CONFIG" > "$TMP"

median() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }

printf '%-7s %-11s %-17s %-15s %-12s %s\n' \
  threads io_wait_s per_query_io_us mean_us/q cells_read "io_wait/round"
for n in 1 2 4 8 12 16 24; do
  export OMP_NUM_THREADS="$n"
  iws=(); musq=(); crs=()
  for r in 1 2 3; do
    out=$(numactl --physcpubind=0-$((n-1)) --membind=0 \
      ./build/search_stats --config "$TMP" --gt "$GT" 2>/dev/null)
    iws+=("$(printf '%s' "$out" | sed -n 's/^nonzero_result_t_io_wait = //p')")
    musq+=("$(printf '%s' "$out" | sed -n 's/^nonzero_result_mean_us_per_query = //p')")
    crs+=("$(printf '%s' "$out" | sed -n 's/^nonzero_result_cells_read = //p')")
  done
  iw=$(median "${iws[@]}"); mq=$(median "${musq[@]}"); cr=$(median "${crs[@]}")
  pq=$(awk -v x="$iw" 'BEGIN{printf "%.1f", x/300*1e6}')
  printf '%-7s %-11s %-17s %-15s %-12s %s\n' "$n" "$iw" "$pq" "$mq" "$cr" "${iws[*]}"
done
