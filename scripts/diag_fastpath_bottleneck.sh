#!/usr/bin/env bash
# Locate the fast-path (fast_path_nprobe_N) bottleneck on SSNPP-1M.
# Uses DJ_BATCHLOG to split oracle(segA) time vs slow-path batch time,
# and DJ_PHASE3_INMEM to isolate Phase-3 8-bit disk pread.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="configs/ssnpp_uint8_sample.config"
GT="experiments/range_gt/output/ssnpp1m_310.2212_gt.json"
export OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores

run() {
  local label="$1"; local nprobe="$2"; shift 2
  local TMP; TMP=$(mktemp /tmp/diag_fp_XXXX.config)
  if grep -q '^fast_path_nprobe_N ' "$CONFIG"; then
    sed "s/^fast_path_nprobe_N .*/fast_path_nprobe_N ${nprobe}/" "$CONFIG" > "$TMP"
  else
    cp "$CONFIG" "$TMP"; printf 'fast_path_nprobe_N %s\n' "$nprobe" >> "$TMP"
  fi
  echo "######## ${label} (nprobe=${nprobe}) extra_env: $* ########"
  env "$@" DJ_BATCHLOG=1 numactl --physcpubind=0-23 --membind=0 \
      ./build/search --config "$TMP" --gt "$GT" 2>diag_stderr.tmp 1>diag_stdout.tmp || true
  echo "--- result ---"
  grep -E '^recall = |^qps = |false_empty|fast_path_skip_rate|dist_comp' diag_stdout.tmp || true
  echo "--- time split (DJ_BATCHLOG) ---"
  grep -E '\[batchlog\] (n_batches|oracle_summed)' diag_stderr.tmp || true
  grep -E '\[fuse-busy\]|\[makespan\]' diag_stderr.tmp | head -3 || true
  rm -f "$TMP"
  echo
}

run "baseline"            128
run "phase3_inmem"        128 DJ_PHASE3_INMEM=1
run "nprobe384"           384
run "nprobe384_inmem"     384 DJ_PHASE3_INMEM=1
echo "ALL DONE"
