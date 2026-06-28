#!/usr/bin/env bash
# CUDA runtime analysis: compute-sanitizer suite over the engine.
# Sanitizers are 10-100x slow, so we use a tiny workload (1 page, few tokens).
# Set GUNDAM=1 to sanitize the high-res tiling path instead of Base.
#   memcheck   - out-of-bounds / misaligned / leaked device allocations
#   racecheck  - shared-memory data races
#   initcheck  - reads of uninitialized device global memory
#   synccheck  - invalid __syncthreads / barrier usage
set -uo pipefail
cd "$(dirname "$0")/.."
CS="${CS:-/usr/local/cuda/bin/compute-sanitizer}"
BIN=./ocr_bin
PDF="${PDF:-$HOME/unlimited-ocr/Unlimited-OCR.pdf}"
TOKENS="${TOKENS:-40}"
ARGS=("$PDF" 1 "$TOKENS")
[ -x "$BIN" ] || { echo "build first:  make KV=fp8"; exit 1; }
command -v "$CS" >/dev/null || { echo "compute-sanitizer not found at $CS"; exit 1; }

fail=0
for tool in memcheck racecheck initcheck synccheck; do
  echo "==================== compute-sanitizer --tool $tool ${GUNDAM:+(GUNDAM)} ===================="
  log="/tmp/cs_${tool}.log"
  if "$CS" --tool "$tool" --error-exitcode 99 "$BIN" "${ARGS[@]}" >"$log" 2>&1; then
    echo "  PASS ($tool) — no errors"
  else
    echo "  FAIL ($tool) — see $log"
    grep -iE "error|race|leak|uninitial|barrier|====" "$log" | grep -ivE "no error|0 error" | head -10
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "ALL CLEAN" || echo "SANITIZER ERRORS FOUND"
exit $fail
