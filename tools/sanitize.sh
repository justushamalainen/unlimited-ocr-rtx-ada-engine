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

# HARD LIMIT (learned 2026-07-03 the OOM way): racecheck's host shadow memory explodes on gundam-sized
# prefills (>900-tok refs) — 126GB RSS, kernel OOM-killed the whole box (twice: 07-02 + 07-03). The ulimit
# makes it abort with bad_alloc instead. racecheck under GUNDAM=1 is expected to die at the limit (0 hazards
# up to that point); gundam/mixed coverage = memcheck+initcheck+synccheck (these fit) + base racecheck.
# Base racecheck itself peaks 60-100GB host — 100GB default fits it while still protecting a 125GB box.
ULIM="${ULIM:-104857600}"  # kB of virtual memory (100GB), env-overridable
fail=0
for tool in memcheck racecheck initcheck synccheck; do
  echo "==================== compute-sanitizer --tool $tool ${GUNDAM:+(GUNDAM)} ===================="
  log="/tmp/cs_${tool}.log"
  if ( ulimit -v "$ULIM"; exec "$CS" --tool "$tool" --error-exitcode 99 "$BIN" "${ARGS[@]}" ) >"$log" 2>&1; then
    echo "  PASS ($tool) — no errors"
  else
    echo "  FAIL ($tool) — see $log"
    grep -iE "error|race|leak|uninitial|barrier|====" "$log" | grep -ivE "no error|0 error" | head -10
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "ALL CLEAN" || echo "SANITIZER ERRORS FOUND"
exit $fail
