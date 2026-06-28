#!/usr/bin/env bash
# Correctness harness: build + vision-fixture verification + determinism gate.
# Use before committing a change to confirm it preserves behavior.
set -uo pipefail
cd "$(dirname "$0")/.."
PDF="${1:-$HOME/unlimited-ocr/Unlimited-OCR.pdf}"
rc=0

echo "== build (make KV=fp8) =="
if make KV=fp8 >/tmp/check_build.log 2>&1; then echo "  OK"; else echo "  BUILD FAIL"; tail -15 /tmp/check_build.log; exit 1; fi

echo "== vision fixture (GUNDAM_VFIX) =="
if [ -f gundam/ref_assembled.bin ]; then
  ma=$(GUNDAM_VFIX=1 ./ocr_bin 2>&1 | grep -oE "mean_abs=[0-9.]+" | head -1 | cut -d= -f2)
  if [ -n "$ma" ] && awk "BEGIN{exit !($ma < 0.01)}"; then echo "  OK (mean_abs=$ma)"; else echo "  FAIL (mean_abs=$ma, want <0.01)"; rc=1; fi
else
  echo "  SKIP (gundam/ref_* fixtures not present — regenerate with gundam/gen_fixtures.py)"
fi

echo "== determinism (Base, 2 runs identical) =="
./ocr_bin "$PDF" 1 60 2>/dev/null | sed -n '/===== OCR/,$p' > /tmp/chk_a.txt
./ocr_bin "$PDF" 1 60 2>/dev/null | sed -n '/===== OCR/,$p' > /tmp/chk_b.txt
if [ -s /tmp/chk_a.txt ] && diff -q /tmp/chk_a.txt /tmp/chk_b.txt >/dev/null; then echo "  OK"; else echo "  NON-DETERMINISTIC / empty output"; rc=1; fi

[ $rc -eq 0 ] && echo "CHECK PASSED" || echo "CHECK FAILED"
exit $rc
