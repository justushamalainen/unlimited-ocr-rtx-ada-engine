#!/usr/bin/env bash
# Static analysis (no GPU / no build needed).
#   cppcheck   - reliable on .cu treated as C++; the primary linter here
#   clang-tidy - best-effort via clang's CUDA front-end (some CUDA/cuBLAS builtins are unknown to it)
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=(engine.cu vision_enc.cu st_loader.h server.cpp server.h)

echo "==================== cppcheck ===================="
if command -v cppcheck >/dev/null; then
  cppcheck --enable=warning,performance,portability --inline-suppr --language=c++ --std=c++17 \
    --suppressions-list=tools/cppcheck-suppress.txt -DKV_FP8 -DOCR_LINK --quiet \
    "${SRC[@]}" 2>&1 | grep -vE "missingInclude|^$" || echo "  (no findings)"
else
  echo "  cppcheck not installed:  sudo apt-get install -y cppcheck"
fi

echo "==================== clang-tidy (best-effort) ===================="
if command -v clang-tidy >/dev/null; then
  for f in engine.cu vision_enc.cu; do
    echo "--- $f ---"
    MUPDF="$HOME/unlimited-ocr/.venv/lib/python3.12/site-packages/pymupdf"
    clang-tidy "$f" --quiet -- -x cuda --cuda-gpu-arch=sm_89 -std=c++17 -DKV_FP8 -DOCR_LINK \
      --no-cuda-version-check -I/usr/local/cuda/include -I"$MUPDF/mupdf-devel/include" 2>/dev/null \
      | grep -E "warning:|error:" | grep -vE "site-packages|cuda_runtime|cublas|unknown-cuda-version|unknown type name '__" | head -25 || echo "  (no parseable findings)"
  done
  echo "--- server.cpp ---"
  clang-tidy server.cpp --quiet -- -x c++ -std=c++17 2>/dev/null \
    | grep -E "warning:|error:" | head -25 || echo "  (no parseable findings)"
else
  echo "  clang-tidy not installed:  sudo apt-get install -y clang-tidy"
fi
