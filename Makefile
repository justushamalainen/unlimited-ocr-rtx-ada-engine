# Unlimited-OCR engine — pure C++/CUDA, single host thread, no Python at runtime.
# Pipeline: PDF -> MuPDF render -> SAM ViT-B + CLIP-L + projector (vision_enc.cu)
#           -> DeepseekV2 MoE decoder (engine.cu) -> byte-level BPE -> text.
# Usage:  make && ./ocr_bin [pdf] [npages] [maxtok]   (defaults: bundled paper, 1 page, run to EOS)
# Pluggable KV cache:  make KV=fp8   (default KV=bf16) — fp8 e4m3 halves the long-context attention bandwidth.

MUPDF := $(HOME)/unlimited-ocr/.venv/lib/python3.12/site-packages/pymupdf
NVCC  := nvcc -O3 -arch=sm_89 -std=c++17 --expt-relaxed-constexpr
INC   := -I$(MUPDF)/mupdf-devel/include
KV    ?= bf16
ifeq ($(KV),fp8)
KVFLAG := -DKV_FP8
endif
# rpath so the binary finds libmupdf without LD_LIBRARY_PATH
LIBS  := -Lmupdflib -lmupdf -lcublas -Xlinker -rpath=$(MUPDF)

ocr_bin: vision_enc.o engine.o server.o mupdflib/libmupdf.so
	nvcc -arch=sm_89 vision_enc.o engine.o server.o -o $@ $(LIBS) -lpthread

# vision needs the per-thread default stream so its forward can be CUDA-graph-captured;
# -DOCR_LINK drops its standalone test main() so engine.cu provides the single main().
vision_enc.o: vision_enc.cu st_loader.h
	$(NVCC) -default-stream per-thread -DOCR_LINK -c vision_enc.cu -o $@ $(INC)

engine.o: engine.cu st_loader.h server.h
	$(NVCC) --expt-extended-lambda $(KVFLAG) -c engine.cu -o $@

# HTTP front-end + job queue (plain C++, zero CUDA/MuPDF — connection threads never touch the GPU)
server.o: server.cpp server.h
	$(NVCC) -Xcompiler -pthread -c server.cpp -o $@

mupdflib/libmupdf.so:
	mkdir -p mupdflib && ln -sf $(MUPDF)/libmupdf.so.27.2 mupdflib/libmupdf.so

clean:
	rm -f *.o ocr_bin

# --- analysis / correctness ---
PDF ?= $(HOME)/unlimited-ocr/Unlimited-OCR.pdf

check: ocr_bin            # build + vision-fixture verify + determinism gate
	@bash tools/check.sh "$(PDF)"

sanitize: ocr_bin         # compute-sanitizer suite (memcheck/racecheck/initcheck/synccheck); GUNDAM=1 for tiling path
	@bash tools/sanitize.sh

lint:                     # cppcheck + clang-tidy (no build / no GPU)
	@bash tools/lint.sh

servercheck: ocr_bin      # server gate: CLI parity, multi-doc concurrency, gundam interlude, error paths
	@bash tools/server_check.sh "$(PDF)"

.PHONY: clean check sanitize lint servercheck
