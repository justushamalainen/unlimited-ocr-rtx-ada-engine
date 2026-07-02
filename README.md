# Unlimited-OCR — RTX 6000 Ada inference engine

A from-scratch, pure **C++/CUDA** inference engine for [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
(DeepseekV2 MoE decoder + SAM ViT-B / CLIP-L vision encoder), optimized for the
**RTX 6000 Ada (sm_89)**. Zero Python at runtime.

## Pipeline

```
PDF → MuPDF render → SAM ViT-B + CLIP-L + projector (vision_enc.cu)
    → DeepseekV2 MoE decoder (engine.cu) → byte-level BPE → text
```

## Build & run

```sh
make                      # default KV cache = bf16
make KV=fp8               # fp8 e4m3 KV cache (halves long-context attention bandwidth)
./ocr_bin [pdf] [npages] [maxtok]
./ocr_bin serve [port]    # HTTP server (default port 8000), multiple documents concurrently
```

## Server

`ocr_bin serve` turns the engine into a persistent multi-document server: weights load once,
then documents from concurrent clients feed the same page-parallel decode window (pages of
different documents co-batch; per-job round-robin admission keeps small documents fast).

```sh
curl --data-binary @doc.pdf 'http://host:8000/ocr'                 # whole document -> text/plain
curl --data-binary @doc.pdf 'http://host:8000/ocr?pages=3'         # first 3 pages
curl --data-binary @doc.pdf 'http://host:8000/ocr?gundam=1'        # high-res tiling (exclusive run)
curl 'http://host:8000/healthz'                                    # liveness + X-Queue/X-Done headers
```

Responses carry `X-Pages`/`X-Tokens`/`X-Truncated-Pages`/`X-Millis`. Errors: 400 bad request,
411 missing Content-Length, 413 too large / gundam page cap, 422 unreadable PDF, 501 chunked
body, 503 queue full (16 jobs) or connection cap (64). One request per connection; send
`Content-Length` (no chunked bodies). A document OCR'd on an idle server is byte-identical to
the CLI; under concurrent load, co-batching may flip rare argmax near-ties (see DESIGN.md).
`GUNDAM_PAGE_CAP` (default 64) bounds gundam requests; `WINDOW` (default 128) bounds resident
page-streams/VRAM. Gate: `make servercheck`.

Requires CUDA (nvcc, sm_89), cuBLAS, and a local `pymupdf` install (the Makefile links
`libmupdf` from `.venv/.../pymupdf`). Model weights are loaded at runtime from the
HF model directory; the int4/fp8 expert quantizations and the small constant fixtures
(`vocab.bin`, `gundam/*pos*.bin`, `gundam/relpos_*.bin`) are produced by the `prep_*`/
`gen_fixtures.py` scripts.

## Modes

- **Base** — 1024px global view, 273 visual tokens/page. Fast; best for clean/large text.
- **Gundam** (`GUNDAM=1`) — high-res dynamic tiling (global view + local 640px tiles) for
  small/dense text.

## Decode architecture (key ideas)

- **Page-parallel decode** — N pages decode as **independent batched streams**, each with its
  own per-page reference `[bos + visual + prompt]`; no cross-page attention. Documents are just
  a grouping of pages, so multi-document batching is the same path (bucket Gundam pages by token
  count). Batch-shrink + per-active-count CUDA graph keep utilization high.
- **R-SWA** (Reference Sliding-Window Attention) — decode attends to the full reference + a
  bounded 128-token window, so per-step cost stays flat with context length.
- **Mixed precision** — int4 MoE experts + int4 lm_head (small-batch tail), bf16 prefill and
  bulk lm_head (accuracy-protected), fp8/cuBLAS projection crossovers tuned by batch size.

## Documentation

- **DESIGN.md** — full architecture and the optimization history.
- **EXPERIMENTS.md** — approaches that were tried and rejected (and why).
