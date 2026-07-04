# Page-parallel decode — experiments that didn't pan out

Two ideas explored on top of the working **page-parallel** path (`PAGEPAR=1`), both aimed at
replacing the per-page approach with a single shared context. Both were implemented and measured;
both are recorded here because the *negative* results are the useful part — they explain why the
current per-page design is the right one.

Context: the shipped `PAGEPAR` path decodes N pages as **independent streams**, each with its own
small reference (`[bos + 273 visual + 4 prompt]` = 278 tokens) and ring. It hits ~2,900 tok/s
(14-page) / ~3,100 tok/s (30-page) via batch-shrink + per-active-count CUDA graph, output 95–96%
identical to the sequential one-shot. The two experiments below tried to do better and didn't.

---

## Experiment 1 — Shared single context + per-stream offset ("anchor each stream to its page")

**Idea.** Instead of N separate per-page references, do **one** prefill over the whole document
(shared reference, full cross-page fidelity, cheap) and launch N decode streams that all attend to
that one reference but each **start at a different offset** so it lands on a different page. The
appeal: one prefill instead of N, full context, and it unifies with the one-shot (which is just
"1 stream at offset 0").

**The crux: anchoring.** With the full reference visible, what makes stream *p* decode page *p*
instead of page 0? The decode state in R-SWA is exactly `(shared reference, last-128 window,
RoPE position, current token)` — there's no stored "page pointer". So to start at page *p* a stream
must reproduce the true state at page *p*'s boundary.

**Tests** (env-gated `JUMPSEED` path in `generate()`; decoupled RoPE position from the ring counter
via `g_ropebase` so the window stays local while the position jumps):

| What we fed | RoPE position | Result |
|---|---|---|
| real page-5 tail + `<PAGE>` seed | decode start (3827) | **page 0** (ignored the seed) |
| real page-5 tail + `<PAGE>` seed | page-6 offset (7528) | **page 6, verbatim** ✅ |
| **generic** `<PAGE>` only | page-6 offset (7528) | **hallucination** (fake titles + page-1 text) |
| generic `<PAGE>` | sweep 6828/7528/8637 | position barely moved it; ~same wrong output |

**Findings.**
- The anchor is a **consistent `(real-prior-tail, position)` pair**. Break either and it fails.
- The **window content is essential**: prime the model with a fake/generic window and it confidently
  generates *fake* content — the visual tokens do **not** override a wrong autoregressive prior.
- R-SWA self-heals a wrong window only by emitting 128 *correct* tokens — but a wrong window emits
  *wrong* tokens, so it never converges. Chicken-and-egg.

**Conclusion.** To start stream *p* in a shared context you need page (p−1)'s **actual tail**, which
only exists *after* decoding page p−1 → a **sequential dependency** that can't be faked. So the clean
single-context parallel isn't buildable as hoped. This is exactly why the **per-page reference**
design works: giving a stream *only* its page's visual tokens removes the "which page" ambiguity at
the source — no prior tail needed. (`<PAGE>` is also not a special token — it's 3 BPE ids
`[100855,16412,32]`; there is no `</PAGE>` end marker.)

---

## Experiment 2 — Batched block-diagonal prefill (one pass instead of N)

**Idea.** Keep the per-page references but build them in **one** forward pass: lay out all N pages'
`[bos+visual+prompt]` (278 each) and run the prefill with **block-diagonal attention** (each page
attends only within itself), writing each page's clean KV into its stream's reference region. One
pass should beat N separate prefills on launch overhead + GPU utilization.

**Implemented** (`BDPREFILL=1`): kernels `k_pageembeds_all`, `k_rope_bd` (per-sequence RoPE),
`k_attn_prefill_bd` (causal within each sequence), `k_seedkv_bd` (strided write to per-stream region),
`k_gather_last`; function `prefill_bd`. Buffers grow to `N*278` rows for the pass.

**Results.**

| | N-seq (default) | block-diag |
|---|---|---|
| 14-page prefill | 817 ms | 825 ms |
| 30-page prefill | 1742 ms | 1783 ms |
| output vs N-seq | — | 99.6% identical (correct) |

**Conclusion.** Correct, but **no speedup at any scale** (marginally slower). The prefill is
**compute-bound on the MoE + projections** — one pass does the same total work as N passes, and the
per-call launch overhead these kernels would save was never the bottleneck. It also needs `N×278`-row
buffers (OOM risk on 100+ page docs). So the default stays **N-sequential**; block-diagonal was left
behind `BDPREFILL=1` until 2026-07-02, when windowed admission removed its upfront-embeds input and the
code was deleted (single-engine policy).

---

## Takeaways
1. **The decode, not the prefill, is the bottleneck** — and it's already optimized (batch-shrink +
   graph). Prefill is ~6% of e2e; batching it doesn't move the needle.
2. **Page independence is a feature, not a limitation.** Every shared-context variant either
   hallucinated (no real prior window) or reduced to sequential. The per-page reference sidesteps the
   anchoring problem entirely, which is why it's the shipped design.
3. For more end-to-end speed, look at the **decode** (e.g. cublasLt fp8 tensor-core GEMMs) or the
   **vision encoder**, not the prefill or the context structure.

Scaffolding: Exp-1's `JUMPSEED`/`g_ropebase` was removed when the one-shot `generate` path was
deleted (page-parallel is now the only decode path). Exp-2's `BDPREFILL` + `*_bd` kernels were removed
2026-07-02 (windowed admission made the upfront-embeds path structurally dead).

## Exp-8: fp8-mma decode projections at batch (2026-07-02) — REJECTED on perf, code removed
Custom m16n8k16 kernels (fp8-e4m3 weights x bf16 acts, fp32 accum) for qkv/o/dense/shared at NA>3,
replacing cuBLAS bf16. LUT dequant version -22%; sm_89 hw-cvt f16 version -4% (14pg) / -9% (Gundam-50).
cuBLAS at tiny-batch GEMV-ish sizes is near the DRAM floor already (96MB L2); halving weight bytes needs
a cutlass-grade cp.async pipeline to show up. Kernels were numerically correct (sub-ULP vs per-token fp8
ref), but output drift vs bf16 was 99.85% on the paper PDF and only 92.4% on the small-text brochure
(850 vs 868 det els) — consistent with fp8 lm_head/vision degrading small text. Rejected on perf; would
also have struggled on accuracy. Code deleted in the single-engine consolidation (never committed);
DESIGN.md "FP8MMA" preserves the full design for a rebuild if NA regimes change.

## Exp-9: k_attn_split_b multi-warp blocks (2026-07-04) — REVERTED (neutral on gundam, -3.6% on base)
Hypothesis: 1-warp blocks cap Ada at 24 blocks/SM = 50% theoretical occupancy (ncu: 26% achieved at
NA=6, DRAM 72%), so packing 4 splits per 128-thread block (per-warp FP sequence unchanged, padding
warps early-exit, sf shrunk 512->256 floats/warp) should lift the small-NA tail toward the roofline.
Measured: gundam-50 decode UNCHANGED (13.29 vs 13.26s — at NA=50 the kernel is already 83-90% DRAM,
extra occupancy adds no bandwidth) and base 14pg decode REGRESSED 4135 -> 3985 tok/s (-3.6%): at
ns=8/NA=14 the 4-warp granularity coarsens scheduling (280 fat blocks vs 1120 slim ones) and doubles
per-block shared. All gates stayed bit-exact — the revert is purely perf. Lesson: the split kernel's
occupancy cap only matters where DRAM% is far from roofline AND the grid is small; that regime (NA<=8
gundam tail) is too small a slice of any real run to pay for hurting the base fast path.

## Exp-10: int4 KV cache, group-of-32 scales (2026-07-04) — REJECTED on accuracy, code removed
Design (recoverable from this session's git diff if regimes change): per-(token,head) row = 80B
(64 packed two's-complement nibbles + 4 inline fp32 group scales at +64), KVTOKE=NH*80 bytes/token,
same single-array alloc/memcpy plumbing; row-wise k_seedkv (32 thr, group absmax butterflies), 32-thread
k_rope_store_b (thread t owns rope pairs (2t,2t+64),(2t+1,2t+65) -> bytes t,t+32, no nibble RMW), and an
int4-specific attention lane mapping (lane owns 4 CONTIGUOUS elems = one u16 + one group scale -> 1 load
+ 4 FMA + 1 mul per key; the first cut with the fp8 lane mapping was SLOWER than fp8 — 4 byte loads + 8
muls turned the DRAM-bound kernel issue-bound, decode 13.2->15.9s).
Perf (v2 mapping): decode 13.2 -> 11.8s (4853 tok/s), gundam-50 25.2s (-1.6s), kernel 479 -> ~415us at
NA=50, ncu DRAM 70% / SM 67% (occupancy-capped, more headroom existed). Base 14pg unchanged.
Accuracy (controlled GSLOTS=8 idle-server, per-page feats vs fp8 baseline): mean p10 0.716 -> 0.689,
mean conf 0.932 -> 0.927, escalate-grade pages 45 -> 48, TOTAL WORDS -4.3% — systematic content loss
including CLEAN pages (p10-0.94 page 32: 525->421 words; page 33: 761->504). Same failure signature as
fp8 lm_head/vision: 4-bit activation precision crushes exactly the dense small text gundam exists for.
REJECTED per the accuracy-first policy; deleted per the single-engine policy. KV stays fp8-e4m3.
Lesson: for KV-bound decode the traffic win is real and the kernel-side cost model works (contiguous
lane mapping matters more than raw bit-width), but K/V activation quantization below 8 bits is not
quality-neutral for this model on its own target workload.
