# unlimited-ocr — focused CUDA decode engine (RTX 6000 Ada / sm_89)

Goal: a bespoke C++/CUDA inference engine for the Unlimited-OCR **language decoder**
(the diagnosed bottleneck), hand-optimized for Ada. Beat SGLang single-stream by
attacking the real limiter (HBM weight bandwidth) with fp8 experts + tight fusion +
full-step CUDA graph and zero per-step CPU work.

## Scope
- **In:** decoder forward — embed → 12× (RMSNorm, MHA+RoPE+R-SWA, RMSNorm, MoE/dense) →
  final norm → lm_head → greedy sample + no-repeat-ngram. Prefill + autoregressive decode.
- **Boundary:** vision encoder (SAM+CLIP) and input-embedding assembly are run ONCE via
  the reference model (.venv) and dumped as an input-embedding sequence. The engine
  consumes that sequence. Vision is not the bottleneck; can be ported later.
- Verification: token-exact vs the reference (HF) decode for the same input embeddings.

## Architecture (from config.json + safetensors)
- hidden=1280, layers=12, vocab=129280, max_pos=32768
- MHA: 10 heads × head_dim 128 (q/k/v/o = 1280×1280, no bias), RoPE θ=10000
- R-SWA: each decode token attends to ALL prefill (reference/visual+prompt) KV + last
  sliding_window=128 decode-KV. Prefill = full causal.
- RMSNorm eps=1e-6 (pre-attn `input_layernorm`, pre-mlp `post_attention_layernorm`, final `model.norm`)
- Layer 0: dense MLP (SwiGLU, intermediate 6848)
- Layers 1–11: MoE — gate [64,1280] → softmax over 64 → top-6 greedy (no renorm,
  routed_scaling=1.0); experts SwiGLU interm=896; + 2 fused shared experts (interm 1792), always on.
- act = SiLU; expert out = down( silu(gate(x)) * up(x) )

## Weight layout
Read original safetensors via mmap + `manifest.tsv` (name, dtype, shape, abs_offset, nbytes).
No bf16 duplication. fp8 path adds `experts_fp8.bin` (e4m3) + per-(expert,out-row) fp32
scales, produced by `prep_weights.py`.

## Kernels (sm_89), staged
1. rmsnorm (+residual) — bf16 io, fp32 accum                [stage 1, this commit]
2. qkv/o GEMM — cuBLAS (prefill) / custom bf16 GEMV (decode)
3. RoPE apply
4. attention: decode flash (online softmax over ref+window) ; prefill causal
5. router (gate GEMV → softmax → top-6) 
6. MoE expert GEMV — **fp8 weight, bf16 act, dequant-in-reg, fused SwiGLU** (the 2× lever)
7. dense MLP (layer 0)
8. lm_head GEMV + argmax + no-repeat-ngram mask
9. full per-step CUDA graph; KV cache bf16 in HBM

## Build
`engine/Makefile` → nvcc -arch=sm_89. Smoke tests compare each kernel to a CPU reference.

## Status checklist
- [x] recon weights + constants
- [x] manifest generation
- [x] C++ loader (mmap + manifest) + RMSNorm kernel + smoke test  (STAGE-1 OK, sm_89)
- [x] attention prefill (causal) + RoPE — verified
- [x] MoE (bf16) + dense — full prefill forward ARGMAX-MATCHES HF (logits diff 0.58, bf16 noise)
      (root-caused a nasty bug: CK() macro's `cudaError_t e` shadowed loop var `e` → garbage offsets)
- [x] decode loop (KV cache) — 37/39 teacher-forced next-token match vs HF (2 = bf16 near-ties)
      free-running matches 10 tokens then diverges (expected greedy bf16 sensitivity)
- [x] device-resident decode (token+argmax on GPU, K/V written into cache, no per-step sync)
- [ ] **CUDA graph capture of decode step** — THE speed fix. Measured bottleneck = launch
      overhead: ~240 tiny serial ops/token (≈18 ms/tok, ~70µs/op), NOT compute/attention
      (fixed-context vs growing-context only 55 vs 49 tok/s). Graph replay should collapse this.
      Needs: device-resident `pos` (KV write offset + attn clen read from device) so one
      captured graph replays for all steps; cuBLAS via stream capture.
- [ ] R-SWA fixed window (prefill + last 128) — caps attn cost AND matches model semantics
- [ ] fp8 experts (bandwidth) — second-order win once launch overhead is gone
- [ ] benchmark single-stream vs SGLang (target: beat 546 tok/s)

- [x] **CUDA graph** capture/replay of device-position-driven decode step
- [x] flash-style decode attention (8 warps/head, register accum, no per-key block sync)
- [x] fused QKV GEMV + fused gate|up GEMV (kill ~36 tiny cuBLAS launches/token)
- [x] fp8 (e4m3) lm_head GEMV — verified argmax == bf16 (FP8-OK)

## Current status — GOAL MET
Engine is NUMERICALLY CORRECT (prefill argmax == HF; decode 37/39 teacher-forced; fp8 lm_head
argmax == bf16) and **BEATS SGLang single-stream: 605 tok/s vs 546 (~+11%)**.

### Optimization journey (single-stream decode tok/s)
| step | tok/s | what |
|---|---|---|
| naive | 49 | per-op launches; k_experts = 90% of time (6 blocks, serial GEMV) |
| fast MoE | 413 | warp-per-output-row expert GEMV (full occupancy, coalesced) |
| flash attn | 535 | 8-warp/head decode attention, register accum, no per-key sync |
| fused QKV | 546 | 3 GEMV -> 1 |
| fused gate|up | 549 | 2 GEMV -> 1 |
| **fp8 lm_head** | **605** | e4m3 lm_head GEMV (331MB -> 165MB read) |

CUDA graph itself gave ~+8% on top of each (collapses ~240 launches/step). Profiled each stage
with nsys; every step verified against HF fixtures.

### Quantization phase (mixed precision)
| step | tok/s | what | accuracy |
|---|---|---|---|
| bf16 weights | 605 | (above) | 37/39 TF |
| fp8 everywhere | 841 | e4m3 on qkvo + shared + dense + experts + lm_head; router stays bf16 | 37/39 TF (no loss) |
| q4 experts (per-row) | **897** | int4 per-row experts (gate/up/down), fp8 elsewhere | 36/39 TF (−1: a near-tie) |
| **q4 experts (group-128)** | 885 | int4 with one scale per 128-elem group | **38/39 TF (best of all configs)** |

Group-128 int4 finding: fine-grained scales recover the per-row loss AND beat fp8 — int4 with
local group scales captures each block's distribution better than e4m3's fixed mantissa. Cost vs
per-row: −1.3% speed (per-group scale lookups in the GEMV loop) + negligible scale storage
(H/128=10 and MOEI/128=7 scales per row). Net: group-128 is the accuracy/speed sweet spot.

### Extending group-128 int4 to qkvo + shared (vs fp8 there)
| precision split | tok/s | TF | note |
|---|---|---|---|
| fp8 qkvo/shared + q4-g128 experts | 885 | **38/39** | DEFAULT — best accuracy, 1.62× SGLang |
| q4-g128 everywhere (qkvo+shared+experts) | **937** | 35/39 | fastest, 1.72× SGLang, −3 tokens |
Finding: **attention projections are more int4-sensitive than experts.** Experts have 64-way
redundancy that absorbs quant noise (int4 even improved them); Q/K/V/O errors feed the softmax
directly with no redundancy, so int4 there costs 3 TF tokens for +6% speed — not worth it.
Rule of thumb confirmed: FFN/MoE quantize hard, attention wants more bits. Default keeps attention fp8.

### Latency-kernel fixes (subagent-guided, profile-verified)
Subagents proposed vectorized uint4 loads (predicted +25-40%) — but it gave only **+1%**:
nsys showed lm_head was already bandwidth-SATURATED (189µs/165MB = 91% peak) and the small
GEMVs are latency-bound, not load-width-bound. Roofline (Σbytes÷BW ≈ 2000 tok/s) was an
unreachable bound: it ignored ~285µs/token of byte-free latency kernels and assumed uniform
saturation (real: lm_head 91%, qkvo 68%, experts 51%). The real wins were the latency kernels:
- **fused residual-add + RMSNorm** (`k_add_rmsnorm`): one kernel + one global round-trip instead
  of two, ×24/token. 961→980 tok/s, 38/39 unchanged.
- **parallel router** (`k_route1`, <<<1,64>>>): parallelize the 64 `expf` (was 1-thread serial).
  980→996 tok/s.
- **gate+route fusion: REVERTED** — replacing cuBLAS gate GEMV with a single-block kernel (1 of
  142 SMs, serial expert dots) was SLOWER than cuBLAS. Lesson: don't hand-roll what cuBLAS
  parallelizes well; only fuse the genuinely latency-bound tiny kernels.
Now: **996 tok/s = 1.82× SGLang**, 38/39, 1.00 ms/token.

### Fusion/occupancy grind round 2 (996 -> 1067, all exact, 38/39 throughout)
- fused rope + K/V store (`k_rope_store`): 3 kernels -> 1.  +~
- fused gate|up + silu (`k_swiglu_fp8`) for shared/dense: removes silu launch + gu round-trip.  996->1021
- split-KV decode attention (`k_attn_split`/`k_attn_merge`, NSPLIT=12): was 10 blocks (7% of SMs)
  -> NH*NSPLIT blocks.  1021->1058
- folded `k_combine` into shared-down GEMV via bias add (`k_gemv_fp8_bias`).  1058->1066
- custom multi-block gate GEMV (`k_gate`) replacing cuBLAS: neutral (cuBLAS wasn't the cost).
**Plateau at ~1067 tok/s (0.937 ms, 1.95× SGLang).** Four micro-opts gave ~0 (fp8-vectorize,
q4-vectorize, no-atomic down, launch_bounds): remaining kernels are bandwidth-SATURATED
(lm_head 175µs @ 943 GB/s = peak) or batch-1 latency-bound (experts 230µs @ 52%, unimprovable by
occupancy/vectorization). Reaching 1100+ needs fewer bytes on lm_head (int4 + exact top-K rescore
to keep argmax exact) or speculative decoding — not more kernel micro-opts.

### Exact-rescore int4 lm_head -> 1154 tok/s (cleared 1100, still 38/39)
lm_head was 175µs at peak BW reading 165MB fp8. Plain int4 (82MB, ~95µs) lost 2 TF tokens.
Fix = two-stage EXACT argmax: (1) int4 full logits (`k_lmhead_q4`, fast); (2) per-block top-4
candidates over those logits (`k_topk_blocks`, NBAM=256 blocks -> 1024 cands); (3) re-score ONLY
those 1024 in fp8 (`k_rescore_fp8`); (4) argmax over candidates. The emitted token is bit-exact
to the fp8 lm_head (the fp8-argmax is always within the int4 top-1024), so **38/39 restored**.
1067 -> **1154 tok/s (0.866 ms) = 2.11x SGLang**, no accuracy loss. Needs both LMH4 (int4) and
LMH8 (fp8) resident (~247MB). This is the textbook "speculative lm_head": cheap pass ranks, exact
pass confirms.

### Prefill optimization (reusing the decode findings)
Prefill was never optimized — still ran the naive 1-block-per-(token,expert) `k_experts`
(1.77ms/layer): **S=19 took 20.8ms, a 256-token page 199ms** (~200ms TTFT). Replaced the routed
experts with batched warp-per-(token,slot,row) kernels (the decode structure + a token dim).
- **bf16 batched experts** (DEFAULT): S=256 prefill **199 -> 66ms (3.0x), exact (38/39)**. The
  naive kernel was slow from parallelization, not precision — bf16 keeps it matching HF.
- q4 batched experts: faster (46ms) but 36/39 — q4-precision prefill perturbs the KV cache the
  decode attends to. Not worth 2 tokens; kept bf16.
Shared/gate already used cuBLAS (fine for S>1); only routed experts needed the fix. (Note: several
now-dead kernels remain — naive k_experts, S=1 bf16 moe, q4_S — harmless, could be pruned.)

### Parallel argmax
`k_argmax` was single-block (1/142 SMs, ~88µs/token scanning 129k logits). Split into
partial (256 blocks, grid-stride) + final reduce: **884 → 954 tok/s, 38/39 unchanged** (exact
argmax, accuracy-free). Default config now: q4-g128 experts + fp8 attn/shared/dense/lm_head +
bf16 router + parallel argmax = **954 tok/s = 1.75× SGLang single-stream**.

Final: **897 tok/s = 1.64× SGLang single-stream**, prefill argmax == HF, fp8 lm_head argmax == bf16.
Router (gate) kept bf16/fp32 — sensitive & tiny. q4 is per-row symmetric; group-128 int4 would
recover the 1 flipped near-tie at slightly more storage if needed. Remaining hot kernels (nsys):
qkvo/shared fp8 GEMVs (23%), lm_head fp8 (190µs), k_argmax (88µs — single-block, easy to parallelize).

Build: `nvcc -O3 -arch=sm_89 --expt-relaxed-constexpr --expt-extended-lambda engine.cu -o engine_bin -lcublas`
Remaining headroom (not needed to beat, but available): fp8 experts (≈559µs bf16) + fp8 qkvo.

### End-to-end (vision -> engine) + 14-page one-shot
- Wired HF vision encoder -> engine: prefill(...,dembeds) takes the merged vision+text embedding
  sequence; --gen CLI runs prefill+decode+detok. e2e.py / e2e_multi.py capture inputs_embeds and
  drive the engine; output matches HF reference text.
- R-SWA implemented (was growing full attention): RoPE at absolute pos, K/V to a 128-slot ring,
  attention over (prefill + min(decode,128)) keys. Correct + bounded + KV cache only prefill+128.
- 14-page one-shot parse works (abstract verbatim). All parts made faster:
  decode (clen~4000): 507 -> 786 tok/s (adaptive attn split scaled to context);
  prefill (3827 tok): 1248 -> 972 ms (flash-style causal attention);
  vision (14 pages, HF): 1.15 s (unchanged).
  Remaining: fp8 KV cache, grouped-GEMM prefill experts, no-repeat-ngram for very long one-shot.

## Full C++/CUDA pipeline (no Python) — vision port, staged
Goal: image (PDF) -> vision encoder -> LM -> text, entirely in C++/CUDA. LM decoder DONE.
Tooling: MuPDF (bundled libmupdf.so) for PDF render; nvjpeg available. Fixtures: dump_vision_fixture.py
dumps HF ground truth (image_ori, sam_out, clip_out, proj_out) to vfix/ for per-stage verification.
- [x] STAGE 1: PDF render (MuPDF) + GPU preprocess (aspect-preserving resize + center-pad 127 +
      normalize pixel/127.5-1) -> [3,1024,1024]. vision.cu. Matches HF image_ori: mean_abs 0.0097
      (pad-exact; residual = bilinear vs PIL bicubic+antialias at text edges, refine later).
- [ ] STAGE 2: SAM ViT-B — patch_embed conv16, abs pos, 12 blocks (windowed/global rel-pos MHA at
      [2,5,8,11] global), neck (2 conv + LN2d), net_2/net_3 downsample convs -> [1024,16,16].
- [ ] STAGE 3: CLIP-L — uses SAM features as patch embeds, +class+pos+pre-LN, 24 layers -> [257,1024].
- [ ] STAGE 4: concat(clip[1:],sam)=2048 -> projector linear -> [256,1280]; reshape+newline+sep -> 273 tok.
- [ ] STAGE 5: BPE tokenizer (encode prompt, decode output) in C++ (127741 merges, byte-level).
- [ ] STAGE 6: glue main: PDF -> vision -> embeds -> prefill -> decode -> detok. Zero Python.

## COMPLETE: full pure-C++/CUDA e2e (no Python at runtime)
Build: nvcc -O3 -arch=sm_89 -DOCR_LINK engine.cu vision_enc.cu -o ocr_bin -I<mupdf> -lmupdf -lcublas
Run:   ./ocr_bin --ocr <pdf> <page> <maxtok>   (LD_LIBRARY_PATH=<pymupdf libs>)
Pipeline, all CUDA: MuPDF render -> GPU preprocess -> SAM ViT-B -> CLIP-L -> projector ->
  273 visual tokens -> build [277] embeds (bos + visual + "document parsing.") -> LM prefill+decode
  -> byte-level BPE decode -> text. Vision verified per-stage vs HF (SAM 0.0017, CLIP 0.0048, proj 0.0031).
Stages 2-6 DONE. Symbol-isolated (vision_enc.cu static + renamed k_add->k_vadd; engine.cu provides main).
- Vision in CUDA (vision_enc.cu): fp32 + TF32 tensor cores. SAM 76ms + CLIP 7ms/page steady-state.
- Tokenizer: byte-level BPE decoder in C++ (vocab.bin build asset). Prompt ids baked (fixed OCR prompt).
- Single-page e2e: vision 243ms (incl 1-time weight load ~150ms) + prefill 155ms + decode 952 tok/s. Correct OCR.
Bug found: k_assemble launched <<<,1280>>> (>1024 max threads) -> silent fail -> garbage visual tokens;
  fixed to 2D grid. (Standalone only verified proj, computed before assembly, so it hid the bug.)
Remaining vision headroom (not done): bf16 compute, flash attention for SAM global blocks (4x 4096^2 S matrices).

## Vision optimization (make vision faster)
Profiled the warm per-page vision (after 1-time weight load): it was LAUNCH-bound, not compute-bound
(~25ms kernel GPU time vs 83ms wall). Fixes:
- TF32 tensor cores (cublasSetMathMode) for the fp32 GEMMs.
- Preallocated all forward buffers (no per-call cudaMalloc/cudaFree).
- **CUDA graph** for the GPU forward (SAM+CLIP+projector): capture once, replay per page. GPU forward
  83ms -> 47ms (launch overhead gone). Build needs vision_enc.cu compiled with `-default-stream
  per-thread` (separate object) so default-stream launches hit the capturable per-thread stream;
  cublasSetStream(CUB, cudaStreamPerThread); capture cudaStreamPerThread. Sync cudaMemcpy->Async in capture.
- Dead end: flash-per-query attention for SAM global blocks was SLOWER (176ms) than explicit cuBLAS —
  with N=4096 queries it re-reads K/V from global per query; cuBLAS tiles and reuses them. Reverted.
Warm vision now ~91ms/page = render 44ms (MuPDF CPU rasterization, inherent) + GPU forward 47ms (graph).
Build: nvcc -default-stream per-thread -DOCR_LINK -c vision_enc.cu; nvcc -c engine.cu; link both.
Remaining headroom: bf16 GPU forward (~halve 47ms), render/GPU overlap across pages, lower render dpi.

## Vision bf16 + render/GPU interleaving
- Full bf16 conversion of the vision encoder (activations bf16, fp32 accum; GEMMs cublasGemmEx/
  GemmStridedBatchedEx CUDA_R_16BF compute_32F; small params LN/bias/rel-pos kept fp32; scores
  bf16). Weights loaded bf16 directly (Wb) - no fp32 conversion, half the weight memory/load.
  Verified vs HF: SAM 0.0035, CLIP 0.0089, proj 0.0058 mean_abs (HF is bf16 too). Warm GPU forward
  47ms -> 35ms. Output still correct.
- Multi-page render/GPU interleaving (--ocrmulti): split MuPDF render (CPU) from GPU forward. Loop
  launches GPU(page i) async (graph), renders page i+1 on CPU concurrently, syncs, accumulates 273
  tokens, uploads i+1. Per-page vision 79ms (render44+GPU35 sequential) -> ~48ms steady-state
  (overlapped, render-bound). Multi-page is one-shot: [bos][N*273][Multi,page,parsing,.] -> prefill
  -> decode. Verified correct on 4 pages (full abstract incl. github URL).

## Per-page vision optimization (focus pass)
Profiled the per-page vision in the interleaved multi-page path:
- Document re-open: vis_render_cpu called fz_open_document EVERY page (re-parsing the PDF). Cached
  ctx+doc (open once) -> rasterize drops 44ms -> ~7ms/page after page 0 (44ms was first-page MuPDF
  font/cache warmup + per-call reparse). Render now fully hidden behind the GPU forward.
- Clean GPU-forward measurement (GBENCH: replay graph 50x): 36 ms/page. The render was NOT the
  per-page bottleneck (it overlaps); the ~34ms GPU forward is the floor.
- k_relpos: was 32-thread blocks re-reading q from global per thread -> 1 block/(qi,bh), q cached in
  shared, 1 thread/key-row. (Distributed in the graph; ~0 measurable.)
- k_conv: terrible occupancy (16 threads/block for net_3) -> flat grid 256 threads. 36 -> 33.7 ms/page.
Per-page vision now ~34ms GPU floor (render hidden); 14-page run 55ms/page amortized (incl 1-time
graph capture ~130ms + p0 warmup). Remaining cost is distributed across ~100 graph kernels (SAM
global attention softmax, CLIP 24 layers); further gains need query-tiled flash or kernel fusion.

## Vision GPU-forward kernel optimization (36 -> 23 ms/page)
Profiled the bf16 GPU forward un-graphed (NOGRAPH env, 20 iters). Targeted the top kernels:
- k_relpos 8.9ms: was 49K tiny blocks each reloading the rel-pos table (uncoalesced). Rewrote as one
  block per (batch, query-row): all W queries in a row share the same Rh rows (loaded once to shared);
  rel_w loads the whole tabW once. 768 blocks for global instead of 49K. -> 2.6ms (opt-in 65KB shared).
- k_conv 5.5ms: naive direct conv at ~1 TFLOP/s. Routed through cuBLAS bf16 tensor cores: 1x1 = direct
  GemmEx; 3x3 = im2col (k_im2col3) + GemmEx. -> ~1ms.
- k_biassoftmax 5.5ms: 3 passes over the 402MB global S. Cache the row in shared (float) -> 1 read +
  1 write. -> 4.9ms (memory-bound floor; also improved accuracy by avoiding bf16 round-trips). Further
  cuts need flash attention, but analysis shows it's not a clear win here: cuBLAS qk^T/Sv are already
  very efficient, and hand-written flash re-reads K/V more than cuBLAS tiles do; only saves S traffic.
Result: GPU forward 36 -> 23.3 ms/page (GBENCH graph replay). 14-page interleaved run 55 -> 43 ms/page.
Floor now: necessary bf16 GEMMs (~8ms) + softmax S traffic (~4ms). Verified vs HF (sam_out 0.0025).

## Flash attention for SAM global blocks — tried, reverted (measured loss)
Implemented a tiled flash kernel (k_flash_global): warp-per-query, BQ=8 queries/block sharing K/V
tiles, one key-row per tile (so bias = rel_h[qi,kt] + rel_w[qi,kw] separates cleanly). Correct
(sam_out mean_abs 0.0021, even better than explicit) but **3.3x SLOWER: 77ms vs 23ms GPU forward**.
Reason: it re-loads K/V across 512 query-tiles/head with a serial per-warp key loop; cuBLAS's qk^T/Sv
GEMMs tile K/V optimally and win decisively. Flash helps for decode (1 query) or when you can't afford
to materialize S; for full-attention prefill against an efficient GEMM lib, explicit > flash. Reverted.

## Cleanup pass (single-threaded, minimal CLI)
- CLI collapsed to one positional form: `ocr_bin [pdf] [npages] [maxtok]` (defaults: bundled paper,
  1 page, 4096 tokens). Removed the --ocr/--ocrmulti/--gen flags. N pages -> one-shot multi-page parse.
- Removed slow/dev/dead code: the --gen file-bridge mode, the teacher-forced fixture benchmark in
  main(), dead kernels (k_conv replaced by cuBLAS convs; k_flash_global reverted flash), and all
  debug env paths (VFIX/NOGRAPH/GBENCH/VR). Deleted obsolete files: main.cu, vision.cu, e2e.py,
  e2e_multi.py (old Python bridge / stage-1 scaffolds). Kept prep_weights.py (manifest build asset).
- Single host thread by design: MuPDF context uses NULL locks (no internal threads); cuBLAS host-side
  single-thread; no OpenMP. (The vision "-default-stream per-thread" is a CUDA-stream mechanism for
  graph capture, not CPU multithreading.)
- Makefile: `make` builds vision_enc.o + engine.o + links (with -rpath so ./ocr_bin runs without
  LD_LIBRARY_PATH). engine.cu 769 -> 627 lines.

## Pluggable KV cache (bf16 / fp8 / KVarN-ready)
KV element is a compile-time backend (engine.cu top): `kvt` + `kvld()`/`kvst()` + `k_seedkv`. These are
the ONLY KV touch points (k_rope_store stores, k_attn_split reads, prefill seeds). `make KV=fp8` ->
fp8 e4m3; default bf16. Verified: bf16 and fp8 both produce correct OCR.
Finding (14-page, 3827-token prefix): fp8 k_attn_split 416 -> 381 us/step (only -9%), decode 792 ->
811 tok/s (+2.4%). The long-context R-SWA attention is COMPUTE-bound (q.k dots + online-softmax exp
over all reference tokens), NOT bandwidth-bound — so KV quantization buys MEMORY, not speed:
fp8 halves the KV cache 243 -> 121 MB (14 pages), which is the lever for fitting MORE pages.
KVarN backend (4-bit K per-channel + 2-bit V per-token + Hadamard + Sinkhorn, 128-tiles): plugs into
the same touch points but needs a richer backend (packed bits + per-channel/per-token scales + a
vector load, plus a query-Hadamard hook). ~4-6x smaller KV (max pages) — worth it for capacity, not
decode speed (attention is compute-bound). Larger CUDA port (their FWHT + Sinkhorn kernels).

## Decode attention compute optimization (791 -> 918 tok/s, 14-page)
Profiling showed the long-context decode is attention-COMPUTE-bound (k_attn_split 358-416us/step:
per-key warp-reduce dot + online-softmax exp over all 3827 reference keys), not bandwidth-bound.
Wins: (1) 2-pass softmax (1 exp/key + breaks the sequential rescale chain) -6%; (2) #pragma unroll 8
both passes (pipeline the per-key shfl-reduce latency) +7%; (3) split-KV ~48 keys/block (was 96) for
more ILP/parallelism +5%. Net decode 791 -> 918 tok/s (+16%), ~12x HF-eager (75 tok/s).
Tried and REVERTED (slower): 3-phase thread-per-key full dots (non-coalesced K reads, 845); shared-K
load (32-way bank conflicts, 755). The dims-split + coalesced reads + per-key warp-reduce IS the
FlashDecoding-optimal structure for single-query decode; tensor cores don't apply (M=1). The attention
is at its practical limit. Reaching >1000 would need the constant kernels (MoE int4 gateup is ~2x its
bandwidth floor; gemv/lm_head are at floor) — i.e. lower precision (accuracy cost, vetoed for attention
projections) or deeper MoE-kernel work, not more attention tuning.

## Breaking 1000 tok/s — ncu-guided (Nsight Compute, not nsys)
nsys gives durations only; ncu reads HW counters (needs RmProfilingAdminOnly=0 or sudo). Key metric:
"Warp Cycles Per Issued Instruction" + SpeedOfLight (Compute% vs DRAM%). Finding: the decode GEMV
kernels are LATENCY-bound (neither compute nor DRAM saturated, <55%), not throughput-bound:
  k_attn_split: 6.1 cyc/issue (healthy);  k_moe_gateup_q4: 13.3 cyc/issue (stalled on load->FMA scoreboard).
Fixes (all accuracy-neutral, ncu-pointed):
  1. MoE gateup/down + lmhead + dot_fp8_vec: 4 independent accumulators -> ILP hides the load->FMA
     scoreboard latency.  gateup 167->123us.
  2. fp8 GEMVs: 256->128 thr/block. ncu showed only 49% occupancy on small GEMVs (o_proj 1280 rows ->
     160 blocks of 8 warps barely fills 142 SMs once = wave-quantization tail). 128-thr packs the tail.
Decode 791 -> 918 (attn) -> 940 (merge) -> 971 (gateup ILP) -> 986 (gemv/lmhead ILP) -> 1000 (128-thr).
Full 14-page doc steady-state: 1033 tok/s, 13498 tok, output correct. ~13.5x HF-eager (75 tok/s).
Lesson: the kernels looked "at floor" by a naive byte count; ncu showed they were latency-bound with
real headroom. Profile with the right tool before declaring a floor.

## Page-parallel batched decode (PAGEPAR=1) — exploits page independence
Validated empirically: a page decoded alone is 99.5% identical to that page inside the 14-page run
(only ~5px coord jitter). So pages are self-contained -> decode them as a BATCH of N independent
single-page streams (each its own 278-tok reference + 128 R-SWA ring), one batch-N forward per step.
Turns batch-1 latency-bound GEMVs into batch-N work that fills the idle GPU.
Implementation (generate_pagepar, env PAGEPAR=1): per-page prefill -> N reference KVs in kcb/vcb;
batched decode reuses prefill bf16 kernels (lin/cuBLAS, mlp_dense, k_rmsnorm) with S=N. New kernels:
k_attn_split_b/merge_b (R-SWA per stream, block-diagonal), k_rope_store_b, k_argmax_b, k_pageembeds,
k_record_b/k_setdone_b (device-resident loop state, sync every 16 steps), and crucially
k_moe_gateup_q4_b/k_moe_down_q4_b (INT4 batched MoE, warp-per-expert-row looping tokens — the reused
bf16 prefill MoE was 74% of time at 4x the int4 bandwidth).
Result (14-page): decode 1033 -> 1623 tok/s, e2e 16.2 -> 9.7s (~1.67x), output 95.6% identical
(184 vs 185 elements; differences = coord jitter + bf16-vs-int4 precision). Correct full document.
Remaining levers (not yet done): MoE still 56% (grouped-GEMM gather-by-expert would beat the
per-(expert,row) token loop); batch-shrink (1696 steps vs ~972 avg page -> ~43% wasted compute on
finished streams); int4 batched lm_head (currently bf16 full GEMM); CUDA-graph capture of the step.
These would push toward the ~3x theoretical. The EXACT speculative-coupling variant (window-coupling
verification) builds on this same batched path.

## Page-parallel tightening (ncu-driven, batch-2 and batch-14)
ncu on the batched kernels found two loose spots:
1. MoE batched kernels launched ALL 64 experts (warp per (expert,row)) so at small batch most warps
   check-and-exit (DRAM 14%, idle). Fix: index by ACTIVE (token,slot,row) = B*TOPK*MOEI warps, all real
   work, + 4-accumulator ILP. batch-14 decode 1623 -> 1989 tok/s.
2. At batch-2 the cuBLAS GEMMs were 53% (bf16 lm_head reads full 330MB LMH/step). Fix: int4 batched
   lm_head (warp-per-row, loop B, row stays L2-resident ~83MB) for B<=4; cuBLAS bf16 GEMM still wins at
   B>=7 (real M). batch-2 823 -> 912.
Decode scaling now: b2=912, b7=1823, b14=1966 tok/s. Full 14-page e2e: 16.2s(seq) -> 8.4s (~1.93x).
Still loose (ncu): k_attn_split_b occupancy 2.3% at small batch (1 warp/block, ~160 blocks);
cuBLAS qkv/o/shared at tiny M (-> fp8 batched gemv); batch-shrink (~43% wasted on finished streams);
graph capture.

## Page-parallel: pushing decode toward 3000 tok/s (1623 -> ~2850)
Three levers, in order of impact:
1. int4 MoE active-warp indexing (k_moe_*_q4_b: warp per ACTIVE (token,slot,row) + 4-acc ILP): 1623->1989.
2. BATCH-SHRINK (the big one): pages finish at different lengths (avg ~972, longest ~1700), so a fixed
   batch wastes ~43% compute on done streams. Added an active-stream compaction: d_act maps active slot
   -> physical stream (KV stays in place), recompact every 16 steps, kernels (attn/rope/record/setdone)
   take the act map. Batch util 57% -> 99%. 1989 -> 2797.
3. Per-active-count CUDA graph (gcache[NA]): the batched path was 24% launch-bound; capture one step-graph
   per NA (the NA=14 bulk reuses one), cuBLAS made capture-safe via cublasSetWorkspace. 2797 -> 2881.
Also fused residual+norm (k_add_rmsnorm batched). Now COMPUTE-bound (~2850, peak 2881): MoE q4 ~36%,
cuBLAS GEMMs ~27% (incl bf16 lm_head 330MB/step), attention ~8%.
Dead ends (all LOSE to cuBLAS bf16 GEMM at batch>=~5): per-token fp8 projections (re-read+re-dequant
per token); dequant-once fp8 (low parallelism / predication waste); int4 lm_head (cuBLAS GEMM wins).
Lesson: at batch the dense projections are GEMM-shaped and cuBLAS's tiling beats hand fp8 kernels despite
2x the bytes; realizing fp8 at batch needs cublasLt fp8 TENSOR-CORE GEMM, not warp kernels.
Final 14-page: decode ~2850 tok/s (2.76x sequential 1033, ~38x HF-eager), e2e ~6.3s vs 16.2s. 95.6% identical.

## vLLM comparison + ncu re-audit (2026-07-01)
Benchmarked vs vLLM main (0.23.1rc1.dev704+g4787f2dd1, precompiled; PyPI 0.24.0 lacks the
UnlimitedOCRForCausalLM arch) serving ./model with the official recipe flags (NGramPerReqLogitsProcessor,
--no-enable-prefix-caching, --mm-processor-cache-gb 0). vLLM's port is faithful (R-SWA via FlexAttention
mask; single-image=gundam, multi-image=base). Same 14pg paper, 300dpi pages, greedy, ngram 35.
Setup lives in .venv-vllm (serve needs ninja on PATH); client = vllm_bench.py; outputs_vllm/.

| workload | engine | vLLM | ratio |
|---|---|---|---|
| 1pg gundam | 2.2s cold / ~0.9s warm, 935 tok/s dec | 4.0s (TTFT 1.1s, 176 tok/s dec) | ~4.4x warm |
| 14pg one-shot (base) | 6.4s e2e, 3114 tok/s dec | 105.9s, 144 tok/s dec | 16.5x |
| 14pg throughput | 6.4s base / 8.0s gundam-batched (2900 tok/s) | 21.8s = 14 conc reqs, 615 tok/s agg | 2.7-3.4x |
| GPU mem | ~10 GB | 45.4 GB (0.9 prealloc) | |

Output parity: 1pg 99.8% identical (coords stripped); 14pg 86.7% (= pagepar-vs-sequential semantics
+ greedy drift; no repetition either side; vLLM 202 det els vs engine 185). Structural reason vLLM
loses: sequential bf16 batch-1 decode of a 14k-token doc; its continuous batching (615 tok/s) can't
express page-parallel decode.

ncu re-audit of the batched Base path (sudo /usr/local/cuda/bin/ncu, DECNOGRAPH=1; decode is 100%
GPU-busy — zero launch slack; 2.68ms/step @ NA=14: MoE 1.26ms=47%, cuBLAS proj+lmhead 800us, attn 228us):
- k_moe_gateup_q4_b / k_moe_down_q4_b: SM 79/83%, warps active 93/86%, DRAM 10/9%, ~16 cyc/inst
  -> compute/ISSUE-bound on dequant+FMA (confirms the register-limited ILP-optimum finding; the
  bandwidth story is over at batch — bytes are NOT the limiter).
- lm_head bf16 GEMM (B>=7): 92.6% DRAM reading 331MB/step = at bf16 floor; only fewer bytes helps.
- projections cutlass 16x16 @ M=14: qkv near BW floor; small ones (o_proj etc) latency-bound (2% occ).
- k_attn_split_b @ NA=14: 15% occupancy, DRAM 48% (bulk ns=8 never re-tuned).

### Remaining levers, ranked (NOT yet tried; accuracy gates = md5 pair + brochure small-text)
[2026-07-02 outcomes: (1) DONE +25.5% = TCMOE below. (2) DONE +6.1% token-exact = LMHMMA below.
(3) TRIED & PARKED as custom mma kernels, -4%, see "FP8MMA" section. (4) TRIED & REVERTED, +1% = noise.]
1. TENSOR-CORE INT4 MoE (est +20-30% decode): MoE is 47% of step at 10% DRAM. Feed RAW int4 values
   to WMMA as bf16 (ints -8..7 are bf16-EXACT), fp32 accum per K=128 group, apply fp32 group scale
   POST-accumulation: same products reassociated (no fp8-style rounding), gather (token,slot) by
   expert into M=16 tiles (avg 1.3 tok/expert -> ~9x pad waste, but TCs have ~30x headroom over the
   5.8 TFLOP/s the ALU path achieves). Sidesteps both "no int4 TCs on Ada" (uses bf16 TCs) and the
   fp8 small-text veto. Verify vs gates before believing any number.
2. BATCHED EXACT-RESCORE INT4 LM_HEAD for B>=7 (est +6-8%): revive the deleted single-stream
   two-stage exact argmax (int4 full logits rank -> bf16 rescore top-1024 -> argmax; provably equal
   if true argmax lands in top-1024 — it always did). Kernels (k_lmhead_q4/k_topk_blocks/k_rescore/
   k_argmax_cand) are in git history; batch them. 331MB -> ~83MB+rescore per step. The earlier
   "int4 lm_head loses at batch" dead end was the PLAIN tiled kernel, not the rescore variant.
3. cublasLt FP8 TENSOR-CORE GEMM for qkvo/shared at B>=5 (est +8-9%): the noted "realizing fp8 at
   batch needs cublasLt fp8 GEMM" lever. fp8 projections already ship at NA<=3, so precedent exists,
   but fp8 flipped small-text tokens on lm_head/vision -> ship ONLY if brochure check passes.
4. Bulk attn split ns 8->16/24 (est +1-3%, cheap): k_attn_split_b occupancy 15% at NA=14.
Not worth re-trying (re-verified): vision at bf16 roofline (SAM SxV GEMM 88% DRAM), vision/prefill
pipelining (reverted, ~4%), qkv GEMM (near floor), launch overhead (graphs already remove it).

## Tensor-core int4 kernels: TCMOE + LMHMMA (2026-07-02, levers 1+2 from the ncu re-audit)
Design chosen by a 3-design/3-judge panel (PTX mma.sync m16n8k16 beat WMMA-API and split-K-by-group variants).
**TCMOE** (`k_moe_bins`/`k_moe_gateup_mma`/`k_moe_down_mma`/`k_moe_comb_b`; dispatch hardwired at NA>=TCMIN=4 —
the TCMOE/TCMOE_MASK env gates existed during bring-up and were removed in the 2026-07-02 single-engine
consolidation after sign-off): replaces the warp-per-(token,slot,row) int4 MoE at NA>=4 (ncu showed it
ISSUE-bound: SM 79-83%, DRAM 9-10%). A-fragment = 16 weight rows (8 gate + 8 up of the SAME r-range -> silu
register-local; down: d,d+8), B = 8 tokens, fp32 accumulators; fp32 group scale applied IN-REGISTER to the
group partial via the C-fragment row map (r = 8j+lane/4) -- no shared round-trip; weights stream DRAM->reg
in fragment order (never touch shared) from a load-time relay of the SAME q4 nibbles (biased +8 = raw^8;
(0x4300|m)-0x4308 = m-8 EXACT in bf16 -> products exact, reassociated only). Deterministic: bins built in
fixed (t,slot) order (`k_moe_bins`), down combine in fixed slot order. Graph-safe: fixed grids, cnt-guarded
early exits. TCDBG=n A/B vs old kernels on live activations (measured: mh sub-bf16-ULP, Yf ~1e-5 = pure
reassociation). racecheck/initcheck/memcheck CLEAN. md5 gates BIT-EXACT (NA=1 keeps old kernels).
**Perf: 14pg decode 3095 -> 3884 tok/s (+25.5%); N=4 +6.6%, N=7 +13.6%, Gundam-batched 14pg 2807 -> 3536 (+26%).**
Output drift (fp32 reassociation flipping greedy near-ties): 14pg = 99.96% coords-stripped similarity —
exactly 3 spots: "Baidu"+"百度" header boxes merged (184 vs 185 els), one TOC dot-leader dropped, one
table-header cell TED5->TEDS1. Deterministic run-to-run. **Drift SIGNED OFF by user 2026-07-02**, after
which the env gate was removed — the tensor-core path IS the engine at NA>=4 (rebuild with TCMIN huge to
reproduce pre-TCMOE output). New 14pg base regression md5 (OCR section): af3a8ae8e348d6b2104b3544363b4f37.
**LMHMMA** (`k_lmhead_mma`/`k_topk_blocks_b`/`k_rescore_b`/`k_argmax_cand_b`; hardwired at na>4, the LMHMMA
env gate removed in the same consolidation; LMHDBG=n token-parity A/B vs bf16 cuBLAS full logits remains):
revives the exact-rescore lm_head (DESIGN 138-146) batched, for na>=5: int4 mma ranking
(83MB, same tile skeleton, per-(row,group) fp32 scales) -> ngram mask on ranked logits (bans can't become
candidates; rescore preserves -1e30) -> per-block top-4 = 1024 cands/token -> bf16 rescore (fp32 accum) ->
argmax. **TOKEN-EXACT vs the 331MB bf16 cuBLAS path: 0 mismatches over the full 14pg run (LMHDBG); during
bring-up, this path combined with the then-existing TCMOE=0 gate reproduced the pre-change reference
output BIT-IDENTICALLY — that historical validation is why it ships ungated.**
**Perf: +6.1% on top of TCMOE -> 4121 tok/s 14pg first measurement, 4148-4175 re-measured on idle GPU
(+34% total vs 3095). na<=4 path untouched (md5 gates).**

## FP8MMA: fp8-mma projections at NA>3 — measured, PARKED (2026-07-02, lever 3)
Attempted the "fp8 projections at batch" lever as custom mma kernels instead of cublasLt (deterministic,
graph-safe, reuses the TCMOE tile skeleton): `k_proj_mma_fp8` (qkv/o/dense-down/shared-down, optional
fused bias folds `k_combine`) + `k_swiglu_mma_fp8` (gate|up interleaved per m16 tile, register SwiGLU),
weights from a load-time relay of the SAME e4m3 bytes + per-row scale pairs (`repack_p8`). Env `FP8MMA`
= bitmask 1 qkv / 2 o / 4 dense / 8 shared (1 -> all), default **0 = OFF**; repacks only allocated when
enabled (~180MB). `FP8DBG=n` A/Bs vs the per-token fp8 kernels on live activations.
- v1 (e4m3->bf16 via 256-entry shared LUT): 3225 tok/s 14pg = **-22%**. Bottleneck: ~1.3k random-indexed
  shared loads/thread (bank-conflict serialized).
- v2 (sm_89 hw `cvt.rn.f16x2.e4m3x2` + f16 mma, exact for weights AND bf16 activations in f16 range):
  3995 tok/s = **-4%**; Gundam-50 (NA=50) 3836 vs 4214 = -9%. Numerics: FP8DBG sub-ULP vs per-token fp8
  on all 5 sites (kernels correct); output vs bf16 path 99.85% sim on the paper PDF but only **92.4% on
  the small-text brochure** (850 vs 868 det els) — fp8-at-batch measurably hurts dense small text, so it
  would have needed a hard accuracy fight even at a perf win.
- Why it loses: nsys shows k_proj_mma_fp8 at 17.2us avg vs 5.6-7.8us for the cuBLAS bf16 GEMMs it
  replaces. At these sizes cuBLAS is already near the DRAM floor (96MB L2 absorbs re-reads), so fp8's 2x
  weight-traffic saving only materializes with a cutlass-grade cp.async multi-stage pipeline — not worth
  the complexity + accuracy sign-off burden for <=+10% ceiling. Grid starvation (20-60 blocks at NA=14)
  was the v1 co-factor; v2 closed most of it, the pipeline gap is what remains.
Verdict: engine keeps cuBLAS bf16 at NA>PPSMALL. **Code REMOVED from the tree** in the same-day
single-engine consolidation (user: "single engine, not multiple gated ones") — it was never committed, so
THIS SECTION is the record: kernels `k_proj_mma_fp8`/`k_swiglu_mma_fp8` (TCMOE tile skeleton + P8FRAGH =
hw `cvt.rn.f16x2.e4m3x2` pairs + `b2h2` bf16->f16 staging + f16 mma), `repack_p8` relay (word = bytes
[rA k, rA k+1, rB k, rB k+1], word1 at k+8, pair=1 interleaves gate|up rows 8j / rows/2+8j, per-row scale
float2 pairs), dispatch was `else if(fp8mma&bit)` between the small-batch and cuBLAS branches. Rebuild
from this + the TCMOE kernels in ~200 lines if NA regimes ever change (e.g. batch-windowing at NA>>50).

## Windowed admission: unlimited pages at flat VRAM (2026-07-02, ROADMAP WS1+WS2)
The spec's lazy page loading + page-level KV eviction collapse, in this engine's page-parallel
architecture, into ONE mechanism: a rolling window of W resident page-streams (`WINDOW` env, default 128).
`generate_pagepar(N,dembeds,po,vpp,encpg)` — with an `encpg(page,dst)` callback (base mode: `enc_base`,
lazy vision with CPU-render lookahead) pages are encoded+prefilled ON ADMISSION into a recycled slot;
without it (Gundam, gundam-mixed fallback) all pages are resident as before. Mechanics: per-SLOT step
counters `d_steps[W]` (k_rope_store_b/k_attn_split_b/k_ngram_mask/k_record_b index dstep[slot];
k_incstep_b increments active slots) so streams admitted at different times coexist in one batched step;
admission at the 16-step sync boundary tops the window back up (encode -> prefill -> seed slot KV -> join
d_act); retirement harvests the slot's outbuf row into its page and frees the slot. Invariants held: a
stream's positions are a pure function of its page (never admission time); slot recycling never renumbers
anything (prefix region overwritten by the seed copy, ring region only read up to the stream's own step);
WIN=128 and all weights untouched. BDPREFILL block-diag prefill deleted (windowing removed its upfront-
embeds input; was a no-speedup experiment, EXPERIMENTS Exp-2).
**Results (112-page doc = 8x paper, base mode; final numbers on idle GPU):** one continuous session,
108,752 tok in 6.0 s decode = **18,136 tok/s**; TTFT 353 ms (page-0 encode+prefill vs whole-doc prefill);
99% batch util maintained by rolling admission; peak process VRAM 11.4 GB (W=16) / 12.1 (W=64) / 12.7
(W=112) — W-bound, not N-bound. Throughput SCALES with resident batch (per-token step cost 56us at NA=112
vs 110us at NA=64) -> default WINDOW=128 (memory flattens past it, full speed below it). Output: W=64
windowed 112pg first block 99.89% identical to the 14pg reference, 185 det els per 14-page block;
WINDOW=4 stress (14pg through 4 slots) 99.53%. All 3 md5 gates BIT-EXACT and gundam-50 BIT-IDENTICAL to
the pre-windowing engine (N<=W reproduces classic full-batch behavior exactly), make check PASS.
**Codegen bit-exactness lesson:** the first windowing build changed gundam-50 output (a coordinate
near-tie flipped at step ~160, cascading to a 4096-step run-on page and -35%% tok/s on that asset) with
ZERO logical change — hoisting `int ps=act[s]` above the `dstep` load made ptxas reschedule the FP
sequence of k_attn_split_b (244 vs 316 FP ops), and sub-ULP rounding differences flip near-ties on
gundam's 3.1k-key accumulations (base's 406-key chains were unaffected -> base md5s alone did NOT catch
it). Fix: keep the exact expression shape (`dstep[act[s]]` swapped in-place for `*dstep`, original `ps`
line untouched) -> FP opcode sequence identical (cuobjdump -sass diff) -> gundam-50 md5 restored.
When touching bit-verified kernels, diff the SASS FP-opcode sequence, and gate on the LONGEST-chain
workload, not just the fastest one.

## Multi-document server: `ocr_bin serve` (2026-07-02)
The one-shot CLI became a persistent server by generalizing WHO feeds windowed admission. The decode
loop was refactored behind a `PageSrc` interface (next/embeds/out/done/wait — hooks run on the engine
thread at admission/16-step sync boundaries only, never inside graph capture): `FixedSrc` reproduces the
CLI byte-for-byte (verified: full-binary SASS identical, all 3 md5 gates + gundam-50 bit-exact after the
refactor), `QueueSrc` feeds pages from an HTTP job queue. Documents ARE just a grouping of pages: pages
of different documents co-batch in one decode window; per-job round-robin admission (one page per job
per free slot) keeps a 1-page doc from queueing behind a 500-pager. `server.cpp` is the dependency-free
HTTP/1.1 front (POST /ocr[?pages=N][&gundam=1] body=PDF -> text/plain; GET /healthz; Expect:100-continue,
lingering close, strict Content-Length, 16-job/64-conn caps -> 503; spool to $TMPDIR/ocr_srv.<pid>).
Connection threads never touch CUDA/MuPDF — the engine thread is the sole consumer (MuPDF ctx has NULL
locks; vision runs on cudaStreamPerThread, so encode must stay on the engine thread forever).
- **Gundam jobs = exclusive interludes, now ALSO windowed**: per-page vpp is heterogeneous across
  DIFFERENT documents, so gundam can't co-batch with base or other docs in one window. But WITHIN a
  uniform-tiling doc, gundam now uses the same windowed admission as base: `ocr_gundam()` checks page
  dims are uniform (cheap `gundam_page_ntok`, no encode), then runs `generate_pagepar(N,nullptr,po,nt0,
  enc_gundam,...,wcap=vram_wcap(nt0))` — per-slot lazy tile-encode streaming through W = min(WINDOW,
  VRAM-safe) slots. VRAM is bounded by W (not N) -> UNLIMITED gundam pages at flat memory; the old
  `GUNDAM_PAGE_CAP` is gone. N<=W reproduces the pre-windowing all-upfront path bit-exactly (gundam-50
  md5 IDENTICAL; verified 68pg@26GB past the old cap). Mixed page sizes fall back to sequential
  per-page (already flat memory). When a gundam job reaches the queue head, residents drain, the core
  returns, the interlude runs, then the base loop re-enters. Bounded head-of-line as before.
- **Determinism contract**: a job on an IDLE server is byte-identical to the CLI (same NA trajectory;
  W=wcap vs min(N,wcap) only caps admission — gated by tools/server_check.sh parity stanzas). Under
  CONCURRENT load, co-batching changes the NA trajectory -> cuBLAS shapes/ns/TCMOE tiling -> argmax
  near-ties can flip (measured: 12 diff lines on a 515-token page co-batched with 64 other pages).
  Same numeric class as windowed NA variation; inherent to batched serving (vLLM behaves the same).
  Do NOT "re-optimize" documents into serialized exclusive runs to erase this — it collapses many-small-
  doc throughput (NA=1-2 vs co-batched NA<=128).
- **Robustness**: fz_try wrappers (fzdoc/render/count — previously zero fz_try: corrupt PDF = process
  abort) fail the job with 422 instead; unrenderable page -> embeds() nullptr -> page skipped, no slot
  consumed, job 422. Doc handles live in an LRU-8 cache keyed by path (round-robin across docs would
  thrash the old single-doc cache); `vis_doc_close()` at job completion so recycled mkstemp paths can
  never serve a stale document; the render-prefetch marker is keyed (pdf,page) for the same reason.
  MuPDF store bounded (FZ_STORE_DEFAULT=256MB, was UNLIMITED). CUDA errors stay fail-fast exit(1)
  (supervisor restarts; ~30s weight reload; readiness = socket bound only after load_weights).
- **Measured** (RTX 6000 Ada, KV=fp8): 14pg+50pg concurrent docs 8.7s/12.1s (64 pages co-batched,
  99% util); 1-page doc arriving mid-decode joins at the next admission boundary and returns in 3.1s
  (vs ~12s drain-then-run); idle 1pg 0.9s e2e. Gates: make check PASS, all 3 md5 BIT-EXACT, gundam-50
  BIT-IDENTICAL, full-SASS diff clean, make servercheck PASS.

## Server-vs-server showdown vs tuned vLLM (2026-07-02/03)
Saturating multi-document fleets against vLLM (git main dev704) serving the same model, vLLM tuned
flags-only to its best (615 tok/s @14 conc -> 1207 @224: `--max-num-seqs 256 --max-num-batched-tokens
16384 --max-model-len 12288 --async-scheduling`; 0.95 util + 32k batched tokens OOM-crashes on vision
transients; fp8 KV rejected by FlexAttention). vLLM = 1 crop-mode request/page (its best case, client
renders pages offline); engine = 1 POST/doc to `ocr_bin serve` (renders internally).
- 16x paper14 (224pg): engine 20.6s, 10,546 tok/s, doc lat 20-21s | vLLM 177.7s, 1,207 tok/s, lat
  174-178s -> **8.7x**. Mixed 2x14+2x50+1x112 (240pg): 27.3s/9,511 vs 225.1s/1,132 -> **8.4x**.
- Gundam mode-matched (serialized interludes, our weak spot): 85.2s, 2,538 tok/s -> still 2.1x.
- Parity coords-stripped: paper 99.4%; brochure gundam-vs-crop 84.4% (cross-stack greedy drift class).
- AWQ W4A16 community checkpoint in vLLM (attention+experts int4): loads (after fixing a vLLM-side
  quant-ignore prefix bug) but +23% degeneration bloat, slower wall than bf16, brochure similarity 0.03
  -> empirically confirms the engine's precision placement (int4 experts + exact-rescore lm_head ONLY;
  attention stays fp8/bf16). Structural gap: vLLM's floor is bf16 MoE decode + FlexAttention R-SWA
  (~10x cudagraph tax documented upstream); fleet clients: ../vllm_fleet.py, ../engine_fleet.py.
