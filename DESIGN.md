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
