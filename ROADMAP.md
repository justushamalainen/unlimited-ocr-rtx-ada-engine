# ROADMAP — Training-Free Extensions to Unlimited-OCR (R-SWA)

Spec provided 2026-07-01 (user). Zero weight changes, zero training — all work in the engine's
KV-cache manager, scheduler, and orchestration layer. Kernel-level perf levers live separately in
DESIGN.md ("Remaining levers, ranked"); this file is the capability roadmap.

## How this maps onto THIS engine (read first)

The spec is written against the paper's semantic: ONE sequential decode stream whose prefix holds
ALL pages. This engine's only decode path is page-parallel batched decode (`generate_pagepar`):
each page is an independent stream attending to ITS OWN ~278-token reference + a 128-slot R-SWA
ring. Consequences, per workstream:

- **Per-stream KV is ALREADY constant and tiny** (1 page + 128 ring ≈ 406 entries/layer/stream —
  below WS2's target C = prompt + k*256 + 128 for k=2). The 32K ceiling never binds per stream.
  What grows with document length here is the NUMBER of resident streams (VRAM + batch width),
  not any prefix. So WS1+WS2's goal (unbounded pages, flat memory) maps to **batch-windowing**:
  decode at most W pages concurrently; when a stream finishes, retire its KV and admit the next
  page (encode overlapped on the vision stream — encoder is per-page independent, and batch-shrink
  compaction via d_act already exists; this adds batch-REFILL). No RoPE surgery needed at all:
  each stream's positions are self-contained. Cheaper and exactly equivalent in memory profile.
- **WS0 (attention probe) is still worth building**, for two reasons: (a) it validates the
  page-independence assumption underlying pagepar on NEW document types (the 99.5% check was one
  doc); (b) it is the gate for running the paper-style sequential one-shot semantic with eviction
  (WS2 as written) if cross-page context ever matters (hyphenation/tables across pages — the known
  pagepar weakness). Probe = ungraphed fallback attention path (DECNOGRAPH-style) dumping per-head
  mass per page-block; must be bit-exact-off in production.
- **WS2 as written** (block-table prefix eviction inside one sequential stream) only applies if we
  re-add a sequential one-shot mode (deleted; in git history). Decision point after WS0: if some
  heads DO reach back across pages, sequential + lazy-load + evict-with-lag is the correct mode
  and pagepar keeps being the fast path for independent pages.
- **WS3 (two-pass adaptive tiling) is fully applicable and the highest-value item**: crops are
  ordinary single-image requests that feed pagepar's batch trivially. It attacks the measured
  accuracy weak spot (small/dense text where Base garbles and fp8 was vetoed) at base-mode cost,
  and its selective-crop budget beats Gundam's uniform tiling cost. Needs: per-block logprob
  export from decode (cheap: log-softmax of emitted token at block spans), garble detector,
  MuPDF region rasterizer (fz_scale to arbitrary clip already available), coordinate-keyed merge.

Priority here: WS3 > WS0 > batch-windowing (WS1/2-equivalent) > sequential+eviction mode (gated).

---

## The spec (verbatim)

Target model: `baidu/Unlimited-OCR` (arXiv:2606.23050), a 3B MoE (500M active) OCR model built on
DeepSeek OCR. Decoder uses Reference Sliding Window Attention (R-SWA): every decode token attends
to the **full visual+prompt prefix** (m tokens, fixed per document) plus a **sliding window of the
last n=128 generated tokens**. Decode-side KV cache is already constant; the **prefix is not** —
it grows ~256 tokens per page (base mode, 1024×1024) and much more in dense-page dynamic-tiling
("Gundam") mode. Max context is 32K, so the prefill ceiling caps document length today.

Execution environment: our own CUDA inference engine, already running the model correctly for
basic (full-prefill) usage. All work below happens in **our engine's KV-cache manager, scheduler,
and orchestration layer. Zero weight changes, zero training.**

Key facts the implementation relies on:
- Attention is a set operation over KV pairs. Cache entries can be inserted or removed at any time
  as long as each entry carries the correct RoPE `position_ids`. The model cannot distinguish a
  lazily-built cache from a full upfront prefill.
- Pages are encoded **independently** by the frozen DeepEncoder (no cross-page attention in the
  encoder), so per-page KV blocks are self-contained and relocatable.
- The model emits a `<page>` separator token between pages, and emits normalized block coordinates
  (0–1000) before each block's content. Both are usable as runtime control signals.
- Prior art worth skimming for the cache-surgery pattern (concepts, not dependencies):
  StreamingLLM (evict-without-renumbering + attention sinks), H2O / SnapKV (attention-guided
  eviction), paged-KV block allocators (vLLM-style block tables).

Implement the four workstreams below in order. Workstream 0 gates workstream 2.

### Workstream 0 — Attention-mass probe (validation tool, build first)

**What it adds:** Empirical evidence for whether completed pages can be safely evicted. Logs
per-layer, per-head attention mass from each decode token onto each page's visual-token block.

**Build:**
- Add a debug/probe mode to the engine's attention path that exposes per-head attention weights
  over the prefix (a slow non-fused fallback path is acceptable; this is offline tooling and must
  not touch the production kernel).
- Group prefix key indices into page blocks using the known layout (256 tokens/page in base mode
  + prompt tokens).
- Run on 5–40 page documents. Output: heatmap of attention mass per (decode-position, page-block),
  aggregated and per-head; plus a summary metric: fraction of attention mass on already-completed
  pages, as a function of pages-behind.

**How it is seen / success criterion:** If attention mass on pages ≥1 behind the current page
collapses to <1–2% for (nearly) all heads, eviction in Workstream 2 is safe with lag=1. If
specific heads keep attending backward, report which, and set the eviction lag accordingly.

### Workstream 1 — Lazy page loading (streamed prefill)

**What it adds: effectively unlimited document length.** Instead of prefilling all pages upfront
(bounded by 32K), encode only the first k pages (k=2 default), start decoding, and when the model
emits `<page>`, encode the next page with the DeepEncoder and **insert its KV into the prefix
segment with the position_ids it would have had in a full prefill**. The 32K ceiling stops binding
on total document length; it only bounds the resident window.

**Build:**
- A paged prefix-cache abstraction in the engine that supports out-of-order insertion at fixed
  position offsets. Page i's visual block always occupies positions
  `[prompt_len + i*256, prompt_len + (i+1)*256)` regardless of when it is encoded. A block-table
  indirection (page block → physical cache blocks) keeps insertion O(1) and avoids compaction.
- Trigger: the scheduler watches the decode stream for the `<page>` token id; on emission, encode
  page `current+k` asynchronously (encoder forward on a separate stream, overlapped with ongoing
  decode) and insert on completion.
- Keep prompt tokens permanently resident (they double as attention-sink anchors).

**How it is seen / metrics:**
- **Unlimited context:** demonstrate a 100+ page document parsed in one continuous session —
  impossible today (100 pages ≈ 25.6K visual tokens + output would exceed the 32K prefill+decode
  budget). Report max pages processed vs. baseline.
- **Memory:** peak GPU memory flat vs. page count once k pages resident (plot peak VRAM vs.
  document pages, baseline vs. lazy).
- **Speed:** time-to-first-token drops dramatically (prefill of k pages instead of all pages);
  report TTFT vs. page count. Steady-state TPS should be unchanged (±2%) if encoding overlaps
  decode; report any decode stalls at page boundaries.
- **Accuracy guard:** edit distance / Distinct-n on the paper's page buckets (2/5/10/20/40+) must
  match full-prefill baseline within noise. The only distribution shift is loss of look-ahead to
  future pages; quantify it.

### Workstream 2 — Page-level KV eviction (sliding reference window)

**What it adds: truly constant total KV cache** — prompt + k resident pages + 128 decode tokens —
independent of document length. Combined with Workstream 1, this is the full sliding reference
window: pages stream in ahead of the cursor and are freed behind it. Gate the default eviction lag
on Workstream 0's findings.

**Build:**
- Extend the paged prefix cache with `evict_page(i)`: return the page's physical blocks to the
  allocator pool; do NOT renumber positions of remaining entries. With a block table this is a
  table update plus a free-list push — no data movement.
- Policy: on `<page>` emission for page p, evict page p − lag (lag from probe results; default 1).
  Configurable resident-page count k and eviction lag.
- Positions stay untouched after eviction — surviving pages keep their original RoPE ids (no
  re-rotation, no shifting; the attention kernel just sees a sparser but correctly-labeled key set
  via the block table).

**How it is seen / metrics:**
- **Memory:** KV cache size is a flat line vs. decode step AND vs. page count (the paper's cache
  is flat vs. decode step only). Plot both; report the constant: `C = prompt + k*256 + 128`
  entries per layer.
- **Speed:** per-step attention cost over the prefix drops from O(total pages) to O(k pages).
  Report per-call kernel latency vs. page position (paper Figure 3 style) — should be flat AND
  lower than the paper's for long docs.
- **Concurrency:** this is where the win compounds. With per-request cache bounded and known a
  priori, the scheduler can pack far more concurrent requests per GPU and admission control
  becomes exact (no headroom reserved for cache growth). Report max concurrency and aggregate TPS
  at fixed VRAM (e.g., 80GB) for 40-page docs: baseline R-SWA (prefix grows with pages) vs.
  evicting version. Expected: multi-x improvement on long documents.
- **Accuracy guard:** same eval as WS1; additionally check cross-page consistency failures (e.g.,
  hyphenated words or tables spanning a page boundary) — these are the expected failure mode if
  lag is too small.

### Workstream 3 — Two-pass adaptive tiling pipeline (dense pages, non-uniform tiles)

**What it adds:** dense-page reading quality at base-mode memory cost, with **content-adaptive,
non-uniform tiling** — without touching the model's internal tiling. Pass 1: run base mode (1024²,
256 tokens) per page to get layout, block coordinates, and coarse text with per-block confidence
proxy (e.g., logprob, or garbage-detection heuristics). Pass 2: for dense / low-confidence blocks
only, crop the region from the original page image at high DPI, snap crop boundaries to whitespace
between detected blocks (quadtree-style subdivision only where needed), and re-run each crop as an
independent image. Merge: replace pass-1 block text with pass-2 text keyed by coordinates.

**Build:**
- Orchestration layer on top of the engine's request API; crops are just ordinary single-image
  requests.
- Selector for re-crop: block area vs. estimated text density (chars emitted per normalized area),
  plus a repetition/garble detector; expose per-block logprobs from the engine if not already
  available.
- Crop renderer: rasterize the coordinate region from the source PDF/image at a DPI chosen so the
  crop fills the model's native input resolution — variable physical zoom at fixed token budget.
  This IS the non-uniform tiling.
- Merger: coordinate-keyed splice preserving reading order from pass 1.

**How it is seen / metrics:**
- **Accuracy:** edit distance on dense document types (newspaper, exam paper, colorful textbook —
  the paper's weakest subcategories) vs. (a) base mode alone, (b) native dense-page tiling mode.
  Target: match or beat the native mode.
- **Memory/speed:** tokens processed per page (pass-1 256 + only-as-needed crops) vs. native
  tiling's full cost; report average token cost per dense page and wall-clock per page. Sparse
  pages pay zero pass-2 cost.
- **Concurrency:** crops are small independent requests — they batch trivially; report throughput
  of pass-2 crop requests under the engine's continuous batching.

### Cross-cutting requirements

- **Deliverables:** (1) paged prefix cache with streamed insertion + eviction in the engine;
  (2) probe mode + plots; (3) scheduler integration (`<page>`-triggered encode/evict, async encoder
  stream); (4) two-pass pipeline CLI; (5) a benchmark harness producing every metric above as
  CSV + plots.
- **Benchmarks:** OmniDocBench v1.5 for single-page accuracy regression; a multi-page set
  (concatenated PDFs at 2/5/10/20/40/100+ pages) for long-horizon metrics; report edit distance,
  Distinct-20/35, peak VRAM, TTFT, steady-state TPS, per-step kernel latency, max concurrency at
  fixed VRAM.
- **Correctness invariants to assert in code:** position_ids of a page block are a pure function
  of page index (never of insertion time); eviction never renumbers surviving entries; the decode
  sliding window of 128 is untouched; encoder and all model weights are byte-identical to the
  released checkpoint; probe mode is bit-exact-off in production builds.
- **Out of scope (requires training):** page-index embeddings, per-page RoPE reset, trained
  page-gated attention, in-pass coordinate-driven tile fetching, changing the n=128 decode window.

---
## WS0 RESULTS (2026-07-02, engine/tools/probe_attn.py — HF sequential decode, 5pg, 3723 steps)
Attention-mass criterion FAILS at lag=1: mass on completed pages = 17.0% mean, 69.8% worst
(layer,head); lag=3 still 11.0%/39.7% — NOT the <1-2% needed to declare eviction invisible.
BUT the structure says the backward mass is not retrieval: per-token mass bos=0.13 (dominant sink),
ring=3.7e-3 (working attention, 42-49% of total), own page=3.7e-4, every OTHER page uniformly
2.4e-4 — identical for pages behind AND AHEAD (future-page mass cannot be content lookback)
=> diffuse background + sinks. Consequences: (1) a sequential+eviction mode must keep bos (+page0,
slightly elevated as secondary sink) and use lag>=3, and even then it redistributes ~7%/page of
softmax mass — unproven; (2) the engine's page-parallel design (each stream sees ONLY its page,
measured ~0.5% output cost) is quantitatively justified: cross-page attention carries ~no content.
Outputs: outputs_probe/{probe_mass.csv,probe_summary.csv,probe_heatmap.png}.

## WS3 RESULTS (2026-07-02, engine/tools/twopass.py — v1)
Two-pass adaptive tiling works as budget middle ground between base and Gundam. 50pg brochure:
pass-1 base 60.8k tok; selector (est glyph height < 13px @1024, or garble heuristics) picked 87/924
blocks; pass-2 re-rendered each at variable zoom filling 1024 (non-uniform tiling) and ran ALL 87
crops as ONE page-parallel engine batch (12.0k tok) — crops batch trivially, as the spec predicted.
Merge gate (reject garbled/too-short crop text; measured: overlap/length-cap variants net WORSE)
replaced 56 blocks. Judged vs Gundam gold: real-word fraction on replaced blocks 0.445 -> 0.603
(51 improved / 21 regressed / 18 neutral). Token cost 1457/page vs Gundam's ~3100 visual tok/page.
Not yet Gundam parity — Gundam remains the quality mode for dense docs; two-pass is ~2x cheaper and
fixes the worst blocks. v1 gaps (future): selector should use engine logprobs (not text heuristics),
crop boundaries should snap to whitespace (6% pad still clips/over-captures), block-anchored merge.
