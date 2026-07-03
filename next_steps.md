# next_steps.md — Fully heterogeneous decode slots (mixed-shape windows)

Goal: let one decode window hold page-streams with **different reference sizes** (base 273-token
refs, and every Gundam tiling — 903, 3113, …) at the same time, each slot starting, stepping, and
retiring independently. This removes the last serialization boundary in the engine: today base and
each Gundam shape run as separate exclusive passes; the endpoint is a single continuous loop where
`documents are a grouping of pages` holds for *all* shapes, not just same-vpp ones.

This is the natural generalization of windowed admission. Windowing already made the per-slot **step
counter** independent (`d_steps[]`); this makes the per-slot **reference length** independent too.

---

## Why it's possible (the enabling fact)

Walk one decode step (`body` lambda in `generate_pagepar_core`, engine.cu):

    k_embed → k_rmsnorm → QKV proj → k_rope_store_b → k_attn_split_b → k_attn_merge_b
            → o proj → k_add_rmsnorm → mlp_block_b (MoE) → lm_head → ngram → argmax → record

Every kernel here operates on the **current token's hidden state alone** — EXCEPT the three that read
the reference KV:

- `k_rope_store_b`  — writes the current K/V into the slot's ring; uses `pf` for RoPE position + ring base.
- `k_attn_split_b`  — the flash-attention key/value loop; `clen = pf + min(step, WIN)`.
- `k_attn_merge_b`  — merges the key-splits (shape-agnostic except via `ns`).

MoE, projections, lm_head, argmax do not depend on reference length at all. So "mixed shapes in one
window" reduces to: **make `pf` per-slot** (a device array indexed like `d_steps[]`) and give the KV
cache a per-slot stride. Nothing else is coupled. There is no theoretical barrier — the uniform-shape
constraint is a memory-layout + one-scalar simplification, not a law.

---

## Current constraint (what forces uniform shape today)

- `pf` (`PF = 1 + vpp + 4`) is a **scalar** kernel argument; `MS = PF + WIN` is uniform.
- `kcb[l]` / `vcb[l]` are rectangular `[W, MS, H]` buffers — slot `s` at offset `s*MS*H`, one stride.
- The per-active-count captured graph bakes the attention grid (`ns`, sized from the single `PF+WIN`).
- Gundam runs its own `ocr_gundam()` → `generate_pagepar` call, one shape per call.

---

## Ordering (lowest-risk-first; each stage ships and gates independently)

### Stage A — Cross-document shape bucketing (NO kernel change) ← do first
Group pending Gundam jobs by vpp; run each vpp-bucket as **one** windowed pass (the existing uniform
path), routing each decoded page back to its owning job via the per-page job pointer `QueueSrc`
already uses for base. Collapses N same-shape Gundam jobs from N exclusive passes to 1.

- Files: `engine.cu` (`engine_serve` / a new `run_gundam_bucket` that drains all queued Gundam jobs,
  buckets by `gundam_page_ntok`, one `ocr_gundam`-style windowed call per bucket with a multi-job
  page source), `server.cpp`/`server.h` (`srv_take_all_gundam`).
- Value: captures most of the concurrency win with **zero** risk to bit-verified kernels.
- De-risks the multi-job routing/lifecycle *before* touching attention. Prerequisite in practice.

### Stage B — Per-slot reference length (the heterogeneous window itself)
Only start once Stage A's routing is proven. Sub-steps, in order:

- **B1 — Data structures.** Replace scalar `pf` with `int pf[W]` (device), `clen` derived per slot.
  KV layout: **start with max-padded** — allocate `[W, MS_max, H]` where `MS_max = PF_max + WIN`, one
  stride, small slots waste the tail. Simple and correctness-first. (Jagged/CSR layout with a per-slot
  base-offset array is a later memory optimization — B-optional, only if padding waste bites.)
- **B2 — Kernels (the bit-sensitive change).** Make `k_rope_store_b`, `k_attn_split_b`, `k_attn_merge_b`
  read `pf[act[s]]` instead of the scalar. **Follow the codegen lesson**: swap the scalar for the array
  load *in place*, do not hoist or reorder — a syntactic change here made ptxas reschedule FP ops and
  flipped near-ties on Gundam's long accumulation chains (base gates missed it; only gundam-50 caught
  it). `ns` (attention split count, attention grid z-dim) must be sized from `PF_max + WIN`; per-slot
  short streams already write NEUTRAL for their empty splits — reuse that path, no new logic.
- **B3 — Admission + graph.** `generate_pagepar_core` admits slots with heterogeneous vpp (each slot
  seeds its own `pf[slot]` + PF-sized reference); one encode callback per shape or a shape-tagged
  source. With `pf` now a *device array read at runtime*, the per-NA captured graph stays valid across
  shapes (grid dims constant at `ns_max`), so no per-shape graph explosion.
- **B4 — Memory bound.** Generalize `vram_wcap` to a per-slot KV budget (sum of admitted slots'
  `MS[slot]`, not `W * MS_uniform`); admission stops when the running KV sum would exceed the budget.
  Keeps VRAM bounded with a heterogeneous mix.

Endpoint: delete the separate Gundam interlude — base + all shapes flow through one windowed loop.

---

## Verification

Split into three buckets; a stage ships only when all three pass.

### 1. Regression — the uniform path must stay BIT-EXACT
The existing gates all run uniform shapes; they are the guard that Stage B's kernel edits didn't perturb
numerics:
- `base 1pg` `84c6420e…`, `gundam 1pg` `d4c62427…`, `base 14pg` `af3a8ae8…` md5 — bit-exact.
- **gundam-50** (`../testdata/reaktor_mkt.pdf`, 3.1k-key chains) bit-identical — the sensitive gate that
  base md5s miss. This is the one that catches ptxas FP-rescheduling.
- `cuobjdump -sass` FP-opcode sequence of `k_attn_split_b` / `k_rope_store_b` / `k_attn_merge_b`
  identical old-vs-new (diff the normalized SASS; a per-shape gate baseline binary is stashable).
- `make check` + all four `compute-sanitizer` tools clean.

### 2. New capability — mixed-shape correctness (TOLERANCE, not bit-exact)
There is no existing mixed-shape reference, so build one. Because co-batch composition changes cuBLAS
reduction shapes, a page's output in a mixed window is NOT bit-identical to the same page decoded alone —
it drifts within the **already-accepted greedy near-tie tolerance** (same class as base multi-doc
co-batching, DESIGN.md "Determinism contract"). So the check is agreement, not equality:
- Decode `{one base page + one Gundam page}` in a mixed window; compare each page's tokens to that page
  decoded in a uniform window. Require ≥99% token agreement per page (base multi-doc is ~99.5–99.9%).
- Determinism: two identical mixed-window runs produce identical output (per-slot positions are a pure
  function of the page, never of neighbours — assert this invariant holds under mixing).
- A `servercheck` stanza: two DIFFERENT-shape Gundam docs submitted concurrently complete in **one**
  window (not two interludes) — assert via the engine log / a single decode-pass marker.

### 3. Safety — memory + no starvation
- VRAM stays under budget for an adversarial mix (many large-ref slots) — `cudaMemGetInfo` sampled
  during a mixed-window soak; `vram_wcap` per-slot budget respected, no OOM/exit(1).
- Attention efficiency: `ns` sized to `PF_max` means small-ref slots waste split work — measure the
  tok/s hit vs uniform; acceptable if within a few %, else revisit `ns` per-slot.
- No admission starvation of small-ref pages behind large-ref ones (fairness of the per-slot budget).

---

## Risks & gotchas (read before touching B2)

- **The codegen lesson is the whole game.** `k_attn_split_b` is the most bit-sensitive kernel in the
  engine. Keep the exact expression shape when swapping scalar `pf` → `pf[act[s]]`; verify the SASS
  FP-opcode sequence is identical on the uniform case and gate on gundam-50, not just the fast gates.
- **Padding waste** (B1 max-padded) can be large if base (273) and 32-tile Gundam (3513) share a window;
  bucket-by-size admission or the jagged layout (B-optional) mitigates it. Correctness first, then optimize.
- **Attention load-balance** goes lumpy with mixed `clen`; a perf cost, not correctness — tune `ns` after
  correctness is proven, don't co-mingle the changes.
- **Accepted nondeterminism widens**: mixing shapes is more co-batch variation, same numeric class as
  today's windowed NA variation — get explicit sign-off that the tolerance check (not bit-exact) is the
  contract for the new capability, exactly as base multi-doc co-batching was signed off.

## Out of scope
- Cross-page shared context / eviction (refuted by WS0, DESIGN.md) — pages stay independent.
- Any change to weights, the 128-token decode window, or per-page RoPE reset (training-required).
