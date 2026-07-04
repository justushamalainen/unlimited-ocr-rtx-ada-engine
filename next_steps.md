# next_steps.md — Fully heterogeneous decode slots — SHIPPED 2026-07-03

The plan below was executed end-to-end in one pass (Stage A folded into Stage B: with per-slot
`pf[]`/`kvoff[]` landing first and gating bit-exact, cross-doc bucketing became unnecessary — all
shapes co-batch directly). Full record: DESIGN.md "Heterogeneous decode windows" (2026-07-03).

## What shipped
- **B1/B2** — per-slot `d_pf[]` + `d_kvoff[]` (two-zone KV pool per user direction: W fixed base-size
  slot regions + G big regions sized `GVPP_MAX=3793`, one allocation/layer, constant offset table).
  In-place scalar->array-load swap in `k_rope_store_b`/`k_attn_split_b`; FP expression shape kept.
  SASS FP sequences shifted (integer scheduling only) — all numeric gates bit-exact, incl. gundam-50.
- **B3** — heterogeneous admission: `PageSrc::next(base_ok,big_ok)/vpp_of/vpp_gmax`; per-page prefill
  size; graphs cached per (NA, ns); QueueSrc expands gundam jobs into per-page items (dims-only ntok);
  gundam interludes + `srv_take_gundam`/`srv_reenqueue_front` DELETED; auto hi-res = gundam items in the
  same rotation. `ocr_gundam` mixed-tiling docs = ONE windowed pass (sequential fallback deleted).
- **B4** — VRAM bound: big-slot count from free VRAM, capped by `GSLOTS` (default 32, capacity knob).
- **Scheduling (beyond the plan)** — no HOL blocking: disjoint slot classes; big admissions capped
  2/boundary; base-only booster sub-window (+48 steps) when classes mix. 1pg base ~8s DURING a 50pg
  gundam job (was 35s+ behind the interlude; idle 0.8s).
- **Per-slot key-splits (beyond the plan)** — `k_attn_split_b` derives its split count from
  `pf[act[s]]` (padding splits NEUTRAL; merge padding is exact +0.0f -> bit-equal to a small launch).
  Uniform windows: nss==ns, bit-identical. Fixes both the plan's "ns sized to PF_max wastes splits"
  risk AND the mixed-window near-tie amplifier (idle-vs-loaded agreement 0.93 -> 0.9987).

## Verification (all green 2026-07-03/04)
Base1/gundam1/base14 md5 BIT-EXACT; gundam-50 BIT-IDENTICAL (WINDOW=48 pinned); make check/lint PASS;
servercheck PASS incl. new stanzas: co-batch agreement >=0.99, no-HOL latency, mixed-tiling one-pass;
mixed-tiling determinism (2 runs byte-identical — positions stay pure functions of the page).
Sanitizers: base 4x CLEAN; mixed-tiling memcheck+initcheck+synccheck CLEAN; racecheck on gundam-sized
prefills = compute-sanitizer host-shadow blowup (126GB RSS, OOM-killed the box 07-02 AND 07-03 — NOT an
engine bug; 0 hazards before abort; sanitize.sh now ulimit-guards it, see ULIM). Perf regression: NONE —
112pg base 18,306 tok/s / TTFT 356ms / 12.4s e2e; gundam-50 34.4s (matches pre-change references).
Assets: testdata/mixed_ratio.pdf.

## Async admission — SHIPPED 2026-07-04 (DESIGN.md "Async big-ref admission")
Encode launches on VS (gundam_encode_begin + event) during the decode window; prefill runs on a second
stream with a pointer-swapped PfCtx (scratch + gm pools + cuBLAS handle) — bit-identical (idle gundam
parity byte-exact through the async path). All loop copies stream-scoped (legacy-stream cudaMemcpy
drains other streams — the overlap killer). Base admission stays sync (parity trajectory). Measured:
base-under-gundam 8.2s -> 2.3s (GSLOTS=8) / 4.5s (GSLOTS=32); gundam-50 36.6s, 112pg 18.3k tok/s — no
regressions. Gotchas fixed en route: NA==0 must pump before srv_wait_work (auto-hires items live in the
rotation, not the HTTP queue -> deadlock); encoded ref must stage at ROW 1 of the embeds buffer (direct
gundam_result() shifted all visual tokens one row — decode still "worked" at conf 0.92; only the
byte-parity gate caught it); vision CK() macro shadows `e` (CK(e) self-inits).

## Remaining headroom (not built)
- Encode pipeline depth 2 (staging buffer for the assembled ref) — admission rate is now bounded by
  one encode per boundary; depth 2 would decouple render(k+1) from prefill(k). Modest.
- Gundam render prefetch (mirror enc_page's CPU-render-during-GPU-encode) — hides ~150ms MuPDF render.
- Base admissions during an in-flight gundam encode wait <=~200ms in vis_gpu_sync (shared VS); a second
  vision buffer set would remove it. Bounded and rare; likely not worth it.
