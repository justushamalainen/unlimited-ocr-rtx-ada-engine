#!/usr/bin/env python3
"""Benchmark harness (ROADMAP cross-cutting deliverable): runs the engine over a workload matrix and
emits every long-horizon metric as CSV + plots.

Metrics per run: TTFT, prefill ms, decode tok / ms / tok/s, steps, per-step ms, batch util,
peak process VRAM (sampled), det elements, 14pg md5 gate check, admissions (windowed runs).
Derived plots: peak VRAM vs pages (windowed vs all-resident), TTFT vs pages, tok/s vs pages,
tok/s + VRAM vs WINDOW at fixed 112 pages, per-step latency vs resident batch.

Usage: .venv/bin/python engine/tools/bench.py [outdir=outputs_bench]   (GPU must be idle)
"""
import subprocess, os, re, csv, sys, time, threading

ROOT   = "/home/janitor/unlimited-ocr"
ENGINE = f"{ROOT}/engine/ocr_bin"
PAPER  = f"{ROOT}/Unlimited-OCR.pdf"
X8     = f"{ROOT}/testdata/paper_x8.pdf"
BRO    = f"{ROOT}/testdata/reaktor_mkt.pdf"
OUT    = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/outputs_bench"
MD5_14 = "af3a8ae8e348d6b2104b3544363b4f37"
os.makedirs(OUT, exist_ok=True)

RUNS = [  # (tag, pdf, npages, WINDOW or None=default128, gundam)
    ("paper2",   PAPER, 2,   None, False), ("paper5",  PAPER, 5,  None, False), ("paper14", PAPER, 14, None, False),
    ("x8_28",    X8,    28,  None, False), ("x8_56",   X8,   56,  None, False), ("x8_112",  X8,  112, None, False),
    ("x8_112w64",X8,    112, 64,   False), ("x8_112w32",X8, 112,  32,   False), ("x8_112w16",X8, 112, 16,  False),
    ("gundam50", BRO,   50,  None, True),
]

def vram_sampler(pid, stop, peak):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout
            for ln in o.splitlines():
                p, m = [x.strip() for x in ln.split(",")]
                if int(p) == pid: peak[0] = max(peak[0], int(m))
        except Exception: pass
        time.sleep(0.3)

def run(tag, pdf, n, window, gundam):
    env = dict(os.environ)
    if window: env["WINDOW"] = str(window)
    if gundam: env["GUNDAM"] = "1"
    t0 = time.time()
    proc = subprocess.Popen([ENGINE, pdf, str(n)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    stop, peak = threading.Event(), [0]
    th = threading.Thread(target=vram_sampler, args=(proc.pid, stop, peak)); th.start()
    out, _ = proc.communicate()
    stop.set(); th.join()
    wall = time.time() - t0
    r = {"tag": tag, "pages": n, "window": window or 128, "gundam": int(gundam), "wall_s": f"{wall:.1f}", "peak_vram_mib": peak[0]}
    m = re.search(r"TTFT \(page 0 .*?\): (\d+) ms", out);                       r["ttft_ms"] = m.group(1) if m else ""
    m = re.search(r"prefill.*?: (\d+)/(\d+) pages in (\d+) ms", out);           r["prefill_ms"] = m.group(3) if m else ""
    m = re.search(r"decode: (\d+) tok in (\d+) ms \((\d+) tok/s\), (\d+) steps, (\d+)% batch util", out)
    if m:
        r.update(tok=m.group(1), decode_ms=m.group(2), tok_s=m.group(3), steps=m.group(4), util_pct=m.group(5))
        r["step_ms"] = f"{int(m.group(2))/int(m.group(4)):.2f}"
    m = re.search(r"windowed: .*? (\d+) admissions (\d+) ms", out);             r["admissions"] = m.group(1) if m else ""
    body = out[out.find("===== OCR"):]
    r["det_els"] = body.count("<|det|>")
    if tag == "paper14":
        import hashlib; r["md5_gate"] = "PASS" if hashlib.md5(body.encode()).hexdigest() == MD5_14 else "FAIL"
    else: r["md5_gate"] = ""
    print(f"{tag:11s} pages={n:3d} W={r['window']:3d} tok/s={r.get('tok_s','?'):>6s} ttft={r['ttft_ms']:>5s}ms "
          f"vram={peak[0]:5d}MiB step={r.get('step_ms','?'):>5s}ms util={r.get('util_pct','?')}% {r['md5_gate']}")
    return r

rows = [run(*a) for a in RUNS]
cols = ["tag","pages","window","gundam","ttft_ms","prefill_ms","tok","decode_ms","tok_s","steps","step_ms",
        "util_pct","admissions","det_els","peak_vram_mib","wall_s","md5_gate"]
with open(f"{OUT}/bench.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k,"") for k in cols})
print(f"-> {OUT}/bench.csv")

# analytic max-concurrency at fixed VRAM (spec metric): slots = (budget - base) / per-slot KV
KVB = 1  # fp8 KV byte
for name, ms in (("base", 278+128), ("gundam_landscape", 3110+128)):
    per_slot = ms*1280*KVB*2*12/2**20
    for budget in (48, 80):
        base_gb = 10.5
        print(f"max concurrent {name} streams @ {budget}GB: {int((budget-base_gb)*1024/per_slot)} (KV {per_slot:.1f} MiB/slot)")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    B = [r for r in rows if not r["gundam"] and (r["window"] == 128)]
    Wv = sorted([r for r in rows if r["tag"].startswith("x8_112")], key=lambda r: r["window"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    pg = [r["pages"] for r in B]
    ax[0][0].plot(pg, [r["peak_vram_mib"]/1024 for r in B], "o-", label="W=128 (default)")
    ax[0][0].plot([112]*len(Wv), [r["peak_vram_mib"]/1024 for r in Wv], "s", label="112pg, W sweep")
    for r in Wv: ax[0][0].annotate(f"W={r['window']}", (112, r["peak_vram_mib"]/1024), fontsize=7, xytext=(3,3), textcoords="offset points")
    ax[0][0].set_xlabel("pages"); ax[0][0].set_ylabel("peak VRAM (GiB)"); ax[0][0].set_title("VRAM is window-bound, not page-bound"); ax[0][0].legend(); ax[0][0].set_ylim(0)
    ax[0][1].plot(pg, [int(r["ttft_ms"]) for r in B if r["ttft_ms"]], "o-")
    ax[0][1].set_xlabel("pages"); ax[0][1].set_ylabel("TTFT (ms)"); ax[0][1].set_title("TTFT flat vs page count (page-0 only)"); ax[0][1].set_ylim(0)
    ax[1][0].plot(pg, [int(r["tok_s"]) for r in B], "o-")
    ax[1][0].set_xlabel("pages"); ax[1][0].set_ylabel("decode tok/s"); ax[1][0].set_title("throughput scales with resident batch"); ax[1][0].set_ylim(0)
    ax[1][1].plot([r["window"] for r in Wv], [int(r["tok_s"]) for r in Wv], "o-", label="tok/s")
    ax2 = ax[1][1].twinx(); ax2.plot([r["window"] for r in Wv], [r["peak_vram_mib"]/1024 for r in Wv], "s--", color="gray", label="VRAM")
    ax[1][1].set_xlabel("WINDOW (112 pages)"); ax[1][1].set_ylabel("tok/s"); ax2.set_ylabel("VRAM (GiB)"); ax[1][1].set_title("memory/throughput trade at fixed 112 pages")
    fig.tight_layout(); fig.savefig(f"{OUT}/bench.png", dpi=120)
    print(f"-> {OUT}/bench.png")
except Exception as e: print("plots skipped:", e)
