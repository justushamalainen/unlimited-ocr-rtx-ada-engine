#!/usr/bin/env python3
"""WS3 two-pass adaptive tiling pipeline (orchestration; the engine binary is unmodified).

Pass 1: engine base mode (1024^2, 273 tok/page) -> layout blocks + coarse text.
Select : blocks whose estimated glyph size is tiny, or whose text looks garbled (repetition /
         low distinct ratio) -> candidates for re-read.
Pass 2: re-render each selected block from the source PDF at a DPI that makes the crop fill the
        model's native 1024 input (variable physical zoom at fixed token budget = non-uniform
        tiling), build a temp crops.pdf, and run ALL crops through the engine in ONE page-parallel
        batch (they are ordinary single-image requests).
Merge : splice pass-2 text over pass-1 blocks, keyed by block identity, preserving reading order.

Usage: .venv/bin/python engine/tools/twopass.py <pdf> <npages> <outdir>
         [--engine engine/ocr_bin] [--glyph-px 13] [--pad 0.02] [--no-pass2]
Writes: outdir/pass1.txt, outdir/crops.pdf, outdir/pass2.txt, outdir/merged.md, outdir/stats.csv
"""
import sys, os, re, csv, subprocess, argparse, time
import pymupdf

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("npages", type=int); ap.add_argument("outdir")
ap.add_argument("--engine", default=os.path.join(os.path.dirname(__file__), "..", "ocr_bin"))
ap.add_argument("--glyph-px", type=float, default=13.0, help="re-crop blocks whose est. glyph height (px @1024) is below this")
ap.add_argument("--pad", type=float, default=0.06, help="crop margin as fraction of block size (too tight clips glyphs)")
ap.add_argument("--no-pass2", action="store_true", help="selection dry-run only")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)

DET = re.compile(r"<\|det\|>(\w+) \[(\d+), (\d+), (\d+), (\d+)\]<\|/det\|>", re.S)

def run_engine(pdf, npages, out, env=None):
    e = dict(os.environ); e.update(env or {})
    t0 = time.time()
    r = subprocess.run([A.engine, pdf, str(npages)], capture_output=True, text=True, env=e)
    open(out, "w").write(r.stdout)
    body = r.stdout[r.stdout.find("===== OCR"):]
    toks = int(re.search(r"\((\d+) page\(s\), (\d+) tokens\)", body).group(2))
    return body, toks, time.time() - t0

def parse_pages(body):
    """-> per page: list of blocks {type,(x0,y0,x1,y1) in 0-1000 padded-square coords, text}"""
    text = body.split("=====", 2)[2]
    pages = text.split("<PAGE>")[1:] if "<PAGE>" in text else [text]
    out = []
    for pg in pages:
        blocks, matches = [], list(DET.finditer(pg))
        for i, m in enumerate(matches):
            end = matches[i+1].start() if i+1 < len(matches) else len(pg)
            blocks.append({"type": m.group(1), "box": tuple(int(m.group(j)) for j in range(2, 6)),
                           "text": pg[m.end():end].strip()})
        out.append(blocks)
    return out

def pad_transform(page):
    """padded-1024-square norm coords (0-1000) -> PDF point rect"""
    w, h = page.rect.width, page.rect.height
    s = 1024.0 / max(w, h)
    offx, offy = (1024 - w*s) / 2, (1024 - h*s) / 2
    def to_pdf(box):
        x0, y0, x1, y1 = [v / 1000.0 * 1024 for v in box]
        return pymupdf.Rect((x0-offx)/s, (y0-offy)/s, (x1-offx)/s, (y1-offy)/s) & page.rect
    return to_pdf, s

def garbled(t):
    """repetition / low-diversity heuristics on the block text"""
    if len(t) < 40: return False
    words = t.split()
    if len(words) >= 8:
        bi = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        if len(set(bi)) / len(bi) < 0.45: return True                 # heavy bigram repetition
    longest = max((len(m.group(0)) for m in re.finditer(r"(.)\1{5,}", t)), default=0)
    return longest >= 8                                               # aaaaaaaa-style runs

def select(blocks, page, s):
    """est glyph height in engine pixels: block px height / text line count"""
    sel = []
    for b in blocks:
        if b["type"] not in ("text", "table", "title", "header", "footer") or not b["text"]: continue
        x0, y0, x1, y1 = b["box"]
        hpx = (y1 - y0) / 1000.0 * 1024
        wpx = (x1 - x0) / 1000.0 * 1024
        if hpx < 8 or wpx < 8: continue
        chars = len(b["text"])
        cpl = max(8.0, wpx / 7.5)                                     # rough chars/line if glyphs ~7.5px wide (readable floor)
        est_lines = max(1.0, chars / cpl)
        glyph = hpx / est_lines
        b["glyph_px"], b["garbled"] = glyph, garbled(b["text"])
        if b["garbled"] or glyph < A.glyph_px: sel.append(b)
    return sel

# ---------- pass 1 ----------
body1, tok1, wall1 = run_engine(A.pdf, A.npages, f"{A.outdir}/pass1.txt")
pages = parse_pages(body1)
doc = pymupdf.open(A.pdf)
crops, per_page_sel = [], []
for p, blocks in enumerate(pages[:A.npages]):
    to_pdf, s = pad_transform(doc[p])
    sel = select(blocks, doc[p], s)
    per_page_sel.append(sel)
    for b in sel:
        r = to_pdf(b["box"])
        m = A.pad * max(r.width, r.height)
        r = pymupdf.Rect(r.x0-m, r.y0-m, r.x1+m, r.y1+m) & doc[p].rect
        if r.width > 4 and r.height > 4: crops.append((p, b, r))
nsel = len(crops)
print(f"pass1: {tok1} tok, {wall1:.1f}s, {sum(len(b) for b in pages)} blocks, {nsel} selected for re-read")
if A.no_pass2 or nsel == 0:
    open(f"{A.outdir}/merged.md","w").write("\n".join("\n".join(f"[{b['type']}] {b['text']}" for b in pg) for pg in pages))
    sys.exit(0)

# ---------- pass 2: crops.pdf, one page per crop, rendered so the crop fills 1024 ----------
cd = pymupdf.open()
for (p, b, r) in crops:
    z = 1024.0 / max(r.width, r.height)                               # variable zoom, fixed token budget
    pix = doc[p].get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r)
    pg = cd.new_page(width=pix.width, height=pix.height)
    pg.insert_image(pg.rect, pixmap=pix)
cd.save(f"{A.outdir}/crops.pdf")
body2, tok2, wall2 = run_engine(f"{A.outdir}/crops.pdf", nsel, f"{A.outdir}/pass2.txt")
cpages = parse_pages(body2)
print(f"pass2: {nsel} crops in ONE batched run, {tok2} tok, {wall2:.1f}s")

# ---------- merge: coordinate-keyed splice, reading order from pass 1 ----------
def strip_dets(blocks): return " ".join(b["text"] for b in blocks if b["text"])
n_rej = 0
for i, (p, b, r) in enumerate(crops):
    if i < len(cpages) and cpages[i]:
        nt = strip_dets(cpages[i])
        # quality gate (runtime signals only): reject crop text that is itself garbled or implausibly short.
        # Measured on the 50pg brochure vs Gundam gold: this gate nets +30 blocks (51 improved / 21 regressed);
        # adding a 3x length cap or overlap-with-original tests LOSES more good recoveries than regressions
        # they catch (garbled originals legitimately expand a lot; low overlap IS the fix, not a defect).
        if nt and not garbled(nt) and len(nt) >= 0.5*len(b["text"]): b["text2"] = nt
        elif nt: n_rej += 1
print(f"merge gate rejected {n_rej} regression crops (kept pass-1 text)")
merged, n_replaced = [], 0
for p, blocks in enumerate(pages[:A.npages]):
    merged.append(f"\n## page {p+1}\n")
    for b in blocks:
        t = b.get("text2", b["text"]); n_replaced += "text2" in b
        merged.append(f"[{b['type']}] {t}")
open(f"{A.outdir}/merged.md","w").write("\n".join(merged))
with open(f"{A.outdir}/stats.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["pass","tokens","wall_s","pages_or_crops"])
    w.writerow(["base", tok1, f"{wall1:.1f}", A.npages]); w.writerow(["crops", tok2, f"{wall2:.1f}", nsel])
    w.writerow(["total", tok1+tok2, f"{wall1+wall2:.1f}", ""])
print(f"merged: {n_replaced}/{nsel} blocks replaced -> {A.outdir}/merged.md")
print(f"token cost: base {tok1} + crops {tok2} = {tok1+tok2} ({(tok1+tok2)/A.npages:.0f}/page)")
