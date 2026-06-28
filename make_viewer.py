#!/usr/bin/env python3
"""Build an interactive OCR viewer: page PNG (left) + extracted elements (right),
hover an element to highlight its bounding box on the image.
Box coords are 0-1000 in the model's letterboxed 1024x1024 input; we invert the
aspect-pad (sc=min(1024/w,1024/h), centered) to map back onto the native page."""
import re, sys, os, html, json
import fitz  # pymupdf

PDF = sys.argv[1] if len(sys.argv) > 1 else "/home/janitor/unlimited-ocr/Unlimited-OCR.pdf"
OCR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ocr_full.txt"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__)) + "/viewer"
DPI = 150
IMG = 1024

os.makedirs(OUT, exist_ok=True)

# --- parse OCR output ---
raw = open(OCR, encoding="utf-8", errors="replace").read()
m = re.search(r"=====\s*OCR.*?=====", raw)
body = raw[m.end():] if m else raw
pages_txt = [p for p in body.split("<PAGE>") if p.strip()]
EL = re.compile(r"<\|det\|>\s*([a-z_]+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]<\|/det\|>(.*?)(?=<\|det\|>|$)", re.S)

def frac(b, W, H):
    # model emits boxes already normalized 0-1000 to the ORIGINAL page (letterbox inverted internally)
    return b[0] / 1000, b[1] / 1000, b[2] / 1000, b[3] / 1000

doc = fitz.open(PDF)
pages = []
for i, ptxt in enumerate(pages_txt):
    pg = doc[i]
    pix = pg.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    pix.save(f"{OUT}/page_{i+1}.png")
    W, H = pix.width, pix.height
    els = []
    for typ, x1, y1, x2, y2, txt in EL.findall(ptxt):
        b = [int(x1), int(y1), int(x2), int(y2)]
        L, T, R, Bo = frac(b, W, H)
        els.append({"type": typ, "box": b,
                    "l": round(L * 100, 3), "t": round(T * 100, 3),
                    "w": round((R - L) * 100, 3), "h": round((Bo - T) * 100, 3),
                    "text": txt.strip()})
    pages.append({"img": f"page_{i+1}.png", "w": W, "h": H, "els": els})
doc.close()

COLORS = {"text":"#2563eb","title":"#7c3aed","header":"#0d9488","image_caption":"#ea580c",
          "ref_text":"#64748b","page_number":"#16a34a","list":"#4f46e5","table":"#dc2626",
          "equation":"#a16207","image":"#db2777"}

DATA = json.dumps(pages, ensure_ascii=False)
COL = json.dumps(COLORS)

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unlimited-OCR viewer</title>
<style>
*{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#1e293b;background:#f1f5f9}
header{background:#0f172a;color:#e2e8f0;padding:10px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:20}
header h1{font-size:15px;margin:0;font-weight:600}
header .sub{font-size:12px;color:#94a3b8}
.pager{display:flex;gap:4px;flex-wrap:wrap;margin-left:auto}
.pager button{background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:12px}
.pager button.on{background:#3b82f6;color:#fff;border-color:#3b82f6}
.wrap{display:flex;gap:14px;padding:14px;align-items:flex-start}
.imgcol{flex:0 0 52%;position:sticky;top:62px}
.imgwrap{position:relative;display:inline-block;width:100%;border:1px solid #cbd5e1;border-radius:8px;overflow:hidden;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.imgwrap img{display:block;width:100%}
.bbox{position:absolute;border:1.5px solid transparent;border-radius:2px;pointer-events:auto;cursor:pointer;transition:background .08s,border-color .08s,box-shadow .08s}
.imgwrap.showall .bbox{border-color:var(--c);opacity:.35}
.bbox.hi{opacity:1!important;border-width:2.5px;background:var(--c2);box-shadow:0 0 0 2px #fff,0 0 14px var(--c)}
.elcol{flex:1 1 48%;min-width:0}
.el{background:#fff;border:1px solid #e2e8f0;border-left:4px solid var(--c);border-radius:7px;padding:8px 11px;margin-bottom:8px;cursor:pointer}
.el:hover,.el.hi{background:#eff6ff;border-color:#93c5fd;box-shadow:0 1px 6px rgba(37,99,235,.18)}
.el .meta{display:flex;gap:8px;align-items:center;margin-bottom:3px}
.el .tag{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#fff;background:var(--c);padding:1px 7px;border-radius:20px}
.el .coord{font-size:11px;color:#94a3b8;font-family:ui-monospace,monospace}
.el .body{white-space:pre-wrap;word-break:break-word;font-size:13px;color:#334155}
.el .body:empty:after{content:"(no text)";color:#cbd5e1;font-style:italic}
.toolbar{display:flex;gap:10px;align-items:center;font-size:12px;color:#64748b;margin:0 0 10px}
.toolbar label{display:flex;gap:5px;align-items:center;cursor:pointer}
.count{color:#94a3b8}
</style></head><body>
<header><h1>Unlimited-OCR</h1><span class="sub" id="sub"></span>
<div class="pager" id="pager"></div></header>
<div class="wrap">
  <div class="imgcol"><div class="imgwrap showall" id="imgwrap"><img id="img"><div id="overlay"></div></div></div>
  <div class="elcol">
    <div class="toolbar"><label><input type="checkbox" id="showall" checked> outline all boxes</label>
      <span class="count" id="count"></span></div>
    <div id="els"></div>
  </div>
</div>
<script>
const PAGES=__DATA__, COL=__COL__;
const imgwrap=document.getElementById('imgwrap'),img=document.getElementById('img'),
      overlay=document.getElementById('overlay'),els=document.getElementById('els'),
      pager=document.getElementById('pager'),sub=document.getElementById('sub'),count=document.getElementById('count');
function lite(hex){return hex+'33';}
let cur=0;
function render(p){
  cur=p; const pg=PAGES[p];
  img.src=pg.img; overlay.innerHTML=''; els.innerHTML='';
  sub.textContent=`page ${p+1} / ${PAGES.length} — ${pg.els.length} elements`;
  count.textContent=`${pg.els.length} elements`;
  pg.els.forEach((e,i)=>{
    const c=COL[e.type]||'#475569';
    const b=document.createElement('div'); b.className='bbox'; b.dataset.i=i;
    b.style.cssText=`left:${e.l}%;top:${e.t}%;width:${e.w}%;height:${e.h}%;--c:${c};--c2:${lite(c)}`;
    overlay.appendChild(b);
    const el=document.createElement('div'); el.className='el'; el.dataset.i=i; el.style.setProperty('--c',c);
    el.innerHTML=`<div class="meta"><span class="tag" style="background:${c}">${e.type}</span>`+
                 `<span class="coord">[${e.box.join(', ')}]</span></div>`+
                 `<div class="body"></div>`;
    el.querySelector('.body').textContent=e.text;
    els.appendChild(el);
    const on=()=>{b.classList.add('hi');el.classList.add('hi');},
          off=()=>{b.classList.remove('hi');el.classList.remove('hi');};
    el.addEventListener('mouseenter',on); el.addEventListener('mouseleave',off);
    b.addEventListener('mouseenter',()=>{on();el.scrollIntoView({block:'nearest',behavior:'smooth'});});
    b.addEventListener('mouseleave',off);
  });
  [...pager.children].forEach((x,i)=>x.classList.toggle('on',i===p));
}
PAGES.forEach((_,i)=>{const btn=document.createElement('button');btn.textContent=i+1;btn.onclick=()=>render(i);pager.appendChild(btn);});
document.getElementById('showall').addEventListener('change',e=>imgwrap.classList.toggle('showall',e.target.checked));
render(0);
</script></body></html>"""

HTML = HTML.replace("__DATA__", DATA).replace("__COL__", COL)
open(f"{OUT}/index.html", "w", encoding="utf-8").write(HTML)
ne = sum(len(p["els"]) for p in pages)
print(f"wrote {OUT}/index.html  ({len(pages)} pages, {ne} elements, PNGs at {DPI}dpi)")
