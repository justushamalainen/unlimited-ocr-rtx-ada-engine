#!/usr/bin/env python3
"""WS0 attention-mass probe (offline tooling; the engine itself is untouched).

Runs the HF reference implementation of baidu/Unlimited-OCR in true SEQUENTIAL multi-page decode
(one shared prefix over all pages) and measures, at every decode step, how much R-SWA attention
mass lands on each page's visual block. This answers the ROADMAP WS0 question — "can completed
pages be evicted?" — and doubles as evidence for the engine's page-parallel design (which gives
each page-stream ONLY its own page).

Usage:  .venv/bin/python engine/tools/probe_attn.py [pdf] [npages] [maxtok] [outdir]
Output: outdir/probe_mass.csv   (step, cur_page, layer, head, target_page, mass)  [aggregated]
        outdir/probe_summary.csv (per (layer,head): mass on cur page block, pages behind >=1, ring, prompt+bos)
        outdir/probe_heatmap.png (decode-page x target-page, mean over layers/heads)
Prereq: GPU. Model weights at /home/janitor/unlimited-ocr/model. venv: /home/janitor/unlimited-ocr/.venv
"""
import sys, os, math, csv
import torch
import pymupdf
from PIL import Image, ImageOps
import torchvision.transforms as T
import numpy as np

ROOT = "/home/janitor/unlimited-ocr"
PDF   = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/Unlimited-OCR.pdf"
NPAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAXTOK= int(sys.argv[3]) if len(sys.argv) > 3 else 1500
OUT   = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/outputs_probe"
os.makedirs(OUT, exist_ok=True)

VPP, RING, PAGE_SEQ = 273, 128, [100855, 16412, 32]   # tokens/page block, R-SWA window, '<PAGE>' BPE ids

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{ROOT}/model", trust_remote_code=True)
model = AutoModel.from_pretrained(f"{ROOT}/model", trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16, attn_implementation="eager").eval().cuda()

# --- hook: SlidingWindowLlamaAttention returns None for weights; patch forward to stash the fp32 softmax ---
SW = type(model.model.layers[0].self_attn)
_orig_sdp = torch.nn.functional.softmax
PROBE = {"on": False, "step": []}     # step: list over layers of [heads, ctx] fp32 (decode q_len==1 only)
_orig_fwd = SW.forward
def _patched(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
             output_attentions=False, use_cache=False, **kw):
    # wrap softmax for the duration of this call to capture the last attention distribution
    cap = {}
    def soft(x, dim=-1, dtype=None):
        r = _orig_sdp(x, dim=dim, dtype=dtype)
        if PROBE["on"] and x.dim() == 4 and x.shape[2] == 1:   # decode: [1, heads, 1, ctx]
            cap["w"] = r.detach().float().squeeze(0).squeeze(1).cpu()   # [heads, ctx]
        return r
    torch.nn.functional.softmax = soft
    try:
        out = _orig_fwd(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions,
                        use_cache=use_cache, **kw)
    finally:
        torch.nn.functional.softmax = _orig_sdp
    if PROBE["on"] and "w" in cap: PROBE["step"].append(cap["w"])
    return out
SW.forward = _patched

# --- build the multi-page base-mode inputs exactly like infer_multi ---
doc = pymupdf.open(PDF)
imgs = []
for p in range(NPAGE):
    pix = doc[p].get_pixmap(dpi=300)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    imgs.append(ImageOps.pad(im, (1024, 1024), color=(127, 127, 127)))
norm = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
images_ori = torch.stack([norm(im) for im in imgs]).to(torch.bfloat16).cuda()          # [N,3,1024,1024]
dummy_crop = torch.zeros((1, 3, 1024, 1024), dtype=torch.bfloat16).cuda()
IMG_TOK = 128815
page_block = ([IMG_TOK]*16 + [IMG_TOK])*16 + [IMG_TOK]                                 # 273 slots
prompt_ids = [37460, 4366, 76466, 16]                                                  # 'Multi page parsing.'
ids = [0] + page_block*NPAGE + prompt_ids
S = len(ids)
assert S == 273*NPAGE + 5
input_ids = torch.tensor([ids], device="cuda")
seq_mask = torch.tensor([[t == IMG_TOK for t in ids]], device="cuda")
crop_flags = [(1, 1)]*NPAGE
print(f"probe: {NPAGE} pages, prefix S={S}, maxtok={MAXTOK}")

model.config._ring_window = RING
saved_sw = model.config.sliding_window
model.config.sliding_window = None

def blocks_of(ctx_len):
    """index ranges: ('bos',0,1), ('page',i,lo,hi)..., ('prompt',S-4,S), ('ring',S,ctx)"""
    b = [("bos", 0, 1)]
    for i in range(NPAGE): b.append((f"page{i}", 1+273*i, 1+273*(i+1)))
    b.append(("prompt", S-4, S)); b.append(("ring", S, ctx_len))
    return b

# --- prefill ---
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(input_ids=input_ids, images=[(dummy_crop, images_ori)], images_seq_mask=seq_mask,
              images_spatial_crop=torch.tensor([crop_flags]), use_cache=True)
past = o.past_key_values
tok_id = int(o.logits[0, -1].argmax())

# --- decode with per-step capture; accumulate mass per (cur_page, layer, head, block) ---
NL, NHD = len(model.model.layers), model.config.num_attention_heads
acc  = np.zeros((NPAGE, NL, NHD, NPAGE+3), dtype=np.float64)   # [...,0:N)=pages, N=bos, N+1=prompt, N+2=ring
cnt  = np.zeros(NPAGE, dtype=np.int64)
hist, cur_page, out_ids = [], -1, []                     # <PAGE> is emitted at the START of each page: first match -> page 0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for step in range(MAXTOK):
        if tok_id == 1: break
        out_ids.append(tok_id); hist.append(tok_id)
        if hist[-3:] == PAGE_SEQ: cur_page = min(cur_page+1, NPAGE-1)
        PROBE["on"], PROBE["step"] = True, []
        o = model(input_ids=torch.tensor([[tok_id]], device="cuda"), past_key_values=past,
                  use_cache=True, position_ids=torch.tensor([[S + step]], device="cuda"))
        PROBE["on"] = False
        past = o.past_key_values
        if len(PROBE["step"]) == NL and 0 <= cur_page < NPAGE:
            for l, w in enumerate(PROBE["step"]):                  # w: [heads, ctx]
                ctx = w.shape[1]; wnp = w.numpy()
                for i in range(NPAGE): acc[cur_page, l, :, i]   += wnp[:, 1+273*i:1+273*(i+1)].sum(1)
                acc[cur_page, l, :, NPAGE]   += wnp[:, 0]          # bos
                acc[cur_page, l, :, NPAGE+1] += wnp[:, S-4:S].sum(1)
                if ctx > S: acc[cur_page, l, :, NPAGE+2] += wnp[:, S:].sum(1)
            cnt[cur_page] += 1
        tok_id = int(o.logits[0, -1].argmax())
model.config.sliding_window = saved_sw
print(f"decoded {len(out_ids)} tokens, reached page {cur_page}, steps per page: {cnt.tolist()}")

# --- outputs ---
mean = acc / np.maximum(cnt, 1)[:, None, None, None]               # per-step mean mass
with open(f"{OUT}/probe_mass.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cur_page","layer","head","target","mass"])
    names = [f"page{i}" for i in range(NPAGE)] + ["bos","prompt","ring"]
    for cp in range(NPAGE):
        if cnt[cp]==0: continue
        for l in range(NL):
            for h in range(NHD):
                for t in range(NPAGE+3): w.writerow([cp,l,h,names[t],f"{mean[cp,l,h,t]:.6f}"])
# summary: worst-case (max over layer,head) mass on pages >= lag behind, per lag
with open(f"{OUT}/probe_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lag","mean_mass_behind","max_mass_behind_layer_head"])
    for lag in (1,2,3):
        m, mx = [], 0.0
        for cp in range(lag, NPAGE):
            if cnt[cp]==0: continue
            behind = mean[cp,:,:,:cp-lag+1].sum(-1)                # mass on pages <= cp-lag
            m.append(behind.mean()); mx = max(mx, float(behind.max()))
        if m: w.writerow([lag, f"{np.mean(m):.6f}", f"{mx:.6f}"])
print(open(f"{OUT}/probe_summary.csv").read())
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    hm = mean[:,:,:,:NPAGE].mean(axis=(1,2))                       # [cur_page, target_page]
    plt.figure(figsize=(6,5)); plt.imshow(hm, cmap="viridis", vmin=0)
    plt.colorbar(label="mean attention mass"); plt.xlabel("target page block"); plt.ylabel("decoding page")
    plt.title(f"R-SWA cross-page attention mass ({NPAGE}pg)"); plt.savefig(f"{OUT}/probe_heatmap.png", dpi=120)
    print(f"heatmap -> {OUT}/probe_heatmap.png")
except Exception as e: print("plot skipped:", e)
