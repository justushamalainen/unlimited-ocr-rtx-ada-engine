"""Dump ground-truth fixtures from the HF reference model for engine verification:
  - input token ids (text-only; the decoder math is identical regardless of image vs text)
  - per-layer hidden states (output_hidden_states) for layer-by-layer checks
  - final logits at the last position
  - greedy-generated token ids (full-attention decode reference)
All saved as little-endian fp32 / int32 binaries under engine/fixture/.
"""
import os, struct, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

MD = "/home/janitor/unlimited-ocr/model"
FX = "/home/janitor/unlimited-ocr/engine/fixture"
os.makedirs(FX, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model = AutoModel.from_pretrained(MD, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16, attn_implementation="eager").eval().cuda()

ids = tok("Recently, end-to-end OCR models, exemplified by DeepSeek OCR, have once again",
          return_tensors="pt").input_ids.cuda()
S = ids.shape[1]
print("seq len:", S, "ids:", ids[0].tolist())

def dump(name, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32) if arr.dtype != np.int32 else arr)
    arr.tofile(os.path.join(FX, name))

with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple len L+1: [embeds, after L0, ..., after L11]
    logits = out.logits[0]  # [S, V]
    print("num hidden_states:", len(hs), "logits:", tuple(logits.shape))
    dump("ids.i32", ids[0].to(torch.int32).cpu().numpy())
    for i, h in enumerate(hs):
        dump(f"hs_{i}.f32", h[0].float().cpu().numpy())   # [S,H]
    dump("logits_last.f32", logits[-1].float().cpu().numpy())  # [V]
    np.array([S, len(hs), logits.shape[-1]], dtype=np.int32).tofile(os.path.join(FX, "meta.i32"))

    # greedy generate (full-attention; reference token stream)
    gen = ids.clone()
    past = None
    cur = ids
    gentoks = []
    for step in range(40):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nt = int(o.logits[0, -1].argmax())
        gentoks.append(nt)
        cur = torch.tensor([[nt]], device=ids.device)
    np.array(gentoks, dtype=np.int32).tofile(os.path.join(FX, "gen.i32"))
    print("ref gen tokens:", gentoks[:12], "...")
    print("decoded:", repr(tok.decode(gentoks)))
print("fixtures written to", FX)
