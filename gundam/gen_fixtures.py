#!/usr/bin/env python3
# Regenerate Gundam runtime fixtures from model weights (bicubic-interpolated pos embeds + rel-pos tables).
# Run from repo root with the model venv: python engine/gundam/gen_fixtures.py
import torch, torch.nn.functional as F, math, numpy as np
from safetensors import safe_open
f=safe_open("model/model-00001-of-000001.safetensors","pt"); GD="engine/gundam/"
def interp_bicubic(x,size): return F.interpolate(x.to(torch.float32),size=size,mode='bicubic',antialias=True,align_corners=False)
# SAM pos embed 64x64 -> 40x40  [1600,768]
sp=f.get_tensor("model.sam_model.pos_embed")                       # [1,64,64,768] bf16
s40=interp_bicubic(sp.permute(0,3,1,2),(40,40)).to(torch.bfloat16).permute(0,2,3,1).reshape(1600,768)
s40.to(torch.float32).numpy().astype(np.float32).tofile(GD+"sam_pos40.bin")
# CLIP pos embed 16x16(+cls) -> 10x10(+cls)  [101,1024]
cp=f.get_tensor("model.vision_model.embeddings.position_embedding.weight")  # [257,1024]
cls,old=cp[:1],cp[1:]; c10=interp_bicubic(old.view(1,16,16,1024).permute(0,3,1,2),(10,10)).to(torch.bfloat16).permute(0,2,3,1).reshape(100,1024)
torch.cat([cls,c10],0).to(torch.float32).numpy().astype(np.float32).tofile(GD+"clip_pos100.bin")
# rel-pos tables 127 -> 79 (linear) for global blocks 2,5,8,11
def relpos40(t):
    rp=t.to(torch.float32).reshape(1,t.shape[0],-1).permute(0,2,1)
    return F.interpolate(rp,size=79,mode="linear").reshape(-1,79).permute(1,0)
for l in [2,5,8,11]:
    for nm in ["h","w"]:
        relpos40(f.get_tensor(f"model.sam_model.blocks.{l}.attn.rel_pos_{nm}")).to(torch.float32).numpy().astype(np.float32).tofile(GD+f"relpos_{nm}40_b{l}.bin")
print("regenerated 10 Gundam fixtures in", GD)
