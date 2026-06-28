"""One-time dev tool: dump ground-truth vision tensors from HF to verify the CUDA vision encoder.
Saves image_ori (preprocessed input), SAM output, CLIP output, projector output, final visual tokens."""
import os, struct, numpy as np, torch, fitz
from transformers import AutoModel, AutoTokenizer
MD="/home/janitor/unlimited-ocr/model"; FX="/home/janitor/unlimited-ocr/engine/vfix"
os.makedirs(FX, exist_ok=True)
tok=AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
model=AutoModel.from_pretrained(MD, trust_remote_code=True, use_safetensors=True,
                                torch_dtype=torch.bfloat16, attn_implementation="eager").eval().cuda()
cap={}
m=model.model
def hk(mod,key,grab_in=False):
    f=mod.forward
    def t(*a,**k):
        if grab_in and a: cap[key+'_in']=a[0].detach()
        r=f(*a,**k); cap[key]=(r[0] if isinstance(r,tuple) else r).detach(); return r
    mod.forward=t
hk(m.sam_model,'sam',grab_in=True); hk(m.vision_model,'clip'); hk(m.projector,'proj')

doc=fitz.open(os.path.join(MD,"Unlimited-OCR.pdf")); d="/tmp"; pg=os.path.join(d,"vp.png")
doc[0].get_pixmap(matrix=fitz.Matrix(120/72,120/72)).save(pg); doc.close()
with torch.no_grad():
    model.infer(tok, prompt="<image>document parsing.", image_file=pg, output_path=d,
                base_size=1024, image_size=1024, crop_mode=False, max_length=300, save_results=False)

def save(name,t):
    a=t.float().cpu().numpy().ravel().astype(np.float32)
    open(os.path.join(FX,name),"wb").write(a.tobytes()); print(name, tuple(t.shape))
save("image_ori.f32", cap['sam_in'])      # [1,3,1024,1024] preprocessed input
save("sam_out.f32",  cap['sam'])           # [1,1024,16,16]
save("clip_out.f32", cap['clip'])          # [1,257,1024]
save("proj_out.f32", cap['proj'])          # [1,256,1280]
# render the SAME page to raw RGB for the C++ MuPDF path to match
import PIL.Image as I
img=I.open(pg).convert("RGB"); arr=np.asarray(img,dtype=np.uint8)
print("source png size", arr.shape)
open(os.path.join(FX,"src_rgb.bin"),"wb").write(struct.pack("<iii",arr.shape[0],arr.shape[1],3)+arr.tobytes())
print("fixtures ->", FX)
