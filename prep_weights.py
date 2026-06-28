"""Emit manifest.tsv: name, dtype, shape, abs_offset(bytes into file), nbytes.
The C++ engine mmaps the original safetensors and indexes tensors via this manifest."""
import json, struct, os, sys

SAFE = "/home/janitor/unlimited-ocr/model/model-00001-of-000001.safetensors"
OUT = "/home/janitor/unlimited-ocr/engine/manifest.tsv"

with open(SAFE, "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(hlen))
hdr.pop("__metadata__", None)
data_start = 8 + hlen

rows = []
for name, m in hdr.items():
    b, e = m["data_offsets"]
    rows.append((name, m["dtype"], ",".join(map(str, m["shape"])), data_start + b, e - b))
rows.sort()
with open(OUT, "w") as f:
    f.write(f"# safetensors={SAFE}\n# data_start={data_start}\n")
    for name, dt, shp, off, nb in rows:
        f.write(f"{name}\t{dt}\t{shp}\t{off}\t{nb}\n")
print(f"wrote {OUT}: {len(rows)} tensors, data_start={data_start}, filesize={os.path.getsize(SAFE)}")
