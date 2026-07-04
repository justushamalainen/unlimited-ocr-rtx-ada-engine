#!/usr/bin/env python3
"""Per-page confidence-scorer evaluation on the IN-REPO corpus (no outside datasets).

Labels: pseudo-CER = normalized word-level edit distance between a page's BASE output and its
GUNDAM output (gundam = quality mode = pseudo-ground-truth), layout/coordinate tokens stripped.
Collection is ISOLATED single-page inference (pages=<p>, one page per request) so co-batch
near-tie noise pollutes neither labels nor scorers.

Usage (from repo root, needs GPU + built engine):
  .venv/bin/python engine/tools/conf_eval.py --collect     # start a test server, cache all pages
  .venv/bin/python engine/tools/conf_eval.py --report      # evaluate scorers from the cache
Outputs: outputs_conf/cache.json, outputs_conf/report.md
"""
import argparse, json, math, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENG  = os.path.join(ROOT, "engine")
OUT  = os.path.join(ROOT, "outputs_conf")
CACHE= os.path.join(OUT, "cache.json")
DOCS = [("paper",    os.path.join(ROOT, "Unlimited-OCR.pdf"),          14),
        ("brochure", os.path.join(ROOT, "testdata/reaktor_mkt.pdf"),   50),
        ("mixed",    os.path.join(ROOT, "testdata/mixed_ratio.pdf"),    3)]
BAD_CER = 0.05          # "bad page" label threshold (also report 0.15)
FEATS = ["conf","lowf","p10","wminp","emean","wment","regp"]  # engine-exported feature order (+degen added python-side)

# ---------- collection ----------
def start_server():
    env = dict(os.environ, GSLOTS="8")
    p = subprocess.Popen(["./ocr_bin","serve","0"], cwd=ENG, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    port = None; t0=time.time()
    while time.time()-t0 < 180:
        line = p.stdout.readline()
        if not line: time.sleep(0.2); continue
        m = re.search(r"ready on port (\d+)", line)
        if m: port = int(m.group(1)); break
    if not port: p.kill(); raise RuntimeError("server never became ready")
    return p, port

def ocr_page(port, pdf, page1, gundam):
    url = f"http://127.0.0.1:{port}/ocr?pages={page1},&auto=0&feats=1" + ("&gundam=1" if gundam else "")
    with open(pdf,"rb") as f: body=f.read()
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        text = r.read().decode("utf-8", "replace")
        h = {k.lower(): v for k,v in r.headers.items()}
    conf = float(h.get("x-page-conf","0")); lowf = float(h.get("x-page-lowconf","0"))
    risk = float(h.get("x-page-risk","0"))
    fv = h.get("x-page-feats","")
    p10=wminp=emean=wment=regp=0.0; ntok=0
    if fv:
        parts = fv.split(";")[0].split(":")
        p10,wminp,emean,wment,regp = map(float, parts[:5]); ntok=int(parts[5])
    return {"text":text,"conf":conf,"lowf":lowf,"risk":risk,"p10":p10,"wminp":wminp,
            "emean":emean,"wment":wment,"regp":regp,"ntok":ntok}

def collect():
    os.makedirs(OUT, exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [(d,p,m) for d,_,n in DOCS for p in range(1,n+1) for m in ("base","gundam")
            if f"{d}/{p}/{m}" not in cache]
    if not todo: print("cache complete"); return
    srv, port = start_server()
    try:
        for i,(doc,page,mode) in enumerate(todo):
            pdf = next(x[1] for x in DOCS if x[0]==doc)
            r = ocr_page(port, pdf, page, mode=="gundam")
            cache[f"{doc}/{page}/{mode}"] = r
            json.dump(cache, open(CACHE,"w"))
            print(f"[{i+1}/{len(todo)}] {doc} p{page} {mode}: conf={r['conf']:.2f} ntok={r['ntok']}")
    finally:
        srv.kill()
    print("collected", len(cache), "entries")

# ---------- labels ----------
DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)   # strip layout labels + coordinates (differ across modes)
def strip_layout(t):
    t = t.replace("<PAGE>"," ")
    t = DET.sub(" ", t)
    return " ".join(t.split())

def edit_dist(a, b):                                 # word-level Levenshtein (char-level too slow in pure python)
    if a==b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def pseudo_cer(base_text, gundam_text):
    a = strip_layout(base_text).split(); b = strip_layout(gundam_text).split()
    if not b: return 0.0 if not a else 1.0
    return min(1.0, edit_dist(a,b)/len(b))

# ---------- H7: text-side sanity (logit-free) ----------
def degen_score(text):                               # higher = worse; mirrors engine is_degenerate + garbage rate
    t = strip_layout(text)
    words=[]; cur=""; indig=False
    for c in t:
        if c.isspace():
            if cur: words.append(cur); cur=""
            indig=False
        elif c.isdigit():
            if not indig: cur+="#"; indig=True
        else: cur+=c; indig=False
    if cur: words.append(cur)
    rep = 0.0
    if len(words)>=40:
        g={" ".join(words[i:i+8]) for i in range(len(words)-7)}
        rep = 1.0 - len(g)/max(1,len(words)-7)       # non-unique 8-shingle fraction
    junk = sum(1 for c in t if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}%&/+-*'\"|<>=_#@€$£~^\\"))
    junkr = junk/max(1,len(t))
    return max(rep, min(1.0, 10*junkr))

# ---------- metrics ----------
def spearman(x, y):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(v):
            j=i
            while j+1<len(v) and v[idx[j+1]]==v[idx[i]]: j+=1
            for k in range(i,j+1): r[idx[k]]=(i+j)/2+1
            i=j+1
        return r
    rx,ry = rank(x),rank(y); n=len(x); mx=sum(rx)/n; my=sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

def auroc(scores_bad_high, labels):                  # scores: higher = more likely bad; labels: 1=bad
    pairs = sorted(zip(scores_bad_high, labels))
    pos = sum(labels); neg = len(labels)-pos
    if not pos or not neg: return float("nan")
    rank_sum=0; i=0
    while i < len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(i+j)/2+1
        rank_sum += avg*sum(1 for k in range(i,j+1) if pairs[k][1])
        i=j+1
    return (rank_sum - pos*(pos+1)/2)/(pos*neg)

def risk_coverage(scores_bad_high, labels, fracs=(0.05,0.10,0.15,0.25)):
    n=len(labels); order=sorted(range(n), key=lambda i:-scores_bad_high[i]); pos=sum(labels)
    out={}
    for f in fracs:
        k=max(1,int(round(f*n)))
        out[f]= (sum(labels[i] for i in order[:k])/pos) if pos else float("nan")
    return out

def bootstrap_win(sa, sb, labels, iters=1000, seed=7):
    import random; rng=random.Random(seed); n=len(labels); wins=0; valid=0
    for _ in range(iters):
        idx=[rng.randrange(n) for _ in range(n)]
        la=[labels[i] for i in idx]
        if not sum(la) or sum(la)==n: continue
        valid+=1
        if auroc([sa[i] for i in idx],la) > auroc([sb[i] for i in idx],la): wins+=1
    return wins/valid if valid else float("nan")

# ---------- H8: tiny logistic regression (numpy-free GD) ----------
def fit_logistic(X, y, iters=4000, lr=0.5, l2=1e-2):
    nf=len(X[0]); mu=[sum(r[k] for r in X)/len(X) for k in range(nf)]
    sd=[max(1e-6, math.sqrt(sum((r[k]-mu[k])**2 for r in X)/len(X))) for k in range(nf)]
    Z=[[(r[k]-mu[k])/sd[k] for k in range(nf)] for r in X]
    w=[0.0]*nf; b=0.0
    for _ in range(iters):
        gw=[0.0]*nf; gb=0.0
        for zi,yi in zip(Z,y):
            p=1/(1+math.exp(-max(-30,min(30, sum(wk*zk for wk,zk in zip(w,zi))+b))))
            d=p-yi
            for k in range(nf): gw[k]+=d*zi[k]
            gb+=d
        for k in range(nf): w[k]-= lr*(gw[k]/len(Z)+l2*w[k])
        b-=lr*gb/len(Z)
    return w,b,mu,sd

def logit_score(w,b,mu,sd,row):
    z=sum(wk*(rk-m)/s for wk,rk,m,s in zip(w,row,mu,sd))+b
    return 1/(1+math.exp(-max(-30,min(30,z))))

# ---------- report ----------
def report():
    cache=json.load(open(CACHE))
    pages=[]
    for doc,_,n in DOCS:
        for p in range(1,n+1):
            b=cache.get(f"{doc}/{p}/base"); g=cache.get(f"{doc}/{p}/gundam")
            if not b or not g: continue
            cer=pseudo_cer(b["text"],g["text"])
            row={"doc":doc,"page":p,"cer":cer,"degen":degen_score(b["text"])}
            row.update({k:b[k] for k in FEATS+["ntok","risk"]})
            pages.append(row)
    # split: even pages calibrate, odd pages test (mixed doc all test)
    cal=[r for r in pages if r["doc"]!="mixed" and r["page"]%2==0]
    tst=[r for r in pages if r["doc"]=="mixed" or r["page"]%2==1]
    # scorers: value = badness score (higher = escalate)
    scorers={
        "H1_conf":       lambda r: 1-r["conf"],
        "H1b_lowfrac":   lambda r: r["lowf"],
        "H2_p10":        lambda r: 1-r["p10"],
        "H3_entropy":    lambda r: r["emean"],
        "H4_winentropy": lambda r: r["wment"],
        "H4b_winminp":   lambda r: 1-r["wminp"],
        "H5_region":     lambda r: 1-r["regp"],
        "H7_textsanity": lambda r: r["degen"],
    }
    feat_row=lambda r:[r[k] for k in FEATS]+[r["degen"]]
    w,b,mu,sd=fit_logistic([feat_row(r) for r in cal],[1.0 if r["cer"]>BAD_CER else 0.0 for r in cal])
    scorers["H8_combined"]=lambda r: logit_score(w,b,mu,sd,feat_row(r))

    lines=["# Confidence scorer comparison (in-repo corpus, gundam pseudo-CER labels)",""]
    lines.append(f"pages: {len(pages)} total, {len(cal)} calibration / {len(tst)} test; bad@{BAD_CER}: "
                 f"{sum(1 for r in tst if r['cer']>BAD_CER)}/{len(tst)} test pages, bad@0.15: {sum(1 for r in tst if r['cer']>0.15)}")
    lines.append("")
    lines.append("| scorer | Spearman(score,CER) | AUROC@0.05 | AUROC@0.15 | catch@5% | @10% | @15% | @25% | boot win vs H1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    cer=[r["cer"] for r in tst]; lab05=[1 if c>BAD_CER else 0 for c in cer]; lab15=[1 if c>0.15 else 0 for c in cer]
    h1=[scorers["H1_conf"](r) for r in tst]
    results={}
    for name,fn in scorers.items():
        s=[fn(r) for r in tst]
        rc=risk_coverage(s,lab05)
        bw=bootstrap_win(s,h1,lab05) if name!="H1_conf" else float("nan")
        results[name]=(spearman(s,cer),auroc(s,lab05),auroc(s,lab15),rc,bw)
        lines.append(f"| {name} | {results[name][0]:.3f} | {results[name][1]:.3f} | {results[name][2]:.3f} | "
                     f"{rc[0.05]:.2f} | {rc[0.10]:.2f} | {rc[0.15]:.2f} | {rc[0.25]:.2f} | "+
                     (f"{bw:.2f}" if bw==bw else "—")+" |")
    lines.append("")
    lines.append(f"H8 weights (standardized, features {FEATS+['degen']}): "+
                 ", ".join(f"{k}={v:+.2f}" for k,v in zip(FEATS+["degen"],w))+f", b={b:+.2f}")
    lines.append(f"H8 raw mu: "+", ".join(f"{v:.4f}" for v in mu))
    lines.append(f"H8 raw sd: "+", ".join(f"{v:.4f}" for v in sd))
    # sentinels + failure pages
    sent=[r for r in tst if r["cer"]>0.15]
    lines.append("")
    lines.append("## Sentinel pages (test split, pseudo-CER > 0.15)")
    for r in sorted(sent,key=lambda r:-r["cer"]):
        lines.append(f"- {r['doc']} p{r['page']}: CER={r['cer']:.2f} conf={r['conf']:.2f} p10={r['p10']:.2f} "
                     f"wminp={r['wminp']:.2f} regp={r['regp']:.2f} degen={r['degen']:.2f}")
    lines.append("")
    lines.append("## Confidently-wrong candidates (high conf, high CER)")
    for r in sorted(pages,key=lambda r:-(r["cer"]*(r["conf"]>0.9)))[:5]:
        if r["cer"]>BAD_CER and r["conf"]>0.9:
            lines.append(f"- {r['doc']} p{r['page']}: CER={r['cer']:.2f} conf={r['conf']:.2f} degen={r['degen']:.2f} "
                         f"emean={r['emean']:.2f} — logits missed it, text/entropy signals must catch it")
    lines.append("")
    lines.append("Cost note: all scorers are computed from a single base decode (near-free, "
                 "traces exported by the engine); no H6 self-agreement (would double inference).")
    os.makedirs(OUT,exist_ok=True)
    open(os.path.join(OUT,"report.md"),"w").write("\n".join(lines)+"\n")
    json.dump({"pages":pages,"h8":{"w":w,"b":b,"mu":mu,"sd":sd}},open(os.path.join(OUT,"eval.json"),"w"),indent=1)
    print("\n".join(lines))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--collect",action="store_true"); ap.add_argument("--report",action="store_true")
    a=ap.parse_args()
    if a.collect: collect()
    if a.report or not a.collect: report()
