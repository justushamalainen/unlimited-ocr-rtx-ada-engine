// server.h — HTTP front-end <-> engine-queue interface (`ocr_bin serve [port]`).
// Threading contract: connection threads (server.cpp) do sockets + files + this queue ONLY.
// The engine thread (engine.cu) is the sole consumer and the only thread that touches CUDA/MuPDF.
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <chrono>

// Per-page decode-confidence features (engine thread computes at page retire from per-step traces).
// All READ-ONLY wrt token selection. p1 = softmax prob of the emitted token; ent = top-k Shannon entropy (nats).
struct PageConf{
    float conf=0;      // mean p1 over emitted tokens incl. prefill token (H1 baseline; header back-compat)
    float lowf=0;      // fraction of tokens with p1<0.5 (header back-compat)
    float p10=0;       // 10th-percentile p1 (quantile signal: a page is bad if its WORST tokens are bad)
    float wminp=1;     // min over 32-token sliding windows of window-mean p1 (errors cluster locally)
    float emean=0;     // mean per-token entropy
    float wment=0;     // max over 32-token sliding windows of window-mean entropy
    float regp=1;      // worst line-region mean p1 (regions split at newline tokens, merged to >=24 tokens)
    int   ntok=0;      // tokens scored (post-EOS steps excluded)
};
struct OcrJob{
    // request (connection thread, immutable after enqueue)
    std::string path;                 // spooled PDF (unlinked by the connection thread after completion)
    int npages=-1;                    // page cap; -1 = all
    std::vector<int> pagelist;        // explicit 0-based pages (?pages=3,7,12 — selective retry); empty = first npages
    bool gundam=false;                // high-res tiling (pages co-batch with base work in the same decode window)
    bool auto_hires=true;             // ?auto=0 to disable: server re-OCRs low-conf/looping base pages in gundam before returning
    bool retried=false;               // the gundam re-pass items of this base job are queued/ran -> don't re-defer
    std::vector<int> pdfpages;        // page_toks[i] <-> this PDF page (base pass; for mapping flagged pages to gundam re-encode)
    std::chrono::steady_clock::time_point t_enq;
    // result (engine thread, before srv_complete)
    int status=200; std::string err;  // 200 | 422 unreadable/render-failed
    std::vector<std::vector<int>> page_toks;  // per page, raw token ids (EOS included; decode via ocr_decode_tokens)
    std::vector<float> page_conf;     // per page: mean top-1 prob of emitted tokens (decode confidence)
    std::vector<float> page_lowfrac;  // per page: fraction of tokens with p1<0.5
    std::vector<PageConf> page_feats; // per page: full confidence feature vector (X-Page-Feats with ?feats=1)
    std::vector<float> page_risk;     // per page: calibrated bad-page risk 0..1 (X-Page-Risk; higher = escalate)
    bool want_feats=false;            // ?feats=1: emit the full feature header (eval harness / power clients)
    int pages=0; int pending=0;       // pending = pages not yet finished (engine-internal)
    long tokens=0; int truncated=0;   // emitted (non-EOS) tokens; pages cut at MAXSTEP without EOS
    // completion handshake
    std::mutex m; std::condition_variable cv; bool done=false;
};

// engine-facing (engine thread only)
std::shared_ptr<OcrJob> srv_take();          // pop head job (base or gundam); null if empty
bool srv_wait_work();                        // block until a job is queued (always true; kept bool for the PageSrc::wait contract)
void srv_complete(std::shared_ptr<OcrJob> j);// publish result + wake the waiting connection thread
// server lifecycle (main)
int  server_start(int port,const char* bind_addr=nullptr);  // spool+listen+accept thread; returns bound port (0 = ephemeral ok); bind_addr default 0.0.0.0
// engine.cu exports for connection threads (read-only global vocab -> thread-safe after load_vocab)
std::string ocr_decode_tokens(const std::vector<int>& toks);   // BPE-decode, skipping EOS(id 1)
