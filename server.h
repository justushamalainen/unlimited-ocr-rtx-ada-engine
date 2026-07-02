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

struct OcrJob{
    // request (connection thread, immutable after enqueue)
    std::string path;                 // spooled PDF (unlinked by the connection thread after completion)
    int npages=-1;                    // page cap; -1 = all
    bool gundam=false;                // high-res tiling (exclusive run; base jobs co-batch)
    std::chrono::steady_clock::time_point t_enq;
    // result (engine thread, before srv_complete)
    int status=200; std::string err;  // 200 | 413 gundam page cap | 422 unreadable/render-failed
    std::vector<std::vector<int>> page_toks;  // per page, raw token ids (EOS included; decode via ocr_decode_tokens)
    int pages=0; int pending=0;       // pending = pages not yet finished (engine-internal)
    long tokens=0; int truncated=0;   // emitted (non-EOS) tokens; pages cut at MAXSTEP without EOS
    // completion handshake
    std::mutex m; std::condition_variable cv; bool done=false;
};

// engine-facing (engine thread only)
std::shared_ptr<OcrJob> srv_take_base();     // pop head job if base; null if empty or head is gundam
std::shared_ptr<OcrJob> srv_take_gundam();   // pop head job if gundam; null otherwise
bool srv_wait_work();                        // block until a job is queued; true = head is base, false = head is gundam
void srv_complete(std::shared_ptr<OcrJob> j);// publish result + wake the waiting connection thread
// server lifecycle (main)
int  server_start(int port);                 // spool dir + listen + accept thread; returns bound port (0 = ephemeral ok)
// engine.cu exports for connection threads (read-only global vocab -> thread-safe after load_vocab)
std::string ocr_decode_tokens(const std::vector<int>& toks);   // BPE-decode, skipping EOS(id 1)
