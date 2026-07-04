// server.cpp — hand-rolled HTTP/1.1 front-end for `ocr_bin serve [port]`. No deps beyond POSIX.
// POST /ocr[?pages=N][&gundam=1]  body=PDF  -> 200 text/plain (blocks until OCR'd) | 4xx/5xx
// GET|HEAD /healthz               -> 200 "ok" (never touches the engine; bound only after weights load)
// One request per connection (Connection: close — no keep-alive, no pipelining, no chunked bodies).
// Threading: accept thread + detached thread per connection; connection threads never touch CUDA/MuPDF.
#include "server.h"
#include <deque>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <csignal>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>

static const size_t BODYCAP=256u<<20;      // max upload
static const size_t HDRCAP=16384;          // max request head
static const int    QCAP=16;               // max queued jobs -> 503
static const int    CONNCAP=64;            // max concurrent connections -> 503
static const int    IOTIMEO=30;            // per-recv/send socket timeout (s)

// ---- job queue (multi-producer connection threads -> single-consumer engine thread) ----
static std::deque<std::shared_ptr<OcrJob>> g_q;
static std::mutex g_qm; static std::condition_variable g_qcv;
static std::atomic<long> g_done{0},g_failed{0};
static std::string g_spool;

std::shared_ptr<OcrJob> srv_take(){
    std::lock_guard<std::mutex> lk(g_qm);
    if(g_q.empty()) return nullptr;
    auto j=g_q.front(); g_q.pop_front(); return j;
}
bool srv_wait_work(){
    std::unique_lock<std::mutex> lk(g_qm);
    g_qcv.wait(lk,[]{return !g_q.empty();});
    return true;                                       // base and gundam jobs feed the same heterogeneous window
}
void srv_complete(std::shared_ptr<OcrJob> j){
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-j->t_enq).count();
    (j->status==200?g_done:g_failed)++;
    float cm=0; for(float c:j->page_conf) cm+=c; if(!j->page_conf.empty()) cm/=j->page_conf.size();
    printf("[job] %s pages=%d tok=%ld trunc=%d conf=%.2f status=%d %.0f ms\n",
           j->retried?"auto-hires":j->gundam?"gundam":"base",j->pages,j->tokens,j->truncated,cm,j->status,ms);
    fflush(stdout);
    { std::lock_guard<std::mutex> lk(j->m); j->done=true; }
    j->cv.notify_all();
}
static bool enqueue(std::shared_ptr<OcrJob> j){
    { std::lock_guard<std::mutex> lk(g_qm);
      if((int)g_q.size()>=QCAP) return false;
      j->t_enq=std::chrono::steady_clock::now(); g_q.push_back(j); }
    g_qcv.notify_one(); return true;
}
static int queue_depth(){ std::lock_guard<std::mutex> lk(g_qm); return (int)g_q.size(); }

// ---- socket helpers ----
static bool send_all(int fd,const char* p,size_t n){
    while(n){ ssize_t k=send(fd,p,n,MSG_NOSIGNAL); if(k<=0){ if(k<0&&errno==EINTR)continue; return false; } p+=k; n-=k; }
    return true;
}
static void linger_close(int fd){                        // drain unread request bytes so the response isn't RST'd away
    shutdown(fd,SHUT_WR);
    char buf[8192]; size_t drained=0; time_t t0=time(nullptr);
    while(drained<(32u<<20) && time(nullptr)-t0<5){ ssize_t k=recv(fd,buf,sizeof buf,0); if(k<=0)break; drained+=k; }
    close(fd);
}
static void resp(int fd,int code,const char* reason,const std::string& body,const std::string& extra="",bool head_only=false,
                 const char* ctype="text/plain; charset=utf-8"){
    char cl[64]; snprintf(cl,sizeof cl,"%zu",body.size());
    std::string h="HTTP/1.1 "; h+=std::to_string(code); h+=' '; h+=reason;
    h+="\r\nConnection: close\r\nContent-Type: "; h+=ctype;
    h+="\r\nContent-Length: "; h+=cl; h+="\r\n"; h+=extra;   // extra can be large (per-page headers) -> std::string, not a fixed buffer
    h+="\r\n";
    send_all(fd,h.data(),h.size()); if(!head_only) send_all(fd,body.data(),body.size());
    linger_close(fd);
}

// ---- request head parsing ----
struct Req{ std::string method,path,query; long long clen=-1; bool te=false,expect100=false,dup_cl=false;
            std::string hdrbuf; size_t body_off=0; };
static bool ieq(const std::string&a,const char*b){ if(a.size()!=strlen(b))return false;
    for(size_t i=0;i<a.size();i++) if(tolower(a[i])!=tolower(b[i]))return false; return true; }
static bool read_head(int fd,Req& r){                    // recv until CRLFCRLF; bytes past it are body carryover
    r.hdrbuf.reserve(4096); char buf[4096]; time_t t0=time(nullptr);
    for(;;){
        size_t ecr=r.hdrbuf.find("\r\n\r\n"), elf=r.hdrbuf.find("\n\n");   // earlier match wins: a bare-LF head's
        if(ecr!=std::string::npos&&(elf==std::string::npos||ecr<elf)){ r.body_off=ecr+4; break; }  // binary body carryover can contain \r\n\r\n
        if(elf!=std::string::npos){ r.body_off=elf+2; break; }
        if(r.hdrbuf.size()>HDRCAP || time(nullptr)-t0>10) return false;
        ssize_t k=recv(fd,buf,sizeof buf,0);
        if(k<=0){ if(k<0&&errno==EINTR)continue; return false; }
        r.hdrbuf.append(buf,k);
    }
    // request line
    size_t eol=r.hdrbuf.find('\n'); std::string line=r.hdrbuf.substr(0,eol);
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    size_t s1=line.find(' '),s2=line.rfind(' ');
    if(s1==std::string::npos||s2==s1||line.compare(s2+1,7,"HTTP/1.")!=0) return false;
    r.method=line.substr(0,s1);
    std::string target=line.substr(s1+1,s2-s1-1);
    if(target.empty()||target[0]!='/') return false;      // origin-form only
    size_t q=target.find('?');
    r.path=target.substr(0,q); if(q!=std::string::npos)r.query=target.substr(q+1);
    // headers
    size_t pos=eol+1;
    while(pos<r.body_off){
        size_t e2=r.hdrbuf.find('\n',pos); if(e2==std::string::npos||e2>=r.body_off)break;
        std::string h=r.hdrbuf.substr(pos,e2-pos); pos=e2+1;
        if(!h.empty()&&h.back()=='\r')h.pop_back();
        if(h.empty())continue;
        size_t c=h.find(':'); if(c==std::string::npos)continue;
        std::string k=h.substr(0,c); std::string v=h.substr(c+1);
        while(!v.empty()&&(v[0]==' '||v[0]=='\t'))v.erase(0,1);
        while(!v.empty()&&(v.back()==' '||v.back()=='\t'))v.pop_back();
        if(ieq(k,"content-length")){
            if(r.clen>=0){ r.dup_cl=true; continue; }
            if(v.empty()||v.size()>19) { r.dup_cl=true; continue; }
            for(char ch:v) if(!isdigit((unsigned char)ch)){ r.dup_cl=true; break; }
            if(!r.dup_cl) r.clen=strtoll(v.c_str(),nullptr,10);
        } else if(ieq(k,"transfer-encoding")) r.te=true;
        else if(ieq(k,"expect")){ std::string lv=v; for(auto&ch:lv)ch=tolower(ch); if(lv.find("100-continue")!=std::string::npos)r.expect100=true; }
    }
    return true;
}
static std::string pctdecode(const std::string& s){     // %XX + '+' -> raw (URLSearchParams encodes ',' as %2C)
    std::string o; o.reserve(s.size());
    for(size_t i=0;i<s.size();i++){
        if(s[i]=='%'&&i+2<s.size()&&isxdigit((unsigned char)s[i+1])&&isxdigit((unsigned char)s[i+2])){
            o+=(char)strtol(s.substr(i+1,2).c_str(),nullptr,16); i+=2;
        } else if(s[i]=='+') o+=' '; else o+=s[i];
    }
    return o;
}
static bool qparam(const std::string& q,const char* key,std::string& out){
    size_t p=0;
    while(p<q.size()){ size_t e=q.find('&',p); if(e==std::string::npos)e=q.size();
        std::string kv=q.substr(p,e-p); size_t eq=kv.find('=');
        std::string k=eq==std::string::npos?kv:kv.substr(0,eq);
        if(k==key){ out=eq==std::string::npos?"":pctdecode(kv.substr(eq+1)); return true; }
        p=e+1; }
    return false;
}

// ---- web assets: <dir of binary>/web (server can be started from any cwd) ----
static std::string g_webdir;
static void webdir_init(){
    char buf[512]; ssize_t n=readlink("/proc/self/exe",buf,sizeof buf-1);
    if(n>0){ buf[n]=0; std::string p(buf); g_webdir=p.substr(0,p.rfind('/'))+"/web"; }
    else g_webdir="web";
}

// ---- per-connection handler (socket timeouts already set by the accept loop) ----
static std::atomic<int> g_conns{0};
static void handle(int fd){
    Req r;
    if(!read_head(fd,r)){ resp(fd,400,"Bad Request","bad request head\n"); return; }
    if(r.path=="/healthz"){
        if(r.method!="GET"&&r.method!="HEAD"){ resp(fd,405,"Method Not Allowed","use GET\n","Allow: GET, HEAD\r\n"); return; }
        char x[128]; snprintf(x,sizeof x,"X-Queue: %d\r\nX-Done: %ld\r\nX-Failed: %ld\r\n",queue_depth(),g_done.load(),g_failed.load());
        resp(fd,200,"OK","ok\n",x,r.method=="HEAD"); return;
    }
    if(r.method=="GET"||r.method=="HEAD"){                  // annotation-viewer statics (fixed allowlist, no traversal)
        const char* ct=nullptr; std::string fn;
        if(r.path=="/"||r.path=="/index.html"){ fn="index.html"; ct="text/html; charset=utf-8"; }
        else if(r.path=="/pdf.mjs"||r.path=="/pdf.worker.mjs"){ fn=r.path.substr(1); ct="text/javascript"; }
        if(ct){
            std::string body; FILE* f=fopen((g_webdir+"/"+fn).c_str(),"rb");
            if(!f){ resp(fd,404,"Not Found","viewer asset missing (web/ next to the binary)\n"); return; }
            char b[65536]; size_t k; while((k=fread(b,1,sizeof b,f))>0) body.append(b,k); fclose(f);
            resp(fd,200,"OK",body,"",r.method=="HEAD",ct); return;
        }
    }
    if(r.path!="/ocr"){ resp(fd,404,"Not Found","unknown path (GET / viewer, POST /ocr, GET /healthz)\n"); return; }
    if(r.method!="POST"){ resp(fd,405,"Method Not Allowed","use POST\n","Allow: POST\r\n"); return; }
    if(r.te){ resp(fd,501,"Not Implemented","chunked bodies unsupported; send Content-Length\n"); return; }
    if(r.dup_cl){ resp(fd,400,"Bad Request","bad Content-Length\n"); return; }
    if(r.clen<0){ resp(fd,411,"Length Required","Content-Length required\n"); return; }
    if((size_t)r.clen>BODYCAP){ resp(fd,413,"Payload Too Large","body over cap\n"); return; }
    // params
    int npages=-1; bool gundam=false; std::string v; std::vector<int> pagelist;
    if(qparam(r.query,"pages",v)){                       // "N" = first N pages | "3,7,12" = exactly those pages (1-based)
        if(v.empty()||v.size()>4096){ resp(fd,400,"Bad Request","bad pages=\n"); return; }
        for(char ch:v) if(!isdigit((unsigned char)ch)&&ch!=','){ resp(fd,400,"Bad Request","bad pages=\n"); return; }
        if(v.find(',')==std::string::npos){
            npages=atoi(v.c_str()); if(npages<1){ resp(fd,400,"Bad Request","pages must be >=1\n"); return; }
        } else {
            size_t p=0;
            while(p<v.size()){ size_t e=v.find(',',p); if(e==std::string::npos)e=v.size();
                std::string tok=v.substr(p,e-p); p=e+1;
                if(tok.empty()||tok.size()>9){ resp(fd,400,"Bad Request","bad pages= list\n"); return; }
                int pg=atoi(tok.c_str()); if(pg<1){ resp(fd,400,"Bad Request","pages entries must be >=1\n"); return; }
                pagelist.push_back(pg-1);
            }
            if(pagelist.empty()||pagelist.size()>512){ resp(fd,400,"Bad Request","pages list: 1..512 entries\n"); return; }
        }
    }
    if(qparam(r.query,"gundam",v)) gundam=(v=="1");
    bool autohires=true; if(qparam(r.query,"auto",v)) autohires=(v!="0");   // server re-OCRs flagged base pages in gundam before returning
    bool feats=false; if(qparam(r.query,"feats",v)) feats=(v=="1");         // emit the full X-Page-Feats vector (eval harness / power clients)
    if(queue_depth()>=QCAP){ resp(fd,503,"Service Unavailable","queue full\n","Retry-After: 10\r\n"); return; }
    if(r.expect100){ const char* c="HTTP/1.1 100 Continue\r\n\r\n"; if(!send_all(fd,c,strlen(c))){ close(fd); return; } }
    // spool body (header carryover first, then stream)
    char path[512]; snprintf(path,sizeof path,"%s/job_XXXXXX",g_spool.c_str());
    int sfd=mkstemp(path);
    if(sfd<0){ resp(fd,500,"Internal Server Error","spool failed\n"); return; }
    size_t got=r.hdrbuf.size()-r.body_off;
    if(got>(size_t)r.clen)got=r.clen;
    bool ok=true,werr=false;                               // werr: OUR disk failed (ENOSPC) — client is owed a 500, not an RST
    if(got&&(size_t)write(sfd,r.hdrbuf.data()+r.body_off,got)!=got){ ok=false; werr=true; }
    long long remain=r.clen-got; char buf[65536];
    time_t t0=time(nullptr); long long deadline=std::max(60LL,r.clen/(1<<20));
    while(ok&&remain>0){
        if(time(nullptr)-t0>deadline){ ok=false; break; }
        ssize_t k=recv(fd,buf,(size_t)std::min<long long>(sizeof buf,remain),0);
        if(k<=0){ if(k<0&&errno==EINTR)continue; ok=false; break; }
        if(write(sfd,buf,k)!=k){ ok=false; werr=true; break; }
        remain-=k;
    }
    int werrno=werr?errno:0;
    close(sfd);
    if(!ok){ unlink(path);
        if(werr){ fprintf(stderr,"[srv] spool write failed: %s\n",strerror(werrno)); g_failed++;
                  resp(fd,500,"Internal Server Error","spool write failed\n"); }
        else close(fd);                                    // client vanished mid-upload: no response owed
        return; }
    // enqueue + wait (v1: no cancel — a disconnected client's job completes and is discarded)
    auto j=std::make_shared<OcrJob>(); j->path=path; j->npages=npages; j->gundam=gundam; j->pagelist=std::move(pagelist); j->auto_hires=autohires; j->want_feats=feats;
    if(!enqueue(j)){ unlink(path); resp(fd,503,"Service Unavailable","queue full\n","Retry-After: 10\r\n"); return; }
    { std::unique_lock<std::mutex> lk(j->m); j->cv.wait(lk,[&]{return j->done;}); }
    unlink(path);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-j->t_enq).count();
    if(j->status!=200){
        const char* rs=j->status==400?"Bad Request":j->status==413?"Payload Too Large":j->status==422?"Unprocessable Entity":"Internal Server Error";
        resp(fd,j->status,rs,j->err+"\n"); return;
    }
    std::string text;                                    // per-page decode joined by form-feed (\f) — a reliable page
    for(size_t i=0;i<j->page_toks.size();i++){           // delimiter, unlike the model-emitted <PAGE> which can be
        std::string pt=ocr_decode_tokens(j->page_toks[i]); // dropped or hallucinated -> segment/conf misalignment
        for(char&c:pt) if(c=='\f')c=' ';                 // reserve \f (byte-BPE effectively never emits it in doc text)
        if(i)text+='\f';
        text+=pt;
    }
    char x[192]; snprintf(x,sizeof x,"X-Pages: %d\r\nX-Tokens: %ld\r\nX-Truncated-Pages: %d\r\nX-Millis: %.0f\r\n",
                          j->pages,j->tokens,j->truncated,ms);
    std::string extra=x;
    if(!j->page_conf.empty() && j->page_conf.size()<=2048){          // per-page decode confidence (header stays bounded)
        char n[48]; std::string c="X-Page-Conf: ", l="X-Page-LowConf: ", k="X-Page-Risk: ";
        for(size_t i=0;i<j->page_conf.size();i++){
            snprintf(n,sizeof n,"%s%.2f",i?",":"",j->page_conf[i]);    c+=n;
            snprintf(n,sizeof n,"%s%.2f",i?",":"",j->page_lowfrac[i]); l+=n;
            snprintf(n,sizeof n,"%s%.2f",i?",":"",i<j->page_risk.size()?j->page_risk[i]:0.f); k+=n;
        }
        extra+=c+"\r\n"+l+"\r\n"+k+"\r\n";
        if(!j->page_mode.empty()){                                   // which mode produced each page (b=base, g=gundam pre-check/retry)
            std::string md="X-Page-Mode: ";
            for(size_t i=0;i<j->page_mode.size();i++){ if(i)md+=','; md+=j->page_mode[i]?'g':'b'; }
            extra+=md+"\r\n";
        }
        if(j->want_feats && j->page_feats.size()<=512){              // full feature vector per page (?feats=1): p10:wminp:emean:wment:regp:ntok
            std::string ftr="X-Page-Feats: ";
            for(size_t i=0;i<j->page_feats.size();i++){ const PageConf& pc=j->page_feats[i];
                snprintf(n,sizeof n,"%s%.3f:%.3f:%.3f:%.3f:%.3f:%d",i?";":"",pc.p10,pc.wminp,pc.emean,pc.wment,pc.regp,pc.ntok); ftr+=n;
            }
            extra+=ftr+"\r\n";
        }
    }
    resp(fd,200,"OK",text,extra);
}

// ---- spool dir: per-pid, sweep dirs of dead pids (fail-fast exit(1) leaves orphans) ----
static void spool_init(){
    const char* base=getenv("TMPDIR"); if(!base||!*base)base="/var/tmp";
    DIR* d=opendir(base);
    if(d){ struct dirent* e;
        while((e=readdir(d))){ int pid=0;
            if(sscanf(e->d_name,"ocr_srv.%d",&pid)==1 && pid>0 && kill(pid,0)!=0 && errno==ESRCH){
                char p[512]; DIR* dd; struct dirent* f;
                snprintf(p,sizeof p,"%s/%s",base,e->d_name);
                if((dd=opendir(p))){ while((f=readdir(dd))){ if(f->d_name[0]!='.'){ char fp[1024]; snprintf(fp,sizeof fp,"%s/%s",p,f->d_name); unlink(fp); } } closedir(dd); }
                rmdir(p);
            } }
        closedir(d); }
    char p[512]; snprintf(p,sizeof p,"%s/ocr_srv.%d",base,(int)getpid());
    mkdir(p,0700); g_spool=p;
}

int server_start(int port,const char* bind_addr){
    signal(SIGPIPE,SIG_IGN);
    webdir_init();
    spool_init();
    int lfd=socket(AF_INET,SOCK_STREAM,0);
    if(lfd<0){ perror("socket"); exit(1); }
    int one=1; setsockopt(lfd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_ANY); a.sin_port=htons((uint16_t)port);
    if(bind_addr && inet_pton(AF_INET,bind_addr,&a.sin_addr)!=1){ fprintf(stderr,"bad bind address: %s\n",bind_addr); exit(1); }  // e.g. 127.0.0.1 behind a TLS proxy
    if(bind(lfd,(sockaddr*)&a,sizeof a)<0){ perror("bind"); exit(1); }
    if(listen(lfd,SOMAXCONN)<0){ perror("listen"); exit(1); }
    socklen_t al=sizeof a; getsockname(lfd,(sockaddr*)&a,&al);
    int bound=ntohs(a.sin_port);
    std::thread([lfd]{
        static std::atomic<int> rejects{0};                 // reject threads are bounded too — an unbounded spawn path
        for(;;){                                            // would let pure connection load throw in std::thread -> terminate
            int fd=accept4(lfd,nullptr,nullptr,SOCK_CLOEXEC);
            if(fd<0){ if(errno==EINTR||errno==ECONNABORTED)continue; if(errno==EMFILE||errno==ENFILE){usleep(10000);continue;} perror("accept"); continue; }
            struct timeval tv{IOTIMEO,0};                   // timeouts BEFORE any use: linger/send on an untimed fd blocks forever
            setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv); setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
            if(g_conns.load()>=CONNCAP){                    // reject off-thread: linger drain must not stall accept()
                if(rejects.load()>=32){ close(fd); continue; }         // extreme overload: drop (RST) beats leaking threads
                rejects++;
                try{ std::thread([fd]{
                        const char* m="HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 5\r\nContent-Length: 0\r\n\r\n";
                        send_all(fd,m,strlen(m)); linger_close(fd);    // drain: bare close RSTs the 503 away mid-upload
                        rejects--;
                    }).detach(); }
                catch(...){ rejects--; close(fd); }
                continue;
            }
            g_conns++;
            try{ std::thread([fd]{ handle(fd); g_conns--; }).detach(); }
            catch(...){ g_conns--; close(fd); }
        }
    }).detach();
    return bound;
}
