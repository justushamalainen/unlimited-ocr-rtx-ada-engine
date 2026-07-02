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

std::shared_ptr<OcrJob> srv_take_base(){
    std::lock_guard<std::mutex> lk(g_qm);
    if(g_q.empty()||g_q.front()->gundam) return nullptr;
    auto j=g_q.front(); g_q.pop_front(); return j;
}
std::shared_ptr<OcrJob> srv_take_gundam(){
    std::lock_guard<std::mutex> lk(g_qm);
    if(g_q.empty()||!g_q.front()->gundam) return nullptr;
    auto j=g_q.front(); g_q.pop_front(); return j;
}
bool srv_wait_work(){
    std::unique_lock<std::mutex> lk(g_qm);
    g_qcv.wait(lk,[]{return !g_q.empty();});
    return !g_q.front()->gundam;
}
void srv_complete(std::shared_ptr<OcrJob> j){
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-j->t_enq).count();
    (j->status==200?g_done:g_failed)++;
    printf("[job] %s pages=%d tok=%ld trunc=%d status=%d %.0f ms\n",
           j->gundam?"gundam":"base",j->pages,j->tokens,j->truncated,j->status,ms);
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
static void resp(int fd,int code,const char* reason,const std::string& body,const std::string& extra="",bool head_only=false){
    char h[512];
    int n=snprintf(h,sizeof h,"HTTP/1.1 %d %s\r\nConnection: close\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %zu\r\n%s\r\n",
                   code,reason,body.size(),extra.c_str());
    send_all(fd,h,n); if(!head_only) send_all(fd,body.data(),body.size());
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
static bool qparam(const std::string& q,const char* key,std::string& out){
    size_t p=0;
    while(p<q.size()){ size_t e=q.find('&',p); if(e==std::string::npos)e=q.size();
        std::string kv=q.substr(p,e-p); size_t eq=kv.find('=');
        std::string k=eq==std::string::npos?kv:kv.substr(0,eq);
        if(k==key){ out=eq==std::string::npos?"":kv.substr(eq+1); return true; }
        p=e+1; }
    return false;
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
    if(r.path!="/ocr"){ resp(fd,404,"Not Found","unknown path (POST /ocr, GET /healthz)\n"); return; }
    if(r.method!="POST"){ resp(fd,405,"Method Not Allowed","use POST\n","Allow: POST\r\n"); return; }
    if(r.te){ resp(fd,501,"Not Implemented","chunked bodies unsupported; send Content-Length\n"); return; }
    if(r.dup_cl){ resp(fd,400,"Bad Request","bad Content-Length\n"); return; }
    if(r.clen<0){ resp(fd,411,"Length Required","Content-Length required\n"); return; }
    if((size_t)r.clen>BODYCAP){ resp(fd,413,"Payload Too Large","body over cap\n"); return; }
    // params
    int npages=-1; bool gundam=false; std::string v;
    if(qparam(r.query,"pages",v)){
        if(v.empty()||v.size()>9){ resp(fd,400,"Bad Request","bad pages=\n"); return; }
        for(char ch:v) if(!isdigit((unsigned char)ch)){ resp(fd,400,"Bad Request","bad pages=\n"); return; }
        npages=atoi(v.c_str()); if(npages<1){ resp(fd,400,"Bad Request","pages must be >=1\n"); return; }
    }
    if(qparam(r.query,"gundam",v)) gundam=(v=="1");
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
    auto j=std::make_shared<OcrJob>(); j->path=path; j->npages=npages; j->gundam=gundam;
    if(!enqueue(j)){ unlink(path); resp(fd,503,"Service Unavailable","queue full\n","Retry-After: 10\r\n"); return; }
    { std::unique_lock<std::mutex> lk(j->m); j->cv.wait(lk,[&]{return j->done;}); }
    unlink(path);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-j->t_enq).count();
    if(j->status!=200){
        const char* rs=j->status==413?"Payload Too Large":j->status==422?"Unprocessable Entity":"Internal Server Error";
        resp(fd,j->status,rs,j->err+"\n"); return;
    }
    std::vector<int> flat; for(auto&p:j->page_toks) flat.insert(flat.end(),p.begin(),p.end());
    std::string text=ocr_decode_tokens(flat);
    char x[192]; snprintf(x,sizeof x,"X-Pages: %d\r\nX-Tokens: %ld\r\nX-Truncated-Pages: %d\r\nX-Millis: %.0f\r\n",
                          j->pages,j->tokens,j->truncated,ms);
    resp(fd,200,"OK",text,x);
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

int server_start(int port){
    signal(SIGPIPE,SIG_IGN);
    spool_init();
    int lfd=socket(AF_INET,SOCK_STREAM,0);
    if(lfd<0){ perror("socket"); exit(1); }
    int one=1; setsockopt(lfd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_ANY); a.sin_port=htons((uint16_t)port);
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
