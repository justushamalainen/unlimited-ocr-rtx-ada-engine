// SAM ViT-B encoder in C++/CUDA (fp32, cuBLAS). Verified vs HF fixtures (vfix/).
// patch16 -> 64x64x768 -> 12 blocks (window=14 rel-pos, global at 2,5,8,11) -> neck -> net_2/net_3 -> [1024,16,16].
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
extern "C" {
#include <mupdf/fitz.h>
}
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float b2f(bf16 x){return __bfloat162float(x);}
__device__ __forceinline__ bf16 f2b(float x){return __float2bfloat16(x);}
#include "st_loader.h"
#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x);if(s){fprintf(stderr,"cublas %s:%d %d\n",__FILE__,__LINE__,s);exit(1);} }while(0)
static const int IMG=1024,PATCH=16,GS=64,C=768,NH=12,HD=64,MLP=3072,WIN=14;
static cublasHandle_t CUB; static SafeTensors ST;
static const float ONE=1.f,ZERO=0.f;
static float* g_dimg; static cudaStream_t VS;  // declared early (used in render_preprocess)

// ---- MuPDF render + GPU aspect-pad resize + normalize (pixel/127.5-1) -> [3,1024,1024] ----
__global__ void k_resize_norm(const unsigned char* src,int h,int w,int nw,int nh,int offx,int offy,float* out){
    int ox=blockIdx.x*blockDim.x+threadIdx.x, oy=blockIdx.y*blockDim.y+threadIdx.y; if(ox>=IMG||oy>=IMG)return;
    int ix=ox-offx, iy=oy-offy;
    if(ix<0||ix>=nw||iy<0||iy>=nh){ for(int c=0;c<3;c++) out[(size_t)c*IMG*IMG+oy*IMG+ox]=127.f/127.5f-1.f; return; }
    float fx=(ix+0.5f)*w/nw-0.5f, fy=(iy+0.5f)*h/nh-0.5f;
    int x0=floorf(fx),y0=floorf(fy); float dx=fx-x0,dy=fy-y0; int x1=x0+1,y1=y0+1;
    x0=max(0,min(w-1,x0));x1=max(0,min(w-1,x1));y0=max(0,min(h-1,y0));y1=max(0,min(h-1,y1));
    for(int c=0;c<3;c++){float p00=src[(y0*w+x0)*3+c],p01=src[(y0*w+x1)*3+c],p10=src[(y1*w+x0)*3+c],p11=src[(y1*w+x1)*3+c];
        float v=p00*(1-dx)*(1-dy)+p01*dx*(1-dy)+p10*(1-dx)*dy+p11*dx*dy; out[(size_t)c*IMG*IMG+oy*IMG+ox]=v/127.5f-1.f;}
}
static unsigned char* g_dsrc=nullptr; static size_t g_dsrcn=0;
static void render_preprocess(const char* pdf,int page,float dpi){  // -> g_dimg
    fz_context* ctx=fz_new_context(NULL,NULL,FZ_STORE_UNLIMITED); fz_register_document_handlers(ctx);
    fz_set_aa_level(ctx,2);  // lower anti-aliasing: faster render, no OCR-quality loss (model downsamples to 1024)
    fz_document* doc=fz_open_document(ctx,pdf); fz_matrix mat=fz_scale(dpi/72.f,dpi/72.f);
    fz_pixmap* pix=fz_new_pixmap_from_page_number(ctx,doc,page,mat,fz_device_rgb(ctx),0);
    int w=fz_pixmap_width(ctx,pix),h=fz_pixmap_height(ctx,pix); unsigned char* sm=fz_pixmap_samples(ctx,pix);
    if((size_t)w*h*3>g_dsrcn){ if(g_dsrc)cudaFree(g_dsrc); CK(cudaMalloc(&g_dsrc,(size_t)w*h*3)); g_dsrcn=(size_t)w*h*3; }
    CK(cudaMemcpy(g_dsrc,sm,(size_t)w*h*3,cudaMemcpyHostToDevice));
    float sc=fminf((float)IMG/w,(float)IMG/h); int nw=(int)lroundf(w*sc),nh=(int)lroundf(h*sc),offx=(IMG-nw)/2,offy=(IMG-nh)/2;
    dim3 b(16,16),g((IMG+15)/16,(IMG+15)/16); k_resize_norm<<<g,b>>>(g_dsrc,h,w,nw,nh,offx,offy,g_dimg);
    fz_drop_pixmap(ctx,pix); fz_drop_document(ctx,doc); fz_drop_context(ctx);
}

// ---- weight loading ----
__global__ void k_bf16f32(const unsigned short* s,float* d,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n){unsigned u=(unsigned)s[i]<<16;float f;memcpy(&f,&u,4);d[i]=f;}}
static unsigned short* g_stage=nullptr; static size_t g_stagen=0;  // reused bf16 staging (no per-tensor cudaFree sync)
static bf16* Wb(const std::string& nm){   // GEMM/embedding weights: keep bf16 (safetensors is bf16) -> direct upload
    const Tensor& t=ST.get(nm); size_t n=1; for(long d:t.shape)n*=d;
    bf16* d; CK(cudaMalloc(&d,n*2)); CK(cudaMemcpy(d,t.data,n*2,cudaMemcpyHostToDevice)); return d;
}
static float* W(const std::string& nm){   // small params (LN, bias, rel_pos): convert to fp32 for precision
    const Tensor& t=ST.get(nm); size_t n=1; for(long d:t.shape)n*=d;
    if(n>g_stagen){ if(g_stage)cudaFree(g_stage); CK(cudaMalloc(&g_stage,n*2)); g_stagen=n; }
    CK(cudaMemcpy(g_stage,t.data,n*2,cudaMemcpyHostToDevice));
    float* d; CK(cudaMalloc(&d,n*4)); k_bf16f32<<<(n+255)/256,256>>>(g_stage,d,n); return d;
}
static size_t numel(const std::string& nm){const Tensor&t=ST.get(nm);size_t n=1;for(long d:t.shape)n*=d;return n;}

// ---- kernels (bf16 activations, fp32 accum; small params fp32) ----
static const float ONE_=1.f,ZERO_=0.f;
__global__ void k_addbias(bf16* y,const float* b,int M,int N){size_t i=blockIdx.x*256+threadIdx.x;if(i<(size_t)M*N)y[i]=f2b(b2f(y[i])+b[i%N]);}
__global__ void k_vadd(bf16* y,const bf16* a,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=f2b(b2f(y[i])+b2f(a[i]));}
__global__ void k_geluE(bf16* x,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n){float v=b2f(x[i]);x[i]=f2b(0.5f*v*(1.f+erff(v*0.70710678f)));}}
__global__ void k_geluE_bias(bf16* x,const float* b,int N,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n){float v=b2f(x[i])+b[i%N];x[i]=f2b(0.5f*v*(1.f+erff(v*0.70710678f)));}}
__global__ void k_ln(const bf16* x,bf16* y,const float* w,const float* b,int N,int Cc,float eps){
    int r=blockIdx.x; if(r>=N)return; const bf16* xr=x+(size_t)r*Cc; bf16* yr=y+(size_t)r*Cc;
    __shared__ float sm,sv; float s=0; for(int i=threadIdx.x;i<Cc;i+=blockDim.x)s+=b2f(xr[i]);
    for(int o=16;o;o>>=1)s+=__shfl_xor_sync(~0u,s,o); __shared__ float ws[32]; if((threadIdx.x&31)==0)ws[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x==0){float m=0;for(int i=0;i<(blockDim.x+31)/32;i++)m+=ws[i];sm=m/Cc;} __syncthreads();
    float v=0; for(int i=threadIdx.x;i<Cc;i+=blockDim.x){float d=b2f(xr[i])-sm;v+=d*d;}
    for(int o=16;o;o>>=1)v+=__shfl_xor_sync(~0u,v,o); if((threadIdx.x&31)==0)ws[threadIdx.x>>5]=v; __syncthreads();
    if(threadIdx.x==0){float m=0;for(int i=0;i<(blockDim.x+31)/32;i++)m+=ws[i];sv=rsqrtf(m/Cc+eps);} __syncthreads();
    for(int i=threadIdx.x;i<Cc;i+=blockDim.x)yr[i]=f2b((b2f(xr[i])-sm)*sv*w[i]+b[i]);
}
// im2col patch16: img fp32 [3,IMG,IMG] -> patches bf16 [GS*GS, 768] order ic*256+ky*16+kx
__global__ void k_im2col(const float* img,bf16* pat,int gs,int imgsz){
    int tok=blockIdx.x, j=threadIdx.x; int oh=tok/gs, ow=tok%gs; int ic=j/256, r=j%256, ky=r/16, kx=r%16;
    pat[(size_t)tok*768+j]=f2b(img[(size_t)ic*imgsz*imgsz+(oh*16+ky)*imgsz+(ow*16+kx)]);
}
// linear y[M,N] = x[M,K] @ W[N,K]^T + bias  (bf16 GEMM, fp32 accumulate)
static void linear(int M,int K,int N,const bf16* x,const bf16* w,const float* b,bf16* y){
    CB(cublasGemmEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&ONE_,w,CUDA_R_16BF,K,x,CUDA_R_16BF,K,&ZERO_,y,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    if(b) k_addbias<<<((size_t)M*N+255)/256,256>>>(y,b,M,N);
}
__global__ void k_splitqkv(const bf16* qkv,bf16* q,bf16* k,bf16* v,int B,int N,int Nh,const float* bias){
    int token=blockIdx.x, c=threadIdx.x; int b=token/N, qi=token%N, h=blockIdx.y;
    const bf16* base=qkv+(size_t)token*3*C + h*HD; size_t o=((size_t)(b*Nh+h)*N+qi)*HD + c;
    q[o]=f2b(b2f(base[c])+bias[h*HD+c]); k[o]=f2b(b2f(base[C+c])+bias[C+h*HD+c]); v[o]=f2b(b2f(base[2*C+c])+bias[2*C+h*HD+c]);
}
// rel-pos: one block per (bh, query-row qh) processes all W queries in that row together.
// All W queries share the Rh rows (same qh); rel_w needs the whole tabW (loaded once). 768 blocks for global.
__global__ void k_relpos(const bf16* q,const float* tabH,const float* tabW,float* rh,float* rw,int Bh,int N,int Hsz,int Wsz){
    int bh=blockIdx.y, qh=blockIdx.x, t=threadIdx.x; int W=Wsz, Hk=Hsz;
    extern __shared__ float sh[]; float* Q=sh; float* Rh=Q+(size_t)W*HD; float* Tw=Rh+(size_t)Hk*HD;
    for(int i=t;i<W*HD;i+=blockDim.x){ int qw=i/HD,c=i%HD; Q[i]=b2f(q[((size_t)(bh*N+qh*W+qw))*HD+c]); }
    for(int i=t;i<Hk*HD;i+=blockDim.x){ int kh=i/HD,c=i%HD; Rh[i]=tabH[(size_t)((qh-kh)+Hk-1)*HD+c]; }  // shared by all qw
    for(int i=t;i<(2*W-1)*HD;i+=blockDim.x){ Tw[i]=tabW[i]; }
    __syncthreads();
    for(int idx=t;idx<W*Hk;idx+=blockDim.x){ int qw=idx/Hk,kh=idx%Hk; float s=0; for(int c=0;c<HD;c++)s+=Q[qw*HD+c]*Rh[kh*HD+c]; rh[((size_t)(bh*N+qh*W+qw))*Hk+kh]=s; }
    for(int idx=t;idx<W*W;idx+=blockDim.x){ int qw=idx/W,kw=idx%W; const float* tw=Tw+(size_t)((qw-kw)+W-1)*HD; float s=0; for(int c=0;c<HD;c++)s+=Q[qw*HD+c]*tw[c]; rw[((size_t)(bh*N+qh*W+qw))*W+kw]=s; }
}
// S bf16 (scores in -> probs out); row cached in shared (1 read + 1 write of S, not 3 passes).
__global__ void k_biassoftmax(bf16* S,const float* rh,const float* rw,int Bh,int N,int Hsz,int Wsz,float scale){
    int bh=blockIdx.y, qi=blockIdx.x; if(qi>=N)return; bf16* s=S+((size_t)bh*N+qi)*N;
    const float* rhp=rh+((size_t)bh*N+qi)*Hsz; const float* rwp=rw+((size_t)bh*N+qi)*Wsz;
    extern __shared__ float sf[]; float m=-1e30f; __shared__ float wm[32],wl[32];
    for(int j=threadIdx.x;j<N;j+=blockDim.x){float v=b2f(s[j])*scale+rhp[j/Wsz]+rwp[j%Wsz];sf[j]=v;m=fmaxf(m,v);}
    for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_xor_sync(~0u,m,o));
    if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=m; __syncthreads();
    if(threadIdx.x==0){float mm=-1e30f;for(int i=0;i<(blockDim.x+31)/32;i++)mm=fmaxf(mm,wm[i]);wm[0]=mm;} __syncthreads(); m=wm[0];
    float l=0; for(int j=threadIdx.x;j<N;j+=blockDim.x){float e=__expf(sf[j]-m);sf[j]=e;l+=e;}
    for(int o=16;o;o>>=1)l+=__shfl_xor_sync(~0u,l,o); if((threadIdx.x&31)==0)wl[threadIdx.x>>5]=l; __syncthreads();
    if(threadIdx.x==0){float ll=0;for(int i=0;i<(blockDim.x+31)/32;i++)ll+=wl[i];wl[0]=ll;} __syncthreads(); l=wl[0];
    for(int j=threadIdx.x;j<N;j+=blockDim.x)s[j]=f2b(sf[j]/l);
}
__global__ void k_mergeheads(const bf16* O,bf16* out,int B,int N,int Nh){
    int token=blockIdx.x, c=threadIdx.x, h=blockIdx.y; int b=token/N, qi=token%N;
    out[(size_t)token*C + h*HD+c]=O[((size_t)(b*Nh+h)*N+qi)*HD+c];
}
__global__ void k_partition(const bf16* h,bf16* w,int pad,int gs){
    int gs2=gs+pad; int nwin=gs2/WIN; int wtok=blockIdx.x, c=threadIdx.x;
    int wb=wtok/(WIN*WIN), lr=wtok%(WIN*WIN), li=lr/WIN, lj=lr%WIN; int wi=wb/nwin, wj=wb%nwin;
    int oh=wi*WIN+li, ow=wj*WIN+lj; bf16 v=f2b(0.f); if(oh<gs&&ow<gs) v=h[((size_t)oh*gs+ow)*C+c];
    w[(size_t)wtok*C+c]=v;
}
__global__ void k_unpartition(const bf16* w,bf16* h,int pad,int gs){
    int gs2=gs+pad; int nwin=gs2/WIN; int oh=blockIdx.x/gs, ow=blockIdx.x%gs, c=threadIdx.x;
    int wi=oh/WIN, li=oh%WIN, wj=ow/WIN, lj=ow%WIN; int wb=wi*nwin+wj; int wtok=wb*(WIN*WIN)+li*WIN+lj;
    h[((size_t)oh*gs+ow)*C+c]=w[(size_t)wtok*C+c];
}
// LayerNorm2d over channel dim: in[Cc,H,W], per pixel normalize over Cc
// conv via cuBLAS (bf16 tensor cores). 1x1 = direct GEMM; 3x3 = im2col + GEMM. out [Cout,HoWo] CHW.
static bf16* g_col;
__global__ void k_im2col3(const bf16* in,bf16* col,int Cin,int H,int Wd,int s,int p,int Ho,int Wo){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)Cin*9*Ho*Wo)return;
    int M=Ho*Wo, pix=idx%M, k=idx/M; int oy=pix/Wo, ox=pix%Wo; int ic=k/9, r=k%9, ky=r/3, kx=r%3;
    int iy=oy*s-p+ky, ix=ox*s-p+kx;
    col[idx]=(iy<0||iy>=H||ix<0||ix>=Wd)?f2b(0.f):in[((size_t)ic*H+iy)*Wd+ix];
}
static void conv1x1(const bf16* in,const bf16* w,bf16* out,int Cin,int Cout,int HW){
    CB(cublasGemmEx(CUB,CUBLAS_OP_N,CUBLAS_OP_N,HW,Cout,Cin,&ONE_,in,CUDA_R_16BF,HW,w,CUDA_R_16BF,Cin,&ZERO_,out,CUDA_R_16BF,HW,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
static void conv3x3(const bf16* in,const bf16* w,bf16* out,int Cin,int Cout,int H,int Wd,int s,int p,int Ho,int Wo){
    int K=Cin*9, M=Ho*Wo; k_im2col3<<<((size_t)K*M+255)/256,256>>>(in,g_col,Cin,H,Wd,s,p,Ho,Wo);
    CB(cublasGemmEx(CUB,CUBLAS_OP_N,CUBLAS_OP_N,M,Cout,K,&ONE_,g_col,CUDA_R_16BF,M,w,CUDA_R_16BF,K,&ZERO_,out,CUDA_R_16BF,M,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
__global__ void k_ln2d(const bf16* in,bf16* out,const float* w,const float* b,int Cc,int HW,float eps){
    int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=HW)return;
    float m=0; for(int c=0;c<Cc;c++)m+=b2f(in[(size_t)c*HW+p]); m/=Cc;
    float v=0; for(int c=0;c<Cc;c++){float d=b2f(in[(size_t)c*HW+p])-m;v+=d*d;} v=rsqrtf(v/Cc+eps);
    for(int c=0;c<Cc;c++)out[(size_t)c*HW+p]=f2b((b2f(in[(size_t)c*HW+p])-m)*v*w[c]+b[c]);
}
__global__ void k_to_chw(const bf16* hwc,bf16* chw,int gs){int p=blockIdx.x,c=threadIdx.x; chw[(size_t)c*gs*gs+p]=hwc[(size_t)p*C+c];}

// scratch (bf16 activations; rel-pos bias fp32)
static bf16 *g_q,*g_k,*g_v,*g_S,*g_O,*g_qkv,*g_t1,*g_t2,*g_t3,*g_win,*g_win2;
static float *g_rh,*g_rw;
static bf16 *g_a,*g_chw,*g_c1,*g_c2,*g_n2,*g_n3,*g_samx,*g_clipx,*g_fused,*g_proj,*g_vis,*g_samout;
struct Blk{float *n1w,*n1b,*qkvb,*projb,*n2w,*n2b,*l1b,*l2b,*rh,*rw,*rh40,*rw40; bf16 *qkvw,*projw,*l1w,*l2w; int global;};
static Blk BL[12];
static bf16 *PEw,*POS,*NK0w,*NK1w,*N2w,*N3w; static float *PEb,*NK0lw,*NK0lb,*NK1lw,*NK1lb;

static void sam_attn(bf16* x,int B,int N,int Hs,Blk& bl,const float* rh,const float* rw){
    int Bh=B*NH;
    linear(B*N,C,3*C,x,bl.qkvw,nullptr,g_qkv);
    dim3 sg(B*N,NH); k_splitqkv<<<sg,HD>>>(g_qkv,g_q,g_k,g_v,B,N,NH,bl.qkvb);
    size_t rsh=((size_t)Hs*HD + Hs*HD + (2*Hs-1)*HD)*sizeof(float);
    k_relpos<<<dim3(Hs,Bh),256,rsh>>>(g_q,rh,rw,g_rh,g_rw,Bh,N,Hs,Hs);
    // explicit cuBLAS qk^T -> bias+softmax -> Sv (beats hand-written flash here: cuBLAS tiles K/V optimally)
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,N,N,HD,&ONE_,g_k,CUDA_R_16BF,HD,(long long)N*HD,g_q,CUDA_R_16BF,HD,(long long)N*HD,&ZERO_,g_S,CUDA_R_16BF,N,(long long)N*N,Bh,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    dim3 bg(N,Bh); k_biassoftmax<<<bg,256,(size_t)N*sizeof(float)>>>(g_S,g_rh,g_rw,Bh,N,Hs,Hs,1.f/sqrtf((float)HD));
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_N,CUBLAS_OP_N,HD,N,N,&ONE_,g_v,CUDA_R_16BF,HD,(long long)N*HD,g_S,CUDA_R_16BF,N,(long long)N*N,&ZERO_,g_O,CUDA_R_16BF,HD,(long long)N*HD,Bh,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    dim3 mg(B*N,NH); k_mergeheads<<<mg,HD>>>(g_O,g_t2,B,N,NH);
    linear(B*N,C,C,g_t2,bl.projw,bl.projb,x);
}

// SAM forward: image_ori[3,1024,1024] -> sam_out[1024,16,16]. x kept as [GS*GS, C].
static bf16* sam_forward(const float* dimg,int gs,int imgsz,const bf16* pos){
    bf16* x=g_samx; size_t NG=(size_t)gs*gs;
    k_im2col<<<gs*gs,768>>>(dimg,g_t1,gs,imgsz);
    linear(gs*gs,768,C,g_t1,PEw,PEb,x);
    k_vadd<<<(NG*C+255)/256,256>>>(x,pos,NG*C);
    for(int l=0;l<12;l++){ Blk& bl=BL[l];
        k_ln<<<gs*gs,256>>>(x,g_t1,bl.n1w,bl.n1b,gs*gs,C,1e-6f);
        if(bl.global){
            CK(cudaMemcpyAsync(g_a,g_t1,NG*C*2,cudaMemcpyDeviceToDevice));
            sam_attn(g_a,1,gs*gs,gs,bl,(gs==64)?bl.rh:bl.rh40,(gs==64)?bl.rw:bl.rw40);
            k_vadd<<<(NG*C+255)/256,256>>>(x,g_a,NG*C);
        } else {
            int pad=(WIN-gs%WIN)%WIN; int gs2=gs+pad, nwin=gs2/WIN, B=nwin*nwin, N=WIN*WIN;
            k_partition<<<B*N,C>>>(g_t1,g_win,pad,gs);
            sam_attn(g_win,B,N,WIN,bl,bl.rh,bl.rw);
            k_unpartition<<<gs*gs,C>>>(g_win,g_win2,pad,gs);
            k_vadd<<<(NG*C+255)/256,256>>>(x,g_win2,NG*C);
        }
        k_ln<<<gs*gs,256>>>(x,g_t1,bl.n2w,bl.n2b,gs*gs,C,1e-6f);
        linear(gs*gs,C,MLP,g_t1,bl.l1w,bl.l1b,g_t3);
        k_geluE<<<(NG*MLP+255)/256,256>>>(g_t3,NG*MLP);
        linear(gs*gs,MLP,C,g_t3,bl.l2w,bl.l2b,g_t1);
        k_vadd<<<(NG*C+255)/256,256>>>(x,g_t1,NG*C);
    }
    int g2=gs/2, g4=gs/4;                                                        // neck: gs -> gs/2 -> gs/4
    k_to_chw<<<gs*gs,C>>>(x,g_chw,gs);
    conv1x1(g_chw,NK0w,g_c1,768,256,gs*gs);                         k_ln2d<<<(gs*gs+127)/128,128>>>(g_c1,g_c2,NK0lw,NK0lb,256,gs*gs,1e-6f);
    conv3x3(g_c2,NK1w,g_c1,256,256,gs,gs,1,1,gs,gs);               k_ln2d<<<(gs*gs+127)/128,128>>>(g_c1,g_c2,NK1lw,NK1lb,256,gs*gs,1e-6f);
    conv3x3(g_c2,N2w,g_n2,256,512,gs,gs,2,1,g2,g2);                // net_2 -> [512,g2,g2]
    conv3x3(g_n2,N3w,g_n3,512,1024,g2,g2,2,1,g4,g4);              // net_3 -> [1024,g4,g4]
    return g_n3;
}
// ===== batched SAM over B tiles (same-resolution tiles batched -> bigger GEMMs) =====
__global__ void k_im2col_b(const float* imgs,bf16* pat,int gs,int imgsz,int B){
    int tok=blockIdx.x, j=threadIdx.x; int b=tok/(gs*gs), lt=tok%(gs*gs); int oh=lt/gs, ow=lt%gs; int ic=j/256, r=j%256, ky=r/16, kx=r%16;
    const float* img=imgs+(size_t)b*3*imgsz*imgsz;
    pat[(size_t)tok*768+j]=f2b(img[(size_t)ic*imgsz*imgsz+(oh*16+ky)*imgsz+(ow*16+kx)]);
}
__global__ void k_addpos_b(bf16* x,const bf16* pos,int gs,int B){     // add per-tile pos (pos[lt]) to each token
    size_t i=(size_t)blockIdx.x*256+threadIdx.x; if(i>=(size_t)B*gs*gs*C)return; int c=i%C; size_t tok=i/C; int lt=tok%(gs*gs);
    x[i]=f2b(b2f(x[i])+b2f(pos[(size_t)lt*C+c]));
}
__global__ void k_partition_b(const bf16* h,bf16* w,int pad,int gs,int B){
    int gs2=gs+pad,nwin=gs2/WIN; int wtok=blockIdx.x, c=threadIdx.x; int per=nwin*nwin*WIN*WIN; int b=wtok/per, wt=wtok%per;
    int wb=wt/(WIN*WIN), lr=wt%(WIN*WIN), li=lr/WIN, lj=lr%WIN; int wi=wb/nwin, wj=wb%nwin; int oh=wi*WIN+li, ow=wj*WIN+lj;
    bf16 v=f2b(0.f); if(oh<gs&&ow<gs) v=h[((size_t)b*gs*gs+oh*gs+ow)*C+c]; w[(size_t)wtok*C+c]=v;
}
__global__ void k_unpartition_b(const bf16* w,bf16* h,int pad,int gs,int B){
    int gs2=gs+pad,nwin=gs2/WIN; size_t idx=blockIdx.x; int b=idx/(gs*gs), p=idx%(gs*gs); int oh=p/gs, ow=p%gs, c=threadIdx.x;
    int wi=oh/WIN, li=oh%WIN, wj=ow/WIN, lj=ow%WIN; int wb=wi*nwin+wj; int wtok=b*(nwin*nwin*WIN*WIN)+wb*(WIN*WIN)+li*WIN+lj;
    h[(size_t)idx*C+c]=w[(size_t)wtok*C+c];
}
// B tiles in[B,3,imgsz,imgsz] -> g_samout[B,1024,g4*g4]
static bf16* sam_forward_batch(const float* in,int B,int gs,int imgsz,const bf16* pos){
    bf16* x=g_samx; size_t NG=(size_t)gs*gs, BN=(size_t)B*NG;
    k_im2col_b<<<B*gs*gs,768>>>(in,g_t1,gs,imgsz,B);
    linear(BN,768,C,g_t1,PEw,PEb,x);
    k_addpos_b<<<(BN*C+255)/256,256>>>(x,pos,gs,B);
    for(int l=0;l<12;l++){ Blk& bl=BL[l];
        k_ln<<<BN,256>>>(x,g_t1,bl.n1w,bl.n1b,BN,C,1e-6f);
        if(bl.global){
            CK(cudaMemcpyAsync(g_a,g_t1,BN*C*2,cudaMemcpyDeviceToDevice));
            sam_attn(g_a,B,gs*gs,gs,bl,(gs==64)?bl.rh:bl.rh40,(gs==64)?bl.rw:bl.rw40);
            k_vadd<<<(BN*C+255)/256,256>>>(x,g_a,BN*C);
        } else {
            int pad=(WIN-gs%WIN)%WIN; int gs2=gs+pad,nwin=gs2/WIN,Bw=B*nwin*nwin,N=WIN*WIN;
            k_partition_b<<<Bw*N,C>>>(g_t1,g_win,pad,gs,B);
            sam_attn(g_win,Bw,N,WIN,bl,bl.rh,bl.rw);
            k_unpartition_b<<<BN,C>>>(g_win,g_win2,pad,gs,B);
            k_vadd<<<(BN*C+255)/256,256>>>(x,g_win2,BN*C);
        }
        k_ln<<<BN,256>>>(x,g_t1,bl.n2w,bl.n2b,BN,C,1e-6f);
        linear(BN,C,MLP,g_t1,bl.l1w,nullptr,g_t3);
        k_geluE_bias<<<(BN*MLP+255)/256,256>>>(g_t3,bl.l1b,MLP,BN*MLP);
        linear(BN,MLP,C,g_t3,bl.l2w,bl.l2b,g_t1);
        k_vadd<<<(BN*C+255)/256,256>>>(x,g_t1,BN*C);
    }
    int g2=gs/2,g4=gs/4;                                              // neck per tile (cheap spatial conv)
    for(int b=0;b<B;b++){ bf16* xb=x+(size_t)b*NG*C;
        k_to_chw<<<gs*gs,C>>>(xb,g_chw,gs);
        conv1x1(g_chw,NK0w,g_c1,768,256,gs*gs); k_ln2d<<<(gs*gs+127)/128,128>>>(g_c1,g_c2,NK0lw,NK0lb,256,gs*gs,1e-6f);
        conv3x3(g_c2,NK1w,g_c1,256,256,gs,gs,1,1,gs,gs); k_ln2d<<<(gs*gs+127)/128,128>>>(g_c1,g_c2,NK1lw,NK1lb,256,gs*gs,1e-6f);
        conv3x3(g_c2,N2w,g_n2,256,512,gs,gs,2,1,g2,g2);
        conv3x3(g_n2,N3w,g_samout+(size_t)b*1024*g4*g4,512,1024,g2,g2,2,1,g4,g4);
    }
    return g_samout;
}
// ================= CLIP-L =================
static const int CN=257,CC=1024,CNH=16,CHD=64,CMLP=4096,CL=24;
struct CBlk{float *n1w,*n1b,*qb,*ob,*n2w,*n2b,*f1b,*f2b; bf16 *qw,*ow,*f1w,*f2w;};
static CBlk CBL[24]; static bf16 *CLS,*POSC; static float *PREw,*PREb;
__global__ void k_quickgelu(bf16* x,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n){float v=b2f(x[i]);x[i]=f2b(v/(1.f+__expf(-1.702f*v)));}}
__global__ void k_quickgelu_bias(bf16* x,const float* b,int N,size_t n){size_t i=blockIdx.x*256+threadIdx.x;if(i<n){float v=b2f(x[i])+b[i%N];x[i]=f2b(v/(1.f+__expf(-1.702f*v)));}}
__global__ void k_clip_emb(const bf16* sam,const bf16* cls,const bf16* pos,bf16* out,int ntok){
    int t=blockIdx.x,c=threadIdx.x; float v=(t==0)?b2f(cls[c]):b2f(sam[(size_t)c*ntok+(t-1)]);
    out[(size_t)t*CC+c]=f2b(v+b2f(pos[(size_t)t*CC+c]));
}
__global__ void k_splitqkv_c(const bf16* qkv,bf16* q,bf16* k,bf16* v,int N,const float* bias){
    int qi=blockIdx.x,c=threadIdx.x,h=blockIdx.y; const bf16* base=qkv+(size_t)qi*3*CC+h*CHD;
    size_t o=((size_t)h*N+qi)*CHD+c; q[o]=f2b(b2f(base[c])+bias[h*CHD+c]);k[o]=f2b(b2f(base[CC+c])+bias[CC+h*CHD+c]);v[o]=f2b(b2f(base[2*CC+c])+bias[2*CC+h*CHD+c]);
}
__global__ void k_mergeheads_c(const bf16* O,bf16* out,int N){
    int qi=blockIdx.x,c=threadIdx.x,h=blockIdx.y; out[(size_t)qi*CC+h*CHD+c]=O[((size_t)h*N+qi)*CHD+c];
}
__global__ void k_softmax(bf16* S,int N,float scale){
    int bh=blockIdx.y,qi=blockIdx.x; bf16* s=S+((size_t)bh*N+qi)*N; float m=-1e30f; __shared__ float wm[32],wl[32];
    for(int j=threadIdx.x;j<N;j+=blockDim.x){float v=b2f(s[j])*scale;s[j]=f2b(v);m=fmaxf(m,v);}
    for(int o=16;o;o>>=1)m=fmaxf(m,__shfl_xor_sync(~0u,m,o));
    if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=m;__syncthreads();
    if(threadIdx.x==0){float mm=-1e30f;for(int i=0;i<(blockDim.x+31)/32;i++)mm=fmaxf(mm,wm[i]);wm[0]=mm;}__syncthreads();m=wm[0];
    float l=0;for(int j=threadIdx.x;j<N;j+=blockDim.x){float e=__expf(b2f(s[j])-m);s[j]=f2b(e);l+=e;}
    for(int o=16;o;o>>=1)l+=__shfl_xor_sync(~0u,l,o); if((threadIdx.x&31)==0)wl[threadIdx.x>>5]=l;__syncthreads();
    if(threadIdx.x==0){float ll=0;for(int i=0;i<(blockDim.x+31)/32;i++)ll+=wl[i];wl[0]=ll;}__syncthreads();l=wl[0];
    for(int j=threadIdx.x;j<N;j+=blockDim.x)s[j]=f2b(b2f(s[j])/l);
}
static void clip_attn(const bf16* in,bf16* out,CBlk& b,int ncn){
    linear(ncn,CC,3*CC,in,b.qw,nullptr,g_qkv);
    dim3 sg(ncn,CNH); k_splitqkv_c<<<sg,CHD>>>(g_qkv,g_q,g_k,g_v,ncn,b.qb);
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,ncn,ncn,CHD,&ONE_,g_k,CUDA_R_16BF,CHD,(long long)ncn*CHD,g_q,CUDA_R_16BF,CHD,(long long)ncn*CHD,&ZERO_,g_S,CUDA_R_16BF,ncn,(long long)ncn*ncn,CNH,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    dim3 bg(ncn,CNH); k_softmax<<<bg,256>>>(g_S,ncn,1.f/sqrtf((float)CHD));
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_N,CUBLAS_OP_N,CHD,ncn,ncn,&ONE_,g_v,CUDA_R_16BF,CHD,(long long)ncn*CHD,g_S,CUDA_R_16BF,ncn,(long long)ncn*ncn,&ZERO_,g_O,CUDA_R_16BF,CHD,(long long)ncn*CHD,CNH,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    k_mergeheads_c<<<sg,CHD>>>(g_O,g_t3,ncn);
    linear(ncn,CC,CC,g_t3,b.ow,b.ob,out);
}
static bf16* clip_forward(const bf16* sam_out,int ntok,const bf16* posc){
    int ncn=ntok+1; size_t NC=(size_t)ncn*CC; bf16* x=g_clipx;
    k_clip_emb<<<ncn,CC>>>(sam_out,CLS,posc,x,ntok);
    k_ln<<<ncn,256>>>(x,g_t2,PREw,PREb,ncn,CC,1e-5f); CK(cudaMemcpyAsync(x,g_t2,NC*2,cudaMemcpyDeviceToDevice,VS));
    for(int l=0;l<CL;l++){ CBlk& b=CBL[l];
        k_ln<<<ncn,256>>>(x,g_t2,b.n1w,b.n1b,ncn,CC,1e-5f);
        clip_attn(g_t2,g_t1,b,ncn); k_vadd<<<(NC+255)/256,256>>>(x,g_t1,NC);
        k_ln<<<ncn,256>>>(x,g_t2,b.n2w,b.n2b,ncn,CC,1e-5f);
        linear(ncn,CC,CMLP,g_t2,b.f1w,b.f1b,g_t3); k_quickgelu<<<((size_t)ncn*CMLP+255)/256,256>>>(g_t3,(size_t)ncn*CMLP);
        linear(ncn,CMLP,CC,g_t3,b.f2w,b.f2b,g_t1); k_vadd<<<(NC+255)/256,256>>>(x,g_t1,NC);
    }
    return x;
}

// ================= projector + token assembly =================
static bf16 *PROJw,*NEWLINE,*VSEP; static float *PROJb;
__global__ void k_fuse(const bf16* clip,const bf16* sam,bf16* out,int ntok){ // [ntok,2048]=cat(clip[1:],sam_flat)
    size_t i=(size_t)blockIdx.x*256+threadIdx.x; if(i>=(size_t)ntok*2048)return; int t=i/2048,c=i%2048;
    out[i]=(c<1024)? clip[(size_t)(1+t)*CC+c] : sam[(size_t)(c-1024)*ntok+t];
}
__global__ void k_assemble(const bf16* proj,const bf16* nl,const bf16* sep,bf16* out){
    int tok=blockIdx.x,c=blockIdx.y*256+threadIdx.x; if(c>=1280)return;
    if(tok==272){out[(size_t)tok*1280+c]=sep[c];return;}
    int gy=tok/17,r=tok%17;
    out[(size_t)tok*1280+c]=(r==16)? nl[c] : proj[(size_t)(gy*16+r)*1280+c];
}
static bf16* project(const bf16* clip,const bf16* sam,int ntok){    // -> [ntok,1280]
    k_fuse<<<((size_t)ntok*2048+255)/256,256>>>(clip,sam,g_fused,ntok);
    linear(ntok,2048,1280,g_fused,PROJw,PROJb,g_proj);
    return g_proj;
}
static bf16* project_assemble(const bf16* clip,const bf16* sam,bf16** proj_out){   // BASE/global: -> 273 assembled
    bf16* pj=project(clip,sam,256); *proj_out=pj;
    k_assemble<<<dim3(273,5),256>>>(pj,NEWLINE,VSEP,g_vis);
    return g_vis;
}

// ===== batched CLIP over B tiles =====
__global__ void k_clip_emb_b(const bf16* sam,const bf16* cls,const bf16* pos,bf16* out,int ntok,int B){
    int t=blockIdx.x,c=threadIdx.x; int ncn=ntok+1; int b=t/ncn, lt=t%ncn;
    float v=(lt==0)?b2f(cls[c]):b2f(sam[(size_t)b*CC*ntok+(size_t)c*ntok+(lt-1)]);
    out[(size_t)t*CC+c]=f2b(v+b2f(pos[(size_t)lt*CC+c]));
}
__global__ void k_splitqkv_c_b(const bf16* qkv,bf16* q,bf16* k,bf16* v,int ncn,int B,const float* bias){
    int qi=blockIdx.x,c=threadIdx.x,h=blockIdx.y; int b=qi/ncn,lq=qi%ncn;
    const bf16* base=qkv+(size_t)qi*3*CC+h*CHD; size_t o=((size_t)(b*CNH+h)*ncn+lq)*CHD+c;
    q[o]=f2b(b2f(base[c])+bias[h*CHD+c]);k[o]=f2b(b2f(base[CC+c])+bias[CC+h*CHD+c]);v[o]=f2b(b2f(base[2*CC+c])+bias[2*CC+h*CHD+c]);
}
__global__ void k_mergeheads_c_b(const bf16* O,bf16* out,int ncn,int B){
    int qi=blockIdx.x,c=threadIdx.x,h=blockIdx.y; int b=qi/ncn,lq=qi%ncn;
    out[(size_t)qi*CC+h*CHD+c]=O[((size_t)(b*CNH+h)*ncn+lq)*CHD+c];
}
static void clip_attn_batch(const bf16* in,bf16* out,CBlk& bl,int ncn,int B){
    int BN=B*ncn, Bh=B*CNH;
    linear(BN,CC,3*CC,in,bl.qw,nullptr,g_qkv);
    dim3 sg(BN,CNH); k_splitqkv_c_b<<<sg,CHD>>>(g_qkv,g_q,g_k,g_v,ncn,B,bl.qb);
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,ncn,ncn,CHD,&ONE_,g_k,CUDA_R_16BF,CHD,(long long)ncn*CHD,g_q,CUDA_R_16BF,CHD,(long long)ncn*CHD,&ZERO_,g_S,CUDA_R_16BF,ncn,(long long)ncn*ncn,Bh,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    dim3 bg(ncn,Bh); k_softmax<<<bg,256>>>(g_S,ncn,1.f/sqrtf((float)CHD));
    CB(cublasGemmStridedBatchedEx(CUB,CUBLAS_OP_N,CUBLAS_OP_N,CHD,ncn,ncn,&ONE_,g_v,CUDA_R_16BF,CHD,(long long)ncn*CHD,g_S,CUDA_R_16BF,ncn,(long long)ncn*ncn,&ZERO_,g_O,CUDA_R_16BF,CHD,(long long)ncn*CHD,Bh,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    k_mergeheads_c_b<<<sg,CHD>>>(g_O,g_t3,ncn,B);
    linear(BN,CC,CC,g_t3,bl.ow,bl.ob,out);
}
static bf16* clip_forward_batch(const bf16* sam,int B,int ntok,const bf16* posc){
    int ncn=ntok+1; size_t BN=(size_t)B*ncn, NC=BN*CC; bf16* x=g_clipx;
    k_clip_emb_b<<<B*ncn,CC>>>(sam,CLS,posc,x,ntok,B);
    k_ln<<<BN,256>>>(x,g_t2,PREw,PREb,BN,CC,1e-5f); CK(cudaMemcpyAsync(x,g_t2,NC*2,cudaMemcpyDeviceToDevice,VS));
    for(int l=0;l<CL;l++){ CBlk& b=CBL[l];
        k_ln<<<BN,256>>>(x,g_t2,b.n1w,b.n1b,BN,CC,1e-5f);
        clip_attn_batch(g_t2,g_t1,b,ncn,B); k_vadd<<<(NC+255)/256,256>>>(x,g_t1,NC);
        k_ln<<<BN,256>>>(x,g_t2,b.n2w,b.n2b,BN,CC,1e-5f);
        linear(BN,CC,CMLP,g_t2,b.f1w,nullptr,g_t3); k_quickgelu_bias<<<(BN*CMLP+255)/256,256>>>(g_t3,b.f1b,CMLP,BN*CMLP);
        linear(BN,CMLP,CC,g_t3,b.f2w,b.f2b,g_t1); k_vadd<<<(NC+255)/256,256>>>(x,g_t1,NC);
    }
    return x;
}
__global__ void k_fuse_b(const bf16* clip,const bf16* sam,bf16* out,int ntok,int B){
    size_t i=(size_t)blockIdx.x*256+threadIdx.x; if(i>=(size_t)B*ntok*2048)return; int ncn=ntok+1;
    int c=i%2048; size_t r=i/2048; int t=r%ntok; int b=r/ntok;
    out[i]=(c<1024)? clip[((size_t)b*ncn+(1+t))*CC+c] : sam[(size_t)b*CC*ntok+(size_t)(c-1024)*ntok+t];
}
static bf16* project_batch(const bf16* clip,const bf16* sam,int B,int ntok){
    k_fuse_b<<<((size_t)B*ntok*2048+255)/256,256>>>(clip,sam,g_fused,ntok,B);
    linear(B*ntok,2048,1280,g_fused,PROJw,PROJb,g_proj);
    return g_proj;
}
static double cmp(const char* fx,const bf16* dev,size_t n){
    std::vector<bf16> mb(n); CK(cudaMemcpy(mb.data(),dev,n*2,cudaMemcpyDeviceToHost));
    std::vector<float> mine(n); for(size_t i=0;i<n;i++)mine[i]=__bfloat162float(mb[i]);
    std::string p=std::string("/home/janitor/unlimited-ocr/engine/vfix/")+fx; FILE* f=fopen(p.c_str(),"rb");
    if(!f){printf("  (no %s)\n",fx);return -1;} std::vector<float> ref(n); if(fread(ref.data(),4,n,f)!=n){} fclose(f);
    double md=0,sum=0; for(size_t i=0;i<n;i++){double d=fabs(mine[i]-ref[i]);md=d>md?d:md;sum+=d;}
    printf("  %-14s max_abs=%.4f mean_abs=%.5f  mine[0..2]=%.4f %.4f %.4f ref=%.4f %.4f %.4f\n",fx,md,sum/n,mine[0],mine[1],mine[2],ref[0],ref[1],ref[2]);
    return md;
}

static bf16 *POS40=nullptr,*POSC100=nullptr;   // precomputed bicubic-interpolated 640-tile pos embeds (Gundam)
static const char* GD="/home/janitor/unlimited-ocr/engine/gundam/";
static float* load_f32bin(const char* fn,size_t n){
    std::string p=std::string(GD)+fn; FILE* f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"missing %s\n",p.c_str());exit(1);}
    std::vector<float> h(n); if(fread(h.data(),4,n,f)!=n){} fclose(f);
    float* d; CK(cudaMalloc(&d,n*4)); CK(cudaMemcpy(d,h.data(),n*4,cudaMemcpyHostToDevice)); return d;
}
static bf16* load_posbin(const char* fn,size_t n){
    std::string p=std::string(GD)+fn; FILE* f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"missing %s\n",p.c_str());exit(1);}
    std::vector<float> h(n); if(fread(h.data(),4,n,f)!=n){} fclose(f);
    std::vector<bf16> hb(n); for(size_t i=0;i<n;i++)hb[i]=__float2bfloat16(h[i]);
    bf16* d; CK(cudaMalloc(&d,n*2)); CK(cudaMemcpy(d,hb.data(),n*2,cudaMemcpyHostToDevice)); return d;
}
static double cmp_gd(const char* fn,const bf16* dev,size_t n){
    std::vector<bf16> mb(n); CK(cudaMemcpy(mb.data(),dev,n*2,cudaMemcpyDeviceToHost));
    std::string p=std::string(GD)+fn; FILE* f=fopen(p.c_str(),"rb"); std::vector<float> ref(n); if(fread(ref.data(),4,n,f)!=n){} fclose(f);
    double md=0,sum=0; for(size_t i=0;i<n;i++){double d=fabs(__bfloat162float(mb[i])-ref[i]);md=d>md?d:md;sum+=d;}
    printf("  %-20s max_abs=%.4f mean_abs=%.5f  mine[0..2]=%.4f %.4f %.4f  ref=%.4f %.4f %.4f\n",fn,md,sum/n,
        __bfloat162float(mb[0]),__bfloat162float(mb[1]),__bfloat162float(mb[2]),ref[0],ref[1],ref[2]);
    return md;
}
static bf16 *POS40_t,*POSC100_t;
static bf16 g_dummy;
static bf16 g_vinit_done;
static bool g_vinit=false;
void init_vision(){
    if(g_vinit)return; g_vinit=true;
    CB(cublasCreate(&CUB)); CB(cublasSetMathMode(CUB,CUBLAS_TF32_TENSOR_OP_MATH)); // TF32 tensor cores
    CK(cudaFuncSetAttribute(k_relpos,cudaFuncAttributeMaxDynamicSharedMemorySize,100000)); // rel-pos uses ~65KB shared (global)
    ST.load("/home/janitor/unlimited-ocr/engine/manifest.tsv");
    const std::string P="model.sam_model.";
    PEw=Wb(P+"patch_embed.proj.weight"); PEb=W(P+"patch_embed.proj.bias"); POS=Wb(P+"pos_embed");
    int gl[4]={2,5,8,11};
    for(int l=0;l<12;l++){ Blk&b=BL[l]; std::string B=P+"blocks."+std::to_string(l)+".";
        b.n1w=W(B+"norm1.weight");b.n1b=W(B+"norm1.bias");b.qkvw=Wb(B+"attn.qkv.weight");b.qkvb=W(B+"attn.qkv.bias");
        b.projw=Wb(B+"attn.proj.weight");b.projb=W(B+"attn.proj.bias");b.n2w=W(B+"norm2.weight");b.n2b=W(B+"norm2.bias");
        b.l1w=Wb(B+"mlp.lin1.weight");b.l1b=W(B+"mlp.lin1.bias");b.l2w=Wb(B+"mlp.lin2.weight");b.l2b=W(B+"mlp.lin2.bias");
        b.rh=W(B+"attn.rel_pos_h");b.rw=W(B+"attn.rel_pos_w"); b.global=0; for(int g=0;g<4;g++)if(gl[g]==l)b.global=1;
    }
    NK0w=Wb(P+"neck.0.weight");NK0lw=W(P+"neck.1.weight");NK0lb=W(P+"neck.1.bias");
    NK1w=Wb(P+"neck.2.weight");NK1lw=W(P+"neck.3.weight");NK1lb=W(P+"neck.3.bias");
    N2w=Wb(P+"net_2.weight");N3w=Wb(P+"net_3.weight");
    int MAXN=GS*GS, MAXBh=NH; size_t MAXTOK=11000, MAXBhN=130000;   // sized for up to GBATCH=6 tiles batched (9600 tok)
    CK(cudaMalloc(&g_qkv,MAXTOK*3*C*2)); CK(cudaMalloc(&g_q,MAXBhN*HD*2)); CK(cudaMalloc(&g_k,MAXBhN*HD*2));
    CK(cudaMalloc(&g_v,MAXBhN*HD*2)); CK(cudaMalloc(&g_S,(size_t)MAXBh*MAXN*MAXN*2)); CK(cudaMalloc(&g_O,MAXBhN*HD*2));
    CK(cudaMalloc(&g_rh,MAXBhN*GS*4)); CK(cudaMalloc(&g_rw,MAXBhN*GS*4));  // rel-pos bias fp32 (Bh*N*Hk)
    CK(cudaMalloc(&g_t1,MAXTOK*MLP*2)); CK(cudaMalloc(&g_t2,MAXTOK*C*2)); CK(cudaMalloc(&g_t3,MAXTOK*MLP*2));
    CK(cudaMalloc(&g_win,(size_t)12000*C*2)); CK(cudaMalloc(&g_win2,MAXTOK*C*2));
    CK(cudaMalloc(&g_a,MAXTOK*C*2)); CK(cudaMalloc(&g_chw,(size_t)C*GS*GS*2)); CK(cudaMalloc(&g_c1,(size_t)256*GS*GS*2));
    CK(cudaMalloc(&g_c2,(size_t)256*GS*GS*2)); CK(cudaMalloc(&g_n2,(size_t)512*32*32*2)); CK(cudaMalloc(&g_n3,(size_t)1024*16*16*2));
    CK(cudaMalloc(&g_dimg,(size_t)3*IMG*IMG*4)); CK(cudaMalloc(&g_samx,MAXTOK*C*2)); CK(cudaMalloc(&g_clipx,(size_t)700*CC*2));
    CK(cudaMalloc(&g_fused,(size_t)700*2048*2)); CK(cudaMalloc(&g_proj,(size_t)700*1280*2)); CK(cudaMalloc(&g_vis,(size_t)273*1280*2));
    CK(cudaMalloc(&g_samout,(size_t)6*1024*256*2));   // batched SAM neck output [B,1024,g4*g4]
    CK(cudaMalloc(&g_col,(size_t)2304*4096*2));   // im2col scratch (max: neck conv3x3 256*9 x 64*64)
    VS=cudaStreamPerThread; CB(cublasSetStream(CUB,VS));   // per-thread default stream (capturable); compile -default-stream per-thread
    const std::string Q="model.vision_model.";
    CLS=Wb(Q+"embeddings.class_embedding"); POSC=Wb(Q+"embeddings.position_embedding.weight");
    PREw=W(Q+"pre_layrnorm.weight"); PREb=W(Q+"pre_layrnorm.bias");
    for(int l=0;l<CL;l++){ CBlk& b=CBL[l]; std::string B=Q+"transformer.layers."+std::to_string(l)+".";
        b.n1w=W(B+"layer_norm1.weight");b.n1b=W(B+"layer_norm1.bias");b.qw=Wb(B+"self_attn.qkv_proj.weight");b.qb=W(B+"self_attn.qkv_proj.bias");
        b.ow=Wb(B+"self_attn.out_proj.weight");b.ob=W(B+"self_attn.out_proj.bias");b.n2w=W(B+"layer_norm2.weight");b.n2b=W(B+"layer_norm2.bias");
        b.f1w=Wb(B+"mlp.fc1.weight");b.f1b=W(B+"mlp.fc1.bias");b.f2w=Wb(B+"mlp.fc2.weight");b.f2b=W(B+"mlp.fc2.bias");
    }
    PROJw=Wb("model.projector.layers.weight"); PROJb=W("model.projector.layers.bias");
    NEWLINE=Wb("model.image_newline"); VSEP=Wb("model.view_seperator");
    POS40=load_posbin("sam_pos40.bin",(size_t)1600*768); POSC100=load_posbin("clip_pos100.bin",(size_t)101*1024);
    { int gl2[4]={2,5,8,11}; char fn[64]; for(int g=0;g<4;g++){ int l=gl2[g];
        sprintf(fn,"relpos_h40_b%d.bin",l); BL[l].rh40=load_f32bin(fn,(size_t)79*64);
        sprintf(fn,"relpos_w40_b%d.bin",l); BL[l].rw40=load_f32bin(fn,(size_t)79*64); } }
}
// PDF page -> 273 visual token embeddings [273,1280] bf16 (device).
static cudaGraphExec_t g_vgraph=nullptr;
static void vision_gpu(){ bf16* s=sam_forward(g_dimg,GS,IMG,POS); bf16* c=clip_forward(s,256,POSC); bf16* pj; project_assemble(c,s,&pj); }
static void ensure_graph(){
    if(g_vgraph)return;
    vision_gpu(); CK(cudaStreamSynchronize(VS));    // warm up cuBLAS/JIT outside capture (g_dimg must be valid)
    cudaGraph_t g; CK(cudaStreamBeginCapture(VS,cudaStreamCaptureModeThreadLocal));
    vision_gpu();
    CK(cudaStreamEndCapture(VS,&g)); CK(cudaGraphInstantiate(&g_vgraph,g,nullptr,nullptr,0));
}
bf16* vision_encode(const char* pdf,int page){   // single-page convenience entry (PDF page -> 273 visual tokens)
    init_vision(); render_preprocess(pdf,page,120.f); ensure_graph();
    CK(cudaGraphLaunch(g_vgraph,VS)); CK(cudaStreamSynchronize(VS));
    return g_vis;
}
// ---- Gundam tiling assembly ----
static bf16 *g_projs=nullptr,*g_gundam=nullptr;   // [P*100,1280] tile projs ; [maxtok,1280] assembled
__global__ void k_gundam_local(const bf16* projs,const bf16* nl,bf16* out,int w,int h){
    int tok=blockIdx.x,c=blockIdx.y*256+threadIdx.x; if(c>=1280)return; int W=w*10+1;
    int R=tok/W, Cc=tok%W;
    if(Cc==w*10){ out[(size_t)tok*1280+c]=nl[c]; return; }                       // row separator
    int hi=R/10, wi=Cc/10, ry=R%10, rx=Cc%10; int p=hi*w+wi, loc=ry*10+rx;       // tile p, local 10x10 pos
    out[(size_t)tok*1280+c]=projs[((size_t)p*100+loc)*1280+c];
}
// global_in[3,1024,1024], tiles_in[P,3,640,640] (device fp32) -> g_gundam [ntok,1280]; returns ntok
static int gundam_assemble(const float* global_in,const float* tiles_in,int w,int h,int P){
    if(!g_projs){ CK(cudaMalloc(&g_projs,(size_t)32*100*1280*2)); CK(cudaMalloc(&g_gundam,(size_t)4096*1280*2)); }
    const int CHUNK=6;                                                          // batch tiles through SAM (bigger GEMMs)
    for(int c0=0;c0<P;c0+=CHUNK){ int B=(P-c0<CHUNK)?(P-c0):CHUNK;
        bf16* sams=sam_forward_batch(tiles_in+(size_t)c0*3*640*640,B,40,640,POS40);   // [B,1024,100]
        bf16* cl=clip_forward_batch(sams,B,100,POSC100); bf16* pj=project_batch(cl,sams,B,100);
        CK(cudaMemcpyAsync(g_projs+(size_t)c0*100*1280,pj,(size_t)B*100*1280*2,cudaMemcpyDeviceToDevice,VS));
    }
    int Lloc=h*10*(w*10+1);
    k_gundam_local<<<dim3(Lloc,5),256,0,VS>>>(g_projs,NEWLINE,g_gundam,w,h);     // local grid + row seps
    bf16* gs=sam_forward(global_in,64,1024,POS); bf16* gc=clip_forward(gs,256,POSC);
    bf16* gpj; bf16* gvis=project_assemble(gc,gs,&gpj);                          // global Base 273 (in g_vis)
    CK(cudaMemcpyAsync(g_gundam+(size_t)Lloc*1280,gvis,(size_t)273*1280*2,cudaMemcpyDeviceToDevice,VS));
    return Lloc+273;
}
// ---- Gundam verification vs HF fixtures ----
void gundam_vfix(){
    init_vision();
    int w,h,P,N; { FILE* f=fopen((std::string(GD)+"ref_meta.txt").c_str(),"r"); fscanf(f,"%d %d %d %d",&w,&h,&P,&N); fclose(f); }
    printf("meta w=%d h=%d P=%d N=%d\n",w,h,P,N);
    float *gin,*tin; CK(cudaMalloc(&gin,(size_t)3*1024*1024*4)); CK(cudaMalloc(&tin,(size_t)P*3*640*640*4));
    { size_t n=(size_t)3*1024*1024; std::vector<float> hbuf(n); FILE* f=fopen((std::string(GD)+"ref_global_in.bin").c_str(),"rb"); if(fread(hbuf.data(),4,n,f)!=n){} fclose(f); CK(cudaMemcpy(gin,hbuf.data(),n*4,cudaMemcpyHostToDevice)); }
    { size_t n=(size_t)P*3*640*640; std::vector<float> hbuf(n); FILE* f=fopen((std::string(GD)+"ref_tiles_in.bin").c_str(),"rb"); if(fread(hbuf.data(),4,n,f)!=n){} fclose(f); CK(cudaMemcpy(tin,hbuf.data(),n*4,cudaMemcpyHostToDevice)); }
    int nt=gundam_assemble(gin,tin,w,h,P); CK(cudaStreamSynchronize(VS));
    printf("assembled %d tokens (ref %d)\n",nt,N);
    cmp_gd("ref_assembled.bin",g_gundam,(size_t)N*1280);
}
// ---- split render(CPU) / GPU for multi-page render/compute interleaving ----
static std::vector<unsigned char> g_hostrgb; static int g_hw[2];
static fz_context* g_ctx=nullptr; static fz_document* g_doc=nullptr; static std::string g_docpath;
void vis_render_cpu(const char* pdf,int page){   // CPU only: MuPDF rasterize -> host RGB (overlaps GPU). Single-threaded ctx.
    if(!g_ctx){ g_ctx=fz_new_context(NULL,NULL,FZ_STORE_UNLIMITED); fz_register_document_handlers(g_ctx); fz_set_aa_level(g_ctx,2); }
    if(g_docpath!=pdf){ if(g_doc)fz_drop_document(g_ctx,g_doc); g_doc=fz_open_document(g_ctx,pdf); g_docpath=pdf; }  // open once, reuse
    fz_matrix mat=fz_scale(120.f/72.f,120.f/72.f);
    fz_pixmap* pix=fz_new_pixmap_from_page_number(g_ctx,g_doc,page,mat,fz_device_rgb(g_ctx),0);
    int w=fz_pixmap_width(g_ctx,pix),h=fz_pixmap_height(g_ctx,pix); unsigned char* sm=fz_pixmap_samples(g_ctx,pix);
    g_hostrgb.assign(sm,sm+(size_t)w*h*3); g_hw[0]=w; g_hw[1]=h;
    fz_drop_pixmap(g_ctx,pix);
}
void vis_upload(){   // host RGB -> g_dimg (GPU, small)
    init_vision(); int w=g_hw[0],h=g_hw[1];
    if((size_t)w*h*3>g_dsrcn){ if(g_dsrc)cudaFree(g_dsrc); CK(cudaMalloc(&g_dsrc,(size_t)w*h*3)); g_dsrcn=(size_t)w*h*3; }
    CK(cudaMemcpyAsync(g_dsrc,g_hostrgb.data(),(size_t)w*h*3,cudaMemcpyHostToDevice,VS));
    float sc=fminf((float)IMG/w,(float)IMG/h); int nw=(int)lroundf(w*sc),nh=(int)lroundf(h*sc),offx=(IMG-nw)/2,offy=(IMG-nh)/2;
    dim3 b(16,16),g((IMG+15)/16,(IMG+15)/16); k_resize_norm<<<g,b>>>(g_dsrc,h,w,nw,nh,offx,offy,g_dimg);
}
void vis_gpu_launch(){ init_vision(); ensure_graph(); CK(cudaGraphLaunch(g_vgraph,VS)); }  // async GPU forward
void vis_gpu_sync(){ CK(cudaStreamSynchronize(VS)); }
bf16* vis_result(){ return g_vis; }

// ---- Gundam tiling preprocessing (PDF -> P x [3,640,640] device fp32) ----
#include <algorithm>
#include <set>
static int g_gw,g_gh,g_gP; static unsigned char* g_dtiles=nullptr; static size_t g_dtilesn=0; static float* g_tilesf=nullptr;
static void gundam_ratio(float pw,float ph,int* ow,int* oh){
    float ar=pw/ph; const int mn=2,mx=32; std::set<std::pair<int,int>> uniq;
    for(int n=mn;n<=mx;n++)for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){int b=i*j; if(b>=mn&&b<=mx)uniq.insert({i,j});}
    std::vector<std::pair<int,int>> rs(uniq.begin(),uniq.end());
    std::stable_sort(rs.begin(),rs.end(),[](const std::pair<int,int>&a,const std::pair<int,int>&b){return a.first*a.second<b.first*b.second;});
    float bd=1e9f; int bw=1,bh=1; double area=(double)pw*ph;
    for(auto&r:rs){ float d=fabsf(ar-(float)r.first/r.second);
        if(d<bd){bd=d;bw=r.first;bh=r.second;}
        else if(d==bd && area>0.5*640.0*640.0*r.first*r.second){bw=r.first;bh=r.second;} }
    *ow=bw;*oh=bh;
}
__global__ void k_gundam_tiles(const unsigned char* src,int sw,int w,float* out){    // src[sh,sw,3] -> out[P,3,640,640]
    size_t i=(size_t)blockIdx.x*256+threadIdx.x; size_t TOT=(size_t)gridDim.x*256; // bound checked by caller via P
    int x=i%640; size_t t=i/640; int y=t%640; t/=640; int c=t%3; int p=t/3;
    int row=p/w, col=p%w; int sy=row*640+y, sx=col*640+x;
    out[((size_t)(p*3+c)*640+y)*640+x]=src[((size_t)sy*sw+sx)*3+c]/127.5f-1.f;
}
static void gundam_render(const char* pdf,int page){
    if(!g_ctx){ g_ctx=fz_new_context(NULL,NULL,FZ_STORE_UNLIMITED); fz_register_document_handlers(g_ctx); fz_set_aa_level(g_ctx,2); }
    if(g_docpath!=pdf){ if(g_doc)fz_drop_document(g_ctx,g_doc); g_doc=fz_open_document(g_ctx,pdf); g_docpath=pdf; }
    fz_page* pg=fz_load_page(g_ctx,g_doc,page); fz_rect r=fz_bound_page(g_ctx,pg); float pw=r.x1-r.x0, ph=r.y1-r.y0;
    gundam_ratio(pw,ph,&g_gw,&g_gh); g_gP=g_gw*g_gh; int SW=640*g_gw, SH=640*g_gh;
    fz_matrix mat=fz_scale((float)SW/pw,(float)SH/ph);                                // anisotropic -> exact (SW,SH)
    fz_pixmap* pix=fz_new_pixmap_from_page(g_ctx,pg,mat,fz_device_rgb(g_ctx),0);
    int w=fz_pixmap_width(g_ctx,pix),h=fz_pixmap_height(g_ctx,pix); unsigned char* sm=fz_pixmap_samples(g_ctx,pix);
    if((size_t)w*h*3>g_dtilesn){ if(g_dtiles)cudaFree(g_dtiles); CK(cudaMalloc(&g_dtiles,(size_t)w*h*3)); g_dtilesn=(size_t)w*h*3; }
    CK(cudaMemcpyAsync(g_dtiles,sm,(size_t)w*h*3,cudaMemcpyHostToDevice,VS));
    if(!g_tilesf) CK(cudaMalloc(&g_tilesf,(size_t)32*3*640*640*4));
    size_t TOT=(size_t)g_gP*3*640*640; k_gundam_tiles<<<(TOT+255)/256,256,0,VS>>>(g_dtiles,w,g_gw,g_tilesf);
    fz_drop_pixmap(g_ctx,pix); fz_drop_page(g_ctx,pg);
}
int gundam_encode(const char* pdf,int page){                                          // PDF page -> g_gundam [ntok,1280]
    init_vision(); render_preprocess(pdf,page,120.f); gundam_render(pdf,page);
    int nt=gundam_assemble(g_dimg,g_tilesf,g_gw,g_gh,g_gP); CK(cudaStreamSynchronize(VS)); return nt;
}
bf16* gundam_result(){ return g_gundam; }

#ifndef OCR_LINK
int main(){
    init_vision();
    float* dimg; CK(cudaMalloc(&dimg,(size_t)3*IMG*IMG*4));
    { FILE* f=fopen("/home/janitor/unlimited-ocr/engine/vfix/image_ori.f32","rb"); std::vector<float> h((size_t)3*IMG*IMG); if(fread(h.data(),4,h.size(),f)!=h.size()){} fclose(f); CK(cudaMemcpy(dimg,h.data(),h.size()*4,cudaMemcpyHostToDevice)); }
    bf16* out=sam_forward(dimg); CK(cudaDeviceSynchronize());
    printf("SAM verify:\n"); cmp("sam_out.f32",out,(size_t)1024*16*16);
    bf16* clip=clip_forward(out); CK(cudaDeviceSynchronize());
    printf("CLIP verify:\n"); cmp("clip_out.f32",clip,(size_t)CN*CC);
    bf16* projout; bf16* vis=project_assemble(clip,out,&projout); CK(cudaDeviceSynchronize());
    printf("Projector verify:\n"); cmp("proj_out.f32",projout,(size_t)256*1280);
    printf("final visual tokens: [273,1280] assembled\n"); return 0;
}
#endif
