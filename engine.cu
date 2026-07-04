// Unlimited-OCR decoder — focused CUDA engine (sm_89). Stage 2: full bf16 prefill,
// verified layer-by-layer against HF fixtures.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include "st_loader.h"
#include "server.h"

#define CK(x) do{cudaError_t _cke_=(x);if(_cke_){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_cke_));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t _cbs_=(x);if(_cbs_){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)_cbs_);exit(1);} }while(0)

using bf16 = __nv_bfloat16;
// ===== Pluggable KV cache element (compile-time backend) =====
// Default bf16. Build with -DKV_FP8 for fp8 e4m3 (halves KV bandwidth -> the long-context attention
// lever from profiling). A richer backend (e.g. KVarN: 4-bit K per-channel + 2-bit V per-token +
// Hadamard rotation) would replace kvt/kvld/kvst with a struct carrying packed bits + tile scales
// and a vector load/store; k_rope_store/k_attn_split/seed are already the only touch points.
#ifdef KV_FP8
using kvt = __nv_fp8_e4m3;
__host__ __device__ __forceinline__ float kvld(kvt x){ return float(x); }
__device__ __forceinline__ kvt  kvst(float x){ return (kvt)x; }
#define KV_NAME "fp8-e4m3"
#else
using kvt = __nv_bfloat16;
__host__ __device__ __forceinline__ float kvld(kvt x){ return __bfloat162float(x); }
__device__ __forceinline__ kvt  kvst(float x){ return __float2bfloat16(x); }
#define KV_NAME "bf16"
#endif
static const int H=1280, NH=10, HD=128, NEXP=64, TOPK=6, MOEI=896, SHI=1792, DENSEI=6848, NL=12, V=129280;
static const int WIN=128;  // R-SWA sliding window (config sliding_window_size)
static const float EPS=1e-6f, ROPE_THETA=10000.f;

static cublasHandle_t CUB;
static SafeTensors ST;
static cudaStream_t GS=0;     // universal stream (captured for the decode graph)

static bf16* up(const std::string& name){
    const Tensor& t=ST.get(name); bf16* d; CK(cudaMalloc(&d,t.nbytes));
    CK(cudaMemcpy(d,t.data,t.nbytes,cudaMemcpyHostToDevice)); return d;
}
__global__ void k_quant_rows(const bf16*,uint8_t*,float*,int);  // fwd decl
static void quant8(bf16* W,int rows,int cols,uint8_t** W8,float** sc);  // fwd decl
struct Layer{ bf16 *in_norm,*post_norm,*q,*k,*v,*o,*gate,*qkv; // qkv = fused [3H,H]
              bf16 *e_gate,*e_up,*e_down;     // experts packed [64,*]
              bf16 *sh_gate,*sh_up,*sh_down;  // shared
              bf16 *d_gate,*d_up,*d_down;     // dense (layer0 only)
              bf16 *d_gu,*sh_gu;              // fused gate|up for dense / shared
              // fp8 (e4m3) weights + per-row scales for decode
              uint8_t *qkv8,*o8,*shgu8,*shd8,*eg8,*eu8,*ed8,*dgu8,*dd8;
              float *qkv_s,*o_s,*shgu_s,*shd_s,*eg_s,*eu_s,*ed_s,*dgu_s,*dd_s;
              uint32_t *egu4t,*ed4t; float2 *sgu2,*sd2;   // mma-tiled int4 repack + float2 group scales (tensor-core MoE)
              bool dense; };
static Layer L[NL];
static bf16 *EMB,*LMH,*FNORM;
static uint8_t* LMH4=nullptr; static float* lmh_s4=nullptr;      // int4 group-128 lm_head + group scales
static uint32_t* LMH4T=nullptr; static float2* LMH_S2=nullptr;   // mma-tiled lm_head relay (LMHMMA path)

// quantize [rows,cols] bf16 -> e4m3 (raw uint8) with per-row scale = maxabs/448
__global__ void k_quant_rows(const bf16* W,uint8_t* W8,float* scale,int cols){
    int row=blockIdx.x; const bf16* wr=W+(size_t)row*cols; uint8_t* o=W8+(size_t)row*cols;
    __shared__ float sm; float mx=0;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x) mx=fmaxf(mx,fabsf(__bfloat162float(wr[c])));
    for(int s=16;s;s>>=1)mx=fmaxf(mx,__shfl_down_sync(~0u,mx,s));
    __shared__ float wm[32]; if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=mx; __syncthreads();
    if(threadIdx.x<32){float v=(threadIdx.x<(blockDim.x+31)/32)?wm[threadIdx.x]:0; for(int s=16;s;s>>=1)v=fmaxf(v,__shfl_down_sync(~0u,v,s)); if(!threadIdx.x)sm=v;}
    __syncthreads(); float sc=(sm>0?sm:1.f)/448.f;
    if(threadIdx.x==0)scale[row]=sc;
    for(int c=threadIdx.x;c<cols;c+=blockDim.x)
        o[c]=__nv_cvt_float_to_fp8(__bfloat162float(wr[c])/sc,__NV_SATFINITE,__NV_E4M3);
}
// dequant one fp8 byte b (0..3) out of a packed uint32 -> float
__device__ __forceinline__ float fp8b(uint32_t p,int b){
    unsigned char c=(unsigned char)((p>>(8*b))&0xFFu); __half h=__nv_cvt_fp8_to_halfraw(c,__NV_E4M3); return __half2float(h);
}
// vectorized fp8 dot: 16 weights per uint4 load (cols%16==0). x scalar bf16 (small/cached).
__device__ __forceinline__ float dot_fp8_vec(const uint8_t* wrow,const bf16* x,int cols){
    const uint4* w4=(const uint4*)wrow; int nck=cols/16, lane=threadIdx.x&31;
    float a[4]={0,0,0,0};   // 4 independent accumulators (one per 4-elem subgroup) -> ILP to hide load->FMA latency
    for(int ck=lane;ck<nck;ck+=32){ uint4 W=w4[ck]; const bf16* xx=x+(size_t)ck*16; uint32_t pp[4]={W.x,W.y,W.z,W.w};
        #pragma unroll
        for(int u=0;u<4;u++){ const bf16* xu=xx+u*4; uint32_t p=pp[u];
            a[u]+=__bfloat162float(xu[0])*fp8b(p,0)+__bfloat162float(xu[1])*fp8b(p,1)
                +__bfloat162float(xu[2])*fp8b(p,2)+__bfloat162float(xu[3])*fp8b(p,3); } }
    float acc=(a[0]+a[1])+(a[2]+a[3]);
    for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o); return acc;
}
#define QG 128
__device__ __forceinline__ int I4LO(uint8_t by){ int v=by&0xF; return v>7?v-16:v; }
__device__ __forceinline__ int I4HI(uint8_t by){ int v=(by>>4)&0xF; return v>7?v-16:v; }
// fused fp8 SwiGLU: out[r]=silu(gate_r·x)*(up_r·x), Wgu=[2*interm,H]. replaces gemv(2*interm)+silu_mul.
static void quant8(bf16* W,int rows,int cols,uint8_t** W8,float** sc){
    CK(cudaMalloc(W8,(size_t)rows*cols)); CK(cudaMalloc(sc,(size_t)rows*4));
    k_quant_rows<<<rows,256,0,GS>>>(W,*W8,*sc,cols);
}
// group-128 int4 symmetric: one scale per 128-element group along the row. blockDim=128, grid=rows.
// cols must be a multiple of 128 (H=1280=10*128, MOEI=896=7*128). scale layout [row*ng + g].
__global__ void k_quant_rows_q4(const bf16* W,uint8_t* W4,float* scale,int cols){
    int row=blockIdx.x, tid=threadIdx.x, ng=cols/QG; const bf16* wr=W+(size_t)row*cols;
    __shared__ float r[QG];
    for(int g=0;g<ng;g++){                                   // per-group scale
        r[tid]=fabsf(__bfloat162float(wr[g*QG+tid])); __syncthreads();
        for(int s=QG/2;s;s>>=1){ if(tid<s)r[tid]=fmaxf(r[tid],r[tid+s]); __syncthreads(); }
        if(tid==0)scale[(size_t)row*ng+g]=(r[0]>0?r[0]:1.f)/7.f; __syncthreads();
    }
    uint8_t* o=W4+(size_t)row*(cols/2);
    for(int b=tid;b<cols/2;b+=QG){ float sc=scale[(size_t)row*ng + (2*b)/QG];
        int lo=__float2int_rn(__bfloat162float(wr[2*b])/sc);   lo=max(-7,min(7,lo));
        int hi=__float2int_rn(__bfloat162float(wr[2*b+1])/sc); hi=max(-7,min(7,hi));
        o[b]=(uint8_t)((lo&0xF)|((hi&0xF)<<4));
    }
}
// q4 (group-128) expert gate|up: per-group scale applied per term. ng = H/QG.
// S>1 (prefill) batched BF16 experts (exact, matches HF): warp per (token,slot,output).
__global__ void k_moe_gateup_bf16_S(const bf16* x,const int* idx,const bf16* Wg,const bf16* Wu,bf16* hbuf,int Sn,int interm){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31;
    if(gw>=(long)Sn*TOPK*interm)return;
    int r=gw%interm; long ts=gw/interm; int slot=ts%TOPK, t=ts/TOPK; int e=idx[t*TOPK+slot];
    const bf16* wg=Wg+((size_t)e*interm+r)*H; const bf16* wu=Wu+((size_t)e*interm+r)*H; const bf16* xt=x+(size_t)t*H; float g=0,u=0;
    for(int c=lane;c<H;c+=32){ float xc=__bfloat162float(xt[c]); g+=xc*__bfloat162float(wg[c]); u+=xc*__bfloat162float(wu[c]); }
    for(int o=16;o;o>>=1){ g+=__shfl_xor_sync(~0u,g,o); u+=__shfl_xor_sync(~0u,u,o); }
    if(lane==0) hbuf[gw]=__float2bfloat16((g/(1.f+__expf(-g)))*u);
}
__global__ void k_moe_down_bf16_S(const bf16* hbuf,const int* idx,const float* w,const bf16* Wd,float* Yf,int Sn,int interm){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31;
    if(gw>=(long)Sn*H)return;                                       // warp per (token,d); loop K slots fixed order -> deterministic
    int d=gw%H; long t=gw/H; float ysum=0;
    for(int slot=0;slot<TOPK;slot++){ int e=idx[t*TOPK+slot];
        const bf16* wd=Wd+((size_t)e*H+d)*interm; const bf16* h=hbuf+(size_t)(t*TOPK+slot)*interm; float acc=0;
        for(int c=lane;c<interm;c+=32)acc+=__bfloat162float(h[c])*__bfloat162float(wd[c]);
        for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o);
        ysum+=w[t*TOPK+slot]*acc; }
    if(lane==0)Yf[(size_t)t*H+d]=ysum;
}
static void quant_q4(bf16* W,int rows,int cols,uint8_t** W4,float** sc){
    CK(cudaMalloc(W4,(size_t)rows*(cols/2))); CK(cudaMalloc(sc,(size_t)rows*(cols/QG)*4));
    k_quant_rows_q4<<<rows,QG,0,GS>>>(W,*W4,*sc,cols);
}
// ===== mma-tile repack (TCMOE): same q4 nibbles relaid fragment-ordered, biased +8 (= raw^8, since m=v+8 = n^8).
// Word (lane,ks) bits: [0:4)=w(rA,k) [4:8)=w(rB,k) [8:12)=w(rA,k+8) [12:16)=w(rB,k+8) [16:20)=w(rA,k+1)
// [20:24)=w(rB,k+1) [24:28)=w(rA,k+9) [28:32)=w(rB,k+9); k=kb+2*(lane&3), kb=g*QG+ks*16, ks=4p+(wi&3).
// gateup tile j: rA=gate row 8j+(lane>>2), rB=up row (same r). down tile j: rA=d=16j+(lane>>2), rB=d+8.
__global__ void k_repack_gu(const uint8_t* G4,const uint8_t* U4,uint32_t* T){        // grid(64,112,10) block 256 (wi = p*128+lane*4+s)
    int e=blockIdx.x,j=blockIdx.y,g=blockIdx.z,wi=threadIdx.x,lane=(wi>>2)&31;
    int kb=g*QG+((wi>>7)*4+(wi&3))*16, klo=kb+2*(lane&3); size_t rg=((size_t)e*MOEI+8*j+(lane>>2))*(H/2);
    uint8_t g0=G4[rg+klo/2],g1=G4[rg+(klo+8)/2],u0=U4[rg+klo/2],u1=U4[rg+(klo+8)/2];
    T[((((size_t)e*112+j)*10+g)<<8)+wi] = ((g0&0xFu)^8u)|(((u0&0xFu)^8u)<<4)|(((g1&0xFu)^8u)<<8)|(((u1&0xFu)^8u)<<12)
        |((uint32_t)((g0>>4)^8u)<<16)|((uint32_t)((u0>>4)^8u)<<20)|((uint32_t)((g1>>4)^8u)<<24)|((uint32_t)((u1>>4)^8u)<<28);
}
__global__ void k_repack_dn(const uint8_t* D4,uint32_t* T){                          // grid(64,80,7) block 256
    int e=blockIdx.x,j=blockIdx.y,g=blockIdx.z,wi=threadIdx.x,lane=(wi>>2)&31;
    int kb=g*QG+((wi>>7)*4+(wi&3))*16, klo=kb+2*(lane&3);
    size_t rA=((size_t)e*H+16*j+(lane>>2))*(MOEI/2), rB=rA+8*(MOEI/2);
    uint8_t a0=D4[rA+klo/2],a1=D4[rA+(klo+8)/2],b0=D4[rB+klo/2],b1=D4[rB+(klo+8)/2];
    T[((((size_t)e*80+j)*7+g)<<8)+wi] = ((a0&0xFu)^8u)|(((b0&0xFu)^8u)<<4)|(((a1&0xFu)^8u)<<8)|(((b1&0xFu)^8u)<<12)
        |((uint32_t)((a0>>4)^8u)<<16)|((uint32_t)((b0>>4)^8u)<<20)|((uint32_t)((a1>>4)^8u)<<24)|((uint32_t)((b1>>4)^8u)<<28);
}
__global__ void k_repack_sgu(const float* GS,const float* US,float2* S2){            // grid(64,10): {gate,up} scale per (g,r)
    int e=blockIdx.x,g=blockIdx.y;
    for(int r=threadIdx.x;r<MOEI;r+=256) S2[((size_t)e*10+g)*MOEI+r]=make_float2(GS[((size_t)e*MOEI+r)*10+g],US[((size_t)e*MOEI+r)*10+g]);
}
__global__ void k_repack_sd(const float* DS,float2* S2){                             // grid(64,7): {sc(d),sc(d+8)} per (g, j*8+rl)
    int e=blockIdx.x,g=blockIdx.y;
    for(int i=threadIdx.x;i<H/2;i+=256){ int d=16*(i>>3)+(i&7);
        S2[((size_t)e*7+g)*(H/2)+i]=make_float2(DS[((size_t)e*H+d)*7+g],DS[((size_t)e*H+d+8)*7+g]); }
}
static void repack_mma(Layer&ly){
    CK(cudaMalloc(&ly.egu4t,(size_t)NEXP*112*10*1024)); CK(cudaMalloc(&ly.ed4t,(size_t)NEXP*80*7*1024));
    CK(cudaMalloc(&ly.sgu2,(size_t)NEXP*10*MOEI*8));    CK(cudaMalloc(&ly.sd2,(size_t)NEXP*7*(H/2)*8));
    k_repack_gu<<<dim3(NEXP,112,10),256,0,GS>>>(ly.eg8,ly.eu8,ly.egu4t);
    k_repack_dn<<<dim3(NEXP,80,7),256,0,GS>>>(ly.ed8,ly.ed4t);
    k_repack_sgu<<<dim3(NEXP,10),256,0,GS>>>(ly.eg_s,ly.eu_s,ly.sgu2);
    k_repack_sd<<<dim3(NEXP,7),256,0,GS>>>(ly.ed_s,ly.sd2);
}
__global__ void k_repack_lmh(const uint8_t* W4,uint32_t* T){                         // grid(V/16,10) block 256: lm_head tiles (r, r+8)
    int j=blockIdx.x,g=blockIdx.y,wi=threadIdx.x,lane=(wi>>2)&31;
    int kb=g*QG+((wi>>7)*4+(wi&3))*16, klo=kb+2*(lane&3);
    size_t rA=((size_t)16*j+(lane>>2))*(H/2), rB=rA+8*(size_t)(H/2);
    uint8_t a0=W4[rA+klo/2],a1=W4[rA+(klo+8)/2],b0=W4[rB+klo/2],b1=W4[rB+(klo+8)/2];
    T[(((size_t)j*10+g)<<8)+wi] = ((a0&0xFu)^8u)|(((b0&0xFu)^8u)<<4)|(((a1&0xFu)^8u)<<8)|(((b1&0xFu)^8u)<<12)
        |((uint32_t)((a0>>4)^8u)<<16)|((uint32_t)((b0>>4)^8u)<<20)|((uint32_t)((a1>>4)^8u)<<24)|((uint32_t)((b1>>4)^8u)<<28);
}
__global__ void k_repack_slmh(const float* S,float2* S2,int Vn){                     // grid(10): {sc(r),sc(r+8)} per (g, j*8+rl)
    int g=blockIdx.x;
    for(int i=threadIdx.x;i<Vn/2;i+=256){ int d=16*(i>>3)+(i&7);
        S2[(size_t)g*(Vn/2)+i]=make_float2(S[(size_t)d*10+g],S[((size_t)d+8)*10+g]); }
}
// int4 group-128 lm_head GEMV (float out): out[v]=dot(x, deq_g128(W4[v])); warp/row (see k_lmhead_q4_b)

// ---- gemm: Y[S,O] = X[S,Hin] · W[O,Hin]^T  (row-major), bf16 io, fp32 accum ----
static void lin(const bf16* X,const bf16* W,bf16* Y,int S,int Hin,int O){
    float a=1,b=0;
    CB(cublasGemmEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,O,S,Hin,&a,
        W,CUDA_R_16BF,Hin, X,CUDA_R_16BF,Hin,&b, Y,CUDA_R_16BF,O,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
static void lin_f32(const bf16* X,const bf16* W,float* Y,int S,int Hin,int O){
    float a=1,b=0;
    CB(cublasGemmEx(CUB,CUBLAS_OP_T,CUBLAS_OP_N,O,S,Hin,&a,
        W,CUDA_R_16BF,Hin, X,CUDA_R_16BF,Hin,&b, Y,CUDA_R_32F,O,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}

__global__ void k_embed(const bf16* emb,const int* ids,bf16* out,int S){
    int t=blockIdx.x, i=threadIdx.x+blockIdx.y*blockDim.x; if(i>=H)return;
    out[(size_t)t*H+i]=emb[(size_t)ids[t]*H+i];
}
__global__ void k_rmsnorm(const bf16* x,const bf16* w,bf16* y,int S,int Hd,float eps){
    int row=blockIdx.x; const bf16* xr=x+(size_t)row*Hd; bf16* yr=y+(size_t)row*Hd;
    __shared__ float red[32]; float acc=0;
    for(int i=threadIdx.x;i<Hd;i+=blockDim.x){float v=__bfloat162float(xr[i]);acc+=v*v;}
    for(int o=16;o;o>>=1)acc+=__shfl_down_sync(~0u,acc,o);
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=acc; __syncthreads();
    if(threadIdx.x<32){float v=(threadIdx.x<(blockDim.x+31)/32)?red[threadIdx.x]:0; for(int o=16;o;o>>=1)v+=__shfl_down_sync(~0u,v,o); if(!threadIdx.x)red[0]=v;}
    __syncthreads(); float rstd=rsqrtf(red[0]/Hd+eps);
    for(int i=threadIdx.x;i<Hd;i+=blockDim.x) yr[i]=__float2bfloat16(__bfloat162float(xr[i])*rstd*__bfloat162float(w[i]));
}
__global__ void k_add(bf16* x,const bf16* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__float2bfloat16(__bfloat162float(x[i])+__bfloat162float(y[i]));}
// fused: x += y (residual, in place), then out = rmsnorm(x)*w. one block per row.
__global__ void k_add_rmsnorm(bf16* x,const bf16* y,const bf16* w,bf16* out,int Hd,float eps){
    int row=blockIdx.x; bf16* xr=x+(size_t)row*Hd; const bf16* yr=y+(size_t)row*Hd; bf16* outr=out+(size_t)row*Hd;
    __shared__ float red[32]; float acc=0;
    for(int i=threadIdx.x;i<Hd;i+=blockDim.x){ float v=__bfloat162float(xr[i])+__bfloat162float(yr[i]); xr[i]=__float2bfloat16(v); acc+=v*v; }
    for(int o=16;o;o>>=1)acc+=__shfl_down_sync(~0u,acc,o);
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=acc; __syncthreads();
    if(threadIdx.x<32){float v=(threadIdx.x<(blockDim.x+31)/32)?red[threadIdx.x]:0; for(int o=16;o;o>>=1)v+=__shfl_down_sync(~0u,v,o); if(!threadIdx.x)red[0]=v;}
    __syncthreads(); float rstd=rsqrtf(red[0]/Hd+eps);
    for(int i=threadIdx.x;i<Hd;i+=blockDim.x) outr[i]=__float2bfloat16(__bfloat162float(xr[i])*rstd*__bfloat162float(w[i]));
}
// RoPE in place on q/k laid out [S, NH*HD] (head-major within row)
__global__ void k_rope(bf16* q,bf16* k,int S,int pos0){
    int t=blockIdx.x, h=blockIdx.y, d=threadIdx.x; // d in 0..HD/2-1
    float inv=powf(ROPE_THETA, -2.f*d/HD); float ang=(pos0+t)*inv; float c=cosf(ang),s=sinf(ang);
    size_t base=((size_t)t*NH+h)*HD;
    auto rot=[&](bf16* x){ float a=__bfloat162float(x[base+d]), b=__bfloat162float(x[base+d+HD/2]);
        x[base+d]=__float2bfloat16(a*c-b*s); x[base+d+HD/2]=__float2bfloat16(b*c+a*s); };
    rot(q); rot(k);
}
// prefill attention: causal, flash-style. q,k,v [S,NH,HD]. one block per (h, query i), 8 warps split keys.
__global__ void k_attn_prefill(const bf16* q,const bf16* k,const bf16* v,bf16* out,int S){
    int h=blockIdx.x, i=blockIdx.y, tid=threadIdx.x, w=tid>>5, lane=tid&31; if(i>=S)return;
    int clen=i+1; float scale=rsqrtf((float)HD);
    __shared__ float qs[HD], sm[8], sl[8], sacc[8][HD];
    size_t qb=((size_t)i*NH+h)*HD;
    for(int x=tid;x<HD;x+=blockDim.x) qs[x]=__bfloat162float(q[qb+x]);
    __syncthreads();
    int d0=lane,d1=lane+32,d2=lane+64,d3=lane+96;
    float q0=qs[d0],q1=qs[d1],q2=qs[d2],q3=qs[d3];
    float m=-1e30f,l=0,a0=0,a1=0,a2=0,a3=0;
    for(int j=w;j<clen;j+=8){ const bf16* kj=k+((size_t)j*NH+h)*HD;
        float p=q0*__bfloat162float(kj[d0])+q1*__bfloat162float(kj[d1])+q2*__bfloat162float(kj[d2])+q3*__bfloat162float(kj[d3]);
        for(int o=16;o;o>>=1)p+=__shfl_xor_sync(~0u,p,o);
        float sc=p*scale,mn=fmaxf(m,sc),cr=__expf(m-mn),pe=__expf(sc-mn);
        const bf16* vj=v+((size_t)j*NH+h)*HD;
        a0=a0*cr+pe*__bfloat162float(vj[d0]); a1=a1*cr+pe*__bfloat162float(vj[d1]);
        a2=a2*cr+pe*__bfloat162float(vj[d2]); a3=a3*cr+pe*__bfloat162float(vj[d3]); l=l*cr+pe; m=mn; }
    if(lane==0){sm[w]=m;sl[w]=l;}
    sacc[w][d0]=a0; sacc[w][d1]=a1; sacc[w][d2]=a2; sacc[w][d3]=a3;
    __syncthreads();
    if(tid<HD){ float mg=-1e30f; for(int ww=0;ww<8;ww++)mg=fmaxf(mg,sm[ww]);
        float lg=0,o=0; for(int ww=0;ww<8;ww++){float f=__expf(sm[ww]-mg); lg+=sl[ww]*f; o+=sacc[ww][tid]*f;}
        out[(size_t)i*NH*HD + h*HD + tid]=__float2bfloat16(o/lg); }
}
// seed reference KV from bf16 prefill K/V into the pluggable KV format (bf16 round-trip / fp8 quantize)
__global__ void k_seedkv(kvt* dst,const bf16* src,size_t n){ size_t i=(size_t)blockIdx.x*256+threadIdx.x; if(i<n) dst[i]=kvst(__bfloat162float(src[i])); }
__global__ void k_silu_mul(const bf16* g,const bf16* u,bf16* o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=__bfloat162float(g[i]); o[i]=__float2bfloat16((x/(1.f+__expf(-x)))*__bfloat162float(u[i]));
}
// softmax(64) + top6 per token. logits[S,64] fp32 -> idx[S,6],w[S,6]
__global__ void k_route(const float* logits,int* idx,float* w,int S){   // block per token, NEXP threads (parallel softmax)
    int t=blockIdx.x, e=threadIdx.x; const float* lg=logits+(size_t)t*NEXP;
    __shared__ float s[NEXP]; __shared__ float mx;
    float v=lg[e]; s[e]=v; __syncthreads();
    if(e==0){ float m=-1e30f; for(int i=0;i<NEXP;i++)m=fmaxf(m,s[i]); mx=m; } __syncthreads();
    s[e]=__expf(v-mx); __syncthreads();
    if(e==0){ float z=0; for(int i=0;i<NEXP;i++)z+=s[i];
        bool used[NEXP]; for(int i=0;i<NEXP;i++)used[i]=false;
        for(int k=0;k<TOPK;k++){int b=-1;float bv=-1; for(int i=0;i<NEXP;i++)if(!used[i]&&s[i]>bv){bv=s[i];b=i;} used[b]=true; idx[t*TOPK+k]=b; w[t*TOPK+k]=s[b]/z; } }
}
// gate GEMV: glog[e]=dot(x, gateW[e]); warp per expert, multi-block. bf16 in / fp32 accum (== cuBLAS).  (see k_gate_b)
// out_bf16 = Yf(float experts) + shared(bf16)
__global__ void k_combine(bf16* out,const float* Yf,const bf16* shared,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2bfloat16(Yf[i]+__bfloat162float(shared[i]));
}

static void load_weights(){
    EMB=up("model.embed_tokens.weight"); LMH=up("lm_head.weight"); FNORM=up("model.norm.weight");
    quant_q4(LMH,V,H,&LMH4,&lmh_s4); CK(cudaStreamSynchronize(GS));   // int4 lm_head (fast full logits)
    CK(cudaMalloc(&LMH4T,(size_t)(V/16)*10*1024)); CK(cudaMalloc(&LMH_S2,(size_t)10*(V/2)*8));   // LMHMMA relay
    k_repack_lmh<<<dim3(V/16,10),256,0,GS>>>(LMH4,LMH4T);
    k_repack_slmh<<<10,256,0,GS>>>(lmh_s4,LMH_S2,V);
    for(int l=0;l<NL;l++){ std::string p="model.layers."+std::to_string(l)+".";
        L[l].in_norm=up(p+"input_layernorm.weight"); L[l].post_norm=up(p+"post_attention_layernorm.weight");
        L[l].q=up(p+"self_attn.q_proj.weight"); L[l].k=up(p+"self_attn.k_proj.weight");
        L[l].v=up(p+"self_attn.v_proj.weight"); L[l].o=up(p+"self_attn.o_proj.weight");
        CK(cudaMalloc(&L[l].qkv,3*(size_t)H*H*2));  // fused QKV for decode
        CK(cudaMemcpy(L[l].qkv,            L[l].q,(size_t)H*H*2,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(L[l].qkv+(size_t)H*H,L[l].k,(size_t)H*H*2,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(L[l].qkv+2*(size_t)H*H,L[l].v,(size_t)H*H*2,cudaMemcpyDeviceToDevice));
        quant8(L[l].qkv,3*H,H,&L[l].qkv8,&L[l].qkv_s); quant8(L[l].o,H,H,&L[l].o8,&L[l].o_s);  // fp8 attn (more bit-sensitive)
        L[l].dense=(l==0);
        if(l==0){ L[l].d_gate=up(p+"mlp.gate_proj.weight"); L[l].d_up=up(p+"mlp.up_proj.weight"); L[l].d_down=up(p+"mlp.down_proj.weight");
            CK(cudaMalloc(&L[l].d_gu,2*(size_t)DENSEI*H*2));
            CK(cudaMemcpy(L[l].d_gu,L[l].d_gate,(size_t)DENSEI*H*2,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(L[l].d_gu+(size_t)DENSEI*H,L[l].d_up,(size_t)DENSEI*H*2,cudaMemcpyDeviceToDevice));
            quant8(L[l].d_gu,2*DENSEI,H,&L[l].dgu8,&L[l].dgu_s); quant8(L[l].d_down,H,DENSEI,&L[l].dd8,&L[l].dd_s); }
        else{
            L[l].gate=up(p+"mlp.gate.weight");
            // pack experts: allocate [64,interm,H] and copy each
            size_t gsz=(size_t)NEXP*MOEI*H*2, dsz=(size_t)NEXP*H*MOEI*2;
            CK(cudaMalloc(&L[l].e_gate,gsz)); CK(cudaMalloc(&L[l].e_up,gsz)); CK(cudaMalloc(&L[l].e_down,dsz));
            for(int e=0;e<NEXP;e++){ std::string pe=p+"mlp.experts."+std::to_string(e)+".";
                size_t off=(size_t)e*MOEI*H;
                const Tensor&g=ST.get(pe+"gate_proj.weight"); CK(cudaMemcpy(L[l].e_gate+off,g.data,g.nbytes,cudaMemcpyHostToDevice));
                const Tensor&u=ST.get(pe+"up_proj.weight");   CK(cudaMemcpy(L[l].e_up+off,  u.data,u.nbytes,cudaMemcpyHostToDevice));
                const Tensor&d=ST.get(pe+"down_proj.weight"); CK(cudaMemcpy(L[l].e_down+off,d.data,d.nbytes,cudaMemcpyHostToDevice)); }
            quant_q4(L[l].e_gate,NEXP*MOEI,H,&L[l].eg8,&L[l].eg_s);   // q4 experts
            quant_q4(L[l].e_up,  NEXP*MOEI,H,&L[l].eu8,&L[l].eu_s);
            quant_q4(L[l].e_down,NEXP*H,MOEI,&L[l].ed8,&L[l].ed_s);
            repack_mma(L[l]);                                         // mma-tiled relay of the SAME nibbles/scales (TCMOE path)
            L[l].sh_gate=up(p+"mlp.shared_experts.gate_proj.weight");
            L[l].sh_up=up(p+"mlp.shared_experts.up_proj.weight");
            L[l].sh_down=up(p+"mlp.shared_experts.down_proj.weight");
            CK(cudaMalloc(&L[l].sh_gu,2*(size_t)SHI*H*2));
            CK(cudaMemcpy(L[l].sh_gu,L[l].sh_gate,(size_t)SHI*H*2,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(L[l].sh_gu+(size_t)SHI*H,L[l].sh_up,(size_t)SHI*H*2,cudaMemcpyDeviceToDevice));
            quant8(L[l].sh_gu,2*SHI,H,&L[l].shgu8,&L[l].shgu_s); quant8(L[l].sh_down,H,SHI,&L[l].shd8,&L[l].shd_s); // fp8 shared
            if(l==1 && getenv("DBG")){ const Tensor&g46=ST.get(p+"mlp.experts.46.gate_proj.weight");
                printf("  [LOAD] mmap e46.gate.data nbytes=%zu host[0]=%.4f host[1]=%.4f\n",g46.nbytes,
                       bf16_to_f32(((const uint16_t*)g46.data)[0]),bf16_to_f32(((const uint16_t*)g46.data)[1]));
                uint16_t dv[2]; CK(cudaMemcpy(dv,L[l].e_gate+(size_t)46*MOEI*H,4,cudaMemcpyDeviceToHost));
                printf("  [LOAD] device e_gate[46*MOEI*H + 0..1]=%.4f %.4f\n",bf16_to_f32(dv[0]),bf16_to_f32(dv[1])); }
        }
    }
}

// scratch
static bf16 *xbuf,*nbuf,*q,*k,*v,*att,*tmp,*mg,*mu,*mh; static float* glog; static int *didx; static float* dw; static float* Yf;
static kvt *kcache[NL],*vcache[NL];
static void alloc(int S){
    size_t sH=(size_t)S*H;
    if(xbuf){ cudaFree(xbuf);cudaFree(nbuf);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(att);cudaFree(tmp);   // free-first: repeated calls (Gundam per-page) reuse, no leak
        cudaFree(mg);cudaFree(mu);cudaFree(mh);cudaFree(glog);cudaFree(didx);cudaFree(dw);cudaFree(Yf);
        for(int l=0;l<NL;l++){cudaFree(kcache[l]);cudaFree(vcache[l]);} }
    CK(cudaMalloc(&xbuf,sH*2));CK(cudaMalloc(&nbuf,sH*2));CK(cudaMalloc(&q,sH*2));CK(cudaMalloc(&k,sH*2));CK(cudaMalloc(&v,sH*2));CK(cudaMalloc(&att,sH*2));CK(cudaMalloc(&tmp,sH*2));
    CK(cudaMalloc(&mg,(size_t)S*DENSEI*2));CK(cudaMalloc(&mu,(size_t)S*DENSEI*2));CK(cudaMalloc(&mh,(size_t)S*DENSEI*2));
    CK(cudaMalloc(&glog,(size_t)S*NEXP*4));CK(cudaMalloc(&didx,(size_t)S*TOPK*4));CK(cudaMalloc(&dw,(size_t)S*TOPK*4));CK(cudaMalloc(&Yf,sH*4));
    for(int l=0;l<NL;l++){ CK(cudaMalloc(&kcache[l],(size_t)S*H*sizeof(kvt))); CK(cudaMalloc(&vcache[l],(size_t)S*H*sizeof(kvt))); }
}
static void mlp_dense(Layer&ly,const bf16* xn,bf16* outY,int S,int interm,bf16* Wg,bf16* Wu,bf16* Wd);
// MoE/dense block for PREFILL (S>1). Page-parallel decode uses mlp_block_b.
// ===== expert-grouped bf16 MoE (prefill): sort tokens by expert -> 1 GEMM/expert (weight read ONCE, tensor cores) =====
__global__ void k_moe_hist(const int* didx,int* cnt,int S){ long i=(long)blockIdx.x*256+threadIdx.x; if(i<(long)S*TOPK)atomicAdd(&cnt[didx[i]],1); }
__global__ void k_moe_scat(const int* didx,const int* off,int* cur,int* gtok,int S){
    long i=(long)blockIdx.x*256+threadIdx.x; if(i>=(long)S*TOPK)return; int e=didx[i]; int p=atomicAdd(&cur[e],1); gtok[off[e]+p]=(int)i; }
__global__ void k_moe_gath(const bf16* x,const int* gtok,bf16* xg,long tot){
    long idx=(long)blockIdx.x*256+threadIdx.x; if(idx>=tot*H)return; long i=idx/H; int c=idx%H; xg[idx]=x[(size_t)(gtok[i]/TOPK)*H+c]; }
__global__ void k_moe_inv(const int* gtok,int* inv,long tot){ long j=(long)blockIdx.x*256+threadIdx.x; if(j<tot)inv[gtok[j]]=(int)j; }
__global__ void k_moe_comb(const bf16* og,const int* inv,const float* dw,float* Yf,int S){   // det: sum K slots in fixed order, 1 write
    long idx=(long)blockIdx.x*256+threadIdx.x; if(idx>=(long)S*H)return; int c=idx%H; long t=idx/H; float s=0;
    for(int slot=0;slot<TOPK;slot++){ int j=inv[t*TOPK+slot]; s+=dw[t*TOPK+slot]*__bfloat162float(og[(size_t)j*H+c]); }
    Yf[idx]=s; }
static int *gm_cnt=0,*gm_off=0,*gm_cur=0,*gm_gtok=0; static bf16 *gm_xg=0,*gm_gate=0,*gm_up=0,*gm_h=0,*gm_og=0; static int* gm_inv=0; static long gm_cap=0;
static void moe_grouped(Layer& ly,int S){
    long tot=(long)S*TOPK;
    if(tot>gm_cap){ if(gm_xg){cudaFree(gm_xg);cudaFree(gm_gate);cudaFree(gm_up);cudaFree(gm_h);cudaFree(gm_og);cudaFree(gm_gtok);}
        CK(cudaMalloc(&gm_xg,(size_t)tot*H*2));CK(cudaMalloc(&gm_gate,(size_t)tot*MOEI*2));CK(cudaMalloc(&gm_up,(size_t)tot*MOEI*2));
        CK(cudaMalloc(&gm_h,(size_t)tot*MOEI*2));CK(cudaMalloc(&gm_og,(size_t)tot*H*2));CK(cudaMalloc(&gm_gtok,(size_t)tot*4));CK(cudaMalloc(&gm_inv,(size_t)tot*4)); gm_cap=tot; }
    if(!gm_cnt){CK(cudaMalloc(&gm_cnt,NEXP*4));CK(cudaMalloc(&gm_off,(NEXP+1)*4));CK(cudaMalloc(&gm_cur,NEXP*4));}
    CK(cudaMemsetAsync(gm_cnt,0,NEXP*4,GS));
    k_moe_hist<<<(tot+255)/256,256,0,GS>>>(didx,gm_cnt,S);
    CK(cudaStreamSynchronize(GS));                                              // hist runs on GS; must finish before host reads counts
    static int hcnt[NEXP],hoff[NEXP+1]; CK(cudaMemcpyAsync(hcnt,gm_cnt,NEXP*4,cudaMemcpyDeviceToHost,GS)); CK(cudaStreamSynchronize(GS)); // stream-scoped: a legacy-stream memcpy would drain the OTHER streams (async-admission overlap)
    hoff[0]=0; for(int e=0;e<NEXP;e++)hoff[e+1]=hoff[e]+hcnt[e]; const int* off=hoff;
    CK(cudaMemcpyAsync(gm_off,hoff,(NEXP+1)*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemsetAsync(gm_cur,0,NEXP*4,GS));
    k_moe_scat<<<(tot+255)/256,256,0,GS>>>(didx,gm_off,gm_cur,gm_gtok,S);
    k_moe_inv<<<(tot+255)/256,256,0,GS>>>(gm_gtok,gm_inv,tot);
    k_moe_gath<<<(tot*H+255)/256,256,0,GS>>>(nbuf,gm_gtok,gm_xg,tot);
    for(int e=0;e<NEXP;e++){ int ne=off[e+1]-off[e]; if(!ne)continue; const bf16* xe=gm_xg+(size_t)off[e]*H;   // gate+up GEMMs/expert
        lin(xe,ly.e_gate+(size_t)e*MOEI*H,gm_gate+(size_t)off[e]*MOEI,ne,H,MOEI);
        lin(xe,ly.e_up  +(size_t)e*MOEI*H,gm_up  +(size_t)off[e]*MOEI,ne,H,MOEI); }
    k_silu_mul<<<(tot*MOEI+255)/256,256,0,GS>>>(gm_gate,gm_up,gm_h,tot*MOEI);
    for(int e=0;e<NEXP;e++){ int ne=off[e+1]-off[e]; if(!ne)continue;                                          // down GEMM/expert
        lin(gm_h+(size_t)off[e]*MOEI,ly.e_down+(size_t)e*H*MOEI,gm_og+(size_t)off[e]*H,ne,MOEI,H); }
    k_moe_comb<<<((size_t)S*H+255)/256,256,0,GS>>>(gm_og,gm_inv,dw,Yf,S);
}
static void mlp_block(Layer&ly,int S){
    if(ly.dense){ mlp_dense(ly,nbuf,tmp,S,DENSEI,ly.d_gate,ly.d_up,ly.d_down); return; }
    lin_f32(nbuf,ly.gate,glog,S,H,NEXP);
    k_route<<<S,NEXP,0,GS>>>(glog,didx,dw,S);
    if(getenv("OLDMOE")){ long t1=(long)S*TOPK*MOEI; k_moe_gateup_bf16_S<<<(t1+7)/8,256,0,GS>>>(nbuf,didx,ly.e_gate,ly.e_up,mh,S,MOEI);
        long t2=(long)S*H; k_moe_down_bf16_S<<<(t2+7)/8,256,0,GS>>>(mh,didx,dw,ly.e_down,Yf,S,MOEI); }
    else moe_grouped(ly,S);
    mlp_dense(ly,nbuf,att,S,SHI,ly.sh_gate,ly.sh_up,ly.sh_down);
    k_combine<<<(S*H+255)/256,256,0,GS>>>(tmp,Yf,att,S*H);
}
static void mlp_dense(Layer&ly,const bf16* xn,bf16* outY,int S,int interm,bf16* Wg,bf16* Wu,bf16* Wd){
    lin(xn,Wg,mg,S,H,interm); lin(xn,Wu,mu,S,H,interm);
    k_silu_mul<<<(S*interm+255)/256,256,0,GS>>>(mg,mu,mh,S*interm);
    lin(mh,Wd,outY,S,interm,H);
}
// ===== async-admission prefill context (heterogeneous windows) =====
// Big-ref (gundam) admissions prefill on their OWN stream + scratch + cuBLAS handle so decode keeps
// replaying on GS. Single engine thread: the file-scope scratch globals are pointer-SWAPPED around the
// admission prefill only — the verified prefill/mlp_block/moe_grouped code is untouched, and decode
// graphs hold pointers captured at graph time, so swaps between windows cannot perturb them.
struct PfCtx{
    bf16 *xbuf,*nbuf,*q,*k,*v,*att,*tmp,*mg,*mu,*mh; float* glog; int* didx; float *dw,*Yf;
    kvt *kcache[NL],*vcache[NL];
    bf16 *gm_xg,*gm_gate,*gm_up,*gm_h,*gm_og; int *gm_gtok,*gm_inv; long gm_cap; int *gm_cnt,*gm_off,*gm_cur;
    cudaStream_t st; cublasHandle_t cub; void* ws;
};
static void pfctx_alloc(PfCtx& c,int S){
    size_t sH=(size_t)S*H;
    CK(cudaMalloc(&c.xbuf,sH*2));CK(cudaMalloc(&c.nbuf,sH*2));CK(cudaMalloc(&c.q,sH*2));CK(cudaMalloc(&c.k,sH*2));CK(cudaMalloc(&c.v,sH*2));CK(cudaMalloc(&c.att,sH*2));CK(cudaMalloc(&c.tmp,sH*2));
    CK(cudaMalloc(&c.mg,(size_t)S*DENSEI*2));CK(cudaMalloc(&c.mu,(size_t)S*DENSEI*2));CK(cudaMalloc(&c.mh,(size_t)S*DENSEI*2));
    CK(cudaMalloc(&c.glog,(size_t)S*NEXP*4));CK(cudaMalloc(&c.didx,(size_t)S*TOPK*4));CK(cudaMalloc(&c.dw,(size_t)S*TOPK*4));CK(cudaMalloc(&c.Yf,sH*4));
    for(int l=0;l<NL;l++){ CK(cudaMalloc(&c.kcache[l],(size_t)S*H*sizeof(kvt))); CK(cudaMalloc(&c.vcache[l],(size_t)S*H*sizeof(kvt))); }
    long tot=(long)S*TOPK; c.gm_cap=tot;                     // gm pools preallocated at PFG size -> no lazy growth in the swapped context
    CK(cudaMalloc(&c.gm_xg,(size_t)tot*H*2));CK(cudaMalloc(&c.gm_gate,(size_t)tot*MOEI*2));CK(cudaMalloc(&c.gm_up,(size_t)tot*MOEI*2));
    CK(cudaMalloc(&c.gm_h,(size_t)tot*MOEI*2));CK(cudaMalloc(&c.gm_og,(size_t)tot*H*2));CK(cudaMalloc(&c.gm_gtok,(size_t)tot*4));CK(cudaMalloc(&c.gm_inv,(size_t)tot*4));
    CK(cudaMalloc(&c.gm_cnt,NEXP*4));CK(cudaMalloc(&c.gm_off,(NEXP+1)*4));CK(cudaMalloc(&c.gm_cur,NEXP*4));
    CK(cudaStreamCreateWithFlags(&c.st,cudaStreamNonBlocking));
    CB(cublasCreate(&c.cub)); CB(cublasSetStream(c.cub,c.st)); CK(cudaMalloc(&c.ws,(size_t)32<<20)); CB(cublasSetWorkspace(c.cub,c.ws,(size_t)32<<20));
}
static void pfctx_swap(PfCtx& c){
    std::swap(xbuf,c.xbuf);std::swap(nbuf,c.nbuf);std::swap(q,c.q);std::swap(k,c.k);std::swap(v,c.v);std::swap(att,c.att);std::swap(tmp,c.tmp);
    std::swap(mg,c.mg);std::swap(mu,c.mu);std::swap(mh,c.mh);std::swap(glog,c.glog);std::swap(didx,c.didx);std::swap(dw,c.dw);std::swap(Yf,c.Yf);
    for(int l=0;l<NL;l++){ std::swap(kcache[l],c.kcache[l]); std::swap(vcache[l],c.vcache[l]); }
    std::swap(gm_xg,c.gm_xg);std::swap(gm_gate,c.gm_gate);std::swap(gm_up,c.gm_up);std::swap(gm_h,c.gm_h);std::swap(gm_og,c.gm_og);
    std::swap(gm_gtok,c.gm_gtok);std::swap(gm_inv,c.gm_inv);std::swap(gm_cap,c.gm_cap);std::swap(gm_cnt,c.gm_cnt);std::swap(gm_off,c.gm_off);std::swap(gm_cur,c.gm_cur);
    std::swap(GS,c.st); std::swap(CUB,c.cub);
}
static void pfctx_free(PfCtx& c){
    cudaFree(c.xbuf);cudaFree(c.nbuf);cudaFree(c.q);cudaFree(c.k);cudaFree(c.v);cudaFree(c.att);cudaFree(c.tmp);
    cudaFree(c.mg);cudaFree(c.mu);cudaFree(c.mh);cudaFree(c.glog);cudaFree(c.didx);cudaFree(c.dw);cudaFree(c.Yf);
    for(int l=0;l<NL;l++){cudaFree(c.kcache[l]);cudaFree(c.vcache[l]);}
    cudaFree(c.gm_xg);cudaFree(c.gm_gate);cudaFree(c.gm_up);cudaFree(c.gm_h);cudaFree(c.gm_og);cudaFree(c.gm_gtok);cudaFree(c.gm_inv);
    cudaFree(c.gm_cnt);cudaFree(c.gm_off);cudaFree(c.gm_cur);
    cublasDestroy(c.cub); cudaStreamDestroy(c.st); cudaFree(c.ws);
}
// returns device logits[last] -> caller. Runs full prefill.
static void prefill(const int* dids,int S,float* dlogits_last,bool check,std::vector<std::vector<float>>* fx,const bf16* dembeds=nullptr){
    if(dembeds && dembeds!=xbuf) CK(cudaMemcpyAsync(xbuf,dembeds,(size_t)S*H*2,cudaMemcpyDeviceToDevice,GS));  // vision/text embeds (skip self-copy if already in xbuf)
    else { dim3 eb((H+255)/256); k_embed<<<dim3(S,eb.x),256,0,GS>>>(EMB,dids,xbuf,S); }
    auto cmp=[&](int idx,const char* tag){ if(!check)return; std::vector<uint16_t> hb((size_t)S*H); CK(cudaMemcpy(hb.data(),xbuf,(size_t)S*H*2,cudaMemcpyDeviceToHost));
        double md=0,ref=0; auto&f=(*fx)[idx]; for(size_t i=0;i<(size_t)S*H;i++){float g=bf16_to_f32(hb[i]);md=fmax(md,fabsf(g-f[i]));ref=fmax(ref,fabsf(f[i]));}
        printf("  [check] %-10s max_abs=%.4f (ref_absmax=%.3f)\n",tag,md,ref); };
    cmp(0,"embeds");
    for(int l=0;l<NL;l++){ Layer&ly=L[l];
        k_rmsnorm<<<S,256,0,GS>>>(xbuf,ly.in_norm,nbuf,S,H,EPS);
        lin(nbuf,ly.q,q,S,H,H); lin(nbuf,ly.k,k,S,H,H); lin(nbuf,ly.v,v,S,H,H);
        k_rope<<<dim3(S,NH),HD/2,0,GS>>>(q,k,S,0);
        k_seedkv<<<((size_t)S*H+255)/256,256,0,GS>>>(kcache[l],k,(size_t)S*H);  // seed reference KV (quantize to KV format)
        k_seedkv<<<((size_t)S*H+255)/256,256,0,GS>>>(vcache[l],v,(size_t)S*H);
        k_attn_prefill<<<dim3(NH,S),256,0,GS>>>(q,k,v,att,S);
        lin(att,ly.o,tmp,S,H,H);
        k_add<<<(S*H+255)/256,256,0,GS>>>(xbuf,tmp,S*H);             // residual
        k_rmsnorm<<<S,256,0,GS>>>(xbuf,ly.post_norm,nbuf,S,H,EPS);
        mlp_block(ly,S);
        k_add<<<(S*H+255)/256,256,0,GS>>>(xbuf,tmp,S*H);             // residual
        if(check){ CK(cudaDeviceSynchronize()); cmp(l+1, ("after L"+std::to_string(l)).c_str()); }
    }
    k_rmsnorm<<<S,256,0,GS>>>(xbuf,FNORM,nbuf,S,H,EPS);
    lin_f32(nbuf+(size_t)(S-1)*H, LMH, dlogits_last, 1, H, V);  // last position only
}

// ===== integrated vision + tokenizer (no python) =====
extern bool vis_render_cpu(const char* pdf,int page);  // split render/GPU for interleaving; false = render failed
extern void vis_upload(); extern void vis_gpu_launch(); extern void vis_gpu_sync(); extern bf16* vis_result();
extern int vis_page_count(const char* pdf);            // -1 = unreadable
extern void vis_doc_close(const char* pdf);            // drop doc-cache entry (recycled server temp paths)
// byte-level BPE decoder (build asset vocab.bin: id -> byte-level utf8 string)
static std::vector<std::string> g_vocab; static std::map<uint32_t,unsigned char> g_bdec;
static std::vector<uint8_t> g_toknl;                     // token id -> decoded piece contains '\n' (line-region splits for confidence features)
static std::string bpe_decode(const std::vector<int>& ids);
static void load_vocab(const char* path){
    FILE* f=fopen(path,"rb"); int n; if(fread(&n,4,1,f)!=1){} g_vocab.resize(n);
    for(int i=0;i<n;i++){ int len; if(fread(&len,4,1,f)!=1){} std::string s(len,0); if(len&&fread(&s[0],1,len,f)!=(size_t)len){} g_vocab[i]=s; } fclose(f);
    std::vector<int> bs; for(int b='!';b<='~';b++)bs.push_back(b); for(int b=0xA1;b<=0xAC;b++)bs.push_back(b); for(int b=0xAE;b<=0xFF;b++)bs.push_back(b);
    std::vector<int> cs=bs; int m=0; for(int b=0;b<256;b++){bool in=false;for(int x:bs)if(x==b){in=true;break;} if(!in){bs.push_back(b);cs.push_back(256+m);m++;}}
    for(size_t i=0;i<bs.size();i++) g_bdec[(uint32_t)cs[i]]=(unsigned char)bs[i];
    g_toknl.assign(n,0); std::vector<int> one(1);
    for(int i=0;i<n;i++){ one[0]=i; if(bpe_decode(one).find('\n')!=std::string::npos) g_toknl[i]=1; }
}
static std::string bpe_decode(const std::vector<int>& ids){
    std::string ob;
    for(int id:ids){ if(id<0||id>=(int)g_vocab.size())continue; const std::string& s=g_vocab[id]; size_t i=0;
        while(i<s.size()){ unsigned char c0=s[i]; uint32_t cp; int len;
            if(c0<0x80){cp=c0;len=1;} else if((c0>>5)==0x6){cp=c0&0x1f;len=2;} else if((c0>>4)==0xe){cp=c0&0xf;len=3;} else {cp=c0&0x7;len=4;}
            for(int k=1;k<len&&i+k<s.size();k++)cp=(cp<<6)|(s[i+k]&0x3f); i+=len;
            auto it=g_bdec.find(cp); if(it!=g_bdec.end())ob.push_back((char)it->second); } }
    return ob;
}
// ===== PAGE-PARALLEL batched decode: N independent single-page streams, R-SWA per stream =====
// Each page is its own single-page OCR (own 278-tok reference + 128 ring), decoded as a batch-N step.
// Reuses prefill bf16 kernels (lin/cuBLAS, mlp_block, k_rmsnorm) with S=N; only attention/rope/argmax are new.
#define NSPLITB 32
static kvt *kcb[NL]={0},*vcb[NL]={0};
static float *atb_pm=0,*atb_pl=0,*atb_pacc=0; static bf16 *qkvbb=0; static float* dlogb=0; static int* d_tokb=0;
__global__ void k_attn_split_b(const bf16* qkvb,const kvt* kc,const kvt* vc,float* pm,float* pl,float* pacc,const int* pf,const int* dstep,int nsplit,int nafill,const int* kvoff,const int* act){
    int h=blockIdx.x, s=blockIdx.y, sp=blockIdx.z, lane=threadIdx.x;
    int dc=dstep[act[s]]+1; int clen=pf[act[s]]+(dc<WIN?dc:WIN);   // per-slot step+ref (heterogeneous window); expression shape kept -> FP codegen identical to the lockstep original
    int nss=min(nsplit,max((pf[act[s]]+WIN)/48,nafill));           // per-slot split count: a slot's softmax reduction shape depends on ITS ref, not the window max
    int chunk=(clen+nss-1)/nss, j0=sp*chunk, j1=min(clen,j0+chunk);   // sp>=nss -> j0>=clen -> NEUTRAL (merge adds exact +0.0f -> bit-equal to an nss-split launch)
    int base=(s*NH+h)*nsplit+sp, d0=lane,d1=lane+32,d2=lane+64,d3=lane+96;
    if(j0>=j1){ if(lane==0){pm[base]=-1e30f;pl[base]=0;}      // empty split (ns>keys): write NEUTRAL so merge ignores it
        pacc[(size_t)base*HD+d0]=0;pacc[(size_t)base*HD+d1]=0;pacc[(size_t)base*HD+d2]=0;pacc[(size_t)base*HD+d3]=0; return; }
    const bf16* q=qkvb+(size_t)s*3*H+(size_t)h*HD;          // rotated q for (stream s, head h)
    int ps=act[s]; const kvt* kcs=kc+(size_t)kvoff[ps]*H; const kvt* vcs=vc+(size_t)kvoff[ps]*H;
    float scale=rsqrtf((float)HD);
    float q0=__bfloat162float(q[d0]),q1=__bfloat162float(q[d1]),q2=__bfloat162float(q[d2]),q3=__bfloat162float(q[d3]);
    __shared__ float sf[512]; float m=-1e30f;
    #pragma unroll 8
    for(int j=j0;j<j1;j++){ const kvt* kj=kcs+((size_t)j*NH+h)*HD;     // ILP: overlap K loads across iters (latency-bound)
        float p=q0*kvld(kj[d0])+q1*kvld(kj[d1])+q2*kvld(kj[d2])+q3*kvld(kj[d3]);
        for(int o=16;o;o>>=1)p+=__shfl_xor_sync(~0u,p,o);
        float sc=p*scale; if(lane==0)sf[j-j0]=sc; m=fmaxf(m,sc); }
    __syncwarp();
    float l=0,a0=0,a1=0,a2=0,a3=0;
    #pragma unroll 8
    for(int j=j0;j<j1;j++){ float e=__expf(sf[j-j0]-m); const kvt* vj=vcs+((size_t)j*NH+h)*HD;
        a0+=e*kvld(vj[d0]); a1+=e*kvld(vj[d1]); a2+=e*kvld(vj[d2]); a3+=e*kvld(vj[d3]); l+=e; }
    if(lane==0){pm[base]=m;pl[base]=l;}
    pacc[(size_t)base*HD+d0]=a0; pacc[(size_t)base*HD+d1]=a1; pacc[(size_t)base*HD+d2]=a2; pacc[(size_t)base*HD+d3]=a3;
}
__global__ void k_attn_merge_b(const float* pm,const float* pl,const float* pacc,bf16* out,int nsplit){
    int h=blockIdx.x, s=blockIdx.y, d=threadIdx.x; int b0=(s*NH+h)*nsplit;   // dense pack (match k_attn_split_b)
    __shared__ float fs[NSPLITB], red[HD], lsh;
    float pmd=(d<nsplit)?pm[b0+d]:-1e30f; red[d]=pmd; __syncthreads();
    for(int st=HD/2;st>0;st>>=1){ if(d<st)red[d]=fmaxf(red[d],red[d+st]); __syncthreads(); }
    float mg=red[0]; __syncthreads();
    float fd=0; if(d<nsplit){ fd=__expf(pmd-mg); fs[d]=fd; }
    red[d]=(d<nsplit)?pl[b0+d]*fd:0; __syncthreads();
    for(int st=HD/2;st>0;st>>=1){ if(d<st)red[d]+=red[d+st]; __syncthreads(); }
    if(d==0)lsh=red[0]; __syncthreads();
    float acc=0; for(int sp=0;sp<nsplit;sp++) acc+=pacc[(size_t)(b0+sp)*HD+d]*fs[sp];
    out[(size_t)s*H+h*HD+d]=__float2bfloat16(acc/lsh);
}
__global__ void k_rope_store_b(bf16* qkvb,kvt* kc,kvt* vc,const int* pf,const int* dstep,const int* kvoff,const int* act){
    int h=blockIdx.x, s=blockIdx.y, d=threadIdx.x; int step=dstep[act[s]];   // per-slot step+ref (heterogeneous window); expression shape kept
    int p=pf[act[s]]+step; float inv=powf(ROPE_THETA,-2.f*d/HD); float ang=p*inv,c=cosf(ang),sn=sinf(ang);
    int rslot=pf[act[s]]+((p-pf[act[s]])%WIN);
    bf16* q=qkvb+(size_t)s*3*H; bf16* k=q+H; bf16* v=q+2*H;
    int ps=act[s]; kvt* kcs=kc+(size_t)kvoff[ps]*H; kvt* vcs=vc+(size_t)kvoff[ps]*H;
    size_t b=(size_t)h*HD, o=(size_t)rslot*H+b;
    float qa=__bfloat162float(q[b+d]),qb=__bfloat162float(q[b+d+HD/2]);
    q[b+d]=__float2bfloat16(qa*c-qb*sn); q[b+d+HD/2]=__float2bfloat16(qb*c+qa*sn);
    float ka=__bfloat162float(k[b+d]),kb=__bfloat162float(k[b+d+HD/2]);
    kcs[o+d]=kvst(ka*c-kb*sn); kcs[o+d+HD/2]=kvst(kb*c+ka*sn);
    vcs[o+d]=kvst(__bfloat162float(v[b+d])); vcs[o+d+HD/2]=kvst(__bfloat162float(v[b+d+HD/2]));
}
// no_repeat_ngram: per stream, ban any token that would complete an n-gram already seen in this stream's
// output (breaks degeneration loops). Large n -> only long exact repeats (loops) caught, legit short repeats kept.
__global__ void k_ngram_mask(float* logits,const int* outbuf,const int* dstep,int maxstep,const int* act,int ngram){
    int a=blockIdx.x, pos=dstep[act[a]]; if(pos<ngram-1)return;
    const int* hist=outbuf+(size_t)act[a]*maxstep; float* lg=logits+(size_t)a*V;
    int base=pos-(ngram-1);                                       // current (ngram-1)-gram = hist[base..pos-1]
    for(int j=threadIdx.x;j<base;j+=blockDim.x){                  // scan earlier (ngram-1)-grams
        bool m=true; for(int k=0;k<ngram-1;k++) if(hist[j+k]!=hist[base+k]){m=false;break;}
        if(m) lg[hist[j+ngram-1]]=-1e30f;                         // ban the token that followed it
    }
}
__global__ void k_argmax_b(const float* logits,int* tok,int Vn){
    int s=blockIdx.x; const float* lg=logits+(size_t)s*Vn;
    __shared__ float bv[256]; __shared__ int bi[256];
    float best=-1e30f; int bidx=0;
    for(int i=threadIdx.x;i<Vn;i+=256){ float x=lg[i]; if(x>best){best=x;bidx=i;} }
    bv[threadIdx.x]=best; bi[threadIdx.x]=bidx; __syncthreads();
    for(int st=128;st>0;st>>=1){ if(threadIdx.x<st && bv[threadIdx.x+st]>bv[threadIdx.x]){bv[threadIdx.x]=bv[threadIdx.x+st];bi[threadIdx.x]=bi[threadIdx.x+st];} __syncthreads(); }
    if(threadIdx.x==0) tok[s]=bi[0];
}
__global__ void k_pageembeds(bf16* out,const bf16* emb,const bf16* dembeds,int p,int bos,int t0,int t1,int t2,int t3,int vpp){
    int slot=blockIdx.x, c=blockIdx.y*256+threadIdx.x; if(c>=H)return;
    if(slot==0){ out[(size_t)slot*H+c]=emb[(size_t)bos*H+c]; return; }
    if(slot<=vpp){ int vi=1+p*vpp+(slot-1); out[(size_t)slot*H+c]=dembeds[(size_t)vi*H+c]; return; }   // vpp visual tokens/page
    int ti=slot-(vpp+1), id=(ti==0)?t0:(ti==1)?t1:(ti==2)?t2:t3;
    out[(size_t)slot*H+c]=emb[(size_t)id*H+c];
}
// batched int4 MoE: warp per (expert, row); loop the B tokens, compute only those routing to this expert.
// int4 weights (4x less bytes than the reused bf16 prefill kernel) -> the page-parallel decode bottleneck.
__global__ void k_moe_gateup_q4_b(const bf16* x,const int* didx,const uint8_t* Wg,const float* sg,const uint8_t* Wu,const float* su,bf16* hbuf,int B,int interm){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31;
    if(gw>=(long)B*TOPK*interm)return;                              // warp per ACTIVE (token,slot,row) -> no idle experts
    int r=gw%interm; long ts=gw/interm; int slot=ts%TOPK, t=ts/TOPK; int e=didx[t*TOPK+slot];
    int ng=H/QG; size_t row=(size_t)e*interm+r;
    const uint8_t* wg=Wg+row*(H/2); const uint8_t* wu=Wu+row*(H/2); const float* sgr=sg+row*ng; const float* sur=su+row*ng;
    const bf16* xt=x+(size_t)t*H; float ga[4]={0,0,0,0},ua[4]={0,0,0,0};
    for(int b=lane;b<H/2;b+=128){
        #pragma unroll
        for(int j=0;j<4;j++){ int bb=b+j*32; if(bb>=H/2)continue; int gi=(2*bb)/QG; float sgg=sgr[gi],sug=sur[gi];
            float x0=__bfloat162float(xt[2*bb]),x1=__bfloat162float(xt[2*bb+1]); uint8_t bg=wg[bb],bu=wu[bb];
            ga[j]+=sgg*(x0*I4LO(bg)+x1*I4HI(bg)); ua[j]+=sug*(x0*I4LO(bu)+x1*I4HI(bu)); } }
    float g=(ga[0]+ga[1])+(ga[2]+ga[3]), u=(ua[0]+ua[1])+(ua[2]+ua[3]);
    for(int o=16;o;o>>=1){ g+=__shfl_xor_sync(~0u,g,o); u+=__shfl_xor_sync(~0u,u,o); }
    if(lane==0) hbuf[(size_t)(t*TOPK+slot)*interm+r]=__float2bfloat16((g/(1.f+__expf(-g)))*u);
}
__global__ void k_moe_down_q4_b(const bf16* hbuf,const int* didx,const float* dw,const uint8_t* Wd,const float* sd,float* Yf,int B,int interm){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31;
    if(gw>=(long)B*H)return;                                        // warp per (token,d); loop K slots in FIXED order -> deterministic
    int d=gw%H; long t=gw/H; int ng=interm/QG; float ysum=0;
    for(int slot=0;slot<TOPK;slot++){
        int e=didx[t*TOPK+slot]; size_t row=(size_t)e*H+d;
        const uint8_t* wd=Wd+row*(interm/2); const float* sdr=sd+row*ng;
        const bf16* h=hbuf+(size_t)(t*TOPK+slot)*interm; float aa[4]={0,0,0,0};
        for(int b=lane;b<interm/2;b+=128){
            #pragma unroll
            for(int j=0;j<4;j++){ int bb=b+j*32; if(bb>=interm/2)continue; float sc=sdr[(2*bb)/QG]; uint8_t by=wd[bb];
                aa[j]+=sc*(__bfloat162float(h[2*bb])*I4LO(by)+__bfloat162float(h[2*bb+1])*I4HI(by)); } }
        float acc=(aa[0]+aa[1])+(aa[2]+aa[3]);
        for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o);
        ysum+=dw[t*TOPK+slot]*acc;
    }
    if(lane==0)Yf[(size_t)t*H+d]=ysum;                             // single write (no atomic, no pre-zero)
}
__global__ void k_gate_b(const bf16* x,const bf16* gateW,float* glog,int B){   // batched bf16 gate GEMV (avoids cuBLAS M=1 launch overhead)
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31; if(gw>=(long)B*NEXP)return;
    int e=gw%NEXP; long t=gw/NEXP; const bf16* w=gateW+(size_t)e*H; const bf16* xt=x+(size_t)t*H; float acc=0;
    for(int c=lane;c<H;c+=32) acc+=__bfloat162float(w[c])*__bfloat162float(xt[c]);
    for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o);
    if(lane==0) glog[(size_t)t*NEXP+e]=acc;
}
// ===== TCMOE: tensor-core int4 routed MoE (mma.sync m16n8k16 bf16, fp32 accum, in-register fp32 group scales).
// A = 16 weight rows (8 gate + 8 up of same r | down: d,d+8), B = 8 tokens. int4 values are EXACT in bf16
// ((0x4300|m)-136 = m-8), scale applied to the fp32 GROUP partial in registers via the C-fragment row map ->
// same products as k_moe_*_q4_b, reassociated only. Dispatch: NA>=TCMIN=4 (below, per-token kernels win on the
// 8-token mma tile's padding waste AND keep the 1pg md5 gates bit-exact). TCDBG=n A/Bs the two on live activations.
__device__ __forceinline__ uint32_t q2b(uint32_t q){ uint32_t r=(q&0x000F000Fu)|0x43004300u; const uint32_t k=0x43084308u;
    __nv_bfloat162 x=__hsub2(*(__nv_bfloat162*)&r,*(const __nv_bfloat162*)&k); return *(uint32_t*)&x; }
__device__ __forceinline__ void mma16816(float* c,uint32_t q,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(q2b(q)),"r"(q2b(q>>4)),"r"(q2b(q>>8)),"r"(q2b(q>>12)),"r"(b0),"r"(b1)); }
__global__ void k_moe_bins(const int* didx,int* cnt,int* bin,int B,int bs){   // <<<1,64>>>: fixed ascending (t,slot) scan -> deterministic bins
    int e=threadIdx.x,c=0,n=B*TOPK;
    for(int i=0;i<n;i++) if(didx[i]==e) bin[e*bs+(c++)]=i;
    cnt[e]=c; for(int i=c;i<bs;i++) bin[e*bs+i]=0;                            // pad (padded lanes are cnt-guarded)
}
// grid (64, NTILES/8, ceil(B/8)), block 128 = 4 warps x 2 m-tiles. GU: KG=10 groups, writes silu(g)*u -> hbuf[ts*MOEI+r].
// DN: KG=7, writes raw fp32 partials -> part[ts*H+d] (routing weight applied in k_moe_comb_b, fixed slot order).
__global__ void k_moe_gateup_mma(const bf16* x,const int* cnt,const int* bin,int bs,const uint4* W,const float2* S2,bf16* hbuf){
    int e=blockIdx.x,w=threadIdx.x>>5,lane=threadIdx.x&31;
    int c=cnt[e],n0=blockIdx.z*8; if(n0>=c)return;
    int nn=min(8,c-n0),j0=blockIdx.y*8+w*2;
    __shared__ bf16 xs[2][8][136]; __shared__ int bn[8];
    if(threadIdx.x<8) bn[threadIdx.x]=(threadIdx.x<nn)?bin[e*bs+n0+threadIdx.x]:0;
    __syncthreads();
    float cf[2][4]={},ac[2][4]={};
    for(int g=0;g<10;g++){
        uint32_t* xw=(uint32_t*)xs[g&1];
        for(int i=threadIdx.x;i<512;i+=128){ int n=i>>6,kw=i&63;
            xw[n*68+kw]=(n<nn)?((const uint32_t*)(x+(size_t)(bn[n]/TOPK)*H+g*QG))[kw]:0u; }
        __syncthreads();
        const uint4* wt0=W+((((size_t)e*112+j0)*10+g)<<6);  uint4 lo0=wt0[lane],hi0=wt0[32+lane];
        const uint4* wt1=W+((((size_t)e*112+j0+1)*10+g)<<6);uint4 lo1=wt1[lane],hi1=wt1[32+lane];
        #pragma unroll
        for(int s=0;s<8;s++){
            uint32_t q0=s<4?((uint32_t*)&lo0)[s]:((uint32_t*)&hi0)[s-4], q1=s<4?((uint32_t*)&lo1)[s]:((uint32_t*)&hi1)[s-4];
            uint32_t b0=xw[(lane>>2)*68+s*8+(lane&3)], b1=xw[(lane>>2)*68+s*8+4+(lane&3)];
            mma16816(cf[0],q0,b0,b1); mma16816(cf[1],q1,b0,b1);
        }
        float2 s0=S2[((size_t)e*10+g)*MOEI+8*j0+(lane>>2)], s1=S2[((size_t)e*10+g)*MOEI+8*(j0+1)+(lane>>2)];
        #pragma unroll
        for(int i=0;i<2;i++){ ac[0][i]+=s0.x*cf[0][i]; ac[0][i+2]+=s0.y*cf[0][i+2]; ac[1][i]+=s1.x*cf[1][i]; ac[1][i+2]+=s1.y*cf[1][i+2];
            cf[0][i]=cf[0][i+2]=cf[1][i]=cf[1][i+2]=0; }
    }
    int t0=2*(lane&3),t1=t0+1;
    #pragma unroll
    for(int q=0;q<2;q++){ int r=8*(j0+q)+(lane>>2); float g0=ac[q][0],g1=ac[q][1];
        if(t0<nn) hbuf[(size_t)bn[t0]*MOEI+r]=__float2bfloat16(g0/(1.f+__expf(-g0))*ac[q][2]);
        if(t1<nn) hbuf[(size_t)bn[t1]*MOEI+r]=__float2bfloat16(g1/(1.f+__expf(-g1))*ac[q][3]); }
}
__global__ void k_moe_down_mma(const bf16* hb,const int* cnt,const int* bin,int bs,const uint4* W,const float2* S2,float* part){
    int e=blockIdx.x,w=threadIdx.x>>5,lane=threadIdx.x&31;
    int c=cnt[e],n0=blockIdx.z*8; if(n0>=c)return;
    int nn=min(8,c-n0),j0=blockIdx.y*8+w*2;
    __shared__ bf16 xs[2][8][136]; __shared__ int bn[8];
    if(threadIdx.x<8) bn[threadIdx.x]=(threadIdx.x<nn)?bin[e*bs+n0+threadIdx.x]:0;
    __syncthreads();
    float cf[2][4]={},ac[2][4]={};
    for(int g=0;g<7;g++){
        uint32_t* xw=(uint32_t*)xs[g&1];
        for(int i=threadIdx.x;i<512;i+=128){ int n=i>>6,kw=i&63;
            xw[n*68+kw]=(n<nn)?((const uint32_t*)(hb+(size_t)bn[n]*MOEI+g*QG))[kw]:0u; }
        __syncthreads();
        const uint4* wt0=W+((((size_t)e*80+j0)*7+g)<<6);  uint4 lo0=wt0[lane],hi0=wt0[32+lane];
        const uint4* wt1=W+((((size_t)e*80+j0+1)*7+g)<<6);uint4 lo1=wt1[lane],hi1=wt1[32+lane];
        #pragma unroll
        for(int s=0;s<8;s++){
            uint32_t q0=s<4?((uint32_t*)&lo0)[s]:((uint32_t*)&hi0)[s-4], q1=s<4?((uint32_t*)&lo1)[s]:((uint32_t*)&hi1)[s-4];
            uint32_t b0=xw[(lane>>2)*68+s*8+(lane&3)], b1=xw[(lane>>2)*68+s*8+4+(lane&3)];
            mma16816(cf[0],q0,b0,b1); mma16816(cf[1],q1,b0,b1);
        }
        float2 s0=S2[((size_t)e*7+g)*(H/2)+j0*8+(lane>>2)], s1=S2[((size_t)e*7+g)*(H/2)+(j0+1)*8+(lane>>2)];
        #pragma unroll
        for(int i=0;i<2;i++){ ac[0][i]+=s0.x*cf[0][i]; ac[0][i+2]+=s0.y*cf[0][i+2]; ac[1][i]+=s1.x*cf[1][i]; ac[1][i+2]+=s1.y*cf[1][i+2];
            cf[0][i]=cf[0][i+2]=cf[1][i]=cf[1][i+2]=0; }
    }
    int t0=2*(lane&3),t1=t0+1;
    #pragma unroll
    for(int q=0;q<2;q++){ int d=16*(j0+q)+(lane>>2);
        if(t0<nn){ part[(size_t)bn[t0]*H+d]=ac[q][0]; part[(size_t)bn[t0]*H+d+8]=ac[q][2]; }
        if(t1<nn){ part[(size_t)bn[t1]*H+d]=ac[q][1]; part[(size_t)bn[t1]*H+d+8]=ac[q][3]; } }
}
__global__ void k_moe_comb_b(const float* part,const float* dw,float* Yf,int B){   // det: fixed slot order, single write
    int t=blockIdx.x,d=blockIdx.y*256+threadIdx.x; if(d>=H)return; float s=0;
    for(int sl=0;sl<TOPK;sl++) s+=dw[t*TOPK+sl]*part[(size_t)(t*TOPK+sl)*H+d];
    Yf[(size_t)t*H+d]=s;
}
static int *tc_cnt=0,*tc_bin=0; static float* tc_part=0; static int tc_bs=0;   // TCMOE scratch (generate_pagepar-owned)
// ===== LMHMMA: exact-rescore int4 lm_head for na>=5 (replaces the 331MB bf16 cuBLAS GEMM/step).
// int4 mma ranking (83MB) -> ngram mask -> per-block top-4 candidates (1024/token) -> bf16 rescore of
// candidates only (fp32 accum) -> argmax. Emitted token == bf16-lm_head argmax iff the true argmax is in the
// int4 top-1024 (single-stream variant: always was). Banned (ngram) tokens keep -1e30 through the rescore.
__global__ void k_lmhead_mma(const bf16* x,const uint4* W,const float2* S2,float* out,int B){   // grid(V/128, ceil(B/8)), block 128
    int w=threadIdx.x>>5,lane=threadIdx.x&31,n0=blockIdx.y*8;
    int nn=min(8,B-n0),j0=blockIdx.x*8+w*2;
    __shared__ bf16 xs[2][8][136];
    float cf[2][4]={},ac[2][4]={};
    for(int g=0;g<10;g++){
        uint32_t* xw=(uint32_t*)xs[g&1];
        for(int i=threadIdx.x;i<512;i+=128){ int n=i>>6,kw=i&63;
            xw[n*68+kw]=(n<nn)?((const uint32_t*)(x+(size_t)(n0+n)*H+g*QG))[kw]:0u; }
        __syncthreads();
        const uint4* wt0=W+(((size_t)j0*10+g)<<6);    uint4 lo0=wt0[lane],hi0=wt0[32+lane];
        const uint4* wt1=W+(((size_t)(j0+1)*10+g)<<6);uint4 lo1=wt1[lane],hi1=wt1[32+lane];
        #pragma unroll
        for(int s=0;s<8;s++){
            uint32_t q0=s<4?((uint32_t*)&lo0)[s]:((uint32_t*)&hi0)[s-4], q1=s<4?((uint32_t*)&lo1)[s]:((uint32_t*)&hi1)[s-4];
            uint32_t b0=xw[(lane>>2)*68+s*8+(lane&3)], b1=xw[(lane>>2)*68+s*8+4+(lane&3)];
            mma16816(cf[0],q0,b0,b1); mma16816(cf[1],q1,b0,b1);
        }
        float2 s0=S2[(size_t)g*(V/2)+j0*8+(lane>>2)], s1=S2[(size_t)g*(V/2)+(j0+1)*8+(lane>>2)];
        #pragma unroll
        for(int i=0;i<2;i++){ ac[0][i]+=s0.x*cf[0][i]; ac[0][i+2]+=s0.y*cf[0][i+2]; ac[1][i]+=s1.x*cf[1][i]; ac[1][i+2]+=s1.y*cf[1][i+2];
            cf[0][i]=cf[0][i+2]=cf[1][i]=cf[1][i+2]=0; }
    }
    int t0=2*(lane&3),t1=t0+1;
    #pragma unroll
    for(int q=0;q<2;q++){ int r=16*(j0+q)+(lane>>2);
        if(t0<nn){ out[(size_t)(n0+t0)*V+r]=ac[q][0]; out[(size_t)(n0+t0)*V+r+8]=ac[q][2]; }
        if(t1<nn){ out[(size_t)(n0+t1)*V+r]=ac[q][1]; out[(size_t)(n0+t1)*V+r+8]=ac[q][3]; } }
}
#define NBAM 256
#define LMTOPK 4
__global__ void k_topk_blocks_b(const float* logits,int* cand,int Vn){   // grid(NBAM,na): per-block top-4 of token by; dyn smem chunk
    int b=blockIdx.x,t=threadIdx.x; const float* lg=logits+(size_t)blockIdx.y*Vn;
    int chunk=(Vn+NBAM-1)/NBAM, base=b*chunk, end=min(Vn,base+chunk), n=end-base;
    extern __shared__ float sv[];
    for(int i=t;i<n;i+=blockDim.x) sv[i]=lg[base+i];
    __syncthreads();
    __shared__ float rv[256]; __shared__ int ri[256];
    for(int k=0;k<LMTOPK;k++){
        float lv=-1e30f; int li=-1;
        for(int i=t;i<n;i+=blockDim.x) if(sv[i]>lv){lv=sv[i];li=i;}
        rv[t]=lv; ri[t]=li; __syncthreads();
        for(int o=128;o;o>>=1){ if(t<o&&rv[t+o]>rv[t]){rv[t]=rv[t+o];ri[t]=ri[t+o];} __syncthreads(); }
        if(t==0){ cand[((size_t)blockIdx.y*NBAM+b)*LMTOPK+k]=(ri[0]>=0)?base+ri[0]:base; if(ri[0]>=0)sv[ri[0]]=-1e30f; }
        __syncthreads();
    }
}
__global__ void k_rescore_b(const bf16* x,const bf16* W,const float* logits,const int* cand,float* out,int B){ // warp/(t,cand); bf16 dot, fp32 accum
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31;
    const int nc=NBAM*LMTOPK; if(gw>=(long)B*nc)return;
    int c=gw%nc; long t=gw/nc; int v=cand[(size_t)t*nc+c];
    if(logits[(size_t)t*V+v]<=-1e29f){ if(!lane)out[(size_t)t*nc+c]=-1e30f; return; }   // ngram ban survives rescore
    const bf16* wr=W+(size_t)v*H; const bf16* xt=x+(size_t)t*H; float aa[4]={0,0,0,0};
    for(int i=lane;i<H;i+=128){
        #pragma unroll
        for(int j=0;j<4;j++){ int ii=i+j*32; if(ii<H) aa[j]+=__bfloat162float(xt[ii])*__bfloat162float(wr[ii]); } }
    float acc=(aa[0]+aa[1])+(aa[2]+aa[3]);
    for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o);
    if(!lane)out[(size_t)t*nc+c]=acc;
}
__global__ void k_argmax_cand_b(const float* val,const int* cand,int* tok){   // grid(na): winner among 1024 candidates
    const int nc=NBAM*LMTOPK; int t=blockIdx.x,x=threadIdx.x; float bv=-1e30f; int bi=0;
    for(int i=x;i<nc;i+=256){ float v=val[(size_t)t*nc+i]; if(v>bv){bv=v;bi=i;} }
    __shared__ float sv[256]; __shared__ int si[256]; sv[x]=bv; si[x]=bi; __syncthreads();
    for(int o=128;o;o>>=1){ if(x<o&&sv[x+o]>sv[x]){sv[x]=sv[x+o];si[x]=si[x+o];} __syncthreads(); }
    if(!x)tok[t]=cand[(size_t)t*nc+si[0]];
}
static int *lmh_cand=0; static float* lmh_cval=0;                             // LMHMMA scratch (generate_pagepar-owned)
#define PPSMALL 3   // NA<=PPSMALL: fp8/int4 per-token kernels (fast at small batch / single page); above: cuBLAS bf16
#define TCMIN   4   // NA>=TCMIN: tensor-core int4 MoE (mma tile = 8 tokens; below, padding waste loses to the per-token kernels)
__global__ void k_gemv_fp8_b(const bf16*,const uint8_t*,const float*,bf16*,int,int,int);          // fwd decl
__global__ void k_gemv_fp8_bias_b(const bf16*,const uint8_t*,const float*,const float*,bf16*,int,int,int);
__global__ void k_swiglu_fp8_b(const bf16*,const uint8_t*,const float*,bf16*,int,int);
#define GEMVBT(rows,B) (((long)(rows)*(B)+7)/8)
#define SWIGBT(it,B)   (((long)(it)*(B)+7)/8)
static void mlp_block_b(Layer&ly,int B){          // int4 routed MoE always; dense/shared: fp8 (small B) or cuBLAS bf16 (large B)
    bool sm=(B<=PPSMALL);
    if(ly.dense){
        if(sm){ k_swiglu_fp8_b<<<SWIGBT(DENSEI,B),256,0,GS>>>(nbuf,ly.dgu8,ly.dgu_s,mh,B,DENSEI);
                k_gemv_fp8_b<<<GEMVBT(H,B),256,0,GS>>>(mh,ly.dd8,ly.dd_s,tmp,B,H,DENSEI); }
        else  { mlp_dense(ly,nbuf,tmp,B,DENSEI,ly.d_gate,ly.d_up,ly.d_down); }
        return;
    }
    k_gate_b<<<(B*NEXP+7)/8,256,0,GS>>>(nbuf,ly.gate,glog,B);
    k_route<<<B,NEXP,0,GS>>>(glog,didx,dw,B);
    static int tcdbg=[](){const char* s=getenv("TCDBG");return s?atoi(s):0;}();  // A/B vs small-batch kernels (needs DECNOGRAPH=1)
    if(B>=TCMIN && tc_bin){                                                      // tensor-core int4 MoE (reassociated fp32; small B keeps per-token kernels + bit-exact 1pg gates)
        k_moe_bins<<<1,NEXP,0,GS>>>(didx,tc_cnt,tc_bin,B,tc_bs);
        int nt=(B+7)/8;
        k_moe_gateup_mma<<<dim3(NEXP,14,nt),128,0,GS>>>(nbuf,tc_cnt,tc_bin,tc_bs,(const uint4*)ly.egu4t,ly.sgu2,mh);
        k_moe_down_mma<<<dim3(NEXP,10,nt),128,0,GS>>>(mh,tc_cnt,tc_bin,tc_bs,(const uint4*)ly.ed4t,ly.sd2,tc_part);
        k_moe_comb_b<<<dim3(B,(H+255)/256),256,0,GS>>>(tc_part,dw,Yf,B);
        if(tcdbg>0){ tcdbg--;                                                    // compare vs old kernels on the same nbuf/didx/dw
            bf16* mh2=(bf16*)mg; float* yf2=(float*)mu;
            k_moe_gateup_q4_b<<<(B*TOPK*MOEI+7)/8,256,0,GS>>>(nbuf,didx,ly.eg8,ly.eg_s,ly.eu8,ly.eu_s,mh2,B,MOEI);
            k_moe_down_q4_b<<<(B*H+7)/8,256,0,GS>>>(mh2,didx,dw,ly.ed8,ly.ed_s,yf2,B,MOEI);
            CK(cudaStreamSynchronize(GS));
            int nmh=B*TOPK*MOEI,nyf=B*H; std::vector<bf16> a(nmh),b(nmh); std::vector<float> ya(nyf),yb(nyf);
            CK(cudaMemcpy(a.data(),mh,(size_t)nmh*2,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(),mh2,(size_t)nmh*2,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ya.data(),Yf,(size_t)nyf*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yb.data(),yf2,(size_t)nyf*4,cudaMemcpyDeviceToHost));
            float dm=0,dy=0; for(int i=0;i<nmh;i++)dm=fmaxf(dm,fabsf(__bfloat162float(a[i])-__bfloat162float(b[i])));
            for(int i=0;i<nyf;i++)dy=fmaxf(dy,fabsf(ya[i]-yb[i]));
            printf("TCDBG L%d B%d mh|Δ|max %.3e Yf|Δ|max %.3e\n",(int)(&ly-L),B,dm,dy); }
    } else {
    k_moe_gateup_q4_b<<<(B*TOPK*MOEI+7)/8,256,0,GS>>>(nbuf,didx,ly.eg8,ly.eg_s,ly.eu8,ly.eu_s,mh,B,MOEI);
    k_moe_down_q4_b<<<(B*H+7)/8,256,0,GS>>>(mh,didx,dw,ly.ed8,ly.ed_s,Yf,B,MOEI);   // deterministic: writes all Yf, no pre-zero
    }
    if(sm){ k_swiglu_fp8_b<<<SWIGBT(SHI,B),256,0,GS>>>(nbuf,ly.shgu8,ly.shgu_s,mh,B,SHI);
            k_gemv_fp8_bias_b<<<GEMVBT(H,B),256,0,GS>>>(mh,ly.shd8,ly.shd_s,Yf,tmp,B,H,SHI); }
    else  { mlp_dense(ly,nbuf,att,B,SHI,ly.sh_gate,ly.sh_up,ly.sh_down); k_combine<<<(B*H+255)/256,256,0,GS>>>(tmp,Yf,att,B*H); }
}
// batched int4 lm_head: warp per row, loop B tokens (row stays L2-resident -> ~83MB int4 vs 330MB bf16 cuBLAS).
__global__ void k_lmhead_q4_b(const bf16* x,const uint8_t* W4,const float* scale,float* out,int B){  // per-token int4 (small batch only)
    int row=blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5), lane=threadIdx.x&31; if(row>=V)return;
    int ng=H/QG; const uint8_t* wr=W4+(size_t)row*(H/2); const float* sr=scale+(size_t)row*ng;
    for(int t=0;t<B;t++){ const bf16* xt=x+(size_t)t*H; float aa[4]={0,0,0,0};
        for(int b=lane;b<H/2;b+=128){
            #pragma unroll
            for(int j=0;j<4;j++){ int bb=b+j*32; if(bb>=H/2)continue; float sc=sr[(2*bb)/QG]; uint8_t by=wr[bb];
                aa[j]+=sc*(__bfloat162float(xt[2*bb])*I4LO(by)+__bfloat162float(xt[2*bb+1])*I4HI(by)); } }
        float acc=(aa[0]+aa[1])+(aa[2]+aa[3]);
        for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(~0u,acc,o);
        if(lane==0)out[(size_t)t*V+row]=acc; }
}
// per-token fp8 projections (warp per (token,row)) — fastest at SMALL batch (used for NA<=PPSMALL / single-page)
__global__ void k_gemv_fp8_b(const bf16* x,const uint8_t* W8,const float* scale,bf16* out,int B,int rows,int cols){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31; if(gw>=(long)B*rows)return;
    int r=gw%rows; long t=gw/rows;
    float acc=dot_fp8_vec(W8+(size_t)r*cols, x+(size_t)t*cols, cols);
    if(lane==0) out[(size_t)t*rows+r]=__float2bfloat16(acc*scale[r]);
}
__global__ void k_gemv_fp8_bias_b(const bf16* x,const uint8_t* W8,const float* scale,const float* bias,bf16* out,int B,int rows,int cols){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31; if(gw>=(long)B*rows)return;
    int r=gw%rows; long t=gw/rows;
    float acc=dot_fp8_vec(W8+(size_t)r*cols, x+(size_t)t*cols, cols);
    if(lane==0) out[(size_t)t*rows+r]=__float2bfloat16(acc*scale[r]+bias[(size_t)t*rows+r]);
}
__global__ void k_swiglu_fp8_b(const bf16* x,const uint8_t* Wgu,const float* scale,bf16* out,int B,int interm){
    long gw=(long)blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5); int lane=threadIdx.x&31; if(gw>=(long)B*interm)return;
    int r=gw%interm; long t=gw/interm; const bf16* xt=x+(size_t)t*H;
    float g=dot_fp8_vec(Wgu+(size_t)r*H,xt,H)*scale[r];
    float u=dot_fp8_vec(Wgu+(size_t)(interm+r)*H,xt,H)*scale[interm+r];
    if(lane==0) out[(size_t)t*interm+r]=__float2bfloat16((g/(1.f+__expf(-g)))*u);
}
__global__ void k_record_b(int* outbuf,const int* tok,const int* dstep,int maxstep,const int* act){ int ps=act[blockIdx.x]; outbuf[(size_t)ps*maxstep+dstep[ps]]=tok[blockIdx.x]; }
__global__ void k_setdone_b(int* done,const int* tok,const int* act){ int s=blockIdx.x; if(threadIdx.x==0 && tok[s]==1) done[act[s]]=1; }
__global__ void k_incstep_b(int* steps,const int* act){ steps[act[blockIdx.x]]++; }   // per-slot decode position (windowed admission)
// Page-parallel batched decode with WINDOWED admission: at most W page-streams resident (slots); pages
// beyond W are encoded+prefilled lazily into slots freed by finished streams -> unlimited page count at
// flat VRAM. The page SOURCE feeds admission: FixedSrc = N pages of one doc (CLI/gundam — byte-identical
// to the pre-server engine; md5 gates run there), QueueSrc = the server's multi-document queue (documents
// are just a grouping of pages; their pages co-batch in the same window).
// HETEROGENEOUS WINDOW: one window holds page-streams of DIFFERENT reference sizes. Slot classes:
// [0,W) hold refs up to the call's vpp (fixed MS=PF+WIN rows each), [W,W+G) are "big" slots (fixed
// MSG=PFG+WIN rows, sized for the largest gundam tiling) fed by pages with vpp_of(id) > vpp. Per-slot
// d_pf[]/d_kvoff[] replace the scalar pf/stride; each slot starts, steps, and retires independently.
// Base pages never wait on big slots and vice versa -> no head-of-line blocking between shapes.
// Invariants: a stream's positions are 0..PF-1 (its own ref) + PF+step — a pure function of its page, never
// of admission time; slot recycling frees whole streams (no renumbering); WIN=128 untouched.
// Source hooks run on the engine thread ONLY, at admission/16-step sync boundaries, never inside graph capture.
#define GVPP_MAX 3793   // largest gundam tiling: h=32,w=1 -> h*10*(w*10+1)+273 (gundam_ratio caps at 32 tiles)
static int gslots_env(){ static int g=[](){const char* s=getenv("GSLOTS");return s?atoi(s):32;}(); return g; }  // big-slot cap (capacity knob, like WINDOW)
struct PageSrc{
    virtual int  next(bool base_ok,bool big_ok)=0;            // next page id admissible NOW given free slot classes; -1 = none
    virtual int  vpp_of(int id){ (void)id; return -1; }       // per-page ref size; -1 = the call's uniform vpp
    virtual int  vpp_gmax(){ return 0; }                      // >0: may yield refs larger than vpp (server gundam) -> allocate big slots
    virtual const bf16* embeds(int id,bf16* scr,int* pidx)=0; // visual embeds (encode into scr, or upfront buffer + *pidx); nullptr = page unrenderable -> skipped, no slot consumed
    virtual std::vector<int>& out(int id)=0;                  // per-page token sink
    virtual void stats(int id,const PageConf& pc){ (void)id;(void)pc; }   // decode-confidence features (before done)
    virtual void done(int id){}                               // page complete (EOS-at-prefill or retire)
    virtual bool wait(){return false;}                        // NA==0 & none available: block for more work? false = drain + return
    virtual bool lazy(){return true;}                         // encode-on-admission (vs upfront dembeds)
    virtual bool verbose(){return true;}                      // CLI stat prints (TTFT/prefill/decode/windowed)
    // ASYNC big-ref admission (server gundam pages): encode launches on the vision stream and the core
    // polls completion at boundaries — decode keeps replaying meanwhile. Depth 1: result valid until next begin.
    virtual int  enc_begin(int id){ (void)id; return -1; }    // launch async encode; ntok or -1 = fail/unsupported
    virtual bool enc_ready(){ return true; }                  // in-flight encode finished on the GPU?
    virtual void enc_wait(){}                                 // block until finished (idle window: nothing to overlap)
    virtual const bf16* enc_result(){ return nullptr; }       // encoded ref (device), valid until the next enc_begin
    virtual bool alive(int id){ (void)id; return true; }      // page's job still live (skip dead work cheaply)
    virtual ~PageSrc(){}
};
// ===== per-token decode confidence (read-only; token selection untouched) =====
// p1 = softmax prob of the emitted token; ent = Shannon entropy of the step distribution (nats).
// Small-NA path: exact over the FULL logits (emitted token is the post-mask max). Rescore path: over
// the 1024 exact-rescored candidates (tail mass negligible — the same property that makes the rescore
// token-exact). Written as per-slot per-STEP traces; features computed host-side at retire.
// Entropy from the same pass: with logits shifted by max mt, H = ln(Z) - (sum e^x * x)/Z.
__global__ void k_conf_b(const float* logits,const int* tok,float* p1t,float* entt,const int* dstep,int maxstep,const int* act,int Vn){  // grid(na) block 1024
    int s=blockIdx.x; const float* lg=logits+(size_t)s*Vn;
    float mt=lg[tok[s]];                                 // emitted token = post-mask argmax -> its logit IS the max (no max pass needed)
    __shared__ float se[1024],sx[1024];
    float e=0,w=0;
    for(int i=threadIdx.x;i<Vn;i+=1024){ float x=lg[i]; if(x>-1e29f){ float ex=__expf(x-mt); e+=ex; w+=ex*(x-mt); } }
    se[threadIdx.x]=e; sx[threadIdx.x]=w; __syncthreads();
    for(int o=512;o;o>>=1){ if(threadIdx.x<o){ se[threadIdx.x]+=se[threadIdx.x+o]; sx[threadIdx.x]+=sx[threadIdx.x+o]; } __syncthreads(); }
    if(!threadIdx.x){ int ps=act[s]; float Z=se[0]; size_t o=(size_t)ps*maxstep+dstep[ps];
        p1t[o]=1.f/Z; entt[o]=__logf(Z)-sx[0]/Z; }
}
__global__ void k_conf_cand_b(const float* cval,float* p1t,float* entt,const int* dstep,int maxstep,const int* act,int nc){ // grid(na) block 256
    int s=blockIdx.x; const float* v=cval+(size_t)s*nc;
    __shared__ float mx[256], se[256], sx[256];
    float m=-1e30f; for(int i=threadIdx.x;i<nc;i+=256) m=fmaxf(m,v[i]);
    mx[threadIdx.x]=m; __syncthreads();
    for(int o=128;o;o>>=1){ if(threadIdx.x<o) mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+o]); __syncthreads(); }
    float mt=mx[0]; float e=0,w=0;
    for(int i=threadIdx.x;i<nc;i+=256){ float x=v[i]; if(x>-1e29f){ float ex=__expf(x-mt); e+=ex; w+=ex*(x-mt); } }
    se[threadIdx.x]=e; sx[threadIdx.x]=w; __syncthreads();
    for(int o=128;o;o>>=1){ if(threadIdx.x<o){ se[threadIdx.x]+=se[threadIdx.x+o]; sx[threadIdx.x]+=sx[threadIdx.x+o]; } __syncthreads(); }
    if(!threadIdx.x){ int ps=act[s]; float Z=se[0]; size_t o=(size_t)ps*maxstep+dstep[ps];
        p1t[o]=1.f/Z; entt[o]=__logf(Z)-sx[0]/Z; }
}
// host-side feature extraction from the traces (at retire). toks aligned with trace steps; post-EOS
// steps (stream keeps stepping to the boundary) are EXCLUDED — cut at the first EOS.
static PageConf conf_features(float conf0,float ent0,const std::vector<float>& p1,const std::vector<float>& ent,const std::vector<int>& toks){
    PageConf pc; int n=(int)p1.size();
    for(int i=0;i<n;i++) if(i<(int)toks.size()&&toks[i]==1){ n=i+1; break; }   // cut at EOS
    std::vector<float> p; p.reserve(n+1); p.push_back(conf0); for(int i=0;i<n;i++) p.push_back(p1[i]);
    std::vector<float> e; e.reserve(n+1); e.push_back(ent0);  for(int i=0;i<n;i++) e.push_back(ent[i]);
    int m=(int)p.size(); pc.ntok=m;
    double ps=0,es=0; int lo=0;
    for(int i=0;i<m;i++){ ps+=p[i]; es+=e[i]; if(p[i]<0.5f)lo++; }
    pc.conf=(float)(ps/m); pc.lowf=(float)lo/m; pc.emean=(float)(es/m);
    std::vector<float> srt=p; std::sort(srt.begin(),srt.end());
    pc.p10=srt[(size_t)(0.10*(m-1))];
    const int WI=32;                                     // sliding windows (errors cluster locally)
    if(m<=WI){ pc.wminp=pc.conf; pc.wment=pc.emean; }
    else{ double wp=0,we=0; for(int i=0;i<WI;i++){ wp+=p[i]; we+=e[i]; }
          double mnp=wp,mxe=we;
          for(int i=WI;i<m;i++){ wp+=p[i]-p[i-WI]; we+=e[i]-e[i-WI]; mnp=std::min(mnp,wp); mxe=std::max(mxe,we); }
          pc.wminp=(float)(mnp/WI); pc.wment=(float)(mxe/WI); }
    pc.regp=pc.conf;                                     // worst line-region mean p1 (fallback: page mean when vocab not loaded)
    if(!g_toknl.empty()){
        const int RMIN=24; float best=1e9f; double rs=0; int rn=0; bool any=false;
        for(int i=0;i<n;i++){ rs+=p1[i]; rn++;
            bool nl=(size_t)toks[i]<g_toknl.size() && g_toknl[toks[i]];
            if((nl&&rn>=RMIN)||i==n-1){ if(rn>=RMIN||!any){ best=std::min(best,(float)(rs/rn)); any=true; } rs=0; rn=0; } }
        if(any) pc.regp=best;
    }
    return pc;
}
static void generate_pagepar_core(PageSrc& src,int vpp,int W,int Ntotal){
    const int PF=1+vpp+4, MS=PF+WIN;                     // ref = 1 bos + vpp visual + 4 prompt (273=Base, larger=Gundam)
    const int GV=src.vpp_gmax();                         // >0: heterogeneous window — G big slots for refs larger than vpp
    const int PFG=GV?1+GV+4:0, MSG=GV?PFG+WIN:0;
    int G=0;
    if(GV){ size_t freeb=0,totalb=0; cudaMemGetInfo(&freeb,&totalb);            // big-slot count from free VRAM (like vram_wcap)
        size_t chunk=(size_t)MSG*H*NL*2*sizeof(kvt);                            // K+V all layers, one big slot
        long g=(long)(((double)freeb-3.5e9)/chunk);                             // reserve: PFG prefill scratch + per-slot decode scratch + headroom
        G=(int)std::max(1L,std::min((long)gslots_env(),g)); }
    const int WT=W+G;                                    // slot ids: [0,W) = vpp-shaped, [W,WT) = big
    alloc(std::max(PF,WT));                              // big-ref prefills run in their own PfCtx (async), so the shared scratch only needs PF
    std::vector<int> hkvoff(WT),hpf(WT,PF);              // per-slot KV row offset (constant) + ref length (set at admission)
    for(int s=0;s<WT;s++) hkvoff[s]=s<W?s*MS:W*MS+(s-W)*MSG;
    for(int s=W;s<WT;s++) hpf[s]=PFG;
    for(int l=0;l<NL;l++){ CK(cudaMalloc(&kcb[l],((size_t)W*MS+(size_t)G*MSG)*H*sizeof(kvt))); CK(cudaMalloc(&vcb[l],((size_t)W*MS+(size_t)G*MSG)*H*sizeof(kvt))); }
    int *d_pf,*d_kvoff; CK(cudaMalloc(&d_pf,(size_t)WT*4)); CK(cudaMalloc(&d_kvoff,(size_t)WT*4));
    CK(cudaMemcpy(d_pf,hpf.data(),(size_t)WT*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_kvoff,hkvoff.data(),(size_t)WT*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&atb_pm,(size_t)WT*NH*NSPLITB*4)); CK(cudaMalloc(&atb_pl,(size_t)WT*NH*NSPLITB*4)); CK(cudaMalloc(&atb_pacc,(size_t)WT*NH*NSPLITB*HD*4));
    tc_bs=std::max(16,WT); CK(cudaMalloc(&tc_cnt,NEXP*4)); CK(cudaMalloc(&tc_bin,(size_t)NEXP*tc_bs*4)); CK(cudaMalloc(&tc_part,(size_t)WT*TOPK*H*4));
    CK(cudaMalloc(&lmh_cand,(size_t)WT*NBAM*LMTOPK*4)); CK(cudaMalloc(&lmh_cval,(size_t)WT*NBAM*LMTOPK*4));
    std::vector<float> conf0(WT,0.f),ent0v(WT,0.f);                  // prefill first-token p1/entropy per slot (host-computed)
    CK(cudaMalloc(&qkvbb,(size_t)WT*3*H*2)); CK(cudaMalloc(&dlogb,(size_t)WT*V*4)); CK(cudaMalloc(&d_tokb,(size_t)WT*4));
    const int MAXSTEP=4096;
    float *d_p1t,*d_entt;                                            // per-slot per-STEP confidence traces (read-only wrt token selection)
    CK(cudaMalloc(&d_p1t,(size_t)WT*MAXSTEP*4)); CK(cudaMalloc(&d_entt,(size_t)WT*MAXSTEP*4));
    int *d_steps,*outbuf,*d_done,*d_act;                             // all slot-indexed; d_act maps active idx -> slot
    CK(cudaMalloc(&d_steps,(size_t)WT*4)); CK(cudaMalloc(&outbuf,(size_t)WT*MAXSTEP*4)); CK(cudaMalloc(&d_done,(size_t)WT*4)); CK(cudaMalloc(&d_act,(size_t)WT*4));
    CK(cudaMemset(outbuf,0,(size_t)WT*MAXSTEP*4));                   // slots are re-read up to each stream's own step only; memset keeps initcheck clean
    CK(cudaMemset(d_steps,0,(size_t)WT*4)); CK(cudaMemset(d_done,0,(size_t)WT*4));
    std::vector<int> active,curtok,hsteps(WT,0),hdone(WT,0),slot2page(WT,0),freesl,freegs;
    for(int s=W-1;s>=0;s--) freesl.push_back(s);
    for(int s=WT-1;s>=W;s--) freegs.push_back(s);
    bool ttft=false; float admms=0; long nadm=0; long total=0;       // cumulative admission (encode+prefill) time; total emitted tokens
    bool vb=src.verbose();
    bf16 *pe,*vis1=nullptr; float* dlog; CK(cudaMalloc(&pe,(size_t)PF*H*2)); CK(cudaMalloc(&dlog,(size_t)V*4));
    if(src.lazy()) CK(cudaMalloc(&vis1,(size_t)(1+(size_t)std::max(vpp,GV))*H*2));
    PfCtx c2{}; bf16 *pe2=nullptr,*vis2=nullptr; float* dlog2=nullptr;   // ASYNC big-ref admission context (server gundam pages)
    if(G){ pfctx_alloc(c2,PFG); CK(cudaMalloc(&pe2,(size_t)PFG*H*2)); CK(cudaMalloc(&vis2,(size_t)(1+(size_t)GV)*H*2)); CK(cudaMalloc(&dlog2,(size_t)V*4)); }
    cudaEvent_t pa,pb,aa,ab; cudaEventCreate(&pa);cudaEventCreate(&pb);cudaEventCreate(&aa);cudaEventCreate(&ab);
    CK(cudaStreamSynchronize(GS)); cudaEventRecord(pa,GS);
    auto admit=[&](){                                                // SYNC admission, base-class slots only: encode -> prefill -> seed slot KV
        if(freesl.empty()) return;                                   // (big-ref pages go through the async pump; this path defines the idle-parity NA trajectory)
        int pid=src.next(true,false); if(pid<0) return;
        cudaEventRecord(aa,GS);
        for(;;){
            int pv=src.vpp_of(pid); if(pv<0)pv=vpp;
            int slot=freesl.back(); int PFp=1+pv+4;
            int pidx=0; const bf16* semb=src.embeds(pid,vis1,&pidx); // slot-local embeds: encoded page lives at index 0
            if(semb){
                k_pageembeds<<<dim3(PFp,(H+255)/256),256,0,GS>>>(pe,EMB,semb,pidx,0,37460,4366,76466,16,pv);
                prefill(nullptr,PFp,dlog,false,nullptr,pe);
                for(int l=0;l<NL;l++){ CK(cudaMemcpyAsync(kcb[l]+(size_t)hkvoff[slot]*H,kcache[l],(size_t)PFp*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS));
                                       CK(cudaMemcpyAsync(vcb[l]+(size_t)hkvoff[slot]*H,vcache[l],(size_t)PFp*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS)); }
                std::vector<float> ll(V); CK(cudaMemcpyAsync(ll.data(),dlog,(size_t)V*4,cudaMemcpyDeviceToHost,GS)); CK(cudaStreamSynchronize(GS));  // stream-scoped: don't drain VS (in-flight async encode)
                int t=0; for(int e=1;e<V;e++) if(ll[e]>ll[t])t=e; src.out(pid).push_back(t); total++;
                double cz=0,sxe=0; for(int e=0;e<V;e++){ double d=(double)ll[e]-ll[t],ex=exp(d); cz+=ex; sxe+=ex*d; }   // first-token p1+entropy (host; logits already here)
                float p1=(float)(1.0/cz),e0=(float)(log(cz)-sxe/cz);
                if(!ttft){ ttft=true; cudaEventRecord(pb,GS); CK(cudaEventSynchronize(pb)); float ms=0; cudaEventElapsedTime(&ms,pa,pb);
                           if(vb)printf("TTFT (page 0 %sprefill): %.0f ms\n",src.lazy()?"encode+":"",ms); }
                if(t!=1){ freesl.pop_back(); slot2page[slot]=pid; hsteps[slot]=0; hdone[slot]=0; active.push_back(slot); curtok.push_back(t);
                          if(hpf[slot]!=PFp){ hpf[slot]=PFp; CK(cudaMemcpyAsync(d_pf+slot,&hpf[slot],4,cudaMemcpyHostToDevice,GS)); }
                          conf0[slot]=p1; ent0v[slot]=e0; }
                else { src.stats(pid,conf_features(p1,e0,{},{},{})); src.done(pid); }   // EOS at prefill: page complete without ever holding a slot
            } else src.done(pid);                                    // unrenderable page: skipped, completes empty
            nadm++;
            if(freesl.empty()) break;
            if((pid=src.next(true,false))<0) break;
        }
        cudaEventRecord(ab,GS); CK(cudaEventSynchronize(ab)); float ms=0; cudaEventElapsedTime(&ms,aa,ab); admms+=ms;
    };
    // ASYNC big-ref admission pump: encode runs on the vision stream, prefill+seed on c2.st with swapped
    // scratch — the decode window keeps replaying on GS throughout. Depth 1 (vision buffers are single-
    // instance); the completed page joins the window at the next boundary via ready[]/activate().
    struct{ int pid=-1,pv=0,stage=0; } ap;                           // stage: 0 idle, 1 encode in flight
    struct Ready{ int slot,pid,tok; float p1,e0; }; std::vector<Ready> ready;
    auto pump=[&](bool block){
        if(!G) return;
        if(ap.stage==1){
            if(block) src.enc_wait();
            if(src.enc_ready()){
                if(!src.alive(ap.pid)){ src.done(ap.pid); nadm++; ap.stage=0; }      // job died while encoding: page completes empty
                else if(!freegs.empty()){
                    int slot=freegs.back(); int PFp=1+ap.pv+4;
                    pfctx_swap(c2);                                  // prefill on the admission stream+scratch (verified code path, swapped pointers)
                    CK(cudaMemcpyAsync(vis2+H,src.enc_result(),(size_t)ap.pv*H*2,cudaMemcpyDeviceToDevice,GS));   // k_pageembeds expects visual tokens at ROW 1 (slot-local embeds convention)
                    k_pageembeds<<<dim3(PFp,(H+255)/256),256,0,GS>>>(pe2,EMB,vis2,0,0,37460,4366,76466,16,ap.pv);
                    prefill(nullptr,PFp,dlog2,false,nullptr,pe2);
                    for(int l=0;l<NL;l++){ CK(cudaMemcpyAsync(kcb[l]+(size_t)hkvoff[slot]*H,kcache[l],(size_t)PFp*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS));
                                           CK(cudaMemcpyAsync(vcb[l]+(size_t)hkvoff[slot]*H,vcache[l],(size_t)PFp*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS)); }
                    std::vector<float> ll(V); CK(cudaMemcpyAsync(ll.data(),dlog2,(size_t)V*4,cudaMemcpyDeviceToHost,GS)); CK(cudaStreamSynchronize(GS));
                    pfctx_swap(c2);                                  // back to the decode context
                    int t=0; for(int e=1;e<V;e++) if(ll[e]>ll[t])t=e; src.out(ap.pid).push_back(t); total++;
                    double cz=0,sxe=0; for(int e=0;e<V;e++){ double d=(double)ll[e]-ll[t],ex=exp(d); cz+=ex; sxe+=ex*d; }
                    float p1=(float)(1.0/cz),e0=(float)(log(cz)-sxe/cz);
                    if(t!=1){ freegs.pop_back(); hpf[slot]=PFp; ready.push_back({slot,ap.pid,t,p1,e0}); }
                    else { src.stats(ap.pid,conf_features(p1,e0,{},{},{})); src.done(ap.pid); }
                    nadm++; ap.stage=0;
                }                                                    // else: no free big slot — keep the encode parked (result valid until next begin)
            }
        }
        if(ap.stage==0 && !freegs.empty()){                          // start the next big-ref encode (its CPU render + GPU overlap the window)
            int pid=src.next(false,true); if(pid<0) return;
            int pv=src.vpp_of(pid);
            if(src.enc_begin(pid)<0){ src.done(pid); nadm++; return; }   // render failed / job dead: page completes empty
            ap.pid=pid; ap.pv=pv; ap.stage=1;
        }
    };
    auto activate=[&](){                                             // async-admitted big slots join the window at a boundary
        for(Ready& r:ready){
            slot2page[r.slot]=r.pid; hsteps[r.slot]=0; hdone[r.slot]=0; active.push_back(r.slot); curtok.push_back(r.tok);
            CK(cudaMemcpyAsync(d_pf+r.slot,&hpf[r.slot],4,cudaMemcpyHostToDevice,GS));
            conf0[r.slot]=r.p1; ent0v[r.slot]=r.e0;
        }
        ready.clear();
    };
    admit();                                             // initial fill: first W pages (windowed) or all (classic)
    if(vb)printf("page-parallel prefill: %ld/%d pages in %.0f ms%s\n",nadm,Ntotal,admms,W<Ntotal?" (windowed)":"");
    int NA=(int)active.size();
    CK(cudaMemcpy(d_act,active.data(),(size_t)NA*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tokb,curtok.data(),(size_t)NA*4,cudaMemcpyHostToDevice));
    int ns=(PF+WIN)/48; if(ns<1)ns=1; if(ns>NSPLITB)ns=NSPLITB;   // base; recomputed per-NA in loop to fill SMs at small NA
    const int ngram=16;                                            // no_repeat_ngram (always on; kills degeneration loops, safe for clean docs)
    static void* cubws=nullptr; if(!cubws)CK(cudaMalloc(&cubws,(size_t)32<<20)); CB(cublasSetWorkspace(CUB,cubws,(size_t)32<<20)); // persistent workspace (cuBLAS holds the ptr -> must NOT free)
    auto body=[&](int na){                                          // one decode step captured as a graph, replayed per window
        k_embed<<<dim3(na,(H+255)/256),256,0,GS>>>(EMB,d_tokb,xbuf,na);
        k_rmsnorm<<<na,256,0,GS>>>(xbuf,L[0].in_norm,nbuf,na,H,EPS);
        bool sm=(na<=PPSMALL);
        for(int l=0;l<NL;l++){ Layer&ly=L[l];
            if(sm) k_gemv_fp8_b<<<GEMVBT(3*H,na),256,0,GS>>>(nbuf,ly.qkv8,ly.qkv_s,qkvbb,na,3*H,H); else lin(nbuf,ly.qkv,qkvbb,na,H,3*H);
            k_rope_store_b<<<dim3(NH,na),HD/2,0,GS>>>(qkvbb,kcb[l],vcb[l],d_pf,d_steps,d_kvoff,d_act);
            k_attn_split_b<<<dim3(NH,na,ns),32,0,GS>>>(qkvbb,kcb[l],vcb[l],atb_pm,atb_pl,atb_pacc,d_pf,d_steps,ns,(284+NH*na-1)/(NH*na),d_kvoff,d_act);
            k_attn_merge_b<<<dim3(NH,na),HD,0,GS>>>(atb_pm,atb_pl,atb_pacc,att,ns);
            if(sm) k_gemv_fp8_b<<<GEMVBT(H,na),256,0,GS>>>(att,ly.o8,ly.o_s,tmp,na,H,H); else lin(att,ly.o,tmp,na,H,H);
            k_add_rmsnorm<<<na,256,0,GS>>>(xbuf,tmp,ly.post_norm,nbuf,H,EPS);   // fused residual+norm
            mlp_block_b(ly,na);
            const bf16* nn=(l+1<NL)?L[l+1].in_norm:FNORM;
            k_add_rmsnorm<<<na,256,0,GS>>>(xbuf,tmp,nn,nbuf,H,EPS);             // fused residual+norm
        }
        static int lmhdbg=[](){const char* s=getenv("LMHDBG");return s?atoi(s):0;}();     // token-parity A/B vs bf16 cuBLAS full logits (needs DECNOGRAPH)
        bool lmx=(na>4 && lmh_cand);                                                      // exact-rescore path (token-exact vs cuBLAS, verified)
        if(lmx) k_lmhead_mma<<<dim3(V/128,(na+7)/8),128,0,GS>>>(nbuf,(const uint4*)LMH4T,LMH_S2,dlogb,na);   // int4 ranking, 83MB vs 331MB
        else    k_lmhead_q4_b<<<(V+7)/8,256,0,GS>>>(nbuf,LMH4,lmh_s4,dlogb,na);
        k_ngram_mask<<<na,256,0,GS>>>(dlogb,outbuf,d_steps,MAXSTEP,d_act,ngram);         // no_repeat_ngram (always on; masks BEFORE candidates)
        if(lmx){
            k_topk_blocks_b<<<dim3(NBAM,na),256,((V+NBAM-1)/NBAM)*4,GS>>>(dlogb,lmh_cand,V);
            k_rescore_b<<<(int)(((long)na*NBAM*LMTOPK+7)/8),256,0,GS>>>(nbuf,LMH,dlogb,lmh_cand,lmh_cval,na);
            k_argmax_cand_b<<<na,256,0,GS>>>(lmh_cval,lmh_cand,d_tokb);
            if(lmhdbg>0){ lmhdbg--;                                                      // old path on same nbuf -> token parity
                static float* dbglog=0; static int* dbgtok=0; static int dbgcap=0;
                if(na>dbgcap){ if(dbglog){cudaFree(dbglog);cudaFree(dbgtok);} CK(cudaMalloc(&dbglog,(size_t)na*V*4));CK(cudaMalloc(&dbgtok,(size_t)na*4)); dbgcap=na; }
                lin_f32(nbuf,LMH,dbglog,na,H,V);
                k_ngram_mask<<<na,256,0,GS>>>(dbglog,outbuf,d_steps,MAXSTEP,d_act,ngram);
                k_argmax_b<<<na,256,0,GS>>>(dbglog,dbgtok,V);
                CK(cudaStreamSynchronize(GS));
                std::vector<int> tn(na),to(na);
                CK(cudaMemcpy(tn.data(),d_tokb,na*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(to.data(),dbgtok,na*4,cudaMemcpyDeviceToHost));
                int mm=0; for(int i=0;i<na;i++) mm+=(tn[i]!=to[i]);
                if(mm) printf("LMHDBG na%d MISMATCH %d/%d\n",na,mm,na); }
        }
        else k_argmax_b<<<na,256,0,GS>>>(dlogb,d_tokb,V);
        if(lmx) k_conf_cand_b<<<na,256,0,GS>>>(lmh_cval,d_p1t,d_entt,d_steps,MAXSTEP,d_act,NBAM*LMTOPK);   // read-only confidence traces
        else    k_conf_b<<<na,1024,0,GS>>>(dlogb,d_tokb,d_p1t,d_entt,d_steps,MAXSTEP,d_act,V);
        k_record_b<<<na,1,0,GS>>>(outbuf,d_tokb,d_steps,MAXSTEP,d_act);
        k_setdone_b<<<na,1,0,GS>>>(d_done,d_tokb,d_act);
        k_incstep_b<<<na,1,0,GS>>>(d_steps,d_act);
    };
    std::map<int,cudaGraphExec_t> gcache;                           // one captured step-graph per (active-count, key-split count) — ns is baked into the grid
    cudaEvent_t da,db; cudaEventCreate(&da);cudaEventCreate(&db); CK(cudaStreamSynchronize(GS)); cudaEventRecord(da,GS);
    long wsteps=0; const int CK_EVERY=16; long stream_steps=0; bool ng=getenv("DECNOGRAPH");
    auto retire=[&](int slot){                                       // stream done: harvest its outbuf row + confidence traces, free the slot
        int n=hsteps[slot]; std::vector<int>& out=src.out(slot2page[slot]);
        std::vector<int> row(n>0?n:0); std::vector<float> p1v(n>0?n:0),env(n>0?n:0);   // copies stream-scoped to GS: a legacy-stream memcpy would drain VS (in-flight async encode)
        if(n>0){ CK(cudaMemcpyAsync(row.data(),outbuf+(size_t)slot*MAXSTEP,(size_t)n*4,cudaMemcpyDeviceToHost,GS));
                 CK(cudaMemcpyAsync(p1v.data(),d_p1t+(size_t)slot*MAXSTEP,(size_t)n*4,cudaMemcpyDeviceToHost,GS));
                 CK(cudaMemcpyAsync(env.data(),d_entt+(size_t)slot*MAXSTEP,(size_t)n*4,cudaMemcpyDeviceToHost,GS)); }
        CK(cudaStreamSynchronize(GS));
        if(n>0) for(int t:row){ out.push_back(t); total++; if(t==1)break; }
        src.stats(slot2page[slot],conf_features(conf0[slot],ent0v[slot],p1v,env,row));
        src.done(slot2page[slot]);
        (slot<W?freesl:freegs).push_back(slot);
    };
    for(;;){
        if(NA==0){
            pump(ap.stage==1);                                       // idle window: drive async admissions FIRST (pending gundam items live in the source's rotation, not the HTTP queue) — block on an in-flight encode (nothing to overlap)
            activate();
            admit(); NA=(int)active.size();
            if(NA==0){
                if(ap.stage||!ready.empty()) continue;               // async admission in flight: don't block on the queue
                if(!src.wait()) break;                               // truly idle: block for new jobs (FixedSrc: drained -> exit)
                continue;
            }
            // server: woke on new work; full device-state re-upload (slots recycled while idle)
            CK(cudaMemcpyAsync(d_act,active.data(),(size_t)NA*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_tokb,curtok.data(),(size_t)NA*4,cudaMemcpyHostToDevice,GS));
            CK(cudaMemcpyAsync(d_steps,hsteps.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_done,hdone.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS));
        }
        int pfmx=0; for(int s:active) pfmx=std::max(pfmx,hpf[s]);    // heterogeneous window: splits sized to the longest ACTIVE ref (uniform window -> exactly the old scalar)
        ns=std::min(NSPLITB,std::max((pfmx+WIN)/48,(284+NH*NA-1)/(NH*NA)));   // more key-splits at small NA to fill the 142 SMs (24 keys/split tried 2026-07: +1% = noise, 0.12% output drift -> reverted)
        int gkey=NA*(NSPLITB+1)+ns;
        if(!ng && !gcache.count(gkey)){ cudaGraph_t g; CK(cudaStreamBeginCapture(GS,cudaStreamCaptureModeThreadLocal)); body(NA);
            CK(cudaStreamEndCapture(GS,&g)); cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)); cudaGraphDestroy(g); gcache[gkey]=ge; }
        int hi=0; for(int s:active) hi=std::max(hi,hsteps[s]);       // outbuf rows are MAXSTEP deep -> cap the window at the furthest stream
        int nstep=std::min(CK_EVERY,MAXSTEP-hi);
        for(int i=0;i<nstep;i++,wsteps++){ stream_steps+=NA; if(ng)body(NA); else CK(cudaGraphLaunch(gcache[gkey],GS)); }
        pump(false);                                                 // OVERLAP: big-ref encode/prefill runs while the queued window replays on GS
        CK(cudaStreamSynchronize(GS));                               // recompact: retire streams that hit EOS (or the per-stream step cap)
        CK(cudaMemcpyAsync(hdone.data(),d_done,(size_t)WT*4,cudaMemcpyDeviceToHost,GS));
        CK(cudaMemcpyAsync(hsteps.data(),d_steps,(size_t)WT*4,cudaMemcpyDeviceToHost,GS));
        std::vector<int> ct(NA); CK(cudaMemcpyAsync(ct.data(),d_tokb,(size_t)NA*4,cudaMemcpyDeviceToHost,GS)); CK(cudaStreamSynchronize(GS));
        std::vector<int> na, nt;
        for(int a=0;a<NA;a++){ int s=active[a]; if(hdone[s]||hsteps[s]>=MAXSTEP) retire(s); else { na.push_back(s); nt.push_back(ct[a]); } }
        active=na; curtok=nt;
        activate();                                                  // async-admitted big slots join here
        admit();                                                     // top the window back up (no-op when all pages already admitted)
        NA=(int)active.size();
        // BOOST: mixed window -> extra base-only sub-window. Base steps are ~10x cheaper (406-row clen,
        // small ns) than the admission stalls big-ref pages inject (~0.5s encode+prefill each), so fast
        // pages would otherwise be rate-limited by gundam admission: latency-class head-of-line blocking.
        // Positions stay per-stream pure functions -> per-page numerics untouched (co-batch NA variation class).
        const int BOOST=48;
        if(NA>0 && G){
            std::vector<int> bi; for(int a=0;a<NA;a++) if(active[a]<W) bi.push_back(a);
            int NB=(int)bi.size();
            if(NB>0 && NB<NA){                                       // both classes active
                std::vector<int> bact(NB),btok(NB);
                for(int i=0;i<NB;i++){ bact[i]=active[bi[i]]; btok[i]=curtok[bi[i]]; }
                int bhi=0,bpf=0; for(int s:bact){ bhi=std::max(bhi,hsteps[s]); bpf=std::max(bpf,hpf[s]); }
                int bstep=std::min(BOOST,MAXSTEP-bhi);
                CK(cudaMemcpyAsync(d_act,bact.data(),(size_t)NB*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_tokb,btok.data(),(size_t)NB*4,cudaMemcpyHostToDevice,GS));
                CK(cudaMemcpyAsync(d_steps,hsteps.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_done,hdone.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS));
                int bns=std::min(NSPLITB,std::max((bpf+WIN)/48,(284+NH*NB-1)/(NH*NB)));
                int bkey=NB*(NSPLITB+1)+bns;
                if(!ng && !gcache.count(bkey)){ cudaGraph_t g; CK(cudaStreamBeginCapture(GS,cudaStreamCaptureModeThreadLocal)); { int sv=ns; ns=bns; body(NB); ns=sv; }
                    CK(cudaStreamEndCapture(GS,&g)); cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)); cudaGraphDestroy(g); gcache[bkey]=ge; }
                if(ng){ int sv=ns; ns=bns; for(int i=0;i<bstep;i++,wsteps++){ stream_steps+=NB; body(NB); } ns=sv; }
                else  for(int i=0;i<bstep;i++,wsteps++){ stream_steps+=NB; CK(cudaGraphLaunch(gcache[bkey],GS)); }
                pump(false);                                         // keep the big-ref pipeline moving during the booster too
                CK(cudaStreamSynchronize(GS));
                CK(cudaMemcpyAsync(hdone.data(),d_done,(size_t)WT*4,cudaMemcpyDeviceToHost,GS));
                CK(cudaMemcpyAsync(hsteps.data(),d_steps,(size_t)WT*4,cudaMemcpyDeviceToHost,GS));
                std::vector<int> bct(NB); CK(cudaMemcpyAsync(bct.data(),d_tokb,(size_t)NB*4,cudaMemcpyDeviceToHost,GS)); CK(cudaStreamSynchronize(GS));
                for(int i=0;i<NB;i++) curtok[bi[i]]=bct[i];
                std::vector<int> na2,nt2;
                for(int a=0;a<NA;a++){ int s=active[a]; if(hdone[s]||hsteps[s]>=MAXSTEP) retire(s); else { na2.push_back(s); nt2.push_back(curtok[a]); } }
                active=na2; curtok=nt2; NA=(int)active.size();
            }
        }
        if(NA>0){ CK(cudaMemcpyAsync(d_act,active.data(),(size_t)NA*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_tokb,curtok.data(),(size_t)NA*4,cudaMemcpyHostToDevice,GS));
                  CK(cudaMemcpyAsync(d_steps,hsteps.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS)); CK(cudaMemcpyAsync(d_done,hdone.data(),(size_t)WT*4,cudaMemcpyHostToDevice,GS)); }
    }
    cudaEventRecord(db,GS); CK(cudaEventSynchronize(db)); float dms=0; cudaEventElapsedTime(&dms,da,db);
    if(vb){
        printf("page-parallel decode: %ld tok in %.0f ms (%.0f tok/s), %ld steps, %.0f%% batch util\n",total,dms,total*1000.0/dms,wsteps,100.0*total/stream_steps);
        if(W<Ntotal) printf("windowed: %d pages through %d slots, %ld admissions %.0f ms interleaved (KV resident: %d x %d entries/layer)\n",Ntotal,W,nadm,admms,W,MS);
    }
    for(auto&kv:gcache) cudaGraphExecDestroy(kv.second);                 // free per-call GPU scratch (Gundam calls this per page)
    if(G){ pfctx_free(c2); cudaFree(pe2); cudaFree(vis2); cudaFree(dlog2); }
    cudaFree(d_pf); cudaFree(d_kvoff);
    for(int l=0;l<NL;l++){ cudaFree(kcb[l]); cudaFree(vcb[l]); kcb[l]=nullptr; vcb[l]=nullptr; }
    cudaFree(atb_pm); cudaFree(atb_pl); cudaFree(atb_pacc); atb_pm=atb_pl=atb_pacc=nullptr;
    cudaFree(tc_cnt); cudaFree(tc_bin); cudaFree(tc_part); tc_cnt=tc_bin=nullptr; tc_part=nullptr;
    cudaFree(lmh_cand); cudaFree(lmh_cval); lmh_cand=nullptr; lmh_cval=nullptr;
    cudaFree(d_p1t); cudaFree(d_entt);
    cudaFree(qkvbb); cudaFree(dlogb); cudaFree(d_tokb); qkvbb=nullptr; dlogb=nullptr; d_tokb=nullptr;
    cudaFree(d_steps); cudaFree(outbuf); cudaFree(d_done); cudaFree(d_act);  // (cubws is persistent: cuBLAS holds it)
    cudaFree(pe); cudaFree(dlog); if(vis1)cudaFree(vis1);
    cudaEventDestroy(pa);cudaEventDestroy(pb);cudaEventDestroy(aa);cudaEventDestroy(ab);cudaEventDestroy(da);cudaEventDestroy(db);
}
// CLI/gundam page source: N pages of one document — byte-identical behavior to the pre-server engine.
struct FixedSrc final : PageSrc{
    int N=0,nextpg=0; const bf16* dembeds=nullptr; std::vector<std::vector<int>>* po=nullptr; void(*encpg)(int,bf16*)=nullptr;
    const std::vector<int>* vpps=nullptr;                            // per-page ref sizes (mixed-tiling gundam doc); null = uniform
    std::vector<float> *conf=nullptr,*lowf=nullptr;                  // optional per-page confidence capture (server gundam path)
    int next(bool bok,bool) override { return bok&&nextpg<N?nextpg++:-1; }   // all pages <= the call vpp -> base class only
    int vpp_of(int id) override { return vpps?(*vpps)[id]:-1; }
    const bf16* embeds(int id,bf16* scr,int* pidx) override {
        if(encpg){ encpg(id,scr+H); *pidx=0; return scr; }           // slot-local embeds: callback page lives at index 0
        *pidx=id; return dembeds;                                    // upfront embeds: page id indexes the caller's buffer
    }
    std::vector<int>& out(int id) override { return (*po)[id]; }
    void stats(int id,const PageConf& pc) override { if(conf)(*conf)[id]=pc.conf; if(lowf)(*lowf)[id]=pc.lowf; }
    bool lazy() override { return encpg!=nullptr; }
};
static int window_env(){ static int w=[](){const char* s=getenv("WINDOW");return s?atoi(s):128;}(); return w; }
static void generate_pagepar(int N,bf16* dembeds,std::vector<std::vector<int>>& po,int vpp=273,void(*encpg)(int,bf16*)=nullptr,
                             std::vector<float>* conf=nullptr,std::vector<float>* lowf=nullptr,int wcap_override=0,
                             const std::vector<int>* vpps=nullptr){
    int wcap=wcap_override>0?wcap_override:window_env();  // resident-stream cap (capacity knob; tok/s scales with batch -> keep high, memory flattens past it)
    const int W=encpg?std::min(N,std::max(1,wcap)):N;    // no callback -> caller built all embeds upfront -> all resident
    po.assign(N,{});
    if(conf)conf->assign(N,0.f); if(lowf)lowf->assign(N,0.f);
    FixedSrc src; src.N=N; src.dembeds=dembeds; src.po=&po; src.encpg=encpg; src.conf=conf; src.lowf=lowf; src.vpps=vpps;
    generate_pagepar_core(src,vpp,W,N);
}
// VRAM-safe resident-slot cap for a given reference size (KV = W * MS * H * NL * (K+V) * sizeof(kvt)).
// Bounds gundam memory by the WINDOW, not the page count -> unlimited gundam pages at flat VRAM.
static int vram_wcap(int vpp){
    size_t freeb=0,totalb=0; cudaMemGetInfo(&freeb,&totalb);
    size_t MS=(size_t)(1+vpp+4)+WIN;
    size_t per_slot=MS*H*NL*2*sizeof(kvt);              // K+V for all layers, one slot
    long w=(long)((double)freeb*0.55/per_slot);         // leave 45% for scratch (tc_part/lmh/dlogb ~ per-slot) + vision + headroom
    if(w<1)w=1; if(w>window_env())w=window_env();
    return (int)w;
}
void gundam_vfix();
// Base-mode page encode with next-page prefetch. Marker (g_pf_*) = "this (pdf,page) is already
// rendered+uploaded to the vision input buffer"; keyed by BOTH pdf and page so multi-document
// admission (server) can never reuse another document's pixels. Invalidate whenever anything else
// touches the vision input (gundam runs) or a marked doc goes away.
static std::string g_pf_pdf; static int g_pf_page=-1;
static void enc_invalidate(){ g_pf_pdf.clear(); g_pf_page=-1; }
static bool enc_page(const char* pdf,int p,const char* npdf,int np,bf16* dst){
    if(!(g_pf_page==p && g_pf_pdf==pdf)){ if(!vis_render_cpu(pdf,p)) return false; vis_upload(); }
    vis_gpu_launch();
    bool pre = npdf && vis_render_cpu(npdf,np);                      // CPU render of the NEXT page overlaps the GPU encode
    vis_gpu_sync();
    CK(cudaMemcpy(dst,vis_result(),(size_t)273*H*2,cudaMemcpyDeviceToDevice));
    if(pre){ vis_upload(); g_pf_pdf=npdf; g_pf_page=np; } else enc_invalidate();
    return true;
}
static const char* g_pdf=nullptr; static int g_N=0;                  // CLI page-encode callback state (one doc)
static void enc_base(int p,bf16* dst){                               // encode page p (base 1024, 273 tok); prefetch-renders p+1 on CPU
    if(!enc_page(g_pdf,p,(p+1<g_N)?g_pdf:nullptr,p+1,dst)){ fprintf(stderr,"MuPDF render failed: %s page %d\n",g_pdf,p); exit(1); }
}
// ===== Gundam OCR of one doc: high-res tiling; WINDOWED decode (same admission as Base, vpp=tiles) =====
// Uniform tiling -> per-slot lazy encode through the windowed loop: VRAM bounded by W (vram_wcap), NOT by
// page count -> UNLIMITED gundam pages at flat memory, no page cap. Mixed page sizes -> sequential per-page
// (already one-page-at-a-time = flat memory). po gets one token stream per page. false = a page failed.
extern int gundam_encode(const char* pdf,int page); extern bf16* gundam_result();
extern int gundam_encode_begin(const char* pdf,int page); extern int gundam_encode_ready(); extern void gundam_encode_wait();
extern int gundam_page_ntok(const char* pdf,int page);   // token count from page dims only (no encode)
extern int vis_page_glyphpx10(const char* pdf,int page); // p25 glyph px at base-1024 scale, x10; -1 = no text layer
static const char* g_gpdf=nullptr; static const std::vector<int>* g_gpgs=nullptr; static const std::vector<int>* g_gvpps=nullptr;  // enc_gundam callback state
static void enc_gundam(int id,bf16* dst){                // encode gundam page pgs[id] into a slot's reference (vpp=vpps[id])
    int nt=gundam_encode(g_gpdf,(*g_gpgs)[id]);
    if(nt!=(*g_gvpps)[id]){ fprintf(stderr,"gundam page %d: %d tok != %d (dims drifted mid-stream)\n",(*g_gpgs)[id],nt,(*g_gvpps)[id]); exit(1); }  // dims pre-checked -> unreachable
    CK(cudaMemcpy(dst,gundam_result(),(size_t)nt*H*2,cudaMemcpyDeviceToDevice));
}
static bool ocr_gundam(const char* pdf,const std::vector<int>& pgs,std::vector<std::vector<int>>& po,
                       std::vector<float>* conf=nullptr,std::vector<float>* lowf=nullptr){
    const int N=(int)pgs.size();
    po.clear();
    std::vector<int> nts(N); int mx=0;                             // cheap dims-based tiling per page (no encode)
    for(int i=0;i<N;i++){ nts[i]=gundam_page_ntok(pdf,pgs[i]); if(nts[i]<0)return false; mx=std::max(mx,nts[i]); }
    bool uniform=true; for(int i=1;i<N;i++) if(nts[i]!=nts[0]){uniform=false;break;}
    g_gpdf=pdf; g_gpgs=&pgs; g_gvpps=&nts;                         // WINDOWED: encode+decode stream through vram_wcap slots
    int wc=vram_wcap(mx);                                          // mixed tilings: slots sized to the doc's largest page, per-slot pf handles the rest
    cudaEvent_t va,vb; cudaEventCreate(&va);cudaEventCreate(&vb); cudaEventRecord(va,0);
    generate_pagepar(N,nullptr,po,mx,enc_gundam,conf,lowf,wc,uniform?nullptr:&nts);
    cudaEventRecord(vb,0); CK(cudaEventSynchronize(vb)); float ms=0; cudaEventElapsedTime(&ms,va,vb);
    if(uniform) printf("[gundam] %d pages x %d visual tokens -> WINDOWED decode (W=%d, %.0f ms)\n",N,mx,std::min(N,wc),ms);
    else        printf("[gundam] %d pages, mixed tilings (max %d visual tokens) -> ONE windowed decode (W=%d, %.0f ms)\n",N,mx,std::min(N,wc),ms);
    cudaEventDestroy(va);cudaEventDestroy(vb);
    return true;
}
// ===== server mode: continuous multi-document serving over the SAME decode loop =====
// Connection threads (server.cpp) enqueue jobs; the engine thread expands them into (job,page) items
// and feeds the windowed admission — pages of DIFFERENT documents co-batch in one decode window
// ("documents are just a grouping of pages"). Round-robin one page per job per free slot, so a 1-page
// doc admits at the next 16-step boundary even behind a 500-pager. A job on an idle server reproduces
// the CLI byte-for-byte (same NA trajectory); under concurrent load, co-batching can flip argmax
// near-ties (same numeric class as windowed NA variation — see DESIGN.md "Serving").
// Gundam jobs co-batch too (heterogeneous window): their pages take big slots, base pages take base
// slots — neither class ever waits on the other (no head-of-line blocking, no exclusive interludes).
std::string ocr_decode_tokens(const std::vector<int>& toks){
    std::vector<int> keep; keep.reserve(toks.size());
    for(int t:toks) if(t!=1) keep.push_back(t);
    return bpe_decode(keep);
}
// ===== page quality gates (drive server-side auto hi-res) =====
// CALIBRATED 2026-07-04 on the in-repo corpus vs gundam pseudo-CER (tools/conf_eval.py, outputs_conf/):
// p10 (10th-percentile token prob) separates bad pages far better than mean conf — all 50 pages with
// pseudo-CER>0.15 have p10<=0.839 (the old conf<0.80||lowf>0.20 gate MISSED 35 of them: base reads
// small print confidently-but-wrong, mean stays high); clean pages sit at p10>=0.96. Threshold 0.85
// catches 50/50 with 0 false escalations on clean docs. Degeneration check stays: confident loops
// (conf 0.98) evade ALL logit signals.
static const float P10_T=0.85f;
// Repetition/degeneration: a page decoded into a loop (e.g. an incrementing counter) evades exact
// no-repeat-ngram AND keeps HIGH token confidence. Digit runs -> '#', then flag if 8-word shingles
// are mostly non-unique. Matches web/index.html degeneratePages().
static bool is_degenerate(const std::string& t){
    if(t.size()<3000) return false;
    std::vector<std::string> w; std::string cur; bool indig=false;
    for(unsigned char c:t){
        if(isspace(c)){ if(!cur.empty()){w.push_back(cur);cur.clear();} indig=false; }
        else if(isdigit(c)){ if(!indig){cur+='#';indig=true;} }
        else { cur+=(char)c; indig=false; }
    }
    if(!cur.empty())w.push_back(cur);
    if(w.size()<200) return false;
    std::unordered_set<std::string> g;
    for(size_t i=0;i+8<=w.size();i++){ std::string s; for(int k=0;k<8;k++){ s+=w[i+k]; s+=' '; } g.insert(s); }
    return (double)g.size()/(double)(w.size()-7) < 0.35;
}
// Calibrated bad-page risk (0..1, higher = escalate; X-Page-Risk). Two detectors, per the calibration:
// quantile signal for hesitant/garbled/sparse pages, text signal for confident loops.
static float page_risk(const PageConf& pc,const std::vector<int>& toks){
    float r=1.f-pc.p10;
    if(is_degenerate(ocr_decode_tokens(toks))) r=std::max(r,0.95f);   // confident loop: logit signals stay high
    return r;
}
static bool page_flagged(const std::vector<int>& toks,const PageConf& pc){ return page_risk(pc,toks)>1.f-P10_T; }
// PRE-DECODE escalation (calibrated 2026-07-04, same corpus): char-weighted p25 glyph size at the base
// 1024 render predicts base failure BEFORE any encode/decode — every pseudo-CER>0.15 page with a text
// layer sits at <=11.6px, every clean page at >=13.1px. Threshold 12px: those pages go STRAIGHT to
// gundam (skips the doomed base pass, ~2x faster than base+retry at equal quality). No text layer
// (scans) or borderline -> base first; the post-decode p10 gate stays as the backstop.
static const int GLYPH_T10=120;
struct QueueSrc final : PageSrc{
    struct Item{ std::shared_ptr<OcrJob> job; int page; int pdfpage; std::vector<int> toks; PageConf pc; int vpp=273; };  // page = job-relative slot; pdfpage = actual page in the PDF
    std::unordered_map<int,Item> items; int nid=0;         // live (admittable/in-flight) items only
    std::deque<std::deque<int>> rr;                        // per-job pending page ids, round-robin
    void expand(std::shared_ptr<OcrJob> j){                // job -> items (page count read here, engine thread)
        int total=vis_page_count(j->path.c_str());
        if(total<=0){ j->status=422; j->err=total<0?"cannot open document":"empty document";
                  vis_doc_close(j->path.c_str()); srv_complete(std::move(j)); return; }   // a 0-page doc still got cached: drop it (pins the unlinked spool otherwise)
        int n;
        if(!j->pagelist.empty()){                          // explicit page list (selective retry)
            for(int p:j->pagelist) if(p>=total){ j->status=400; j->err="page "+std::to_string(p+1)+" out of range (document has "+std::to_string(total)+")";
                                                 vis_doc_close(j->path.c_str()); srv_complete(std::move(j)); return; }
            n=(int)j->pagelist.size();
        } else { n=total; if(j->npages>0 && j->npages<n) n=j->npages; }
        j->pages=n; j->pending=n; j->page_toks.assign(n,{});
        j->page_conf.assign(n,0.f); j->page_lowfrac.assign(n,0.f); j->pdfpages.assign(n,0);
        j->page_feats.assign(n,{}); j->page_risk.assign(n,0.f); j->page_mode.assign(n,0);
        std::deque<int> pend;
        for(int p=0;p<n;p++){ int pdfp=j->pagelist.empty()?p:j->pagelist[p]; j->pdfpages[p]=pdfp;
            int pv=273;
            if(j->gundam){ pv=gundam_page_ntok(j->path.c_str(),pdfp);   // per-page tiling from dims only (no encode)
                if(pv<0){ j->status=422; j->err="cannot read page "+std::to_string(pdfp+1);
                          vis_doc_close(j->path.c_str()); srv_complete(std::move(j)); return; } }
            else if(j->auto_hires) pv=-2;                  // PRE-DECODE check pending: decided lazily at first rotation touch
            items[nid]=Item{j,p,pdfp,{},{},pv}; pend.push_back(nid); nid++; }
        rr.push_back(std::move(pend));
    }
    void pump(){ std::shared_ptr<OcrJob> j; while((j=srv_take())) expand(std::move(j)); }
    void decide(Item& it){                                 // PRE-DECODE escalation: small print never survives base -> straight to gundam
        it.vpp=273;
        int g10=vis_page_glyphpx10(it.job->path.c_str(),it.pdfpage);
        if(g10>=0 && g10<GLYPH_T10){ int pv=gundam_page_ntok(it.job->path.c_str(),it.pdfpage); if(pv>273) it.vpp=pv; }
    }
    int next(bool bok,bool gok) override {
        pump();                                            // arrivals join the rotation before each admission
        int nq=(int)rr.size();                             // one full rotation; jobs whose next page has no free slot class go to the back
        for(int i=0;i<nq;i++){
            if(rr.empty()) break;
            std::deque<int> q=std::move(rr.front()); rr.pop_front();
            if(q.empty()) continue;
            Item& hd=items.at(q.front());
            if(hd.vpp==-2) decide(hd);                     // lazy: one ~20ms fz_stext per admission touch, amortized
            bool big=hd.vpp>273;
            if(big?gok:bok){
                int id=q.front(); q.pop_front();
                if(!q.empty()) rr.push_back(std::move(q)); // rotate: next admission goes to the next job
                return id;
            }
            rr.push_back(std::move(q));                    // wrong class for the free slots: skip, keep rotating
        }
        return -1;
    }
    bool peek(const char** pdf,int* page){                 // next BASE admission target (render prefetch; gundam encodes bypass this path)
        for(auto& q:rr) if(!q.empty()){ Item& it=items.at(q.front()); if(it.vpp>273) continue; if(it.job->status!=200)return false;
                                        *pdf=it.job->path.c_str(); *page=it.pdfpage; return true; }
        return false;
    }
    const bf16* embeds(int id,bf16* scr,int* pidx) override {
        Item& it=items.at(id);
        if(it.job->status!=200) return nullptr;            // job already failed: skip its remaining pages fast
        if(it.vpp>273){                                    // gundam page: high-res tiling encode (clobbers the base-prefetch vision input)
            int nt=gundam_encode(it.job->path.c_str(),it.pdfpage);
            enc_invalidate();
            if(nt!=it.vpp){ it.job->status=422; it.job->err="page render failed"; return nullptr; }
            CK(cudaMemcpy(scr+H,gundam_result(),(size_t)it.vpp*H*2,cudaMemcpyDeviceToDevice));
            *pidx=0; return scr;
        }
        const char* npdf=nullptr; int np=0; peek(&npdf,&np);
        if(!enc_page(it.job->path.c_str(),it.pdfpage,npdf,np,scr+H)){ it.job->status=422; it.job->err="page render failed"; return nullptr; }
        *pidx=0; return scr;
    }
    std::vector<int>& out(int id) override { return items.at(id).toks; }
    void stats(int id,const PageConf& pc) override { items.at(id).pc=pc; }
    void done(int id) override {
        auto node=items.extract(id); Item& it=node.mapped(); std::shared_ptr<OcrJob> j=it.job;
        j->page_toks[it.page]=std::move(it.toks);
        j->page_conf[it.page]=it.pc.conf; j->page_lowfrac[it.page]=it.pc.lowf; j->page_feats[it.page]=it.pc;
        if(it.vpp>273) j->page_mode[it.page]=1;            // decoded in gundam (pre-escalated or retry)
        if(--j->pending==0){
            if(j->auto_hires && !j->gundam && !j->retried){   // AUTO HI-RES: re-queue flagged pages as gundam items (co-batched — no interlude, no reenqueue)
                std::deque<int> pend;
                for(int i=0;i<j->pages && (int)pend.size()<512;i++){
                    if(j->page_mode[i]) continue;             // already gundam-decoded (pre-check): a same-mode re-run can't improve it
                    if(!page_flagged(j->page_toks[i],j->page_feats[i])) continue;
                    int pv=gundam_page_ntok(j->path.c_str(),j->pdfpages[i]);
                    if(pv<0) continue;                        // dims unreadable: keep the base result
                    items[nid]=Item{j,i,j->pdfpages[i],{},{},pv}; pend.push_back(nid); nid++;   // page=i -> done() overwrites page_toks[i] = the merge
                }
                if(!pend.empty()){ j->retried=true; j->pending=(int)pend.size(); rr.push_back(std::move(pend)); return; }
            }
            long tk=0; j->truncated=0;                     // recount from final page_toks (retried pages would double-count incrementally)
            for(auto&p:j->page_toks){ if(!p.empty()&&p.back()!=1)j->truncated++; for(int t:p) if(t!=1)tk++; }
            j->tokens=tk;
            for(int i=0;i<j->pages;i++) j->page_risk[i]=page_risk(j->page_feats[i],j->page_toks[i]);   // calibrated escalation signal (X-Page-Risk)
            vis_doc_close(j->path.c_str());                // spool paths get recycled: never serve a stale doc
            if(g_pf_pdf==j->path) enc_invalidate();
            srv_complete(std::move(j));
        }
    }
    bool wait() override { return srv_wait_work(); }       // blocks until a job is queued (all jobs feed the one window)
    bool verbose() override { return false; }
    int vpp_of(int id) override { return items.at(id).vpp; }
    int vpp_gmax() override { return GVPP_MAX; }           // server may see any gundam tiling -> allocate big slots
    int enc_begin(int id) override {                       // ASYNC gundam encode: launch render+SAM/CLIP on the vision stream
        Item& it=items.at(id);
        if(it.job->status!=200) return -1;
        int nt=gundam_encode_begin(it.job->path.c_str(),it.pdfpage);
        enc_invalidate();                                  // the render clobbered the base-prefetch vision input
        if(nt!=it.vpp){ it.job->status=422; it.job->err="page render failed"; return -1; }
        return nt;
    }
    bool enc_ready() override { return gundam_encode_ready(); }
    void enc_wait() override { gundam_encode_wait(); }
    const bf16* enc_result() override { return gundam_result(); }
    bool alive(int id) override { return items.at(id).job->status==200; }
};
static void engine_serve(){                                // never returns (fail-fast: CUDA errors exit(1), supervisor restarts)
    static int wcap=[](){const char* s=getenv("WINDOW");return s?atoi(s):128;}();
    QueueSrc src; generate_pagepar_core(src,273,std::max(1,wcap),-1);   // ONE continuous heterogeneous window: base + gundam pages co-batch
}
int main(int argc,char**argv){
    if(getenv("GUNDAM_VFIX")){ gundam_vfix(); return 0; }
    CB(cublasCreate(&CUB));
    CK(cudaStreamCreate(&GS)); CB(cublasSetStream(CUB,GS));
    ST.load("/home/janitor/unlimited-ocr/engine/manifest.tsv");
    printf("KV cache backend: %s\n", KV_NAME);
    printf("loading weights to GPU...\n"); load_weights(); CK(cudaDeviceSynchronize());

    if(argc>1 && !strcmp(argv[1],"serve")){                   // ===== server: continuous multi-document OCR =====
        int port=argc>2?atoi(argv[2]):8000;
        load_vocab("/home/janitor/unlimited-ocr/engine/vocab.bin");
        extern void init_vision(); init_vision();             // front-load vision weights: readiness = socket bound
        int bound=server_start(port,argc>3?argv[3]:nullptr);   // serve [port] [bind-addr] — bind 127.0.0.1 behind a TLS proxy
        printf("[serve] ready on port %d (POST /ocr[?pages=N][&gundam=1], GET /healthz)\n",bound); fflush(stdout);
        engine_serve();
        return 0;
    }
    // ===== OCR (single host thread): ocr_bin [pdf] [npages] | ocr_bin serve [port] =====
    // Defaults: bundled paper, 1 page. Decode always runs to EOS (per-page, page-parallel). N pages parsed together.
    const char* pdf = argc>1 ? argv[1] : "/home/janitor/unlimited-ocr/Unlimited-OCR.pdf";
    int N      = argc>2 ? atoi(argv[2]) : 1;
    if(getenv("GUNDAM")){
        load_vocab("/home/janitor/unlimited-ocr/engine/vocab.bin");
        std::vector<std::vector<int>> po;
        std::vector<int> pgs; for(int p=0;p<N;p++)pgs.push_back(p);
        if(!ocr_gundam(pdf,pgs,po)){ fprintf(stderr,"gundam encode failed: %s\n",pdf); return 1; }
        std::vector<int> allkeep; for(auto&pgo:po) for(int t:pgo) if(t!=1) allkeep.push_back(t);
        printf("\n===== OCR (GUNDAM, %d page(s), %d tokens) =====\n%s\n",N,(int)allkeep.size(),bpe_decode(allkeep).c_str());
        return 0;
    }
    load_vocab("/home/janitor/unlimited-ocr/engine/vocab.bin");  // before decode: line-region confidence features need the newline-token mask
    g_pdf=pdf; g_N=N; enc_invalidate();                       // vision is encoded lazily per admission (enc_base), overlapped CPU render
    std::vector<std::vector<int>> po; generate_pagepar(N,nullptr,po,273,enc_base);   // the only decode path: page-parallel, windowed admission
    std::vector<int> keep; for(auto&pg:po) for(int t:pg) if(t!=1) keep.push_back(t);
    printf("\n===== OCR (%d page(s), %d tokens) =====\n%s\n", N, (int)keep.size(), bpe_decode(keep).c_str());
    return 0;
}
