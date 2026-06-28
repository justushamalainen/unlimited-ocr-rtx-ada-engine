// Unlimited-OCR decoder — focused CUDA engine (sm_89). Stage 2: full bf16 prefill,
// verified layer-by-layer against HF fixtures.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include "st_loader.h"

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
static int* d_pos=nullptr;    // device-resident decode position

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
              bool dense; };
static Layer L[NL];
static bf16 *EMB,*LMH,*FNORM;
static uint8_t* LMH4=nullptr; static float* lmh_s4=nullptr;      // int4 group-128 lm_head + group scales

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
// lm_head GEMV in fp8: out[v]=scale[v]*dot(x, deq(W8[v])); one warp per vocab row
__global__ void k_lmhead_fp8(const bf16* x,const uint8_t* W8,const float* scale,float* out,int rows){
    int gw=blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5), lane=threadIdx.x&31; if(gw>=rows)return;
    float acc=dot_fp8_vec(W8+(size_t)gw*H,x,H);
    if(lane==0)out[gw]=acc*scale[gw];
}
// generic fp8 GEMV (bf16 out): out[r]=scale[r]*dot(x[cols], deq(W8[r])); warp/row. <<<(rows+7)/8,256>>>
// fp8 GEMV with float bias add (folds k_combine into shared down): out[r]=bias[r]+scale[r]*dot
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
// q4 (group-128) expert down: warp per (slot,d), atomic-accumulate into Yf (needs Yf pre-zeroed).
static void quant_q4(bf16* W,int rows,int cols,uint8_t** W4,float** sc){
    CK(cudaMalloc(W4,(size_t)rows*(cols/2))); CK(cudaMalloc(sc,(size_t)rows*(cols/QG)*4));
    k_quant_rows_q4<<<rows,QG,0,GS>>>(W,*W4,*sc,cols);
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
// device-pos counter for graph-captured decode (incremented once per step; wraps in [lo,hi))
__global__ void k_incpos(int* pos,int lo,int hi){ if(threadIdx.x==0){ int p=*pos+1; if(p>=hi)p=lo; *pos=p; } }
// fused RoPE + KV store: rope q in qkvb (in place), rope k -> kcache[*pos], copy v -> vcache[*pos].
// qkvb layout [3H]: q=[0,H) k=[H,2H) v=[2H,3H). <<<NH, HD/2>>>.
// R-SWA: RoPE at absolute position *pos; K/V written to ring slot pf + ((*pos-pf) mod WIN).
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
static kvt *kcache[NL],*vcache[NL]; static int MAXLEN=0; static bf16* qkvb=nullptr; static bf16* gu=nullptr;
static void alloc(int S){
    size_t sH=(size_t)S*H;
    if(xbuf){ cudaFree(xbuf);cudaFree(nbuf);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(att);cudaFree(tmp);   // free-first: repeated calls (Gundam per-page) reuse, no leak
        cudaFree(mg);cudaFree(mu);cudaFree(mh);cudaFree(glog);cudaFree(didx);cudaFree(dw);cudaFree(Yf);
        for(int l=0;l<NL;l++){cudaFree(kcache[l]);cudaFree(vcache[l]);} cudaFree(qkvb);cudaFree(gu); }
    CK(cudaMalloc(&xbuf,sH*2));CK(cudaMalloc(&nbuf,sH*2));CK(cudaMalloc(&q,sH*2));CK(cudaMalloc(&k,sH*2));CK(cudaMalloc(&v,sH*2));CK(cudaMalloc(&att,sH*2));CK(cudaMalloc(&tmp,sH*2));
    CK(cudaMalloc(&mg,(size_t)S*DENSEI*2));CK(cudaMalloc(&mu,(size_t)S*DENSEI*2));CK(cudaMalloc(&mh,(size_t)S*DENSEI*2));
    CK(cudaMalloc(&glog,(size_t)S*NEXP*4));CK(cudaMalloc(&didx,(size_t)S*TOPK*4));CK(cudaMalloc(&dw,(size_t)S*TOPK*4));CK(cudaMalloc(&Yf,sH*4));
    MAXLEN=S; for(int l=0;l<NL;l++){ CK(cudaMalloc(&kcache[l],(size_t)S*H*sizeof(kvt))); CK(cudaMalloc(&vcache[l],(size_t)S*H*sizeof(kvt))); }
    CK(cudaMalloc(&qkvb,3*(size_t)H*2)); CK(cudaMalloc(&gu,2*(size_t)DENSEI*2));
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
    static int hcnt[NEXP],hoff[NEXP+1]; CK(cudaMemcpy(hcnt,gm_cnt,NEXP*4,cudaMemcpyDeviceToHost));
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
extern bf16* vision_encode(const char* pdf,int page);  // -> device [273,1280] bf16 (vision_enc.cu)
extern void vis_render_cpu(const char* pdf,int page);  // split render/GPU for interleaving
extern void vis_upload(); extern void vis_gpu_launch(); extern void vis_gpu_sync(); extern bf16* vis_result();
// write only the non-visual slots ([bos] + [Multi,page,parsing,.]) directly into the embed buffer;
// the N*273 visual slots are filled by vision writing straight into the buffer (no staging copies).
__global__ void k_embed_slots(bf16* out,const bf16* emb,int nvis,int bos,int t0,int t1,int t2,int t3){
    int slot=blockIdx.x, c=blockIdx.y*256+threadIdx.x; if(c>=H)return;
    int pos,id; if(slot==0){pos=0;id=bos;} else {int j=slot-1;pos=1+nvis+j;id=(j==0)?t0:(j==1)?t1:(j==2)?t2:t3;}
    out[(size_t)pos*H+c]=emb[(size_t)id*H+c];
}
// build the 277-token prompt embeds: [bos=0][273 visual][document, parsing, .]
__global__ void k_buildembeds(bf16* out,const bf16* emb,const bf16* vis,int bos,int t0,int t1,int t2){
    int tok=blockIdx.x,c=blockIdx.y*256+threadIdx.x; if(c>=H)return; size_t o=(size_t)tok*H+c;
    if(tok==0) out[o]=emb[(size_t)bos*H+c];
    else if(tok<=273) out[o]=vis[(size_t)(tok-1)*H+c];
    else { int ti=tok-274,id=(ti==0)?t0:(ti==1)?t1:t2; out[o]=emb[(size_t)id*H+c]; }
}
// byte-level BPE decoder (build asset vocab.bin: id -> byte-level utf8 string)
static std::vector<std::string> g_vocab; static std::map<uint32_t,unsigned char> g_bdec;
static void load_vocab(const char* path){
    FILE* f=fopen(path,"rb"); int n; if(fread(&n,4,1,f)!=1){} g_vocab.resize(n);
    for(int i=0;i<n;i++){ int len; if(fread(&len,4,1,f)!=1){} std::string s(len,0); if(len&&fread(&s[0],1,len,f)!=(size_t)len){} g_vocab[i]=s; } fclose(f);
    std::vector<int> bs; for(int b='!';b<='~';b++)bs.push_back(b); for(int b=0xA1;b<=0xAC;b++)bs.push_back(b); for(int b=0xAE;b<=0xFF;b++)bs.push_back(b);
    std::vector<int> cs=bs; int m=0; for(int b=0;b<256;b++){bool in=false;for(int x:bs)if(x==b){in=true;break;} if(!in){bs.push_back(b);cs.push_back(256+m);m++;}}
    for(size_t i=0;i<bs.size();i++) g_bdec[(uint32_t)cs[i]]=(unsigned char)bs[i];
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
__global__ void k_attn_split_b(const bf16* qkvb,const kvt* kc,const kvt* vc,float* pm,float* pl,float* pacc,int pf,const int* dstep,int nsplit,int maxseq,const int* act){
    int h=blockIdx.x, s=blockIdx.y, sp=blockIdx.z, lane=threadIdx.x;
    int dc=*dstep+1; int clen=pf+(dc<WIN?dc:WIN);
    int chunk=(clen+nsplit-1)/nsplit, j0=sp*chunk, j1=min(clen,j0+chunk);
    int base=(s*NH+h)*nsplit+sp, d0=lane,d1=lane+32,d2=lane+64,d3=lane+96;
    if(j0>=j1){ if(lane==0){pm[base]=-1e30f;pl[base]=0;}      // empty split (ns>keys): write NEUTRAL so merge ignores it
        pacc[(size_t)base*HD+d0]=0;pacc[(size_t)base*HD+d1]=0;pacc[(size_t)base*HD+d2]=0;pacc[(size_t)base*HD+d3]=0; return; }
    const bf16* q=qkvb+(size_t)s*3*H+(size_t)h*HD;          // rotated q for (stream s, head h)
    int ps=act[s]; const kvt* kcs=kc+(size_t)ps*maxseq*H; const kvt* vcs=vc+(size_t)ps*maxseq*H;
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
__global__ void k_rope_store_b(bf16* qkvb,kvt* kc,kvt* vc,int pf,const int* dstep,int maxseq,const int* act){
    int h=blockIdx.x, s=blockIdx.y, d=threadIdx.x; int step=*dstep;
    int p=pf+step; float inv=powf(ROPE_THETA,-2.f*d/HD); float ang=p*inv,c=cosf(ang),sn=sinf(ang);
    int rslot=pf+((p-pf)%WIN);
    bf16* q=qkvb+(size_t)s*3*H; bf16* k=q+H; bf16* v=q+2*H;
    int ps=act[s]; kvt* kcs=kc+(size_t)ps*maxseq*H; kvt* vcs=vc+(size_t)ps*maxseq*H;
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
    int a=blockIdx.x, pos=*dstep; if(pos<ngram-1)return;
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
// ===== batched BLOCK-DIAGONAL prefill: all N pages [bos+visual+prompt] in one pass, each page attends only itself
#define PFL 278   // per-page sequence length (1 bos + 273 visual + 4 prompt)
__global__ void k_pageembeds_all(bf16* out,const bf16* emb,const bf16* dembeds,int N,int bos,int t0,int t1,int t2,int t3){
    int seq=blockIdx.x, slot=blockIdx.y, c=blockIdx.z*256+threadIdx.x; if(c>=H)return;
    size_t o=((size_t)seq*PFL+slot)*H+c;
    if(slot==0){ out[o]=emb[(size_t)bos*H+c]; return; }
    if(slot<=273){ out[o]=dembeds[(size_t)(1+seq*273+(slot-1))*H+c]; return; }
    int ti=slot-274, id=(ti==0)?t0:(ti==1)?t1:(ti==2)?t2:t3; out[o]=emb[(size_t)id*H+c];
}
__global__ void k_rope_bd(bf16* q,bf16* k,int L){   // per-sequence RoPE position = token_idx % L
    int t=blockIdx.x, h=blockIdx.y, d=threadIdx.x; int pos=t%L;
    float inv=powf(ROPE_THETA,-2.f*d/HD); float ang=pos*inv,c=cosf(ang),s=sinf(ang); size_t base=((size_t)t*NH+h)*HD;
    auto rot=[&](bf16* x){ float a=__bfloat162float(x[base+d]),b=__bfloat162float(x[base+d+HD/2]);
        x[base+d]=__float2bfloat16(a*c-b*s); x[base+d+HD/2]=__float2bfloat16(b*c+a*s); };
    rot(q); rot(k);
}
__global__ void k_attn_prefill_bd(const bf16* q,const bf16* k,const bf16* v,bf16* out,int L){   // causal WITHIN each sequence
    int h=blockIdx.x, i=blockIdx.y, tid=threadIdx.x, w=tid>>5, lane=tid&31; int s0=(i/L)*L;
    float scale=rsqrtf((float)HD); __shared__ float qs[HD],sm[8],sl[8],sacc[8][HD];
    size_t qb=((size_t)i*NH+h)*HD; for(int x=tid;x<HD;x+=blockDim.x) qs[x]=__bfloat162float(q[qb+x]); __syncthreads();
    int d0=lane,d1=lane+32,d2=lane+64,d3=lane+96; float q0=qs[d0],q1=qs[d1],q2=qs[d2],q3=qs[d3];
    float m=-1e30f,l=0,a0=0,a1=0,a2=0,a3=0;
    for(int j=s0+w;j<=i;j+=8){ const bf16* kj=k+((size_t)j*NH+h)*HD;
        float p=q0*__bfloat162float(kj[d0])+q1*__bfloat162float(kj[d1])+q2*__bfloat162float(kj[d2])+q3*__bfloat162float(kj[d3]);
        for(int o=16;o;o>>=1)p+=__shfl_xor_sync(~0u,p,o);
        float sc=p*scale,mn=fmaxf(m,sc),cr=__expf(m-mn),pe=__expf(sc-mn); const bf16* vj=v+((size_t)j*NH+h)*HD;
        a0=a0*cr+pe*__bfloat162float(vj[d0]); a1=a1*cr+pe*__bfloat162float(vj[d1]);
        a2=a2*cr+pe*__bfloat162float(vj[d2]); a3=a3*cr+pe*__bfloat162float(vj[d3]); l=l*cr+pe; m=mn; }
    if(lane==0){sm[w]=m;sl[w]=l;} sacc[w][d0]=a0;sacc[w][d1]=a1;sacc[w][d2]=a2;sacc[w][d3]=a3; __syncthreads();
    if(tid<HD){ float mg=-1e30f; for(int ww=0;ww<8;ww++)mg=fmaxf(mg,sm[ww]);
        float lg=0,o=0; for(int ww=0;ww<8;ww++){float f=__expf(sm[ww]-mg); lg+=sl[ww]*f; o+=sacc[ww][tid]*f;}
        out[(size_t)i*NH*HD+h*HD+tid]=__float2bfloat16(o/lg); }
}
__global__ void k_seedkv_bd(kvt* dst,const bf16* src,int N,int L,int MS){   // (seq s, pos i) -> stream s ref region, slot i
    size_t g=(size_t)blockIdx.x*256+threadIdx.x; if(g>=(size_t)N*L*H)return;
    int c=g%H; size_t ti=g/H; int i=ti%L, s=ti/L;
    dst[(size_t)(s*MS+i)*H+c]=kvst(__bfloat162float(src[ti*H+c]));
}
__global__ void k_gather_last(bf16* out,const bf16* nbuf,int L){   // out[s] = nbuf[s*L + L-1] (each seq's last hidden)
    int s=blockIdx.x, c=blockIdx.y*256+threadIdx.x; if(c>=H)return;
    out[(size_t)s*H+c]=nbuf[(size_t)(s*PFL+PFL-1)*H+c];
}
// one block-diagonal prefill of all N pages -> per-page reference KV in kcb/vcb + first decode token tok0[]
static void prefill_bd(int N,bf16* emball,kvt** kcb,kvt** vcb,int MS,std::vector<int>& tok0,float* dlog){
    int S=N*PFL;
    CK(cudaMemcpyAsync(xbuf,emball,(size_t)S*H*2,cudaMemcpyDeviceToDevice,GS));
    for(int l=0;l<NL;l++){ Layer&ly=L[l];
        k_rmsnorm<<<S,256,0,GS>>>(xbuf,ly.in_norm,nbuf,S,H,EPS);
        lin(nbuf,ly.q,q,S,H,H); lin(nbuf,ly.k,k,S,H,H); lin(nbuf,ly.v,v,S,H,H);
        k_rope_bd<<<dim3(S,NH),HD/2,0,GS>>>(q,k,PFL);
        k_seedkv_bd<<<((size_t)S*H+255)/256,256,0,GS>>>(kcb[l],k,N,PFL,MS);
        k_seedkv_bd<<<((size_t)S*H+255)/256,256,0,GS>>>(vcb[l],v,N,PFL,MS);
        k_attn_prefill_bd<<<dim3(NH,S),256,0,GS>>>(q,k,v,att,PFL);
        lin(att,ly.o,tmp,S,H,H);
        k_add<<<(S*H+255)/256,256,0,GS>>>(xbuf,tmp,S*H);
        k_rmsnorm<<<S,256,0,GS>>>(xbuf,ly.post_norm,nbuf,S,H,EPS);
        mlp_block(ly,S);
        k_add<<<(S*H+255)/256,256,0,GS>>>(xbuf,tmp,S*H);
    }
    k_rmsnorm<<<S,256,0,GS>>>(xbuf,FNORM,nbuf,S,H,EPS);
    k_gather_last<<<dim3(N,(H+255)/256),256,0,GS>>>(att,nbuf,PFL);   // att[0..N) = each page's last hidden
    lin_f32(att,LMH,dlog,N,H,V);                                     // logits[N,V]
    std::vector<float> ll((size_t)N*V); CK(cudaMemcpy(ll.data(),dlog,(size_t)N*V*4,cudaMemcpyDeviceToHost));
    tok0.assign(N,0); for(int p=0;p<N;p++){ const float* r=ll.data()+(size_t)p*V; int t=0; for(int e=1;e<V;e++) if(r[e]>r[t])t=e; tok0[p]=t; }
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
#define PPSMALL 3   // NA<=PPSMALL: fp8/int4 per-token kernels (fast at small batch / single page); above: cuBLAS bf16
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
    k_moe_gateup_q4_b<<<(B*TOPK*MOEI+7)/8,256,0,GS>>>(nbuf,didx,ly.eg8,ly.eg_s,ly.eu8,ly.eu_s,mh,B,MOEI);
    k_moe_down_q4_b<<<(B*H+7)/8,256,0,GS>>>(mh,didx,dw,ly.ed8,ly.ed_s,Yf,B,MOEI);   // deterministic: writes all Yf, no pre-zero
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
// DEQUANT-ONCE batched fp8 projections: warp per (row, token-chunk). Unpack each fp8 weight element ONCE,
// reuse across the chunk's tokens (acc[] in regs). Fixes "fp8 loses when batched" — per-token kernels
// re-unpack the weight B times; here unpack cost is amortized across the chunk, so fp8's byte saving holds.
// grid = rows * ceil(B/TBG) warps.  PPNCH(B,TB) = ceil(B/TB) (defined above mlp_block_b).
__device__ __forceinline__ float deq8(uint8_t c){ return __half2float(__nv_cvt_fp8_to_halfraw(c,__NV_E4M3)); }
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
__global__ void k_record_b(int* outbuf,const int* tok,const int* dstep,int maxstep,const int* act){ int s=blockIdx.x; outbuf[(size_t)act[s]*maxstep+*dstep]=tok[s]; }
__global__ void k_setdone_b(int* done,const int* tok,const int* act){ int s=blockIdx.x; if(threadIdx.x==0 && tok[s]==1) done[act[s]]=1; }
static void generate_pagepar(int N,bf16* dembeds,std::vector<std::vector<int>>& po,int vpp=273){
    const int PF=1+vpp+4, MS=PF+WIN;                     // ref = 1 bos + vpp visual + 4 prompt (273=Base, larger=Gundam)
    bool bd=getenv("BDPREFILL") && vpp==273;                        // bd prefill assumes PFL=278; N-seq for Gundam
    alloc(bd?N*PFL:std::max(PF,N));
    for(int l=0;l<NL;l++){ CK(cudaMalloc(&kcb[l],(size_t)N*MS*H*sizeof(kvt))); CK(cudaMalloc(&vcb[l],(size_t)N*MS*H*sizeof(kvt))); }
    CK(cudaMalloc(&atb_pm,(size_t)N*NH*NSPLITB*4)); CK(cudaMalloc(&atb_pl,(size_t)N*NH*NSPLITB*4)); CK(cudaMalloc(&atb_pacc,(size_t)N*NH*NSPLITB*HD*4));
    CK(cudaMalloc(&qkvbb,(size_t)N*3*H*2)); CK(cudaMalloc(&dlogb,(size_t)N*V*4)); CK(cudaMalloc(&d_tokb,(size_t)N*4));
    std::vector<int> tok0(N);
    cudaEvent_t pa,pb; cudaEventCreate(&pa);cudaEventCreate(&pb); CK(cudaStreamSynchronize(GS)); cudaEventRecord(pa,GS);
    if(!bd){                                             // DEFAULT: N sequential per-page prefills
        bf16* pe; CK(cudaMalloc(&pe,(size_t)PF*H*2)); float* dlog; CK(cudaMalloc(&dlog,(size_t)V*4));
        for(int p=0;p<N;p++){
            k_pageembeds<<<dim3(PF,(H+255)/256),256,0,GS>>>(pe,EMB,dembeds,p,0,37460,4366,76466,16,vpp);
            prefill(nullptr,PF,dlog,false,nullptr,pe);
            for(int l=0;l<NL;l++){ CK(cudaMemcpyAsync(kcb[l]+(size_t)p*MS*H,kcache[l],(size_t)PF*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS));
                                   CK(cudaMemcpyAsync(vcb[l]+(size_t)p*MS*H,vcache[l],(size_t)PF*H*sizeof(kvt),cudaMemcpyDeviceToDevice,GS)); }
            std::vector<float> ll(V); CK(cudaMemcpy(ll.data(),dlog,(size_t)V*4,cudaMemcpyDeviceToHost));
            int t=0; for(int e=1;e<V;e++) if(ll[e]>ll[t])t=e; tok0[p]=t;
        }
        cudaFree(pe); cudaFree(dlog);
    } else {                                             // ONE block-diagonal prefill of all pages
        bf16* emball; CK(cudaMalloc(&emball,(size_t)N*PFL*H*2));
        k_pageembeds_all<<<dim3(N,PFL,(H+255)/256),256,0,GS>>>(emball,EMB,dembeds,N,0,37460,4366,76466,16);
        float* dlogN; CK(cudaMalloc(&dlogN,(size_t)N*V*4));
        prefill_bd(N,emball,kcb,vcb,MS,tok0,dlogN);
        cudaFree(emball); cudaFree(dlogN);
    }
    cudaEventRecord(pb,GS); CK(cudaEventSynchronize(pb)); float pms=0; cudaEventElapsedTime(&pms,pa,pb);
    printf("page-parallel prefill (%s): %d pages in %.0f ms\n",bd?"block-diag":"N-seq",N,pms);
    int ns=(PF+WIN)/48; if(ns<1)ns=1; if(ns>NSPLITB)ns=NSPLITB;   // base; recomputed per-NA in loop to fill SMs at small NA
    const int MAXSTEP=4096;
    int *d_step,*outbuf,*d_done,*d_act;                              // batch-shrink: d_act maps active slot -> physical stream
    CK(cudaMalloc(&d_step,4)); CK(cudaMalloc(&outbuf,(size_t)N*MAXSTEP*4)); CK(cudaMalloc(&d_done,(size_t)N*4)); CK(cudaMalloc(&d_act,(size_t)N*4));
    CK(cudaMemset(outbuf,0,(size_t)N*MAXSTEP*4));                    // init: the bulk D2H read below copies all MAXSTEP rows, but only `step` are written (rest read only up to each stream's EOS) — keeps initcheck clean
    int z=0; CK(cudaMemcpy(d_step,&z,4,cudaMemcpyHostToDevice)); CK(cudaMemset(d_done,0,(size_t)N*4));
    std::vector<int> active, curtok;                                 // initial active set = streams whose first token isn't EOS
    for(int p=0;p<N;p++) if(tok0[p]!=1){ active.push_back(p); curtok.push_back(tok0[p]); }
    int NA=(int)active.size();
    CK(cudaMemcpy(d_act,active.data(),(size_t)NA*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tokb,curtok.data(),(size_t)NA*4,cudaMemcpyHostToDevice));
    const int ngram=16;                                            // no_repeat_ngram (always on; kills degeneration loops, safe for clean docs)
    static void* cubws=nullptr; if(!cubws)CK(cudaMalloc(&cubws,(size_t)32<<20)); CB(cublasSetWorkspace(CUB,cubws,(size_t)32<<20)); // persistent workspace (cuBLAS holds the ptr -> must NOT free)
    auto body=[&](int na){                                          // one decode step captured as a graph, replayed per window
        k_embed<<<dim3(na,(H+255)/256),256,0,GS>>>(EMB,d_tokb,xbuf,na);
        k_rmsnorm<<<na,256,0,GS>>>(xbuf,L[0].in_norm,nbuf,na,H,EPS);
        bool sm=(na<=PPSMALL);
        for(int l=0;l<NL;l++){ Layer&ly=L[l];
            if(sm) k_gemv_fp8_b<<<GEMVBT(3*H,na),256,0,GS>>>(nbuf,ly.qkv8,ly.qkv_s,qkvbb,na,3*H,H); else lin(nbuf,ly.qkv,qkvbb,na,H,3*H);
            k_rope_store_b<<<dim3(NH,na),HD/2,0,GS>>>(qkvbb,kcb[l],vcb[l],PF,d_step,MS,d_act);
            k_attn_split_b<<<dim3(NH,na,ns),32,0,GS>>>(qkvbb,kcb[l],vcb[l],atb_pm,atb_pl,atb_pacc,PF,d_step,ns,MS,d_act);
            k_attn_merge_b<<<dim3(NH,na),HD,0,GS>>>(atb_pm,atb_pl,atb_pacc,att,ns);
            if(sm) k_gemv_fp8_b<<<GEMVBT(H,na),256,0,GS>>>(att,ly.o8,ly.o_s,tmp,na,H,H); else lin(att,ly.o,tmp,na,H,H);
            k_add_rmsnorm<<<na,256,0,GS>>>(xbuf,tmp,ly.post_norm,nbuf,H,EPS);   // fused residual+norm
            mlp_block_b(ly,na);
            const bf16* nn=(l+1<NL)?L[l+1].in_norm:FNORM;
            k_add_rmsnorm<<<na,256,0,GS>>>(xbuf,tmp,nn,nbuf,H,EPS);             // fused residual+norm
        }
        if(na<=4) k_lmhead_q4_b<<<(V+7)/8,256,0,GS>>>(nbuf,LMH4,lmh_s4,dlogb,na);
        else      lin_f32(nbuf,LMH,dlogb,na,H,V);
        k_ngram_mask<<<na,256,0,GS>>>(dlogb,outbuf,d_step,MAXSTEP,d_act,ngram);          // no_repeat_ngram (always on)
        k_argmax_b<<<na,256,0,GS>>>(dlogb,d_tokb,V);
        k_record_b<<<na,1,0,GS>>>(outbuf,d_tokb,d_step,MAXSTEP,d_act);
        k_setdone_b<<<na,1,0,GS>>>(d_done,d_tokb,d_act);
        k_incpos<<<1,1,0,GS>>>(d_step,0,1<<30);
    };
    std::map<int,cudaGraphExec_t> gcache;                           // one captured step-graph per active-count (NA=14 bulk reuses one)
    cudaEvent_t da,db; cudaEventCreate(&da);cudaEventCreate(&db); CK(cudaStreamSynchronize(GS)); cudaEventRecord(da,GS);
    int step=0; const int CK_EVERY=16; long stream_steps=0; bool ng=getenv("DECNOGRAPH");
    while(NA>0 && step<MAXSTEP){
        ns=std::min(NSPLITB,std::max((PF+WIN)/48,(284+NH*NA-1)/(NH*NA)));   // more key-splits at small NA to fill the 142 SMs
        if(!ng && !gcache.count(NA)){ cudaGraph_t g; CK(cudaStreamBeginCapture(GS,cudaStreamCaptureModeThreadLocal)); body(NA);
            CK(cudaStreamEndCapture(GS,&g)); cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)); cudaGraphDestroy(g); gcache[NA]=ge; }
        int nstep=std::min(CK_EVERY,MAXSTEP-step);
        for(int i=0;i<nstep;i++,step++){ stream_steps+=NA; if(ng)body(NA); else CK(cudaGraphLaunch(gcache[NA],GS)); }
        CK(cudaStreamSynchronize(GS));                               // recompact: drop streams that hit EOS
        std::vector<int> dn(N); CK(cudaMemcpy(dn.data(),d_done,(size_t)N*4,cudaMemcpyDeviceToHost));
        std::vector<int> ct(NA); CK(cudaMemcpy(ct.data(),d_tokb,(size_t)NA*4,cudaMemcpyDeviceToHost));
        std::vector<int> na, nt;
        for(int a=0;a<NA;a++){ int s=active[a]; if(!dn[s]){ na.push_back(s); nt.push_back(ct[a]); } }
        active=na; NA=(int)active.size();
        if(NA>0){ CK(cudaMemcpy(d_act,active.data(),(size_t)NA*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_tokb,nt.data(),(size_t)NA*4,cudaMemcpyHostToDevice)); }
    }
    cudaEventRecord(db,GS); CK(cudaEventSynchronize(db)); float dms=0; cudaEventElapsedTime(&dms,da,db);
    std::vector<int> ob((size_t)N*MAXSTEP); CK(cudaMemcpy(ob.data(),outbuf,(size_t)N*MAXSTEP*4,cudaMemcpyDeviceToHost));
    po.assign(N,{}); long total=0;
    for(int p=0;p<N;p++){ po[p].push_back(tok0[p]); if(tok0[p]!=1) for(int j=0;j<step;j++){ int t=ob[(size_t)p*MAXSTEP+j]; po[p].push_back(t); if(t==1)break; } total+=po[p].size(); }
    printf("page-parallel decode: %ld tok in %.0f ms (%.0f tok/s), %d steps, %.0f%% batch util\n",total,dms,total*1000.0/dms,step,100.0*total/stream_steps);
    for(auto&kv:gcache) cudaGraphExecDestroy(kv.second);                 // free per-call GPU scratch (Gundam calls this per page)
    for(int l=0;l<NL;l++){ cudaFree(kcb[l]); cudaFree(vcb[l]); kcb[l]=nullptr; vcb[l]=nullptr; }
    cudaFree(atb_pm); cudaFree(atb_pl); cudaFree(atb_pacc); atb_pm=atb_pl=atb_pacc=nullptr;
    cudaFree(qkvbb); cudaFree(dlogb); cudaFree(d_tokb); qkvbb=nullptr; dlogb=nullptr; d_tokb=nullptr;
    cudaFree(d_step); cudaFree(outbuf); cudaFree(d_done); cudaFree(d_act);   // (cubws is persistent: cuBLAS holds it)
}
void gundam_vfix();
int main(int argc,char**argv){
    if(getenv("GUNDAM_VFIX")){ gundam_vfix(); return 0; }
    CB(cublasCreate(&CUB));
    CK(cudaStreamCreate(&GS)); CB(cublasSetStream(CUB,GS)); CK(cudaMalloc(&d_pos,4));
    ST.load("/home/janitor/unlimited-ocr/engine/manifest.tsv");
    printf("KV cache backend: %s\n", KV_NAME);
    printf("loading weights to GPU...\n"); load_weights(); CK(cudaDeviceSynchronize());

    // ===== OCR (single host thread): ocr_bin [pdf] [npages] =====
    // Defaults: bundled paper, 1 page. Decode always runs to EOS (per-page, page-parallel). N pages parsed together.
    const char* pdf = argc>1 ? argv[1] : "/home/janitor/unlimited-ocr/Unlimited-OCR.pdf";
    int N      = argc>2 ? atoi(argv[2]) : 1;
    if(getenv("GUNDAM")){                                    // ===== Gundam: high-res tiling; BATCHED decode (identical path to Base, vpp=tiles) =====
        extern int gundam_encode(const char* pdf,int page); extern bf16* gundam_result();
        load_vocab("/home/janitor/unlimited-ocr/engine/vocab.bin");
        std::vector<int> allkeep;
        cudaEvent_t va,vb; cudaEventCreate(&va);cudaEventCreate(&vb); cudaEventRecord(va,0);
        int nt0=gundam_encode(pdf,0);                        // assume uniform page size (same tiling -> same vpp)
        bf16* de; CK(cudaMalloc(&de,(size_t)(1+(size_t)N*nt0)*H*2));
        CK(cudaMemcpy(de+(size_t)1*H,gundam_result(),(size_t)nt0*H*2,cudaMemcpyDeviceToDevice));
        bool uniform=true; int pg=1;
        for(;pg<N;pg++){ int nt=gundam_encode(pdf,pg);
            if(nt!=nt0){ uniform=false; printf("[gundam] page %d: %d tok != %d -> mixed sizes, sequential fallback\n",pg,nt,nt0); break; }
            CK(cudaMemcpy(de+(size_t)(1+(size_t)pg*nt0)*H,gundam_result(),(size_t)nt0*H*2,cudaMemcpyDeviceToDevice)); }
        cudaEventRecord(vb,0); CK(cudaEventSynchronize(vb)); float vms=0; cudaEventElapsedTime(&vms,va,vb);
        if(uniform){
            printf("[gundam] %d pages x %d visual tokens (vision %.0f ms) -> BATCHED decode\n",N,nt0,vms);
            std::vector<std::vector<int>> po; generate_pagepar(N,de,po,nt0);          // SAME batched decode as Base, just bigger references
            for(auto&pgo:po) for(int t:pgo) if(t!=1) allkeep.push_back(t);
        } else {                                             // mixed page sizes: per-size batching is future work; sequential for now
            for(int p=0;p<N;p++){ int nt=gundam_encode(pdf,p); bf16* gv=gundam_result();
                bf16* d1; CK(cudaMalloc(&d1,(size_t)(1+nt)*H*2)); CK(cudaMemcpy(d1+(size_t)1*H,gv,(size_t)nt*H*2,cudaMemcpyDeviceToDevice));
                std::vector<std::vector<int>> po; generate_pagepar(1,d1,po,nt);
                for(int t:po[0]) if(t!=1) allkeep.push_back(t); cudaFree(d1); }
        }
        cudaFree(de);
        printf("\n===== OCR (GUNDAM, %d page(s), %d tokens) =====\n%s\n",N,(int)allkeep.size(),bpe_decode(allkeep).c_str());
        return 0;
    }
    int nvis=N*273, Se=1+nvis+4;                              // [bos][N*273 visual][Multi page parsing.]
    bf16* dembeds; CK(cudaMalloc(&dembeds,(size_t)Se*H*2));
    k_embed_slots<<<dim3(5,(H+255)/256),256>>>(dembeds,EMB,nvis,0,37460,4366,76466,16);
    cudaEvent_t va,vb; cudaEventCreate(&va); cudaEventCreate(&vb); cudaEventRecord(va,0);
    vis_render_cpu(pdf,0); vis_upload();                      // render page 0
    for(int i=0;i<N;i++){
        vis_gpu_launch();                                    // GPU encodes page i (async, CUDA graph)
        if(i+1<N) vis_render_cpu(pdf,i+1);                   // CPU renders page i+1 -- overlaps GPU(i)
        vis_gpu_sync();
        CK(cudaMemcpy(dembeds+(size_t)(1+i*273)*H,vis_result(),(size_t)273*H*2,cudaMemcpyDeviceToDevice));
        if(i+1<N) vis_upload();
    }
    CK(cudaDeviceSynchronize());
    cudaEventRecord(vb,0); CK(cudaEventSynchronize(vb)); float vms=0; cudaEventElapsedTime(&vms,va,vb);
    printf("vision: %.0f ms (%.0f ms/page)\n", vms, vms/N);
    std::vector<std::vector<int>> po; generate_pagepar(N,dembeds,po);   // the only decode path: page-parallel (works for N>=1)
    load_vocab("/home/janitor/unlimited-ocr/engine/vocab.bin");
    std::vector<int> keep; for(auto&pg:po) for(int t:pg) if(t!=1) keep.push_back(t);
    printf("\n===== OCR (%d page(s), %d tokens) =====\n%s\n", N, (int)keep.size(), bpe_decode(keep).c_str());
    return 0;
}
