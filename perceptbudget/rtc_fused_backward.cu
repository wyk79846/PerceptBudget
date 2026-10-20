// Input-only BF16 VJP kernels. Frozen style, no parameter gradients.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void ada_dx(const __nv_bfloat16* x, const __nv_bfloat16* style,
                       const __nv_bfloat16* dy, __nv_bfloat16* dx, int dim) {
    __shared__ float sums[256], dots[256];
    int row=blockIdx.x, tid=threadIdx.x;
    float ss=0.f, dot=0.f;
    for(int j=tid;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]);
        float z=__bfloat162float(dy[row*dim+j])*(1.f+__bfloat162float(style[row*3*dim+j]));
        ss+=v*v; dot+=z*v;
    }
    sums[tid]=ss; dots[tid]=dot; __syncthreads();
    for(int s=128;s;s>>=1) {
        if(tid<s) { sums[tid]+=sums[tid+s]; dots[tid]+=dots[tid+s]; }
        __syncthreads();
    }
    float r=rsqrtf(sums[0]/dim+1e-6f), mean=dots[0]/dim;
    for(int j=tid;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]);
        float z=__bfloat162float(dy[row*dim+j])*(1.f+__bfloat162float(style[row*3*dim+j]));
        dx[row*dim+j]=__float2bfloat16_rn(z*r-v*(r*r*r)*mean);
    }
}

__global__ void geglu_dx(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                        const __nv_bfloat16* dy, __nv_bfloat16* dg,
                        __nv_bfloat16* du, int n) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    float g=__bfloat162float(gate[j]), u=__bfloat162float(up[j]), d=__bfloat162float(dy[j]);
    float p=1.f/(1.f+expf(-1.5957691216057308f*g*(1.f+0.044715f*g*g)));
    float derivative=p+g*p*(1.f-p)*1.5957691216057308f*(1.f+3.f*0.044715f*g*g);
    dg[j]=__float2bfloat16_rn(d*u*derivative);
    du[j]=__float2bfloat16_rn(d*g*p);
}

// tanh-form GELU VJP: matches transformers' gelu_pytorch_tanh used by the
// pi05 suffix (the sigmoid variant above belongs to the old experimental path
// and is kept only for its historical gate records).
__global__ void geglu_tanh_dx(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                              const __nv_bfloat16* dy, __nv_bfloat16* dg,
                              __nv_bfloat16* du, int n) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    float g=__bfloat162float(gate[j]), u=__bfloat162float(up[j]), d=__bfloat162float(dy[j]);
    const float c=0.7978845608028654f;
    float z=c*g*(1.f+0.044715f*g*g);
    float t=tanhf(z);
    float act=0.5f*g*(1.f+t);
    float dact=0.5f*(1.f+t)+0.5f*g*(1.f-t*t)*c*(1.f+3.f*0.044715f*g*g);
    dg[j]=__float2bfloat16_rn(d*u*dact);
    du[j]=__float2bfloat16_rn(d*act);
}

extern "C" int rtc_ada_dx(const void* x,const void* style,const void* dy,
                          void* dx,int rows,int dim,void* stream) {
    ada_dx<<<rows,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,
        (const __nv_bfloat16*)style,(const __nv_bfloat16*)dy,(__nv_bfloat16*)dx,dim);
    return (int)cudaGetLastError();
}
extern "C" int rtc_geglu_dx(const void* gate,const void* up,const void* dy,
                            void* dg,void* du,int n,void* stream) {
    geglu_dx<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,(const __nv_bfloat16*)dy,
        (__nv_bfloat16*)dg,(__nv_bfloat16*)du,n);
    return (int)cudaGetLastError();
}
extern "C" int rtc_geglu_tanh_dx(const void* gate,const void* up,const void* dy,
                                 void* dg,void* du,int n,void* stream) {
    geglu_tanh_dx<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,(const __nv_bfloat16*)dy,
        (__nv_bfloat16*)dg,(__nv_bfloat16*)du,n);
    return (int)cudaGetLastError();
}

__global__ void residual_kernel(const __nv_bfloat16* x,const __nv_bfloat16* p,
    const __nv_bfloat16* style,__nv_bfloat16* y,int rows,int dim,bool backward) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=rows*dim) return;
    int row=j/dim,col=j%dim;
    float gate=__bfloat162float(style[row*3*dim+2*dim+col]);
    float value=__bfloat162float(x[j]);
    y[j]=__float2bfloat16_rn(backward ? value*gate : value+__bfloat162float(p[j])*gate);
}

__global__ void qkv_rope_kernel(const __nv_bfloat16* input,const __nv_bfloat16* rope,
    __nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,int rows) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=rows*2560) return;
    int row=j/2560,col=j%2560;
    if(col>=2304) { v[row*256+col-2304]=input[j]; return; }
    int offset=col%256, base=col-(offset%2);
    float a=__bfloat162float(input[row*2560+base]);
    float b=__bfloat162float(input[row*2560+base+1]);
    float c=__bfloat162float(rope[row*256+offset-(offset%2)]);
    float s=__bfloat162float(rope[row*256+offset-(offset%2)+1]);
    __nv_bfloat16 out=__float2bfloat16_rn(offset%2 ? b*c+a*s : a*c-b*s);
    if(col<2048) q[row*2048+col]=out; else k[row*256+col-2048]=out;
}

__global__ void qkv_rope_dx_kernel(const __nv_bfloat16* dq,const __nv_bfloat16* dk,
    const __nv_bfloat16* dv,const __nv_bfloat16* rope,__nv_bfloat16* dx,int rows) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=rows*2560) return;
    int row=j/2560,col=j%2560;
    if(col>=2304) { dx[j]=dv[row*256+col-2304]; return; }
    int offset=col%256, base=col-(offset%2);
    const __nv_bfloat16* source=col<2048 ? dq+row*2048+base : dk+row*256+base-2048;
    float a=__bfloat162float(source[0]), b=__bfloat162float(source[1]);
    float c=__bfloat162float(rope[row*256+offset-(offset%2)]);
    float s=__bfloat162float(rope[row*256+offset-(offset%2)+1]);
    dx[j]=__float2bfloat16_rn(offset%2 ? b*c-a*s : a*c+b*s);
}

extern "C" int rtc_residual(const void* x,const void* p,const void* style,
    void* y,int rows,int dim,int backward,void* stream) {
    residual_kernel<<<(rows*dim+255)/256,256,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(const __nv_bfloat16*)p,(const __nv_bfloat16*)style,
        (__nv_bfloat16*)y,rows,dim,backward!=0);
    return (int)cudaGetLastError();
}
extern "C" int rtc_qkv_rope(const void* x,const void* rope,void* q,void* k,void* v,
    int rows,void* stream) {
    qkv_rope_kernel<<<(rows*2560+255)/256,256,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(const __nv_bfloat16*)rope,(__nv_bfloat16*)q,
        (__nv_bfloat16*)k,(__nv_bfloat16*)v,rows);
    return (int)cudaGetLastError();
}
extern "C" int rtc_qkv_rope_dx(const void* q,const void* k,const void* v,const void* rope,
    void* dx,int rows,void* stream) {
    qkv_rope_dx_kernel<<<(rows*2560+255)/256,256,0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
        (const __nv_bfloat16*)rope,(__nv_bfloat16*)dx,rows);
    return (int)cudaGetLastError();
}
