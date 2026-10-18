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
