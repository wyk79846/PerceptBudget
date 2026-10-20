// Original pi05 AdaRMS: BF16 hidden/output, FP32 modulation and reduction.
// No weight/style gradients. Keep the direct and variance BF16 gradient
// branches separately rounded, rather than replacing them by one FP32 sum.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void ada_forward(const __nv_bfloat16* x,const float* style,
    __nv_bfloat16* y,float* inverse,int dim,float eps) {
    __shared__ float sum[256];
    int row=blockIdx.x,t=threadIdx.x;
    float ss=0.f;
    for(int j=t;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]); ss+=v*v;
    }
    sum[t]=ss; __syncthreads();
    for(int s=128;s;s>>=1) {
        if(t<s) sum[t]+=sum[t+s]; __syncthreads();
    }
    float r=rsqrtf(sum[0]/dim+eps);
    if(t==0) inverse[row]=r;
    for(int j=t;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]);
        float normalized=v*r;
        y[row*dim+j]=__float2bfloat16_rn(normalized*(1.f+style[j])+style[dim+j]);
    }
}

__global__ void ada_backward(const __nv_bfloat16* x,const float* style,
    const float* inverse,const __nv_bfloat16* dy,__nv_bfloat16* dx,int dim) {
    __shared__ float dot[256];
    int row=blockIdx.x,t=threadIdx.x;
    float local=0.f;
    for(int j=t;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]);
        float z=__bfloat162float(dy[row*dim+j])*(1.f+style[j]);
        local+=z*v;
    }
    dot[t]=local; __syncthreads();
    for(int s=128;s;s>>=1) {
        if(t<s) dot[t]+=dot[t+s]; __syncthreads();
    }
    float r=inverse[row];
    float dvar=(dot[0]*-.5f)*(r*r*r);
    float dmean=dvar/dim;
    for(int j=t;j<dim;j+=256) {
        float v=__bfloat162float(x[row*dim+j]);
        float z=__bfloat162float(dy[row*dim+j])*(1.f+style[j]);
        __nv_bfloat16 direct=__float2bfloat16_rn(z*r);
        __nv_bfloat16 variance=__float2bfloat16_rn(dmean*(2.f*v));
        dx[row*dim+j]=__float2bfloat16_rn(__bfloat162float(direct)+__bfloat162float(variance));
    }
}

extern "C" int pb_norm_forward(const void* x,const void* style,void* y,void* inverse,
    int rows,int dim,float eps,void* stream) {
    ada_forward<<<rows,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,
        (const float*)style,(__nv_bfloat16*)y,(float*)inverse,dim,eps);
    return (int)cudaGetLastError();
}
extern "C" int pb_norm_backward(const void* x,const void* style,const void* inverse,
    const void* dy,void* dx,int rows,int dim,void* stream) {
    ada_backward<<<rows,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,
        (const float*)style,(const float*)inverse,(const __nv_bfloat16*)dy,
        (__nv_bfloat16*)dx,dim);
    return (int)cudaGetLastError();
}
