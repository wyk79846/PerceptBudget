#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
//64 independent2x2 blocks. Feature dispersion after per-token L2 normalization.
__global__ void score_blocks(const __nv_bfloat16* x,float* scores){
 int camera=blockIdx.x/64,b=blockIdx.x%64,t=threadIdx.x,r=(b/8)*2,c=(b%8)*2;
 x+=camera*256*1152;
 int idx[4]={r*16+c,r*16+c+1,(r+1)*16+c,(r+1)*16+c+1};
 __shared__ float sums[4][256];__shared__ float inverse[4];
 for(int q=0;q<4;q++){
  float s=0;for(int d=t;d<1152;d+=256){float v=__bfloat162float(x[idx[q]*1152+d]);s+=v*v;}sums[q][t]=s;
 }
 __syncthreads();
 for(int n=128;n;n/=2){if(t<n)for(int q=0;q<4;q++)sums[q][t]+=sums[q][t+n];__syncthreads();}
 if(t<4)inverse[t]=1.f/fmaxf(sqrtf(sums[t][0]),1e-8f);
 __syncthreads();float v=0;
 for(int d=t;d<1152;d+=256){float a[4],m=0;for(int q=0;q<4;q++){a[q]=__bfloat162float(x[idx[q]*1152+d])*inverse[q];m+=a[q]*.25f;}for(int q=0;q<4;q++){float z=a[q]-m;v+=z*z*.25f;}}
 sums[0][t]=v;__syncthreads();
 for(int n=128;n;n/=2){if(t<n)sums[0][t]+=sums[0][t+n];__syncthreads();}
 if(t==0)scores[camera*64+b]=sums[0][0];
}
template<int MERGED> __global__ void choose_blocks(const float* score,int* selected,int* mapping){
 if(threadIdx.x)return;
 score+=blockIdx.x*64;selected+=blockIdx.x*64;mapping+=blockIdx.x*(256-3*MERGED);
 for(int i=0;i<64;i++){int rank=0;for(int j=0;j<64;j++)if(score[j]<score[i]||(score[j]==score[i]&&j<i))rank++;selected[i]=(rank<MERGED);}
 int k=0;for(int i=0;i<256;i++){int r=i/16,c=i%16,b=(r/2)*8+c/2;if(!selected[b]||(!(r%2)&&!(c%2)))mapping[k++]=i;}
}
template<int MERGED> __global__ void pack_both_tokens(const __nv_bfloat16* x,const int* selected,const int* mapping,__nv_bfloat16* y){
 const int per_camera=256-3*MERGED;
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=2*per_camera*1152)return;
 int token=j/1152,d=j%1152,camera=token/per_camera,i=mapping[token];
 int r=i/16,c=i%16,b=camera*64+(r/2)*8+c/2;x+=camera*256*1152;
 if(!selected[b]){y[j]=x[i*1152+d];return;}
 float v=__bfloat162float(x[i*1152+d])+__bfloat162float(x[(i+1)*1152+d]);
 v+=__bfloat162float(x[(i+16)*1152+d]);v+=__bfloat162float(x[(i+17)*1152+d]);y[j]=__float2bfloat16_rn(v*.25f);
}
__global__ void pack_tokens(const __nv_bfloat16* x,const int* selected,const int* mapping,__nv_bfloat16* y){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=416*1152)return;int token=j/1152,d=j%1152;
 if(token>=160){y[j]=x[(256+token-160)*1152+d];return;}
 int i=mapping[token],r=i/16,c=i%16,b=(r/2)*8+c/2;
 if(!selected[b]){y[j]=x[i*1152+d];return;}
 float v=__bfloat162float(x[i*1152+d])+__bfloat162float(x[(i+1)*1152+d]);
 v+=__bfloat162float(x[(i+16)*1152+d]);v+=__bfloat162float(x[(i+17)*1152+d]);y[j]=__float2bfloat16_rn(v*.25f);
}
extern "C" int selective_pool416(const void* x,void* scores,void* selected,void* mapping,void* y,void* stream){
 cudaStream_t s=(cudaStream_t)stream;
 score_blocks<<<64,256,0,s>>>((const __nv_bfloat16*)x,(float*)scores);
 choose_blocks<32><<<1,1,0,s>>>((float*)scores,(int*)selected,(int*)mapping);
 pack_tokens<<<(416*1152+255)/256,256,0,s>>>((const __nv_bfloat16*)x,(int*)selected,(int*)mapping,(__nv_bfloat16*)y);
 return (int)cudaGetLastError();
}
extern "C" int selective_pool320(const void* x,void* scores,void* selected,void* mapping,void* y,void* stream){
 cudaStream_t s=(cudaStream_t)stream;
 score_blocks<<<128,256,0,s>>>((const __nv_bfloat16*)x,(float*)scores);
 choose_blocks<32><<<2,1,0,s>>>((float*)scores,(int*)selected,(int*)mapping);
 pack_both_tokens<32><<<(320*1152+255)/256,256,0,s>>>((const __nv_bfloat16*)x,(int*)selected,(int*)mapping,(__nv_bfloat16*)y);
 return (int)cudaGetLastError();
}
extern "C" int selective_pool464(const void* x,void* scores,void* selected,void* mapping,void* y,void* stream){
 cudaStream_t s=(cudaStream_t)stream;
 score_blocks<<<128,256,0,s>>>((const __nv_bfloat16*)x,(float*)scores);
 choose_blocks<8><<<2,1,0,s>>>((float*)scores,(int*)selected,(int*)mapping);
 pack_both_tokens<8><<<(464*1152+255)/256,256,0,s>>>((const __nv_bfloat16*)x,(int*)selected,(int*)mapping,(__nv_bfloat16*)y);
 return (int)cudaGetLastError();
}
