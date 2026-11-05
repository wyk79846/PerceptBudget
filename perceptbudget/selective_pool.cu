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

// Restore the original visual multiplicity before attention. These are exact
// bit copies: RoPE is still applied by FlashRT's unchanged native split kernel.
__global__ void expand_rows(const unsigned short* x,const int* selected,
 const int* mapping,unsigned short* y,int visual,int prompt,int columns){
 int row=blockIdx.x,per=visual/2;__shared__ int source;
 if(threadIdx.x==0){
  if(row>=512)source=visual+row-512;
  else{
   int camera=row/256,i=row%256,r=i/16,c=i%16,b=camera*64+(r/2)*8+c/2;
   int representative=selected[b]?((r/2)*2*16+(c/2)*2):i;
   int lo=0,hi=per;
   while(lo<hi){int mid=(lo+hi)/2;if(mapping[camera*per+mid]<representative)lo=mid+1;else hi=mid;}
   source=camera*per+lo;
  }
 }
 __syncthreads();
 for(int c=threadIdx.x;c<columns;c+=blockDim.x)y[row*columns+c]=x[source*columns+c];
}
__global__ void gather_rows(const unsigned short* x,const int* mapping,
 unsigned short* y,int visual,int columns){
 int row=blockIdx.x,per=visual/2;
 int source=row<visual?(row/per)*256+mapping[row]:512+row-visual;
 for(int c=threadIdx.x;c<columns;c+=blockDim.x)y[row*columns+c]=x[source*columns+c];
}
extern "C" int expand_selected_rows(const void* x,const void* selected,const void* mapping,
 void* y,int visual,int prompt,int columns,void* stream){
 if((visual!=320&&visual!=464&&visual!=512)||prompt<0||prompt>200||columns<=0)return (int)cudaErrorInvalidValue;
 expand_rows<<<512+prompt,256,0,(cudaStream_t)stream>>>((const unsigned short*)x,
  (const int*)selected,(const int*)mapping,(unsigned short*)y,visual,prompt,columns);
 return (int)cudaGetLastError();
}
extern "C" int gather_selected_rows(const void* x,const void* mapping,void* y,
 int visual,int prompt,int columns,void* stream){
 if((visual!=320&&visual!=464&&visual!=512)||prompt<0||prompt>200||columns<=0)return (int)cudaErrorInvalidValue;
 gather_rows<<<visual+prompt,256,0,(cudaStream_t)stream>>>((const unsigned short*)x,
  (const int*)mapping,(unsigned short*)y,visual,columns);
 return (int)cudaGetLastError();
}
// Keep original camera/text RoPE coordinates after removing redundant tokens.
__global__ void gather_selective_rope(const __nv_bfloat16* original,const int* mapping,
 __nv_bfloat16* output,int visual,int sequence){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=sequence*256)return;
 int token=j/256,d=j%256,per_camera=visual/2;
 int position=token<visual?mapping[token]+(token/per_camera)*256:512+token-visual;
 output[j]=original[position*256+d];
}
extern "C" int selective_rope(const void* original,const void* mapping,void* output,
 int visual,int sequence,void* stream){
 if((visual!=320&&visual!=464)||sequence<visual)return (int)cudaErrorInvalidValue;
 gather_selective_rope<<<(sequence*256+255)/256,256,0,(cudaStream_t)stream>>>(
  (const __nv_bfloat16*)original,(const int*)mapping,(__nv_bfloat16*)output,visual,sequence);
 return (int)cudaGetLastError();
}
