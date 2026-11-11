#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// 256 external tokens, projected width 2048; wrist is never selected/pruned.
__global__ void norms(const __nv_bfloat16* x, float* inverse) {
    int i = blockIdx.x, t = threadIdx.x;
    __shared__ float sums[256];
    float value = 0;
    for (int d = t; d < 2048; d += 256) {
        float v = __bfloat162float(x[i * 2048 + d]);
        value += v * v;
    }
    sums[t] = value; __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n) sums[t] += sums[t + n];
        __syncthreads();
    }
    if (!t) inverse[i] = 1.f / fmaxf(sqrtf(sums[0]), 1e-8f);
}

__global__ void cosine_distances(const __nv_bfloat16* x, const float* inverse, float* distances) {
    int i = blockIdx.x, j = blockIdx.y, t = threadIdx.x;
    if (i > j) return;
    if (i == j) { if (!t) distances[i * 256 + j] = INFINITY; return; }
    __shared__ float sums[256];
    float dot = 0;
    for (int d = t; d < 2048; d += 256)
        dot += __bfloat162float(x[i * 2048 + d]) * __bfloat162float(x[j * 2048 + d]);
    sums[t] = dot; __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (t < n) sums[t] += sums[t + n];
        __syncthreads();
    }
    if (!t) {
        float distance = 1.f - sums[0] * inverse[i] * inverse[j];
        distances[i * 256 + j] = distance;
        distances[j * 256 + i] = distance;
    }
}

__global__ void max_min_select(const float* distances, int* mapping, const int* seeds=nullptr) {
    int t = threadIdx.x;
    __shared__ float best_value[256];
    __shared__ int best_index[256], selected[256], choice;
    selected[t] = 0;
    float nearest = INFINITY;
    for (int j = 0; j < 256; ++j) nearest = fminf(nearest, distances[t * 256 + j]);
    __syncthreads();
    for (int k = 0; k < 64; ++k) {
        if (seeds && k < seeds[0]) {
            if (!t) { choice = seeds[1 + k]; selected[choice] = 1; }
            __syncthreads();
            float d = distances[t * 256 + choice];
            nearest = k == 0 ? d : fminf(nearest, d);
            continue;
        }
        best_value[t] = selected[t] ? -INFINITY : nearest;
        best_index[t] = t;
        __syncthreads();
        for (int n = 128; n; n /= 2) {
            if (t < n && (best_value[t + n] > best_value[t] ||
                (best_value[t + n] == best_value[t] && best_index[t + n] < best_index[t]))) {
                best_value[t] = best_value[t + n];
                best_index[t] = best_index[t + n];
            }
            __syncthreads();
        }
        if (!t) { choice = best_index[0]; selected[choice] = 1; }
        __syncthreads();
        float d = distances[t * 256 + choice];
        nearest = k == 0 ? d : fminf(nearest, d);
    }
    __syncthreads();
    // Restore spatial order, while the separate RoPE gather retains exact
    // original positions. Never compact wrist coordinates down to 64..319.
    if (!t) {
        int k = 0;
        for (int i = 0; i < 256; ++i) if (selected[i]) mapping[k++] = i;
    }
    mapping[64 + t] = 256 + t;
}

__global__ void gather_tokens(const __nv_bfloat16* source, const int* mapping, __nv_bfloat16* output) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 320 * 1152) output[j] = source[mapping[j / 1152] * 1152 + j % 1152];
}

__global__ void gather_rope(const __nv_bfloat16* original, const int* mapping,
                            __nv_bfloat16* output, int sequence) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= sequence * 256) return;
    int token = j / 256;
    int position = token < 320 ? mapping[token] : 512 + token - 320;
    output[j] = original[position * 256 + j % 256];
}

extern "C" int external_divprune(const void* projected, void* inverse, void* distances,
                                void* mapping, const void* source, void* output, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    norms<<<256, 256, 0, s>>>((const __nv_bfloat16*)projected, (float*)inverse);
    cosine_distances<<<dim3(256, 256), 256, 0, s>>>((const __nv_bfloat16*)projected,
        (float*)inverse, (float*)distances);
    max_min_select<<<1, 256, 0, s>>>((const float*)distances, (int*)mapping);
    gather_tokens<<<(320 * 1152 + 255) / 256, 256, 0, s>>>(
        (const __nv_bfloat16*)source, (const int*)mapping, (__nv_bfloat16*)output);
    return (int)cudaGetLastError();
}

extern "C" int external_divprune_rope(const void* original, const void* mapping,
                                     void* output, int sequence, void* stream) {
    if (sequence < 320) return (int)cudaErrorInvalidValue;
    gather_rope<<<(sequence * 256 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)original, (const int*)mapping, (__nv_bfloat16*)output, sequence);
    return (int)cudaGetLastError();
}

extern "C" int external_divprune_seeded(const void* projected, void* inverse, void* distances,
                                       void* mapping, const void* source, void* output,
                                       const void* seeds, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    norms<<<256, 256, 0, s>>>((const __nv_bfloat16*)projected, (float*)inverse);
    cosine_distances<<<dim3(256, 256), 256, 0, s>>>((const __nv_bfloat16*)projected,
        (float*)inverse, (float*)distances);
    max_min_select<<<1, 256, 0, s>>>((const float*)distances, (int*)mapping, (const int*)seeds);
    gather_tokens<<<(320 * 1152 + 255) / 256, 256, 0, s>>>(
        (const __nv_bfloat16*)source, (const int*)mapping, (__nv_bfloat16*)output);
    return (int)cudaGetLastError();
}
