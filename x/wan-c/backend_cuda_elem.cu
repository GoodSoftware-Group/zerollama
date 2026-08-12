/* CUDA elementwise kernels for DiT FFN half-block. */
#include "backend_ops.h"

#include <cuda_runtime.h>
#include <math.h>

#ifndef WAN_HAVE_CUDA
#define WAN_HAVE_CUDA 1
#endif

__global__ static void k_layernorm(float *y, const float *x, const float *w,
                                   int N, int D, float eps) {
  int n = blockIdx.x;
  if (n >= N) return;
  const float *xr = x + (size_t)n * D;
  float *yr = y + (size_t)n * D;
  float mean = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) mean += xr[d];
  __shared__ float red[256];
  red[threadIdx.x] = mean;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  mean = red[0] / (float)D;
  float var = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float t = xr[d] - mean;
    var += t * t;
  }
  red[threadIdx.x] = var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float v = (xr[d] - mean) * inv;
    yr[d] = w ? v * w[d] : v;
  }
}

__global__ static void k_affine(float *y, const float *x, const float *scale,
                                const float *shift, int N, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t ntot = (size_t)N * (size_t)D;
  if (i >= ntot) return;
  int d = (int)(i % (size_t)D);
  y[i] = x[i] * (1.f + scale[d]) + shift[d];
}

__global__ static void k_gelu(float *y, const float *x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  y[i] = 0.5f * v * (1.f + tanhf(c));
}

__global__ static void k_gated_res(float *y, const float *x, const float *delta,
                                   const float *gate, int N, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t ntot = (size_t)N * (size_t)D;
  if (i >= ntot) return;
  int d = (int)(i % (size_t)D);
  y[i] = x[i] + delta[i] * gate[d];
}

extern "C" int wan_cuda_layernorm_f32(float *y, const float *x, const float *w,
                                      int N, int D, float eps) {
  if (N < 1 || D < 1) return -1;
  int thr = D < 256 ? D : 256;
  /* thr must be power-of-2 for reduction — snap down. */
  int p2 = 1;
  while (p2 * 2 <= thr) p2 *= 2;
  k_layernorm<<<N, p2>>>(y, x, w, N, D, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int wan_cuda_affine_mul_add_f32(float *y, const float *x,
                                           const float *scale, const float *shift,
                                           int N, int D) {
  size_t n = (size_t)N * (size_t)D;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_affine<<<blk, thr>>>(y, x, scale, shift, N, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int wan_cuda_gelu_tanh_f32(float *y, const float *x, size_t n) {
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_gelu<<<blk, thr>>>(y, x, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int wan_cuda_gated_residual_f32(float *y, const float *x,
                                           const float *delta, const float *gate,
                                           int N, int D) {
  size_t n = (size_t)N * (size_t)D;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_gated_res<<<blk, thr>>>(y, x, delta, gate, N, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ static void k_rmsnorm(float *y, const float *x, const float *w, int N,
                                 int D, float eps) {
  int n = blockIdx.x;
  if (n >= N) return;
  const float *xr = x + (size_t)n * D;
  float *yr = y + (size_t)n * D;
  float ss = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) ss += xr[d] * xr[d];
  __shared__ float red[256];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float v = xr[d] * inv;
    yr[d] = w ? v * w[d] : v;
  }
}

__global__ static void k_bias_add(float *y, const float *x, const float *bias,
                                  int N, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t ntot = (size_t)N * (size_t)D;
  if (i >= ntot) return;
  int d = (int)(i % (size_t)D);
  y[i] = x[i] + bias[d];
}

extern "C" int wan_cuda_rmsnorm_f32(float *y, const float *x, const float *w,
                                    int N, int D, float eps) {
  if (N < 1 || D < 1) return -1;
  int thr = D < 256 ? D : 256;
  int p2 = 1;
  while (p2 * 2 <= thr) p2 *= 2;
  k_rmsnorm<<<N, p2>>>(y, x, w, N, D, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int wan_cuda_bias_add_f32(float *y, const float *x, const float *bias,
                                     int N, int D) {
  size_t n = (size_t)N * (size_t)D;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_bias_add<<<blk, thr>>>(y, x, bias, N, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ static void k_scale_bias(float *y, const float *x, const float *scale,
                                    const float *bias, int N, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t ntot = (size_t)N * (size_t)D;
  if (i >= ntot) return;
  int d = (int)(i % (size_t)D);
  y[i] = x[i] * scale[d] + bias[d];
}

extern "C" int wan_cuda_scale_bias_f32(float *y, const float *x,
                                       const float *scale, const float *bias,
                                       int N, int D) {
  size_t n = (size_t)N * (size_t)D;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_scale_bias<<<blk, thr>>>(y, x, scale, bias, N, D);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ static void k_head_rms(float *y, const float *x, const float *w,
                                  int rows, int HD, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (size_t)r * HD;
  float *yr = y + (size_t)r * HD;
  float ss = 0.f;
  for (int d = threadIdx.x; d < HD; d += blockDim.x) ss += xr[d] * xr[d];
  __shared__ float red[256];
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)HD + eps);
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float v = xr[d] * inv;
    yr[d] = w ? v * w[d] : v;
  }
}

extern "C" int wan_cuda_head_rmsnorm_f32(float *y, const float *x,
                                         const float *w, int T, int H, int HD,
                                         float eps) {
  int rows = T * H;
  if (rows < 1 || HD < 1) return -1;
  int thr = HD < 256 ? HD : 256;
  int p2 = 1;
  while (p2 * 2 <= thr) p2 *= 2;
  k_head_rms<<<rows, p2>>>(y, x, w, rows, HD, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* One block = one (t,h) query row; naive SDPA for lab T. */
__global__ static void k_attn_row(float *out, const float *q, const float *k,
                                  const float *v, int T, int Tk, int H, int HD,
                                  float scale) {
  int idx = blockIdx.x; /* t * H + h */
  if (idx >= T * H) return;
  int t = idx / H;
  int h = idx % H;
  const float *qr = q + ((size_t)t * H + h) * HD;
  extern __shared__ float sm[];
  float *scores = sm; /* Tk */
  float maxv = -1e30f;
  for (int s = threadIdx.x; s < Tk; s += blockDim.x) {
    const float *kr = k + ((size_t)s * H + h) * HD;
    float dot = 0.f;
    for (int d = 0; d < HD; d++) dot += qr[d] * kr[d];
    scores[s] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int s = 0; s < Tk; s++)
      if (scores[s] > maxv) maxv = scores[s];
    float sum = 0.f;
    for (int s = 0; s < Tk; s++) {
      scores[s] = expf(scores[s] - maxv);
      sum += scores[s];
    }
    float inv = 1.f / sum;
    for (int s = 0; s < Tk; s++) scores[s] *= inv;
  }
  __syncthreads();
  float *orow = out + ((size_t)t * H + h) * HD;
  for (int d = threadIdx.x; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s < Tk; s++) {
      const float *vr = v + ((size_t)s * H + h) * HD;
      acc += scores[s] * vr[d];
    }
    orow[d] = acc;
  }
}

extern "C" int wan_cuda_attn_sdpa_f32(float *out, const float *q, const float *k,
                                      const float *v, int T, int Tk, int H,
                                      int HD) {
  if (T < 1 || Tk < 1 || H < 1 || HD < 1) return -1;
  float scale = 1.f / sqrtf((float)HD);
  int thr = HD < 128 ? HD : 128;
  if (thr < 1) thr = 1;
  size_t shmem = (size_t)Tk * sizeof(float);
  k_attn_row<<<T * H, thr, shmem>>>(out, q, k, v, T, Tk, H, HD, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
