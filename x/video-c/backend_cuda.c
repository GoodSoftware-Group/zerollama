/* In-process CUDA wan_backend — cuBLAS GEMM + elementwise DiT ops. */
#include "backend_ops.h"
#include "wan_backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CUDA_CAP = 256 };

typedef struct {
  char name[96];
  void *dev;
  size_t bytes;
  int used;
  int is_bank;
  int is_alias; /* shares another slot's device ptr; do not cudaFree */
} cuda_slot;

typedef struct {
  int device;
  cublasHandle_t handle;
  cuda_slot slots[CUDA_CAP];
  size_t live_bytes;
  size_t peak_bytes;
} cuda_impl;

static cuda_slot *find_slot(cuda_impl *im, const char *name) {
  for (int i = 0; i < CUDA_CAP; i++)
    if (im->slots[i].used && strcmp(im->slots[i].name, name) == 0)
      return &im->slots[i];
  return NULL;
}

static cuda_slot *alloc_slot(cuda_impl *im, const char *name) {
  cuda_slot *s = find_slot(im, name);
  if (s) return s;
  for (int i = 0; i < CUDA_CAP; i++) {
    if (!im->slots[i].used) {
      s = &im->slots[i];
      memset(s, 0, sizeof(*s));
      snprintf(s->name, sizeof(s->name), "%s", name);
      s->used = 1;
      return s;
    }
  }
  return NULL;
}

static void note_bytes(cuda_impl *im) {
  if (im->live_bytes > im->peak_bytes) im->peak_bytes = im->live_bytes;
}

/* Drop aliases that point at the same device pointer (after owner free). */
static void clear_aliases_to(cuda_impl *im, void *dev) {
  if (!dev) return;
  for (int i = 0; i < CUDA_CAP; i++) {
    if (im->slots[i].used && im->slots[i].is_alias && im->slots[i].dev == dev)
      memset(&im->slots[i], 0, sizeof(im->slots[i]));
  }
}

static int free_owned_dev(cuda_impl *im, cuda_slot *s) {
  if (!s || !s->used) return 0;
  if (s->is_alias) {
    memset(s, 0, sizeof(*s));
    return 0;
  }
  void *dev = s->dev;
  size_t bytes = s->bytes;
  if (dev) {
    cudaFree(dev);
    im->live_bytes -= bytes;
    clear_aliases_to(im, dev);
  }
  memset(s, 0, sizeof(*s));
  return 0;
}

static void cuda_destroy(wan_backend *b) {
  if (!b) return;
  cuda_impl *im = (cuda_impl *)b->impl;
  if (im) {
    for (int i = 0; i < CUDA_CAP; i++) {
      if (!im->slots[i].used) continue;
      if (im->slots[i].is_alias) {
        memset(&im->slots[i], 0, sizeof(im->slots[i]));
        continue;
      }
      if (im->slots[i].dev) cudaFree(im->slots[i].dev);
    }
    if (im->handle) cublasDestroy(im->handle);
    free(im);
  }
  free(b);
}

static int cuda_buf_alloc(wan_backend *b, const char *name, size_t bytes) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = alloc_slot(im, name);
  if (!s) return -1;
  if (s->is_alias) {
    /* Replacing an alias with an owned buffer. */
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->used = 1;
  }
  if (s->dev && !s->is_alias && s->bytes == bytes) return 0;
  if (s->dev && !s->is_alias) {
    cudaFree(s->dev);
    im->live_bytes -= s->bytes;
    clear_aliases_to(im, s->dev);
    s->dev = NULL;
  }
  if (cudaMalloc(&s->dev, bytes) != cudaSuccess) return -1;
  s->bytes = bytes;
  s->is_bank = 0;
  s->is_alias = 0;
  im->live_bytes += bytes;
  note_bytes(im);
  return 0;
}

static int cuda_buf_free(wan_backend *b, const char *name) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = find_slot(im, name);
  if (!s) return 0;
  return free_owned_dev(im, s);
}

static int cuda_buf_put(wan_backend *b, const char *name, const void *host,
                        size_t bytes) {
  if (cuda_buf_alloc(b, name, bytes) != 0) return -1;
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = find_slot(im, name);
  if (!s || !s->dev) return -1;
  return cudaMemcpy(s->dev, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess
             ? 0
             : -1;
}

static int cuda_buf_get(wan_backend *b, const char *name, void *host,
                        size_t bytes) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = find_slot(im, name);
  if (!s || !s->dev || s->bytes < bytes) return -1;
  return cudaMemcpy(host, s->dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
             ? 0
             : -1;
}

static int cuda_bank_put(wan_backend *b, const char *name, const void *host,
                         size_t bytes) {
  if (cuda_buf_put(b, name, host, bytes) != 0) return -1;
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = find_slot(im, name);
  if (s) {
    s->is_bank = 1;
    s->is_alias = 0;
  }
  return 0;
}

static int cuda_bank_bind(wan_backend *b, const char *bank, const char *as_buf) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sb = find_slot(im, bank);
  if (!sb || !sb->dev) return -1;
  cuda_slot *sa = find_slot(im, as_buf);
  if (sa && !sa->is_alias && sa->dev && sa->dev != sb->dev) {
    /* Drop previous owned buffer under this working name. */
    free_owned_dev(im, sa);
    sa = NULL;
  }
  if (!sa) {
    sa = alloc_slot(im, as_buf);
    if (!sa) return -1;
  }
  sa->dev = sb->dev;
  sa->bytes = sb->bytes;
  sa->is_alias = 1;
  sa->is_bank = 0;
  return 0;
}

static int cuda_bank_evict(wan_backend *b, const char *name) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *s = find_slot(im, name);
  if (!s) return 0;
  return free_owned_dev(im, s);
}

static int cuda_gemm_f32(wan_backend *b, const char *a, const char *bname,
                         const char *y, int M, int N, int K) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sa = find_slot(im, a);
  cuda_slot *sb = find_slot(im, bname);
  size_t ybytes = (size_t)M * (size_t)N * sizeof(float);
  if (cuda_buf_alloc(b, y, ybytes) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sa || !sb || !sy || !sa->dev || !sb->dev || !sy->dev) return -1;

  /* Row-major Y = A[M,K] * B[K,N] via column-major cuBLAS:
   * C^T = B^T * A^T  → gemm(N, M, K, B, N, A, K, C, N) with no-trans. */
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasSgemm(im->handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, (const float *)sb->dev, N,
                                  (const float *)sa->dev, K, &beta,
                                  (float *)sy->dev, N);
  return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int cuda_layernorm(wan_backend *b, const char *x, const char *y,
                          const char *w, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  cuda_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy || !sx->dev || !sy->dev) return -1;
  return wan_cuda_layernorm_f32((float *)sy->dev, (const float *)sx->dev,
                                sw && sw->dev ? (const float *)sw->dev : NULL, N,
                                D, 1e-6f);
}

static int cuda_affine(wan_backend *b, const char *x, const char *y,
                       const char *scale, const char *shift, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  cuda_slot *ss = find_slot(im, scale);
  cuda_slot *sh = find_slot(im, shift);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sx || !sy || !ss || !sh || !sx->dev || !sy->dev || !ss->dev || !sh->dev)
    return -1;
  return wan_cuda_affine_mul_add_f32((float *)sy->dev, (const float *)sx->dev,
                                     (const float *)ss->dev,
                                     (const float *)sh->dev, N, D);
}

static int cuda_gelu(wan_backend *b, const char *x, const char *y, size_t n) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  if (cuda_buf_alloc(b, y, n * sizeof(float)) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sx || !sy || !sx->dev || !sy->dev) return -1;
  return wan_cuda_gelu_tanh_f32((float *)sy->dev, (const float *)sx->dev, n);
}

static int cuda_gated_res(wan_backend *b, const char *x, const char *delta,
                          const char *gate, const char *y, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  cuda_slot *sd = find_slot(im, delta);
  cuda_slot *sg = find_slot(im, gate);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sx || !sd || !sg || !sy || !sx->dev || !sd->dev || !sg->dev || !sy->dev)
    return -1;
  return wan_cuda_gated_residual_f32((float *)sy->dev, (const float *)sx->dev,
                                     (const float *)sd->dev,
                                     (const float *)sg->dev, N, D);
}

static int cuda_rmsnorm(wan_backend *b, const char *x, const char *y,
                        const char *w, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  cuda_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy || !sx->dev || !sy->dev) return -1;
  return wan_cuda_rmsnorm_f32((float *)sy->dev, (const float *)sx->dev,
                              sw && sw->dev ? (const float *)sw->dev : NULL, N, D,
                              1e-6f);
}

static int cuda_bias_add(wan_backend *b, const char *x, const char *y,
                         const char *bias, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  cuda_slot *sb = find_slot(im, bias);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sx || !sb || !sy || !sx->dev || !sb->dev || !sy->dev) return -1;
  return wan_cuda_bias_add_f32((float *)sy->dev, (const float *)sx->dev,
                               (const float *)sb->dev, N, D);
}

static int cuda_scale_bias(wan_backend *b, const char *x, const char *y,
                           const char *scale, const char *bias, int N, int D) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  cuda_slot *ss = find_slot(im, scale);
  cuda_slot *sb = find_slot(im, bias);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  if (!sx || !ss || !sb || !sy || !sx->dev || !ss->dev || !sb->dev || !sy->dev)
    return -1;
  return wan_cuda_scale_bias_f32((float *)sy->dev, (const float *)sx->dev,
                                 (const float *)ss->dev, (const float *)sb->dev,
                                 N, D);
}

static int cuda_head_rms(wan_backend *b, const char *x, const char *y,
                         const char *w, int T, int H, int HD) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sx = find_slot(im, x);
  size_t yb = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  if (cuda_buf_alloc(b, y, yb) != 0) return -1;
  cuda_slot *sy = find_slot(im, y);
  cuda_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy || !sx->dev || !sy->dev) return -1;
  return wan_cuda_head_rmsnorm_f32((float *)sy->dev, (const float *)sx->dev,
                                   sw && sw->dev ? (const float *)sw->dev : NULL,
                                   T, H, HD, 1e-6f);
}

/* RoPE matches Wan host reference — D2H / apply / H2D (cheap vs GEMM). */
static int cuda_rope3(wan_backend *b, const char *x, const char *y, int T, int H,
                      int HD, int gt, int gh, int gw) {
  size_t yb = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  float *tmp = malloc(yb);
  float *out = malloc(yb);
  if (!tmp || !out) {
    free(tmp);
    free(out);
    return -1;
  }
  int rc = -1;
  if (cuda_buf_get(b, x, tmp, yb) != 0) goto done;
  if (wan_op_rope3_f32(out, tmp, T, H, HD, gt, gh, gw) != 0) goto done;
  if (cuda_buf_put(b, y, out, yb) != 0) goto done;
  rc = 0;
done:
  free(tmp);
  free(out);
  return rc;
}

static int cuda_attn(wan_backend *b, const char *q, const char *k, const char *v,
                     const char *out, int T, int Tk, int H, int HD) {
  cuda_impl *im = (cuda_impl *)b->impl;
  cuda_slot *sq = find_slot(im, q);
  cuda_slot *sk = find_slot(im, k);
  cuda_slot *sv = find_slot(im, v);
  size_t ob = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  if (cuda_buf_alloc(b, out, ob) != 0) return -1;
  cuda_slot *so = find_slot(im, out);
  if (!sq || !sk || !sv || !so || !sq->dev || !sk->dev || !sv->dev || !so->dev)
    return -1;
  return wan_cuda_attn_sdpa_f32((float *)so->dev, (const float *)sq->dev,
                                (const float *)sk->dev, (const float *)sv->dev, T,
                                Tk, H, HD);
}

static int cuda_sync(wan_backend *b) {
  (void)b;
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

static size_t cuda_device_bytes(wan_backend *b) {
  cuda_impl *im = (cuda_impl *)b->impl;
  return im ? im->peak_bytes : 0;
}

static const wan_backend_vtable cuda_vt = {
    .name = "cuda",
    .destroy = cuda_destroy,
    .buf_alloc = cuda_buf_alloc,
    .buf_free = cuda_buf_free,
    .buf_put = cuda_buf_put,
    .buf_get = cuda_buf_get,
    .bank_put = cuda_bank_put,
    .bank_bind = cuda_bank_bind,
    .bank_evict = cuda_bank_evict,
    .gemm_f32 = cuda_gemm_f32,
    .layernorm = cuda_layernorm,
    .affine_mul_add = cuda_affine,
    .gelu_tanh = cuda_gelu,
    .gated_residual = cuda_gated_res,
    .rmsnorm = cuda_rmsnorm,
    .bias_add = cuda_bias_add,
    .scale_bias = cuda_scale_bias,
    .head_rmsnorm = cuda_head_rms,
    .rope3 = cuda_rope3,
    .attn_sdpa = cuda_attn,
    .sync = cuda_sync,
    .device_bytes = cuda_device_bytes,
};

wan_backend *wan_backend_cuda_create(int device) {
  if (cudaSetDevice(device) != cudaSuccess) return NULL;
  cuda_impl *im = calloc(1, sizeof(*im));
  wan_backend *b = calloc(1, sizeof(*b));
  if (!im || !b) {
    free(im);
    free(b);
    return NULL;
  }
  im->device = device;
  if (cublasCreate(&im->handle) != CUBLAS_STATUS_SUCCESS) {
    free(im);
    free(b);
    return NULL;
  }
  b->vt = &cuda_vt;
  b->impl = im;
  return b;
}
