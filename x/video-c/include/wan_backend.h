/* Thin compute backend for wan-c — UMA or in-process CUDA.
 * See docs/cuda-uma-toolkit.md */
#ifndef WAN_BACKEND_H
#define WAN_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wan_backend wan_backend;

typedef struct wan_backend_vtable {
  const char *name;
  void (*destroy)(wan_backend *b);
  int (*buf_alloc)(wan_backend *b, const char *name, size_t bytes);
  int (*buf_free)(wan_backend *b, const char *name);
  int (*buf_put)(wan_backend *b, const char *name, const void *host, size_t bytes);
  int (*buf_get)(wan_backend *b, const char *name, void *host, size_t bytes);
  /*
   * BANK: persistent weight blob keyed by name.
   * Contract: layer keys (e.g. "Wu_L3"); bank_evict before put on pager miss.
   */
  int (*bank_put)(wan_backend *b, const char *name, const void *host, size_t bytes);
  int (*bank_bind)(wan_backend *b, const char *bank, const char *as_buf);
  int (*bank_evict)(wan_backend *b, const char *name);
  /* Y[M,N] = A[M,K] * B[K,N]  (row-major f32). */
  int (*gemm_f32)(wan_backend *be, const char *a, const char *bmat, const char *y, int M,
                  int N, int K);
  /* DiT elementwise (phase-2 FFN / AdaLN). w may be NULL for LN. */
  int (*layernorm)(wan_backend *b, const char *x, const char *y, const char *w, int N,
                   int D);
  /* y = x * (1+scale) + shift ; scale/shift are length-D buffers. */
  int (*affine_mul_add)(wan_backend *b, const char *x, const char *y, const char *scale,
                        const char *shift, int N, int D);
  int (*gelu_tanh)(wan_backend *b, const char *x, const char *y, size_t n);
  /* y = x + delta * gate ; gate length D. */
  int (*gated_residual)(wan_backend *b, const char *x, const char *delta,
                        const char *gate, const char *y, int N, int D);
  /* WanRMSNorm over last dim D (weight length D). */
  int (*rmsnorm)(wan_backend *b, const char *x, const char *y, const char *w, int N,
                 int D);
  int (*bias_add)(wan_backend *b, const char *x, const char *y, const char *bias,
                  int N, int D);
  /* y = x * scale + bias (Wan LayerNorm affine); scale/bias length D. */
  int (*scale_bias)(wan_backend *b, const char *x, const char *y, const char *scale,
                    const char *bias, int N, int D);
  /* Attention path: buffers are [T,H,HD] = [T, D] with D=H*HD. */
  int (*head_rmsnorm)(wan_backend *b, const char *x, const char *y, const char *w,
                      int T, int H, int HD);
  int (*rope3)(wan_backend *b, const char *x, const char *y, int T, int H, int HD,
               int grid_t, int grid_h, int grid_w);
  int (*attn_sdpa)(wan_backend *b, const char *q, const char *k, const char *v,
                   const char *out, int T, int Tk, int H, int HD);
  int (*sync)(wan_backend *b);
  size_t (*device_bytes)(wan_backend *b);
} wan_backend_vtable;

struct wan_backend {
  const wan_backend_vtable *vt;
  void *impl;
};

static inline void wan_backend_destroy(wan_backend *b) {
  if (b && b->vt && b->vt->destroy) b->vt->destroy(b);
}

wan_backend *wan_backend_cuda_create(int device);
wan_backend *wan_backend_host_create(void);
#if defined(__APPLE__)
wan_backend *wan_backend_uma_create(const char *sock);
#endif

#ifdef __cplusplus
}
#endif

#endif
