/* Darwin UMA adapter — wraps uma_client + uma_buf_pool when linked on Mac.
 * Linux: not used by cuda-lab (CUDA twin is the phase-1 unlock). */
#include "wan_backend.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include "uma_buf_load.h"
#include "uma/client.h"

#include <string.h>

typedef struct {
  UmaClient *uma;
  uma_buf_pool *pool;
  size_t peak_bytes;
  size_t live_bytes;
} uma_impl;

static void uma_destroy(wan_backend *b) {
  if (!b) return;
  uma_impl *im = (uma_impl *)b->impl;
  if (im) {
    if (im->pool) uma_buf_pool_destroy(im->pool);
    if (im->uma) uma_client_close(im->uma);
    free(im);
  }
  free(b);
}

static int uma_buf_alloc(wan_backend *b, const char *name, size_t bytes) {
  uma_impl *im = (uma_impl *)b->impl;
  int rc = uma_buf_pool_alloc(im->pool, name, bytes);
  if (rc == 0) {
    im->live_bytes += bytes; /* best-effort; pool may replace */
    if (im->live_bytes > im->peak_bytes) im->peak_bytes = im->live_bytes;
  }
  return rc;
}

static int uma_buf_free_vt(wan_backend *b, const char *name) {
  uma_impl *im = (uma_impl *)b->impl;
  return uma_buf_pool_free(im->pool, name);
}

static int uma_buf_put_vt(wan_backend *b, const char *name, const void *host,
                          size_t bytes) {
  uma_impl *im = (uma_impl *)b->impl;
  return uma_buf_pool_ensure_put(im->pool, name, host, bytes);
}

static int uma_buf_get_vt(wan_backend *b, const char *name, void *host,
                          size_t bytes) {
  uma_impl *im = (uma_impl *)b->impl;
  char resp[256];
  size_t got = 0;
  if (uma_client_buf_get(im->uma, name, host, bytes, &got, resp, sizeof(resp)) !=
      0)
    return -1;
  return got == bytes ? 0 : -1;
}

static int uma_bank_put_vt(wan_backend *b, const char *name, const void *host,
                           size_t bytes) {
  uma_impl *im = (uma_impl *)b->impl;
  return uma_buf_pool_bank_put(im->pool, name, host, bytes);
}

static int uma_bank_bind_vt(wan_backend *b, const char *bank, const char *as_buf) {
  uma_impl *im = (uma_impl *)b->impl;
  return uma_buf_pool_bank_bind(im->pool, bank, as_buf);
}

static int uma_bank_evict_vt(wan_backend *b, const char *name) {
  /* UMA pool has no BANK_EVICT yet — free the bound working name if any.
   * Layer-keyed banks persist until daemon TTL; Mac eviction lands with BANK API. */
  uma_impl *im = (uma_impl *)b->impl;
  (void)name;
  (void)im;
  return 0; /* no-op until uma_buf_pool gains bank_evict */
}

static int uma_gemm_f32(wan_backend *b, const char *a, const char *bmat,
                        const char *y, int M, int N, int K) {
  /* Phase-1: host round-trip GEMM so Darwin lab can share the vtable without
   * GRAPH recipe work. Real GEMM_F16 path stays in wan_graph until incremental
   * adoption. */
  size_t abytes = (size_t)M * (size_t)K * sizeof(float);
  size_t bbytes = (size_t)K * (size_t)N * sizeof(float);
  size_t ybytes = (size_t)M * (size_t)N * sizeof(float);
  float *A = malloc(abytes);
  float *B = malloc(bbytes);
  float *Y = malloc(ybytes);
  if (!A || !B || !Y) {
    free(A);
    free(B);
    free(Y);
    return -1;
  }
  int rc = -1;
  if (uma_buf_get_vt(b, a, A, abytes) != 0 ||
      uma_buf_get_vt(b, bmat, B, bbytes) != 0)
    goto done;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
      Y[i * N + j] = s;
    }
  if (uma_buf_put_vt(b, y, Y, ybytes) != 0) goto done;
  rc = 0;
done:
  free(A);
  free(B);
  free(Y);
  return rc;
}

static int uma_unimplemented(wan_backend *b, ...) {
  (void)b;
  fprintf(stderr, "wan_backend_uma: elementwise op not wired — use CUDA/host\n");
  return -1;
}

static int uma_layernorm(wan_backend *b, const char *x, const char *y,
                         const char *w, int N, int D) {
  (void)x;
  (void)y;
  (void)w;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}
static int uma_affine(wan_backend *b, const char *x, const char *y,
                      const char *scale, const char *shift, int N, int D) {
  (void)x;
  (void)y;
  (void)scale;
  (void)shift;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}
static int uma_gelu(wan_backend *b, const char *x, const char *y, size_t n) {
  (void)x;
  (void)y;
  (void)n;
  return uma_unimplemented(b);
}
static int uma_gated(wan_backend *b, const char *x, const char *delta,
                     const char *gate, const char *y, int N, int D) {
  (void)x;
  (void)delta;
  (void)gate;
  (void)y;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}

static int uma_rmsnorm(wan_backend *b, const char *x, const char *y,
                       const char *w, int N, int D) {
  (void)x;
  (void)y;
  (void)w;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}
static int uma_bias(wan_backend *b, const char *x, const char *y,
                    const char *bias, int N, int D) {
  (void)x;
  (void)y;
  (void)bias;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}
static int uma_scale_bias(wan_backend *b, const char *x, const char *y,
                          const char *scale, const char *bias, int N, int D) {
  (void)x;
  (void)y;
  (void)scale;
  (void)bias;
  (void)N;
  (void)D;
  return uma_unimplemented(b);
}

static int uma_head_rms(wan_backend *b, const char *x, const char *y,
                        const char *w, int T, int H, int HD) {
  (void)x;
  (void)y;
  (void)w;
  (void)T;
  (void)H;
  (void)HD;
  return uma_unimplemented(b);
}
static int uma_rope3(wan_backend *b, const char *x, const char *y, int T, int H,
                     int HD, int gt, int gh, int gw) {
  (void)x;
  (void)y;
  (void)T;
  (void)H;
  (void)HD;
  (void)gt;
  (void)gh;
  (void)gw;
  return uma_unimplemented(b);
}
static int uma_attn(wan_backend *b, const char *q, const char *k, const char *v,
                    const char *out, int T, int Tk, int H, int HD) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)T;
  (void)Tk;
  (void)H;
  (void)HD;
  return uma_unimplemented(b);
}

static int uma_sync(wan_backend *b) {
  (void)b;
  return 0;
}

static size_t uma_device_bytes(wan_backend *b) {
  uma_impl *im = (uma_impl *)b->impl;
  return im ? im->peak_bytes : 0;
}

static const wan_backend_vtable uma_vt = {
    .name = "uma",
    .destroy = uma_destroy,
    .buf_alloc = uma_buf_alloc,
    .buf_free = uma_buf_free_vt,
    .buf_put = uma_buf_put_vt,
    .buf_get = uma_buf_get_vt,
    .bank_put = uma_bank_put_vt,
    .bank_bind = uma_bank_bind_vt,
    .bank_evict = uma_bank_evict_vt,
    .gemm_f32 = uma_gemm_f32,
    .layernorm = uma_layernorm,
    .affine_mul_add = uma_affine,
    .gelu_tanh = uma_gelu,
    .gated_residual = uma_gated,
    .rmsnorm = uma_rmsnorm,
    .bias_add = uma_bias,
    .scale_bias = uma_scale_bias,
    .head_rmsnorm = uma_head_rms,
    .rope3 = uma_rope3,
    .attn_sdpa = uma_attn,
    .sync = uma_sync,
    .device_bytes = uma_device_bytes,
};

wan_backend *wan_backend_uma_create(const char *sock) {
  uma_impl *im = calloc(1, sizeof(*im));
  wan_backend *b = calloc(1, sizeof(*b));
  if (!im || !b) {
    free(im);
    free(b);
    return NULL;
  }
  im->uma = uma_client_connect(sock && sock[0] ? sock : NULL);
  if (!im->uma) {
    free(im);
    free(b);
    return NULL;
  }
  im->pool = uma_buf_pool_create(im->uma);
  if (!im->pool) {
    uma_client_close(im->uma);
    free(im);
    free(b);
    return NULL;
  }
  b->vt = &uma_vt;
  b->impl = im;
  return b;
}
#endif /* __APPLE__ */
