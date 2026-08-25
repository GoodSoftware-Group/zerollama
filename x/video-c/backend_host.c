/* Host (malloc) wan_backend — unlocks Linux tests without CUDA; slow GEMM. */
#include "backend_ops.h"
#include "wan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HOST_CAP = 256 };

typedef struct {
  char name[96];
  void *ptr;
  size_t bytes;
  int used;
  int is_alias;
} host_slot;

typedef struct {
  host_slot slots[HOST_CAP];
  size_t live_bytes;
  size_t peak_bytes;
} host_impl;

static host_slot *find_slot(host_impl *im, const char *name) {
  for (int i = 0; i < HOST_CAP; i++)
    if (im->slots[i].used && strcmp(im->slots[i].name, name) == 0)
      return &im->slots[i];
  return NULL;
}

static host_slot *alloc_slot(host_impl *im, const char *name) {
  host_slot *s = find_slot(im, name);
  if (s) return s;
  for (int i = 0; i < HOST_CAP; i++) {
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

static void clear_aliases_to(host_impl *im, void *ptr) {
  if (!ptr) return;
  for (int i = 0; i < HOST_CAP; i++) {
    if (im->slots[i].used && im->slots[i].is_alias && im->slots[i].ptr == ptr)
      memset(&im->slots[i], 0, sizeof(im->slots[i]));
  }
}

static int free_owned(host_impl *im, host_slot *s) {
  if (!s || !s->used) return 0;
  if (s->is_alias) {
    memset(s, 0, sizeof(*s));
    return 0;
  }
  void *ptr = s->ptr;
  size_t bytes = s->bytes;
  free(ptr);
  if (bytes) im->live_bytes -= bytes;
  clear_aliases_to(im, ptr);
  memset(s, 0, sizeof(*s));
  return 0;
}

static void host_destroy(wan_backend *b) {
  if (!b) return;
  host_impl *im = (host_impl *)b->impl;
  if (im) {
    for (int i = 0; i < HOST_CAP; i++) {
      if (!im->slots[i].used) continue;
      if (im->slots[i].is_alias) {
        memset(&im->slots[i], 0, sizeof(im->slots[i]));
        continue;
      }
      free(im->slots[i].ptr);
    }
    free(im);
  }
  free(b);
}

static int host_buf_alloc(wan_backend *b, const char *name, size_t bytes) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *s = alloc_slot(im, name);
  if (!s) return -1;
  if (s->is_alias) {
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->used = 1;
  }
  if (s->ptr && !s->is_alias && s->bytes == bytes) return 0;
  if (s->ptr && !s->is_alias) {
    void *old = s->ptr;
    free(old);
    im->live_bytes -= s->bytes;
    clear_aliases_to(im, old);
    s->ptr = NULL;
  }
  s->ptr = malloc(bytes);
  if (!s->ptr) return -1;
  s->bytes = bytes;
  s->is_alias = 0;
  im->live_bytes += bytes;
  if (im->live_bytes > im->peak_bytes) im->peak_bytes = im->live_bytes;
  return 0;
}

static int host_buf_free(wan_backend *b, const char *name) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *s = find_slot(im, name);
  if (!s) return 0;
  return free_owned(im, s);
}

static int host_buf_put(wan_backend *b, const char *name, const void *host,
                        size_t bytes) {
  if (host_buf_alloc(b, name, bytes) != 0) return -1;
  host_impl *im = (host_impl *)b->impl;
  host_slot *s = find_slot(im, name);
  if (!s || !s->ptr) return -1;
  memcpy(s->ptr, host, bytes);
  return 0;
}

static int host_buf_get(wan_backend *b, const char *name, void *host,
                        size_t bytes) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *s = find_slot(im, name);
  if (!s || !s->ptr || s->bytes < bytes) return -1;
  memcpy(host, s->ptr, bytes);
  return 0;
}

static int host_bank_put(wan_backend *b, const char *name, const void *host,
                         size_t bytes) {
  return host_buf_put(b, name, host, bytes);
}

static int host_bank_bind(wan_backend *b, const char *bank, const char *as_buf) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sb = find_slot(im, bank);
  if (!sb || !sb->ptr) return -1;
  host_slot *sa = find_slot(im, as_buf);
  if (sa && !sa->is_alias && sa->ptr && sa->ptr != sb->ptr) {
    free_owned(im, sa);
    sa = NULL;
  }
  if (!sa) {
    sa = alloc_slot(im, as_buf);
    if (!sa) return -1;
  }
  sa->ptr = sb->ptr;
  sa->bytes = sb->bytes;
  sa->is_alias = 1;
  return 0;
}

static int host_bank_evict(wan_backend *b, const char *name) {
  return host_buf_free(b, name);
}

static int host_gemm_f32(wan_backend *b, const char *a, const char *bname,
                         const char *y, int M, int N, int K) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sa = find_slot(im, a);
  host_slot *sb = find_slot(im, bname);
  size_t ybytes = (size_t)M * (size_t)N * sizeof(float);
  if (host_buf_alloc(b, y, ybytes) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sa || !sb || !sy) return -1;
  const float *A = (const float *)sa->ptr;
  const float *B = (const float *)sb->ptr;
  float *Y = (float *)sy->ptr;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
      Y[i * N + j] = sum;
    }
  }
  return 0;
}

static int host_layernorm(wan_backend *b, const char *x, const char *y,
                          const char *w, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  host_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy || !sx->ptr) return -1;
  wan_op_layernorm_f32((float *)sy->ptr, (const float *)sx->ptr,
                       sw ? (const float *)sw->ptr : NULL, N, D, 1e-6f);
  return 0;
}

static int host_affine(wan_backend *b, const char *x, const char *y,
                       const char *scale, const char *shift, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  host_slot *ss = find_slot(im, scale);
  host_slot *sh = find_slot(im, shift);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !sy || !ss || !sh) return -1;
  wan_op_affine_mul_add_f32((float *)sy->ptr, (const float *)sx->ptr,
                            (const float *)ss->ptr, (const float *)sh->ptr, N,
                            D);
  return 0;
}

static int host_gelu(wan_backend *b, const char *x, const char *y, size_t n) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  if (host_buf_alloc(b, y, n * sizeof(float)) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !sy) return -1;
  wan_op_gelu_tanh_f32((float *)sy->ptr, (const float *)sx->ptr, n);
  return 0;
}

static int host_gated_res(wan_backend *b, const char *x, const char *delta,
                          const char *gate, const char *y, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  host_slot *sd = find_slot(im, delta);
  host_slot *sg = find_slot(im, gate);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !sd || !sg || !sy) return -1;
  wan_op_gated_residual_f32((float *)sy->ptr, (const float *)sx->ptr,
                            (const float *)sd->ptr, (const float *)sg->ptr, N,
                            D);
  return 0;
}

static int host_rmsnorm(wan_backend *b, const char *x, const char *y,
                        const char *w, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  host_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy) return -1;
  wan_op_rmsnorm_f32((float *)sy->ptr, (const float *)sx->ptr,
                     sw ? (const float *)sw->ptr : NULL, N, D, 1e-6f);
  return 0;
}

static int host_bias_add(wan_backend *b, const char *x, const char *y,
                         const char *bias, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  host_slot *sb = find_slot(im, bias);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !sb || !sy) return -1;
  wan_op_bias_add_f32((float *)sy->ptr, (const float *)sx->ptr,
                      (const float *)sb->ptr, N, D);
  return 0;
}

static int host_scale_bias(wan_backend *b, const char *x, const char *y,
                           const char *scale, const char *bias, int N, int D) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  host_slot *ss = find_slot(im, scale);
  host_slot *sb = find_slot(im, bias);
  size_t yb = (size_t)N * (size_t)D * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !ss || !sb || !sy) return -1;
  wan_op_scale_bias_f32((float *)sy->ptr, (const float *)sx->ptr,
                        (const float *)ss->ptr, (const float *)sb->ptr, N, D);
  return 0;
}

static int host_head_rms(wan_backend *b, const char *x, const char *y,
                         const char *w, int T, int H, int HD) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  size_t yb = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  host_slot *sw = (w && w[0]) ? find_slot(im, w) : NULL;
  if (!sx || !sy) return -1;
  wan_op_head_rmsnorm_f32((float *)sy->ptr, (const float *)sx->ptr,
                          sw ? (const float *)sw->ptr : NULL, T, H, HD, 1e-6f);
  return 0;
}

static int host_rope3(wan_backend *b, const char *x, const char *y, int T, int H,
                      int HD, int gt, int gh, int gw) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sx = find_slot(im, x);
  size_t yb = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  if (host_buf_alloc(b, y, yb) != 0) return -1;
  host_slot *sy = find_slot(im, y);
  if (!sx || !sy) return -1;
  return wan_op_rope3_f32((float *)sy->ptr, (const float *)sx->ptr, T, H, HD, gt,
                          gh, gw);
}

static int host_attn(wan_backend *b, const char *q, const char *k, const char *v,
                     const char *out, int T, int Tk, int H, int HD) {
  host_impl *im = (host_impl *)b->impl;
  host_slot *sq = find_slot(im, q);
  host_slot *sk = find_slot(im, k);
  host_slot *sv = find_slot(im, v);
  size_t ob = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  if (host_buf_alloc(b, out, ob) != 0) return -1;
  host_slot *so = find_slot(im, out);
  if (!sq || !sk || !sv || !so) return -1;
  wan_op_attn_sdpa_f32((float *)so->ptr, (const float *)sq->ptr,
                       (const float *)sk->ptr, (const float *)sv->ptr, T, Tk, H,
                       HD);
  return 0;
}

static int host_sync(wan_backend *b) {
  (void)b;
  return 0;
}

static size_t host_device_bytes(wan_backend *b) {
  host_impl *im = (host_impl *)b->impl;
  return im ? im->peak_bytes : 0;
}

static const wan_backend_vtable host_vt = {
    .name = "host",
    .destroy = host_destroy,
    .buf_alloc = host_buf_alloc,
    .buf_free = host_buf_free,
    .buf_put = host_buf_put,
    .buf_get = host_buf_get,
    .bank_put = host_bank_put,
    .bank_bind = host_bank_bind,
    .bank_evict = host_bank_evict,
    .gemm_f32 = host_gemm_f32,
    .layernorm = host_layernorm,
    .affine_mul_add = host_affine,
    .gelu_tanh = host_gelu,
    .gated_residual = host_gated_res,
    .rmsnorm = host_rmsnorm,
    .bias_add = host_bias_add,
    .scale_bias = host_scale_bias,
    .head_rmsnorm = host_head_rms,
    .rope3 = host_rope3,
    .attn_sdpa = host_attn,
    .sync = host_sync,
    .device_bytes = host_device_bytes,
};

wan_backend *wan_backend_host_create(void) {
  host_impl *im = calloc(1, sizeof(*im));
  wan_backend *b = calloc(1, sizeof(*b));
  if (!im || !b) {
    free(im);
    free(b);
    return NULL;
  }
  b->vt = &host_vt;
  b->impl = im;
  return b;
}
