#include "wan_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *strip_prefix(const char *name, const char *prefix) {
  size_t n = strlen(prefix);
  if (strncmp(name, prefix, n) == 0)
    return name + n;
  return NULL;
}

float *wan_load_tensor_f32(wan_ctx *ctx, const char *name, size_t *nelems_out) {
  if (nelems_out)
    *nelems_out = 0;
  if (!ctx || !name)
    return NULL;

  /* Prefer on-disk safetensors for dit.* (no GGUF copy). */
  if (ctx->st) {
    const char *raw = strip_prefix(name, "dit.");
    const st_tensor_t *t =
        st_find_tensor(ctx->st, raw ? raw : name);
    if (t) {
      size_t n = st_tensor_nelems(t);
      float *buf = calloc(n, sizeof(float));
      if (!buf)
        return NULL;
      if (st_tensor_to_f32(ctx->st, t, buf, n) != 0) {
        free(buf);
        return NULL;
      }
      if (nelems_out)
        *nelems_out = n;
      return buf;
    }
  }

  /* T5 embed/norm from torch .pth via zip index. */
  if (ctx->t5_zip) {
    const zw_tensor_t *t = zw_find_tensor(ctx->t5_zip, name);
    if (t) {
      size_t n = zw_tensor_nelems(t);
      float *buf = calloc(n, sizeof(float));
      if (!buf)
        return NULL;
      if (zw_tensor_to_f32(ctx->t5_zip, t, buf, n) != 0) {
        free(buf);
        return NULL;
      }
      if (nelems_out)
        *nelems_out = n;
      return buf;
    }
  }

  /* VAE decoder / conv* from Wan2.1_VAE.pth. */
  if (ctx->vae_zip) {
    const zw_tensor_t *t = zw_find_tensor(ctx->vae_zip, name);
    if (t) {
      size_t n = zw_tensor_nelems(t);
      float *buf = calloc(n, sizeof(float));
      if (!buf)
        return NULL;
      if (zw_tensor_to_f32(ctx->vae_zip, t, buf, n) != 0) {
        free(buf);
        return NULL;
      }
      if (nelems_out)
        *nelems_out = n;
      return buf;
    }
  }

  if (ctx->gguf) {
    const gguf_tensor_t *t = gguf_find_tensor(ctx->gguf, name);
    if (t) {
      size_t n = gguf_tensor_nelems(t);
      float *buf = calloc(n, sizeof(float));
      if (!buf)
        return NULL;
      if (gguf_tensor_to_f32(ctx->gguf, t, buf, n) != 0) {
        free(buf);
        return NULL;
      }
      if (nelems_out)
        *nelems_out = n;
      return buf;
    }
  }
  return NULL;
}

int wan_gguf_has(wan_ctx *ctx, const char *name) {
  if (!ctx || !name)
    return 0;
  if (ctx->st) {
    const char *raw = strip_prefix(name, "dit.");
    if (st_find_tensor(ctx->st, raw ? raw : name))
      return 1;
  }
  if (ctx->t5_zip && zw_find_tensor(ctx->t5_zip, name))
    return 1;
  if (ctx->vae_zip && zw_find_tensor(ctx->vae_zip, name))
    return 1;
  if (ctx->gguf && gguf_find_tensor(ctx->gguf, name))
    return 1;
  return 0;
}

void wan_fill_eye_nt(float *w, int N, int K) {
  if (!w || N < 1 || K < 1)
    return;
  memset(w, 0, (size_t)N * (size_t)K * sizeof(float));
  int d = N < K ? N : K;
  for (int i = 0; i < d; i++)
    w[i * K + i] = 1.0f;
}

int wan_put_weight_or_eye(wan_ctx *ctx, const char *buf_name,
                          const char *bank_key, const char *gguf_name, int N,
                          int K) {
  if (!ctx || !ctx->bufs || !buf_name || N < 1 || K < 1)
    return -1;
  size_t need = (size_t)N * (size_t)K;
  size_t got = 0;
  float *w = NULL;
  int from_gguf = 0;
  if (gguf_name && gguf_name[0])
    w = wan_load_tensor_f32(ctx, gguf_name, &got);
  if (w && got == need) {
    from_gguf = 1;
  } else {
    free(w);
    w = calloc(need, sizeof(float));
    if (!w)
      return -1;
    wan_fill_eye_nt(w, N, K);
  }
  int rc = uma_buf_pool_put_weight(ctx->bufs, buf_name, bank_key, w,
                                   need * sizeof(float));
  free(w);
  if (rc != 0) {
    fprintf(stderr, "wan-c: put_weight failed name=%s key=%s %dx%d (%.1f MiB)\n",
            buf_name, bank_key ? bank_key : "-", N, K,
            (need * 4) / (1024.0 * 1024.0));
    return rc;
  }
  if (from_gguf) {
    const char *logw = getenv("WAN_WEIGHT_LOG");
    if (logw && (logw[0] == '1' || logw[0] == 'y'))
      fprintf(stderr, "wan-c: weight %s ← %s (%dx%d)\n", buf_name, gguf_name, N,
              K);
  }
  return 0;
}

int wan_put_weight_raw(wan_ctx *ctx, const char *buf_name, const char *bank_key,
                       const char *gguf_name, size_t expect_nelems) {
  if (!ctx || !ctx->bufs || !buf_name || !gguf_name)
    return -1;
  size_t got = 0;
  float *w = wan_load_tensor_f32(ctx, gguf_name, &got);
  if (!w || (expect_nelems > 0 && got != expect_nelems)) {
    free(w);
    return -1;
  }
  int rc = uma_buf_pool_put_weight(ctx->bufs, buf_name, bank_key, w,
                                   got * sizeof(float));
  free(w);
  if (rc == 0)
    fprintf(stderr, "wan-c: weight %s ← %s (%zu f32)\n", buf_name, gguf_name,
            got);
  return rc;
}
