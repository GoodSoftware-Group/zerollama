#include "wan_internal.h"
#include "wan_lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *strip_prefix(const char *name, const char *prefix) {
  size_t n = strlen(prefix);
  if (strncmp(name, prefix, n) == 0)
    return name + n;
  return NULL;
}

int wan_ctx_set_lora(wan_ctx *ctx, const char *path, float scale) {
  if (!ctx || !path || !path[0])
    return -1;
  if (ctx->lora) {
    fprintf(stderr, "wan-c: lora already set — one adapter per run\n");
    return -1;
  }
  ctx->lora = wan_lora_open(path);
  if (!ctx->lora)
    return -1;
  ctx->lora_scale = scale;
  return 0;
}

float *wan_load_tensor_f32(wan_ctx *ctx, const char *name, size_t *nelems_out) {
  if (nelems_out)
    *nelems_out = 0;
  if (!ctx || !name)
    return NULL;
  const char *raw_key = strip_prefix(name, "dit.");
  if (!raw_key)
    raw_key = name;

  /* Prefer on-disk safetensors for dit.* (no GGUF copy). */
  if (ctx->st) {
    const st_tensor_t *t =
        st_find_tensor(ctx->st, raw_key);
    if (t) {
      size_t n = st_tensor_nelems(t);
      float *buf = calloc(n, sizeof(float));
      if (!buf)
        return NULL;
      if (st_tensor_to_f32(ctx->st, t, buf, n) != 0) {
        free(buf);
        return NULL;
      }
      if (ctx->lora &&
          wan_lora_apply((const wan_lora *)ctx->lora, raw_key, buf, n,
                         ctx->lora_scale) < 0)
        fprintf(stderr, "wan-c: lora apply failed on %s (using base)\n", name);
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
      if (ctx->lora &&
          wan_lora_apply((const wan_lora *)ctx->lora, raw_key, buf, n,
                         ctx->lora_scale) < 0)
        fprintf(stderr, "wan-c: lora apply failed on %s (using base)\n", name);
      if (nelems_out)
        *nelems_out = n;
      return buf;
    }
  }
  return NULL;
}

typedef struct wan_wcache_ent {
  char name[160];
  float *data;
  size_t n;
  struct wan_wcache_ent *next;
} wan_wcache_ent;

typedef struct wan_wcache_hdr {
  wan_wcache_ent *head;
  size_t bytes;
  size_t hits;
  size_t misses;
  int nent;
} wan_wcache_hdr;

const float *wan_borrow_tensor_f32(wan_ctx *ctx, const char *name,
                                   size_t *nelems_out) {
  if (nelems_out)
    *nelems_out = 0;
  if (!ctx || !name)
    return NULL;
  wan_wcache_hdr *h = (wan_wcache_hdr *)ctx->weight_cache;
  if (!h) {
    h = calloc(1, sizeof(*h));
    if (!h)
      return NULL;
    ctx->weight_cache = h;
    fprintf(stderr, "wan-c: host weight borrow-cache enabled\n");
  }
  for (wan_wcache_ent *e = h->head; e; e = e->next) {
    if (strcmp(e->name, name) == 0) {
      h->hits++;
      if (nelems_out)
        *nelems_out = e->n;
      return e->data;
    }
  }
  size_t n = 0;
  float *buf = wan_load_tensor_f32(ctx, name, &n);
  if (!buf)
    return NULL;
  wan_wcache_ent *ent = calloc(1, sizeof(*ent));
  if (!ent) {
    free(buf);
    return NULL;
  }
  snprintf(ent->name, sizeof(ent->name), "%s", name);
  ent->data = buf;
  ent->n = n;
  ent->next = h->head;
  h->head = ent;
  h->bytes += n * sizeof(float);
  h->misses++;
  h->nent++;
  if (h->misses == 1 || (h->misses % 60) == 0)
    fprintf(stderr,
            "wan-c: weight-cache miss=%zu hit=%zu ents=%d ~%.1f MiB\n",
            h->misses, h->hits, h->nent, (double)h->bytes / (1024.0 * 1024.0));
  if (nelems_out)
    *nelems_out = n;
  return buf;
}

void wan_weight_cache_clear(wan_ctx *ctx) {
  if (!ctx || !ctx->weight_cache)
    return;
  wan_wcache_hdr *h = (wan_wcache_hdr *)ctx->weight_cache;
  wan_wcache_ent *e = h->head;
  while (e) {
    wan_wcache_ent *n = e->next;
    free(e->data);
    free(e);
    e = n;
  }
  if (h->misses || h->hits)
    fprintf(stderr,
            "wan-c: weight-cache final miss=%zu hit=%zu ents=%d ~%.1f MiB\n",
            h->misses, h->hits, h->nent, (double)h->bytes / (1024.0 * 1024.0));
  free(h);
  ctx->weight_cache = NULL;
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
