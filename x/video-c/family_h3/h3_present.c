#include "h3_present.h"

#include "h3_adaln_host.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t *values;
  size_t count;
  size_t capacity;
} idbuf;

static int ids_reserve(idbuf *ids, size_t extra) {
  if (extra > SIZE_MAX - ids->count)
    return 0;
  size_t needed = ids->count + extra;
  if (needed <= ids->capacity)
    return 1;
  size_t cap = ids->capacity ? ids->capacity : 32;
  while (cap < needed) {
    if (cap > SIZE_MAX / 2) {
      cap = needed;
      break;
    }
    cap *= 2;
  }
  uint32_t *v = (uint32_t *)realloc(ids->values, cap * sizeof(*v));
  if (!v)
    return 0;
  ids->values = v;
  ids->capacity = cap;
  return 1;
}

static int ids_append(idbuf *ids, const uint32_t *values, size_t count) {
  if (!ids_reserve(ids, count))
    return 0;
  if (count)
    memcpy(ids->values + ids->count, values, count * sizeof(*values));
  ids->count += count;
  return 1;
}

static int ids_push(idbuf *ids, uint32_t value) {
  return ids_append(ids, &value, 1);
}

void h3_presentation_free(h3_presentation *p) {
  if (!p)
    return;
  free(p->ids);
  free(p->tags);
  free(p->pos);
  free(p->spans);
  memset(p, 0, sizeof(*p));
}

int h3_present_mrope(const h3_present_span *spans, size_t n_spans, size_t seq,
                     uint32_t *pos) {
  if (!pos || seq < 1)
    return -1;
  if (!n_spans) {
    for (size_t axis = 0; axis < 3; axis++)
      for (size_t i = 0; i < seq; i++)
        pos[axis * seq + i] = (uint32_t)i;
    return 0;
  }
  if (!spans)
    return -1;
  memset(pos, 0, 3 * seq * sizeof(uint32_t));
  int64_t offset = 0;
  for (size_t image = 0; image < n_spans; image++) {
    const h3_present_span *sp = &spans[image];
    size_t start = sp->start;
    size_t end = start + sp->tokens;
    if (!sp->tokens || start > seq || end > seq || sp->merged_h < 1 ||
        sp->merged_w < 1 ||
        (size_t)sp->merged_h * (size_t)sp->merged_w != sp->tokens)
      return -1;
    if (image == 0) {
      for (size_t axis = 0; axis < 3; axis++)
        for (size_t i = 0; i < start; i++)
          pos[axis * seq + i] = (uint32_t)i;
    }
    int64_t length_max =
        sp->merged_h > sp->merged_w ? sp->merged_h : sp->merged_w;
    int64_t base = (int64_t)start + offset;
    int64_t next = (int64_t)start + length_max + offset;
    if (base < 0 || next < 0)
      return -1;
    for (size_t i = start; i < end; i++)
      pos[i] = (uint32_t)base;
    size_t cursor = start;
    for (int row = 0; row < sp->merged_h; row++) {
      for (int col = 0; col < sp->merged_w; col++) {
        pos[seq + cursor] = (uint32_t)(base + row);
        pos[2 * seq + cursor] = (uint32_t)(base + col);
        cursor++;
      }
    }
    if (cursor != end)
      return -1;
    for (size_t axis = 0; axis < 3; axis++)
      for (size_t i = end; i < seq; i++)
        pos[axis * seq + i] = (uint32_t)(next + (int64_t)(i - end));
    offset += length_max - (int64_t)sp->tokens;
  }
  return 0;
}

static int tokenize_append(const h3_tokenizer *tokenizer, const char *text,
                           idbuf *ids, char *error, size_t error_size) {
  uint32_t *values = NULL;
  size_t count = 0;
  if (!h3_tokenizer_encode(tokenizer, text, 0, &values, &count, error,
                           error_size))
    return 0;
  int ok = ids_append(ids, values, count);
  h3_tokenizer_ids_free(values);
  if (!ok && error && error_size)
    snprintf(error, error_size, "oom appending tokens");
  return ok;
}

int h3_present_fl2va(const h3_tokenizer *tok, const char *prompt,
                     const int *merged_h, const int *merged_w, size_t n_images,
                     h3_presentation *out, char *error, size_t error_size) {
  if (out)
    memset(out, 0, sizeof(*out));
  if (!tok || !prompt || !out || (n_images && (!merged_h || !merged_w))) {
    if (error && error_size)
      snprintf(error, error_size, "invalid FL2VA presentation arguments");
    return 0;
  }
  idbuf ids = {0};
  h3_present_span *spans = NULL;
  if (n_images) {
    spans = (h3_present_span *)calloc(n_images, sizeof(*spans));
    if (!spans)
      goto oom;
  }
  for (size_t i = 0; i < n_images; i++) {
    if (merged_h[i] < 1 || merged_w[i] < 1)
      goto bad;
    char prefix[64];
    int n = snprintf(prefix, sizeof(prefix), "<Picture %zu>: ", i + 1);
    if (n < 0 || (size_t)n >= sizeof(prefix))
      goto bad;
    if (!tokenize_append(tok, prefix, &ids, error, error_size))
      goto fail;
    if (!ids_push(&ids, H3_VISION_START_ID))
      goto oom;
    spans[i].start = ids.count;
    spans[i].merged_h = merged_h[i];
    spans[i].merged_w = merged_w[i];
    spans[i].tokens = (size_t)merged_h[i] * (size_t)merged_w[i];
    if (!ids_reserve(&ids, spans[i].tokens + 1))
      goto oom;
    for (size_t t = 0; t < spans[i].tokens; t++)
      ids.values[ids.count++] = H3_IMAGE_PAD_ID;
    ids.values[ids.count++] = H3_VISION_END_ID;
  }
  if (!tokenize_append(tok, prompt, &ids, error, error_size))
    goto fail;
  if (!ids.count)
    goto bad;
  uint8_t *tags = (uint8_t *)malloc(ids.count);
  uint32_t *pos = (uint32_t *)calloc(3 * ids.count, sizeof(uint32_t));
  uint32_t *idcopy = (uint32_t *)malloc(ids.count * sizeof(uint32_t));
  if (!tags || !pos || !idcopy)
    goto oom_inner;
  memcpy(idcopy, ids.values, ids.count * sizeof(uint32_t));
  memset(tags, (uint8_t)H3_ADALN_TAG_TEXT, ids.count);
  for (size_t i = 0; i < n_images; i++) {
    size_t first = spans[i].start - 1;
    size_t count = spans[i].tokens + 2;
    if (first > ids.count || count > ids.count - first)
      goto bad_inner;
    memset(tags + first, (uint8_t)H3_ADALN_TAG_VIDEO, count);
  }
  if (h3_present_mrope(spans, n_images, ids.count, pos) != 0)
    goto bad_inner;
  out->ids = idcopy;
  out->tags = tags;
  out->pos = pos;
  out->count = ids.count;
  out->spans = spans;
  out->n_spans = n_images;
  free(ids.values);
  return 1;

bad_inner:
  free(idcopy);
  free(tags);
  free(pos);
bad:
  if (error && error_size)
    snprintf(error, error_size, "invalid FL2VA presentation");
  goto fail;
oom_inner:
  free(idcopy);
  free(tags);
  free(pos);
oom:
  if (error && error_size)
    snprintf(error, error_size, "oom constructing FL2VA presentation");
fail:
  free(ids.values);
  free(spans);
  return 0;
}
