/* Thin wrap of sibling BMTL C tokenizer (Gigatoken lessons). Encode-only.
 * This bmtl tree declares bmtl_tokenizer_load_bmtl but does not define it;
 * we load BMTLTK01 here and encode with byte_to_id + NFC (stock encode
 * currently passes NULL byte_to_id). */
#include "h3_tokenizer.h"

#include "bmtl/bpe.h"
#include "bmtl/nfc.h"
#include "bmtl/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FLAG_BYTE_MAP 1u
#define FLAG_SPECIALS 2u
#define FLAG_NFC 4u

struct h3_tokenizer {
  bmtl_tokenizer tok;
  int nfc;
};

static int ends_with(const char *s, const char *suf) {
  size_t n = strlen(s), m = strlen(suf);
  return n >= m && memcmp(s + n - m, suf, m) == 0;
}

static int set_error(char *error, size_t error_size, const char *msg) {
  if (error && error_size)
    snprintf(error, error_size, "%s", msg);
  return 0;
}

static const char *resolve_blob(const char *path, char *buf, size_t buf_size) {
  if (path && ends_with(path, ".bmtl_tok") && access(path, R_OK) == 0)
    return path;
  const char *env = getenv("H3_BMTL_TOK");
  if (env && env[0] && access(env, R_OK) == 0)
    return env;
  const char *home = getenv("HOME");
  if (home) {
    snprintf(buf, buf_size, "%s/.zerollama/third_party/h3/minimax_h3.bmtl_tok",
             home);
    if (access(buf, R_OK) == 0)
      return buf;
  }
  return NULL;
}

static int read_u32(FILE *f, uint32_t *out) {
  unsigned char b[4];
  if (fread(b, 1, 4, f) != 4)
    return 0;
  *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
  return 1;
}

static bmtl_status load_bmtl_blob(bmtl_tokenizer *tok, int *nfc_out,
                                  const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return BMTL_ERROR_IO;
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "BMTLTK01", 8) != 0) {
    fclose(f);
    return BMTL_ERROR_FORMAT;
  }
  uint32_t version, vocab_size, n_merges, scheme, flags, n_specials;
  if (!read_u32(f, &version) || !read_u32(f, &vocab_size) ||
      !read_u32(f, &n_merges) || !read_u32(f, &scheme) || !read_u32(f, &flags) ||
      !read_u32(f, &n_specials) || version != 1) {
    fclose(f);
    return BMTL_ERROR_FORMAT;
  }
  bmtl_tokenizer_init(tok);
  tok->vocab_size = vocab_size;
  tok->type = (char *)malloc(4);
  if (!tok->type) {
    fclose(f);
    return BMTL_ERROR_OUT_OF_MEMORY;
  }
  memcpy(tok->type, "BPE", 4);
  bmtl_pretokenizer_init(&tok->pretokenizer, (bmtl_pretoken_scheme)scheme);
  bmtl_status s = bmtl_pair_rank_table_init(&tok->prt, BMTL_TOK_SPARSE_CAP);
  if (s != BMTL_OK) {
    fclose(f);
    bmtl_tokenizer_destroy(tok);
    return s;
  }
  tok->prt_owned = 1;
  s = bmtl_pretoken_cache_init(&tok->cache, BMTL_TOK_CACHE_CAP);
  if (s != BMTL_OK) {
    fclose(f);
    bmtl_tokenizer_destroy(tok);
    return s;
  }
  if (flags & FLAG_BYTE_MAP) {
    if (fread(tok->byte_to_id, sizeof(uint32_t), 256, f) != 256) {
      fclose(f);
      bmtl_tokenizer_destroy(tok);
      return BMTL_ERROR_FORMAT;
    }
    tok->has_byte_map = 1;
  }
  for (uint32_t i = 0; i < n_merges; i++) {
    uint32_t a, b, m;
    if (!read_u32(f, &a) || !read_u32(f, &b) || !read_u32(f, &m)) {
      fclose(f);
      bmtl_tokenizer_destroy(tok);
      return BMTL_ERROR_FORMAT;
    }
    s = bmtl_pair_rank_table_insert(&tok->prt, a, b, m);
    if (s != BMTL_OK) {
      fclose(f);
      bmtl_tokenizer_destroy(tok);
      return s;
    }
  }
  if (flags & FLAG_SPECIALS) {
    for (uint32_t i = 0; i < n_specials; i++) {
      uint32_t id, blen, sflags;
      if (!read_u32(f, &id) || !read_u32(f, &blen) || !read_u32(f, &sflags) ||
          blen > 4096) {
        fclose(f);
        bmtl_tokenizer_destroy(tok);
        return BMTL_ERROR_FORMAT;
      }
      uint8_t *bytes = (uint8_t *)malloc(blen ? blen : 1);
      if (!bytes || (blen && fread(bytes, 1, blen, f) != blen)) {
        free(bytes);
        fclose(f);
        bmtl_tokenizer_destroy(tok);
        return BMTL_ERROR_FORMAT;
      }
      s = bmtl_pretokenizer_add_special_ex(&tok->pretokenizer, bytes, blen,
                                           (int32_t)id, (sflags & 1) ? 1 : 0);
      free(bytes);
      if (s != BMTL_OK) {
        fclose(f);
        bmtl_tokenizer_destroy(tok);
        return s;
      }
    }
  }
  fclose(f);
  *nfc_out = (flags & FLAG_NFC) ? 1 : 0;
  return BMTL_OK;
}

static uint32_t encode_mapped(bmtl_tokenizer *tokenizer, int nfc,
                              const char *text, size_t text_len,
                              bmtl_token_id *ids, uint32_t ids_cap) {
  if (!tokenizer || !text || !ids || ids_cap == 0)
    return 0;
  const uint8_t *src = (const uint8_t *)text;
  size_t src_len = text_len;
  const uint8_t *nfc_ptr = src;
  size_t nfc_len = src_len;
  int owned = 0;
  if (bmtl_utf8_nfc_maybe(nfc, src, src_len, &nfc_ptr, &nfc_len, &owned) < 0)
    return 0;
  const uint32_t *bmap = tokenizer->has_byte_map ? tokenizer->byte_to_id : NULL;
  bmtl_pretokenizer *pre = &tokenizer->pretokenizer;
  bmtl_pretokenizer_reset(pre, nfc_ptr, nfc_len);
  pre->pack_on_push = 1;
  pre->pack_cache = &tokenizer->cache;
  uint32_t total = 0;
  while (total < ids_cap) {
    size_t pos0 = pre->pos;
    uint32_t n_batch = bmtl_pretokenizer_refill(pre);
    if (n_batch == 0)
      break;
    bmtl_pretokenizer_pack_batch(pre);
    for (uint32_t i = 0; i < n_batch && total < ids_cap; i++) {
      size_t start = pre->batch_starts[i];
      size_t end = pre->batch_ends[i];
      uint32_t byte_len = (uint32_t)(end - start);
      int32_t special = pre->batch_special_ids[i];
      if (special >= 0) {
        ids[total++] = special;
        continue;
      }
      if (byte_len == 0)
        continue;
      uint32_t n =
          bmtl_bpe_encode(&tokenizer->prt, nfc_ptr + start, byte_len, bmap,
                          ids + total, ids_cap - total);
      if (n == 0)
        break;
      total += n;
    }
    if (pre->pos <= pos0)
      break;
    if (pre->pos >= nfc_len)
      break;
  }
  if (owned)
    free((void *)nfc_ptr);
  return total;
}

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json, char *error,
                                size_t error_size) {
  char def[768];
  const char *blob = resolve_blob(tokenizer_json, def, sizeof(def));
  if (!blob) {
    set_error(error, error_size,
              "missing BMTL blob (export tokenizer.json --scheme qwen2 to "
              "$HOME/.zerollama/third_party/h3/minimax_h3.bmtl_tok or "
              "H3_BMTL_TOK)");
    return NULL;
  }
  h3_tokenizer *t = (h3_tokenizer *)calloc(1, sizeof(*t));
  if (!t) {
    set_error(error, error_size, "oom");
    return NULL;
  }
  bmtl_status st = load_bmtl_blob(&t->tok, &t->nfc, blob);
  if (st != BMTL_OK) {
    if (error && error_size)
      snprintf(error, error_size, "load BMTLTK01(%s): %s", blob,
               bmtl_status_string(st));
    free(t);
    return NULL;
  }
  return t;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
  if (!tokenizer)
    return;
  bmtl_tokenizer_destroy(&tokenizer->tok);
  free(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
  if (!ids || !count)
    return set_error(error, error_size, "null out");
  *ids = NULL;
  *count = 0;
  if (!tokenizer)
    return set_error(error, error_size, "null tokenizer");
  if (!utf8)
    utf8 = "";
  size_t len = strlen(utf8);
  if (len == 0) {
    if (!pad_empty)
      return 1;
    uint32_t *out = (uint32_t *)malloc(sizeof(uint32_t));
    if (!out)
      return set_error(error, error_size, "oom");
    out[0] = H3_PAD_TOKEN_ID;
    *ids = out;
    *count = 1;
    return 1;
  }
  uint32_t cap = (uint32_t)(len + 32);
  if (cap < 64)
    cap = 64;
  bmtl_token_id *tmp = NULL;
  uint32_t n = 0;
  for (;;) {
    bmtl_token_id *grow =
        (bmtl_token_id *)realloc(tmp, (size_t)cap * sizeof(bmtl_token_id));
    if (!grow) {
      free(tmp);
      return set_error(error, error_size, "oom");
    }
    tmp = grow;
    n = encode_mapped((bmtl_tokenizer *)&tokenizer->tok, tokenizer->nfc, utf8,
                      len, tmp, cap);
    if (n > 0 && n < cap)
      break;
    if (cap >= (1u << 20)) {
      free(tmp);
      return set_error(error, error_size, "encode overflow");
    }
    cap *= 2;
  }
  uint32_t *out = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
  if (!out) {
    free(tmp);
    return set_error(error, error_size, "oom");
  }
  for (uint32_t i = 0; i < n; i++)
    out[i] = (uint32_t)tmp[i];
  free(tmp);
  *ids = out;
  *count = n;
  return 1;
}

void h3_tokenizer_ids_free(uint32_t *ids) { free(ids); }

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer, const uint32_t *ids,
                          size_t count, char *error, size_t error_size) {
  (void)tokenizer;
  (void)ids;
  (void)count;
  set_error(error, error_size, "bmtl tokenizer is encode-only");
  return NULL;
}
