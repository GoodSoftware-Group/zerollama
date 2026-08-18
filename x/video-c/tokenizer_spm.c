#include "tokenizer_spm.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WAN_VOCAB_MAGIC 0x564f4357u /* "WCOV" */

typedef struct {
  char *bytes;
  int len;
  float score;
} spm_piece_t;

struct tokenizer_spm {
  spm_piece_t *pieces;
  int n_pieces;
  int unk_id;
  int bos_id;
  int eos_id;
};

static int read_varint(const uint8_t *buf, size_t n, size_t *i, uint64_t *out) {
  uint64_t x = 0;
  int shift = 0;
  while (*i < n && shift < 64) {
    uint8_t b = buf[(*i)++];
    x |= (uint64_t)(b & 0x7fu) << shift;
    if ((b & 0x80u) == 0) {
      *out = x;
      return 0;
    }
    shift += 7;
  }
  return -1;
}

static void apply_special_ids(tokenizer_spm *tok) {
  tok->unk_id = 0;
  tok->bos_id = 0;
  tok->eos_id = 1;
  for (int i = 0; i < tok->n_pieces; i++) {
    const char *s = tok->pieces[i].bytes;
    if (!s)
      continue;
    if (strcmp(s, "<unk>") == 0)
      tok->unk_id = i;
    if (strcmp(s, "<s>") == 0)
      tok->bos_id = i;
    if (strcmp(s, "</s>") == 0)
      tok->eos_id = i;
  }
}

static int load_binary_vocab(tokenizer_spm *tok, const uint8_t *data, size_t n) {
  if (n < 8)
    return -1;
  uint32_t magic = 0, np = 0;
  memcpy(&magic, data, 4);
  memcpy(&np, data + 4, 4);
  if (magic != WAN_VOCAB_MAGIC || np == 0 || np > 500000)
    return -1;
  size_t off = 8;
  tok->n_pieces = (int)np;
  tok->pieces = calloc(np, sizeof(spm_piece_t));
  if (!tok->pieces)
    return -1;
  for (uint32_t i = 0; i < np; i++) {
    if (off + 8 > n)
      return -1;
    spm_piece_t *p = &tok->pieces[i];
    memcpy(&p->score, data + off, 4);
    off += 4;
    uint32_t len = 0;
    memcpy(&len, data + off, 4);
    off += 4;
    if (len > 4096 || off + len > n)
      return -1;
    p->len = (int)len;
    p->bytes = malloc(len + 1);
    if (!p->bytes)
      return -1;
    memcpy(p->bytes, data + off, len);
    p->bytes[len] = 0;
    off += len;
  }
  apply_special_ids(tok);
  return 0;
}

/* Minimal ModelProto: repeated SentencePiece pieces = 1 { piece=1, score=2 }. */
static int load_sentencepiece_model(tokenizer_spm *tok, const uint8_t *data,
                                    size_t n) {
  /* Count pieces (field 1 length-delimited). */
  size_t i = 0;
  int count = 0;
  while (i < n) {
    uint64_t tag = 0;
    if (read_varint(data, n, &i, &tag) != 0)
      return -1;
    int field = (int)(tag >> 3);
    int wt = (int)(tag & 7);
    if (wt == 0) {
      uint64_t v = 0;
      if (read_varint(data, n, &i, &v) != 0)
        return -1;
    } else if (wt == 1) {
      if (i + 8 > n)
        return -1;
      i += 8;
    } else if (wt == 5) {
      if (i + 4 > n)
        return -1;
      i += 4;
    } else if (wt == 2) {
      uint64_t ln = 0;
      if (read_varint(data, n, &i, &ln) != 0 || i + ln > n)
        return -1;
      if (field == 1)
        count++;
      i += (size_t)ln;
    } else {
      return -1;
    }
  }
  if (count < 1 || count > 500000)
    return -1;

  tok->n_pieces = count;
  tok->pieces = calloc((size_t)count, sizeof(spm_piece_t));
  if (!tok->pieces)
    return -1;

  i = 0;
  int pi = 0;
  while (i < n && pi < count) {
    uint64_t tag = 0;
    if (read_varint(data, n, &i, &tag) != 0)
      return -1;
    int field = (int)(tag >> 3);
    int wt = (int)(tag & 7);
    if (wt == 0) {
      uint64_t v = 0;
      if (read_varint(data, n, &i, &v) != 0)
        return -1;
      continue;
    }
    if (wt == 1) {
      if (i + 8 > n)
        return -1;
      i += 8;
      continue;
    }
    if (wt == 5) {
      if (i + 4 > n)
        return -1;
      i += 4;
      continue;
    }
    if (wt != 2)
      return -1;
    uint64_t ln = 0;
    if (read_varint(data, n, &i, &ln) != 0 || i + ln > n)
      return -1;
    const uint8_t *msg = data + i;
    size_t msg_n = (size_t)ln;
    i += msg_n;
    if (field != 1)
      continue;

    spm_piece_t *p = &tok->pieces[pi];
    p->score = 0.0f;
    size_t j = 0;
    while (j < msg_n) {
      uint64_t t2 = 0;
      if (read_varint(msg, msg_n, &j, &t2) != 0)
        return -1;
      int f2 = (int)(t2 >> 3);
      int w2 = (int)(t2 & 7);
      if (w2 == 2) {
        uint64_t l2 = 0;
        if (read_varint(msg, msg_n, &j, &l2) != 0 || j + l2 > msg_n)
          return -1;
        if (f2 == 1) {
          if (l2 > 4096)
            return -1;
          p->len = (int)l2;
          p->bytes = malloc((size_t)l2 + 1);
          if (!p->bytes)
            return -1;
          memcpy(p->bytes, msg + j, (size_t)l2);
          p->bytes[l2] = 0;
        }
        j += (size_t)l2;
      } else if (w2 == 5) {
        if (j + 4 > msg_n)
          return -1;
        if (f2 == 2)
          memcpy(&p->score, msg + j, 4);
        j += 4;
      } else if (w2 == 0) {
        uint64_t v = 0;
        if (read_varint(msg, msg_n, &j, &v) != 0)
          return -1;
      } else if (w2 == 1) {
        if (j + 8 > msg_n)
          return -1;
        j += 8;
      } else {
        return -1;
      }
    }
    if (!p->bytes) {
      p->bytes = calloc(1, 1);
      p->len = 0;
    }
    pi++;
  }
  if (pi != count)
    return -1;
  apply_special_ids(tok);
  return 0;
}

tokenizer_spm *tokenizer_spm_load(const char *path) {
  if (!path)
    return NULL;
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long sz = ftell(f);
  if (sz < 8 || sz > 64L * 1024L * 1024L) {
    fclose(f);
    return NULL;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *data = malloc((size_t)sz);
  if (!data) {
    fclose(f);
    return NULL;
  }
  if (fread(data, 1, (size_t)sz, f) != (size_t)sz) {
    free(data);
    fclose(f);
    return NULL;
  }
  fclose(f);

  tokenizer_spm *tok = calloc(1, sizeof(*tok));
  if (!tok) {
    free(data);
    return NULL;
  }

  uint32_t magic = 0;
  memcpy(&magic, data, 4);
  int rc;
  if (magic == WAN_VOCAB_MAGIC)
    rc = load_binary_vocab(tok, data, (size_t)sz);
  else
    rc = load_sentencepiece_model(tok, data, (size_t)sz);

  free(data);
  if (rc != 0) {
    tokenizer_spm_free(tok);
    fprintf(stderr,
            "wan-c: tokenizer failed to load %s (need .vocab or SPM .model)\n",
            path);
    return NULL;
  }
  return tok;
}

void tokenizer_spm_free(tokenizer_spm *tok) {
  if (!tok)
    return;
  for (int i = 0; i < tok->n_pieces; i++)
    free(tok->pieces[i].bytes);
  free(tok->pieces);
  free(tok);
}

int tokenizer_spm_vocab_size(const tokenizer_spm *tok) {
  return tok ? tok->n_pieces : 0;
}

/* Unigram Viterbi: maximize sum of piece scores over UTF-8 byte spans.
 * umt5 / SentencePiece: map ASCII space → U+2581 (▁), add dummy ▁ prefix
 * when vocab uses ▁ (add_dummy_prefix). Emit eos only (no bos) to rematch
 * HF AutoTokenizer(umt5-xxl, add_special_tokens=True). */
int tokenizer_spm_encode(tokenizer_spm *tok, const char *text, int32_t *ids,
                         size_t cap, size_t *n_out) {
  if (!tok || !text || !ids || !n_out || cap < 1)
    return -1;

  static const char UBLK[3] = {(char)0xe2, (char)0x96, (char)0x81}; /* ▁ */
  int use_ublk = 0;
  for (int p = 0; p < tok->n_pieces; p++) {
    if (tok->pieces[p].len >= 3 &&
        memcmp(tok->pieces[p].bytes, UBLK, 3) == 0) {
      use_ublk = 1;
      break;
    }
  }

  size_t raw_n = strlen(text);
  if (raw_n > 4096)
    raw_n = 4096;

  /* Worst case: every byte → ▁ + byte (3+1), plus leading ▁. */
  char norm[4096 * 4 + 8];
  size_t L = 0;
  if (use_ublk) {
    if (raw_n == 0) {
      /* empty → eos only below */
      L = 0;
    } else {
      int need_prefix = 1;
      if (raw_n >= 3 && memcmp(text, UBLK, 3) == 0)
        need_prefix = 0;
      if (need_prefix) {
        memcpy(norm + L, UBLK, 3);
        L += 3;
      }
      for (size_t i = 0; i < raw_n; i++) {
        if (text[i] == ' ') {
          if (L + 3 > sizeof(norm))
            break;
          memcpy(norm + L, UBLK, 3);
          L += 3;
        } else {
          if (L + 1 > sizeof(norm))
            break;
          norm[L++] = text[i];
        }
      }
    }
  } else {
    if (raw_n > sizeof(norm))
      raw_n = sizeof(norm);
    memcpy(norm, text, raw_n);
    L = raw_n;
  }
  norm[L] = 0;

  if (L == 0) {
    ids[0] = (int32_t)tok->eos_id;
    *n_out = 1;
    return 0;
  }

  float *best = malloc((L + 1) * sizeof(float));
  int *back_len = malloc((L + 1) * sizeof(int));
  int *back_id = malloc((L + 1) * sizeof(int));
  if (!best || !back_len || !back_id) {
    free(best);
    free(back_len);
    free(back_id);
    return -1;
  }
  for (size_t i = 0; i <= L; i++) {
    best[i] = -FLT_MAX;
    back_len[i] = -1;
    back_id[i] = tok->unk_id;
  }
  best[0] = 0.0f;

  for (size_t i = 0; i < L; i++) {
    if (best[i] <= -FLT_MAX / 2)
      continue;
    int matched = 0;
    for (int p = 0; p < tok->n_pieces; p++) {
      int plen = tok->pieces[p].len;
      if (plen <= 0 || i + (size_t)plen > L)
        continue;
      if (memcmp(norm + i, tok->pieces[p].bytes, (size_t)plen) != 0)
        continue;
      float cand = best[i] + tok->pieces[p].score;
      size_t j = i + (size_t)plen;
      if (cand > best[j]) {
        best[j] = cand;
        back_len[j] = plen;
        back_id[j] = p;
      }
      matched = 1;
    }
    if (!matched && i + 1 <= L) {
      float cand = best[i] - 10.0f;
      if (cand > best[i + 1]) {
        best[i + 1] = cand;
        back_len[i + 1] = 1;
        back_id[i + 1] = tok->unk_id;
      }
    }
  }

  int tmp[4096];
  int nt = 0;
  size_t pos = L;
  while (pos > 0 && nt < 4096) {
    int plen = back_len[pos];
    if (plen <= 0)
      break;
    tmp[nt++] = back_id[pos];
    pos -= (size_t)plen;
  }

  size_t n = 0;
  for (int i = nt - 1; i >= 0 && n + 1 < cap; i--)
    ids[n++] = (int32_t)tmp[i];
  if (n < cap)
    ids[n++] = (int32_t)tok->eos_id;
  *n_out = n;

  free(best);
  free(back_len);
  free(back_id);
  return 0;
}
