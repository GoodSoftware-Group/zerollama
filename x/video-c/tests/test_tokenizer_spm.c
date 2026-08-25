#include "tokenizer_spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal ModelProto with two pieces: "a"(score0), "b"(score-1). */
static const unsigned char k_mini_model[] = {
    0x0a, 0x0a, 0x0a, 0x01, 'a', 0x15, 0x00, 0x00, 0x00, 0x00, 0x18, 0x01,
    0x0a, 0x08, 0x0a, 0x01, 'b', 0x15, 0x00, 0x00, 0x80, 0xbf,
};

static int write_tmp(const char *path, const void *data, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  if (fwrite(data, 1, n, f) != n) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

static int expect_ids(const char *tag, const int32_t *got, size_t n,
                      const int32_t *want, size_t wn) {
  if (n != wn) {
    fprintf(stderr, "FAIL %s n=%zu want=%zu ids:", tag, n, wn);
    for (size_t i = 0; i < n; i++)
      fprintf(stderr, " %d", got[i]);
    fprintf(stderr, "\n");
    return -1;
  }
  for (size_t i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      fprintf(stderr, "FAIL %s mismatch at %zu got=%d want=%d\n", tag, i,
              got[i], want[i]);
      return -1;
    }
  }
  printf("PASS: %s\n", tag);
  return 0;
}

int main(void) {
  const char *path = "tests/tmp_mini_spm.model";
  if (write_tmp(path, k_mini_model, sizeof(k_mini_model)) != 0) {
    fprintf(stderr, "write tmp model failed\n");
    return 1;
  }
  tokenizer_spm *tok = tokenizer_spm_load(path);
  if (!tok) {
    fprintf(stderr, "load .model failed\n");
    return 1;
  }
  if (tokenizer_spm_vocab_size(tok) != 2) {
    fprintf(stderr, "want 2 pieces got %d\n", tokenizer_spm_vocab_size(tok));
    return 1;
  }
  int32_t ids[64];
  size_t n = 0;
  if (tokenizer_spm_encode(tok, "ab", ids, 64, &n) != 0 || n < 2) {
    fprintf(stderr, "encode mini failed n=%zu\n", n);
    return 1;
  }
  printf("PASS: mini encode n=%zu\n", n);
  tokenizer_spm_free(tok);

  const char *real =
      "/Users/user1/.zerollama/third_party/wan/Wan2.1-T2V-1.3B/"
      "google/umt5-xxl/spiece.model";
  FILE *rf = fopen(real, "rb");
  if (!rf) {
    fprintf(stderr, "test_tokenizer_spm OK (mini only; no umt5)\n");
    return 0;
  }
  fclose(rf);

  tok = tokenizer_spm_load(real);
  if (!tok) {
    fprintf(stderr, "load real spiece.model failed\n");
    return 1;
  }
  int vs = tokenizer_spm_vocab_size(tok);
  if (vs != 256000) {
    fprintf(stderr, "umt5 vocab want 256000 got %d\n", vs);
    tokenizer_spm_free(tok);
    return 1;
  }

  /* Gold: SentencePieceProcessor encode + eos (== HF add_special_tokens). */
  static const int32_t w_cat[] = {289, 6283, 19429, 1};
  static const int32_t w_cine[] = {320, 17443, 8378, 36215, 329, 289, 6283,
                                   369, 289, 197877, 1};
  static const int32_t w_hello[] = {154424, 3914, 1};
  static const int32_t w_fox[] = {517, 24598, 47178, 273, 56209, 48150, 281,
                                  702, 312, 298, 3065, 6833, 1};
  static const int32_t w_dogs[] = {16349, 100574, 25588, 301, 312, 45540, 367,
                                   364, 1377, 1};
  static const int32_t w_zh[] = {273, 4635, 2199, 4503, 1};
  static const int32_t w_empty[] = {1};

  struct {
    const char *prompt;
    const int32_t *want;
    size_t wn;
  } cases[] = {
      {"a cat running", w_cat, 4},
      {"A cinematic shot of a cat on a skateboard", w_cine, 11},
      {"hello world", w_hello, 3},
      {"The quick brown fox jumps over the lazy dog", w_fox, 13},
      {"Two dogs playing in the snow at dusk", w_dogs, 10},
      {"你好世界", w_zh, 5},
      {"", w_empty, 1},
  };

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    n = 0;
    if (tokenizer_spm_encode(tok, cases[c].prompt, ids, 64, &n) != 0) {
      fprintf(stderr, "encode failed %s\n", cases[c].prompt);
      tokenizer_spm_free(tok);
      return 1;
    }
    if (expect_ids(cases[c].prompt[0] ? cases[c].prompt : "<empty>", ids, n,
                   cases[c].want, cases[c].wn) != 0) {
      tokenizer_spm_free(tok);
      return 1;
    }
  }

  tokenizer_spm_free(tok);
  fprintf(stderr, "test_tokenizer_spm OK (mini + umt5 rematch %d)\n", vs);
  return 0;
}
