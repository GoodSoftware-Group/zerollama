/* Rematch antirez h3.c tests/test_tokenizer.c encode IDs (BMTL, no decode). */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int check_case(h3_tokenizer *tokenizer, const char *text,
                      const uint32_t *expected, size_t expected_count) {
  char error[256];
  uint32_t *ids = NULL;
  size_t count = 0;
  CHECK(h3_tokenizer_encode(tokenizer, text, 1, &ids, &count, error,
                            sizeof(error)));
  if (count != expected_count) {
    fprintf(stderr, "FAIL count want %zu got %zu for %s\n", expected_count,
            count, text);
    h3_tokenizer_ids_free(ids);
    return 1;
  }
  for (size_t i = 0; i < count; i++) {
    if (ids[i] != expected[i]) {
      fprintf(stderr, "FAIL id[%zu] want %u got %u for %s\n", i, expected[i],
              ids[i], text);
      h3_tokenizer_ids_free(ids);
      return 1;
    }
  }
  h3_tokenizer_ids_free(ids);
  return 0;
}

int main(int argc, char **argv) {
  char def[768];
  const char *path = argc > 1 ? argv[1] : NULL;
  if (!path) {
    const char *env = getenv("H3_BMTL_TOK");
    if (env && env[0])
      path = env;
    else {
      const char *home = getenv("HOME");
      if (!home) {
        fprintf(stderr, "test_h3_tokenizer SKIP (no HOME)\n");
        return 0;
      }
      snprintf(def, sizeof(def),
               "%s/.zerollama/third_party/h3/minimax_h3.bmtl_tok", home);
      path = def;
    }
  }
  if (access(path, R_OK) != 0) {
    fprintf(stderr, "test_h3_tokenizer SKIP (no blob at %s)\n", path);
    return 0;
  }
  char error[512];
  h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
  if (!tokenizer) {
    fprintf(stderr, "FAIL test_h3_tokenizer: %s\n", error);
    return 1;
  }
  const uint32_t fox[] = {32, 2518, 38835, 11435, 1526, 11794};
  if (check_case(tokenizer, "A red fox walking through snow", fox,
                 sizeof(fox) / sizeof(fox[0])))
    return 1;
  const uint32_t unicode[] = {9707, 11, 50891, 0, 220, 220, 17, 15, 17, 21, 198,
                              104811, 51950, 594};
  if (check_case(tokenizer, "Hello, WORLD!  2026\n中文 café's", unicode,
                 sizeof(unicode) / sizeof(unicode[0])))
    return 1;
  const uint32_t emoji[] = {145080, 64, 4894};
  if (check_case(tokenizer, "🙂a!\n", emoji, sizeof(emoji) / sizeof(emoji[0])))
    return 1;
  const uint32_t special[] = {151644};
  if (check_case(tokenizer, "<|im_start|>", special, 1))
    return 1;
  const uint32_t cinematic[] = {32, 64665, 3265, 5239, 315, 264, 8866, 1778,
                                11958, 4633, 10971, 13};
  if (check_case(tokenizer, "A cinematic close-up of a clockwork bird taking flight.",
                 cinematic, sizeof(cinematic) / sizeof(cinematic[0])))
    return 1;

  uint32_t *ids = NULL;
  size_t count = 99;
  CHECK(h3_tokenizer_encode(tokenizer, "", 0, &ids, &count, error,
                            sizeof(error)));
  CHECK(count == 0 && ids == NULL);
  CHECK(h3_tokenizer_encode(tokenizer, "", 1, &ids, &count, error,
                            sizeof(error)));
  CHECK(count == 1 && ids[0] == H3_PAD_TOKEN_ID);
  h3_tokenizer_ids_free(ids);

  h3_tokenizer_free(tokenizer);
  fprintf(stderr, "test_h3_tokenizer OK (%d checks vs antirez Qwen IDs)\n",
          checks);
  return 0;
}
