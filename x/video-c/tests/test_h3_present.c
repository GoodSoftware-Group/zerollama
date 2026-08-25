#include "h3_present.h"

#include "h3_adaln_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_mrope_text_only(void) {
  uint32_t pos[3 * 4];
  CHECK(h3_present_mrope(NULL, 0, 4, pos) == 0);
  for (int a = 0; a < 3; a++)
    for (int i = 0; i < 4; i++)
      CHECK(pos[a * 4 + i] == (uint32_t)i);
  return 0;
}

static int test_mrope_one_image(void) {
  /* label 3 tokens, then start, 2×2 pads, end, prompt 2 → seq 3+1+4+1+2=11
   * span.start = 4 (after start), tokens=4, mh=mw=2 */
  h3_present_span sp = {.start = 4, .tokens = 4, .merged_h = 2, .merged_w = 2};
  const size_t seq = 11;
  uint32_t pos[3 * 11];
  CHECK(h3_present_mrope(&sp, 1, seq, pos) == 0);
  for (size_t i = 0; i < 4; i++) {
    CHECK(pos[i] == (uint32_t)i);
    CHECK(pos[seq + i] == (uint32_t)i);
    CHECK(pos[2 * seq + i] == (uint32_t)i);
  }
  /* pads: t=base=4, h=4+row, w=4+col */
  CHECK(pos[4] == 4 && pos[5] == 4 && pos[6] == 4 && pos[7] == 4);
  CHECK(pos[seq + 4] == 4 && pos[seq + 5] == 4 && pos[seq + 6] == 5 &&
        pos[seq + 7] == 5);
  CHECK(pos[2 * seq + 4] == 4 && pos[2 * seq + 5] == 5 &&
        pos[2 * seq + 6] == 4 && pos[2 * seq + 7] == 5);
  /* after end (index 8): next = start + max(h,w) = 4+2=6; pos = 6+(i-8) */
  CHECK(pos[8] == 6);
  CHECK(pos[9] == 7);
  CHECK(pos[10] == 8);
  CHECK(pos[seq + 10] == 8);
  return 0;
}

static int test_fl2va_t2va(h3_tokenizer *tok) {
  h3_presentation p;
  char err[256];
  CHECK(h3_present_fl2va(tok, "A red fox walking through snow", NULL, NULL, 0,
                         &p, err, sizeof(err)));
  const uint32_t fox[] = {32, 2518, 38835, 11435, 1526, 11794};
  CHECK(p.count == 6);
  CHECK(p.n_spans == 0);
  for (size_t i = 0; i < 6; i++) {
    CHECK(p.ids[i] == fox[i]);
    CHECK(p.tags[i] == (uint8_t)H3_ADALN_TAG_TEXT);
    CHECK(p.pos[i] == (uint32_t)i);
  }
  h3_presentation_free(&p);
  return 0;
}

static int test_fl2va_picture(h3_tokenizer *tok) {
  h3_presentation p;
  char err[256];
  int mh = 2, mw = 2;
  CHECK(h3_present_fl2va(tok, "hi", &mh, &mw, 1, &p, err, sizeof(err)));
  CHECK(p.n_spans == 1);
  CHECK(p.spans[0].tokens == 4);
  size_t vs = p.spans[0].start - 1;
  CHECK(p.ids[vs] == H3_VISION_START_ID);
  CHECK(p.ids[vs + 1] == H3_IMAGE_PAD_ID);
  CHECK(p.ids[vs + 5] == H3_VISION_END_ID);
  CHECK(p.tags[vs] == (uint8_t)H3_ADALN_TAG_VIDEO);
  CHECK(p.tags[vs + 5] == (uint8_t)H3_ADALN_TAG_VIDEO);
  CHECK(p.tags[0] == (uint8_t)H3_ADALN_TAG_TEXT); /* Picture label */
  CHECK(p.tags[p.count - 1] == (uint8_t)H3_ADALN_TAG_TEXT);
  h3_presentation_free(&p);
  return 0;
}

int main(void) {
  if (test_mrope_text_only())
    return 1;
  if (test_mrope_one_image())
    return 1;
  char def[768];
  const char *path = getenv("H3_BMTL_TOK");
  if (!path) {
    const char *home = getenv("HOME");
    if (!home) {
      fprintf(stderr, "test_h3_present SKIP (no HOME)\n");
      return 0;
    }
    snprintf(def, sizeof(def),
             "%s/.zerollama/third_party/h3/minimax_h3.bmtl_tok", home);
    path = def;
  }
  if (access(path, R_OK) != 0) {
    fprintf(stderr, "test_h3_present SKIP (no blob)\n");
    return 0;
  }
  char error[512];
  h3_tokenizer *tok = h3_tokenizer_load(path, error, sizeof(error));
  if (!tok) {
    fprintf(stderr, "FAIL load: %s\n", error);
    return 1;
  }
  if (test_fl2va_t2va(tok) || test_fl2va_picture(tok)) {
    h3_tokenizer_free(tok);
    return 1;
  }
  h3_tokenizer_free(tok);
  fprintf(stderr, "test_h3_present OK\n");
  return 0;
}
