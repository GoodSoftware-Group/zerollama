/* MiniMax-H3 FL2VA / t2va prompt presentation (no chat template). */
#ifndef H3_PRESENT_H
#define H3_PRESENT_H

#include "h3_tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VISION_START_ID UINT32_C(151652)
#define H3_VISION_END_ID UINT32_C(151653)
#define H3_IMAGE_PAD_ID UINT32_C(151655)
#define H3_VIDEO_PAD_ID UINT32_C(151656)

typedef struct {
  size_t start;     /* first image_pad (after vision_start) */
  size_t tokens;    /* merged_h * merged_w */
  int merged_h;
  int merged_w;
} h3_present_span;

typedef struct {
  uint32_t *ids;
  uint8_t *tags;  /* H3_ADALN_TAG_TEXT or VIDEO */
  uint32_t *pos;  /* axis-major [3, count] */
  size_t count;
  h3_present_span *spans;
  size_t n_spans;
} h3_presentation;

void h3_presentation_free(h3_presentation *p);

/*
 * mRoPE positions for vision spans (antirez h3_multimodal.c).
 * pos is [3, seq]. Sequential text when n_spans==0.
 */
int h3_present_mrope(const h3_present_span *spans, size_t n_spans, size_t seq,
                     uint32_t *pos);

/*
 * t2va: prompt only. fl2va: each image prepends "<Picture i>: " +
 * vision_start + image_pad×(mh*mw) + vision_end. No specials on the prompt.
 * merged_h/w may be NULL when n_images==0.
 */
int h3_present_fl2va(const h3_tokenizer *tok, const char *prompt,
                     const int *merged_h, const int *merged_w, size_t n_images,
                     h3_presentation *out, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_PRESENT_H */
