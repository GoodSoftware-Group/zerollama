/* Prompt → ClipProj cond [nt, 5120] (Qwen3-VL-4B TE or hash fallback). */
#ifndef H3_TEXT_COND_H
#define H3_TEXT_COND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float *cond; /* [nt, H3_CLIPPROJ_DOUT] */
  int *tags;   /* [nt] AdaLN modality (0 video / 1 text / 2 audio); optional */
  int nt;
  int used_4b;
  int used_dump; /* 1 = H3_TEXT_COND / --text-cond f32 dump (32B TE) */
} h3_text_cond;

void h3_text_cond_free(h3_text_cond *c);

/*
 * t2va/fl2va present + TE + ClipProj. clipproj_path NULL → celeb-mlp cache.
 * n_images 0 → text-only. TE dir: H3_QWEN_TE_DIR or ~/.zerollama/models/Qwen3-VL-4B-Instruct.
 */
int h3_text_cond_from_prompt(const char *prompt, const int *merged_h,
                             const int *merged_w, size_t n_images,
                             const char *clipproj_path, h3_text_cond *out,
                             char *error, size_t error_size);

/*
 * Raw cond dump: magic "H3TE" + uint32le nt + uint32le dim + float32[nt*dim]
 * + optional uint8[nt] MiniMax token tags (Comfy minimax_token_tags).
 * dim must be 5120. From tools/h3_dump_comfy_te32.py (Comfy NVFP4 32B TE).
 */
int h3_text_cond_from_bin(const char *path, h3_text_cond *out, char *error,
                          size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_TEXT_COND_H */
