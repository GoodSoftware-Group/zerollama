/* NicoLab28 ClipProj — map small Qwen3-VL hidden → MiniMax-H3 cond [*, 5120]. */
#ifndef H3_CLIPPROJ_HOST_H
#define H3_CLIPPROJ_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  H3_CLIPPROJ_DOUT = 5120, /* MiniMax-H3 conditioning width */
  H3_CLIPPROJ_DIN_4B = 2560,
  H3_CLIPPROJ_DIN_8B = 4096,
  H3_CLIPPROJ_MLP_HIDDEN = 16384
};

typedef struct h3_clipproj h3_clipproj;

/* Load mmh3-*-ClipProj*.safetensors (W, mean/std in/out; optional sink_out, mlp.*). */
h3_clipproj *h3_clipproj_load(const char *path, char *error, size_t error_size);
void h3_clipproj_free(h3_clipproj *proj);

int h3_clipproj_din(const h3_clipproj *proj);
int h3_clipproj_dout(const h3_clipproj *proj);
int h3_clipproj_has_sink(const h3_clipproj *proj);
int h3_clipproj_has_mlp(const h3_clipproj *proj);

/*
 * Project hidden [seq, din] → cond [seq, dout] (row-major).
 *   xn = (h - mean_in) / std_in
 *   yn = xn @ W  (+ optional mlp(xn) in standardized space)
 *   cond = yn * std_out + mean_out
 *   if sink_out present and seq>0: cond[0,:] = sink_out
 * Returns 0 on success.
 */
int h3_clipproj_apply(const h3_clipproj *proj, const float *hidden, int seq,
                      float *cond_out, char *error, size_t error_size);

/* Host-only affine (no file): same formula; W is [din, dout] row-major. */
int h3_clipproj_apply_affine(const float *hidden, int seq, int din, int dout,
                             const float *W, const float *mean_in,
                             const float *std_in, const float *mean_out,
                             const float *std_out, const float *sink_out,
                             float *cond_out);

#ifdef __cplusplus
}
#endif

#endif /* H3_CLIPPROJ_HOST_H */
