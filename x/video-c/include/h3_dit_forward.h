/* Packed MiniMax-H3 DiT: patch projs + token refiner + streamed blocks + heads. */
#ifndef H3_DIT_FORWARD_H
#define H3_DIT_FORWARD_H

#include "h3_st_store.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Packed forward. video [nv, 96], audio [na, 32], text [nt, 5120].
 * indices are positions in the packed sequence of length `seq`.
 * tags[seq], position_ids[seq*3], timestep in [0,1].
 * n_layers 0..50 (0 = refiner + final only).
 * video_out [nv, 96], audio_out [na, 32].
 */
int h3_dit_forward(const h3_st_store *store, const float *video, int nv,
                   const float *audio, int na, const float *text, int nt,
                   const int *video_index, const int *audio_index,
                   const int *text_index, const int *tags,
                   const float *position_ids, int seq, float timestep,
                   const float *row_t, int n_layers, float *video_out,
                   float *audio_out, char *error, size_t error_size);

/*
 * Rectified-flow denoise. Default Euler (`x += (σ−σ′)·v`). Lab
 * `H3_SAMPLER=res_multistep` is Comfy sample_res_multistep (η=0, CONST
 * denoised = x+σv). `reuse_interval` ≤1 evaluates every step.
 * video shift 12 / audio 3. adaln_t_sigma: 0 = t=1-σ, 1 = t=σ, <0 = env.
 */
int h3_dit_denoise(const h3_st_store *store, float *video, int nv, float *audio,
                   int na, const float *text, int nt, const int *video_index,
                   const int *audio_index, const int *text_index,
                   const int *tags, const float *position_ids, int seq,
                   int steps, int n_layers, int reuse_interval,
                   int adaln_t_sigma, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_DIT_FORWARD_H */
