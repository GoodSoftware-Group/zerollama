/* AdaLN ModulationCache sizing helpers (weightless DiT planning). */
#ifndef H3_ADALN_HOST_H
#define H3_ADALN_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  H3_ADALN_MODALITY_NUM = 3, /* video, text, audio */
  H3_ADALN_TAG_VIDEO = 0,
  H3_ADALN_TAG_TEXT = 1,
  H3_ADALN_TAG_AUDIO = 2,
  H3_ADALN_TAG_PAD = -1,
  H3_ADALN_HIDDEN_SIZE = 5376,
  H3_ADALN_NUM_LAYERS = 50,
  H3_ADALN_TENSORS_PER_BLOCK = 6, /* shift/scale/gate × msa/mlp */
  H3_ADALN_OUT_FEATURES = 96768,  /* 6 * 3 * 5376 */
  H3_ADALN_TIME_EMBED_DIM = 2688,
  H3_ADALN_FINAL_OUT_FEATURES = 10752 /* 2 * 5376 */
};

/* Row in the (timestep × modality) modulation table. PAD maps to video (0). */
int h3_adaln_modality_row(int timestep_index, int tag);

/*
 * Distinct timesteps presented to DiT: union of schedule sigmas + conditioning
 * level. Writes sorted unique values into dst; returns count, or -1 on error.
 */
int h3_adaln_schedule_timesteps(const float *sigmas, int n_sigmas,
                                float conditioning_level, float *dst,
                                int dst_cap);

/*
 * Distinct per-token AdaLN times, sorted like Comfy unique_t.
 * row_t NULL → broadcast timestep. Writes uniq[0..n) and tslots[seq].
 */
int h3_adaln_collect_timesteps(const float *row_t, float timestep, int seq,
                               float *uniq, int uniq_cap, int *tslots);

/* Cache value count / bf16 bytes for T distinct timesteps (release geometry). */
unsigned long long h3_adaln_cache_values(int num_timesteps);
unsigned long long h3_adaln_cache_bf16_nbytes(int num_timesteps);

/* Per-block adaln_proj bf16 footprint (weight [out,in] + bias [out]). */
unsigned long long h3_adaln_proj_bf16_nbytes(void);

/*
 * Split AdaLN linear output [T, 6*3*H] into 6 contiguous tensors of [T*3, H]
 * (MLX AdaLayerNormModulation). dst must hold 6 * T * 3 * H floats, laid out
 * as dst[k * (T*3*H) + ...] for k in 0..5.
 */
int h3_adaln_split_block(const float *proj_out, int num_timesteps, int hidden,
                         float *dst_six);

/* final_layer.adaln_proj: [T, 2*H] → shift[T,H], scale[T,H]. */
int h3_adaln_split_final(const float *proj_out, int num_timesteps, int hidden,
                         float *shift, float *scale);

/*
 * Pruned AdaLN timestep table [grid, rank] (e.g. 1001×64). t in [0,1] is
 * linearly interpolated. The stored table is already mean-centered.
 */
int h3_adaln_table_embed(const float *table, int grid, int rank, float t,
                         float *out);

#ifdef __cplusplus
}
#endif

#endif /* H3_ADALN_HOST_H */
