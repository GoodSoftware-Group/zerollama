/* Stream one MiniMax-H3 DiT transformer block from a safetensors store. */
#ifndef H3_DIT_BLOCK_H
#define H3_DIT_BLOCK_H

#include "h3_st_store.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { H3_ADALN_TABLE_GRID = 1001, H3_ADALN_TABLE_RANK = 64 };

/*
 * Packed-sequence block: x/out [seq, hidden].
 * tags[seq] = H3_ADALN_TAG_*; timestep in [0,1] (data-time, 1-sigma).
 * row_t NULL → broadcast `timestep`; else per-row [seq] (video vs audio schedules).
 * table may be NULL → load `adaln_t_table` (grid×rank).
 */
int h3_dit_block_forward(const h3_st_store *store, int block, const float *x,
                         int seq, const int *tags, float timestep,
                         const float *row_t, const float *position_ids,
                         const float *table, int grid, int rank, float *out,
                         char *error, size_t error_size);

/* y = x @ W^T + b; bname may be NULL. */
int h3_dit_linear_named(const h3_st_store *store, const char *weight_name,
                        const char *bias_name, const float *x, int rows,
                        int in_dim, int out_dim, float *y, char *error,
                        size_t error_size);

/*
 * Token-refiner block: pre-norm attn+MLP, no AdaLN, no RoPE.
 * prefix e.g. "token_refiner.blocks.0."
 */
int h3_dit_plain_block_forward(const h3_st_store *store, const char *prefix,
                               const float *x, int seq, float *out, char *error,
                               size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_DIT_BLOCK_H */
