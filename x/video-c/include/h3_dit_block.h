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
 * Per-forward scratch arena. The block path otherwise mallocs/frees ~800 MB
 * per block-step (qkv/fused/hid/q/k/v/attn); Darwin mmaps those sizes, so
 * every allocation page-faults fresh pages. Create once per forward pass and
 * reuse across all blocks. All functions accept NULL (allocate per call).
 */
typedef struct h3_dit_ws h3_dit_ws;
h3_dit_ws *h3_dit_ws_create(int max_seq);
void h3_dit_ws_free(h3_dit_ws *ws);
/* Shared RoPE table slots ([max_seq, H3_DIT_ROPE_DIM] each) — build once per
 * forward via h3_dit_rope_tables and pass to every block. */
float *h3_dit_ws_cos(h3_dit_ws *ws);
float *h3_dit_ws_sin(h3_dit_ws *ws);

/* cos/sin RoPE tables for position_ids [seq,3]: [seq, H3_DIT_ROPE_DIM] each.
 * Position ids are identical across blocks and steps — compute once, share.
 * Uses the pack's rope.inv_freq (theta formula fallback). */
int h3_dit_rope_tables(const h3_st_store *store, const float *position_ids,
                       int seq, float *cos, float *sin, char *error,
                       size_t error_size);

/*
 * Packed-sequence block: x/out [seq, hidden].
 * tags[seq] = H3_ADALN_TAG_*; timestep in [0,1] (data-time, 1-sigma).
 * row_t NULL → broadcast `timestep`; else per-row [seq] (video vs audio schedules).
 * table may be NULL → load `adaln_t_table` (grid×rank).
 * rope_cos/rope_sin may be NULL → build from position_ids (per call).
 */
int h3_dit_block_forward(const h3_st_store *store, int block, const float *x,
                         int seq, const int *tags, float timestep,
                         const float *row_t, const float *position_ids,
                         const float *rope_cos, const float *rope_sin,
                         const float *table, int grid, int rank, float *out,
                         h3_dit_ws *ws, char *error, size_t error_size);

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
                               const float *x, int seq, float *out,
                               h3_dit_ws *ws, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_DIT_BLOCK_H */
