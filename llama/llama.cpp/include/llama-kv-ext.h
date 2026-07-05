#pragma once

/*
 * Staging C API for zerollama Phase 15 v20 — external PA page bind onto llama KV cells.
 *
 * WHY separate from llama.h: cell-index and tensor introspection are not stable upstream
 * yet; zerollama links this when built with ZEROLLAMA_KV_DECODE_LOOP=1.
 */

#include "llama.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_KV_EXT_OK           0
#define LLAMA_KV_EXT_UNSUPPORTED -1
#define LLAMA_KV_EXT_NOT_FOUND   -2
#define LLAMA_KV_EXT_ARG         -3

typedef struct llama_kv_cell_bind {
    llama_pos  pos;
    uint32_t   cell_idx;
    uint32_t   stream;
} llama_kv_cell_bind;

typedef struct llama_kv_tensor_info {
    int32_t  layer;
    uint64_t k_data;
    uint64_t v_data;
    uint64_t k_size_bytes;
    uint64_t v_size_bytes;
    int32_t  ok;
} llama_kv_tensor_info;

/* Resolved attn KV view behind llama_memory_t (for operator / probe visibility). */
typedef enum llama_kv_ext_mem_kind {
    LLAMA_KV_EXT_MEM_NONE = 0,
    LLAMA_KV_EXT_MEM_KV_CACHE,
    LLAMA_KV_EXT_MEM_ISWA_BASE,
    LLAMA_KV_EXT_MEM_HYBRID_ATTN,
    LLAMA_KV_EXT_MEM_HYBRID_ISWA_BASE,
    LLAMA_KV_EXT_MEM_UNSUPPORTED,
} llama_kv_ext_mem_kind;

/* Classify memory; returns LLAMA_KV_EXT_UNSUPPORTED for recurrent-only layouts. */
LLAMA_API int32_t llama_memory_kv_ext_classify(
        llama_memory_t         mem,
        llama_kv_ext_mem_kind * out_kind);

/* Resolve llama KV cell index for one (seq_id, pos). */
LLAMA_API int32_t llama_memory_kv_cell_for_pos(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_pos      pos,
        uint32_t *     out_cell_idx,
        uint32_t *     out_stream);

/*
 * Fill out[] with cell bindings for positions [pos_start, pos_end).
 * out_count receives the number written on success.
 */
LLAMA_API int32_t llama_memory_kv_cell_map_range(
        llama_memory_t       mem,
        llama_seq_id         seq_id,
        llama_pos            pos_start,
        llama_pos            pos_end,
        llama_kv_cell_bind * out,
        uint32_t             out_cap,
        uint32_t *           out_count);

/* Read-only K/V tensor backing store for one KV layer (kv layer index, not model layer).
 * stream is the per-sequence stream index from llama_kv_cell_bind.stream.
 * Pass 0 for single-stream (n_stream==1) contexts. */
LLAMA_API int32_t llama_memory_kv_tensor_info(
        llama_memory_t         mem,
        int32_t                kv_layer,
        uint32_t               stream,
        llama_kv_tensor_info * out);

/* Number of KV layers in the resolved attn cache (for multi-layer bind fan-out). */
LLAMA_API int32_t llama_memory_kv_n_layers(
        llama_memory_t mem,
        uint32_t *     out_n);

/*
 * Writable span for one PA page mapped onto llama KV cells (layer 0 export).
 * WHY layer 0 first: operators validate bind path before multi-layer fan-out;
 * higher layers share the same cell index layout.
 */
typedef struct llama_kv_page_map {
    llama_pos  pos_start;
    llama_pos  pos_end;          /* exclusive */
    uint32_t   n_cells;
    uint32_t   cell_idx_first;
    uint32_t   cell_idx_last;
    uint32_t   stream;
    uint64_t   k_data;           /* writable base for contiguous cell span */
    uint64_t   v_data;
    uint64_t   k_span_bytes;
    uint64_t   v_span_bytes;
    int32_t    kv_layer;
    int32_t    ok;
} llama_kv_page_map;

/*
 * Resolve writable K/V tensor spans for token positions
 * [seq_pos_min + page_index*block_size, +block_size) on attn KV cache.
 * Returns LLAMA_KV_EXT_NOT_FOUND when page has no live cells yet.
 */
LLAMA_API int32_t llama_memory_kv_page_map(
        llama_memory_t     mem,
        llama_seq_id       seq_id,
        llama_pos          seq_pos_min,
        uint32_t           page_index,
        uint32_t           block_size,
        int32_t            kv_layer,
        llama_kv_page_map * out);

/*
 * Probe whether writable cross-allocator PA→tensor page bind is available.
 * out_available: 1 when staging or upstream writable page-map API is linked.
 * out_api_name: detected API name (e.g. "llama_memory_kv_page_map") or "none".
 */
LLAMA_API int32_t llama_memory_kv_ext_writable_bind_probe(
        int32_t  * out_available,
        char     * out_api_name,
        uint32_t   name_cap);

#ifdef __cplusplus
}
#endif

