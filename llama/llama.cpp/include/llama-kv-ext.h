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

/*
 * Number of KV layers in the resolved attn cache.
 * WHY needed: models with per-layer K/V scaling, MLA, or split-device offload may
 * allocate different tensor shapes per layer; verifying only layer 0 would silently
 * pass the tensor bind probe while deeper layers are unmapped. Callers should loop
 * 0..out_n-1 and fail the bind if any layer returns NOT_FOUND or UNSUPPORTED.
 */
LLAMA_API int32_t llama_memory_kv_n_layers(
        llama_memory_t mem,
        uint32_t *     out_n);

/*
 * Cache-level layout constants for the resolved attn KV.
 * WHY v_transposed: non-FA caches store V tokens at dim-1 cell positions but with
 * embedding values interleaved across n_embd_v_gqa rows at row stride kv_size.
 * A page-map consumer must check v_transposed before choosing its access pattern:
 *   v_transposed=0 (FA)    : V is [n_embd, n_cells] — contiguous per-cell buffer.
 *   v_transposed=1 (non-FA): V is [n_embd rows × kv_size cols] — scatter/gather required.
 * This struct is a cache constant; query it once rather than inferring from tensor dims.
 */
typedef struct llama_kv_cache_layout {
    uint32_t kv_size;        /* total cell slots in the KV cache */
    uint32_t n_stream;       /* per-sequence stream count (== llama_parallel_slots) */
    int32_t  v_transposed;   /* 1 when V uses transposed cell indexing (non-FA) */
    int32_t  ok;
} llama_kv_cache_layout;

/* Resolve cache-level layout constants (kv_size, n_stream, v_transposed). */
LLAMA_API int32_t llama_memory_kv_cache_layout(
        llama_memory_t         mem,
        llama_kv_cache_layout * out);

/*
 * Writable span for one PA page mapped onto llama KV cells.
 * Call llama_memory_kv_page_map() once per (page, layer) pair to fan out the bind
 * across all KV layers (v34+). All layers share the same cell_idx layout; only the
 * tensor base pointers (k_data, v_data) differ.
 *
 * IMPORTANT — v_transposed semantics (v35):
 *   v_transposed=0 (Flash Attention): V cells are contiguous.
 *     v_data..v_data+v_span_bytes is a flat [n_cells × n_embd_v_gqa] buffer.
 *   v_transposed=1 (non-FA, default): V cells are scattered across rows.
 *     v_data is the start of the cell block in dim-1, but each cell's embedding
 *     is split across n_embd_v_gqa rows at row stride kv_size.
 *     Do NOT memcpy v_span_bytes as a flat cell buffer — use scatter/gather.
 *
 * v_data is 0 (and v_span_bytes is 0) for MLA models where the V cache is absent.
 */
typedef struct llama_kv_page_map {
    llama_pos  pos_start;
    llama_pos  pos_end;          /* exclusive */
    uint32_t   n_cells;
    uint32_t   cell_idx_first;
    uint32_t   cell_idx_last;
    uint32_t   stream;
    uint64_t   k_data;           /* writable base for contiguous K cell span */
    uint64_t   v_data;           /* writable base for V cells; 0 for MLA (no V) */
    uint64_t   k_span_bytes;
    uint64_t   v_span_bytes;     /* byte extent; interpret with v_transposed flag */
    int32_t    kv_layer;
    int32_t    v_transposed;     /* 1 when V requires scatter/gather (non-FA) */
    int32_t    ok;
} llama_kv_page_map;

/*
 * Resolve writable K/V tensor spans for one PA page on one KV layer.
 * Covers token positions [seq_pos_min + page_index*block_size, +block_size).
 * Returns LLAMA_KV_EXT_NOT_FOUND when the page has no live cells yet.
 *
 * WHY per-layer: v34 fan-out calls this once per (page, kv_layer) pair.
 * All layers share the same cell_idx layout; kv_data pointers differ.
 * WHY multi-stream guard: cells belonging to different streams cannot form a
 * contiguous physical page; callers should always map positions from one stream.
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

/*
 * Phase 15 v47 — external PA buffer alias staging (probe + validate; no tensor mutation).
 *
 * WHY separate from page_map: page_map discovers llama-owned spans; alias_validate
 * checks whether external pool pointers can zero-copy alias those spans without copy.
 * True ggml allocator overlay bind remains a follow-on slice; v47 only classifies feasibility.
 */
typedef enum llama_kv_ext_alias_mode {
    LLAMA_KV_EXT_ALIAS_NONE = 0,
    LLAMA_KV_EXT_ALIAS_SAME_POINTER,    /* ext ptrs already match llama page_map targets */
    LLAMA_KV_EXT_ALIAS_HOST_REBASE,     /* host tensors; full-tensor rebase feasible (staging) */
    LLAMA_KV_EXT_ALIAS_BLOCKED_DEVICE,  /* KV tensors on non-host buffer (Metal/CUDA) */
    LLAMA_KV_EXT_ALIAS_BLOCKED_V_TRANS, /* non-FA V layout blocks flat external alias */
    LLAMA_KV_EXT_ALIAS_BLOCKED_SPAN,    /* ext byte spans != llama page_map spans */
    LLAMA_KV_EXT_ALIAS_BLOCKED_NO_PAGE, /* page_map failed (no live cells / misaligned page) */
    LLAMA_KV_EXT_ALIAS_BLOCKED_UNSUPPORTED, /* recurrent-only / unsupported memory layout */
} llama_kv_ext_alias_mode;

typedef struct llama_kv_ext_external_alias_probe {
    int32_t available;    /* 1 when LLAMA_KV_EXT_EXTERNAL_ALIAS is linked */
    int32_t validate_api; /* 1 when llama_memory_kv_page_alias_validate is linked */
    char    api_name[64];
} llama_kv_ext_external_alias_probe;

/* Static/build probe — no live ctx required. */
LLAMA_API int32_t llama_memory_kv_ext_external_alias_probe(
        llama_kv_ext_external_alias_probe * out);

typedef struct llama_kv_page_alias_plan {
    int32_t  ok;
    int32_t  alias_ready;       /* 1 when zero-copy alias is feasible now (SAME_POINTER) */
    int32_t  alias_mode;        /* llama_kv_ext_alias_mode */
    int32_t  buffer_host;       /* llama K tensor on host ggml buffer */
    int32_t  k_spans_match;
    int32_t  v_spans_match;
    int32_t  k_same_pointer;
    int32_t  v_same_pointer;
    int32_t  v_transposed;
    uint64_t llama_k_data;
    uint64_t llama_v_data;
    uint64_t k_span_bytes;
    uint64_t v_span_bytes;
    char     blocker[128];
} llama_kv_page_alias_plan;

/*
 * Validate external K/V pointers against llama_memory_kv_page_map geometry.
 * Does not mutate tensors — feasibility check only.
 *
 * alias_ready=1 only when ext pointers exactly match llama page_map targets
 * (SAME_POINTER). HOST_REBASE is reported when host spans match but pointers
 * differ; callers must not treat that as alias_ready until overlay bind ships.
 */
LLAMA_API int32_t llama_memory_kv_page_alias_validate(
        llama_memory_t            mem,
        llama_seq_id              seq_id,
        llama_pos                 seq_pos_min,
        uint32_t                  page_index,
        uint32_t                  block_size,
        int32_t                   kv_layer,
        uint64_t                  ext_k_data,
        uint64_t                  ext_k_span_bytes,
        uint64_t                  ext_v_data,
        uint64_t                  ext_v_span_bytes,
        llama_kv_page_alias_plan * out);

#ifdef __cplusplus
}
#endif

