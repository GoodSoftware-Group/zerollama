#ifndef KV_TENSOR_PROBE_H
#define KV_TENSOR_PROBE_H

#include <stdint.h>

#include "kv_page_bind_internal.h"

#ifdef ZEROLLAMA_KV_DECODE_LOOP

/*
 * Result of one kv_tensor_probe_run() call.
 *
 * Populate order inside kv_tensor_probe_run:
 *   1. memory_non_null, memory_kind, can_shift, seq_pos_min/max  — from llama_get_memory
 *   2. kv_v_transposed, kv_cache_kv_size, kv_cache_n_stream      — from llama_memory_kv_cache_layout (v35)
 *   3. pa_pages_registered, pa_block_size                         — from KvPageBind registry
 *   4. llama_token_cells, pages_fit, aligned                      — computed from 1+3
 *   5. cell_pages_bound, tensor_pages_bound, kv_n_layers,
 *      tensor_layers_verified, kv_k_data_layer0, kv_v_data_layer0 — from kv_tensor_bind_attempt (v20/v34)
 *   6. physical_pages_bound, physical_pages_mapped                — from kv_page_bind_materialize_writable (v33/v34)
 *
 * WHY kv_v_transposed is set early (step 2): the layout is a cache constant;
 * it should be visible on /health even if the bind attempt fails partway through.
 */
typedef struct {
    /* llama memory state */
    int memory_non_null;
    int can_shift;
    int32_t seq_pos_min;
    int32_t seq_pos_max;
    int32_t llama_token_cells;
    int32_t memory_kind;        /* llama_kv_ext_mem_kind enum value */

    /* PA page table */
    int32_t pa_pages_registered;
    int32_t pa_block_size;
    int pages_fit;
    int aligned;

    /* bind result */
    int cell_pages_bound;
    int tensor_pages_bound;
    int32_t blocker_code;       /* KV_TENSOR_BLOCKER_* — most specific failure reason */
    int32_t kv_stream;          /* stream index used for this sequence */

    /* v34: multi-layer verify */
    int32_t kv_n_layers;
    int32_t tensor_layers_verified;

    /* v35: cache layout (populated before bind attempt) */
    int32_t kv_v_transposed;    /* 1 when non-FA transposed-V layout; callers must scatter/gather */
    uint32_t kv_cache_kv_size;
    uint32_t kv_cache_n_stream;

    /* layer 0 tensor pointers (for operator snapshot) */
    uint64_t kv_k_data_layer0;
    uint64_t kv_v_data_layer0;

    /* v33: writable page-map result */
    int physical_pages_bound;
    int physical_pages_mapped;
} KvTensorProbeResult;

int kv_tensor_probe_run(void *ctx, int32_t seq_id, int32_t kv_slot, KvTensorProbeResult *out);

/*
 * v35: last-probe snapshot.
 *
 * WHY: page_bind_clear() fires on generation complete. After that, /health has no
 * running request to probe, so kv_page_bind.status would downgrade to "partial"
 * even though the decode was fully bound. The last-probe snapshot preserves the
 * most recent successful tensor probe so operators can see post-generate layout
 * data without keeping a request alive just for health polling.
 *
 * WHY indexed by bind-table position (not kv_slot value):
 *   kv_slot is an arbitrary scheduler integer and can exceed KV_MAX_PAGE_BINDS.
 *   Using kv_slot directly as an array index would overflow. The snapshot table
 *   mirrors g_page_binds[] layout and stores kv_slot inside the entry.
 *
 * kv_tensor_probe_last_get:          look up by kv_slot value (O(n) scan).
 *                                    Returns 0=found, 1=not found, -1=bad arg.
 * kv_tensor_probe_last_get_by_index: iterate table by position idx (0-based).
 *                                    out_kv_slot receives the stored kv_slot value.
 *                                    Returns 0=ok, 1=slot unused, -1=bad arg.
 */
int kv_tensor_probe_last_get(int kv_slot, KvTensorProbeResult *out);
int kv_tensor_probe_last_get_by_index(int idx, int *out_kv_slot, KvTensorProbeResult *out);

#endif /* ZEROLLAMA_KV_DECODE_LOOP */
#endif /* KV_TENSOR_PROBE_H */
