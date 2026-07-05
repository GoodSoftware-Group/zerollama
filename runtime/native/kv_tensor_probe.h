#ifndef KV_TENSOR_PROBE_H
#define KV_TENSOR_PROBE_H

#include <stdint.h>

#include "kv_page_bind_internal.h"

#ifdef ZEROLLAMA_KV_DECODE_LOOP

typedef struct {
    int memory_non_null;
    int can_shift;
    int32_t seq_pos_min;
    int32_t seq_pos_max;
    int32_t llama_token_cells;
    int32_t pa_pages_registered;
    int32_t pa_block_size;
    int pages_fit;
    int aligned;
    int cell_pages_bound;
    int tensor_pages_bound;
    int32_t blocker_code;
    int32_t kv_stream;
    int32_t memory_kind;
    uint64_t kv_k_data_layer0;
    uint64_t kv_v_data_layer0;
    int32_t kv_n_layers;
    int32_t tensor_layers_verified;
    int32_t kv_v_transposed;
    uint32_t kv_cache_kv_size;
    uint32_t kv_cache_n_stream;
    int physical_pages_bound;
    int physical_pages_mapped;
} KvTensorProbeResult;

int kv_tensor_probe_run(void *ctx, int32_t seq_id, int32_t kv_slot, KvTensorProbeResult *out);

/* v35: survives page_bind_clear so /health can show last decode bind. */
int kv_tensor_probe_last_get(int kv_slot, KvTensorProbeResult *out);

#endif /* ZEROLLAMA_KV_DECODE_LOOP */
#endif /* KV_TENSOR_PROBE_H */
