/* Phase 15 v12–v15 — optional libllama link probe + decode loop (see setup.py). */

#ifndef KV_DECODE_LOOP_H
#define KV_DECODE_LOOP_H

#include <stddef.h>
#include <stdint.h>

#ifdef ZEROLLAMA_KV_DECODE_LOOP

/*
 * Link probe — called by decode_loop_status() Python binding.
 */
size_t kv_decode_loop_llama_max_devices(void);

/*
 * Run all prefill batches for a single sequence.
 *
 * ctx          : llama_context* (opaque pointer, owned by Python)
 * tokens       : token ids, length n_tokens
 * n_tokens     : number of prompt tokens
 * seq_id       : llama sequence id (same as kv_slot in in-process mode)
 * block_size   : PA page size; 0 → single batch (no page-chunking)
 * pos_start    : first llama write position (v14: remaining prefill resume)
 * steps_out    : incremented by number of llama_decode calls made
 *
 * WHY ctx is void*: Python holds the ctypes pointer as a c_void_p integer;
 * we cast it to llama_context* inside the function.
 *
 * Returns 0 on success, -1 on llama_decode failure, -2 on page bind validation failure.
 */
int kv_decode_loop_run_prefill(
    void          *ctx,
    const int32_t *tokens,
    int32_t        n_tokens,
    int32_t        seq_id,
    int32_t        block_size,
    int32_t        pos_start,
    int32_t       *steps_out
);

/*
 * Run one decode step at position current_pos.
 *
 * smpl         : optional llama_sampler*; when non-NULL, sampled_out is set
 * sampled_out  : token id from llama_sampler_sample after decode (v15)
 *
 * Returns 0 on success, -1 on llama_decode failure, -2 on page bind validation failure.
 */
int kv_decode_loop_run_step(
    void    *ctx,
    int32_t  token,
    int32_t  seq_id,
    int32_t  current_pos,
    void    *smpl,
    int32_t *sampled_out,
    int32_t *steps_out
);

/*
 * Sample from the last forward pass (post-prefill or post-decode).
 *
 * WHY separate from run_step: first generated token is sampled after prefill
 * batches with logits_last=0 — no decode step precedes it.
 */
int32_t kv_decode_loop_sample(void *smpl, void *ctx);

/* WHY in header: kv_block_pool.c (Python binding) enforces the same limit. */
#define KV_DECODE_LOOP_BATCH_MAX 64

/*
 * Continuous batch decode: one llama_decode for N single-token rows (v26).
 *
 * tokens[], seq_ids[], positions[] — parallel arrays length n_entries (>= 1).
 * Each row is one in-flight sequence's next decode token at its llama position.
 * Page-bind validated per row before llama_decode.
 *
 * sampled_out — optional array of n_entries; when smpl_ptrs is non-NULL, filled via
 * llama_sampler_sample(smpl_ptrs[i], ctx, i) for each batch index i.
 *
 * steps_out — set to 1 on success (one llama_decode call, not n_entries).
 *
 * Returns 0 on success, -1 on llama_decode failure, -2 on page bind failure.
 */
int kv_decode_loop_run_batch_step(
    void                *ctx,
    const int32_t       *tokens,
    const int32_t       *seq_ids,
    const int32_t       *positions,
    int32_t              n_entries,
    const void *const   *smpl_ptrs,
    int32_t             *sampled_out,
    int32_t             *steps_out
);

/*
 * Run the post-prefill tensor probe with the GIL held.
 * Must be called after Py_END_ALLOW_THREADS in the Python binding.
 */
void kv_decode_loop_post_prefill_probe(void *ctx, int32_t seq_id);

#endif /* ZEROLLAMA_KV_DECODE_LOOP */
#endif /* KV_DECODE_LOOP_H */
