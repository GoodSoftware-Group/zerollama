/*
 * Phase 15 v12–v15 — libllama link probe + decode loop.
 *
 * WHY separate file: keeps kv_block_pool.c free of llama.h when the optional
 * decode-loop build flag is off (default CI / operator builds).
 *
 * v12: link probe only (kv_decode_loop_llama_max_devices).
 * v13: prefill + single-step decode via llama_decode, consuming decode_work
 *      plans built by kv_decode_prefill_plan / kv_decode_step_plan.
 * v15: llama_sampler_sample in C (optional smpl on run_step + run_sample).
 * v24: kv_page_bind_validate_range before each llama_decode; post-prefill
 *      tensor probe moved out of run_prefill into a separate helper so the
 *      registry write happens with the GIL held (data-race fix).
 * v26: kv_decode_loop_run_batch_step — N single-token rows, one llama_decode.
 * v30: smpl_ptrs[] — one sampler per row so llama_sampler_sample uses logit
 *      index i and accept state does not bleed across sequences (v27 audit).
 * v31: chunked-prefill abort flag — process-global atomic_int checked between
 *      page-aligned chunks; Python calls kv_decode_loop_abort_set() while the
 *      GIL is released; returns KV_DECODE_LOOP_ERR_ABORT (-3) on cancel.
 */

#ifdef ZEROLLAMA_KV_DECODE_LOOP
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#include "llama.h"
#include "kv_decode_loop.h"
#include "kv_page_bind_internal.h"
#include "kv_tensor_probe.h"

/* WHY weak stub: patched libllama exports a real invalidator on CUDA builds;
 * vendor/pinned trees without the hook still link _kv_native for Metal decode. */
__attribute__((weak)) int
llama_context_cuda_graph_invalidate(struct llama_context * ctx)
{
    (void)ctx;
    return 0;
}

/* WHY -2 for page bind: distinct from llama_decode failure (-1) so Python can
 * surface the same LlamaServerError as ctypes page-bind validation. */
#define KV_DECODE_LOOP_ERR_BIND -2
/* KV_DECODE_LOOP_ERR_ABORT defined in header (-3): abort between prefill chunks. */

/*
 * Process-global abort flag (v31).
 *
 * WHY atomic_int: the GIL is released during prefill (Py_BEGIN_ALLOW_THREADS);
 * a Python signal-handler or cancellation thread calls kv_decode_loop_abort_set
 * concurrently.  atomic_int with RELAXED ordering is sufficient — we only need
 * visibility within one prefill call, not cross-thread ordering of llama state.
 */
static atomic_int _kv_prefill_abort = 0;

void
kv_decode_loop_abort_set(void)
{
    atomic_store_explicit(&_kv_prefill_abort, 1, memory_order_relaxed);
}

void
kv_decode_loop_abort_clear(void)
{
    atomic_store_explicit(&_kv_prefill_abort, 0, memory_order_relaxed);
}

int
kv_decode_loop_abort_check(void)
{
    return atomic_load_explicit(&_kv_prefill_abort, memory_order_relaxed);
}

size_t
kv_decode_loop_llama_max_devices(void)
{
    return llama_max_devices();
}

/*
 * WHY manual llama_batch: llama_batch_init allocates pos[] and leaves other
 * arrays NULL-initialised; llama.cpp reads them all so we must populate every
 * field.  llama_batch_get_one is stack-unsafe for chunked calls.  We reuse a
 * single heap batch across chunks to avoid repeated alloc/free on the hot path.
 */
/*
 * WHY n_seq_max is kept as a parameter despite being always 1 today: future
 * multi-sequence batches will size the per-token seq_id[i] inner arrays here.
 * For now, every token has exactly one sequence id so inner arrays are length 1.
 */
static llama_batch *
kv_batch_alloc(int32_t max_tokens, int32_t n_seq_max)
{
    (void)n_seq_max; /* reserved for multi-seq future; inner arrays are length 1 */
    llama_batch *b = (llama_batch *)malloc(sizeof(llama_batch));
    if (!b) return NULL;
    memset(b, 0, sizeof(*b));
    b->token    = (llama_token  *)malloc((size_t)max_tokens * sizeof(llama_token));
    b->pos      = (llama_pos    *)malloc((size_t)max_tokens * sizeof(llama_pos));
    b->n_seq_id = (int32_t      *)malloc((size_t)max_tokens * sizeof(int32_t));
    b->seq_id   = (llama_seq_id **)malloc((size_t)max_tokens * sizeof(llama_seq_id *));
    b->logits   = (int8_t        *)malloc((size_t)max_tokens * sizeof(int8_t));
    if (!b->token || !b->pos || !b->n_seq_id || !b->seq_id || !b->logits) {
        free(b->token); free(b->pos); free(b->n_seq_id);
        free(b->seq_id); free(b->logits); free(b);
        return NULL;
    }
    /* Allocate the per-token seq_id[i] arrays (each is length 1 here). */
    for (int32_t i = 0; i < max_tokens; i++) {
        b->seq_id[i] = (llama_seq_id *)malloc(sizeof(llama_seq_id));
        if (!b->seq_id[i]) {
            for (int32_t j = 0; j < i; j++) free(b->seq_id[j]);
            free(b->token); free(b->pos); free(b->n_seq_id);
            free(b->seq_id); free(b->logits); free(b);
            return NULL;
        }
    }
    b->embd = NULL; /* text-token path; not embedding input */
    return b;
}

static void
kv_batch_free(llama_batch *b, int32_t max_tokens)
{
    if (!b) return;
    if (b->seq_id) {
        for (int32_t i = 0; i < max_tokens; i++) {
            free(b->seq_id[i]);
        }
    }
    free(b->token); free(b->pos); free(b->n_seq_id);
    free(b->seq_id); free(b->logits); free(b);
}

static void
kv_batch_fill_entries(
    llama_batch *b,
    const int32_t *tokens,
    const int32_t *seq_ids,
    const int32_t *positions,
    int32_t n,
    int logits_all)
{
    b->n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        b->token[i]      = (llama_token)tokens[i];
        b->pos[i]        = (llama_pos)positions[i];
        b->n_seq_id[i]   = 1;
        b->seq_id[i][0]  = (llama_seq_id)seq_ids[i];
        b->logits[i]     = (int8_t)(logits_all ? 1 : 0);
    }
}

static void
kv_batch_fill(
    llama_batch *b,
    const int32_t *tokens,
    int32_t n,
    int32_t seq_id,
    int32_t pos_start,
    int logits_last)          /* 0 = all zeros, 1 = last-only; WHY prefill=0: no
                              * sampling logits until decode single-token batch */
{
    b->n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        b->token[i]    = (llama_token)tokens[i];
        b->pos[i]      = (llama_pos)(pos_start + i);
        b->n_seq_id[i] = 1;
        b->seq_id[i][0] = (llama_seq_id)seq_id;
        b->logits[i]   = (int8_t)(logits_last && (i == n - 1) ? 1 : 0);
    }
}

int
kv_decode_loop_run_prefill(
    void          *ctx,
    const int32_t *tokens,
    int32_t        n_tokens,
    int32_t        seq_id,
    int32_t        block_size,
    int32_t        pos_start,
    int32_t       *steps_out)
{
    /*
     * WHY block_size > 0 for page-aligned chunking: mirrors decode_prefill_chunks
     * and the Python ctypes path.  When block_size <= 0, one batch for all tokens.
     * WHY pos_start: v14 resume — tokens[] may be remaining prompt slice; llama
     * positions are pos_start + tok_off (not tok_off alone).
     */
    if (n_tokens <= 0) {
        if (steps_out) *steps_out = 0;
        return 0;
    }

    struct llama_context *lctx = (struct llama_context *)ctx;
    int chunk_sz = (block_size > 0 && n_tokens > block_size) ? block_size : n_tokens;

    llama_batch *b = kv_batch_alloc(chunk_sz, 1);
    if (!b) return -1;

    int32_t tok_off = 0;
    int32_t remaining = n_tokens;
    while (remaining > 0) {
        /* v31: check abort flag between chunks so a Python cancellation thread
         * can interrupt long prefill without waiting for the next llama_decode. */
        if (kv_decode_loop_abort_check()) {
            kv_batch_free(b, chunk_sz);
            return KV_DECODE_LOOP_ERR_ABORT;
        }
        int32_t n = remaining < chunk_sz ? remaining : chunk_sz;
        /* WHY validate each chunk: long prefill may span multiple pages; fail
         * before llama_decode writes past the PA-reserved page table. */
        if (kv_page_bind_validate_range(seq_id, pos_start + tok_off, n) != 0) {
            kv_batch_free(b, chunk_sz);
            return KV_DECODE_LOOP_ERR_BIND;
        }
        /* WHY logits_last on final chunk only: prefill must leave logits for the
         * first llama_sampler_sample after prompt ingest (Python decode loop). */
        const int logits_last = (remaining <= n) ? 1 : 0;
        kv_batch_fill(b, tokens + tok_off, n, seq_id, pos_start + tok_off,
                      logits_last);
        if (llama_decode(lctx, *b) != 0) {
            kv_batch_free(b, chunk_sz);
            return -1;
        }
        if (steps_out) (*steps_out)++;
        tok_off   += n;
        remaining -= n;
    }
    kv_batch_free(b, chunk_sz);
    return 0;
}

/*
 * Run the tensor probe for a given seq_id / kv_slot after prefill completes.
 * Called from the Python binding after Py_END_ALLOW_THREADS so that writes to
 * the shared bind registry (cell_pages_bound, tensor_pages_bound_slot) happen
 * with the GIL held — the same condition under which page_bind_slots() reads them.
 *
 * WHY separate from run_prefill: moving the probe outside the GIL-released block
 * prevents a data race between the C registry write and Python threads calling
 * page_bind_stats / page_bind_slots.  The probe result is best-effort; errors
 * are silently ignored.
 */
void
kv_decode_loop_post_prefill_probe(void *ctx, int32_t seq_id)
{
    KvTensorProbeResult probe;
    kv_tensor_probe_run(ctx, seq_id, seq_id, &probe);
}

int
kv_decode_loop_run_step(
    void    *ctx,
    int32_t  token,
    int32_t  seq_id,
    int32_t  current_pos,
    void    *smpl,
    int32_t *sampled_out,
    int32_t *steps_out)
{
    struct llama_context *lctx = (struct llama_context *)ctx;
    if (kv_page_bind_validate_range(seq_id, current_pos, 1) != 0) {
        return KV_DECODE_LOOP_ERR_BIND;
    }
    llama_batch *b = kv_batch_alloc(1, 1);
    if (!b) return -1;

    kv_batch_fill(b, &token, 1, seq_id, current_pos, 1 /* logits_last=true */);
    if (llama_decode(lctx, *b) != 0) {
        kv_batch_free(b, 1);
        return -1;
    }
    if (steps_out) (*steps_out)++;
    kv_batch_free(b, 1);
    if (smpl != NULL && sampled_out != NULL) {
        *sampled_out = (int32_t)llama_sampler_sample(
            (struct llama_sampler *)smpl, lctx, -1);
    }
    return 0;
}

int
kv_decode_loop_run_batch_step(
    void                *ctx,
    const int32_t       *tokens,
    const int32_t       *seq_ids,
    const int32_t       *positions,
    int32_t              n_entries,
    const void *const   *smpl_ptrs,
    int32_t             *sampled_out,
    int32_t             *steps_out)
{
    /*
     * WHY one llama_decode for N rows: continuous batching under llama_parallel_slots>1
     * — each active sequence contributes one token; logits on every row for sampling.
     */
    if (n_entries <= 0) {
        if (steps_out) {
            *steps_out = 0;
        }
        return 0;
    }
    if (n_entries > KV_DECODE_LOOP_BATCH_MAX) {
        return -1;
    }

    struct llama_context *lctx = (struct llama_context *)ctx;
    for (int32_t i = 0; i < n_entries; i++) {
        if (kv_page_bind_validate_range(seq_ids[i], positions[i], 1) != 0) {
            return KV_DECODE_LOOP_ERR_BIND;
        }
    }

    llama_batch *b = kv_batch_alloc(n_entries, 1);
    if (!b) {
        return -1;
    }

    kv_batch_fill_entries(b, tokens, seq_ids, positions, n_entries, 1);
    if (llama_decode(lctx, *b) != 0) {
        kv_batch_free(b, n_entries);
        return -1;
    }
    if (steps_out) {
        *steps_out = 1;
    }
    kv_batch_free(b, n_entries);

    if (smpl_ptrs != NULL && sampled_out != NULL) {
        for (int32_t i = 0; i < n_entries; i++) {
            const void *s = smpl_ptrs[i];
            if (s != NULL) {
                sampled_out[i] = (int32_t)llama_sampler_sample(
                    (struct llama_sampler *)s, lctx, i);
            } else {
                sampled_out[i] = -1;
            }
        }
    }
    return 0;
}

int32_t
kv_decode_loop_sample(void *smpl, void *ctx)
{
    /* WHY NULL check: Python callers guard smpl/ctx before calling, but
     * direct C callers must not crash; return -1 (invalid token) on bad args. */
    if (!smpl || !ctx) return -1;
    return (int32_t)llama_sampler_sample(
        (struct llama_sampler *)smpl, (struct llama_context *)ctx, -1);
}

int
kv_decode_loop_invalidate_cuda_graphs(void *ctx)
{
    /* WHY: L3 slot clear changes KV while ggml-cuda may reuse a captured graph
     * keyed by cgraph topology. In-process path calls this wrapper; subprocess
     * uses llama-server POST /cuda-graph/invalidate on ctx_tgt instead. */
    if (!ctx) {
        return 0;
    }
    return (int)llama_context_cuda_graph_invalidate((struct llama_context *)ctx);
}

#endif /* ZEROLLAMA_KV_DECODE_LOOP */
