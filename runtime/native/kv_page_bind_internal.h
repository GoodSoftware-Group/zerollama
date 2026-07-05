/* Internal page bind registry (Phase 15 v8/v19). Shared by kv_block_pool.c and kv_tensor_probe.c. */

#ifndef KV_PAGE_BIND_INTERNAL_H
#define KV_PAGE_BIND_INTERNAL_H

#define KV_MAX_PAGE_BINDS 32
#define KV_MAX_PAGES_PER_BIND 8192  /* 131072 ctx @ block_size=16 */

#define KV_TENSOR_BLOCKER_NONE 0
#define KV_TENSOR_BLOCKER_NO_PAGE_API 1
#define KV_TENSOR_BLOCKER_UNSUPPORTED_MEM 2
#define KV_TENSOR_BLOCKER_CELL_GAP 3
#define KV_TENSOR_BLOCKER_NO_TENSOR 4
#define KV_TENSOR_BLOCKER_MISALIGNED 5

typedef struct {
    int active;
    int kv_slot;
    int block_size;
    int num_pages;
    int block_ids[KV_MAX_PAGES_PER_BIND];
    int cell_pages_bound;          /* v20: PA pages resolved to llama cells */
    int tensor_pages_bound_slot;   /* v20: K/V tensor backing verified */
    int physical_pages_bound;      /* v33: writable tensor spans for all live pages */
    int physical_pages_mapped;     /* v33: count of pages with resolved writable spans */
} KvPageBind;

KvPageBind *kv_find_page_bind(int kv_slot);

/*
 * Validate token positions [token_start, token_start + n_tokens) against the
 * registered PA page table for kv_slot.
 *
 * Returns  0 — ok, or no bind registered (nothing to check)
 *         -2 — position exceeds bound pages
 *
 * WHY endpoint-only check: positions are contiguous; validating first and last
 * index is sufficient (matches Python validate_token_positions).
 */
int kv_page_bind_validate_range(int kv_slot, int token_start, int n_tokens);

#endif /* KV_PAGE_BIND_INTERNAL_H */
