/*
 * Phase 15 v20/v21 — PA page → llama KV cell + tensor bind (llama-kv-ext.h).
 */

#ifdef ZEROLLAMA_KV_DECODE_LOOP

#include <string.h>

#include "llama.h"
#include "llama-kv-ext.h"

#include "kv_tensor_probe.h"   /* pulls in kv_page_bind_internal.h */

#define KV_TENSOR_BIND_MAX_CELLS 512

#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
static int kv_page_bind_materialize_writable(
    void *ctx,
    int32_t seq_id,
    int32_t kv_slot,
    KvTensorProbeResult *out);
#endif

static void
kv_tensor_bind_attempt(
    struct llama_context *lctx,
    int32_t seq_id,
    int32_t kv_slot,
    KvTensorProbeResult *out)
{
    /*
     * Order matters:
     *   1. Guard on pre-conditions; set accurate blocker before returning.
     *   2. Find PA registry entry.
     *   3. Clear stale bind flags.
     *   4. Fetch llama memory.
     *   5. Range-map cells page by page.
     *   6. Resolve stream, verify tensor backing.
     */

    if (!out) {
        return;
    }

    /* Non-null memory is required; lctx==NULL is a caller bug. */
    if (!lctx) {
        out->blocker_code = KV_TENSOR_BLOCKER_NO_PAGE_API;
        return;
    }
    if (!out->memory_non_null) {
        out->blocker_code = KV_TENSOR_BLOCKER_NO_PAGE_API;
        return;
    }

    /* WHY: accurate blocker — misalignment means PA cap was exceeded, not that
     * the ext API is absent.  Operator correlates with "misaligned" status. */
    if (!out->aligned) {
        out->blocker_code = KV_TENSOR_BLOCKER_MISALIGNED;
        return;
    }

    KvPageBind *bind = kv_find_page_bind((int)kv_slot);
    if (bind == NULL || !bind->active || bind->num_pages <= 0 || bind->block_size <= 0) {
        /* No registered PA bind for this slot — nothing to attempt. */
        return;
    }

    /* WHY clear before attempt: stale flags from a prior probe must not leak
     * into page_bind_stats() when this attempt fails partway through. */
    bind->cell_pages_bound = 0;
    bind->tensor_pages_bound_slot = 0;
    bind->physical_pages_bound = 0;
    bind->physical_pages_mapped = 0;

    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        out->blocker_code = KV_TENSOR_BLOCKER_NO_PAGE_API;
        return;
    }

    const int32_t bs = bind->block_size;
    /* WHY: block_size > 512 would overflow the stack array; guard here rather
     * than silently producing a wrong result. */
    if (bs > KV_TENSOR_BIND_MAX_CELLS) {
        out->blocker_code = KV_TENSOR_BLOCKER_CELL_GAP;
        return;
    }

    /* No live llama cells yet — cannot establish a cell bind. */
    if (out->llama_token_cells <= 0) {
        return;
    }

    /* Only probe the pages that are actually populated by live llama cells.
     * Tail pages beyond llama_token_cells are reserved PA but empty KV. */
    int32_t pages_live = (out->llama_token_cells + bs - 1) / bs;
    if (pages_live > bind->num_pages) {
        pages_live = bind->num_pages;
    }

    const llama_pos base = (out->seq_pos_min >= 0) ? (llama_pos)out->seq_pos_min : 0;
    /* WHY guard against INT32_MAX overflow: llama_pos is int32_t;
     * seq_pos_max + 1 would wrap.  Use the actual cell count instead. */
    const llama_pos seq_end = base + (llama_pos)out->llama_token_cells;

    llama_kv_cell_bind stack_cells[KV_TENSOR_BIND_MAX_CELLS];
    int all_cells = 1;

    for (int p = 0; p < pages_live && all_cells; p++) {
        const llama_pos t0 = base + (llama_pos)(p * bs);
        llama_pos t1 = t0 + (llama_pos)bs;
        if (t1 > seq_end) {
            t1 = seq_end;
        }
        if (t1 <= t0) {
            continue;
        }
        const uint32_t need = (uint32_t)(t1 - t0);
        /* Redundant guard: bs <= KV_TENSOR_BIND_MAX_CELLS checked above,
         * last page may be smaller; need <= bs <= 512. */
        if (need > KV_TENSOR_BIND_MAX_CELLS) {
            all_cells = 0;
            out->blocker_code = KV_TENSOR_BLOCKER_CELL_GAP;
            break;
        }
        uint32_t n = 0;
        const int32_t rc = llama_memory_kv_cell_map_range(
            mem,
            (llama_seq_id)seq_id,
            t0,
            t1,
            stack_cells,
            need,
            &n);
        if (rc != LLAMA_KV_EXT_OK || n != need) {
            all_cells = 0;
            out->blocker_code = (rc == LLAMA_KV_EXT_UNSUPPORTED)
                ? KV_TENSOR_BLOCKER_UNSUPPORTED_MEM
                : KV_TENSOR_BLOCKER_CELL_GAP;
        }
    }

    if (!all_cells) {
        bind->cell_pages_bound = 0;
        bind->tensor_pages_bound_slot = 0;
        bind->physical_pages_bound = 0;
        bind->physical_pages_mapped = 0;
        return;
    }

    out->cell_pages_bound = 1;
    bind->cell_pages_bound = 1;

    /* Resolve the stream index for n_stream>1 contexts so we get the correct
     * per-stream 2D K/V tensor view rather than the full 3D parent tensor. */
    uint32_t seq_stream = 0;
    if (out->seq_pos_min >= 0) {
        uint32_t found_cell = 0;
        uint32_t found_stream = 0;
        if (llama_memory_kv_cell_for_pos(
                mem,
                (llama_seq_id)seq_id,
                (llama_pos)out->seq_pos_min,
                &found_cell,
                &found_stream) == LLAMA_KV_EXT_OK) {
            seq_stream = found_stream;
        }
    }
    out->kv_stream = (int32_t)seq_stream;

    uint32_t n_layers = 0;
    if (llama_memory_kv_n_layers(mem, &n_layers) != LLAMA_KV_EXT_OK || n_layers == 0) {
        bind->tensor_pages_bound_slot = 0;
        out->blocker_code = KV_TENSOR_BLOCKER_NO_TENSOR;
        return;
    }
    out->kv_n_layers = (int32_t)n_layers;

    int32_t verified = 0;
    for (uint32_t layer = 0; layer < n_layers; layer++) {
        llama_kv_tensor_info info;
        if (llama_memory_kv_tensor_info(mem, (int32_t)layer, seq_stream, &info) != LLAMA_KV_EXT_OK
            || !info.ok) {
            break;
        }
        verified++;
        if (layer == 0) {
            out->kv_k_data_layer0 = info.k_data;
            out->kv_v_data_layer0 = info.v_data;
        }
    }
    out->tensor_layers_verified = verified;

    if (verified != (int32_t)n_layers) {
        bind->tensor_pages_bound_slot = 0;
        out->blocker_code = KV_TENSOR_BLOCKER_NO_TENSOR;
        return;
    }

    out->tensor_pages_bound = 1;
    bind->tensor_pages_bound_slot = 1;
    out->blocker_code = KV_TENSOR_BLOCKER_NONE;
#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
    kv_page_bind_materialize_writable(lctx, seq_id, kv_slot, out);
#endif
    return;
}

/*
 * Last-probe snapshot: indexed by bind-table position (0..KV_MAX_PAGE_BINDS-1),
 * NOT by kv_slot value. WHY: kv_slot is an opaque scheduler integer that can be
 * arbitrarily large; using it as an array index would overflow. We store the
 * kv_slot value inside the entry so callers can filter or return it accurately.
 */
typedef struct {
    int valid;
    int kv_slot;
    KvTensorProbeResult probe;
} KvLastProbeSlot;

static KvLastProbeSlot g_last_probes[KV_MAX_PAGE_BINDS];

static void
kv_tensor_probe_last_save(int kv_slot, const KvTensorProbeResult *probe)
{
    if (probe == NULL) {
        return;
    }
    /* Find the bind-table index for this kv_slot to use as the array index. */
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_last_probes[i].valid && g_last_probes[i].kv_slot == kv_slot) {
            g_last_probes[i].probe = *probe;
            return;
        }
    }
    /* No existing entry — find a free slot. */
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (!g_last_probes[i].valid) {
            g_last_probes[i].valid    = 1;
            g_last_probes[i].kv_slot  = kv_slot;
            g_last_probes[i].probe    = *probe;
            return;
        }
    }
    /* Table full: overwrite the first entry (oldest). */
    g_last_probes[0].kv_slot = kv_slot;
    g_last_probes[0].probe   = *probe;
}

int
kv_tensor_probe_last_get(int kv_slot, KvTensorProbeResult *out)
{
    if (out == NULL) {
        return -1;
    }
    for (int i = 0; i < KV_MAX_PAGE_BINDS; i++) {
        if (g_last_probes[i].valid && g_last_probes[i].kv_slot == kv_slot) {
            *out = g_last_probes[i].probe;
            return 0;
        }
    }
    return 1; /* not found */
}

int
kv_tensor_probe_last_get_by_index(int idx, int *out_kv_slot, KvTensorProbeResult *out)
{
    if (out == NULL || out_kv_slot == NULL || idx < 0 || idx >= KV_MAX_PAGE_BINDS) {
        return -1;
    }
    if (!g_last_probes[idx].valid) {
        return 1;
    }
    *out_kv_slot = g_last_probes[idx].kv_slot;
    *out         = g_last_probes[idx].probe;
    return 0;
}

int
kv_tensor_probe_run(void *ctx, int32_t seq_id, int32_t kv_slot, KvTensorProbeResult *out)
{
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->seq_pos_min = -1;
    out->seq_pos_max = -1;
    out->blocker_code = KV_TENSOR_BLOCKER_NO_PAGE_API;

    if (!ctx) {
        return 0;
    }

    struct llama_context *lctx = (struct llama_context *)ctx;
    llama_memory_t mem = llama_get_memory(lctx);
    out->memory_non_null = (mem != NULL) ? 1 : 0;
    if (mem) {
        llama_kv_ext_mem_kind kind = LLAMA_KV_EXT_MEM_NONE;
        llama_memory_kv_ext_classify(mem, &kind);
        out->memory_kind = (int32_t)kind;
        out->can_shift = llama_memory_can_shift(mem) ? 1 : 0;
        out->seq_pos_min = (int32_t)llama_memory_seq_pos_min(mem, (llama_seq_id)seq_id);
        out->seq_pos_max = (int32_t)llama_memory_seq_pos_max(mem, (llama_seq_id)seq_id);
        llama_kv_cache_layout layout;
        if (llama_memory_kv_cache_layout(mem, &layout) == LLAMA_KV_EXT_OK && layout.ok) {
            out->kv_v_transposed = layout.v_transposed;
            out->kv_cache_kv_size = layout.kv_size;
            out->kv_cache_n_stream = layout.n_stream;
        }
    }

    KvPageBind *bind = kv_find_page_bind((int)kv_slot);
    if (bind != NULL && bind->active) {
        out->pa_pages_registered = bind->num_pages;
        out->pa_block_size = bind->block_size;
    }

    if (out->seq_pos_max >= 0) {
        int32_t lo = out->seq_pos_min >= 0 ? out->seq_pos_min : 0;
        out->llama_token_cells = out->seq_pos_max - lo + 1;
        if (out->pa_block_size > 0 && out->pa_pages_registered > 0) {
            int32_t pages_need =
                (out->llama_token_cells + out->pa_block_size - 1) / out->pa_block_size;
            out->pages_fit = (pages_need <= out->pa_pages_registered) ? 1 : 0;
            int32_t pa_cap = out->pa_pages_registered * out->pa_block_size;
            out->aligned = (out->llama_token_cells <= pa_cap) ? 1 : 0;
        } else {
            /* No PA bind registered — accounting cannot verify, treat as aligned. */
            out->pages_fit = 1;
            out->aligned = 1;
        }
    } else {
        /* Sequence has no cells yet — treat as aligned (nothing to check). */
        out->llama_token_cells = 0;
        out->pages_fit = 1;
        out->aligned = 1;
    }

    kv_tensor_bind_attempt(lctx, seq_id, kv_slot, out);
    if (out->tensor_pages_bound) {
        kv_tensor_probe_last_save(kv_slot, out);
    }
    return 0;
}

#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
static int
kv_page_bind_materialize_writable(
    void *ctx,
    int32_t seq_id,
    int32_t kv_slot,
    KvTensorProbeResult *out)
{
    if (!ctx || !out) {
        return -1;
    }

    KvPageBind *bind = kv_find_page_bind((int)kv_slot);
    if (bind == NULL || !bind->active || bind->num_pages <= 0 || bind->block_size <= 0) {
        return 0;
    }
    if (!out->tensor_pages_bound || out->llama_token_cells <= 0) {
        return 0;
    }

    struct llama_context *lctx = (struct llama_context *)ctx;
    llama_memory_t mem = llama_get_memory(lctx);
    if (!mem) {
        return 0;
    }

    const int32_t bs = bind->block_size;
    int32_t pages_live = (out->llama_token_cells + bs - 1) / bs;
    if (pages_live > bind->num_pages) {
        pages_live = bind->num_pages;
    }

    const llama_pos base = (out->seq_pos_min >= 0) ? (llama_pos)out->seq_pos_min : 0;
    int mapped_pages = 0;

    uint32_t n_layers = 0;
    if (llama_memory_kv_n_layers(mem, &n_layers) != LLAMA_KV_EXT_OK || n_layers == 0) {
        return 0;
    }

    for (int p = 0; p < pages_live; p++) {
        int layers_ok = 0;
        for (uint32_t layer = 0; layer < n_layers; layer++) {
            llama_kv_page_map page_map;
            const int32_t rc = llama_memory_kv_page_map(
                mem,
                (llama_seq_id)seq_id,
                base,
                (uint32_t)p,
                (uint32_t)bs,
                (int32_t)layer,
                &page_map);
            if (rc == LLAMA_KV_EXT_OK && page_map.ok) {
                layers_ok++;
            }
        }
        if (layers_ok == (int)n_layers) {
            mapped_pages++;
        }
    }

    bind->physical_pages_mapped = mapped_pages;
    bind->physical_pages_bound = (mapped_pages > 0 && mapped_pages == pages_live) ? 1 : 0;
    out->physical_pages_mapped = mapped_pages;
    out->physical_pages_bound = bind->physical_pages_bound;
    return 0;
}
#endif /* LLAMA_KV_EXT_WRITABLE_PAGE_MAP */

#endif /* ZEROLLAMA_KV_DECODE_LOOP */
