/*
 * Staging llama.cpp KV cell / tensor introspection for zerollama Phase 15 v20/v31/v47.
 *
 * WHY resolve hybrid/iSWA: PA page bind targets attn KV cells. Hybrid and iSWA
 * models wrap llama_kv_cache (or iswa base cache) — dynamic_cast to the wrapper
 * alone returned UNSUPPORTED even though get_base()/get_mem_attn() expose the
 * same cell layout as standard models.
 *
 * WHY v47 alias validate: page_map discovers llama-owned spans; external PA pools
 * must prove zero-copy alias feasibility before v48 overlay bind mutates tensor->data.
 */

#include "llama-kv-ext.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

static llama_kv_cache * llama_kv_ext_resolve_cache(
        llama_memory_t         mem,
        llama_kv_ext_mem_kind * kind_out) {
    if (kind_out) {
        *kind_out = LLAMA_KV_EXT_MEM_NONE;
    }
    if (!mem) {
        return nullptr;
    }

    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        if (kind_out) {
            *kind_out = LLAMA_KV_EXT_MEM_KV_CACHE;
        }
        return kv;
    }

    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        /* WHY base cache: holds full-context attn KV; SWA cache is windowed. */
        if (kind_out) {
            *kind_out = LLAMA_KV_EXT_MEM_ISWA_BASE;
        }
        return iswa->get_base();
    }

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        llama_kv_cache * attn = hybrid->get_mem_attn();
        if (kind_out) {
            *kind_out = attn ? LLAMA_KV_EXT_MEM_HYBRID_ATTN : LLAMA_KV_EXT_MEM_UNSUPPORTED;
        }
        return attn;
    }

    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        llama_kv_cache_iswa * attn = hybrid_iswa->get_mem_attn();
        if (!attn) {
            if (kind_out) {
                *kind_out = LLAMA_KV_EXT_MEM_UNSUPPORTED;
            }
            return nullptr;
        }
        if (kind_out) {
            *kind_out = LLAMA_KV_EXT_MEM_HYBRID_ISWA_BASE;
        }
        return attn->get_base();
    }

    if (kind_out) {
        *kind_out = LLAMA_KV_EXT_MEM_UNSUPPORTED;
    }
    return nullptr;
}

int32_t llama_memory_kv_ext_classify(
        llama_memory_t         mem,
        llama_kv_ext_mem_kind * out_kind) {
    if (!out_kind) {
        return LLAMA_KV_EXT_ARG;
    }
    llama_kv_ext_resolve_cache(mem, out_kind);
    if (*out_kind == LLAMA_KV_EXT_MEM_UNSUPPORTED) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }
    if (*out_kind == LLAMA_KV_EXT_MEM_NONE) {
        return LLAMA_KV_EXT_ARG;
    }
    return LLAMA_KV_EXT_OK;
}

int32_t llama_memory_kv_cell_for_pos(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_pos      pos,
        uint32_t *     out_cell_idx,
        uint32_t *     out_stream) {
    if (!out_cell_idx) {
        return LLAMA_KV_EXT_ARG;
    }

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    const int32_t cell = kv->cell_index_for(seq_id, pos, out_stream);
    if (cell < 0) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    *out_cell_idx = (uint32_t) cell;
    return LLAMA_KV_EXT_OK;
}

int32_t llama_memory_kv_cell_map_range(
        llama_memory_t       mem,
        llama_seq_id         seq_id,
        llama_pos            pos_start,
        llama_pos            pos_end,
        llama_kv_cell_bind * out,
        uint32_t             out_cap,
        uint32_t *           out_count) {
    if (!out || !out_count || pos_end <= pos_start) {
        return LLAMA_KV_EXT_ARG;
    }

    const uint32_t need = (uint32_t) (pos_end - pos_start);
    if (need > out_cap) {
        return LLAMA_KV_EXT_ARG;
    }

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    uint32_t n = 0;
    for (llama_pos pos = pos_start; pos < pos_end; ++pos) {
        uint32_t stream = 0;
        const int32_t cell = kv->cell_index_for(seq_id, pos, &stream);
        if (cell < 0) {
            return LLAMA_KV_EXT_NOT_FOUND;
        }
        out[n].pos      = pos;
        out[n].cell_idx = (uint32_t) cell;
        out[n].stream   = stream;
        ++n;
    }

    *out_count = n;
    return LLAMA_KV_EXT_OK;
}

static uint64_t llama_kv_ext_tensor_bytes(const ggml_tensor * t) {
    if (!t) {
        return 0;
    }
    return (uint64_t) ggml_nbytes(t);
}

int32_t llama_memory_kv_tensor_info(
        llama_memory_t         mem,
        int32_t                kv_layer,
        uint32_t               stream,
        llama_kv_tensor_info * out) {
    if (!out) {
        return LLAMA_KV_EXT_ARG;
    }

    std::memset(out, 0, sizeof(*out));
    out->layer = kv_layer;

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    ggml_tensor * k = kv->kv_tensor_k(kv_layer, stream);
    ggml_tensor * v = kv->kv_tensor_v(kv_layer, stream);
    if (!k || !v || !k->data || !v->data) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    out->k_data        = (uint64_t) (uintptr_t) k->data;
    out->v_data        = (uint64_t) (uintptr_t) v->data;
    out->k_size_bytes  = llama_kv_ext_tensor_bytes(k);
    out->v_size_bytes  = llama_kv_ext_tensor_bytes(v);
    out->ok            = 1;
    return LLAMA_KV_EXT_OK;
}

int32_t llama_memory_kv_n_layers(
        llama_memory_t mem,
        uint32_t *     out_n) {
    if (!out_n) {
        return LLAMA_KV_EXT_ARG;
    }
    *out_n = 0;

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    *out_n = kv->n_kv_layers();
    return LLAMA_KV_EXT_OK;
}

int32_t llama_memory_kv_cache_layout(
        llama_memory_t         mem,
        llama_kv_cache_layout * out) {
    if (!out) {
        return LLAMA_KV_EXT_ARG;
    }

    std::memset(out, 0, sizeof(*out));

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    out->kv_size      = kv->get_size();
    out->n_stream     = kv->get_n_stream();
    out->v_transposed = kv->get_v_trans() ? 1 : 0;
    out->ok           = 1;
    return LLAMA_KV_EXT_OK;
}

static bool llama_kv_ext_cells_contiguous(
        const llama_kv_cell_bind * cells,
        uint32_t                   n) {
    if (n <= 1) {
        return true;
    }
    for (uint32_t i = 1; i < n; ++i) {
        if (cells[i].cell_idx != cells[i - 1].cell_idx + 1) {
            return false;
        }
    }
    return true;
}

static int32_t llama_kv_ext_page_map_contiguous(
        llama_kv_cache *           kv,
        ggml_tensor *              k,
        ggml_tensor *              v,      /* may be null for MLA (no V cache) */
        const llama_kv_cell_bind * cells,
        uint32_t                   n_cells,
        llama_kv_page_map *        out) {
    if (!kv || !k || !cells || n_cells == 0 || !out) {
        return LLAMA_KV_EXT_ARG;
    }
    if (!k->data) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    const uint64_t kv_size = kv->get_size();
    if ((uint64_t) k->ne[1] != kv_size) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }
    if (v && (uint64_t) v->ne[1] != kv_size) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    if (!llama_kv_ext_cells_contiguous(cells, n_cells)) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    for (uint32_t i = 1; i < n_cells; ++i) {
        if (cells[i].stream != cells[0].stream) {
            return LLAMA_KV_EXT_NOT_FOUND;
        }
    }

    const uint32_t cell0 = cells[0].cell_idx;
    const uint32_t cell1 = cells[n_cells - 1].cell_idx;
    if (cell1 >= kv_size) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    const uint64_t k_off = (uint64_t) cell0 * (uint64_t) k->nb[1];
    const uint64_t span  = (uint64_t) n_cells * (uint64_t) k->nb[1];

    out->n_cells         = n_cells;
    out->cell_idx_first  = cell0;
    out->cell_idx_last   = cell1;
    out->stream          = cells[0].stream;
    out->k_data          = (uint64_t) (uintptr_t) ((const char *) k->data + k_off);
    out->k_span_bytes    = span;
    out->v_transposed    = kv->get_v_trans() ? 1 : 0;

    if (v && v->data) {
        const uint64_t v_off  = (uint64_t) cell0 * (uint64_t) v->nb[1];
        const uint64_t vspan  = (uint64_t) n_cells * (uint64_t) v->nb[1];
        out->v_data       = (uint64_t) (uintptr_t) ((const char *) v->data + v_off);
        /* WHY v_span_bytes when v_transposed=1: v->nb[1] is the byte stride between
         * consecutive cell indices (dim 1 stride). v_off..v_off+vspan covers the
         * contiguous byte range that holds this cell block's data, but the embedding
         * values for each cell are NOT laid out contiguously — they are interleaved
         * across n_embd_v_gqa rows at stride kv_size (see set_input_v_idxs).
         * Callers must check v_transposed and use the appropriate scatter/gather
         * pattern; do not memcpy(v_data, vspan) as a flat cell buffer. */
        out->v_span_bytes = vspan;
    } else {
        out->v_data       = 0;
        out->v_span_bytes = 0;
    }

    out->ok              = 1;
    return LLAMA_KV_EXT_OK;
}

int32_t llama_memory_kv_page_map(
        llama_memory_t     mem,
        llama_seq_id       seq_id,
        llama_pos          seq_pos_min,
        uint32_t           page_index,
        uint32_t           block_size,
        int32_t            kv_layer,
        llama_kv_page_map * out) {
    if (!out || block_size == 0 || page_index > 8192) {
        return LLAMA_KV_EXT_ARG;
    }

    std::memset(out, 0, sizeof(*out));
    out->kv_layer = kv_layer;

    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, nullptr);
    if (!kv) {
        return LLAMA_KV_EXT_UNSUPPORTED;
    }

    if (kv_layer < 0 || (uint32_t) kv_layer >= kv->n_kv_layers()) {
        return LLAMA_KV_EXT_ARG;
    }

    const llama_pos seq_max = llama_memory_seq_pos_max(mem, seq_id);
    if (seq_max < 0) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    const llama_pos base = seq_pos_min >= 0 ? seq_pos_min : 0;
    const llama_pos t0   = base + (llama_pos) page_index * (llama_pos) block_size;
    llama_pos       t1   = t0 + (llama_pos) block_size;
    if (t0 > seq_max) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }
    if (t1 > seq_max + 1) {
        t1 = seq_max + 1;
    }
    if (t1 <= t0) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }

    const uint32_t need = (uint32_t) (t1 - t0);
    if (need > 512) {
        return LLAMA_KV_EXT_ARG;
    }

    llama_kv_cell_bind cells[512];
    uint32_t n = 0;
    const int32_t rc = llama_memory_kv_cell_map_range(
        mem, seq_id, t0, t1, cells, need, &n);
    if (rc != LLAMA_KV_EXT_OK || n != need) {
        return rc != LLAMA_KV_EXT_OK ? rc : LLAMA_KV_EXT_NOT_FOUND;
    }

    ggml_tensor * k = kv->kv_tensor_k(kv_layer, cells[0].stream);
    ggml_tensor * v = kv->kv_tensor_v(kv_layer, cells[0].stream);

    out->pos_start = t0;
    out->pos_end   = t1;

    return llama_kv_ext_page_map_contiguous(kv, k, v, cells, n, out);
}

static bool llama_kv_ext_tensor_on_host(const ggml_tensor * t) {
    if (!t || !t->buffer) {
        return false;
    }
    return ggml_backend_buffer_is_host(t->buffer);
}

static void llama_kv_ext_alias_set_blocker(
        llama_kv_page_alias_plan * plan,
        const char *               msg) {
    if (!plan || !msg) {
        return;
    }
    std::snprintf(plan->blocker, sizeof(plan->blocker), "%s", msg);
}

int32_t llama_memory_kv_ext_external_alias_probe(
        llama_kv_ext_external_alias_probe * out) {
    if (!out) {
        return LLAMA_KV_EXT_ARG;
    }
    std::memset(out, 0, sizeof(*out));
#ifdef LLAMA_KV_EXT_EXTERNAL_ALIAS
    out->available    = 1;
    out->validate_api = 1;
    std::snprintf(out->api_name, sizeof(out->api_name), "llama_memory_kv_page_alias_validate");
    return LLAMA_KV_EXT_OK;
#else
    std::snprintf(out->api_name, sizeof(out->api_name), "none");
    return LLAMA_KV_EXT_OK;
#endif
}

int32_t llama_memory_kv_page_alias_validate(
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
        llama_kv_page_alias_plan * out) {
    if (!out) {
        return LLAMA_KV_EXT_ARG;
    }
    if (block_size == 0 || page_index > 8192) {
        return LLAMA_KV_EXT_ARG;
    }

    std::memset(out, 0, sizeof(*out));

#ifndef LLAMA_KV_EXT_EXTERNAL_ALIAS
    llama_kv_ext_alias_set_blocker(out, "external_alias_api_not_linked");
    return LLAMA_KV_EXT_UNSUPPORTED;
#else
    llama_kv_ext_mem_kind kind = LLAMA_KV_EXT_MEM_NONE;
    llama_kv_cache * kv = llama_kv_ext_resolve_cache(mem, &kind);
    if (!kv || kind == LLAMA_KV_EXT_MEM_UNSUPPORTED) {
        out->alias_mode = LLAMA_KV_EXT_ALIAS_BLOCKED_UNSUPPORTED;
        llama_kv_ext_alias_set_blocker(out, "kv_cache_unsupported");
        out->ok = 1;
        return LLAMA_KV_EXT_OK;
    }

    llama_kv_page_map page_map;
    std::memset(&page_map, 0, sizeof(page_map));
    const int32_t map_rc = llama_memory_kv_page_map(
        mem, seq_id, seq_pos_min, page_index, block_size, kv_layer, &page_map);
    if (map_rc != LLAMA_KV_EXT_OK || !page_map.ok) {
        out->alias_mode = (map_rc == LLAMA_KV_EXT_UNSUPPORTED)
            ? LLAMA_KV_EXT_ALIAS_BLOCKED_UNSUPPORTED
            : LLAMA_KV_EXT_ALIAS_BLOCKED_NO_PAGE;
        llama_kv_ext_alias_set_blocker(
            out,
            map_rc == LLAMA_KV_EXT_UNSUPPORTED ? "kv_cache_unsupported" : "page_map_failed");
        return map_rc != LLAMA_KV_EXT_OK ? map_rc : LLAMA_KV_EXT_NOT_FOUND;
    }

    out->llama_k_data   = page_map.k_data;
    out->llama_v_data   = page_map.v_data;
    out->k_span_bytes   = page_map.k_span_bytes;
    out->v_span_bytes   = page_map.v_span_bytes;
    out->v_transposed   = page_map.v_transposed;

    ggml_tensor * k = kv->kv_tensor_k(kv_layer, page_map.stream);
    if (!k || !k->data) {
        out->alias_mode = LLAMA_KV_EXT_ALIAS_BLOCKED_NO_PAGE;
        llama_kv_ext_alias_set_blocker(out, "kv_tensor_not_materialized");
        out->ok = 1;
        return LLAMA_KV_EXT_OK;
    }
    const bool host = llama_kv_ext_tensor_on_host(k);
    out->buffer_host = host ? 1 : 0;

    out->k_spans_match = (ext_k_span_bytes == page_map.k_span_bytes) ? 1 : 0;
    const bool v_absent = page_map.v_data == 0 || page_map.v_span_bytes == 0;
    if (v_absent) {
        out->v_spans_match = (ext_v_data == 0 && ext_v_span_bytes == 0) ? 1 : 0;
    } else {
        out->v_spans_match = (ext_v_span_bytes == page_map.v_span_bytes) ? 1 : 0;
    }

    out->k_same_pointer = (ext_k_data != 0 && ext_k_data == page_map.k_data) ? 1 : 0;
    if (v_absent) {
        out->v_same_pointer = (ext_v_data == 0) ? 1 : 0;
    } else {
        out->v_same_pointer = (ext_v_data != 0 && ext_v_data == page_map.v_data) ? 1 : 0;
    }

    if (!out->k_spans_match || !out->v_spans_match) {
        out->alias_mode = LLAMA_KV_EXT_ALIAS_BLOCKED_SPAN;
        llama_kv_ext_alias_set_blocker(out, "ext_span_mismatch");
        out->ok = 1;
        return LLAMA_KV_EXT_OK;
    }

    if (!host) {
        out->alias_mode = LLAMA_KV_EXT_ALIAS_BLOCKED_DEVICE;
        llama_kv_ext_alias_set_blocker(out, "kv_tensors_on_device_buffer");
        out->ok = 1;
        return LLAMA_KV_EXT_OK;
    }

    if (page_map.v_transposed && !v_absent) {
        out->alias_mode = LLAMA_KV_EXT_ALIAS_BLOCKED_V_TRANS;
        llama_kv_ext_alias_set_blocker(out, "v_transposed_scatter_gather_required");
        out->ok = 1;
        return LLAMA_KV_EXT_OK;
    }

    if (out->k_same_pointer && out->v_same_pointer) {
        out->alias_mode  = LLAMA_KV_EXT_ALIAS_SAME_POINTER;
        out->alias_ready = 1;
        out->ok          = 1;
        return LLAMA_KV_EXT_OK;
    }

    out->alias_mode = LLAMA_KV_EXT_ALIAS_HOST_REBASE;
    llama_kv_ext_alias_set_blocker(out, "host_rebase_overlay_bind_not_implemented");
    out->ok = 1;
    return LLAMA_KV_EXT_OK;
#endif /* LLAMA_KV_EXT_EXTERNAL_ALIAS */
}

int32_t llama_memory_kv_ext_writable_bind_probe(
        int32_t  * out_available,
        char     * out_api_name,
        uint32_t   name_cap) {
    if (!out_available) {
        return LLAMA_KV_EXT_ARG;
    }
    *out_available = 0;
    if (out_api_name && name_cap > 0) {
        out_api_name[0] = '\0';
    }

#ifdef LLAMA_KV_EXT_WRITABLE_PAGE_MAP
    *out_available = 1;
    if (out_api_name && name_cap > 0) {
        std::snprintf(out_api_name, name_cap, "llama_memory_kv_page_map");
    }
    return LLAMA_KV_EXT_OK;
#else
    if (out_api_name && name_cap > 0) {
        std::snprintf(out_api_name, name_cap, "none");
    }
    return LLAMA_KV_EXT_OK;
#endif
}

/*
 * Phase 15 v48 — CPU-only donor-buffer registry.
 *
 * WHY a static registry instead of threading a parameter through
 * llama_context_params -> llama_model::create_memory -> every arch's
 * llama_kv_cache constructor call site: that would be a public-signature
 * change touching ~10+ call sites in llama-model.cpp. An additive
 * process-level registration hook matches how every other Phase 15 staging
 * API (v20/v34/v47) has extended llama-kv-ext.h without changing core
 * constructors.
 *
 * Consumed exactly once per registered donor by the first CPU-buft KV cache
 * allocation group that fits (see llama_kv_ext_donor_try_consume, called from
 * llama_kv_cache's ctor allocation loop in llama-kv-cache.cpp).
 */
#ifdef LLAMA_KV_EXT_DONOR_BUFFER

struct llama_kv_ext_donor_entry {
    void *   ptr        = nullptr;
    uint64_t size        = 0;
    bool     registered = false;
    bool     bound      = false;
    uint64_t bytes_used  = 0;
};

static std::mutex                                             g_donor_mutex;
static llama_kv_ext_donor_entry                                g_donor_registry[LLAMA_KV_EXT_DONOR_MAX];
static uint32_t                                                g_donor_next_id = 1;
static std::map<uint32_t, size_t>                              g_donor_id_to_slot;

int32_t llama_kv_ext_register_donor_buffer(
        void *     ptr,
        uint64_t   size,
        uint32_t * out_donor_id) {
    if (!ptr || size == 0 || !out_donor_id) {
        return LLAMA_KV_EXT_ARG;
    }

    std::lock_guard<std::mutex> lock(g_donor_mutex);

    size_t slot = LLAMA_KV_EXT_DONOR_MAX;
    for (size_t i = 0; i < LLAMA_KV_EXT_DONOR_MAX; ++i) {
        if (!g_donor_registry[i].registered) {
            slot = i;
            break;
        }
    }
    if (slot == LLAMA_KV_EXT_DONOR_MAX) {
        return LLAMA_KV_EXT_ARG;
    }

    g_donor_registry[slot] = llama_kv_ext_donor_entry{ptr, size, true, false, 0};
    const uint32_t id = g_donor_next_id++;
    g_donor_id_to_slot[id] = slot;
    *out_donor_id = id;
    return LLAMA_KV_EXT_OK;
}

int32_t llama_kv_ext_unregister_donor_buffer(uint32_t donor_id) {
    std::lock_guard<std::mutex> lock(g_donor_mutex);

    auto it = g_donor_id_to_slot.find(donor_id);
    if (it == g_donor_id_to_slot.end()) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }
    g_donor_registry[it->second] = llama_kv_ext_donor_entry{};
    g_donor_id_to_slot.erase(it);
    return LLAMA_KV_EXT_OK;
}

int32_t llama_kv_ext_donor_buffer_status(
        uint32_t    donor_id,
        int32_t *   out_bound,
        uint64_t *  out_bytes_used) {
    if (!out_bound) {
        return LLAMA_KV_EXT_ARG;
    }
    std::lock_guard<std::mutex> lock(g_donor_mutex);

    auto it = g_donor_id_to_slot.find(donor_id);
    if (it == g_donor_id_to_slot.end()) {
        return LLAMA_KV_EXT_NOT_FOUND;
    }
    const auto & entry = g_donor_registry[it->second];
    *out_bound = entry.bound ? 1 : 0;
    if (out_bytes_used) {
        *out_bytes_used = entry.bytes_used;
    }
    return LLAMA_KV_EXT_OK;
}

/*
 * Called from llama_kv_cache's CPU-buft allocation loop. Finds the first
 * unused donor with size >= required_size, wraps it via
 * ggml_backend_cpu_buffer_from_ptr, and marks it consumed. Returns nullptr
 * (no side effects) when no donor fits — caller falls through to normal
 * ggml_backend_alloc_ctx_tensors_from_buft allocation.
 */
ggml_backend_buffer_t llama_kv_ext_donor_try_consume(size_t required_size) {
    std::lock_guard<std::mutex> lock(g_donor_mutex);

    for (size_t i = 0; i < LLAMA_KV_EXT_DONOR_MAX; ++i) {
        auto & entry = g_donor_registry[i];
        if (entry.registered && !entry.bound && entry.size >= required_size) {
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(entry.ptr, entry.size);
            if (!buf) {
                continue;
            }
            entry.bound      = true;
            entry.bytes_used = required_size;
            return buf;
        }
    }
    return nullptr;
}

/*
 * Phase 15 v49 — device-buft donor consume (Metal unified-memory zero-copy).
 *
 * WHY this exists: v48 only tried ggml_backend_cpu_buffer_from_ptr for
 * ggml_backend_buft_is_host(buft) groups. On Apple Silicon, the Metal device
 * ALSO exposes a host-ptr-to-buffer wrapper (caps.buffer_from_host_ptr,
 * ggml_backend_dev_buffer_from_host_ptr -> newBufferWithBytesNoCopy +
 * MTLResourceStorageModeShared) — the exact same mechanism llama-model.cpp's
 * mmap-weight-loading path already uses in production (see the
 * buffer_from_host_ptr_supported branch in llama_model::load_tensors). Metal's
 * KV buft groups report is_host()==false (they are NOT literally "host"
 * buffers — ggml_backend_buffer_is_host gates the CPU compute-op dispatch
 * path, unrelated to whether the *backing memory* happens to be host RAM), so
 * v48's is_host() check never reaches this path; a separate device-capability
 * check is required.
 *
 * CUDA does not implement buffer_from_host_ptr (discrete VRAM, no unified
 * memory) — dev->iface.buffer_from_host_ptr is NULL there, so this function
 * naturally returns nullptr for CUDA bufts without any CUDA-specific branch.
 *
 * max_tensor_size mirrors llama-model.cpp's ggml_get_max_tensor_size(ctx) —
 * Metal buffers above the device's max_buffer_size are split into
 * overlapping windows internally by ggml_metal_buffer_map; passing the
 * largest single tensor size lets that split remain safe (no tensor straddles
 * a window boundary).
 */
ggml_backend_buffer_t llama_kv_ext_donor_try_consume_dev(
        ggml_backend_dev_t dev,
        size_t             required_size,
        size_t             max_tensor_size) {
    if (!dev) {
        return nullptr;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_donor_mutex);

    for (size_t i = 0; i < LLAMA_KV_EXT_DONOR_MAX; ++i) {
        auto & entry = g_donor_registry[i];
        if (entry.registered && !entry.bound && entry.size >= required_size) {
            ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(
                    dev, entry.ptr, entry.size, max_tensor_size);
            if (!buf) {
                continue;
            }
            entry.bound      = true;
            entry.bytes_used = required_size;
            return buf;
        }
    }
    return nullptr;
}

#else /* !LLAMA_KV_EXT_DONOR_BUFFER */

int32_t llama_kv_ext_register_donor_buffer(
        void *, uint64_t, uint32_t *) {
    return LLAMA_KV_EXT_UNSUPPORTED;
}

int32_t llama_kv_ext_unregister_donor_buffer(uint32_t) {
    return LLAMA_KV_EXT_UNSUPPORTED;
}

int32_t llama_kv_ext_donor_buffer_status(
        uint32_t, int32_t *, uint64_t *) {
    return LLAMA_KV_EXT_UNSUPPORTED;
}

ggml_backend_buffer_t llama_kv_ext_donor_try_consume_dev(
        ggml_backend_dev_t, size_t, size_t) {
    return nullptr;
}

#endif /* LLAMA_KV_EXT_DONOR_BUFFER */
