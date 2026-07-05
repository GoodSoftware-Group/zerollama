/*
 * Staging llama.cpp KV cell / tensor introspection for zerollama Phase 15 v20/v31.
 *
 * WHY resolve hybrid/iSWA: PA page bind targets attn KV cells. Hybrid and iSWA
 * models wrap llama_kv_cache (or iswa base cache) — dynamic_cast to the wrapper
 * alone returned UNSUPPORTED even though get_base()/get_mem_attn() expose the
 * same cell layout as standard models.
 */

#include "llama-kv-ext.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>

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
        ggml_tensor *              v,
        const llama_kv_cell_bind * cells,
        uint32_t                   n_cells,
        llama_kv_page_map *        out) {
    if (!kv || !k || !v || !cells || n_cells == 0 || !out) {
        return LLAMA_KV_EXT_ARG;
    }
    if (!k->data || !v->data) {
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
        out->v_data           = (uint64_t) (uintptr_t) ((const char *) v->data + v_off);
        /* WHY same span for v_trans: cells are contiguous along dim 1; full V page
         * requires strided access with row stride kv_size when v_transposed=1. */
        out->v_span_bytes     = vspan;
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
