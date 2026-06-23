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
