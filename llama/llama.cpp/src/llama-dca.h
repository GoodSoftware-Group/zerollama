#pragma once

// Dual Chunk Attention helpers (Qwen long-ctx / 1M). Dense path only.
// RoPE remaps match SGLang DualChunkRotaryEmbedding (rope_variant.py).

#include "llama-hparams.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace llama_dca {

inline uint32_t chunk_len(const llama_hparams & hp) {
    return hp.dca_chunk_len();
}

// P_k = pos % chunk_len
inline llama_pos pos_k(llama_pos pos, uint32_t c_len) {
    return c_len ? (pos % (llama_pos) c_len) : pos;
}

// P_q_intra = pos % chunk_len
inline llama_pos pos_q_intra(llama_pos pos, uint32_t c_len) {
    return pos_k(pos, c_len);
}

// P_q_succ = min((pos % chunk_len) + chunk_len, chunk_size)
inline llama_pos pos_q_succ(llama_pos pos, uint32_t c_len, uint32_t chunk_size) {
    if (!c_len) {
        return pos;
    }
    const llama_pos local = pos % (llama_pos) c_len;
    return std::min(local + (llama_pos) c_len, (llama_pos) chunk_size);
}

// P_q_inter = chunk_size (constant; dense path)
inline llama_pos pos_q_inter(uint32_t chunk_size) {
    return (llama_pos) chunk_size;
}

// s(L) = max(1, 0.1 * log(L / L0) + 1); L0 from dca_orig_ctx
inline float length_scale(uint32_t seq_len, uint32_t l0) {
    if (l0 == 0 || seq_len == 0) {
        return 1.0f;
    }
    return std::max(1.0f, 0.1f * std::log(float(seq_len) / float(l0)) + 1.0f);
}

// Decode / per-query chunk index n = (S-1) // chunk_len with S = pos+1 → n = pos // chunk_len
inline uint32_t chunk_index(llama_pos pos, uint32_t c_len) {
    return c_len ? uint32_t(pos) / c_len : 0;
}

enum class mask_kind {
    INTRA, // same chunk as query; causal (key_pos <= query_pos)
    SUCC,  // previous chunk only (n>=1); non-causal within that chunk
    INTER, // older chunks [0, (n-1)*c) (n>=2); non-causal
};

// Return true if key at key_pos should attend for query at query_pos under DCA stage.
inline bool allow_key(
        mask_kind kind,
        llama_pos query_pos,
        llama_pos key_pos,
        uint32_t  c_len) {
    if (!c_len) {
        return key_pos <= query_pos;
    }
    const uint32_t n = chunk_index(query_pos, c_len);
    const uint32_t kc = uint32_t(key_pos) / c_len;

    switch (kind) {
        case mask_kind::INTRA:
            return kc == n && key_pos <= query_pos;
        case mask_kind::SUCC:
            return n >= 1 && kc == (n - 1);
        case mask_kind::INTER:
            return n >= 2 && uint32_t(key_pos) < (n - 1) * c_len;
    }
    return false;
}

} // namespace llama_dca
