#pragma once

#include <cstdint>

// Spacing of prompt marks on a model that cannot rewind.
// `mark` is one recurrent cell. `per_token` is one token of the attention
// cache. The gap is wide enough that a mark costs about as much memory as
// the attention cache between two marks, and it stays a multiple of 256
// between 256 and 8192. An unknown mark keeps the 256-token minimum.
// A recurrent model with no attention cache uses one byte per token, so the
// gap widens to the cap.

// True when this cache is a sliding window that overwrites old positions.
// `cells` is the cache length. A full-size cache is longer than one window
// per sequence plus a batch of slack, and that cache rewinds in place.
inline bool llama_mark_window(uint32_t n_swa, uint32_t n_seq, uint64_t cells) {
    if (n_swa == 0 || n_seq == 0 || cells == 0) {
        return false;
    }
    const uint64_t ring = (uint64_t) n_swa * (uint64_t) n_seq + 2048;
    return cells <= ring;
}

inline int32_t llama_mark_spacing(uint64_t mark, uint64_t per_token) {
    if (mark == 0) {
        return 256;
    }
    if (per_token == 0) {
        per_token = 1;
    }
    uint64_t n = mark / per_token;
    if (n >= 8192) {
        return 8192;
    }
    n = ((n + 255) / 256) * 256;
    if (n < 256) {
        n = 256;
    }
    return (int32_t) n;
}

#if defined(LLAMA_MARK_GRID_TEST)
#include <cstdio>

int main() {
    if (llama_mark_spacing(0, 1000) != 256) {
        return 1;
    }
    if (llama_mark_spacing(100, 1000) != 256) {
        return 1;
    }
    if (llama_mark_spacing(256ull * 1000, 1000) != 256) {
        return 1;
    }
    if (llama_mark_spacing(257ull * 1000, 1000) != 512) {
        return 1;
    }
    if (llama_mark_spacing(512ull * 1000, 1000) != 512) {
        return 1;
    }
    if (llama_mark_spacing(100000ull * 1000, 1000) != 8192) {
        return 1;
    }
    if (llama_mark_spacing(8192, 1) != 8192) {
        return 1;
    }
    if (llama_mark_spacing(5000, 0) != 5120) {
        return 1;
    }
    if (llama_mark_window(0, 1, 4608) || llama_mark_window(4096, 0, 4608) || llama_mark_window(4096, 1, 0)) {
        return 1;
    }
    // A ring of one window plus a batch.
    if (!llama_mark_window(4096, 1, 4608)) {
        return 1;
    }
    // One window per sequence, unified.
    if (!llama_mark_window(4096, 4, 4096ull * 4 + 512)) {
        return 1;
    }
    // A full context is longer than the window and rewinds in place.
    if (llama_mark_window(4096, 1, 8192)) {
        return 1;
    }
    std::printf("mark grid ok\n");
    return 0;
}
#endif
