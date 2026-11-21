#pragma once

// Whether a longer cached prompt makes a shorter one redundant.
// Window 0 rewinds to any shorter length. Window -1 returns only to a saved
// mark or to its own end. A positive window is a sliding window: a state no
// longer than it still rewinds, and a longer one has started to overwrite.
// LLAMA_WARM_COVER_TEST compiles the checks below.

#include <cstddef>
#include <cstdint>

// Longest length at or before `len` that this entry can restore. `end` is
// the saved state's length. `marks` are checkpoint lengths.
inline int64_t llama_warm_reach(bool rewinds, int64_t end, const int64_t * marks, size_t n_marks, int64_t len) {
    if (len > end) {
        len = end;
    }
    if (len <= 0) {
        return 0;
    }
    if (rewinds || len == end) {
        return len;
    }
    int64_t best = 0;
    if (marks != nullptr) {
        for (size_t i = 0; i < n_marks; i++) {
            if (marks[i] <= len && marks[i] > best) {
                best = marks[i];
            }
        }
    }
    return best;
}

// A state of `n_tokens` can be cut to any shorter length.
inline bool llama_warm_in_place(int32_t window, size_t n_tokens) {
    if (window < 0) {
        return false;
    }
    if (window == 0) {
        return true;
    }
    return n_tokens <= (size_t) window;
}

// True when `neu` starts with `old` and can restore every length `old` can,
// including `old`'s end. A short state that still rewinds is not covered by
// a longer one that has wrapped: the short one is the only way back to the
// lengths in between. An equal-length entry is not covered.
inline bool llama_warm_covers(
        int32_t window,
        const int32_t * old_tokens, size_t old_n,
        const int64_t * old_marks, size_t old_marks_n,
        const int32_t * new_tokens, size_t new_n,
        const int64_t * new_marks, size_t new_marks_n) {
    if (old_n == 0 || old_n >= new_n || old_tokens == nullptr || new_tokens == nullptr) {
        return false;
    }
    for (size_t i = 0; i < old_n; i++) {
        if (old_tokens[i] != new_tokens[i]) {
            return false;
        }
    }
    const bool old_place = llama_warm_in_place(window, old_n);
    const bool new_place = llama_warm_in_place(window, new_n);
    if (old_place && !new_place) {
        return false;
    }
    auto reachable = [&](int64_t m) {
        return llama_warm_reach(new_place, (int64_t) new_n, new_marks, new_marks_n, m) == m;
    };
    if (!reachable((int64_t) old_n)) {
        return false;
    }
    if (old_marks != nullptr) {
        for (size_t i = 0; i < old_marks_n; i++) {
            if (!reachable(old_marks[i])) {
                return false;
            }
        }
    }
    return true;
}

#if defined(LLAMA_WARM_COVER_TEST)
#include <cstdio>

int main() {
    const int32_t old_t[] = { 1, 2, 3 };
    const int32_t neu_t[] = { 1, 2, 3, 4, 5 };
    const int32_t other[] = { 1, 9, 3, 4, 5 };
    if (!llama_warm_covers(0, old_t, 3, nullptr, 0, neu_t, 5, nullptr, 0)) {
        return 1;
    }
    if (llama_warm_covers(0, old_t, 3, nullptr, 0, other, 5, nullptr, 0)) {
        return 1;
    }
    if (llama_warm_covers(0, old_t, 3, nullptr, 0, old_t, 3, nullptr, 0)) {
        return 1;
    }
    // No mark at token 3, so the shorter state is the only way back there.
    if (llama_warm_covers(-1, old_t, 3, nullptr, 0, neu_t, 5, nullptr, 0)) {
        return 1;
    }
    const int64_t at_end[] = { 3 };
    if (!llama_warm_covers(-1, old_t, 3, nullptr, 0, neu_t, 5, at_end, 1)) {
        return 1;
    }
    const int32_t wrapped[] = { 1, 2, 3, 4, 5, 6 };
    // Three tokens still fit in a window of 4. Six have wrapped, so they
    // cannot replace the shorter state.
    if (llama_warm_covers(4, old_t, 3, nullptr, 0, wrapped, 6, nullptr, 0)) {
        return 1;
    }
    if (!llama_warm_covers(8, old_t, 3, nullptr, 0, neu_t, 5, nullptr, 0)) {
        return 1;
    }
    const int64_t old_marks[] = { 2 };
    const int64_t grid[] = { 2, 4 };
    const int32_t long_t[] = { 1, 2, 3, 4, 5, 6 };
    // The new prompt reaches 2 and 4, not the old end at 3.
    if (llama_warm_covers(-1, old_t, 3, old_marks, 1, long_t, 6, grid, 2)) {
        return 1;
    }
    const int64_t both[] = { 2, 3 };
    if (!llama_warm_covers(-1, old_t, 3, old_marks, 1, long_t, 6, both, 2)) {
        return 1;
    }
    std::printf("warm cover ok\n");
    return 0;
}
#endif
