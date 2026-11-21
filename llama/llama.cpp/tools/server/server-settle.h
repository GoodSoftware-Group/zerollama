#pragma once

// Where a GPU sequence may be resumed.
//
// A prompt batch and a decode step do not round the same way. The resume
// point is the last full batch at or before the end of the prompt. Tokens
// after that, including the reply, are run again as prompt batches. The CPU
// keeps the whole sequence. LLAMA_SETTLE=0 leaves that behavior in place.

#include <cstdint>

// Wait this long after a reply before rewriting it, so a request that is
// already on its way is not held behind the rewrite.
inline constexpr int32_t LLAMA_SETTLE_IDLE_MS = 250;

// Largest multiple of grid that is <= n_prompt. 0 when the prompt has not
// filled a batch yet.
inline int32_t llama_settle_mark(int32_t n_prompt, int32_t grid) {
    if (grid <= 1 || n_prompt <= 0) {
        return 0;
    }
    return (n_prompt / grid) * grid;
}

// Largest multiple of grid that is <= n_tokens, and not before mark.
inline int32_t llama_settle_end(int32_t mark, int32_t n_tokens, int32_t grid) {
    if (mark < 0) {
        mark = 0;
    }
    if (grid <= 1 || n_tokens <= mark) {
        return mark;
    }
    const int32_t end = (n_tokens / grid) * grid;
    return end < mark ? mark : end;
}

#ifdef LLAMA_SETTLE_TEST
#include <cstdio>
#include <cstdlib>
int main() {
    if (llama_settle_mark(0, 512) != 0 || llama_settle_mark(100, 512) != 0) {
        std::fputs("a short prompt has no full batch to resume from\n", stderr);
        return 1;
    }
    if (llama_settle_mark(512, 512) != 512 || llama_settle_mark(600, 512) != 512) {
        std::fputs("the resume point is the last full batch inside the prompt\n", stderr);
        return 1;
    }
    if (llama_settle_end(512, 600, 512) != 512) {
        std::fputs("a short reply does not finish a batch\n", stderr);
        return 1;
    }
    if (llama_settle_end(512, 2000, 512) != 1536) {
        std::fputs("the rewrite stops on the last full batch of the sequence\n", stderr);
        return 1;
    }
    if (llama_settle_end(0, 100, 512) != 0 || llama_settle_end(0, 512, 512) != 512) {
        std::fputs("the first batch ends on the grid\n", stderr);
        return 1;
    }
    if (LLAMA_SETTLE_IDLE_MS != 250) {
        std::fputs("idle wait is not 250 ms\n", stderr);
        return 1;
    }
    std::puts("settle ok");
    return 0;
}
#endif
