#pragma once

// Whether a cached prompt is worth restoring. Copying a sequence state is not
// free: a later prompt should save more time than the copy costs. Until a
// prefill has been timed, every hit is taken. LLAMA_WARM_COST=0 turns the
// check off.

#include <cstdint>
#include <cstddef>

// Seconds saved by skipping `hit` tokens, minus the seconds to copy `bytes`.
// A non-positive token rate means no prefill has been timed yet.
inline double llama_warm_gain(size_t hit, double seconds_per_token, size_t bytes, double seconds_per_byte) {
    return (double) hit * seconds_per_token - (double) bytes * seconds_per_byte;
}

// True when restoring `hit` tokens is faster than running them, or when no
// prefill has been timed yet. A hit of zero never pays.
inline bool llama_warm_pays(size_t hit, double seconds_per_token, size_t bytes, double seconds_per_byte) {
    if (hit == 0) {
        return false;
    }
    if (seconds_per_token <= 0.0) {
        return true;
    }
    return llama_warm_gain(hit, seconds_per_token, bytes, seconds_per_byte) > 0.0;
}

// True when a later resume of the first `tokens` tokens is expected to save
// more than copying the state out and back. Until a prefill is timed, keep it.
inline bool llama_warm_worth(size_t tokens, double seconds_per_token, size_t bytes, double seconds_per_byte) {
    if (seconds_per_token <= 0.0) {
        return true;
    }
    if (tokens == 0) {
        return false;
    }
    return (double) tokens * seconds_per_token >= 2.0 * (double) bytes * seconds_per_byte;
}

// Host copy before any restore has been timed: 8 GB/s. Disk read before any
// read has been timed: 500 MB/s.
struct llama_warm_cost {
    static constexpr double k_restore_s = 1.0 / 8e9;
    static constexpr double k_load_s    = 1.0 / 5e8;

    bool   on        = true;
    double token_s   = 0.0;
    double restore_s = 0.0;
    double load_s    = k_load_s;

    bool timed() const {
        return on && token_s > 0.0;
    }

    double restore() const {
        return restore_s > 0.0 ? restore_s : k_restore_s;
    }

    bool worth(size_t tokens, size_t bytes) const {
        if (!on) {
            return true;
        }
        return llama_warm_worth(tokens, token_s, bytes, restore());
    }

    bool pays(size_t hit, size_t bytes, double extra_s_per_byte) const {
        if (!on) {
            return hit > 0;
        }
        return llama_warm_pays(hit, token_s, bytes, restore() + extra_s_per_byte);
    }

    // Latest prefill, seconds per token that was actually run.
    void note_prefill(uint64_t n_tokens, int64_t microseconds) {
        if (!on || n_tokens == 0 || microseconds <= 0) {
            return;
        }
        token_s = ((double) microseconds / 1e6) / (double) n_tokens;
    }

    // Running mean of seconds per byte, first sample replaces the default.
    void note_restore(size_t bytes, int64_t microseconds) {
        if (!on || bytes == 0 || microseconds <= 0) {
            return;
        }
        const double measured = ((double) microseconds / 1e6) / (double) bytes;
        restore_s = restore_s > 0.0 ? 0.5 * (restore_s + measured) : measured;
    }

    void note_load(size_t bytes, int64_t microseconds) {
        if (!on || bytes == 0 || microseconds <= 0) {
            return;
        }
        const double measured = ((double) microseconds / 1e6) / (double) bytes;
        load_s = 0.5 * (load_s + measured);
    }
};

#ifdef LLAMA_WARM_COST_TEST
#include <iostream>

static int warm_cost_fail(const char * msg) {
    std::cerr << msg << "\n";
    return 1;
}

int main() {
    if (!llama_warm_pays(10, 0.0, 1000000000, 1e-6)) {
        return warm_cost_fail("untimed hit");
    }
    if (llama_warm_pays(0, 0.0, 1, 1e-9)) {
        return warm_cost_fail("empty hit");
    }
    // 100 tokens at 1 ms each, 1000 bytes at 1 ns each: the copy is noise.
    if (!llama_warm_pays(100, 0.001, 1000, 1e-9)) {
        return warm_cost_fail("cheap copy");
    }
    // 1 token at 1 ms, 1e9 bytes at 10 ns: ten seconds of copying.
    if (llama_warm_pays(1, 0.001, 1000000000, 1e-8)) {
        return warm_cost_fail("expensive copy");
    }
    if (llama_warm_gain(100, 0.001, 1000, 1e-9) <= llama_warm_gain(10, 0.001, 1000, 1e-9)) {
        return warm_cost_fail("longer hit saves more");
    }
    if (!llama_warm_worth(1000, 0.0, 100000000, 1e-9)) {
        return warm_cost_fail("untimed keep");
    }
    // 1000 * 1e-4 = 0.1; 2 * 1e6 * 1e-9 = 0.002
    if (!llama_warm_worth(1000, 1e-4, 1000000, 1e-9)) {
        return warm_cost_fail("worth keeping");
    }
    // 10 * 1e-4 = 0.001; 2 * 1e8 * 1e-9 = 0.2
    if (llama_warm_worth(10, 1e-4, 100000000, 1e-9)) {
        return warm_cost_fail("not worth keeping");
    }

    llama_warm_cost cost;
    if (cost.timed() || !cost.worth(1, 1)) {
        return warm_cost_fail("default");
    }
    cost.note_prefill(1000, 1000000); // 1 ms/token
    if (!cost.timed() || cost.token_s < 0.0009 || cost.token_s > 0.0011) {
        return warm_cost_fail("prefill rate");
    }
    cost.note_restore(1000, 1000); // 1 us / 1000 bytes = 1e-6 s/byte
    cost.note_restore(1000, 3000); // average with 3e-6 → 2e-6
    if (cost.restore() < 1.9e-6 || cost.restore() > 2.1e-6) {
        return warm_cost_fail("restore average");
    }
    const double load_before = cost.load_s;
    cost.note_load(1000, 1000); // 1e-6 s/byte, slower than the 500 MB/s default
    if (!(cost.load_s > load_before && cost.load_s < 1e-6)) {
        return warm_cost_fail("load average");
    }
    cost.on = false;
    if (cost.timed() || !cost.worth(1, 100000000000ull)) {
        return warm_cost_fail("disabled");
    }

    std::cout << "warm cost ok\n";
    return 0;
}
#endif
