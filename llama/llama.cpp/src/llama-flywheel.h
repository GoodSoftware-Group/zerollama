#pragma once

// Which cold experts of one batch run on the host.
//
// The experts are ordered with the most tokens first. Each one is copied to
// the device, or computed on the host, whichever keeps the two running totals
// from falling behind. A long prompt pays for the copy once. A single token
// stays on the host when the copy would take longer.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// Device memory left unused so the driver and later buffers still fit.
inline constexpr size_t LLAMA_FLYWHEEL_RESERVE = 384ull << 20;

// Seconds per byte. host is the first token, host_per_token each token after it.
struct llama_flywheel_rates {
    double host           = 0.0;
    double host_per_token = 0.0;
    double copy           = 0.0;
    double slowdown       = 1.0;
};

struct llama_flywheel_miss {
    int32_t expert = 0;
    int32_t tokens = 1;
};

inline void llama_flywheel_place(const llama_flywheel_rates & rates, size_t bytes,
        const std::vector<llama_flywheel_miss> & misses,
        std::vector<int32_t> & host, std::vector<int32_t> & device) {
    host.clear();
    device.clear();
    const double slow = rates.slowdown > 0.0 ? rates.slowdown : 1.0;
    double host_time = 0.0;
    double copy_time = 0.0;
    for (const llama_flywheel_miss & m : misses) {
        const int32_t tokens = std::max(m.tokens, 1);
        const double host_s = slow * (double) bytes * (rates.host + (double) (tokens - 1) * rates.host_per_token);
        const double copy_s = (double) bytes * rates.copy;
        if (copy_time + copy_s >= host_time + host_s) {
            host.push_back(m.expert);
            host_time += host_s;
        } else {
            device.push_back(m.expert);
            copy_time += copy_s;
        }
    }
}

#ifdef LLAMA_FLYWHEEL_TEST
#include <cstdio>
#include <cstdlib>
int main() {
    const llama_flywheel_rates slow_copy { 1e-9, 1e-10, 1e-8, 1.0 };
    const size_t bytes = 1000;
    std::vector<int32_t> host, device;

    llama_flywheel_place(slow_copy, bytes, { { 3, 1 } }, host, device);
    if (host.size() != 1 || host[0] != 3 || !device.empty()) {
        std::fputs("one token should stay on the host when the copy is slower\n", stderr);
        return 1;
    }

    llama_flywheel_place(slow_copy, bytes, { { 7, 100 } }, host, device);
    if (device.size() != 1 || device[0] != 7 || !host.empty()) {
        std::fputs("a long prompt should pay for the copy\n", stderr);
        return 1;
    }

    // Most tokens first, as the caller sorts them.
    llama_flywheel_place(slow_copy, bytes, { { 7, 100 }, { 3, 1 } }, host, device);
    if (device.size() != 1 || device[0] != 7 || host.size() != 1 || host[0] != 3) {
        std::fputs("the long expert is copied and the single token stays on the host\n", stderr);
        return 1;
    }

    const llama_flywheel_rates fast_copy { 1e-8, 1e-9, 1e-12, 1.0 };
    llama_flywheel_place(fast_copy, bytes, { { 1, 1 }, { 2, 1 } }, host, device);
    if (!host.empty() || device.size() != 2) {
        std::fputs("a fast copy takes every expert\n", stderr);
        return 1;
    }

    if (LLAMA_FLYWHEEL_RESERVE != (384ull << 20)) {
        std::fputs("reserve is not 384 MiB\n", stderr);
        return 1;
    }
    std::puts("flywheel ok");
    return 0;
}
#endif
