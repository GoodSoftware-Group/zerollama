#pragma once

// A miss may take a cache slot only from an expert used fewer times. One
// used as often stays, and the miss is copied into a one-batch slot.

#include <cstdint>

// True when the expert already in the slot has been used strictly fewer
// times than the one that missed. An empty slot is not a use count.
inline bool llama_moe_may_evict(uint32_t victim_uses, uint32_t newcomer_uses) {
    return victim_uses < newcomer_uses;
}

// Slot index of one-batch copy `transit_i`, after the cached slots and the
// zero slots the host misses read.
inline int32_t llama_moe_transit_slot(int32_t n_slots, int32_t n_zero, int32_t transit_i) {
    return n_slots + n_zero + transit_i;
}

// One-batch slots reserved when the cache budget can spare them.
inline constexpr int32_t LLAMA_MOE_TRANSIT_SLOTS = 128;

#ifdef LLAMA_MOE_TRANSIT_TEST
#include <cstdio>
int main() {
    if (!llama_moe_may_evict(0, 1) || !llama_moe_may_evict(4, 5)) {
        std::fputs("a miss can replace an expert used less often\n", stderr);
        return 1;
    }
    if (llama_moe_may_evict(1, 1) || llama_moe_may_evict(5, 4)) {
        std::fputs("a miss does not replace an expert used as often\n", stderr);
        return 1;
    }
    if (llama_moe_transit_slot(10, 2, 0) != 12 || llama_moe_transit_slot(10, 2, 3) != 15) {
        std::fputs("one-batch slots follow the cache and the zero slots\n", stderr);
        return 1;
    }
    if (LLAMA_MOE_TRANSIT_SLOTS != 128) {
        std::fputs("128 one-batch slots\n", stderr);
        return 1;
    }
    std::puts("transit ok");
    return 0;
}
#endif
