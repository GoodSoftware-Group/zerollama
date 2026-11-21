#pragma once

// A cache that has a slot for every expert can be filled at load. The first
// tokens then find the expert already on the device.

#include <cstdint>

// True when n_slots can hold every expert of every layer in the group.
inline bool llama_moe_holds_all(int32_t n_slots, int32_t n_layers, int32_t n_expert) {
    if (n_slots <= 0 || n_layers <= 0 || n_expert <= 0) {
        return false;
    }
    const int64_t need = (int64_t) n_layers * (int64_t) n_expert;
    return (int64_t) n_slots >= need;
}

// Slot of one expert when the group is packed layer by layer.
inline int32_t llama_moe_resident_slot(int32_t layer_in_group, int32_t expert, int32_t n_expert) {
    return layer_in_group * n_expert + expert;
}

#ifdef LLAMA_MOE_RESIDENT_TEST
#include <cstdio>
#include <cstdlib>
int main() {
    if (llama_moe_holds_all(0, 2, 4) || llama_moe_holds_all(8, 0, 4) || llama_moe_holds_all(8, 2, 0)) {
        std::fputs("an empty group does not hold every expert\n", stderr);
        return 1;
    }
    if (llama_moe_holds_all(7, 2, 4)) {
        std::fputs("seven slots cannot hold two layers of four experts\n", stderr);
        return 1;
    }
    if (!llama_moe_holds_all(8, 2, 4) || !llama_moe_holds_all(9, 2, 4)) {
        std::fputs("eight slots hold two layers of four experts\n", stderr);
        return 1;
    }
    if (llama_moe_resident_slot(0, 0, 4) != 0 || llama_moe_resident_slot(1, 3, 4) != 7) {
        std::fputs("experts are packed layer by layer\n", stderr);
        return 1;
    }
    std::puts("resident ok");
    return 0;
}
#endif
