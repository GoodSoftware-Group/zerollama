// FFN SwiGLU weight-name helpers (no ggml dependency — usable from lab smokes).
#include "ane_ffn_swiglu_fuse.h"

#include <string.h>

static bool name_has(const char * n, const char * sub) {
    return n && sub && strstr(n, sub) != NULL;
}

bool ane_ffn_name_is_ffn_up(const char * weight_name) {
    if (name_has(weight_name, "ffn_up_exps")) {
        return false;
    }
    return name_has(weight_name, "ffn_up_shexp") || name_has(weight_name, "ffn_up.weight");
}

bool ane_ffn_name_is_ffn_gate(const char * weight_name) {
    if (name_has(weight_name, "ffn_gate_exps")) {
        return false;
    }
    return name_has(weight_name, "ffn_gate_shexp") || name_has(weight_name, "ffn_gate.weight");
}

bool ane_ffn_name_is_ffn_down(const char * weight_name) {
    if (name_has(weight_name, "ffn_down_exps")) {
        return false;
    }
    return name_has(weight_name, "ffn_down_shexp") || name_has(weight_name, "ffn_down.weight");
}

bool ane_ffn_name_is_ffn_swiglu_weight(const char * weight_name) {
    return ane_ffn_name_is_ffn_up(weight_name) ||
           ane_ffn_name_is_ffn_gate(weight_name) ||
           ane_ffn_name_is_ffn_down(weight_name);
}
