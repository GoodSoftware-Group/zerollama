// kv_ext_donor_stubs.cpp — weak-symbol fallbacks for llama-kv-ext donor hooks.
//
// The donor registry (llama_kv_ext_donor_try_consume, llama_kv_ext_donor_try_consume_dev)
// lives in llama-memory-kv-ext.cpp which is compiled into llama-server but not into the
// zerollama Go binary. Provide weak definitions so the Go binary links cleanly.
// The Go path never activates the kv-ext donor path at runtime; these are no-ops.

#include "llama.h"
#include "ggml.h"

// These are C++-only internal hooks (declared outside extern "C" in llama-kv-ext.h),
// so they have C++ linkage and must be defined without extern "C" here.

__attribute__((weak))
struct ggml_backend_buffer * llama_kv_ext_donor_try_consume(size_t required_size) {
    (void)required_size;
    return nullptr;
}

__attribute__((weak))
struct ggml_backend_buffer * llama_kv_ext_donor_try_consume_dev(
        struct ggml_backend_device * dev,
        size_t required_size,
        size_t max_tensor_size) {
    (void)dev;
    (void)required_size;
    (void)max_tensor_size;
    return nullptr;
}
