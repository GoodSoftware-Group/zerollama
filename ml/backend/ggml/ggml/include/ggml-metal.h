// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(
        ggml_backend_dev_t device,
        uint32_t surface_id,
        size_t size,
        size_t max_tensor_size);

// P70 (lab): pause/resume the background residency-set keep-alive heartbeat for this
// Metal device. Callers holding a long host-side compute window (no GPU submissions on
// this device) can pause the heartbeat to avoid racing its requestResidency calls
// against the next command encoder/resource-list mutation, then must resume it
// afterwards (including on early-return / error paths) via a scope guard. Ref-counted;
// safe to call from any thread. No-op if device is not Metal or has no residency-set
// support.
GGML_BACKEND_API void ggml_backend_metal_dev_rsets_pause (ggml_backend_dev_t device);
GGML_BACKEND_API void ggml_backend_metal_dev_rsets_resume(ggml_backend_dev_t device);

#ifdef __cplusplus
}
#endif
