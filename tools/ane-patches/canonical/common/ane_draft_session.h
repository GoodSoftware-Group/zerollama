#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// In-process ANE draft session for llama-server (B1+).
// Why in-process: ANE bridge IOSurface IDs are not IOSurfaceLookup-able from another PID;
// ggml handoff must map the same surface the kernel compiled against.

// True when compiled with Apple ANE bridge (libane_bridge) available.
bool ane_draft_session_supported(void);

// Compile ANE draft kernel once; holds IOSurface input for process lifetime.
// weight_path may be NULL for synthetic weights; gamma_path optional (B3 ffn norm mul).
bool ane_draft_session_init(int channels, int spatial, const char * weight_path, const char * gamma_path);

bool ane_draft_session_ready(void);

uint32_t ane_draft_session_surface_id(void);
size_t   ane_draft_session_surface_bytes(void);
int      ane_draft_session_channels(void);
int      ane_draft_session_spatial(void);

// Metal stub fill + ANE eval (B1 smoke path when no draft activations available).
bool ane_draft_session_step_once(float fill_val);

// ANE eval only — caller must populate input IOSurface first (ggml handoff path).
bool ane_draft_session_eval(void);

// Block until any in-flight ane_draft_session_eval_async() completes.
void ane_draft_session_eval_sync(void);

// Queue ANE eval on a serial background queue (overlap with Metal draft decode).
// on_done runs on the eval queue when eval finishes (optional).
typedef void (*ane_draft_eval_async_fn)(bool ok);
bool ane_draft_session_eval_async(ane_draft_eval_async_fn on_done);

// Default async for matmul unless ZEROLLAMA_ANE_DRAFT_EVAL_ASYNC=0.
bool ane_draft_session_eval_async_enabled(void);

// Copy last ANE output (fp32, channel-major × spatial) into dst; returns bytes copied.
size_t ane_draft_session_read_output(float * dst, size_t dst_floats);

int ane_draft_session_step_count(void);

// ZEROLLAMA_ANE_DRAFT_CONV_DEPTH cap (0 = unlimited); active conv1 kernels compiled after init.
int ane_draft_session_conv_depth_cap(void);
int ane_draft_session_active_conv_count(void);

// Output channel width (matmul oc or conv ch).
int ane_draft_session_output_channels(void);

// True when ZEROLLAMA_ANE_DRAFT_KERNEL=matmul compiled successfully.
bool ane_draft_session_matmul_active(void);

// True when matmul uses dynamic MIL (weights in IOSurface input, prefill-style).
bool ane_draft_session_matmul_dynamic(void);

// Pack activation slice only into dynamic matmul IOSurface [ic × seq]; weights primed at init.
bool ane_draft_session_pack_matmul_activations(float * dst, const float * hidden, int hidden_len);

// 0, 1 (gate), 2 (gate+silu+up), 3 (SwiGLU+ffn_down), or 4 (+attn_gate) when matmul kernel active.
int ane_draft_session_matmul_chain_depth(void);

// Full n_embd after ffn_down (768 on 2B); valid when chain depth >= 3.
int ane_draft_session_matmul_ffn_embd(void);

// Stashed ffn_down vector for B7 drive when chain4 output is attn_gate width.
size_t ane_draft_session_read_ffn_down(float * dst, size_t dst_floats);

bool ane_draft_session_using_conv2(void);

void ane_draft_session_shutdown(void);

#ifdef __cplusplus
}
#endif
