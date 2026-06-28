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

// Copy last ANE output (fp32, channel-major × spatial) into dst; returns bytes copied.
size_t ane_draft_session_read_output(float * dst, size_t dst_floats);

int ane_draft_session_step_count(void);

// True when B6 two-conv MIL compiled successfully (not conv1 fallback).
bool ane_draft_session_using_conv2(void);

void ane_draft_session_shutdown(void);

#ifdef __cplusplus
}
#endif
