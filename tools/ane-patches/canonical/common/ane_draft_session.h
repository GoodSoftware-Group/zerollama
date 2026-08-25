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

// 0 … 17 (P16 output_norm + lm_head = chain 17; P15 ffn SwiGLU+down = chain 16) when matmul kernel active.
int ane_draft_session_matmul_chain_depth(void);

// True when MATMUL_CHAIN=8 / native dflash_fc path (input from ctx_tgt target_hidden).
bool ane_draft_session_dflash_fc_active(void);

// True when MATMUL_CHAIN=11 — dflash_fc + hidden_norm + blk.0 attn_q (P10).
bool ane_draft_session_dflash_chain11_active(void);

// True when MATMUL_CHAIN=12 — dflash_fc + hidden_norm + blk.0 attn_q/k/v (P11; output = v).
bool ane_draft_session_dflash_chain12_active(void);

// True when MATMUL_CHAIN=13 — chain 12 + host cross-attn softmax/KV combine (P12).
bool ane_draft_session_dflash_chain13_active(void);

// True when MATMUL_CHAIN=14 — chain 13 + blk.0 attn_output (wo) on ANE (P13).
bool ane_draft_session_dflash_chain14_active(void);

// True when MATMUL_CHAIN=15 — chain 14 + blk.0 ffn_gate on ANE (P14).
bool ane_draft_session_dflash_chain15_active(void);

// True when MATMUL_CHAIN=16 — chain 15 + blk.0 ffn_up/SwiGLU/down (P15).
bool ane_draft_session_dflash_chain16_active(void);

// True when MATMUL_CHAIN=17 — chain 16 + host output_norm for tied-embed lm_head (P16).
bool ane_draft_session_dflash_chain17_active(void);

// ANE eval attn_out @ wo after host cross-attn wrote outBuf (chain 14+).
bool ane_draft_session_eval_dflash_attn_wo(void);

// ANE eval post-attn hidden @ ffn_gate after wo wrote outBuf (chain 15+).
bool ane_draft_session_eval_dflash_ffn_gate(void);

// ANE ffn_up + host SwiGLU + host fp32 ffn_down after gate (chain 16+).
bool ane_draft_session_eval_dflash_ffn_up_swiglu_down(void);

// Host RMS output_norm on ffn_down hidden (chain 17).
bool ane_draft_session_eval_dflash_output_norm(void);

// Host RMS attn_post_norm on post-attn residual before ffn_gate (chain 15+).
bool ane_draft_session_eval_dflash_attn_post_norm(void);

// Host fp32 dflash_fc output (full target export @ native W) — skips ANE kernel1 eval when set.
bool ane_draft_session_set_dflash_fc_host(const float * fc, int n);
void ane_draft_session_clear_dflash_fc_host(void);

// Overwrite last ANE output with host attn vector (chain 13).
bool ane_draft_session_write_dflash_attn_out(const float * src, int n);

// Spatial-mean row of current outBuf; add delta row to all seq slots (dflash residuals).
bool ane_draft_session_snapshot_output_row(float * row, int n);
bool ane_draft_session_add_output_row(const float * delta, int n);
bool ane_draft_session_write_output_row(const float * row, int n);

// Read stashed q/k/v noise projections after chain 12/13 eval.
// Read host/ANE Q/K/V row vectors; oc_q and oc_kv may differ under GQA.
bool ane_draft_session_read_dflash_qkv(float * q, int oc_q, float * k, float * v, int oc_kv);

// P24: stash host-computed Q/K/V before cross-attn (oc_q / oc_kv row vectors).
bool ane_draft_session_set_dflash_qkv(const float * q, const float * k, const float * v, int oc_q, int oc_kv);

bool ane_draft_session_dflash_qkv_host_fp32(void);

// Full n_embd after ffn_down (768 on 2B); valid when chain depth >= 3.
int ane_draft_session_matmul_ffn_embd(void);

// blk.1 SwiGLU width (192 on 2B proxy); valid when chain depth >= 9.
int ane_draft_session_matmul9_oc(void);

// Stashed attn_qkv prefix (P6) after handoff hidden pack.
size_t ane_draft_session_read_qkv_prefix(float * dst, size_t dst_floats);

// Stashed ffn_down vector for B7 drive when chain4+ output is not full n_embd.
size_t ane_draft_session_read_ffn_down(float * dst, size_t dst_floats);

bool ane_draft_session_using_conv2(void);

void ane_draft_session_shutdown(void);

#ifdef __cplusplus
}
#endif
