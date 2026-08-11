// Lab-only ANE prefill FFN-slice session (expert geometry).
// Compile-once + IOSurface I/O — same contract as ane_draft_session but
// rectangular IC×OC (e.g. 2048×512) and no draft/dflash coupling.
//
// Architecture (do not fold into ane_draft_session):
//   - Dense / shared-expert FFN → Metal MUL_MAT (clean intercept target)
//   - Routed MoE experts → Metal MUL_MAT_ID (separate path; needs dyn weight stream)
//   - Env namespace when hooked: ZEROLLAMA_ANE_FFN_* (not ZEROLLAMA_ANE_DRAFT_*)
// Modes: fp16-blob matmul, int8-conv, fused SwiGLU (fp16 / int8 / fused-gate-up)
//
// Not wired into ggml/llama-server yet. Use ane-prefill-ffn-slice-smoke to validate.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEPrefillSession ANEPrefillSession;

// Single matmul y = x @ W. weight_oc_ic: row-major [oc][ic], or NULL for synth.
ANEPrefillSession *ane_prefill_session_create(int ic, int oc, int seq,
                                              const float *weight_oc_ic);

// Same geometry as create(), but int8 BLOBFILE weights via constexpr_affine_dequantize + 1×1 conv.
ANEPrefillSession *ane_prefill_session_create_int8(int ic, int oc, int seq,
                                                   const float *weight_oc_ic);

// Fused SwiGLU: y = (silu(x@Wg) * (x@Wu)) @ Wd
// Wg/Wu: [hidden×ic], Wd: [ic×hidden], or NULL for synth (fp16-safe scale).
ANEPrefillSession *ane_prefill_session_create_swiglu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden);

// SwiGLU with int8 BLOBFILE weights (three constexpr_affine_dequantize).
ANEPrefillSession *ane_prefill_session_create_swiglu_int8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden);

// SwiGLU with gate∥up fused into one 1×1 conv [2H,IC] + slice_by_size (fp16 weights).
ANEPrefillSession *ane_prefill_session_create_swiglu_fused_gu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden);

// int8 weights + fused gate∥up + optional W8A8 quant/dequant on hid before down.
// act_scale: MIL fp16 scale for quantize/dequantize when w8a8_hid (ignored otherwise).
ANEPrefillSession *ane_prefill_session_create_swiglu_int8_fused(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    bool w8a8_hid,
    float act_scale);

// int8 SwiGLU (3 weights) + optional W8A8 on hid and/or input x.
// hid_scale>0 enables hid quant; x_scale>0 enables input quant (or dequant if int8_input).
// sp0*sp1 must equal seq (1×seq strip, or e.g. 2×256 tiling — same buffer layout).
// int8_input: MIL takes tensor<int8> acts (host writes int8); x_scale is dequant scale.
ANEPrefillSession *ane_prefill_session_create_swiglu_int8_w8a8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    float hid_scale,
    float x_scale,
    int sp0, int sp1,
    bool int8_input);

// Pick a good [sp0,sp1] with sp0*sp1==seq for W8A8 tiling (falls back to 1×seq).
void ane_prefill_session_pick_tile(int seq, int *sp0, int *sp1);

bool ane_prefill_session_is_int8_input(const ANEPrefillSession *s);

// Write int8 activations [ic × seq] when session was created with int8_input.
bool ane_prefill_session_write_acts_int8(ANEPrefillSession *s,
                                         const void *int8_ic_seq,
                                         size_t bytes);

void ane_prefill_session_destroy(ANEPrefillSession *s);

bool ane_prefill_session_ready(const ANEPrefillSession *s);

int ane_prefill_session_ic(const ANEPrefillSession *s);
int ane_prefill_session_oc(const ANEPrefillSession *s); // matmul OC, or IC for swiglu out
int ane_prefill_session_hidden(const ANEPrefillSession *s); // 0 if matmul-only
int ane_prefill_session_seq(const ANEPrefillSession *s);
bool ane_prefill_session_is_swiglu(const ANEPrefillSession *s);
bool ane_prefill_session_is_int8(const ANEPrefillSession *s);
float ane_prefill_session_int8_scale(const ANEPrefillSession *s); // 0 if not int8

uint32_t ane_prefill_session_input_surface_id(const ANEPrefillSession *s);
uint32_t ane_prefill_session_output_surface_id(const ANEPrefillSession *s);
size_t   ane_prefill_session_input_bytes(const ANEPrefillSession *s);
size_t   ane_prefill_session_output_bytes(const ANEPrefillSession *s);

// Write fp16 activations [ic × seq] channel-major into input IOSurface.
bool ane_prefill_session_write_acts_fp16(ANEPrefillSession *s,
                                         const void *fp16_ic_seq,
                                         size_t bytes);

// ANE eval only — caller must populate input surface first (ggml map path).
bool ane_prefill_session_eval(ANEPrefillSession *s);

// Read fp16 output channel-major: [oc×seq] matmul, or [ic×seq] swiglu.
bool ane_prefill_session_read_out_fp16(ANEPrefillSession *s, void *dst, size_t bytes);

// Lock output IOSurface and convert fp16→f32 in one pass (no staging memcpy).
// n = oc*seq (matmul) or ic*seq (swiglu).
bool ane_prefill_session_read_out_f32(ANEPrefillSession *s, float *dst, size_t n);

double ane_prefill_session_compile_ms(const ANEPrefillSession *s);
int    ane_prefill_session_eval_count(const ANEPrefillSession *s);

#ifdef __cplusplus
}
#endif
