// Lab ANE host replace for ane_ffn_force_try_mul_mat_host.
// Compile-once session cache keyed by (ic, oc, seq, weight pointer, mode).
// SwiGLU opt-in (env, read at create): INT8 / W8A8 / W8A8_X / INT8_IN.
// INT8_IN hot path: ane_ffn_force_replace_swiglu_int8 (pre-packed channel int8).
// Not linked into ggml-metal by default (keeps Metal free of ane_bridge).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register ane_ffn_force_replace_mul_mat as the policy host replace fn.
void ane_ffn_force_register_host_replace(void);

// Direct replace (also used as the registered callback).
bool ane_ffn_force_replace_mul_mat(
    int ic, int oc, int seq,
    const float *W_oc_ic,
    const float *X_ic_seq,
    float *Y_oc_seq);

// Fused SwiGLU: y = (silu(x@Wg)*(x@Wu))@Wd
// Wg/Wu: [hidden×ic], Wd: [ic×hidden], X/Y: [ic×seq] channel-major.
bool ane_ffn_force_replace_swiglu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const float *X_ic_seq,
    float *Y_ic_seq);

// Same session cache; X/Y are IEEE fp16 channel-major bytes.
bool ane_ffn_force_replace_swiglu_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *X_ic_seq_f16,
    void *Y_ic_seq_f16);

// Pre-packed channel-major int8 X (scale must match session x_scale).
// Requires INT8_IN session already created (via replace_swiglu / _fp16 first).
// Y is f32 channel-major. No per-eval quantize.
bool ane_ffn_force_replace_swiglu_int8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    float *Y_ic_seq);

// Same as _int8 but Y is IEEE fp16 channel-major (ggml hot path; no f32 materialize).
bool ane_ffn_force_replace_swiglu_int8_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    void *Y_ic_seq_f16);

// Cached x_scale after INT8_IN / W8A8_X create (0 if unset).
float ane_ffn_force_swiglu_x_scale(void);

// Stable scache identity (ggml weight ->data pointers). Set before replace/activate
// so session LRU does not false-hit when wcache float buffers are malloc-recycled.
void ane_ffn_force_swiglu_set_weight_ids(
    const void *wg_data, const void *wu_data, const void *wd_data);

// Make g_active the scache entry for these weights (ids + float keys). false = miss.
bool ane_ffn_force_swiglu_activate(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden);

// Surface ids for Metal layout pack/unpack (0 if no session).
uint32_t ane_ffn_force_swiglu_input_surface_id(void);
uint32_t ane_ffn_force_swiglu_output_surface_id(void);
// Padded session seq (0 if no session). Metal pack/unpack must use this.
int ane_ffn_force_swiglu_session_seq(void);

// Slice-like: write int8 acts once, then reeval (eval+read) without rewrite.
// `seq` is the logical length; padded to session seq internally.
bool ane_ffn_force_swiglu_write_int8(const int8_t *X_ic_seq_i8, int seq);
bool ane_ffn_force_swiglu_reeval_f32(float *Y_ic_seq, int seq);
// Eval only (no read) — for ANE wall vs slice smoke.
bool ane_ffn_force_swiglu_eval_only(void);

#ifdef __cplusplus
}
#endif
