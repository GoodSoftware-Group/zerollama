// Lab-only ANE FFN intercept policy — fail-closed, no Metal replace by default.
// Namespace: ZEROLLAMA_ANE_FFN_* (never ZEROLLAMA_ANE_DRAFT_*).
//
// Canonical copy used by ggml-metal shadow telemetry and tools/ane-prefill smokes.
// First target: GGML_OP_MUL_MAT for dense/shexp geometry (not MUL_MAT_ID).

#pragma once

#include "ane_ffn_swiglu_fuse.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANE_FFN_NAME_PAT_MAX  16
#define ANE_FFN_NAME_PAT_LEN  48

typedef enum {
    ANE_FFN_MODE_OFF    = 0,
    ANE_FFN_MODE_SHADOW = 1, // log / count only — Metal still runs
    ANE_FFN_MODE_FORCE  = 2, // skip Metal only when force_try succeeds (lab); refuses prod ports
} ane_ffn_mode_t;

typedef enum {
    ANE_FFN_OP_NONE       = 0,
    ANE_FFN_OP_MUL_MAT    = 1,
    ANE_FFN_OP_MUL_MAT_ID = 2, // never intercept in v1
} ane_ffn_op_t;

typedef struct {
    bool enabled;
    ane_ffn_mode_t mode;
    int ic;       // 0 = any
    int oc;       // 0 = any
    int seq_max;  // 0 = any
    int lab_port;
    bool telemetry;
    bool refuse_production_ports;
    // Weight-name substrings (src0). 0 = any name. Presets via ZEROLLAMA_ANE_FFN_NAME.
    int n_name_pats;
    char name_pats[ANE_FFN_NAME_PAT_MAX][ANE_FFN_NAME_PAT_LEN];
} ane_ffn_policy_t;

typedef struct {
    bool allow;
    bool geometry_match;
    bool port_ok;
    bool name_match;
    const char *reason;
} ane_ffn_verdict_t;

bool ane_ffn_policy_load(ane_ffn_policy_t *out);
ane_ffn_mode_t ane_ffn_policy_mode(void);
bool ane_ffn_policy_enabled(void);
bool ane_ffn_policy_telemetry(void);

int ane_ffn_policy_parse_host_port(const char *host_or_url);
bool ane_ffn_policy_is_production_port(int port);

// weight_name: ggml weight tensor name (src0), e.g. "blk.0.ffn_up_shexp.weight".
// NULL/empty fails when a name filter is configured.
bool ane_ffn_policy_name_matches(const ane_ffn_policy_t *pol, const char *weight_name);

ane_ffn_verdict_t ane_ffn_policy_decide(
    const ane_ffn_policy_t *pol,
    ane_ffn_op_t op,
    int ic, int oc, int seq,
    int serve_port,
    const char *weight_name);

// ggml-metal shadow hook: count + optional stderr log. Never skips Metal.
// ic/oc/seq from MUL_MAT locals (ne00 / ne01 / ne11); weight_name from src0->name.
void ane_ffn_shadow_note_mul_mat(int ic, int oc, int seq, const char *weight_name);

// Shadow/force telemetry when clean up→gate→GLU→down fuse matches (Metal may still run).
void ane_ffn_shadow_note_swiglu_fuse(int ic, int hidden, int seq, const char *weight_name);

// Lab: why a force attempt did not replace (sync / shared / wcache / pack / …).
void ane_ffn_force_note_bail(
    const char *reason, int ic, int hidden, int seq, const char *weight_name);

// Lab: multi-slot weight-cache hit/miss (ZEROLLAMA_ANE_FFN_WCACHE_SLOTS).
void ane_ffn_force_note_wcache(
    int hit, int slots, int ic, int hidden, const char *weight_name);

// Lab profile (ZEROLLAMA_ANE_FFN_PROFILE=1): accumulate phase ms; dumps every N replaces.
void ane_ffn_profile_add_ms(const char *phase, double ms);
void ane_ffn_profile_tick_replace(void); // call once per successful replace
double ane_ffn_profile_now_ms(void);

// Force path: returns true only when ANE successfully replaced this MUL_MAT.
// Dims-only entry (Metal hook): normally false (deferred).
// Extra gate: ZEROLLAMA_ANE_FFN_FORCE_ENABLE=1 required in addition to MODE=force.
bool ane_ffn_force_try_mul_mat(int ic, int oc, int seq, const char *weight_name);

// True when force policy would allow a replace attempt (mode/enable/geometry/port/name).
// No deferred counter side effects.
bool ane_ffn_force_want_try(int ic, int oc, int seq, const char *weight_name);

// Host-buffer force replace (lab / smoke). Layouts match ane_prefill_session:
//   W[oc][ic], X[ic][seq], Y[oc][seq] channel-major floats.
// Requires MODE=force + FORCE_ENABLE + registered replace fn. Never on prod ports.
bool ane_ffn_force_try_mul_mat_host(
    int ic, int oc, int seq,
    const float *W_oc_ic,
    const float *X_ic_seq,
    float *Y_oc_seq);

// Optional ANE replace implementation (linked from tools/ane-prefill force replace).
// When NULL, force always defers (Metal keeps running).
typedef bool (*ane_ffn_host_replace_fn)(
    int ic, int oc, int seq,
    const float *W_oc_ic,
    const float *X_ic_seq,
    float *Y_oc_seq);

void ane_ffn_force_set_host_replace(ane_ffn_host_replace_fn fn);
ane_ffn_host_replace_fn ane_ffn_force_get_host_replace(void);

typedef bool (*ane_ffn_swiglu_replace_fn)(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const float *X_ic_seq,
    float *Y_ic_seq);

void ane_ffn_force_set_swiglu_replace(ane_ffn_swiglu_replace_fn fn);

// Lab: dlopen ZEROLLAMA_ANE_FFN_REPLACE_DYLIB and bind
// ane_ffn_force_replace_mul_mat into this process's host-replace slot.
void ane_ffn_force_autoload_host_replace(void);

// Metal/ggml entry: pack host-coherent F16/F32 tensors → ANE host replace → unpack.
// Caller must ensure CPU-visible acts (encoder sync-and-resume, or FORCE_HOST=1).
struct ggml_tensor;
bool ane_ffn_force_try_mul_mat_tensors(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst);

// Fused SwiGLU host replace (lab). Wg/Wu [hidden×ic], Wd [ic×hidden], X/Y [ic×seq].
// Opt-in: ZEROLLAMA_ANE_FFN_SWIGLU=1. Uses dlsym ane_ffn_force_replace_swiglu.
bool ane_ffn_force_try_swiglu_host(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const float *X_ic_seq,
    float *Y_ic_seq);

// Same as try_swiglu_host but X/Y are IEEE fp16 channel-major (skips f32↔fp16 in dylib).
// Falls back to false if dylib lacks ane_ffn_force_replace_swiglu_fp16.
bool ane_ffn_force_try_swiglu_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *X_ic_seq_f16,
    void *Y_ic_seq_f16);

// Pre-packed channel int8 X + fp16 Y. Requires INT8_IN session already created.
bool ane_ffn_force_try_swiglu_int8_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    void *Y_ic_seq_f16);

// x_scale from dylib session cache (0 if unset / no dylib).
float ane_ffn_force_query_x_scale(void);

// Stable scache keys (ggml weight data ptrs) + activate existing session for Metal layout.
void ane_ffn_force_swiglu_bind_weight_ids(
    const void *wg_data, const void *wu_data, const void *wd_data);
bool ane_ffn_force_swiglu_activate_session(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden);

// Metal layout path (dylib): pack ggml→i8 into input surface, eval, unpack out→ggml f16.
// Returns false if dylib/Metal symbols missing (caller falls back).
bool ane_ffn_force_try_swiglu_metal_layout(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *src_ggml_acts, // f16 or f32 per acts_is_f16
    int acts_is_f16,
    void *dst_ggml_f16);

// Pack fuse result (optional scales folded into W) and run SwiGLU replace into fuse->dst.
// Returns true on success. Requires shared host-visible tensors.
bool ane_ffn_force_try_swiglu_tensors(const ane_ffn_swiglu_fuse_t * fuse);

// Telemetry counters (process-wide).
uint64_t ane_ffn_shadow_match_count(void);
uint64_t ane_ffn_shadow_seen_count(void);
uint64_t ane_ffn_shadow_swiglu_fuse_count(void);
uint64_t ane_ffn_force_deferred_count(void);
uint64_t ane_ffn_force_replaced_count(void);

#ifdef __cplusplus
}
#endif
