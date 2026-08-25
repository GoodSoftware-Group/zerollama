// Lab-only ANE SwiGLU FFN session (expert / dense shexp geometry).
// One compile: gate + up matmul → silu(gate)*up → down, three BLOBFILE weights.
//
// Do not fold into ane_draft_session. Shares IOSurface I/O contract with
// ane_prefill_session (acts [dim×seq] in, [dim×seq] out).
//
// Not wired into ggml/llama-server. Use ane-prefill-swiglu-smoke for parity.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEPrefillSwiGLUSession ANEPrefillSwiGLUSession;

// Weights are row-major float32:
//   Wg, Wu: [dim][hidden]  (index i*hidden + h)
//   Wd:     [hidden][dim]  (index h*dim + o)
// Pass NULL for any weight to fill synthetic 0.05 constants.
ANEPrefillSwiGLUSession *ane_prefill_swiglu_session_create(
    int dim, int hidden, int seq,
    const float *Wg_dim_hidden,
    const float *Wu_dim_hidden,
    const float *Wd_hidden_dim);
void ane_prefill_swiglu_session_destroy(ANEPrefillSwiGLUSession *s);

bool ane_prefill_swiglu_session_ready(const ANEPrefillSwiGLUSession *s);

int ane_prefill_swiglu_session_dim(const ANEPrefillSwiGLUSession *s);
int ane_prefill_swiglu_session_hidden(const ANEPrefillSwiGLUSession *s);
int ane_prefill_swiglu_session_seq(const ANEPrefillSwiGLUSession *s);

uint32_t ane_prefill_swiglu_session_input_surface_id(const ANEPrefillSwiGLUSession *s);
size_t   ane_prefill_swiglu_session_input_bytes(const ANEPrefillSwiGLUSession *s);
size_t   ane_prefill_swiglu_session_output_bytes(const ANEPrefillSwiGLUSession *s);

bool ane_prefill_swiglu_session_write_acts_fp16(ANEPrefillSwiGLUSession *s,
                                                const void *fp16_dim_seq,
                                                size_t bytes);
bool ane_prefill_swiglu_session_eval(ANEPrefillSwiGLUSession *s);
bool ane_prefill_swiglu_session_read_out_fp16(ANEPrefillSwiGLUSession *s,
                                              void *dst, size_t bytes);

double ane_prefill_swiglu_session_compile_ms(const ANEPrefillSwiGLUSession *s);
int    ane_prefill_swiglu_session_eval_count(const ANEPrefillSwiGLUSession *s);

#ifdef __cplusplus
}
#endif
