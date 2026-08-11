// Detect parallel SwiGLU FFN chain (build_ffn PAR / LLM_FFN_SILU→swiglu_split):
//   build order:  MUL_MAT(up) [→scale] → MUL_MAT(gate) [→scale] → GLU(SWIGLU) → MUL_MAT(down) [→scale]
//   Metal topo:   MUL_MAT(gate) [→scale] → MUL_MAT(up) [→scale] → GLU(SWIGLU) → MUL_MAT(down) [→scale]
// MoE shexp may interleave router/shared-gate ops between up and GLU / GLU and down;
// matcher skip-scans for GLU/down by dataflow. When holey, Metal encode skips only the
// contiguous gate+up[+scales] prefix and clears COMPUTE on glu/down(+scale).
// Anchor at whichever of ffn_up / ffn_gate appears first; siblings share activation src1.
// Optional post-mm MUL scales are folded into weights at replace time.
// Types (F16/F32) are enforced at force-replace, not here (shadow may match Q4).

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

typedef struct {
    int n_fuse;          // chain op count (4..7): up/gate/glu/down + optional scales
    int n_encode_skip;   // Metal encode advance from anchor (full span if contiguous)
    int holey;           // 1 if glu/down lie past n_encode_skip (interleaved MoE topo)
    int ic;              // embedding / restore width
    int hidden;          // intermediate (gate/up OC)
    int seq;
    const struct ggml_tensor * up;         // MUL_MAT
    const struct ggml_tensor * up_scale;   // optional MUL, else NULL
    const struct ggml_tensor * gate;       // MUL_MAT
    const struct ggml_tensor * gate_scale; // optional MUL, else NULL
    const struct ggml_tensor * glu;        // GLU SWIGLU
    const struct ggml_tensor * down;       // MUL_MAT
    const struct ggml_tensor * down_scale; // optional MUL after down, else NULL
    const struct ggml_tensor * dst;        // write target: down_scale or down
} ane_ffn_swiglu_fuse_t;

// nodes[0..n_available) starting at candidate up or gate mul_mat.
// Dense: lookahead ≥4 (≤7 with scales). MoE shexp: widen to ~48 for interleaved topo.
bool ane_ffn_swiglu_fuse_match(
    const struct ggml_tensor * const * nodes,
    int n_available,
    ane_ffn_swiglu_fuse_t * out);

// True if weight name is dense/shexp ffn_up (not _exps).
bool ane_ffn_name_is_ffn_up(const char * weight_name);
bool ane_ffn_name_is_ffn_gate(const char * weight_name);
bool ane_ffn_name_is_ffn_down(const char * weight_name);
// True for dense/shexp up|gate|down (not routed _exps).
bool ane_ffn_name_is_ffn_swiglu_weight(const char * weight_name);

#ifdef __cplusplus
}
#endif
