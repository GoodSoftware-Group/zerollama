// SwiGLU FFN fuse matcher — clean chain + optional post-mm scale MULs (lab ANE force).
// Accepts build_ffn PAR order (up→gate→glu→down) and Metal topo order (gate→up→glu→down).
// MoE shexp: skip-scan past interleaved non-chain ops to GLU/down by dataflow.
#include "ane_ffn_swiglu_fuse.h"

#include "ggml.h"

#include <string.h>

// Post-mm scale: dst = prev * scale, with scale a leaf (weight) tensor.
static bool is_post_mm_scale(
    const struct ggml_tensor * t,
    const struct ggml_tensor * prev) {
    if (!t || !prev || t->op != GGML_OP_MUL) {
        return false;
    }
    if (t->src[0] != prev || !t->src[1]) {
        return false;
    }
    // Scale must not be another compute node in this chain.
    if (t->src[1]->op != GGML_OP_NONE && t->src[1]->op != GGML_OP_VIEW &&
        t->src[1]->op != GGML_OP_RESHAPE && t->src[1]->op != GGML_OP_PERMUTE &&
        t->src[1]->op != GGML_OP_TRANSPOSE) {
        // Weights are typically GGML_OP_NONE (params).
        if (t->src[1]->src[0] != NULL) {
            return false;
        }
    }
    if (!ggml_is_contiguous(t) || !ggml_is_contiguous(t->src[1])) {
        return false;
    }
    // Broadcast: scalar, or per-output-channel vector (prev ne0) — foldable into W.
    const int64_t nscale = ggml_nelements(t->src[1]);
    if (nscale != 1 && nscale != prev->ne[0]) {
        return false;
    }
    return true;
}

static bool finish_fuse(
    ane_ffn_swiglu_fuse_t * out,
    int n_fuse,
    int n_encode_skip,
    int holey,
    const struct ggml_tensor * up,
    const struct ggml_tensor * up_scale,
    const struct ggml_tensor * gate,
    const struct ggml_tensor * gate_scale,
    const struct ggml_tensor * glu,
    const struct ggml_tensor * down,
    const struct ggml_tensor * down_scale,
    const struct ggml_tensor * dst) {
    if (!ggml_is_contiguous(up->src[0]) || !ggml_is_contiguous(gate->src[0]) ||
        !ggml_is_contiguous(down->src[0]) || !ggml_is_contiguous(up->src[1]) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    if (up_scale && !ggml_is_contiguous(up_scale->src[1])) {
        return false;
    }
    if (gate_scale && !ggml_is_contiguous(gate_scale->src[1])) {
        return false;
    }
    if (down_scale && !ggml_is_contiguous(down_scale->src[1])) {
        return false;
    }

    const int ic     = (int)up->src[0]->ne[0];
    const int hidden = (int)up->src[0]->ne[1];
    const int seq    = (int)up->src[1]->ne[1];
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return false;
    }
    if (gate->src[0]->ne[0] != ic || gate->src[0]->ne[1] != hidden) {
        return false;
    }
    if (down->src[0]->ne[0] != hidden || down->src[0]->ne[1] != ic) {
        return false;
    }
    if (dst->ne[0] != ic || dst->ne[1] != seq) {
        return false;
    }
    if (up->src[0]->ne[2] != 1 || up->src[1]->ne[2] != 1 || dst->ne[2] != 1) {
        return false;
    }
    // Weight/act types checked at force-replace time (F16/F32 pack). Shadow may match Q4.

    out->n_fuse = n_fuse;
    out->n_encode_skip = n_encode_skip;
    out->holey = holey;
    out->ic = ic;
    out->hidden = hidden;
    out->seq = seq;
    out->up = up;
    out->up_scale = up_scale;
    out->gate = gate;
    out->gate_scale = gate_scale;
    out->glu = glu;
    out->down = down;
    out->down_scale = down_scale;
    out->dst = dst;
    return true;
}

// From *i, skip-scan for GLU(SWIGLU) with gate_branch/up_branch, then down[+scale].
// prefix_len = contiguous encode skip (gate+up[+scales] only).
static bool match_glu_down(
    const struct ggml_tensor * const * nodes,
    int n_available,
    int * i,
    int prefix_len,
    const struct ggml_tensor * up,
    const struct ggml_tensor * up_branch,
    const struct ggml_tensor * up_scale,
    const struct ggml_tensor * gate,
    const struct ggml_tensor * gate_branch,
    const struct ggml_tensor * gate_scale,
    ane_ffn_swiglu_fuse_t * out) {
    int holey = 0;

    const struct ggml_tensor * glu = NULL;
    while (*i < n_available) {
        const struct ggml_tensor * cand = nodes[*i];
        if (cand && cand->op == GGML_OP_GLU && cand->src[0] && cand->src[1] &&
            ggml_get_glu_op(cand) == GGML_GLU_OP_SWIGLU &&
            cand->src[0] == gate_branch && cand->src[1] == up_branch) {
            glu = cand;
            break;
        }
        holey = 1;
        (*i)++;
    }
    if (!glu) {
        return false;
    }
    (*i)++;

    const struct ggml_tensor * down = NULL;
    while (*i < n_available) {
        const struct ggml_tensor * cand = nodes[*i];
        if (cand && cand->op == GGML_OP_MUL_MAT && cand->src[0] && cand->src[1] &&
            ane_ffn_name_is_ffn_down(cand->src[0]->name) &&
            cand->src[1] == glu) {
            down = cand;
            break;
        }
        holey = 1;
        (*i)++;
    }
    if (!down) {
        return false;
    }
    (*i)++;

    const struct ggml_tensor * down_scale = NULL;
    const struct ggml_tensor * dst = down;
    if (*i < n_available && is_post_mm_scale(nodes[*i], down)) {
        down_scale = nodes[(*i)++];
        dst = down_scale;
    }

    int n_chain = 4 + (up_scale != NULL) + (gate_scale != NULL) + (down_scale != NULL);
    int n_encode_skip = holey ? prefix_len : *i;
    return finish_fuse(
        out, n_chain, n_encode_skip, holey,
        up, up_scale, gate, gate_scale, glu, down, down_scale, dst);
}

static bool match_up_first(
    const struct ggml_tensor * const * nodes,
    int n_available,
    ane_ffn_swiglu_fuse_t * out) {
    int i = 0;
    const struct ggml_tensor * up = nodes[i++];
    if (!up || up->op != GGML_OP_MUL_MAT || !up->src[0] || !up->src[1]) {
        return false;
    }
    if (!ane_ffn_name_is_ffn_up(up->src[0]->name)) {
        return false;
    }

    const struct ggml_tensor * up_branch = up;
    const struct ggml_tensor * up_scale = NULL;
    if (i < n_available && is_post_mm_scale(nodes[i], up)) {
        up_scale = nodes[i++];
        up_branch = up_scale;
    }

    if (i >= n_available) {
        return false;
    }
    const struct ggml_tensor * gate = nodes[i++];
    if (!gate || gate->op != GGML_OP_MUL_MAT || !gate->src[0] || !gate->src[1]) {
        return false;
    }
    if (!ane_ffn_name_is_ffn_gate(gate->src[0]->name)) {
        return false;
    }
    if (gate->src[1] != up->src[1]) {
        return false;
    }

    const struct ggml_tensor * gate_branch = gate;
    const struct ggml_tensor * gate_scale = NULL;
    if (i < n_available && is_post_mm_scale(nodes[i], gate)) {
        gate_scale = nodes[i++];
        gate_branch = gate_scale;
    }

    const int prefix_len = i;
    return match_glu_down(
        nodes, n_available, &i, prefix_len,
        up, up_branch, up_scale, gate, gate_branch, gate_scale, out);
}

static bool match_gate_first(
    const struct ggml_tensor * const * nodes,
    int n_available,
    ane_ffn_swiglu_fuse_t * out) {
    int i = 0;
    const struct ggml_tensor * gate = nodes[i++];
    if (!gate || gate->op != GGML_OP_MUL_MAT || !gate->src[0] || !gate->src[1]) {
        return false;
    }
    if (!ane_ffn_name_is_ffn_gate(gate->src[0]->name)) {
        return false;
    }

    const struct ggml_tensor * gate_branch = gate;
    const struct ggml_tensor * gate_scale = NULL;
    if (i < n_available && is_post_mm_scale(nodes[i], gate)) {
        gate_scale = nodes[i++];
        gate_branch = gate_scale;
    }

    if (i >= n_available) {
        return false;
    }
    const struct ggml_tensor * up = nodes[i++];
    if (!up || up->op != GGML_OP_MUL_MAT || !up->src[0] || !up->src[1]) {
        return false;
    }
    if (!ane_ffn_name_is_ffn_up(up->src[0]->name)) {
        return false;
    }
    if (up->src[1] != gate->src[1]) {
        return false;
    }

    const struct ggml_tensor * up_branch = up;
    const struct ggml_tensor * up_scale = NULL;
    if (i < n_available && is_post_mm_scale(nodes[i], up)) {
        up_scale = nodes[i++];
        up_branch = up_scale;
    }

    const int prefix_len = i;
    return match_glu_down(
        nodes, n_available, &i, prefix_len,
        up, up_branch, up_scale, gate, gate_branch, gate_scale, out);
}

bool ane_ffn_swiglu_fuse_match(
    const struct ggml_tensor * const * nodes,
    int n_available,
    ane_ffn_swiglu_fuse_t * out) {
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (!nodes || n_available < 4 || !out) {
        return false;
    }

    if (match_up_first(nodes, n_available, out)) {
        return true;
    }
    memset(out, 0, sizeof(*out));
    return match_gate_first(nodes, n_available, out);
}
