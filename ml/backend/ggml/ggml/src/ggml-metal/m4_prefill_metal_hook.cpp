// Opt-in Metal fused Gate+Up SwiGLU for prefill (m4-prefill-engine borrowings).
#include "m4_prefill_metal_hook.h"

#include "ane_ffn_swiglu_fuse.h"

#include "ggml-backend-impl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static ggml_metal_buffer_id m4_prefill_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;
    return ggml_metal_buffer_get_id(ctx, t);
}

static bool m4_prefill_swiglu_want(void) {
    const char * e = getenv("ZEROLLAMA_M4_PREFILL_SWIGLU");
    if (!e || !e[0] || e[0] == '0') {
        return false;
    }
    if (e[0] == 'f' || e[0] == 'F' || e[0] == 'n' || e[0] == 'N') {
        return false;
    }
    return true;
}

static int m4_prefill_skip_through_glu(
    const ggml_tensor * const * nodes,
    int n_look,
    const ggml_tensor * glu) {
    for (int k = 0; k < n_look; ++k) {
        if (nodes[k] == glu) {
            return k + 1;
        }
    }
    return 0;
}

int m4_prefill_metal_op_mul_mat_try(
    ggml_metal_op_t ctx,
    ggml_metal_library_t lib,
    ggml_metal_encoder_t enc,
    int idx,
    ggml_tensor * op,
    int ic,
    int oc,
    int seq) {
    if (!m4_prefill_swiglu_want() || !ctx || !lib || !enc || !op) {
        return 0;
    }

    const char * wname = op->src[0] ? op->src[0]->name : nullptr;
    if (!ane_ffn_name_is_ffn_up(wname) && !ane_ffn_name_is_ffn_gate(wname)) {
        return 0;
    }

    // Prefill-oriented: need a full tile row for the PoC 32-wide kernel.
    if (seq < 32) {
        return 0;
    }
    if (ic <= 0 || (ic % 32) != 0) {
        return 0;
    }

    if (idx + 3 >= ggml_metal_op_n_nodes(ctx)) {
        return 0;
    }

    const int n_avail = ggml_metal_op_n_nodes(ctx) - idx;
    const int n_look = n_avail > 48 ? 48 : n_avail;
    const ggml_tensor * nodes[48];
    for (int k = 0; k < n_look; ++k) {
        nodes[k] = ggml_metal_op_node(ctx, idx + k);
    }

    ane_ffn_swiglu_fuse_t fuse;
    memset(&fuse, 0, sizeof(fuse));
    if (!ane_ffn_swiglu_fuse_match(nodes, n_look, &fuse)) {
        return 0;
    }

    // v1: dense contiguous chain only; no post-mm scales; leave down to stock Metal.
    if (fuse.holey || fuse.up_scale || fuse.gate_scale || !fuse.up || !fuse.gate || !fuse.glu) {
        return 0;
    }
    if (fuse.ic != ic || fuse.seq != seq) {
        return 0;
    }
    if (fuse.hidden <= 0 || (fuse.hidden % 32) != 0) {
        return 0;
    }

    const ggml_tensor * w_gate = fuse.gate->src[0];
    const ggml_tensor * w_up   = fuse.up->src[0];
    const ggml_tensor * act    = fuse.gate->src[1];
    const ggml_tensor * glu    = fuse.glu;
    if (!w_gate || !w_up || !act || !glu) {
        return 0;
    }
    if (w_gate->type != GGML_TYPE_Q4_0 || w_up->type != GGML_TYPE_Q4_0) {
        return 0;
    }
    if (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(act) || !ggml_is_contiguous(glu)) {
        return 0;
    }
    if (!ggml_is_contiguous(w_gate) || !ggml_is_contiguous(w_up)) {
        return 0;
    }
    // Shared activation between gate and up.
    if (fuse.up->src[1] != act) {
        return 0;
    }
    if ((int) act->ne[0] != fuse.ic || (int) act->ne[1] != fuse.seq) {
        return 0;
    }
    if ((int) glu->ne[0] != fuse.hidden || (int) glu->ne[1] != fuse.seq) {
        return 0;
    }
    if ((int) w_gate->ne[0] != fuse.ic || (int) w_gate->ne[1] != fuse.hidden) {
        return 0;
    }
    if ((int) w_up->ne[0] != fuse.ic || (int) w_up->ne[1] != fuse.hidden) {
        return 0;
    }

    const int skip = m4_prefill_skip_through_glu(nodes, n_look, glu);
    if (skip < 3) {
        return 0;
    }

    const char * kname = "fused_gate_up_swiglu_q4_0_f32";
    ggml_metal_pipeline_with_params pipe = ggml_metal_library_get_pipeline(lib, kname);
    if (!pipe.pipeline) {
        pipe = ggml_metal_library_compile_pipeline(lib, kname, kname, nullptr);
    }
    if (!pipe.pipeline) {
        static bool once = false;
        if (!once) {
            once = true;
            fprintf(stderr, "m4_prefill: pipeline '%s' missing — fall through\n", kname);
        }
        return 0;
    }

    uint32_t M     = (uint32_t) fuse.seq;
    uint32_t N_mlp = (uint32_t) fuse.hidden;
    uint32_t K     = (uint32_t) fuse.ic;

    ggml_metal_encoder_set_pipeline(enc, pipe);
    ggml_metal_encoder_set_buffer(enc, m4_prefill_buffer_id(act),    0);
    ggml_metal_encoder_set_buffer(enc, m4_prefill_buffer_id(w_gate), 1);
    ggml_metal_encoder_set_buffer(enc, m4_prefill_buffer_id(w_up),   2);
    ggml_metal_encoder_set_buffer(enc, m4_prefill_buffer_id(glu),    3);
    ggml_metal_encoder_set_bytes(enc, &M,     sizeof(M),     4);
    ggml_metal_encoder_set_bytes(enc, &N_mlp, sizeof(N_mlp), 5);
    ggml_metal_encoder_set_bytes(enc, &K,     sizeof(K),     6);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, 4096, 0);

    const int ngx = (int) ((N_mlp + 31u) / 32u);
    const int ngy = (int) ((M + 31u) / 32u);
    ggml_metal_encoder_dispatch_threadgroups(enc, ngx, ngy, 1, 32, 1, 1);

    {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "m4_prefill: fused SWIGLU opt-in active (experimental; "
                    "lab A/B on 0.5B Q4_0 was ~16%% slower than stock Metal mul_mm+swiglu)\n");
        }
    }

    if (getenv("ZEROLLAMA_M4_PREFILL_TELEMETRY") &&
        getenv("ZEROLLAMA_M4_PREFILL_TELEMETRY")[0] == '1') {
        fprintf(stderr,
                "m4_prefill: fused swiglu ic=%d hidden=%d seq=%d skip=%d oc_anchor=%d name=%s\n",
                fuse.ic, fuse.hidden, fuse.seq, skip, oc, wname ? wname : "");
    }

    return skip;
}
