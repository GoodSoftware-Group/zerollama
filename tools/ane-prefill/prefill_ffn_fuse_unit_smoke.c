// Unit smoke: ane_ffn_swiglu_fuse_match — clean + optional scale MULs (no ANE).
#include "ane_ffn_swiglu_fuse.h"
#include "ane_ffn_force_pack.h"
#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// --- ggml stubs (smoke only) ---
size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_F32: return 4;
        default: return 0;
    }
}

int64_t ggml_blck_size(enum ggml_type type) {
    (void)type;
    return 1;
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return 0;
    }
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    size_t expected = ggml_type_size(tensor->type);
    if (expected == 0 || ggml_blck_size(tensor->type) != 1) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1 && tensor->nb[i] != expected) {
            return false;
        }
        expected *= (size_t)tensor->ne[i];
    }
    return true;
}

enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * tensor) {
    return (enum ggml_glu_op) tensor->op_params[0];
}

static void set_contig(struct ggml_tensor * t, enum ggml_type type, int64_t ne0, int64_t ne1) {
    t->type = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = t->nb[0] * (size_t)ne0;
    t->nb[2] = t->nb[1] * (size_t)ne1;
    t->nb[3] = t->nb[2];
}

static void set_name(struct ggml_tensor * t, const char * name) {
    snprintf(t->name, sizeof(t->name), "%s", name);
}

static int expect_match(
    const struct ggml_tensor * const * nodes, int n,
    int want_n_fuse, int want_encode_skip, int want_holey,
    int ic, int hidden, int seq, int want_scales) {
    ane_ffn_swiglu_fuse_t fuse;
    if (!ane_ffn_swiglu_fuse_match(nodes, n, &fuse)) {
        return 0;
    }
    if (fuse.n_fuse != want_n_fuse || fuse.n_encode_skip != want_encode_skip ||
        fuse.holey != want_holey || fuse.ic != ic || fuse.hidden != hidden ||
        fuse.seq != seq) {
        return 0;
    }
    int have = (fuse.up_scale != NULL) + (fuse.gate_scale != NULL) + (fuse.down_scale != NULL);
    if (have != want_scales) {
        return 0;
    }
    if (fuse.dst != (fuse.down_scale ? fuse.down_scale : fuse.down)) {
        return 0;
    }
    return 1;
}

int main(void) {
    const int ic = 256;
    const int hidden = 128;
    const int seq = 64;

    struct ggml_tensor Wu, Wg, Wd, X, Su, Sg, Sd, up, up_s, gate, gate_s, glu, down, down_s;
    memset(&Wu, 0, sizeof(Wu));
    memset(&Wg, 0, sizeof(Wg));
    memset(&Wd, 0, sizeof(Wd));
    memset(&X, 0, sizeof(X));
    memset(&Su, 0, sizeof(Su));
    memset(&Sg, 0, sizeof(Sg));
    memset(&Sd, 0, sizeof(Sd));
    memset(&up, 0, sizeof(up));
    memset(&up_s, 0, sizeof(up_s));
    memset(&gate, 0, sizeof(gate));
    memset(&gate_s, 0, sizeof(gate_s));
    memset(&glu, 0, sizeof(glu));
    memset(&down, 0, sizeof(down));
    memset(&down_s, 0, sizeof(down_s));

    set_contig(&Wu, GGML_TYPE_F16, ic, hidden);
    set_contig(&Wg, GGML_TYPE_F16, ic, hidden);
    set_contig(&Wd, GGML_TYPE_F16, hidden, ic);
    set_contig(&X,  GGML_TYPE_F16, ic, seq);
    set_contig(&Su, GGML_TYPE_F32, 1, 1); // scalar scales
    set_contig(&Sg, GGML_TYPE_F32, 1, 1);
    set_contig(&Sd, GGML_TYPE_F32, 1, 1);
    set_name(&Wu, "blk.0.ffn_up_shexp.weight");
    set_name(&Wg, "blk.0.ffn_gate_shexp.weight");
    set_name(&Wd, "blk.0.ffn_down_shexp.weight");
    Su.op = GGML_OP_NONE;
    Sg.op = GGML_OP_NONE;
    Sd.op = GGML_OP_NONE;

    up.op = GGML_OP_MUL_MAT;
    up.src[0] = &Wu;
    up.src[1] = &X;
    set_contig(&up, GGML_TYPE_F16, hidden, seq);

    gate.op = GGML_OP_MUL_MAT;
    gate.src[0] = &Wg;
    gate.src[1] = &X;
    set_contig(&gate, GGML_TYPE_F16, hidden, seq);

    glu.op = GGML_OP_GLU;
    glu.op_params[0] = (int32_t) GGML_GLU_OP_SWIGLU;
    glu.src[0] = &gate;
    glu.src[1] = &up;
    set_contig(&glu, GGML_TYPE_F16, hidden, seq);

    down.op = GGML_OP_MUL_MAT;
    down.src[0] = &Wd;
    down.src[1] = &glu;
    set_contig(&down, GGML_TYPE_F16, ic, seq);

    const struct ggml_tensor * clean[4] = { &up, &gate, &glu, &down };
    if (!expect_match(clean, 4, 4, 4, 0, ic, hidden, seq, 0)) {
        printf("{\"ok\":false,\"error\":\"clean fuse failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
        return 1;
    }

    // Metal topo often schedules independent gate before up.
    const struct ggml_tensor * gate_first[4] = { &gate, &up, &glu, &down };
    if (!expect_match(gate_first, 4, 4, 4, 0, ic, hidden, seq, 0)) {
        printf("{\"ok\":false,\"error\":\"gate-first fuse failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
        return 1;
    }

    // up_s / gate_s / down_s intercalated
    up_s.op = GGML_OP_MUL;
    up_s.src[0] = &up;
    up_s.src[1] = &Su;
    set_contig(&up_s, GGML_TYPE_F16, hidden, seq);

    gate_s.op = GGML_OP_MUL;
    gate_s.src[0] = &gate;
    gate_s.src[1] = &Sg;
    set_contig(&gate_s, GGML_TYPE_F16, hidden, seq);

    glu.src[0] = &gate_s;
    glu.src[1] = &up_s;

    down_s.op = GGML_OP_MUL;
    down_s.src[0] = &down;
    down_s.src[1] = &Sd;
    set_contig(&down_s, GGML_TYPE_F16, ic, seq);

    const struct ggml_tensor * scaled[7] = {
        &up, &up_s, &gate, &gate_s, &glu, &down, &down_s,
    };
    if (!expect_match(scaled, 7, 7, 7, 0, ic, hidden, seq, 3)) {
        printf("{\"ok\":false,\"error\":\"scaled fuse failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
        return 1;
    }

    // MoE shexp: router / ARGSORT interleaved between up and GLU / GLU and down.
    {
        struct ggml_tensor junk0, junk1, junk2;
        memset(&junk0, 0, sizeof(junk0));
        memset(&junk1, 0, sizeof(junk1));
        memset(&junk2, 0, sizeof(junk2));
        junk0.op = GGML_OP_ARGSORT;
        junk1.op = GGML_OP_MUL_MAT;
        junk2.op = GGML_OP_UNARY;
        // Restore glu srcs after scaled test mutated them.
        glu.src[0] = &gate;
        glu.src[1] = &up;
        const struct ggml_tensor * holey[7] = {
            &gate, &up, &junk1, &junk0, &glu, &junk2, &down,
        };
        if (!expect_match(holey, 7, 4, 2, 1, ic, hidden, seq, 0)) {
            printf("{\"ok\":false,\"error\":\"holey moe fuse failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
            return 1;
        }
    }

    // Fold helper: W rows *= scalar
    float W[4] = { 1, 2, 3, 4 }; // oc=2, ic=2
    float s = 2.0f;
    if (!ane_ffn_fold_out_scale_f32(W, 2, 2, &s, 0, 1) ||
        fabsf(W[0] - 2.f) > 1e-6f || fabsf(W[3] - 8.f) > 1e-6f) {
        printf("{\"ok\":false,\"error\":\"fold scale failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
        return 1;
    }

    // ggml [seq][ic] → fp16 channel [ic][seq] transpose
    {
        float ggml_x[6] = { /*t0*/ 1, 2, /*t1*/ 3, 4, /*t2*/ 5, 6 }; // ic=2 seq=3
        uint16_t out[6];
        if (!ane_ffn_pack_acts_ggml_to_channel_f16(ggml_x, 0, 2, 3, out)) {
            printf("{\"ok\":false,\"error\":\"fp16 acts pack failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
            return 1;
        }
        float back[6];
        if (!ane_ffn_unpack_dst_channel_f16_to_ggml(out, 0, 2, 3, back) ||
            fabsf(back[0] - 1.f) > 1e-2f || fabsf(back[1] - 2.f) > 1e-2f) {
            printf("{\"ok\":false,\"error\":\"fp16 acts roundtrip failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
            return 1;
        }
    }

    // ggml [seq][ic] → int8 channel with scale
    {
        float ggml_x[4] = { 1.f, -2.f, 3.f, -4.f }; // ic=2 seq=2
        float scale = 4.f / 127.f;
        int8_t out[4];
        if (!ane_ffn_pack_acts_ggml_to_channel_i8(ggml_x, 0, 2, 2, scale, out)) {
            printf("{\"ok\":false,\"error\":\"i8 acts pack failed\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
            return 1;
        }
        // channel [i][t]: i0 → 1,3 ; i1 → -2,-4
        float r0 = (float)out[0] * scale; // ~1
        float r1 = (float)out[2] * scale; // ~-2
        if (fabsf(r0 - 1.f) > 0.05f || fabsf(r1 + 2.f) > 0.05f) {
            printf("{\"ok\":false,\"error\":\"i8 acts pack values\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
            return 1;
        }
    }

    if (ane_ffn_name_is_ffn_up("blk.0.ffn_up_exps.weight") ||
        !ane_ffn_name_is_ffn_swiglu_weight("blk.0.ffn_gate_shexp.weight")) {
        printf("{\"ok\":false,\"error\":\"name helper\",\"mode\":\"ane_ffn_fuse_unit\"}\n");
        return 1;
    }

    printf("{\"ok\":true,\"mode\":\"ane_ffn_fuse_unit\",\"n_fuse_clean\":4,\"n_fuse_gate_first\":4,"
           "\"n_fuse_scaled\":7,\"n_fuse_holey\":4,\"n_encode_skip_holey\":2,\"ic\":%d,\"hidden\":%d,\"seq\":%d,"
           "\"note\":\"up-first+gate-first+holey; scales fold into W; fp16+i8 acts pack\"}\n",
           ic, hidden, seq);
    return 0;
}
