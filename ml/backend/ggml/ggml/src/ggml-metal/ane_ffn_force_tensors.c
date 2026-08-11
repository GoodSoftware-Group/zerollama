// Metal/ggml tensor entry for lab ANE force replace.
#include "ane_ffn_policy.h"
#include "ane_ffn_force_pack.h"

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int type_is_f16(enum ggml_type t) {
    return t == GGML_TYPE_F16;
}

static int type_ok_act(enum ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_F32;
}

static bool pack_weight_tensor(
    const struct ggml_tensor * w, int ic, int oc, float * dst_oc_ic) {
    if (!w || !w->data || !dst_oc_ic || ic <= 0 || oc <= 0) {
        return false;
    }
    if (w->ne[0] != ic || w->ne[1] != oc || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (w->type == GGML_TYPE_F16) {
        return ane_ffn_pack_weight_to_f32(w->data, 1, ic, oc, dst_oc_ic);
    }
    if (w->type == GGML_TYPE_F32) {
        return ane_ffn_pack_weight_to_f32(w->data, 0, ic, oc, dst_oc_ic);
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(w->type);
    if (!tr || !tr->to_float || !tr->is_quantized) {
        return false;
    }
    const int64_t blck = ggml_blck_size(w->type);
    if (blck <= 0 || (ic % (int)blck) != 0) {
        return false;
    }
    for (int r = 0; r < oc; r++) {
        const void * row = (const char *)w->data + (size_t)r * w->nb[1];
        tr->to_float(row, dst_oc_ic + (size_t)r * (size_t)ic, ic);
    }
    return true;
}

static bool tensor_host_shared(const struct ggml_tensor * t) {
    if (!t || !t->data) {
        return false;
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buffer) {
        return true; // CPU tensor
    }
    // Metal shared buffers expose tensor->data to the CPU after cmd_buf sync.
    return ggml_metal_buffer_is_shared((ggml_metal_buffer_t) buffer->context);
}

bool ane_ffn_force_try_mul_mat_tensors(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst) {
    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (!tensor_host_shared(src0) || !tensor_host_shared(src1) || !tensor_host_shared(dst)) {
        return false;
    }
    if (!type_ok_act(src1->type) || !type_ok_act(dst->type)) {
        return false;
    }
    // v1: activations and dst same float width; weight may be F16/F32 or quantized.
    if (src1->type != dst->type) {
        return false;
    }
    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) {
        const struct ggml_type_traits * tr = ggml_get_type_traits(src0->type);
        if (!tr || !tr->to_float || !tr->is_quantized) {
            return false;
        }
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int ic  = (int)src0->ne[0];
    const int oc  = (int)src0->ne[1];
    const int seq = (int)src1->ne[1];
    if (src1->ne[0] != ic || dst->ne[0] != oc || dst->ne[1] != seq) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2]  != 1 || dst->ne[3]  != 1) {
        return false;
    }

    ane_ffn_force_autoload_host_replace();

    float *W = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
    float *X = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
    float *Y = (float *)malloc((size_t)oc * (size_t)seq * sizeof(float));
    if (!W || !X || !Y) {
        free(W); free(X); free(Y);
        return false;
    }

    bool ok = pack_weight_tensor(src0, ic, oc, W)
           && ane_ffn_pack_acts_ggml_to_channel(src1->data, type_is_f16(src1->type), ic, seq, X);
    if (!ok) {
        free(W); free(X); free(Y);
        return false;
    }

    ok = ane_ffn_force_try_mul_mat_host(ic, oc, seq, W, X, Y);
    if (ok) {
        ok = ane_ffn_unpack_dst_channel_to_ggml(Y, type_is_f16(dst->type), oc, seq, dst->data);
    }

    free(W);
    free(X);
    free(Y);
    return ok;
}
