// Metal ggml↔channel layout helpers for ANE FFN force path (lab).
// Cuts host CPU transpose tax (F0742): pack/unpack via compute into IOSurface.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lazy init Metal device/queue/pipelines. Returns false if Metal unavailable.
bool ane_ffn_layout_metal_ready(void);

// channel-major fp16 [oc×seq] on ANE output IOSurface → ggml [seq×oc] fp16 dst.
bool ane_ffn_layout_metal_unpack_out_f16(
    uint32_t out_surface_id,
    int oc, int seq,
    void *dst_ggml_f16);

// ggml [seq×ic] f32 → channel-major int8 [ic×seq] on ANE input IOSurface.
// scale: same as MIL dequant (max/127); kernel uses inv_scale = 1/scale.
bool ane_ffn_layout_metal_pack_in_i8_f32(
    uint32_t in_surface_id,
    const float *src_ggml_f32,
    int ic, int seq,
    float scale);

// ggml [seq×ic] fp16 → channel int8 on input surface.
bool ane_ffn_layout_metal_pack_in_i8_f16(
    uint32_t in_surface_id,
    const void *src_ggml_f16,
    int ic, int seq,
    float scale);

#ifdef __cplusplus
}
#endif
