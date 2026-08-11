// Layout pack helpers: ggml ne0-first ↔ ANE channel-major (no ggml/ANE deps).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// is_f16: 0 = float32 source/dest, 1 = IEEE fp16 (ggml_fp16_t / _Float16 layout).
// Weight ggml contiguous [oc][ic] (ne0=ic, ne1=oc) → float W[oc*ic] same order.
bool ane_ffn_pack_weight_to_f32(
    const void *src, int is_f16, int ic, int oc, float *dst_oc_ic);

// Acts ggml contiguous [seq][ic] (ne0=ic, ne1=seq) → channel-major X[ic][seq].
bool ane_ffn_pack_acts_ggml_to_channel(
    const void *src, int is_f16, int ic, int seq, float *dst_ic_seq);

// Same transpose, direct to IEEE fp16 channel-major (skips f32 staging).
bool ane_ffn_pack_acts_ggml_to_channel_f16(
    const void *src, int is_f16, int ic, int seq, void *dst_ic_seq_f16);

// ggml [seq][ic] → channel-major int8 [ic][seq] with global symmetric scale.
// scale must be > 0 (typically max(|x|)/127 from calibration).
bool ane_ffn_pack_acts_ggml_to_channel_i8(
    const void *src, int is_f16, int ic, int seq, float scale, int8_t *dst_ic_seq);

// Channel-major Y[oc][seq] → ggml contiguous [seq][oc] (ne0=oc, ne1=seq).
bool ane_ffn_unpack_dst_channel_to_ggml(
    const float *src_oc_seq, int is_f16, int oc, int seq, void *dst);

// IEEE fp16 channel-major → ggml (skips f32 staging when dst is f16).
bool ane_ffn_unpack_dst_channel_f16_to_ggml(
    const void *src_oc_seq_f16, int dst_is_f16, int oc, int seq, void *dst);

// Fold post-mm scale into weight rows: W[o,i] *= s[o] (or scalar s[0]).
// nscale: 1 (broadcast) or oc. is_f16 applies to scale storage.
bool ane_ffn_fold_out_scale_f32(
    float *W_oc_ic, int ic, int oc,
    const void *scale, int scale_is_f16, int nscale);

#ifdef __cplusplus
}
#endif
