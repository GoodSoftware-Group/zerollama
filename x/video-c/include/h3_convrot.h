/* Comfy-kitchen int8_tensorwise + ConvRot (comfy_quant_v1) host dequant. */
#ifndef H3_CONVROT_H
#define H3_CONVROT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Normalized FWHT (Walsh-Hadamard / sqrt(n)); n power of two. In-place. */
int h3_fwht_normalized(float *x, int n);

/*
 * Inverse ConvRot on row-major [rows, cols]: each gs-block along cols is
 * multiplied by H^T. H is orthonormal so this is the same as the forward
 * rotate used at quantize time.
 */
int h3_convrot_unrotate(float *w, int rows, int cols, int gs);

/*
 * q int8 [rows, cols], scale f32 [rows] (or [rows,1]), dst f32 [rows, cols].
 * dst may alias a caller buffer only (not q). gs==0 or 1 skips rotation.
 */
int h3_convrot_dequant_i8(const int8_t *q, int rows, int cols,
                          const float *scale, int gs, float *dst);

/*
 * Comfy int8_linear activation fake-quant: rotate (gs>1), per-row int8
 * round (amax/127), unrotate. In-place. gs<=1 skips rotation.
 */
int h3_convrot_fakequant_act(float *x, int rows, int cols, int gs);

/* Parse comfy_quant JSON bytes. Sets *out_gs (0 if rotation off). Returns 0. */
int h3_comfy_quant_parse(const uint8_t *bytes, size_t n, int *out_gs);

#ifdef __cplusplus
}
#endif

#endif /* H3_CONVROT_H */
