/* Shared f32 elementwise DiT ops — host reference + CUDA launches. */
#ifndef WAN_BACKEND_OPS_H
#define WAN_BACKEND_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Row-wise LN over last dim D. eps=1e-6. Optional w[D] (NULL → no affine weight). */
void wan_op_layernorm_f32(float *y, const float *x, const float *w, int N, int D,
                          float eps);

/* y = x * (1 + scale) + shift ; scale/shift length D (broadcast over N rows). */
void wan_op_affine_mul_add_f32(float *y, const float *x, const float *scale,
                               const float *shift, int N, int D);

/* GELU tanh approx (Wan / GPT-2 style). */
void wan_op_gelu_tanh_f32(float *y, const float *x, size_t n);

/* y[t,d] = x[t,d] + delta[t,d] * gate[d] */
void wan_op_gated_residual_f32(float *y, const float *x, const float *delta,
                               const float *gate, int N, int D);

/* Token RMSNorm over last dim D (WanRMSNorm). w length D (NULL → 1). */
void wan_op_rmsnorm_f32(float *y, const float *x, const float *w, int N, int D,
                        float eps);

/* y[n,d] = x[n,d] + bias[d] */
void wan_op_bias_add_f32(float *y, const float *x, const float *bias, int N,
                         int D);

/* y = x * scale + bias */
void wan_op_scale_bias_f32(float *y, const float *x, const float *scale,
                           const float *bias, int N, int D);

/* Transpose PyTorch Linear [out,in] → gemm B [in,out]. */
void wan_op_transpose_oi_f32(float *dst, const float *src, int out, int in);

/* Per-head RMSNorm: x/y layout [T, H, HD] row-major; w length HD (NULL → 1). */
void wan_op_head_rmsnorm_f32(float *y, const float *x, const float *w, int T,
                             int H, int HD, float eps);

/* Wan-style 3-axis RoPE in-place/out on [T,H,HD]; grid_t*grid_h*grid_w == T. */
int wan_op_rope3_f32(float *y, const float *x, int T, int H, int HD, int grid_t,
                     int grid_h, int grid_w);

/* SDPA: q[T,H,HD], k/v[Tk,H,HD] → out[T,H,HD]. scale = 1/sqrt(HD). */
void wan_op_attn_sdpa_f32(float *out, const float *q, const float *k,
                          const float *v, int T, int Tk, int H, int HD);

/*
 * DiT patch_embedding: Conv3d (stride=kernel, pad=0) + NCDHW→tokens [T,Cout].
 * x [Cin,Tin,Hin,Win], w PyTorch [Cout,Cin,kt,kh,kw], bias [Cout] or NULL.
 * Requires Tin%kt==0, Hin%kh==0, Win%kw==0. T_out = (Tin/kt)*(Hin/kh)*(Win/kw).
 */
int wan_op_patch_embed_f32(float *tok, const float *x, const float *w,
                           const float *bias, int Cin, int Tin, int Hin, int Win,
                           int Cout, int kt, int kh, int kw);

/* Device variants. */
int wan_cuda_layernorm_f32(float *y, const float *x, const float *w, int N,
                           int D, float eps);
int wan_cuda_affine_mul_add_f32(float *y, const float *x, const float *scale,
                                const float *shift, int N, int D);
int wan_cuda_gelu_tanh_f32(float *y, const float *x, size_t n);
int wan_cuda_gated_residual_f32(float *y, const float *x, const float *delta,
                                const float *gate, int N, int D);
int wan_cuda_rmsnorm_f32(float *y, const float *x, const float *w, int N, int D,
                         float eps);
int wan_cuda_bias_add_f32(float *y, const float *x, const float *bias, int N,
                          int D);
int wan_cuda_scale_bias_f32(float *y, const float *x, const float *scale,
                            const float *bias, int N, int D);
int wan_cuda_head_rmsnorm_f32(float *y, const float *x, const float *w, int T,
                              int H, int HD, float eps);
int wan_cuda_attn_sdpa_f32(float *out, const float *q, const float *k,
                           const float *v, int T, int Tk, int H, int HD);

#ifdef __cplusplus
}
#endif

#endif
