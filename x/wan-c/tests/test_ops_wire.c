/* Host-only wire: LN → AdaLN → RoPE3 → GEMM → GroupNorm → Conv2d → Unpatchify */
#include "wan_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  enum { rows = 4, D = 8, C = 4, S = 4 };
  float x[rows * D], y[rows * D], scale[D], shift[D], w[D * D];
  float gn_in[C * S], gn_out[C * S], conv_w[C * C], conv_out[C * S];
  float up_in[C * S], up_out[C * S];

  for (int i = 0; i < rows * D; i++)
    x[i] = 0.1f * (float)(i - 7);
  for (int i = 0; i < D; i++) {
    scale[i] = 0.05f;
    shift[i] = 0.01f;
  }
  wan_fill_eye_nt(w, D, D);

  uma_wan_layernorm_f32(y, x, NULL, NULL, rows, D, 1e-6f);
  uma_wan_affine_mul_add_f32(y, y, scale, shift, rows, D);
  if (wan_rope3_tokens(y, rows, 1, D) != 0) {
    fprintf(stderr, "FAIL rope\n");
    return 1;
  }
  float z[rows * D];
  uma_wan_gemm_f32(z, y, w, rows, D, D);

  for (int i = 0; i < C * S; i++)
    gn_in[i] = 0.2f * (float)((i % 5) - 2);
  uma_wan_groupnorm_f32(gn_out, gn_in, NULL, NULL, 1, C, S, 2, 1e-6f);
  wan_fill_eye_nt(conv_w, C, C);
  uma_wan_conv2d_f32(conv_out, gn_out, conv_w, NULL, 1, C, 2, 2, C, 1, 1, 1, 0);
  memcpy(up_in, conv_out, sizeof(up_in));
  uma_wan_unpatchify3d_f32(up_out, up_in, 1, C, 1, 2, 2, 1, 1, 1);

  float acc = 0.f;
  for (int i = 0; i < rows * D; i++)
    acc += fabsf(z[i]);
  for (int i = 0; i < C * S; i++)
    acc += fabsf(up_out[i]);
  if (!(acc > 0.f) || !isfinite(acc)) {
    fprintf(stderr, "FAIL acc=%g\n", acc);
    return 1;
  }
  printf("test_ops_wire OK acc=%.3g\n", acc);
  return 0;
}
