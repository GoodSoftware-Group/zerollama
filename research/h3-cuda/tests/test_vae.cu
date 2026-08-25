/* Video VAE pad / conv3d / group_norm+silu smoke. */
#include "h3_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int die(const char *m) {
  std::fprintf(stderr, "FAIL: %s\n", m);
  return 1;
}
#define OP(c, l)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "FAIL %s: %s\n", l, h3_gpu_error(gpu));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);

  /* Pad: B=1 D=2 H=4 W=4 C=2; front=1, spatial before/after=1 */
  const uint32_t B = 1, D = 2, H = 4, W = 4, C = 2;
  const uint32_t df = 1, hb = 1, ha = 1, wb = 1, wa = 1;
  std::vector<float> in(B * D * H * W * C);
  for (size_t i = 0; i < in.size(); i++) in[i] = 0.1f * (float)(i + 1);
  uint32_t od = D + df, oh = H + hb + ha, ow = W + wb + wa;
  auto *tin = h3_gpu_tensor_from_f32(gpu, in.data(), in.size());
  auto *tpadded = h3_gpu_tensor_new_f32(gpu, B * od * oh * ow * C);
  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_vae_encoder_pad_f32(gpu, tpadded, tin, B, D, H, W, C, df, hb, ha,
                                wb, wa),
     "pad");
  OP(h3_gpu_submit(gpu), "submit pad");
  std::vector<float> pad(B * od * oh * ow * C);
  OP(h3_gpu_tensor_read_f32(tpadded, pad.data(), pad.size()), "read pad");
  /* Temporal front plane must be zero. */
  for (uint32_t y = 0; y < oh; y++)
    for (uint32_t x = 0; x < ow; x++)
      for (uint32_t c = 0; c < C; c++) {
        size_t i = (((size_t)0 * od + 0) * oh + y) * ow * C + x * C + c;
        if (pad[i] != 0.f) return die("front pad not zero");
      }
  /* Interior (t=1 maps to source t=0), spatial (1,1) → source (0,0) */
  size_t dest =
      (((size_t)0 * od + 1) * oh + hb) * ow * C + wb * C + 0;
  if (std::fabs(pad[dest] - in[0]) > 1e-6f) return die("pad copy mismatch");
  std::printf("ok vae_encoder_pad_f32\n");

  /* Conv3d: small 2x2x2 -> 1x1x1 with k=2 s=1, Cin=2 Cout=3 */
  const uint32_t kd = 2, kh = 2, kw = 2, Cout = 3;
  std::vector<float> w(Cout * C * kd * kh * kw), bias(Cout, 0.05f);
  for (size_t i = 0; i < w.size(); i++) w[i] = 0.01f * (float)((i % 7) + 1);
  /* Use unpadded tiny volume for clearer reference */
  const uint32_t d2 = 2, h2 = 2, w2 = 2;
  std::vector<float> xin(B * d2 * h2 * w2 * C);
  for (size_t i = 0; i < xin.size(); i++) xin[i] = 0.2f * (float)(i + 1);
  uint32_t od2 = 1, oh2 = 1, ow2 = 1;
  std::vector<float> yref(B * od2 * oh2 * ow2 * Cout, 0.f);
  for (uint32_t oc = 0; oc < Cout; oc++) {
    float s = bias[oc];
    for (uint32_t ic = 0; ic < C; ic++)
      for (uint32_t zd = 0; zd < kd; zd++)
        for (uint32_t yh = 0; yh < kh; yh++)
          for (uint32_t xw = 0; xw < kw; xw++) {
            size_t ii =
                ((((size_t)0 * d2 + zd) * h2 + yh) * w2 + xw) * C + ic;
            size_t wi =
                ((((size_t)oc * C + ic) * kd + zd) * kh + yh) * kw + xw;
            s += xin[ii] * w[wi];
          }
    yref[oc] = s;
  }
  auto *tx = h3_gpu_tensor_from_f32(gpu, xin.data(), xin.size());
  auto *tw = h3_gpu_tensor_from_f32(gpu, w.data(), w.size());
  auto *tb = h3_gpu_tensor_from_f32(gpu, bias.data(), bias.size());
  auto *ty = h3_gpu_tensor_new_f32(gpu, Cout);
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_conv3d_f32(gpu, ty, tx, tw, tb, B, d2, h2, w2, C, Cout, kd, kh, kw,
                       1, 1, 1),
     "conv3d");
  OP(h3_gpu_submit(gpu), "submit conv");
  std::vector<float> y(Cout);
  OP(h3_gpu_tensor_read_f32(ty, y.data(), y.size()), "read conv");
  float max_abs = 0.f;
  for (uint32_t i = 0; i < Cout; i++)
    max_abs = std::fmax(max_abs, std::fabs(y[i] - yref[i]));
  if (max_abs > 1e-5f) {
    std::fprintf(stderr, "FAIL conv3d max_abs=%g\n", max_abs);
    return 1;
  }
  std::printf("ok conv3d_f32 max_abs=%g\n", max_abs);

  /* Group norm + silu */
  std::vector<float> nw(C, 1.f), nb(C, 0.f);
  auto *tnw = h3_gpu_tensor_from_f32(gpu, nw.data(), nw.size());
  auto *tnb = h3_gpu_tensor_from_f32(gpu, nb.data(), nb.size());
  auto *tnorm = h3_gpu_tensor_new_f32(gpu, xin.size());
  OP(h3_gpu_begin(gpu), "begin3");
  OP(h3_gpu_vae_encoder_group_norm_silu_f32(gpu, tnorm, tx, tnw, tnb, B, d2, h2,
                                           w2, C, 2, 1e-5f),
     "gn");
  OP(h3_gpu_submit(gpu), "submit gn");
  std::vector<float> gn(xin.size());
  OP(h3_gpu_tensor_read_f32(tnorm, gn.data(), gn.size()), "read gn");
  float gsum = 0.f;
  for (float v : gn) gsum += std::fabs(v);
  if (!(gsum > 0.f) || !std::isfinite(gsum)) return die("gn bad");
  std::printf("ok vae_group_norm_silu  |out|=%.4f\n", gsum);

  h3_gpu_free(gpu);
  std::printf("vae smoke passed\n");
  return 0;
}
