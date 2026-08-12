/* Token-reduction fusions + AudioVAE conv1d smoke. */
#include "h3_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f2b(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}
static float b2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
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

  const uint32_t in_rows = 4, pooled = 2, dim = 16, slots = 2;
  std::vector<uint16_t> src(in_rows * dim);
  for (size_t i = 0; i < src.size(); i++) src[i] = f2b(0.2f * (float)(i + 1));
  std::vector<uint32_t> pairs = {0, 1, 2, 3};
  std::vector<uint32_t> bidx = {0, 1};
  std::vector<uint16_t> nw(dim, f2b(1.f)), mod(slots * dim, f2b(0.f));
  std::vector<uint32_t> rmap(pooled, 0);

  auto *tsrc = h3_gpu_tensor_from_bf16(gpu, src.data(), src.size());
  auto *tpairs = h3_gpu_tensor_from_u32(gpu, pairs.data(), pairs.size());
  auto *tbidx = h3_gpu_tensor_from_u32(gpu, bidx.data(), bidx.size());
  auto *tnw = h3_gpu_tensor_from_bf16(gpu, nw.data(), nw.size());
  auto *tmod = h3_gpu_tensor_from_bf16(gpu, mod.data(), mod.size());
  auto *trmap = h3_gpu_tensor_from_u32(gpu, rmap.data(), rmap.size());
  auto *tres = h3_gpu_tensor_new_bf16(gpu, pooled * dim);
  auto *tout = h3_gpu_tensor_new_bf16(gpu, pooled * dim);
  auto *torig = h3_gpu_tensor_new_bf16(gpu, in_rows * dim);
  auto *tbase = h3_gpu_tensor_new_bf16(gpu, pooled * dim);

  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_token_pool_adaln_bf16(gpu, tres, tout, tsrc, 0, torig, 0, tbase, 0,
                                  tbidx, tpairs, tnw, tmod, trmap, in_rows,
                                  pooled, pooled, dim, slots, 0, 1, 1e-5f),
     "pool_adaln");
  OP(h3_gpu_submit(gpu), "submit1");
  std::vector<uint16_t> res(pooled * dim), adn(pooled * dim);
  OP(h3_gpu_tensor_read_bf16(tres, res.data(), res.size()), "read res");
  OP(h3_gpu_tensor_read_bf16(tout, adn.data(), adn.size()), "read adn");
  float avg = 0.5f * (b2f(src[0]) + b2f(src[dim]));
  if (std::fabs(b2f(res[0]) - avg) > 0.05f) return die("pool residual");
  if (!(std::fabs(b2f(adn[0])) > 0.f)) return die("adaln zero");
  std::printf("ok token_pool_adaln\n");

  /* Expand: parents map full rows → reduced parents */
  std::vector<uint32_t> parents = {0, 0, 1, 1};
  auto *tpar = h3_gpu_tensor_from_u32(gpu, parents.data(), parents.size());
  auto *texp = h3_gpu_tensor_new_bf16(gpu, in_rows * dim);
  /* Use residual as reduced (pooled values) */
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_token_expand_delta_bf16(gpu, texp, torig, 0, tres, tbase, 0, tbidx,
                                   tpar, in_rows, pooled, pooled, dim, 0, 1.f),
     "expand_delta");
  OP(h3_gpu_submit(gpu), "submit2");
  std::vector<uint16_t> expv(in_rows * dim);
  OP(h3_gpu_tensor_read_bf16(texp, expv.data(), expv.size()), "read exp");
  /* With update_scale=1 and reduced==baseline, update=0 → restore original */
  if (std::fabs(b2f(expv[0]) - b2f(src[0])) > 0.05f) return die("expand orig");
  std::printf("ok token_expand_delta\n");

  auto *tres2 = h3_gpu_tensor_new_bf16(gpu, in_rows * dim);
  auto *tadn2 = h3_gpu_tensor_new_bf16(gpu, in_rows * dim);
  std::vector<uint32_t> rmap2(in_rows, 0);
  auto *trmap2 = h3_gpu_tensor_from_u32(gpu, rmap2.data(), rmap2.size());
  OP(h3_gpu_begin(gpu), "begin3");
  OP(h3_gpu_token_expand_adaln_bf16(gpu, tres2, tadn2, torig, 0, tres, tbase, 0,
                                   tbidx, tpar, tnw, tmod, trmap2, in_rows,
                                   pooled, pooled, dim, 0, 1.f, slots, 0, 1,
                                   1e-5f),
     "expand_adaln");
  OP(h3_gpu_submit(gpu), "submit3");
  std::printf("ok token_expand_adaln\n");

  /* conv1d: B=1 L=8 Cin=2 Cout=3 K=3 pad=1 → out_len=8 */
  const uint32_t B = 1, L = 8, Cin = 2, Cout = 3, K = 3, pad = 1;
  std::vector<float> xin(B * L * Cin), w(Cout * Cin * K), bias(Cout, 0.1f);
  for (size_t i = 0; i < xin.size(); i++) xin[i] = 0.05f * (float)(i + 1);
  for (size_t i = 0; i < w.size(); i++) w[i] = 0.02f * (float)((i % 5) + 1);
  uint32_t out_len = L; /* stride 1, pad 1, k 3 */
  std::vector<float> yref(B * out_len * Cout, 0.f);
  for (uint32_t t = 0; t < out_len; t++)
    for (uint32_t oc = 0; oc < Cout; oc++) {
      float s = bias[oc];
      for (uint32_t ic = 0; ic < Cin; ic++)
        for (uint32_t k = 0; k < K; k++) {
          int it = (int)t - (int)pad + (int)k;
          if (it < 0 || it >= (int)L) continue;
          s += xin[(size_t)it * Cin + ic] * w[((size_t)oc * Cin + ic) * K + k];
        }
      yref[(size_t)t * Cout + oc] = s;
    }
  auto *tx = h3_gpu_tensor_from_f32(gpu, xin.data(), xin.size());
  auto *tw = h3_gpu_tensor_from_f32(gpu, w.data(), w.size());
  auto *tb = h3_gpu_tensor_from_f32(gpu, bias.data(), bias.size());
  auto *ty = h3_gpu_tensor_new_f32(gpu, B * out_len * Cout);
  OP(h3_gpu_begin(gpu), "begin4");
  OP(h3_gpu_conv1d_f32(gpu, ty, tx, tw, tb, B, L, Cin, Cout, K, pad, 1),
     "conv1d");
  OP(h3_gpu_submit(gpu), "submit4");
  std::vector<float> y(B * out_len * Cout);
  OP(h3_gpu_tensor_read_f32(ty, y.data(), y.size()), "read conv");
  float max_abs = 0.f;
  for (size_t i = 0; i < y.size(); i++)
    max_abs = std::fmax(max_abs, std::fabs(y[i] - yref[i]));
  if (max_abs > 1e-5f) {
    std::fprintf(stderr, "FAIL conv max_abs=%g\n", max_abs);
    return 1;
  }
  std::printf("ok conv1d_f32 max_abs=%g\n", max_abs);

  h3_gpu_free(gpu);
  std::printf("token+conv smoke passed\n");
  return 0;
}
