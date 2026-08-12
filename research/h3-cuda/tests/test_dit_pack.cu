/* Pack/final DiT ops: linear_f32, F32→BF16 patch, map scatter, adaln_linear. */
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

  const uint32_t rows = 4, in = 8, out = 6, width = 8, slots = 2;

  /* --- linear_f32 --- */
  std::vector<float> x(rows * in), w(out * in), b(out), yref(rows * out);
  for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (float)((i % 7) + 1);
  for (size_t i = 0; i < w.size(); i++) w[i] = 0.05f * (float)((i % 5) + 1);
  for (uint32_t o = 0; o < out; o++) b[o] = 0.01f * (float)(o + 1);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t o = 0; o < out; o++) {
      float acc = b[o];
      for (uint32_t k = 0; k < in; k++)
        acc += x[r * in + k] * w[o * in + k];
      yref[r * out + o] = acc;
    }
  auto *tx = h3_gpu_tensor_from_f32(gpu, x.data(), x.size());
  auto *tw = h3_gpu_tensor_from_f32(gpu, w.data(), w.size());
  auto *tb = h3_gpu_tensor_from_f32(gpu, b.data(), b.size());
  auto *ty = h3_gpu_tensor_new_f32(gpu, rows * out);
  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_linear_f32(gpu, ty, tx, tw, tb, rows, in, out), "linear_f32");
  OP(h3_gpu_submit(gpu), "submit f32");
  std::vector<float> y(rows * out);
  OP(h3_gpu_tensor_read_f32(ty, y.data(), y.size()), "read f32");
  float max_abs = 0.f;
  for (size_t i = 0; i < y.size(); i++)
    max_abs = std::fmax(max_abs, std::fabs(y[i] - yref[i]));
  if (max_abs > 1e-4f) {
    std::fprintf(stderr, "FAIL linear_f32 max_abs=%g\n", max_abs);
    return 1;
  }
  std::printf("ok linear_f32 max_abs=%g\n", max_abs);

  /* --- patch_linear F32→BF16 --- */
  auto *tp = h3_gpu_tensor_new_bf16(gpu, rows * out);
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_patch_linear_bf16(gpu, tp, tx, tw, tb, rows, in, out), "patch");
  OP(h3_gpu_submit(gpu), "submit patch");
  std::vector<uint16_t> pb(rows * out);
  OP(h3_gpu_tensor_read_bf16(tp, pb.data(), pb.size()), "read patch");
  float pmax = 0.f;
  for (size_t i = 0; i < pb.size(); i++)
    pmax = std::fmax(pmax, std::fabs(b2f(pb[i]) - yref[i]));
  if (pmax > 0.05f) {
    std::fprintf(stderr, "FAIL patch max_abs=%g\n", pmax);
    return 1;
  }
  std::printf("ok patch_linear_bf16 max_abs=%g\n", pmax);

  /* --- patch map scatter --- */
  const uint32_t out_rows = 8;
  std::vector<uint32_t> map = {1, 3, 5, 7};
  auto *tmap = h3_gpu_tensor_from_u32(gpu, map.data(), map.size());
  auto *tscatter = h3_gpu_tensor_new_bf16(gpu, out_rows * out);
  /* zero destination */
  std::vector<uint16_t> z(out_rows * out, 0);
  /* no write API for full tensor zero — use from_bf16 recreate */
  h3_gpu_tensor_free(tscatter);
  tscatter = h3_gpu_tensor_from_bf16(gpu, z.data(), z.size());
  OP(h3_gpu_begin(gpu), "begin3");
  OP(h3_gpu_patch_linear_bf16_map(gpu, tscatter, tx, tw, tb, tmap, out_rows,
                                 rows, in, out),
     "patch map");
  OP(h3_gpu_submit(gpu), "submit map");
  std::vector<uint16_t> sm(out_rows * out);
  OP(h3_gpu_tensor_read_bf16(tscatter, sm.data(), sm.size()), "read map");
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t o = 0; o < out; o++) {
      float got = b2f(sm[map[r] * out + o]);
      if (std::fabs(got - yref[r * out + o]) > 0.05f) {
        std::fprintf(stderr, "FAIL map r=%u o=%u got=%g ref=%g\n", r, o, got,
                     yref[r * out + o]);
        return 1;
      }
    }
  }
  std::printf("ok patch_linear_bf16_map\n");

  /* --- adaln_linear --- */
  std::vector<uint16_t> hin(rows * width), nw(width), mod(slots * width),
      lw(out * width), lb(out);
  for (size_t i = 0; i < hin.size(); i++)
    hin[i] = f2b(0.05f * (float)((i % 9) + 1));
  for (uint32_t i = 0; i < width; i++) nw[i] = f2b(1.f);
  for (size_t i = 0; i < mod.size(); i++) mod[i] = f2b(0.01f * (float)(i % 5));
  for (size_t i = 0; i < lw.size(); i++)
    lw[i] = f2b(0.02f * (float)((i % 6) + 1));
  for (uint32_t i = 0; i < out; i++) lb[i] = f2b(0.f);
  std::vector<uint32_t> amap(rows);
  for (uint32_t r = 0; r < rows; r++) amap[r] = 0; /* single mod row */
  auto *thin = h3_gpu_tensor_from_bf16(gpu, hin.data(), hin.size());
  auto *tnw = h3_gpu_tensor_from_bf16(gpu, nw.data(), nw.size());
  auto *tmod = h3_gpu_tensor_from_bf16(gpu, mod.data(), mod.size());
  auto *tlw = h3_gpu_tensor_from_bf16(gpu, lw.data(), lw.size());
  auto *tlb = h3_gpu_tensor_from_bf16(gpu, lb.data(), lb.size());
  auto *tamap = h3_gpu_tensor_from_u32(gpu, amap.data(), amap.size());
  auto *tinv = h3_gpu_tensor_new_f32(gpu, rows);
  auto *tout = h3_gpu_tensor_new_bf16(gpu, rows * out);
  OP(h3_gpu_begin(gpu), "begin4");
  OP(h3_gpu_adaln_linear_bf16(gpu, tout, tinv, thin, 0, tnw, tmod, tamap, tlw,
                             tlb, rows, width, out, slots, 0, 1, 1e-5f),
     "adaln_linear");
  OP(h3_gpu_submit(gpu), "submit adaln_linear");
  std::vector<uint16_t> aout(rows * out);
  std::vector<float> inv(rows);
  OP(h3_gpu_tensor_read_bf16(tout, aout.data(), aout.size()), "read adaln out");
  OP(h3_gpu_tensor_read_f32(tinv, inv.data(), inv.size()), "read inv");
  float asum = 0.f;
  for (auto v : aout) asum += std::fabs(b2f(v));
  for (auto v : inv)
    if (!(v > 0.f) || !std::isfinite(v)) return die("bad inverse rms");
  if (!(asum > 0.f) || !std::isfinite(asum)) return die("bad adaln_linear out");
  std::printf("ok adaln_linear_bf16  |out|=%.4f  inv0=%.4f\n", asum, inv[0]);

  h3_gpu_free(gpu);
  std::printf("dit pack/final smoke passed\n");
  return 0;
}
