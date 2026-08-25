/* Microbench: BF16 cublas linear vs int8 (cublas path) on mid-size GEMM. */
#include "h3_gpu.h"

#include <chrono>
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

  const uint32_t rows = 128, in_dim = 512, out_dim = 256;
  const int warmup = 5, iters = 20;

  std::vector<uint16_t> xb(rows * in_dim), wb(out_dim * in_dim);
  for (size_t i = 0; i < xb.size(); i++)
    xb[i] = f2b(0.01f * (float)((int)(i % 17) - 8));
  for (size_t i = 0; i < wb.size(); i++)
    wb[i] = f2b(0.01f * (float)((int)(i % 13) - 6));

  auto *tx = h3_gpu_tensor_from_bf16(gpu, xb.data(), xb.size());
  auto *tw = h3_gpu_tensor_from_bf16(gpu, wb.data(), wb.size());
  auto *ty_bf = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);
  auto *tw_i8 = h3_gpu_tensor_new_i8(gpu, wb.size());
  auto *tw_s = h3_gpu_tensor_new_f32(gpu, out_dim);
  auto *tqi = h3_gpu_tensor_new_i8(gpu, rows * in_dim);
  auto *tis = h3_gpu_tensor_new_f32(gpu, rows);
  auto *ty_i8 = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);

  OP(h3_gpu_begin(gpu), "begin qw");
  OP(h3_gpu_quantize_weight_int8(gpu, tw_i8, tw_s, tw, out_dim, in_dim), "qw");
  OP(h3_gpu_submit(gpu), "submit qw");

  auto run_bf16 = [&]() {
    OP(h3_gpu_begin(gpu), "bf begin");
    OP(h3_gpu_linear_bf16(gpu, ty_bf, tx, tw, nullptr, rows, in_dim, out_dim),
       "bf linear");
    OP(h3_gpu_submit(gpu), "bf submit");
    return 0;
  };
  auto run_i8 = [&]() {
    OP(h3_gpu_begin(gpu), "i8 begin");
    OP(h3_gpu_linear_int8_bf16(gpu, ty_i8, tqi, tis, tx, tw_i8, tw_s, rows,
                               in_dim, out_dim, 0),
       "i8 linear");
    OP(h3_gpu_submit(gpu), "i8 submit");
    return 0;
  };

  for (int i = 0; i < warmup; i++) {
    if (run_bf16()) return 1;
    if (run_i8()) return 1;
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++)
    if (run_bf16()) return 1;
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++)
    if (run_i8()) return 1;
  auto t2 = std::chrono::steady_clock::now();

  double bf_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double i8_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;

  /* FLOPs ~ 2*rows*in*out */
  double flops = 2.0 * rows * in_dim * out_dim;
  std::printf("ok bench linear  shape=%ux%ux%u  bf16=%.3f ms (%.1f GFLOP/s)  "
              "int8=%.3f ms (%.1f GFLOP/s)  speedup=%.2fx\n",
              rows, in_dim, out_dim, bf_ms, flops / (bf_ms * 1e6), i8_ms,
              flops / (i8_ms * 1e6), bf_ms / i8_ms);

  h3_gpu_stats st{};
  h3_gpu_get_stats(gpu, &st);
  std::printf("stats direct=%llu linear=%llu\n",
              (unsigned long long)st.direct_dispatches,
              (unsigned long long)st.mps_linear_dispatches);
  h3_gpu_free(gpu);
  return 0;
}
