/* P2 elem + token_pool + multi-chunk pinned stream smoke. */
#include "h3_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
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

  const uint32_t n = 64, rows = 4, width = 16;
  std::vector<float> x(n), y(n);
  for (uint32_t i = 0; i < n; i++) x[i] = -2.f + 0.1f * (float)i;
  auto *tx = h3_gpu_tensor_from_f32(gpu, x.data(), x.size());
  auto *ty = h3_gpu_tensor_new_f32(gpu, n);
  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_silu_f32(gpu, ty, tx, n), "silu");
  OP(h3_gpu_clip_f32(gpu, ty, ty, n, -1.f, 1.f), "clip");
  OP(h3_gpu_submit(gpu), "submit1");
  OP(h3_gpu_tensor_read_f32(ty, y.data(), y.size()), "read");
  for (uint32_t i = 0; i < n; i++) {
    float silu = x[i] / (1.f + std::exp(-x[i]));
    float clip = std::fmin(1.f, std::fmax(-1.f, silu));
    if (std::fabs(y[i] - clip) > 1e-5f) return die("silu/clip mismatch");
  }
  std::printf("ok silu_f32+clip\n");

  std::vector<float> in(rows * width), w(width, 1.f), b(width, 0.f),
      out(rows * width);
  for (size_t i = 0; i < in.size(); i++) in[i] = 0.1f * (float)(i % 11);
  auto *tin = h3_gpu_tensor_from_f32(gpu, in.data(), in.size());
  auto *tw = h3_gpu_tensor_from_f32(gpu, w.data(), w.size());
  auto *tb = h3_gpu_tensor_from_f32(gpu, b.data(), b.size());
  auto *tout = h3_gpu_tensor_new_f32(gpu, rows * width);
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_layer_norm_f32(gpu, tout, tin, tw, tb, rows, width, 1e-5f), "ln");
  OP(h3_gpu_rms_norm_f32(gpu, tout, tin, tw, rows, width, 1e-5f), "rms");
  OP(h3_gpu_submit(gpu), "submit2");
  std::printf("ok layer_norm_f32+rms_norm_f32\n");

  /* token pool: 4 input rows → 2 pooled */
  const uint32_t in_rows = 4, pooled = 2, dim = 8;
  std::vector<uint16_t> src(in_rows * dim);
  for (size_t i = 0; i < src.size(); i++) src[i] = f2b(0.25f * (float)(i + 1));
  std::vector<uint32_t> pairs = {0, 1, 2, 3};
  std::vector<uint32_t> bidx = {0, 1};
  auto *tsrc = h3_gpu_tensor_from_bf16(gpu, src.data(), src.size());
  auto *tpairs = h3_gpu_tensor_from_u32(gpu, pairs.data(), pairs.size());
  auto *tbidx = h3_gpu_tensor_from_u32(gpu, bidx.data(), bidx.size());
  auto *tpool = h3_gpu_tensor_new_bf16(gpu, pooled * dim);
  auto *torig = h3_gpu_tensor_new_bf16(gpu, in_rows * dim);
  auto *tbase = h3_gpu_tensor_new_bf16(gpu, pooled * dim);
  OP(h3_gpu_begin(gpu), "begin3");
  OP(h3_gpu_token_pool_bf16(gpu, tpool, tsrc, 0, torig, 0, tbase, 0, tbidx,
                           tpairs, in_rows, pooled, pooled, dim),
     "token_pool");
  OP(h3_gpu_submit(gpu), "submit3");
  std::vector<uint16_t> pool(pooled * dim);
  OP(h3_gpu_tensor_read_bf16(tpool, pool.data(), pool.size()), "read pool");
  float avg0 = 0.5f * (b2f(src[0]) + b2f(src[dim]));
  if (std::fabs(b2f(pool[0]) - avg0) > 0.02f) return die("token pool avg");
  std::printf("ok token_pool_bf16\n");

  /* Multi-chunk pinned stream: 3 MiB+ of BF16 (> half of 4 MiB pin to exercise
   * at least one full chunk; use 2.1M elems ≈ 4.2 MiB for 2 chunks). */
  const size_t elems = (size_t)(2.1e6);
  std::vector<uint16_t> big(elems);
  for (size_t i = 0; i < elems; i++) big[i] = f2b((float)(i % 17));
  char path[] = "/tmp/h3_cuda_stream_XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) return die("mkstemp");
  if (write(fd, big.data(), big.size() * 2) != (ssize_t)(big.size() * 2)) {
    close(fd);
    unlink(path);
    return die("write fixture");
  }
  close(fd);
  auto *tdev = h3_gpu_tensor_new_bf16(gpu, elems);
  char ferr[256]{};
  if (!h3_gpu_tensor_stream_file_bf16(tdev, path, 0, elems, ferr, sizeof(ferr))) {
    std::fprintf(stderr, "FAIL stream: %s\n", ferr);
    unlink(path);
    return 1;
  }
  unlink(path);
  std::vector<uint16_t> got(256);
  OP(h3_gpu_tensor_read_bf16(tdev, got.data(), got.size()), "read stream head");
  for (size_t i = 0; i < got.size(); i++)
    if (got[i] != big[i]) return die("stream mismatch");
  std::printf("ok pinned multi-chunk stream (%zu elems)\n", elems);

  h3_gpu_free(gpu);
  std::printf("p2 elem smoke passed\n");
  return 0;
}
