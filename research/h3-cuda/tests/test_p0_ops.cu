/* P0 smoke: linear_bf16, swiglu, adaln, gate, sdpa, file-stream. */
#include "h3_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static int fail(const char *msg) {
  std::fprintf(stderr, "FAIL: %s\n", msg);
  return 1;
}

static int test_linear() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return fail(err);

  const uint32_t rows = 4, in = 8, out = 4;
  std::vector<uint16_t> x(rows * in), w(out * in), yref(rows * out);
  for (uint32_t i = 0; i < rows * in; i++)
    x[i] = f32_to_bf16((float)((i % 7) + 1) * 0.1f);
  for (uint32_t i = 0; i < out * in; i++)
    w[i] = f32_to_bf16((float)((i % 5) + 1) * 0.05f);

  /* CPU: Y = X @ W.T */
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t o = 0; o < out; o++) {
      float acc = 0.f;
      for (uint32_t k = 0; k < in; k++)
        acc += bf16_to_f32(x[r * in + k]) * bf16_to_f32(w[o * in + k]);
      yref[r * out + o] = f32_to_bf16(acc);
    }
  }

  h3_gpu_tensor *tx = h3_gpu_tensor_from_bf16(gpu, x.data(), x.size());
  h3_gpu_tensor *tw = h3_gpu_tensor_from_bf16(gpu, w.data(), w.size());
  h3_gpu_tensor *ty = h3_gpu_tensor_new_bf16(gpu, rows * out);
  if (!tx || !tw || !ty) return fail(h3_gpu_error(gpu));
  if (!h3_gpu_begin(gpu) || !h3_gpu_linear_bf16(gpu, ty, tx, tw, nullptr, rows, in, out) ||
      !h3_gpu_submit(gpu))
    return fail(h3_gpu_error(gpu));

  std::vector<uint16_t> y(rows * out);
  if (!h3_gpu_tensor_read_bf16(ty, y.data(), y.size())) return fail("read y");

  float max_abs = 0.f;
  for (size_t i = 0; i < y.size(); i++)
    max_abs = std::fmax(max_abs, std::fabs(bf16_to_f32(y[i]) - bf16_to_f32(yref[i])));
  h3_gpu_tensor_free(tx);
  h3_gpu_tensor_free(tw);
  h3_gpu_tensor_free(ty);
  h3_gpu_free(gpu);
  if (max_abs > 2e-2f) {
    std::fprintf(stderr, "linear max_abs=%g\n", max_abs);
    return fail("linear_bf16 vs CPU");
  }
  std::printf("ok linear_bf16 max_abs=%g\n", max_abs);
  return 0;
}

static int test_adaln_gate_swiglu() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return fail(err);
  const uint32_t rows = 2, width = 64, slots = 3;
  std::vector<uint16_t> x(rows * width), w(width), mod(rows * slots * width);
  std::vector<uint32_t> map = {0, 1};
  for (size_t i = 0; i < x.size(); i++) x[i] = f32_to_bf16(0.5f);
  for (size_t i = 0; i < w.size(); i++) w[i] = f32_to_bf16(1.f);
  for (size_t i = 0; i < mod.size(); i++) mod[i] = f32_to_bf16(0.1f);

  auto *tx = h3_gpu_tensor_from_bf16(gpu, x.data(), x.size());
  auto *tw = h3_gpu_tensor_from_bf16(gpu, w.data(), w.size());
  auto *tm = h3_gpu_tensor_from_bf16(gpu, mod.data(), mod.size());
  auto *tmap = h3_gpu_tensor_from_u32(gpu, map.data(), map.size());
  auto *tout = h3_gpu_tensor_new_bf16(gpu, rows * width);
  auto *tbranch = h3_gpu_tensor_from_bf16(gpu, x.data(), x.size());
  auto *tgated = h3_gpu_tensor_new_bf16(gpu, rows * width);

  std::vector<uint16_t> fused(rows * width * 2);
  for (size_t i = 0; i < fused.size(); i++) fused[i] = f32_to_bf16(0.25f);
  auto *tf = h3_gpu_tensor_from_bf16(gpu, fused.data(), fused.size());
  auto *tact = h3_gpu_tensor_new_bf16(gpu, rows * width);

  if (!h3_gpu_begin(gpu) ||
      !h3_gpu_adaln_bf16(gpu, tout, tx, tw, tm, tmap, rows, width, slots, 0, 1,
                         1e-5f) ||
      !h3_gpu_gate_bf16(gpu, tgated, tx, tbranch, tm, tmap, rows, width, slots,
                        2) ||
      !h3_gpu_swiglu_bf16(gpu, tact, tf, rows, width) || !h3_gpu_submit(gpu))
    return fail(h3_gpu_error(gpu));

  std::vector<uint16_t> out(rows * width);
  h3_gpu_tensor_read_bf16(tout, out.data(), out.size());
  h3_gpu_tensor_read_bf16(tact, out.data(), out.size());
  std::printf("ok adaln/gate/swiglu\n");
  h3_gpu_tensor_free(tx);
  h3_gpu_tensor_free(tw);
  h3_gpu_tensor_free(tm);
  h3_gpu_tensor_free(tmap);
  h3_gpu_tensor_free(tout);
  h3_gpu_tensor_free(tbranch);
  h3_gpu_tensor_free(tgated);
  h3_gpu_tensor_free(tf);
  h3_gpu_tensor_free(tact);
  h3_gpu_free(gpu);
  return 0;
}

static int test_sdpa() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return fail(err);
  const uint32_t seq = 8, heads = 2, dim = 16;
  size_t n = (size_t)seq * heads * dim;
  std::vector<uint16_t> q(n), k(n), v(n);
  for (size_t i = 0; i < n; i++) {
    q[i] = f32_to_bf16(0.01f * (float)(i % 17));
    k[i] = f32_to_bf16(0.01f * (float)((i + 3) % 17));
    v[i] = f32_to_bf16(0.01f * (float)((i + 5) % 17));
  }
  auto *tq = h3_gpu_tensor_from_bf16(gpu, q.data(), n);
  auto *tk = h3_gpu_tensor_from_bf16(gpu, k.data(), n);
  auto *tv = h3_gpu_tensor_from_bf16(gpu, v.data(), n);
  auto *to = h3_gpu_tensor_new_bf16(gpu, n);
  float scale = 1.f / std::sqrt((float)dim);
  if (!h3_gpu_begin(gpu) ||
      !h3_gpu_sdpa_bf16(gpu, to, tq, tk, tv, seq, heads, dim, scale) ||
      !h3_gpu_submit(gpu))
    return fail(h3_gpu_error(gpu));
  std::vector<uint16_t> out(n);
  if (!h3_gpu_tensor_read_bf16(to, out.data(), n)) return fail("sdpa read");
  std::printf("ok sdpa_bf16\n");
  h3_gpu_tensor_free(tq);
  h3_gpu_tensor_free(tk);
  h3_gpu_tensor_free(tv);
  h3_gpu_tensor_free(to);
  h3_gpu_free(gpu);
  return 0;
}

static int test_file_stream() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return fail(err);
  const size_t n = 1024;
  std::vector<uint16_t> host(n);
  for (size_t i = 0; i < n; i++) host[i] = f32_to_bf16((float)i * 0.001f);
  char path[] = "/tmp/h3_cuda_stream_XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) return fail("mkstemp");
  if (write(fd, host.data(), n * 2) != (ssize_t)(n * 2)) return fail("write");
  close(fd);

  auto *t = h3_gpu_tensor_new_bf16(gpu, n);
  if (!h3_gpu_tensor_stream_file_bf16(t, path, 0, n, err, sizeof(err))) {
    unlink(path);
    return fail(err);
  }
  std::vector<uint16_t> back(n);
  h3_gpu_tensor_read_bf16(t, back.data(), n);
  unlink(path);
  for (size_t i = 0; i < n; i++) {
    if (back[i] != host[i]) {
      h3_gpu_tensor_free(t);
      h3_gpu_free(gpu);
      return fail("file-stream mismatch");
    }
  }
  std::printf("ok file-stream bf16 (%zu elems)\n", n);
  h3_gpu_tensor_free(t);
  h3_gpu_free(gpu);
  return 0;
}

int main() {
  if (test_linear()) return 1;
  if (test_adaln_gate_swiglu()) return 1;
  if (test_sdpa()) return 1;
  if (test_file_stream()) return 1;
  std::printf("P0 smoke passed\n");
  return 0;
}
