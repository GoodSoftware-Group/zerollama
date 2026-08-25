/* P1 smoke: embedding, head RMS, text RoPE, GQA causal. */
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

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);

  const uint32_t seq = 8, qh = 4, kvh = 2, dim = 32, vocab = 64, width = 32;
  /* embedding */
  std::vector<uint16_t> w(vocab * width);
  for (size_t i = 0; i < w.size(); i++) w[i] = f2b(0.01f * (float)(i % 17));
  std::vector<uint32_t> ids = {1, 2, 3, 4};
  auto *tw = h3_gpu_tensor_from_bf16(gpu, w.data(), w.size());
  auto *tids = h3_gpu_tensor_from_u32(gpu, ids.data(), ids.size());
  auto *temb = h3_gpu_tensor_new_bf16(gpu, ids.size() * width);

  /* GQA tensors */
  size_t qn = (size_t)seq * qh * dim;
  size_t kn = (size_t)seq * kvh * dim;
  std::vector<uint16_t> q(qn), k(kn), v(kn);
  for (size_t i = 0; i < qn; i++) q[i] = f2b(0.02f * (float)((i % 13) + 1));
  for (size_t i = 0; i < kn; i++) {
    k[i] = f2b(0.02f * (float)((i % 11) + 1));
    v[i] = f2b(0.02f * (float)((i % 7) + 1));
  }
  auto *tq = h3_gpu_tensor_from_bf16(gpu, q.data(), q.size());
  auto *tk = h3_gpu_tensor_from_bf16(gpu, k.data(), k.size());
  auto *tv = h3_gpu_tensor_from_bf16(gpu, v.data(), v.size());
  auto *to = h3_gpu_tensor_new_bf16(gpu, qn);

  std::vector<uint16_t> wn(dim, f2b(1.f));
  auto *twn = h3_gpu_tensor_from_bf16(gpu, wn.data(), wn.size());

  uint32_t half = dim / 2;
  std::vector<float> cos((size_t)seq * half), sinv((size_t)seq * half);
  for (uint32_t r = 0; r < seq; r++)
    for (uint32_t d = 0; d < half; d++) {
      float a = 0.05f * (float)(r * half + d);
      cos[r * half + d] = std::cos(a);
      sinv[r * half + d] = std::sin(a);
    }
  auto *tcos = h3_gpu_tensor_from_f32(gpu, cos.data(), cos.size());
  auto *tsin = h3_gpu_tensor_from_f32(gpu, sinv.data(), sinv.size());

#define OP(c, l)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "FAIL %s: %s\n", l, h3_gpu_error(gpu));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_embedding_bf16(gpu, temb, tw, tids, (uint32_t)ids.size(), vocab,
                           width),
     "embedding");
  OP(h3_gpu_head_rms_norm_bf16(gpu, tq, twn, seq, qh, dim, 1e-5f), "head rms");
  OP(h3_gpu_rope_text_bf16(gpu, tq, tk, tcos, tsin, seq, qh, kvh, dim),
     "rope text");
  OP(h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, seq, qh, kvh, dim,
                            1.f / std::sqrt((float)dim)),
     "gqa causal");
  OP(h3_gpu_submit(gpu), "submit");

  std::vector<uint16_t> out(qn);
  if (!h3_gpu_tensor_read_bf16(to, out.data(), out.size())) return die("read");
  float sum = 0.f;
  for (auto b : out) sum += std::fabs(b2f(b));
  if (!(sum > 0.f) || !std::isfinite(sum)) return die("gqa output bad");

  /* Causal: row 0 can only attend to itself — finite check already done. */
  std::printf("ok P1 text/GQA  seq=%u qh=%u kvh=%u dim=%u  |out|=%.4f\n", seq,
              qh, kvh, dim, sum);
  h3_gpu_free(gpu);
  return 0;
}
