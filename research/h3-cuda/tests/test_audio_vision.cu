/* AudioVAE snake/SDPA + vision QKV RoPE smoke. */
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

  const uint32_t B = 1, L = 8, C = 4;
  std::vector<float> x(B * L * C), a(C), y(B * L * C);
  for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (float)(i + 1);
  for (uint32_t i = 0; i < C; i++) a[i] = 0.5f + 0.1f * (float)i;
  auto *tx = h3_gpu_tensor_from_f32(gpu, x.data(), x.size());
  auto *ta = h3_gpu_tensor_from_f32(gpu, a.data(), a.size());
  auto *ty = h3_gpu_tensor_new_f32(gpu, x.size());
  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_snake1d_f32(gpu, ty, tx, ta, B, L, C), "snake");
  OP(h3_gpu_submit(gpu), "submit1");
  OP(h3_gpu_tensor_read_f32(ty, y.data(), y.size()), "read snake");
  float wave = std::sin(a[0] * x[0]);
  float ref0 = x[0] + wave * wave / (a[0] + 1e-9f);
  if (std::fabs(y[0] - ref0) > 1e-5f) return die("snake mismatch");
  std::printf("ok snake1d_f32\n");

  /* alias-free: just ensure it runs with 12-tap filters */
  std::vector<float> al(C, 0.f), be(C, 0.f), up(12, 1.f / 12.f),
      down(12, 1.f / 12.f);
  auto *tal = h3_gpu_tensor_from_f32(gpu, al.data(), al.size());
  auto *tbe = h3_gpu_tensor_from_f32(gpu, be.data(), be.size());
  auto *tup = h3_gpu_tensor_from_f32(gpu, up.data(), up.size());
  auto *tdn = h3_gpu_tensor_from_f32(gpu, down.data(), down.size());
  auto *taf = h3_gpu_tensor_new_f32(gpu, x.size());
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_alias_free_snake_f32(gpu, taf, tx, tal, tbe, tup, tdn, B, L, C),
     "alias");
  OP(h3_gpu_submit(gpu), "submit2");
  std::printf("ok alias_free_snake_f32\n");

  /* QKV split + causal SDPA */
  const uint32_t heads = 2, dim = 8, seq = 4;
  const uint32_t width = heads * dim;
  std::vector<float> qkv(B * seq * width * 3), qb(width, 0.f), kb(width, 0.f),
      vb(width, 0.f);
  for (size_t i = 0; i < qkv.size(); i++) qkv[i] = 0.01f * (float)((i % 9) + 1);
  auto *tqkv = h3_gpu_tensor_from_f32(gpu, qkv.data(), qkv.size());
  auto *tqb = h3_gpu_tensor_from_f32(gpu, qb.data(), qb.size());
  auto *tkb = h3_gpu_tensor_from_f32(gpu, kb.data(), kb.size());
  auto *tvb = h3_gpu_tensor_from_f32(gpu, vb.data(), vb.size());
  auto *tq = h3_gpu_tensor_new_f32(gpu, B * seq * width);
  auto *tk = h3_gpu_tensor_new_f32(gpu, B * seq * width);
  auto *tv = h3_gpu_tensor_new_f32(gpu, B * seq * width);
  auto *to = h3_gpu_tensor_new_f32(gpu, B * seq * width);
  OP(h3_gpu_begin(gpu), "begin3");
  OP(h3_gpu_audio_qkv_split_f32(gpu, tq, tk, tv, tqkv, tqb, tkb, tvb, B, seq,
                               heads, dim),
     "qkv");
  OP(h3_gpu_sdpa_causal_f32(gpu, to, tq, tk, tv, B, seq, heads, dim,
                            1.f / std::sqrt((float)dim)),
     "sdpa");
  OP(h3_gpu_submit(gpu), "submit3");
  std::vector<float> out(B * seq * width);
  OP(h3_gpu_tensor_read_f32(to, out.data(), out.size()), "read sdpa");
  float sum = 0.f;
  for (float v : out) sum += std::fabs(v);
  if (!(sum > 0.f) || !std::isfinite(sum)) return die("sdpa bad");
  std::printf("ok audio_qkv+sdpa_causal  |out|=%.4f\n", sum);

  auto *tpool = h3_gpu_tensor_new_f32(gpu, B * seq * (dim / 2));
  OP(h3_gpu_begin(gpu), "begin4");
  OP(h3_gpu_audio_attention_pool_f32(gpu, tpool, to, B, seq, heads, dim,
                                    dim / 2),
     "pool");
  OP(h3_gpu_submit(gpu), "submit4");
  std::printf("ok audio_attention_pool\n");

  /* vision qkv rope */
  const uint32_t vseq = 4, vh = 2, vd = 8, half = 4;
  std::vector<uint16_t> vqkv(vseq * vh * vd * 3);
  for (size_t i = 0; i < vqkv.size(); i++)
    vqkv[i] = f2b(0.05f * (float)((i % 11) + 1));
  std::vector<uint16_t> cos(vseq * half), sinv(vseq * half);
  for (uint32_t r = 0; r < vseq; r++)
    for (uint32_t d = 0; d < half; d++) {
      float ang = 0.1f * (float)(r * half + d);
      cos[r * half + d] = f2b(std::cos(ang));
      sinv[r * half + d] = f2b(std::sin(ang));
    }
  auto *tvq = h3_gpu_tensor_from_bf16(gpu, vqkv.data(), vqkv.size());
  auto *tcos = h3_gpu_tensor_from_bf16(gpu, cos.data(), cos.size());
  auto *tsin = h3_gpu_tensor_from_bf16(gpu, sinv.data(), sinv.size());
  auto *tq2 = h3_gpu_tensor_new_bf16(gpu, vseq * vh * vd);
  auto *tk2 = h3_gpu_tensor_new_bf16(gpu, vseq * vh * vd);
  auto *tv2 = h3_gpu_tensor_new_bf16(gpu, vseq * vh * vd);
  OP(h3_gpu_begin(gpu), "begin5");
  OP(h3_gpu_vision_qkv_rope_bf16(gpu, tq2, tk2, tv2, tvq, tcos, tsin, vseq, vh,
                                vd, half),
     "vision rope");
  OP(h3_gpu_submit(gpu), "submit5");
  std::vector<uint16_t> qout(vseq * vh * vd);
  OP(h3_gpu_tensor_read_bf16(tq2, qout.data(), qout.size()), "read vision");
  if (!(std::fabs(b2f(qout[0])) > 0.f)) return die("vision q zero");
  std::printf("ok vision_qkv_rope_bf16\n");

  h3_gpu_free(gpu);
  std::printf("audio/vision smoke passed\n");
  return 0;
}
