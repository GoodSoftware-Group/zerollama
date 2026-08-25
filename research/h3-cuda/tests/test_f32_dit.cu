/* F32 DiT helpers + ungrouped qkv_rope_bf16 smoke. */
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

static float rms_inv(const float *x, uint32_t n, float eps) {
  float s = 0.f;
  for (uint32_t i = 0; i < n; i++) s = std::fmaf(x[i], x[i], s);
  return 1.f / std::sqrt(s / (float)n + eps);
}

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);

  const uint32_t rows = 4, width = 8, slots = 6;
  const float eps = 1e-5f;
  std::vector<float> xin(rows * width), nw(width, 1.f), mod(2 * slots * width);
  std::vector<uint32_t> map(rows);
  for (uint32_t i = 0; i < rows * width; i++) xin[i] = 0.1f * (float)(i + 1);
  for (uint32_t i = 0; i < rows; i++) map[i] = i % 2;
  for (size_t i = 0; i < mod.size(); i++) mod[i] = 0.01f * (float)((i % 11) + 1);

  auto *tin = h3_gpu_tensor_from_f32(gpu, xin.data(), xin.size());
  auto *tnw = h3_gpu_tensor_from_f32(gpu, nw.data(), nw.size());
  auto *tmod = h3_gpu_tensor_from_f32(gpu, mod.data(), mod.size());
  auto *tmap = h3_gpu_tensor_from_u32(gpu, map.data(), map.size());
  auto *tout = h3_gpu_tensor_new_f32(gpu, rows * width);
  if (!tin || !tnw || !tmod || !tmap || !tout) return die("alloc adaln");

  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_adaln_f32(gpu, tout, tin, tnw, tmod, tmap, rows, width, slots, 0, 1,
                      eps),
     "adaln");
  OP(h3_gpu_submit(gpu), "submit adaln");
  std::vector<float> y(rows * width);
  OP(h3_gpu_tensor_read_f32(tout, y.data(), y.size()), "read adaln");
  for (uint32_t r = 0; r < rows; r++) {
    const float *xr = xin.data() + r * width;
    float inv = rms_inv(xr, width, eps);
    size_t base = (size_t)map[r] * slots * width;
    for (uint32_t c = 0; c < width; c++) {
      float n = xr[c] * inv * nw[c];
      float shift = mod[base + 0 * width + c];
      float scale = mod[base + 1 * width + c];
      float ref = n * (1.f + scale) + shift;
      if (std::fabs(y[r * width + c] - ref) > 1e-5f) return die("adaln mismatch");
    }
  }
  std::printf("ok adaln_f32\n");

  std::vector<float> res(xin), br(xin.size());
  for (size_t i = 0; i < br.size(); i++) br[i] = 0.05f * (float)(i + 3);
  auto *tres = h3_gpu_tensor_from_f32(gpu, res.data(), res.size());
  auto *tbr = h3_gpu_tensor_from_f32(gpu, br.data(), br.size());
  auto *tg = h3_gpu_tensor_new_f32(gpu, rows * width);
  OP(h3_gpu_begin(gpu), "begin gate");
  OP(h3_gpu_gate_f32(gpu, tg, tres, tbr, tmod, tmap, rows, width, slots, 2),
     "gate");
  OP(h3_gpu_submit(gpu), "submit gate");
  std::vector<float> gy(rows * width);
  OP(h3_gpu_tensor_read_f32(tg, gy.data(), gy.size()), "read gate");
  for (uint32_t r = 0; r < rows; r++) {
    size_t base = (size_t)map[r] * slots * width;
    for (uint32_t c = 0; c < width; c++) {
      float gate = mod[base + 2 * width + c];
      float ref = res[r * width + c] + br[r * width + c] * gate;
      if (std::fabs(gy[r * width + c] - ref) > 1e-5f) return die("gate mismatch");
    }
  }
  std::printf("ok gate_f32\n");

  /* QKV RoPE F32: seq=2 heads=2 dim=4 rope_half=2 */
  const uint32_t seq = 2, heads = 2, dim = 4, rh = 2;
  size_t inner = (size_t)heads * dim;
  std::vector<float> qkv(seq * inner * 3), qn(dim, 1.f), kn(dim, 1.f);
  std::vector<float> cos(seq * rh), sin(seq * rh);
  for (size_t i = 0; i < qkv.size(); i++) qkv[i] = 0.1f * (float)((i % 13) + 1);
  for (uint32_t r = 0; r < seq; r++)
    for (uint32_t d = 0; d < rh; d++) {
      cos[r * rh + d] = std::cos(0.1f * (float)(r + d + 1));
      sin[r * rh + d] = std::sin(0.1f * (float)(r + d + 1));
    }
  auto *tqkv = h3_gpu_tensor_from_f32(gpu, qkv.data(), qkv.size());
  auto *tqn = h3_gpu_tensor_from_f32(gpu, qn.data(), qn.size());
  auto *tkn = h3_gpu_tensor_from_f32(gpu, kn.data(), kn.size());
  auto *tcos = h3_gpu_tensor_from_f32(gpu, cos.data(), cos.size());
  auto *tsin = h3_gpu_tensor_from_f32(gpu, sin.data(), sin.size());
  auto *tq = h3_gpu_tensor_new_f32(gpu, seq * inner);
  auto *tk = h3_gpu_tensor_new_f32(gpu, seq * inner);
  auto *tv = h3_gpu_tensor_new_f32(gpu, seq * inner);
  OP(h3_gpu_begin(gpu), "begin qkv");
  OP(h3_gpu_qkv_rope_f32(gpu, tq, tk, tv, tqkv, tqn, tkn, tcos, tsin, seq, heads,
                         dim, rh, eps),
     "qkv_rope");
  OP(h3_gpu_submit(gpu), "submit qkv");
  std::vector<float> q(seq * inner), k(seq * inner), v(seq * inner);
  OP(h3_gpu_tensor_read_f32(tq, q.data(), q.size()), "read q");
  OP(h3_gpu_tensor_read_f32(tk, k.data(), k.size()), "read k");
  OP(h3_gpu_tensor_read_f32(tv, v.data(), v.size()), "read v");
  for (uint32_t r = 0; r < seq; r++) {
    size_t row_base = (size_t)r * inner * 3;
    for (uint32_t h = 0; h < heads; h++) {
      size_t qb = row_base + h * dim;
      size_t kb = qb + inner;
      size_t vb = qb + inner * 2;
      float qinv = rms_inv(qkv.data() + qb, dim, eps);
      float kinv = rms_inv(qkv.data() + kb, dim, eps);
      float qo[4], ko[4];
      for (uint32_t d = 0; d < dim; d++) {
        qo[d] = qkv[qb + d] * qinv * qn[d];
        ko[d] = qkv[kb + d] * kinv * kn[d];
        float vref = qkv[vb + d];
        size_t oi = ((size_t)r * heads + h) * dim + d;
        if (std::fabs(v[oi] - vref) > 1e-5f) return die("qkv V mismatch");
      }
      for (uint32_t d = 0; d < rh; d++) {
        float c = cos[r * rh + d], s = sin[r * rh + d];
        float q0 = qo[d], q1 = qo[d + rh];
        float k0 = ko[d], k1 = ko[d + rh];
        qo[d] = q0 * c - q1 * s;
        qo[d + rh] = q0 * s + q1 * c;
        ko[d] = k0 * c - k1 * s;
        ko[d + rh] = k0 * s + k1 * c;
      }
      for (uint32_t d = 0; d < dim; d++) {
        size_t oi = ((size_t)r * heads + h) * dim + d;
        if (std::fabs(q[oi] - qo[d]) > 1e-5f) return die("qkv Q mismatch");
        if (std::fabs(k[oi] - ko[d]) > 1e-5f) return die("qkv K mismatch");
      }
    }
  }
  std::printf("ok qkv_rope_f32\n");

  /* Video QKV: interleaved */
  std::vector<float> vqkv(seq * heads * dim * 3);
  for (size_t i = 0; i < vqkv.size(); i++)
    vqkv[i] = 0.07f * (float)((i % 17) + 1);
  auto *tvqkv = h3_gpu_tensor_from_f32(gpu, vqkv.data(), vqkv.size());
  auto *tvq = h3_gpu_tensor_new_f32(gpu, seq * inner);
  auto *tvk = h3_gpu_tensor_new_f32(gpu, seq * inner);
  auto *tvv = h3_gpu_tensor_new_f32(gpu, seq * inner);
  OP(h3_gpu_begin(gpu), "begin video");
  OP(h3_gpu_video_qkv_rope_f32(gpu, tvq, tvk, tvv, tvqkv, tcos, tsin, seq, heads,
                               dim, rh, eps),
     "video_qkv");
  OP(h3_gpu_submit(gpu), "submit video");
  std::vector<float> vq(seq * inner), vk(seq * inner), vv(seq * inner);
  OP(h3_gpu_tensor_read_f32(tvq, vq.data(), vq.size()), "read vq");
  OP(h3_gpu_tensor_read_f32(tvk, vk.data(), vk.size()), "read vk");
  OP(h3_gpu_tensor_read_f32(tvv, vv.data(), vv.size()), "read vv");
  for (uint32_t r = 0; r < seq; r++)
    for (uint32_t h = 0; h < heads; h++) {
      size_t base = ((size_t)r * heads + h) * dim * 3;
      float qinv = rms_inv(vqkv.data() + base, dim, eps);
      float kinv = rms_inv(vqkv.data() + base + dim, dim, eps);
      float qo[4], ko[4];
      for (uint32_t d = 0; d < dim; d++) {
        qo[d] = vqkv[base + d] * qinv;
        ko[d] = vqkv[base + dim + d] * kinv;
        size_t oi = ((size_t)r * heads + h) * dim + d;
        if (std::fabs(vv[oi] - vqkv[base + 2 * dim + d]) > 1e-5f)
          return die("video V mismatch");
      }
      for (uint32_t d = 0; d < rh; d++) {
        float c = cos[r * rh + d], s = sin[r * rh + d];
        float q0 = qo[d], q1 = qo[d + rh];
        float k0 = ko[d], k1 = ko[d + rh];
        qo[d] = q0 * c - q1 * s;
        qo[d + rh] = q0 * s + q1 * c;
        ko[d] = k0 * c - k1 * s;
        ko[d + rh] = k0 * s + k1 * c;
      }
      for (uint32_t d = 0; d < dim; d++) {
        size_t oi = ((size_t)r * heads + h) * dim + d;
        if (std::fabs(vq[oi] - qo[d]) > 1e-5f) return die("video Q mismatch");
        if (std::fabs(vk[oi] - ko[d]) > 1e-5f) return die("video K mismatch");
      }
    }
  std::printf("ok video_qkv_rope_f32\n");

  /* SDPA F32: identity-ish — Q=K=V ones, expect uniform avg of V */
  const uint32_t s2 = 3, h2 = 1, d2 = 4;
  std::vector<float> ones((size_t)s2 * h2 * d2, 1.f);
  auto *sq = h3_gpu_tensor_from_f32(gpu, ones.data(), ones.size());
  auto *sk = h3_gpu_tensor_from_f32(gpu, ones.data(), ones.size());
  auto *sv = h3_gpu_tensor_from_f32(gpu, ones.data(), ones.size());
  auto *so = h3_gpu_tensor_new_f32(gpu, ones.size());
  float scale = 1.f / std::sqrt((float)d2);
  OP(h3_gpu_begin(gpu), "begin sdpa");
  OP(h3_gpu_sdpa_f32(gpu, so, sq, sk, sv, s2, h2, d2, scale), "sdpa");
  OP(h3_gpu_submit(gpu), "submit sdpa");
  std::vector<float> sout(ones.size());
  OP(h3_gpu_tensor_read_f32(so, sout.data(), sout.size()), "read sdpa");
  for (float v : sout)
    if (std::fabs(v - 1.f) > 1e-4f) return die("sdpa ones mismatch");
  std::printf("ok sdpa_f32\n");

  /* Ungrouped BF16 QKV: cast tiny F32 path through bf16 API */
  std::vector<uint16_t> qkv_b(qkv.size()), qn_b(dim), kn_b(dim), cos_b(cos.size()),
      sin_b(sin.size());
  auto f2b = [](float f) -> uint16_t {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
  };
  for (size_t i = 0; i < qkv.size(); i++) qkv_b[i] = f2b(qkv[i]);
  for (uint32_t i = 0; i < dim; i++) {
    qn_b[i] = f2b(1.f);
    kn_b[i] = f2b(1.f);
  }
  for (size_t i = 0; i < cos.size(); i++) {
    cos_b[i] = f2b(cos[i]);
    sin_b[i] = f2b(sin[i]);
  }
  auto *bqkv = h3_gpu_tensor_from_bf16(gpu, qkv_b.data(), qkv_b.size());
  auto *bqn = h3_gpu_tensor_from_bf16(gpu, qn_b.data(), qn_b.size());
  auto *bkn = h3_gpu_tensor_from_bf16(gpu, kn_b.data(), kn_b.size());
  auto *bcos = h3_gpu_tensor_from_bf16(gpu, cos_b.data(), cos_b.size());
  auto *bsin = h3_gpu_tensor_from_bf16(gpu, sin_b.data(), sin_b.size());
  auto *bq = h3_gpu_tensor_new_bf16(gpu, seq * inner);
  auto *bk = h3_gpu_tensor_new_bf16(gpu, seq * inner);
  auto *bv = h3_gpu_tensor_new_bf16(gpu, seq * inner);
  OP(h3_gpu_begin(gpu), "begin bf16 qkv");
  OP(h3_gpu_qkv_rope_bf16(gpu, bq, bk, bv, bqkv, bqn, bkn, bcos, bsin, seq,
                          heads, dim, rh, eps),
     "qkv_rope_bf16");
  OP(h3_gpu_submit(gpu), "submit bf16");
  std::vector<uint16_t> bqraw(seq * inner);
  OP(h3_gpu_tensor_read_bf16(bq, bqraw.data(), bqraw.size()), "read bf16 q");
  float absmax = 0.f;
  for (uint16_t bits : bqraw) {
    uint32_t u = ((uint32_t)bits) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    absmax = std::fmax(absmax, std::fabs(x));
  }
  if (!(absmax > 0.f) || !std::isfinite(absmax)) return die("bf16 qkv empty");
  std::printf("ok qkv_rope_bf16 absmax=%g\n", absmax);

  h3_gpu_free(gpu);
  std::printf("ok f32_dit all\n");
  return 0;
}
