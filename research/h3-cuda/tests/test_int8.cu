/* Portable int8 quantize + linear + mlp smoke. */
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

static uint16_t f2b(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}
static float b2f(uint16_t bits) {
  uint32_t u = ((uint32_t)bits) << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);
  if (!h3_gpu_has_int8_mlp(gpu)) return die("has_int8_mlp expected 1");
  if (h3_gpu_has_nax_mlp(gpu)) return die("has_nax_mlp expected 0");

  const uint32_t rows = 4, in_dim = 8, out_dim = 4;
  std::vector<float> wh(out_dim * in_dim), xh(rows * in_dim);
  for (size_t i = 0; i < wh.size(); i++)
    wh[i] = b2f(f2b(0.05f * (float)((int)(i % 9) - 4)));
  for (size_t i = 0; i < xh.size(); i++)
    xh[i] = b2f(f2b(0.1f * (float)((int)(i % 7) - 3)));

  std::vector<uint16_t> wb(wh.size()), xb(xh.size());
  for (size_t i = 0; i < wh.size(); i++) wb[i] = f2b(wh[i]);
  for (size_t i = 0; i < xh.size(); i++) xb[i] = f2b(xh[i]);

  auto *tw = h3_gpu_tensor_from_bf16(gpu, wb.data(), wb.size());
  auto *tw_i8 = h3_gpu_tensor_new_i8(gpu, wb.size());
  auto *tw_s = h3_gpu_tensor_new_f32(gpu, out_dim);
  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_quantize_weight_int8(gpu, tw_i8, tw_s, tw, out_dim, in_dim),
     "qweight");
  OP(h3_gpu_submit(gpu), "submit qw");

  std::vector<float> wscales(out_dim);
  OP(h3_gpu_tensor_read_f32(tw_s, wscales.data(), wscales.size()), "read ws");
  /* CPU reference scales */
  for (uint32_t o = 0; o < out_dim; o++) {
    float m = 0.f;
    for (uint32_t k = 0; k < in_dim; k++)
      m = std::fmax(m, std::fabs(wh[o * in_dim + k]));
    float ref = m > 0.f ? m / 127.f : 1.f / 127.f;
    if (std::fabs(wscales[o] - ref) > 1e-4f) return die("weight scale");
  }
  std::printf("ok quantize_weight_int8\n");

  auto *tx = h3_gpu_tensor_from_bf16(gpu, xb.data(), xb.size());
  auto *tqi = h3_gpu_tensor_new_i8(gpu, rows * in_dim);
  auto *tis = h3_gpu_tensor_new_f32(gpu, rows);
  auto *ty = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);
  OP(h3_gpu_begin(gpu), "begin2");
  OP(h3_gpu_linear_int8_bf16(gpu, ty, tqi, tis, tx, tw_i8, tw_s, rows, in_dim,
                             out_dim, 0),
     "linear_int8");
  OP(h3_gpu_submit(gpu), "submit lin");
  std::vector<uint16_t> yb(rows * out_dim);
  OP(h3_gpu_tensor_read_bf16(ty, yb.data(), yb.size()), "read y");
  std::vector<float> xs_gpu(rows);
  OP(h3_gpu_tensor_read_f32(tis, xs_gpu.data(), xs_gpu.size()), "read xs");

  /* Reference with rintf to match CUDA quantize kernel. */
  std::vector<int8_t> xq(rows * in_dim), wq(out_dim * in_dim);
  for (uint32_t r = 0; r < rows; r++) {
    float m = 0.f;
    for (uint32_t k = 0; k < in_dim; k++)
      m = std::fmax(m, std::fabs(xh[r * in_dim + k]));
    float inv = m > 0.f ? 127.f / m : 127.f;
    for (uint32_t k = 0; k < in_dim; k++) {
      int q = (int)std::rintf(xh[r * in_dim + k] * inv);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      xq[r * in_dim + k] = (int8_t)q;
    }
  }
  for (uint32_t o = 0; o < out_dim; o++) {
    float m = 0.f;
    for (uint32_t k = 0; k < in_dim; k++)
      m = std::fmax(m, std::fabs(wh[o * in_dim + k]));
    float inv = m > 0.f ? 127.f / m : 127.f;
    for (uint32_t k = 0; k < in_dim; k++) {
      int q = (int)std::rintf(wh[o * in_dim + k] * inv);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      wq[o * in_dim + k] = (int8_t)q;
    }
  }
  float max_abs = 0.f;
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t o = 0; o < out_dim; o++) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < in_dim; k++)
        acc += (int32_t)xq[r * in_dim + k] * (int32_t)wq[o * in_dim + k];
      float ref = (float)acc * xs_gpu[r] * wscales[o];
      float got = b2f(yb[r * out_dim + o]);
      float e = std::fabs(got - ref);
      if (e > max_abs) max_abs = e;
      if (e > 0.05f)
        std::fprintf(stderr, "diff r=%u o=%u got=%g ref=%g acc=%d sa=%g sw=%g\n",
                     r, o, got, ref, (int)acc, xs_gpu[r], wscales[o]);
    }
  if (max_abs > 0.05f) {
    std::fprintf(stderr, "FAIL linear max_abs=%g\n", max_abs);
    return 1;
  }
  std::printf("ok linear_int8_bf16 max_abs=%g\n", max_abs);

  /* Mid-size path should hit cuBLAS int8 (in>=64, out>=16, in%4==0). */
  {
    const uint32_t r2 = 16, k2 = 128, o2 = 64;
    std::vector<float> x2(r2 * k2), w2(o2 * k2);
    for (size_t i = 0; i < x2.size(); i++)
      x2[i] = b2f(f2b(0.02f * (float)((int)(i % 11) - 5)));
    for (size_t i = 0; i < w2.size(); i++)
      w2[i] = b2f(f2b(0.02f * (float)((int)(i % 9) - 4)));
    std::vector<uint16_t> xb2(x2.size()), wb2(w2.size());
    for (size_t i = 0; i < x2.size(); i++) xb2[i] = f2b(x2[i]);
    for (size_t i = 0; i < w2.size(); i++) wb2[i] = f2b(w2[i]);
    auto *tx2 = h3_gpu_tensor_from_bf16(gpu, xb2.data(), xb2.size());
    auto *tw2 = h3_gpu_tensor_from_bf16(gpu, wb2.data(), wb2.size());
    auto *tw2i = h3_gpu_tensor_new_i8(gpu, wb2.size());
    auto *tw2s = h3_gpu_tensor_new_f32(gpu, o2);
    auto *tqi2b = h3_gpu_tensor_new_i8(gpu, r2 * k2);
    auto *tis2b = h3_gpu_tensor_new_f32(gpu, r2);
    auto *ty2b = h3_gpu_tensor_new_bf16(gpu, r2 * o2);
    OP(h3_gpu_begin(gpu), "begin cublas-size");
    OP(h3_gpu_quantize_weight_int8(gpu, tw2i, tw2s, tw2, o2, k2), "qw2");
    OP(h3_gpu_linear_int8_bf16(gpu, ty2b, tqi2b, tis2b, tx2, tw2i, tw2s, r2, k2,
                               o2, 0),
       "lin cublas-size");
    OP(h3_gpu_submit(gpu), "submit cublas-size");
    std::vector<uint16_t> y2b(r2 * o2);
    OP(h3_gpu_tensor_read_bf16(ty2b, y2b.data(), y2b.size()), "read y2");
    float amax = 0.f;
    for (auto b : y2b) amax = std::fmax(amax, std::fabs(b2f(b)));
    if (!std::isfinite(amax)) return die("cublas-size nan");
    h3_gpu_stats st{};
    h3_gpu_get_stats(gpu, &st);
    std::printf("ok linear_int8 cublas-size absmax=%g linear_dispatches=%llu\n",
                amax, (unsigned long long)st.mps_linear_dispatches);
  }

  /* Head-major linear: layout [heads, rows, dim] */
  const uint32_t heads = 2, hdim = 4;
  std::vector<float> hm(heads * rows * hdim);
  for (size_t i = 0; i < hm.size(); i++)
    hm[i] = 0.08f * (float)((int)(i % 5) - 2);
  std::vector<uint16_t> hmb(hm.size());
  for (size_t i = 0; i < hm.size(); i++) hmb[i] = f2b(hm[i]);
  auto *thm = h3_gpu_tensor_from_bf16(gpu, hmb.data(), hmb.size());
  auto *tqi2 = h3_gpu_tensor_new_i8(gpu, rows * heads * hdim);
  auto *tis2 = h3_gpu_tensor_new_f32(gpu, rows);
  auto *ty2 = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);
  /* Reuse same weight shape out x (heads*hdim)=out x 8 */
  OP(h3_gpu_begin(gpu), "begin hm");
  OP(h3_gpu_linear_int8_head_major_bf16(gpu, ty2, tqi2, tis2, thm, tw_i8, tw_s,
                                         rows, heads, hdim, out_dim),
     "head_major");
  OP(h3_gpu_submit(gpu), "submit hm");
  std::vector<uint16_t> y2(rows * out_dim);
  OP(h3_gpu_tensor_read_bf16(ty2, y2.data(), y2.size()), "read hm");
  float hm_abs = 0.f;
  for (float v : std::vector<float>{b2f(y2[0]), b2f(y2[1])})
    hm_abs = std::fmax(hm_abs, std::fabs(v));
  if (!std::isfinite(hm_abs)) return die("head_major nan");
  std::printf("ok linear_int8_head_major absmax=%g\n", hm_abs);

  /* Tiny MLP int8: in=8 hidden=4 out=4 */
  const uint32_t hid = 4;
  std::vector<float> fc1(hid * 2 * in_dim), fc2(out_dim * hid);
  for (size_t i = 0; i < fc1.size(); i++)
    fc1[i] = 0.04f * (float)((int)(i % 6) - 2);
  for (size_t i = 0; i < fc2.size(); i++)
    fc2[i] = 0.03f * (float)((int)(i % 5) - 1);
  std::vector<uint16_t> fc1b(fc1.size()), fc2b(fc2.size());
  for (size_t i = 0; i < fc1.size(); i++) fc1b[i] = f2b(fc1[i]);
  for (size_t i = 0; i < fc2.size(); i++) fc2b[i] = f2b(fc2[i]);
  auto *tfc1 = h3_gpu_tensor_from_bf16(gpu, fc1b.data(), fc1b.size());
  auto *tfc2 = h3_gpu_tensor_from_bf16(gpu, fc2b.data(), fc2b.size());
  auto *tfc1i = h3_gpu_tensor_new_i8(gpu, fc1.size());
  auto *tfc1s = h3_gpu_tensor_new_f32(gpu, hid * 2);
  auto *tfc2i = h3_gpu_tensor_new_i8(gpu, fc2.size());
  auto *tfc2s = h3_gpu_tensor_new_f32(gpu, out_dim);
  OP(h3_gpu_begin(gpu), "begin qmlp");
  OP(h3_gpu_quantize_weight_int8(gpu, tfc1i, tfc1s, tfc1, hid * 2, in_dim),
     "qfc1");
  OP(h3_gpu_quantize_weight_int8(gpu, tfc2i, tfc2s, tfc2, out_dim, hid),
     "qfc2");
  OP(h3_gpu_submit(gpu), "submit qmlp");

  auto *tact = h3_gpu_tensor_new_bf16(gpu, rows * hid);
  auto *tqa = h3_gpu_tensor_new_i8(gpu, rows * std::max(in_dim, hid * 2));
  auto *tas = h3_gpu_tensor_new_f32(gpu, rows);
  auto *tout = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);
  OP(h3_gpu_begin(gpu), "begin mlp");
  OP(h3_gpu_mlp_int8_bf16(gpu, tout, tact, tqa, tas, tx, tfc1i, tfc1s, tfc2i,
                           tfc2s, nullptr, nullptr, rows, in_dim, hid, out_dim,
                           0, 0, 1, 0),
     "mlp_int8");
  OP(h3_gpu_submit(gpu), "submit mlp");
  std::vector<uint16_t> mout(rows * out_dim);
  OP(h3_gpu_tensor_read_bf16(tout, mout.data(), mout.size()), "read mlp");
  float mabs = 0.f;
  for (auto bits : mout) mabs = std::fmax(mabs, std::fabs(b2f(bits)));
  if (!std::isfinite(mabs)) return die("mlp nan");
  std::printf("ok mlp_int8_bf16 absmax=%g\n", mabs);

  /* gate_adaln_quantize_int8 smoke */
  const uint32_t width = 8, slots = 6;
  std::vector<float> res(rows * width), br(rows * width), nw(width, 1.f),
      mod(2 * slots * width);
  std::vector<uint32_t> map(rows);
  for (uint32_t i = 0; i < rows * width; i++) {
    res[i] = 0.1f * (float)(i + 1);
    br[i] = 0.05f * (float)(i + 2);
  }
  for (uint32_t i = 0; i < rows; i++) map[i] = i % 2;
  for (size_t i = 0; i < mod.size(); i++) mod[i] = 0.01f * (float)((i % 9) + 1);
  std::vector<uint16_t> resb(res.size()), brb(br.size()), nwb(width),
      modb(mod.size());
  for (size_t i = 0; i < res.size(); i++) {
    resb[i] = f2b(res[i]);
    brb[i] = f2b(br[i]);
  }
  for (uint32_t i = 0; i < width; i++) nwb[i] = f2b(1.f);
  for (size_t i = 0; i < mod.size(); i++) modb[i] = f2b(mod[i]);
  auto *tres = h3_gpu_tensor_from_bf16(gpu, resb.data(), resb.size());
  auto *tbr = h3_gpu_tensor_from_bf16(gpu, brb.data(), brb.size());
  auto *tnw = h3_gpu_tensor_from_bf16(gpu, nwb.data(), nwb.size());
  auto *tgmod = h3_gpu_tensor_from_bf16(gpu, modb.data(), modb.size());
  auto *tnmod = h3_gpu_tensor_from_bf16(gpu, modb.data(), modb.size());
  auto *tmap = h3_gpu_tensor_from_u32(gpu, map.data(), map.size());
  auto *tgated = h3_gpu_tensor_new_bf16(gpu, rows * width);
  auto *tqout = h3_gpu_tensor_new_i8(gpu, rows * width);
  auto *tqsc = h3_gpu_tensor_new_f32(gpu, rows);
  OP(h3_gpu_begin(gpu), "begin gateq");
  OP(h3_gpu_gate_adaln_quantize_int8(gpu, tgated, tqout, tqsc, tres, tbr, tnw,
                                      tgmod, tnmod, tmap, rows, rows, width,
                                      slots, 2, 0, 1, 1e-5f),
     "gate_adaln_q");
  OP(h3_gpu_submit(gpu), "submit gateq");
  std::vector<float> qsc(rows);
  OP(h3_gpu_tensor_read_f32(tqsc, qsc.data(), qsc.size()), "read qsc");
  if (!(qsc[0] > 0.f)) return die("gate quant scales");
  std::printf("ok gate_adaln_quantize_int8 scale0=%g\n", qsc[0]);

  /* grouped_qkv_linear_rope_int8 tiny */
  const uint32_t qh = 2, qd = 4, rh = 2;
  std::vector<float> qw(qh * qd * 3 * in_dim);
  for (size_t i = 0; i < qw.size(); i++)
    qw[i] = 0.02f * (float)((int)(i % 8) - 3);
  std::vector<uint16_t> qwb(qw.size());
  for (size_t i = 0; i < qw.size(); i++) qwb[i] = f2b(qw[i]);
  auto *tqw = h3_gpu_tensor_from_bf16(gpu, qwb.data(), qwb.size());
  auto *tqwi = h3_gpu_tensor_new_i8(gpu, qw.size());
  auto *tqws = h3_gpu_tensor_new_f32(gpu, qh * qd * 3);
  OP(h3_gpu_begin(gpu), "begin qkv w");
  OP(h3_gpu_quantize_weight_int8(gpu, tqwi, tqws, tqw, qh * qd * 3, in_dim),
     "qkv w");
  OP(h3_gpu_submit(gpu), "submit qkv w");
  std::vector<uint16_t> ones(qd, f2b(1.f));
  std::vector<float> cos(rows * rh), sinv(rows * rh);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t d = 0; d < rh; d++) {
      cos[r * rh + d] = std::cos(0.1f * (r + d + 1));
      sinv[r * rh + d] = std::sin(0.1f * (r + d + 1));
    }
  std::vector<uint16_t> cosb(cos.size()), sinb(sinv.size());
  for (size_t i = 0; i < cos.size(); i++) {
    cosb[i] = f2b(cos[i]);
    sinb[i] = f2b(sinv[i]);
  }
  auto *tqn = h3_gpu_tensor_from_bf16(gpu, ones.data(), ones.size());
  auto *tkn = h3_gpu_tensor_from_bf16(gpu, ones.data(), ones.size());
  auto *tcos = h3_gpu_tensor_from_bf16(gpu, cosb.data(), cosb.size());
  auto *tsin = h3_gpu_tensor_from_bf16(gpu, sinb.data(), sinb.size());
  auto *tq = h3_gpu_tensor_new_bf16(gpu, rows * qh * qd);
  auto *tk = h3_gpu_tensor_new_bf16(gpu, rows * qh * qd);
  auto *tv = h3_gpu_tensor_new_bf16(gpu, rows * qh * qd);
  auto *tqi3 = h3_gpu_tensor_new_i8(gpu, rows * in_dim);
  auto *tis3 = h3_gpu_tensor_new_f32(gpu, rows);
  OP(h3_gpu_begin(gpu), "begin qkv");
  OP(h3_gpu_grouped_qkv_linear_rope_int8(
         gpu, tq, tk, tv, tqi3, tis3, tx, tqwi, tqws, tqn, tkn, tcos, tsin,
         rows, in_dim, qh, qd, rh, 1e-5f, 0, 0, 0, 0),
     "qkv_int8");
  OP(h3_gpu_submit(gpu), "submit qkv");
  std::vector<uint16_t> qq(rows * qh * qd);
  OP(h3_gpu_tensor_read_bf16(tq, qq.data(), qq.size()), "read q");
  float qabs = 0.f;
  for (auto b : qq) qabs = std::fmax(qabs, std::fabs(b2f(b)));
  if (!(qabs > 0.f) || !std::isfinite(qabs)) return die("qkv empty");
  std::printf("ok grouped_qkv_linear_rope_int8 absmax=%g\n", qabs);

  /* NAX fallback still callable */
  auto *tnax = h3_gpu_tensor_new_bf16(gpu, rows * out_dim);
  auto *tact2 = h3_gpu_tensor_new_bf16(gpu, rows * hid);
  OP(h3_gpu_begin(gpu), "begin nax");
  OP(h3_gpu_mlp_nax_bf16(gpu, tnax, tact2, tx, tfc1, tfc2, rows, in_dim, hid,
                         out_dim),
     "nax_fallback");
  OP(h3_gpu_submit(gpu), "submit nax");
  std::printf("ok mlp_nax_bf16 fallback\n");

  h3_gpu_free(gpu);
  std::printf("ok int8 all  has_int8=%d\n", 1);
  return 0;
}
