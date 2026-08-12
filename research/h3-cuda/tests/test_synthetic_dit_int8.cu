/* Synthetic DiT block on the portable int8 path:
 * AdaLN → QKV int8+RoPE → SDPA → linear_int8 → gate_adaln_quantize →
 * mlp_int8 → gate. Small dims; no checkpoint. */
#include "h3_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum {
  ROWS = 8,
  HIDDEN = 64,
  HEADS = 2,
  HEAD_DIM = 32,
  INNER = HEADS * HEAD_DIM,
  FFN = 64,
  SLOTS = 6,
  ROPE_HALF = 16
};

static uint16_t f2b(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

static float b2f(uint16_t bits) {
  uint32_t u = ((uint32_t)bits) << 16;
  float x;
  memcpy(&x, &u, 4);
  return x;
}

static int die(const char *m) {
  std::fprintf(stderr, "FAIL: %s\n", m);
  return 1;
}

static h3_gpu_tensor *fill_bf16(h3_gpu *gpu, size_t n, float v) {
  std::vector<uint16_t> h(n, f2b(v));
  return h3_gpu_tensor_from_bf16(gpu, h.data(), n);
}

#define OP(call, label)                                                        \
  do {                                                                         \
    if (!(call)) {                                                             \
      std::fprintf(stderr, "FAIL %s: %s\n", label, h3_gpu_error(gpu));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);
  if (!h3_gpu_has_int8_mlp(gpu)) return die("int8 path disabled");

  auto *hidden = fill_bf16(gpu, ROWS * HIDDEN, 0.1f);
  auto *norm1 = fill_bf16(gpu, HIDDEN, 1.f);
  auto *norm2 = fill_bf16(gpu, HIDDEN, 1.f);
  auto *qkv_w_bf = fill_bf16(gpu, (size_t)INNER * 3 * HIDDEN, 0.02f);
  auto *out_w_bf = fill_bf16(gpu, (size_t)HIDDEN * INNER, 0.02f);
  auto *fc1_w_bf = fill_bf16(gpu, (size_t)FFN * 2 * HIDDEN, 0.02f);
  auto *fc2_w_bf = fill_bf16(gpu, (size_t)HIDDEN * FFN, 0.02f);
  auto *q_norm = fill_bf16(gpu, HEAD_DIM, 1.f);
  auto *k_norm = fill_bf16(gpu, HEAD_DIM, 1.f);
  auto *mod = fill_bf16(gpu, (size_t)ROWS * SLOTS * HIDDEN, 0.05f);
  std::vector<uint32_t> map(ROWS);
  for (uint32_t i = 0; i < ROWS; i++) map[i] = i;
  auto *row_map = h3_gpu_tensor_from_u32(gpu, map.data(), map.size());

  std::vector<uint16_t> cos(ROWS * ROPE_HALF), sinv(ROWS * ROPE_HALF);
  for (uint32_t r = 0; r < ROWS; r++)
    for (uint32_t d = 0; d < ROPE_HALF; d++) {
      float ang = 0.01f * (float)(r * ROPE_HALF + d);
      cos[r * ROPE_HALF + d] = f2b(std::cos(ang));
      sinv[r * ROPE_HALF + d] = f2b(std::sin(ang));
    }
  auto *rope_cos = h3_gpu_tensor_from_bf16(gpu, cos.data(), cos.size());
  auto *rope_sin = h3_gpu_tensor_from_bf16(gpu, sinv.data(), sinv.size());

  /* Quantized weights */
  auto *qkv_w = h3_gpu_tensor_new_i8(gpu, (size_t)INNER * 3 * HIDDEN);
  auto *qkv_s = h3_gpu_tensor_new_f32(gpu, INNER * 3);
  auto *out_w = h3_gpu_tensor_new_i8(gpu, (size_t)HIDDEN * INNER);
  auto *out_s = h3_gpu_tensor_new_f32(gpu, HIDDEN);
  auto *fc1_w = h3_gpu_tensor_new_i8(gpu, (size_t)FFN * 2 * HIDDEN);
  auto *fc1_s = h3_gpu_tensor_new_f32(gpu, FFN * 2);
  auto *fc2_w = h3_gpu_tensor_new_i8(gpu, (size_t)HIDDEN * FFN);
  auto *fc2_s = h3_gpu_tensor_new_f32(gpu, HIDDEN);

  OP(h3_gpu_begin(gpu), "begin qw");
  OP(h3_gpu_quantize_weight_int8(gpu, qkv_w, qkv_s, qkv_w_bf, INNER * 3, HIDDEN),
     "q qkv");
  OP(h3_gpu_quantize_weight_int8(gpu, out_w, out_s, out_w_bf, HIDDEN, INNER),
     "q out");
  OP(h3_gpu_quantize_weight_int8(gpu, fc1_w, fc1_s, fc1_w_bf, FFN * 2, HIDDEN),
     "q fc1");
  OP(h3_gpu_quantize_weight_int8(gpu, fc2_w, fc2_s, fc2_w_bf, HIDDEN, FFN),
     "q fc2");
  OP(h3_gpu_submit(gpu), "submit qw");

  auto *mod_attn = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *query = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *key = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *value = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *heads = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *attn_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *mod_mlp = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *mlp_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *qi = h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * HIDDEN * 2);
  auto *is = h3_gpu_tensor_new_f32(gpu, ROWS);
  auto *qi2 = h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * INNER);
  auto *is2 = h3_gpu_tensor_new_f32(gpu, ROWS);
  auto *act = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * FFN);
  auto *qact = h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * FFN * 2);
  auto *as = h3_gpu_tensor_new_f32(gpu, ROWS);
  auto *gated = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *q_adaln = h3_gpu_tensor_new_i8(gpu, ROWS * HIDDEN);
  auto *q_scales = h3_gpu_tensor_new_f32(gpu, ROWS);

  OP(h3_gpu_begin(gpu), "begin block");
  OP(h3_gpu_adaln_bf16(gpu, mod_attn, hidden, norm1, mod, row_map, ROWS, HIDDEN,
                       SLOTS, 0, 1, 1e-5f),
     "attn AdaLN");
  OP(h3_gpu_grouped_qkv_linear_rope_int8(
         gpu, query, key, value, qi, is, mod_attn, qkv_w, qkv_s, q_norm, k_norm,
         rope_cos, rope_sin, ROWS, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f, 0,
         0, 0, 0),
     "QKV int8+RoPE");
  OP(h3_gpu_sdpa_bf16(gpu, heads, query, key, value, ROWS, HEADS, HEAD_DIM,
                      1.f / std::sqrt((float)HEAD_DIM)),
     "SDPA");
  OP(h3_gpu_linear_int8_bf16(gpu, attn_out, qi2, is2, heads, out_w, out_s, ROWS,
                             INNER, HIDDEN, 0),
     "attn out int8");
  OP(h3_gpu_gate_adaln_quantize_int8(gpu, gated, q_adaln, q_scales, hidden,
                                      attn_out, norm2, mod, mod, row_map, ROWS,
                                      ROWS, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
     "gate+AdaLN+q");
  /* Feed quantized AdaLN into MLP via prequantized flag: need BF16 for mlp
   * input_is_quantized=0 path using mod_mlp from AdaLN alone. */
  OP(h3_gpu_adaln_bf16(gpu, mod_mlp, gated, norm2, mod, row_map, ROWS, HIDDEN,
                       SLOTS, 3, 4, 1e-5f),
     "mlp AdaLN");
  OP(h3_gpu_mlp_int8_bf16(gpu, mlp_out, act, qact, as, mod_mlp, fc1_w, fc1_s,
                           fc2_w, fc2_s, nullptr, nullptr, ROWS, HIDDEN, FFN,
                           HIDDEN, 0, 0, 1, 0),
     "MLP int8");
  OP(h3_gpu_gate_bf16(gpu, hidden, gated, mlp_out, mod, row_map, ROWS, HIDDEN,
                      SLOTS, 5),
     "MLP gate");
  OP(h3_gpu_submit(gpu), "submit");

  std::vector<uint16_t> out(ROWS * HIDDEN);
  if (!h3_gpu_tensor_read_bf16(hidden, out.data(), out.size()))
    return die("read hidden");
  float sum = 0.f;
  for (auto b : out) {
    float f = b2f(b);
    if (!std::isfinite(f)) return die("non-finite hidden");
    sum += std::fabs(f);
  }
  if (!(sum > 0.f)) return die("hidden zero");

  h3_gpu_stats st{};
  h3_gpu_get_stats(gpu, &st);
  std::printf("ok synthetic int8 DiT block  rows=%d hidden=%d  |hidden|=%.4f  "
              "dispatches=%llu peak_bytes=%llu has_int8=%d\n",
              ROWS, HIDDEN, sum, (unsigned long long)st.direct_dispatches,
              (unsigned long long)st.peak_live_bytes, h3_gpu_has_int8_mlp(gpu));
  h3_gpu_free(gpu);
  return 0;
}
