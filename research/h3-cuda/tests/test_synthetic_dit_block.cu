/* Synthetic DiT block: AdaLN → QKV/RoPE → SDPA → linear → gate_adaln → MLP.
 * Small dims; exercises the BF16 close-reference op chain. */
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
  FFN = 128,
  SLOTS = 6,
  ROPE_HALF = 16
};

static uint16_t f2b(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

static int die(const char *m) {
  std::fprintf(stderr, "FAIL: %s\n", m);
  return 1;
}

static h3_gpu_tensor *fill_bf16(h3_gpu *gpu, size_t n, float v) {
  std::vector<uint16_t> h(n, f2b(v));
  return h3_gpu_tensor_from_bf16(gpu, h.data(), n);
}

int main() {
  char err[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);

  auto *hidden = fill_bf16(gpu, ROWS * HIDDEN, 0.1f);
  auto *norm1 = fill_bf16(gpu, HIDDEN, 1.f);
  auto *norm2 = fill_bf16(gpu, HIDDEN, 1.f);
  auto *qkv_w = fill_bf16(gpu, (size_t)INNER * 3 * HIDDEN, 0.01f);
  auto *out_w = fill_bf16(gpu, (size_t)HIDDEN * INNER, 0.01f);
  auto *fc1_w = fill_bf16(gpu, (size_t)FFN * 2 * HIDDEN, 0.01f);
  auto *fc2_w = fill_bf16(gpu, (size_t)HIDDEN * FFN, 0.01f);
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

  auto *mod_attn = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER * 3);
  auto *query = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *key = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *value = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *heads = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * INNER);
  auto *attn_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *mod_mlp = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
  auto *mlp_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);

#define OP(call, label)                                                        \
  do {                                                                         \
    if (!(call)) {                                                             \
      std::fprintf(stderr, "FAIL %s: %s\n", label, h3_gpu_error(gpu));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

  OP(h3_gpu_begin(gpu), "begin");
  OP(h3_gpu_adaln_bf16(gpu, mod_attn, hidden, norm1, mod, row_map, ROWS, HIDDEN,
                       SLOTS, 0, 1, 1e-5f),
     "attn AdaLN");
  OP(h3_gpu_grouped_qkv_linear_rope_bf16(
         gpu, query, key, value, qkv, mod_attn, qkv_w, q_norm, k_norm, rope_cos,
         rope_sin, ROWS, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
     "QKV+RoPE");
  OP(h3_gpu_sdpa_bf16(gpu, heads, query, key, value, ROWS, HEADS, HEAD_DIM,
                      1.f / std::sqrt((float)HEAD_DIM)),
     "SDPA");
  OP(h3_gpu_linear_bf16(gpu, attn_out, heads, out_w, nullptr, ROWS, INNER,
                        HIDDEN),
     "attn out");
  OP(h3_gpu_gate_adaln_bf16(gpu, hidden, mod_mlp, hidden, attn_out, norm2, mod,
                            mod, row_map, ROWS, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
     "gate+AdaLN");
  OP(h3_gpu_mlp_bf16(gpu, mlp_out, mod_mlp, fc1_w, fc2_w, ROWS, HIDDEN, FFN,
                     HIDDEN),
     "MLP");
  OP(h3_gpu_gate_bf16(gpu, hidden, hidden, mlp_out, mod, row_map, ROWS, HIDDEN,
                      SLOTS, 5),
     "MLP gate");
  OP(h3_gpu_submit(gpu), "submit");

  std::vector<uint16_t> out(ROWS * HIDDEN);
  if (!h3_gpu_tensor_read_bf16(hidden, out.data(), out.size()))
    return die("read hidden");
  float sum = 0.f;
  for (auto b : out) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    sum += std::fabs(f);
  }
  if (!(sum > 0.f) || !std::isfinite(sum)) return die("hidden not finite/nonzero");

  h3_gpu_stats st{};
  h3_gpu_get_stats(gpu, &st);
  std::printf("ok synthetic DiT block  rows=%d hidden=%d  |hidden|=%.4f  "
              "dispatches=%llu peak_bytes=%llu\n",
              ROWS, HIDDEN, sum, (unsigned long long)st.direct_dispatches,
              (unsigned long long)st.peak_live_bytes);
  h3_gpu_free(gpu);
  return 0;
}
