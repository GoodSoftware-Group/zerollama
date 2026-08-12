/* Toy DiT fixture parity: load generated safetensors, run F32 block on CUDA,
 * compare to CPU/MLX-style goldens (x.h_out / x.attn_out / x.mlp_out). */
#include "h3_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "h3_safetensors.h"
#ifdef __cplusplus
}
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum {
  SEQUENCE = 32,
  HIDDEN = 256,
  HEADS = 4,
  HEAD_DIM = 32,
  INNER = HEADS * HEAD_DIM,
  FFN = 128,
  T_ROWS = 2,
  T_DIM = 32,
  MODALITIES = 3,
  MODULATION_SLOTS = 6,
  ROPE_HALF = 12
};

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

static void *load_exact(const h3_st_header *fx, const char *name, h3_dtype dt,
                        size_t elems, char *err, size_t errn) {
  const h3_st_tensor *t = h3_st_find(fx, name);
  if (!t || t->dtype != dt || h3_st_tensor_elements(t) != elems) {
    std::snprintf(err, errn, "bad tensor %s", name);
    return nullptr;
  }
  size_t bytes = elems * h3_dtype_size(dt);
  void *p = std::malloc(bytes);
  if (!p || !h3_st_read_data(fx, t, p, bytes, err, errn)) {
    std::free(p);
    return nullptr;
  }
  return p;
}

static double rel_l2(const float *got, const float *want, size_t n, double *abs_out) {
  double num = 0, den = 0, amax = 0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
    amax = std::fmax(amax, std::fabs(d));
  }
  if (abs_out) *abs_out = amax;
  return std::sqrt(num) / (std::sqrt(den) + 1e-12);
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1]
                              : "misc/fixtures/h3_dit_toy_f32.safetensors";
  char err[512]{};
  h3_st_header fx{};
  if (!h3_st_read_header(path, &fx, err, sizeof(err))) return die(err);

  h3_gpu *gpu = h3_gpu_create(nullptr, err, sizeof(err));
  if (!gpu) return die(err);

  auto load_f32 = [&](const char *name, size_t n) -> h3_gpu_tensor * {
    float *h = (float *)load_exact(&fx, name, H3_DTYPE_F32, n, err, sizeof(err));
    if (!h) {
      std::fprintf(stderr, "FAIL load %s: %s\n", name, err);
      return nullptr;
    }
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, h, n);
    std::free(h);
    return t;
  };

  h3_gpu_tensor *h_in = load_f32("x.h_in", SEQUENCE * HIDDEN);
  h3_gpu_tensor *attn_in = load_f32("x.attn_in", SEQUENCE * HIDDEN);
  h3_gpu_tensor *t_emb = load_f32("x.t_emb", T_ROWS * T_DIM);
  h3_gpu_tensor *rope_cos = load_f32("x.rope_cos", SEQUENCE * ROPE_HALF);
  h3_gpu_tensor *rope_sin = load_f32("x.rope_sin", SEQUENCE * ROPE_HALF);
  h3_gpu_tensor *norm1 = load_f32("norm1.weight", HIDDEN);
  h3_gpu_tensor *norm2 = load_f32("norm2.weight", HIDDEN);
  h3_gpu_tensor *adaln_w =
      load_f32("adaln_proj.linear.weight",
               (size_t)MODALITIES * MODULATION_SLOTS * HIDDEN * T_DIM);
  h3_gpu_tensor *adaln_b =
      load_f32("adaln_proj.linear.bias",
               (size_t)MODALITIES * MODULATION_SLOTS * HIDDEN);
  h3_gpu_tensor *qkv_w = load_f32("attn.qkv_proj.weight", (size_t)INNER * 3 * HIDDEN);
  h3_gpu_tensor *q_norm = load_f32("attn.q_norm.weight", HEAD_DIM);
  h3_gpu_tensor *k_norm = load_f32("attn.k_norm.weight", HEAD_DIM);
  h3_gpu_tensor *out_w = load_f32("attn.out_proj.weight", (size_t)HIDDEN * INNER);
  h3_gpu_tensor *fc1_w = load_f32("mlp.fc1.weight", (size_t)FFN * 2 * HIDDEN);
  h3_gpu_tensor *fc2_w = load_f32("mlp.fc2.weight", (size_t)HIDDEN * FFN);
  if (!h_in || !attn_in || !t_emb || !rope_cos || !rope_sin || !norm1 || !norm2 ||
      !adaln_w || !adaln_b || !qkv_w || !q_norm || !k_norm || !out_w || !fc1_w ||
      !fc2_w)
    return 1;

  int32_t *runs =
      (int32_t *)load_exact(&fx, "x.runs", H3_DTYPE_I32, 12, err, sizeof(err));
  if (!runs) return die(err);
  std::vector<uint32_t> row_map(SEQUENCE, 0xffffffffu);
  for (int run = 0; run < 4; run++) {
    int32_t start = runs[run * 3], stop = runs[run * 3 + 1], row = runs[run * 3 + 2];
    for (int32_t i = start; i < stop; i++) row_map[i] = (uint32_t)row;
  }
  std::free(runs);
  h3_gpu_tensor *gpu_row_map =
      h3_gpu_tensor_from_u32(gpu, row_map.data(), row_map.size());

  auto *plain_qkv = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER * 3);
  auto *plain_q = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *plain_k = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *plain_v = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *plain_sdpa = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *plain_attn_out = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *plain_fc1 = h3_gpu_tensor_new_f32(gpu, SEQUENCE * FFN * 2);
  auto *plain_swiglu = h3_gpu_tensor_new_f32(gpu, SEQUENCE * FFN);
  auto *plain_mlp_out = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);

  auto *t_silu = h3_gpu_tensor_new_f32(gpu, T_ROWS * T_DIM);
  auto *modulation = h3_gpu_tensor_new_f32(
      gpu, (size_t)T_ROWS * MODALITIES * MODULATION_SLOTS * HIDDEN);
  auto *mod_attn = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *full_qkv = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER * 3);
  auto *full_q = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *full_k = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *full_v = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *full_sdpa = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
  auto *full_attn_out = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *after_attn = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *mod_mlp = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *full_fc1 = h3_gpu_tensor_new_f32(gpu, SEQUENCE * FFN * 2);
  auto *full_swiglu = h3_gpu_tensor_new_f32(gpu, SEQUENCE * FFN);
  auto *full_mlp_out = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);
  auto *h_out = h3_gpu_tensor_new_f32(gpu, SEQUENCE * HIDDEN);

  float scale = 1.f / std::sqrt((float)HEAD_DIM);

  OP(h3_gpu_begin(gpu), "begin");
  /* Plain path */
  OP(h3_gpu_linear_f32(gpu, plain_qkv, attn_in, qkv_w, nullptr, SEQUENCE, HIDDEN,
                       INNER * 3),
     "plain qkv");
  OP(h3_gpu_qkv_rope_f32(gpu, plain_q, plain_k, plain_v, plain_qkv, q_norm,
                         k_norm, rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                         ROPE_HALF, 1e-5f),
     "plain rope");
  OP(h3_gpu_sdpa_f32(gpu, plain_sdpa, plain_q, plain_k, plain_v, SEQUENCE, HEADS,
                     HEAD_DIM, scale),
     "plain sdpa");
  OP(h3_gpu_linear_f32(gpu, plain_attn_out, plain_sdpa, out_w, nullptr, SEQUENCE,
                       INNER, HIDDEN),
     "plain attn out");
  OP(h3_gpu_linear_f32(gpu, plain_fc1, attn_in, fc1_w, nullptr, SEQUENCE, HIDDEN,
                       FFN * 2),
     "plain fc1");
  OP(h3_gpu_swiglu_f32(gpu, plain_swiglu, plain_fc1, SEQUENCE, FFN), "plain swiglu");
  OP(h3_gpu_linear_f32(gpu, plain_mlp_out, plain_swiglu, fc2_w, nullptr, SEQUENCE,
                       FFN, HIDDEN),
     "plain mlp out");

  /* Full block */
  OP(h3_gpu_silu_f32(gpu, t_silu, t_emb, T_ROWS * T_DIM), "silu");
  OP(h3_gpu_linear_f32(gpu, modulation, t_silu, adaln_w, adaln_b, T_ROWS, T_DIM,
                       MODALITIES * MODULATION_SLOTS * HIDDEN),
     "adaln proj");
  OP(h3_gpu_adaln_f32(gpu, mod_attn, h_in, norm1, modulation, gpu_row_map,
                      SEQUENCE, HIDDEN, MODULATION_SLOTS, 0, 1, 1e-5f),
     "attn adaln");
  OP(h3_gpu_linear_f32(gpu, full_qkv, mod_attn, qkv_w, nullptr, SEQUENCE, HIDDEN,
                       INNER * 3),
     "full qkv");
  OP(h3_gpu_qkv_rope_f32(gpu, full_q, full_k, full_v, full_qkv, q_norm, k_norm,
                         rope_cos, rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                         ROPE_HALF, 1e-5f),
     "full rope");
  OP(h3_gpu_sdpa_f32(gpu, full_sdpa, full_q, full_k, full_v, SEQUENCE, HEADS,
                     HEAD_DIM, scale),
     "full sdpa");
  OP(h3_gpu_linear_f32(gpu, full_attn_out, full_sdpa, out_w, nullptr, SEQUENCE,
                       INNER, HIDDEN),
     "full attn out");
  OP(h3_gpu_gate_f32(gpu, after_attn, h_in, full_attn_out, modulation,
                     gpu_row_map, SEQUENCE, HIDDEN, MODULATION_SLOTS, 2),
     "attn gate");
  OP(h3_gpu_adaln_f32(gpu, mod_mlp, after_attn, norm2, modulation, gpu_row_map,
                      SEQUENCE, HIDDEN, MODULATION_SLOTS, 3, 4, 1e-5f),
     "mlp adaln");
  OP(h3_gpu_linear_f32(gpu, full_fc1, mod_mlp, fc1_w, nullptr, SEQUENCE, HIDDEN,
                       FFN * 2),
     "full fc1");
  OP(h3_gpu_swiglu_f32(gpu, full_swiglu, full_fc1, SEQUENCE, FFN), "full swiglu");
  OP(h3_gpu_linear_f32(gpu, full_mlp_out, full_swiglu, fc2_w, nullptr, SEQUENCE,
                       FFN, HIDDEN),
     "full mlp out");
  OP(h3_gpu_gate_f32(gpu, h_out, after_attn, full_mlp_out, modulation,
                     gpu_row_map, SEQUENCE, HIDDEN, MODULATION_SLOTS, 5),
     "mlp gate");
  OP(h3_gpu_submit(gpu), "submit");

  float *want_attn =
      (float *)load_exact(&fx, "x.attn_out", H3_DTYPE_F32, SEQUENCE * HIDDEN, err,
                          sizeof(err));
  float *want_mlp =
      (float *)load_exact(&fx, "x.mlp_out", H3_DTYPE_F32, SEQUENCE * HIDDEN, err,
                          sizeof(err));
  float *want_h =
      (float *)load_exact(&fx, "x.h_out", H3_DTYPE_F32, SEQUENCE * HIDDEN, err,
                          sizeof(err));
  if (!want_attn || !want_mlp || !want_h) return die(err);

  std::vector<float> got_attn(SEQUENCE * HIDDEN), got_mlp(SEQUENCE * HIDDEN),
      got_h(SEQUENCE * HIDDEN);
  OP(h3_gpu_tensor_read_f32(plain_attn_out, got_attn.data(), got_attn.size()),
     "read attn");
  OP(h3_gpu_tensor_read_f32(plain_mlp_out, got_mlp.data(), got_mlp.size()),
     "read mlp");
  OP(h3_gpu_tensor_read_f32(h_out, got_h.data(), got_h.size()), "read h");

  double a_abs, m_abs, b_abs;
  double a_rel = rel_l2(got_attn.data(), want_attn, got_attn.size(), &a_abs);
  double m_rel = rel_l2(got_mlp.data(), want_mlp, got_mlp.size(), &m_abs);
  double b_rel = rel_l2(got_h.data(), want_h, got_h.size(), &b_abs);
  std::free(want_attn);
  std::free(want_mlp);
  std::free(want_h);

  std::printf("CUDA/CPU toy fixture: attention rel %.3g abs %.3g; "
              "MLP rel %.3g abs %.3g; block rel %.3g abs %.3g\n",
              a_rel, a_abs, m_rel, m_abs, b_rel, b_abs);
  if (a_rel > 5e-3 || m_rel > 5e-3 || b_rel > 5e-3)
    return die("exceeds error bound (5e-3 rel)");

  h3_st_free_header(&fx);
  h3_gpu_free(gpu);
  std::puts("ok: CUDA toy DiT block matches fixture goldens");
  return 0;
}
