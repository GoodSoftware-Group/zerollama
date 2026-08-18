/* Qwen3-VL language decoder host math (H3 text encoder). No 32B/4B shards. */
#ifndef H3_QWEN_TE_HOST_H
#define H3_QWEN_TE_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MiniMax-H3 stock conditioner: first 50 of 64 Qwen3-VL-32B layers. */
enum {
  H3_QWEN_TE_HIDDEN_32B = 5120,
  H3_QWEN_TE_LAYERS_READ = 50,
  H3_QWEN_TE_HEADS_32B = 64,
  H3_QWEN_TE_KV_32B = 8,
  H3_QWEN_TE_HEAD_DIM = 128,
  H3_QWEN_TE_FFN_32B = 25600,
  H3_QWEN_TE_VOCAB = 151936,
  H3_QWEN_TE_MROPE_FIRST = 60 /* half-dims using interleaved t/h/w axes */
};

/* ClipProj-4B path (Qwen3-VL-4B hidden → 5120). */
enum {
  H3_QWEN_TE_HIDDEN_4B = 2560,
  H3_QWEN_TE_HEADS_4B = 32,
  H3_QWEN_TE_KV_4B = 8,
  H3_QWEN_TE_FFN_4B = 9728,
  H3_QWEN_TE_LAYERS_4B = 36,
  /* ClipProj celeb-mlp is fitted on this decoder tap, not last_hidden_state. */
  H3_QWEN_TE_CLIPPROJ_TAP = 24
};

#define H3_QWEN_TE_RMS_EPS 1e-6f
#define H3_QWEN_TE_ROPE_THETA 5000000.0f

/*
 * RoPE tables [tokens, head_dim/2].
 * position_ids NULL → sequential 0..T-1 on axis 0.
 * Else axis-major [3, tokens] (antirez / Qwen mRoPE).
 * mrope: for freq index < H3_QWEN_TE_MROPE_FIRST, axis = index % 3.
 */
int h3_qwen_te_rope_tables(size_t tokens, int head_dim, float theta,
                           const uint32_t *position_ids, float *cos,
                           float *sin);

/* rotate-half on last dim: x [tokens, heads, head_dim], tables [tokens, half]. */
int h3_qwen_te_apply_rope(float *x, size_t tokens, int heads, int head_dim,
                          const float *cos, const float *sin);

/* RMSNorm per head: x [tokens, heads, dim], weight [dim] (may be NULL). */
int h3_qwen_te_head_rmsnorm(float *x, size_t tokens, int heads, int dim,
                            float eps, const float *weight);

/* Repeat KV heads: src [T, n_kv, D] → dst [T, n_q, D]. n_q % n_kv == 0. */
int h3_qwen_te_repeat_kv(const float *src, size_t tokens, int n_kv, int n_q,
                         int head_dim, float *dst);

/*
 * One decoder layer (pre-norm). Layouts row-major:
 *   hidden [T, hidden]
 *   Wq [Hq*D, hidden], Wk/Wv [Hkv*D, hidden], Wo [hidden, Hq*D]
 *   q_norm/k_norm [D], input_norm/post_norm [hidden]
 *   Wgate/Wup [ffn, hidden], Wdown [hidden, ffn]
 */
int h3_qwen_te_layer(float *hidden, size_t tokens, int hidden_dim, int n_q,
                     int n_kv, int head_dim, int ffn,
                     const float *input_norm, const float *Wq, const float *Wk,
                     const float *Wv, const float *Wo, const float *q_norm,
                     const float *k_norm, const float *post_norm,
                     const float *Wgate, const float *Wup, const float *Wdown,
                     const float *rope_cos, const float *rope_sin);

/* Deterministic 4B-width stand-in until Qwen3-VL-4B shards exist. */
void h3_qwen_te_hash_embed(const uint32_t *ids, size_t n, int dim,
                           float *hidden);

#ifdef __cplusplus
}
#endif

#endif /* H3_QWEN_TE_HOST_H */
