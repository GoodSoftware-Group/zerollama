#include "h3_qwen_te_4b.h"

#include "h3_dit_host.h"
#include "h3_qwen_te_host.h"
#include "h3_st_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HID H3_QWEN_TE_HIDDEN_4B
#define NQ H3_QWEN_TE_HEADS_4B
#define NKV H3_QWEN_TE_KV_4B
#define HD H3_QWEN_TE_HEAD_DIM
#define FFN H3_QWEN_TE_FFN_4B
#define LAYERS H3_QWEN_TE_LAYERS_4B
#define VOCAB H3_QWEN_TE_VOCAB

int h3_qwen_te_4b_forward(const char *model_dir, const uint32_t *ids, size_t n,
                          const uint32_t *position_ids, int apply_final_norm,
                          float *hidden_out, char *error, size_t error_size) {
  if (!model_dir || !ids || !hidden_out || n < 1) {
    if (error && error_size)
      snprintf(error, error_size, "invalid Qwen3-VL-4B forward args");
    return 0;
  }
  for (size_t i = 0; i < n; i++) {
    if (ids[i] >= (uint32_t)VOCAB) {
      if (error && error_size)
        snprintf(error, error_size, "token id %u >= vocab %d", ids[i], VOCAB);
      return 0;
    }
  }
  int n_layers = LAYERS;
  const char *el = getenv("H3_QWEN_TE_LAYERS");
  if (el && el[0]) {
    int k = atoi(el);
    if (k >= 1 && k <= LAYERS)
      n_layers = k;
  }
  h3_st_store *st = h3_st_store_open(model_dir, error, error_size);
  if (!st)
    return 0;

  const float *embed = h3_st_store_get_f32(
      st, "model.language_model.embed_tokens.weight", NULL, error, error_size);
  if (!embed) {
    h3_st_store_free(st);
    return 0;
  }
  for (size_t t = 0; t < n; t++)
    memcpy(hidden_out + t * (size_t)HID, embed + (size_t)ids[t] * (size_t)HID,
           (size_t)HID * sizeof(float));

  float *cos = (float *)malloc(n * (size_t)(HD / 2) * sizeof(float));
  float *sin = (float *)malloc(n * (size_t)(HD / 2) * sizeof(float));
  if (!cos || !sin) {
    free(cos);
    free(sin);
    h3_st_store_free(st);
    if (error && error_size)
      snprintf(error, error_size, "oom rope tables");
    return 0;
  }
  if (h3_qwen_te_rope_tables(n, HD, H3_QWEN_TE_ROPE_THETA, position_ids, cos,
                             sin) != 0) {
    free(cos);
    free(sin);
    h3_st_store_free(st);
    if (error && error_size)
      snprintf(error, error_size, "rope tables failed");
    return 0;
  }

  const size_t qn = (size_t)NQ * HD;
  const size_t kvn = (size_t)NKV * HD;
  const size_t wq_n = qn * (size_t)HID;
  const size_t wkv_n = kvn * (size_t)HID;
  const size_t wo_n = (size_t)HID * qn;
  const size_t wffn_n = (size_t)FFN * (size_t)HID;
  const size_t wd_n = (size_t)HID * (size_t)FFN;

  for (int layer = 0; layer < n_layers; layer++) {
    char name[160];
    const float *in_n = NULL, *post = NULL, *qnrm = NULL, *knrm = NULL;
    const float *Wq = NULL, *Wk = NULL, *Wv = NULL, *Wo = NULL;
    const float *Wg = NULL, *Wu = NULL, *Wd = NULL;
#define GET(dst, suffix, ne)                                                   \
  do {                                                                         \
    size_t nout = 0;                                                           \
    snprintf(name, sizeof(name), "model.language_model.layers.%d." suffix,     \
             layer);                                                           \
    (dst) = h3_st_store_get_f32(st, name, &nout, error, error_size);           \
    if (!(dst) || nout != (ne)) {                                              \
      h3_st_store_free(st);                                                    \
      return 0;                                                                \
    }                                                                          \
  } while (0)
    GET(in_n, "input_layernorm.weight", (size_t)HID);
    GET(post, "post_attention_layernorm.weight", (size_t)HID);
    GET(qnrm, "self_attn.q_norm.weight", (size_t)HD);
    GET(knrm, "self_attn.k_norm.weight", (size_t)HD);
    GET(Wq, "self_attn.q_proj.weight", wq_n);
    GET(Wk, "self_attn.k_proj.weight", wkv_n);
    GET(Wv, "self_attn.v_proj.weight", wkv_n);
    GET(Wo, "self_attn.o_proj.weight", wo_n);
    GET(Wg, "mlp.gate_proj.weight", wffn_n);
    GET(Wu, "mlp.up_proj.weight", wffn_n);
    GET(Wd, "mlp.down_proj.weight", wd_n);
#undef GET
    int rc = h3_qwen_te_layer(hidden_out, n, HID, NQ, NKV, HD, FFN, in_n, Wq,
                              Wk, Wv, Wo, qnrm, knrm, post, Wg, Wu, Wd, cos,
                              sin);
    if (rc != 0) {
      free(cos);
      free(sin);
      h3_st_store_free(st);
      if (error && error_size)
        snprintf(error, error_size, "decoder layer %d failed", layer);
      return 0;
    }
  }
  free(cos);
  free(sin);

  if (apply_final_norm) {
    const float *nw = h3_st_store_get_f32(st, "model.language_model.norm.weight",
                                          NULL, error, error_size);
    float *tmp = (float *)malloc(n * (size_t)HID * sizeof(float));
    if (!nw || !tmp) {
      free(tmp);
      h3_st_store_free(st);
      if (error && error_size)
        snprintf(error, error_size, "oom final norm");
      return 0;
    }
    int rc = h3_dit_rmsnorm(hidden_out, (int)n, HID, H3_QWEN_TE_RMS_EPS, nw,
                            tmp);
    if (rc == 0)
      memcpy(hidden_out, tmp, n * (size_t)HID * sizeof(float));
    free(tmp);
    if (rc != 0) {
      h3_st_store_free(st);
      if (error && error_size)
        snprintf(error, error_size, "final RMSNorm failed");
      return 0;
    }
  }
  h3_st_store_free(st);
  return 1;
}
