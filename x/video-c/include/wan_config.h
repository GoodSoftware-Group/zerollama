/*
 * wan_config.h — Wan 2.1 T2V 1.3B hyperparameters and validation.
 */
#pragma once

#include "wan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wan_model_config {
  int dim;
  int num_layers;
  int num_heads;
  int ffn_dim;
  int patch_t;
  int patch_h;
  int patch_w;
  int vae_stride_t;
  int vae_stride_h;
  int vae_stride_w;
  int z_channels;
  int text_dim;
  int text_ffn; /* F1020: umt5 FFN intermed dim (T5FeedForward) */
  int text_len;
  int default_width;
  int default_height;
  int default_frames;
  int default_steps;
  float default_cfg;
} wan_model_config;

const wan_model_config *wan_model_config_1_3b(void);

int wan_validate_resolution(int width, int height, int frames, char *err,
                            size_t err_n);
int wan_validate_steps_cfg(int steps, float cfg, char *err, size_t err_n);

#ifdef __cplusplus
}
#endif
