#include "wan_config.h"

#include <stdio.h>
#include <string.h>

static const wan_model_config g_wan_1_3b = {
    .dim = 1536,
    .num_layers = 30,
    .num_heads = 12,
    .ffn_dim = 8960,
    .patch_t = 1,
    .patch_h = 2,
    .patch_w = 2,
    .vae_stride_t = 4,
    .vae_stride_h = 8,
    .vae_stride_w = 8,
    .z_channels = 16,
    .text_dim = 4096,
    .text_ffn = 10240,
    .text_len = 512,
    .default_width = 832,
    .default_height = 480,
    .default_frames = 49,
    .default_steps = 25,
    .default_cfg = 5.0f,
};

const wan_model_config *wan_model_config_1_3b(void) { return &g_wan_1_3b; }

int wan_validate_resolution(int width, int height, int frames, char *err,
                            size_t err_n) {
  const wan_model_config *c = wan_model_config_1_3b();
  if (width <= 0 || height <= 0) {
    snprintf(err, err_n, "width/height must be positive");
    return -1;
  }
  if (width % c->vae_stride_w != 0 || height % c->vae_stride_h != 0) {
    snprintf(err, err_n,
             "width must be multiple of %d, height multiple of %d",
             c->vae_stride_w, c->vae_stride_h);
    return -1;
  }
  if (frames < 5) {
    snprintf(err, err_n, "frames must be >= 5");
    return -1;
  }
  if (((frames - 1) % c->vae_stride_t) != 0) {
    snprintf(err, err_n, "frames-1 must be divisible by %d (got %d)",
             c->vae_stride_t, frames);
    return -1;
  }
  return 0;
}

int wan_validate_steps_cfg(int steps, float cfg, char *err, size_t err_n) {
  if (steps < 1 || steps > 100) {
    snprintf(err, err_n, "steps must be in [1,100]");
    return -1;
  }
  if (cfg < 1.0f || cfg > 30.0f) {
    snprintf(err, err_n, "cfg_scale must be in [1,30]");
    return -1;
  }
  return 0;
}

int wan_validate_params(const wan_gen_params *p, char *err, size_t err_n) {
  if (!p) {
    snprintf(err, err_n, "params is NULL");
    return -1;
  }
  if (!p->prompt || !p->prompt[0]) {
    snprintf(err, err_n, "prompt is required");
    return -1;
  }
  if (wan_validate_resolution(p->width, p->height, p->frames, err, err_n) != 0)
    return -1;
  if (wan_validate_steps_cfg(p->steps, p->cfg_scale, err, err_n) != 0)
    return -1;
  if (p->fps < 1 || p->fps > 120) {
    snprintf(err, err_n, "fps must be in [1,120]");
    return -1;
  }
  return 0;
}
