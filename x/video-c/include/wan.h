/*
 * wan.h — Pure-C Wan 2.1 T2V client (1.3B scaffold)
 *
 * Links libuma_client for GRAPH dispatch; loads weights via minimal GGUF reader.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  WAN_SOLVER_UNIPC = 0,
  WAN_SOLVER_DPMPP = 1
} wan_solver_t;

typedef enum {
  WAN_DTYPE_F32 = 0,
  WAN_DTYPE_F16 = 1
} wan_dtype_t;

typedef struct wan_gen_params {
  const char *prompt;
  const char *negative_prompt;
  int width;
  int height;
  int frames;
  int steps;
  float cfg_scale;
  float shift;
  int seed;
  wan_solver_t solver;
  wan_dtype_t dtype;
  int fps;
  const char *vocab_path;
} wan_gen_params;

typedef struct wan_ctx wan_ctx;

wan_ctx *wan_ctx_open(const char *ckpt_dir, const char *uma_sock);
void wan_ctx_close(wan_ctx *ctx);

int wan_generate_t2v(wan_ctx *ctx, const wan_gen_params *p, const char *out_mp4);
int wan_validate_params(const wan_gen_params *p, char *err, size_t err_n);

#ifdef __cplusplus
}
#endif
