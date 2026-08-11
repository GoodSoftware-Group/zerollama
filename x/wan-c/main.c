#include "wan.h"
#include "wan_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s --ckpt-dir DIR [options]\n"
          "\n"
          "  --ckpt-dir PATH       Wan checkpoint directory\n"
          "  --uma-sock PATH       UMA broker socket (default: UMA_SOCK env)\n"
          "  --prompt TEXT         Positive prompt (required)\n"
          "  --negative-prompt T   Negative prompt\n"
          "  --width N             Output width (default 832)\n"
          "  --height N            Output height (default 480)\n"
          "  --frames N            Frame count (default 49)\n"
          "  --steps N             Diffusion steps (default 25)\n"
          "  --cfg F               CFG scale (default 5.0)\n"
          "  --shift F             Flow sigma shift (default 5.0)\n"
          "  --seed N              RNG seed (0=auto)\n"
          "  --solver unipc|dpmpp  Solver (default unipc)\n"
          "  --dtype f32|f16       Compute dtype hint (default f16)\n"
          "  --fps N               Output fps (default 16)\n"
          "  --vocab PATH          Binary umt5.vocab from export_umt5_spm.py\n"
          "  --out PATH            Output mp4 (required)\n"
          "  --validate-only       Check params and exit\n"
          "\n"
          "Env: UMA_WAN_LOCAL=1  host uma_wan_ops (no broker)\n"
          "     UMA_WAN_EXT=1    prefer EXT_CALL for LN/AdaLN/GN (needs opworker)\n"
          "     UMA_EXT_SOCK     EXT worker socket (default /tmp/uma_ext_wan.sock)\n"
          "     WAN_DIT_NO_PERSIST=1  skip F0994 block BANK (host FFN)\n"
          "     WAN_DIT_HOST_FFN=1    force host FFN (implies no FFN_GELU)\n"
          "     WAN_DIT_MIRROR=1      per-block token GET/PUT (debug)\n"
          "     WAN_DIT_QCHUNK=N      ATTN t= window rows (default 5460 if T>)\n"
          "     WAN_DIT_FFN_CHUNK=N   FFN_GELU t= window rows (default 4096)\n"
          "     WAN_VAE_NO_HEADT=1    skip broker HEADT (F1001–F1004)\n"
          "     WAN_VAE_WARM_HEADT=1  warm feat_cache on broker (F1012 mid/out;\n"
          "                          opt-in — default-on miss @832). SHUTTLE=1=old\n"
          "     WAN_VAE_NO_WARM_HEADT=1  force host warm after cold\n"
          "     WAN_VAE_NO_RESID_FUSE=1  legacy dual HEADT + host ADD\n"
          "     WAN_DIT_NO_GATED_RESID=1  legacy AFFINE+RESIDUAL (no Metal gate)\n"
          "     WAN_PROFILE=1         stage wall timers (stderr summary)\n"
          "     WAN_VAE_STAGE_PROF=1  with WAN_PROFILE: VAE tip stage map\n"
          "     WAN_BUF_STICKY=0      re-assert BUF_ALLOC every call\n",
          argv0);
}

static int parse_solver(const char *s, wan_solver_t *out) {
  if (!s || !out)
    return -1;
  if (!strcmp(s, "unipc")) {
    *out = WAN_SOLVER_UNIPC;
    return 0;
  }
  if (!strcmp(s, "dpmpp")) {
    *out = WAN_SOLVER_DPMPP;
    return 0;
  }
  return -1;
}

static int parse_dtype(const char *s, wan_dtype_t *out) {
  if (!s || !out)
    return -1;
  if (!strcmp(s, "f32")) {
    *out = WAN_DTYPE_F32;
    return 0;
  }
  if (!strcmp(s, "f16")) {
    *out = WAN_DTYPE_F16;
    return 0;
  }
  return -1;
}

int main(int argc, char **argv) {
  const wan_model_config *defs = wan_model_config_1_3b();
  wan_gen_params p = {
      .prompt = NULL,
      .negative_prompt = "",
      .width = defs->default_width,
      .height = defs->default_height,
      .frames = defs->default_frames,
      .steps = defs->default_steps,
      .cfg_scale = defs->default_cfg,
      .shift = 5.0f,
      .seed = 0,
      .solver = WAN_SOLVER_UNIPC,
      .dtype = WAN_DTYPE_F16,
      .fps = 16,
      .vocab_path = NULL,
  };

  const char *ckpt_dir = NULL;
  const char *uma_sock = NULL;
  const char *out_mp4 = NULL;
  int validate_only = 0;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "--ckpt-dir") && i + 1 < argc)
      ckpt_dir = argv[++i];
    else if (!strcmp(a, "--uma-sock") && i + 1 < argc)
      uma_sock = argv[++i];
    else if (!strcmp(a, "--prompt") && i + 1 < argc)
      p.prompt = argv[++i];
    else if (!strcmp(a, "--negative-prompt") && i + 1 < argc)
      p.negative_prompt = argv[++i];
    else if (!strcmp(a, "--width") && i + 1 < argc)
      p.width = atoi(argv[++i]);
    else if (!strcmp(a, "--height") && i + 1 < argc)
      p.height = atoi(argv[++i]);
    else if (!strcmp(a, "--frames") && i + 1 < argc)
      p.frames = atoi(argv[++i]);
    else if (!strcmp(a, "--steps") && i + 1 < argc)
      p.steps = atoi(argv[++i]);
    else if (!strcmp(a, "--cfg") && i + 1 < argc)
      p.cfg_scale = (float)atof(argv[++i]);
    else if (!strcmp(a, "--shift") && i + 1 < argc)
      p.shift = (float)atof(argv[++i]);
    else if (!strcmp(a, "--seed") && i + 1 < argc)
      p.seed = atoi(argv[++i]);
    else if (!strcmp(a, "--solver") && i + 1 < argc) {
      if (parse_solver(argv[++i], &p.solver) != 0) {
        fprintf(stderr, "bad --solver\n");
        return 2;
      }
    } else if (!strcmp(a, "--dtype") && i + 1 < argc) {
      if (parse_dtype(argv[++i], &p.dtype) != 0) {
        fprintf(stderr, "bad --dtype\n");
        return 2;
      }
    } else if (!strcmp(a, "--fps") && i + 1 < argc)
      p.fps = atoi(argv[++i]);
    else if (!strcmp(a, "--vocab") && i + 1 < argc)
      p.vocab_path = argv[++i];
    else if (!strcmp(a, "--out") && i + 1 < argc)
      out_mp4 = argv[++i];
    else if (!strcmp(a, "--validate-only"))
      validate_only = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", a);
      usage(argv[0]);
      return 2;
    }
  }

  char err[256];
  if (wan_validate_params(&p, err, sizeof(err)) != 0) {
    fprintf(stderr, "wan-c: %s\n", err);
    return 1;
  }
  if (validate_only) {
    fprintf(stderr, "wan-c: params OK\n");
    return 0;
  }

  if (!ckpt_dir || !p.prompt || !out_mp4) {
    usage(argv[0]);
    return 2;
  }

  wan_ctx *ctx = wan_ctx_open(ckpt_dir, uma_sock);
  if (!ctx)
    return 1;

  int rc = wan_generate_t2v(ctx, &p, out_mp4);
  wan_ctx_close(ctx);
  return rc == 0 ? 0 : 1;
}
