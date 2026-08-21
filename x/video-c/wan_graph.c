#include "wan_internal.h"
#include "wan_profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wan_env_local(void) {
  const char *e = getenv("UMA_WAN_LOCAL");
  return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
}

const char *wan_gemm_role(const wan_ctx *ctx) {
  (void)ctx;
  const char *e = getenv("WAN_GEMM_CPU");
  if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y'))
    return "CPU";
  return "GPU"; /* F0909 Metal dense GEMM_F16 tip */
}

int wan_env_prefer_ext(void) {
  const char *e = getenv("UMA_WAN_EXT");
  return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
}

static int resp_ok(const char *resp) {
  if (!resp)
    return 0;
  if (strncmp(resp, "OK", 2) == 0)
    return 1;
  return strstr(resp, "GEMM_F16") || strstr(resp, "LAYERNORM") ||
         strstr(resp, "AFFINE_MUL") || strstr(resp, "GROUP_NORM") ||
         strstr(resp, "EXT_CALL") || strstr(resp, "MODULATE6") ||
         strstr(resp, "ROPE3") || strstr(resp, "CONV2D") ||
         strstr(resp, "CONV_TRANSPOSE") || strstr(resp, "CT2D") ||
         strstr(resp, "CONV3D") || strstr(resp, "UNPATCHIFY") ||
         strstr(resp, "NCDHW_TOKENS") || strstr(resp, "TOK3") ||
         strstr(resp, "TOKENS_NCDHW") || strstr(resp, "NCDHW3") ||
         strstr(resp, "SINUSOID") || strstr(resp, "TIMESTEP_EMB") ||
         strstr(resp, "CONV_TRANSPOSE3D") || strstr(resp, "CT3D") ||
         strstr(resp, "ATTN_NAMED") ||
         strstr(resp, "SILU_MUL") || strstr(resp, "RESIDUAL_ADD") ||
         strstr(resp, "COPY") || strstr(resp, "C2D") || strstr(resp, "UP3");
}

int wan_submit_graph(UmaClient *c, const char *nodes) {
  char req[8192];
  char job[8300];
  char resp[512];
  char body[7800];
  char mech_body[7800];
  if (!c || !nodes || !nodes[0])
    return -1;
  double t_prof0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  /* Daemon header ends at first ';'. Smokes use "GRAPH … ; OP@…". */
  if (nodes[0] == ';')
    snprintf(body, sizeof(body), "%s", nodes);
  else
    snprintf(body, sizeof(body), "; %s", nodes);

  const char *level = getenv("WAN_GRAPH_LEVEL");
  if (!level || !level[0])
    level = "abs";
  const char *form = getenv("WAN_GRAPH_FORM");
  if (!form || !form[0])
    form = "chain";
  int ngen = 0;
  const char *ngen_e = getenv("WAN_GRAPH_NGEN");
  if (ngen_e && ngen_e[0])
    ngen = atoi(ngen_e);

  const char *nodes_use = body;
  if (strcmp(level, "mech") == 0) {
    /* F0891: mech drops abs !/? duty marks. */
    size_t j = 0;
    for (size_t i = 0; body[i] && j + 1 < sizeof(mech_body); i++) {
      if (body[i] == '!' || body[i] == '?')
        continue;
      mech_body[j++] = body[i];
    }
    mech_body[j] = '\0';
    nodes_use = mech_body;
  }

  int rc_fmt;
  if (strcmp(form, "repeat") == 0 && ngen > 0)
    rc_fmt = uma_client_format_graph_ex(req, sizeof(req), level, 1, form,
                                        nodes_use, ngen, -1, NULL);
  else
    rc_fmt = uma_client_format_graph_level(req, sizeof(req), level, 1, form,
                                           nodes_use);
  if (rc_fmt != 0) {
    fprintf(stderr, "wan-c: GRAPH format failed (level=%s form=%s)\n", level,
            form);
    return -1;
  }
  if (getenv("WAN_GRAPH_DUMP")) {
    static int dumped;
    if (!dumped) {
      dumped = 1;
      fprintf(stderr, "wan-c: GRAPH_DUMP len=%zu\n%.*s\n", strlen(req),
              (int)(strlen(req) > 4000 ? 4000 : strlen(req)), req);
    }
  }
  /* F0793: Wan DiT/VAE jobs use qos=batch so they don't hog interactive. */
  snprintf(job, sizeof(job), "qos=batch %s", req);
  uint64_t ticket = 0;
  if (uma_client_submit(c, "wan-c", job, &ticket) != 0) {
    fprintf(stderr, "wan-c: GRAPH submit failed (job_chars=%zu)\n",
            strlen(job));
    return -1;
  }
  double wait_s = 600.0;
  const char *we = getenv("WAN_GRAPH_WAIT");
  if (we && we[0])
    wait_s = atof(we);
  if (wait_s < 30.0)
    wait_s = 30.0;
  if (uma_client_wait(c, ticket, wait_s, resp, sizeof(resp)) != 0) {
    fprintf(stderr, "wan-c: GRAPH wait failed: %.200s\n", resp);
    return -1;
  }
  if (!resp_ok(resp)) {
    fprintf(stderr, "wan-c: GRAPH failed: %.200s\n", resp);
    return -1;
  }
  if (wan_profile_on())
    wan_profile_add_ms("graph", wan_profile_now_ms() - t_prof0);
  return 0;
}

int wan_probe_caps(wan_ctx *ctx) {
  if (!ctx || !ctx->uma)
    return -1;
  char info[4096];
  memset(&ctx->caps, 0, sizeof(ctx->caps));
  ctx->caps.prefer_ext = wan_env_prefer_ext();
  if (uma_client_info(ctx->uma, info, sizeof(info)) != 0) {
    if (uma_client_cmd(ctx->uma, "HELP", info, sizeof(info)) != 0) {
      fprintf(stderr, "wan-c: cannot probe broker HELP/INFO\n");
      ctx->caps.gemm_f16 = 1;
      ctx->caps.layernorm = 1;
      ctx->caps.affine = 1;
      ctx->caps.group_norm = 1;
      ctx->caps.rope3 = 1;
      ctx->caps.conv2d = 1;
      ctx->caps.ct2d = 1;
      ctx->caps.conv3d = 1;
      ctx->caps.unpatchify = 1;
      ctx->caps.attn_full = 1;
      ctx->caps.silu_mul = 1;
      ctx->caps.residual_add = 1;
      ctx->caps.tok3 = 1;
      ctx->caps.ncdhw3 = 1;
      ctx->caps.sinusoid = 1;
      ctx->caps.ct3d = 1;
      ctx->caps.channel_rms = 1;
      ctx->caps.silu = 1;
      ctx->caps.causal_pad3d = 1;
      ctx->caps.nearest = 1;
      return 0;
    }
  }
  ctx->caps.gemm_f16 = strstr(info, "GEMM_F16") != NULL;
  ctx->caps.layernorm = strstr(info, "LAYERNORM_MUL") != NULL;
  ctx->caps.affine = strstr(info, "AFFINE_MUL_ADD") != NULL ||
                     strstr(info, "MODULATE6") != NULL;
  ctx->caps.group_norm = strstr(info, "GROUP_NORM") != NULL;
  ctx->caps.rope3 = strstr(info, "ROPE3") != NULL;
  ctx->caps.conv2d = strstr(info, "CONV2D") != NULL;
  ctx->caps.ct2d = strstr(info, "CONV_TRANSPOSE2D") != NULL ||
                   strstr(info, "CT2D") != NULL;
  ctx->caps.conv3d = strstr(info, "CONV3D") != NULL;
  ctx->caps.unpatchify = strstr(info, "UNPATCHIFY3D") != NULL;
  ctx->caps.attn_full = strstr(info, "ATTN_NAMED_kind=full") != NULL ||
                        strstr(info, "graph_wan=") != NULL;
  ctx->caps.attn_tc = strstr(info, "ATTN_NAMED_tc") != NULL;
  ctx->caps.attn_bias = strstr(info, "kind=bias") != NULL ||
                        strstr(info, "kind=unscaled") != NULL ||
                        strstr(info, "|bias") != NULL ||
                        strstr(info, "graph_wan=") != NULL;
  ctx->caps.gelu_tanh_mul = strstr(info, "GELU_TANH_MUL") != NULL ||
                            strstr(info, "GELU_MUL") != NULL;
  ctx->caps.silu_mul = strstr(info, "SILU_MUL") != NULL;
  ctx->caps.residual_add = strstr(info, "RESIDUAL_ADD") != NULL;
  /* F0826: mainline layout (also listed in graph_wan=). */
  ctx->caps.tok3 = strstr(info, "NCDHW_TOKENS") != NULL ||
                   strstr(info, "TOK3") != NULL;
  ctx->caps.ncdhw3 = strstr(info, "TOKENS_NCDHW") != NULL ||
                     strstr(info, "NCDHW3") != NULL;
  ctx->caps.form_repeat = strstr(info, "graph_form=") != NULL &&
                          strstr(info, "repeat") != NULL;
  ctx->caps.mech = strstr(info, "graph_level=abs|mech") != NULL ||
                   strstr(info, "graph_mech=") != NULL;
  /* F0906 / F0905 */
  ctx->caps.sinusoid = strstr(info, "SINUSOID") != NULL ||
                       strstr(info, "TIMESTEP_EMB") != NULL;
  ctx->caps.ct3d = strstr(info, "CONV_TRANSPOSE3D") != NULL ||
                   strstr(info, "CT3D") != NULL;
  ctx->caps.gelu = strstr(info, "GELU") != NULL;
  ctx->caps.row_copy = strstr(info, "ROW_COPY") != NULL ||
                       strstr(info, "RCOPY") != NULL;
  ctx->caps.ffn_gelu = strstr(info, "FFN_GELU") != NULL;
  ctx->caps.head_rmsnorm = strstr(info, "HEAD_RMSNORM") != NULL;
  ctx->caps.channel_rms = strstr(info, "CHANNEL_RMS") != NULL ||
                          strstr(info, "RMS_CHANNEL") != NULL;
  ctx->caps.silu = strstr(info, "SILU") != NULL || strstr(info, "SWISH") != NULL;
  ctx->caps.causal_pad3d = strstr(info, "CAUSAL_PAD3D") != NULL ||
                           strstr(info, "CPAD3D") != NULL;
  ctx->caps.nearest = strstr(info, "NEAREST") != NULL;
  /* F0900: long SUBMIT/WAIT tip-plane — raise job TTL when available. */
  {
    char ttl_resp[256];
    int ttl = 3600;
    const char *e = getenv("WAN_JOB_TTL");
    if (e && e[0])
      ttl = atoi(e);
    if (ttl < 60)
      ttl = 60;
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "SET_JOB_TTL %d", ttl);
    if (uma_client_cmd(ctx->uma, cmd, ttl_resp, sizeof(ttl_resp)) == 0)
      fprintf(stderr, "wan-c: %s OK\n", cmd);
  }
  fprintf(stderr,
          "wan-c: caps gemm=%d ln=%d affine=%d gn=%d rope3=%d c2d=%d ct2d=%d "
          "c3d=%d ct3d=%d up3=%d attn_full=%d silu_mul=%d residual=%d tok3=%d "
          "ncdhw3=%d sinusoid=%d form_repeat=%d mech=%d gelu=%d row_copy=%d "
          "ffn_gelu=%d head_rms=%d ch_rms=%d silu=%d cpad3d=%d nearest=%d "
          "prefer_ext=%d gemm_role=%s\n",
          ctx->caps.gemm_f16, ctx->caps.layernorm, ctx->caps.affine,
          ctx->caps.group_norm, ctx->caps.rope3, ctx->caps.conv2d,
          ctx->caps.ct2d, ctx->caps.conv3d, ctx->caps.ct3d,
          ctx->caps.unpatchify, ctx->caps.attn_full, ctx->caps.silu_mul,
          ctx->caps.residual_add, ctx->caps.tok3, ctx->caps.ncdhw3,
          ctx->caps.sinusoid, ctx->caps.form_repeat, ctx->caps.mech,
          ctx->caps.gelu, ctx->caps.row_copy, ctx->caps.ffn_gelu,
          ctx->caps.head_rmsnorm, ctx->caps.channel_rms, ctx->caps.silu,
          ctx->caps.causal_pad3d, ctx->caps.nearest, ctx->caps.prefer_ext,
          wan_gemm_role(ctx));
  return 0;
}

static int ext_register_kind(wan_ctx *ctx, const char *kind) {
  char line[384];
  char resp[512];
  snprintf(line, sizeof(line), "EXT_REGISTER name=%s sock=%s", kind,
           ctx->ext_sock);
  if (uma_client_cmd(ctx->uma, line, resp, sizeof(resp)) != 0)
    return -1;
  return strstr(resp, "OK") ? 0 : -1;
}

int wan_ext_setup(wan_ctx *ctx) {
  if (!ctx || !ctx->uma || !ctx->caps.prefer_ext)
    return 0;
  const char *sock = getenv("UMA_EXT_SOCK");
  if (!sock || !sock[0])
    sock = "/tmp/uma_ext_wan.sock";
  snprintf(ctx->ext_sock, sizeof(ctx->ext_sock), "%s", sock);

  /* F0777–F0828 kinds (layout tip + timestep sinusoid EXT). */
  const char *kinds[] = {"EXT_LAYERNORM",
                         "EXT_AFFINE_MUL_ADD",
                         "EXT_GROUP_NORM_C16_G8",
                         "EXT_ROPE3_H1",
                         "EXT_TOK3_1_16_2_8_8",
                         "EXT_NCDHW3_1_16_2_8_8",
                         "EXT_SIN_1_16",
                         "EXT_SINUSOID_1_16"};
  int ok = 0;
  for (int i = 0; i < 8; i++) {
    if (ext_register_kind(ctx, kinds[i]) == 0)
      ok++;
  }
  ctx->caps.ext_ready = ok > 0;
  ctx->caps.rope3_ext = (ext_register_kind(ctx, "EXT_ROPE3_H1") == 0);
  ctx->caps.layout_ext =
      (ext_register_kind(ctx, "EXT_TOK3_1_16_2_8_8") == 0) ||
      (ext_register_kind(ctx, "EXT_NCDHW3_1_16_2_8_8") == 0);
  fprintf(stderr,
          "wan-c: EXT register %d/8 via %s (ready=%d rope3=%d layout=%d)\n", ok,
          ctx->ext_sock, ctx->caps.ext_ready, ctx->caps.rope3_ext,
          ctx->caps.layout_ext);
  return ctx->caps.ext_ready ? 0 : -1;
}

static int graph_ext(wan_ctx *ctx, const char *kind, const char *bx,
                     const char *by, const char *bw, const char *bgate,
                     const char *bup, int N, int D) {
  char nodes[768];
  if (!ctx || !ctx->caps.ext_ready)
    return -1;
  if (ext_register_kind(ctx, kind) != 0) {
    fprintf(stderr, "wan-c: EXT_REGISTER failed for %s\n", kind);
    return -1;
  }
  snprintf(nodes, sizeof(nodes),
           "EXT_CALL@CPU! kind=%s x=%s y=%s w=%s gate=%s up=%s N=%d D=%d ; "
           "MARK@GPU?",
           kind, bx, by, (bw && bw[0]) ? bw : "-",
           (bgate && bgate[0]) ? bgate : "-", (bup && bup[0]) ? bup : "-", N, D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_gemm_f32(wan_ctx *ctx, const char *bx, const char *by,
                       const char *bw, int M, int N, int K) {
  char nodes[512];
  if (!ctx || !ctx->uma || M < 1 || N < 1 || K < 1)
    return -1;
  const char *role = wan_gemm_role(ctx);
  /* Prefer MARK@GPU? so sticky CB does not bounce after a GPU GEMM. */
  const char *mark = (strcmp(role, "CPU") == 0) ? "CPU" : "GPU";
  snprintf(nodes, sizeof(nodes),
           "GEMM_F16@%s! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@%s?", role, bx,
           by, bw, M, N, K, mark);
  if (wan_submit_graph(ctx->uma, nodes) == 0)
    return 0;
  /* Fall back to host GEMM if Metal path rejects. */
  if (strcmp(role, "CPU") != 0) {
    snprintf(nodes, sizeof(nodes),
             "GEMM_F16@CPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@CPU?", bx, by,
             bw, M, N, K);
    return wan_submit_graph(ctx->uma, nodes);
  }
  return -1;
}

int wan_graph_layernorm(wan_ctx *ctx, const char *bx, const char *by,
                        const char *bw, int rows, int D) {
  char nodes[512];
  if (!ctx || !ctx->uma || rows < 1 || D < 1)
    return -1;
  if (ctx->caps.prefer_ext && ctx->caps.ext_ready)
    return graph_ext(ctx, "EXT_LAYERNORM", bx, by, bw, NULL, NULL, rows, D);
  if (bw && bw[0])
    snprintf(nodes, sizeof(nodes),
             "LAYERNORM_MUL@CPU! x=%s y=%s w=%s N=%d D=%d ; MARK@GPU?", bx, by,
             bw, rows, D);
  else
    snprintf(nodes, sizeof(nodes),
             "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; MARK@GPU?", bx, by, rows,
             D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_affine(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bscale, const char *bshift, int rows, int D) {
  char nodes[640];
  if (!ctx || !ctx->uma || rows < 1 || D < 1)
    return -1;
  if (ctx->caps.prefer_ext && ctx->caps.ext_ready)
    return graph_ext(ctx, "EXT_AFFINE_MUL_ADD", bx, by, NULL, bscale, bshift,
                     rows, D);
  snprintf(nodes, sizeof(nodes),
           "AFFINE_MUL_ADD@CPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; MARK@GPU?",
           bx, by, bscale, bshift, rows, D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_groupnorm(wan_ctx *ctx, const char *bx, const char *by, int N,
                        int C, int spatial, int G) {
  char nodes[512];
  if (!ctx || !ctx->uma || N < 1 || C < 1 || spatial < 1 || G < 1)
    return -1;
  if (ctx->caps.prefer_ext && ctx->caps.ext_ready) {
    /* F0778: kind EXT_GROUP_NORM_C#_G#; N=batch; D=C*spatial. */
    char kind[80];
    snprintf(kind, sizeof(kind), "EXT_GROUP_NORM_C%d_G%d", C, G);
    return graph_ext(ctx, kind, bx, by, NULL, NULL, NULL, N, C * spatial);
  }
  snprintf(nodes, sizeof(nodes),
           "GROUP_NORM@CPU! x=%s y=%s N=%d D=%d K=%d H=%d ; MARK@GPU?", bx, by,
           N, C, spatial, G);
  return wan_submit_graph(ctx->uma, nodes);
}

/* Match uma_wan_rope3_f32 HD split (complex pairs → float dims). */
static void wan_rope_hd_split(int HD, int *d0, int *d1, int *d2) {
  int c = HD / 2;
  *d0 = 2 * (c - 2 * (c / 3));
  *d1 = 2 * (c / 3);
  *d2 = 2 * (c / 3);
  if (*d0 < 2) {
    *d0 = (HD / 6) * 2;
    *d1 = (HD / 6) * 2;
    *d2 = HD - *d0 - *d1;
  }
}

int wan_rope_axis_dim(int HD) {
  if (HD < 2)
    return 0;
  int d0, d1, d2;
  wan_rope_hd_split(HD, &d0, &d1, &d2);
  if (d0 < 2)
    return HD;
  return d0 > d1 ? d0 : d1;
}

void wan_fill_rope_freqs(float *freq, int npos, int HD) {
  if (!freq || npos < 1 || HD < 2)
    return;
  /* Wan rope_params(theta=10000): complex polar on pairs. */
  int half = HD / 2;
  for (int p = 0; p < npos; p++) {
    for (int i = 0; i < half; i++) {
      float ang = (float)p * powf(10000.f, -2.f * (float)i / (float)HD);
      freq[p * HD + 2 * i] = cosf(ang);
      freq[p * HD + 2 * i + 1] = sinf(ang);
    }
  }
}

static void wan_rope_axis_apply(float *x, int HD, const float *freq, int pos) {
  int half = HD / 2;
  const float *f = freq + (size_t)pos * (size_t)HD;
  for (int i = 0; i < half; i++) {
    float cosv = f[2 * i];
    float sinv = f[2 * i + 1];
    float a = x[2 * i];
    float b = x[2 * i + 1];
    x[2 * i] = a * cosv - b * sinv;
    x[2 * i + 1] = a * sinv + b * cosv;
  }
}

int wan_rope3_tokens_grid(float *tokens, int T, int H, int HD, int grid_t,
                          int grid_h, int grid_w) {
  if (!tokens || T < 1 || H < 1 || HD < 2 || (HD % 2) != 0)
    return -1;
  if (grid_t < 1)
    grid_t = T;
  if (grid_h < 1)
    grid_h = 1;
  if (grid_w < 1)
    grid_w = 1;
  if ((size_t)grid_t * (size_t)grid_h * (size_t)grid_w != (size_t)T)
    return -1;

  int d0, d1, d2;
  wan_rope_hd_split(HD, &d0, &d1, &d2);
  float *ft = calloc((size_t)grid_t * (size_t)(d0 > 0 ? d0 : 2), sizeof(float));
  float *fh = calloc((size_t)grid_h * (size_t)(d1 > 0 ? d1 : 2), sizeof(float));
  float *fw = calloc((size_t)grid_w * (size_t)(d2 > 0 ? d2 : 2), sizeof(float));
  if (!ft || !fh || !fw) {
    free(ft);
    free(fh);
    free(fw);
    return -1;
  }
  if (d0 >= 2)
    wan_fill_rope_freqs(ft, grid_t, d0);
  if (d1 >= 2)
    wan_fill_rope_freqs(fh, grid_h, d1);
  if (d2 >= 2)
    wan_fill_rope_freqs(fw, grid_w, d2);

  /* Match ncdhw_to_tokens / Wan flatten: W fastest, then H, then T. */
  int spat = grid_h * grid_w;
  for (int idx = 0; idx < T; idx++) {
    int ow = idx % grid_w;
    int oh = (idx / grid_w) % grid_h;
    int od = spat > 0 ? (idx / spat) : 0;
    for (int h = 0; h < H; h++) {
      float *row =
          tokens + ((size_t)idx * (size_t)H + (size_t)h) * (size_t)HD;
      if (d0 >= 2)
        wan_rope_axis_apply(row, d0, ft, od);
      if (d1 >= 2)
        wan_rope_axis_apply(row + d0, d1, fh, oh);
      if (d2 >= 2)
        wan_rope_axis_apply(row + d0 + d1, d2, fw, ow);
    }
  }
  free(ft);
  free(fh);
  free(fw);
  return 0;
}

int wan_rope3_tokens(float *tokens, int T, int H, int HD) {
  return wan_rope3_tokens_grid(tokens, T, H, HD, T, 1, 1);
}

/* Compact Wan axis freqs + pad to T*HD (broker ROPE3 buf_size upper bound). */
static int rope3_put_freqs(wan_ctx *ctx, int T, int HD, int gt, int gh, int gw,
                           const char *bft, const char *bfh, const char *bfw) {
  int d0, d1, d2;
  wan_rope_hd_split(HD, &d0, &d1, &d2);
  if (d0 < 2)
    return -1;
  if (gt < 1)
    gt = T;
  if (gh < 1)
    gh = 1;
  if (gw < 1)
    gw = 1;

  /* Broker checks freq nbytes >= T*HD*4; real tables are Gt*d0 / Gh*d1 / Gw*d2. */
  size_t nfreq = (size_t)T * (size_t)HD;
  size_t fbytes = nfreq * sizeof(float);
  float *ft = calloc(nfreq, sizeof(float));
  float *fh = calloc(nfreq, sizeof(float));
  float *fw = calloc(nfreq, sizeof(float));
  if (!ft || !fh || !fw) {
    free(ft);
    free(fh);
    free(fw);
    return -1;
  }
  if (d0 >= 2)
    wan_fill_rope_freqs(ft, gt, d0);
  if (d1 >= 2)
    wan_fill_rope_freqs(fh, gh, d1);
  if (d2 >= 2)
    wan_fill_rope_freqs(fw, gw, d2);

  /*
   * Prefer PUT-overwrite on existing freq bufs. Free/realloc between Q and K
   * RoPE calls was failing the second ROPE3 under CFG (silent quality gap) and
   * churning broker slots across multi-step runs.
   * Geometry-sticky: skip host fill + BUF_PUT when T/HD/grid unchanged.
   */
  static size_t live_fbytes;
  static int live_gt, live_gh, live_gw, live_T, live_HD;
  int rc = -1;
  if (live_fbytes == fbytes && live_gt == gt && live_gh == gh &&
      live_gw == gw && live_T == T && live_HD == HD) {
    /* Bufs already hold matching freqs — still ensure alloc sticky. */
    if (uma_buf_pool_alloc(ctx->bufs, bft, fbytes) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, bfh, fbytes) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, bfw, fbytes) == 0) {
      free(ft);
      free(fh);
      free(fw);
      wan_profile_add_count("rope_freq_skip", 1);
      return 0;
    }
  }
  if (live_fbytes != fbytes) {
    (void)uma_buf_pool_free(ctx->bufs, bft);
    (void)uma_buf_pool_free(ctx->bufs, bfh);
    (void)uma_buf_pool_free(ctx->bufs, bfw);
    live_fbytes = 0;
  }
  if (uma_buf_pool_alloc(ctx->bufs, bft, fbytes) == 0 &&
      uma_buf_pool_alloc(ctx->bufs, bfh, fbytes) == 0 &&
      uma_buf_pool_alloc(ctx->bufs, bfw, fbytes) == 0 &&
      uma_buf_pool_put(ctx->bufs, bft, ft, fbytes) == 0 &&
      uma_buf_pool_put(ctx->bufs, bfh, fh, fbytes) == 0 &&
      uma_buf_pool_put(ctx->bufs, bfw, fw, fbytes) == 0) {
    live_fbytes = fbytes;
    live_gt = gt;
    live_gh = gh;
    live_gw = gw;
    live_T = T;
    live_HD = HD;
    rc = 0;
  }
  free(ft);
  free(fh);
  free(fw);
  return rc;
}

int wan_graph_rope3(wan_ctx *ctx, const char *bx, const char *by, int T, int H,
                    int HD) {
  if (!ctx || !bx || !by || T < 1 || H < 1 || HD < 2 || (HD % 2) != 0)
    return -1;

  size_t nbytes = (size_t)T * (size_t)H * (size_t)HD * sizeof(float);
  const char *bft = "x_rope_ft";
  const char *bfh = "x_rope_fh";
  const char *bfw = "x_rope_fw";

  int have_grid =
      (ctx->gen_tp > 0 && ctx->gen_hp > 0 && ctx->gen_wp > 0 &&
       (size_t)ctx->gen_tp * (size_t)ctx->gen_hp * (size_t)ctx->gen_wp ==
           (size_t)T);
  int gt = have_grid ? ctx->gen_tp : T;
  int gh = have_grid ? ctx->gen_hp : 1;
  int gw = have_grid ? ctx->gen_wp : 1;

  /* F0934: in-daemon ROPE3 with compact axis freqs + Gt/Gh/Gw. */
  if (ctx->uma && ctx->bufs && ctx->caps.rope3 &&
      !(ctx->caps.prefer_ext && ctx->caps.ext_ready)) {
    if (rope3_put_freqs(ctx, T, HD, gt, gh, gw, bft, bfh, bfw) != 0)
      return -1;
    char nodes[768];
    snprintf(nodes, sizeof(nodes),
             "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
             "Gt=%d Gh=%d Gw=%d ; MARK@GPU?",
             bx, by, bft, bfh, bfw, T, H, HD, gt, gh, gw);
    int rc = wan_submit_graph(ctx->uma, nodes);
    if (rc == 0 && have_grid) {
      static int logged_grid;
      if (!logged_grid) {
        fprintf(stderr, "wan-c: DiT RoPE3D grid t×h×w=%d×%d×%d (broker freqs)\n",
                gt, gh, gw);
        logged_grid = 1;
      }
    }
    return rc;
  }

  if (ctx->uma && ctx->bufs && ctx->caps.prefer_ext && ctx->caps.ext_ready) {
    if (rope3_put_freqs(ctx, T, HD, gt, gh, gw, bft, bfh, bfw) != 0)
      return -1;
    char kind[96];
    snprintf(kind, sizeof(kind), "EXT_ROPE3_H%d_Gt%d_Gh%d_Gw%d", H, gt, gh, gw);
    if (graph_ext(ctx, kind, bx, by, bft, bfh, bfw, T * H, HD) != 0)
      return -1;
    ctx->caps.rope3_ext = 1;
    if (have_grid) {
      static int logged_grid_ext;
      if (!logged_grid_ext) {
        fprintf(stderr, "wan-c: DiT RoPE3D grid t×h×w=%d×%d×%d (ext freqs)\n",
                gt, gh, gw);
        logged_grid_ext = 1;
      }
    }
    return 0;
  }

  /* Host RoPE: pull bx → rotate → put by. */
  if (!ctx->uma || !ctx->bufs)
    return -1;
  char resp[512];
  size_t got = 0;
  float *tok = calloc((size_t)T * (size_t)H * (size_t)HD, sizeof(float));
  if (!tok)
    return -1;
  if (uma_client_buf_get(ctx->uma, bx, tok, nbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != nbytes) {
    free(tok);
    return -1;
  }
  if (wan_rope3_tokens_grid(tok, T, H, HD, gt, gh, gw) != 0 ||
      uma_buf_pool_put(ctx->bufs, by, tok, nbytes) != 0) {
    free(tok);
    return -1;
  }
  free(tok);
  if (have_grid) {
    static int logged_grid_host;
    if (!logged_grid_host) {
      fprintf(stderr, "wan-c: DiT RoPE3D grid t×h×w=%d×%d×%d (host)\n", gt, gh,
              gw);
      logged_grid_host = 1;
    }
  }
  return 0;
}

/* Speed-gap: one GRAPH for Q and K RoPE (shared freqs PUT). */
int wan_rope3_ensure_freqs(wan_ctx *ctx, int T, int HD, int *gt, int *gh,
                           int *gw) {
  if (!ctx || !ctx->bufs || T < 1 || HD < 2 || (HD % 2) != 0)
    return -1;
  const char *bft = "x_rope_ft";
  const char *bfh = "x_rope_fh";
  const char *bfw = "x_rope_fw";
  int have_grid =
      (ctx->gen_tp > 0 && ctx->gen_hp > 0 && ctx->gen_wp > 0 &&
       (size_t)ctx->gen_tp * (size_t)ctx->gen_hp * (size_t)ctx->gen_wp ==
           (size_t)T);
  int g_t = have_grid ? ctx->gen_tp : T;
  int g_h = have_grid ? ctx->gen_hp : 1;
  int g_w = have_grid ? ctx->gen_wp : 1;
  if (rope3_put_freqs(ctx, T, HD, g_t, g_h, g_w, bft, bfh, bfw) != 0)
    return -1;
  if (gt)
    *gt = g_t;
  if (gh)
    *gh = g_h;
  if (gw)
    *gw = g_w;
  return 0;
}

int wan_graph_rope3_qk(wan_ctx *ctx, const char *bq, const char *bqr,
                       const char *bk, const char *bkr, int T, int H, int HD) {
  if (!ctx || !bq || !bqr || !bk || !bkr || T < 1 || H < 1 || HD < 2 ||
      (HD % 2) != 0)
    return -1;
  if (!ctx->uma || !ctx->bufs || !ctx->caps.rope3 ||
      (ctx->caps.prefer_ext && ctx->caps.ext_ready))
    return -1;
  const char *bft = "x_rope_ft";
  const char *bfh = "x_rope_fh";
  const char *bfw = "x_rope_fw";
  int gt, gh, gw;
  if (wan_rope3_ensure_freqs(ctx, T, HD, &gt, &gh, &gw) != 0)
    return -1;
  char nodes[1024];
  int n = snprintf(
      nodes, sizeof(nodes),
      "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
      "Gt=%d Gh=%d Gw=%d ; "
      "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
      "Gt=%d Gh=%d Gw=%d ; MARK@GPU?",
      bq, bqr, bft, bfh, bfw, T, H, HD, gt, gh, gw, bk, bkr, bft, bfh, bfw, T, H,
      HD, gt, gh, gw);
  if (n < 0 || (size_t)n >= sizeof(nodes))
    return -1;
  int rc = wan_submit_graph(ctx->uma, nodes);
  if (rc == 0 && gt > 0 &&
      (ctx->gen_tp > 0 && ctx->gen_hp > 0 && ctx->gen_wp > 0 &&
       (size_t)ctx->gen_tp * (size_t)ctx->gen_hp * (size_t)ctx->gen_wp ==
           (size_t)T)) {
    static int logged_grid;
    if (!logged_grid) {
      fprintf(stderr,
              "wan-c: DiT RoPE3D grid t×h×w=%d×%d×%d (broker freqs, Q+K)\n", gt,
              gh, gw);
      logged_grid = 1;
    }
  }
  return rc;
}

int wan_graph_attn_full(wan_ctx *ctx, const char *bq, const char *bk,
                        const char *bv, const char *bout, int T, int Tk, int H,
                        int KV, int HD) {
  return wan_graph_attn_full_row(ctx, bq, bk, bv, bout, T, Tk, H, KV, HD, -1);
}

/* F0945 umt5: ATTN_NAMED kind=bias — no 1/√HD, per-head [H,T,Tk] bias mid=. */
int wan_graph_attn_bias(wan_ctx *ctx, const char *bq, const char *bk,
                        const char *bv, const char *bbias, const char *bout,
                        int T, int Tk, int H, int KV, int HD) {
  char nodes[640];
  if (!ctx || !ctx->uma || !bq || !bk || !bv || !bbias || !bout || T < 1 ||
      Tk < 1 || H < 1 || KV < 1 || HD < 1)
    return -1;
  if (!ctx->caps.attn_bias)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "ATTN_NAMED@GPU! q=%s k=%s v=%s mid=%s out=%s B=1 T=%d Tk=%d H=%d "
           "KV=%d HD=%d kind=bias ; MARK@GPU?",
           bq, bk, bv, bbias, bout, T, Tk, H, KV, HD);
  return wan_submit_graph(ctx->uma, nodes);
}

/* F0987 gated GELU(tanh): y = gelu(gate) * up (T5FeedForward). */
int wan_graph_gelu_tanh_mul(wan_ctx *ctx, const char *bgate, const char *bup,
                            const char *by, int D) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bgate || !bup || !by || D < 1)
    return -1;
  if (!ctx->caps.gelu_tanh_mul)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "GELU_TANH_MUL@CPU! gate=%s up=%s y=%s D=%d ; MARK@GPU?", bgate, bup,
           by, D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_attn_full_row(wan_ctx *ctx, const char *bq, const char *bk,
                            const char *bv, const char *bout, int T, int Tk,
                            int H, int KV, int HD, int t_row) {
  char nodes[576];
  if (!ctx || !ctx->uma || !bq || !bk || !bv || !bout || T < 1 || Tk < 1 ||
      H < 1 || KV < 1 || HD < 1)
    return -1;
  if (!ctx->caps.attn_full)
    return -1;
  /* F1156: BF16 TensorOps flash when the daemon advertises it. Daemon falls
   * through to the standard path when ineligible (bias / HD>128). */
  const char *tc = ctx->caps.attn_tc ? " tc=1" : "";
  if (t_row >= 0)
    snprintf(nodes, sizeof(nodes),
             "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
             "HD=%d kind=full%s t=%d ; MARK@GPU?",
             bq, bk, bv, bout, T, Tk, H, KV, HD, tc, t_row);
  else
    snprintf(nodes, sizeof(nodes),
             "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
             "HD=%d kind=full%s ; MARK@GPU?",
             bq, bk, bv, bout, T, Tk, H, KV, HD, tc);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_silu_mul(wan_ctx *ctx, const char *bgate, const char *bup,
                       const char *by, int D) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bgate || !bup || !by || D < 1)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "SILU_MUL@GPU! gate=%s up=%s y=%s D=%d ; MARK@GPU?", bgate, bup, by,
           D);
  return wan_submit_graph(ctx->uma, nodes);
}

/* Wan DiT FFN: GELU(tanh) — never FFN_SILU. */
int wan_graph_gelu(wan_ctx *ctx, const char *bx, const char *by, int D) {
  char nodes[320];
  if (!ctx || !ctx->uma || !bx || !by || D < 1)
    return -1;
  snprintf(nodes, sizeof(nodes), "GELU_TANH@CPU! x=%s y=%s D=%d ; MARK@CPU?", bx,
           by, D);
  return wan_submit_graph(ctx->uma, nodes);
}

/* F0993: dense FFN_GELU with optional row window t=. */
int wan_graph_ffn_gelu(wan_ctx *ctx, const char *bx, const char *by,
                       const char *bwu, const char *bwd, const char *bmid,
                       int M, int D, int ffn, int t_row) {
  char nodes[512];
  if (!ctx || !ctx->uma || !bx || !by || !bwu || !bwd || !bmid || M < 1 ||
      D < 1 || ffn < 1)
    return -1;
  if (!ctx->caps.ffn_gelu)
    return -1;
  if (t_row >= 0)
    snprintf(nodes, sizeof(nodes),
             "FFN_GELU@GPU! x=%s y=%s wu=%s wd=%s mid=%s M=%d D=%d ffn=%d t=%d "
             "; MARK@GPU?",
             bx, by, bwu, bwd, bmid, M, D, ffn, t_row);
  else
    snprintf(nodes, sizeof(nodes),
             "FFN_GELU@GPU! x=%s y=%s wu=%s wd=%s mid=%s M=%d D=%d ffn=%d ; "
             "MARK@GPU?",
             bx, by, bwu, bwd, bmid, M, D, ffn);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_head_rmsnorm(wan_ctx *ctx, const char *bx, const char *bw, int H,
                           int HD) {
  char nodes[320];
  if (!ctx || !ctx->uma || !bx || !bw || H < 1 || HD < 1)
    return -1;
  if (!ctx->caps.head_rmsnorm)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; MARK@GPU?", bx, bw, H, HD);
  return wan_submit_graph(ctx->uma, nodes);
}

/* ROW_COPY: dst[dst_row:dst_row+N,:] ← src[src_row:src_row+N,:] */
int wan_graph_row_copy(wan_ctx *ctx, const char *bx, const char *by, int N,
                       int D, int src_row, int dst_row) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bx || !by || N < 1 || D < 1 || src_row < 0 ||
      dst_row < 0)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "ROW_COPY@CPU! x=%s y=%s N=%d D=%d t=%d Tk=%d ; MARK@CPU?", bx, by, N,
           D, src_row, dst_row);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_residual_add(wan_ctx *ctx, const char *by, const char *bx,
                           int D) {
  char nodes[320];
  if (!ctx || !ctx->uma || !by || !bx || D < 1)
    return -1;
  snprintf(nodes, sizeof(nodes), "RESIDUAL_ADD@GPU! y=%s x=%s D=%d ; MARK@GPU?",
           by, bx, D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_copy(wan_ctx *ctx, const char *by, const char *bx, int D) {
  char nodes[320];
  if (!ctx || !ctx->uma || !by || !bx || D < 1)
    return -1;
  snprintf(nodes, sizeof(nodes), "COPY@GPU! y=%s x=%s D=%d ; MARK@GPU?", by, bx,
           D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_dit_ln_affine_gemm(wan_ctx *ctx, const char *ba, const char *bb,
                                 const char *bw, const char *bs, const char *bt,
                                 int rows, int D) {
  char nodes[1024];
  if (!ctx || !ctx->uma)
    return -1;
  if (!(ctx->caps.prefer_ext && ctx->caps.ext_ready)) {
    const char *gr = wan_gemm_role(ctx);
    snprintf(nodes, sizeof(nodes),
             "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; "
             "AFFINE_MUL_ADD@CPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
             "GEMM_F16@%s! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
             ba, bb, rows, D, bb, ba, bs, bt, rows, D, gr, ba, bb, bw, rows, D,
             D);
    return wan_submit_graph(ctx->uma, nodes);
  }
  if (wan_graph_layernorm(ctx, ba, bb, NULL, rows, D) != 0)
    return -1;
  if (wan_graph_affine(ctx, bb, ba, bs, bt, rows, D) != 0)
    return -1;
  return wan_graph_gemm_f32(ctx, ba, bb, bw, rows, D, D);
}

/* F0784 / F0779: CONV2D k_ops, else EXT_C2D, else host. */
int wan_graph_conv2d(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bw, const char *bbias, int N, int Cin, int Hin,
                     int Win, int Cout, int KH, int KW, int stride, int pad) {
  if (!ctx || N < 1 || Cin < 1 || Hin < 1 || Win < 1 || Cout < 1 || KH < 1 ||
      KW < 1)
    return -1;
  if (stride < 1)
    stride = 1;

  size_t xin = (size_t)N * (size_t)Cin * (size_t)Hin * (size_t)Win;
  size_t wne = (size_t)Cout * (size_t)Cin * (size_t)KH * (size_t)KW;
  int hout = (Hin + 2 * pad - KH) / stride + 1;
  int wout = (Win + 2 * pad - KW) / stride + 1;
  size_t yne = (size_t)N * (size_t)Cout * (size_t)hout * (size_t)wout;

  if (ctx->uma && ctx->bufs && ctx->caps.conv2d &&
      !(ctx->caps.prefer_ext && ctx->caps.ext_ready)) {
    char nodes[768];
    if (bbias && bbias[0])
      snprintf(nodes, sizeof(nodes),
               "CONV2D@CPU! x=%s y=%s w=%s gate=%s N=%d D=%d H=%d T=%d V=%d "
               "K=%d ffn=%d HD=%d KV=%d ; MARK@GPU?",
               bx, by, bw, bbias, N, Cin, Hin, Win, Cout, KH, KW, stride, pad);
    else
      snprintf(nodes, sizeof(nodes),
               "CONV2D@CPU! x=%s y=%s w=%s N=%d D=%d H=%d T=%d V=%d K=%d ffn=%d "
               "HD=%d KV=%d ; MARK@GPU?",
               bx, by, bw, N, Cin, Hin, Win, Cout, KH, KW, stride, pad);
    return wan_submit_graph(ctx->uma, nodes);
  }

  if (ctx->uma && ctx->bufs && ctx->caps.prefer_ext && ctx->caps.ext_ready &&
      wne > 0 && (xin % wne) == 0 && yne <= xin) {
    char kind[96];
    snprintf(kind, sizeof(kind), "EXT_C2D_%d_%d_%d_%d_%d_%d_%d_%d_%d", N, Cin,
             Hin, Win, Cout, KH, KW, stride, pad);
    if (strlen(kind) >= 64)
      goto host_c2d;
    int Nb = (int)(xin / wne);
    int Db = (int)wne;
    return graph_ext(ctx, kind, bx, by, bw, bbias, NULL, Nb, Db);
  }

host_c2d:
  /* Host path: get x/w[/b], conv, put y. */
  if (!ctx->uma || !ctx->bufs)
    return -1;
  {
    char resp[512];
    size_t got = 0;
    float *x = calloc(xin, sizeof(float));
    float *w = calloc(wne, sizeof(float));
    float *b = bbias ? calloc((size_t)Cout, sizeof(float)) : NULL;
    float *y = calloc(yne > xin ? yne : xin, sizeof(float));
    if (!x || !w || !y || (bbias && !b)) {
      free(x);
      free(w);
      free(b);
      free(y);
      return -1;
    }
    if (uma_client_buf_get(ctx->uma, bx, x, xin * 4, &got, resp, sizeof(resp)) !=
            0 ||
        got != xin * 4 ||
        uma_client_buf_get(ctx->uma, bw, w, wne * 4, &got, resp, sizeof(resp)) !=
            0 ||
        got != wne * 4) {
      free(x);
      free(w);
      free(b);
      free(y);
      return -1;
    }
    if (bbias && b) {
      if (uma_client_buf_get(ctx->uma, bbias, b, (size_t)Cout * 4, &got, resp,
                             sizeof(resp)) != 0)
        memset(b, 0, (size_t)Cout * sizeof(float));
    }
    uma_wan_conv2d_f32(y, x, w, b, N, Cin, Hin, Win, Cout, KH, KW, stride, pad);
    size_t out_b = (yne <= xin ? xin : yne) * 4;
    int rc = uma_buf_pool_put(ctx->bufs, by, y, out_b);
    free(x);
    free(w);
    free(b);
    free(y);
    return rc;
  }
}

int wan_graph_ct2d(wan_ctx *ctx, const char *bx, const char *by, const char *bw,
                   const char *bbias, int N, int Cin, int Hin, int Win,
                   int Cout, int KH, int KW, int stride, int pad, int out_pad) {
  if (!ctx || !ctx->uma || N < 1 || Cin < 1 || Hin < 1 || Win < 1 || Cout < 1 ||
      KH < 1 || KW < 1)
    return -1;
  if (stride < 1)
    stride = 1;
  if (pad < 0)
    pad = 0;
  if (out_pad < 0)
    out_pad = 0;
  if (!ctx->caps.ct2d)
    return -1;

  int hout = (Hin - 1) * stride - 2 * pad + KH + out_pad;
  int wout = (Win - 1) * stride - 2 * pad + KW + out_pad;
  if (hout < 1 || wout < 1)
    return -1;

  size_t xin = (size_t)N * (size_t)Cin * (size_t)Hin * (size_t)Win;
  size_t wne = (size_t)Cin * (size_t)Cout * (size_t)KH * (size_t)KW;
  size_t yne = (size_t)N * (size_t)Cout * (size_t)hout * (size_t)wout;

  char nodes[768];
  if (bbias && bbias[0])
    snprintf(nodes, sizeof(nodes),
             "CONV_TRANSPOSE2D@CPU! x=%s y=%s w=%s gate=%s N=%d D=%d H=%d T=%d "
             "V=%d K=%d ffn=%d HD=%d KV=%d M=%d ; MARK@CPU?",
             bx, by, bw, bbias, N, Cin, Hin, Win, Cout, KH, KW, stride, pad,
             out_pad);
  else
    snprintf(nodes, sizeof(nodes),
             "CONV_TRANSPOSE2D@CPU! x=%s y=%s w=%s N=%d D=%d H=%d T=%d V=%d "
             "K=%d ffn=%d HD=%d KV=%d M=%d ; MARK@CPU?",
             bx, by, bw, N, Cin, Hin, Win, Cout, KH, KW, stride, pad, out_pad);
  if (wan_submit_graph(ctx->uma, nodes) == 0)
    return 0;

  /* Host fallback rematch path. */
  if (!ctx->bufs)
    return -1;
  char resp[512];
  size_t got = 0;
  float *x = calloc(xin, sizeof(float));
  float *w = calloc(wne, sizeof(float));
  float *b = bbias ? calloc((size_t)Cout, sizeof(float)) : NULL;
  float *y = calloc(yne, sizeof(float));
  if (!x || !w || !y || (bbias && !b)) {
    free(x);
    free(w);
    free(b);
    free(y);
    return -1;
  }
  if (uma_client_buf_get(ctx->uma, bx, x, xin * 4, &got, resp, sizeof(resp)) !=
          0 ||
      got != xin * 4 ||
      uma_client_buf_get(ctx->uma, bw, w, wne * 4, &got, resp, sizeof(resp)) !=
          0 ||
      got != wne * 4) {
    free(x);
    free(w);
    free(b);
    free(y);
    return -1;
  }
  if (bbias && b &&
      uma_client_buf_get(ctx->uma, bbias, b, (size_t)Cout * 4, &got, resp,
                         sizeof(resp)) != 0)
    memset(b, 0, (size_t)Cout * sizeof(float));
  uma_wan_conv_transpose2d_f32(y, x, w, b, N, Cin, Hin, Win, Cout, KH, KW,
                               stride, pad, out_pad);
  int rc = uma_buf_pool_alloc(ctx->bufs, by, yne * 4) ||
           uma_buf_pool_put(ctx->bufs, by, y, yne * 4);
  free(x);
  free(w);
  free(b);
  free(y);
  return rc;
}

int wan_graph_tok3(wan_ctx *ctx, const char *bx, const char *by,
                   const char *kind) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bx || !by || !kind || !kind[0])
    return -1;
  if (!ctx->caps.tok3)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "NCDHW_TOKENS@CPU! x=%s y=%s kind=%s ; MARK@CPU?", bx, by, kind);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_ncdhw3(wan_ctx *ctx, const char *bx, const char *by,
                     const char *kind) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bx || !by || !kind || !kind[0])
    return -1;
  if (!ctx->caps.ncdhw3)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "TOKENS_NCDHW@CPU! x=%s y=%s kind=%s ; MARK@CPU?", bx, by, kind);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_conv3d(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bw, const char *kind) {
  char nodes[512];
  if (!ctx || !ctx->uma || !bx || !by || !bw || !kind || !kind[0])
    return -1;
  if (!ctx->caps.conv3d)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "CONV3D@CPU! x=%s y=%s w=%s kind=%s ; MARK@CPU?", bx, by, bw, kind);
  return wan_submit_graph(ctx->uma, nodes);
}

/* F1001: CHANNEL_RMS→SILU→CAUSAL_PAD3D(pd=ph=pw=1)→CONV3D k=3 pad0 (+bias). */
int wan_graph_vae_headt(wan_ctx *ctx, const char *bx, const char *by,
                        const char *bw, const char *bgamma, const char *bbias,
                        const char *brms, const char *bsil, const char *bpad,
                        int Cin, int Cout, int T, int H, int W) {
  return wan_graph_vae_headt_cache(ctx, bx, by, bw, bgamma, bbias, brms, bsil,
                                   bpad, NULL, 0, Cin, Cout, T, H, W);
}

/* F1012: same HEADT with optional sticky feat_cache on CAUSAL_PAD3D. */
int wan_graph_vae_headt_cache(wan_ctx *ctx, const char *bx, const char *by,
                              const char *bw, const char *bgamma,
                              const char *bbias, const char *brms,
                              const char *bsil, const char *bpad,
                              const char *bcache, int cache_t, int Cin,
                              int Cout, int T, int H, int W) {
  char nodes[1280];
  if (!ctx || !ctx->uma || !bx || !by || !bw || !bgamma || !brms || !bsil ||
      !bpad || Cin < 1 || Cout < 1 || T < 1 || H < 1 || W < 1)
    return -1;
  if (!ctx->caps.channel_rms || !ctx->caps.silu || !ctx->caps.causal_pad3d ||
      !ctx->caps.conv3d)
    return -1;
  if (cache_t < 0)
    cache_t = 0;
  if (cache_t > 2)
    cache_t = 2;
  int Dp = T + 2, Hp = H + 2, Wp = W + 2;
  size_t nd = (size_t)Cin * (size_t)T * (size_t)H * (size_t)W;
  int n;
  char pad_extra[96];
  pad_extra[0] = 0;
  if (bcache && bcache[0])
    snprintf(pad_extra, sizeof(pad_extra), " mid=%s out=%s t=%d", bcache,
             bcache, cache_t);
  if (bbias && bbias[0]) {
    n = snprintf(
        nodes, sizeof(nodes),
        "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_%d_%d_%d ; "
        "SILU@CPU! x=%s y=%s D=%d ; "
        "CAUSAL_PAD3D@CPU! x=%s y=%s%s kind=1_%d_%d_%d_%d_1_1_1 ; "
        "CONV3D@CPU! x=%s y=%s w=%s gate=%s "
        "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_0_0 ; MARK@CPU?",
        bx, brms, bgamma, Cin, T, H, W, brms, bsil, (int)nd, bsil, bpad,
        pad_extra, Cin, T, H, W, bpad, by, bw, bbias, Cin, Dp, Hp, Wp, Cout);
  } else {
    n = snprintf(
        nodes, sizeof(nodes),
        "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_%d_%d_%d ; "
        "SILU@CPU! x=%s y=%s D=%d ; "
        "CAUSAL_PAD3D@CPU! x=%s y=%s%s kind=1_%d_%d_%d_%d_1_1_1 ; "
        "CONV3D@CPU! x=%s y=%s w=%s "
        "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_0_0 ; MARK@CPU?",
        bx, brms, bgamma, Cin, T, H, W, brms, bsil, (int)nd, bsil, bpad,
        pad_extra, Cin, T, H, W, bpad, by, bw, Cin, Dp, Hp, Wp, Cout);
  }
  if (n < 0 || (size_t)n >= sizeof(nodes))
    return -1;
  return wan_submit_graph(ctx->uma, nodes);
}

/* Append one CHANNEL_RMS→SILU→CAUSAL_PAD3D→CONV3D segment (no MARK). */
static int vae_headt_seg(char *dst, size_t n, int *off, const char *bx,
                         const char *by, const char *bw, const char *bgamma,
                         const char *bbias, const char *brms, const char *bsil,
                         const char *bpad, const char *bcache, int cache_t,
                         int Cin, int Cout, int T, int H, int W) {
  if (!dst || !off || *off < 0 || Cin < 1 || Cout < 1 || T < 1 || H < 1 || W < 1)
    return -1;
  if (cache_t < 0)
    cache_t = 0;
  if (cache_t > 2)
    cache_t = 2;
  int Dp = T + 2, Hp = H + 2, Wp = W + 2;
  size_t nd = (size_t)Cin * (size_t)T * (size_t)H * (size_t)W;
  char pad_extra[96];
  pad_extra[0] = 0;
  if (bcache && bcache[0])
    snprintf(pad_extra, sizeof(pad_extra), " mid=%s out=%s t=%d", bcache,
             bcache, cache_t);
  int k;
  if (bbias && bbias[0])
    k = snprintf(
        dst + *off, n - (size_t)*off,
        "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_%d_%d_%d ; "
        "SILU@CPU! x=%s y=%s D=%d ; "
        "CAUSAL_PAD3D@CPU! x=%s y=%s%s kind=1_%d_%d_%d_%d_1_1_1 ; "
        "CONV3D@CPU! x=%s y=%s w=%s gate=%s "
        "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_0_0 ; ",
        bx, brms, bgamma, Cin, T, H, W, brms, bsil, (int)nd, bsil, bpad,
        pad_extra, Cin, T, H, W, bpad, by, bw, bbias, Cin, Dp, Hp, Wp, Cout);
  else
    k = snprintf(
        dst + *off, n - (size_t)*off,
        "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_%d_%d_%d ; "
        "SILU@CPU! x=%s y=%s D=%d ; "
        "CAUSAL_PAD3D@CPU! x=%s y=%s%s kind=1_%d_%d_%d_%d_1_1_1 ; "
        "CONV3D@CPU! x=%s y=%s w=%s "
        "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_0_0 ; ",
        bx, brms, bgamma, Cin, T, H, W, brms, bsil, (int)nd, bsil, bpad,
        pad_extra, Cin, T, H, W, bpad, by, bw, Cin, Dp, Hp, Wp, Cout);
  if (k < 0 || (size_t)k >= n - (size_t)*off)
    return -1;
  *off += k;
  return 0;
}

/* Brick 7: dual HEADT (+ optional resid) one GRAPH — same Cin==Cout. */
int wan_graph_vae_resblock_dual_headt(
    wan_ctx *ctx, const char *bx, const char *bmid, const char *by,
    const char *bw1, const char *bg1, const char *bb1, const char *brms1,
    const char *bsil1, const char *bpad1, const char *bcache1, int cache_t1,
    const char *bw2, const char *bg2, const char *bb2, const char *brms2,
    const char *bsil2, const char *bpad2, const char *bcache2, int cache_t2,
    int C, int T, int H, int W, int add_resid) {
  char nodes[2560];
  if (!ctx || !ctx->uma || !bx || !bmid || !by || !bw1 || !bg1 || !brms1 ||
      !bsil1 || !bpad1 || !bw2 || !bg2 || !brms2 || !bsil2 || !bpad2 || C < 1 ||
      T < 1 || H < 1 || W < 1)
    return -1;
  if (!ctx->caps.channel_rms || !ctx->caps.silu || !ctx->caps.causal_pad3d ||
      !ctx->caps.conv3d)
    return -1;
  if (add_resid && !ctx->caps.residual_add)
    return -1;
  int off = 0;
  if (vae_headt_seg(nodes, sizeof(nodes), &off, bx, bmid, bw1, bg1, bb1, brms1,
                    bsil1, bpad1, bcache1, cache_t1, C, C, T, H, W) != 0)
    return -1;
  if (vae_headt_seg(nodes, sizeof(nodes), &off, bmid, by, bw2, bg2, bb2, brms2,
                    bsil2, bpad2, bcache2, cache_t2, C, C, T, H, W) != 0)
    return -1;
  int k;
  if (add_resid) {
    size_t flat = (size_t)C * (size_t)T * (size_t)H * (size_t)W;
    k = snprintf(nodes + off, sizeof(nodes) - (size_t)off,
                 "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?", by, bx,
                 (int)flat);
  } else {
    k = snprintf(nodes + off, sizeof(nodes) - (size_t)off, "MARK@CPU?");
  }
  if (k < 0 || (size_t)k >= sizeof(nodes) - (size_t)off)
    return -1;
  return wan_submit_graph(ctx->uma, nodes);
}

/* F1001 resample tip: NEAREST×2 then CONV2D k=3 pad=1. */
int wan_graph_vae_nearest_conv2d(wan_ctx *ctx, const char *blo, const char *bhi,
                                 const char *by, const char *bw,
                                 const char *bbias, int Cin, int Cout, int H,
                                 int W) {
  char nodes[768];
  if (!ctx || !ctx->uma || !blo || !bhi || !by || !bw || Cin < 1 || Cout < 1 ||
      H < 1 || W < 1)
    return -1;
  if (!ctx->caps.nearest || !ctx->caps.conv2d)
    return -1;
  int H2 = H * 2, W2 = W * 2;
  int n;
  if (bbias && bbias[0]) {
    n = snprintf(nodes, sizeof(nodes),
                 "NEAREST@CPU! x=%s y=%s kind=1_%d_%d_%d_2_2 ; "
                 "CONV2D@CPU! x=%s y=%s w=%s gate=%s "
                 "N=1 D=%d H=%d T=%d V=%d K=%d ffn=%d HD=1 KV=1 ; MARK@CPU?",
                 blo, bhi, Cin, H, W, bhi, by, bw, bbias, Cin, H2, W2, Cout, 3,
                 3);
  } else {
    n = snprintf(nodes, sizeof(nodes),
                 "NEAREST@CPU! x=%s y=%s kind=1_%d_%d_%d_2_2 ; "
                 "CONV2D@CPU! x=%s y=%s w=%s "
                 "N=1 D=%d H=%d T=%d V=%d K=%d ffn=%d HD=1 KV=1 ; MARK@CPU?",
                 blo, bhi, Cin, H, W, bhi, by, bw, Cin, H2, W2, Cout, 3, 3);
  }
  if (n < 0 || (size_t)n >= sizeof(nodes))
    return -1;
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_ct3d(wan_ctx *ctx, const char *bx, const char *by, const char *bw,
                   const char *kind) {
  char nodes[640];
  if (!ctx || !ctx->uma || !bx || !by || !bw || !kind || !kind[0])
    return -1;
  if (!ctx->caps.ct3d)
    return -1;
  snprintf(nodes, sizeof(nodes),
           "CONV_TRANSPOSE3D@CPU! x=%s y=%s w=%s kind=%s ; MARK@CPU?", bx, by,
           bw, kind);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_sinusoid(wan_ctx *ctx, const char *bt, const char *by, int N,
                       int D) {
  char nodes[384];
  if (!ctx || !ctx->uma || !bt || !by || N < 1 || D < 2 || (D % 2) != 0)
    return -1;
  if (!ctx->caps.sinusoid)
    return -1;
  /* F0906: prefer @GPU (Metal tip F0907); CPU also accepted. */
  snprintf(nodes, sizeof(nodes),
           "SINUSOID@GPU! x=%s y=%s N=%d D=%d ; MARK@CPU?", bt, by, N, D);
  if (wan_submit_graph(ctx->uma, nodes) == 0)
    return 0;
  snprintf(nodes, sizeof(nodes),
           "TIMESTEP_EMB@CPU! x=%s y=%s N=%d D=%d ; MARK@CPU?", bt, by, N, D);
  return wan_submit_graph(ctx->uma, nodes);
}

int wan_graph_unpatchify3d(wan_ctx *ctx, const char *bx, const char *by, int B,
                           int C, int T, int H, int W, int pt, int ph, int pw) {
  if (!ctx || B < 1 || C < 1 || T < 1 || H < 1 || W < 1 || pt < 1 || ph < 1 ||
      pw < 1)
    return -1;
  if ((T % pt) || (H % ph) || (W % pw))
    return -1;

  size_t ne = (size_t)B * (size_t)C * (size_t)T * (size_t)H * (size_t)W;
  size_t nbytes = ne * sizeof(float);

  if (ctx->uma && ctx->bufs && ctx->caps.unpatchify &&
      !(ctx->caps.prefer_ext && ctx->caps.ext_ready)) {
    char nodes[384];
    snprintf(nodes, sizeof(nodes),
             "UNPATCHIFY3D@CPU! x=%s y=%s kind=%d_%d_%d_%d_%d_%d_%d_%d ; "
             "MARK@GPU?",
             bx, by, B, C, T, H, W, pt, ph, pw);
    return wan_submit_graph(ctx->uma, nodes);
  }

  if (ctx->uma && ctx->bufs && ctx->caps.prefer_ext && ctx->caps.ext_ready) {
    char kind[80];
    snprintf(kind, sizeof(kind), "EXT_UP3_%d_%d_%d_%d_%d_%d_%d_%d", B, C, T, H,
             W, pt, ph, pw);
    if (strlen(kind) < 64) {
      int Nb = B * T * H * W;
      int Db = C;
      if ((size_t)Nb * (size_t)Db == ne &&
          graph_ext(ctx, kind, bx, by, NULL, NULL, NULL, Nb, Db) == 0)
        return 0;
    }
  }

  if (!ctx->uma || !ctx->bufs)
    return -1;
  {
    char resp[512];
    size_t got = 0;
    float *tok = calloc(ne, sizeof(float));
    float *out = calloc(ne, sizeof(float));
    if (!tok || !out) {
      free(tok);
      free(out);
      return -1;
    }
    if (uma_client_buf_get(ctx->uma, bx, tok, nbytes, &got, resp, sizeof(resp)) !=
            0 ||
        got != nbytes) {
      free(tok);
      free(out);
      return -1;
    }
    uma_wan_unpatchify3d_f32(out, tok, B, C, T, H, W, pt, ph, pw);
    int rc = uma_buf_pool_put(ctx->bufs, by, out, nbytes);
    free(tok);
    free(out);
    return rc;
  }
}
