#include "wan.h"
#include "wan_internal.h"
#include "wan_lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *join_path(const char *dir, const char *leaf, char *out,
                       size_t out_n) {
  size_t dlen = strlen(dir);
  int need_slash = dlen > 0 && dir[dlen - 1] != '/';
  snprintf(out, out_n, "%s%s%s", dir, need_slash ? "/" : "", leaf);
  return out;
}

static int try_index_path(char *out, size_t out_n, const char *ckpt_dir,
                          const char *leaf) {
  join_path(ckpt_dir, leaf, out, out_n);
  if (access(out, R_OK) == 0)
    return 0;
  const char *env = getenv("WAN_WEIGHT_INDEX_DIR");
  if (env && env[0]) {
    join_path(env, leaf, out, out_n);
    if (access(out, R_OK) == 0)
      return 0;
  }
  /* Repo-relative from common cwd layouts. */
  const char *cands[] = {
      "x/wan-c/indices",
      "indices",
      "../x/wan-c/indices",
      "../../x/wan-c/indices",
      NULL};
  for (int i = 0; cands[i]; i++) {
    snprintf(out, out_n, "%s/%s", cands[i], leaf);
    if (access(out, R_OK) == 0)
      return 0;
  }
  return -1;
}

wan_ctx *wan_ctx_open(const char *ckpt_dir, const char *uma_sock) {
  if (!ckpt_dir)
    return NULL;

  wan_ctx *ctx = calloc(1, sizeof(*ctx));
  if (!ctx)
    return NULL;

  snprintf(ctx->ckpt_dir, sizeof(ctx->ckpt_dir), "%s", ckpt_dir);
  if (uma_sock)
    snprintf(ctx->uma_sock, sizeof(ctx->uma_sock), "%s", uma_sock);

  ctx->cfg = *wan_model_config_1_3b();
  ctx->local_mode = wan_env_local();

  if (!ctx->local_mode) {
    ctx->uma = uma_client_connect(ctx->uma_sock[0] ? ctx->uma_sock : NULL);
    if (!ctx->uma) {
      fprintf(stderr,
              "wan-c: UMA broker unavailable; set UMA_WAN_LOCAL=1 for host "
              "kernels\n");
      free(ctx);
      return NULL;
    }
    ctx->bufs = uma_buf_pool_create(ctx->uma);
    if (!ctx->bufs) {
      uma_client_close(ctx->uma);
      free(ctx);
      return NULL;
    }
    wan_probe_caps(ctx);
    if (ctx->caps.prefer_ext)
      (void)wan_ext_setup(ctx);
  } else {
    fprintf(stderr, "wan-c: UMA_WAN_LOCAL=1 — host uma_wan_ops path\n");
  }

  /* DiT weights: mmap safetensors in place (no GGUF duplicate). */
  char st_path[4096];
  join_path(ckpt_dir, "diffusion_pytorch_model.safetensors", st_path,
            sizeof(st_path));
  ctx->st = st_open(st_path);
  if (ctx->st)
    fprintf(stderr, "wan-c: safetensors mmap (%d tensors) %s\n",
            st_tensor_count(ctx->st), st_path);
  else
    fprintf(stderr, "wan-c: no safetensors at %s\n", st_path);

  /* T5 embed: mmap torch .pth via small JSON index (no extract). */
  char t5_pth[4096], t5_idx[4096];
  join_path(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth", t5_pth, sizeof(t5_pth));
  if (access(t5_pth, R_OK) == 0 &&
      try_index_path(t5_idx, sizeof(t5_idx), ckpt_dir, "t5_embed_index.json") ==
          0) {
    ctx->t5_zip = zw_open(t5_pth, t5_idx);
    if (ctx->t5_zip)
      fprintf(stderr, "wan-c: T5 .pth mmap via %s\n", t5_idx);
  }

  /* VAE: mmap Wan2.1_VAE.pth via packed-storage index. */
  char vae_pth[4096], vae_idx[4096];
  join_path(ckpt_dir, "Wan2.1_VAE.pth", vae_pth, sizeof(vae_pth));
  if (access(vae_pth, R_OK) == 0 &&
      try_index_path(vae_idx, sizeof(vae_idx), ckpt_dir, "vae_index.json") ==
          0) {
    ctx->vae_zip = zw_open(vae_pth, vae_idx);
    if (ctx->vae_zip)
      fprintf(stderr, "wan-c: VAE .pth mmap via %s\n", vae_idx);
  }

  /* Optional GGUF fallback (legacy / partial slices). */
  char gguf_path[4096];
  join_path(ckpt_dir, "wan_t2v_1.3b.gguf", gguf_path, sizeof(gguf_path));
  ctx->gguf = gguf_open(gguf_path);
  if (ctx->gguf)
    fprintf(stderr, "wan-c: GGUF loaded (%d tensors) [fallback]\n",
            gguf_tensor_count(ctx->gguf));

  if (!ctx->st && !ctx->gguf && !ctx->t5_zip && !ctx->vae_zip)
    fprintf(stderr, "wan-c: warning — no weight sources opened under %s\n",
            ckpt_dir);

  return ctx;
}

void wan_ctx_close(wan_ctx *ctx) {
  if (!ctx)
    return;
  wan_weight_cache_clear(ctx);
  wan_lora_close((wan_lora *)ctx->lora);
  ctx->lora = NULL;
  for (int i = 0; i < 2; i++) {
    free(ctx->dit_tctx_pack[i]);
    ctx->dit_tctx_pack[i] = NULL;
  }
  zw_close(ctx->vae_zip);
  zw_close(ctx->t5_zip);
  st_close(ctx->st);
  gguf_close(ctx->gguf);
  uma_buf_pool_destroy_keep_bank(ctx->bufs);
  uma_client_close(ctx->uma);
  free(ctx);
}
