/* Probe MiniMax-H3 on-disk layout — aligned with antirez h3_load_dir. */
#include "h3_info.h"
#include "h3_adaln_host.h"
#include "h3_audio_vae_host.h"
#include "h3_clipproj_host.h"
#include "h3_dit_block.h"
#include "h3_dit_host.h"
#include "h3_host.h"
#include "h3_st_store.h"
#include "h3_video_vae_host.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  int files;
  int tensors; /* -1 if not counted */
  unsigned long long bytes;
} h3_inv;

static int is_dir(const char *path) {
  struct stat st;
  return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int is_file(const char *path) {
  struct stat st;
  return path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void report(const char *label, int ok, const char *detail) {
  printf("  %-28s %s%s%s\n", label, ok ? "ok" : "MISSING",
         detail && detail[0] ? " — " : "", detail ? detail : "");
}

/* Count .safetensors files and sum sizes (header inventory; no parse). */
static int inventory_dir(const char *dir, h3_inv *out) {
  memset(out, 0, sizeof(*out));
  out->tensors = -1;
  DIR *d = opendir(dir);
  if (!d)
    return 0;
  struct dirent *ent;
  char path[1100];
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.')
      continue;
    snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
    struct stat st;
    if (stat(path, &st) != 0)
      continue;
    if (S_ISREG(st.st_mode)) {
      const char *dot = strrchr(ent->d_name, '.');
      if (dot && !strcmp(dot, ".safetensors")) {
        out->files++;
        out->bytes += (unsigned long long)st.st_size;
      }
    } else if (S_ISDIR(st.st_mode)) {
      h3_inv sub;
      if (inventory_dir(path, &sub)) {
        out->files += sub.files;
        out->bytes += sub.bytes;
      }
    }
  }
  closedir(d);
  return out->files > 0;
}

static void report_inv(const char *label, const char *dir, int required) {
  char detail[192];
  if (!is_dir(dir)) {
    report(label, 0, required ? "directory missing" : "optional");
    return;
  }
  char err[128];
  h3_st_store *store = h3_st_store_open(dir, err, sizeof(err));
  if (store) {
    snprintf(detail, sizeof(detail), "%zu shard(s), %zu tensors, %.2f GiB",
             h3_st_store_shards(store), h3_st_store_tensors(store),
             h3_st_store_bytes(store) / (1024.0 * 1024.0 * 1024.0));
    report(label, 1, detail);
    h3_st_store_free(store);
    return;
  }
  h3_inv inv;
  int ok = inventory_dir(dir, &inv);
  if (ok)
    snprintf(detail, sizeof(detail), "%d shard(s), %.2f GiB", inv.files,
             inv.bytes / (1024.0 * 1024.0 * 1024.0));
  else
    snprintf(detail, sizeof(detail), "no .safetensors under %s", dir);
  report(label, ok, detail);
}

static int peek_st_hw2(const char *path, const char *tensor, long long *d0,
                       long long *d1) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1 || n < 16 || n > 64ull * 1024ull * 1024ull) {
    fclose(f);
    return -1;
  }
  char *j = (char *)malloc((size_t)n + 1);
  if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) {
    free(j);
    fclose(f);
    return -1;
  }
  fclose(f);
  j[n] = 0;
  char key[160];
  snprintf(key, sizeof(key), "\"%s\"", tensor);
  char *p = strstr(j, key);
  char *sh = p ? strstr(p, "\"shape\"") : NULL;
  long long a = 0, b = 0;
  int ok = 0;
  if (sh) {
    char *br = strchr(sh, '[');
    if (br)
      ok = sscanf(br, "[%lld ,%lld", &a, &b) == 2 ||
           sscanf(br, "[%lld,%lld", &a, &b) == 2;
  }
  free(j);
  if (!ok)
    return -1;
  if (d0)
    *d0 = a;
  if (d1)
    *d1 = b;
  return 0;
}

static void fill_dit_pack_path(char *buf, size_t n) {
  if (!h3_resolve_dit_pack_path(buf, n))
    buf[0] = 0;
}

static void fill_qwen4b_dir(char *buf, size_t n) {
  if (!h3_resolve_qwen4b_dir(buf, n))
    buf[0] = 0;
}

int h3_checkpoint_info(const char *model_dir) {
  if (!model_dir || !model_dir[0]) {
    fprintf(stderr, "video-c: h3 --info requires -d / --ckpt-dir\n");
    return 1;
  }
  if (!is_dir(model_dir)) {
    fprintf(stderr, "video-c: h3 model dir not found: %s\n", model_dir);
    return 1;
  }

  char fl2va[1024], ref2va[1024], path[1100];
  snprintf(fl2va, sizeof(fl2va), "%s/FL2VA", model_dir);
  snprintf(ref2va, sizeof(ref2va), "%s/Ref2VA", model_dir);

  int fl2 = is_dir(fl2va);
  printf("video-c h3 checkpoint info (M4 / Darwin host path)\n");
  printf("  model_dir                    %s\n", model_dir);
  report("FL2VA/", fl2, fl2 ? fl2va : "required released tree");

  int missing = 0;
  if (!fl2) {
    fprintf(stderr, "video-c: h3 layout incomplete — need %s/FL2VA\n", model_dir);
    return 1;
  }

  snprintf(path, sizeof(path), "%s/transformer/config.json", fl2va);
  int cfg = is_file(path);
  report("FL2VA/transformer/config.json", cfg, "required for --info probe");
  if (!cfg)
    missing = 1;

  snprintf(path, sizeof(path), "%s/tokenizer/tokenizer.json", fl2va);
  int tok = is_file(path);
  report("FL2VA/tokenizer/tokenizer.json", tok, "required");
  if (!tok)
    missing = 1;

  snprintf(path, sizeof(path), "%s/transformer", fl2va);
  report_inv("FL2VA/transformer shards", path, 0);
  int have_stock_dit = 0;
  {
    h3_inv inv;
    have_stock_dit = inventory_dir(path, &inv);
  }

  char dit_pack[768];
  fill_dit_pack_path(dit_pack, sizeof(dit_pack));
  int have_dit = dit_pack[0] && is_file(dit_pack);
  {
    char detail[192];
    if (have_dit)
      snprintf(detail, sizeof(detail), "%s", dit_pack);
    else if (dit_pack[0])
      snprintf(detail, sizeof(detail), "missing %s (H3_DIT_ST)", dit_pack);
    else
      snprintf(detail, sizeof(detail), "set H3_DIT_ST or HOME");
    report("pruned DiT pack", have_dit, detail);
  }
  if (have_dit) {
    long long g = 0, r = 0;
    if (peek_st_hw2(dit_pack, "adaln_t_table", &g, &r) == 0)
      printf("    adaln_t_table [%lld,%lld] (need [%d,%d])\n", g, r,
             H3_ADALN_TABLE_GRID, H3_ADALN_TABLE_RANK);
  }
  {
    const char *home = getenv("HOME");
    char bf16[768];
    bf16[0] = 0;
    if (home && home[0])
      snprintf(bf16, sizeof(bf16),
               "%s/.zerollama/third_party/h3/dit/diffusion_models/"
               "minimax_h3_fl2va_pruned_bf16.safetensors",
               home);
    if (bf16[0] && is_file(bf16)) {
      char detail[192] = "Comfy only; not video-c";
      long long g = 0, r = 0;
      if (peek_st_hw2(bf16, "adaln_t_table", &g, &r) == 0)
        snprintf(detail, sizeof(detail),
                 "Comfy only: adaln_t_table [%lld,%lld] (video-c needs %d×%d)",
                 g, r, H3_ADALN_TABLE_GRID, H3_ADALN_TABLE_RANK);
      report("Comfy pruned BF16", 1, detail);
    }
  }

  snprintf(path, sizeof(path), "%s/text_encoder", fl2va);
  report_inv("FL2VA/text_encoder (32B, unused)", path, 0);

  char te_dir[768];
  fill_qwen4b_dir(te_dir, sizeof(te_dir));
  int have_te = 0;
  if (te_dir[0]) {
    char shard[900];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             te_dir);
    have_te = is_file(shard);
    report("Qwen3-VL-4B TE", have_te, have_te ? te_dir : shard);
  } else {
    report("Qwen3-VL-4B TE", 0, "set H3_QWEN_TE_DIR or HOME");
  }

  snprintf(path, sizeof(path), "%s/video_vae/source", fl2va);
  report_inv("FL2VA/video_vae/source", path, 1);
  {
    h3_inv inv;
    if (!inventory_dir(path, &inv))
      missing = 1;
  }

  snprintf(path, sizeof(path), "%s/audio_vae", fl2va);
  report_inv("FL2VA/audio_vae", path, 1);
  {
    h3_inv inv;
    if (!inventory_dir(path, &inv))
      missing = 1;
  }

  int ref = is_dir(ref2va);
  report("Ref2VA/", ref, ref ? "optional Ref2VA tree" : "optional");
  if (ref) {
    snprintf(path, sizeof(path), "%s/transformer", ref2va);
    report_inv("Ref2VA/transformer", path, 0);
  }

  /* Host geometry smoke (no weights): 512² / 22 frames balanced preset. */
  {
    int lw, lh;
    h3_latent_canvas(512, 512, &lw, &lh);
    h3_temporal_shape t = h3_temporal(22);
    h3_sigma_schedule sched;
    int sk = h3_serving_schedule_build(20, &sched);
    printf("\n  host geometry (no weights)\n");
    printf("  canvas 512x512 → latent %dx%d  frames22→Tvideo=%d Taudio=%d\n", lw,
           lh, t.video_t, t.audio_t);
    printf("  audio VAE hop=%d → Taudio=%d → pcm=%d @ %d Hz\n",
           H3_AUDIO_VAE_HOP_LENGTH, t.audio_t,
           h3_audio_vae_pcm_samples(t.audio_t), H3_AUDIO_VAE_SAMPLE_RATE);
    printf("  video VAE spatial=%dx temporal=%d clip=%d drop=%d decoder=%dx%d\n",
           h3_video_vae_spatial_ratio(), h3_video_vae_temporal_ratio(),
           H3_VIDEO_VAE_CLIP_LENGTH, H3_VIDEO_VAE_TOKEN_DROP,
           H3_VIDEO_VAE_DECODER_LAYERS, h3_video_vae_decoder_dim());
    printf("  serving schedule steps=20 %s\n", sk ? "ok" : "FAIL");
    if (!sk)
      missing = 1;
    if (h3_audio_vae_hop_from_rates() != H3_AUDIO_VAE_HOP_LENGTH)
      missing = 1;
  }

  printf("\n  DiT host constants\n");
  printf("  hidden=%d heads=%d head_dim=%d inner=%d layers=%d (generate default %d)\n",
         H3_DIT_HIDDEN_SIZE, H3_DIT_NUM_HEADS, H3_DIT_HEAD_DIM, H3_DIT_INNER_DIM,
         H3_DIT_NUM_LAYERS, H3_DIT_DEFAULT_GENERATE_LAYERS);
  printf("  text_dim=%d patch_dim=%d rope_dim=%d adaln_out=%d\n",
         H3_DIT_TEXT_DIM, H3_DIT_VIDEO_PATCH_DIM, H3_DIT_ROPE_DIM,
         H3_ADALN_OUT_FEATURES);
  printf("  ModulationCache T=20 → %.1f MiB bf16 (vs ~%.0f GiB adaln_proj)\n",
         h3_adaln_cache_bf16_nbytes(20) / (1024.0 * 1024.0),
         h3_adaln_proj_bf16_nbytes() / (1024.0 * 1024.0 * 1024.0));

  {
    const char *home = getenv("HOME");
    char clip[768];
    int found = 0;
    if (home) {
      static const char *names[] = {
          "mmh3-4b-ClipProj-celeb-mlp.safetensors",
          "mmh3-8b-ClipProj-celeb-mlp.safetensors",
          "mmh3-ClipProj-control-zero.safetensors",
          "mmh3-ClipProj-control-identity.safetensors",
      };
      printf("\n  ClipProj (optional TE shrink)\n");
      for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        snprintf(clip, sizeof(clip),
                 "%s/.zerollama/third_party/h3/clipproj/%s", home, names[i]);
        if (access(clip, R_OK) == 0) {
          char err[128];
          h3_clipproj *p = h3_clipproj_load(clip, err, sizeof(err));
          if (p) {
            printf("  %-40s din=%d dout=%d sink=%d mlp=%d\n", names[i],
                   h3_clipproj_din(p), h3_clipproj_dout(p),
                   h3_clipproj_has_sink(p), h3_clipproj_has_mlp(p));
            h3_clipproj_free(p);
            found = 1;
          }
        }
      }
      if (!found)
        printf("  (none under ~/.zerollama/third_party/h3/clipproj/)\n");
    }
  }

  printf("\n");
  if (have_dit && have_te)
    printf("  generate-ready: pruned DiT + 4B TE + ClipProj path "
           "(--generate default %d/%d layers; 50L host is rank-1)\n",
           H3_DIT_DEFAULT_GENERATE_LAYERS, H3_DIT_NUM_LAYERS);
  else if (have_dit)
    printf("  generate: pruned DiT present; no 4B TE → hash text (not a fox)\n");
  else
    printf("  probe-only: generate needs pruned DiT "
           "(~/.zerollama/third_party/h3/dit or H3_DIT_ST)\n");
  if (have_stock_dit)
    printf("  Metal ../h3.c: stock BF16 transformer shards present\n");
  else
    printf("  Metal ../h3.c: no stock BF16 transformer shards "
           "(int8 ConvRot pack is video-c only)\n");
  printf("  audio decode: --family h3 --decode-audio -d DIR -o out.wav\n");
  printf("  audio encode: --family h3 --encode-audio -d DIR\n");
  printf("  rematch: make -C x/video-c test ; make -C x/video-c test-h3-weights\n");

  if (missing) {
    fprintf(stderr,
            "video-c: h3 probe incomplete — need tokenizer + audio_vae + video_vae/source\n");
    return 1;
  }
  return 0;
}
