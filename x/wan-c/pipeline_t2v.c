#include "encode_mp4.h"
#include "sched_unipc.h"
#include "tokenizer_spm.h"
#include "wan_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t latent_elems(const wan_model_config *c, int w, int h, int frames) {
  int lt = (frames - 1) / c->vae_stride_t + 1;
  int lh = h / c->vae_stride_h;
  int lw = w / c->vae_stride_w;
  return (size_t)c->z_channels * (size_t)lt * (size_t)lh * (size_t)lw;
}

static void fill_noise(float *x, size_t n, int seed) {
  /* Match Wan / torch.randn: i.i.d. N(0,1). Prior LCG uniform[-1,1] was a
   * quality gap (blob-like frames even at multi-step). */
  unsigned s = (unsigned)seed;
  if (s == 0)
    s = (unsigned)time(NULL);
  for (size_t i = 0; i < n; i += 2) {
    float u1, u2;
    do {
      s = s * 1103515245u + 12345u;
      u1 = (float)(s & 0xffffff) / 16777216.0f;
    } while (u1 <= 1e-7f);
    s = s * 1103515245u + 12345u;
    u2 = (float)(s & 0xffffff) / 16777216.0f;
    float r = sqrtf(-2.0f * logf(u1));
    float th = 6.28318530718f * u2;
    x[i] = r * cosf(th);
    if (i + 1 < n)
      x[i + 1] = r * sinf(th);
  }
}

int wan_pipeline_t2v(wan_ctx *ctx, const wan_gen_params *p, float *rgb_out,
                     size_t rgb_cap, size_t *rgb_len) {
  if (!ctx || !p)
    return -1;

  char err[256];
  if (wan_validate_params(p, err, sizeof(err)) != 0) {
    fprintf(stderr, "wan-c: %s\n", err);
    return -1;
  }

  const wan_model_config *mc = &ctx->cfg;
  size_t latent_n = latent_elems(mc, p->width, p->height, p->frames);
  size_t rgb_n = (size_t)p->width * (size_t)p->height * (size_t)p->frames * 3;

  ctx->gen_lt = (p->frames - 1) / mc->vae_stride_t + 1;
  ctx->gen_lh = p->height / mc->vae_stride_h;
  ctx->gen_lw = p->width / mc->vae_stride_w;

  float *latent = calloc(latent_n, sizeof(float));
  float *model_out = calloc(latent_n, sizeof(float));
  size_t text_n = (size_t)mc->text_len * (size_t)mc->text_dim;
  float *text_emb = calloc(text_n, sizeof(float));
  float *neg_emb = calloc(text_n, sizeof(float));
  float *rgb = calloc(rgb_n, sizeof(float));
  if (!latent || !model_out || !text_emb || !neg_emb || !rgb) {
    free(latent);
    free(model_out);
    free(text_emb);
    free(neg_emb);
    free(rgb);
    return -1;
  }

  char vocab_auto[4096];
  const char *vocab = p->vocab_path;
  if (!vocab || !vocab[0]) {
    static const char *k_vocab_candidates[] = {
        "%s/umt5.vocab",
        "%s/google/umt5-xxl/spiece.model",
        "%s/spiece.model",
        NULL,
    };
    vocab = NULL;
    for (int vi = 0; k_vocab_candidates[vi]; vi++) {
      snprintf(vocab_auto, sizeof(vocab_auto), k_vocab_candidates[vi],
               ctx->ckpt_dir);
      FILE *vf = fopen(vocab_auto, "rb");
      if (vf) {
        fclose(vf);
        vocab = vocab_auto;
        break;
      }
    }
  }

  int32_t ids[512];
  size_t n_ids = 0;
  int32_t neg_ids[512];
  size_t n_neg = 0;
  if (vocab) {
    tokenizer_spm *tok = tokenizer_spm_load(vocab);
    if (tok) {
      tokenizer_spm_encode(tok, p->prompt, ids, 512, &n_ids);
      if (p->negative_prompt && p->negative_prompt[0])
        tokenizer_spm_encode(tok, p->negative_prompt, neg_ids, 512, &n_neg);
      fprintf(stderr, "wan-c: SPM encode prompt ids=%zu vocab=%s\n", n_ids,
              vocab);
      tokenizer_spm_free(tok);
    }
  }

  if (n_ids > 0) {
    if (wan_t5_encode_ids(ctx, ids, n_ids, text_emb, text_n) != 0) {
      fprintf(stderr, "wan-c: T5 encode(ids) failed\n");
      free(latent);
      free(model_out);
      free(text_emb);
      free(neg_emb);
      free(rgb);
      return -1;
    }
  } else if (wan_t5_encode(ctx, p->prompt, text_emb, text_n) != 0) {
    fprintf(stderr, "wan-c: T5 encode failed\n");
    free(latent);
    free(model_out);
    free(text_emb);
    free(neg_emb);
    free(rgb);
    return -1;
  }
  if (p->negative_prompt && p->negative_prompt[0]) {
    if (n_neg > 0)
      wan_t5_encode_ids(ctx, neg_ids, n_neg, neg_emb, text_n);
    else
      wan_t5_encode(ctx, p->negative_prompt, neg_emb, text_n);
  } else if (p->cfg_scale > 1.0001f) {
    /* Wan CFG uses empty/neg prompt — not a second cond pass. */
    int32_t empty_ids[8];
    size_t n_empty = 0;
    if (vocab) {
      tokenizer_spm *tok = tokenizer_spm_load(vocab);
      if (tok) {
        tokenizer_spm_encode(tok, "", empty_ids, 8, &n_empty);
        tokenizer_spm_free(tok);
      }
    }
    if (n_empty > 0) {
      if (wan_t5_encode_ids(ctx, empty_ids, n_empty, neg_emb, text_n) != 0)
        memset(neg_emb, 0, text_n * sizeof(float));
    } else {
      memset(neg_emb, 0, text_n * sizeof(float));
    }
    static int logged_cfg;
    if (!logged_cfg) {
      fprintf(stderr, "wan-c: CFG uncond = empty prompt encoding\n");
      logged_cfg = 1;
    }
  } else {
    memset(neg_emb, 0, text_n * sizeof(float));
    static int logged_nocfg;
    if (!logged_nocfg) {
      fprintf(stderr, "wan-c: CFG≈1 — skip uncond T5/DiT\n");
      logged_nocfg = 1;
    }
  }

  fill_noise(latent, latent_n, p->seed);

  {
    const char *dump = getenv("WAN_DUMP_DIR");
    if (dump && dump[0]) {
      char path[768];
      FILE *f;
      /* Active T5 rows = leading non-zero (trim-by-default matches Wan u[:seq]). */
      int D = mc->text_dim;
      int rows = (D > 0) ? (int)(text_n / (size_t)D) : 0;
      int active = 0;
      for (int r = 0; r < rows; r++) {
        const float *row = text_emb + (size_t)r * (size_t)D;
        int nz = 0;
        for (int i = 0; i < D; i++) {
          if (row[i] != 0.f) {
            nz = 1;
            break;
          }
        }
        if (!nz)
          break;
        active = r + 1;
      }
      if (active < 1)
        active = (n_ids > 0 && (int)n_ids <= rows) ? (int)n_ids : rows;
      snprintf(path, sizeof(path), "%s/t5_emb.f32", dump);
      f = fopen(path, "wb");
      if (f) {
        fwrite(text_emb, sizeof(float), (size_t)active * (size_t)D, f);
        fclose(f);
      }
      snprintf(path, sizeof(path), "%s/noise.f32", dump);
      f = fopen(path, "wb");
      if (f) {
        fwrite(latent, sizeof(float), latent_n, f);
        fclose(f);
      }
      snprintf(path, sizeof(path), "%s/meta.json", dump);
      f = fopen(path, "w");
      if (f) {
        fprintf(f,
                "{\n  \"mode\": \"wan-c\",\n  \"prompt\": \"%s\",\n"
                "  \"t5_shape\": [%d, %d],\n  \"latent_elems\": %zu,\n"
                "  \"latent_shape\": [%d, %d, %d, %d],\n  \"seed\": %d,\n"
                "  \"width\": %d, \"height\": %d, \"frames\": %d\n}\n",
                p->prompt ? p->prompt : "", active, D, latent_n, mc->z_channels,
                ctx->gen_lt, ctx->gen_lh, ctx->gen_lw, p->seed, p->width,
                p->height, p->frames);
        fclose(f);
      }
      fprintf(stderr, "wan-c: WAN_DUMP_DIR=%s t5=[%d,%d] noise_n=%zu\n", dump,
              active, D, latent_n);
    }
  }

  sched_unipc *sched = sched_unipc_create(p->steps, p->shift > 0 ? p->shift : 5.0f);
  if (!sched) {
    free(latent);
    free(model_out);
    free(text_emb);
    free(neg_emb);
    free(rgb);
    return -1;
  }

  for (int step = 0; step < p->steps; step++) {
    float *cond = model_out;
    const float *sigmas = sched_unipc_sigmas(sched);
    if (sigmas)
      ctx->gen_t = sigmas[step] * 1000.f;
    else
      ctx->gen_t = 1000.f * fmaxf(0.f, 1.f - (float)step / (float)p->steps);

    fprintf(stderr, "wan-c: UniPC step %d/%d sigma_t=%.4f cfg=%.2f\n", step + 1,
            p->steps, ctx->gen_t / 1000.f, p->cfg_scale);
    fflush(stderr);

    memcpy(cond, latent, latent_n * sizeof(float));
    if (wan_dit_denoise(ctx, cond, latent_n, step, text_emb, text_n) != 0) {
      fprintf(stderr, "wan-c: DiT cond failed at step %d\n", step);
      sched_unipc_destroy(sched);
      free(latent);
      free(model_out);
      free(text_emb);
      free(neg_emb);
      free(rgb);
      return -1;
    }

    if (p->cfg_scale > 1.0001f) {
      float *uncond = calloc(latent_n, sizeof(float));
      if (!uncond) {
        sched_unipc_destroy(sched);
        free(latent);
        free(model_out);
        free(text_emb);
        free(neg_emb);
        free(rgb);
        return -1;
      }
      memcpy(uncond, latent, latent_n * sizeof(float));
      if (wan_dit_denoise(ctx, uncond, latent_n, step, neg_emb, text_n) != 0) {
        fprintf(stderr, "wan-c: DiT uncond failed at step %d\n", step);
        free(uncond);
        sched_unipc_destroy(sched);
        free(latent);
        free(model_out);
        free(text_emb);
        free(neg_emb);
        free(rgb);
        return -1;
      }
      sched_unipc_cfg_combine(uncond, cond, model_out, latent_n, p->cfg_scale);
      free(uncond);
    }
    /* else model_out already holds cond (== CFG scale 1) */

    /* Snapshot DiT pred before UniPC mutates sample (pred buffer is const). */
    if (step == 0) {
      const char *dump = getenv("WAN_DUMP_DIR");
      if (dump && dump[0]) {
        char path[768];
        FILE *f;
        snprintf(path, sizeof(path), "%s/dit_pred.f32", dump);
        f = fopen(path, "wb");
        if (f) {
          fwrite(model_out, sizeof(float), latent_n, f);
          fclose(f);
        }
      }
    }

    if (sched_unipc_step(sched, step, model_out, latent, latent_n) != 0) {
      sched_unipc_destroy(sched);
      free(latent);
      free(model_out);
      free(text_emb);
      free(neg_emb);
      free(rgb);
      return -1;
    }

    /* First-step DiT A/B dumps (latent after UniPC + meta). */
    if (step == 0) {
      const char *dump = getenv("WAN_DUMP_DIR");
      if (dump && dump[0]) {
        char path[768];
        FILE *f;
        snprintf(path, sizeof(path), "%s/latent_s1.f32", dump);
        f = fopen(path, "wb");
        if (f) {
          fwrite(latent, sizeof(float), latent_n, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/meta.json", dump);
        f = fopen(path, "w");
        if (f) {
          int D = mc->text_dim;
          int rows = (D > 0) ? (int)(text_n / (size_t)D) : 0;
          int active = 0;
          for (int r = 0; r < rows; r++) {
            const float *row = text_emb + (size_t)r * (size_t)D;
            int nz = 0;
            for (int i = 0; i < D; i++) {
              if (row[i] != 0.f) {
                nz = 1;
                break;
              }
            }
            if (!nz)
              break;
            active = r + 1;
          }
          if (active < 1)
            active = (n_ids > 0 && (int)n_ids <= rows) ? (int)n_ids : rows;
          fprintf(f,
                  "{\n  \"mode\": \"wan-c\",\n  \"prompt\": \"%s\",\n"
                  "  \"t5_shape\": [%d, %d],\n  \"latent_elems\": %zu,\n"
                  "  \"latent_shape\": [%d, %d, %d, %d],\n  \"seed\": %d,\n"
                  "  \"width\": %d, \"height\": %d, \"frames\": %d,\n"
                  "  \"steps\": %d, \"cfg_scale\": %.4f, \"shift\": %.4f,\n"
                  "  \"sigma0\": %.8f, \"gen_t\": %.4f,\n"
                  "  \"dumped\": [\"t5_emb.f32\", \"noise.f32\", "
                  "\"dit_pred.f32\", \"latent_s1.f32\"]\n}\n",
                  p->prompt ? p->prompt : "", active, D, latent_n,
                  mc->z_channels, ctx->gen_lt, ctx->gen_lh, ctx->gen_lw, p->seed,
                  p->width, p->height, p->frames, p->steps, p->cfg_scale,
                  p->shift > 0 ? p->shift : 5.0f,
                  sigmas ? sigmas[0] : ctx->gen_t / 1000.f, ctx->gen_t);
          fclose(f);
        }
        fprintf(stderr,
                "wan-c: WAN_DUMP_DIR DiT step0 pred+latent_s1 n=%zu t=%.2f\n",
                latent_n, ctx->gen_t);
      }
    }
  }
  sched_unipc_destroy(sched);

  if (wan_vae_decode(ctx, latent, latent_n, rgb, rgb_n, p->width, p->height,
                     p->frames) != 0) {
    fprintf(stderr, "wan-c: VAE decode failed\n");
    free(latent);
    free(model_out);
    free(text_emb);
    free(neg_emb);
    free(rgb);
    return -1;
  }

  if (rgb_out && rgb_cap >= rgb_n)
    memcpy(rgb_out, rgb, rgb_n * sizeof(float));
  if (rgb_len)
    *rgb_len = rgb_n;

  free(latent);
  free(model_out);
  free(text_emb);
  free(neg_emb);
  free(rgb);
  return 0;
}

int wan_generate_t2v(wan_ctx *ctx, const wan_gen_params *p,
                     const char *out_mp4) {
  if (!ctx || !p || !out_mp4)
    return -1;

  const wan_model_config *mc = &ctx->cfg;
  size_t rgb_n =
      (size_t)p->width * (size_t)p->height * (size_t)p->frames * 3;
  float *rgb = calloc(rgb_n, sizeof(float));
  if (!rgb)
    return -1;

  size_t rgb_len = 0;
  int rc = wan_pipeline_t2v(ctx, p, rgb, rgb_n, &rgb_len);
  if (rc != 0) {
    free(rgb);
    return rc;
  }

  rc = encode_mp4_from_rgb(out_mp4, p->width, p->height, p->frames, p->fps,
                           rgb, rgb_len);
  free(rgb);
  if (rc != 0) {
    fprintf(stderr, "wan-c: ffmpeg encode failed (is ffmpeg installed?)\n");
    return -1;
  }
  fprintf(stderr, "wan-c: wrote %s (%dx%d x%d frames)\n", out_mp4, p->width,
          p->height, p->frames);
  (void)mc;
  return 0;
}
