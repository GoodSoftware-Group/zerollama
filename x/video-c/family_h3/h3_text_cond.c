#include "h3_text_cond.h"

#include "h3_clipproj_host.h"
#include "h3_host.h"
#include "h3_present.h"
#include "h3_qwen_te_4b.h"
#include "h3_qwen_te_host.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int clipproj_te_layers(void) {
  const char *el = getenv("H3_QWEN_TE_LAYERS");
  if (el && el[0]) {
    int k = atoi(el);
    if (k >= 1 && k <= H3_QWEN_TE_LAYERS_4B)
      return k;
  }
  return H3_QWEN_TE_CLIPPROJ_TAP;
}

void h3_text_cond_free(h3_text_cond *c) {
  if (!c)
    return;
  free(c->cond);
  free(c->tags);
  memset(c, 0, sizeof(*c));
}

int h3_text_cond_from_prompt(const char *prompt, const int *merged_h,
                             const int *merged_w, size_t n_images,
                             const char *clipproj_path, h3_text_cond *out,
                             char *error, size_t error_size) {
  if (out)
    memset(out, 0, sizeof(*out));
  if (error && error_size)
    error[0] = 0;
  if (!prompt || !prompt[0] || !out)
    return -1;

  char err[1024];
  const char *blob = getenv("H3_BMTL_TOK");
  h3_tokenizer *tok = h3_tokenizer_load(blob, err, sizeof(err));
  if (!tok) {
    if (error && error_size)
      snprintf(error, error_size, "tokenizer: %s", err);
    return -1;
  }
  h3_presentation pres;
  if (!h3_present_fl2va(tok, prompt, n_images ? merged_h : NULL,
                        n_images ? merged_w : NULL, n_images, &pres, err,
                        sizeof(err))) {
    if (error && error_size)
      snprintf(error, error_size, "present: %s", err);
    h3_tokenizer_free(tok);
    return -1;
  }
  h3_tokenizer_free(tok);

  size_t n = pres.count;
  const int din = H3_QWEN_TE_HIDDEN_4B;
  const int dout = H3_CLIPPROJ_DOUT;
  float *hidden = (float *)calloc(n * (size_t)din, sizeof(float));
  float *cond = (float *)calloc(n * (size_t)dout, sizeof(float));
  if (!hidden || !cond) {
    free(hidden);
    free(cond);
    h3_presentation_free(&pres);
    return -1;
  }

  int used_4b = 0;
  char te_dir[768];
  if (!h3_resolve_qwen4b_dir(te_dir, sizeof(te_dir)))
    te_dir[0] = 0;
  if (te_dir[0]) {
    char shard[900];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
             te_dir);
    if (access(shard, R_OK) == 0) {
      char tap[16];
      snprintf(tap, sizeof(tap), "%d", clipproj_te_layers());
      setenv("H3_QWEN_TE_LAYERS", tap, 1);
      int apply_norm = getenv("H3_QWEN_TE_FINAL_NORM") ? 1 : 0;
      if (!h3_qwen_te_4b_forward(te_dir, pres.ids, n, pres.pos, apply_norm,
                                 hidden, err, sizeof(err))) {
        if (error && error_size)
          snprintf(error, error_size, "4B TE: %s", err);
        free(hidden);
        free(cond);
        h3_presentation_free(&pres);
        return -1;
      }
      used_4b = 1;
    }
  }
  if (!used_4b)
    h3_qwen_te_hash_embed(pres.ids, n, din, hidden);
  int *tags = (int *)malloc(n * sizeof(int));
  if (!tags) {
    free(hidden);
    free(cond);
    h3_presentation_free(&pres);
    return -1;
  }
  for (size_t i = 0; i < n; i++)
    tags[i] = (int)pres.tags[i];
  h3_presentation_free(&pres);

  char def[768];
  if (!clipproj_path || !clipproj_path[0]) {
    const char *home = getenv("HOME");
    if (!home) {
      free(hidden);
      free(cond);
      free(tags);
      if (error && error_size)
        snprintf(error, error_size, "ClipProj needs PATH or HOME");
      return -1;
    }
    snprintf(def, sizeof(def),
             "%s/.zerollama/third_party/h3/clipproj/"
             "mmh3-4b-ClipProj-celeb-mlp.safetensors",
             home);
    clipproj_path = def;
  }
  h3_clipproj *proj = h3_clipproj_load(clipproj_path, err, sizeof(err));
  if (!proj) {
    if (error && error_size)
      snprintf(error, error_size, "clipproj: %s", err);
    free(hidden);
    free(cond);
    free(tags);
    return -1;
  }
  int rc = h3_clipproj_apply(proj, hidden, (int)n, cond, err, sizeof(err));
  {
    double hsq = 0, csq = 0, tsq = 0;
    size_t hn = n * (size_t)din;
    size_t cn = n * (size_t)dout;
    for (size_t i = 0; i < hn; i++)
      hsq += (double)hidden[i] * hidden[i];
    for (size_t i = 0; i < cn; i++)
      csq += (double)cond[i] * cond[i];
    if (n > 1) {
      for (size_t i = (size_t)dout; i < cn; i++)
        tsq += (double)cond[i] * cond[i];
    }
    fprintf(stderr,
            "video-c: text cond tap=%d te_rms=%.4g cond_rms=%.4g "
            "text_rms=%.4g nt=%d (%s)\n",
            used_4b ? clipproj_te_layers() : 0, sqrt(hsq / (double)hn),
            sqrt(csq / (double)cn),
            n > 1 ? sqrt(tsq / (double)((n - 1) * (size_t)dout)) : 0.0, (int)n,
            used_4b ? "4B" : "hash");
  }
  h3_clipproj_free(proj);
  free(hidden);
  if (rc != 0) {
    if (error && error_size)
      snprintf(error, error_size, "clipproj apply: %s", err);
    free(cond);
    free(tags);
    return -1;
  }
  out->cond = cond;
  out->tags = tags;
  out->nt = (int)n;
  out->used_4b = used_4b;
  out->used_dump = 0;
  return 0;
}

int h3_text_cond_from_bin(const char *path, h3_text_cond *out, char *error,
                          size_t error_size) {
  if (out)
    memset(out, 0, sizeof(*out));
  if (error && error_size)
    error[0] = 0;
  if (!path || !path[0] || !out)
    return -1;
  FILE *f = fopen(path, "rb");
  if (!f) {
    if (error && error_size)
      snprintf(error, error_size, "open %s", path);
    return -1;
  }
  char mag[4];
  uint32_t nt = 0, dim = 0;
  if (fread(mag, 1, 4, f) != 4 || memcmp(mag, "H3TE", 4) != 0 ||
      fread(&nt, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1) {
    fclose(f);
    if (error && error_size)
      snprintf(error, error_size, "bad H3TE header in %s", path);
    return -1;
  }
  if (nt < 1 || nt > 8192 || dim != (uint32_t)H3_CLIPPROJ_DOUT) {
    fclose(f);
    if (error && error_size)
      snprintf(error, error_size, "H3TE nt=%u dim=%u (need dim=%d)", nt, dim,
               H3_CLIPPROJ_DOUT);
    return -1;
  }
  size_t n = (size_t)nt * (size_t)dim;
  float *cond = (float *)malloc(n * sizeof(float));
  if (!cond) {
    fclose(f);
    return -1;
  }
  if (fread(cond, sizeof(float), n, f) != n) {
    fclose(f);
    free(cond);
    if (error && error_size)
      snprintf(error, error_size, "short H3TE body in %s", path);
    return -1;
  }
  int *tags = NULL;
  uint8_t *raw = (uint8_t *)malloc((size_t)nt);
  if (raw && fread(raw, 1, (size_t)nt, f) == (size_t)nt) {
    tags = (int *)malloc((size_t)nt * sizeof(int));
    if (tags) {
      for (uint32_t i = 0; i < nt; i++) {
        int v = (int)raw[i];
        tags[i] = (v >= 0 && v <= 2) ? v : 1;
      }
    }
  }
  free(raw);
  fclose(f);
  double csq = 0;
  for (size_t i = 0; i < n; i++)
    csq += (double)cond[i] * cond[i];
  fprintf(stderr, "video-c: text cond dump nt=%d dim=%d cond_rms=%.4g tags=%s (%s)\n",
          (int)nt, (int)dim, sqrt(csq / (double)n), tags ? "yes" : "no", path);
  out->cond = cond;
  out->tags = tags;
  out->nt = (int)nt;
  out->used_dump = 1;
  return 0;
}
