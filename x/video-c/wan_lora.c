/*
 * wan_lora.c — see wan_lora.h for the why.
 */
#include "wan_lora.h"
#include "safetensors_min.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WAN_LORA_MAX 512

typedef struct lora_pair {
  char base[224];
  const st_tensor_t *a; /* [rank, in] */
  const st_tensor_t *b; /* [out, rank] */
  const st_tensor_t *alpha;
} lora_pair;

struct wan_lora {
  st_file *sf;
  lora_pair pairs[WAN_LORA_MAX];
  int npairs;
};

static const char *k_prefixes[] = {"diffusion_model.", "transformer.",
                                   "model.diffusion_model.", NULL};

/* Classify a key: fills base (stripped of prefix + suffix) and kind.
 * kind: 1 = A, 2 = B, 3 = alpha, 0 = unsupported. */
static int classify(const char *key, char *base, size_t cap, int *is_lower) {
  for (int i = 0; k_prefixes[i]; i++) {
    size_t n = strlen(k_prefixes[i]);
    if (strncmp(key, k_prefixes[i], n) == 0) {
      key += n;
      break;
    }
  }
  static const char *sufs[][2] = {
      {".lora_A.weight", "1"}, {".lora_B.weight", "2"},
      {".lora_A.default.weight", "1"}, {".lora_B.default.weight", "2"},
      {".alpha", "3"},
      {".lora_down.weight", "1"}, /* diffusers lowercase alias */
      {".lora_up.weight", "2"},
      {NULL, NULL},
  };
  *is_lower = 0;
  for (int i = 0; sufs[i][0]; i++) {
    size_t n = strlen(sufs[i][0]);
    size_t k = strlen(key);
    if (k > n && strcmp(key + k - n, sufs[i][0]) == 0) {
      if (n >= cap)
        return 0;
      snprintf(base, cap, "%.*s", (int)(k - n), key);
      if (!strcmp(sufs[i][1], "1") && !strcmp(sufs[i][0], ".lora_down.weight"))
        *is_lower = 1;
      return atoi(sufs[i][1]);
    }
  }
  return 0;
}

static lora_pair *find_pair(wan_lora *L, const char *base) {
  for (int i = 0; i < L->npairs; i++)
    if (strcmp(L->pairs[i].base, base) == 0)
      return &L->pairs[i];
  if (L->npairs >= WAN_LORA_MAX)
    return NULL;
  lora_pair *p = &L->pairs[L->npairs++];
  memset(p, 0, sizeof(*p));
  snprintf(p->base, sizeof(p->base), "%s", base);
  return p;
}

wan_lora *wan_lora_open(const char *path) {
  if (!path || !path[0])
    return NULL;
  st_file *sf = st_open(path);
  if (!sf) {
    fprintf(stderr, "wan-c: lora open failed: %s\n", path);
    return NULL;
  }
  wan_lora *L = calloc(1, sizeof(*L));
  if (!L) {
    st_close(sf);
    return NULL;
  }
  L->sf = sf;
  int nt = st_tensor_count(sf);
  int skipped = 0;
  for (int i = 0; i < nt; i++) {
    const st_tensor_t *t = st_tensor_at(sf, i);
    if (!t)
      continue;
    char base[224];
    int lower = 0;
    int kind = classify(t->name, base, sizeof(base), &lower);
    if (kind == 0) {
      skipped++;
      continue;
    }
    lora_pair *p = find_pair(L, base);
    if (!p) {
      skipped++;
      continue;
    }
    if (kind == 3) {
      p->alpha = t;
      continue;
    }
    /* lora_A / lora_down are both [rank, in]. */
    if (kind == 1)
      p->a = t;
    else if (kind == 2)
      p->b = t;
  }
  /* Keep only complete pairs. */
  int kept = 0;
  for (int i = 0; i < L->npairs; i++) {
    lora_pair *p = &L->pairs[i];
    if (p->a && p->b)
      L->pairs[kept++] = *p;
  }
  L->npairs = kept;
  fprintf(stderr,
          "wan-c: lora %s targets=%d keys=%d skipped=%d\n",
          path, kept, nt, skipped);
  if (kept == 0) {
    fprintf(stderr,
            "wan-c: lora WARNING no complete A/B pairs — "
            "unsupported naming? (v1 supports dotted PEFT/ComfyUI style)\n");
  }
  return L;
}

void wan_lora_close(wan_lora *L) {
  if (!L)
    return;
  if (L->sf)
    st_close(L->sf);
  free(L);
}

int wan_lora_targets(const wan_lora *L) { return L ? L->npairs : 0; }

int wan_lora_apply(const wan_lora *L, const char *name, float *w, size_t n,
                   float cli_scale) {
  if (!L || !name || !w)
    return 0;
  /* Pairs are indexed by base ("blocks.N.x.y"); lookups arrive as weight
   * names ("blocks.N.x.y.weight") — align before compare. */
  char base[224];
  snprintf(base, sizeof(base), "%s", name);
  size_t bl = strlen(base);
  if (bl > 7 && strcmp(base + bl - 7, ".weight") == 0)
    base[bl - 7] = '\0';
  int applied = 0;
  for (int i = 0; i < L->npairs; i++) {
    const lora_pair *p = &L->pairs[i];
    if (strcmp(p->base, base) != 0)
      continue;
    long long rank = p->a->shape[0];
    long long in_f = p->a->shape[1];
    long long out_f = p->b->shape[0];
    if (p->b->shape[1] != rank || out_f * in_f != (long long)n ||
        rank < 1 || in_f < 1) {
      fprintf(stderr,
              "wan-c: lora shape mismatch %s: W=%zu B=[%lld,%lld] A=[%lld,"
              "%lld]\n",
              name, n, out_f, p->b->shape[1], rank, in_f);
      return -1;
    }
    float alpha = 1.0f;
    if (p->alpha && st_tensor_nelems(p->alpha) >= 1)
      if (st_tensor_to_f32(L->sf, p->alpha, &alpha, 1) != 0)
        alpha = 1.0f;
    float s = cli_scale * alpha / (float)rank;
    /* Load A/B once (f16/f32 → f32 scratch), then rows: w[n,:] += s·B[n,r]·A[r,:]. */
    size_t na = (size_t)rank * (size_t)in_f;
    size_t nb = (size_t)out_f * (size_t)rank;
    float *A = malloc(na * sizeof(float));
    float *B = malloc(nb * sizeof(float));
    if (!A || !B) {
      free(A);
      free(B);
      return -1;
    }
    if (st_tensor_to_f32(L->sf, p->a, A, na) != 0 ||
        st_tensor_to_f32(L->sf, p->b, B, nb) != 0) {
      free(A);
      free(B);
      return -1;
    }
    for (long long o = 0; o < out_f; o++) {
      float *wrow = w + (size_t)o * (size_t)in_f;
      const float *brow = B + (size_t)o * (size_t)rank;
      for (long long r = 0; r < rank; r++) {
        float bv = s * brow[r];
        if (bv == 0.0f)
          continue;
        const float *arow = A + (size_t)r * (size_t)in_f;
        for (long long k = 0; k < in_f; k++)
          wrow[k] += bv * arow[k];
      }
    }
    free(A);
    free(B);
    static int n_applied, logged_once;
    n_applied++;
    if (!logged_once || (n_applied % 50) == 0) {
      fprintf(stderr, "wan-c: lora applied x%d last=%s\n", n_applied, name);
      logged_once = 1;
    }
    applied = 1;
  }
  return applied;
}
