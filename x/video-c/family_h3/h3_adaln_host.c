#include "h3_adaln_host.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int h3_adaln_modality_row(int timestep_index, int tag) {
  if (timestep_index < 0)
    return -1;
  int t = tag;
  if (t < 0)
    t = 0;
  return timestep_index * H3_ADALN_MODALITY_NUM + t;
}

static int cmp_float(const void *a, const void *b) {
  float x = *(const float *)a;
  float y = *(const float *)b;
  if (x < y)
    return -1;
  if (x > y)
    return 1;
  return 0;
}

int h3_adaln_schedule_timesteps(const float *sigmas, int n_sigmas,
                                float conditioning_level, float *dst,
                                int dst_cap) {
  if (!dst || dst_cap < 1 || n_sigmas < 0 || (n_sigmas > 0 && !sigmas))
    return -1;
  int cap = n_sigmas + 1;
  if (cap > dst_cap)
    return -1;
  float *tmp = malloc((size_t)cap * sizeof(float));
  if (!tmp)
    return -1;
  int n = 0;
  tmp[n++] = conditioning_level;
  for (int i = 0; i < n_sigmas; i++)
    tmp[n++] = sigmas[i];
  qsort(tmp, (size_t)n, sizeof(float), cmp_float);
  int out = 0;
  for (int i = 0; i < n; i++) {
    if (out > 0 && fabsf(tmp[i] - dst[out - 1]) < 5e-10f)
      continue;
    dst[out++] = tmp[i];
  }
  free(tmp);
  return out;
}

int h3_adaln_collect_timesteps(const float *row_t, float timestep, int seq,
                               float *uniq, int uniq_cap, int *tslots) {
  if (!uniq || !tslots || seq < 1 || uniq_cap < 1)
    return -1;
  int n = 0;
  for (int s = 0; s < seq; s++) {
    float t = row_t ? row_t[s] : timestep;
    int found = 0;
    for (int u = 0; u < n; u++) {
      if (fabsf(uniq[u] - t) < 1e-7f) {
        found = 1;
        break;
      }
    }
    if (found)
      continue;
    if (n >= uniq_cap)
      return -1;
    uniq[n++] = t;
  }
  qsort(uniq, (size_t)n, sizeof(float), cmp_float);
  for (int s = 0; s < seq; s++) {
    float t = row_t ? row_t[s] : timestep;
    int found = -1;
    for (int u = 0; u < n; u++) {
      if (fabsf(uniq[u] - t) < 1e-7f) {
        found = u;
        break;
      }
    }
    if (found < 0)
      return -1;
    tslots[s] = found;
  }
  return n;
}

unsigned long long h3_adaln_cache_values(int num_timesteps) {
  if (num_timesteps < 1)
    return 0;
  return (unsigned long long)H3_ADALN_NUM_LAYERS *
         (unsigned long long)H3_ADALN_TENSORS_PER_BLOCK *
         (unsigned long long)num_timesteps *
         (unsigned long long)H3_ADALN_MODALITY_NUM *
         (unsigned long long)H3_ADALN_HIDDEN_SIZE;
}

unsigned long long h3_adaln_cache_bf16_nbytes(int num_timesteps) {
  return h3_adaln_cache_values(num_timesteps) * 2ull;
}

unsigned long long h3_adaln_proj_bf16_nbytes(void) {
  unsigned long long w =
      (unsigned long long)H3_ADALN_OUT_FEATURES *
      (unsigned long long)H3_ADALN_TIME_EMBED_DIM;
  unsigned long long b = (unsigned long long)H3_ADALN_OUT_FEATURES;
  return (w + b) * 2ull * (unsigned long long)H3_ADALN_NUM_LAYERS;
}

int h3_adaln_split_block(const float *proj_out, int num_timesteps, int hidden,
                         float *dst_six) {
  if (!proj_out || !dst_six || num_timesteps < 1 || hidden < 1)
    return -1;
  const int M = H3_ADALN_MODALITY_NUM;
  const int K = H3_ADALN_TENSORS_PER_BLOCK;
  size_t rows = (size_t)num_timesteps * (size_t)M;
  size_t per = rows * (size_t)hidden;
  for (int t = 0; t < num_timesteps; t++) {
    const float *src_t = proj_out + (size_t)t * (size_t)(K * M * hidden);
    for (int m = 0; m < M; m++) {
      const float *src_m = src_t + (size_t)m * (size_t)(K * hidden);
      size_t row = (size_t)t * (size_t)M + (size_t)m;
      for (int k = 0; k < K; k++) {
        const float *chunk = src_m + (size_t)k * (size_t)hidden;
        float *dst = dst_six + (size_t)k * per + row * (size_t)hidden;
        memcpy(dst, chunk, (size_t)hidden * sizeof(float));
      }
    }
  }
  return 0;
}

int h3_adaln_split_final(const float *proj_out, int num_timesteps, int hidden,
                         float *shift, float *scale) {
  if (!proj_out || !shift || !scale || num_timesteps < 1 || hidden < 1)
    return -1;
  for (int t = 0; t < num_timesteps; t++) {
    const float *row = proj_out + (size_t)t * (size_t)(2 * hidden);
    memcpy(shift + (size_t)t * (size_t)hidden, row,
           (size_t)hidden * sizeof(float));
    memcpy(scale + (size_t)t * (size_t)hidden, row + hidden,
           (size_t)hidden * sizeof(float));
  }
  return 0;
}

int h3_adaln_table_embed(const float *table, int grid, int rank, float t,
                         float *out) {
  if (!table || !out || grid < 2 || rank < 1)
    return -1;
  if (t < 0.f)
    t = 0.f;
  if (t > 1.f)
    t = 1.f;
  float x = t * (float)(grid - 1);
  int i0 = (int)x;
  if (i0 >= grid - 1) {
    memcpy(out, table + (size_t)(grid - 1) * (size_t)rank,
           (size_t)rank * sizeof(float));
    return 0;
  }
  float a = x - (float)i0;
  const float *a0 = table + (size_t)i0 * (size_t)rank;
  const float *a1 = a0 + rank;
  for (int i = 0; i < rank; i++)
    out[i] = a0[i] * (1.f - a) + a1[i] * a;
  return 0;
}
