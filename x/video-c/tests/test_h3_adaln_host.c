/* Rematch AdaLN ModulationCache sizing vs minimax-h3-mlx.adaln docs. */
#include "h3_adaln_host.h"

#include <stdio.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_modality(void) {
  CHECK(h3_adaln_modality_row(0, H3_ADALN_TAG_VIDEO) == 0);
  CHECK(h3_adaln_modality_row(0, H3_ADALN_TAG_TEXT) == 1);
  CHECK(h3_adaln_modality_row(0, H3_ADALN_TAG_AUDIO) == 2);
  CHECK(h3_adaln_modality_row(1, H3_ADALN_TAG_VIDEO) == 3);
  CHECK(h3_adaln_modality_row(5, H3_ADALN_TAG_AUDIO) == 17);
  CHECK(h3_adaln_modality_row(2, H3_ADALN_TAG_PAD) == 6); /* PAD → video */
  CHECK(h3_adaln_modality_row(-1, 0) == -1);
  return 0;
}

static int test_schedule(void) {
  float sigmas[] = {1.0f, 0.5f, 0.0f, 0.5f};
  float dst[8];
  int n = h3_adaln_schedule_timesteps(sigmas, 4, 0.0f, dst, 8);
  CHECK(n == 3);
  CHECK(dst[0] == 0.0f && dst[1] == 0.5f && dst[2] == 1.0f);
  n = h3_adaln_schedule_timesteps(sigmas, 4, 0.25f, dst, 8);
  CHECK(n == 4);
  CHECK(dst[0] == 0.0f && dst[1] == 0.25f && dst[2] == 0.5f && dst[3] == 1.0f);
  return 0;
}

static int test_cache_size(void) {
  /* MLX adaln.py: 50×6×(40×3)×5376 = 193536000 values ≈ 387 MB bf16 */
  CHECK(h3_adaln_cache_values(40) == 193536000ull);
  CHECK(h3_adaln_cache_bf16_nbytes(40) == 387072000ull);
  CHECK(H3_ADALN_OUT_FEATURES ==
        H3_ADALN_TENSORS_PER_BLOCK * H3_ADALN_MODALITY_NUM *
            H3_ADALN_HIDDEN_SIZE);
  unsigned long long proj = h3_adaln_proj_bf16_nbytes();
  /* ~26 GiB class: 50 * (96768*2688 + 96768) * 2 */
  CHECK(proj > 25ull * 1000ull * 1000ull * 1000ull);
  CHECK(proj < 28ull * 1000ull * 1000ull * 1000ull);
  /* Cache ≪ projections (~67×). */
  CHECK(h3_adaln_cache_bf16_nbytes(40) * 60ull < proj);
  return 0;
}

static int test_split(void) {
  /* Tiny: T=2, H=4, out features = 6*3*4 = 72 */
  const int T = 2, H = 4, M = 3;

  float proj[2 * 72];
  for (int i = 0; i < 2 * 72; i++)
    proj[i] = (float)i;
  float dst[6 * 2 * 3 * 4];
  CHECK(h3_adaln_split_block(proj, T, H, dst) == 0);
  /* Row for t=1,m=2 → index 1*3+2=5; k=3 chunk starts at
   * proj[1*72 + 2*(6*4) + 3*4] = 72+48+12 = 132 */
  int row = h3_adaln_modality_row(1, H3_ADALN_TAG_AUDIO);
  CHECK(row == 5);
  const float *k3 = dst + 3 * (T * M * H) + row * H;
  CHECK(k3[0] == 132.f && k3[3] == 135.f);
  /* t=0,m=0,k=0 → 0..3 */
  const float *k0 = dst + 0 * (T * M * H) + 0 * H;
  CHECK(k0[0] == 0.f && k0[3] == 3.f);
  return 0;
}

static int test_split_final(void) {
  float proj[] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f};
  float shift[4], scale[4];
  CHECK(h3_adaln_split_final(proj, 2, 2, shift, scale) == 0);
  CHECK(shift[0] == 1.f && shift[1] == 2.f);
  CHECK(scale[0] == 3.f && scale[1] == 4.f);
  CHECK(shift[2] == 10.f && scale[3] == 40.f);
  return 0;
}

static int test_collect_sorted(void) {
  /* First-seen 0.8 then 0.2 → Comfy sorted unique_t [0.2, 0.8]. */
  float row_t[] = {0.8f, 0.2f, 0.8f, 0.2f};
  float uniq[8];
  int slots[4];
  int n = h3_adaln_collect_timesteps(row_t, 0.f, 4, uniq, 8, slots);
  CHECK(n == 2);
  CHECK(uniq[0] == 0.2f && uniq[1] == 0.8f);
  CHECK(slots[0] == 1 && slots[1] == 0 && slots[2] == 1 && slots[3] == 0);
  CHECK(h3_adaln_modality_row(slots[0], H3_ADALN_TAG_VIDEO) == 3);
  CHECK(h3_adaln_modality_row(slots[1], H3_ADALN_TAG_AUDIO) == 2);
  return 0;
}

int main(void) {
  if (test_modality())
    return 1;
  if (test_schedule())
    return 1;
  if (test_cache_size())
    return 1;
  if (test_split())
    return 1;
  if (test_split_final())
    return 1;
  if (test_collect_sorted())
    return 1;
  printf("test_h3_adaln_host OK\n");
  return 0;
}
