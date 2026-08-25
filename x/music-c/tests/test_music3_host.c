#include "music_chunk.h"
#include "music_dav.h"
#include "music_prompt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_prompt(void) {
  CHECK(MUSIC3_IM_START == 151644);
  CHECK(MUSIC3_AUDIO_CFG == 151654);
  CHECK(MUSIC3_AUDIO_CODE_OFFSET == 151675);
  char *own = NULL;
  CHECK(music3_normalize_lyrics("[Verse]\nWalking down the street", &own));
  CHECK(strstr(own, "[start]\n") == own);
  CHECK(strstr(own, "[verse]") != NULL);
  CHECK(strstr(own, "Walking down the street") != NULL);
  free(own);
  char *same = NULL;
  CHECK(music3_normalize_lyrics("[Verse] Walking down the street", &same));
  CHECK(strstr(same, "Walking down the street") == NULL);
  free(same);
  char *p = NULL;
  CHECK(music3_build_prompt("Warm acoustic pop", "[Verse]\nHi", &p));
  CHECK(strncmp(p, "<|im_start|><|caption_start|>", 29) == 0);
  CHECK(strstr(p, "<|audio_start|>") != NULL);
  free(p);
  return 0;
}

static int test_chunk(void) {
  CHECK(music3_aligned_mel_length(250) == 861);
  music3_chunk_window w[8];
  CHECK(music3_chunk_windows(250, w, 8) == 2);
  CHECK(w[0].start == 0 && w[0].end == 200 && w[0].is_first);
  CHECK(w[1].start == 100 && w[1].end == 250 && w[1].is_last);
  CHECK(music3_chunk_windows(150, w, 8) == 1);
  CHECK(w[0].end == 150 && w[0].is_first && w[0].is_last);
  return 0;
}

static int test_snake_dav(void) {
  const float in[] = {0.2f, -0.3f, 0.5f, 0.1f};
  const float a[] = {0.5f, 1.2f};
  float out[4];
  CHECK(music3_snake1d_f32(out, in, a, 1, 2, 2) == 0);
  /* x + sin(a x)^2 / (a+1e-9) */
  float s0 = sinf(0.5f * 0.2f);
  float want0 = 0.2f + (1.f / (0.5f + 1e-9f)) * s0 * s0;
  CHECK(fabsf(out[0] - want0) < 1e-6f);
  music3_wav_host wav = {0};
  char err[64];
  CHECK(music3_dav_synthetic_decode(2, &wav, err, sizeof(err)));
  CHECK(wav.samples == 1024 && wav.channels == 2 && wav.sample_rate == 44100);
  music3_wav_host_free(&wav);
  return 0;
}

int main(void) {
  if (test_prompt())
    return 1;
  if (test_chunk())
    return 1;
  if (test_snake_dav())
    return 1;
  printf("ok music3 weightless\n");
  return 0;
}
