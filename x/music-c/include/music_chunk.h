#ifndef MUSIC_CHUNK_H
#define MUSIC_CHUNK_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
  MUSIC3_AR_CHUNK_FRAMES = 200,
  MUSIC3_AR_CHUNK_HOP_FRAMES = 100,
  MUSIC3_AR_HIDDEN_SIZE = 32768,
  MUSIC3_DAV_SAMPLE_RATE = 44100,
  MUSIC3_OUTPUT_SAMPLE_RATE = 32000,
  MUSIC3_DAV_UPSAMPLE = 512,
  MUSIC3_SR_INPUT = 24000,
  MUSIC3_HOP_IN = 960,
  MUSIC3_HOP_OUT = 512,
  MUSIC3_LATENT_CHANNELS = 128
};

typedef struct {
  int index;
  int start;
  int end;
  int is_first;
  int is_last;
} music3_chunk_window;

int music3_aligned_mel_length(int frames);

/* Fill up to max_windows. Returns count, or -1 if max_windows is too small. */
int music3_chunk_windows(int frames, music3_chunk_window *out, int max_windows);

#ifdef __cplusplus
}
#endif

#endif
