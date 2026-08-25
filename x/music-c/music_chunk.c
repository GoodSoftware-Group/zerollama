#include "music_chunk.h"

int music3_aligned_mel_length(int frames) {
  if (frames < 1)
    return 1;
	/* Omni: int(frames * 44100 / 24000 * 960 / 512) with float intermediates.
	 * Why double: integer-only truncates 250 → 859 instead of Omni 861.
	 */
  int n = (int)((double)frames * (double)MUSIC3_DAV_SAMPLE_RATE /
                (double)MUSIC3_SR_INPUT * (double)MUSIC3_HOP_IN /
                (double)MUSIC3_HOP_OUT);
  return n < 1 ? 1 : n;
}

int music3_chunk_windows(int frames, music3_chunk_window *out, int max_windows) {
  if (frames < 0 || !out || max_windows < 1)
    return -1;
  if (frames == 0)
    return 0;
  if (frames <= MUSIC3_AR_CHUNK_FRAMES) {
    out[0].index = 0;
    out[0].start = 0;
    out[0].end = frames;
    out[0].is_first = 1;
    out[0].is_last = 1;
    return 1;
  }
  int n = 0;
  int index = 0;
  int start = 0;
  while (start < frames) {
    if (n >= max_windows)
      return -1;
    int end = start + MUSIC3_AR_CHUNK_FRAMES;
    if (end > frames)
      end = frames;
    out[n].index = index;
    out[n].start = start;
    out[n].end = end;
    out[n].is_first = index == 0;
    out[n].is_last = end >= frames;
    n++;
    if (end >= frames)
      break;
    index++;
    start += MUSIC3_AR_CHUNK_HOP_FRAMES;
  }
  return n;
}
