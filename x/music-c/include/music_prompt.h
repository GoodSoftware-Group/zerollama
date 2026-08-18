#ifndef MUSIC_PROMPT_H
#define MUSIC_PROMPT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  MUSIC3_IM_START = 151644,
  MUSIC3_IM_END = 151645,
  MUSIC3_AUDIO_CFG = 151654,
  MUSIC3_AUDIO_START = 151669,
  MUSIC3_AUDIO_END = 151670,
  MUSIC3_CAPTION_START = 151671,
  MUSIC3_CAPTION_END = 151672,
  MUSIC3_LYRICS_START = 151673,
  MUSIC3_LYRICS_END = 151674,
  MUSIC3_AUDIO_CODE_OFFSET = 151675,
  MUSIC3_MAX_PROMPT_TOKENS = 5000,
  MUSIC3_MAX_AUDIO_FRAMES = 9000
};

/* Omni lyrics normalize + caption markdown strip. Caller frees *out.
 * Why Omni not Comfy: Comfy regex-splits tags; Omni drops same-line lyrics after [Verse].
 */
int music3_normalize_lyrics(const char *lyrics, char **out);
int music3_clean_caption(const char *caption, char **out);
int music3_build_prompt(const char *caption, const char *lyrics, char **out);

#ifdef __cplusplus
}
#endif

#endif
