/*
 * encode_mp4.h — write RGB frames via ffmpeg shell-out.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int encode_mp4_from_rgb(const char *out_mp4, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats);

#ifdef __cplusplus
}
#endif
