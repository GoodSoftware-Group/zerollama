/*
 * encode_mp4.h — write RGB frames via ffmpeg shell-out.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int encode_avi_from_rgb(const char *out_avi, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats,
                        const float *pcm_ch_major, int pcm_channels,
                        int pcm_samples, int pcm_rate);

int encode_mp4_from_rgb(const char *out_mp4, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats);

/* Optional stereo channel-major PCM; muxes AVI when ffmpeg cannot. */
int encode_mp4_from_rgb_pcm(const char *out_mp4, int width, int height,
                            int frames, int fps, const float *rgb,
                            size_t nfloats, const float *pcm_ch_major,
                            int pcm_channels, int pcm_samples, int pcm_rate);

#ifdef __cplusplus
}
#endif
