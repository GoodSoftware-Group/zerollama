#include "encode_mp4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int clamp_u8(float v) {
  if (v < 0.0f)
    return 0;
  if (v > 255.0f)
    return 255;
  return (int)(v + 0.5f);
}

static int clamp_s16(float v) {
  int x = (int)(v * 32767.f + (v >= 0.f ? 0.5f : -0.5f));
  if (x < -32768)
    return -32768;
  if (x > 32767)
    return 32767;
  return x;
}

static int wr4(FILE *fp, const char *tag) {
  return fwrite(tag, 1, 4, fp) == 4 ? 0 : -1;
}

static int wr32(FILE *fp, uint32_t v) {
  unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                        (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
  return fwrite(b, 1, 4, fp) == 4 ? 0 : -1;
}

static int wr16(FILE *fp, uint16_t v) {
  unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
  return fwrite(b, 1, 2, fp) == 2 ? 0 : -1;
}

static int patch32(FILE *fp, long pos, uint32_t v) {
  long cur = ftell(fp);
  if (cur < 0 || fseek(fp, pos, SEEK_SET) != 0 || wr32(fp, v) != 0 ||
      fseek(fp, cur, SEEK_SET) != 0)
    return -1;
  return 0;
}

static uint32_t row_stride(int width) {
  return (uint32_t)((width * 3 + 3) & ~3);
}

int encode_avi_from_rgb(const char *out_avi, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats,
                        const float *pcm_ch_major, int pcm_channels,
                        int pcm_samples, int pcm_rate) {
  size_t need = (size_t)width * (size_t)height * (size_t)frames * 3;
  if (!out_avi || !rgb || nfloats < need || width < 2 || height < 2 ||
      frames < 1 || fps < 1)
    return -1;
  int has_audio = pcm_ch_major && pcm_channels == 2 && pcm_samples > 0 &&
                  pcm_rate > 0;
  uint32_t stride = row_stride(width);
  uint32_t frame_bytes = stride * (uint32_t)height;
  uint32_t frame_chunk = (frame_bytes + 1u) & ~1u;
  uint32_t pcm_bytes =
      has_audio ? (uint32_t)pcm_samples * (uint32_t)pcm_channels * 2u : 0;
  uint32_t pcm_chunk = (pcm_bytes + 1u) & ~1u;
  int streams = has_audio ? 2 : 1;

  FILE *fp = fopen(out_avi, "wb");
  if (!fp)
    return -1;

  if (wr4(fp, "RIFF") != 0 || wr32(fp, 0) != 0 || wr4(fp, "AVI ") != 0)
    goto fail;
  long hdrl_list = ftell(fp);
  if (wr4(fp, "LIST") != 0 || wr32(fp, 0) != 0 || wr4(fp, "hdrl") != 0)
    goto fail;
  if (wr4(fp, "avih") != 0 || wr32(fp, 56) != 0)
    goto fail;
  uint32_t usec = (uint32_t)(1000000 / fps);
  if (wr32(fp, usec) != 0 || wr32(fp, frame_bytes * (uint32_t)fps) != 0 ||
      wr32(fp, 0) != 0 || wr32(fp, 0x10) != 0 || wr32(fp, (uint32_t)frames) != 0 ||
      wr32(fp, 0) != 0 || wr32(fp, (uint32_t)streams) != 0 ||
      wr32(fp, frame_chunk) != 0 || wr32(fp, (uint32_t)width) != 0 ||
      wr32(fp, (uint32_t)height) != 0)
    goto fail;
  for (int i = 0; i < 4; i++)
    if (wr32(fp, 0) != 0)
      goto fail;

  if (wr4(fp, "LIST") != 0 || wr32(fp, 116) != 0 || wr4(fp, "strl") != 0)
    goto fail;
  if (wr4(fp, "strh") != 0 || wr32(fp, 56) != 0 || wr4(fp, "vids") != 0 ||
      wr4(fp, "DIB ") != 0)
    goto fail;
  if (wr32(fp, 0) != 0 || wr32(fp, 0) != 0 || wr32(fp, 0) != 0 || wr32(fp, 1) != 0 ||
      wr32(fp, (uint32_t)fps) != 0 || wr32(fp, 0) != 0 ||
      wr32(fp, (uint32_t)frames) != 0 || wr32(fp, frame_chunk) != 0 ||
      wr32(fp, 0xffffffffu) != 0 || wr32(fp, 0) != 0 || wr16(fp, 0) != 0 ||
      wr16(fp, 0) != 0 || wr16(fp, (uint16_t)width) != 0 ||
      wr16(fp, (uint16_t)height) != 0)
    goto fail;
  if (wr4(fp, "strf") != 0 || wr32(fp, 40) != 0)
    goto fail;
  if (wr32(fp, 40) != 0 || wr32(fp, (uint32_t)width) != 0 ||
      wr32(fp, (uint32_t)height) != 0 || wr16(fp, 1) != 0 || wr16(fp, 24) != 0 ||
      wr32(fp, 0) != 0 || wr32(fp, frame_bytes) != 0 || wr32(fp, 0) != 0 ||
      wr32(fp, 0) != 0 || wr32(fp, 0) != 0 || wr32(fp, 0) != 0)
    goto fail;

  if (has_audio) {
    if (wr4(fp, "LIST") != 0 || wr32(fp, 92) != 0 || wr4(fp, "strl") != 0)
      goto fail;
    if (wr4(fp, "strh") != 0 || wr32(fp, 56) != 0 || wr4(fp, "auds") != 0 ||
        wr32(fp, 1) != 0)
      goto fail;
    if (wr32(fp, 0) != 0 || wr32(fp, 0) != 0 || wr32(fp, 0) != 0 || wr32(fp, 1) != 0 ||
        wr32(fp, (uint32_t)pcm_rate) != 0 || wr32(fp, 0) != 0 ||
        wr32(fp, (uint32_t)pcm_samples) != 0 || wr32(fp, pcm_chunk) != 0 ||
        wr32(fp, 0xffffffffu) != 0 || wr32(fp, 0) != 0 || wr16(fp, 0) != 0 ||
        wr16(fp, 0) != 0 || wr16(fp, 0) != 0 || wr16(fp, 0) != 0)
      goto fail;
    if (wr4(fp, "strf") != 0 || wr32(fp, 16) != 0)
      goto fail;
    if (wr16(fp, 1) != 0 || wr16(fp, 2) != 0 || wr32(fp, (uint32_t)pcm_rate) != 0 ||
        wr32(fp, (uint32_t)pcm_rate * 4u) != 0 || wr16(fp, 4) != 0 ||
        wr16(fp, 16) != 0)
      goto fail;
  }

  long hdrl_end = ftell(fp);
  if (hdrl_end < 0 ||
      patch32(fp, hdrl_list + 4, (uint32_t)(hdrl_end - hdrl_list - 8)) != 0)
    goto fail;

  long movi_list = ftell(fp);
  if (wr4(fp, "LIST") != 0 || wr32(fp, 0) != 0 || wr4(fp, "movi") != 0)
    goto fail;

  uint32_t *idx_off = (uint32_t *)malloc((size_t)(frames + (has_audio ? 1 : 0)) *
                                         sizeof(uint32_t));
  uint32_t *idx_sz = (uint32_t *)malloc((size_t)(frames + (has_audio ? 1 : 0)) *
                                        sizeof(uint32_t));
  if (!idx_off || !idx_sz)
    goto fail_idx;
  int nidx = 0;
  unsigned char *row = (unsigned char *)malloc(stride);
  if (!row)
    goto fail_idx;
  memset(row, 0, stride);

  for (int f = 0; f < frames; f++) {
    long chunk_pos = ftell(fp);
    if (chunk_pos < 0)
      goto fail_row;
    idx_off[nidx] = (uint32_t)(chunk_pos - movi_list);
    idx_sz[nidx] = frame_bytes;
    nidx++;
    if (wr4(fp, "00dc") != 0 || wr32(fp, frame_bytes) != 0)
      goto fail_row;
    const float *src = rgb + (size_t)f * (size_t)width * (size_t)height * 3;
    for (int y = height - 1; y >= 0; y--) {
      const float *line = src + (size_t)y * (size_t)width * 3;
      for (int x = 0; x < width; x++) {
        row[x * 3 + 0] = (unsigned char)clamp_u8(line[x * 3 + 2]);
        row[x * 3 + 1] = (unsigned char)clamp_u8(line[x * 3 + 1]);
        row[x * 3 + 2] = (unsigned char)clamp_u8(line[x * 3 + 0]);
      }
      if (fwrite(row, 1, stride, fp) != stride)
        goto fail_row;
    }
    if (frame_chunk > frame_bytes) {
      unsigned char z = 0;
      if (fwrite(&z, 1, 1, fp) != 1)
        goto fail_row;
    }
  }
  if (has_audio) {
    long chunk_pos = ftell(fp);
    if (chunk_pos < 0)
      goto fail_row;
    idx_off[nidx] = (uint32_t)(chunk_pos - movi_list);
    idx_sz[nidx] = pcm_bytes;
    nidx++;
    if (wr4(fp, "01wb") != 0 || wr32(fp, pcm_bytes) != 0)
      goto fail_row;
    for (int s = 0; s < pcm_samples; s++) {
      int16_t l = (int16_t)clamp_s16(pcm_ch_major[s]);
      int16_t r = (int16_t)clamp_s16(pcm_ch_major[(size_t)pcm_samples + (size_t)s]);
      unsigned char b[4] = {(unsigned char)l, (unsigned char)((uint16_t)l >> 8),
                            (unsigned char)r, (unsigned char)((uint16_t)r >> 8)};
      if (fwrite(b, 1, 4, fp) != 4)
        goto fail_row;
    }
    if (pcm_chunk > pcm_bytes) {
      unsigned char z = 0;
      if (fwrite(&z, 1, 1, fp) != 1)
        goto fail_row;
    }
  }
  free(row);
  row = NULL;
  long movi_end = ftell(fp);
  if (movi_end < 0 ||
      patch32(fp, movi_list + 4, (uint32_t)(movi_end - movi_list - 8)) != 0)
    goto fail_idx;

  if (wr4(fp, "idx1") != 0 || wr32(fp, (uint32_t)nidx * 16u) != 0)
    goto fail_idx;
  for (int i = 0; i < nidx; i++) {
    if (wr4(fp, i + 1 == nidx && has_audio ? "01wb" : "00dc") != 0)
      goto fail_idx;
    if (wr32(fp, 0x10) != 0 || wr32(fp, idx_off[i]) != 0 || wr32(fp, idx_sz[i]) != 0)
      goto fail_idx;
  }
  long end = ftell(fp);
  if (end < 0 || patch32(fp, 4, (uint32_t)(end - 8)) != 0)
    goto fail_idx;
  free(idx_off);
  free(idx_sz);
  fclose(fp);
  return 0;

fail_row:
  free(row);
fail_idx:
  free(idx_off);
  free(idx_sz);
fail:
  fclose(fp);
  remove(out_avi);
  return -1;
}

static int find_ffmpeg(char *dst, size_t dst_n) {
  const char *home = getenv("HOME");
  const char *cands[8];
  int n = 0;
  cands[n++] = "ffmpeg";
  if (home) {
    static char hb[512];
    snprintf(hb, sizeof(hb), "%s/.homebrew/bin/ffmpeg", home);
    cands[n++] = hb;
  }
  cands[n++] = "/opt/homebrew/bin/ffmpeg";
  cands[n++] = "/usr/local/bin/ffmpeg";
  for (int i = 0; i < n; i++) {
    if (strchr(cands[i], '/')) {
      if (access(cands[i], X_OK) == 0) {
        snprintf(dst, dst_n, "%s", cands[i]);
        return 0;
      }
    } else {
      char which[640];
      snprintf(which, sizeof(which), "command -v %s >/dev/null 2>&1", cands[i]);
      if (system(which) == 0) {
        snprintf(dst, dst_n, "%s", cands[i]);
        return 0;
      }
    }
  }
  return -1;
}

static int write_pcm_s16le_interleaved(const char *path, const float *pcm,
                                       int channels, int samples) {
  FILE *fp = fopen(path, "wb");
  if (!fp)
    return -1;
  for (int s = 0; s < samples; s++) {
    for (int c = 0; c < channels; c++) {
      int16_t v = (int16_t)clamp_s16(pcm[(size_t)c * (size_t)samples + (size_t)s]);
      unsigned char b[2] = {(unsigned char)v, (unsigned char)((uint16_t)v >> 8)};
      if (fwrite(b, 1, 2, fp) != 2) {
        fclose(fp);
        return -1;
      }
    }
  }
  fclose(fp);
  return 0;
}

static void avi_path_from_mp4(const char *out_mp4, char *dst, size_t dst_n) {
  snprintf(dst, dst_n, "%s", out_mp4);
  size_t n = strlen(dst);
  if (n >= 4) {
    const char *ext = dst + n - 4;
    if (!strcmp(ext, ".mp4") || !strcmp(ext, ".MP4")) {
      dst[n - 3] = 'a';
      dst[n - 2] = 'v';
      dst[n - 1] = 'i';
      return;
    }
  }
  snprintf(dst, dst_n, "%s.avi", out_mp4);
}

int encode_mp4_from_rgb(const char *out_mp4, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats) {
  return encode_mp4_from_rgb_pcm(out_mp4, width, height, frames, fps, rgb,
                                 nfloats, NULL, 0, 0, 0);
}

int encode_mp4_from_rgb_pcm(const char *out_mp4, int width, int height,
                            int frames, int fps, const float *rgb,
                            size_t nfloats, const float *pcm_ch_major,
                            int pcm_channels, int pcm_samples, int pcm_rate) {
  size_t need = (size_t)width * (size_t)height * (size_t)frames * 3;
  if (!out_mp4 || !rgb || nfloats < need)
    return -1;

  int has_audio = pcm_ch_major && pcm_channels == 2 && pcm_samples > 0 &&
                  pcm_rate > 0;

  char ff[512];
  if (find_ffmpeg(ff, sizeof(ff)) == 0) {
    char raw_path[] = "/tmp/wan_frames_XXXXXX";
    int fd = mkstemp(raw_path);
    if (fd < 0)
      return -1;
    FILE *raw = fdopen(fd, "wb");
    if (!raw) {
      close(fd);
      remove(raw_path);
      return -1;
    }
    for (size_t i = 0; i < need; i += 3) {
      unsigned char px[3];
      px[0] = (unsigned char)clamp_u8(rgb[i + 0]);
      px[1] = (unsigned char)clamp_u8(rgb[i + 1]);
      px[2] = (unsigned char)clamp_u8(rgb[i + 2]);
      if (fwrite(px, 1, 3, raw) != 3) {
        fclose(raw);
        remove(raw_path);
        return -1;
      }
    }
    fclose(raw);

    char pcm_path[] = "/tmp/wan_pcm_XXXXXX";
    int pfd = -1;
    if (has_audio) {
      pfd = mkstemp(pcm_path);
      if (pfd < 0) {
        remove(raw_path);
        return -1;
      }
      close(pfd);
      if (write_pcm_s16le_interleaved(pcm_path, pcm_ch_major, pcm_channels,
                                      pcm_samples) != 0) {
        remove(raw_path);
        remove(pcm_path);
        return -1;
      }
    }

    char cmd[4096];
    if (has_audio)
      snprintf(cmd, sizeof(cmd),
               "'%s' -y -loglevel error -f rawvideo -pix_fmt rgb24 "
               "-s %dx%d -r %d -i '%s' -f s16le -ar %d -ac 2 -i '%s' "
               "-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '%s'",
               ff, width, height, fps, raw_path, pcm_rate, pcm_path, out_mp4);
    else
      snprintf(cmd, sizeof(cmd),
               "'%s' -y -loglevel error -f rawvideo -pix_fmt rgb24 "
               "-s %dx%d -r %d -i '%s' -c:v libx264 -pix_fmt yuv420p '%s'",
               ff, width, height, fps, raw_path, out_mp4);

    int rc = system(cmd);
    remove(raw_path);
    if (has_audio)
      remove(pcm_path);
    if (rc == 0)
      return 0;
  }

  char avi[1024];
  avi_path_from_mp4(out_mp4, avi, sizeof(avi));
  if (encode_avi_from_rgb(avi, width, height, frames, fps, rgb, nfloats,
                          has_audio ? pcm_ch_major : NULL,
                          has_audio ? pcm_channels : 0,
                          has_audio ? pcm_samples : 0,
                          has_audio ? pcm_rate : 0) != 0)
    return -1;
  fprintf(stderr,
          "video-c: ffmpeg unavailable; wrote %s "
          "(%dx%d x %d @ %d fps AVI). Install ffmpeg for mp4.\n",
          avi, width, height, frames, fps);
  return 0;
}
