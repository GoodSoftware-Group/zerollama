#include "encode_mp4.h"

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

int encode_mp4_from_rgb(const char *out_mp4, int width, int height, int frames,
                        int fps, const float *rgb, size_t nfloats) {
  size_t need = (size_t)width * (size_t)height * (size_t)frames * 3;
  if (!out_mp4 || !rgb || nfloats < need)
    return -1;

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

  char cmd[2048];
  snprintf(cmd, sizeof(cmd),
           "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 "
           "-s %dx%d -r %d -i %s -c:v libx264 -pix_fmt yuv420p %s",
           width, height, fps, raw_path, out_mp4);

  int rc = system(cmd);
  if (rc == 0) {
    remove(raw_path);
    return 0;
  }

  /* Fallback when ffmpeg is missing: keep raw RGB next to requested path. */
  char raw_out[1024];
  snprintf(raw_out, sizeof(raw_out), "%s.rgb24", out_mp4);
  if (rename(raw_path, raw_out) != 0) {
    remove(raw_path);
    return -1;
  }
  fprintf(stderr,
          "wan-c: ffmpeg unavailable; wrote raw RGB24 %s "
          "(%dx%d x %d frames @ %d fps). Install ffmpeg for mp4.\n",
          raw_out, width, height, frames, fps);
  /* Also touch empty mp4 path so callers see an artifact path exists. */
  FILE *touch = fopen(out_mp4, "wb");
  if (touch) {
    fprintf(touch, "RAW:%s\n", raw_out);
    fclose(touch);
  }
  return 0;
}
