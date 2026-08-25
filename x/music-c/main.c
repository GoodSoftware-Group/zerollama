#include "music_chunk.h"
#include "music_dav.h"
#include "music_prompt.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(void) {
  fprintf(stderr,
          "music-cli --info|--tokenize|--decode-audio [options]\n"
          "  --info           geometry + inventory of DIR (safetensors/.pth)\n"
          "  --tokenize       pack Omni prompt string (no BPE / BMTL yet)\n"
          "  --decode-audio   synthetic zeros WAV (not dav.pth until export)\n"
          "  --ckpt-dir DIR   MiniMax-Music3 or mlx pack\n"
          "  --caption TEXT\n"
          "  --lyrics TEXT\n"
          "  --latent-t N     synthetic DAV frames (default 2)\n"
          "  --out PATH       WAV output\n");
}

static int cmd_info(const char *dir) {
  printf("family=music3\n");
  printf("ar_hidden=%d dav_sr=%d out_sr=%d upsample=%d\n", MUSIC3_AR_HIDDEN_SIZE,
         MUSIC3_DAV_SAMPLE_RATE, MUSIC3_OUTPUT_SAMPLE_RATE, MUSIC3_DAV_UPSAMPLE);
  printf("special im_start=%d audio_cfg=%d audio_start=%d audio_code_offset=%d\n",
         MUSIC3_IM_START, MUSIC3_AUDIO_CFG, MUSIC3_AUDIO_START,
         MUSIC3_AUDIO_CODE_OFFSET);
  if (!dir || !dir[0])
    return 0;
  printf("ckpt_dir=%s\n", dir);
  DIR *d = opendir(dir);
  if (!d) {
    printf("ckpt_dir_readable=no\n");
    return 0;
  }
  struct dirent *ent;
  int n_st = 0, n_pth = 0;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.')
      continue;
    char path[1400];
    snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
    size_t n = strlen(ent->d_name);
    if (n > 12 && !strcmp(ent->d_name + n - 12, ".safetensors")) {
      printf("safetensors=%s\n", ent->d_name);
      n_st++;
    } else if (n > 4 && !strcmp(ent->d_name + n - 4, ".pth")) {
      printf("pth=%s readable=%s\n", ent->d_name,
             access(path, R_OK) == 0 ? "yes" : "no");
      n_pth++;
    } else if (!strcmp(ent->d_name, "config.json") ||
               !strcmp(ent->d_name, "tokenizer") ||
               !strcmp(ent->d_name, "qwen_7B") ||
               !strcmp(ent->d_name, "dav.pth")) {
      printf("entry=%s\n", ent->d_name);
    }
  }
  closedir(d);
  printf("inventory safetensors=%d pth=%d\n", n_st, n_pth);
  return 0;
}

static int cmd_tokenize(const char *caption, const char *lyrics) {
  char *prompt = NULL;
  if (!music3_build_prompt(caption ? caption : "", lyrics ? lyrics : "",
                           &prompt)) {
    fprintf(stderr, "build_prompt failed\n");
    return 1;
  }
  fputs(prompt, stdout);
  fputc('\n', stdout);
  free(prompt);
  return 0;
}

static int cmd_decode(int latent_t, const char *out) {
  music3_wav_host wav = {0};
  char err[256];
  if (!music3_dav_synthetic_decode(latent_t, &wav, err, sizeof(err))) {
    fprintf(stderr, "decode: %s\n", err);
    return 1;
  }
  printf("synthetic dav latent_t=%d samples=%d sr=%d\n", latent_t, wav.samples,
         wav.sample_rate);
  if (out && out[0]) {
    if (!music3_wav_write(&wav, out, err, sizeof(err))) {
      fprintf(stderr, "wav: %s\n", err);
      music3_wav_host_free(&wav);
      return 1;
    }
    printf("wrote %s\n", out);
  }
  music3_wav_host_free(&wav);
  return 0;
}

int main(int argc, char **argv) {
  const char *ckpt = NULL;
  const char *caption = "Warm acoustic pop";
  const char *lyrics = "[Verse]\nMorning light";
  const char *out = NULL;
  int latent_t = 2;
  int do_info = 0, do_tok = 0, do_dec = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--info"))
      do_info = 1;
    else if (!strcmp(argv[i], "--tokenize"))
      do_tok = 1;
    else if (!strcmp(argv[i], "--decode-audio"))
      do_dec = 1;
    else if (!strcmp(argv[i], "--ckpt-dir") && i + 1 < argc)
      ckpt = argv[++i];
    else if (!strcmp(argv[i], "--caption") && i + 1 < argc)
      caption = argv[++i];
    else if (!strcmp(argv[i], "--lyrics") && i + 1 < argc)
      lyrics = argv[++i];
    else if (!strcmp(argv[i], "--out") && i + 1 < argc)
      out = argv[++i];
    else if (!strcmp(argv[i], "--latent-t") && i + 1 < argc)
      latent_t = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      usage();
      return 0;
    } else {
      fprintf(stderr, "unknown arg %s\n", argv[i]);
      usage();
      return 2;
    }
  }
  if (!do_info && !do_tok && !do_dec) {
    usage();
    return 2;
  }
  int rc = 0;
  if (do_info)
    rc |= cmd_info(ckpt);
  if (do_tok)
    rc |= cmd_tokenize(caption, lyrics);
  if (do_dec)
    rc |= cmd_decode(latent_t, out);
  return rc;
}
