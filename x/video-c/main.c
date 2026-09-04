#include "encode_mp4.h"
#include "h3_adaln_host.h"
#include "h3_audio_vae_decode.h"
#include "h3_audio_vae_host.h"
#include "h3_clipproj_host.h"
#include "h3_dit_forward.h"
#include "h3_dit_pack.h"
#include "h3_dit_host.h"
#include "h3_host.h"
#include "h3_info.h"
#include "h3_prof.h"
#include "h3_video_vae_decode.h"
#include "h3_video_vae_encode.h"
#include "h3_st_store.h"
#include "h3_present.h"
#include "h3_text_cond.h"
#include "h3_qwen_te_4b.h"
#include "h3_qwen_te_host.h"
#include "h3_tokenizer.h"
#include "h3_video_vae_host.h"
#include "wan.h"
#include "wan_profile.h"
#include "wan_config.h"

#include <math.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

typedef enum {
  VIDEO_FAMILY_WAN = 0,
  VIDEO_FAMILY_H3 = 1,
} video_family_t;

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s --ckpt-dir DIR [options]\n"
          "\n"
          "  --family wan|h3     Model family (default: wan)\n"
          "  --info              Probe checkpoint layout and exit (h3: VAE+tokenizer)\n"
          "  --decode-audio      H3: host AudioVAE decode → WAV (needs FL2VA/audio_vae)\n"
          "  --encode-audio      H3: host AudioVAE encode 800-sample unit PCM → latent stats\n"
          "  --encode-video      H3: host video VAE CNN encode (default 32x32x1; max 512, tiled >tile)\n"
          "                      --in frame.ppm uses that image; with -o: pad T=2, decode, write mp4/ppm\n"
          "  --decode-video      H3: host ViT decode; -o out.mp4 or frame.ppm\n"
          "  --clipproj [PATH]   H3: apply ClipProj to unit hidden (default celeb-mlp cache)\n"
          "  --tokenize          H3: BMTL/Qwen encode --prompt (blob in third_party/h3)\n"
          "  --embed             H3: present + Qwen3-VL-4B TE (or hash) + ClipProj\n"
          "  --present           H3: print FL2VA/t2va token ids, AdaLN tags, mRoPE\n"
          "  --pictures N        H3 present/embed: N dummy keyframes (<Picture i> + pads)\n"
          "  --merge-h N --merge-w N  Merged Qwen grid per picture (default 2)\n"
          "  --latent-t N        Audio decode T (default 4); video decode 2 or T>=7, (T-2) mod 5 = 0\n"
          "  -d / --ckpt-dir PATH  Checkpoint / MiniMax-H3 directory\n"
          "  --uma-sock PATH     UMA broker socket (default: UMA_SOCK env)\n"
          "  --prompt TEXT       Positive prompt (wan generate)\n"
          "  --negative-prompt T Negative prompt\n"
          "  --width N           Output width (default 832)\n"
          "  --height N          Output height (default 480)\n"
          "  --frames N          Frame count (default 49)\n"
          "  --steps N           Diffusion steps (default 25)\n"
          "  --cfg F             Wan CFG (H3 is Comfy BasicGuider: cond only)\n"
          "  --shift F           Flow sigma shift (default 5.0)\n"
          "  --seed N            RNG seed (0=auto)\n"
          "  --lora PATH         Wan: merge LoRA safetensors into DiT weights\n"
          "                      (dotted PEFT/ComfyUI keys; merge-at-load)\n"
          "  --lora-scale F      Wan: LoRA strength multiplier (default 1.0)\n"
          "  --solver unipc|dpmpp  Solver (default unipc)\n"
          "  --dtype f32|f16     Compute dtype hint (default f16)\n"
          "  --fps N             Output fps (default 16)\n"
          "  --vocab PATH        Binary umt5.vocab from export_umt5_spm.py\n"
          "  --dit-denoise       H3: 4-token T2VA Euler (layout RoPE; --steps/--layers/--seed)\n"
          "  --generate          H3: T2VA (default 5x32; --width 768 → canvas; --prompt TE+ClipProj)\n"
          "  --text-cond PATH    H3: load [nt,5120] dump (H3TE) instead of 4B+ClipProj; or H3_TEXT_COND\n"
          "  --serve-sock PATH   H3: run as resident daemon keeping all weight caches in RAM\n"
          "                      (serves tab-separated requests on a Unix socket; H3_MLOCK=1 pins them)\n"
          "                      request: out_mp4\\tprompt\\tframes\\twidth\\theight\\tseed\\tsteps\\tlayers\\treuse\\tadaln_t_sigma\n"
          "  --layers N          H3 DiT layers (--generate default 50; --dit-denoise default 1)\n"
          "  --reuse N           H3 velocity reuse: 1 every step, 2 fast, 3 aggressive\n"
          "  --adaln-t-sigma N   H3 AdaLN time index: 0 = t=1-σ (default), 1 = t=σ, -1 = H3_ADALN_T_SIGMA env\n"
          "  --ssd-streaming     H3 SSD BF16 stream (Metal/h3.c; ignored on host ConvRot)\n"
          "  --out / -o PATH     Output mp4 (wan, h3 --decode-video, --encode-video), wav, or ppm\n"
          "  --in PATH           H3 --encode-video: binary P6 PPM (RGB)\n"
          "  --validate-only     Check wan params and exit\n"
          "\n"
          "Clients pick a model tag via /v1/videos; operators may set\n"
          "ZEROLLAMA_VIDEO_CLI (or ZEROLLAMA_WAN_CLI) for this binary.\n"
          "\n"
          "Env: UMA_WAN_LOCAL=1  host uma_wan_ops (no broker)\n"
          "     WAN_T5_CACHE=0   disable prompt→embed disk cache (t5_cache.c;\n"
          "                      default on, ~/.zerollama/cache/wan_t5)\n"
          "     WAN_T5_CACHE_DIR PATH  override cache dir\n"
          "     UMA_WAN_EXT=1    prefer EXT_CALL for LN/AdaLN/GN (needs opworker)\n"
          "     UMA_EXT_SOCK     EXT worker socket (default /tmp/uma_ext_wan.sock)\n"
          "     WAN_DIT_NO_PERSIST=1  skip F0994 block BANK (host FFN)\n"
          "     WAN_DIT_HOST_FFN=1    force host FFN (implies no FFN_GELU)\n"
          "     WAN_DIT_MIRROR=1      per-block token GET/PUT (debug)\n"
          "     WAN_DIT_QCHUNK=N      ATTN t= window rows (default 5460 if T>)\n"
          "     WAN_DIT_FFN_CHUNK=N   FFN_GELU t= window rows (default 4096)\n"
          "     WAN_VAE_NO_HEADT=1    skip broker HEADT (F1001–F1004)\n"
          "     WAN_VAE_WARM_HEADT=1  warm feat_cache on broker (F1012 mid/out;\n"
          "                          opt-in — default-on miss @832). SHUTTLE=1=old\n"
          "     WAN_VAE_NO_WARM_HEADT=1  force host warm after cold\n"
          "     WAN_VAE_NO_RESID_FUSE=1  legacy dual HEADT + host ADD\n"
          "     WAN_DIT_NO_GATED_RESID=1  legacy AFFINE+RESIDUAL (no Metal gate)\n"
          "     WAN_PROFILE=1         stage wall timers (stderr summary)\n"
          "     WAN_VAE_STAGE_PROF=1  with WAN_PROFILE: VAE tip stage map\n"
          "     WAN_BUF_STICKY=0      re-assert BUF_ALLOC every call\n",
          argv0);
}

static int parse_solver(const char *s, wan_solver_t *out) {
  if (!s || !out)
    return -1;
  if (!strcmp(s, "unipc")) {
    *out = WAN_SOLVER_UNIPC;
    return 0;
  }
  if (!strcmp(s, "dpmpp")) {
    *out = WAN_SOLVER_DPMPP;
    return 0;
  }
  return -1;
}

static int parse_dtype(const char *s, wan_dtype_t *out) {
  if (!s || !out)
    return -1;
  if (!strcmp(s, "f32")) {
    *out = WAN_DTYPE_F32;
    return 0;
  }
  if (!strcmp(s, "f16")) {
    *out = WAN_DTYPE_F16;
    return 0;
  }
  return -1;
}

static int path_has_ext(const char *path, const char *ext) {
  if (!path || !ext)
    return 0;
  size_t n = strlen(path), e = strlen(ext);
  return n >= e && !strcasecmp(path + n - e, ext);
}

static int write_h3_video_media(const char *ckpt_dir, const char *out_path,
                                const h3_video_frames_host *frames, int fps) {
  char error[1024];
  if (path_has_ext(out_path, ".ppm")) {
    if (!h3_video_frames_write_ppm(frames, 0, out_path, error, sizeof(error))) {
      fprintf(stderr, "video-c: ppm write failed: %s\n", error);
      return 1;
    }
    printf("video-c: wrote %s (frame 0, %dx%d)\n", out_path, frames->width,
           frames->height);
    return 0;
  }
  size_t pn = (size_t)frames->frames * frames->height * frames->width * 3;
  float *scaled = (float *)malloc(pn * sizeof(float));
  if (!scaled)
    return 1;
  for (size_t i = 0; i < pn; i++)
    scaled[i] = frames->rgb[i] * 255.f;
  h3_audio_waveform_host wav;
  memset(&wav, 0, sizeof(wav));
  char avae[1100];
  snprintf(avae, sizeof(avae), "%s/FL2VA/audio_vae/model.safetensors", ckpt_dir);
  int at = (frames->frames * 40 + fps - 1) / fps;
  if (at < 1)
    at = 1;
  if (access(avae, R_OK) == 0) {
    snprintf(avae, sizeof(avae), "%s/FL2VA/audio_vae", ckpt_dir);
    size_t an =
        (size_t)H3_AUDIO_VAE_LATENT_CHANNELS * H3_AUDIO_VAE_STEREO * (size_t)at;
    float *alatent = (float *)calloc(an, sizeof(float));
    if (alatent) {
      h3_audio_vae_fill_unit_latent(alatent, an);
      if (!h3_audio_vae_decode_host(avae, alatent, at, &wav, error,
                                    sizeof(error))) {
        fprintf(stderr, "video-c: audio mux skipped: %s\n", error);
        h3_audio_waveform_host_free(&wav);
        memset(&wav, 0, sizeof(wav));
      }
      free(alatent);
    }
  }
  int rc = encode_mp4_from_rgb_pcm(out_path, frames->width, frames->height,
                                   frames->frames, fps, scaled, pn, wav.pcm,
                                   wav.channels, wav.samples, wav.sample_rate);
  h3_audio_waveform_host_free(&wav);
  free(scaled);
  if (rc != 0) {
    fprintf(stderr, "video-c: media encode failed\n");
    return 1;
  }
  printf("video-c: wrote media for %s (%dx%d x%d @ %d fps)\n", out_path,
         frames->width, frames->height, frames->frames, fps);
  return 0;
}

static int parse_family(const char *s, video_family_t *out) {
  if (!s || !out)
    return -1;
  if (!strcmp(s, "wan")) {
    *out = VIDEO_FAMILY_WAN;
    return 0;
  }
  if (!strcmp(s, "h3")) {
    *out = VIDEO_FAMILY_H3;
    return 0;
  }
  return -1;
}

static int h3_run_generate(const char *prompt, int frames, int width, int height,
                           uint64_t seed, int steps, int layers, int reuse,
                           int adaln_t_sigma, const char *ckpt_dir,
                           const char *clipproj_path, const char *out_mp4,
                           int fps, char *error, size_t error_size) {
  char path[768];
  if (!h3_resolve_dit_pack_path(path, sizeof(path))) {
    fprintf(stderr, "video-c: HOME unset (need H3_DIT_ST)\n");
    return 2;
  }
  int req_frames = H3_DIT_TINY_FRAMES;
  if (frames > 0 && frames != 49)
    req_frames = frames;
  int pw = H3_DIT_TINY_PIXEL, ph = H3_DIT_TINY_PIXEL;
  if (width > 0 || height > 0) {
    pw = width > 0 ? width : height;
    ph = height > 0 ? height : width;
  }
  h3_dit_t2va_geom geom;
  if (h3_dit_t2va_geom_build(pw, ph, req_frames, &geom) != 0) {
    fprintf(stderr, "video-c: bad H3 canvas %dx%d frames=%d\n", pw, ph,
            req_frames);
    return 2;
  }
  int n_layers = layers > 0 ? layers : H3_DIT_DEFAULT_GENERATE_LAYERS;
  if (layers <= 0) {
    const char *hl = getenv("VIDEO_H3_LAYERS");
    if (!hl || !hl[0])
      hl = getenv("H3_DIT_LAYERS");
    if (hl && hl[0]) {
      int n = atoi(hl);
      if (n > 0)
        n_layers = n;
    }
  }
  if (n_layers > H3_DIT_NUM_LAYERS)
    n_layers = H3_DIT_NUM_LAYERS;
  {
    const char *eo = getenv("H3_EMBED_ONLY");
    if (eo && eo[0] && eo[0] != '0')
      n_layers = 0;
  }
  int steps_n = steps > 0 ? steps : 2;
  if (steps_n < 1)
    steps_n = 1;
  if (geom.nv > 8)
    fprintf(stderr,
            "video-c: H3 canvas %dx%d latent %dx%dx%d nv=%d seq~%d "
            "steps=%d layers=%d (host-slow)\n",
            geom.pixel_w, geom.pixel_h, geom.latent_t, geom.latent_h,
            geom.latent_w, geom.nv, geom.nv + geom.na + 12, steps_n, n_layers);
  if (geom.pixel_w < 768 || geom.pixel_h < 768)
    fprintf(stderr,
            "video-c: short edge %d<%d (H3 was released for 768); "
            "expect a field, not a readable clip\n",
            geom.pixel_w < geom.pixel_h ? geom.pixel_w : geom.pixel_h, 768);
  setvbuf(stderr, NULL, _IONBF, 0);
  setvbuf(stdout, NULL, _IONBF, 0);
  uint64_t seed_v = seed ? seed : 1;
  h3_text_cond tc;
  memset(&tc, 0, sizeof(tc));
  float *text = NULL;
  int nt = 12;
  int own_text = 0;
  double t_stage = wan_profile_now_ms();
#define H3_PROF_TICK(name)                                                     \
  do {                                                                         \
    if (t_stage > 0)                                                           \
      wan_profile_add_ms(name, wan_profile_now_ms() - t_stage);                \
    t_stage = wan_profile_now_ms();                                            \
  } while (0)
  const char *dump = getenv("H3_TEXT_COND");
  if (dump && dump[0]) {
    if (h3_text_cond_from_bin(dump, &tc, error, error_size) != 0) {
      fprintf(stderr, "video-c: text cond failed: %s\n", error);
      return 1;
    }
    text = tc.cond;
    nt = tc.nt;
  } else if (prompt && prompt[0]) {
    if (h3_text_cond_from_prompt(prompt, NULL, NULL, 0, clipproj_path, &tc,
                                 error, error_size) != 0) {
      fprintf(stderr, "video-c: text cond failed: %s\n", error);
      return 1;
    }
    text = tc.cond;
    nt = tc.nt;
  } else {
    text = (float *)malloc((size_t)nt * (size_t)H3_DIT_TEXT_DIM *
                           sizeof(float));
    if (!text)
      return 1;
    own_text = 1;
    for (int i = 0; i < nt * H3_DIT_TEXT_DIM; i++)
      text[i] = 0.01f * sinf((float)i * 0.02f);
  }
  H3_PROF_TICK("h3_text_cond");
  float *video = (float *)malloc(geom.video_n * sizeof(float));
  float *audio = (float *)malloc(geom.audio_n * sizeof(float));
  if (!video || !audio) {
    free(video);
    free(audio);
    if (own_text)
      free(text);
    h3_text_cond_free(&tc);
    return 1;
  }
  h3_st_store *st = h3_st_store_open(path, error, error_size);
  if (!st) {
    fprintf(stderr, "video-c: DiT open failed: %s (%s)\n", error, path);
    free(video);
    free(audio);
    if (own_text)
      free(text);
    h3_text_cond_free(&tc);
    return 1;
  }
  h3_st_store_set_prof_tag(st, "h3_dit_wload");
  H3_PROF_TICK("h3_dit_open");
  if (h3_dit_t2va(st, text, nt, tc.tags, steps_n, n_layers,
                  reuse > 0 ? reuse : 1, adaln_t_sigma, seed_v, &geom, video,
                  audio, error, error_size) != 0) {
    fprintf(stderr, "video-c: generate failed: %s\n", error);
    h3_st_store_free(st);
    free(video);
    free(audio);
    if (own_text)
      free(text);
    h3_text_cond_free(&tc);
    return 1;
  }
H3_PROF_TICK("h3_dit_t2va");
  if (getenv("H3_STORE_DBG")) {
    h3_st_store_debug(st, "dit-after-t2va");
  }
  if (own_text)
    free(text);
  int used_4b = tc.used_4b;
  int used_dump = tc.used_dump;
  int have_prompt = (prompt && prompt[0]) || used_dump;
  h3_text_cond_free(&tc);
  h3_st_store_free(st);
  double vsq = 0, asq = 0;
  for (size_t i = 0; i < geom.video_n; i++)
    vsq += (double)video[i] * video[i];
  for (size_t i = 0; i < geom.audio_n; i++)
    asq += (double)audio[i] * audio[i];
  printf("video-c: generate T2VA %dx%dx%d latent %dx%dx%d nv=%d steps=%d "
         "layers=%d seed=%llu nt=%d latent_rms=%.6g a_rms=%.6g (%s)\n",
         geom.frames, geom.pixel_h, geom.pixel_w, geom.latent_t,
         geom.latent_h, geom.latent_w, geom.nv, steps_n, n_layers,
         (unsigned long long)seed_v, nt, sqrt(vsq / (double)geom.video_n),
         sqrt(asq / (double)geom.audio_n),
         have_prompt ? (used_dump ? "Qwen3-VL-32B dump"
                        : used_4b   ? "Qwen3-VL-4B+ClipProj"
                                    : "hash TE+ClipProj")
                     : "dummy text");
  {
    const char *da = getenv("H3_DUMP_AUDIO_LATENT");
    if (da && da[0]) {
      FILE *af = fopen(da, "wb");
      if (af) {
        fwrite(audio, sizeof(float), geom.audio_n, af);
        fclose(af);
        fprintf(stderr, "video-c: dumped audio latent %s n=%zu\n", da,
                geom.audio_n);
      }
    }
  }
  h3_dit_log_latent_spatial(video, H3_VIDEO_VAE_LATENT_CHANNELS, geom.latent_t,
                            geom.latent_h, geom.latent_w);
  {
    const char *prev = getenv("H3_LATENT_PREVIEW");
    char pgm[1100];
    if (prev && prev[0]) {
      snprintf(pgm, sizeof(pgm), "%s", prev);
    } else if (out_mp4 && path_has_ext(out_mp4, ".ppm")) {
      snprintf(pgm, sizeof(pgm), "%s", out_mp4);
      size_t n = strlen(pgm);
      if (n > 4)
        memcpy(pgm + n - 4, ".pgm", 5);
      else
        pgm[0] = 0;
    } else {
      pgm[0] = 0;
    }
    if (pgm[0])
      h3_dit_write_latent_pgm(video, H3_VIDEO_VAE_LATENT_CHANNELS,
                              geom.latent_t, geom.latent_h, geom.latent_w, pgm);
  }
  if (!ckpt_dir && !out_mp4) {
    wan_profile_report("h3-generate");
    free(video);
    free(audio);
    return 0;
  }
  if (!ckpt_dir) {
    fprintf(stderr, "video-c: VAE decode needs -d MiniMax-H3\n");
    free(video);
    free(audio);
    return 2;
  }
  char vae[1100];
  snprintf(vae, sizeof(vae), "%s/FL2VA/video_vae/source", ckpt_dir);
  char avae[1100];
  snprintf(avae, sizeof(avae), "%s/FL2VA/audio_vae", ckpt_dir);
  int Ac = H3_AUDIO_VAE_LATENT_CHANNELS;
  int AT = geom.audio_t;
  float *audio_c2t = (float *)malloc(geom.audio_n * sizeof(float));
  if (!audio_c2t) {
    free(video);
    free(audio);
    return 1;
  }
  for (int ch = 0; ch < 2; ch++)
    for (int c = 0; c < Ac; c++)
      for (int t = 0; t < AT; t++)
        audio_c2t[(c * 2 + ch) * AT + t] = audio[(ch * Ac + c) * AT + t];
  __block h3_video_frames_host vf;
  memset(&vf, 0, sizeof(vf));
  __block h3_audio_waveform_host wav;
  memset(&wav, 0, sizeof(wav));
  const char *par = getenv("H3_PARALLEL_VAE");
  int do_par = par && par[0] && par[0] != '0';
  if (do_par) {
    const char *vae_p = vae, *avae_p = avae;
    char *verr = (char *)calloc(1, 1024);
    char *aerr = (char *)calloc(1, 1024);
    dispatch_apply(2,
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t i) {
                     if (i == 0) {
                       if (!h3_video_vae_decode_host(vae_p, video, geom.latent_t,
                                                     geom.latent_h, geom.latent_w,
                                                     &vf, verr, 1024))
                         vf.frames = 0;
                     } else {
                       if (!h3_audio_vae_decode_host(avae_p, audio_c2t,
                                                     geom.audio_t, &wav, aerr,
                                                     1024))
                         wav.samples = 0;
                     }
                   });
    if (!vf.frames && verr[0])
      snprintf(error, error_size, "%s", verr);
    else if (!wav.samples && aerr[0])
      snprintf(error, error_size, "%s", aerr);
    free(verr);
    free(aerr);
  } else {
    if (!h3_video_vae_decode_host(vae, video, geom.latent_t, geom.latent_h,
                                  geom.latent_w, &vf, error, error_size))
      vf.frames = 0;
    if (!h3_audio_vae_decode_host(avae, audio_c2t, geom.audio_t, &wav, error,
                                  error_size))
      wav.samples = 0;
  }
  if (!vf.frames) {
    fprintf(stderr, "video-c: video VAE decode failed: %s\n", error);
    h3_video_frames_host_free(&vf);
    h3_audio_waveform_host_free(&wav);
    free(audio_c2t);
    free(video);
    free(audio);
    return 1;
  }
  printf("video-c: VAE decode %dx%dx%d\n", vf.frames, vf.height,
         vf.width);
  if (!wav.samples)
    fprintf(stderr, "video-c: audio VAE skipped: %s\n", error);
  {
    const char *ad = getenv("H3_AUDIO_DUMP");
    if (ad && ad[0]) {
      double mx = 0, sq = 0;
      int clip = 0;
      for (int i = 0; i < wav.samples * wav.channels; i++) {
        double v = fabs((double)wav.pcm[i]);
        if (v > mx)
          mx = v;
        sq += (double)wav.pcm[i] * (double)wav.pcm[i];
        if (v >= 0.999)
          clip++;
      }
      size_t n = (size_t)wav.samples * wav.channels;
      fprintf(stderr,
              "video-c: audio dump samples=%d max=%.4g rms=%.4g clipped=%d/%zu\n",
              wav.samples, mx, n ? sqrt(sq / (double)n) : 0.0, clip, n);
      fflush(stderr);
    }
  }
  /* Profile the decode pair: parallel mode reports one combined wall bucket
   * (h3_vae_pair) so it can be compared against serial video+audio. */
  if (do_par)
    H3_PROF_TICK("h3_vae_pair");
  else {
    H3_PROF_TICK("h3_vae_video");
    H3_PROF_TICK("h3_vae_audio");
  }
  if (!out_mp4) {
    wan_profile_report("h3-generate");
    h3_video_frames_host_free(&vf);
    h3_audio_waveform_host_free(&wav);
    free(audio_c2t);
    free(video);
    free(audio);
    return 0;
  }
  if (path_has_ext(out_mp4, ".ppm")) {
    int wrc = write_h3_video_media(ckpt_dir, out_mp4, &vf,
                                   fps > 0 ? fps : H3_FPS);
    H3_PROF_TICK("h3_media_encode");
    wan_profile_report("h3-generate");
    h3_video_frames_host_free(&vf);
    h3_audio_waveform_host_free(&wav);
    free(audio_c2t);
    free(video);
    free(audio);
    return wrc;
  }
  free(audio_c2t);
  free(video);
  free(audio);
  size_t pn = (size_t)vf.frames * vf.height * vf.width * 3;
  float *scaled = (float *)malloc(pn * sizeof(float));
  if (!scaled) {
    h3_video_frames_host_free(&vf);
    h3_audio_waveform_host_free(&wav);
    return 1;
  }
  for (size_t i = 0; i < pn; i++)
    scaled[i] = vf.rgb[i] * 255.f;
  int fps_n = fps > 0 ? fps : H3_FPS;
  int rc = encode_mp4_from_rgb_pcm(out_mp4, vf.width, vf.height,
                                   vf.frames, fps_n, scaled, pn, wav.pcm,
                                   wav.channels, wav.samples, wav.sample_rate);
  H3_PROF_TICK("h3_media_encode");
  wan_profile_report("h3-generate");
  free(scaled);
  h3_audio_waveform_host_free(&wav);
  h3_video_frames_host_free(&vf);
  if (rc != 0) {
    fprintf(stderr, "video-c: media encode failed\n");
    return 1;
  }
  printf("video-c: wrote %s\n", out_mp4);
  return 0;
}

static int h3_serve(const char *sock_path, const char *ckpt_dir,
                    const char *clipproj_path, char *error, size_t error_size) {
  char dit[768], te[768], vvae[1100], avae[1100];
  if (!h3_resolve_dit_pack_path(dit, sizeof(dit))) {
    snprintf(error, error_size, "video-c: HOME unset (need H3_DIT_ST)");
    return 2;
  }
  te[0] = 0;
  h3_resolve_qwen4b_dir(te, sizeof(te));
  snprintf(vvae, sizeof(vvae), "%s/FL2VA/video_vae/source", ckpt_dir);
  snprintf(avae, sizeof(avae), "%s/FL2VA/audio_vae", ckpt_dir);
  h3_st_store *s_dit = h3_st_store_open(dit, error, error_size);
  h3_st_store *s_te = NULL;
  if (te[0])
    s_te = h3_st_store_open(te, error, error_size);
  h3_st_store *s_vvae = h3_st_store_open(vvae, error, error_size);
  h3_st_store *s_avae = h3_st_store_open(avae, error, error_size);
  if (!s_dit || !s_vvae || !s_avae) {
    snprintf(error, error_size, "video-c: h3 serve store open failed");
    h3_st_store_free(s_avae);
    h3_st_store_free(s_vvae);
    h3_st_store_free(s_te);
    h3_st_store_free(s_dit);
    return 1;
  }
  h3_st_store_set_prof_tag(s_dit, "h3_dit_wload");
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    snprintf(error, error_size, "video-c: socket() failed");
    h3_st_store_free(s_avae);
    h3_st_store_free(s_vvae);
    h3_st_store_free(s_te);
    h3_st_store_free(s_dit);
    return 1;
  }
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path);
  unlink(sock_path);
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    snprintf(error, error_size, "video-c: bind %s failed", sock_path);
    close(fd);
    h3_st_store_free(s_avae);
    h3_st_store_free(s_vvae);
    h3_st_store_free(s_te);
    h3_st_store_free(s_dit);
    return 1;
  }
  if (listen(fd, 4) != 0) {
    snprintf(error, error_size, "video-c: listen failed");
    close(fd);
    h3_st_store_free(s_avae);
    h3_st_store_free(s_vvae);
    h3_st_store_free(s_te);
    h3_st_store_free(s_dit);
    return 1;
  }
  fprintf(stderr, "video-c: h3 serve on %s (DiT %s, TE %s)\n", sock_path, dit,
          te[0] ? te : "(hash TE)");
  fprintf(stderr,
          "video-c: stores resident (DiT %.2f GiB + VAE + TE); first request "
          "pays cold load, later requests reuse cached weights\n",
          (double)h3_st_store_bytes(s_dit) / (1024.0 * 1024.0 * 1024.0));
  for (;;) {
    int cfd = accept(fd, NULL, NULL);
    if (cfd < 0)
      continue;
    char line[8192];
    ssize_t n = read(cfd, line, sizeof(line) - 1);
    if (n <= 0) {
      close(cfd);
      continue;
    }
    line[n] = 0;
    char *fields[10] = {0};
    char *p = line;
    int fi = 0;
    while (p && fi < 10) {
      char *tab = strchr(p, '\t');
      if (tab)
        *tab = 0;
      fields[fi++] = p;
      p = tab ? tab + 1 : NULL;
    }
    if (fi < 8) {
      const char *bad = "err: malformed request\n";
      write(cfd, bad, (int)strlen(bad));
      close(cfd);
      continue;
    }
    /* fields: out_mp4, prompt, frames, width, height, seed, steps, layers,
     *         reuse, adaln_t_sigma. */
    int reuse = fi > 8 && fields[8] && fields[8][0] ? atoi(fields[8]) : 1;
    int adaln = fi > 9 && fields[9] && fields[9][0] ? atoi(fields[9]) : -1;
    char err[1024];
    int rc = h3_run_generate(fields[1], atoi(fields[2]), atoi(fields[3]),
                             atoi(fields[4]), (uint64_t)strtoull(fields[5], NULL,
                                                                 10),
                             atoi(fields[6]), atoi(fields[7]), reuse, adaln,
                             ckpt_dir, clipproj_path, fields[0], 0, err,
                             sizeof(err));
    if (rc == 0) {
      const char *ok = "ok\n";
      write(cfd, ok, 3);
    } else {
      char resp[1200];
      int m = snprintf(resp, sizeof(resp), "err: %s\n", err);
      write(cfd, resp, m);
    }
    close(cfd);
    {
      const char *ml = getenv("H3_MLOCK");
      if (ml && ml[0] && ml[0] != '0') {
        unsigned long long n = h3_st_store_mlock_all();
        if (n)
          fprintf(stderr,
                  "video-c: mlocked %.1f GiB of resident weight cache\n",
                  (double)n / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
      }
    }
  }
}

int main(int argc, char **argv) {
  h3_prof_now_ms = wan_profile_now_ms;
  h3_prof_add_ms = wan_profile_add_ms;
  const wan_model_config *defs = wan_model_config_1_3b();
  wan_gen_params p = {
      .prompt = NULL,
      .negative_prompt = "",
      .width = defs->default_width,
      .height = defs->default_height,
      .frames = defs->default_frames,
      .steps = defs->default_steps,
      .cfg_scale = defs->default_cfg,
      .shift = 5.0f,
      .seed = 0,
      .solver = WAN_SOLVER_UNIPC,
      .dtype = WAN_DTYPE_F16,
      .fps = 16,
      .vocab_path = NULL,
  };

  const char *ckpt_dir = NULL;
  const char *uma_sock = NULL;
  const char *lora_path = NULL;
  float lora_scale = 1.0f;
  const char *out_mp4 = NULL;
  const char *in_path = NULL;
  int validate_only = 0;
  int info_only = 0;
  video_family_t family = VIDEO_FAMILY_WAN;
  int h3_layers = 0;
  int h3_reuse = 0;
  int h3_adaln_sigma = -1;
  int h3_ssd = 0;
  int decode_audio = 0;
  int encode_audio = 0;
  int encode_video = 0;
  int decode_video = 0;
  int clipproj = 0;
  int tokenize = 0;
  int embed = 0;
  int present = 0;
  int dit_denoise = 0;
  int h3_generate = 0;
  int pictures = 0;
  int merge_h = 2;
  int merge_w = 2;
  const char *clipproj_path = NULL;
  const char *text_cond_path = NULL;
  const char *serve_sock = NULL;
  int latent_t = 4;
  int latent_t_set = 0;
  int fps_set = 0;
  int steps_set = 0;
  int width_set = 0;
  int height_set = 0;
  int frames_set = 0;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if ((!strcmp(a, "--ckpt-dir") || !strcmp(a, "-d")) && i + 1 < argc)
      ckpt_dir = argv[++i];
    else if (!strcmp(a, "--family") && i + 1 < argc) {
      if (parse_family(argv[++i], &family) != 0) {
        fprintf(stderr, "video-c: bad --family (want wan|h3)\n");
        return 2;
      }
    } else if (!strcmp(a, "--info"))
      info_only = 1;
    else if (!strcmp(a, "--decode-audio"))
      decode_audio = 1;
    else if (!strcmp(a, "--encode-audio"))
      encode_audio = 1;
    else if (!strcmp(a, "--encode-video"))
      encode_video = 1;
    else if (!strcmp(a, "--decode-video"))
      decode_video = 1;
    else if (!strcmp(a, "--clipproj")) {
      clipproj = 1;
      if (i + 1 < argc && argv[i + 1][0] != '-')
        clipproj_path = argv[++i];
    } else if (!strcmp(a, "--text-cond") && i + 1 < argc)
      text_cond_path = argv[++i];
    else if (!strcmp(a, "--tokenize"))
      tokenize = 1;
    else if (!strcmp(a, "--embed"))
      embed = 1;
    else if (!strcmp(a, "--present"))
      present = 1;
    else if (!strcmp(a, "--dit-denoise"))
      dit_denoise = 1;
    else if (!strcmp(a, "--generate"))
      h3_generate = 1;
    else if (!strcmp(a, "--serve-sock") && i + 1 < argc)
      serve_sock = argv[++i];
    else if (!strcmp(a, "--pictures") && i + 1 < argc)
      pictures = atoi(argv[++i]);
    else if (!strcmp(a, "--merge-h") && i + 1 < argc)
      merge_h = atoi(argv[++i]);
    else if (!strcmp(a, "--merge-w") && i + 1 < argc)
      merge_w = atoi(argv[++i]);
    else if (!strcmp(a, "--latent-t") && i + 1 < argc) {
      latent_t = atoi(argv[++i]);
      latent_t_set = 1;
    }
    else if (!strcmp(a, "--uma-sock") && i + 1 < argc)
      uma_sock = argv[++i];
    else if (!strcmp(a, "--prompt") && i + 1 < argc)
      p.prompt = argv[++i];
    else if (!strcmp(a, "--negative-prompt") && i + 1 < argc)
      p.negative_prompt = argv[++i];
    else if (!strcmp(a, "--width") && i + 1 < argc) {
      p.width = atoi(argv[++i]);
      width_set = 1;
    } else if (!strcmp(a, "--height") && i + 1 < argc) {
      p.height = atoi(argv[++i]);
      height_set = 1;
    } else if (!strcmp(a, "--frames") && i + 1 < argc) {
      p.frames = atoi(argv[++i]);
      frames_set = 1;
    }
    else if (!strcmp(a, "--steps") && i + 1 < argc) {
      p.steps = atoi(argv[++i]);
      steps_set = 1;
    }
    else if (!strcmp(a, "--cfg") && i + 1 < argc)
      p.cfg_scale = (float)atof(argv[++i]);
    else if (!strcmp(a, "--shift") && i + 1 < argc)
      p.shift = (float)atof(argv[++i]);
    else if (!strcmp(a, "--seed") && i + 1 < argc)
      p.seed = atoi(argv[++i]);
    else if (!strcmp(a, "--lora") && i + 1 < argc)
      lora_path = argv[++i];
    else if (!strcmp(a, "--lora-scale") && i + 1 < argc)
      lora_scale = (float)atof(argv[++i]);
    else if (!strcmp(a, "--solver") && i + 1 < argc) {
      if (parse_solver(argv[++i], &p.solver) != 0) {
        fprintf(stderr, "bad --solver\n");
        return 2;
      }
    } else if (!strcmp(a, "--dtype") && i + 1 < argc) {
      if (parse_dtype(argv[++i], &p.dtype) != 0) {
        fprintf(stderr, "bad --dtype\n");
        return 2;
      }
    } else if (!strcmp(a, "--fps") && i + 1 < argc) {
      p.fps = atoi(argv[++i]);
      fps_set = 1;
    }
    else if (!strcmp(a, "--vocab") && i + 1 < argc)
      p.vocab_path = argv[++i];
    else if (!strcmp(a, "--layers") && i + 1 < argc)
      h3_layers = atoi(argv[++i]);
    else if (!strcmp(a, "--reuse") && i + 1 < argc)
      h3_reuse = atoi(argv[++i]);
    else if (!strcmp(a, "--adaln-t-sigma") && i + 1 < argc)
      h3_adaln_sigma = atoi(argv[++i]);
    else if (!strcmp(a, "--ssd-streaming"))
      h3_ssd = 1;
    else if ((!strcmp(a, "--out") || !strcmp(a, "-o")) && i + 1 < argc)
      out_mp4 = argv[++i];
    else if (!strcmp(a, "--in") && i + 1 < argc)
      in_path = argv[++i];
    else if (!strcmp(a, "--validate-only"))
      validate_only = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", a);
      usage(argv[0]);
      return 2;
    }
  }

  if (family == VIDEO_FAMILY_H3) {
    if (text_cond_path && text_cond_path[0])
      setenv("H3_TEXT_COND", text_cond_path, 1);
    if (!h3_generate && out_mp4 && p.prompt && p.prompt[0] && !info_only &&
        !decode_audio && !encode_audio && !encode_video && !decode_video &&
        !clipproj && !tokenize && !embed && !present && !dit_denoise)
      h3_generate = 1;
    if (h3_ssd)
      fprintf(stderr, "video-c: --ssd-streaming ignored (host ConvRot pack stays mapped)\n");
    if (info_only)
      return h3_checkpoint_info(ckpt_dir);
    if (dit_denoise) {
      char path[768];
      if (!h3_resolve_dit_pack_path(path, sizeof(path))) {
        fprintf(stderr, "video-c: HOME unset (need H3_DIT_ST)\n");
        return 2;
      }
      int layers = h3_layers > 0 ? h3_layers : 1;
      if (layers > H3_DIT_NUM_LAYERS)
        layers = H3_DIT_NUM_LAYERS;
      int steps = steps_set ? p.steps : 2;
      if (steps < 1)
        steps = 1;
      uint64_t seed = p.seed ? (uint64_t)p.seed : 1;
      char error[1024];
      /* 1 video patch (T=1,H=2,W=2) + stereo audio_t=1 + 1 text row. */
      h3_layout_spec spec = {1, 1, 2, 2, 1, 5, NULL, 0, NULL, 0};
      h3_layout layout;
      if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "video-c: dit-denoise layout: %s\n", error);
        return 1;
      }
      h3_dit_seq_plan plan;
      if (h3_dit_seq_plan_from_layout(&layout, &plan) != 0 || plan.seq != 4 ||
          plan.nv != 1 || plan.na != 2 || plan.nt != 1) {
        fprintf(stderr, "video-c: dit-denoise plan mismatch seq=%d nv=%d na=%d\n",
                plan.seq, plan.nv, plan.na);
        h3_dit_seq_plan_free(&plan);
        h3_layout_free(&layout);
        return 1;
      }
      h3_st_store *st = h3_st_store_open(path, error, sizeof(error));
      if (!st) {
        fprintf(stderr, "video-c: DiT open failed: %s (%s)\n", error, path);
        h3_dit_seq_plan_free(&plan);
        h3_layout_free(&layout);
        return 1;
      }
      float video[96], audio[64], text[H3_DIT_TEXT_DIM];
      h3_rng rng;
      h3_rng_seed(&rng, seed);
      h3_rng_fill_normal(&rng, video, 96);
      h3_rng_fill_normal(&rng, audio, 64);
      for (int i = 0; i < H3_DIT_TEXT_DIM; i++)
        text[i] = 0.01f * sinf((float)i * 0.02f);
      if (h3_dit_denoise(st, video, plan.nv, audio, plan.na, text, plan.nt,
                         plan.video_index, plan.audio_index, plan.text_index,
                         plan.tags, plan.position_ids, plan.seq, steps, layers,
                         h3_reuse > 0 ? h3_reuse : 1, h3_adaln_sigma, 0, 0, 0, 0,
                         error, sizeof(error)) != 0) {
        fprintf(stderr, "video-c: dit-denoise failed: %s\n", error);
        h3_st_store_free(st);
        h3_dit_seq_plan_free(&plan);
        h3_layout_free(&layout);
        return 1;
      }
      double vsq = 0, asq = 0;
      for (int i = 0; i < 96; i++)
        vsq += (double)video[i] * video[i];
      for (int i = 0; i < 64; i++)
        asq += (double)audio[i] * audio[i];
      printf("video-c: dit-denoise steps=%d layers=%d seed=%llu v_rms=%.6g "
             "a_rms=%.6g\n",
             steps, layers, (unsigned long long)seed, sqrt(vsq / 96.0),
             sqrt(asq / 64.0));
      h3_st_store_free(st);
      h3_dit_seq_plan_free(&plan);
      h3_layout_free(&layout);
      return 0;
    }
    if (h3_generate && serve_sock && serve_sock[0]) {
      char error[1024];
      return h3_serve(serve_sock, ckpt_dir, clipproj_path, error,
                      sizeof(error));
    }
    if (h3_generate) {
      char error[1024];
      int req_frames = H3_DIT_TINY_FRAMES;
      if (frames_set && p.frames != 49)
        req_frames = p.frames;
      int width = width_set ? p.width : 0;
      int height = height_set ? p.height : 0;
      int steps = steps_set ? p.steps : 2;
      if (!steps_set && (width >= 256 || height >= 256))
        steps = 8;
      int fps = fps_set ? p.fps : H3_FPS;
      return h3_run_generate(p.prompt, req_frames, width, height,
                             p.seed ? (uint64_t)p.seed : 1, steps, h3_layers,
                             h3_reuse > 0 ? h3_reuse : 1, h3_adaln_sigma,
                             ckpt_dir, clipproj_path, out_mp4, fps, error,
                             sizeof(error));
    }
    if (present || embed) {
      char error[1024];
      if (pictures < 0 || pictures > 16 || merge_h < 1 || merge_w < 1) {
        fprintf(stderr, "video-c: bad --pictures/--merge-h/--merge-w\n");
        return 2;
      }
      const char *blob = getenv("H3_BMTL_TOK");
      h3_tokenizer *tok = h3_tokenizer_load(blob, error, sizeof(error));
      if (!tok) {
        fprintf(stderr, "video-c: tokenizer load failed: %s\n", error);
        return 1;
      }
      const char *text = p.prompt ? p.prompt : "";
      if (!*text) {
        fprintf(stderr, "video-c: --present/--embed need --prompt\n");
        h3_tokenizer_free(tok);
        return 2;
      }
      int mh[16], mw[16];
      for (int i = 0; i < pictures; i++) {
        mh[i] = merge_h;
        mw[i] = merge_w;
      }
      h3_presentation pres;
      if (!h3_present_fl2va(tok, text, pictures ? mh : NULL,
                            pictures ? mw : NULL, (size_t)pictures, &pres,
                            error, sizeof(error))) {
        fprintf(stderr, "video-c: present failed: %s\n", error);
        h3_tokenizer_free(tok);
        return 1;
      }
      h3_tokenizer_free(tok);
      if (present && !embed) {
        printf("video-c: present n=%zu pictures=%d merge=%dx%d\n", pres.count,
               pictures, merge_h, merge_w);
        printf("ids");
        for (size_t i = 0; i < pres.count; i++)
          printf(" %u", pres.ids[i]);
        printf("\ntags");
        for (size_t i = 0; i < pres.count; i++)
          printf(" %u", (unsigned)pres.tags[i]);
        printf("\npos_t");
        for (size_t i = 0; i < pres.count; i++)
          printf(" %u", pres.pos[i]);
        printf("\npos_h");
        for (size_t i = 0; i < pres.count; i++)
          printf(" %u", pres.pos[pres.count + i]);
        printf("\npos_w");
        for (size_t i = 0; i < pres.count; i++)
          printf(" %u", pres.pos[2 * pres.count + i]);
        printf("\n");
        h3_presentation_free(&pres);
        return 0;
      }
      size_t n = pres.count;
      const int din = H3_QWEN_TE_HIDDEN_4B;
      const int dout = H3_CLIPPROJ_DOUT;
      float *hidden = (float *)calloc(n * (size_t)din, sizeof(float));
      float *cond = (float *)calloc(n * (size_t)dout, sizeof(float));
      if (!hidden || !cond) {
        free(hidden);
        free(cond);
        h3_presentation_free(&pres);
        return 1;
      }
      int used_4b = 0;
      char te_dir[768];
      if (!h3_resolve_qwen4b_dir(te_dir, sizeof(te_dir)))
        te_dir[0] = '\0';
      if (te_dir[0]) {
        char shard[900];
        snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors",
                 te_dir);
        if (access(shard, R_OK) == 0) {
          if (!getenv("H3_QWEN_TE_LAYERS")) {
            char tap[16];
            snprintf(tap, sizeof(tap), "%d", H3_QWEN_TE_CLIPPROJ_TAP);
            setenv("H3_QWEN_TE_LAYERS", tap, 1);
          }
          int apply_norm = getenv("H3_QWEN_TE_FINAL_NORM") ? 1 : 0;
          if (!h3_qwen_te_4b_forward(te_dir, pres.ids, n, pres.pos, apply_norm,
                                     hidden, error, sizeof(error))) {
            fprintf(stderr, "video-c: 4B TE failed: %s\n", error);
            free(hidden);
            free(cond);
            h3_presentation_free(&pres);
            return 1;
          }
          used_4b = 1;
        }
      }
      if (!used_4b)
        h3_qwen_te_hash_embed(pres.ids, n, din, hidden);
      char def[768];
      if (!clipproj_path) {
        const char *home = getenv("HOME");
        if (!home) {
          fprintf(stderr, "video-c: --embed needs ClipProj PATH or $HOME\n");
          free(hidden);
          free(cond);
          h3_presentation_free(&pres);
          return 2;
        }
        snprintf(def, sizeof(def),
                 "%s/.zerollama/third_party/h3/clipproj/"
                 "mmh3-4b-ClipProj-celeb-mlp.safetensors",
                 home);
        clipproj_path = def;
      }
      h3_clipproj *proj = h3_clipproj_load(clipproj_path, error, sizeof(error));
      if (!proj) {
        fprintf(stderr, "video-c: clipproj load failed: %s\n", error);
        free(hidden);
        free(cond);
        h3_presentation_free(&pres);
        return 1;
      }
      if (h3_clipproj_apply(proj, hidden, (int)n, cond, error, sizeof(error)) !=
          0) {
        fprintf(stderr, "video-c: clipproj apply failed: %s\n", error);
        h3_clipproj_free(proj);
        free(hidden);
        free(cond);
        h3_presentation_free(&pres);
        return 1;
      }
      double sum = 0.0, sq = 0.0;
      size_t cn = n * (size_t)dout;
      for (size_t i = 0; i < cn; i++) {
        sum += cond[i];
        sq += (double)cond[i] * cond[i];
      }
      size_t n_video = 0;
      for (size_t i = 0; i < n; i++)
        if (pres.tags[i] == 0)
          n_video++;
      printf("video-c: embed n=%zu video_rows=%zu din=%d dout=%d mean=%.6g rms=%.6g (%s)\n",
             n, n_video, din, dout, sum / (double)cn, sqrt(sq / (double)cn),
             used_4b ? "Qwen3-VL-4B" : "hash TE");
      h3_clipproj_free(proj);
      free(hidden);
      free(cond);
      h3_presentation_free(&pres);
      return 0;
    }
    if (tokenize) {
      char error[1024];
      const char *blob = getenv("H3_BMTL_TOK");
      h3_tokenizer *tok = h3_tokenizer_load(blob, error, sizeof(error));
      if (!tok) {
        fprintf(stderr, "video-c: tokenizer load failed: %s\n", error);
        return 1;
      }
      const char *text = p.prompt ? p.prompt : "";
      uint32_t *ids = NULL;
      size_t n = 0;
      if (!h3_tokenizer_encode(tok, text, 1, &ids, &n, error, sizeof(error))) {
        fprintf(stderr, "video-c: encode failed: %s\n", error);
        h3_tokenizer_free(tok);
        return 1;
      }
      printf("video-c: tokenize n=%zu", n);
      for (size_t i = 0; i < n; i++)
        printf(" %u", ids[i]);
      printf("\n");
      h3_tokenizer_ids_free(ids);
      h3_tokenizer_free(tok);
      return 0;
    }
    if (clipproj) {
      char def[768];
      if (!clipproj_path) {
        const char *home = getenv("HOME");
        if (!home) {
          fprintf(stderr, "video-c: --clipproj needs PATH or $HOME\n");
          return 2;
        }
        snprintf(def, sizeof(def),
                 "%s/.zerollama/third_party/h3/clipproj/"
                 "mmh3-4b-ClipProj-celeb-mlp.safetensors",
                 home);
        clipproj_path = def;
      }
      char error[1024];
      h3_clipproj *proj = h3_clipproj_load(clipproj_path, error, sizeof(error));
      if (!proj) {
        fprintf(stderr, "video-c: clipproj load failed: %s\n", error);
        return 1;
      }
      const int seq = 8;
      int din = h3_clipproj_din(proj);
      int dout = h3_clipproj_dout(proj);
      float *h = (float *)calloc((size_t)seq * (size_t)din, sizeof(float));
      float *cond = (float *)calloc((size_t)seq * (size_t)dout, sizeof(float));
      if (!h || !cond) {
        h3_clipproj_free(proj);
        free(h);
        free(cond);
        return 1;
      }
      h3_audio_vae_fill_unit_latent(h, (size_t)seq * (size_t)din);
      if (h3_clipproj_apply(proj, h, seq, cond, error, sizeof(error)) != 0) {
        fprintf(stderr, "video-c: clipproj apply failed: %s\n", error);
        h3_clipproj_free(proj);
        free(h);
        free(cond);
        return 1;
      }
      size_t n = (size_t)seq * (size_t)dout;
      double sum = 0.0, square = 0.0;
      for (size_t i = 0; i < n; i++) {
        sum += cond[i];
        square += (double)cond[i] * cond[i];
      }
      printf("video-c: clipproj din=%d dout=%d seq=%d sink=%d mlp=%d mean=%.6g rms=%.6g\n",
             din, dout, seq, h3_clipproj_has_sink(proj),
             h3_clipproj_has_mlp(proj), sum / (double)n,
             sqrt(square / (double)n));
      h3_clipproj_free(proj);
      free(h);
      free(cond);
      return 0;
    }
    if (decode_audio) {
      if (!ckpt_dir || !out_mp4) {
        fprintf(stderr, "video-c: --decode-audio needs -d DIR and --out out.wav\n");
        return 2;
      }
      if (latent_t < 1) {
        fprintf(stderr, "video-c: --latent-t must be >= 1\n");
        return 2;
      }
      char vae[1100];
      snprintf(vae, sizeof(vae), "%s/FL2VA/audio_vae", ckpt_dir);
      size_t n = (size_t)H3_AUDIO_VAE_LATENT_CHANNELS * H3_AUDIO_VAE_STEREO *
                 (size_t)latent_t;
      float *latent = (float *)calloc(n, sizeof(float));
      if (!latent)
        return 1;
      {
        const char *al = getenv("H3_AUDIO_LATENT");
        if (al && al[0]) {
          FILE *af = fopen(al, "rb");
          if (!af) {
            fprintf(stderr, "video-c: H3_AUDIO_LATENT open failed: %s\n", al);
            free(latent);
            return 1;
          }
          float *raw = (float *)malloc(n * sizeof(float));
          size_t nr = raw ? fread(raw, sizeof(float), n, af) : 0;
          fclose(af);
          if (!raw || nr != n) {
            fprintf(stderr, "video-c: H3_AUDIO_LATENT size %zu want %zu\n", nr,
                    n);
            free(raw);
            free(latent);
            return 1;
          }
          int Ac = H3_AUDIO_VAE_LATENT_CHANNELS, AT = latent_t;
          for (int ch = 0; ch < 2; ch++)
            for (int c = 0; c < Ac; c++)
              for (int t = 0; t < AT; t++)
                latent[(c * 2 + ch) * AT + t] = raw[(ch * Ac + c) * AT + t];
          free(raw);
          fprintf(stderr, "video-c: loaded audio latent %s\n", al);
        } else {
          h3_audio_vae_fill_unit_latent(latent, n);
        }
      }
      char error[1024];
      h3_audio_waveform_host wav;
      memset(&wav, 0, sizeof(wav));
      if (!h3_audio_vae_decode_host(vae, latent, latent_t, &wav, error,
                                    sizeof(error))) {
        fprintf(stderr, "video-c: audio decode failed: %s\n", error);
        free(latent);
        return 1;
      }
      if (!h3_audio_waveform_write_wav(&wav, out_mp4, error, sizeof(error))) {
        fprintf(stderr, "video-c: wav write failed: %s\n", error);
        h3_audio_waveform_host_free(&wav);
        free(latent);
        return 1;
      }
      printf("video-c: wrote %s (%d ch, %d samples @ %d Hz, T=%d)\n", out_mp4,
             wav.channels, wav.samples, wav.sample_rate, latent_t);
      h3_audio_waveform_host_free(&wav);
      free(latent);
      return 0;
    }
    if (encode_audio) {
      if (!ckpt_dir) {
        fprintf(stderr, "video-c: --encode-audio needs -d DIR\n");
        return 2;
      }
      char vae[1100];
      snprintf(vae, sizeof(vae), "%s/FL2VA/audio_vae", ckpt_dir);
      int samples = H3_AUDIO_VAE_HOP_LENGTH;
      float *pcm = (float *)calloc((size_t)H3_AUDIO_VAE_STEREO * (size_t)samples,
                                   sizeof(float));
      if (!pcm)
        return 1;
      h3_audio_vae_fill_unit_latent(pcm, (size_t)H3_AUDIO_VAE_STEREO *
                                             (size_t)samples);
      char error[1024];
      h3_audio_latent_host z;
      memset(&z, 0, sizeof(z));
      if (!h3_audio_vae_encode_host(vae, pcm, samples, &z, error,
                                    sizeof(error))) {
        fprintf(stderr, "video-c: audio encode failed: %s\n", error);
        free(pcm);
        return 1;
      }
      size_t n = (size_t)z.channels * (size_t)z.stereo * (size_t)z.length;
      double sum = 0.0, square = 0.0;
      for (size_t i = 0; i < n; i++) {
        sum += z.values[i];
        square += (double)z.values[i] * z.values[i];
      }
      printf("video-c: encode T=%d ch=%d stereo=%d mean=%.6g rms=%.6g\n",
             z.length, z.channels, z.stereo, sum / (double)n,
             sqrt(square / (double)n));
      h3_audio_latent_host_free(&z);
      free(pcm);
      return 0;
    }
    if (encode_video) {
      if (!ckpt_dir) {
        fprintf(stderr, "video-c: --encode-video needs -d DIR\n");
        return 2;
      }
      int eh = (p.height == defs->default_height) ? 32 : p.height;
      int ew = (p.width == defs->default_width) ? 32 : p.width;
      int ef = (p.frames == defs->default_frames) ? 1 : p.frames;
      char vae[1100];
      snprintf(vae, sizeof(vae), "%s/FL2VA/video_vae/source", ckpt_dir);
      char error[1024];
      float *pix = NULL;
      size_t n = 0;
      if (in_path) {
        h3_video_frames_host src;
        memset(&src, 0, sizeof(src));
        if (!h3_video_frames_read_ppm(in_path, &src, error, sizeof(error))) {
          fprintf(stderr, "video-c: cannot read --in: %s\n", error);
          return 1;
        }
        eh = src.height;
        ew = src.width;
        n = (size_t)3 * (size_t)ef * (size_t)eh * (size_t)ew;
        pix = (float *)calloc(n, sizeof(float));
        if (!pix || h3_video_frames_to_cthw(&src, ef, pix) != 0) {
          fprintf(stderr, "video-c: cannot pack --in PPM\n");
          h3_video_frames_host_free(&src);
          free(pix);
          return 1;
        }
        h3_video_frames_host_free(&src);
      } else {
        n = (size_t)3 * (size_t)ef * (size_t)eh * (size_t)ew;
        pix = (float *)calloc(n, sizeof(float));
        if (!pix)
          return 1;
        h3_audio_vae_fill_unit_latent(pix, n);
        for (size_t i = 0; i < n; i++) {
          float v = pix[i] * 0.25f + 0.5f;
          if (v < 0.f)
            v = 0.f;
          if (v > 1.f)
            v = 1.f;
          pix[i] = v;
        }
      }
      h3_video_latent_host z;
      memset(&z, 0, sizeof(z));
      if (!h3_video_vae_encode_host(vae, pix, ef, eh, ew, &z, error,
                                    sizeof(error))) {
        fprintf(stderr, "video-c: video encode failed: %s\n", error);
        free(pix);
        return 1;
      }
      size_t zn = (size_t)z.channels * z.time * z.height * z.width;
      double sum = 0.0, square = 0.0;
      for (size_t i = 0; i < zn; i++) {
        sum += z.values[i];
        square += (double)z.values[i] * z.values[i];
      }
      printf("video-c: video encode %dx%dx%d -> C=%d T=%d %dx%d mean=%.6g rms=%.6g\n",
             ef, eh, ew, z.channels, z.time, z.height, z.width, sum / (double)zn,
             sqrt(square / (double)zn));
      if (out_mp4) {
        int pad_t = z.time < 2 ? 2 : z.time;
        if (h3_video_vae_output_frames(pad_t) < 1)
          pad_t = 2;
        float *padded = (float *)calloc(
            (size_t)z.channels * (size_t)pad_t * (size_t)z.height *
                (size_t)z.width,
            sizeof(float));
        if (!padded ||
            h3_video_vae_repeat_last_time(z.values, z.channels, z.time,
                                          z.height, z.width, pad_t,
                                          padded) != 0) {
          fprintf(stderr, "video-c: cannot pad latent T=%d→%d\n", z.time, pad_t);
          free(padded);
          h3_video_latent_host_free(&z);
          free(pix);
          return 1;
        }
        h3_video_frames_host vf;
        memset(&vf, 0, sizeof(vf));
        if (!h3_video_vae_decode_host(vae, padded, pad_t, z.height, z.width,
                                       &vf, error, sizeof(error))) {
          fprintf(stderr, "video-c: video decode failed: %s\n", error);
          h3_video_frames_host_free(&vf);
          free(padded);
          h3_video_latent_host_free(&z);
          free(pix);
          return 1;
        }
        size_t pn = (size_t)vf.frames * vf.height * vf.width * 3;
        double dsum = 0.0, dsq = 0.0;
        for (size_t i = 0; i < pn; i++) {
          dsum += vf.rgb[i];
          dsq += (double)vf.rgb[i] * vf.rgb[i];
        }
        printf("video-c: roundtrip decode T=%d %dx%d -> %dx%dx%d mean=%.6g rms=%.6g\n",
               pad_t, z.height, z.width, vf.frames, vf.height,
               vf.width, dsum / (double)pn, sqrt(dsq / (double)pn));
        int fps = fps_set ? p.fps : H3_FPS;
        int wrc = write_h3_video_media(ckpt_dir, out_mp4, &vf, fps);
        h3_video_frames_host_free(&vf);
        free(padded);
        h3_video_latent_host_free(&z);
        free(pix);
        return wrc;
      }
      h3_video_latent_host_free(&z);
      free(pix);
      return 0;
    }
    if (decode_video) {
      if (!ckpt_dir) {
        fprintf(stderr, "video-c: --decode-video needs -d DIR\n");
        return 2;
      }
      int vt = latent_t_set ? latent_t : 2;
      if (h3_video_vae_output_frames(vt) < 1) {
        fprintf(stderr,
                "video-c: --decode-video --latent-t must be 2 or >=7 with (T-2)%%5==0\n");
        return 2;
      }
      char vae[1100];
      snprintf(vae, sizeof(vae), "%s/FL2VA/video_vae/source", ckpt_dir);
      const int C = H3_VIDEO_VAE_LATENT_CHANNELS, H = 2, W = 2;
      size_t n = (size_t)C * (size_t)vt * H * W;
      float *z = (float *)calloc(n, sizeof(float));
      if (!z)
        return 1;
      h3_audio_vae_fill_unit_latent(z, n);
      char error[1024];
      h3_video_frames_host vf;
      memset(&vf, 0, sizeof(vf));
      if (!h3_video_vae_decode_host(vae, z, vt, H, W,  &vf, error,
                                    sizeof(error))) {
        fprintf(stderr, "video-c: video decode failed: %s\n", error);
        free(z);
        return 1;
      }
      size_t pn = (size_t)vf.frames * vf.height * vf.width * 3;
      double sum = 0.0, square = 0.0;
      for (size_t i = 0; i < pn; i++) {
        sum += vf.rgb[i];
        square += (double)vf.rgb[i] * vf.rgb[i];
      }
      printf("video-c: video decode T=%d 2x2 -> %dx%dx%d mean=%.6g rms=%.6g\n",
             vt, vf.frames, vf.height, vf.width, sum / (double)pn,
             sqrt(square / (double)pn));
      if (out_mp4) {
        int fps = fps_set ? p.fps : H3_FPS;
        if (write_h3_video_media(ckpt_dir, out_mp4, &vf, fps) != 0) {
          h3_video_frames_host_free(&vf);
          free(z);
          return 1;
        }
      }
      h3_video_frames_host_free(&vf);
      free(z);
      return 0;
    }
    fprintf(stderr,
            "video-c: --family h3: pick a tool flag\n"
            "  probe:  %s --family h3 --info -d DIR\n"
            "  audio:  %s --family h3 --decode-audio -d DIR -o out.wav\n"
            "  encode: %s --family h3 --encode-audio -d DIR\n"
            "  clip:   %s --family h3 --clipproj [PATH]\n"
            "  tok:    %s --family h3 --tokenize --prompt TEXT\n"
            "  embed:  %s --family h3 --embed --prompt TEXT [--pictures N]\n"
            "  pres:   %s --family h3 --present --prompt TEXT [--pictures N]\n"
            "  denoise: %s --family h3 --dit-denoise [--steps N] [--layers N]\n"
            "  gen:    %s --family h3 --generate [-d DIR -o out.ppm]\n"
            "  video:  %s --family h3 --encode-video -d DIR [-o out.mp4]\n"
            "  decode: %s --family h3 --decode-video -d DIR [-o out.mp4]\n",
            argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0],
            argv[0], argv[0], argv[0], argv[0]);
    return 2;
  }

  /* family wan */
  if (info_only) {
    if (!ckpt_dir) {
      fprintf(stderr, "video-c: --info requires -d / --ckpt-dir\n");
      return 1;
    }
    printf("video-c wan checkpoint dir: %s\n", ckpt_dir);
    printf("  (wan --info: directory present; use --validate-only for geometry)\n");
    return 0;
  }

  char err[256];
  if (wan_validate_params(&p, err, sizeof(err)) != 0) {
    fprintf(stderr, "video-c: %s\n", err);
    return 1;
  }
  if (validate_only) {
    fprintf(stderr, "video-c: params OK\n");
    return 0;
  }

  if (!ckpt_dir || !p.prompt || !out_mp4) {
    usage(argv[0]);
    return 2;
  }

  wan_ctx *ctx = wan_ctx_open(ckpt_dir, uma_sock);
  if (!ctx)
    return 1;

  if (lora_path && lora_path[0] &&
      wan_ctx_set_lora(ctx, lora_path, lora_scale) != 0) {
    fprintf(stderr, "wan-c: --lora failed: %s\n", lora_path);
    wan_ctx_close(ctx);
    return 1;
  }

  int rc = wan_generate_t2v(ctx, &p, out_mp4);
  wan_ctx_close(ctx);
  return rc == 0 ? 0 : 1;
}
