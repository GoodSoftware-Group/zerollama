/* MiniMax-H3 checkpoint layout probe (video-c --family h3 --info). */
#ifndef VIDEO_C_H3_INFO_H
#define VIDEO_C_H3_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Print layout summary to stdout.
 * Probe success (exit 0): FL2VA + transformer/config.json + tokenizer +
 * audio_vae + video_vae/source shards. Stock transformer/text_encoder shards
 * are optional. Generate uses the pruned ConvRot DiT (`H3_DIT_ST` or
 * ~/.zerollama/third_party/h3/dit/…) plus Qwen3-VL-4B + ClipProj. */
int h3_checkpoint_info(const char *model_dir);

#ifdef __cplusplus
}
#endif

#endif /* VIDEO_C_H3_INFO_H */
