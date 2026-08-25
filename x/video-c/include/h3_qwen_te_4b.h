/* Qwen3-VL-4B host forward (stream layers from safetensors). */
#ifndef H3_QWEN_TE_4B_H
#define H3_QWEN_TE_4B_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Language stack only (no vision tower). hidden_out is [n, 2560] f32.
 * position_ids NULL → sequential 0..n-1 on all mRoPE axes.
 * apply_final_norm=1 matches HF last_hidden_state.
 * ClipProj-4B uses tap 24 / no final norm (H3_QWEN_TE_LAYERS, default 24 on
 * the generate/embed path). H3_QWEN_TE_LAYERS=K selects the decoder tap.
 */
int h3_qwen_te_4b_forward(const char *model_dir, const uint32_t *ids, size_t n,
                          const uint32_t *position_ids, int apply_final_norm,
                          float *hidden_out, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_QWEN_TE_4B_H */
