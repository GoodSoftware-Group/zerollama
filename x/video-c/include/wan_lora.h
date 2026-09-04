/*
 * wan_lora.h — LoRA adapters merged into loaded weights (merge-at-load).
 *
 * Why: character consistency for series generation needs adapter weights
 * (CivitAI/diffusers Wan LoRAs). Merging W += s·B@A at tensor-load time
 * costs nothing per step and needs no broker/recipe changes — every path
 * (host ops, BANK_PUT, borrow cache) flows through wan_load_tensor_f32.
 *
 * Key format (PEFT / ComfyUI Wan LoRAs): dotted names, optional
 * `diffusion_model.` / `transformer.` prefix:
 *   <base>.lora_A.weight [rank, in]  <base>.lora_B.weight [out, rank]
 *   <base>.alpha  scalar (optional; default scale = 1/rank)
 * effective_scale = cli_scale × alpha/rank.
 *
 * Set the adapter BEFORE the first generate: the borrow cache must not
 * hold pre-merge copies of adapted tensors.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wan_lora wan_lora;

/* Opens a LoRA safetensors file. NULL on error (reasons on stderr). */
wan_lora *wan_lora_open(const char *path);
void wan_lora_close(wan_lora *L);

/* Number of distinct base tensors that have an A/B pair. */
int wan_lora_targets(const wan_lora *L);

/* Applies all pairs targeting `name` onto w (n floats, [out,in] row-major).
 * Returns: 1 applied, 0 not targeted, -1 shape/error. */
int wan_lora_apply(const wan_lora *L, const char *name, float *w, size_t n,
                   float cli_scale);

#ifdef __cplusplus
}
#endif
