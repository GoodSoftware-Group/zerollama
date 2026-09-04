/*
 * t5_cache.h — persistent prompt → umt5 text-embed disk cache.
 *
 * Why: the host T5 path mmaps the ~11 GB umt5 pth and encodes cond +
 * empty-uncond on every invocation (~46 s combined at any resolution,
 * WAN_PROFILE "t5"). The anime-shorts workflow re-renders the same shot
 * many times, so the embeds are pure waste after the first pass.
 *
 * Key = FNV-1a64(prompt || '\0' || ckpt_dir). Files are self-describing
 * (magic + text_len/text_dim) so a ckpt/config change invalidates by
 * mismatch, not just by key. Disable: WAN_T5_CACHE=0. Override dir:
 * WAN_T5_CACHE_DIR (default ~/.zerollama/cache/wan_t5).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wan_ctx;

/* Returns 0 and fills emb (n floats, exactly cfg.text_len*cfg.text_dim)
 * on hit; non-zero on miss/disabled/error. */
int wan_t5_cache_get(const struct wan_ctx *ctx, const char *prompt, float *emb,
                     size_t n);

/* Best-effort store; non-zero only on hard errors worth surfacing. */
int wan_t5_cache_put(const struct wan_ctx *ctx, const char *prompt,
                     const float *emb, size_t n);

/* Test hook: resolve the cache path for a prompt without touching files. */
int wan_t5_cache_path(const struct wan_ctx *ctx, const char *prompt, char *out,
                      size_t cap);

#ifdef __cplusplus
}
#endif
