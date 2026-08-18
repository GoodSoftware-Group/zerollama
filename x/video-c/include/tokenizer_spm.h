/*
 * tokenizer_spm.h — UMT5 Unigram loader/encode (binary .vocab or SPM .model).
 * Encode: ▁ space map + dummy prefix; eos only (HF rematch; F0940).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_spm tokenizer_spm;

tokenizer_spm *tokenizer_spm_load(const char *path);
void tokenizer_spm_free(tokenizer_spm *tok);

int tokenizer_spm_encode(tokenizer_spm *tok, const char *text, int32_t *ids,
                         size_t cap, size_t *n_out);
int tokenizer_spm_vocab_size(const tokenizer_spm *tok);

#ifdef __cplusplus
}
#endif
