/* MiniMax-H3 Qwen BPE via sibling BMTL C tokenizer (not a from-scratch BPE). */
#ifndef H3_TOKENIZER_H
#define H3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_PAD_TOKEN_ID UINT32_C(151643)

typedef struct h3_tokenizer h3_tokenizer;

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json, char *error,
                                size_t error_size);
void h3_tokenizer_free(h3_tokenizer *tokenizer);

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size);
void h3_tokenizer_ids_free(uint32_t *ids);

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer, const uint32_t *ids,
                          size_t count, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_TOKENIZER_H */
