/*
 * uma_buf_load.h — track UMA BUF_ALLOC / PUT / FREE for Wan stages.
 */
#pragma once

#include "uma/client.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct uma_buf_pool uma_buf_pool;

uma_buf_pool *uma_buf_pool_create(UmaClient *c);
void uma_buf_pool_destroy(uma_buf_pool *pool);

int uma_buf_pool_alloc(uma_buf_pool *pool, const char *name, size_t nbytes);
int uma_buf_pool_put(uma_buf_pool *pool, const char *name, const void *data,
                     size_t nbytes);
/* F0793: large weights via BANK then BIND as name (hot PUT stays ≤4MiB). */
int uma_buf_pool_put_weight(uma_buf_pool *pool, const char *name,
                            const char *bank_key, const void *data,
                            size_t nbytes);
int uma_buf_pool_free(uma_buf_pool *pool, const char *name);
void uma_buf_pool_free_all(uma_buf_pool *pool);

#ifdef __cplusplus
}
#endif
