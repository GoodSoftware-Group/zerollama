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
void uma_buf_pool_destroy_keep_bank(uma_buf_pool *pool);

int uma_buf_pool_alloc(uma_buf_pool *pool, const char *name, size_t nbytes);
int uma_buf_pool_put(uma_buf_pool *pool, const char *name, const void *data,
                     size_t nbytes);
/* Alloc (re-assert) + PUT; recreates name if broker dropped the slot. */
int uma_buf_pool_ensure_put(uma_buf_pool *pool, const char *name,
                            const void *data, size_t nbytes);
/* F0793: large weights via BANK then BIND as name (hot PUT stays ≤4MiB). */
int uma_buf_pool_put_weight(uma_buf_pool *pool, const char *name,
                            const char *bank_key, const void *data,
                            size_t nbytes);
/* F0994: BANK_PUT without bind (persist once, bind per block). */
int uma_buf_pool_bank_put(uma_buf_pool *pool, const char *bank_key,
                          const void *data, size_t nbytes);
int uma_buf_pool_bank_bind(uma_buf_pool *pool, const char *bank_key,
                           const char *as_name);
/* F0703: one IPC for many key:as pairs (comma-separated, no spaces). */
int uma_buf_pool_bank_binds(uma_buf_pool *pool, const char *pairs);
int uma_buf_pool_free(uma_buf_pool *pool, const char *name);
void uma_buf_pool_free_all(uma_buf_pool *pool);
int uma_buf_pool_ensure_bank_open(uma_buf_pool *pool);

#ifdef __cplusplus
}
#endif
