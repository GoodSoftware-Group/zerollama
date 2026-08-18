/* Multi-shard safetensors inventory for H3 component dirs (FL2VA/transformer, …). */
#ifndef H3_ST_STORE_H
#define H3_ST_STORE_H

#include "safetensors_min.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_st_store h3_st_store;

/* Open every *.safetensors under directory; recursive=1 walks subdirs. */
h3_st_store *h3_st_store_open_ex(const char *directory, int recursive,
                                 char *error, size_t error_size);
/* Non-recursive (component leaf dirs). */
h3_st_store *h3_st_store_open(const char *directory, char *error,
                              size_t error_size);
void h3_st_store_free(h3_st_store *store);

size_t h3_st_store_shards(const h3_st_store *store);
size_t h3_st_store_tensors(const h3_st_store *store);
unsigned long long h3_st_store_bytes(const h3_st_store *store);

/* Find tensor by name across shards. Sets *out_file if non-NULL. */
const st_tensor_t *h3_st_store_find(const h3_st_store *store, const char *name,
                                    const st_file **out_file);

/* Decode tensor to host f32 (BF16/F16/F32, or I8 ConvRot + weight_scale). */
int h3_st_store_load_f32(const h3_st_store *store, const char *name, float *dst,
                         size_t dst_n, char *error, size_t error_size);

/* WAN_PROFILE bucket tag for load_f32 wall time on this store (default h3_wload). */
void h3_st_store_set_prof_tag(h3_st_store *store, const char *tag);

/*
 * Cached read-only access: returns a pointer owned by the store's weight cache
 * (no per-call copy). Loads+dequants on first access. NULL on failure
 * (caller should fall back to h3_st_store_load_f32).
 */
const float *h3_st_store_get_f32(const h3_st_store *store, const char *name,
                                 size_t *n_out, char *error, size_t error_size);

/* True if ptr is a store-owned weight-cache buffer (caller must not free). */
int h3_st_store_owns(const h3_st_store *store, const void *ptr);

/* Debug: print store pointer/refcount/weight-cache stats (WAN_PROFILE dev aid). */
void h3_st_store_debug(const h3_st_store *store, const char *label);

/* mlock() every resident store's weight-cache buffers; returns total bytes
 * locked. Opt-in via H3_MLOCK=1 (see main.c serve loop). */
unsigned long long h3_st_store_mlock_all(void);

#ifdef __cplusplus
}
#endif

#endif /* H3_ST_STORE_H */
