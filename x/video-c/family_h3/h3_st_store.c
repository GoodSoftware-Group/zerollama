#include "h3_st_store.h"

#include "h3_convrot.h"
#include "h3_prof.h"

double (*h3_prof_now_ms)(void) = NULL;
void (*h3_prof_add_ms)(const char *bucket, double ms) = NULL;

#include <dirent.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>

typedef struct {
  char *name;
  float *data;
  size_t n;
} h3_wc_ent;

/* Open-addressing name hash over the weight cache (FNV-1a). Slots hold
 * wc-index+1; 0 = empty, UINT_MAX = tombstone. Rebuilt when the entry array
 * grows. Replaces the old O(n) strcmp scan (~600 tensors × thousands of
 * lookups per generate). */
struct h3_st_store {
  st_file **shards;
  size_t n_shards;
  size_t n_tensors;
  unsigned long long bytes;
  h3_wc_ent *wc;
  size_t wc_n;
  size_t wc_cap;
  size_t wc_bytes;
  size_t wc_limit;
  unsigned *htab;
  size_t htab_cap; /* power of two, >= 2 * wc_cap */
  unsigned long wc_hits;
  unsigned long wc_misses;
  int wc_full_logged;
  char prof_tag[24];
  char *key;
  int refcount;
};

static unsigned h3_wc_hash(const char *s) {
  unsigned h = 2166136261u;
  while (*s) {
    h ^= (unsigned)(unsigned char)*s++;
    h *= 16777619u;
  }
  return h;
}

static int h3_wc_htab_init(h3_st_store *store, size_t min_cap) {
  size_t cap = 64;
  while (cap < min_cap * 2)
    cap <<= 1;
  unsigned *ht = (unsigned *)calloc(cap, sizeof(unsigned));
  if (!ht)
    return -1;
  free(store->htab);
  store->htab = ht;
  store->htab_cap = cap;
  return 0;
}

static void h3_wc_htab_rebuild(h3_st_store *store) {
  if (h3_wc_htab_init(store, store->wc_cap) != 0)
    return;
  for (size_t i = 0; i < store->wc_n; i++) {
    unsigned h = h3_wc_hash(store->wc[i].name) & (unsigned)(store->htab_cap - 1);
    while (store->htab[h] && store->htab[h] != UINT_MAX)
      h = (h + 1) & (unsigned)(store->htab_cap - 1);
    store->htab[h] = (unsigned)(i + 1);
  }
}

/* Returns wc index+1 for `name` with nelems elements, or 0. */
static unsigned h3_wc_htab_find(const h3_st_store *store, const char *name,
                                size_t nelems) {
  if (!store->htab || !store->htab_cap)
    return 0;
  unsigned h = h3_wc_hash(name) & (unsigned)(store->htab_cap - 1);
  for (;;) {
    unsigned slot = store->htab[h];
    if (!slot)
      return 0;
    if (slot != UINT_MAX) {
      const h3_wc_ent *e = &store->wc[slot - 1];
      if (e->n == nelems && e->name && strcmp(e->name, name) == 0)
        return slot;
    }
    h = (h + 1) & (unsigned)(store->htab_cap - 1);
  }
}

static void h3_wc_htab_insert(h3_st_store *store, const char *name,
                              unsigned idx1) {
  if (!store->htab && h3_wc_htab_init(store, store->wc_cap ? store->wc_cap : 64) != 0)
    return;
  /* Grow (rehash) when half full relative to entries. */
  if (store->wc_n * 2 >= store->htab_cap)
    h3_wc_htab_rebuild(store);
  if (!store->htab)
    return;
  unsigned h = h3_wc_hash(name) & (unsigned)(store->htab_cap - 1);
  while (store->htab[h] && store->htab[h] != UINT_MAX)
    h = (h + 1) & (unsigned)(store->htab_cap - 1);
  store->htab[h] = idx1;
}

/* Shared-store registry: h3_st_store_open of the same directory returns the
 * same store (refcounted) so a resident daemon can keep weight caches warm
 * across requests while per-call open/free pairs stay balanced. Guarded by a
 * mutex so independent stores (DiT vs VAE vs TE) can be opened concurrently
 * when the VAE decodes overlap. */
typedef struct {
  char *key;
  h3_st_store *store;
} h3_resident_ent;

static h3_resident_ent *residents;
static size_t residents_n;
static size_t residents_cap;
static pthread_mutex_t g_res_lock = PTHREAD_MUTEX_INITIALIZER;

static h3_st_store *resident_lookup_unlocked(const char *key) {
  for (size_t i = 0; i < residents_n; i++) {
    if (residents[i].key && strcmp(residents[i].key, key) == 0)
      return residents[i].store;
  }
  return NULL;
}

static h3_st_store *resident_lookup(const char *key) {
  pthread_mutex_lock(&g_res_lock);
  h3_st_store *r = resident_lookup_unlocked(key);
  pthread_mutex_unlock(&g_res_lock);
  return r;
}

static void resident_add(const char *key, h3_st_store *store) {
  pthread_mutex_lock(&g_res_lock);
  if (resident_lookup_unlocked(key)) {
    pthread_mutex_unlock(&g_res_lock);
    return;
  }
  if (residents_n >= residents_cap) {
    size_t ncap = residents_cap ? residents_cap * 2 : 8;
    h3_resident_ent *nw = realloc(residents, ncap * sizeof(*nw));
    if (!nw) {
      pthread_mutex_unlock(&g_res_lock);
      return;
    }
    residents = nw;
    residents_cap = ncap;
  }
  char *k = strdup(key);
  if (!k) {
    pthread_mutex_unlock(&g_res_lock);
    return;
  }
  residents[residents_n].key = k;
  residents[residents_n].store = store;
  residents_n++;
  pthread_mutex_unlock(&g_res_lock);
}

static void resident_remove_unlocked(h3_st_store *store) {
  for (size_t i = 0; i < residents_n; i++) {
    if (residents[i].store == store) {
      free(residents[i].key);
      residents[i] = residents[residents_n - 1];
      residents_n--;
      return;
    }
  }
}

static void resident_remove(h3_st_store *store) {
  pthread_mutex_lock(&g_res_lock);
  resident_remove_unlocked(store);
  pthread_mutex_unlock(&g_res_lock);
}

static size_t h3_weight_cache_limit(void) {
  const char *e = getenv("H3_WEIGHT_CACHE_MB");
  if (e && e[0]) {
    long mb = atol(e);
    if (mb <= 0)
      return 0;
    return (size_t)mb * 1024ull * 1024ull;
  }
  /* 48 GiB: this Mac has 128 GiB; I8 ConvRot expands ~4×. Set 0 to disable. */
  return 80ull * 1024ull * 1024ull * 1024ull;
}

static const float *wc_find(h3_st_store *store, const char *name, size_t n) {
  if (!store || !store->wc_limit)
    return NULL;
  unsigned slot = h3_wc_htab_find(store, name, n);
  return slot ? store->wc[slot - 1].data : NULL;
}

static void wc_insert(h3_st_store *store, const char *name, const float *src,
                      size_t n) {
  if (!store || !store->wc_limit || !name || !src || n < 1)
    return;
  size_t add = n * sizeof(float);
  if (store->wc_bytes + add > store->wc_limit) {
    if (!store->wc_full_logged) {
      fprintf(stderr,
              "video-c: DiT weight cache full (%.1f GiB, %zu tensors); later "
              "loads stay cold\n",
              (double)store->wc_bytes / (1024.0 * 1024.0 * 1024.0),
              store->wc_n);
      store->wc_full_logged = 1;
    }
    return;
  }
  if (store->wc_n >= store->wc_cap) {
    size_t ncap = store->wc_cap ? store->wc_cap * 2 : 64;
    h3_wc_ent *nw = realloc(store->wc, ncap * sizeof(*nw));
    if (!nw)
      return;
    store->wc = nw;
    store->wc_cap = ncap;
  }
  char *nm = strdup(name);
  float *copy = (float *)malloc(add);
  if (!nm || !copy) {
    free(nm);
    free(copy);
    return;
  }
  memcpy(copy, src, add);
  store->wc[store->wc_n].name = nm;
  store->wc[store->wc_n].data = copy;
  store->wc[store->wc_n].n = n;
  store->wc_n++;
  store->wc_bytes += add;
  h3_wc_htab_insert(store, nm, (unsigned)store->wc_n);
}

/* Pin resident weight-cache pages (opt-in H3_MLOCK=1) so memory pressure
 * cannot swap them out between daemon requests. Returns bytes locked. */
static unsigned long long wc_mlock(h3_st_store *store) {
  unsigned long long total = 0;
  if (!store)
    return 0;
  for (size_t i = 0; i < store->wc_n; i++) {
    const void *p = store->wc[i].data;
    size_t n = store->wc[i].n * sizeof(float);
    if (p && n && mlock(p, n) == 0)
      total += n;
  }
  return total;
}

unsigned long long h3_st_store_mlock_all(void) {
  unsigned long long total = 0;
  for (size_t i = 0; i < residents_n; i++) {
    unsigned long long n = wc_mlock(residents[i].store);
    if (getenv("H3_MLOCK_DBG")) {
      fprintf(stderr,
              "video-c: mlock %-60s wc_n=%zu tensors=%zu %.2f GiB\n",
              residents[i].key ? residents[i].key : "(null)",
              residents[i].store->wc_n, residents[i].store->n_tensors,
              (double)n / (1024.0 * 1024.0 * 1024.0));
      fflush(stderr);
    }
    total += n;
  }
  return total;
}

/* Insert an already-allocated f32 buffer, taking ownership (no copy). Returns 1
 * on success, 0 if the cache is disabled/full (caller keeps ownership). */
static int wc_insert_take(h3_st_store *store, const char *name, float *data,
                          size_t n) {
  if (!store || !store->wc_limit || !name || !data || n < 1)
    return 0;
  size_t add = n * sizeof(float);
  if (store->wc_bytes + add > store->wc_limit) {
    if (!store->wc_full_logged) {
      fprintf(stderr,
              "video-c: DiT weight cache full (%.1f GiB, %zu tensors); later "
              "loads stay cold\n",
              (double)store->wc_bytes / (1024.0 * 1024.0 * 1024.0),
              store->wc_n);
      store->wc_full_logged = 1;
    }
    return 0;
  }
  if (store->wc_n >= store->wc_cap) {
    size_t ncap = store->wc_cap ? store->wc_cap * 2 : 64;
    h3_wc_ent *nw = realloc(store->wc, ncap * sizeof(*nw));
    if (!nw)
      return 0;
    store->wc = nw;
    store->wc_cap = ncap;
  }
  char *nm = strdup(name);
  if (!nm)
    return 0;
  store->wc[store->wc_n].name = nm;
  store->wc[store->wc_n].data = data;
  store->wc[store->wc_n].n = n;
  store->wc_n++;
  store->wc_bytes += add;
  h3_wc_htab_insert(store, nm, (unsigned)store->wc_n);
  return 1;
}

typedef struct {
  char **paths;
  size_t n;
  size_t cap;
} path_list;

static int ends_with_st(const char *name) {
  size_t n = strlen(name);
  return n > 12 && strcmp(name + n - 12, ".safetensors") == 0;
}

static int path_list_push(path_list *pl, const char *path) {
  if (pl->n >= pl->cap) {
    size_t ncap = pl->cap ? pl->cap * 2 : 8;
    char **np = realloc(pl->paths, ncap * sizeof(char *));
    if (!np)
      return -1;
    pl->paths = np;
    pl->cap = ncap;
  }
  char *copy = strdup(path);
  if (!copy)
    return -1;
  pl->paths[pl->n++] = copy;
  return 0;
}

static void path_list_free(path_list *pl) {
  if (!pl)
    return;
  for (size_t i = 0; i < pl->n; i++)
    free(pl->paths[i]);
  free(pl->paths);
  pl->paths = NULL;
  pl->n = pl->cap = 0;
}

static int collect_st(const char *directory, int recursive, path_list *pl,
                      char *error, size_t error_size) {
  DIR *d = opendir(directory);
  if (!d) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store: cannot open %s", directory);
    return -1;
  }
  struct dirent *ent;
  char path[1400];
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.')
      continue;
    int n = snprintf(path, sizeof(path), "%s/%s", directory, ent->d_name);
    if (n < 0 || (size_t)n >= sizeof(path))
      continue;
    struct stat st;
    if (stat(path, &st) != 0)
      continue;
    if (S_ISDIR(st.st_mode)) {
      if (recursive && collect_st(path, 1, pl, error, error_size) != 0) {
        closedir(d);
        return -1;
      }
      continue;
    }
    if (!S_ISREG(st.st_mode) || !ends_with_st(ent->d_name))
      continue;
    if (path_list_push(pl, path) != 0) {
      if (error && error_size)
        snprintf(error, error_size, "h3_st_store: OOM collecting %s", path);
      closedir(d);
      return -1;
    }
  }
  closedir(d);
  return 0;
}

h3_st_store *h3_st_store_open_ex(const char *directory, int recursive,
                                 char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!directory || !directory[0]) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store: empty directory");
    return NULL;
  }
  h3_st_store *shared = resident_lookup(directory);
  if (shared) {
    pthread_mutex_lock(&g_res_lock);
    shared->refcount++;
    pthread_mutex_unlock(&g_res_lock);
    return shared;
  }

  path_list pl = {0};
  struct stat stpath;
  if (stat(directory, &stpath) == 0 && S_ISREG(stpath.st_mode) &&
      ends_with_st(directory)) {
    if (path_list_push(&pl, directory) != 0) {
      path_list_free(&pl);
      return NULL;
    }
  } else if (collect_st(directory, recursive ? 1 : 0, &pl, error, error_size) !=
             0) {
    path_list_free(&pl);
    return NULL;
  }
  if (pl.n == 0) {
    path_list_free(&pl);
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store: no .safetensors in %s",
               directory);
    return NULL;
  }

  h3_st_store *store = calloc(1, sizeof(*store));
  if (!store) {
    path_list_free(&pl);
    return NULL;
  }
  store->wc_limit = h3_weight_cache_limit();
  snprintf(store->prof_tag, sizeof(store->prof_tag), "h3_wload");
  store->refcount = 1;
  store->key = strdup(directory);
  if (!store->key) {
    h3_st_store_free(store);
    path_list_free(&pl);
    return NULL;
  }
  store->shards = calloc(pl.n, sizeof(st_file *));
  if (!store->shards) {
    free(store->key);
    free(store);
    path_list_free(&pl);
    return NULL;
  }

  for (size_t i = 0; i < pl.n; i++) {
    st_file *sf = st_open(pl.paths[i]);
    if (!sf) {
      if (error && error_size)
        snprintf(error, error_size, "h3_st_store: failed to open %s",
                 pl.paths[i]);
      h3_st_store_free(store);
      path_list_free(&pl);
      return NULL;
    }
    store->shards[store->n_shards++] = sf;
    store->n_tensors += (size_t)st_tensor_count(sf);
    struct stat st;
    if (stat(pl.paths[i], &st) == 0)
      store->bytes += (unsigned long long)st.st_size;
  }
  path_list_free(&pl);
  resident_add(directory, store);
  return store;
}

h3_st_store *h3_st_store_open(const char *directory, char *error,
                              size_t error_size) {
  return h3_st_store_open_ex(directory, 0, error, error_size);
}

void h3_st_store_free(h3_st_store *store) {
  if (!store)
    return;
  pthread_mutex_lock(&g_res_lock);
  if (store->refcount > 0) {
    store->refcount--;
    if (store->refcount > 0) {
      pthread_mutex_unlock(&g_res_lock);
      return;
    }
  }
  resident_remove_unlocked(store);
  pthread_mutex_unlock(&g_res_lock);
  free(store->key);
  if (store->wc_n)
    fprintf(stderr,
            "video-c: DiT weight cache hits=%lu misses=%lu tensors=%zu "
            "%.2f GiB\n",
            store->wc_hits, store->wc_misses, store->wc_n,
            (double)store->wc_bytes / (1024.0 * 1024.0 * 1024.0));
  for (size_t i = 0; i < store->wc_n; i++) {
    free(store->wc[i].name);
    free(store->wc[i].data);
  }
  free(store->wc);
  free(store->htab);
  for (size_t i = 0; i < store->n_shards; i++)
    st_close(store->shards[i]);
  free(store->shards);
  free(store);
}

size_t h3_st_store_shards(const h3_st_store *store) {
  return store ? store->n_shards : 0;
}
size_t h3_st_store_tensors(const h3_st_store *store) {
  return store ? store->n_tensors : 0;
}
unsigned long long h3_st_store_bytes(const h3_st_store *store) {
  return store ? store->bytes : 0;
}

const st_tensor_t *h3_st_store_find(const h3_st_store *store, const char *name,
                                    const st_file **out_file) {
  if (out_file)
    *out_file = NULL;
  if (!store || !name)
    return NULL;
  for (size_t i = 0; i < store->n_shards; i++) {
    const st_tensor_t *t = st_find_tensor(store->shards[i], name);
    if (t) {
      if (out_file)
        *out_file = store->shards[i];
      return t;
    }
  }
  return NULL;
}

/* Decode a tensor to host f32 (no weight-cache interaction). */
static int h3_st_store_decode_into(const h3_st_store *store, const char *name,
                                   float *dst, size_t dst_n, char *error,
                                   size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!store || !name || !dst) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: bad args");
    return -1;
  }
  const st_file *sf = NULL;
  const st_tensor_t *t = h3_st_store_find(store, name, &sf);
  if (!t || !sf) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: missing %s", name);
    return -1;
  }
  if (st_tensor_to_f32(sf, t, dst, dst_n) == 0) {
    return 0;
  }
  if (t->dtype != ST_DTYPE_I8 || t->ndim != 2) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: decode failed for %s",
               name);
    return -1;
  }

  int rows = (int)t->shape[0];
  int cols = (int)t->shape[1];
  if (rows < 1 || cols < 1 || dst_n < (size_t)rows * (size_t)cols) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: bad I8 shape for %s",
               name);
    return -1;
  }

  char scale_name[280];
  char cq_name[280];
  snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
  cq_name[0] = 0;
  size_t nlen = strlen(name);
  const char *w = ".weight";
  size_t wl = 7;
  if (nlen > wl && strcmp(name + nlen - wl, w) == 0) {
    snprintf(cq_name, sizeof(cq_name), "%.*s.comfy_quant", (int)(nlen - wl),
             name);
  }

  const st_file *sf_s = NULL;
  const st_tensor_t *ts = h3_st_store_find(store, scale_name, &sf_s);
  if (!ts || !sf_s || st_tensor_nelems(ts) < (size_t)rows) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: missing %s",
               scale_name);
    return -1;
  }
  float *scale = (float *)malloc((size_t)rows * sizeof(float));
  if (!scale)
    return -1;
  if (st_tensor_to_f32(sf_s, ts, scale, (size_t)rows) != 0) {
    free(scale);
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: scale decode %s",
               scale_name);
    return -1;
  }

  int gs = 0;
  if (cq_name[0]) {
    const st_file *sf_c = NULL;
    const st_tensor_t *tc = h3_st_store_find(store, cq_name, &sf_c);
    if (tc && sf_c && tc->dtype == ST_DTYPE_U8) {
      const uint8_t *raw = st_tensor_ptr(sf_c, tc);
      if (raw)
        h3_comfy_quant_parse(raw, tc->nbytes, &gs);
    }
  }

  const uint8_t *qraw = st_tensor_ptr(sf, t);
  if (!qraw || t->nbytes < (size_t)rows * (size_t)cols) {
    free(scale);
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: I8 payload %s", name);
    return -1;
  }
  int rc = h3_convrot_dequant_i8((const int8_t *)qraw, rows, cols, scale, gs,
                                 dst);
  free(scale);
  if (rc != 0) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: convrot dequant %s",
               name);
    return -1;
  }
  return 0;
}

static int h3_st_store_load_f32_impl(const h3_st_store *store, const char *name,
                                     float *dst, size_t dst_n, char *error,
                                     size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!store || !name || !dst) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_load_f32: bad args");
    return -1;
  }
  h3_st_store *mut = (h3_st_store *)store;
  if (mut->wc_limit) {
    const float *hit = wc_find(mut, name, dst_n);
    if (hit) {
      mut->wc_hits++;
      memcpy(dst, hit, dst_n * sizeof(float));
      return 0;
    }
    mut->wc_misses++;
  }
  if (h3_st_store_decode_into(store, name, dst, dst_n, error, error_size) != 0)
    return -1;
  wc_insert(mut, name, dst, dst_n);
  return 0;
}

int h3_st_store_load_f32(const h3_st_store *store, const char *name, float *dst,
                         size_t dst_n, char *error, size_t error_size) {
  if (h3_prof_now_ms) {
    double t0 = h3_prof_now_ms();
    int rc = h3_st_store_load_f32_impl(store, name, dst, dst_n, error, error_size);
    if (t0 > 0)
      h3_prof_add_ms(((const h3_st_store *)store)->prof_tag,
                     h3_prof_now_ms() - t0);
    return rc;
  }
  return h3_st_store_load_f32_impl(store, name, dst, dst_n, error, error_size);
}

void h3_st_store_set_prof_tag(h3_st_store *store, const char *tag) {
  if (!store || !tag || !tag[0])
    return;
  snprintf(store->prof_tag, sizeof(store->prof_tag), "%s", tag);
}

const float *h3_st_store_get_f32(const h3_st_store *store, const char *name,
                                 size_t *n_out, char *error, size_t error_size) {
  if (!store || !name)
    return NULL;
  if (error && error_size)
    error[0] = 0;
  const st_file *sf = NULL;
  const st_tensor_t *t = h3_st_store_find(store, name, &sf);
  if (!t || !sf) {
    if (error && error_size)
      snprintf(error, error_size, "h3_st_store_get_f32: missing %s", name);
    return NULL;
  }
  size_t nelems = st_tensor_nelems(t);
  if (n_out)
    *n_out = nelems;
  h3_st_store *mut = (h3_st_store *)store;
  const float *hit = wc_find(mut, name, nelems);
  if (hit)
    return hit;
  float *buf = (float *)malloc(nelems * sizeof(float));
  if (!buf)
    return NULL;
  double t0 = h3_prof_now_ms ? h3_prof_now_ms() : 0;
  int rc = h3_st_store_decode_into(store, name, buf, nelems, error,
                                   error_size);
  if (t0 > 0 && h3_prof_add_ms)
    h3_prof_add_ms(((const h3_st_store *)store)->prof_tag, h3_prof_now_ms() - t0);
  if (rc != 0) {
    free(buf);
    return NULL;
  }
  if (!wc_insert_take(mut, name, buf, nelems)) {
    free(buf);
    return NULL;
  }
  return buf;
}

int h3_st_store_owns(const h3_st_store *store, const void *ptr) {
  if (!store || !ptr)
    return 0;
  for (size_t i = 0; i < store->wc_n; i++) {
    if (store->wc[i].data == ptr)
      return 1;
  }
  return 0;
}

void h3_st_store_debug(const h3_st_store *store, const char *label) {
  if (!store)
    return;
  fprintf(stderr,
          "video-c: [dbg] %s st=%p ref=%d wc_hits=%lu wc_misses=%lu "
          "tensors=%zu %.2f GiB\n",
          label ? label : "", (const void *)store, store->refcount,
          store->wc_hits, store->wc_misses, store->wc_n,
          (double)store->wc_bytes / (1024.0 * 1024.0 * 1024.0));
}
