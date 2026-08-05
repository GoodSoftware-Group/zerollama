#include "uma_buf_load.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef UMA_WAN_HOT_PUT_MAX
#define UMA_WAN_HOT_PUT_MAX (4u << 20) /* F0701/F0793 default 4 MiB */
#endif

typedef struct uma_buf_entry {
  char name[128];
  struct uma_buf_entry *next;
} uma_buf_entry;

struct uma_buf_pool {
  UmaClient *client;
  uma_buf_entry *head;
  char bank[64];
  int bank_open;
};

static char g_resp[512];

uma_buf_pool *uma_buf_pool_create(UmaClient *c) {
  uma_buf_pool *pool = calloc(1, sizeof(*pool));
  if (!pool)
    return NULL;
  pool->client = c;
  snprintf(pool->bank, sizeof(pool->bank), "wanc");
  return pool;
}

void uma_buf_pool_destroy(uma_buf_pool *pool) {
  if (!pool)
    return;
  uma_buf_pool_free_all(pool);
  if (pool->bank_open)
    (void)uma_client_bank_close(pool->client, pool->bank, g_resp,
                                sizeof(g_resp));
  free(pool);
}

static uma_buf_entry *pool_find(uma_buf_pool *pool, const char *name) {
  for (uma_buf_entry *e = pool->head; e; e = e->next) {
    if (!strcmp(e->name, name))
      return e;
  }
  return NULL;
}

static int pool_track(uma_buf_pool *pool, const char *name) {
  if (pool_find(pool, name))
    return 0;
  uma_buf_entry *e = calloc(1, sizeof(*e));
  if (!e)
    return -1;
  snprintf(e->name, sizeof(e->name), "%s", name);
  e->next = pool->head;
  pool->head = e;
  return 0;
}

static void pool_untrack(uma_buf_pool *pool, const char *name) {
  uma_buf_entry **pp = &pool->head;
  while (*pp) {
    if (!strcmp((*pp)->name, name)) {
      uma_buf_entry *dead = *pp;
      *pp = dead->next;
      free(dead);
      return;
    }
    pp = &(*pp)->next;
  }
}

int uma_buf_pool_alloc(uma_buf_pool *pool, const char *name, size_t nbytes) {
  if (!pool || !pool->client || !name)
    return -1;
  if (pool_find(pool, name))
    return 0;
  if (uma_client_buf_alloc(pool->client, name, nbytes, g_resp, sizeof(g_resp)) !=
      0)
    return -1;
  return pool_track(pool, name);
}

int uma_buf_pool_put(uma_buf_pool *pool, const char *name, const void *data,
                     size_t nbytes) {
  if (!pool || !pool->client || !name)
    return -1;
  if (uma_client_buf_put(pool->client, name, data, nbytes, g_resp,
                         sizeof(g_resp)) != 0)
    return -1;
  return pool_track(pool, name);
}

int uma_buf_pool_put_weight(uma_buf_pool *pool, const char *name,
                            const char *bank_key, const void *data,
                            size_t nbytes) {
  if (!pool || !pool->client || !name || !data || nbytes < 1)
    return -1;
  if (nbytes <= UMA_WAN_HOT_PUT_MAX) {
    if (uma_buf_pool_alloc(pool, name, nbytes) != 0) {
      fprintf(stderr, "wan-c: BUF_ALLOC fail name=%s nbytes=%zu resp=%s\n", name,
              nbytes, g_resp);
      return -1;
    }
    if (uma_buf_pool_put(pool, name, data, nbytes) != 0) {
      fprintf(stderr, "wan-c: BUF_PUT fail name=%s nbytes=%zu resp=%s\n", name,
              nbytes, g_resp);
      return -1;
    }
    return 0;
  }
  /* F0793 BANK path for dense Wan weights. */
  if (!pool->bank_open) {
    if (uma_client_bank_open(pool->client, pool->bank, g_resp, sizeof(g_resp)) !=
        0)
      return -1;
    pool->bank_open = 1;
  }
  const char *key = bank_key && bank_key[0] ? bank_key : name;
  (void)uma_client_bank_unbind(pool->client, name, g_resp, sizeof(g_resp));
  if (uma_client_bank_alloc(pool->client, pool->bank, key, nbytes, g_resp,
                            sizeof(g_resp)) != 0)
    return -1;
  if (uma_client_bank_put(pool->client, pool->bank, key, data, nbytes, g_resp,
                          sizeof(g_resp)) != 0)
    return -1;
  if (uma_client_bank_bind(pool->client, pool->bank, key, name, g_resp,
                           sizeof(g_resp)) != 0)
    return -1;
  return pool_track(pool, name);
}

int uma_buf_pool_free(uma_buf_pool *pool, const char *name) {
  if (!pool || !pool->client || !name)
    return -1;
  (void)uma_client_bank_unbind(pool->client, name, g_resp, sizeof(g_resp));
  if (uma_client_buf_free(pool->client, name, g_resp, sizeof(g_resp)) != 0)
    return -1;
  pool_untrack(pool, name);
  return 0;
}

void uma_buf_pool_free_all(uma_buf_pool *pool) {
  if (!pool)
    return;
  while (pool->head) {
    uma_buf_entry *e = pool->head;
    (void)uma_client_bank_unbind(pool->client, e->name, g_resp, sizeof(g_resp));
    uma_client_buf_free(pool->client, e->name, g_resp, sizeof(g_resp));
    pool->head = e->next;
    free(e);
  }
}
