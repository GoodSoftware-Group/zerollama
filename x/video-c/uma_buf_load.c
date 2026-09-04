#include "uma_buf_load.h"
#include "wan_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef UMA_WAN_HOT_PUT_MAX
#define UMA_WAN_HOT_PUT_MAX (4u << 20) /* F0701/F0793 default 4 MiB */
#endif

typedef struct uma_buf_entry {
  char name[128];
  size_t nbytes;
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

void uma_buf_pool_destroy_keep_bank(uma_buf_pool *pool) {
  if (!pool)
    return;
  uma_buf_pool_free_all(pool);
  free(pool);
}

int uma_buf_pool_ensure_bank_open(uma_buf_pool *pool) {
  if (!pool || !pool->client)
    return -1;
  if (pool->bank_open)
    return 0;
  if (uma_client_bank_open(pool->client, pool->bank, g_resp, sizeof(g_resp)) !=
      0) {
    // If bank already exists, open will succeed (returns OK). If not, it will create.
    // Try force open as fallback.
    if (uma_client_bank_force_open(pool->client, pool->bank, g_resp,
                                   sizeof(g_resp)) != 0)
      return -1;
  }
  pool->bank_open = 1;
  return 0;
}

static uma_buf_entry *pool_find(uma_buf_pool *pool, const char *name) {
  for (uma_buf_entry *e = pool->head; e; e = e->next) {
    if (!strcmp(e->name, name))
      return e;
  }
  return NULL;
}

static int pool_track_sz(uma_buf_pool *pool, const char *name, size_t nbytes) {
  uma_buf_entry *e = pool_find(pool, name);
  if (e) {
    e->nbytes = nbytes;
    return 0;
  }
  e = calloc(1, sizeof(*e));
  if (!e)
    return -1;
  snprintf(e->name, sizeof(e->name), "%s", name);
  e->nbytes = nbytes;
  e->next = pool->head;
  pool->head = e;
  return 0;
}

static int pool_track(uma_buf_pool *pool, const char *name) {
  return pool_track_sz(pool, name, 0);
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
  static int sticky = -1;
  if (sticky < 0) {
    const char *e = getenv("WAN_BUF_STICKY");
    /* Default on: skip re-assert when size matches. WAN_BUF_STICKY=0 restores
     * always-reassert (safer under extreme CAP pressure). */
    if (e && e[0] && (e[0] == '0' || e[0] == 'n' || e[0] == 'N' ||
                       e[0] == 'f' || e[0] == 'F'))
      sticky = 0;
    else
      sticky = 1;
  }
  uma_buf_entry *e = pool_find(pool, name);
  if (e && sticky && e->nbytes == nbytes && nbytes > 0) {
    wan_profile_add_count("buf_skip", 1);
    return 0;
  }
  if (e) {
    if (!sticky && e->nbytes == nbytes && nbytes > 0) {
      /* Sticky off: re-assert in place without free (legacy CAP recovery). */
      double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
      if (uma_client_buf_alloc(pool->client, name, nbytes, g_resp,
                               sizeof(g_resp)) == 0) {
        if (wan_profile_on())
          wan_profile_add_ms("buf_alloc", wan_profile_now_ms() - t0);
        e->nbytes = nbytes;
        return 0;
      }
    }
    (void)uma_client_bank_unbind(pool->client, name, g_resp, sizeof(g_resp));
    (void)uma_client_buf_free(pool->client, name, g_resp, sizeof(g_resp));
    pool_untrack(pool, name);
  }
  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  if (uma_client_buf_alloc(pool->client, name, nbytes, g_resp, sizeof(g_resp)) !=
      0) {
    fprintf(stderr, "wan-c: BUF_ALLOC fail name=%s nbytes=%zu resp=%.120s\n",
            name, nbytes, g_resp);
    return -1;
  }
  if (wan_profile_on())
    wan_profile_add_ms("buf_alloc", wan_profile_now_ms() - t0);
  return pool_track_sz(pool, name, nbytes);
}

int uma_buf_pool_ensure_put(uma_buf_pool *pool, const char *name,
                            const void *data, size_t nbytes) {
  if (uma_buf_pool_alloc(pool, name, nbytes) != 0)
    return -1;
  if (uma_buf_pool_put(pool, name, data, nbytes) == 0)
    return 0;
  /* Broker may have dropped the slot — force free + fresh alloc. */
  (void)uma_buf_pool_free(pool, name);
  if (uma_buf_pool_alloc(pool, name, nbytes) != 0)
    return -1;
  if (uma_buf_pool_put(pool, name, data, nbytes) == 0)
    return 0;
  fprintf(stderr, "wan-c: BUF_PUT fail name=%s nbytes=%zu resp=%.120s\n", name,
          nbytes, g_resp);
  return -1;
}

int uma_buf_pool_put(uma_buf_pool *pool, const char *name, const void *data,
                     size_t nbytes) {
  if (!pool || !pool->client || !name)
    return -1;
  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  if (uma_client_buf_put(pool->client, name, data, nbytes, g_resp,
                         sizeof(g_resp)) != 0)
    return -1;
  if (wan_profile_on())
    wan_profile_add_ms("buf_put", wan_profile_now_ms() - t0);
  return pool_track_sz(pool, name, nbytes);
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
  return pool_track_sz(pool, name, nbytes);
}

/* F0994: persist weight bytes under bank_key; no bind yet. */
int uma_buf_pool_bank_put(uma_buf_pool *pool, const char *bank_key,
                          const void *data, size_t nbytes) {
  if (!pool || !pool->client || !bank_key || !bank_key[0] || !data || nbytes < 1)
    return -1;
  if (!pool->bank_open) {
    if (uma_client_bank_open(pool->client, pool->bank, g_resp, sizeof(g_resp)) !=
        0)
      return -1;
    pool->bank_open = 1;
  }
  if (uma_client_bank_alloc(pool->client, pool->bank, bank_key, nbytes, g_resp,
                            sizeof(g_resp)) != 0)
    return -1;
  if (uma_client_bank_put(pool->client, pool->bank, bank_key, data, nbytes,
                          g_resp, sizeof(g_resp)) != 0)
    return -1;
  return 0;
}

int uma_buf_pool_bank_bind(uma_buf_pool *pool, const char *bank_key,
                           const char *as_name) {
  if (!pool || !pool->client || !bank_key || !bank_key[0] || !as_name ||
      !as_name[0])
    return -1;
  if (!pool->bank_open)
    return -1;
  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  (void)uma_client_bank_unbind(pool->client, as_name, g_resp, sizeof(g_resp));
  if (uma_client_bank_bind(pool->client, pool->bank, bank_key, as_name, g_resp,
                           sizeof(g_resp)) != 0)
    return -1;
  if (wan_profile_on())
    wan_profile_add_ms("bank_bind", wan_profile_now_ms() - t0);
  return pool_track(pool, as_name);
}

int uma_buf_pool_bank_binds(uma_buf_pool *pool, const char *pairs) {
  if (!pool || !pool->client || !pairs || !pairs[0] || !pool->bank_open)
    return -1;
  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  if (uma_client_bank_binds(pool->client, pool->bank, pairs, g_resp,
                            sizeof(g_resp)) != 0) {
    fprintf(stderr, "wan-c: BANK_BINDS fail bank=%s pairs=%.80s resp=%.120s\n",
            pool->bank, pairs, g_resp);
    return -1;
  }
  if (wan_profile_on())
    wan_profile_add_ms("bank_bind", wan_profile_now_ms() - t0);
  /* Track each as= alias (pairs = key:as,key:as,…). */
  const char *p = pairs;
  while (*p) {
    const char *colon = strchr(p, ':');
    if (!colon) {
      fprintf(stderr, "wan-c: BANK_BINDS track fail no colon pairs=%.80s\n",
              pairs);
      return -1;
    }
    const char *as = colon + 1;
    const char *comma = strchr(as, ',');
    char name[128];
    size_t n = comma ? (size_t)(comma - as) : strlen(as);
    if (n >= sizeof(name))
      n = sizeof(name) - 1;
    memcpy(name, as, n);
    name[n] = '\0';
    if (name[0])
      (void)pool_track(pool, name);
    if (!comma)
      break;
    p = comma + 1;
  }
  return 0;
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
