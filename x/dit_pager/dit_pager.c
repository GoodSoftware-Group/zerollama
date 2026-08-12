#include "dit_pager.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
  int layer; /* -1 empty */
  uint64_t tick;
  size_t bytes;
} dit_pager_slot;

struct dit_pager {
  unsigned n_slots;
  uint64_t clock;
  dit_pager_slot *slots;
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  size_t peak_bytes;
};

dit_pager *dit_pager_create(unsigned n_slots) {
  if (n_slots < 1) n_slots = 2;
  dit_pager *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->n_slots = n_slots;
  p->slots = calloc(n_slots, sizeof(*p->slots));
  if (!p->slots) {
    free(p);
    return NULL;
  }
  for (unsigned i = 0; i < n_slots; i++) p->slots[i].layer = -1;
  return p;
}

void dit_pager_destroy(dit_pager *p) {
  if (!p) return;
  free(p->slots);
  free(p);
}

static size_t resident_sum(const dit_pager *p) {
  size_t s = 0;
  for (unsigned i = 0; i < p->n_slots; i++)
    if (p->slots[i].layer >= 0) s += p->slots[i].bytes;
  return s;
}

int dit_pager_touch(dit_pager *p, unsigned layer_id, int *evicted_layer_out) {
  if (!p || !p->slots) return -1;
  if (evicted_layer_out) *evicted_layer_out = -1;
  p->clock++;

  for (unsigned i = 0; i < p->n_slots; i++) {
    if (p->slots[i].layer == (int)layer_id) {
      p->slots[i].tick = p->clock;
      p->hits++;
      return (int)i;
    }
  }

  p->misses++;
  /* Prefer empty slot. */
  for (unsigned i = 0; i < p->n_slots; i++) {
    if (p->slots[i].layer < 0) {
      p->slots[i].layer = (int)layer_id;
      p->slots[i].tick = p->clock;
      p->slots[i].bytes = 0;
      return (int)i;
    }
  }

  /* Evict LRU (smallest tick). */
  unsigned victim = 0;
  for (unsigned i = 1; i < p->n_slots; i++) {
    if (p->slots[i].tick < p->slots[victim].tick) victim = i;
  }
  if (evicted_layer_out) *evicted_layer_out = p->slots[victim].layer;
  p->evictions++;
  p->slots[victim].layer = (int)layer_id;
  p->slots[victim].tick = p->clock;
  p->slots[victim].bytes = 0;
  return (int)victim;
}

void dit_pager_set_slot_bytes(dit_pager *p, int slot, size_t bytes) {
  if (!p || slot < 0 || (unsigned)slot >= p->n_slots) return;
  p->slots[slot].bytes = bytes;
  size_t cur = resident_sum(p);
  if (cur > p->peak_bytes) p->peak_bytes = cur;
}

size_t dit_pager_peak_bytes(const dit_pager *p) {
  return p ? p->peak_bytes : 0;
}

size_t dit_pager_resident_bytes(const dit_pager *p) {
  return p ? resident_sum(p) : 0;
}

dit_pager_stats dit_pager_get_stats(const dit_pager *p) {
  dit_pager_stats s = {0};
  if (!p) return s;
  s.hits = p->hits;
  s.misses = p->misses;
  s.evictions = p->evictions;
  s.n_slots = p->n_slots;
  for (unsigned i = 0; i < p->n_slots; i++)
    if (p->slots[i].layer >= 0) s.occupied++;
  return s;
}

int dit_pager_slot_layer(const dit_pager *p, int slot) {
  if (!p || slot < 0 || (unsigned)slot >= p->n_slots) return -1;
  return p->slots[slot].layer;
}
