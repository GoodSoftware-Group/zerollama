/* Portable DiT N-slot LRU pager — no CUDA/UMA deps. See docs/dit-pager.md */
#ifndef DIT_PAGER_H
#define DIT_PAGER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dit_pager dit_pager;

typedef struct dit_pager_stats {
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  unsigned n_slots;
  unsigned occupied;
} dit_pager_stats;

/* n_slots < 1 → defaults to 2. */
dit_pager *dit_pager_create(unsigned n_slots);
void dit_pager_destroy(dit_pager *p);

/*
 * Touch layer_id. Returns slot index [0, n_slots) on success, -1 on error.
 * If a layer was evicted, *evicted_layer_out is set to that id; else -1.
 * Pass evicted_layer_out=NULL to ignore.
 */
int dit_pager_touch(dit_pager *p, unsigned layer_id, int *evicted_layer_out);

/* Optional: associate bytes with a slot after a successful touch (for peak tracking). */
void dit_pager_set_slot_bytes(dit_pager *p, int slot, size_t bytes);

/* Peak sum of slot bytes observed after set_slot_bytes. */
size_t dit_pager_peak_bytes(const dit_pager *p);
size_t dit_pager_resident_bytes(const dit_pager *p);

dit_pager_stats dit_pager_get_stats(const dit_pager *p);

/* Which layer currently owns slot, or -1 if empty. */
int dit_pager_slot_layer(const dit_pager *p, int slot);

#ifdef __cplusplus
}
#endif

#endif
