#include "dit_pager.h"

#include <stdio.h>
#include <stdlib.h>

static int fail(const char *msg) {
  fprintf(stderr, "FAIL dit_pager: %s\n", msg);
  return 1;
}

int main(void) {
  dit_pager *p = dit_pager_create(2);
  if (!p) return fail("create");

  int ev = -1;
  int s0 = dit_pager_touch(p, 0, &ev);
  if (s0 < 0 || ev != -1) return fail("touch 0");
  dit_pager_set_slot_bytes(p, s0, 1000);

  int s1 = dit_pager_touch(p, 1, &ev);
  if (s1 < 0 || s1 == s0 || ev != -1) return fail("touch 1");
  dit_pager_set_slot_bytes(p, s1, 2000);

  /* Hit layer 0 */
  int s0b = dit_pager_touch(p, 0, &ev);
  if (s0b != s0 || ev != -1) return fail("hit 0");

  /* Miss layer 2 → evict LRU (layer 1 has older tick after hit on 0) */
  int s2 = dit_pager_touch(p, 2, &ev);
  if (s2 < 0 || ev != 1) return fail("evict 1 for 2");
  dit_pager_set_slot_bytes(p, s2, 3000);

  dit_pager_stats st = dit_pager_get_stats(p);
  if (st.hits != 1 || st.misses != 3 || st.evictions != 1)
    return fail("stats");
  if (st.occupied != 2 || st.n_slots != 2) return fail("occupied");

  size_t peak = dit_pager_peak_bytes(p);
  /* After set bytes: 1000+2000=3000, then 1000+3000=4000 */
  if (peak < 3000) return fail("peak too small");

  /* load-all would be 1000+2000+3000=6000; N=2 peak should be < 80% of that */
  size_t load_all = 6000;
  if (peak >= (size_t)(0.8 * load_all)) return fail("kill: peak too close to load-all");

  dit_pager_destroy(p);
  puts("ok: dit_pager LRU");
  return 0;
}
