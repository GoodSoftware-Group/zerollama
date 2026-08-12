/* Parse WAN_DIT_RESIDENT for wan-c / CUDA lab (0 = off / full BANK). */
#include "dit_resident.h"

#include <stdlib.h>

int wan_dit_resident_slots(void) {
  const char *e = getenv("WAN_DIT_RESIDENT");
  if (!e || !*e) return 0;
  int n = atoi(e);
  return n < 0 ? 0 : n;
}
