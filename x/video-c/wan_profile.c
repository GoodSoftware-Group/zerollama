#include "wan_profile.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  char name[32];
  double ms;
  long count;
} wan_prof_slot;

enum { WAN_PROF_SLOTS = 48 };

static wan_prof_slot g_slots[WAN_PROF_SLOTS];
static int g_nslots;
static int g_on = -1;
static int g_vae_stage = -1;
static pthread_mutex_t g_prof_lock = PTHREAD_MUTEX_INITIALIZER;

int wan_profile_on(void) {
  if (g_on < 0) {
    const char *e = getenv("WAN_PROFILE");
    g_on = (e && e[0] && e[0] != '0' && e[0] != 'n' && e[0] != 'N' &&
            e[0] != 'f' && e[0] != 'F')
               ? 1
               : 0;
  }
  return g_on;
}

int wan_profile_vae_stage_on(void) {
  if (g_vae_stage < 0) {
    if (!wan_profile_on()) {
      g_vae_stage = 0;
    } else {
      const char *e = getenv("WAN_VAE_STAGE_PROF");
      g_vae_stage = (e && e[0] == '1') ? 1 : 0;
    }
  }
  return g_vae_stage;
}

void wan_profile_reset(void) {
  pthread_mutex_lock(&g_prof_lock);
  memset(g_slots, 0, sizeof(g_slots));
  g_nslots = 0;
  pthread_mutex_unlock(&g_prof_lock);
}

double wan_profile_now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return 0.0;
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static wan_prof_slot *slot_for(const char *bucket) {
  if (!bucket || !bucket[0])
    return NULL;
  for (int i = 0; i < g_nslots; i++) {
    if (!strcmp(g_slots[i].name, bucket))
      return &g_slots[i];
  }
  if (g_nslots >= WAN_PROF_SLOTS)
    return NULL;
  wan_prof_slot *s = &g_slots[g_nslots++];
  snprintf(s->name, sizeof(s->name), "%s", bucket);
  s->ms = 0.0;
  s->count = 0;
  return s;
}

void wan_profile_add_ms(const char *bucket, double ms) {
  if (!wan_profile_on() || ms < 0.0)
    return;
  pthread_mutex_lock(&g_prof_lock);
  wan_prof_slot *s = slot_for(bucket);
  if (s) {
    s->ms += ms;
    s->count++;
  }
  pthread_mutex_unlock(&g_prof_lock);
}

void wan_profile_add_count(const char *bucket, long n) {
  if (!wan_profile_on() || n == 0)
    return;
  pthread_mutex_lock(&g_prof_lock);
  wan_prof_slot *s = slot_for(bucket);
  if (s)
    s->count += n;
  pthread_mutex_unlock(&g_prof_lock);
}

void wan_profile_report(const char *label) {
  if (!wan_profile_on() || g_nslots < 1)
    return;
  pthread_mutex_lock(&g_prof_lock);
  double total = 0.0;
  for (int i = 0; i < g_nslots; i++)
    total += g_slots[i].ms;
  fprintf(stderr, "wan-c: PROFILE %s total_ms=%.1f\n",
          label && label[0] ? label : "run", total);
  for (int i = 0; i < g_nslots; i++) {
    double pct = total > 0.0 ? 100.0 * g_slots[i].ms / total : 0.0;
    fprintf(stderr, "wan-c: PROFILE   %-16s ms=%8.1f  n=%ld  (%.1f%%)\n",
            g_slots[i].name, g_slots[i].ms, g_slots[i].count, pct);
  }
  fflush(stderr);
  pthread_mutex_unlock(&g_prof_lock);
}
