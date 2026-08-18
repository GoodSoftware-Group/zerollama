/*
 * wan_profile.h — WAN_PROFILE=1 wall timers (Phase 0 speed gap).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int wan_profile_on(void);
/* WAN_PROFILE=1 and WAN_VAE_STAGE_PROF=1 — Brick 11 tip stage map. */
int wan_profile_vae_stage_on(void);
void wan_profile_reset(void);
void wan_profile_add_ms(const char *bucket, double ms);
void wan_profile_add_count(const char *bucket, long n);
void wan_profile_report(const char *label);

/* Monotonic ms since boot (or 0 if unavailable). */
double wan_profile_now_ms(void);

#ifdef __cplusplus
}
#endif
