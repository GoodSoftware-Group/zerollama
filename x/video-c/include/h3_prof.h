/*
 * h3_prof.h — optional WAN_PROFILE hooks for library TUs.
 *
 * video-cli wires these to wan_profile_* in main(); test binaries that skip
 * wan_profile.o leave them NULL and profiling is a no-op. Keeps the DiT/VAE
 * weight-store objects linkable without wan_profile.o.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern double (*h3_prof_now_ms)(void);
extern void (*h3_prof_add_ms)(const char *bucket, double ms);

#ifdef __cplusplus
}
#endif