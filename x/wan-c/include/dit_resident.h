/* WAN_DIT_RESIDENT → N-slot count for dit_pager. See docs/dit-pager.md */
#ifndef WAN_DIT_RESIDENT_H
#define WAN_DIT_RESIDENT_H

#ifdef __cplusplus
extern "C" {
#endif

/* 0 = unset/off (full BANK / pager disabled). >0 = max resident DiT blocks. */
int wan_dit_resident_slots(void);

#ifdef __cplusplus
}
#endif

#endif
