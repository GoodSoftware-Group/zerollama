// ZEROLLAMA_LINUX_ANE_DRAFT_STUB
// WHY: ANE dflash draft is Darwin/lab-only (IOSurface + private ANE). Linux builds need
// the same symbols linked into llama-common without pulling Metal/ANE. Full source is
// kept beside this file as ane_draft_hook.cpp.darwin for Mac sync.
// See docs/ane-draft-inprocess.md.
#include "ane_draft_hook.h"

#include <atomic>

bool common_ane_draft_enabled() { return false; }

void common_ane_draft_bind_target_ctx(struct llama_context *) {}

void common_ane_draft_sync_target_cross(
        struct llama_context *,
        struct llama_context *,
        const llama_batch &) {}

void common_ane_draft_log_init(common_speculative_type, int) {}

void common_ane_draft_handoff_after_decode(struct llama_context *, int32_t) {}

void common_ane_draft_note_handoff_token(llama_token) {}

void common_ane_draft_note_handoff_pos(llama_pos) {}

llama_token common_ane_draft_last_handoff_token(void) { return LLAMA_TOKEN_NULL; }

void common_ane_draft_reset_handoff(void) {}

enum common_ane_draft_drive_mode common_ane_draft_get_drive_mode(void) {
    return COMMON_ANE_DRAFT_DRIVE_OFF;
}

bool common_ane_draft_try_drive_token(
        struct llama_context *, int32_t, llama_token *, float *, float *) {
    return false;
}

bool common_ane_draft_metal_ref_token(struct llama_context *, int32_t, llama_token *) {
    return false;
}

int32_t common_ane_draft_handoff_i_batch(void) { return -1; }

bool common_ane_draft_rebind_drive_slot(struct llama_context *, int32_t) { return false; }

void common_ane_draft_shutdown_iosurface_cache(void) {}

bool ane_dflash_qkv_host_fp32_enabled(void) { return false; }

bool ane_dflash_fill_session_qkv_host_fp32(int, int) { return false; }
