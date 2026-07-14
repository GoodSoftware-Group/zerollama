#!/usr/bin/env bash
# Linux vendor build: replace Darwin ANE draft hook with no-op stubs.
#
# WHY: ane_draft_hook.cpp (patches 0023/0054/0066) references Metal/IOSurface
# symbols only declared under __APPLE__. CUDA/Vulkan llama-server must not
# require libane_bridge. Darwin keeps the full lab hook via patch series.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "${ROOT}" || ! -f "${ROOT}/common/ane_draft_hook.cpp" ]]; then
  echo "patch_vendor_linux_ane_hook: bad root" >&2
  exit 1
fi

FILE="${ROOT}/common/ane_draft_hook.cpp"
MARK="ZEROLLAMA_LINUX_ANE_DRAFT_STUB"

if grep -q "${MARK}" "${FILE}"; then
  exit 0
fi

# Preserve Darwin source for operators who rebuild on Mac from the same tree.
if [[ ! -f "${FILE}.darwin" ]]; then
  cp -f "${FILE}" "${FILE}.darwin"
fi

cat > "${FILE}" <<'EOF'
// ZEROLLAMA_LINUX_ANE_DRAFT_STUB
// Linux no-op stub for ANE dflash draft hook (Darwin lab only).
// Full implementation: common/ane_draft_hook.cpp.darwin
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
EOF

echo "patched ${FILE} (Linux ANE stub; Darwin saved as ${FILE}.darwin)"
