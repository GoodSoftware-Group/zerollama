#pragma once

#include "speculative.h"

struct llama_context;

// Lab-only hooks for ZEROLLAMA_ANE_DRAFT hybrid draft-on-ANE research (see docs/ane-draft-inprocess.md).
//
// Why default off: production dflash must not pay ANE map+eval overhead or depend on libane_bridge
// until B7 routes draft tokens from ANE. IOSurface handoff requires same PID as ane_draft_session.
//
// Why pre-norm handoff: draft hidden state before norm matches sidecar proxy extract geometry;
// post-norm/logits path is the wrong tensor for conv MIL input.

bool common_ane_draft_enabled();

void common_ane_draft_log_init(common_speculative_type type, int draft_n_embd);

// After each draft-model decode: pack hidden state into ANE IOSurface via ggml map + eval.
// B5: called on every draft decode step (not once). B7: optional drive samples token from ANE output.
void common_ane_draft_handoff_after_decode(struct llama_context * ctx_dft, int32_t i_batch);

// B7 lab: reset per draft() call handoff counter and cached ANE output state.
void common_ane_draft_reset_handoff(void);

enum common_ane_draft_drive_mode {
    COMMON_ANE_DRAFT_DRIVE_OFF    = 0,
    COMMON_ANE_DRAFT_DRIVE_SHADOW = 1, // log ANE vs Metal token; still sample Metal
    COMMON_ANE_DRAFT_DRIVE_FORCE  = 2, // use ANE tied-embed argmax token when ready
};

enum common_ane_draft_drive_mode common_ane_draft_get_drive_mode(void);

// Sample draft token from last ANE eval + host tied-embed argmax. Returns false when drive off or head missing.
bool common_ane_draft_try_drive_token(struct llama_context * ctx_dft, int32_t i_batch, llama_token * out_id, float * out_p);
