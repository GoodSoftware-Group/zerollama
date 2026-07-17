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

// B8: target context for dflash_fc target_hidden handoff (set from draft_simple ctor).
void common_ane_draft_bind_target_ctx(struct llama_context * ctx_tgt);

// B8: after target decode, stage ctx_tgt pre-norm → draft cross.v_embd before draft llama_decode.
// No-op unless spec-type dflash, dflash-draft arch, or ZEROLLAMA_ANE_DRAFT=1.
void common_ane_draft_sync_target_cross(
        struct llama_context * ctx_dft,
        struct llama_context * ctx_tgt,
        const llama_batch & batch);

void common_ane_draft_log_init(common_speculative_type type, int draft_n_embd);

// After each draft-model decode: pack hidden state into ANE IOSurface via ggml map + eval.
// B5: called on every draft decode step (not once). B7: optional drive samples token from ANE output.
void common_ane_draft_handoff_after_decode(struct llama_context * ctx_dft, int32_t i_batch);

// P17: token decoded immediately before handoff — dflash blk.0 attn residual is tok_embd[token].
void common_ane_draft_note_handoff_token(llama_token tok);

// P25: batch.pos for the handoff decode — host cross-attn RoPE must match Metal inp_pos.
void common_ane_draft_note_handoff_pos(llama_pos pos);

// Last token noted before handoff (B7 shadow diagnostics).
llama_token common_ane_draft_last_handoff_token(void);

// B7 lab: reset per draft() call handoff counter and cached ANE output state.
void common_ane_draft_reset_handoff(void);

enum common_ane_draft_drive_mode {
    COMMON_ANE_DRAFT_DRIVE_OFF    = 0,
    COMMON_ANE_DRAFT_DRIVE_SHADOW = 1, // log ANE vs Metal token; still sample Metal
    COMMON_ANE_DRAFT_DRIVE_FORCE  = 2, // use ANE tied-embed argmax token when ready
};

enum common_ane_draft_drive_mode common_ane_draft_get_drive_mode(void);

// Sample draft token from last ANE eval + host tied-embed argmax. Returns false when drive off or head missing.
// out_hidden_cos optional: cosine(ANE drive hidden, Metal draft hidden) on first oc dims when shadow mode.
bool common_ane_draft_try_drive_token(struct llama_context * ctx_dft, int32_t i_batch, llama_token * out_id, float * out_p, float * out_hidden_cos);

// B7 shadow: Metal reference token via tied-embed argmax on draft ctx post-forward hidden (matches ANE drive path).
bool common_ane_draft_metal_ref_token(struct llama_context * ctx_dft, int32_t i_batch, llama_token * out_id);

// Batch row used for the last ANE handoff/eval (-1 if none).
int32_t common_ane_draft_handoff_i_batch(void);

// P66: rebind drive hidden to a noise-block row without ANE rematmul / Metal synchronize.
// Host-only: copies already-exported embeddings (+ output_norm) into the session outBuf so B7
// can score every draft row after a stable handoff@0. Full ANE chain parity remains in golden legs.
bool common_ane_draft_rebind_drive_slot(struct llama_context * ctx_dft, int32_t i_batch);

// P67: free the cached IOSurface Metal buffer (and its Metal residency set) created by the
// handoff path. Call from every common_speculative_impl destructor that may have enabled the
// ANE hook — otherwise ggml_metal_rsets_free() asserts `[rsets->data count] == 0` at process
// exit ("cleaning up before exit"). No-op if the hook never mapped a buffer.
void common_ane_draft_shutdown_iosurface_cache(void);

// P24: host fp32 Q/K/V noise matmuls (P21 path) — skip ANE fp16 attn_q/k/v when true.
bool ane_dflash_qkv_host_fp32_enabled(void);

// Fill session lastDflashQ/K/V from host fp32 (P18/P20 per P21). Returns false when disabled or weights missing.
bool ane_dflash_fill_session_qkv_host_fp32(int oc_q, int oc_kv);
