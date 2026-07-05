#include "ane_draft_session.h"

bool ane_draft_session_supported(void) {
    return false;
}

bool ane_draft_session_init(int /*channels*/, int /*spatial*/, const char * /*weight_path*/, const char * /*gamma_path*/) {
    return false;
}

bool ane_draft_session_ready(void) {
    return false;
}

uint32_t ane_draft_session_surface_id(void) {
    return 0;
}

size_t ane_draft_session_surface_bytes(void) {
    return 0;
}

int ane_draft_session_channels(void) {
    return 0;
}

int ane_draft_session_spatial(void) {
    return 0;
}

bool ane_draft_session_step_once(float /*fill_val*/) {
    return false;
}

bool ane_draft_session_eval(void) {
    return false;
}

void ane_draft_session_eval_sync(void) {
}

bool ane_draft_session_eval_async(ane_draft_eval_async_fn /*on_done*/) {
    return false;
}

bool ane_draft_session_eval_async_enabled(void) {
    return false;
}

size_t ane_draft_session_read_output(float * /*dst*/, size_t /*dst_floats*/) {
    return 0;
}

int ane_draft_session_step_count(void) {
    return 0;
}

int ane_draft_session_conv_depth_cap(void) {
    return 0;
}

int ane_draft_session_active_conv_count(void) {
    return 0;
}

int ane_draft_session_output_channels(void) {
    return 0;
}

bool ane_draft_session_matmul_active(void) {
    return false;
}

bool ane_draft_session_matmul_dynamic(void) {
    return false;
}

bool ane_draft_session_pack_matmul_activations(float * /*dst*/, const float * /*hidden*/, int /*hidden_len*/) {
    return false;
}

int ane_draft_session_matmul_chain_depth(void) {
    return 0;
}

int ane_draft_session_matmul_ffn_embd(void) {
    return 0;
}

int ane_draft_session_matmul9_oc(void) {
    return 0;
}

size_t ane_draft_session_read_ffn_down(float * /*dst*/, size_t /*dst_floats*/) {
    return 0;
}

size_t ane_draft_session_read_qkv_prefix(float * /*dst*/, size_t /*dst_floats*/) {
    return 0;
}

bool ane_draft_session_using_conv2(void) {
    return false;
}

bool ane_draft_session_dflash_fc_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain11_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain12_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain13_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain14_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain15_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain16_active(void) {
    return false;
}

bool ane_draft_session_dflash_chain17_active(void) {
    return false;
}

bool ane_draft_session_eval_dflash_ffn_gate(void) {
    return false;
}

bool ane_draft_session_eval_dflash_ffn_up_swiglu_down(void) {
    return false;
}

bool ane_draft_session_eval_dflash_output_norm(void) {
    return false;
}

bool ane_draft_session_set_dflash_fc_host(const float *, int) {
    return false;
}

void ane_draft_session_clear_dflash_fc_host(void) {
}

bool ane_draft_session_eval_dflash_attn_post_norm(void) {
    return false;
}

bool ane_draft_session_eval_dflash_attn_wo(void) {
    return false;
}

bool ane_draft_session_write_dflash_attn_out(const float * src, int n) {
    (void) src;
    (void) n;
    return false;
}

bool ane_draft_session_snapshot_output_row(float * row, int n) {
    (void) row;
    (void) n;
    return false;
}

bool ane_draft_session_add_output_row(const float * delta, int n) {
    (void) delta;
    (void) n;
    return false;
}

bool ane_draft_session_read_dflash_qkv(float * q, float * k, float * v, int n) {
    (void) q;
    (void) k;
    (void) v;
    (void) n;
    return false;
}

void ane_draft_session_shutdown(void) {
}
