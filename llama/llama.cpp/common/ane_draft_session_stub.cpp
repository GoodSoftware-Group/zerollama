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

size_t ane_draft_session_read_output(float * /*dst*/, size_t /*dst_floats*/) {
    return 0;
}

int ane_draft_session_step_count(void) {
    return 0;
}

bool ane_draft_session_using_conv2(void) {
    return false;
}

void ane_draft_session_shutdown(void) {
}
