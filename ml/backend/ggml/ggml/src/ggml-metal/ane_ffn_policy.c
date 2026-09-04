// Fail-closed ANE FFN intercept policy (lab). See ane_ffn_policy.h.
#include "ane_ffn_policy.h"

#include <ctype.h>
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static _Atomic uint64_t g_shadow_seen = 0;
static _Atomic uint64_t g_shadow_match = 0;
static _Atomic uint64_t g_shadow_logged = 0;
static _Atomic uint64_t g_shadow_swiglu_fuse = 0;
static _Atomic uint64_t g_shadow_swiglu_logged = 0;
static _Atomic uint64_t g_force_deferred = 0;
static _Atomic uint64_t g_force_replaced = 0;
static _Atomic uint64_t g_force_bail = 0;

static ane_ffn_host_replace_fn g_host_replace = NULL;
static ane_ffn_swiglu_replace_fn g_swiglu_replace = NULL;
typedef bool (*ane_ffn_swiglu_fp16_replace_fn)(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *X_ic_seq_f16,
    void *Y_ic_seq_f16);
typedef bool (*ane_ffn_swiglu_int8_fp16_replace_fn)(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    void *Y_ic_seq_f16);
typedef float (*ane_ffn_swiglu_x_scale_fn)(void);
typedef uint32_t (*ane_ffn_surface_id_fn)(void);
typedef bool (*ane_ffn_eval_only_fn)(void);
typedef bool (*ane_ffn_metal_pack_f16_fn)(
    uint32_t in_sid, const void *src, int ic, int seq, float scale);
typedef bool (*ane_ffn_metal_pack_f32_fn)(
    uint32_t in_sid, const float *src, int ic, int seq, float scale);
typedef bool (*ane_ffn_metal_unpack_fn)(
    uint32_t out_sid, int oc, int seq, void *dst);
typedef int (*ane_ffn_session_seq_fn)(void);
typedef void (*ane_ffn_set_weight_ids_fn)(const void *, const void *, const void *);
typedef bool (*ane_ffn_activate_fn)(
    int ic, int hidden, int seq,
    const float *Wg, const float *Wu, const float *Wd);
typedef bool (*ane_ffn_pack_eval_async_fn)(
    const void *src_ggml_acts, int ic, int seq, int acts_is_f16);
typedef bool (*ane_ffn_pack_eval_async_wait_fn)(void);
static ane_ffn_swiglu_fp16_replace_fn g_swiglu_fp16_replace = NULL;
static ane_ffn_swiglu_int8_fp16_replace_fn g_swiglu_int8_fp16_replace = NULL;
static ane_ffn_swiglu_x_scale_fn g_swiglu_x_scale = NULL;
static ane_ffn_surface_id_fn g_in_sid = NULL;
static ane_ffn_surface_id_fn g_out_sid = NULL;
static ane_ffn_eval_only_fn g_eval_only = NULL;
static ane_ffn_metal_pack_f16_fn g_metal_pack_f16 = NULL;
static ane_ffn_metal_pack_f32_fn g_metal_pack_f32 = NULL;
static ane_ffn_metal_unpack_fn g_metal_unpack = NULL;
static ane_ffn_session_seq_fn g_session_seq = NULL;
static ane_ffn_set_weight_ids_fn g_set_weight_ids = NULL;
static ane_ffn_activate_fn g_activate = NULL;
static ane_ffn_pack_eval_async_fn g_pack_eval_async = NULL;
static ane_ffn_pack_eval_async_wait_fn g_pack_eval_async_wait = NULL;
static void * g_replace_dylib = NULL;

static bool env_truthy(const char *v) {
    if (!v || !v[0]) {
        return false;
    }
    if (strcmp(v, "1") == 0) {
        return true;
    }
    char buf[8] = {0};
    for (int i = 0; i < 7 && v[i]; i++) {
        buf[i] = (char)tolower((unsigned char)v[i]);
    }
    return strcmp(buf, "true") == 0 || strcmp(buf, "yes") == 0 || strcmp(buf, "on") == 0;
}

static int env_int(const char *name, int def) {
    const char *v = getenv(name);
    if (!v || !v[0]) {
        return def;
    }
    return atoi(v);
}

// Mirror ANE FFN telemetry to a file so lab runs without `tee` still leave evidence.
// Default path: /tmp/ane-ffn-force.log  Override: ZEROLLAMA_ANE_FFN_LOG=/path
static void ane_ffn_telem_v(const char *fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    vfprintf(stderr, fmt, ap);
    const char *path = getenv("ZEROLLAMA_ANE_FFN_LOG");
    if (!path || !path[0]) {
        if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_TELEMETRY"))) {
            va_end(ap2);
            return;
        }
        path = "/tmp/ane-ffn-force.log";
    }
    FILE *f = fopen(path, "a");
    if (f) {
        vfprintf(f, fmt, ap2);
        fclose(f);
    }
    va_end(ap2);
}

static void ane_ffn_telem(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ane_ffn_telem_v(fmt, ap);
    va_end(ap);
}

static void add_name_pat(ane_ffn_policy_t *out, const char *pat) {
    if (!out || !pat || !pat[0] || out->n_name_pats >= ANE_FFN_NAME_PAT_MAX) {
        return;
    }
    snprintf(out->name_pats[out->n_name_pats], ANE_FFN_NAME_PAT_LEN, "%s", pat);
    out->n_name_pats++;
}

static void parse_name_filter(ane_ffn_policy_t *out, const char *raw) {
    out->n_name_pats = 0;
    if (!raw || !raw[0]) {
        return;
    }
    // Presets
    if (strcmp(raw, "shexp") == 0) {
        add_name_pat(out, "ffn_gate_shexp");
        add_name_pat(out, "ffn_up_shexp");
        add_name_pat(out, "ffn_down_shexp");
        return;
    }
    if (strcmp(raw, "ffn") == 0 || strcmp(raw, "dense") == 0) {
        // Prefer ".weight" suffix so we don't match *_shexp / *_exps.
        add_name_pat(out, "ffn_gate.weight");
        add_name_pat(out, "ffn_up.weight");
        add_name_pat(out, "ffn_down.weight");
        return;
    }
    if (strcmp(raw, "any") == 0 || strcmp(raw, "*") == 0) {
        return;
    }

    // Comma-separated substrings
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", raw);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok && isspace((unsigned char)*tok)) {
            tok++;
        }
        char *end = tok + strlen(tok);
        while (end > tok && isspace((unsigned char)end[-1])) {
            *--end = '\0';
        }
        if (tok[0]) {
            add_name_pat(out, tok);
        }
    }
}

bool ane_ffn_policy_is_production_port(int port) {
    return port == 11434 || port == 8081;
}

int ane_ffn_policy_parse_host_port(const char *host_or_url) {
    if (!host_or_url || !host_or_url[0]) {
        return 0;
    }
    const char *p = strrchr(host_or_url, ':');
    if (!p || !p[1]) {
        return 0;
    }
    return atoi(p + 1);
}

bool ane_ffn_policy_load(ane_ffn_policy_t *out) {
    if (!out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->refuse_production_ports = true;
    out->enabled = env_truthy(getenv("ZEROLLAMA_ANE_FFN"));
    out->telemetry = env_truthy(getenv("ZEROLLAMA_ANE_FFN_TELEMETRY"));
    out->ic = env_int("ZEROLLAMA_ANE_FFN_IC", 0);
    out->oc = env_int("ZEROLLAMA_ANE_FFN_OC", 0);
    out->seq_max = env_int("ZEROLLAMA_ANE_FFN_SEQ_MAX", 0);
    out->lab_port = env_int("ZEROLLAMA_ANE_FFN_LAB_PORT", 0);
    if (out->lab_port == 0) {
        out->lab_port = env_int("ZEROLLAMA_ANE_FFN_PORT", 0);
    }

    const char *mode = getenv("ZEROLLAMA_ANE_FFN_MODE");
    out->mode = ANE_FFN_MODE_SHADOW;
    if (mode && mode[0]) {
        if (strcmp(mode, "off") == 0) {
            out->mode = ANE_FFN_MODE_OFF;
        } else if (strcmp(mode, "force") == 0) {
            out->mode = ANE_FFN_MODE_FORCE;
        } else {
            out->mode = ANE_FFN_MODE_SHADOW;
        }
    }
    parse_name_filter(out, getenv("ZEROLLAMA_ANE_FFN_NAME"));
    if (!out->enabled) {
        out->mode = ANE_FFN_MODE_OFF;
    }
    return out->enabled;
}

bool ane_ffn_policy_enabled(void) {
    ane_ffn_policy_t p;
    return ane_ffn_policy_load(&p);
}

ane_ffn_mode_t ane_ffn_policy_mode(void) {
    ane_ffn_policy_t p;
    ane_ffn_policy_load(&p);
    return p.mode;
}

bool ane_ffn_policy_telemetry(void) {
    ane_ffn_policy_t p;
    ane_ffn_policy_load(&p);
    return p.telemetry;
}

bool ane_ffn_policy_name_matches(const ane_ffn_policy_t *pol, const char *weight_name) {
    if (!pol || pol->n_name_pats <= 0) {
        return true;
    }
    if (!weight_name || !weight_name[0]) {
        return false;
    }
    for (int i = 0; i < pol->n_name_pats; i++) {
        if (strstr(weight_name, pol->name_pats[i]) != NULL) {
            return true;
        }
    }
    return false;
}

ane_ffn_verdict_t ane_ffn_policy_decide(
    const ane_ffn_policy_t *pol,
    ane_ffn_op_t op,
    int ic, int oc, int seq,
    int serve_port,
    const char *weight_name) {
    ane_ffn_verdict_t v = {0};
    v.reason = "ok";
    v.name_match = true;
    if (!pol || !pol->enabled || pol->mode == ANE_FFN_MODE_OFF) {
        v.reason = "disabled";
        return v;
    }
    if (op == ANE_FFN_OP_MUL_MAT_ID) {
        v.reason = "mul_mat_id_not_supported";
        return v;
    }
    if (op != ANE_FFN_OP_MUL_MAT) {
        v.reason = "unsupported_op";
        return v;
    }
    if (ic <= 0 || oc <= 0 || seq <= 0) {
        v.reason = "invalid_geometry";
        return v;
    }

    v.port_ok = true;
    if (pol->refuse_production_ports && ane_ffn_policy_is_production_port(serve_port)) {
        v.port_ok = false;
        v.reason = "production_port_refused";
        return v;
    }
    if (pol->lab_port > 0 && serve_port > 0 && serve_port != pol->lab_port) {
        v.port_ok = false;
        v.reason = "lab_port_mismatch";
        return v;
    }
    if (pol->lab_port > 0 && serve_port == 0) {
        v.port_ok = false;
        v.reason = "serve_port_unknown";
        return v;
    }
    if (pol->lab_port == 0 && serve_port == 0) {
        v.port_ok = true;
    }

    v.geometry_match = true;
    if (pol->ic > 0 && ic != pol->ic) {
        v.geometry_match = false;
        v.reason = "ic_mismatch";
        return v;
    }
    if (pol->oc > 0 && oc != pol->oc) {
        v.geometry_match = false;
        v.reason = "oc_mismatch";
        return v;
    }
    if (pol->seq_max > 0 && seq > pol->seq_max) {
        v.geometry_match = false;
        v.reason = "seq_above_max";
        return v;
    }

    if (!ane_ffn_policy_name_matches(pol, weight_name)) {
        v.name_match = false;
        v.reason = (!weight_name || !weight_name[0]) ? "name_missing" : "name_mismatch";
        return v;
    }

    v.allow = true;
    if (pol->mode == ANE_FFN_MODE_FORCE) {
        v.reason = "force_match";
    } else {
        v.reason = "shadow_match";
    }
    return v;
}

uint64_t ane_ffn_shadow_match_count(void) {
    return atomic_load(&g_shadow_match);
}

uint64_t ane_ffn_shadow_seen_count(void) {
    return atomic_load(&g_shadow_seen);
}

uint64_t ane_ffn_shadow_swiglu_fuse_count(void) {
    return atomic_load(&g_shadow_swiglu_fuse);
}

uint64_t ane_ffn_force_deferred_count(void) {
    return atomic_load(&g_force_deferred);
}

uint64_t ane_ffn_force_replaced_count(void) {
    return atomic_load(&g_force_replaced);
}

void ane_ffn_force_set_host_replace(ane_ffn_host_replace_fn fn) {
    g_host_replace = fn;
}

ane_ffn_host_replace_fn ane_ffn_force_get_host_replace(void) {
    return g_host_replace;
}

void ane_ffn_force_set_swiglu_replace(ane_ffn_swiglu_replace_fn fn) {
    g_swiglu_replace = fn;
}

void ane_ffn_force_autoload_host_replace(void) {
    static int int8_syms_tried = 0;
    if (g_host_replace && g_swiglu_replace && g_swiglu_fp16_replace && int8_syms_tried) {
        return;
    }
    const char *path = getenv("ZEROLLAMA_ANE_FFN_REPLACE_DYLIB");
    if (!path || !path[0]) {
        return;
    }
    if (!g_replace_dylib) {
        g_replace_dylib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!g_replace_dylib) {
            fprintf(stderr, "ane_ffn_force: dlopen(%s) failed: %s\n", path, dlerror());
            return;
        }
    }
    if (!g_host_replace) {
        g_host_replace = (ane_ffn_host_replace_fn)dlsym(
            g_replace_dylib, "ane_ffn_force_replace_mul_mat");
        if (!g_host_replace) {
            fprintf(stderr, "ane_ffn_force: dlsym(mul_mat) failed: %s\n", dlerror());
        }
    }
    if (!g_swiglu_replace) {
        g_swiglu_replace = (ane_ffn_swiglu_replace_fn)dlsym(
            g_replace_dylib, "ane_ffn_force_replace_swiglu");
    }
    if (!g_swiglu_fp16_replace) {
        g_swiglu_fp16_replace = (ane_ffn_swiglu_fp16_replace_fn)dlsym(
            g_replace_dylib, "ane_ffn_force_replace_swiglu_fp16");
    }
    if (!int8_syms_tried) {
        int8_syms_tried = 1;
        if (!g_swiglu_int8_fp16_replace) {
            g_swiglu_int8_fp16_replace = (ane_ffn_swiglu_int8_fp16_replace_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_replace_swiglu_int8_fp16");
        }
        if (!g_swiglu_x_scale) {
            g_swiglu_x_scale = (ane_ffn_swiglu_x_scale_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_x_scale");
        }
        if (!g_in_sid) {
            g_in_sid = (ane_ffn_surface_id_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_input_surface_id");
        }
        if (!g_out_sid) {
            g_out_sid = (ane_ffn_surface_id_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_output_surface_id");
        }
        if (!g_eval_only) {
            g_eval_only = (ane_ffn_eval_only_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_eval_only");
        }
        if (!g_metal_pack_f16) {
            g_metal_pack_f16 = (ane_ffn_metal_pack_f16_fn)dlsym(
                g_replace_dylib, "ane_ffn_layout_metal_pack_in_i8_f16");
        }
        if (!g_metal_pack_f32) {
            g_metal_pack_f32 = (ane_ffn_metal_pack_f32_fn)dlsym(
                g_replace_dylib, "ane_ffn_layout_metal_pack_in_i8_f32");
        }
        if (!g_metal_unpack) {
            g_metal_unpack = (ane_ffn_metal_unpack_fn)dlsym(
                g_replace_dylib, "ane_ffn_layout_metal_unpack_out_f16");
        }
        if (!g_session_seq) {
            g_session_seq = (ane_ffn_session_seq_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_session_seq");
        }
        if (!g_set_weight_ids) {
            g_set_weight_ids = (ane_ffn_set_weight_ids_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_set_weight_ids");
        }
        if (!g_activate) {
            g_activate = (ane_ffn_activate_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_activate");
        }
        if (!g_pack_eval_async) {
            g_pack_eval_async = (ane_ffn_pack_eval_async_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_pack_eval_async");
        }
        if (!g_pack_eval_async_wait) {
            g_pack_eval_async_wait = (ane_ffn_pack_eval_async_wait_fn)dlsym(
                g_replace_dylib, "ane_ffn_force_swiglu_pack_eval_async_wait");
        }
    }
}

static bool force_policy_allows(
    ane_ffn_policy_t *pol_out,
    int ic, int oc, int seq,
    const char *weight_name,
    bool check_name,
    int *port_out) {
    ane_ffn_policy_t pol;
    if (!ane_ffn_policy_load(&pol) || pol.mode != ANE_FFN_MODE_FORCE) {
        return false;
    }
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_FORCE_ENABLE"))) {
        return false;
    }
    int serve_port = ane_ffn_policy_parse_host_port(getenv("OLLAMA_HOST"));
    if (serve_port == 0) {
        serve_port = pol.lab_port;
    }
    // Host-buffer smoke path may skip name (no ggml tensor). Metal always checks.
    const char *name_arg = weight_name;
    int saved_pats = pol.n_name_pats;
    if (!check_name) {
        pol.n_name_pats = 0;
    }
    ane_ffn_verdict_t v = ane_ffn_policy_decide(
        &pol, ANE_FFN_OP_MUL_MAT, ic, oc, seq, serve_port, name_arg);
    pol.n_name_pats = saved_pats;
    if (!v.allow) {
        return false;
    }
    if (pol_out) {
        *pol_out = pol;
    }
    if (port_out) {
        *port_out = serve_port;
    }
    return true;
}

bool ane_ffn_force_want_try(int ic, int oc, int seq, const char *weight_name) {
    return force_policy_allows(NULL, ic, oc, seq, weight_name, true, NULL);
}

bool ane_ffn_force_try_mul_mat_host(
    int ic, int oc, int seq,
    const float *W_oc_ic,
    const float *X_ic_seq,
    float *Y_oc_seq) {
    ane_ffn_policy_t pol;
    int serve_port = 0;
    if (!force_policy_allows(&pol, ic, oc, seq, NULL, false, &serve_port)) {
        return false;
    }
    if (!W_oc_ic || !X_ic_seq || !Y_oc_seq) {
        return false;
    }
    ane_ffn_force_autoload_host_replace();
    ane_ffn_host_replace_fn fn = g_host_replace;
    if (!fn) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: deferred#%llu ic=%d oc=%d seq=%d port=%d "
                    "(no host replace registered — Metal still runs)\n",
                    (unsigned long long)n, ic, oc, seq, serve_port);
        }
        return false;
    }
    if (!fn(ic, oc, seq, W_oc_ic, X_ic_seq, Y_oc_seq)) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: replace_failed#%llu ic=%d oc=%d seq=%d port=%d "
                    "(Metal still runs)\n",
                    (unsigned long long)n, ic, oc, seq, serve_port);
        }
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_replaced, 1) + 1;
    if (pol.telemetry || n <= 8 || (n % 1024ull) == 0) {
        fprintf(stderr,
                "ane_ffn_force: replaced#%llu ic=%d oc=%d seq=%d port=%d\n",
                (unsigned long long)n, ic, oc, seq, serve_port);
    }
    return true;
}

bool ane_ffn_force_try_mul_mat(int ic, int oc, int seq, const char *weight_name) {
    ane_ffn_policy_t pol;
    int serve_port = 0;
    if (!force_policy_allows(&pol, ic, oc, seq, weight_name, true, &serve_port)) {
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
    if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
        fprintf(stderr,
                "ane_ffn_force: deferred#%llu ic=%d oc=%d seq=%d port=%d name=%s "
                "(dims-only hook — use host replace for ANE skip)\n",
                (unsigned long long)n, ic, oc, seq, serve_port,
                weight_name && weight_name[0] ? weight_name : "-");
    }
    return false;
}

void ane_ffn_shadow_note_mul_mat(int ic, int oc, int seq, const char *weight_name) {
    ane_ffn_policy_t pol;
    if (!ane_ffn_policy_load(&pol)) {
        return;
    }

    atomic_fetch_add(&g_shadow_seen, 1);

    int serve_port = ane_ffn_policy_parse_host_port(getenv("OLLAMA_HOST"));
    if (serve_port == 0) {
        serve_port = pol.lab_port;
    }

    ane_ffn_verdict_t v = ane_ffn_policy_decide(
        &pol, ANE_FFN_OP_MUL_MAT, ic, oc, seq, serve_port, weight_name);
    if (!v.allow) {
        return;
    }

    uint64_t n = atomic_fetch_add(&g_shadow_match, 1) + 1;

    uint64_t logged = atomic_load(&g_shadow_logged);
    bool should_log = pol.telemetry || n <= 8 || (n % 1024ull) == 0;
    if (should_log && logged < n) {
        atomic_store(&g_shadow_logged, n);
        fprintf(stderr,
                "ane_ffn_shadow: match#%llu %s ic=%d oc=%d seq=%d port=%d mode=%s name=%s "
                "(Metal still runs)\n",
                (unsigned long long)n,
                v.reason ? v.reason : "?",
                ic, oc, seq, serve_port,
                pol.mode == ANE_FFN_MODE_FORCE ? "force" : "shadow",
                weight_name && weight_name[0] ? weight_name : "-");
    }
}

void ane_ffn_shadow_note_swiglu_fuse(int ic, int hidden, int seq, const char *weight_name) {
    ane_ffn_policy_t pol;
    if (!ane_ffn_policy_load(&pol)) {
        return;
    }
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_SWIGLU"))) {
        return;
    }

    int serve_port = ane_ffn_policy_parse_host_port(getenv("OLLAMA_HOST"));
    if (serve_port == 0) {
        serve_port = pol.lab_port;
    }
    // Geometry check uses hidden as OC (expert-up width).
    ane_ffn_verdict_t v = ane_ffn_policy_decide(
        &pol, ANE_FFN_OP_MUL_MAT, ic, hidden, seq, serve_port, weight_name);
    if (!v.allow) {
        return;
    }

    uint64_t n = atomic_fetch_add(&g_shadow_swiglu_fuse, 1) + 1;
    uint64_t logged = atomic_load(&g_shadow_swiglu_logged);
    bool should_log = pol.telemetry || n <= 8 || (n % 1024ull) == 0;
    if (should_log && logged < n) {
        atomic_store(&g_shadow_swiglu_logged, n);
        ane_ffn_telem(
                "ane_ffn_shadow: swiglu_fuse#%llu ic=%d hidden=%d seq=%d port=%d mode=%s "
                "name=%s (Metal still runs unless force replace)\n",
                (unsigned long long)n, ic, hidden, seq, serve_port,
                pol.mode == ANE_FFN_MODE_FORCE ? "force" : "shadow",
                weight_name && weight_name[0] ? weight_name : "-");
    }
}

void ane_ffn_force_note_bail(
    const char *reason, int ic, int hidden, int seq, const char *weight_name) {
    ane_ffn_policy_t pol;
    if (!ane_ffn_policy_load(&pol) || pol.mode != ANE_FFN_MODE_FORCE) {
        return;
    }
    uint64_t n = atomic_fetch_add(&g_force_bail, 1) + 1;
    if (!(pol.telemetry || n <= 32 || (n % 256ull) == 0)) {
        return;
    }
    ane_ffn_telem(
            "ane_ffn_force: bail#%llu reason=%s ic=%d hidden=%d seq=%d name=%s\n",
            (unsigned long long)n,
            reason && reason[0] ? reason : "-",
            ic, hidden, seq,
            weight_name && weight_name[0] ? weight_name : "-");
}

static _Atomic uint64_t g_force_wcache_hit = 0;
static _Atomic uint64_t g_force_wcache_miss = 0;

void ane_ffn_force_note_wcache(
    int hit, int slots, int ic, int hidden, const char *weight_name) {
    ane_ffn_policy_t pol;
    if (!ane_ffn_policy_load(&pol) || pol.mode != ANE_FFN_MODE_FORCE) {
        return;
    }
    uint64_t n = hit
        ? atomic_fetch_add(&g_force_wcache_hit, 1) + 1
        : atomic_fetch_add(&g_force_wcache_miss, 1) + 1;
    if (!(pol.telemetry || n <= 16 || (n % 256ull) == 0)) {
        return;
    }
    ane_ffn_telem(
            "ane_ffn_force: wcache_%s#%llu slots=%d ic=%d hidden=%d name=%s\n",
            hit ? "hit" : "miss",
            (unsigned long long)n, slots, ic, hidden,
            weight_name && weight_name[0] ? weight_name : "-");
}

// ---- hot-path profile (ZEROLLAMA_ANE_FFN_PROFILE=1) ----
enum {
    ANE_PROF_SYNC = 0,
    ANE_PROF_PACK,
    ANE_PROF_WRITE,
    ANE_PROF_EVAL,
    ANE_PROF_READ,
    ANE_PROF_UNPACK,
    ANE_PROF_DYLIB, // write+eval+read if split unavailable
    ANE_PROF_N
};

static const char * const g_prof_names[ANE_PROF_N] = {
    "sync", "pack", "write", "eval", "read", "unpack", "dylib",
};

static double g_prof_ms[ANE_PROF_N];
static uint64_t g_prof_n[ANE_PROF_N];
static _Atomic uint64_t g_prof_replaces = 0;
static int g_prof_on = -1;

static int profile_enabled(void) {
    if (g_prof_on < 0) {
        g_prof_on = env_truthy(getenv("ZEROLLAMA_ANE_FFN_PROFILE")) ? 1 : 0;
    }
    return g_prof_on;
}

double ane_ffn_profile_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

void ane_ffn_profile_add_ms(const char *phase, double ms) {
    if (!profile_enabled() || !phase || !(ms >= 0) || !isfinite(ms)) {
        return;
    }
    int idx = -1;
    for (int i = 0; i < ANE_PROF_N; i++) {
        if (strcmp(phase, g_prof_names[i]) == 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        return;
    }
    g_prof_ms[idx] += ms;
    g_prof_n[idx] += 1;
}

void ane_ffn_profile_tick_replace(void) {
    if (!profile_enabled()) {
        return;
    }
    uint64_t n = atomic_fetch_add(&g_prof_replaces, 1) + 1;
    // Dump after warmup fills (~24) and every 96 replaces (~4 decode tokens).
    if (n != 48 && n != 96 && (n % 384ull) != 0) {
        return;
    }
    double sum = 0;
    for (int i = 0; i < ANE_PROF_N; i++) {
        sum += g_prof_ms[i];
    }
    ane_ffn_telem(
            "ane_ffn_profile: after#%llu replaces — "
            "sync=%.1fms/%llu pack=%.1fms/%llu write=%.1fms/%llu "
            "eval=%.1fms/%llu read=%.1fms/%llu unpack=%.1fms/%llu "
            "dylib=%.1fms/%llu sum_phases=%.1fms\n",
            (unsigned long long)n,
            g_prof_ms[ANE_PROF_SYNC], (unsigned long long)g_prof_n[ANE_PROF_SYNC],
            g_prof_ms[ANE_PROF_PACK], (unsigned long long)g_prof_n[ANE_PROF_PACK],
            g_prof_ms[ANE_PROF_WRITE], (unsigned long long)g_prof_n[ANE_PROF_WRITE],
            g_prof_ms[ANE_PROF_EVAL], (unsigned long long)g_prof_n[ANE_PROF_EVAL],
            g_prof_ms[ANE_PROF_READ], (unsigned long long)g_prof_n[ANE_PROF_READ],
            g_prof_ms[ANE_PROF_UNPACK], (unsigned long long)g_prof_n[ANE_PROF_UNPACK],
            g_prof_ms[ANE_PROF_DYLIB], (unsigned long long)g_prof_n[ANE_PROF_DYLIB],
            sum);
}

bool ane_ffn_force_try_swiglu_host(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const float *X_ic_seq,
    float *Y_ic_seq) {
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_SWIGLU"))) {
        return false;
    }
    ane_ffn_policy_t pol;
    int serve_port = 0;
    // Skip name filter here — Metal already matched ffn_up; smoke leaves NAME unset.
    if (!force_policy_allows(&pol, ic, hidden, seq, NULL, false, &serve_port)) {
        return false;
    }
    if (!Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !X_ic_seq || !Y_ic_seq) {
        return false;
    }
    ane_ffn_force_autoload_host_replace();
    if (!g_swiglu_replace) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: swiglu_deferred#%llu ic=%d hidden=%d seq=%d "
                    "(no swiglu replace in dylib)\n",
                    (unsigned long long)n, ic, hidden, seq);
        }
        return false;
    }
    if (!g_swiglu_replace(ic, hidden, seq, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden,
                          X_ic_seq, Y_ic_seq)) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: swiglu_replace_failed#%llu ic=%d hidden=%d seq=%d\n",
                    (unsigned long long)n, ic, hidden, seq);
        }
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_replaced, 1) + 1;
    if (pol.telemetry || n <= 8 || (n % 1024ull) == 0) {
        fprintf(stderr,
                "ane_ffn_force: swiglu_replaced#%llu ic=%d hidden=%d seq=%d port=%d\n",
                (unsigned long long)n, ic, hidden, seq, serve_port);
    }
    return true;
}

bool ane_ffn_force_try_swiglu_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *X_ic_seq_f16,
    void *Y_ic_seq_f16) {
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_SWIGLU"))) {
        return false;
    }
    ane_ffn_policy_t pol;
    int serve_port = 0;
    if (!force_policy_allows(&pol, ic, hidden, seq, NULL, false, &serve_port)) {
        return false;
    }
    if (!Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !X_ic_seq_f16 || !Y_ic_seq_f16) {
        return false;
    }
    ane_ffn_force_autoload_host_replace();
    if (!g_swiglu_fp16_replace) {
        return false;
    }
    if (!g_swiglu_fp16_replace(ic, hidden, seq, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden,
                               X_ic_seq_f16, Y_ic_seq_f16)) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            ane_ffn_telem(
                    "ane_ffn_force: swiglu_fp16_replace_failed#%llu ic=%d hidden=%d seq=%d\n",
                    (unsigned long long)n, ic, hidden, seq);
        }
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_replaced, 1) + 1;
    if (pol.telemetry || n <= 8 || (n % 1024ull) == 0) {
        ane_ffn_telem(
                "ane_ffn_force: swiglu_fp16_replaced#%llu ic=%d hidden=%d seq=%d port=%d\n",
                (unsigned long long)n, ic, hidden, seq, serve_port);
    }
    return true;
}

bool ane_ffn_force_try_swiglu_int8_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    void *Y_ic_seq_f16) {
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_SWIGLU"))) {
        return false;
    }
    ane_ffn_policy_t pol;
    int serve_port = 0;
    if (!force_policy_allows(&pol, ic, hidden, seq, NULL, false, &serve_port)) {
        return false;
    }
    if (!Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !X_ic_seq_i8 || !Y_ic_seq_f16) {
        return false;
    }
    ane_ffn_force_autoload_host_replace();
    if (!g_swiglu_int8_fp16_replace) {
        return false;
    }
    if (!g_swiglu_int8_fp16_replace(ic, hidden, seq, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden,
                                    X_ic_seq_i8, Y_ic_seq_f16)) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: swiglu_int8_fp16_replace_failed#%llu ic=%d hidden=%d seq=%d\n",
                    (unsigned long long)n, ic, hidden, seq);
        }
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_replaced, 1) + 1;
    if (pol.telemetry || n <= 8 || (n % 1024ull) == 0) {
        fprintf(stderr,
                "ane_ffn_force: swiglu_int8_fp16_replaced#%llu ic=%d hidden=%d seq=%d port=%d\n",
                (unsigned long long)n, ic, hidden, seq, serve_port);
    }
    return true;
}

float ane_ffn_force_query_x_scale(void) {
    ane_ffn_force_autoload_host_replace();
    if (g_swiglu_x_scale) {
        return g_swiglu_x_scale();
    }
    return 0.f;
}

void ane_ffn_force_swiglu_bind_weight_ids(
    const void *wg_data, const void *wu_data, const void *wd_data) {
    ane_ffn_force_autoload_host_replace();
    if (g_set_weight_ids) {
        g_set_weight_ids(wg_data, wu_data, wd_data);
    }
}

bool ane_ffn_force_swiglu_activate_session(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden) {
    ane_ffn_force_autoload_host_replace();
    if (!g_activate) {
        return false;
    }
    return g_activate(ic, hidden, seq, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden);
}

bool ane_ffn_force_try_swiglu_metal_layout(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *src_ggml_acts,
    int acts_is_f16,
    void *dst_ggml_f16) {
    if (!env_truthy(getenv("ZEROLLAMA_ANE_FFN_SWIGLU"))) {
        return false;
    }
    ane_ffn_policy_t pol;
    int serve_port = 0;
    if (!force_policy_allows(&pol, ic, hidden, seq, NULL, false, &serve_port)) {
        return false;
    }
    if (!Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !src_ggml_acts || !dst_ggml_f16) {
        return false;
    }
    ane_ffn_force_autoload_host_replace();
    if (!g_in_sid || !g_out_sid || !g_eval_only || !g_metal_unpack ||
        !(g_metal_pack_f16 || g_metal_pack_f32) || !g_swiglu_x_scale || !g_session_seq) {
        return false;
    }
    float xscale = g_swiglu_x_scale();
    if (!(xscale > 0)) {
        return false; // session not created yet
    }
    const int sess_seq = g_session_seq();
    if (sess_seq <= 0 || seq > sess_seq) {
        return false;
    }
    uint32_t in_sid = g_in_sid();
    uint32_t out_sid = g_out_sid();
    if (in_sid == 0 || out_sid == 0) {
        return false;
    }

    // Session may be seq-padded (×32) while ggml fuse uses logical seq.
    const void *pack_src = src_ggml_acts;
    void *unpack_dst = dst_ggml_f16;
    void *pad_src = NULL;
    void *pad_dst = NULL;
    if (seq != sess_seq) {
        const size_t n_pad = (size_t)ic * (size_t)sess_seq;
        if (acts_is_f16) {
            pad_src = calloc(n_pad, sizeof(uint16_t));
            if (!pad_src) {
                return false;
            }
            const uint16_t *s = (const uint16_t *)src_ggml_acts;
            uint16_t *d = (uint16_t *)pad_src;
            for (int t = 0; t < seq; t++) {
                memcpy(d + (size_t)t * (size_t)ic, s + (size_t)t * (size_t)ic,
                       (size_t)ic * sizeof(uint16_t));
            }
        } else {
            pad_src = calloc(n_pad, sizeof(float));
            if (!pad_src) {
                return false;
            }
            const float *s = (const float *)src_ggml_acts;
            float *d = (float *)pad_src;
            for (int t = 0; t < seq; t++) {
                memcpy(d + (size_t)t * (size_t)ic, s + (size_t)t * (size_t)ic,
                       (size_t)ic * sizeof(float));
            }
        }
        pad_dst = malloc(n_pad * sizeof(uint16_t));
        if (!pad_dst) {
            free(pad_src);
            return false;
        }
        pack_src = pad_src;
        unpack_dst = pad_dst;
    }

    bool packed = false;
    if (acts_is_f16 && g_metal_pack_f16) {
        packed = g_metal_pack_f16(in_sid, pack_src, ic, sess_seq, xscale);
    } else if (!acts_is_f16 && g_metal_pack_f32) {
        packed = g_metal_pack_f32(in_sid, (const float *)pack_src, ic, sess_seq, xscale);
    }
    bool ok = packed && g_eval_only() && g_metal_unpack(out_sid, ic, sess_seq, unpack_dst);
    if (ok && pad_dst) {
        const uint16_t *s = (const uint16_t *)pad_dst;
        uint16_t *d = (uint16_t *)dst_ggml_f16;
        for (int t = 0; t < seq; t++) {
            memcpy(d + (size_t)t * (size_t)ic, s + (size_t)t * (size_t)ic,
                   (size_t)ic * sizeof(uint16_t));
        }
    }
    free(pad_src);
    free(pad_dst);
    if (!ok) {
        uint64_t n = atomic_fetch_add(&g_force_deferred, 1) + 1;
        if (pol.telemetry || n <= 4 || (n % 1024ull) == 0) {
            fprintf(stderr,
                    "ane_ffn_force: metal_layout_failed#%llu ic=%d hidden=%d seq=%d sess_seq=%d\n",
                    (unsigned long long)n, ic, hidden, seq, sess_seq);
        }
        return false;
    }
    uint64_t n = atomic_fetch_add(&g_force_replaced, 1) + 1;
    if (pol.telemetry || n <= 8 || (n % 1024ull) == 0) {
        fprintf(stderr,
                "ane_ffn_force: metal_layout_replaced#%llu ic=%d hidden=%d seq=%d sess_seq=%d port=%d\n",
                (unsigned long long)n, ic, hidden, seq, sess_seq, serve_port);
    }
    (void)Wg_hidden_ic;
    (void)Wu_hidden_ic;
    (void)Wd_ic_hidden;
    return true;
}

bool ane_ffn_force_swiglu_pack_eval_async_host(
    const void *src_ggml_acts, int ic, int seq, int acts_is_f16) {
    ane_ffn_force_autoload_host_replace();
    if (!g_pack_eval_async || !src_ggml_acts || ic <= 0 || seq <= 0) {
        return false;
    }
    return g_pack_eval_async(src_ggml_acts, ic, seq, acts_is_f16);
}

bool ane_ffn_force_swiglu_pack_eval_async_wait_host(void) {
    ane_ffn_force_autoload_host_replace();
    if (!g_pack_eval_async_wait) {
        return false;
    }
    return g_pack_eval_async_wait();
}

bool ane_ffn_force_swiglu_unpack_metal_to_ggml_f16(
    int ic, int seq, void *dst_ggml_f16) {
    ane_ffn_force_autoload_host_replace();
    if (!g_out_sid || !g_metal_unpack || !g_session_seq || !dst_ggml_f16) {
        return false;
    }
    const int sess_seq = g_session_seq();
    uint32_t out_sid = g_out_sid();
    if (sess_seq <= 0 || seq > sess_seq || out_sid == 0) {
        return false;
    }
    if (seq == sess_seq) {
        return g_metal_unpack(out_sid, ic, sess_seq, dst_ggml_f16);
    }
    const size_t n_pad = (size_t)ic * (size_t)sess_seq;
    void *pad_dst = malloc(n_pad * sizeof(uint16_t));
    if (!pad_dst) {
        return false;
    }
    bool ok = g_metal_unpack(out_sid, ic, sess_seq, pad_dst);
    if (ok) {
        const uint16_t *s = (const uint16_t *)pad_dst;
        uint16_t *d = (uint16_t *)dst_ggml_f16;
        for (int t = 0; t < seq; t++) {
            memcpy(d + (size_t)t * (size_t)ic, s + (size_t)t * (size_t)ic,
                   (size_t)ic * sizeof(uint16_t));
        }
    }
    free(pad_dst);
    return ok;
}
