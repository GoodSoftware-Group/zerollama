// Host-buffer ANE replace for ZEROLLAMA_ANE_FFN force path (lab).
// Default: fp16-blob SwiGLU/matmul.
// Opt-in best expert path (env):
//   ZEROLLAMA_ANE_FFN_INT8=1
//   ZEROLLAMA_ANE_FFN_W8A8=1          // hid W8A8 (implies int8 weights)
//   ZEROLLAMA_ANE_FFN_W8A8_X=1        // dual W8A8 x+hid
//   ZEROLLAMA_ANE_FFN_INT8_IN=1       // host int8 acts (implies W8A8_X)
// Session LRU: ZEROLLAMA_ANE_FFN_SCACHE_SLOTS (default 64). Decode reuses a
// prefill session when sess.seq >= padded decode seq (avoids 2× slots).
// Keys: optional stable ggml weight data ids (set_weight_ids) + float staging ptrs.
#import <Foundation/Foundation.h>

#include "ane_ffn_force_replace.h"
#include "ane_prefill_session.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int ic;
    int oc;       // matmul OC, or hidden for swiglu cache key
    int seq;
    int hidden;   // 0 = matmul
    int mode;     // 0=fp16, 1=int8, 2=w8a8(+opts)
    int int8_in;
    float hid_scale;
    float x_scale;
    const float *weight_key;
    const float *wg_key;
    const float *wu_key;
    const float *wd_key;
    // Stable ggml weight ->data (avoids malloc ABA when wcache recycles floats).
    const void *id_wg;
    const void *id_wu;
    const void *id_wd;
    ANEPrefillSession *sess;
    uint64_t tick;
} ForceCacheEntry;

#define SCACHE_SLOTS_MAX 128
#define SCACHE_SLOTS_DEFAULT 64

static ForceCacheEntry g_slots[SCACHE_SLOTS_MAX];
static int g_slots_n = 0;
static uint64_t g_sclock = 1;
static ForceCacheEntry *g_active = NULL; // last used (surface / eval_only APIs)

// Caller-provided stable weight identity (ggml tensor data pointers).
static const void *g_id_wg = NULL;
static const void *g_id_wu = NULL;
static const void *g_id_wd = NULL;

// Shared staging (one eval at a time).
static int8_t *g_qin = NULL;
static size_t g_qin_bytes = 0;
static _Float16 *g_xin = NULL;
static size_t g_xin_bytes = 0;

static int scache_slots(void) {
    if (g_slots_n > 0) return g_slots_n;
    int n = SCACHE_SLOTS_DEFAULT;
    const char *e = getenv("ZEROLLAMA_ANE_FFN_SCACHE_SLOTS");
    if (e && e[0]) {
        int v = atoi(e);
        if (v >= 1 && v <= SCACHE_SLOTS_MAX) n = v;
    }
    g_slots_n = n;
    return n;
}

static void scache_telem(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    const char *telem = getenv("ZEROLLAMA_ANE_FFN_TELEMETRY");
    if (!telem || !telem[0] || !(telem[0] == '1' || telem[0] == 't' || telem[0] == 'T' ||
                                  telem[0] == 'y' || telem[0] == 'Y')) {
        return;
    }
    const char *path = getenv("ZEROLLAMA_ANE_FFN_LOG");
    if (!path || !path[0]) path = "/tmp/ane-ffn-force.log";
    FILE *f = fopen(path, "a");
    if (!f) return;
    va_list ap2;
    va_start(ap2, fmt);
    vfprintf(f, fmt, ap2);
    va_end(ap2);
    fclose(f);
}

static void slot_clear(ForceCacheEntry *e) {
    if (!e) return;
    if (e->sess) {
        ane_prefill_session_destroy(e->sess);
        e->sess = NULL;
    }
    if (g_active == e) g_active = NULL;
    memset(e, 0, sizeof(*e));
}

__attribute__((unused))
static void cache_clear(void) {
    const int n = scache_slots();
    for (int i = 0; i < n; i++) slot_clear(&g_slots[i]);
    free(g_qin); g_qin = NULL; g_qin_bytes = 0;
    free(g_xin); g_xin = NULL; g_xin_bytes = 0;
}

static bool cache_ensure_qin(size_t bytes) {
    if (g_qin && g_qin_bytes >= bytes) return true;
    free(g_qin);
    g_qin = (int8_t *)malloc(bytes);
    g_qin_bytes = g_qin ? bytes : 0;
    return g_qin != NULL;
}

static bool cache_ensure_xin(size_t bytes) {
    if (g_xin && g_xin_bytes >= bytes) return true;
    free(g_xin);
    g_xin = (_Float16 *)malloc(bytes);
    g_xin_bytes = g_xin ? bytes : 0;
    return g_xin != NULL;
}

static ForceCacheEntry *scache_victim(void) {
    const int n = scache_slots();
    for (int i = 0; i < n; i++) {
        if (!g_slots[i].sess) return &g_slots[i];
    }
    int v = 0;
    for (int i = 1; i < n; i++) {
        if (g_slots[i].tick < g_slots[v].tick) v = i;
    }
    slot_clear(&g_slots[v]);
    return &g_slots[v];
}

void ane_ffn_force_swiglu_set_weight_ids(
    const void *wg_data, const void *wu_data, const void *wd_data) {
    g_id_wg = wg_data;
    g_id_wu = wu_data;
    g_id_wd = wd_data;
}

static bool scache_keys_match(
    const ForceCacheEntry *e,
    const float *Wg, const float *Wu, const float *Wd) {
    if (!e) return false;
    // Prefer stable ggml ids when the caller set them (multi-layer serve).
    if (g_id_wg || g_id_wu || g_id_wd) {
        return e->id_wg == g_id_wg && e->id_wu == g_id_wu && e->id_wd == g_id_wd;
    }
    return e->wg_key == Wg && e->wu_key == Wu && e->wd_key == Wd;
}

static ForceCacheEntry *scache_find_swiglu(
    int ic, int hidden, int min_seq, int mode, int int8_in,
    const float *Wg, const float *Wu, const float *Wd) {
    const int n = scache_slots();
    ForceCacheEntry *best = NULL;
    for (int i = 0; i < n; i++) {
        ForceCacheEntry *e = &g_slots[i];
        if (!e->sess || e->hidden != hidden || e->ic != ic ||
            e->mode != mode || e->int8_in != int8_in ||
            !scache_keys_match(e, Wg, Wu, Wd)) {
            continue;
        }
        if (e->seq < min_seq) continue;
        if (!best || e->seq < best->seq) best = e;
    }
    return best;
}

static void scache_touch(ForceCacheEntry *e, int hit) {
    if (!e) return;
    e->tick = ++g_sclock;
    g_active = e;
    static uint64_t hits = 0, misses = 0;
    uint64_t n = hit ? ++hits : ++misses;
    // Rate-limit: first 24 + every 64th (lab: one miss/layer on cold prefill).
    if (n <= 24 || (n % 64ull) == 0) {
        scache_telem(
            "ane_ffn_force: scache_%s#%llu slots=%d ic=%d hidden=%d seq=%d\n",
            hit ? "hit" : "miss",
            (unsigned long long)n, scache_slots(), e->ic, e->hidden, e->seq);
    }
}

static bool env_on(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return false;
    if (v[0] == '1') return true;
    if (v[0] == 't' || v[0] == 'T') return true; // true
    if (v[0] == 'y' || v[0] == 'Y') return true; // yes
    if ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N')) return true; // on
    return false;
}

// Lab kernel mode for SwiGLU replace.
static void swiglu_mode_from_env(bool *want_int8, bool *w8a8_hid, bool *w8a8_x, bool *int8_in) {
    *want_int8 = env_on("ZEROLLAMA_ANE_FFN_INT8");
    *w8a8_hid = env_on("ZEROLLAMA_ANE_FFN_W8A8");
    *w8a8_x = env_on("ZEROLLAMA_ANE_FFN_W8A8_X");
    *int8_in = env_on("ZEROLLAMA_ANE_FFN_INT8_IN");
    if (*int8_in) {
        *w8a8_x = true;
        *w8a8_hid = true;
        *want_int8 = true;
    } else if (*w8a8_x) {
        *w8a8_hid = true;
        *want_int8 = true;
    } else if (*w8a8_hid) {
        *want_int8 = true;
    }
}

static float silu_f(float x) {
    return x / (1.f + expf(-x));
}

static void cpu_matmul(const float *W, const float *X, float *Y, int ic, int oc, int seq) {
    for (int o = 0; o < oc; o++) {
        for (int s = 0; s < seq; s++) {
            float acc = 0;
            for (int i = 0; i < ic; i++) {
                acc += W[o * ic + i] * X[i * seq + s];
            }
            Y[o * seq + s] = acc;
        }
    }
}

static float quantize_dequant_inplace(float *w, int n) {
    float maxAbs = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / 127.0f;
    if (!(scale > 0)) scale = 1.0f;
    for (int i = 0; i < n; i++) {
        float v = w[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        int8_t q = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        w[i] = (float)q * scale;
    }
    return scale;
}

static void apply_act_int8_roundtrip(float *x, int n, float scale) {
    if (!(scale > 0)) return;
    for (int i = 0; i < n; i++) {
        float v = x[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        int8_t q = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        x[i] = (float)q * scale;
    }
}

static float cpu_hid_max_abs(const float *Wg, const float *Wu, const float *X,
                             int ic, int hidden, int seq) {
    float *gate = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    float *up = (float *)malloc((size_t)hidden * (size_t)seq * sizeof(float));
    if (!gate || !up) {
        free(gate);
        free(up);
        return 1.0f;
    }
    cpu_matmul(Wg, X, gate, ic, hidden, seq);
    cpu_matmul(Wu, X, up, ic, hidden, seq);
    float mx = 0;
    for (int i = 0; i < hidden * seq; i++) {
        float a = fabsf(silu_f(gate[i]) * up[i]);
        if (a > mx) mx = a;
    }
    free(gate);
    free(up);
    return mx > 0 ? mx : 1.0f;
}

static float max_abs_f32(const float *x, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

static void quantize_f32_to_int8(const float *x, int8_t *q, int n, float scale) {
    float s = scale > 0 ? scale : 1.0f;
    for (int i = 0; i < n; i++) {
        float v = x[i] / s;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
}

static void quantize_f16_to_int8(const _Float16 *x, int8_t *q, int n, float scale) {
    float s = scale > 0 ? scale : 1.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)x[i] / s;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
}

// Calibrate scales for W8A8 session create; returns false on alloc failure.
static bool calibrate_w8a8_scales(int ic, int hidden, int seq,
                                  const float *Wg, const float *Wu,
                                  const float *X_f32,
                                  bool w8a8_x, bool int8_in,
                                  float *hid_scale_out, float *x_scale_out) {
    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nAct = (size_t)ic * (size_t)seq;
    float *Gdq = (float *)malloc(nGate * sizeof(float));
    float *Udq = (float *)malloc(nGate * sizeof(float));
    if (!Gdq || !Udq) {
        free(Gdq);
        free(Udq);
        return false;
    }
    memcpy(Gdq, Wg, nGate * sizeof(float));
    memcpy(Udq, Wu, nGate * sizeof(float));
    (void)quantize_dequant_inplace(Gdq, (int)nGate);
    (void)quantize_dequant_inplace(Udq, (int)nGate);

    float xScale = 0;
    float *Xcal = (float *)X_f32;
    float *Xrt = NULL;
    if (w8a8_x || int8_in) {
        xScale = max_abs_f32(X_f32, (int)nAct) / 127.0f;
        if (!(xScale > 0)) xScale = 1.0f;
        Xrt = (float *)malloc(nAct * sizeof(float));
        if (!Xrt) {
            free(Gdq);
            free(Udq);
            return false;
        }
        memcpy(Xrt, X_f32, nAct * sizeof(float));
        apply_act_int8_roundtrip(Xrt, (int)nAct, xScale);
        Xcal = Xrt;
    }

    float hidMax = cpu_hid_max_abs(Gdq, Udq, Xcal, ic, hidden, seq);
    float hidScale = hidMax / 127.0f;
    if (!(hidScale > 0)) hidScale = 1.0f;

    free(Xrt);
    free(Gdq);
    free(Udq);
    *hid_scale_out = hidScale;
    *x_scale_out = xScale;
    return true;
}

static ANEPrefillSession *create_swiglu_for_mode(
    int ic, int hidden, int seq,
    const float *Wg, const float *Wu, const float *Wd,
    const float *X_f32,
    bool want_int8, bool w8a8_hid, bool w8a8_x, bool int8_in,
    float *hid_scale_out, float *x_scale_out) {
    *hid_scale_out = 0;
    *x_scale_out = 0;
    if (w8a8_hid || w8a8_x || int8_in) {
        float hidScale = 0, xScale = 0;
        if (!calibrate_w8a8_scales(ic, hidden, seq, Wg, Wu, X_f32,
                                   w8a8_x || int8_in, int8_in,
                                   &hidScale, &xScale)) {
            return NULL;
        }
        int sp0 = 0, sp1 = 0;
        ane_prefill_session_pick_tile(seq, &sp0, &sp1);
        *hid_scale_out = hidScale;
        *x_scale_out = xScale;
        return ane_prefill_session_create_swiglu_int8_w8a8(
            ic, hidden, seq, Wg, Wu, Wd,
            hidScale, (w8a8_x || int8_in) ? xScale : 0,
            sp0, sp1, int8_in);
    }
    if (want_int8) {
        return ane_prefill_session_create_swiglu_int8(ic, hidden, seq, Wg, Wu, Wd);
    }
    return ane_prefill_session_create_swiglu(ic, hidden, seq, Wg, Wu, Wd);
}

static int mode_tag(bool want_int8, bool w8a8_hid, bool w8a8_x, bool int8_in) {
    if (w8a8_hid || w8a8_x || int8_in) return 2;
    if (want_int8) return 1;
    return 0;
}

// ANE SwiGLU MIL at large IC×H rejects many seq lengths. Lab (2048→512 int8-in):
// lengths ≡ 32 (mod 64) fail compile (32,96,160,224,…); multiples of 64 work.
// Pad channel-major acts up to a multiple of 64 (min 64).
static int swiglu_pad_seq(int seq) {
    const int align = 64;
    if (seq <= 0) {
        return 0;
    }
    int p = ((seq + align - 1) / align) * align;
    return p < align ? align : p;
}

static void pad_channel_major_i8(
    const int8_t *src, int ic, int seq_src, int8_t *dst, int seq_dst) {
    for (int i = 0; i < ic; i++) {
        for (int t = 0; t < seq_dst; t++) {
            dst[(size_t)i * (size_t)seq_dst + (size_t)t] =
                (t < seq_src) ? src[(size_t)i * (size_t)seq_src + (size_t)t] : 0;
        }
    }
}

static void pad_channel_major_f16(
    const _Float16 *src, int ic, int seq_src, _Float16 *dst, int seq_dst) {
    for (int i = 0; i < ic; i++) {
        for (int t = 0; t < seq_dst; t++) {
            dst[(size_t)i * (size_t)seq_dst + (size_t)t] =
                (t < seq_src) ? src[(size_t)i * (size_t)seq_src + (size_t)t]
                              : (_Float16)0;
        }
    }
}

static void pad_channel_major_f32(
    const float *src, int ic, int seq_src, float *dst, int seq_dst) {
    for (int i = 0; i < ic; i++) {
        for (int t = 0; t < seq_dst; t++) {
            dst[(size_t)i * (size_t)seq_dst + (size_t)t] =
                (t < seq_src) ? src[(size_t)i * (size_t)seq_src + (size_t)t] : 0.f;
        }
    }
}

static void crop_channel_major_f16(
    const _Float16 *src, int ic, int seq_src, _Float16 *dst, int seq_dst) {
    for (int i = 0; i < ic; i++) {
        memcpy(dst + (size_t)i * (size_t)seq_dst,
               src + (size_t)i * (size_t)seq_src,
               (size_t)seq_dst * sizeof(_Float16));
    }
}

static void crop_channel_major_f32(
    const float *src, int ic, int seq_src, float *dst, int seq_dst) {
    for (int i = 0; i < ic; i++) {
        memcpy(dst + (size_t)i * (size_t)seq_dst,
               src + (size_t)i * (size_t)seq_src,
               (size_t)seq_dst * sizeof(float));
    }
}

static bool write_acts_for_sess(ANEPrefillSession *sess, const float *X_f32,
                                int ic, int seq, float x_scale) {
    size_t inBytes = ane_prefill_session_input_bytes(sess);
    size_t n = (size_t)ic * (size_t)seq;
    if (ane_prefill_session_is_int8_input(sess)) {
        if (inBytes != n * sizeof(int8_t)) return false;
        if (!cache_ensure_qin(inBytes)) return false;
        quantize_f32_to_int8(X_f32, g_qin, (int)n, x_scale);
        return ane_prefill_session_write_acts_int8(sess, g_qin, inBytes);
    }
    if (inBytes != n * sizeof(_Float16)) return false;
    if (!cache_ensure_xin(inBytes)) return false;
    for (size_t i = 0; i < n; i++) {
        g_xin[i] = (_Float16)X_f32[i];
    }
    return ane_prefill_session_write_acts_fp16(sess, g_xin, inBytes);
}

static bool write_acts_f16_for_sess(ANEPrefillSession *sess, const void *X_f16,
                                    int ic, int seq, float x_scale) {
    size_t inBytes = ane_prefill_session_input_bytes(sess);
    size_t n = (size_t)ic * (size_t)seq;
    if (ane_prefill_session_is_int8_input(sess)) {
        if (inBytes != n * sizeof(int8_t)) return false;
        if (!cache_ensure_qin(inBytes)) return false;
        quantize_f16_to_int8((const _Float16 *)X_f16, g_qin, (int)n, x_scale);
        return ane_prefill_session_write_acts_int8(sess, g_qin, inBytes);
    }
    if (inBytes != n * sizeof(_Float16)) return false;
    return ane_prefill_session_write_acts_fp16(sess, X_f16, inBytes);
}

static bool output_finite_nonzero_f32(const float *Y, size_t n) {
    float mx = 0;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(Y[i]);
        if (a > mx) mx = a;
    }
    return (mx > 0) && isfinite(mx);
}

// Ensure a SwiGLU session for (weights, mode) with sess.seq >= min_seq.
// Prefers exact seq, else smallest compatible (decode reuses prefill pad).
static ForceCacheEntry *scache_ensure_swiglu(
    int ic, int hidden, int min_seq,
    const float *Wg, const float *Wu, const float *Wd,
    const float *X_cal, // may be NULL for fp16-only create
    bool want_int8, bool w8a8_hid, bool w8a8_x, bool int8_in) {
    int tag = mode_tag(want_int8, w8a8_hid, w8a8_x, int8_in);
    int i8 = int8_in ? 1 : 0;
    ForceCacheEntry *e = scache_find_swiglu(ic, hidden, min_seq, tag, i8, Wg, Wu, Wd);
    if (e) {
        scache_touch(e, 1);
        return e;
    }

    float hidScale = 0, xScale = 0;
    const float *Xcal = X_cal;
    float *Xones = NULL;
    size_t n = (size_t)ic * (size_t)min_seq;
    if (!Xcal && (w8a8_hid || w8a8_x || int8_in)) {
        return NULL;
    }
    if (!Xcal) {
        Xones = (float *)malloc(n * sizeof(float));
        if (!Xones) return NULL;
        for (size_t i = 0; i < n; i++) Xones[i] = 1.0f;
        Xcal = Xones;
    }
    ANEPrefillSession *sess = create_swiglu_for_mode(
        ic, hidden, min_seq, Wg, Wu, Wd, Xcal,
        want_int8, w8a8_hid, w8a8_x, int8_in, &hidScale, &xScale);
    free(Xones);
    if (!sess || !ane_prefill_session_ready(sess)) {
        if (sess) ane_prefill_session_destroy(sess);
        return NULL;
    }
    e = scache_victim();
    e->ic = ic;
    e->oc = ic;
    e->seq = min_seq;
    e->hidden = hidden;
    e->mode = tag;
    e->int8_in = i8;
    e->hid_scale = hidScale;
    e->x_scale = xScale;
    e->wg_key = Wg;
    e->wu_key = Wu;
    e->wd_key = Wd;
    e->id_wg = g_id_wg;
    e->id_wu = g_id_wu;
    e->id_wd = g_id_wd;
    e->sess = sess;
    scache_touch(e, 0);
    return e;
}

bool ane_ffn_force_replace_mul_mat(
    int ic, int oc, int seq,
    const float *W_oc_ic,
    const float *X_ic_seq,
    float *Y_oc_seq) {
    if (ic <= 0 || oc <= 0 || seq <= 0 || !W_oc_ic || !X_ic_seq || !Y_oc_seq) {
        return false;
    }

    ForceCacheEntry *e = NULL;
    const int nslots = scache_slots();
    for (int i = 0; i < nslots; i++) {
        ForceCacheEntry *c = &g_slots[i];
        if (c->sess && c->hidden == 0 && c->ic == ic && c->oc == oc &&
            c->seq == seq && c->weight_key == W_oc_ic) {
            e = c;
            break;
        }
    }
    if (!e) {
        ANEPrefillSession *sess = ane_prefill_session_create(ic, oc, seq, W_oc_ic);
        if (!sess || !ane_prefill_session_ready(sess)) {
            if (sess) ane_prefill_session_destroy(sess);
            return false;
        }
        e = scache_victim();
        e->ic = ic;
        e->oc = oc;
        e->seq = seq;
        e->hidden = 0;
        e->weight_key = W_oc_ic;
        e->sess = sess;
        scache_touch(e, 0);
    } else {
        scache_touch(e, 1);
    }

    ANEPrefillSession *sess = e->sess;
    size_t inBytes = ane_prefill_session_input_bytes(sess);
    size_t outBytes = ane_prefill_session_output_bytes(sess);
    size_t nIn = (size_t)ic * (size_t)seq;
    size_t nOut = (size_t)oc * (size_t)seq;
    if (inBytes != nIn * sizeof(_Float16) || outBytes != nOut * sizeof(_Float16)) {
        return false;
    }

    _Float16 *X16 = (_Float16 *)malloc(inBytes);
    _Float16 *Y16 = (_Float16 *)malloc(outBytes);
    if (!X16 || !Y16) {
        free(X16);
        free(Y16);
        return false;
    }
    for (size_t i = 0; i < nIn; i++) {
        X16[i] = (_Float16)X_ic_seq[i];
    }

    bool ok = ane_prefill_session_write_acts_fp16(sess, X16, inBytes)
           && ane_prefill_session_eval(sess)
           && ane_prefill_session_read_out_fp16(sess, Y16, outBytes);
    if (ok) {
        for (size_t i = 0; i < nOut; i++) {
            Y_oc_seq[i] = (float)Y16[i];
        }
    }
    free(X16);
    free(Y16);
    return ok;
}

bool ane_ffn_force_replace_swiglu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const float *X_ic_seq,
    float *Y_ic_seq) {
    if (ic <= 0 || hidden <= 0 || seq <= 0 ||
        !Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !X_ic_seq || !Y_ic_seq) {
        return false;
    }

    const int min_pad = swiglu_pad_seq(seq);
    bool want_int8 = false, w8a8_hid = false, w8a8_x = false, int8_in = false;
    swiglu_mode_from_env(&want_int8, &w8a8_hid, &w8a8_x, &int8_in);

    ForceCacheEntry *e = scache_ensure_swiglu(
        ic, hidden, min_pad, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden, X_ic_seq,
        want_int8, w8a8_hid, w8a8_x, int8_in);
    if (!e) return false;

    const int sess_seq = e->seq;
    ANEPrefillSession *sess = e->sess;
    size_t nSess = (size_t)ic * (size_t)sess_seq;
    size_t outBytes = ane_prefill_session_output_bytes(sess);
    if (outBytes != nSess * sizeof(_Float16)) {
        return false;
    }

    const float *Xuse = X_ic_seq;
    float *Xp = NULL;
    float *Yp = NULL;
    if (sess_seq != seq) {
        Xp = (float *)calloc(nSess, sizeof(float));
        Yp = (float *)malloc(nSess * sizeof(float));
        if (!Xp || !Yp) {
            free(Xp); free(Yp);
            return false;
        }
        pad_channel_major_f32(X_ic_seq, ic, seq, Xp, sess_seq);
        Xuse = Xp;
    }

    bool ok = write_acts_for_sess(sess, Xuse, ic, sess_seq, e->x_scale)
           && ane_prefill_session_eval(sess)
           && ane_prefill_session_read_out_f32(sess, Yp ? Yp : Y_ic_seq, nSess);
    if (ok) {
        ok = output_finite_nonzero_f32(Yp ? Yp : Y_ic_seq, nSess);
    }
    if (ok && Yp) {
        crop_channel_major_f32(Yp, ic, sess_seq, Y_ic_seq, seq);
    }
    free(Xp);
    free(Yp);
    return ok;
}

bool ane_ffn_force_replace_swiglu_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const void *X_ic_seq_f16,
    void *Y_ic_seq_f16) {
    if (ic <= 0 || hidden <= 0 || seq <= 0 ||
        !Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden || !X_ic_seq_f16 || !Y_ic_seq_f16) {
        return false;
    }

    const int min_pad = swiglu_pad_seq(seq);
    bool want_int8 = false, w8a8_hid = false, w8a8_x = false, int8_in = false;
    swiglu_mode_from_env(&want_int8, &w8a8_hid, &w8a8_x, &int8_in);

    // Calibration X only needed on miss for W8A8 modes — build at min_pad.
    float *Xf = NULL;
    ForceCacheEntry *existing = scache_find_swiglu(
        ic, hidden, min_pad, mode_tag(want_int8, w8a8_hid, w8a8_x, int8_in),
        int8_in ? 1 : 0, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden);
    if (!existing && (w8a8_hid || w8a8_x || int8_in)) {
        size_t n = (size_t)ic * (size_t)min_pad;
        Xf = (float *)calloc(n, sizeof(float));
        if (!Xf) return false;
        const _Float16 *X16in = (const _Float16 *)X_ic_seq_f16;
        for (int t = 0; t < seq; t++) {
            for (int i = 0; i < ic; i++) {
                Xf[(size_t)i * (size_t)min_pad + (size_t)t] =
                    (float)X16in[(size_t)i * (size_t)seq + (size_t)t];
            }
        }
    }

    ForceCacheEntry *e = scache_ensure_swiglu(
        ic, hidden, min_pad, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden, Xf,
        want_int8, w8a8_hid, w8a8_x, int8_in);
    free(Xf);
    if (!e) return false;

    const int sess_seq = e->seq;
    ANEPrefillSession *sess = e->sess;
    size_t nSess = (size_t)ic * (size_t)sess_seq;
    size_t outBytes = ane_prefill_session_output_bytes(sess);
    if (outBytes != nSess * sizeof(_Float16)) {
        return false;
    }

    const void *Xuse = X_ic_seq_f16;
    _Float16 *Xp = NULL;
    _Float16 *Yp = NULL;
    if (sess_seq != seq) {
        Xp = (_Float16 *)calloc(nSess, sizeof(_Float16));
        Yp = (_Float16 *)malloc(nSess * sizeof(_Float16));
        if (!Xp || !Yp) {
            free(Xp); free(Yp);
            return false;
        }
        pad_channel_major_f16((const _Float16 *)X_ic_seq_f16, ic, seq, Xp, sess_seq);
        Xuse = Xp;
    }

    const bool do_prof = env_on("ZEROLLAMA_ANE_FFN_PROFILE");
    CFAbsoluteTime t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if (do_prof) t0 = CFAbsoluteTimeGetCurrent();
    bool ok = write_acts_f16_for_sess(sess, Xuse, ic, sess_seq, e->x_scale);
    if (do_prof) t1 = CFAbsoluteTimeGetCurrent();
    if (ok) ok = ane_prefill_session_eval(sess);
    if (do_prof) t2 = CFAbsoluteTimeGetCurrent();
    if (ok) {
        ok = ane_prefill_session_read_out_fp16(
                  sess, Yp ? Yp : Y_ic_seq_f16, outBytes);
    }
    if (do_prof) t3 = CFAbsoluteTimeGetCurrent();
    if (do_prof && ok) {
        static double wms, ems, rms;
        static uint64_t pn;
        wms += (t1 - t0) * 1e3;
        ems += (t2 - t1) * 1e3;
        rms += (t3 - t2) * 1e3;
        pn++;
        if (pn == 48 || pn == 96 || (pn % 384ull) == 0) {
            scache_telem(
                "ane_ffn_profile_dylib: after#%llu write=%.1fms eval=%.1fms read=%.1fms "
                "(avg write=%.3f eval=%.3f read=%.3f)\n",
                (unsigned long long)pn, wms, ems, rms,
                wms / (double)pn, ems / (double)pn, rms / (double)pn);
        }
    }
    if (ok) {
        const _Float16 *Y16 = (const _Float16 *)(Yp ? Yp : Y_ic_seq_f16);
        float mx = 0;
        for (size_t i = 0; i < nSess; i++) {
            float a = fabsf((float)Y16[i]);
            if (a > mx) mx = a;
        }
        if (!(mx > 0) || !isfinite(mx)) {
            ok = false;
        }
    }
    if (ok && Yp) {
        crop_channel_major_f16(Yp, ic, sess_seq, (_Float16 *)Y_ic_seq_f16, seq);
    }
    free(Xp);
    free(Yp);
    return ok;
}

bool ane_ffn_force_replace_swiglu_int8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    float *Y_ic_seq) {
    if (ic <= 0 || hidden <= 0 || seq <= 0 ||
        !Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden ||
        !X_ic_seq_i8 || !Y_ic_seq) {
        return false;
    }
    const int min_pad = swiglu_pad_seq(seq);
    ForceCacheEntry *e = scache_find_swiglu(
        ic, hidden, min_pad, 2, 1, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden);
    if (!e || !e->int8_in || !ane_prefill_session_is_int8_input(e->sess)) {
        return false;
    }
    scache_touch(e, 1);
    const int sess_seq = e->seq;
    ANEPrefillSession *sess = e->sess;
    size_t nSess = (size_t)ic * (size_t)sess_seq;
    size_t inBytes = ane_prefill_session_input_bytes(sess);
    size_t outBytes = ane_prefill_session_output_bytes(sess);
    if (inBytes != nSess * sizeof(int8_t) || outBytes != nSess * sizeof(_Float16)) {
        return false;
    }
    const int8_t *Xuse = X_ic_seq_i8;
    int8_t *Xp = NULL;
    float *Yp = NULL;
    if (sess_seq != seq) {
        Xp = (int8_t *)calloc(nSess, 1);
        Yp = (float *)malloc(nSess * sizeof(float));
        if (!Xp || !Yp) {
            free(Xp); free(Yp);
            return false;
        }
        pad_channel_major_i8(X_ic_seq_i8, ic, seq, Xp, sess_seq);
        Xuse = Xp;
    }
    bool ok = ane_prefill_session_write_acts_int8(sess, Xuse, inBytes)
           && ane_prefill_session_eval(sess)
           && ane_prefill_session_read_out_f32(sess, Yp ? Yp : Y_ic_seq, nSess);
    if (ok) {
        ok = output_finite_nonzero_f32(Yp ? Yp : Y_ic_seq, nSess);
    }
    if (ok && Yp) {
        crop_channel_major_f32(Yp, ic, sess_seq, Y_ic_seq, seq);
    }
    free(Xp);
    free(Yp);
    return ok;
}

bool ane_ffn_force_replace_swiglu_int8_fp16(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    const int8_t *X_ic_seq_i8,
    void *Y_ic_seq_f16) {
    if (ic <= 0 || hidden <= 0 || seq <= 0 ||
        !Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden ||
        !X_ic_seq_i8 || !Y_ic_seq_f16) {
        return false;
    }
    const int min_pad = swiglu_pad_seq(seq);
    ForceCacheEntry *e = scache_find_swiglu(
        ic, hidden, min_pad, 2, 1, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden);
    if (!e || !e->int8_in || !ane_prefill_session_is_int8_input(e->sess)) {
        return false;
    }
    scache_touch(e, 1);
    const int sess_seq = e->seq;
    ANEPrefillSession *sess = e->sess;
    size_t nSess = (size_t)ic * (size_t)sess_seq;
    size_t inBytes = ane_prefill_session_input_bytes(sess);
    size_t outBytes = ane_prefill_session_output_bytes(sess);
    if (inBytes != nSess * sizeof(int8_t) || outBytes != nSess * sizeof(_Float16)) {
        return false;
    }
    const int8_t *Xuse = X_ic_seq_i8;
    int8_t *Xp = NULL;
    _Float16 *Yp = NULL;
    if (sess_seq != seq) {
        Xp = (int8_t *)calloc(nSess, 1);
        Yp = (_Float16 *)malloc(nSess * sizeof(_Float16));
        if (!Xp || !Yp) {
            free(Xp); free(Yp);
            return false;
        }
        pad_channel_major_i8(X_ic_seq_i8, ic, seq, Xp, sess_seq);
        Xuse = Xp;
    }
    bool ok = ane_prefill_session_write_acts_int8(sess, Xuse, inBytes)
           && ane_prefill_session_eval(sess)
           && ane_prefill_session_read_out_fp16(sess, Yp ? Yp : Y_ic_seq_f16, outBytes);
    if (ok) {
        const _Float16 *Y16 = (const _Float16 *)(Yp ? Yp : Y_ic_seq_f16);
        float mx = 0;
        for (size_t i = 0; i < nSess; i++) {
            float a = fabsf((float)Y16[i]);
            if (a > mx) mx = a;
        }
        if (!(mx > 0) || !isfinite(mx)) {
            ok = false;
        }
    }
    if (ok && Yp) {
        crop_channel_major_f16(Yp, ic, sess_seq, (_Float16 *)Y_ic_seq_f16, seq);
    }
    free(Xp);
    free(Yp);
    return ok;
}

float ane_ffn_force_swiglu_x_scale(void) {
    return g_active ? g_active->x_scale : 0.f;
}

bool ane_ffn_force_swiglu_activate(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden) {
    if (ic <= 0 || hidden <= 0 || seq <= 0 ||
        !Wg_hidden_ic || !Wu_hidden_ic || !Wd_ic_hidden) {
        return false;
    }
    bool want_int8 = false, w8a8_hid = false, w8a8_x = false, int8_in = false;
    swiglu_mode_from_env(&want_int8, &w8a8_hid, &w8a8_x, &int8_in);
    const int min_pad = swiglu_pad_seq(seq);
    ForceCacheEntry *e = scache_find_swiglu(
        ic, hidden, min_pad, mode_tag(want_int8, w8a8_hid, w8a8_x, int8_in),
        int8_in ? 1 : 0, Wg_hidden_ic, Wu_hidden_ic, Wd_ic_hidden);
    if (!e) {
        return false;
    }
    scache_touch(e, 1);
    return true;
}

uint32_t ane_ffn_force_swiglu_input_surface_id(void) {
    return g_active && g_active->sess
        ? ane_prefill_session_input_surface_id(g_active->sess) : 0;
}

uint32_t ane_ffn_force_swiglu_output_surface_id(void) {
    return g_active && g_active->sess
        ? ane_prefill_session_output_surface_id(g_active->sess) : 0;
}

int ane_ffn_force_swiglu_session_seq(void) {
    return g_active ? g_active->seq : 0;
}

bool ane_ffn_force_swiglu_write_int8(const int8_t *X_ic_seq_i8, int seq) {
    if (!g_active || !g_active->sess || !g_active->int8_in || !X_ic_seq_i8 || seq <= 0 ||
        !ane_prefill_session_is_int8_input(g_active->sess)) {
        return false;
    }
    const int pad = g_active->seq;
    if (seq > pad) {
        return false;
    }
    size_t inBytes = ane_prefill_session_input_bytes(g_active->sess);
    if (inBytes != (size_t)g_active->ic * (size_t)pad) {
        return false;
    }
    if (seq == pad) {
        return ane_prefill_session_write_acts_int8(g_active->sess, X_ic_seq_i8, inBytes);
    }
    int8_t *Xp = (int8_t *)calloc((size_t)g_active->ic * (size_t)pad, 1);
    if (!Xp) {
        return false;
    }
    pad_channel_major_i8(X_ic_seq_i8, g_active->ic, seq, Xp, pad);
    bool ok = ane_prefill_session_write_acts_int8(g_active->sess, Xp, inBytes);
    free(Xp);
    return ok;
}

bool ane_ffn_force_swiglu_reeval_f32(float *Y_ic_seq, int seq) {
    if (!g_active || !g_active->sess || !Y_ic_seq || seq <= 0) {
        return false;
    }
    const int pad = g_active->seq;
    if (seq > pad) {
        return false;
    }
    size_t nPad = (size_t)g_active->ic * (size_t)pad;
    if (seq == pad) {
        bool ok = ane_prefill_session_eval(g_active->sess)
               && ane_prefill_session_read_out_f32(g_active->sess, Y_ic_seq, nPad);
        if (ok) {
            ok = output_finite_nonzero_f32(Y_ic_seq, nPad);
        }
        return ok;
    }
    float *Yp = (float *)malloc(nPad * sizeof(float));
    if (!Yp) {
        return false;
    }
    bool ok = ane_prefill_session_eval(g_active->sess)
           && ane_prefill_session_read_out_f32(g_active->sess, Yp, nPad);
    if (ok) {
        ok = output_finite_nonzero_f32(Yp, nPad);
    }
    if (ok) {
        crop_channel_major_f32(Yp, g_active->ic, pad, Y_ic_seq, seq);
    }
    free(Yp);
    return ok;
}

bool ane_ffn_force_swiglu_eval_only(void) {
    if (!g_active || !g_active->sess) return false;
    return ane_prefill_session_eval(g_active->sess);
}

void ane_ffn_force_register_host_replace(void) {
    // Smoke / same-image: call ane_ffn_force_set_host_replace(ane_ffn_force_replace_mul_mat).
    // Dylib lab path: policy dlsyms ane_ffn_force_replace_mul_mat (no policy in dylib).
}
