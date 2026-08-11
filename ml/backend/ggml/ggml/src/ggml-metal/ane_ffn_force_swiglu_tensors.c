// Pack fused SwiGLU ggml nodes → fold optional scales → ANE host replace → unpack dst.
// Weights cached by ggml data pointers so the dylib session cache can hit.
// Acts: ggml→fp16 channel pack by default; INT8_IN → ggml→int8 pack + int8_fp16 replace.
#include "ane_ffn_policy.h"
#include "ane_ffn_force_pack.h"
#include "ane_ffn_swiglu_fuse.h"

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int type_is_f16(enum ggml_type t) {
    return t == GGML_TYPE_F16;
}

static bool env_on(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return false;
    if (v[0] == '1') return true;
    if (v[0] == 't' || v[0] == 'T') return true;
    if (v[0] == 'y' || v[0] == 'Y') return true;
    if ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N')) return true;
    return false;
}

static bool tensor_host_shared(const struct ggml_tensor * t) {
    if (!t || !t->data) {
        return false;
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buffer) {
        return true;
    }
    return ggml_metal_buffer_is_shared((ggml_metal_buffer_t) buffer->context);
}

static bool fold_scale(
    float *W_oc_ic, int ic, int oc,
    const struct ggml_tensor * scale_mul) {
    if (!scale_mul) {
        return true;
    }
    const struct ggml_tensor * s = scale_mul->src[1];
    if (!s || !s->data || !tensor_host_shared(s)) {
        return false;
    }
    const int nscale = (int)ggml_nelements(s);
    return ane_ffn_fold_out_scale_f32(
        W_oc_ic, ic, oc, s->data, type_is_f16(s->type), nscale);
}

typedef struct {
    const void * wg_src;
    const void * wu_src;
    const void * wd_src;
    const void * su_src;
    const void * sg_src;
    const void * sd_src;
    int ic;
    int hidden;
    float *Wg;
    float *Wu;
    float *Wd;
    uint64_t tick; // LRU stamp
} SwigluWeightCache;

// Multi-slot LRU: Q4_K_M dense FFN is ~150MB/layer; a single slot thrash-dequants
// every layer on every token. Default 32 covers eliza-1.2B (24 layers) + headroom.
// Override: ZEROLLAMA_ANE_FFN_WCACHE_SLOTS (1..128). Default 64 covers MoE shexp layers.
#define WCACHE_SLOTS_MAX 128
#define WCACHE_SLOTS_DEFAULT 64

static SwigluWeightCache g_wslots[WCACHE_SLOTS_MAX];
static int g_wslots_n = 0; // 0 = uninitialized
static uint64_t g_wclock = 1;

typedef struct {
    int ic;
    int seq;
    int8_t *Xi8;
    uint16_t *X16;
    uint16_t *Y16;
    size_t n;
} SwigluActCache;

static SwigluActCache g_acache = {0};

static int wcache_slots_configured(void) {
    if (g_wslots_n > 0) {
        return g_wslots_n;
    }
    int n = WCACHE_SLOTS_DEFAULT;
    const char *e = getenv("ZEROLLAMA_ANE_FFN_WCACHE_SLOTS");
    if (e && e[0]) {
        int v = atoi(e);
        if (v >= 1 && v <= WCACHE_SLOTS_MAX) {
            n = v;
        }
    }
    g_wslots_n = n;
    return n;
}

static void wcache_slot_clear(SwigluWeightCache *s) {
    free(s->Wg);
    free(s->Wu);
    free(s->Wd);
    memset(s, 0, sizeof(*s));
}

static void acache_clear(void) {
    free(g_acache.Xi8);
    free(g_acache.X16);
    free(g_acache.Y16);
    memset(&g_acache, 0, sizeof(g_acache));
}

static bool acache_ensure(int ic, int seq) {
    size_t n = (size_t)ic * (size_t)seq;
    if (g_acache.Y16 && g_acache.ic == ic && g_acache.seq == seq && g_acache.n == n) {
        return true;
    }
    acache_clear();
    g_acache.X16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    g_acache.Y16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    g_acache.Xi8 = (int8_t *)malloc(n * sizeof(int8_t));
    if (!g_acache.X16 || !g_acache.Y16 || !g_acache.Xi8) {
        acache_clear();
        return false;
    }
    g_acache.ic = ic;
    g_acache.seq = seq;
    g_acache.n = n;
    return true;
}

static const void * scale_data(const struct ggml_tensor * scale_mul) {
    return scale_mul && scale_mul->src[1] ? scale_mul->src[1]->data : NULL;
}

// Pack weight tensor to float [oc][ic]. F16/F32 memcpy path; else ggml to_float dequant
// (Q4_K_M etc.) so force can run on the same GGUFs shadow already matches.
static bool pack_weight_tensor(
    const struct ggml_tensor * w, int ic, int oc, float * dst_oc_ic) {
    if (!w || !w->data || !dst_oc_ic || ic <= 0 || oc <= 0) {
        return false;
    }
    if (w->ne[0] != ic || w->ne[1] != oc || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (w->type == GGML_TYPE_F16) {
        return ane_ffn_pack_weight_to_f32(w->data, 1, ic, oc, dst_oc_ic);
    }
    if (w->type == GGML_TYPE_F32) {
        return ane_ffn_pack_weight_to_f32(w->data, 0, ic, oc, dst_oc_ic);
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(w->type);
    if (!tr || !tr->to_float || !tr->is_quantized) {
        return false;
    }
    const int64_t blck = ggml_blck_size(w->type);
    if (blck <= 0 || (ic % (int)blck) != 0) {
        return false;
    }
    for (int r = 0; r < oc; r++) {
        const void * row = (const char *)w->data + (size_t)r * w->nb[1];
        tr->to_float(row, dst_oc_ic + (size_t)r * (size_t)ic, ic);
    }
    return true;
}

static bool wcache_ensure(
    const ane_ffn_swiglu_fuse_t * fuse,
    float **out_Wg, float **out_Wu, float **out_Wd) {
    const void * wg = fuse->gate->src[0]->data;
    const void * wu = fuse->up->src[0]->data;
    const void * wd = fuse->down->src[0]->data;
    const void * su = scale_data(fuse->up_scale);
    const void * sg = scale_data(fuse->gate_scale);
    const void * sd = scale_data(fuse->down_scale);
    const int ic = fuse->ic;
    const int hidden = fuse->hidden;
    const char *wname = fuse->up->src[0] ? fuse->up->src[0]->name : NULL;
    const int nslots = wcache_slots_configured();

    int hit_i = -1;
    for (int i = 0; i < nslots; i++) {
        SwigluWeightCache *s = &g_wslots[i];
        if (s->Wg &&
            s->wg_src == wg && s->wu_src == wu && s->wd_src == wd &&
            s->su_src == su && s->sg_src == sg && s->sd_src == sd &&
            s->ic == ic && s->hidden == hidden) {
            hit_i = i;
            break;
        }
    }
    if (hit_i >= 0) {
        g_wslots[hit_i].tick = ++g_wclock;
        *out_Wg = g_wslots[hit_i].Wg;
        *out_Wu = g_wslots[hit_i].Wu;
        *out_Wd = g_wslots[hit_i].Wd;
        ane_ffn_force_note_wcache(1, nslots, ic, hidden, wname);
        return true;
    }

    // Miss: prefer empty slot, else LRU.
    int victim = -1;
    for (int i = 0; i < nslots; i++) {
        if (!g_wslots[i].Wg) {
            victim = i;
            break;
        }
    }
    if (victim < 0) {
        victim = 0;
        for (int i = 1; i < nslots; i++) {
            if (g_wslots[i].tick < g_wslots[victim].tick) {
                victim = i;
            }
        }
    }

    SwigluWeightCache *s = &g_wslots[victim];
    wcache_slot_clear(s);
    float *Wg = (float *)malloc((size_t)hidden * (size_t)ic * sizeof(float));
    float *Wu = (float *)malloc((size_t)hidden * (size_t)ic * sizeof(float));
    float *Wd = (float *)malloc((size_t)ic * (size_t)hidden * sizeof(float));
    if (!Wg || !Wu || !Wd) {
        free(Wg); free(Wu); free(Wd);
        return false;
    }
    if (!pack_weight_tensor(fuse->gate->src[0], ic, hidden, Wg) ||
        !pack_weight_tensor(fuse->up->src[0], ic, hidden, Wu) ||
        !pack_weight_tensor(fuse->down->src[0], hidden, ic, Wd) ||
        !fold_scale(Wu, ic, hidden, fuse->up_scale) ||
        !fold_scale(Wg, ic, hidden, fuse->gate_scale) ||
        !fold_scale(Wd, hidden, ic, fuse->down_scale)) {
        free(Wg); free(Wu); free(Wd);
        return false;
    }
    s->wg_src = wg;
    s->wu_src = wu;
    s->wd_src = wd;
    s->su_src = su;
    s->sg_src = sg;
    s->sd_src = sd;
    s->ic = ic;
    s->hidden = hidden;
    s->Wg = Wg;
    s->Wu = Wu;
    s->Wd = Wd;
    s->tick = ++g_wclock;
    *out_Wg = Wg;
    *out_Wu = Wu;
    *out_Wd = Wd;
    ane_ffn_force_note_wcache(0, nslots, ic, hidden, wname);
    return true;
}

bool ane_ffn_force_try_swiglu_tensors(const ane_ffn_swiglu_fuse_t * fuse) {
    if (!fuse || !fuse->up || !fuse->gate || !fuse->down || !fuse->dst || fuse->n_fuse < 4) {
        ane_ffn_force_note_bail("bad_fuse", 0, 0, 0, NULL);
        return false;
    }
    const struct ggml_tensor * up = fuse->up;
    struct ggml_tensor * dst = (struct ggml_tensor *)fuse->dst;
    const char *wname = up->src[0] ? up->src[0]->name : NULL;
    const int ic0 = fuse->ic;
    const int hid0 = fuse->hidden;
    const int seq0 = fuse->seq;

    if (!tensor_host_shared(up->src[0])) {
        ane_ffn_force_note_bail("shared_Wu", ic0, hid0, seq0, wname);
        return false;
    }
    if (!tensor_host_shared(fuse->gate->src[0])) {
        ane_ffn_force_note_bail("shared_Wg", ic0, hid0, seq0, wname);
        return false;
    }
    if (!tensor_host_shared(fuse->down->src[0])) {
        ane_ffn_force_note_bail("shared_Wd", ic0, hid0, seq0, wname);
        return false;
    }
    if (!tensor_host_shared(up->src[1])) {
        ane_ffn_force_note_bail("shared_X", ic0, hid0, seq0, wname);
        return false;
    }
    if (!tensor_host_shared(dst)) {
        ane_ffn_force_note_bail("shared_Y", ic0, hid0, seq0, wname);
        return false;
    }
    if (fuse->up_scale && !tensor_host_shared(fuse->up_scale->src[1])) {
        ane_ffn_force_note_bail("shared_up_s", ic0, hid0, seq0, wname);
        return false;
    }
    if (fuse->gate_scale && !tensor_host_shared(fuse->gate_scale->src[1])) {
        ane_ffn_force_note_bail("shared_gate_s", ic0, hid0, seq0, wname);
        return false;
    }
    if (fuse->down_scale && !tensor_host_shared(fuse->down_scale->src[1])) {
        ane_ffn_force_note_bail("shared_down_s", ic0, hid0, seq0, wname);
        return false;
    }

    const enum ggml_type ta = up->src[1]->type;
    // Acts must be F16/F32 (host pack). Weights may be mixed quant (Q4_K_M: Q4_K
    // up/gate + Q6_K down on some layers) — pack_weight_tensor dequants each.
    if ((ta != GGML_TYPE_F16 && ta != GGML_TYPE_F32) || dst->type != ta) {
        ane_ffn_force_note_bail("act_type", ic0, hid0, seq0, wname);
        return false;
    }
    {
        const struct ggml_tensor * ws[3] = {
            up->src[0], fuse->gate->src[0], fuse->down->src[0],
        };
        for (int wi = 0; wi < 3; wi++) {
            const enum ggml_type tw = ws[wi]->type;
            if (tw == GGML_TYPE_F16 || tw == GGML_TYPE_F32) {
                continue;
            }
            const struct ggml_type_traits * tr = ggml_get_type_traits(tw);
            if (!tr || !tr->to_float || !tr->is_quantized) {
                ane_ffn_force_note_bail("no_dequant", ic0, hid0, seq0, wname);
                return false;
            }
        }
    }

    const int ic     = fuse->ic;
    const int hidden = fuse->hidden;
    const int seq    = fuse->seq;
    const int a_f16  = type_is_f16(up->src[1]->type);
    const bool want_int8_in = env_on("ZEROLLAMA_ANE_FFN_INT8_IN");

    float *Wg = NULL;
    float *Wu = NULL;
    float *Wd = NULL;
    if (!wcache_ensure(fuse, &Wg, &Wu, &Wd)) {
        ane_ffn_force_note_bail("wcache", ic, hidden, seq, wname);
        return false;
    }
    if (!acache_ensure(ic, seq)) {
        ane_ffn_force_note_bail("acache", ic, hidden, seq, wname);
        return false;
    }

    // Stable scache identity (ggml weight data) — required for MoE multi-layer.
    ane_ffn_force_swiglu_bind_weight_ids(
        fuse->gate->src[0]->data, fuse->up->src[0]->data, fuse->down->src[0]->data);

    // INT8_IN: create session per weight id via fp16 path, then Metal/host i8.
    if (want_int8_in) {
        const bool have_sess = ane_ffn_force_swiglu_activate_session(
            ic, hidden, seq, Wg, Wu, Wd);
        if (!have_sess) {
            // Miss: create INT8_IN session (fp16 acts → host quant in dylib).
            if (!ane_ffn_pack_acts_ggml_to_channel_f16(
                    up->src[1]->data, a_f16, ic, seq, g_acache.X16)) {
                return false;
            }
            if (!ane_ffn_force_try_swiglu_fp16(
                    ic, hidden, seq, Wg, Wu, Wd,
                    g_acache.X16, g_acache.Y16)) {
                return false;
            }
            return ane_ffn_unpack_dst_channel_f16_to_ggml(
                g_acache.Y16, a_f16, ic, seq, dst->data);
        }
        // Prefer Metal pack→eval→unpack into ggml layout (F0743).
        if (a_f16) {
            if (ane_ffn_force_try_swiglu_metal_layout(
                    ic, hidden, seq, Wg, Wu, Wd,
                    up->src[1]->data, 1, dst->data)) {
                return true;
            }
        } else {
            // Metal writes ggml-f16 into Y16; widen to f32 dst [seq][ic].
            if (ane_ffn_force_try_swiglu_metal_layout(
                    ic, hidden, seq, Wg, Wu, Wd,
                    up->src[1]->data, 0, g_acache.Y16)) {
                const uint16_t *s = g_acache.Y16;
                float *d = (float *)dst->data;
                const size_t n = (size_t)ic * (size_t)seq;
                for (size_t i = 0; i < n; i++) {
                    // Soft f16→f32 (same layout ggml).
                    uint16_t h = s[i];
                    const uint32_t sign = (uint32_t)(h >> 15) << 31;
                    const uint32_t exp  = (h >> 10) & 0x1f;
                    const uint32_t mant = h & 0x3ff;
                    uint32_t out;
                    if (exp == 0) {
                        out = sign;
                    } else if (exp == 31) {
                        out = sign | 0x7f800000u | (mant << 13);
                    } else {
                        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                    }
                    union { uint32_t u; float f; } u = { .u = out };
                    d[i] = u.f;
                }
                return true;
            }
        }
        float xscale = ane_ffn_force_query_x_scale();
        if (!ane_ffn_pack_acts_ggml_to_channel_i8(
                up->src[1]->data, a_f16, ic, seq, xscale, g_acache.Xi8)) {
            return false;
        }
        if (ane_ffn_force_try_swiglu_int8_fp16(
                ic, hidden, seq, Wg, Wu, Wd,
                g_acache.Xi8, g_acache.Y16)) {
            return ane_ffn_unpack_dst_channel_f16_to_ggml(
                g_acache.Y16, a_f16, ic, seq, dst->data);
        }
        // Fall back to fp16 acts path below.
    }

    {
        const double t0 = ane_ffn_profile_now_ms();
        if (!ane_ffn_pack_acts_ggml_to_channel_f16(
                up->src[1]->data, a_f16, ic, seq, g_acache.X16)) {
            ane_ffn_force_note_bail("pack_acts", ic, hidden, seq, wname);
            return false;
        }
        ane_ffn_profile_add_ms("pack", ane_ffn_profile_now_ms() - t0);
    }

    {
        const double t0 = ane_ffn_profile_now_ms();
        const bool ok = ane_ffn_force_try_swiglu_fp16(
            ic, hidden, seq, Wg, Wu, Wd,
            g_acache.X16, g_acache.Y16);
        ane_ffn_profile_add_ms("dylib", ane_ffn_profile_now_ms() - t0);
        if (ok) {
            const double t1 = ane_ffn_profile_now_ms();
            if (!ane_ffn_unpack_dst_channel_f16_to_ggml(
                    g_acache.Y16, a_f16, ic, seq, dst->data)) {
                ane_ffn_force_note_bail("unpack", ic, hidden, seq, wname);
                return false;
            }
            ane_ffn_profile_add_ms("unpack", ane_ffn_profile_now_ms() - t1);
            ane_ffn_profile_tick_replace();
            return true;
        }
    }

    // Legacy f32 host path (smoke / older dylib without fp16 entry).
    float *X = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
    float *Y = (float *)malloc((size_t)ic * (size_t)seq * sizeof(float));
    if (!X || !Y) {
        free(X); free(Y);
        ane_ffn_force_note_bail("oom_xy", ic, hidden, seq, wname);
        return false;
    }
    bool ok = ane_ffn_pack_acts_ggml_to_channel(up->src[1]->data, a_f16, ic, seq, X)
           && ane_ffn_force_try_swiglu_host(
                  ic, hidden, seq, Wg, Wu, Wd, X, Y);
    if (ok) {
        ok = ane_ffn_unpack_dst_channel_to_ggml(Y, a_f16, ic, seq, dst->data);
        if (!ok) {
            ane_ffn_force_note_bail("unpack_f32", ic, hidden, seq, wname);
        }
    } else {
        ane_ffn_force_note_bail("host_or_fp16_fail", ic, hidden, seq, wname);
    }
    free(X); free(Y);
    return ok;
}
