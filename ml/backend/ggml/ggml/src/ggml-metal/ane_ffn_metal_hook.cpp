// Lab ANE FFN Metal mul_mat intercept (separate TU — see ane_ffn_metal_hook.h).
#include "ane_ffn_metal_hook.h"

#include "ane_ffn_policy.h"
#include "ane_ffn_swiglu_fuse.h"

#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ane_ffn_holey_slot {
    ane_ffn_swiglu_fuse_t fuse;
    bool live;
    bool async_inflight;
};
static ane_ffn_holey_slot g_ane_ffn_holey[64];

static bool ane_ffn_overlap_want(void) {
    const char * e = getenv("ZEROLLAMA_ANE_FFN_OVERLAP");
    if (!e || !e[0] || e[0] == '0') {
        return false;
    }
    if (e[0] == 'f' || e[0] == 'F') {
        return false;
    }
    return true;
}

int ane_ffn_metal_op_mul_mat_try(
    ggml_metal_op_t ctx, int idx, ggml_tensor * op, int ic, int oc, int seq) {
    const char *v = getenv("ZEROLLAMA_ANE_FFN");
    if (!v || v[0] != '1') {
        return 0;
    }
    if (!ctx || !op) {
        return 0;
    }

    const char *wname = op->src[0] ? op->src[0]->name : NULL;
    const bool swiglu_on = getenv("ZEROLLAMA_ANE_FFN_SWIGLU") &&
        getenv("ZEROLLAMA_ANE_FFN_SWIGLU")[0] == '1';
    const bool is_swiglu_w = ane_ffn_name_is_ffn_swiglu_weight(wname);
    const bool fuse_anchor =
        ane_ffn_name_is_ffn_up(wname) || ane_ffn_name_is_ffn_gate(wname);
    const bool overlap_on = ane_ffn_overlap_want();

    if (swiglu_on && ane_ffn_name_is_ffn_down(wname)) {
        for (int pi = 0; pi < 64; ++pi) {
            if (!g_ane_ffn_holey[pi].live) {
                continue;
            }
            ane_ffn_swiglu_fuse_t & fuse = g_ane_ffn_holey[pi].fuse;
            if (fuse.down != op && fuse.dst != op) {
                continue;
            }
            if (!ane_ffn_force_want_try(fuse.ic, fuse.hidden, fuse.seq, wname)) {
                g_ane_ffn_holey[pi].live = false;
                g_ane_ffn_holey[pi].async_inflight = false;
                break;
            }
            ane_ffn_force_autoload_host_replace();
            bool ok = false;
            if (g_ane_ffn_holey[pi].async_inflight) {
                ok = ane_ffn_force_try_swiglu_tensors_finish(&fuse);
                g_ane_ffn_holey[pi].async_inflight = false;
                if (!ok) {
                    // Never fall through to Metal-down-only after holey skip.
                    const bool skip_sync = getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST") &&
                        getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST")[0] == '1';
                    bool host_ready = skip_sync || ggml_metal_op_sync_for_ane_host(ctx);
                    if (host_ready) {
                        ok = ane_ffn_force_try_swiglu_tensors(&fuse);
                    }
                    if (!ok) {
                        ane_ffn_force_note_bail(
                            "overlap_fallback", fuse.ic, fuse.hidden, fuse.seq, wname);
                    }
                }
            } else {
                const bool skip_sync = getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST") &&
                    getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST")[0] == '1';
                bool host_ready = skip_sync || ggml_metal_op_sync_for_ane_host(ctx);
                if (!host_ready) {
                    ane_ffn_force_note_bail(
                        "sync_deferred", fuse.ic, fuse.hidden, fuse.seq, wname);
                    g_ane_ffn_holey[pi].live = false;
                    break;
                }
                ok = ane_ffn_force_try_swiglu_tensors(&fuse);
            }
            if (ok) {
                g_ane_ffn_holey[pi].live = false;
                if (fuse.down_scale && fuse.dst == fuse.down_scale) {
                    const_cast<ggml_tensor *>(fuse.down_scale)->flags &=
                        ~GGML_TENSOR_FLAG_COMPUTE;
                }
                return 1;
            }
            ane_ffn_force_note_bail(
                "deferred_replace", fuse.ic, fuse.hidden, fuse.seq, wname);
            g_ane_ffn_holey[pi].live = false;
            break;
        }
    }

    if (swiglu_on && fuse_anchor && idx + 3 < ggml_metal_op_n_nodes(ctx)) {
        const int n_avail = ggml_metal_op_n_nodes(ctx) - idx;
        const int n_look = n_avail > 48 ? 48 : n_avail;
        const ggml_tensor * nodes[48];
        for (int k = 0; k < n_look; ++k) {
            nodes[k] = ggml_metal_op_node(ctx, idx + k);
        }
        ane_ffn_swiglu_fuse_t fuse;
        if (ane_ffn_swiglu_fuse_match(nodes, n_look, &fuse)) {
            if (ane_ffn_force_want_try(ic, oc, seq, wname)) {
                ane_ffn_force_autoload_host_replace();
                const bool holey_off = fuse.holey && getenv("ZEROLLAMA_ANE_FFN_HOLEY") &&
                    getenv("ZEROLLAMA_ANE_FFN_HOLEY")[0] == '0';
                if (holey_off) {
                    ane_ffn_force_note_bail(
                        "holey_disabled", fuse.ic, fuse.hidden, fuse.seq, wname);
                } else if (fuse.holey) {
                    int slot = -1;
                    for (int pi = 0; pi < 64; ++pi) {
                        if (!g_ane_ffn_holey[pi].live) {
                            slot = pi;
                            break;
                        }
                    }
                    if (slot < 0) {
                        ane_ffn_force_note_bail(
                            "holey_full", fuse.ic, fuse.hidden, fuse.seq, wname);
                    } else {
                        bool kicked = false;
                        if (overlap_on) {
                            const bool skip_sync = getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST") &&
                                getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST")[0] == '1';
                            bool host_ready = skip_sync || ggml_metal_op_sync_for_ane_host(ctx);
                            if (!host_ready) {
                                ane_ffn_force_note_bail(
                                    "sync_overlap", fuse.ic, fuse.hidden, fuse.seq, wname);
                            } else {
                                kicked = ane_ffn_force_try_swiglu_tensors_kick(&fuse);
                                if (!kicked) {
                                    ane_ffn_force_note_bail(
                                        "overlap_kick", fuse.ic, fuse.hidden, fuse.seq, wname);
                                }
                            }
                        }
                        g_ane_ffn_holey[slot].fuse = fuse;
                        g_ane_ffn_holey[slot].live = true;
                        g_ane_ffn_holey[slot].async_inflight = kicked;
                        if (fuse.glu) {
                            const_cast<ggml_tensor *>(fuse.glu)->flags &=
                                ~GGML_TENSOR_FLAG_COMPUTE;
                        }
                        return fuse.n_encode_skip > 0 ? fuse.n_encode_skip : 2;
                    }
                } else {
                    const bool skip_sync = getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST") &&
                        getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST")[0] == '1';
                    bool host_ready = skip_sync || ggml_metal_op_sync_for_ane_host(ctx);
                    if (!host_ready) {
                        ane_ffn_force_note_bail(
                            "sync", fuse.ic, fuse.hidden, fuse.seq, wname);
                    } else if (ane_ffn_force_try_swiglu_tensors(&fuse)) {
                        return fuse.n_encode_skip > 0 ? fuse.n_encode_skip : fuse.n_fuse;
                    }
                }
            } else {
                ane_ffn_force_note_bail(
                    "policy", fuse.ic, fuse.hidden, fuse.seq, wname);
            }
            ane_ffn_shadow_note_swiglu_fuse(fuse.ic, fuse.hidden, fuse.seq, wname);
        } else if (getenv("ZEROLLAMA_ANE_FFN_TELEMETRY") &&
                   getenv("ZEROLLAMA_ANE_FFN_TELEMETRY")[0] == '1') {
            static std::atomic<uint64_t> g_fuse_miss{0};
            uint64_t n = g_fuse_miss.fetch_add(1) + 1;
            if (n <= 8 || (n % 1024ull) == 0) {
                char opsbuf[384];
                size_t o = 0;
                opsbuf[0] = '\0';
                const int n_show = n_look > 16 ? 16 : n_look;
                for (int k = 0; k < n_show && o + 24 < sizeof(opsbuf); ++k) {
                    const char * on = nodes[k] ? ggml_op_name(nodes[k]->op) : "?";
                    const char * wn = (nodes[k] && nodes[k]->op == GGML_OP_MUL_MAT &&
                                      nodes[k]->src[0] && nodes[k]->src[0]->name[0])
                        ? nodes[k]->src[0]->name : "";
                    int w = snprintf(opsbuf + (int)o, sizeof(opsbuf) - o, "%s%s%s%s",
                                     k ? "," : "", on,
                                     wn[0] ? ":" : "", wn[0] ? wn : "");
                    if (w > 0) {
                        o += (size_t)w;
                    }
                }
                fprintf(stderr,
                        "ane_ffn_force: fuse_miss#%llu ic=%d oc=%d seq=%d name=%s look=%d [%s]\n",
                        (unsigned long long)n, ic, oc, seq, wname ? wname : "",
                        n_look, opsbuf);
            }
        }
    }

    if (!(swiglu_on && is_swiglu_w) && ane_ffn_force_want_try(ic, oc, seq, wname)) {
        ane_ffn_force_autoload_host_replace();
        const bool have_replace = ane_ffn_force_get_host_replace() != NULL;
        const bool skip_sync = getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST") &&
            getenv("ZEROLLAMA_ANE_FFN_FORCE_HOST")[0] == '1';
        bool host_ready = skip_sync;
        if (have_replace && !skip_sync) {
            host_ready = ggml_metal_op_sync_for_ane_host(ctx);
        }
        if (have_replace && host_ready &&
            ane_ffn_force_try_mul_mat_tensors(op->src[0], op->src[1], op)) {
            return 1;
        }
        (void) ane_ffn_force_try_mul_mat(ic, oc, seq, wname);
    } else if (!(swiglu_on && is_swiglu_w) &&
               ane_ffn_force_try_mul_mat(ic, oc, seq, wname)) {
        return 1;
    }
    ane_ffn_shadow_note_mul_mat(ic, oc, seq, wname);
    return 0;
}
