// ANE dflash draft hook — packs Metal draft activations into in-process ANE IOSurface (lab).
// See docs/ane-draft-inprocess.md. Env-gated: ZEROLLAMA_ANE_DRAFT=1 (default off).
#include "ane_draft_hook.h"

#include "ane_draft_session.h"
#include "log.h"

#include "ggml-backend.h"
#include "llama.h"
#include "../src/llama-ext.h"

#if defined(__APPLE__)
#include "ggml-metal.h"
#endif

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static bool env_truthy(const char * v) {
    if (!v || !v[0]) {
        return false;
    }
    return strcmp(v, "1") == 0 ||
           strcmp(v, "true") == 0 ||
           strcmp(v, "yes") == 0 ||
           strcmp(v, "on") == 0;
}

static int env_int_or(const char * name, int def) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) {
        return def;
    }
    return std::atoi(v);
}

static bool golden_conv_enabled(int conv_index) {
    const int cap = env_int_or("ZEROLLAMA_ANE_DRAFT_CONV_DEPTH", 0);
    if (cap <= 0) {
        return true;
    }
    return conv_index <= cap;
}

bool common_ane_draft_enabled() {
    return env_truthy(std::getenv("ZEROLLAMA_ANE_DRAFT"));
}

static bool ane_draft_telemetry_enabled() {
    return env_truthy(std::getenv("ZEROLLAMA_ANE_DRAFT_TELEMETRY"));
}

static int ane_handoff_stride() {
    const int s = env_int_or("ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE", 1);
    return s > 0 ? s : 1;
}

common_ane_draft_drive_mode common_ane_draft_get_drive_mode() {
    const char * v = std::getenv("ZEROLLAMA_ANE_DRAFT_DRIVE");
    if (!v || !v[0]) {
        return COMMON_ANE_DRAFT_DRIVE_OFF;
    }
    if (strcmp(v, "shadow") == 0) {
        return COMMON_ANE_DRAFT_DRIVE_SHADOW;
    }
    if (strcmp(v, "force") == 0 || env_truthy(v)) {
        return COMMON_ANE_DRAFT_DRIVE_FORCE;
    }
    return COMMON_ANE_DRAFT_DRIVE_OFF;
}

#if defined(__APPLE__)
static float fp16_bits_to_float32(uint16_t h);
static bool load_gamma_scales(const char * path, int channels, std::vector<float> & out);
static bool load_matmul_golden_weights(const char * path, int ic, int oc, std::vector<float> & W);
static void matmul_golden_reference(const float * input, int ic, int oc,
                                     const std::vector<float> & W, std::vector<float> & ref);
static float ane_silu_host(float x);
static void ane_stash_handoff_hidden(const float * h, int n);
static const float * ane_handoff_hidden_for_golden(struct llama_context * ctx_dft, int32_t i_batch, int * out_len);
static bool ane_fill_golden_input(std::vector<float> & inp, int ic, struct llama_context * ctx_dft, int32_t i_batch);

static void log_ane_golden_telemetry(const float * input, int ch, int sp, int step);
static void ane_handoff_eval_done(bool ok);

static std::atomic<bool> g_ane_output_ready { false };
static std::atomic<int>  g_handoff_step { 0 };
static std::vector<float> g_last_handoff_hidden;
static std::vector<float> g_async_golden_emb;
static int g_async_golden_step = 0;
static bool g_async_golden_telemetry = false;
#endif

void common_ane_draft_reset_handoff(void) {
#if defined(__APPLE__)
    if (ane_draft_session_eval_async_enabled()) {
        ane_draft_session_eval_sync();
    }
    g_ane_output_ready.store(false);
    g_handoff_step.store(0);
    g_last_handoff_hidden.clear();
    g_async_golden_emb.clear();
    g_async_golden_step = 0;
    g_async_golden_telemetry = false;
#endif
}

#if defined(__APPLE__)
struct ane_drive_head {
    const uint16_t * embed_fp16 = nullptr;
    const float    * out_norm   = nullptr;
    size_t           embed_bytes = 0;
    int              n_embd      = 0;
    int              n_vocab     = 0;
    void *           embed_map   = nullptr;
    size_t           embed_map_sz = 0;
    std::vector<float> out_norm_owned;
    bool             ok = false;
};

static ane_drive_head g_drive_head;

static bool ane_drive_metrics_hidden_only(void) {
    if (common_ane_draft_get_drive_mode() != COMMON_ANE_DRAFT_DRIVE_SHADOW) {
        return false;
    }
    const char * m = std::getenv("ZEROLLAMA_ANE_DRAFT_DRIVE_METRICS");
    if (m && m[0]) {
        if (strcmp(m, "tokens") == 0 || strcmp(m, "both") == 0) {
            return false;
        }
        if (strcmp(m, "hidden") == 0) {
            return true;
        }
        return env_truthy(m);
    }
    if (ane_draft_session_matmul_active()) {
        return true;
    }
    const char * km = std::getenv("ZEROLLAMA_ANE_DRAFT_KERNEL");
    return km && strcmp(km, "matmul") == 0;
}

static bool drive_head_load(const char * embd_path, const char * norm_path) {
    if (!embd_path || !embd_path[0]) {
        LOG_WRN("%s: TOKEN_EMBD_FILE missing\n", __func__);
        return false;
    }
    if (g_drive_head.ok) {
        return true;
    }

    const int n_embd  = env_int_or("ZEROLLAMA_ANE_DRAFT_N_EMBD", 0);
    const int n_vocab = env_int_or("ZEROLLAMA_ANE_DRAFT_N_VOCAB", 0);
    if (n_embd <= 0 || n_vocab <= 0) {
        LOG_WRN("%s: N_EMBD/N_VOCAB env missing (need ZEROLLAMA_ANE_DRAFT_N_EMBD/N_VOCAB)\n", __func__);
        return false;
    }

    const int fd = open(embd_path, O_RDONLY);
    if (fd < 0) {
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 16) {
        close(fd);
        return false;
    }
    void * map = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        return false;
    }
    const uint8_t * hdr = (const uint8_t *) map;
    if (memcmp(hdr, "ZANE1", 5) != 0) {
        LOG_WRN("%s: embed magic mismatch\n", __func__);
        munmap(map, (size_t) st.st_size);
        return false;
    }
    const size_t payload = (size_t) st.st_size - 16;
    const size_t need = (size_t) n_embd * (size_t) n_vocab * 2;
    if (payload < need) {
        LOG_WRN("%s: embed size %zu < need %zu\n", __func__, payload, need);
        munmap(map, (size_t) st.st_size);
        return false;
    }

    g_drive_head.embed_map    = map;
    g_drive_head.embed_map_sz = (size_t) st.st_size;
    g_drive_head.embed_fp16   = (const uint16_t *) (hdr + 16);
    g_drive_head.embed_bytes  = need;
    g_drive_head.n_embd       = n_embd;
    g_drive_head.n_vocab      = n_vocab;

    if (norm_path && norm_path[0]) {
        FILE * nf = std::fopen(norm_path, "rb");
        if (nf) {
            g_drive_head.out_norm_owned.resize((size_t) n_embd);
            const size_t nread = std::fread(g_drive_head.out_norm_owned.data(), sizeof(float), (size_t) n_embd, nf);
            std::fclose(nf);
            if (nread == (size_t) n_embd) {
                g_drive_head.out_norm = g_drive_head.out_norm_owned.data();
            }
        }
    }

    g_drive_head.ok = true;
    LOG_INF("%s: B7 drive head loaded n_embd=%d n_vocab=%d embed=%s norm=%s\n",
            __func__, n_embd, n_vocab, embd_path, norm_path && norm_path[0] ? norm_path : "(none)");
    return true;
}

static void drive_apply_rms_norm(std::vector<float> & h, const float * norm_w, float eps) {
    double sumsq = 0.0;
    for (float v : h) {
        sumsq += (double) v * (double) v;
    }
    const float rms = (float) std::sqrt(sumsq / (double) h.size() + (double) eps);
    if (rms <= 0.f) {
        return;
    }
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = h[i] / rms * (norm_w ? norm_w[i] : 1.f);
    }
}

static llama_token drive_argmax_tied(const std::vector<float> & h) {
    if (!g_drive_head.ok || !g_drive_head.embed_fp16) {
        return 0;
    }
    const int n_embd  = g_drive_head.n_embd;
    const int n_vocab = g_drive_head.n_vocab;
    const uint16_t * W = g_drive_head.embed_fp16;

    llama_token best_id = 0;
    double best_logit = -INFINITY;

    int n_vocab_scan = n_vocab;
    const int cap_default = (common_ane_draft_get_drive_mode() != COMMON_ANE_DRAFT_DRIVE_OFF) ? 8192 : 0;
    if (const int cap = env_int_or("ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP", cap_default); cap > 0 && cap < n_vocab_scan) {
        n_vocab_scan = cap;
    }

    constexpr int k_chunk = 4096;
    for (int v0 = 0; v0 < n_vocab_scan; v0 += k_chunk) {
        const int vend = v0 + k_chunk < n_vocab_scan ? v0 + k_chunk : n_vocab_scan;
        for (int v = v0; v < vend; ++v) {
            double dot = 0.0;
            for (int e = 0; e < n_embd; ++e) {
                const uint16_t bits = W[(size_t) e * (size_t) n_vocab + (size_t) v];
                dot += (double) h[(size_t) e] * (double) fp16_bits_to_float32(bits);
            }
            if (dot > best_logit) {
                best_logit = dot;
                best_id = (llama_token) v;
            }
        }
    }
    return best_id;
}

bool common_ane_draft_try_drive_token(struct llama_context * ctx_dft, int32_t i_batch, llama_token * out_id, float * out_p, float * out_hidden_cos) {
    if (!out_id || !out_p || !common_ane_draft_enabled()) {
        return false;
    }
    if (common_ane_draft_get_drive_mode() == COMMON_ANE_DRAFT_DRIVE_OFF) {
        return false;
    }
    if (!g_ane_output_ready.load() || !ane_draft_session_ready()) {
        LOG_DBG("%s: ANE output not ready\n", __func__);
        return false;
    }

    if (ane_draft_session_eval_async_enabled()) {
        ane_draft_session_eval_sync();
    }

    if (!g_ane_output_ready.load()) {
        LOG_DBG("%s: ANE output not ready after eval sync\n", __func__);
        return false;
    }

    const bool metrics_only = ane_drive_metrics_hidden_only();
    const bool matmul = ane_draft_session_matmul_active();

    if (!metrics_only) {
        const char * embd_path = std::getenv("ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE");
        const char * norm_path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
        if (!drive_head_load(embd_path, norm_path)) {
            return false;
        }
    }

    const int oc_out = ane_draft_session_output_channels();
    const int sp = ane_draft_session_spatial();
    const int ic = matmul ? ane_draft_session_channels() : oc_out;
    const int chain_depth = matmul ? ane_draft_session_matmul_chain_depth() : 0;

    std::vector<float> gate;
    int oc = oc_out;
    if (matmul && chain_depth >= 4) {
        const int ffn_embd = ane_draft_session_matmul_ffn_embd();
        if (ffn_embd <= 0) {
            return false;
        }
        gate.assign((size_t) ffn_embd, 0.f);
        if (ane_draft_session_read_ffn_down(gate.data(), gate.size()) == 0) {
            return false;
        }
        oc = ffn_embd;
    } else {
        const size_t nfloats = (size_t) oc_out * (size_t) sp;
        std::vector<float> ane_out(nfloats);
        if (ane_draft_session_read_output(ane_out.data(), nfloats) == 0) {
            return false;
        }
        gate.assign((size_t) oc_out, 0.f);
        for (int o = 0; o < oc_out; ++o) {
            double sum = 0.0;
            for (int s = 0; s < sp; ++s) {
                sum += (double) ane_out[(size_t) o * (size_t) sp + (size_t) s];
            }
            gate[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
        }
        oc = oc_out;
    }

    if (metrics_only) {
        if (out_hidden_cos && ctx_dft && i_batch >= 0 && matmul) {
            *out_hidden_cos = 0.f;
            const int oc_gate = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_out);
            const int oc_ffn  = (chain_depth >= 4) ? ane_draft_session_matmul_ffn_embd() : oc;
            std::vector<float> inp;
            if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                if (ane_draft_session_matmul_chain_depth() >= 3) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        chain_ok.store(w1 && w2 && w3 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down));
                    }
                    if (chain_ok.load()) {
                        std::vector<float> gvec;
                        std::vector<float> uvec;
                        matmul_golden_reference(inp.data(), ic, oc_gate, w_gate, gvec);
                        matmul_golden_reference(inp.data(), ic, oc_gate, w_up, uvec);
                        for (int i = 0; i < oc_gate; ++i) {
                            gvec[(size_t) i] = ane_silu_host(gvec[(size_t) i]) * uvec[(size_t) i];
                        }
                        std::vector<float> down_ref;
                        matmul_golden_reference(gvec.data(), oc_gate, oc_ffn, w_down, down_ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc; ++o) {
                            const double r = (double) down_ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 2) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        chain_ok.store(w1 && w2 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, oc_gate, oc, w_up));
                    }
                    if (chain_ok.load()) {
                        std::vector<float> gref;
                        matmul_golden_reference(inp.data(), ic, oc_gate, w_gate, gref);
                        for (float & v : gref) {
                            v = ane_silu_host(v);
                        }
                        std::vector<float> up_ref;
                        matmul_golden_reference(gref.data(), oc_gate, oc, w_up, up_ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc; ++o) {
                            const double r = (double) up_ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                } else {
                    static std::vector<float> golden_w;
                    static std::atomic<bool> golden_loaded { false };
                    static std::atomic<bool> golden_ok { false };
                    if (!golden_loaded.exchange(true)) {
                        const char * wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        golden_ok.store(load_matmul_golden_weights(wpath, ic, oc_gate, golden_w));
                    }
                    if (golden_ok.load()) {
                        std::vector<float> ref;
                        matmul_golden_reference(inp.data(), ic, oc_gate, golden_w, ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc_gate; ++o) {
                            const double r = (double) ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                }
            }
        }
        *out_id = 0;
        *out_p  = 0.f;
        return true;
    }

    const int n_embd = g_drive_head.n_embd;
    std::vector<float> h((size_t) n_embd, 0.f);
    for (int i = 0; i < oc && i < n_embd; ++i) {
        h[(size_t) i] = gate[(size_t) i];
    }

    const float eps = 1e-6f;
    drive_apply_rms_norm(h, g_drive_head.out_norm, eps);

    if (out_hidden_cos && ctx_dft && i_batch >= 0) {
        *out_hidden_cos = 0.f;
        if (matmul) {
            const int oc_gate = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_out);
            const int oc_ffn  = (chain_depth >= 4) ? ane_draft_session_matmul_ffn_embd() : oc;
            std::vector<float> inp;
            if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                if (ane_draft_session_matmul_chain_depth() >= 3) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        chain_ok.store(w1 && w2 && w3 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down));
                    }
                    if (chain_ok.load()) {
                        std::vector<float> gvec;
                        std::vector<float> uvec;
                        matmul_golden_reference(inp.data(), ic, oc_gate, w_gate, gvec);
                        matmul_golden_reference(inp.data(), ic, oc_gate, w_up, uvec);
                        for (int i = 0; i < oc_gate; ++i) {
                            gvec[(size_t) i] = ane_silu_host(gvec[(size_t) i]) * uvec[(size_t) i];
                        }
                        std::vector<float> down_ref;
                        matmul_golden_reference(gvec.data(), oc_gate, oc_ffn, w_down, down_ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc; ++o) {
                            const double r = (double) down_ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                } else {
                    static std::vector<float> golden_w;
                    static std::atomic<bool> golden_loaded { false };
                    static std::atomic<bool> golden_ok { false };
                    if (!golden_loaded.exchange(true)) {
                        const char * wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        golden_ok.store(load_matmul_golden_weights(wpath, ic, oc_gate, golden_w));
                    }
                    if (golden_ok.load()) {
                        std::vector<float> ref;
                        matmul_golden_reference(inp.data(), ic, oc_gate, golden_w, ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc_gate; ++o) {
                            const double r = (double) ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                }
            }
        } else {
            const float * metal = llama_get_embeddings_ith(ctx_dft, i_batch);
            if (metal) {
                const int nd = oc < n_embd ? oc : n_embd;
                double dot = 0.0;
                double ref = 0.0;
                double outv = 0.0;
                for (int e = 0; e < nd; ++e) {
                    const double m = (double) metal[e];
                    const double a = (double) h[(size_t) e];
                    dot  += m * a;
                    ref  += m * m;
                    outv += a * a;
                }
                if (ref > 0.0 && outv > 0.0) {
                    *out_hidden_cos = (float) (dot / (std::sqrt(ref) * std::sqrt(outv)));
                }
            }
        }
    }

    const llama_token id = drive_argmax_tied(h);
    *out_id = id;
    *out_p  = 0.95f; // lab proxy confidence — not full softmax over vocab
    return true;
}
#else
bool common_ane_draft_try_drive_token(struct llama_context *, int32_t, llama_token *, float *, float *) {
    return false;
}
#endif

#if defined(__APPLE__)
static float fp16_bits_to_float32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.f : 0.f;
        }
        const float f = (float) mant / 1024.f * std::pow(2.f, -14.f);
        return sign ? -f : f;
    }
    if (exp == 31) {
        if (mant) {
            return std::nanf("");
        }
        return sign ? -INFINITY : INFINITY;
    }
    const float f = std::ldexp(1.f + (float) mant / 1024.f, (int) exp - 15);
    return sign ? -f : f;
}

static bool load_gamma_scales(const char * path, int channels, std::vector<float> & out);

static bool load_conv_golden_weights(const char * path, int ch, std::vector<float> & W) {
    if (!path || !path[0] || ch <= 0) {
        return false;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t expected = 64 + 64 + (size_t) ch * (size_t) ch * 2;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0 || (size_t) sz != expected) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(expected);
    if (std::fread(buf.data(), 1, expected, f) != expected) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    W.resize((size_t) ch * (size_t) ch);
    const uint8_t * fp16 = buf.data() + 128;
    for (int oc = 0; oc < ch; ++oc) {
        for (int ic = 0; ic < ch; ++ic) {
            const size_t idx = (size_t) oc * (size_t) ch + (size_t) ic;
            uint16_t bits = (uint16_t) fp16[idx * 2] | ((uint16_t) fp16[idx * 2 + 1] << 8);
            W[idx] = fp16_bits_to_float32(bits);
        }
    }
    return true;
}

static void conv_golden_reference(const float * input, int ch, const std::vector<float> & W, std::vector<float> & ref) {
    ref.assign((size_t) ch, 0.f);
    if (!input || (int) W.size() < ch * ch) {
        return;
    }
    for (int oc = 0; oc < ch; ++oc) {
        double sum = 0.0;
        for (int ic = 0; ic < ch; ++ic) {
            sum += (double) W[(size_t) oc * (size_t) ch + (size_t) ic] * (double) input[ic];
        }
        ref[(size_t) oc] = (float) sum;
    }
}

static bool load_matmul_golden_weights(const char * path, int ic, int oc, std::vector<float> & W) {
    if (!path || !path[0] || ic <= 0 || oc <= 0) {
        return false;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t expected = 64 + 64 + (size_t) ic * (size_t) oc * 2;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0 || (size_t) sz != expected) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(expected);
    if (std::fread(buf.data(), 1, expected, f) != expected) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    W.resize((size_t) ic * (size_t) oc);
    const uint8_t * fp16 = buf.data() + 128;
    for (int i = 0; i < ic; ++i) {
        for (int o = 0; o < oc; ++o) {
            const size_t idx = (size_t) i * (size_t) oc + (size_t) o;
            uint16_t bits = (uint16_t) fp16[idx * 2] | ((uint16_t) fp16[idx * 2 + 1] << 8);
            W[idx] = fp16_bits_to_float32(bits);
        }
    }
    return true;
}

static void matmul_golden_reference(const float * input, int ic, int oc,
                                      const std::vector<float> & W, std::vector<float> & ref) {
    ref.assign((size_t) oc, 0.f);
    if (!input || (int) W.size() < ic * oc) {
        return;
    }
    for (int o = 0; o < oc; ++o) {
        double sum = 0.0;
        for (int i = 0; i < ic; ++i) {
            sum += (double) input[i] * (double) W[(size_t) i * (size_t) oc + (size_t) o];
        }
        ref[(size_t) o] = (float) sum;
    }
}

static float ane_silu_host(float x) {
    return x / (1.f + std::exp(-x));
}

static void ane_stash_handoff_hidden(const float * h, int n) {
    if (!h || n <= 0) {
        g_last_handoff_hidden.clear();
        return;
    }
    g_last_handoff_hidden.assign(h, h + n);
}

// Returns IOSurface-packed hidden (gamma applied) when available — matches ANE matmul input.
static const float * ane_handoff_hidden_for_golden(struct llama_context * ctx_dft, int32_t i_batch, int * out_len) {
    if (!g_last_handoff_hidden.empty()) {
        if (out_len) {
            *out_len = (int) g_last_handoff_hidden.size();
        }
        return g_last_handoff_hidden.data();
    }
    const float * pre = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
    if (!pre) {
        pre = llama_get_embeddings_ith(ctx_dft, i_batch);
    }
    if (out_len && pre && ctx_dft) {
        *out_len = llama_model_n_embd(llama_get_model(ctx_dft));
    }
    return pre;
}

static bool ane_fill_golden_input(std::vector<float> & inp, int ic, struct llama_context * ctx_dft, int32_t i_batch) {
    if (!ctx_dft || ic <= 0) {
        return false;
    }
    int pre_len = 0;
    const float * pre = ane_handoff_hidden_for_golden(ctx_dft, i_batch, &pre_len);
    if (!pre) {
        return false;
    }
    const bool from_stash = !g_last_handoff_hidden.empty() && pre == g_last_handoff_hidden.data();
    inp.assign((size_t) ic, 0.f);
    for (int i = 0; i < ic; ++i) {
        inp[(size_t) i] = (i < pre_len) ? pre[i] : 0.f;
    }
    if (!from_stash) {
        std::vector<float> gamma;
        if (load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic, gamma)) {
            for (int i = 0; i < ic && i < (int) gamma.size(); ++i) {
                inp[(size_t) i] *= gamma[(size_t) i];
            }
        }
    }
    return true;
}

static void log_ane_matmul_chain2_golden_telemetry(const float * input, int ic1, int oc1, int oc2, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc2 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        ok.store(w1 && w2 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, oc1, oc2, w_up));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic1);
    for (int i = 0; i < ic1; ++i) {
        inp[(size_t) i] = input[i];
    }
    std::vector<float> gamma;
    if (load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic1, gamma)) {
        for (int i = 0; i < ic1 && i < (int) gamma.size(); ++i) {
            inp[(size_t) i] *= gamma[(size_t) i];
        }
    }

    std::vector<float> gate;
    matmul_golden_reference(inp.data(), ic1, oc1, w_gate, gate);
    for (float & v : gate) {
        v = ane_silu_host(v);
    }
    std::vector<float> ref;
    matmul_golden_reference(gate.data(), oc1, oc2, w_up, ref);

    const size_t out_floats = (size_t) oc2 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc2; ++o) {
        const float r = ref[(size_t) o];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) o * (size_t) sp + (size_t) s];
        }
        const float o_avg = o_sum / (float) (sp > 0 ? sp : 1);
        mse += (double) (o_avg - r) * (o_avg - r);
        dot_ref += (double) r * r;
        dot_out += (double) o_avg * o_avg;
        dot_cross += (double) r * o_avg;
    }
    mse /= oc2 > 0 ? oc2 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain2 mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d oc_gate=%d oc_up=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc2, sp);
}

static void log_ane_matmul_chain3_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        ok.store(w1 && w2 && w3 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic1);
    for (int i = 0; i < ic1; ++i) {
        inp[(size_t) i] = input[i];
    }
    std::vector<float> gamma;
    if (load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic1, gamma)) {
        for (int i = 0; i < ic1 && i < (int) gamma.size(); ++i) {
            inp[(size_t) i] *= gamma[(size_t) i];
        }
    }

    std::vector<float> gate;
    std::vector<float> up;
    matmul_golden_reference(inp.data(), ic1, oc1, w_gate, gate);
    matmul_golden_reference(inp.data(), ic1, oc1, w_up, up);
    for (int i = 0; i < oc1; ++i) {
        gate[(size_t) i] = ane_silu_host(gate[(size_t) i]) * up[(size_t) i];
    }
    std::vector<float> ref;
    matmul_golden_reference(gate.data(), oc1, oc3, w_down, ref);

    const size_t out_floats = (size_t) oc3 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc3; ++o) {
        const float r = ref[(size_t) o];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) o * (size_t) sp + (size_t) s];
        }
        const float o_avg = o_sum / (float) (sp > 0 ? sp : 1);
        mse += (double) (o_avg - r) * (o_avg - r);
        dot_ref += (double) r * r;
        dot_out += (double) o_avg * o_avg;
        dot_cross += (double) r * o_avg;
    }
    mse /= oc3 > 0 ? oc3 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain3 mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d oc=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc3, sp);
}

static void log_ane_matmul_chain4_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int oc4, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || oc4 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::vector<float> w_attn;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        ok.store(w1 && w2 && w3 && w4 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down) &&
                 load_matmul_golden_weights(w4, oc3, oc4, w_attn));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic1);
    for (int i = 0; i < ic1; ++i) {
        inp[(size_t) i] = input[i];
    }
    std::vector<float> gamma;
    if (load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic1, gamma)) {
        for (int i = 0; i < ic1 && i < (int) gamma.size(); ++i) {
            inp[(size_t) i] *= gamma[(size_t) i];
        }
    }

    std::vector<float> gate;
    std::vector<float> up;
    matmul_golden_reference(inp.data(), ic1, oc1, w_gate, gate);
    matmul_golden_reference(inp.data(), ic1, oc1, w_up, up);
    for (int i = 0; i < oc1; ++i) {
        gate[(size_t) i] = ane_silu_host(gate[(size_t) i]) * up[(size_t) i];
    }
    std::vector<float> down;
    matmul_golden_reference(gate.data(), oc1, oc3, w_down, down);
    std::vector<float> ref;
    matmul_golden_reference(down.data(), oc3, oc4, w_attn, ref);

    const size_t out_floats = (size_t) oc4 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc4; ++o) {
        const float r = ref[(size_t) o];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) o * (size_t) sp + (size_t) s];
        }
        const float o_avg = o_sum / (float) (sp > 0 ? sp : 1);
        mse += (double) (o_avg - r) * (o_avg - r);
        dot_ref += (double) r * r;
        dot_out += (double) o_avg * o_avg;
        dot_cross += (double) r * o_avg;
    }
    mse /= oc4 > 0 ? oc4 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain4 mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d attn=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc4, sp);
}

static void log_ane_matmul_chain5_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int oc5, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || oc5 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::vector<float> w_ssm;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        ok.store(w1 && w2 && w3 && w5 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down) &&
                 load_matmul_golden_weights(w5, oc3, oc5, w_ssm));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic1);
    for (int i = 0; i < ic1; ++i) {
        inp[(size_t) i] = input[i];
    }
    std::vector<float> gamma;
    if (load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic1, gamma)) {
        for (int i = 0; i < ic1 && i < (int) gamma.size(); ++i) {
            inp[(size_t) i] *= gamma[(size_t) i];
        }
    }

    std::vector<float> gate;
    std::vector<float> up;
    matmul_golden_reference(inp.data(), ic1, oc1, w_gate, gate);
    matmul_golden_reference(inp.data(), ic1, oc1, w_up, up);
    for (int i = 0; i < oc1; ++i) {
        gate[(size_t) i] = ane_silu_host(gate[(size_t) i]) * up[(size_t) i];
    }
    std::vector<float> down;
    matmul_golden_reference(gate.data(), oc1, oc3, w_down, down);
    std::vector<float> ref;
    matmul_golden_reference(down.data(), oc3, oc5, w_ssm, ref);

    const size_t out_floats = (size_t) oc5 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc5; ++o) {
        const float r = ref[(size_t) o];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) o * (size_t) sp + (size_t) s];
        }
        const float o_avg = o_sum / (float) (sp > 0 ? sp : 1);
        mse += (double) (o_avg - r) * (o_avg - r);
        dot_ref += (double) r * r;
        dot_out += (double) o_avg * o_avg;
        dot_cross += (double) r * o_avg;
    }
    mse /= oc5 > 0 ? oc5 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain5 mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d ssm=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc5, sp);
}

static void log_ane_matmul_golden_telemetry(const float * input, int ic, int oc, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic <= 0 || oc <= 0 || sp <= 0) {
        return;
    }

    static std::vector<float> golden_w;
    static std::atomic<bool> golden_loaded { false };
    static std::atomic<bool> golden_ok { false };
    if (!golden_loaded.exchange(true)) {
        const char * wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        golden_ok.store(load_matmul_golden_weights(wpath, ic, oc, golden_w));
    }
    if (!golden_ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic);
    for (int i = 0; i < ic; ++i) {
        inp[(size_t) i] = input[i];
    }
    std::vector<float> gamma;
    const char * gamma_path = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    if (load_gamma_scales(gamma_path, ic, gamma)) {
        for (int i = 0; i < ic && i < (int) gamma.size(); ++i) {
            inp[(size_t) i] *= gamma[(size_t) i];
        }
    }

    std::vector<float> ref;
    matmul_golden_reference(inp.data(), ic, oc, golden_w, ref);

    const size_t out_floats = (size_t) oc * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc; ++o) {
        const float r = ref[(size_t) o];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) o * (size_t) sp + (size_t) s];
        }
        const float o_avg = o_sum / (float) (sp > 0 ? sp : 1);
        mse += (double) (o_avg - r) * (o_avg - r);
        dot_ref += (double) r * r;
        dot_out += (double) o_avg * o_avg;
        dot_cross += (double) r * o_avg;
    }
    mse /= oc > 0 ? oc : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d oc=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic, oc, sp);
}

static void log_ane_golden_telemetry(const float * input, int ch, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ch <= 0 || sp <= 0) {
        return;
    }
    if (ane_draft_session_matmul_active()) {
        const int oc1 = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", ch);
        if (ane_draft_session_matmul_chain_depth() >= 5) {
            log_ane_matmul_chain5_golden_telemetry(
                input, ch, oc1, ane_draft_session_matmul_ffn_embd(),
                ane_draft_session_output_channels(), sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 4) {
            log_ane_matmul_chain4_golden_telemetry(
                input, ch, oc1, ane_draft_session_matmul_ffn_embd(),
                ane_draft_session_output_channels(), sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 3) {
            log_ane_matmul_chain3_golden_telemetry(input, ch, oc1, ane_draft_session_output_channels(), sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 2) {
            log_ane_matmul_chain2_golden_telemetry(input, ch, oc1, ane_draft_session_output_channels(), sp, step);
            return;
        }
        log_ane_matmul_golden_telemetry(input, ch, oc1, sp, step);
        return;
    }

    static std::vector<float> golden_w;
    static std::vector<float> golden_w2;
    static std::vector<float> golden_w3;
    static std::vector<float> golden_w4;
    static std::vector<float> golden_w5;
    static std::vector<float> golden_w6;
    static std::vector<float> golden_w7;
    static std::vector<float> golden_w8;
    static std::atomic<bool> golden_loaded { false };
    static std::atomic<bool> golden_ok { false };
    if (!golden_loaded.exchange(true)) {
        const char * wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        golden_ok.store(load_conv_golden_weights(wpath, ch, golden_w));
        const char * w2path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        if (w2path && w2path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w2path, ch, golden_w2));
        }
        const char * w3path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        if (w3path && w3path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w3path, ch, golden_w3));
        }
        const char * w4path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        if (w4path && w4path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w4path, ch, golden_w4));
        }
        const char * w5path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        if (w5path && w5path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w5path, ch, golden_w5));
        }
        const char * w6path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        if (w6path && w6path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w6path, ch, golden_w6));
        }
        const char * w7path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        if (w7path && w7path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w7path, ch, golden_w7));
        }
        const char * w8path = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        if (w8path && w8path[0]) {
            golden_ok.store(golden_ok.load() && load_conv_golden_weights(w8path, ch, golden_w8));
        }
    }
    if (!golden_ok.load()) {
        return;
    }

    std::vector<float> ref;
    std::vector<float> mid;
    std::vector<float> inp((size_t) ch);
    for (int c = 0; c < ch; ++c) {
        inp[(size_t) c] = input[c];
    }
    std::vector<float> gamma;
    const char * gamma_path = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    if (load_gamma_scales(gamma_path, ch, gamma)) {
        for (int c = 0; c < ch && c < (int) gamma.size(); ++c) {
            inp[(size_t) c] *= gamma[(size_t) c];
        }
    }
    conv_golden_reference(inp.data(), ch, golden_w, mid);
    if (!golden_w2.empty() && golden_conv_enabled(2)) {
        conv_golden_reference(mid.data(), ch, golden_w2, ref);
    } else {
        ref = mid;
    }
    if (!golden_w3.empty() && golden_conv_enabled(3)) {
        conv_golden_reference(ref.data(), ch, golden_w3, mid);
        ref = mid;
    }
    if (!golden_w4.empty() && golden_conv_enabled(4)) {
        conv_golden_reference(ref.data(), ch, golden_w4, mid);
        ref = mid;
    }
    if (!golden_w5.empty() && golden_conv_enabled(5)) {
        conv_golden_reference(ref.data(), ch, golden_w5, mid);
        ref = mid;
    }
    if (!golden_w6.empty() && golden_conv_enabled(6)) {
        conv_golden_reference(ref.data(), ch, golden_w6, mid);
        ref = mid;
    }
    if (!golden_w7.empty() && golden_conv_enabled(7)) {
        conv_golden_reference(ref.data(), ch, golden_w7, mid);
        ref = mid;
    }
    if (!golden_w8.empty() && golden_conv_enabled(8)) {
        conv_golden_reference(ref.data(), ch, golden_w8, mid);
        ref = mid;
    }
    if (!golden_w2.empty() && !ane_draft_session_using_conv2()) {
        ref = mid;
    }

    const size_t out_floats = (size_t) ch * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int c = 0; c < ch; ++c) {
        const float r = ref[(size_t) c];
        float o_sum = 0.f;
        for (int s = 0; s < sp; ++s) {
            o_sum += out[(size_t) c * (size_t) sp + (size_t) s];
        }
        const float o = o_sum / (float) sp;
        mse += (double) (o - r) * (o - r);
        dot_ref += (double) r * r;
        dot_out += (double) o * o;
        dot_cross += (double) r * o;
    }
    mse /= ch > 0 ? ch : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    const char * mode = "conv1";
    const int active_golden = ane_draft_session_active_conv_count();
    if (active_golden >= 8 && golden_conv_enabled(8) && !golden_w8.empty()) {
        mode = "conv8";
    } else if (active_golden >= 7 && golden_conv_enabled(7) && !golden_w7.empty()) {
        mode = "conv7";
    } else if (active_golden >= 6 && golden_conv_enabled(6) && !golden_w6.empty()) {
        mode = "conv6";
    } else if (active_golden >= 5 && golden_conv_enabled(5) && !golden_w5.empty()) {
        mode = "conv5";
    } else if (active_golden >= 4 && golden_conv_enabled(4) && !golden_w4.empty()) {
        mode = "conv4";
    } else if (active_golden >= 3 && golden_conv_enabled(3) && !golden_w3.empty()) {
        mode = "conv3";
    } else if (active_golden >= 2 && golden_conv_enabled(2) && ane_draft_session_using_conv2()) {
        mode = "conv2";
    }
    LOG_INF("%s: B6 golden step=%d mode=%s mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d\n",
            __func__, step, mode, mse, cosine, ane_draft_session_step_count());
}
#endif

void common_ane_draft_log_init(common_speculative_type type, int draft_n_embd) {
    if (!common_ane_draft_enabled()) {
        return;
    }

    static std::atomic<bool> logged { false };
    if (logged.exchange(true)) {
        return;
    }

    LOG_INF("%s: ZEROLLAMA_ANE_DRAFT=1 — lab hook active for %s draft (n_embd=%d)\n",
            __func__,
            common_speculative_type_to_str(type).c_str(),
            draft_n_embd);

    if (!ane_draft_session_supported()) {
        LOG_WRN("%s: in-process ANE session unavailable (build without libane_bridge)\n", __func__);
        return;
    }

    const char * weight = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
    const char * gamma  = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    const int init_ch = env_int_or("ZEROLLAMA_ANE_DRAFT_CHANNELS", 64);
    const int init_sp = env_int_or("ZEROLLAMA_ANE_DRAFT_SPATIAL", 16);

    if (ane_draft_session_init(init_ch, init_sp, weight, gamma)) {
        const char * weight2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * weight3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * weight4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        const char * weight5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const char * weight6 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        const char * weight7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * weight8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        LOG_INF("%s: in-process ANE session ready channels=%d spatial=%d surface_id=%u bytes=%zu weight=%s weight2=%s weight3=%s weight4=%s weight5=%s weight6=%s weight7=%s weight8=%s gamma=%s\n",
                __func__,
                ane_draft_session_channels(),
                ane_draft_session_spatial(),
                ane_draft_session_surface_id(),
                ane_draft_session_surface_bytes(),
                weight && weight[0] ? weight : "(synthetic)",
                weight2 && weight2[0] ? weight2 : "(none)",
                weight3 && weight3[0] ? weight3 : "(none)",
                weight4 && weight4[0] ? weight4 : "(none)",
                weight5 && weight5[0] ? weight5 : "(none)",
                weight6 && weight6[0] ? weight6 : "(none)",
                weight7 && weight7[0] ? weight7 : "(none)",
                weight8 && weight8[0] ? weight8 : "(none)",
                gamma && gamma[0] ? gamma : "(none)");
        const int depth_cap = ane_draft_session_conv_depth_cap();
        const int active_convs = ane_draft_session_active_conv_count();
        if (depth_cap > 0) {
            LOG_INF("%s: conv depth cap=%d active_convs=%d (WEIGHT_FILE..%d compiled; higher slots ignored)\n",
                    __func__, depth_cap, active_convs, depth_cap);
        }
        const bool matmul_kernel = getenv("ZEROLLAMA_ANE_DRAFT_KERNEL") &&
                                   strcmp(getenv("ZEROLLAMA_ANE_DRAFT_KERNEL"), "matmul") == 0;
        if (matmul_kernel) {
            const int chain = ane_draft_session_matmul_chain_depth();
            const char * chain_note = "";
            if (chain >= 5) {
                chain_note = " chain5=swiglu+down+attn_gate+ssm_out";
            } else if (chain >= 4) {
                chain_note = " chain4=swiglu+down+attn_gate";
            } else if (chain >= 3) {
                chain_note = " chain3=swiglu+down";
            } else if (chain >= 2) {
                chain_note = " chain2=gate+silu+up";
            }
            LOG_INF("%s: P1 matmul kernel active (blk.0 ffn_gate h@W, seq=%d oc=%d%s%s)\n",
                    __func__,
                    ane_draft_session_spatial(),
                    ane_draft_session_output_channels(),
                    ane_draft_session_matmul_dynamic() ? " dynamic_mil" : "",
                    chain_note);
        } else switch (active_convs) {
        case 8:
            LOG_INF("%s: B13 oct conv1 chain active (block0 quad + blk.1 full quad)\n", __func__);
            break;
        case 7:
            LOG_INF("%s: B12 hept conv1 chain active (block0 quad + blk.1 gate/up/attn_gate)\n", __func__);
            break;
        case 6:
            LOG_INF("%s: B11 hex conv1 chain active (block0 quad + blk.1 gate/up)\n", __func__);
            break;
        case 5:
            LOG_INF("%s: B10 pent conv1 chain active (block0 quad + blk.1 ffn_gate)\n", __func__);
            break;
        case 4:
            LOG_INF("%s: B9 quad conv1 chain active (WEIGHT_FILE2..4 — gate/up/attn_gate/ffn_down proxy)\n", __func__);
            break;
        case 3:
            LOG_INF("%s: B8 triple conv1 chain active (WEIGHT_FILE2 + WEIGHT_FILE3)\n", __func__);
            break;
        case 2:
            LOG_INF("%s: B6 dual conv1 chain active (WEIGHT_FILE2)\n", __func__);
            break;
        default:
            break;
        }
    } else {
        LOG_WRN("%s: in-process ANE session init failed (channels=%d spatial=%d weight=%s)\n",
                __func__,
                init_ch,
                init_sp,
                weight && weight[0] ? weight : "(synthetic)");
    }

    LOG_INF("%s: B6 handoff stride=%d — IOSurface each %d decode step(s); TELEMETRY=1 for golden ref vs ANE\n",
            __func__, ane_handoff_stride(), ane_handoff_stride());

    const common_ane_draft_drive_mode drive = common_ane_draft_get_drive_mode();
    if (drive != COMMON_ANE_DRAFT_DRIVE_OFF) {
        if (ane_drive_metrics_hidden_only()) {
            LOG_INF("%s: B7 drive mode=shadow metrics=hidden — matmul gate cosine only (skip tied-embed argmax)\n", __func__);
        } else {
            LOG_INF("%s: B7 drive mode=%s — TOKEN_EMBD_FILE + OUTPUT_NORM_FILE for tied-embed argmax\n",
                    __func__,
                    drive == COMMON_ANE_DRAFT_DRIVE_FORCE ? "force" : "shadow");
        }
    }
}

#if defined(__APPLE__)
static bool load_gamma_scales(const char * path, int channels, std::vector<float> & out) {
    if (!path || !path[0] || channels <= 0) {
        return false;
    }
    static std::vector<float> cached;
    static std::string cached_path;
    static int cached_ch = 0;
    if (cached_path == path && cached_ch == channels && !cached.empty()) {
        out = cached;
        return true;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    const size_t expected = 64 + 64 + (size_t) channels * 2;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::rewind(f);
    if (sz < 0 || (size_t) sz != expected) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(expected);
    if (std::fread(buf.data(), 1, expected, f) != expected) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    out.resize((size_t) channels);
    const uint8_t * fp16 = buf.data() + 128;
    for (int i = 0; i < channels; ++i) {
        uint16_t bits = (uint16_t) fp16[i * 2] | ((uint16_t) fp16[i * 2 + 1] << 8);
        out[(size_t) i] = fp16_bits_to_float32(bits);
    }
    cached = out;
    cached_path = path;
    cached_ch = channels;
    return true;
}

static bool pack_draft_hidden_into_iosurface(const float * src, int src_len) {
    const uint32_t surface_id = ane_draft_session_surface_id();
    const size_t surface_bytes = ane_draft_session_surface_bytes();
    const int ch = ane_draft_session_channels();
    const int sp = ane_draft_session_spatial();
    if (surface_id == 0 || surface_bytes == 0 || ch <= 0 || sp <= 0 || !src) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        LOG_WRN("%s: no GPU backend device for iosurface handoff\n", __func__);
        return false;
    }

    // Reuse ggml IOSurface map across handoffs — why: alloc/free each step added e2e overhead.
    struct cached_iosurface_buf {
        uint32_t sid = 0;
        size_t bytes = 0;
        ggml_backend_buffer_t buf = nullptr;
    };
    static cached_iosurface_buf cache;

    if (!cache.buf || cache.sid != surface_id || cache.bytes != surface_bytes) {
        if (cache.buf) {
            ggml_backend_buffer_free(cache.buf);
            cache.buf = nullptr;
        }
        cache.buf = ggml_backend_dev_buffer_from_iosurface(
                dev, surface_id, surface_bytes, surface_bytes);
        cache.sid = surface_id;
        cache.bytes = surface_bytes;
    }
    ggml_backend_buffer_t buf = cache.buf;
    if (!buf) {
        LOG_WRN("%s: ggml_backend_dev_buffer_from_iosurface failed surface_id=%u\n", __func__, surface_id);
        return false;
    }

    float * dst = (float *) ggml_backend_buffer_get_base(buf);
    if (!dst) {
        LOG_WRN("%s: iosurface buffer base is null\n", __func__);
        return false;
    }

    // Proxy layout [1, ch, 1, sp]: channel-major; optional B3 sidecar norm gamma scales activations.
    std::vector<float> gamma;
    const char * gamma_path = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    const bool have_gamma = load_gamma_scales(gamma_path, ch, gamma);

    if (ane_draft_session_matmul_dynamic()) {
        std::vector<float> scaled((size_t) src_len);
        for (int c = 0; c < src_len; ++c) {
            float v = src[c];
            if (have_gamma && c < (int) gamma.size()) {
                v *= gamma[(size_t) c];
            }
            scaled[(size_t) c] = v;
        }
        ane_stash_handoff_hidden(scaled.data(), src_len);
        return ane_draft_session_pack_matmul_activations(dst, scaled.data(), src_len);
    }

    for (int c = 0; c < ch; ++c) {
        float v = (c < src_len) ? src[c] : 0.f;
        if (have_gamma && c < (int) gamma.size()) {
            v *= gamma[(size_t) c];
        }
        for (int s = 0; s < sp; ++s) {
            dst[(size_t) c * (size_t) sp + (size_t) s] = v;
        }
    }

    return true;
}
#endif

#if defined(__APPLE__)
static void ane_handoff_eval_done(bool ok) {
    if (!ok) {
        LOG_WRN("%s: async ANE eval failed\n", __func__);
        return;
    }
    g_ane_output_ready.store(true);
    if (g_async_golden_telemetry && !g_async_golden_emb.empty()) {
        log_ane_golden_telemetry(
            g_async_golden_emb.data(),
            ane_draft_session_channels(),
            ane_draft_session_spatial(),
            g_async_golden_step);
    }
}
#endif

void common_ane_draft_handoff_after_decode(struct llama_context * ctx_dft, int32_t i_batch) {
    if (!common_ane_draft_enabled() || !ane_draft_session_ready() || !ctx_dft) {
        return;
    }

    const int step = g_handoff_step.fetch_add(1) + 1;
    const int stride = ane_handoff_stride();
    if ((step - 1) % stride != 0) {
        return;
    }
    const bool log_info = step <= 3 || ane_draft_telemetry_enabled();

    const float * emb = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
    if (!emb) {
        emb = llama_get_embeddings_ith(ctx_dft, i_batch);
    }
    if (!emb) {
        if (log_info) {
            LOG_WRN("%s: step=%d draft hidden unavailable at i_batch=%d — stub ANE fill\n",
                    __func__, step, i_batch);
        }
        if (ane_draft_session_step_once(0.01f)) {
            LOG_INF("%s: stub ANE step ok (no draft hidden state)\n", __func__);
        }
        return;
    }

    const int n_embd = llama_model_n_embd(llama_get_model(ctx_dft));

#if defined(__APPLE__)
    if (ane_draft_session_eval_async_enabled()) {
        ane_draft_session_eval_sync();
    }

    if (!pack_draft_hidden_into_iosurface(emb, n_embd)) {
        if (log_info) {
            LOG_WRN("%s: step=%d iosurface pack failed — stub ANE fill\n", __func__, step);
        }
        if (ane_draft_session_step_once(0.01f)) {
            LOG_INF("%s: stub ANE step ok after handoff failure\n", __func__);
        }
        return;
    }

    g_ane_output_ready.store(false);
    if (ane_draft_session_eval_async_enabled()) {
        g_async_golden_telemetry = ane_draft_telemetry_enabled();
        g_async_golden_step = step;
        if (g_async_golden_telemetry) {
            g_async_golden_emb.assign(emb, emb + n_embd);
        } else {
            g_async_golden_emb.clear();
        }
        if (!ane_draft_session_eval_async(ane_handoff_eval_done)) {
            LOG_WRN("%s: step=%d ANE async eval dispatch failed\n", __func__, step);
            return;
        }
        if (log_info) {
            LOG_INF("%s: step=%d iosurface handoff ok — async ANE eval queued n_embd=%d\n",
                    __func__, step, n_embd);
        }
        return;
    }

    if (!ane_draft_session_eval()) {
        LOG_WRN("%s: step=%d ANE eval failed after ggml iosurface handoff\n", __func__, step);
        return;
    }

    g_ane_output_ready.store(true);

    log_ane_golden_telemetry(emb, ane_draft_session_channels(), ane_draft_session_spatial(), step);

    if (log_info) {
        LOG_INF("%s: step=%d ggml iosurface handoff ok — n_embd=%d surface %ux%u%s, eval ok\n",
                __func__,
                step,
                n_embd,
                (unsigned) ane_draft_session_channels(),
                (unsigned) ane_draft_session_spatial(),
                (std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE") ? " (sidecar gamma)" : ""));
    } else {
        LOG_DBG("%s: step=%d handoff+eval ok\n", __func__, step);
    }
#else
    GGML_UNUSED(step);
    GGML_UNUSED(log_info);
    GGML_UNUSED(emb);
    GGML_UNUSED(n_embd);
    LOG_WRN("%s: handoff skipped (not Apple platform)\n", __func__);
#endif
}
