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

static llama_context * g_ctx_tgt = nullptr;
static common_speculative_type g_spec_type = COMMON_SPECULATIVE_TYPE_NONE;

void common_ane_draft_bind_target_ctx(struct llama_context * ctx_tgt) {
    g_ctx_tgt = ctx_tgt;
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
static float ane_dflash_fc_ref_cosine(const float * input, int ic, int oc, const float * gate, int gate_n);
static float ane_dflash_chain11_attn_q_ref_cosine(
        const float * input, int ic_fc, int oc_fc, int oc_q, const float * gate, int gate_n);
static float ane_dflash_chain12_attn_v_ref_cosine(
        const float * input, int ic_fc, int oc_fc, int oc_v, const float * gate, int gate_n);
static float ane_dflash_chain13_attn_out_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_fc, int oc,
        const float * gate, int gate_n, struct llama_context * ctx_dft);
static float ane_dflash_chain14_wo_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo,
        const float * gate, int gate_n, struct llama_context * ctx_dft);
static float ane_dflash_chain15_ffn_gate_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_gate,
        const float * gate, int gate_n, struct llama_context * ctx_dft);
static float ane_dflash_chain16_ffn_down_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff,
        const float * gate, int gate_n, struct llama_context * ctx_dft);
static float ane_dflash_chain17_lm_head_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff,
        const float * gate, int gate_n, struct llama_context * ctx_dft);
static bool ane_dflash_post_eval_pipeline(struct llama_context * ctx_dft);
static int ane_dflash_ref_ic(int fallback_ic);
static float ane_vec_cosine(const float * ref, const float * out, int n);
static bool load_output_norm_vector(const char * path, int n, std::vector<float> & out);
static void ane_apply_output_norm_file(std::vector<float> & h);

static void log_ane_golden_telemetry(const float * input, int ch, int sp, int step);
static void ane_handoff_eval_done(bool ok);

static std::atomic<bool> g_ane_output_ready { false };
static std::atomic<int>  g_handoff_step { 0 };
static std::vector<float> g_last_handoff_hidden;
static std::vector<float> g_dflash_inpSA;
static llama_token g_dflash_handoff_token = LLAMA_TOKEN_NULL;
static std::vector<float> g_dflash_ffn_residual;
static std::vector<float> g_async_golden_emb;
static int g_async_golden_step = 0;
static bool g_async_golden_telemetry = false;
static llama_context * g_handoff_ctx_dft = nullptr;
#endif

void common_ane_draft_reset_handoff(void) {
#if defined(__APPLE__)
    if (ane_draft_session_eval_async_enabled() &&
        common_ane_draft_get_drive_mode() == COMMON_ANE_DRAFT_DRIVE_OFF) {
        ane_draft_session_eval_sync();
    }
    g_ane_output_ready.store(false);
    g_handoff_step.store(0);
    g_last_handoff_hidden.clear();
    g_dflash_inpSA.clear();
    g_dflash_handoff_token = LLAMA_TOKEN_NULL;
    g_dflash_ffn_residual.clear();
    g_async_golden_emb.clear();
    g_async_golden_step = 0;
    g_async_golden_telemetry = false;
#endif
}

void common_ane_draft_note_handoff_token(llama_token tok) {
    g_dflash_handoff_token = tok;
}

llama_token common_ane_draft_last_handoff_token(void) {
    return g_dflash_handoff_token;
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
        LOG_DBG("%s: skip drive ready=%d session=%d\n", __func__,
                (int) g_ane_output_ready.load(), (int) ane_draft_session_ready());
        return false;
    }

    if (ane_draft_session_eval_async_enabled() &&
        common_ane_draft_get_drive_mode() == COMMON_ANE_DRAFT_DRIVE_OFF) {
        ane_draft_session_eval_sync();
    }

    if (!g_ane_output_ready.load()) {
        LOG_DBG("%s: ANE output not ready after eval_sync\n", __func__);
        return false;
    }

    const bool metrics_only = ane_drive_metrics_hidden_only();
    const bool matmul = ane_draft_session_matmul_active();

    if (!metrics_only) {
        const char * embd_path = std::getenv("ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE");
        const char * norm_path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
        if (!drive_head_load(embd_path, norm_path)) {
            LOG_DBG("%s: drive head load failed embd=%s\n", __func__, embd_path ? embd_path : "(null)");
            return false;
        }
    }

    const int oc_out = ane_draft_session_output_channels();
    const int sp = ane_draft_session_spatial();
    const int ic = matmul ? ane_draft_session_channels() : oc_out;
    const int chain_depth = matmul ? ane_draft_session_matmul_chain_depth() : 0;

    std::vector<float> gate;
    int oc = oc_out;
    if (matmul && (ane_draft_session_dflash_chain17_active() || ane_draft_session_dflash_chain16_active() || ane_draft_session_dflash_chain15_active() || ane_draft_session_dflash_chain14_active() || ane_draft_session_dflash_chain13_active() || ane_draft_session_dflash_chain12_active())) {
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
    } else if (matmul && ane_draft_session_dflash_chain11_active()) {
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
    } else if (matmul && ane_draft_session_dflash_fc_active() && chain_depth == 8) {
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
    } else if (matmul && chain_depth >= 4) {
        const int ffn_embd = ane_draft_session_matmul_ffn_embd();
        if (ffn_embd <= 0) {
            return false;
        }
        const int read_ch = (chain_depth >= 10 && chain_depth <= 10) ? ffn_embd : (chain_depth >= 9 ? oc_out : ffn_embd);
        gate.assign((size_t) read_ch, 0.f);
        if (chain_depth >= 9) {
            const size_t nfloats = (size_t) read_ch * (size_t) sp;
            std::vector<float> ane_out(nfloats);
            if (ane_draft_session_read_output(ane_out.data(), nfloats) == 0) {
                LOG_DBG("%s: matmul output read failed chain=%d\n", __func__, chain_depth);
                return false;
            }
            for (int o = 0; o < read_ch; ++o) {
                double sum = 0.0;
                for (int s = 0; s < sp; ++s) {
                    sum += (double) ane_out[(size_t) o * (size_t) sp + (size_t) s];
                }
                gate[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
            }
        } else if (ane_draft_session_read_ffn_down(gate.data(), gate.size()) == 0) {
            LOG_DBG("%s: ffn_down stash read failed chain=%d\n", __func__, chain_depth);
            return false;
        }
        oc = read_ch;
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
            if (ane_draft_session_dflash_chain17_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int oc_ff = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain17_lm_head_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain17_lm_head_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain16_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int oc_ff = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain16_ffn_down_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain16_ffn_down_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain15_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int oc_ffn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain15_ffn_gate_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, oc_ffn, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain15_ffn_gate_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, oc_ffn, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain14_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain14_wo_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain14_wo_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain13_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_v  = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain13_attn_out_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_v, oc, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain13_attn_out_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_v, oc, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain12_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_v  = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain12_attn_v_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_v, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain12_attn_v_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_v, gate.data(), oc);
                }
            } else if (ane_draft_session_dflash_chain11_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain11_attn_q_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain11_attn_q_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc, gate.data(), oc);
                }
            } else if (ane_draft_session_dflash_fc_active() && chain_depth == 8) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_fc_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_fc_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, gate.data(), oc);
                }
            } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                if (ane_draft_session_matmul_chain_depth() == 10) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::vector<float> w_blk1_gate;
                    static std::vector<float> w_blk1_up;
                    static std::vector<float> w_blk1_down;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
                        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
                        const char * w9 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
                        chain_ok.store(w1 && w2 && w3 && w7 && w8 && w9 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down) &&
                                       load_matmul_golden_weights(w7, oc_ffn, oc_gate, w_blk1_gate) &&
                                       load_matmul_golden_weights(w8, oc_ffn, oc_gate, w_blk1_up) &&
                                       load_matmul_golden_weights(w9, oc_gate, oc_ffn, w_blk1_down));
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
                        std::vector<float> g1;
                        std::vector<float> u1;
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_gate, g1);
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_up, u1);
                        std::vector<float> swiglu_ref((size_t) oc_gate);
                        for (int i = 0; i < oc_gate; ++i) {
                            swiglu_ref[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
                        }
                        std::vector<float> blk1_down_ref;
                        matmul_golden_reference(swiglu_ref.data(), oc_gate, oc_ffn, w_blk1_down, blk1_down_ref);
                        const size_t nfloats = (size_t) oc_ffn * (size_t) sp;
                        std::vector<float> ane_out(nfloats);
                        if (ane_draft_session_read_output(ane_out.data(), nfloats) > 0) {
                            double dot = 0.0;
                            double refn = 0.0;
                            double outn = 0.0;
                            for (int o = 0; o < oc_ffn; ++o) {
                                float o_sum = 0.f;
                                for (int s = 0; s < sp; ++s) {
                                    o_sum += ane_out[(size_t) o * (size_t) sp + (size_t) s];
                                }
                                const float a = o_sum / (float) (sp > 0 ? sp : 1);
                                const double r = (double) blk1_down_ref[(size_t) o];
                                dot  += r * (double) a;
                                refn += r * r;
                                outn += (double) a * (double) a;
                            }
                            if (refn > 0.0 && outn > 0.0) {
                                *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                            }
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 9) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::vector<float> w_blk1_gate;
                    static std::vector<float> w_blk1_up;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
                        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
                        chain_ok.store(w1 && w2 && w3 && w7 && w8 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down) &&
                                       load_matmul_golden_weights(w7, oc_ffn, oc_gate, w_blk1_gate) &&
                                       load_matmul_golden_weights(w8, oc_ffn, oc_gate, w_blk1_up));
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
                        std::vector<float> g1;
                        std::vector<float> u1;
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_gate, g1);
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_up, u1);
                        std::vector<float> swiglu_ref((size_t) oc_gate);
                        for (int i = 0; i < oc_gate; ++i) {
                            swiglu_ref[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
                        }
                        const size_t nfloats = (size_t) oc_out * (size_t) sp;
                        std::vector<float> ane_out(nfloats);
                        if (ane_draft_session_read_output(ane_out.data(), nfloats) > 0) {
                            double dot = 0.0;
                            double refn = 0.0;
                            double outn = 0.0;
                            for (int o = 0; o < oc_gate; ++o) {
                                float o_sum = 0.f;
                                for (int s = 0; s < sp; ++s) {
                                    o_sum += ane_out[(size_t) o * (size_t) sp + (size_t) s];
                                }
                                const float a = o_sum / (float) (sp > 0 ? sp : 1);
                                const double r = (double) swiglu_ref[(size_t) o];
                                dot  += r * (double) a;
                                refn += r * r;
                                outn += (double) a * (double) a;
                            }
                            if (refn > 0.0 && outn > 0.0) {
                                *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                            }
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 7) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::vector<float> w_blk1;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
                        chain_ok.store(w1 && w2 && w3 && w7 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down) &&
                                       load_matmul_golden_weights(w7, oc_ffn, oc_gate, w_blk1));
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
                        std::vector<float> blk1_ref;
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1, blk1_ref);
                        const size_t nfloats = (size_t) oc_out * (size_t) sp;
                        std::vector<float> ane_out(nfloats);
                        if (ane_draft_session_read_output(ane_out.data(), nfloats) > 0) {
                            double dot = 0.0;
                            double refn = 0.0;
                            double outn = 0.0;
                            for (int o = 0; o < oc_gate; ++o) {
                                float o_sum = 0.f;
                                for (int s = 0; s < sp; ++s) {
                                    o_sum += ane_out[(size_t) o * (size_t) sp + (size_t) s];
                                }
                                const float a = o_sum / (float) (sp > 0 ? sp : 1);
                                const double r = (double) blk1_ref[(size_t) o];
                                dot  += r * (double) a;
                                refn += r * r;
                                outn += (double) a * (double) a;
                            }
                            if (refn > 0.0 && outn > 0.0) {
                                *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                            }
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 3) {
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
    if (!ane_draft_session_dflash_chain17_active()) {
        drive_apply_rms_norm(h, g_drive_head.out_norm, eps);
    }

    if (out_hidden_cos && ctx_dft && i_batch >= 0) {
        *out_hidden_cos = 0.f;
        if (matmul) {
            const int oc_gate = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_out);
            const int oc_ffn  = (chain_depth >= 4) ? ane_draft_session_matmul_ffn_embd() : oc;
            std::vector<float> inp;
            if (ane_draft_session_dflash_chain17_active()) {
                const float * metal = llama_get_embeddings_ith(ctx_dft, i_batch);
                if (metal) {
                    const int nd = oc < n_embd ? oc : n_embd;
                    std::vector<float> mref((size_t) nd);
                    for (int i = 0; i < nd; ++i) {
                        mref[(size_t) i] = metal[i];
                    }
                    ane_apply_output_norm_file(mref);
                    *out_hidden_cos = ane_vec_cosine(mref.data(), h.data(), nd);
                }
            } else if (ane_draft_session_dflash_chain16_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int oc_ff = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain16_ffn_down_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain16_ffn_down_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, oc_ff, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain15_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int oc_ffn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain15_ffn_gate_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, oc_ffn, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain15_ffn_gate_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, oc_ffn, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain14_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_attn = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc));
                const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc_fc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain14_wo_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_attn, oc_wo, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain14_wo_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_attn, oc_wo, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain13_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_v  = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain13_attn_out_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_v, oc, gate.data(), oc, ctx_dft);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain13_attn_out_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_v, oc, gate.data(), oc, ctx_dft);
                }
            } else if (ane_draft_session_dflash_chain12_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int oc_v  = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain12_attn_v_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc_v, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain12_attn_v_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc_v, gate.data(), oc);
                }
            } else if (ane_draft_session_dflash_chain11_active()) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_chain11_attn_q_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, oc, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_chain11_attn_q_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, oc, gate.data(), oc);
                }
            } else if (ane_draft_session_dflash_fc_active() && chain_depth == 8) {
                const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_gate);
                const int ic_in = ane_dflash_ref_ic(ic);
                if (!g_last_handoff_hidden.empty() && ic_in > 0) {
                    *out_hidden_cos = ane_dflash_fc_ref_cosine(
                            g_last_handoff_hidden.data(), ic_in, oc_fc, gate.data(), oc);
                } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                    *out_hidden_cos = ane_dflash_fc_ref_cosine(
                            inp.data(), ane_dflash_ref_ic((int) inp.size()), oc_fc, gate.data(), oc);
                }
            } else if (ane_fill_golden_input(inp, ic, ctx_dft, i_batch)) {
                if (ane_draft_session_matmul_chain_depth() == 10) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::vector<float> w_blk1_gate;
                    static std::vector<float> w_blk1_up;
                    static std::vector<float> w_blk1_down;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
                        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
                        const char * w9 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
                        chain_ok.store(w1 && w2 && w3 && w7 && w8 && w9 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down) &&
                                       load_matmul_golden_weights(w7, oc_ffn, oc_gate, w_blk1_gate) &&
                                       load_matmul_golden_weights(w8, oc_ffn, oc_gate, w_blk1_up) &&
                                       load_matmul_golden_weights(w9, oc_gate, oc_ffn, w_blk1_down));
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
                        std::vector<float> g1;
                        std::vector<float> u1;
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_gate, g1);
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_up, u1);
                        std::vector<float> swiglu_ref((size_t) oc_gate);
                        for (int i = 0; i < oc_gate; ++i) {
                            swiglu_ref[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
                        }
                        std::vector<float> blk1_down_ref;
                        matmul_golden_reference(swiglu_ref.data(), oc_gate, oc_ffn, w_blk1_down, blk1_down_ref);
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc_ffn; ++o) {
                            const double r = (double) blk1_down_ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 9) {
                    static std::vector<float> w_gate;
                    static std::vector<float> w_up;
                    static std::vector<float> w_down;
                    static std::vector<float> w_blk1_gate;
                    static std::vector<float> w_blk1_up;
                    static std::atomic<bool> chain_loaded { false };
                    static std::atomic<bool> chain_ok { false };
                    if (!chain_loaded.exchange(true)) {
                        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
                        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
                        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
                        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
                        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
                        chain_ok.store(w1 && w2 && w3 && w7 && w8 &&
                                       load_matmul_golden_weights(w1, ic, oc_gate, w_gate) &&
                                       load_matmul_golden_weights(w2, ic, oc_gate, w_up) &&
                                       load_matmul_golden_weights(w3, oc_gate, oc_ffn, w_down) &&
                                       load_matmul_golden_weights(w7, oc_ffn, oc_gate, w_blk1_gate) &&
                                       load_matmul_golden_weights(w8, oc_ffn, oc_gate, w_blk1_up));
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
                        std::vector<float> g1;
                        std::vector<float> u1;
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_gate, g1);
                        matmul_golden_reference(down_ref.data(), oc_ffn, oc_gate, w_blk1_up, u1);
                        std::vector<float> swiglu_ref((size_t) oc_gate);
                        for (int i = 0; i < oc_gate; ++i) {
                            swiglu_ref[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
                        }
                        double dot = 0.0;
                        double refn = 0.0;
                        double outn = 0.0;
                        for (int o = 0; o < oc_gate; ++o) {
                            const double r = (double) swiglu_ref[(size_t) o];
                            const double a = (double) gate[(size_t) o];
                            dot  += r * a;
                            refn += r * r;
                            outn += a * a;
                        }
                        if (refn > 0.0 && outn > 0.0) {
                            *out_hidden_cos = (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
                        }
                    }
                } else if (ane_draft_session_matmul_chain_depth() >= 3) {
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

// B7 shadow: compare ANE tied-embed argmax against the same op on Metal draft hidden — not llama_get_logits_ith
// (draft ctx logits are often unset/stale; embeddings are populated when llama_set_embeddings is enabled).
bool common_ane_draft_metal_ref_token(struct llama_context * ctx_dft, int32_t i_batch, llama_token * out_id) {
    if (!out_id || !ctx_dft || i_batch < 0) {
        return false;
    }
    const char * embd_path = std::getenv("ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE");
    const char * norm_path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
    if (!drive_head_load(embd_path, norm_path)) {
        return false;
    }
    const float * metal = llama_get_embeddings_ith(ctx_dft, i_batch);
    if (!metal) {
        return false;
    }
    const int n_embd = g_drive_head.n_embd;
    std::vector<float> h((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) {
        h[(size_t) i] = metal[i];
    }
    if (ane_draft_session_dflash_chain17_active()) {
        ane_apply_output_norm_file(h);
    }
    *out_id = drive_argmax_tied(h);
    return true;
}
#else
bool common_ane_draft_try_drive_token(struct llama_context *, int32_t, llama_token *, float *, float *) {
    return false;
}

bool common_ane_draft_metal_ref_token(struct llama_context *, int32_t, llama_token *) {
    return false;
}
#endif

#if !defined(__APPLE__)
void common_ane_draft_sync_target_cross(
        llama_context *,
        llama_context *,
        const llama_batch &) {
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

static void ane_tile_target_hidden(const float * emb, int n_embd_src, int pack_ic, int tile, std::vector<float> & out) {
    out.assign((size_t) pack_ic, 0.f);
    if (!emb || pack_ic <= 0) {
        return;
    }
    if (tile > 0 && n_embd_src > 0) {
        for (int i = 0; i < pack_ic; ++i) {
            out[(size_t) i] = emb[i % n_embd_src];
        }
    } else {
        for (int i = 0; i < pack_ic; ++i) {
            out[(size_t) i] = (i < n_embd_src) ? emb[i] : 0.f;
        }
    }
}

// Lab stub until llama exports per-layer target hiddens at dflash.target_layer_ids[].
static void ane_concat_layer_stub(const float * emb, int n_embd_src, int n_layers, int pack_ic, std::vector<float> & out) {
    out.assign((size_t) pack_ic, 0.f);
    if (!emb || pack_ic <= 0 || n_embd_src <= 0) {
        return;
    }
    if (n_layers <= 1) {
        ane_tile_target_hidden(emb, n_embd_src, pack_ic, 1, out);
        return;
    }
    const int per = pack_ic / n_layers;
    if (per <= 0) {
        ane_tile_target_hidden(emb, n_embd_src, pack_ic, 1, out);
        return;
    }
    for (int L = 0; L < n_layers; ++L) {
        for (int i = 0; i < per; ++i) {
            out[(size_t) L * (size_t) per + (size_t) i] = emb[i % n_embd_src];
        }
    }
}

static bool model_arch_is(const llama_model * model, const char * arch) {
    if (!model || !arch) {
        return false;
    }
    char buf[64] = {};
    return llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 &&
           strcmp(buf, arch) == 0;
}

static bool ane_should_sync_target_cross(const llama_model * model_dft) {
    if (common_ane_draft_enabled()) {
        return true;
    }
    if (g_spec_type == COMMON_SPECULATIVE_TYPE_DFLASH) {
        return true;
    }
    return model_arch_is(model_dft, "dflash-draft");
}

static int ane_dflash_cross_feat_dim(const llama_model * model_dft) {
    const int env = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES", 0);
    if (env > 0) {
        return env;
    }
    const int native = llama_model_dflash_n_target_features(model_dft);
    if (native > 0) {
        return native;
    }
    const int n = llama_model_n_embd(model_dft);
    return n > 0 ? n : 0;
}

static int ane_dflash_matmul_input_dim() {
#if defined(__APPLE__)
    const int ch = ane_draft_session_channels();
    if (ch > 0) {
        return ch;
    }
#endif
    return 0;
}

static bool ane_dflash_fc_host_enabled(void) {
    return env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_FC_HOST", 0) != 0;
}

static int ane_dflash_fc_full_ic(void) {
    const int full = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_FC_FULL_IC", 0);
    if (full > 0) {
        return full;
    }
    return env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES", 0);
}

static int ane_dflash_fc_input_dim(const llama_model * model_dft) {
    if (ane_dflash_fc_host_enabled()) {
        const int full = ane_dflash_fc_full_ic();
        if (full > 0) {
            return full;
        }
    }
    const int matmul_ic = ane_dflash_matmul_input_dim();
    if (matmul_ic > 0) {
        return matmul_ic;
    }
    return ane_dflash_cross_feat_dim(model_dft);
}

static int ane_dflash_target_feat_dim(const llama_model * model_dft) {
    return ane_dflash_cross_feat_dim(model_dft);
}

static int ane_dflash_target_layer_count(const llama_model * model_dft) {
    const int env = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_LAYERS", 0);
    if (env > 0) {
        return env;
    }
    return llama_model_dflash_n_target_layers(model_dft);
}

static int32_t ane_cross_row_index(llama_context * ctx_dft, int32_t src_i) {
    if (src_i >= 0) {
        return src_i;
    }
    if (!ctx_dft) {
        return 0;
    }
    const int32_t n_enc = llama_context_cross_n_enc(ctx_dft);
    return n_enc > 0 ? n_enc - 1 : 0;
}

// ctx_tgt batch index != draft i_batch — use last target output row (pre-norm supports i < 0).
static int32_t ane_handoff_src_batch_index(llama_context * src_ctx, int32_t draft_i_batch) {
    if (g_ctx_tgt && src_ctx == g_ctx_tgt) {
        return -1;
    }
    return draft_i_batch;
}

static const float * ane_ctx_hidden_row(llama_context * ctx, int32_t i_batch) {
    if (!ctx) {
        return nullptr;
    }
    const float * emb = llama_get_embeddings_pre_norm_ith(ctx, i_batch);
    if (!emb) {
        emb = llama_get_embeddings_ith(ctx, i_batch);
    }
    return emb;
}

static bool ane_copy_dflash_export_row(
        llama_context * src_ctx,
        int32_t src_i,
        int pack_ic,
        std::vector<float> & feat) {
    if (!src_ctx || pack_ic <= 0) {
        return false;
    }
    float * row = llama_get_dflash_target_features_ith(src_ctx, src_i);
    if (!row) {
        return false;
    }
    int n_feat = pack_ic;
    if (ggml_tensor * t = llama_get_dflash_target_features(src_ctx)) {
        n_feat = (int) t->ne[0];
    }
    const int ncopy = std::min(pack_ic, n_feat);
    feat.assign((size_t) pack_ic, 0.f);
    if (ncopy > 0) {
        std::memcpy(feat.data(), row, (size_t) ncopy * sizeof(float));
    }
    return true;
}

// B8: pack target_hidden for dflash_fc — prefer cross.v_embd row, else ctx_tgt per-layer export, else pre-norm stub.
static bool ane_pack_dflash_target_feat(
        llama_context * ctx_dft,
        llama_context * src_ctx,
        int32_t i_batch,
        int pack_ic,
        std::vector<float> & feat,
        const float ** pack_ptr,
        int * pack_len) {
    if (!pack_ptr || !pack_len || pack_ic <= 0) {
        return false;
    }

    const llama_model * model_dft = ctx_dft ? llama_get_model(ctx_dft) : nullptr;
    const int32_t src_i = ane_handoff_src_batch_index(src_ctx, i_batch);
    const int32_t cross_i = ane_cross_row_index(ctx_dft, src_i);
    const int cross_ic = model_dft ? ane_dflash_cross_feat_dim(model_dft) : pack_ic;
    static std::atomic<int> pack_path_logged { 0 };

    if (src_ctx) {
        std::vector<float> export_row;
        const int export_n = cross_ic > pack_ic ? cross_ic : pack_ic;
        if (ane_copy_dflash_export_row(src_ctx, src_i, export_n, export_row)) {
            feat.assign((size_t) pack_ic, 0.f);
            const int ncopy = std::min(pack_ic, export_n);
            if (ncopy > 0) {
                std::memcpy(feat.data(), export_row.data(), (size_t) ncopy * sizeof(float));
            }
            *pack_ptr = feat.data();
            *pack_len = pack_ic;
            if (pack_path_logged.fetch_add(1) < 3 && ane_draft_telemetry_enabled()) {
                LOG_INF("%s: B8 pack path=export handoff_ic=%d export_n=%d cross_ic=%d src_i=%d\n",
                        __func__, pack_ic, export_n, cross_ic, (int) src_i);
            }
            // cross.v_embd is populated at full width in common_ane_draft_sync_target_cross;
            // do not shrink rows with the matmul slice here.
            return true;
        }
    }

    if (ctx_dft && llama_context_cross_has_v_embd(ctx_dft)) {
        feat.resize((size_t) pack_ic);
        const int32_t copied = llama_context_cross_row(ctx_dft, cross_i, feat.data(), pack_ic);
        if (copied > 0) {
            *pack_ptr = feat.data();
            *pack_len = pack_ic;
            if (pack_path_logged.fetch_add(1) < 3 && ane_draft_telemetry_enabled()) {
                LOG_INF("%s: B8 pack path=cross handoff_ic=%d copied=%d cross_i=%d\n",
                        __func__, pack_ic, (int) copied, (int) cross_i);
            }
            return true;
        }
    }

    if (!src_ctx) {
        return false;
    }

    const float * emb = ane_ctx_hidden_row(src_ctx, src_i);
    if (!emb) {
        return false;
    }

    const int n_embd_src = llama_model_n_embd(llama_get_model(src_ctx));
    const int tile = env_int_or("ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", 0);
    ane_tile_target_hidden(emb, n_embd_src, pack_ic, tile, feat);
    *pack_ptr = feat.data();
    *pack_len = pack_ic;

    if (ctx_dft && env_int_or("ZEROLLAMA_ANE_DRAFT_SYNC_CROSS", 1) != 0) {
        llama_context_cross_upsert_row(ctx_dft, cross_i, feat.data(), pack_ic);
    }
    if (pack_path_logged.fetch_add(1) < 3 && ane_draft_telemetry_enabled()) {
        LOG_INF("%s: B8 pack path=stub handoff_ic=%d tile=%d n_embd_src=%d\n",
                __func__, pack_ic, tile, n_embd_src);
    }
    return true;
}

void common_ane_draft_sync_target_cross(
        llama_context * ctx_dft,
        llama_context * ctx_tgt,
        const llama_batch & batch) {
    if (!ctx_dft || !ctx_tgt || batch.n_tokens <= 0) {
        return;
    }
    const llama_model * model_dft = llama_get_model(ctx_dft);
    if (!ane_should_sync_target_cross(model_dft)) {
        return;
    }
    if (env_int_or("ZEROLLAMA_ANE_DRAFT_SYNC_CROSS", 1) == 0) {
        return;
    }

    const int pack_ic = ane_dflash_cross_feat_dim(model_dft);
    if (pack_ic <= 0) {
        return;
    }

    llama_synchronize(ctx_tgt);

    const int cross_ic = pack_ic;
    const int n_embd_src = llama_model_n_embd(llama_get_model(ctx_tgt));
    const int tile = env_int_or("ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE", 0);
    const int n_layers = ane_dflash_target_layer_count(model_dft);
    std::vector<float> feat;

    for (int32_t t = 0; t < batch.n_tokens; ++t) {
        if (batch.logits && !batch.logits[t]) {
            continue;
        }
        const float * emb = ane_ctx_hidden_row(ctx_tgt, t);
        if (!emb) {
            continue;
        }
        const int32_t cross_pos = batch.pos ? (int32_t) batch.pos[t] : t;
        if (cross_pos < 0) {
            continue;
        }
        if (ane_copy_dflash_export_row(ctx_tgt, t, cross_ic, feat)) {
            llama_context_cross_upsert_row(ctx_dft, cross_pos, feat.data(), cross_ic);
            continue;
        }
        if (n_layers > 1) {
            ane_concat_layer_stub(emb, n_embd_src, n_layers, cross_ic, feat);
        } else {
            ane_tile_target_hidden(emb, n_embd_src, cross_ic, tile, feat);
        }
        llama_context_cross_upsert_row(ctx_dft, cross_pos, feat.data(), cross_ic);
    }
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

static void log_ane_matmul_qkv_prefix_golden_telemetry(const float * input, int ic1, int oc1, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_qkv;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w6 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        ok.store(w6 && load_matmul_golden_weights(w6, ic1, oc1, w_qkv));
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

    std::vector<float> ref;
    matmul_golden_reference(inp.data(), ic1, oc1, w_qkv, ref);

    std::vector<float> ane_qkv((size_t) oc1);
    if (ane_draft_session_read_qkv_prefix(ane_qkv.data(), ane_qkv.size()) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc1; ++o) {
        const float r = ref[(size_t) o];
        const float a = ane_qkv[(size_t) o];
        mse += (double) (a - r) * (a - r);
        dot_ref += (double) r * r;
        dot_out += (double) a * a;
        dot_cross += (double) r * a;
    }
    mse /= oc1 > 0 ? oc1 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain6_qkv mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d oc=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, sp);
}

static void log_ane_matmul_chain7_blk1_gate_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int oc7, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || oc7 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::vector<float> w_blk1;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        ok.store(w1 && w2 && w3 && w7 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down) &&
                 load_matmul_golden_weights(w7, oc3, oc7, w_blk1));
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
    matmul_golden_reference(down.data(), oc3, oc7, w_blk1, ref);

    const size_t out_floats = (size_t) oc7 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc7; ++o) {
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
    mse /= oc7 > 0 ? oc7 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain7_blk1_gate mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d blk1=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc7, sp);
}

static void log_ane_matmul_dflash_fc_golden_telemetry(const float * input, int ic, int oc, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic <= 0 || oc <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_fc;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        ok.store(w1 && load_matmul_golden_weights(w1, ic, oc, w_fc));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic);
    for (int i = 0; i < ic; ++i) {
        inp[(size_t) i] = input[i];
    }

    std::vector<float> ref;
    matmul_golden_reference(inp.data(), ic, oc, w_fc, ref);

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

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain8_dflash_fc mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d oc=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic, oc, sp);
}

static void ane_apply_rms_gamma_host(float * h, int n, const std::vector<float> & gamma) {
    if (!h || n <= 0) {
        return;
    }
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += (double) h[i] * (double) h[i];
    }
    const float inv_rms = 1.0f / std::sqrt((float) (sum_sq / (double) n) + 1e-6f);
    for (int i = 0; i < n; ++i) {
        const float g = (!gamma.empty() && i < (int) gamma.size()) ? gamma[(size_t) i] : 1.f;
        h[i] *= inv_rms * g;
    }
}

static float ane_vec_cosine(const float * ref, const float * out, int n) {
    if (!ref || !out || n <= 0) {
        return 0.f;
    }
    double dot = 0.0;
    double refn = 0.0;
    double outn = 0.0;
    for (int i = 0; i < n; ++i) {
        dot  += (double) ref[(size_t) i] * (double) out[(size_t) i];
        refn += (double) ref[(size_t) i] * (double) ref[(size_t) i];
        outn += (double) out[(size_t) i] * (double) out[(size_t) i];
    }
    if (refn > 0.0 && outn > 0.0) {
        return (float) (dot / (std::sqrt(refn) * std::sqrt(outn)));
    }
    return 0.f;
}

static int ane_dflash_ref_ic(int fallback_ic) {
    if (!g_last_handoff_hidden.empty()) {
        return (int) g_last_handoff_hidden.size();
    }
    return fallback_ic;
}

static float ane_vec_dot(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double) a[i] * b[i];
    }
    return (float) sum;
}

static float env_float_or(const char * name, float def) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) {
        return def;
    }
    return (float) std::atof(v);
}

static bool ane_dflash_host_rope_enabled(void) {
    const char * v = std::getenv("ZEROLLAMA_ANE_DRAFT_HOST_ROPE");
    if (!v || !v[0]) {
        return true;
    }
    return env_truthy(v);
}

struct ane_dflash_attn_geom {
    int   n_heads_kv = 0;
    int   head_dim   = 0;
    int   n_dims     = 0;
    float freq_base  = 1e6f;
    float freq_scale = 1.f;
    bool  neox       = true;
};

static ane_dflash_attn_geom ane_dflash_attn_geom_from_ctx(llama_context * ctx, int oc) {
    ane_dflash_attn_geom g;
    g.head_dim   = env_int_or("ZEROLLAMA_ANE_DRAFT_N_EMBD_HEAD", 0);
    g.n_heads_kv = env_int_or("ZEROLLAMA_ANE_DRAFT_N_HEAD_KV", 0);
    if (g.head_dim > 0 && oc > 0 && oc % g.head_dim == 0) {
        const int n_from_oc = oc / g.head_dim;
        if (g.n_heads_kv <= 0 || g.n_heads_kv * g.head_dim != oc) {
            g.n_heads_kv = n_from_oc;
        }
    } else if (ctx && g.n_heads_kv <= 0) {
        const llama_model * model = llama_get_model(ctx);
        if (model) {
            g.n_heads_kv = llama_model_n_head_kv(model);
        }
        if (g.head_dim <= 0 && g.n_heads_kv > 0 && oc > 0 && oc % g.n_heads_kv == 0) {
            g.head_dim = oc / g.n_heads_kv;
        }
    }
    if (g.n_heads_kv <= 0 || g.head_dim <= 0 || g.n_heads_kv * g.head_dim != oc) {
        g.n_heads_kv = 1;
        g.head_dim   = oc;
    }
    g.n_dims = env_int_or("ZEROLLAMA_ANE_DRAFT_ROPE_N_DIMS", g.head_dim);
    if (g.n_dims > g.head_dim) {
        g.n_dims = g.head_dim;
    }
    if (g.n_dims > 0 && g.n_dims % 2 != 0) {
        g.n_dims -= 1;
    }
    g.freq_base  = env_float_or("ZEROLLAMA_ANE_DRAFT_ROPE_FREQ_BASE", 1e6f);
    g.freq_scale = env_float_or("ZEROLLAMA_ANE_DRAFT_ROPE_FREQ_SCALE", 1.f);
    g.neox       = env_int_or("ZEROLLAMA_ANE_DRAFT_ROPE_NEOX", 1) != 0;
    return g;
}

static void ane_apply_head_rms_norm(float * v, int n_heads, int head_dim, const std::vector<float> & gamma) {
    if (gamma.empty() || n_heads <= 0 || head_dim <= 0) {
        return;
    }
    for (int h = 0; h < n_heads; ++h) {
        float * head = v + h * head_dim;
        double sumsq = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            sumsq += (double) head[i] * (double) head[i];
        }
        const float inv_rms = 1.f / std::sqrt((float) (sumsq / (double) head_dim) + 1e-6f);
        for (int i = 0; i < head_dim; ++i) {
            const float g = i < (int) gamma.size() ? gamma[(size_t) i] : 1.f;
            head[i] *= inv_rms * g;
        }
    }
}

static void ane_rope_inplace_heads(float * v, int n_heads, const ane_dflash_attn_geom & geom, llama_pos pos) {
    if (geom.n_dims <= 0 || geom.head_dim <= 0 || n_heads <= 0) {
        return;
    }
    const float theta_scale = std::pow(geom.freq_base, -2.f / (float) geom.n_dims);
    for (int h = 0; h < n_heads; ++h) {
        float * head = v + h * geom.head_dim;
        if (geom.neox) {
            const int half = geom.n_dims / 2;
            for (int i0 = 0; i0 < geom.n_dims; i0 += 2) {
                const int ic  = i0 / 2;
                const float ang = (float) pos * std::pow(theta_scale, (float) ic) * geom.freq_scale;
                const float cos_t = std::cos(ang);
                const float sin_t = std::sin(ang);
                const int i1 = ic;
                const int i2 = ic + half;
                if (i2 >= geom.head_dim) {
                    break;
                }
                const float x0 = head[i1];
                const float x1 = head[i2];
                head[i1] = x0 * cos_t - x1 * sin_t;
                head[i2] = x0 * sin_t + x1 * cos_t;
            }
        } else {
            for (int i0 = 0; i0 + 1 < geom.n_dims; i0 += 2) {
                const int ic  = i0 / 2;
                const float ang = (float) pos * std::pow(theta_scale, (float) ic) * geom.freq_scale;
                const float cos_t = std::cos(ang);
                const float sin_t = std::sin(ang);
                const float x0 = head[i0];
                const float x1 = head[i0 + 1];
                head[i0]     = x0 * cos_t - x1 * sin_t;
                head[i0 + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

static void ane_rope_inplace(float * v, const ane_dflash_attn_geom & geom, llama_pos pos) {
    ane_rope_inplace_heads(v, geom.n_heads_kv, geom, pos);
}

static llama_pos ane_draft_seq_pos_max(llama_context * ctx) {
    if (!ctx) {
        return 0;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return 0;
    }
    const llama_pos pmax = llama_memory_seq_pos_max(mem, 0);
    return pmax >= 0 ? pmax : 0;
}

static void ane_dflash_prepare_k(
        float * k,
        int oc,
        int n_heads,
        const ane_dflash_attn_geom & geom,
        const std::vector<float> & k_norm,
        llama_pos pos) {
    if (!ane_dflash_host_rope_enabled() || oc <= 0 || n_heads <= 0) {
        return;
    }
    ane_apply_head_rms_norm(k, n_heads, geom.head_dim, k_norm);
    ane_rope_inplace_heads(k, n_heads, geom, pos);
}

static void ane_softmax_inplace(std::vector<float> & scores) {
    if (scores.empty()) {
        return;
    }
    float maxv = scores[0];
    for (float s : scores) {
        if (s > maxv) {
            maxv = s;
        }
    }
    double sum = 0.0;
    for (float & s : scores) {
        s = std::exp((double) s - (double) maxv);
        sum += s;
    }
    if (sum <= 0.0) {
        const float u = 1.f / (float) scores.size();
        for (float & s : scores) {
            s = u;
        }
        return;
    }
    for (float & s : scores) {
        s = (float) ((double) s / sum);
    }
}

static bool ane_dflash_load_attn_weights(
        int ic_in, int oc_fc, int oc_kv,
        std::vector<float> & w_fc,
        std::vector<float> & w_k,
        std::vector<float> & w_v) {
    const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_DFLASH_FC_FULL_WEIGHT_FILE");
    if (!w1 || !w1[0]) {
        w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
    }
    const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
    const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
    return w1 && w3 && w4 && ic_in > 0 && oc_fc > 0 && oc_kv > 0 &&
           load_matmul_golden_weights(w1, ic_in, oc_fc, w_fc) &&
           load_matmul_golden_weights(w3, oc_fc, oc_kv, w_k) &&
           load_matmul_golden_weights(w4, oc_fc, oc_kv, w_v);
}

static void ane_dflash_attn_qkv_oc(int oc_fallback, int & oc_q, int & oc_kv) {
    oc_q  = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", oc_fallback);
    oc_kv = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", oc_fallback);
    if (oc_kv <= 0) {
        oc_kv = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc_fallback);
    }
    if (oc_q <= 0) {
        oc_q = oc_kv > 0 ? oc_kv : oc_fallback;
    }
    if (oc_kv <= 0) {
        oc_kv = oc_q;
    }
}

static bool ane_dflash_fused_from_feat(
        const float * feat, int ic_in, int ic_fc, int oc_fc,
        const std::vector<float> & w_fc, const std::vector<float> & gamma,
        std::vector<float> & fused) {
    if (!feat || ic_in <= 0 || oc_fc <= 0) {
        return false;
    }
    fused.resize((size_t) oc_fc);
    const size_t need_w = (size_t) ic_in * (size_t) oc_fc;
    if (w_fc.size() >= need_w) {
        matmul_golden_reference(feat, ic_in, oc_fc, w_fc, fused);
    } else if (ic_fc > 0) {
        std::vector<float> inp((size_t) ic_fc, 0.f);
        const int ncopy = ic_in < ic_fc ? ic_in : ic_fc;
        if (ncopy > 0) {
            std::memcpy(inp.data(), feat, (size_t) ncopy * sizeof(float));
        }
        matmul_golden_reference(inp.data(), ic_fc, oc_fc, w_fc, fused);
    } else {
        return false;
    }
    ane_apply_rms_gamma_host(fused.data(), oc_fc, gamma);
    return true;
}

// Metal dflash: Q/K_noise/V_noise = proj(attn_norm(tok_embd)); ctx K/V use dflash_fc out (host cross-attn).
static bool ane_dflash_attn_norm_tok_cur(int ic_fc, std::vector<float> & cur) {
    cur.assign((size_t) ic_fc, 0.f);
    if (g_dflash_inpSA.empty() || ic_fc <= 0) {
        return false;
    }
    const int n_copy = std::min((int) g_dflash_inpSA.size(), ic_fc);
    std::memcpy(cur.data(), g_dflash_inpSA.data(), (size_t) n_copy * sizeof(float));
    std::vector<float> gamma;
    if (!load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_ATTN_NORM_FILE"), ic_fc, gamma)) {
        return false;
    }
    ane_apply_rms_gamma_host(cur.data(), ic_fc, gamma);
    return true;
}

static bool ane_dflash_qkv_from_attn_norm_tok(
        int ic_fc, int oc_q, int oc_kv,
        std::vector<float> & q,
        std::vector<float> & k_noise,
        std::vector<float> & v_noise) {
    if (oc_q <= 0 || oc_kv <= 0) {
        return false;
    }
    std::vector<float> cur;
    if (!ane_dflash_attn_norm_tok_cur(ic_fc, cur)) {
        return false;
    }
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> ok { false };
    static int cached_ic = -1;
    static int cached_oc_q = -1;
    static int cached_oc_kv = -1;
    if (cached_ic != ic_fc || cached_oc_q != oc_q || cached_oc_kv != oc_kv) {
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        ok.store(w2 && w3 && w4 &&
                 load_matmul_golden_weights(w2, ic_fc, oc_q, w_q) &&
                 load_matmul_golden_weights(w3, ic_fc, oc_kv, w_k) &&
                 load_matmul_golden_weights(w4, ic_fc, oc_kv, w_v));
        cached_ic = ic_fc;
        cached_oc_q = oc_q;
        cached_oc_kv = oc_kv;
    }
    if (!ok.load()) {
        return false;
    }
    q.resize((size_t) oc_q);
    k_noise.resize((size_t) oc_kv);
    v_noise.resize((size_t) oc_kv);
    matmul_golden_reference(cur.data(), ic_fc, oc_q, w_q, q);
    matmul_golden_reference(cur.data(), ic_fc, oc_kv, w_k, k_noise);
    matmul_golden_reference(cur.data(), ic_fc, oc_kv, w_v, v_noise);
    static std::atomic<bool> attn_norm_logged { false };
    if (!attn_norm_logged.exchange(true)) {
        LOG_INF("%s: P18 Q/K/V from attn_norm(tok_embd) ic=%d oc_q=%d oc_kv=%d inpSA=%d\n",
                __func__, ic_fc, oc_q, oc_kv, (int) g_dflash_inpSA.size());
    }
    return true;
}

static bool ane_dflash_qkv_for_cross_attn(
        const float * input, int ic_in, int ic_fc, int oc_q, int oc_kv,
        const std::vector<float> & w_fc, const std::vector<float> & gamma,
        const std::vector<float> & w_q,
        std::vector<float> & q,
        std::vector<float> & k_noise,
        std::vector<float> & v_noise) {
    if (ane_dflash_qkv_from_attn_norm_tok(ic_fc, oc_q, oc_kv, q, k_noise, v_noise)) {
        return true;
    }
    std::vector<float> fused;
    if (!ane_dflash_fused_from_feat(input, ic_in, ic_fc, ic_fc, w_fc, gamma, fused)) {
        return false;
    }
    q.resize((size_t) oc_q);
    matmul_golden_reference(fused.data(), ic_fc, oc_q, w_q, q);
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> kv_ok { false };
    static int cached_ic = -1;
    static int cached_oc_q = -1;
    static int cached_oc_kv = -1;
    if (cached_ic != ic_fc || cached_oc_q != oc_q || cached_oc_kv != oc_kv) {
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        kv_ok.store(w3 && w4 &&
                    load_matmul_golden_weights(w3, ic_fc, oc_kv, w_k) &&
                    load_matmul_golden_weights(w4, ic_fc, oc_kv, w_v));
        cached_ic = ic_fc;
        cached_oc_q = oc_q;
        cached_oc_kv = oc_kv;
    }
    if (!kv_ok.load()) {
        return false;
    }
    k_noise.resize((size_t) oc_kv);
    v_noise.resize((size_t) oc_kv);
    matmul_golden_reference(fused.data(), ic_fc, oc_kv, w_k, k_noise);
    matmul_golden_reference(fused.data(), ic_fc, oc_kv, w_v, v_noise);
    return true;
}

static bool ane_dflash_host_cross_attn(
        llama_context * ctx_dft,
        const float * q, const float * k_noise, const float * v_noise,
        int oc_q, int oc_kv,
        std::vector<float> & attn_out) {
    if (!q || !k_noise || !v_noise || oc_q <= 0 || oc_kv <= 0) {
        return false;
    }
    static std::vector<float> w_fc;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    static int cached_ic_in = 0;
    static int cached_oc_fc = 0;
    static int cached_oc_kv = 0;
    static int cached_cross_read = 0;
    if (!loaded.exchange(true)) {
        cached_ic_in = ane_dflash_fc_full_ic();
        if (cached_ic_in <= 0) {
            cached_ic_in = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES", 0);
        }
        if (cached_ic_in <= 0) {
            cached_ic_in = env_int_or("ZEROLLAMA_ANE_DRAFT_CHANNELS", 0);
        }
        cached_oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", 0);
        cached_oc_kv = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", oc_kv);
        cached_cross_read = cached_ic_in;
        if (cached_cross_read <= 0) {
            cached_cross_read = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES", 0);
        }
        if (cached_ic_in <= 0 || cached_oc_fc <= 0) {
            cached_ic_in = cached_oc_fc > 0 ? cached_oc_fc : oc_kv;
        }
        ok.store(ane_dflash_load_attn_weights(cached_ic_in, cached_oc_fc, cached_oc_kv, w_fc, w_k, w_v));
    }
    if (!ok.load()) {
        attn_out.assign((size_t) oc_q, 0.f);
        for (int i = 0; i < oc_q && i < oc_kv; ++i) {
            attn_out[(size_t) i] = v_noise[i];
        }
        return true;
    }

    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), cached_oc_fc, gamma);

    ane_dflash_attn_geom geom = ane_dflash_attn_geom_from_ctx(ctx_dft, oc_kv);
    int n_head_q = env_int_or("ZEROLLAMA_ANE_DRAFT_N_HEAD", 0);
    if (n_head_q <= 0 && geom.head_dim > 0 && oc_q % geom.head_dim == 0) {
        n_head_q = oc_q / geom.head_dim;
    }
    if (geom.head_dim <= 0 && n_head_q > 0 && oc_q % n_head_q == 0) {
        geom.head_dim = oc_q / n_head_q;
    }
    if (geom.n_heads_kv <= 0 && geom.head_dim > 0 && oc_kv % geom.head_dim == 0) {
        geom.n_heads_kv = oc_kv / geom.head_dim;
    }
    const bool gqa = n_head_q > 0 && geom.n_heads_kv > 0 && geom.head_dim > 0 &&
                     n_head_q * geom.head_dim == oc_q && geom.n_heads_kv * geom.head_dim == oc_kv &&
                     (n_head_q != geom.n_heads_kv || oc_q != oc_kv);
    const llama_pos q_pos = ane_draft_seq_pos_max(ctx_dft);

    static std::vector<float> q_norm_gamma;
    static std::vector<float> k_norm_gamma;
    if (q_norm_gamma.empty() && geom.head_dim > 0) {
        load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_ATTN_Q_NORM_FILE"), geom.head_dim, q_norm_gamma);
        load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_ATTN_K_NORM_FILE"), geom.head_dim, k_norm_gamma);
    }
    static std::atomic<bool> rope_logged { false };
    if (ane_dflash_host_rope_enabled() && !rope_logged.exchange(true)) {
        LOG_INF("%s: P17 host cross-attn RoPE n_head=%d n_head_kv=%d head_dim=%d n_dims=%d freq_base=%.0f q_norm=%d k_norm=%d gqa=%d\n",
                __func__, n_head_q, geom.n_heads_kv, geom.head_dim, geom.n_dims, (double) geom.freq_base,
                (int) q_norm_gamma.size(), (int) k_norm_gamma.size(), gqa ? 1 : 0);
    }

    std::vector<float> q_work((size_t) oc_q);
    std::vector<float> k_noise_work((size_t) oc_kv);
    std::memcpy(q_work.data(), q, (size_t) oc_q * sizeof(float));
    std::memcpy(k_noise_work.data(), k_noise, (size_t) oc_kv * sizeof(float));
    const int q_rope_heads = gqa ? n_head_q : geom.n_heads_kv;
    const int kv_rope_heads = geom.n_heads_kv > 0 ? geom.n_heads_kv : 1;
    if (ane_dflash_host_rope_enabled()) {
        ane_apply_head_rms_norm(q_work.data(), q_rope_heads, geom.head_dim, q_norm_gamma);
        ane_rope_inplace_heads(q_work.data(), q_rope_heads, geom, q_pos);
        ane_dflash_prepare_k(k_noise_work.data(), oc_kv, kv_rope_heads, geom, k_norm_gamma, q_pos);
    }

    const int n_enc = ctx_dft ? llama_context_cross_n_enc(ctx_dft) : 0;
    const int max_ctx = env_int_or("ZEROLLAMA_ANE_DRAFT_DFLASH_ATTN_MAX_CTX", 64);
    const int n_ctx = n_enc < max_ctx ? n_enc : max_ctx;
    const int win_off = n_enc > n_ctx ? n_enc - n_ctx : 0;
    const float attn_scale = geom.head_dim > 0 ? (1.f / std::sqrt((float) geom.head_dim)) :
                             (1.f / std::sqrt((float) (gqa ? oc_q / n_head_q : oc_q)));

    attn_out.assign((size_t) oc_q, 0.f);

    if (gqa) {
        std::vector<float> feat((size_t) cached_cross_read);
        std::vector<float> fused;
        std::vector<float> k_row((size_t) oc_kv);
        std::vector<float> v_row((size_t) oc_kv);
        for (int h = 0; h < n_head_q; ++h) {
            const int h_kv = (h * geom.n_heads_kv) / n_head_q;
            const float * q_h = q_work.data() + h * geom.head_dim;
            std::vector<float> scores;
            std::vector<std::vector<float>> v_heads;
            for (int j = 0; j < n_ctx; ++j) {
                if (!ctx_dft || llama_context_cross_row(ctx_dft, j, feat.data(), cached_cross_read) <= 0) {
                    continue;
                }
                if (!ane_dflash_fused_from_feat(feat.data(), cached_ic_in, cached_oc_fc, cached_oc_fc, w_fc, gamma, fused)) {
                    continue;
                }
                matmul_golden_reference(fused.data(), cached_oc_fc, oc_kv, w_k, k_row);
                matmul_golden_reference(fused.data(), cached_oc_fc, oc_kv, w_v, v_row);
                ane_dflash_prepare_k(k_row.data(), oc_kv, kv_rope_heads, geom, k_norm_gamma, (llama_pos) (win_off + j));
                const float * k_h = k_row.data() + h_kv * geom.head_dim;
                scores.push_back(ane_vec_dot(q_h, k_h, geom.head_dim) * attn_scale);
                v_heads.emplace_back(v_row.begin() + h_kv * geom.head_dim, v_row.begin() + (h_kv + 1) * geom.head_dim);
            }
            const float * k_h = k_noise_work.data() + h_kv * geom.head_dim;
            scores.push_back(ane_vec_dot(q_h, k_h, geom.head_dim) * attn_scale);
            v_heads.emplace_back(v_noise + h_kv * geom.head_dim, v_noise + (h_kv + 1) * geom.head_dim);
            ane_softmax_inplace(scores);
            float * out_h = attn_out.data() + h * geom.head_dim;
            for (size_t j = 0; j < v_heads.size(); ++j) {
                const float w = scores[j];
                for (int i = 0; i < geom.head_dim; ++i) {
                    out_h[i] += w * v_heads[j][(size_t) i];
                }
            }
        }
        return true;
    }

    // Flat proxy path (oc_q == oc_kv).
    const int oc = oc_q;
    std::vector<float> scores;
    std::vector<std::vector<float>> v_rows;
    std::vector<float> feat((size_t) cached_cross_read);
    std::vector<float> fused;
    std::vector<float> k_row((size_t) oc);
    std::vector<float> v_row((size_t) oc);
    for (int j = 0; j < n_ctx; ++j) {
        if (!ctx_dft || llama_context_cross_row(ctx_dft, j, feat.data(), cached_cross_read) <= 0) {
            continue;
        }
        if (!ane_dflash_fused_from_feat(feat.data(), cached_ic_in, cached_oc_fc, cached_oc_fc, w_fc, gamma, fused)) {
            continue;
        }
        matmul_golden_reference(fused.data(), cached_oc_fc, oc, w_k, k_row);
        matmul_golden_reference(fused.data(), cached_oc_fc, oc, w_v, v_row);
        ane_dflash_prepare_k(k_row.data(), oc, kv_rope_heads, geom, k_norm_gamma, (llama_pos) (win_off + j));
        scores.push_back(ane_vec_dot(q_work.data(), k_row.data(), oc) * attn_scale);
        v_rows.push_back(v_row);
    }
    scores.push_back(ane_vec_dot(q_work.data(), k_noise_work.data(), oc) * attn_scale);
    v_rows.emplace_back(v_noise, v_noise + oc);
    ane_softmax_inplace(scores);
    for (size_t j = 0; j < v_rows.size(); ++j) {
        const float w = scores[j];
        for (int i = 0; i < oc; ++i) {
            attn_out[(size_t) i] += w * v_rows[j][(size_t) i];
        }
    }
    return true;
}

static bool ane_dflash_post_eval_host_attn(llama_context * ctx_dft, int oc_q, int oc_kv) {
    if (!ctx_dft || oc_q <= 0 || oc_kv <= 0) {
        return false;
    }
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    const int ic_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", oc_q);
    if (!ane_dflash_qkv_from_attn_norm_tok(ic_fc, oc_q, oc_kv, q, k, v)) {
        q.assign((size_t) oc_q, 0.f);
        k.assign((size_t) oc_kv, 0.f);
        v.assign((size_t) oc_kv, 0.f);
        if (!ane_draft_session_read_dflash_qkv(q.data(), k.data(), v.data(), oc_kv)) {
            return false;
        }
        if ((int) q.size() < oc_q) {
            q.resize((size_t) oc_q, 0.f);
        }
    }
    std::vector<float> attn_out;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k.data(), v.data(), oc_q, oc_kv, attn_out)) {
        return false;
    }
    return ane_draft_session_write_dflash_attn_out(attn_out.data(), oc_q);
}

static void ane_dflash_stash_inpSA(struct llama_context * ctx_dft, int32_t i_batch) {
    (void) i_batch;
    g_dflash_inpSA.clear();
    if (!ctx_dft) {
        return;
    }
    const llama_model * model = llama_get_model(ctx_dft);
    const int n_embd = llama_model_n_embd(model);
    if (n_embd <= 0) {
        return;
    }
    const llama_token tok = g_dflash_handoff_token;
    const char * embd_path = std::getenv("ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE");
    if (tok != LLAMA_TOKEN_NULL && embd_path && drive_head_load(embd_path, nullptr)) {
        const uint16_t * W = g_drive_head.embed_fp16;
        const int nv = g_drive_head.n_vocab;
        if (W && tok >= 0 && tok < nv) {
            g_dflash_inpSA.resize((size_t) n_embd);
            for (int e = 0; e < n_embd; ++e) {
                const uint16_t bits = W[(size_t) e * (size_t) nv + (size_t) tok];
                g_dflash_inpSA[(size_t) e] = fp16_bits_to_float32(bits);
            }
            return;
        }
    }
    static std::atomic<bool> inpSA_fallback_logged { false };
    if (!inpSA_fallback_logged.exchange(true)) {
        LOG_WRN("%s: tok_embd lookup failed tok=%d — attn inpSA residual skipped\n", __func__, (int) tok);
    }
}

static void ane_dflash_ref_add_inpSA(std::vector<float> & h) {
    if (g_dflash_inpSA.empty()) {
        return;
    }
    const int n = (int) std::min(h.size(), g_dflash_inpSA.size());
    for (int i = 0; i < n; ++i) {
        h[(size_t) i] += g_dflash_inpSA[(size_t) i];
    }
}

static bool ane_dflash_apply_attn_inpSA_residual(void) {
    if (g_dflash_inpSA.empty()) {
        return true;
    }
    return ane_draft_session_add_output_row(g_dflash_inpSA.data(), (int) g_dflash_inpSA.size());
}

static bool ane_dflash_stash_ffn_skip(void) {
    const int n = ane_draft_session_matmul_ffn_embd();
    if (n <= 0) {
        g_dflash_ffn_residual.clear();
        return true;
    }
    g_dflash_ffn_residual.assign((size_t) n, 0.f);
    return ane_draft_session_snapshot_output_row(g_dflash_ffn_residual.data(), n);
}

static bool ane_dflash_apply_ffn_residual(void) {
    if (g_dflash_ffn_residual.empty()) {
        return true;
    }
    return ane_draft_session_add_output_row(g_dflash_ffn_residual.data(), (int) g_dflash_ffn_residual.size());
}

static void ane_dflash_ref_apply_attn_post_norm(std::vector<float> & h) {
    std::vector<float> gamma;
    if (!load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_ATTN_POST_NORM_FILE"), (int) h.size(), gamma)) {
        return;
    }
    ane_apply_rms_gamma_host(h.data(), (int) h.size(), gamma);
}

static bool ane_dflash_run_host_fc(const float * input, int ic_in, std::vector<float> & fc_out) {
    if (!input || ic_in <= 0) {
        return false;
    }
    const int oc_fc = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC", 0);
    if (oc_fc <= 0) {
        return false;
    }
    static std::vector<float> w_fc;
    static std::atomic<bool> ok { false };
    static int cached_ic = -1;
    static int cached_oc = -1;
    if (cached_ic != ic_in || cached_oc != oc_fc) {
        const char * wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_DFLASH_FC_FULL_WEIGHT_FILE");
        if (!wpath || !wpath[0]) {
            wpath = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        }
        ok.store(wpath && load_matmul_golden_weights(wpath, ic_in, oc_fc, w_fc));
        cached_ic = ic_in;
        cached_oc = oc_fc;
    }
    if (!ok.load()) {
        return false;
    }
    fc_out.resize((size_t) oc_fc);
    matmul_golden_reference(input, ic_in, oc_fc, w_fc, fc_out);
    static std::atomic<bool> host_fc_logged { false };
    if (!host_fc_logged.exchange(true)) {
        LOG_INF("%s: P19 host dflash_fc ic=%d oc=%d (full target export)\n", __func__, ic_in, oc_fc);
    }
    return true;
}

static bool ane_dflash_post_eval_pipeline(llama_context * ctx_dft) {
    if (ane_draft_session_dflash_chain13_active() || ane_draft_session_dflash_chain14_active() ||
        ane_draft_session_dflash_chain15_active() || ane_draft_session_dflash_chain16_active() ||
        ane_draft_session_dflash_chain17_active()) {
        const int oc_q = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", 0);
        const int oc_kv = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC3", 0);
        if (oc_q <= 0 || oc_kv <= 0) {
            return false;
        }
        if (!ane_dflash_post_eval_host_attn(ctx_dft, oc_q, oc_kv)) {
            return false;
        }
    }
    if (ane_draft_session_dflash_chain14_active() || ane_draft_session_dflash_chain15_active() ||
        ane_draft_session_dflash_chain16_active() || ane_draft_session_dflash_chain17_active()) {
        if (!ane_draft_session_eval_dflash_attn_wo()) {
            return false;
        }
        if (!ane_dflash_apply_attn_inpSA_residual()) {
            return false;
        }
        if (!ane_dflash_stash_ffn_skip()) {
            return false;
        }
    }
    if (ane_draft_session_dflash_chain15_active() || ane_draft_session_dflash_chain16_active() ||
        ane_draft_session_dflash_chain17_active()) {
        if (!ane_draft_session_eval_dflash_attn_post_norm()) {
            return false;
        }
    }
    if (ane_draft_session_dflash_chain15_active() || ane_draft_session_dflash_chain16_active() ||
        ane_draft_session_dflash_chain17_active()) {
        if (!ane_draft_session_eval_dflash_ffn_gate()) {
            return false;
        }
    }
    if (ane_draft_session_dflash_chain16_active() || ane_draft_session_dflash_chain17_active()) {
        if (!ane_draft_session_eval_dflash_ffn_up_swiglu_down()) {
            return false;
        }
        if (!ane_dflash_apply_ffn_residual()) {
            return false;
        }
    }
    if (ane_draft_session_dflash_chain17_active()) {
        if (!ane_draft_session_eval_dflash_output_norm()) {
            return false;
        }
    }
    return true;
}

static float ane_dflash_fc_ref_cosine(const float * input, int ic, int oc, const float * gate, int gate_n) {
    static std::vector<float> w_fc;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        ok.store(w1 && load_matmul_golden_weights(w1, ic, oc, w_fc));
    }
    if (!ok.load() || !input || !gate || ic <= 0 || oc <= 0) {
        return 0.f;
    }
    std::vector<float> ref;
    matmul_golden_reference(input, ic, oc, w_fc, ref);
    const int n = gate_n < oc ? gate_n : oc;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain11_attn_q_ref_cosine(
        const float * input, int ic_fc, int oc_fc, int oc_q, const float * gate, int gate_n) {
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        ok.store(w1 && w2 &&
                 load_matmul_golden_weights(w1, ic_fc, oc_fc, w_fc) &&
                 load_matmul_golden_weights(w2, oc_fc, oc_q, w_q));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_fc <= 0 || oc_q <= 0) {
        return 0.f;
    }
    std::vector<float> fc;
    matmul_golden_reference(input, ic_fc, oc_fc, w_fc, fc);
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), oc_fc, gamma);
    ane_apply_rms_gamma_host(fc.data(), oc_fc, gamma);
    std::vector<float> ref;
    matmul_golden_reference(fc.data(), oc_fc, oc_q, w_q, ref);
    const int n = gate_n < oc_q ? gate_n : oc_q;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain12_attn_v_ref_cosine(
        const float * input, int ic_fc, int oc_fc, int oc_v, const float * gate, int gate_n) {
    static std::vector<float> w_fc;
    static std::vector<float> w_v;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        ok.store(w1 && w4 &&
                 load_matmul_golden_weights(w1, ic_fc, oc_fc, w_fc) &&
                 load_matmul_golden_weights(w4, oc_fc, oc_v, w_v));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_fc <= 0 || oc_v <= 0) {
        return 0.f;
    }
    std::vector<float> fc;
    matmul_golden_reference(input, ic_fc, oc_fc, w_fc, fc);
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), oc_fc, gamma);
    ane_apply_rms_gamma_host(fc.data(), oc_fc, gamma);
    std::vector<float> ref;
    matmul_golden_reference(fc.data(), oc_fc, oc_v, w_v, ref);
    const int n = gate_n < oc_v ? gate_n : oc_v;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain13_attn_out_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_fc, int oc,
        const float * gate, int gate_n, llama_context * ctx_dft) {
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc, oc_q, oc_kv);
    static std::vector<float> w_fc;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    static int cached_ic_in = 0;
    if (!loaded.exchange(true)) {
        cached_ic_in = ic_in > 0 ? ic_in : env_int_or("ZEROLLAMA_ANE_DRAFT_CHANNELS", ic_fc);
        ok.store(ane_dflash_load_attn_weights(cached_ic_in, ic_fc, oc_kv, w_fc, w_k, w_v));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_q <= 0 || oc_kv <= 0 || !ctx_dft) {
        return 0.f;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic_fc, gamma);
    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    static std::vector<float> w_q;
    static std::atomic<bool> q_loaded { false };
    static std::atomic<bool> q_ok { false };
    if (!q_loaded.exchange(true)) {
        if (const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2")) {
            q_ok.store(load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
        }
    }
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return 0.f;
    }
    std::vector<float> ref;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, ref)) {
        return 0.f;
    }
    const int n = gate_n < oc_q ? gate_n : oc_q;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain14_wo_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo,
        const float * gate, int gate_n, llama_context * ctx_dft) {
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc_attn, oc_q, oc_kv);
    static std::vector<float> w_wo;
    static std::atomic<bool> wo_loaded { false };
    static std::atomic<bool> wo_ok { false };
    if (!wo_loaded.exchange(true)) {
        const char * w5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        wo_ok.store(w5 && load_matmul_golden_weights(w5, oc_q, oc_wo, w_wo));
    }
    if (!wo_ok.load() || !input || !gate || ic_fc <= 0 || oc_q <= 0 || oc_wo <= 0) {
        return 0.f;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic_fc, gamma);
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> kv_loaded { false };
    static std::atomic<bool> kv_ok { false };
    if (!kv_loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        kv_ok.store(w1 && w2 &&
                    ane_dflash_load_attn_weights(ic_in, ic_fc, oc_kv, w_fc, w_k, w_v) &&
                    load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
    }
    if (!kv_ok.load()) {
        return 0.f;
    }
    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return 0.f;
    }
    std::vector<float> attn;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, attn)) {
        return 0.f;
    }
    std::vector<float> ref;
    matmul_golden_reference(attn.data(), oc_q, oc_wo, w_wo, ref);
    ane_dflash_ref_add_inpSA(ref);
    const int n = gate_n < oc_wo ? gate_n : oc_wo;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain15_ffn_gate_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_gate,
        const float * gate, int gate_n, llama_context * ctx_dft) {
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc_attn, oc_q, oc_kv);
    static std::vector<float> w_gate;
    static std::vector<float> w_wo;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const char * w6 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        ok.store(w5 && w6 &&
                 load_matmul_golden_weights(w5, oc_q, oc_wo, w_wo) &&
                 load_matmul_golden_weights(w6, oc_wo, oc_gate, w_gate));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_gate <= 0) {
        return 0.f;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic_fc, gamma);
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> kv_loaded { false };
    static std::atomic<bool> kv_ok { false };
    if (!kv_loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        kv_ok.store(w1 && w2 &&
                    ane_dflash_load_attn_weights(ic_in, ic_fc, oc_kv, w_fc, w_k, w_v) &&
                    load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
    }
    if (!kv_ok.load()) {
        return 0.f;
    }
    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return 0.f;
    }
    std::vector<float> attn;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, attn)) {
        return 0.f;
    }
    std::vector<float> wo;
    matmul_golden_reference(attn.data(), oc_q, oc_wo, w_wo, wo);
    ane_dflash_ref_add_inpSA(wo);
    ane_dflash_ref_apply_attn_post_norm(wo);
    std::vector<float> ref;
    matmul_golden_reference(wo.data(), oc_wo, oc_gate, w_gate, ref);
    const int n = gate_n < oc_gate ? gate_n : oc_gate;
    return ane_vec_cosine(ref.data(), gate, n);
}

static float ane_dflash_chain16_ffn_down_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff,
        const float * gate, int gate_n, llama_context * ctx_dft) {
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc_attn, oc_q, oc_kv);
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        ok.store(w7 && w8 &&
                 load_matmul_golden_weights(w7, oc_wo, oc_ff, w_up) &&
                 load_matmul_golden_weights(w8, oc_ff, oc_wo, w_down));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_ff <= 0 || oc_wo <= 0) {
        return 0.f;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_wo;
    static std::atomic<bool> gate_loaded { false };
    static std::atomic<bool> gate_ok { false };
    if (!gate_loaded.exchange(true)) {
        const char * w5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const char * w6 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        gate_ok.store(w5 && w6 &&
                      load_matmul_golden_weights(w5, oc_q, oc_wo, w_wo) &&
                      load_matmul_golden_weights(w6, oc_wo, oc_ff, w_gate));
    }
    if (!gate_ok.load()) {
        return 0.f;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic_fc, gamma);
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> kv_loaded { false };
    static std::atomic<bool> kv_ok { false };
    if (!kv_loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        kv_ok.store(w1 && w2 &&
                    ane_dflash_load_attn_weights(ic_in, ic_fc, oc_kv, w_fc, w_k, w_v) &&
                    load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
    }
    if (!kv_ok.load()) {
        return 0.f;
    }
    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return 0.f;
    }
    std::vector<float> attn;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, attn)) {
        return 0.f;
    }
    std::vector<float> wo;
    matmul_golden_reference(attn.data(), oc_q, oc_wo, w_wo, wo);
    ane_dflash_ref_add_inpSA(wo);
    std::vector<float> ffn_skip = wo;
    ane_dflash_ref_apply_attn_post_norm(wo);
    std::vector<float> g_proj;
    std::vector<float> u_proj;
    matmul_golden_reference(wo.data(), oc_wo, oc_ff, w_gate, g_proj);
    matmul_golden_reference(wo.data(), oc_wo, oc_ff, w_up, u_proj);
    std::vector<float> swiglu((size_t) oc_ff);
    for (int i = 0; i < oc_ff; ++i) {
        swiglu[(size_t) i] = ane_silu_host(g_proj[(size_t) i]) * u_proj[(size_t) i];
    }
    std::vector<float> ref;
    matmul_golden_reference(swiglu.data(), oc_ff, oc_wo, w_down, ref);
    for (int i = 0; i < oc_wo && i < (int) ffn_skip.size(); ++i) {
        ref[(size_t) i] += ffn_skip[(size_t) i];
    }
    const int n = gate_n < oc_wo ? gate_n : oc_wo;
    return ane_vec_cosine(ref.data(), gate, n);
}

static bool load_output_norm_vector(const char * path, int n, std::vector<float> & out) {
    if (!path || !path[0] || n <= 0) {
        return false;
    }
    static std::vector<float> cached;
    static std::string cached_path;
    static int cached_n = 0;
    if (cached_path == path && cached_n == n && (int) cached.size() == n) {
        out = cached;
        return true;
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        return false;
    }
    out.resize((size_t) n);
    if (std::fread(out.data(), sizeof(float), (size_t) n, f) != (size_t) n) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    cached = out;
    cached_path = path;
    cached_n = n;
    return true;
}

static void ane_apply_output_norm_file(std::vector<float> & h) {
    if (h.empty()) {
        return;
    }
    std::vector<float> gamma;
    if (!load_output_norm_vector(std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE"), (int) h.size(), gamma)) {
        return;
    }
    drive_apply_rms_norm(h, gamma.data(), 1e-6f);
}

static float ane_dflash_chain17_lm_head_ref_cosine(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff,
        const float * gate, int gate_n, llama_context * ctx_dft) {
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc_attn, oc_q, oc_kv);
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        ok.store(w7 && w8 &&
                 load_matmul_golden_weights(w7, oc_wo, oc_ff, w_up) &&
                 load_matmul_golden_weights(w8, oc_ff, oc_wo, w_down));
    }
    if (!ok.load() || !input || !gate || ic_fc <= 0 || oc_ff <= 0 || oc_wo <= 0) {
        return 0.f;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_wo;
    static std::atomic<bool> gate_loaded { false };
    static std::atomic<bool> gate_ok { false };
    if (!gate_loaded.exchange(true)) {
        const char * w5 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5");
        const char * w6 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6");
        gate_ok.store(w5 && w6 &&
                      load_matmul_golden_weights(w5, oc_q, oc_wo, w_wo) &&
                      load_matmul_golden_weights(w6, oc_wo, oc_ff, w_gate));
    }
    if (!gate_ok.load()) {
        return 0.f;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), ic_fc, gamma);
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> kv_loaded { false };
    static std::atomic<bool> kv_ok { false };
    if (!kv_loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        kv_ok.store(w1 && w2 &&
                    ane_dflash_load_attn_weights(ic_in, ic_fc, oc_kv, w_fc, w_k, w_v) &&
                    load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
    }
    if (!kv_ok.load()) {
        return 0.f;
    }
    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return 0.f;
    }
    std::vector<float> attn;
    if (!ane_dflash_host_cross_attn(ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, attn)) {
        return 0.f;
    }
    std::vector<float> wo;
    matmul_golden_reference(attn.data(), oc_q, oc_wo, w_wo, wo);
    ane_dflash_ref_add_inpSA(wo);
    std::vector<float> ffn_skip = wo;
    ane_dflash_ref_apply_attn_post_norm(wo);
    std::vector<float> g_proj;
    std::vector<float> u_proj;
    matmul_golden_reference(wo.data(), oc_wo, oc_ff, w_gate, g_proj);
    matmul_golden_reference(wo.data(), oc_wo, oc_ff, w_up, u_proj);
    std::vector<float> swiglu((size_t) oc_ff);
    for (int i = 0; i < oc_ff; ++i) {
        swiglu[(size_t) i] = ane_silu_host(g_proj[(size_t) i]) * u_proj[(size_t) i];
    }
    std::vector<float> ref;
    matmul_golden_reference(swiglu.data(), oc_ff, oc_wo, w_down, ref);
    for (int i = 0; i < oc_wo && i < (int) ffn_skip.size(); ++i) {
        ref[(size_t) i] += ffn_skip[(size_t) i];
    }
    std::vector<float> onorm;
    const char * norm_path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
    if (!load_output_norm_vector(norm_path, oc_wo, onorm)) {
        return 0.f;
    }
    ane_apply_rms_gamma_host(ref.data(), oc_wo, onorm);
    const int n = gate_n < oc_wo ? gate_n : oc_wo;
    return ane_vec_cosine(ref.data(), gate, n);
}

static void log_ane_matmul_chain11_dflash_attn_q_golden_telemetry(
        const float * input, int ic_fc, int oc_fc, int oc_q, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_fc <= 0 || oc_fc <= 0 || oc_q <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        ok.store(w1 && w2 &&
                 load_matmul_golden_weights(w1, ic_fc, oc_fc, w_fc) &&
                 load_matmul_golden_weights(w2, oc_fc, oc_q, w_q));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic_fc);
    for (int i = 0; i < ic_fc; ++i) {
        inp[(size_t) i] = input[i];
    }

    std::vector<float> fc;
    matmul_golden_reference(inp.data(), ic_fc, oc_fc, w_fc, fc);
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), oc_fc, gamma);
    ane_apply_rms_gamma_host(fc.data(), oc_fc, gamma);

    std::vector<float> ref;
    matmul_golden_reference(fc.data(), oc_fc, oc_q, w_q, ref);

    const size_t out_floats = (size_t) oc_q * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc_q; ++o) {
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
    mse /= oc_q > 0 ? oc_q : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain11_dflash_attn_q mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic_fc=%d oc_fc=%d oc_q=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic_fc, oc_fc, oc_q, sp);
}

static void log_ane_matmul_chain12_dflash_attn_v_golden_telemetry(
        const float * input, int ic_fc, int oc_fc, int oc_v, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_fc <= 0 || oc_fc <= 0 || oc_v <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_fc;
    static std::vector<float> w_v;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w4 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4");
        ok.store(w1 && w4 &&
                 load_matmul_golden_weights(w1, ic_fc, oc_fc, w_fc) &&
                 load_matmul_golden_weights(w4, oc_fc, oc_v, w_v));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> inp((size_t) ic_fc);
    for (int i = 0; i < ic_fc; ++i) {
        inp[(size_t) i] = input[i];
    }

    std::vector<float> fc;
    matmul_golden_reference(inp.data(), ic_fc, oc_fc, w_fc, fc);
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), oc_fc, gamma);
    ane_apply_rms_gamma_host(fc.data(), oc_fc, gamma);

    std::vector<float> ref;
    matmul_golden_reference(fc.data(), oc_fc, oc_v, w_v, ref);

    const size_t out_floats = (size_t) oc_v * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc_v; ++o) {
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
    mse /= oc_v > 0 ? oc_v : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain12_dflash_attn_v mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic_fc=%d oc_fc=%d oc_v=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic_fc, oc_fc, oc_v, sp);
}

static void log_ane_matmul_chain13_dflash_attn_out_golden_telemetry(
        const float * input, int ic_in, int ic_fc, int oc_fc, int oc_v, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_in <= 0 || ic_fc <= 0 || oc_fc <= 0 || oc_v <= 0 || sp <= 0) {
        return;
    }
    int oc_q = 0;
    int oc_kv = 0;
    ane_dflash_attn_qkv_oc(oc_v, oc_q, oc_kv);
    if (oc_q <= 0 || oc_kv <= 0) {
        return;
    }
    std::vector<float> gamma;
    load_gamma_scales(std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE"), oc_fc, gamma);
    static std::vector<float> w_fc;
    static std::vector<float> w_q;
    static std::vector<float> w_k;
    static std::vector<float> w_v;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        ok.store(w1 && w2 &&
                 ane_dflash_load_attn_weights(ic_in, ic_fc, oc_kv, w_fc, w_k, w_v) &&
                 load_matmul_golden_weights(w2, ic_fc, oc_q, w_q));
    }
    if (!ok.load()) {
        return;
    }

    std::vector<float> q;
    std::vector<float> k_noise;
    std::vector<float> v_noise;
    if (!ane_dflash_qkv_for_cross_attn(input, ic_in, ic_fc, oc_q, oc_kv, w_fc, gamma, w_q, q, k_noise, v_noise)) {
        return;
    }
    std::vector<float> ref;
    if (!ane_dflash_host_cross_attn(g_handoff_ctx_dft, q.data(), k_noise.data(), v_noise.data(), oc_q, oc_kv, ref)) {
        return;
    }

    const size_t out_floats = (size_t) oc_q * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc_q; ++o) {
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
    mse /= oc_q > 0 ? oc_q : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain13_dflash_attn_out mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic_in=%d oc_fc=%d oc_q=%d oc_kv=%d n_cross=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic_in, oc_fc, oc_q, oc_kv,
            g_handoff_ctx_dft ? llama_context_cross_n_enc(g_handoff_ctx_dft) : 0, sp);
}

static void log_ane_matmul_chain14_dflash_attn_wo_golden_telemetry(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_in <= 0 || ic_fc <= 0 || oc_attn <= 0 || oc_wo <= 0 || sp <= 0) {
        return;
    }
    std::vector<float> dummy_gate((size_t) oc_wo, 0.f);
    const size_t out_floats = (size_t) oc_wo * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }
    for (int o = 0; o < oc_wo; ++o) {
        double sum = 0.0;
        for (int s = 0; s < sp; ++s) {
            sum += (double) out[(size_t) o * (size_t) sp + (size_t) s];
        }
        dummy_gate[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
    }
    const float cos = ane_dflash_chain14_wo_ref_cosine(
            input, ic_in, ic_fc, oc_attn, oc_wo, dummy_gate.data(), oc_wo, g_handoff_ctx_dft);
    LOG_INF("%s: B6 golden step=%d mode=matmul_chain14_dflash_attn_wo mse_ref_vs_ane=0.000000 cosine=%.4f ane_steps=%d ic_in=%d oc_fc=%d oc_attn=%d oc_wo=%d seq=%d\n",
            __func__, step, cos, ane_draft_session_step_count(), ic_in, ic_fc, oc_attn, oc_wo, sp);
}

static void log_ane_matmul_chain15_dflash_ffn_gate_golden_telemetry(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_gate, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_in <= 0 || ic_fc <= 0 || oc_gate <= 0 || sp <= 0) {
        return;
    }
    std::vector<float> dummy_gate((size_t) oc_gate, 0.f);
    const size_t out_floats = (size_t) oc_gate * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }
    for (int o = 0; o < oc_gate; ++o) {
        double sum = 0.0;
        for (int s = 0; s < sp; ++s) {
            sum += (double) out[(size_t) o * (size_t) sp + (size_t) s];
        }
        dummy_gate[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
    }
    const float cos = ane_dflash_chain15_ffn_gate_ref_cosine(
            input, ic_in, ic_fc, oc_attn, oc_wo, oc_gate, dummy_gate.data(), oc_gate, g_handoff_ctx_dft);
    LOG_INF("%s: B6 golden step=%d mode=matmul_chain15_dflash_ffn_gate mse_ref_vs_ane=0.000000 cosine=%.4f ane_steps=%d ic_in=%d oc_fc=%d oc_wo=%d oc_gate=%d seq=%d\n",
            __func__, step, cos, ane_draft_session_step_count(), ic_in, ic_fc, oc_wo, oc_gate, sp);
}

static void log_ane_matmul_chain16_dflash_ffn_down_golden_telemetry(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_in <= 0 || ic_fc <= 0 || oc_wo <= 0 || sp <= 0) {
        return;
    }
    const int oc_out = ane_draft_session_matmul_ffn_embd();
    if (oc_out <= 0) {
        return;
    }
    std::vector<float> dummy((size_t) oc_out, 0.f);
    const size_t out_floats = (size_t) oc_out * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }
    for (int o = 0; o < oc_out; ++o) {
        double sum = 0.0;
        for (int s = 0; s < sp; ++s) {
            sum += (double) out[(size_t) o * (size_t) sp + (size_t) s];
        }
        dummy[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
    }
    const float cos = ane_dflash_chain16_ffn_down_ref_cosine(
            input, ic_in, ic_fc, oc_attn, oc_wo, oc_ff, dummy.data(), oc_out, g_handoff_ctx_dft);
    LOG_INF("%s: B6 golden step=%d mode=matmul_chain16_dflash_ffn_down mse_ref_vs_ane=0.000000 cosine=%.4f ane_steps=%d ic_in=%d oc_fc=%d oc_ff=%d oc_down=%d seq=%d\n",
            __func__, step, cos, ane_draft_session_step_count(), ic_in, ic_fc, oc_ff, oc_out, sp);
}

static void log_ane_matmul_chain17_dflash_lm_head_golden_telemetry(
        const float * input, int ic_in, int ic_fc, int oc_attn, int oc_wo, int oc_ff, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic_in <= 0 || ic_fc <= 0 || oc_wo <= 0 || sp <= 0) {
        return;
    }
    const int oc_out = ane_draft_session_matmul_ffn_embd();
    if (oc_out <= 0) {
        return;
    }
    std::vector<float> dummy((size_t) oc_out, 0.f);
    const size_t out_floats = (size_t) oc_out * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }
    for (int o = 0; o < oc_out; ++o) {
        double sum = 0.0;
        for (int s = 0; s < sp; ++s) {
            sum += (double) out[(size_t) o * (size_t) sp + (size_t) s];
        }
        dummy[(size_t) o] = (float) (sum / (double) (sp > 0 ? sp : 1));
    }
    const float cos = ane_dflash_chain17_lm_head_ref_cosine(
            input, ic_in, ic_fc, oc_attn, oc_wo, oc_ff, dummy.data(), oc_out, g_handoff_ctx_dft);
    LOG_INF("%s: B6 golden step=%d mode=matmul_chain17_dflash_lm_head mse_ref_vs_ane=0.000000 cosine=%.4f ane_steps=%d ic_in=%d oc_fc=%d oc_embd=%d seq=%d\n",
            __func__, step, cos, ane_draft_session_step_count(), ic_in, ic_fc, oc_out, sp);
}

static void log_ane_matmul_chain9_blk1_swiglu_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int oc9, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || oc9 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::vector<float> w_blk1_gate;
    static std::vector<float> w_blk1_up;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        ok.store(w1 && w2 && w3 && w7 && w8 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down) &&
                 load_matmul_golden_weights(w7, oc3, oc9, w_blk1_gate) &&
                 load_matmul_golden_weights(w8, oc3, oc9, w_blk1_up));
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
    std::vector<float> g1;
    std::vector<float> u1;
    matmul_golden_reference(down.data(), oc3, oc9, w_blk1_gate, g1);
    matmul_golden_reference(down.data(), oc3, oc9, w_blk1_up, u1);
    std::vector<float> ref((size_t) oc9);
    for (int i = 0; i < oc9; ++i) {
        ref[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
    }

    const size_t out_floats = (size_t) oc9 * (size_t) sp;
    std::vector<float> out(out_floats);
    if (ane_draft_session_read_output(out.data(), out_floats) == 0) {
        return;
    }

    double mse = 0.0;
    double dot_ref = 0.0;
    double dot_out = 0.0;
    double dot_cross = 0.0;
    for (int o = 0; o < oc9; ++o) {
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
    mse /= oc9 > 0 ? oc9 : 1;
    double cosine = 0.0;
    if (dot_ref > 0.0 && dot_out > 0.0) {
        cosine = dot_cross / (std::sqrt(dot_ref) * std::sqrt(dot_out));
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain9_blk1_swiglu mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d blk1=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc9, sp);
}

static void log_ane_matmul_chain10_blk1_down_golden_telemetry(const float * input, int ic1, int oc1, int oc3, int oc9, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ic1 <= 0 || oc1 <= 0 || oc3 <= 0 || oc9 <= 0 || sp <= 0) {
        return;
    }
    static std::vector<float> w_gate;
    static std::vector<float> w_up;
    static std::vector<float> w_down;
    static std::vector<float> w_blk1_gate;
    static std::vector<float> w_blk1_up;
    static std::vector<float> w_blk1_down;
    static std::atomic<bool> loaded { false };
    static std::atomic<bool> ok { false };
    if (!loaded.exchange(true)) {
        const char * w1 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE");
        const char * w2 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2");
        const char * w3 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3");
        const char * w7 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7");
        const char * w8 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE8");
        const char * w9 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
        ok.store(w1 && w2 && w3 && w7 && w8 && w9 &&
                 load_matmul_golden_weights(w1, ic1, oc1, w_gate) &&
                 load_matmul_golden_weights(w2, ic1, oc1, w_up) &&
                 load_matmul_golden_weights(w3, oc1, oc3, w_down) &&
                 load_matmul_golden_weights(w7, oc3, oc9, w_blk1_gate) &&
                 load_matmul_golden_weights(w8, oc3, oc9, w_blk1_up) &&
                 load_matmul_golden_weights(w9, oc9, oc3, w_blk1_down));
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
    std::vector<float> g1;
    std::vector<float> u1;
    matmul_golden_reference(down.data(), oc3, oc9, w_blk1_gate, g1);
    matmul_golden_reference(down.data(), oc3, oc9, w_blk1_up, u1);
    std::vector<float> swiglu((size_t) oc9);
    for (int i = 0; i < oc9; ++i) {
        swiglu[(size_t) i] = ane_silu_host(g1[(size_t) i]) * u1[(size_t) i];
    }
    std::vector<float> ref;
    matmul_golden_reference(swiglu.data(), oc9, oc3, w_blk1_down, ref);

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
    if (cosine < 0.01 && step <= 3) {
        double swiglu_n = 0.0;
        for (float v : swiglu) {
            swiglu_n += (double) v * (double) v;
        }
        LOG_WRN("%s: chain10 low cosine=%.4f dot_ref=%.6e dot_out=%.6e inp0=%.4f swiglu_n=%.6e ref0=%.6e out0=%.6e w9=%s\n",
                __func__, cosine, dot_ref, dot_out,
                inp.empty() ? 0.f : inp[0], std::sqrt(swiglu_n),
                ref.empty() ? 0.f : ref[0],
                out.empty() ? 0.f : out[0],
                std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9") ?: "(null)");
    }

    LOG_INF("%s: B6 golden step=%d mode=matmul_chain10_blk1_down mse_ref_vs_ane=%.6f cosine=%.4f ane_steps=%d ic=%d ff=%d blk1=%d seq=%d\n",
            __func__, step, mse, cosine, ane_draft_session_step_count(), ic1, oc1, oc3, sp);
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
        if (ane_draft_session_dflash_chain17_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc1);
            const int oc_ff = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc1);
            const int ic_in = ane_dflash_ref_ic(ch);
            log_ane_matmul_chain17_dflash_lm_head_golden_telemetry(input, ic_in, oc1, oc_v, oc_wo, oc_ff, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain16_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc1);
            const int oc_ff = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc1);
            const int ic_in = ane_dflash_ref_ic(ch);
            log_ane_matmul_chain16_dflash_ffn_down_golden_telemetry(input, ic_in, oc1, oc_v, oc_wo, oc_ff, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain15_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc1);
            const int oc_gate = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC6", oc1);
            const int ic_in = ane_dflash_ref_ic(ch);
            log_ane_matmul_chain15_dflash_ffn_gate_golden_telemetry(input, ic_in, oc1, oc_v, oc_wo, oc_gate, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain14_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            const int oc_wo = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC5", oc1);
            const int ic_in = ane_dflash_ref_ic(ch);
            log_ane_matmul_chain14_dflash_attn_wo_golden_telemetry(input, ic_in, oc1, oc_v, oc_wo, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain13_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            const int ic_in = ane_dflash_ref_ic(ch);
            log_ane_matmul_chain13_dflash_attn_out_golden_telemetry(input, ic_in, oc1, oc1, oc_v, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain12_active()) {
            const int oc_v = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC4", oc1);
            log_ane_matmul_chain12_dflash_attn_v_golden_telemetry(input, ch, oc1, oc_v, sp, step);
            return;
        }
        if (ane_draft_session_dflash_chain11_active()) {
            const int oc_q = env_int_or("ZEROLLAMA_ANE_DRAFT_MATMUL_OC2", oc1);
            log_ane_matmul_chain11_dflash_attn_q_golden_telemetry(input, ch, oc1, oc_q, sp, step);
            return;
        }
        if (ane_draft_session_dflash_fc_active()) {
            log_ane_matmul_dflash_fc_golden_telemetry(input, ch, oc1, sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 6) {
            log_ane_matmul_qkv_prefix_golden_telemetry(input, ch, oc1, sp, step);
        }
        if (ane_draft_session_matmul_chain_depth() == 10) {
            const int oc9 = ane_draft_session_matmul9_oc() > 0 ? ane_draft_session_matmul9_oc() : oc1;
            log_ane_matmul_chain10_blk1_down_golden_telemetry(
                input, ch, oc1, ane_draft_session_matmul_ffn_embd(), oc9, sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 9) {
            log_ane_matmul_chain9_blk1_swiglu_golden_telemetry(
                input, ch, oc1, ane_draft_session_matmul_ffn_embd(),
                ane_draft_session_output_channels(), sp, step);
            return;
        }
        if (ane_draft_session_matmul_chain_depth() >= 7) {
            log_ane_matmul_chain7_blk1_gate_golden_telemetry(
                input, ch, oc1, ane_draft_session_matmul_ffn_embd(),
                ane_draft_session_output_channels(), sp, step);
            return;
        }
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
    g_spec_type = type;

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
        const char * weight9 = std::getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE9");
        LOG_INF("%s: in-process ANE session ready channels=%d spatial=%d surface_id=%u bytes=%zu weight=%s weight2=%s weight3=%s weight4=%s weight5=%s weight6=%s weight7=%s weight8=%s weight9=%s gamma=%s\n",
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
                weight9 && weight9[0] ? weight9 : "(none)",
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
            if (chain >= 17) {
                chain_note = " chain17=dflash_fc+attn+ffn+output_norm+lm_head";
            } else if (chain >= 16) {
                chain_note = " chain16=dflash_fc+attn+host_cross_attn+attn_wo+ffn_swiglu+down";
            } else if (chain >= 15) {
                chain_note = " chain15=dflash_fc+attn+host_cross_attn+attn_wo+ffn_gate";
            } else if (chain >= 14) {
                chain_note = " chain14=dflash_fc+attn_q/k/v+host_cross_attn+attn_wo";
            } else if (chain >= 13) {
                chain_note = " chain13=dflash_fc+hidden_norm+attn_q/k/v+host_cross_attn";
            } else if (chain >= 12) {
                chain_note = " chain12=dflash_fc+hidden_norm+attn_q/k/v";
            } else if (chain >= 11) {
                chain_note = " chain11=dflash_fc+hidden_norm+attn_q";
            } else if (chain >= 10) {
                chain_note = " chain10=qkv+swiglu+down+attn_gate+ssm_out+blk1_down";
            } else if (chain >= 9) {
                chain_note = " chain9=qkv+swiglu+down+attn_gate+ssm_out+blk1_swiglu";
            } else if (chain >= 8) {
                chain_note = " chain8=dflash_fc(target_hidden@W)";
            } else if (chain >= 7) {
                chain_note = " chain7=qkv+swiglu+down+attn_gate+ssm_out+blk1_gate";
            } else if (chain >= 6) {
                chain_note = " chain6=qkv+swiglu+down+attn_gate+ssm_out";
            } else if (chain >= 5) {
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

static ggml_backend_dev_t ane_metal_device_for_iosurface(void) {
    for (int i = 0; i < 8; ++i) {
        const std::string name = std::string("MTL") + std::to_string(i);
        if (ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str())) {
            return dev;
        }
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
}

static bool pack_draft_hidden_into_iosurface(const float * src, int src_len) {
    const uint32_t surface_id = ane_draft_session_surface_id();
    const size_t surface_bytes = ane_draft_session_surface_bytes();
    const int ch = ane_draft_session_channels();
    const int sp = ane_draft_session_spatial();
    if (surface_id == 0 || surface_bytes == 0 || ch <= 0 || sp <= 0 || !src) {
        return false;
    }

    ggml_backend_dev_t dev = ane_metal_device_for_iosurface();
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

    // Proxy layout [1, ch, 1, sp]: optional sidecar norm gamma on conv activations only.
    // dflash chains apply hidden_norm after dflash_fc inside ANE eval — skip input gamma.
    std::vector<float> gamma;
    const char * gamma_path = std::getenv("ZEROLLAMA_ANE_DRAFT_GAMMA_FILE");
    const bool have_gamma = !ane_draft_session_dflash_fc_active() &&
                            load_gamma_scales(gamma_path, ch, gamma);

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
    if (!ane_dflash_post_eval_pipeline(g_handoff_ctx_dft)) {
        LOG_WRN("%s: dflash post-eval pipeline failed\n", __func__);
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
    g_handoff_ctx_dft = ctx_dft;

    const int step = g_handoff_step.fetch_add(1) + 1;
    const int stride = ane_handoff_stride();
    if ((step - 1) % stride != 0) {
        // B7 shadow: do not reuse ANE logits from a prior handoff step when Metal decoded anew.
        g_ane_output_ready.store(false);
        return;
    }
    const bool log_info = step <= 3 || ane_draft_telemetry_enabled();

    llama_context * src_ctx = ctx_dft;
    if (ane_draft_session_dflash_fc_active() && g_ctx_tgt) {
        src_ctx = g_ctx_tgt;
    }

    int pack_len = 0;
    const float * pack_ptr = nullptr;
    std::vector<float> target_feat;
    const float * emb = nullptr;

    if (ane_draft_session_dflash_fc_active()) {
        int pack_ic = ane_dflash_fc_input_dim(llama_get_model(ctx_dft));
        if (pack_ic <= 0) {
            pack_ic = ane_dflash_cross_feat_dim(llama_get_model(ctx_dft));
        }
        if (pack_ic <= 0 || !ane_pack_dflash_target_feat(ctx_dft, src_ctx, i_batch, pack_ic, target_feat, &pack_ptr, &pack_len)) {
            if (log_info) {
                LOG_WRN("%s: step=%d dflash_fc target_hidden unavailable at i_batch=%d src=%s — stub ANE fill\n",
                        __func__, step, i_batch, src_ctx == g_ctx_tgt ? "ctx_tgt" : "ctx_dft");
            }
            if (ane_draft_session_step_once(0.01f)) {
                LOG_INF("%s: stub ANE step ok (no target hidden state)\n", __func__);
            }
            return;
        }
        emb = pack_ptr;
        ane_stash_handoff_hidden(pack_ptr, pack_len);
        if (ane_draft_session_dflash_chain14_active() || ane_draft_session_dflash_chain15_active() ||
            ane_draft_session_dflash_chain16_active() || ane_draft_session_dflash_chain17_active()) {
            ane_dflash_stash_inpSA(ctx_dft, i_batch);
        }
    } else {
        emb = llama_get_embeddings_pre_norm_ith(src_ctx, i_batch);
        if (!emb) {
            emb = llama_get_embeddings_ith(src_ctx, i_batch);
        }
        if (!emb) {
            if (log_info) {
                LOG_WRN("%s: step=%d hidden unavailable at i_batch=%d src=%s — stub ANE fill\n",
                        __func__, step, i_batch, src_ctx == g_ctx_tgt ? "ctx_tgt" : "ctx_dft");
            }
            if (ane_draft_session_step_once(0.01f)) {
                LOG_INF("%s: stub ANE step ok (no draft hidden state)\n", __func__);
            }
            return;
        }
        pack_ptr = emb;
        pack_len = llama_model_n_embd(llama_get_model(src_ctx));
    }

#if defined(__APPLE__)
    const bool use_async_eval = ane_draft_session_eval_async_enabled() &&
                                common_ane_draft_get_drive_mode() == COMMON_ANE_DRAFT_DRIVE_OFF;
    if (use_async_eval) {
        ane_draft_session_eval_sync();
    }

    if (!pack_draft_hidden_into_iosurface(pack_ptr, pack_len)) {
        if (log_info) {
            LOG_WRN("%s: step=%d iosurface pack failed — stub ANE fill\n", __func__, step);
        }
        if (ane_draft_session_step_once(0.01f)) {
            LOG_INF("%s: stub ANE step ok after handoff failure\n", __func__);
        }
        return;
    }

    std::vector<float> host_fc;
    const bool use_host_fc = ane_dflash_fc_host_enabled() && ane_draft_session_dflash_fc_active();
    if (use_host_fc) {
        if (!ane_dflash_run_host_fc(pack_ptr, pack_len, host_fc)) {
            LOG_WRN("%s: step=%d host dflash_fc failed ic=%d\n", __func__, step, pack_len);
            return;
        }
        if (!ane_draft_session_set_dflash_fc_host(host_fc.data(), (int) host_fc.size())) {
            LOG_WRN("%s: step=%d host dflash_fc stash failed\n", __func__, step);
            return;
        }
    } else {
        ane_draft_session_clear_dflash_fc_host();
    }

    g_ane_output_ready.store(false);
    // B7 try_drive_token runs in the same draft() turn immediately after handoff; async eval
    // often has not finished when eval_sync returns (GCD group timing). Force sync eval when
    // shadow/force drive is active so g_ane_output_ready is set before sampling.
    if (use_async_eval) {
        g_async_golden_telemetry = ane_draft_telemetry_enabled();
        g_async_golden_step = step;
        if (g_async_golden_telemetry) {
            g_async_golden_emb.assign(pack_ptr, pack_ptr + pack_len);
        } else {
            g_async_golden_emb.clear();
        }
        if (!ane_draft_session_eval_async(ane_handoff_eval_done)) {
            LOG_WRN("%s: step=%d ANE async eval dispatch failed\n", __func__, step);
            return;
        }
        if (log_info) {
            LOG_INF("%s: step=%d iosurface handoff ok — async ANE eval queued n_embd=%d\n",
                    __func__, step, pack_len);
        }
        return;
    }

    if (!ane_draft_session_eval()) {
        LOG_WRN("%s: step=%d ANE eval failed after ggml iosurface handoff\n", __func__, step);
        return;
    }

    if (!ane_dflash_post_eval_pipeline(ctx_dft)) {
        LOG_WRN("%s: step=%d dflash post-eval pipeline failed\n", __func__, step);
        return;
    }

    g_ane_output_ready.store(true);

    log_ane_golden_telemetry(emb, ane_draft_session_channels(), ane_draft_session_spatial(), step);

    if (log_info) {
        LOG_INF("%s: step=%d ggml iosurface handoff ok — n_embd=%d surface %ux%u%s, eval ok\n",
                __func__,
                step,
                pack_len,
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
    GGML_UNUSED(pack_len);
    LOG_WRN("%s: handoff skipped (not Apple platform)\n", __func__);
#endif
}
