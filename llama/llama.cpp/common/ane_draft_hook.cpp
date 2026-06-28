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

static std::atomic<bool> g_ane_output_ready { false };
static std::atomic<int>  g_handoff_step { 0 };
#endif

void common_ane_draft_reset_handoff(void) {
#if defined(__APPLE__)
    g_ane_output_ready.store(false);
    g_handoff_step.store(0);
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

bool common_ane_draft_try_drive_token(struct llama_context * ctx_dft, int32_t /*i_batch*/, llama_token * out_id, float * out_p) {
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

    const char * embd_path = std::getenv("ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE");
    const char * norm_path = std::getenv("ZEROLLAMA_ANE_DRAFT_OUTPUT_NORM_FILE");
    if (!drive_head_load(embd_path, norm_path)) {
        return false;
    }

    const int ch = ane_draft_session_channels();
    const int sp = ane_draft_session_spatial();
    const size_t nfloats = (size_t) ch * (size_t) sp;
    std::vector<float> ane_out(nfloats);
    if (ane_draft_session_read_output(ane_out.data(), nfloats) == 0) {
        return false;
    }

    const int n_embd = g_drive_head.n_embd;
    std::vector<float> h((size_t) n_embd, 0.f);
    for (int c = 0; c < ch && c < n_embd; ++c) {
        double sum = 0.0;
        for (int s = 0; s < sp; ++s) {
            sum += (double) ane_out[(size_t) c * (size_t) sp + (size_t) s];
        }
        h[(size_t) c] = (float) (sum / (double) (sp > 0 ? sp : 1));
    }

    const float eps = 1e-6f;
    drive_apply_rms_norm(h, g_drive_head.out_norm, eps);

    const llama_token id = drive_argmax_tied(h);
    *out_id = id;
    *out_p  = 0.95f; // lab proxy confidence — not full softmax over vocab
    GGML_UNUSED(ctx_dft);
    return true;
}
#else
bool common_ane_draft_try_drive_token(struct llama_context *, int32_t, llama_token *, float *) {
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

static void log_ane_golden_telemetry(const float * input, int ch, int sp, int step) {
    if (!ane_draft_telemetry_enabled() || !input || ch <= 0 || sp <= 0) {
        return;
    }

    static std::vector<float> golden_w;
    static std::vector<float> golden_w2;
    static std::vector<float> golden_w3;
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
    if (!golden_w2.empty()) {
        conv_golden_reference(mid.data(), ch, golden_w2, ref);
    } else {
        ref = mid;
    }
    if (!golden_w3.empty()) {
        conv_golden_reference(ref.data(), ch, golden_w3, mid);
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

    const char * mode = golden_w3.empty() ? (ane_draft_session_using_conv2() ? "conv2" : "conv1") : "conv3";
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
        LOG_INF("%s: in-process ANE session ready channels=%d spatial=%d surface_id=%u bytes=%zu weight=%s weight2=%s weight3=%s gamma=%s\n",
                __func__,
                ane_draft_session_channels(),
                ane_draft_session_spatial(),
                ane_draft_session_surface_id(),
                ane_draft_session_surface_bytes(),
                weight && weight[0] ? weight : "(synthetic)",
                weight2 && weight2[0] ? weight2 : "(none)",
                weight3 && weight3[0] ? weight3 : "(none)",
                gamma && gamma[0] ? gamma : "(none)");
        if (weight3 && weight3[0] && ane_draft_session_using_conv2()) {
            LOG_INF("%s: B8 triple conv1 chain active (WEIGHT_FILE2 + WEIGHT_FILE3)\n", __func__);
        } else if (ane_draft_session_using_conv2()) {
            LOG_INF("%s: B6 dual conv1 chain active (WEIGHT_FILE2)\n", __func__);
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
        LOG_INF("%s: B7 drive mode=%s — TOKEN_EMBD_FILE + OUTPUT_NORM_FILE for tied-embed argmax\n",
                __func__,
                drive == COMMON_ANE_DRAFT_DRIVE_FORCE ? "force" : "shadow");
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
    if (!pack_draft_hidden_into_iosurface(emb, n_embd)) {
        if (log_info) {
            LOG_WRN("%s: step=%d iosurface pack failed — stub ANE fill\n", __func__, step);
        }
        if (ane_draft_session_step_once(0.01f)) {
            LOG_INF("%s: stub ANE step ok after handoff failure\n", __func__);
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
