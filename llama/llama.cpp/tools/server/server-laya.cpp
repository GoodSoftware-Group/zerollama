#include "server-laya.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

namespace {

// Worker threads for the CPU decision head (LLAMA_LAYA_THREADS overrides).
// WHY threads + batched linears: the head is a small Transformer run on CPU after the GPU
// encode; as per-token scalar GEMVs on one core it was 94-97% of /v1/decisions latency
// (~400 ms for 28 tokens on a Zen 4 box). See zerollama docs/auto-router-findings.md.
static int laya_threads() {
    static const int n = [] {
        if (const char * e = std::getenv("LLAMA_LAYA_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) {
                return v;
            }
        }
        const unsigned hc = std::thread::hardware_concurrency();
        return (int) std::max(1u, std::min(16u, hc ? hc / 2 : 4u));
    }();
    return n;
}

// laya_pool is a persistent worker pool for the head. WHY not std::thread per call: the head
// runs ~10 parallel sections per input, and spawning 16 threads for each cost more than the
// work (measured: 16 spawned threads 55 ms vs 4 threads 43 ms for a 91-token head).
class laya_pool {
public:
    explicit laya_pool(int n) : n_threads(std::max(1, n)) {
        for (int i = 1; i < n_threads; ++i) {
            workers.emplace_back([this, i] { loop(i); });
        }
    }
    ~laya_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            ++gen;
        }
        cv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }
    int size() const { return n_threads; }
    // run calls job(worker_index) on every worker (index 0 on the caller) and waits.
    void run(const std::function<void(int)> & job) {
        if (n_threads == 1) {
            job(0);
            return;
        }
        std::lock_guard<std::mutex> run_lk(run_mu); // one section at a time
        {
            std::lock_guard<std::mutex> lk(mu);
            cur     = &job;
            pending = n_threads - 1;
            ++gen;
        }
        cv.notify_all();
        job(0);
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [this] { return pending == 0; });
        cur = nullptr;
    }

private:
    void loop(int idx) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> * job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return gen != seen; });
                seen = gen;
                if (stop) {
                    return;
                }
                job = cur;
            }
            (*job)(idx);
            std::lock_guard<std::mutex> lk(mu);
            if (--pending == 0) {
                done_cv.notify_one();
            }
        }
    }

    const int                        n_threads;
    std::vector<std::thread>         workers;
    std::mutex                       mu, run_mu;
    std::condition_variable          cv, done_cv;
    const std::function<void(int)> * cur     = nullptr;
    int                              pending = 0;
    uint64_t                         gen     = 0;
    bool                             stop    = false;
};

static laya_pool & laya_workers() {
    static laya_pool pool(laya_threads());
    return pool;
}

// parallel_for splits [0, n) into contiguous chunks; fn(begin, end) runs on each worker.
template <typename F>
static void parallel_for(int n, F && fn) {
    laya_pool & pool = laya_workers();
    const int nt = std::min(pool.size(), n);
    if (nt <= 1) {
        fn(0, n);
        return;
    }
    const int chunk = (n + nt - 1) / nt;
    pool.run([&](int w) {
        const int b = w * chunk;
        const int e = std::min(n, b + chunk);
        if (b < e) {
            fn(b, e);
        }
    });
}

// dot_f32 keeps 16 independent accumulators so the compiler can vectorize without
// -ffast-math (each lane is its own sum; only the final reduction order changes).
static inline float dot_f32(const float * a, const float * b, int64_t n) {
    constexpr int L = 16;
    float acc[L] = {};
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        for (int j = 0; j < L; ++j) {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    float s = 0.0f;
    for (int j = 0; j < L; ++j) {
        s += acc[j];
    }
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

static bool laya_timing() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_LAYA_TIMING");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

static bool tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        return false;
    }

    const int64_t ne = ggml_nelements(t);
    out.resize((size_t) ne);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t) ne * sizeof(float));
        return true;
    }

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out.data(), ne);
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out.data(), ne);
    } else if (ggml_is_quantized(t->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out.data(), ne);
    } else {
        return false;
    }
    return true;
}

static bool load_linear(const llama_model * model, const char * wname, const char * bname, laya_linear & out) {
    ggml_tensor * tw = llama_model_get_tensor(model, wname);
    if (tw == nullptr || tw->ne[0] <= 0 || tw->ne[1] <= 0) {
        return false;
    }
    out.n_in  = tw->ne[0];
    out.n_out = tw->ne[1];
    if (!tensor_to_f32(tw, out.weight)) {
        return false;
    }
    out.bias.clear();
    if (bname != nullptr) {
        ggml_tensor * tb = llama_model_get_tensor(model, bname);
        if (tb != nullptr) {
            if (!tensor_to_f32(tb, out.bias) || (int64_t) out.bias.size() != out.n_out) {
                return false;
            }
        }
    }
    return true;
}

static bool load_ln(const llama_model * model, const char * wname, const char * bname, laya_ln & out) {
    ggml_tensor * tw = llama_model_get_tensor(model, wname);
    if (tw == nullptr) {
        return false;
    }
    if (!tensor_to_f32(tw, out.weight)) {
        return false;
    }
    out.bias.clear();
    if (bname != nullptr) {
        ggml_tensor * tb = llama_model_get_tensor(model, bname);
        if (tb != nullptr && !tensor_to_f32(tb, out.bias)) {
            return false;
        }
    }
    return true;
}

static void linear_forward(const laya_linear & L, const float * x, float * y) {
    for (int64_t o = 0; o < L.n_out; ++o) {
        const float b = L.bias.empty() ? 0.0f : L.bias[(size_t) o];
        y[o] = b + dot_f32(L.weight.data() + o * L.n_in, x, L.n_in);
    }
}

// linear_forward_batch: Y[t] = W·X[t] + b for all tokens. Threads split output rows and each
// row is reused across every token while it is hot in L1 (W is read once per call, not once
// per token).
static void linear_forward_batch(const laya_linear & L, const float * X, int n_tokens, float * Y) {
    parallel_for((int) L.n_out, [&](int ob, int oe) {
        for (int o = ob; o < oe; ++o) {
            const float * w = L.weight.data() + (size_t) o * L.n_in;
            const float   b = L.bias.empty() ? 0.0f : L.bias[(size_t) o];
            for (int t = 0; t < n_tokens; ++t) {
                Y[(size_t) t * L.n_out + o] = b + dot_f32(w, X + (size_t) t * L.n_in, L.n_in);
            }
        }
    });
}

static void layernorm(const laya_ln & ln, const float * x, float * y, int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= (float) n;

    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= (float) n;
    const float inv = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < n; ++i) {
        float v = (x[i] - mean) * inv;
        if (!ln.weight.empty()) {
            v *= ln.weight[(size_t) i];
        }
        if (!ln.bias.empty()) {
            v += ln.bias[(size_t) i];
        }
        y[i] = v;
    }
}

static float gelu(float x) {
    // tanh approximation
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
}

static void softmax_inplace(std::vector<float> & x) {
    if (x.empty()) {
        return;
    }
    float m = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        m = std::max(m, x[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] *= inv;
    }
}

static void mha_forward(
        const float * x, // [T, d] after LN
        const laya_linear & qkv,
        const laya_linear & out_proj,
        int n_tokens,
        int n_embd,
        int n_head,
        std::vector<float> & y) { // [T, d]
    const int head_dim = n_embd / n_head;
    GGML_ASSERT(head_dim * n_head == n_embd);
    GGML_ASSERT(qkv.n_in == n_embd && qkv.n_out == 3 * n_embd);
    GGML_ASSERT(out_proj.n_in == n_embd && out_proj.n_out == n_embd);

    std::vector<float> qkv_buf((size_t) n_tokens * (size_t) qkv.n_out);
    linear_forward_batch(qkv, x, n_tokens, qkv_buf.data());

    const float scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> attn_out((size_t) n_tokens * (size_t) n_embd, 0.0f);

    parallel_for(n_head, [&](int hb, int he) {
    std::vector<float> scores((size_t) n_tokens);
    for (int h = hb; h < he; ++h) {
        for (int t = 0; t < n_tokens; ++t) {
            const float * q = qkv_buf.data() + (size_t) t * qkv.n_out + (size_t) h * head_dim;
            float m = -INFINITY;
            for (int s = 0; s < n_tokens; ++s) {
                const float * k = qkv_buf.data() + (size_t) s * qkv.n_out + n_embd + (size_t) h * head_dim;
                scores[(size_t) s] = dot_f32(q, k, head_dim) * scale;
                m = std::max(m, scores[(size_t) s]);
            }
            float sum = 0.0f;
            for (int s = 0; s < n_tokens; ++s) {
                scores[(size_t) s] = std::exp(scores[(size_t) s] - m);
                sum += scores[(size_t) s];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            float * o = attn_out.data() + (size_t) t * n_embd + (size_t) h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                o[d] = 0.0f;
            }
            for (int s = 0; s < n_tokens; ++s) {
                const float a = scores[(size_t) s] * inv;
                const float * v = qkv_buf.data() + (size_t) s * qkv.n_out + 2 * n_embd + (size_t) h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    o[d] += a * v[d];
                }
            }
        }
    }
    });

    y.resize((size_t) n_tokens * (size_t) n_embd);
    linear_forward_batch(out_proj, attn_out.data(), n_tokens, y.data());
}


} // namespace

bool laya_model_has_head(const llama_model * model) {
    return model != nullptr && llama_model_get_tensor(model, "laya.type_embd.weight") != nullptr;
}

bool laya_head_load(const llama_model * model, laya_head_weights & out, std::string & err) {
    out = laya_head_weights{};
    if (model == nullptr) {
        err = "null model";
        return false;
    }

    out.n_embd = llama_model_n_embd(model);
    out.n_head = std::max(1, llama_model_n_head(model));
    if (out.n_embd % out.n_head != 0) {
        out.n_head = 1;
    }

    ggml_tensor * type_t = llama_model_get_tensor(model, "laya.type_embd.weight");
    if (type_t == nullptr) {
        err = "missing laya.type_embd.weight";
        return false;
    }
    if (type_t->ne[0] != out.n_embd || type_t->ne[1] < 1) {
        err = "invalid laya.type_embd.weight shape";
        return false;
    }
    if (!tensor_to_f32(type_t, out.type_embd)) {
        err = "failed to read laya.type_embd.weight";
        return false;
    }

    // Count head layers by probing attn_qkv weights.
    out.n_head_layers = 0;
    for (int i = 0; i < 64; ++i) {
        const std::string name = string_format("laya.head.%d.attn_qkv.weight", i);
        if (llama_model_get_tensor(model, name.c_str()) == nullptr) {
            break;
        }
        out.n_head_layers++;
    }
    if (out.n_head_layers <= 0) {
        err = "no laya.head.*.attn_qkv.weight tensors";
        return false;
    }

    out.layers.resize((size_t) out.n_head_layers);
    for (int i = 0; i < out.n_head_layers; ++i) {
        auto & layer = out.layers[(size_t) i];
        const std::string p = string_format("laya.head.%d.", i);
        if (!load_ln(model, (p + "attn_norm.weight").c_str(), (p + "attn_norm.bias").c_str(), layer.attn_norm) ||
            !load_linear(model, (p + "attn_qkv.weight").c_str(), (p + "attn_qkv.bias").c_str(), layer.attn_qkv) ||
            !load_linear(model, (p + "attn_out.weight").c_str(), (p + "attn_out.bias").c_str(), layer.attn_out) ||
            !load_ln(model, (p + "ffn_norm.weight").c_str(), (p + "ffn_norm.bias").c_str(), layer.ffn_norm) ||
            !load_linear(model, (p + "ffn_up.weight").c_str(), (p + "ffn_up.bias").c_str(), layer.ffn_up) ||
            !load_linear(model, (p + "ffn_down.weight").c_str(), (p + "ffn_down.bias").c_str(), layer.ffn_down)) {
            err = string_format("failed to load laya.head.%d tensors", i);
            return false;
        }
    }

    if (!load_ln(model, "laya.scorer.norm.weight", "laya.scorer.norm.bias", out.scorer_norm) ||
        !load_linear(model, "laya.scorer.fc1.weight", "laya.scorer.fc1.bias", out.scorer_fc1) ||
        !load_linear(model, "laya.scorer.fc2.weight", "laya.scorer.fc2.bias", out.scorer_fc2)) {
        err = "failed to load laya.scorer.* tensors";
        return false;
    }

    if (!load_linear(model, "laya.act.fc1.weight", "laya.act.fc1.bias", out.act_fc1) ||
        !load_linear(model, "laya.act.fc2.weight", "laya.act.fc2.bias", out.act_fc2)) {
        err = "failed to load laya.act.* tensors";
        return false;
    }
    out.n_act = (int) out.act_fc2.n_out;
    if (out.n_act <= 0) {
        err = "invalid act head output dim";
        return false;
    }

    // Optional eps from GGUF metadata
    char buf[64];
    if (llama_model_meta_val_str(model, "laya.norm_eps", buf, sizeof(buf)) >= 0 ||
        llama_model_meta_val_str(model, "attention.layernorm_eps", buf, sizeof(buf)) >= 0) {
        try {
            out.eps = std::stof(buf);
        } catch (...) {
            // keep default
        }
    }

    return true;
}

bool laya_encode_embeddings(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        std::vector<float> & embd_out,
        std::string & err) {
    if (ctx == nullptr) {
        err = "null context";
        return false;
    }
    if (tokens.empty()) {
        err = "empty tokens";
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    const int n_embd = llama_model_n_embd(model);
    const int n_tokens = (int) tokens.size();
    const int n_ubatch = (int) llama_n_ubatch(ctx);

    if (n_tokens > n_ubatch) {
        err = string_format("input (%d tokens) exceeds n_ubatch (%d); increase -ub", n_tokens, n_ubatch);
        return false;
    }
    if (n_tokens > (int) llama_n_ctx(ctx)) {
        err = string_format("input (%d tokens) exceeds n_ctx (%u)", n_tokens, llama_n_ctx(ctx));
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    llama_set_embeddings(ctx, true);

    // tip b11351+: common_batch + llama_process (common_batch_add removed)
    common_batch batch(ctx);
    for (int i = 0; i < n_tokens; ++i) {
        batch.add(tokens[(size_t) i], i, /*seq_id*/ 0, /*output*/ true);
    }

    const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) {
        err = string_format("llama_process failed with code %d", rc);
        return false;
    }

    embd_out.resize((size_t) n_tokens * (size_t) n_embd);
    for (int i = 0; i < n_tokens; ++i) {
        float * e = llama_get_embeddings_ith(ctx, i);
        if (e == nullptr) {
            err = string_format("llama_get_embeddings_ith failed at token %d", i);
            return false;
        }
        std::memcpy(embd_out.data() + (size_t) i * n_embd, e, (size_t) n_embd * sizeof(float));
    }

    return true;
}

bool laya_head_apply(
        const laya_head_weights & w,
        const float * H_in,
        int n_tokens,
        const laya_decision_input & input,
        laya_decision_result & out,
        std::string & err) {
    out = laya_decision_result{};
    out.n_tokens = n_tokens;

    if (H_in == nullptr || n_tokens <= 0) {
        err = "empty embeddings";
        return false;
    }
    if (input.marker_pos.empty()) {
        err = "marker_pos must be non-empty";
        return false;
    }
    if (input.qtype < 0 || input.qtype > 2) {
        err = "qtype must be 0..2";
        return false;
    }

    const int n_embd = w.n_embd;
    const size_t type_cols = w.type_embd.size() / (size_t) n_embd;
    if (type_cols < 3 || (size_t) input.qtype >= type_cols) {
        err = "type_embd missing qtype column";
        return false;
    }

    for (int32_t p : input.marker_pos) {
        if (p < 0 || p >= n_tokens) {
            err = string_format("marker_pos %d out of range for n_tokens=%d", (int) p, n_tokens);
            return false;
        }
    }

    // Working copy of H
    std::vector<float> H((size_t) n_tokens * (size_t) n_embd);
    std::memcpy(H.data(), H_in, H.size() * sizeof(float));

    // a. H[t] += type_embd[qtype]
    const float * te = w.type_embd.data() + (size_t) input.qtype * n_embd;
    for (int t = 0; t < n_tokens; ++t) {
        float * row = H.data() + (size_t) t * n_embd;
        for (int i = 0; i < n_embd; ++i) {
            row[i] += te[i];
        }
    }

    // b. head layers: pre-LN MHA + ReLU FFN
    std::vector<float> tmp_ln((size_t) n_embd);
    std::vector<float> tmp_attn;
    std::vector<float> tmp_ffn;

    for (const auto & layer : w.layers) {
        std::vector<float> ln_all((size_t) n_tokens * (size_t) n_embd);
        for (int t = 0; t < n_tokens; ++t) {
            layernorm(layer.attn_norm, H.data() + (size_t) t * n_embd, ln_all.data() + (size_t) t * n_embd, n_embd, w.eps);
        }
        mha_forward(ln_all.data(), layer.attn_qkv, layer.attn_out, n_tokens, n_embd, w.n_head, tmp_attn);
        for (int t = 0; t < n_tokens; ++t) {
            float * row = H.data() + (size_t) t * n_embd;
            const float * a = tmp_attn.data() + (size_t) t * n_embd;
            for (int i = 0; i < n_embd; ++i) {
                row[i] += a[i];
            }
        }

        // FFN over all tokens at once (batched up / down projections).
        for (int t = 0; t < n_tokens; ++t) {
            layernorm(layer.ffn_norm, H.data() + (size_t) t * n_embd, ln_all.data() + (size_t) t * n_embd, n_embd, w.eps);
        }
        std::vector<float> mid((size_t) n_tokens * (size_t) layer.ffn_up.n_out);
        linear_forward_batch(layer.ffn_up, ln_all.data(), n_tokens, mid.data());
        for (float & v : mid) {
            v = v > 0.0f ? v : 0.0f;
        }
        tmp_ffn.resize((size_t) n_tokens * (size_t) n_embd);
        linear_forward_batch(layer.ffn_down, mid.data(), n_tokens, tmp_ffn.data());
        for (size_t i = 0; i < H.size(); ++i) {
            H[i] += tmp_ffn[i];
        }
    }

    // c+d. scorer on marker positions
    const int k = (int) input.marker_pos.size();
    out.logits.assign((size_t) k, 0.0f);
    std::vector<float> sc_mid((size_t) w.scorer_fc1.n_out);
    std::vector<float> sc_out(1);

    for (int i = 0; i < k; ++i) {
        const float * hm = H.data() + (size_t) input.marker_pos[(size_t) i] * n_embd;
        layernorm(w.scorer_norm, hm, tmp_ln.data(), n_embd, w.eps);
        linear_forward(w.scorer_fc1, tmp_ln.data(), sc_mid.data());
        for (size_t j = 0; j < sc_mid.size(); ++j) {
            sc_mid[j] = gelu(sc_mid[j]);
        }
        linear_forward(w.scorer_fc2, sc_mid.data(), sc_out.data());
        out.logits[(size_t) i] = sc_out[0];
    }

    // e. softmax + dist features (mask unused with -1e4 — N/A when only real markers)
    std::vector<float> probs = out.logits;
    // If caller pads beyond k, they would send fewer markers; we only have k.
    softmax_inplace(probs);

    float top1 = 0.0f;
    float top2 = 0.0f;
    for (float p : probs) {
        if (p >= top1) {
            top2 = top1;
            top1 = p;
        } else if (p > top2) {
            top2 = p;
        }
    }
    float ent = 0.0f;
    for (float p : probs) {
        if (p > 0.0f) {
            ent -= p * std::log(p);
        }
    }
    const float logk = k > 1 ? std::log((float) k) : 1.0f;
    const float feats[4] = {
        top1,
        top1 - top2,
        ent / logk,
        (float) k / 255.0f,
    };

    // f. act_head on concat(H[0], feats)
    std::vector<float> act_in((size_t) n_embd + 4);
    std::memcpy(act_in.data(), H.data(), (size_t) n_embd * sizeof(float));
    for (int i = 0; i < 4; ++i) {
        act_in[(size_t) n_embd + i] = feats[i];
    }
    if ((int64_t) act_in.size() != w.act_fc1.n_in) {
        err = string_format("act_fc1 expects in=%lld got %zu", (long long) w.act_fc1.n_in, act_in.size());
        return false;
    }

    std::vector<float> act_hid((size_t) w.act_fc1.n_out);
    linear_forward(w.act_fc1, act_in.data(), act_hid.data());
    for (size_t i = 0; i < act_hid.size(); ++i) {
        act_hid[i] = gelu(act_hid[i]);
    }
    out.act.assign((size_t) w.n_act, 0.0f);
    linear_forward(w.act_fc2, act_hid.data(), out.act.data());
    softmax_inplace(out.act);

    return true;
}

// laya_encode_embeddings_batch encodes several inputs with as few llama_process calls as
// possible: each input is its own sequence (seq_id 0..n-1, positions restart at 0), chunked by
// llama_n_seq_max and n_ubatch. WHY: per-question encodes paid the encode + memory-clear cost
// once per question (multi-question decisions, e.g. router jev batches, paid it 3-4x).
bool laya_encode_embeddings_batch(
        llama_context * ctx,
        const std::vector<laya_decision_input> & inputs,
        std::vector<std::vector<float>> & embd_out,
        std::string & err) {
    if (ctx == nullptr) {
        err = "null context";
        return false;
    }
    const llama_model * model = llama_get_model(ctx);
    const int n_embd    = llama_model_n_embd(model);
    const int n_ubatch  = (int) llama_n_ubatch(ctx);
    const int n_seq_max = std::max(1, (int) llama_n_seq_max(ctx));
    embd_out.assign(inputs.size(), {});

    size_t next = 0;
    while (next < inputs.size()) {
        // greedy chunk: as many whole inputs as fit in one ubatch / seq budget
        size_t end = next;
        int tokens = 0;
        while (end < inputs.size() && (int) (end - next) < n_seq_max &&
               tokens + (int) inputs[end].tokens.size() <= n_ubatch) {
            tokens += (int) inputs[end].tokens.size();
            ++end;
        }
        if (end == next) { // single oversize input: let the single-input path report it
            if (!laya_encode_embeddings(ctx, inputs[next].tokens, embd_out[next], err)) {
                return false;
            }
            ++next;
            continue;
        }
        if (end - next == 1) {
            if (!laya_encode_embeddings(ctx, inputs[next].tokens, embd_out[next], err)) {
                return false;
            }
            ++next;
            continue;
        }

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        llama_set_embeddings(ctx, true);
        common_batch batch(ctx);
        for (size_t k = next; k < end; ++k) {
            const auto & toks = inputs[k].tokens;
            if (toks.empty()) {
                err = "empty tokens";
                return false;
            }
            for (size_t i = 0; i < toks.size(); ++i) {
                batch.add(toks[i], (llama_pos) i, /*seq_id*/ (llama_seq_id) (k - next), /*output*/ true);
            }
        }
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            err = string_format("llama_process failed with code %d", rc);
            return false;
        }
        int idx = 0;
        for (size_t k = next; k < end; ++k) {
            const int n = (int) inputs[k].tokens.size();
            embd_out[k].resize((size_t) n * (size_t) n_embd);
            for (int i = 0; i < n; ++i, ++idx) {
                float * e = llama_get_embeddings_ith(ctx, idx);
                if (e == nullptr) {
                    err = string_format("llama_get_embeddings_ith failed at token %d", idx);
                    return false;
                }
                std::memcpy(embd_out[k].data() + (size_t) i * n_embd, e, (size_t) n_embd * sizeof(float));
            }
        }
        next = end;
    }
    return true;
}

bool laya_decide_batch(
        llama_context * ctx,
        const laya_head_weights & w,
        const std::vector<laya_decision_input> & inputs,
        std::vector<laya_decision_result> & out,
        std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> embd;
    if (!laya_encode_embeddings_batch(ctx, inputs, embd, err)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    out.assign(inputs.size(), {});
    int total = 0;
    for (size_t k = 0; k < inputs.size(); ++k) {
        total += (int) inputs[k].tokens.size();
        if (!laya_head_apply(w, embd[k].data(), (int) inputs[k].tokens.size(), inputs[k], out[k], err)) {
            return false;
        }
    }
    if (laya_timing()) {
        const auto t2 = std::chrono::steady_clock::now();
        LOG_INF("laya decisions: inputs=%zu tokens=%d threads=%d encode_ms=%.2f head_ms=%.2f\n", inputs.size(), total,
                laya_threads(), std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    return true;
}

bool laya_decide(
        llama_context * ctx,
        const laya_head_weights & w,
        const laya_decision_input & input,
        laya_decision_result & out,
        std::string & err) {
    std::vector<float> embd;
    if (!laya_encode_embeddings(ctx, input.tokens, embd, err)) {
        return false;
    }
    return laya_head_apply(w, embd.data(), (int) input.tokens.size(), input, out, err);
}
