#include "server-laya.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

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
        float sum = L.bias.empty() ? 0.0f : L.bias[(size_t) o];
        const float * col = L.weight.data() + o * L.n_in;
        for (int64_t i = 0; i < L.n_in; ++i) {
            sum += col[i] * x[i];
        }
        y[o] = sum;
    }
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
    for (int t = 0; t < n_tokens; ++t) {
        linear_forward(qkv, x + (size_t) t * n_embd, qkv_buf.data() + (size_t) t * qkv.n_out);
    }

    const float scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> attn_out((size_t) n_tokens * (size_t) n_embd, 0.0f);
    std::vector<float> scores((size_t) n_tokens);

    for (int h = 0; h < n_head; ++h) {
        for (int t = 0; t < n_tokens; ++t) {
            const float * q = qkv_buf.data() + (size_t) t * qkv.n_out + (size_t) h * head_dim;
            float m = -INFINITY;
            for (int s = 0; s < n_tokens; ++s) {
                const float * k = qkv_buf.data() + (size_t) s * qkv.n_out + n_embd + (size_t) h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q[d] * k[d];
                }
                scores[(size_t) s] = dot * scale;
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

    y.resize((size_t) n_tokens * (size_t) n_embd);
    for (int t = 0; t < n_tokens; ++t) {
        linear_forward(out_proj, attn_out.data() + (size_t) t * n_embd, y.data() + (size_t) t * n_embd);
    }
}

static void ffn_relu_forward(
        const float * x,
        const laya_linear & up,
        const laya_linear & down,
        int n_embd,
        std::vector<float> & y) {
    std::vector<float> mid((size_t) up.n_out);
    linear_forward(up, x, mid.data());
    for (size_t i = 0; i < mid.size(); ++i) {
        mid[i] = mid[i] > 0.0f ? mid[i] : 0.0f;
    }
    y.resize((size_t) n_embd);
    linear_forward(down, mid.data(), y.data());
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
    if (llama_model_meta_val_str(model, "laya.attention.layer_norm_epsilon", buf, sizeof(buf)) >= 0 ||
        llama_model_meta_val_str(model, "laya.norm_eps", buf, sizeof(buf)) >= 0) {
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

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[(size_t) i], i, {0}, /*logits*/ true);
    }

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        err = string_format("llama_decode failed with code %d", rc);
        return false;
    }

    embd_out.resize((size_t) n_tokens * (size_t) n_embd);
    for (int i = 0; i < n_tokens; ++i) {
        float * e = llama_get_embeddings_ith(ctx, i);
        if (e == nullptr) {
            llama_batch_free(batch);
            err = string_format("llama_get_embeddings_ith failed at token %d", i);
            return false;
        }
        std::memcpy(embd_out.data() + (size_t) i * n_embd, e, (size_t) n_embd * sizeof(float));
    }

    llama_batch_free(batch);
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

        for (int t = 0; t < n_tokens; ++t) {
            float * row = H.data() + (size_t) t * n_embd;
            layernorm(layer.ffn_norm, row, tmp_ln.data(), n_embd, w.eps);
            ffn_relu_forward(tmp_ln.data(), layer.ffn_up, layer.ffn_down, n_embd, tmp_ffn);
            for (int i = 0; i < n_embd; ++i) {
                row[i] += tmp_ffn[(size_t) i];
            }
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
