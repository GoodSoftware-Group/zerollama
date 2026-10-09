// Strands Decider pointer head (Apache-2.0 algorithm from strands_decider /
// zerollama x/models/strands). Consumes per-token hidden states from llama-server.
#include "strands.h"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

namespace {
void require(bool ok, const char * msg) {
    if (!ok) {
        throw std::runtime_error(msg);
    }
}

struct vec {
    int n = 0;
    std::vector<float> data;
    explicit vec(int n_ = 0) : n(n_), data(size_t(n_)) {}
    float & operator[](int i) { return data[size_t(i)]; }
    float operator[](int i) const { return data[size_t(i)]; }
};

struct mat {
    int rows = 0, cols = 0;
    std::vector<float> data;
    mat() = default;
    mat(int r, int c) : rows(r), cols(c), data(size_t(r) * c) {}
    float & at(int r, int c) { return data[size_t(r) * cols + c]; }
    float at(int r, int c) const { return data[size_t(r) * cols + c]; }
};

vec layer_norm(const vec & x, const vec & w, const vec & b, float eps = 1e-5f) {
    require(x.n == w.n && x.n == b.n, "Strands: LayerNorm shape mismatch");
    double mean = 0, var = 0;
    for (int i = 0; i < x.n; ++i) {
        mean += x[i];
    }
    mean /= x.n;
    for (int i = 0; i < x.n; ++i) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= x.n;
    const float inv = float(1.0 / std::sqrt(var + eps));
    vec out(x.n);
    for (int i = 0; i < x.n; ++i) {
        out[i] = (float(x[i] - mean) * inv) * w[i] + b[i];
    }
    return out;
}

// weight is [out, in] row-major (same as torch Linear).
vec linear(const vec & x, const mat & w, const vec & b) {
    require(w.cols == x.n && w.rows == b.n, "Strands: linear shape mismatch");
    vec out(w.rows);
    for (int o = 0; o < w.rows; ++o) {
        double acc = b[o];
        for (int i = 0; i < w.cols; ++i) {
            acc += double(w.at(o, i)) * x[i];
        }
        out[o] = float(acc);
    }
    return out;
}

float temperature_for(int kind, float base, float t_noul, float t_choice, float t_score) {
    float t = base;
    if (kind == 0 && t_noul > 0) {
        t = t_noul;
    } else if (kind == 1 && t_choice > 0) {
        t = t_choice;
    } else if (kind == 2 && t_score > 0) {
        t = t_score;
    }
    return t > 1e-6f ? t : 1e-6f;
}
} // namespace

struct strands_head::impl {
    FILE * file = nullptr;
    gguf_context * meta = nullptr;
    ggml_context * tensors = nullptr;
    mat q_w, k_w;
    vec q_b, k_b, norm_w, norm_b;
    int hidden_size = 0;
    int pointer_dim = 0;
    float temperature = 1.f;
    float t_noul = 0.f, t_choice = 0.f, t_score = 0.f;

    void read(int index, void * data, size_t bytes) {
        const size_t position = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index);
#if defined(_WIN32)
        const int seek_rc = _fseeki64(file, static_cast<__int64>(position), SEEK_SET);
#else
        const int seek_rc = fseeko(file, static_cast<off_t>(position), SEEK_SET);
#endif
        require(seek_rc == 0 && std::fread(data, 1, bytes, file) == bytes, "Strands: truncated tensor data");
    }

    vec load_vec(const char * name, int expect) {
        const int idx = gguf_find_tensor(meta, name);
        require(idx >= 0, "Strands: missing head tensor");
        auto * t = ggml_get_tensor(tensors, name);
        require(t && t->type == GGML_TYPE_F32 && ggml_nelements(t) == expect, "Strands: bad head vector");
        vec out(expect);
        read(idx, out.data.data(), size_t(expect) * sizeof(float));
        return out;
    }

    mat load_mat(const char * name, int rows, int cols) {
        const int idx = gguf_find_tensor(meta, name);
        require(idx >= 0, "Strands: missing head matrix");
        auto * t = ggml_get_tensor(tensors, name);
        require(t && t->type == GGML_TYPE_F32 && ggml_nelements(t) == int64_t(rows) * cols,
                "Strands: bad head matrix");
        // GGUF stores ne[0]=cols, ne[1]=rows for 2D tensors.
        require(t->ne[0] == cols && t->ne[1] == rows, "Strands: unexpected matrix layout");
        mat out(rows, cols);
        read(idx, out.data.data(), size_t(rows) * cols * sizeof(float));
        return out;
    }

    explicit impl(const std::string & path) : file(ggml_fopen(path.c_str(), "rb")) {
        try {
            meta = gguf_init_from_file(path.c_str(), {true, &tensors});
            require(meta && tensors && file, "Strands: cannot open model");
            const int arch_key = gguf_find_key(meta, "general.architecture");
            require(arch_key >= 0 && gguf_get_kv_type(meta, arch_key) == GGUF_TYPE_STRING,
                    "Strands: missing architecture");
            const std::string prefix = std::string(gguf_get_val_str(meta, arch_key)) + ".decision.";
            auto u32 = [&](const char * key, uint32_t fallback = 0) -> uint32_t {
                const int i = gguf_find_key(meta, (prefix + key).c_str());
                if (i < 0) {
                    return fallback;
                }
                require(gguf_get_kv_type(meta, i) == GGUF_TYPE_UINT32, "Strands: bad u32 meta");
                return gguf_get_val_u32(meta, i);
            };
            auto f32 = [&](const char * key, float fallback = 0.f) -> float {
                const int i = gguf_find_key(meta, (prefix + key).c_str());
                if (i < 0) {
                    return fallback;
                }
                const auto ty = gguf_get_kv_type(meta, i);
                if (ty == GGUF_TYPE_FLOAT32) {
                    return gguf_get_val_f32(meta, i);
                }
                if (ty == GGUF_TYPE_FLOAT64) {
                    return float(gguf_get_val_f64(meta, i));
                }
                return fallback;
            };
            pointer_dim = int(u32("pointer_dim"));
            require(pointer_dim > 0 && pointer_dim <= 4096, "Strands: missing pointer_dim");
            temperature = f32("temperature", 1.f);
            t_noul = f32("temperature.noul");
            t_choice = f32("temperature.choice");
            t_score = f32("temperature.score");

            auto * nw = ggml_get_tensor(tensors, "strands.norm.weight");
            require(nw && nw->type == GGML_TYPE_F32, "Strands: missing strands.norm.weight");
            hidden_size = int(ggml_nelements(nw));
            require(hidden_size > 0 && hidden_size <= 16384, "Strands: bad hidden size");

            norm_w = load_vec("strands.norm.weight", hidden_size);
            norm_b = load_vec("strands.norm.bias", hidden_size);
            q_w = load_mat("strands.q.weight", pointer_dim, hidden_size);
            q_b = load_vec("strands.q.bias", pointer_dim);
            k_w = load_mat("strands.k.weight", pointer_dim, hidden_size);
            k_b = load_vec("strands.k.bias", pointer_dim);
        } catch (...) {
            if (tensors) {
                ggml_free(tensors);
                tensors = nullptr;
            }
            if (meta) {
                gguf_free(meta);
                meta = nullptr;
            }
            if (file) {
                std::fclose(file);
                file = nullptr;
            }
            throw;
        }
    }

    ~impl() {
        if (tensors) {
            ggml_free(tensors);
        }
        if (meta) {
            gguf_free(meta);
        }
        if (file) {
            std::fclose(file);
        }
    }

    std::vector<float> score_row(const std::vector<std::vector<float>> & states, int query_i,
                                 const std::vector<int> & pointers, int kind) {
        require(query_i >= 0 && query_i < int(states.size()), "Strands: bad query index");
        require(int(states[size_t(query_i)].size()) == hidden_size, "Strands: hidden width mismatch");
        vec q_in(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            q_in[i] = states[size_t(query_i)][size_t(i)];
        }
        const vec q = linear(layer_norm(q_in, norm_w, norm_b), q_w, q_b);
        const float scale = float(std::sqrt(double(pointer_dim))) *
                            temperature_for(kind, temperature, t_noul, t_choice, t_score);
        std::vector<float> logits;
        logits.reserve(pointers.size());
        for (int pidx : pointers) {
            require(pidx >= 0 && pidx < int(states.size()), "Strands: bad pointer index");
            require(int(states[size_t(pidx)].size()) == hidden_size, "Strands: option hidden width mismatch");
            vec k_in(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                k_in[i] = states[size_t(pidx)][size_t(i)];
            }
            const vec k = linear(layer_norm(k_in, norm_w, norm_b), k_w, k_b);
            double dot = 0;
            for (int i = 0; i < pointer_dim; ++i) {
                dot += double(k[i]) * q[i];
            }
            logits.push_back(float(dot / scale));
        }
        return logits;
    }
};

strands_head::strands_head(const std::string & path) : p(std::make_unique<impl>(path)) {}
strands_head::~strands_head() = default;

std::vector<std::vector<float>> strands_head::score(const std::vector<std::vector<float>> & hidden,
                                                    const std::vector<int32_t> & tokens,
                                                    const common_json & rows) {
    require(rows.is_array() && !rows.empty(), "Strands: pointer_rows required");
    require(!hidden.empty() && hidden.size() == tokens.size(), "Strands: hidden/token length mismatch");
    const int query_i = int(tokens.size()) - 1;
    std::vector<std::vector<float>> out;
    out.reserve(rows.size());
    for (const auto & row : rows) {
        require(row.contains("type") && row.contains("pointers"), "Strands: bad pointer row");
        const int kind = row.at("type").get<int>();
        require(kind >= 0 && kind <= 2, "Strands: bad question type");
        std::vector<int> pointers;
        for (const auto & pjson : row.at("pointers")) {
            pointers.push_back(pjson.get<int>());
        }
        require(pointers.size() >= 2 && pointers.size() <= 255, "Strands: bad option count");
        if (kind == 0) {
            require(pointers.size() == 2, "Strands: noul needs 2 options");
        }
        if (kind == 2) {
            require(pointers.size() <= 10, "Strands: score needs ≤10 options");
        }
        out.push_back(p->score_row(hidden, query_i, pointers, kind));
    }
    return out;
}
