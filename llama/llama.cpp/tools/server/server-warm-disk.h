#pragma once

// Disk copy of a prompt-cache entry. The RAM cache is lost when the process
// exits; a file written here can restore the same sequence state after a restart.
// LLAMA_WARM_DISK=0 disables it. Files that belong to another model, context,
// or device are left unread. CPU and GPU runs round differently, so the key
// includes the offload and the devices that wrote the state. It also includes
// the model file's size, modification time and leading megabyte, so a file
// replaced at the same path starts a new cache. It also includes the release,
// the physical prompt batch and the RoPE overrides, which change the state
// a later token continues from. A file also keeps the seconds per token of
// the prefill that wrote it, so a later ranking uses that speed. Version 2 removes
// version 1 files: those kept tokens written one at a time, which a GPU resume
// must not restore. The oldest files in the directory go first once it passes
// its byte budget, whichever model wrote them. Restoring a file makes it the
// newest, so a prompt that was just used is not the one the budget drops.

#include "server-warm-cover.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct llama_warm_ckpt {
    int64_t n_tokens = 0;
    int32_t id_task  = -1;
    int32_t pos_min  = 0;
    int32_t pos_max  = 0;
    std::vector<uint8_t> tgt;
    std::vector<uint8_t> dft;
    std::vector<uint8_t> spec;
};

struct llama_warm_entry {
    std::vector<int32_t> tokens;
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
    std::vector<llama_warm_ckpt> ckpts;

    // Logits after the last token. Absent on files written before this trailer.
    // logits_k < 0 is the whole vocabulary.
    std::vector<float> logits;
    int32_t  logits_n    = 0;
    int32_t  logits_k    = 0;
    uint64_t logits_hash = 0;

    // Seconds per token of the prefill that wrote this state. 0 when that
    // prefill was not timed, and the ranking then uses the latest one.
    double prefill_s = 0;
};

struct llama_warm_pick {
    std::filesystem::path path;
    float    f_keep = 0.0f;
    float    f_sim  = 0.0f;
    int      lcp    = 0;
    uint64_t bytes  = 0;
    double   prefill_s = 0;
};

inline uint64_t llama_warm_mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= 0x100000001b3ull;
    return h;
}

inline uint64_t llama_warm_mix_bytes(uint64_t h, const void * data, size_t n) {
    const auto * b = (const unsigned char *) data;
    for (size_t i = 0; i < n; i++) {
        h = llama_warm_mix(h, b[i]);
    }
    return h;
}

// The placement that produced a state. n_gpu_layers is how many layers left
// the CPU (0 is a CPU run). devices are backend names and ids, in order.
// tensor_split may be null. The same placement mixes to the same key.
inline uint64_t llama_warm_device(uint64_t id, uint32_t n_gpu_layers, int32_t split_mode,
        const float * tensor_split, size_t n_split,
        const char * const * devices, size_t n_devices) {
    id = llama_warm_mix(id, n_gpu_layers);
    id = llama_warm_mix(id, (uint64_t) (int64_t) split_mode);
    id = llama_warm_mix(id, n_split);
    if (tensor_split != nullptr) {
        for (size_t i = 0; i < n_split; i++) {
            uint32_t bits = 0;
            memcpy(&bits, &tensor_split[i], sizeof(bits));
            id = llama_warm_mix(id, bits);
        }
    }
    id = llama_warm_mix(id, n_devices);
    for (size_t i = 0; i < n_devices; i++) {
        const char * name = devices != nullptr && devices[i] != nullptr ? devices[i] : "";
        id = llama_warm_mix_bytes(id, name, strlen(name));
        id = llama_warm_mix(id, 0x1f);
    }
    return id;
}

inline uint64_t llama_warm_mix_f32(uint64_t id, float v) {
    static_assert(sizeof(float) == sizeof(uint32_t), "float is 32 bits");
    uint32_t bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return llama_warm_mix(id, bits);
}

// The release and the settings that change how a prompt is rounded into a
// state. version may be null. n_ubatch is the physical prompt batch.
inline uint64_t llama_warm_forward(uint64_t id, const char * version, int32_t n_ubatch,
        float rope_freq_base, float rope_freq_scale,
        float yarn_ext_factor, float yarn_attn_factor,
        float yarn_beta_fast, float yarn_beta_slow,
        int32_t yarn_orig_ctx, int32_t rope_scaling_type) {
    if (version == nullptr) {
        version = "";
    }
    id = llama_warm_mix_bytes(id, version, strlen(version));
    id = llama_warm_mix(id, (uint64_t) (int64_t) n_ubatch);
    id = llama_warm_mix_f32(id, rope_freq_base);
    id = llama_warm_mix_f32(id, rope_freq_scale);
    id = llama_warm_mix_f32(id, yarn_ext_factor);
    id = llama_warm_mix_f32(id, yarn_attn_factor);
    id = llama_warm_mix_f32(id, yarn_beta_fast);
    id = llama_warm_mix_f32(id, yarn_beta_slow);
    id = llama_warm_mix(id, (uint64_t) (int64_t) yarn_orig_ctx);
    id = llama_warm_mix(id, (uint64_t) (int64_t) rope_scaling_type);
    return id;
}

// Identity of the model file, folded into the cache key. A new file at the
// same path has a different size, time or leading megabyte and must not
// resume a state the previous file left behind. A missing file mixes as 0.
inline uint64_t llama_warm_file_id(uint64_t id, const std::filesystem::path & path) {
    std::error_code ec;
    if (path.empty()) {
        return llama_warm_mix(id, 0);
    }
    const auto nbytes = std::filesystem::file_size(path, ec);
    if (ec) {
        return llama_warm_mix(id, 0);
    }
    id = llama_warm_mix(id, (uint64_t) nbytes);
    const auto stamp = std::filesystem::last_write_time(path, ec);
    if (ec) {
        id = llama_warm_mix(id, 0);
    } else {
        id = llama_warm_mix(id, (uint64_t) stamp.time_since_epoch().count());
    }
    std::ifstream in(path, std::ios::binary);
    constexpr size_t k_head = 1u << 20;
    char buf[4096];
    size_t left = k_head;
    while (in && left > 0) {
        in.read(buf, (std::streamsize) std::min(left, sizeof(buf)));
        const size_t got = (size_t) in.gcount();
        if (got == 0) {
            break;
        }
        id = llama_warm_mix_bytes(id, buf, got);
        left -= got;
    }
    return id;
}

// Oldest files are dropped until the remaining bytes fit. protect is never dropped.
inline std::vector<size_t> llama_warm_trim(const std::vector<uint64_t> & bytes, const std::vector<int64_t> & mtime, uint64_t budget, size_t protect) {
    std::vector<size_t> order(bytes.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (mtime[a] != mtime[b]) {
            return mtime[a] < mtime[b];
        }
        return a < b;
    });

    uint64_t sum = 0;
    for (uint64_t b : bytes) {
        sum += b;
    }

    std::vector<size_t> drop;
    for (size_t i : order) {
        if (sum <= budget) {
            break;
        }
        if (i == protect) {
            continue;
        }
        drop.push_back(i);
        sum -= bytes[i];
    }
    return drop;
}

class llama_warm_disk {
public:
    static constexpr uint64_t k_budget = 2ull << 30;
    static constexpr char     k_magic[8] = { 'Z', 'R', 'W', 'A', 'R', 'M', 0, 0 };
    static constexpr uint64_t k_version = 2;
    static constexpr uint32_t k_max_tokens = 1u << 20;
    static constexpr uint32_t k_max_ckpts  = 4096;
    static constexpr uint32_t k_max_logits = 1u << 22;

    llama_warm_disk(std::filesystem::path dir, uint64_t model_id, uint64_t budget = k_budget)
        : dir_(std::move(dir)), model_id_(model_id), budget_(budget) {
        std::filesystem::create_directories(dir_);
        for (const auto & ent : std::filesystem::directory_iterator(dir_)) {
            std::error_code ec;
            if (!ent.is_regular_file(ec)) {
                continue;
            }
            const std::string name = ent.path().filename().string();
            if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
                std::filesystem::remove(ent.path(), ec);
                continue;
            }
            if (name.size() < 6 || name.compare(name.size() - 6, 6, ".zwarm") != 0) {
                continue;
            }
            file_meta meta;
            const meta_status st = read_meta(ent.path(), meta);
            if (st == meta_status::bad) {
                std::filesystem::remove(ent.path(), ec);
                continue;
            }
            if (st == meta_status::ok && meta.model_id == model_id_) {
                files_.push_back(std::move(meta));
            }
        }
        size_t protect = (size_t) -1;
        if (!files_.empty()) {
            protect = 0;
            for (size_t i = 1; i < files_.size(); i++) {
                if (files_[i].mtime >= files_[protect].mtime) {
                    protect = i;
                }
            }
        }
        trim_files(protect);
    }

    // The file was restored. Its timestamp becomes now, in the index and on
    // disk, so a later trim keeps it ahead of files that have not been used.
    void note_used(const std::filesystem::path & path) {
        std::error_code ec;
        std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
        if (ec) {
            return;
        }
        const int64_t stamp = mtime_of(path);
        std::lock_guard<std::mutex> g(mu_);
        for (auto & f : files_) {
            if (f.path == path) {
                f.mtime = stamp;
                return;
            }
        }
    }

    ~llama_warm_disk() {
        flush();
    }

    llama_warm_disk(const llama_warm_disk &) = delete;
    llama_warm_disk & operator=(const llama_warm_disk &) = delete;

    uint64_t budget() const {
        return budget_;
    }

    // 0 rewinds to any shorter length, -1 only to a mark, and a positive
    // value is the sliding window. Set once, before the first store.
    void set_rewind_window(int32_t window) {
        rewind_window_ = window;
    }

    size_t count() const {
        std::lock_guard<std::mutex> g(mu_);
        return files_.size();
    }

    uint64_t bytes() const {
        std::lock_guard<std::mutex> g(mu_);
        uint64_t sum = 0;
        for (const auto & f : files_) {
            sum += f.file_bytes;
        }
        return sum;
    }

    void flush() {
        if (writer_.joinable()) {
            writer_.join();
        }
    }

    void store(llama_warm_entry entry) {
        if (entry.tokens.empty() || entry.tokens.size() > k_max_tokens || entry.main.empty()) {
            return;
        }
        if (entry.ckpts.size() > k_max_ckpts) {
            return;
        }
        uint64_t payload = entry.main.size() + entry.drft.size();
        for (const auto & c : entry.ckpts) {
            payload += c.tgt.size() + c.dft.size() + c.spec.size();
        }
        if (payload > budget_) {
            return;
        }
        flush();
        writer_ = std::thread([this, entry = std::move(entry)]() mutable {
            try {
                write_entry(entry);
            } catch (const std::exception & e) {
                fprintf(stderr, "warm disk: %s\n", e.what());
            } catch (...) {
                fprintf(stderr, "warm disk: write failed\n");
            }
        });
    }

    // A cached prompt wins when it keeps a larger fraction of itself than f_keep_best
    // and covers a larger fraction of the query than f_sim_best. A prompt that would
    // throw away more than three quarters of its state is ignored, matching the RAM cache.
    std::optional<llama_warm_pick> better(const int32_t * query, size_t n, float f_keep_best, float f_sim_best) const {
        if (query == nullptr || n == 0) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> g(mu_);
        std::optional<llama_warm_pick> best;
        for (const auto & f : files_) {
            size_t lcp = 0;
            const size_t lim = std::min(n, f.tokens.size());
            while (lcp < lim && query[lcp] == f.tokens[lcp]) {
                lcp++;
            }
            if (lcp == 0 || f.tokens.empty()) {
                continue;
            }
            const float f_keep = (float) lcp / (float) f.tokens.size();
            const float f_sim  = (float) lcp / (float) n;
            if (f_keep < 0.25f) {
                continue;
            }
            if (!(f_keep_best < f_keep && f_sim_best < f_sim)) {
                continue;
            }
            f_keep_best = f_keep;
            f_sim_best  = f_sim;
            llama_warm_pick pick;
            pick.path   = f.path;
            pick.f_keep = f_keep;
            pick.f_sim  = f_sim;
            pick.lcp       = (int) lcp;
            pick.bytes     = f.file_bytes;
            pick.prefill_s = f.prefill_s;
            best = std::move(pick);
        }
        return best;
    }

    // Every cached prompt that shares a prefix. The caller drops a restore
    // that costs more than running the shared tokens.
    std::vector<llama_warm_pick> matches(const int32_t * query, size_t n) const {
        std::vector<llama_warm_pick> out;
        if (query == nullptr || n == 0) {
            return out;
        }
        std::lock_guard<std::mutex> g(mu_);
        for (const auto & f : files_) {
            size_t lcp = 0;
            const size_t lim = std::min(n, f.tokens.size());
            while (lcp < lim && query[lcp] == f.tokens[lcp]) {
                lcp++;
            }
            if (lcp == 0 || f.tokens.empty()) {
                continue;
            }
            const float f_keep = (float) lcp / (float) f.tokens.size();
            llama_warm_pick pick;
            pick.path   = f.path;
            pick.f_keep = f_keep;
            pick.f_sim  = (float) lcp / (float) n;
            pick.lcp       = (int) lcp;
            pick.bytes     = f.file_bytes;
            pick.prefill_s = f.prefill_s;
            out.push_back(std::move(pick));
        }
        return out;
    }

    std::optional<llama_warm_entry> read(const std::filesystem::path & path) const {
        try {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                return std::nullopt;
            }
            file_meta meta;
            if (read_meta_stream(in, meta) != meta_status::ok || meta.model_id != model_id_) {
                return std::nullopt;
            }
            llama_warm_entry entry;
            entry.tokens = std::move(meta.tokens);
            if (!read_vec(in, entry.main, meta.main_size) || !read_vec(in, entry.drft, meta.drft_size)) {
                return std::nullopt;
            }
            entry.ckpts.resize(meta.n_ckpt);
            for (uint32_t i = 0; i < meta.n_ckpt; i++) {
                auto & c = entry.ckpts[i];
                uint64_t tgt = 0, dft = 0, spec = 0;
                if (!read_i64(in, c.n_tokens) || !read_i32(in, c.id_task) || !read_i32(in, c.pos_min) || !read_i32(in, c.pos_max)) {
                    return std::nullopt;
                }
                if (!read_u64(in, tgt) || !read_u64(in, dft) || !read_u64(in, spec)) {
                    return std::nullopt;
                }
                if (!read_vec(in, c.tgt, tgt) || !read_vec(in, c.dft, dft) || !read_vec(in, c.spec, spec)) {
                    return std::nullopt;
                }
            }
            // A file written without a trailer ends here. ZLOG is the logits
            // after the last token. ZRAT is the seconds per token of the
            // prefill that wrote the state. Either may be absent.
            if (!read_trailer(in, &entry, nullptr)) {
                return std::nullopt;
            }
            return entry;
        } catch (const std::exception &) {
            return std::nullopt;
        }
    }

    void drop(const std::filesystem::path & path) {
        std::lock_guard<std::mutex> g(mu_);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        files_.erase(std::remove_if(files_.begin(), files_.end(), [&](const file_meta & f) {
            return f.path == path;
        }), files_.end());
    }

private:
    enum class meta_status { ok, other, bad };

    struct file_meta {
        uint64_t model_id = 0;
        std::vector<int32_t> tokens;
        uint32_t n_ckpt = 0;
        // Checkpoint lengths. Empty when the file has none. The saved state
        // itself is at tokens.size(), which is not repeated here.
        std::vector<int64_t> marks;
        uint64_t main_size = 0;
        uint64_t drft_size = 0;
        uint64_t file_bytes = 0;
        int64_t  mtime = 0;
        double   prefill_s = 0;
        std::filesystem::path path;
    };

    // Positioned at the optional trailer. A missing trailer is success.
    // An unknown tag is left unread by the caller only when this returns
    // true without consuming a rate: the state is still usable.
    static bool read_trailer(std::istream & in, llama_warm_entry * entry, double * rate_out) {
        if (entry != nullptr) {
            entry->prefill_s = 0;
        }
        if (rate_out != nullptr) {
            *rate_out = 0;
        }
        if (in.peek() == std::char_traits<char>::eof()) {
            return true;
        }
        char tag[4];
        if (!read_raw(in, tag, 4)) {
            return false;
        }
        if (std::memcmp(tag, "ZLOG", 4) == 0) {
            int32_t n_row = 0;
            int32_t k_row = 0;
            uint64_t hash = 0;
            uint32_t n_f = 0;
            if (!read_i32(in, n_row) || !read_i32(in, k_row) || !read_u64(in, hash) || !read_u32(in, n_f)) {
                return false;
            }
            if (hash == 0 || n_f == 0 || n_f > k_max_logits) {
                return false;
            }
            if (entry != nullptr && n_row != (int32_t) entry->tokens.size()) {
                return false;
            }
            if (entry != nullptr) {
                entry->logits.resize(n_f);
                if (!read_raw(in, entry->logits.data(), (size_t) n_f * sizeof(float))) {
                    return false;
                }
                entry->logits_n    = n_row;
                entry->logits_k    = k_row;
                entry->logits_hash = hash;
            } else if (!in.seekg((std::streamoff) ((size_t) n_f * sizeof(float)), std::ios::cur) || !in) {
                return false;
            }
            if (in.peek() == std::char_traits<char>::eof()) {
                return true;
            }
            if (!read_raw(in, tag, 4)) {
                return false;
            }
        }
        if (std::memcmp(tag, "ZRAT", 4) == 0) {
            uint64_t bits = 0;
            if (!read_u64(in, bits)) {
                return false;
            }
            double rate = 0;
            memcpy(&rate, &bits, sizeof(rate));
            if (rate > 0.0 && rate == rate && rate < 1.0e6) {
                if (entry != nullptr) {
                    entry->prefill_s = rate;
                }
                if (rate_out != nullptr) {
                    *rate_out = rate;
                }
            }
            return true;
        }
        return true;
    }

    static void write_u64(std::ostream & out, uint64_t v) {
        unsigned char b[8];
        for (int i = 0; i < 8; i++) {
            b[i] = (unsigned char) ((v >> (8 * i)) & 0xff);
        }
        out.write((const char *) b, 8);
    }

    static void write_u32(std::ostream & out, uint32_t v) {
        unsigned char b[4];
        for (int i = 0; i < 4; i++) {
            b[i] = (unsigned char) ((v >> (8 * i)) & 0xff);
        }
        out.write((const char *) b, 4);
    }

    static void write_i32(std::ostream & out, int32_t v) {
        write_u32(out, (uint32_t) v);
    }

    static void write_i64(std::ostream & out, int64_t v) {
        write_u64(out, (uint64_t) v);
    }

    static bool read_raw(std::istream & in, void * dst, size_t n) {
        if (n == 0) {
            return true;
        }
        in.read((char *) dst, (std::streamsize) n);
        return (bool) in;
    }

    static bool read_u64(std::istream & in, uint64_t & v) {
        unsigned char b[8];
        if (!read_raw(in, b, 8)) {
            return false;
        }
        v = 0;
        for (int i = 0; i < 8; i++) {
            v |= (uint64_t) b[i] << (8 * i);
        }
        return true;
    }

    static bool read_u32(std::istream & in, uint32_t & v) {
        unsigned char b[4];
        if (!read_raw(in, b, 4)) {
            return false;
        }
        v = 0;
        for (int i = 0; i < 4; i++) {
            v |= (uint32_t) b[i] << (8 * i);
        }
        return true;
    }

    static bool read_i32(std::istream & in, int32_t & v) {
        uint32_t u = 0;
        if (!read_u32(in, u)) {
            return false;
        }
        v = (int32_t) u;
        return true;
    }

    static bool read_i64(std::istream & in, int64_t & v) {
        uint64_t u = 0;
        if (!read_u64(in, u)) {
            return false;
        }
        v = (int64_t) u;
        return true;
    }

    static bool read_vec(std::istream & in, std::vector<uint8_t> & dst, uint64_t n) {
        if (n > (uint64_t) SIZE_MAX) {
            return false;
        }
        dst.resize((size_t) n);
        return read_raw(in, dst.data(), (size_t) n);
    }

    static uint64_t hash_tokens(uint64_t model, const std::vector<int32_t> & tokens) {
        uint64_t h = model;
        for (size_t i = 0; i < tokens.size(); i++) {
            h = llama_warm_mix(h, (uint64_t) (uint32_t) tokens[i] + (uint64_t) i * 0x9e3779b97f4a7c15ull);
        }
        return h;
    }

    static int64_t mtime_of(const std::filesystem::path & path) {
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(path, ec);
        if (ec) {
            return 0;
        }
        return (int64_t) t.time_since_epoch().count();
    }

    meta_status read_meta_stream(std::istream & in, file_meta & meta) const {
        char magic[8];
        if (!read_raw(in, magic, 8)) {
            return meta_status::bad;
        }
        if (std::memcmp(magic, k_magic, 8) != 0) {
            return meta_status::other;
        }
        uint64_t version = 0;
        if (!read_u64(in, version)) {
            return meta_status::bad;
        }
        if (version != k_version) {
            return meta_status::bad;
        }
        if (!read_u64(in, meta.model_id)) {
            return meta_status::bad;
        }
        uint32_t n_tokens = 0;
        if (!read_u32(in, n_tokens) || n_tokens > k_max_tokens) {
            return meta_status::bad;
        }
        meta.tokens.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (!read_i32(in, meta.tokens[i])) {
                return meta_status::bad;
            }
        }
        if (!read_u32(in, meta.n_ckpt) || meta.n_ckpt > k_max_ckpts) {
            return meta_status::bad;
        }
        if (!read_u64(in, meta.main_size) || !read_u64(in, meta.drft_size)) {
            return meta_status::bad;
        }
        if (meta.main_size > budget_ || meta.drft_size > budget_) {
            return meta_status::bad;
        }
        return meta_status::ok;
    }

    // Checkpoint lengths, after read_meta_stream. The state bytes stay unread:
    // read() continues from there. A bad file returns false.
    bool read_marks(std::istream & in, file_meta & meta) const {
        if (!in.seekg((std::streamoff) (meta.main_size + meta.drft_size), std::ios::cur) || !in) {
            return false;
        }
        meta.marks.clear();
        meta.marks.reserve(meta.n_ckpt);
        for (uint32_t i = 0; i < meta.n_ckpt; i++) {
            int64_t n_tok = 0;
            int32_t id_task = 0;
            int32_t pos_min = 0;
            int32_t pos_max = 0;
            uint64_t n_tgt = 0;
            uint64_t n_dft = 0;
            uint64_t n_spec = 0;
            if (!read_i64(in, n_tok) || !read_i32(in, id_task) || !read_i32(in, pos_min) || !read_i32(in, pos_max) ||
                    !read_u64(in, n_tgt) || !read_u64(in, n_dft) || !read_u64(in, n_spec)) {
                return false;
            }
            if (n_tgt > budget_ || n_dft > budget_ || n_spec > budget_) {
                return false;
            }
            meta.marks.push_back(n_tok);
            const uint64_t skip = n_tgt + n_dft + n_spec;
            if (skip > 0 && (!in.seekg((std::streamoff) skip, std::ios::cur) || !in)) {
                return false;
            }
        }
        return true;
    }

    meta_status read_meta(const std::filesystem::path & path, file_meta & meta) const {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return meta_status::bad;
        }
        const meta_status st = read_meta_stream(in, meta);
        if (st != meta_status::ok) {
            return st;
        }
        if (!read_marks(in, meta)) {
            return meta_status::bad;
        }
        double rate = 0;
        if (!read_trailer(in, nullptr, &rate)) {
            return meta_status::bad;
        }
        meta.prefill_s = rate;
        std::error_code ec;
        meta.file_bytes = (uint64_t) std::filesystem::file_size(path, ec);
        if (ec) {
            return meta_status::bad;
        }
        meta.mtime = mtime_of(path);
        meta.path  = path;
        return meta_status::ok;
    }

    void write_entry(const llama_warm_entry & entry) {
        const uint64_t h = hash_tokens(model_id_, entry.tokens);
        char name[32];
        snprintf(name, sizeof(name), "%016llx.zwarm", (unsigned long long) h);
        const std::filesystem::path dst = dir_ / name;
        const std::filesystem::path tmp = dir_ / (std::string(name) + ".tmp");

        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to create " + tmp.string());
            }
            out.write(k_magic, 8);
            write_u64(out, k_version);
            write_u64(out, model_id_);
            write_u32(out, (uint32_t) entry.tokens.size());
            for (int32_t t : entry.tokens) {
                write_i32(out, t);
            }
            write_u32(out, (uint32_t) entry.ckpts.size());
            write_u64(out, entry.main.size());
            write_u64(out, entry.drft.size());
            if (!entry.main.empty()) {
                out.write((const char *) entry.main.data(), (std::streamsize) entry.main.size());
            }
            if (!entry.drft.empty()) {
                out.write((const char *) entry.drft.data(), (std::streamsize) entry.drft.size());
            }
            for (const auto & c : entry.ckpts) {
                write_i64(out, c.n_tokens);
                write_i32(out, c.id_task);
                write_i32(out, c.pos_min);
                write_i32(out, c.pos_max);
                write_u64(out, c.tgt.size());
                write_u64(out, c.dft.size());
                write_u64(out, c.spec.size());
                if (!c.tgt.empty()) {
                    out.write((const char *) c.tgt.data(), (std::streamsize) c.tgt.size());
                }
                if (!c.dft.empty()) {
                    out.write((const char *) c.dft.data(), (std::streamsize) c.dft.size());
                }
                if (!c.spec.empty()) {
                    out.write((const char *) c.spec.data(), (std::streamsize) c.spec.size());
                }
            }
            if (entry.logits_hash != 0 && entry.logits_n == (int32_t) entry.tokens.size() &&
                    !entry.logits.empty() && entry.logits.size() <= k_max_logits) {
                out.write("ZLOG", 4);
                write_i32(out, entry.logits_n);
                write_i32(out, entry.logits_k);
                write_u64(out, entry.logits_hash);
                write_u32(out, (uint32_t) entry.logits.size());
                out.write((const char *) entry.logits.data(), (std::streamsize) (entry.logits.size() * sizeof(float)));
            }
            if (entry.prefill_s > 0.0 && entry.prefill_s == entry.prefill_s && entry.prefill_s < 1.0e6) {
                out.write("ZRAT", 4);
                uint64_t bits = 0;
                memcpy(&bits, &entry.prefill_s, sizeof(bits));
                write_u64(out, bits);
            }
            if (!out) {
                std::error_code ec;
                std::filesystem::remove(tmp, ec);
                throw std::runtime_error("failed to write " + tmp.string());
            }
        }

        std::error_code ec;
        std::filesystem::rename(tmp, dst, ec);
        if (ec) {
            std::filesystem::remove(dst, ec);
            std::filesystem::rename(tmp, dst, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                throw std::runtime_error("failed to rename " + tmp.string());
            }
        }

        file_meta meta;
        if (read_meta(dst, meta) != meta_status::ok) {
            std::filesystem::remove(dst, ec);
            throw std::runtime_error("failed to read back " + dst.string());
        }

        std::vector<int64_t> new_marks;
        new_marks.reserve(entry.ckpts.size());
        for (const auto & c : entry.ckpts) {
            new_marks.push_back(c.n_tokens);
        }

        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::filesystem::path> dead;
        for (auto it = files_.begin(); it != files_.end(); ) {
            const bool covered = llama_warm_covers(
                    rewind_window_,
                    it->tokens.data(), it->tokens.size(), it->marks.data(), it->marks.size(),
                    entry.tokens.data(), entry.tokens.size(), new_marks.data(), new_marks.size());
            if (it->path == dst || covered) {
                if (it->path != dst) {
                    dead.push_back(it->path);
                }
                it = files_.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & p : dead) {
            std::filesystem::remove(p, ec);
        }
        files_.push_back(std::move(meta));
        trim_files(files_.size() - 1);
    }

    // Caller holds mu_, or is the constructor before the object is shared.
    // Files of other models count toward the same budget. A file newer than
    // the one protect names is left in place: that is the state just written
    // or the newest one this model can still resume.
    void trim_files(size_t protect) {
        std::vector<uint64_t> sizes;
        std::vector<int64_t>  times;
        std::vector<std::filesystem::path> extras;
        sizes.reserve(files_.size());
        times.reserve(files_.size());
        for (const auto & f : files_) {
            sizes.push_back(f.file_bytes);
            times.push_back(f.mtime);
        }
        std::error_code ec;
        for (const auto & ent : std::filesystem::directory_iterator(dir_, ec)) {
            if (ec || !ent.is_regular_file(ec)) {
                continue;
            }
            const std::string name = ent.path().filename().string();
            if (name.size() < 6 || name.compare(name.size() - 6, 6, ".zwarm") != 0) {
                continue;
            }
            bool known = false;
            for (const auto & f : files_) {
                if (f.path == ent.path()) {
                    known = true;
                    break;
                }
            }
            if (known) {
                continue;
            }
            const uint64_t bytes = (uint64_t) std::filesystem::file_size(ent.path(), ec);
            if (ec) {
                continue;
            }
            extras.push_back(ent.path());
            sizes.push_back(bytes);
            times.push_back(mtime_of(ent.path()));
        }
        std::vector<size_t> drop = llama_warm_trim(sizes, times, budget_, protect);
        if (protect < times.size()) {
            const int64_t keep_mtime = times[protect];
            drop.erase(std::remove_if(drop.begin(), drop.end(), [&](size_t i) {
                return i != protect && times[i] > keep_mtime;
            }), drop.end());
        }
        std::vector<char> gone(files_.size(), 0);
        for (size_t i : drop) {
            if (i < files_.size()) {
                gone[i] = 1;
                std::filesystem::remove(files_[i].path, ec);
            } else {
                std::filesystem::remove(extras[i - files_.size()], ec);
            }
        }
        std::vector<file_meta> keep;
        keep.reserve(files_.size());
        for (size_t i = 0; i < files_.size(); i++) {
            if (!gone[i]) {
                keep.push_back(std::move(files_[i]));
            }
        }
        files_.swap(keep);
    }

    std::filesystem::path dir_;
    uint64_t model_id_ = 0;
    uint64_t budget_ = k_budget;
    int32_t rewind_window_ = 0;
    mutable std::mutex mu_;
    std::vector<file_meta> files_;
    std::thread writer_;
};

#ifdef LLAMA_WARM_DISK_TEST
#include <chrono>
#include <iostream>

static int warm_fail(const char * msg) {
    std::cerr << msg << "\n";
    return 1;
}

int main() {
    {
        const uint64_t base = 42;
        const char * cpu[] = { "cpu" };
        const char * metal[] = { "Metal", "0" };
        const uint64_t cpu_id = llama_warm_device(base, 0, 0, nullptr, 0, cpu, 1);
        if (cpu_id != llama_warm_device(base, 0, 0, nullptr, 0, cpu, 1)) {
            return warm_fail("device id stable");
        }
        const uint64_t metal_id = llama_warm_device(base, 33, 1, nullptr, 0, metal, 2);
        if (cpu_id == metal_id) {
            return warm_fail("cpu and metal share a cache");
        }
        if (metal_id == llama_warm_device(base, 10, 1, nullptr, 0, metal, 2)) {
            return warm_fail("layer count is part of the device");
        }
        float even = 0.0f;
        float front = 0.5f;
        if (llama_warm_device(base, 33, 1, &even, 1, metal, 2) ==
                llama_warm_device(base, 33, 1, &front, 1, metal, 2)) {
            return warm_fail("tensor split is part of the device");
        }
    }
    {
        const std::filesystem::path files = "/tmp/llama-warm-file-id";
        std::filesystem::remove_all(files);
        std::filesystem::create_directories(files);
        const std::filesystem::path one = files / "model-a.bin";
        const std::filesystem::path two = files / "model-b.bin";
        const std::filesystem::path copy = files / "model-a-copy.bin";
        {
            std::ofstream out(one, std::ios::binary);
            out << "weights-a";
        }
        {
            std::ofstream out(two, std::ios::binary);
            out << "weights-b";
        }
        const auto stamp = std::filesystem::last_write_time(one);
        std::filesystem::last_write_time(two, stamp);
        const uint64_t id_a = llama_warm_file_id(7, one);
        if (id_a != llama_warm_file_id(7, one)) {
            return warm_fail("file id stable");
        }
        if (id_a == llama_warm_file_id(7, two)) {
            return warm_fail("replaced file shares a cache");
        }
        std::filesystem::copy_file(one, copy);
        std::filesystem::last_write_time(copy, stamp);
        if (llama_warm_file_id(7, copy) != id_a) {
            return warm_fail("same file bytes share a cache");
        }
        std::filesystem::last_write_time(one, stamp - std::chrono::hours(1));
        if (llama_warm_file_id(7, one) == id_a) {
            return warm_fail("file mtime is part of the key");
        }
        if (llama_warm_file_id(7, files / "missing.bin") != llama_warm_mix(7, 0)) {
            return warm_fail("missing file");
        }
        std::filesystem::remove_all(files);
    }
    {
        const uint64_t same = llama_warm_forward(7, "b12557", 512, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0, 0);
        if (same != llama_warm_forward(7, "b12557", 512, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0, 0)) {
            return warm_fail("forward id stable");
        }
        if (same == llama_warm_forward(7, "b12558", 512, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0, 0)) {
            return warm_fail("release shares a cache");
        }
        if (same == llama_warm_forward(7, "b12557", 256, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0, 0)) {
            return warm_fail("prompt batch shares a cache");
        }
        if (same == llama_warm_forward(7, "b12557", 512, 0.0f, 0.5f, -1.0f, -1.0f, -1.0f, -1.0f, 0, 0)) {
            return warm_fail("rope scale shares a cache");
        }
    }
    {
        const std::vector<uint64_t> bytes = { 100, 100, 100 };
        const std::vector<int64_t>  mtime = { 1, 2, 3 };
        const auto drop = llama_warm_trim(bytes, mtime, 250, (size_t) -1);
        if (drop.size() != 1 || drop[0] != 0) {
            return warm_fail("trim oldest");
        }
        const auto kept = llama_warm_trim(bytes, mtime, 250, 0);
        if (kept.size() != 1 || kept[0] != 1) {
            return warm_fail("trim protects");
        }
        if (!llama_warm_trim(bytes, mtime, 1000, (size_t) -1).empty()) {
            return warm_fail("trim under budget");
        }
    }

    const std::filesystem::path dir = "/tmp/llama-warm-disk-test";
    std::filesystem::remove_all(dir);

    llama_warm_entry ent;
    ent.tokens = { 1, 2, 3, 4 };
    ent.main   = { 9, 8, 7 };
    ent.drft   = { 4 };
    llama_warm_ckpt ck;
    ck.n_tokens = 4;
    ck.id_task  = 3;
    ck.pos_min  = 0;
    ck.pos_max  = 3;
    ck.tgt      = { 1, 1 };
    ck.spec     = { 5 };
    ent.ckpts.push_back(ck);

    {
        llama_warm_disk disk(dir, 42, 100000);
        disk.store(ent);
        disk.flush();
        if (disk.count() != 1) {
            return warm_fail("store count");
        }
    }

    {
        llama_warm_disk disk(dir, 42, 100000);
        if (disk.count() != 1) {
            return warm_fail("reopen count");
        }
        const int32_t query[] = { 1, 2, 3, 4, 5 };
        const auto pick = disk.better(query, 5, 0.0f, 0.0f);
        if (!pick || pick->lcp != 4) {
            return warm_fail("better");
        }
        const auto got = disk.read(pick->path);
        if (!got || got->tokens != ent.tokens || got->main != ent.main || got->drft != ent.drft) {
            return warm_fail("roundtrip");
        }
        if (got->ckpts.size() != 1 || got->ckpts[0].n_tokens != 4 || got->ckpts[0].id_task != 3 || got->ckpts[0].tgt != ck.tgt || got->ckpts[0].spec != ck.spec) {
            return warm_fail("checkpoint");
        }
        if (!got->logits.empty() || got->logits_hash != 0 || got->prefill_s != 0.0) {
            return warm_fail("logits absent");
        }
        const int32_t shortq[] = { 1 };
        llama_warm_entry long_ent = ent;
        long_ent.tokens = { 1, 2, 3, 4, 5, 6, 7, 8 };
        disk.store(long_ent);
        disk.flush();
        if (disk.better(shortq, 1, 0.0f, 0.0f)) {
            return warm_fail("f_keep");
        }
    }

    {
        llama_warm_disk other(dir, 99, 100000);
        if (other.count() != 0) {
            return warm_fail("other model");
        }
    }

    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 42, 100000);
        llama_warm_entry a;
        a.tokens = { 1, 2, 3 };
        a.main   = { 1 };
        llama_warm_entry b = a;
        b.tokens = { 1, 2, 3, 4 };
        disk.store(a);
        disk.flush();
        disk.store(b);
        disk.flush();
        if (disk.count() != 1) {
            return warm_fail("prefix drop count");
        }
        const int32_t q[] = { 1, 2, 3, 4 };
        const auto pick = disk.better(q, 4, -1.0f, -1.0f);
        const auto got = pick ? disk.read(pick->path) : std::nullopt;
        if (!got || got->tokens != b.tokens) {
            return warm_fail("prefix drop tokens");
        }
    }

    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 42, 100000);
        disk.set_rewind_window(-1);
        llama_warm_entry a;
        a.tokens = { 1, 2, 3 };
        a.main   = { 1 };
        llama_warm_entry b = a;
        b.tokens = { 1, 2, 3, 4, 5, 6 };
        disk.store(a);
        disk.flush();
        disk.store(b);
        disk.flush();
        if (disk.count() != 2) {
            return warm_fail("mark keep count");
        }
        llama_warm_ckpt at;
        at.n_tokens = 3;
        b.ckpts.push_back(at);
        disk.store(b);
        disk.flush();
        if (disk.count() != 1) {
            return warm_fail("mark drop count");
        }
    }

    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 42, 100000);
        disk.set_rewind_window(4);
        llama_warm_entry a;
        a.tokens = { 1, 2, 3 };
        a.main   = { 1 };
        llama_warm_entry b = a;
        b.tokens = { 1, 2, 3, 4, 5, 6 };
        disk.store(a);
        disk.flush();
        disk.store(b);
        disk.flush();
        if (disk.count() != 2) {
            return warm_fail("window keep count");
        }
        disk.set_rewind_window(8);
        llama_warm_entry c = a;
        c.tokens = { 1, 2, 3, 4, 5 };
        disk.store(c);
        disk.flush();
        // a and c both fit in the window, so c replaces a. b has wrapped.
        if (disk.count() != 2) {
            return warm_fail("window drop count");
        }
    }

    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 42, 100000);
        llama_warm_entry row;
        row.tokens = { 1, 2, 3, 4 };
        row.main   = { 9 };
        row.logits = { 0.25f, -1.5f, 3.0f };
        row.logits_n = 4;
        row.logits_k = -1;
        row.logits_hash = 0x1234;
        row.prefill_s = 0.012;
        disk.store(row);
        disk.flush();
        const int32_t q[] = { 1, 2, 3, 4 };
        const auto pick = disk.better(q, 4, -1.0f, -1.0f);
        const auto got = pick ? disk.read(pick->path) : std::nullopt;
        if (!got || got->main != row.main || got->logits != row.logits || got->logits_n != 4 || got->logits_k != -1 || got->logits_hash != 0x1234 || got->prefill_s != 0.012) {
            return warm_fail("logits trailer");
        }
        if (!pick || pick->prefill_s != 0.012) {
            return warm_fail("prefill rate indexed");
        }
    }

    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 42, 100000);
        llama_warm_entry long_row;
        long_row.tokens = { 1, 2, 3, 4, 5, 6, 7, 8 };
        long_row.main   = { 1, 2, 3, 4 };
        disk.store(long_row);
        disk.flush();
        const int32_t one[] = { 1, 9 };
        const auto few = disk.matches(one, 2);
        if (few.size() != 1 || few[0].lcp != 1 || !(few[0].f_keep < 0.25f)) {
            return warm_fail("short prefix is a candidate");
        }
        if (disk.better(one, 2, 0.0f, 0.0f)) {
            return warm_fail("untimed short prefix");
        }
    }

    uint64_t one = 0;
    {
        std::filesystem::remove_all(dir);
        llama_warm_disk disk(dir, 7, 100000);
        llama_warm_entry a;
        a.tokens = { 8 };
        a.main   = { 1, 2, 3, 4 };
        disk.store(a);
        disk.flush();
        one = disk.bytes();
        if (one == 0 || disk.count() != 1) {
            return warm_fail("budget seed");
        }
    }
    {
        llama_warm_disk disk(dir, 7, one);
        llama_warm_entry b;
        b.tokens = { 9 };
        b.main   = { 1, 2, 3, 4 };
        disk.store(b);
        disk.flush();
        if (disk.count() != 1) {
            return warm_fail("budget trim");
        }
        const int32_t oldq[] = { 8 };
        const int32_t newq[] = { 9 };
        if (disk.better(oldq, 1, -1.0f, -1.0f)) {
            return warm_fail("budget kept old");
        }
        if (!disk.better(newq, 1, -1.0f, -1.0f)) {
            return warm_fail("budget kept new");
        }
    }

    {
        std::filesystem::remove_all(dir);
        std::filesystem::path older;
        std::filesystem::path newer;
        uint64_t one_file = 0;
        {
            llama_warm_disk disk(dir, 42, 100000);
            llama_warm_entry a;
            a.tokens = { 1 };
            a.main   = { 1, 2, 3, 4 };
            llama_warm_entry b;
            b.tokens = { 2 };
            b.main   = { 1, 2, 3, 4 };
            disk.store(a);
            disk.flush();
            disk.store(b);
            disk.flush();
            const int32_t qa[] = { 1 };
            const int32_t qb[] = { 2 };
            const auto pa = disk.better(qa, 1, -1.0f, -1.0f);
            const auto pb = disk.better(qb, 1, -1.0f, -1.0f);
            if (!pa || !pb) {
                return warm_fail("used paths");
            }
            older = pb->path;
            newer = pa->path;
            disk.note_used(newer);
            const auto written = std::filesystem::last_write_time(older);
            std::filesystem::last_write_time(older, written - std::chrono::hours(1));
            one_file = (uint64_t) std::filesystem::file_size(newer);
        }
        {
            llama_warm_disk disk(dir, 42, one_file);
            if (disk.count() != 1) {
                return warm_fail("used trim count");
            }
            const int32_t qa[] = { 1 };
            const int32_t qb[] = { 2 };
            if (!disk.better(qa, 1, -1.0f, -1.0f) || disk.better(qb, 1, -1.0f, -1.0f)) {
                return warm_fail("used file kept");
            }
        }
    }

    {
        std::filesystem::remove_all(dir);
        uint64_t one_file = 0;
        {
            llama_warm_disk first(dir, 42, 100000);
            llama_warm_entry a;
            a.tokens = { 3 };
            a.main   = { 1, 2, 3, 4 };
            first.store(a);
            first.flush();
        }
        {
            llama_warm_disk second(dir, 99, 100000);
            llama_warm_entry b;
            b.tokens = { 4 };
            b.main   = { 1, 2, 3, 4 };
            second.store(b);
            second.flush();
            const int32_t q[] = { 4 };
            const auto pick = second.better(q, 1, -1.0f, -1.0f);
            if (!pick) {
                return warm_fail("other model path");
            }
            one_file = (uint64_t) std::filesystem::file_size(pick->path);
            for (const auto & ent : std::filesystem::directory_iterator(dir)) {
                if (ent.path() == pick->path) {
                    continue;
                }
                const auto written = std::filesystem::last_write_time(ent.path());
                std::filesystem::last_write_time(ent.path(), written - std::chrono::hours(1));
            }
        }
        {
            llama_warm_disk disk(dir, 99, one_file);
            if (disk.count() != 1) {
                return warm_fail("directory trim kept");
            }
        }
        {
            llama_warm_disk disk(dir, 42, 100000);
            if (disk.count() != 0) {
                return warm_fail("directory trim dropped other");
            }
        }
    }

    std::filesystem::remove_all(dir);
    std::cout << "warm disk ok\n";
    return 0;
}
#endif
