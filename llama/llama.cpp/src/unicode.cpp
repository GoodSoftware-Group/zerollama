#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

// Shared GPT-2 byte→UTF-8 table (0117). Built once.
struct unicode_byte_enc {
    char    d[4];
    uint8_t n;
};

static const unicode_byte_enc * unicode_byte_enc_table() {
    static unicode_byte_enc table[256];
    static bool init = false;
    if (!init) {
        const auto map = unicode_byte_to_utf8_map();
        for (int ch = 0; ch < 256; ++ch) {
            const std::string & s = map.at((uint8_t) ch);
            unicode_byte_enc & e = table[ch];
            e.n = (uint8_t) s.size();
            assert(e.n <= 4);
            for (uint8_t i = 0; i < e.n; ++i) {
                e.d[i] = s[i];
            }
        }
        for (int ch = 0x21; ch <= 0x7E; ++ch) {
            assert(table[ch].n == 1 && table[ch].d[0] == (char) ch);
        }
        init = true;
    }
    return table;
}

static inline bool unicode_word_is_printable_ascii(const unsigned char * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] < 0x21u || p[i] > 0x7Eu) {
            return false;
        }
    }
    return true;
}

// WHY (0125): GPT-2/Qwen pretok is usually ` ?\p{L}+` — a leading space + printable rest.
// Space (0x20) is NOT identity under bytes↔unicode (→ U+0120, 2-byte UTF-8), so the
// 0117 printable skip misses these words and remaps every letter byte-by-byte. Remap
// space once, memcpy the rest (identity).
static inline bool unicode_word_is_space_plus_printable(const unsigned char * p, size_t n) {
    return n >= 1 && p[0] == ' ' && unicode_word_is_printable_ascii(p + 1, n - 1);
}

static inline void unicode_byte_enc_append_word(
        std::string & storage, const unicode_byte_enc * table,
        const unsigned char * p, size_t n) {
    if (unicode_word_is_printable_ascii(p, n)) {
        storage.append((const char *) p, n);
        return;
    }
    if (unicode_word_is_space_plus_printable(p, n)) {
        const unicode_byte_enc & sp = table[(unsigned char) ' '];
        storage.append(sp.d, sp.n);
        storage.append((const char *) (p + 1), n - 1);
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const unicode_byte_enc & e = table[p[i]];
        storage.append(e.d, e.n);
    }
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    // WHY (0117): stock did unordered_map lookup + temporary std::string per byte, then
    // `+=` into the output. Printable ASCII 0x21..0x7E maps to itself (1-byte UTF-8) —
    // those words are identity. Flattened LUT + append(len) for the rest.
    // WHY (0125): leading-space + printable also skips per-letter remap.
    // LLAMA_BPE_NO_BYTE_ENC_FAST=1 keeps the old map+= path for identity A/B.
    if (getenv("LLAMA_BPE_NO_BYTE_ENC_FAST") != nullptr) {
        std::vector<std::string> bpe_encoded_words;
        bpe_encoded_words.reserve(bpe_words.size());
        for (const auto & word : bpe_words) {
            std::string encoded_token;
            encoded_token.reserve(word.size() * 2);
            for (unsigned char c : word) {
                encoded_token += unicode_byte_to_utf8(c);
            }
            bpe_encoded_words.emplace_back(std::move(encoded_token));
        }
        return bpe_encoded_words;
    }

    const unicode_byte_enc * table = unicode_byte_enc_table();
    std::vector<std::string> bpe_encoded_words;
    bpe_encoded_words.reserve(bpe_words.size());
    for (const auto & word : bpe_words) {
        const unsigned char * p = (const unsigned char *) word.data();
        const size_t n = word.size();
        if (unicode_word_is_printable_ascii(p, n)) {
            bpe_encoded_words.push_back(word);
            continue;
        }
        std::string encoded_token;
        encoded_token.reserve(n + 2);
        unicode_byte_enc_append_word(encoded_token, table, p, n);
        bpe_encoded_words.emplace_back(std::move(encoded_token));
    }
    return bpe_encoded_words;
}

// WHY (0109): English megaprompts are ASCII-dominant. Custom pretok called
// unicode_cpt_flags_from_cpt per codepoint (indexes a ~2MiB table). Arithmetic
// predicates match that table for c < 128 — identity-safe, much cheaper in the
// letter/digit consume loops that dominate Qwen/GPT-2/Llama pretok.
static inline bool unicode_cpt_is_ascii_letter(uint32_t cpt) {
    return cpt < 128u && ((cpt | 0x20u) - static_cast<uint32_t>('a')) < 26u;
}

static inline bool unicode_cpt_is_ascii_digit(uint32_t cpt) {
    return (cpt - static_cast<uint32_t>('0')) < 10u;
}

static inline size_t unicode_consume_letters(const std::vector<uint32_t> & cpts, size_t pos, size_t end) {
    while (pos < end) {
        const uint32_t c = cpts[pos];
        if (c < 128u) {
            if (!unicode_cpt_is_ascii_letter(c)) {
                break;
            }
        } else if (!unicode_cpt_flags_from_cpt(c).is_letter) {
            break;
        }
        ++pos;
    }
    return pos;
}

static inline size_t unicode_consume_letters_or_marks(const std::vector<uint32_t> & cpts, size_t pos, size_t end) {
    while (pos < end) {
        const uint32_t c = cpts[pos];
        if (c < 128u) {
            if (!unicode_cpt_is_ascii_letter(c)) {
                break;
            }
        } else {
            const auto f = unicode_cpt_flags_from_cpt(c);
            if (!(f.is_letter | f.is_accent_mark)) {
                break;
            }
        }
        ++pos;
    }
    return pos;
}

static inline size_t unicode_consume_digits(const std::vector<uint32_t> & cpts, size_t pos, size_t end) {
    while (pos < end) {
        const uint32_t c = cpts[pos];
        if (c < 128u) {
            if (!unicode_cpt_is_ascii_digit(c)) {
                break;
            }
        } else if (!unicode_cpt_flags_from_cpt(c).is_number) {
            break;
        }
        ++pos;
    }
    return pos;
}

// WHY (0110): After 0106–0109, Qwen English megaprompts are still dominated by the
// custom pretok scanner (not merge). When a segment is pure ASCII, replace
// per-cpt unicode_cpt_flags_from_cpt (~2MiB table) + lambdas with a 128-byte
// class LUT and a tight loop. Identity: LLAMA_BPE_NO_ASCII_PRETOK=1 forces the
// legacy unicode path in the same binary.
static const uint8_t * unicode_ascii_cls_lut() {
    static uint8_t cls[128];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 128; ++i) {
            const auto f = unicode_cpt_flags_from_cpt(i);
            uint8_t v = 0;
            if (f.is_whitespace) {
                v |= 1;
            }
            if (f.is_letter) {
                v |= 2;
            }
            if (f.is_number) {
                v |= 4;
            }
            // Matches: !(ws|L|N) && flags.as_uint() — punctuation/symbol/control path
            if (!(f.is_whitespace | f.is_letter | f.is_number) && f.as_uint()) {
                v |= 8;
            }
            cls[i] = v;
            // WHY (0126): SIMD consume uses arithmetic A–Z/a–z and 0–9 — must match LUT.
            const bool arith_L = ((i | 0x20u) - (uint32_t) 'a') < 26u;
            const bool arith_N = (i - (uint32_t) '0') < 10u;
            assert(arith_L == (bool) (v & 2) && arith_N == (bool) (v & 4));
            (void) arith_L;
            (void) arith_N;
        }
        init = true;
    }
    return cls;
}

// Raw unicode_cpt_flags::as_uint() for ASCII — hybrid scanner uses this when c < 128
// so mostly-ASCII megaprompts (sparse CJK/emoji) skip the ~2MiB flags table.
static const uint16_t * unicode_ascii_flags_u16() {
    static uint16_t raw[128];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 128; ++i) {
            raw[i] = unicode_cpt_flags_from_cpt(i).as_uint();
        }
        init = true;
    }
    return raw;
}

static inline unicode_cpt_flags unicode_flags_for_cpt_hot(uint32_t cpt) {
    if (cpt < 128u) {
        return unicode_cpt_flags(unicode_ascii_flags_u16()[cpt]);
    }
    return unicode_cpt_flags_from_cpt(cpt);
}

static inline bool unicode_want_ascii_pretok() {
    return getenv("LLAMA_BPE_NO_ASCII_PRETOK") == nullptr;
}

static inline bool unicode_seg_is_ascii(const uint32_t * cpts, size_t ini, size_t end) {
    for (size_t i = ini; i < end; ++i) {
        if (cpts[i] >= 128u) {
            return false;
        }
    }
    return true;
}

static inline uint32_t unicode_ascii_tolower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? (c | 0x20u) : c;
}

// Qwen2 pretok on a proven-ASCII [ini, end) slice. Emits codepoint lengths into bpe_offsets.
static void unicode_regex_split_qwen2_ascii_seg(
        const uint32_t * cpts, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const uint32_t cpt = cpts[pos];
        const uint8_t c = cls[cpt];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = unicode_ascii_tolower(cpts[pos + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t n2 = unicode_ascii_tolower(cpts[pos + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || (c & 4))) {
            const bool next_letter = (pos + 1 < end) && (cls[cpts[pos + 1]] & 2);
            if ((c & 2) || next_letter) {
                ++pos;
                while (pos + 8 <= end) {
                    if (!((cls[cpts[pos]] & 2) && (cls[cpts[pos + 1]] & 2) && (cls[cpts[pos + 2]] & 2) &&
                          (cls[cpts[pos + 3]] & 2) && (cls[cpts[pos + 4]] & 2) && (cls[cpts[pos + 5]] & 2) &&
                          (cls[cpts[pos + 6]] & 2) && (cls[cpts[pos + 7]] & 2))) {
                        break;
                    }
                    pos += 8;
                }
                while (pos < end && (cls[cpts[pos]] & 2)) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        // \p{N}
        if (c & 4) {
            add_token(++pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        {
            const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[cpts[pos + 1]]
                               : (cpt == ' ' ? 0 : c);
            if (c2 & 8) {
                pos += (cpt == ' ');
                while (pos < end && (cls[cpts[pos]] & 8)) {
                    ++pos;
                }
                while (pos < end && (cpts[pos] == '\r' || cpts[pos] == '\n')) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        size_t num_ws = 0;
        size_t last_rn = 0;
        while (pos + num_ws < end && (cls[cpts[pos + num_ws]] & 1)) {
            const uint32_t w = cpts[pos + num_ws];
            if (w == '\r' || w == '\n') {
                last_rn = pos + num_ws + 1;
            }
            ++num_ws;
        }

        if (last_rn > 0) {
            pos = last_rn;
            add_token(pos);
            continue;
        }

        // \s+(?!\S) — trailing run with something after → keep last ws for next
        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }

        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }

        add_token(++pos);
    }
}

// WHY (0114): all-ASCII Qwen2 megaprompts still paid for uint32 decode + 4×RAM before the
// 0110 ascii_seg scanner. When text bytes are all < 0x80, pretok offsets == byte lengths —
// scan the original string and substr words (identity vs cpt path).
static inline bool unicode_bytes_are_ascii(const std::string & text) {
    const unsigned char * p = (const unsigned char *) text.data();
    size_t n = text.size();
    size_t i = 0;
    // SWAR: any high bit set ⇒ non-ASCII
    while (i + 8 <= n) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        if (v & 0x8080808080808080ULL) {
            return false;
        }
        i += 8;
    }
    while (i < n) {
        if (p[i++] & 0x80u) {
            return false;
        }
    }
    return true;
}

static inline bool unicode_is_qwen2_regex_expr(const std::string & regex_expr) {
    return regex_expr ==
           "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
}

// WHY (0116): Qwen3.5 adds [\p{L}\p{M}]+ / exclude \p{M} from punct. No ASCII codepoint is
// \p{M}, so all-ASCII pretok matches the Qwen2 scanner bit-for-bit.
static inline bool unicode_is_qwen35_regex_expr(const std::string & regex_expr) {
    return regex_expr ==
           "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
}

static inline bool unicode_is_gpt2_regex_expr(const std::string & regex_expr) {
    return regex_expr ==
           "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
}

static inline bool unicode_is_llama3_regex_expr(const std::string & regex_expr) {
    return regex_expr ==
               "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
           regex_expr ==
               "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
}

static inline bool unicode_want_simd_pretok() {
    // WHY (0126): SWAR/NEON ASCII letter+digit consume. Opt-out keeps the 0114 8-wide LUT loop.
    return getenv("LLAMA_BPE_NO_SIMD_PRETOK") == nullptr;
}

static inline bool unicode_ascii_is_letter_byte(unsigned char b) {
    return ((b | 0x20u) - (unsigned char) 'a') < 26u;
}

static inline bool unicode_ascii_is_digit_byte(unsigned char b) {
    return (b - (unsigned char) '0') < 10u;
}

// Borrow-safe SWAR (bithacks hasless/hasgreater): high bit set ⇒ lane is NOT a letter.
static inline uint64_t unicode_swar64_hasless(uint64_t x, uint8_t n) {
    return (x - UINT64_C(0x0101010101010101) * n) & ~x & UINT64_C(0x8080808080808080);
}

static inline uint64_t unicode_swar64_hasgreater(uint64_t x, uint8_t n) {
    return ((x + UINT64_C(0x0101010101010101) * (uint64_t) (127u - n)) | x) &
           UINT64_C(0x8080808080808080);
}

static inline uint64_t unicode_swar64_letter_nonmask(uint64_t w) {
    const uint64_t lowered = w | UINT64_C(0x2020202020202020);
    return unicode_swar64_hasless(lowered, (uint8_t) 'a') |
           unicode_swar64_hasgreater(lowered, (uint8_t) 'z');
}

static inline uint64_t unicode_swar64_digit_nonmask(uint64_t w) {
    return unicode_swar64_hasless(w, (uint8_t) '0') | unicode_swar64_hasgreater(w, (uint8_t) '9');
}

#if defined(__aarch64__)
#include <arm_neon.h>

// Returns true if all 16 bytes are ASCII letters (vmax of inverted mask — not vmin).
static inline bool unicode_neon16_all_letters(uint8x16_t v) {
    const uint8x16_t lo = vorrq_u8(v, vdupq_n_u8(0x20));
    const uint8x16_t is_let =
            vandq_u8(vcgeq_u8(lo, vdupq_n_u8('a')), vcleq_u8(lo, vdupq_n_u8('z')));
    return vmaxvq_u8(vmvnq_u8(is_let)) == 0;
}

static inline bool unicode_neon16_all_digits(uint8x16_t v) {
    const uint8x16_t is_dig =
            vandq_u8(vcgeq_u8(v, vdupq_n_u8('0')), vcleq_u8(v, vdupq_n_u8('9')));
    return vmaxvq_u8(vmvnq_u8(is_dig)) == 0;
}
#endif

static inline void unicode_ascii_consume_letters_bytes(
        const unsigned char * text, size_t & pos, size_t end, const uint8_t * cls) {
    if (!unicode_want_simd_pretok()) {
        while (pos + 8 <= end) {
            if (!((cls[text[pos]] & 2) && (cls[text[pos + 1]] & 2) && (cls[text[pos + 2]] & 2) &&
                  (cls[text[pos + 3]] & 2) && (cls[text[pos + 4]] & 2) && (cls[text[pos + 5]] & 2) &&
                  (cls[text[pos + 6]] & 2) && (cls[text[pos + 7]] & 2))) {
                break;
            }
            pos += 8;
        }
        while (pos < end && (cls[text[pos]] & 2)) {
            ++pos;
        }
        return;
    }

    // WHY (0126): SWAR-first (short English words). NEON only after a full 8-letter hit.
    while (pos + 8 <= end) {
        uint64_t w;
        memcpy(&w, text + pos, 8);
        const uint64_t non = unicode_swar64_letter_nonmask(w);
        if (non) {
            pos += (size_t) (__builtin_ctzll(non) >> 3);
            return;
        }
        pos += 8;
#if defined(__aarch64__)
        while (pos + 16 <= end) {
            const uint8x16_t v = vld1q_u8(text + pos);
            if (!unicode_neon16_all_letters(v)) {
                break;
            }
            pos += 16;
        }
#endif
    }
    while (pos < end && unicode_ascii_is_letter_byte(text[pos])) {
        ++pos;
    }
}

static inline void unicode_ascii_consume_digits_bytes(
        const unsigned char * text, size_t & pos, size_t end, const uint8_t * cls) {
    if (!unicode_want_simd_pretok()) {
        while (pos + 8 <= end) {
            if (!((cls[text[pos]] & 4) && (cls[text[pos + 1]] & 4) && (cls[text[pos + 2]] & 4) &&
                  (cls[text[pos + 3]] & 4) && (cls[text[pos + 4]] & 4) && (cls[text[pos + 5]] & 4) &&
                  (cls[text[pos + 6]] & 4) && (cls[text[pos + 7]] & 4))) {
                break;
            }
            pos += 8;
        }
        while (pos < end && (cls[text[pos]] & 4)) {
            ++pos;
        }
        return;
    }

    while (pos + 8 <= end) {
        uint64_t w;
        memcpy(&w, text + pos, 8);
        const uint64_t non = unicode_swar64_digit_nonmask(w);
        if (non) {
            pos += (size_t) (__builtin_ctzll(non) >> 3);
            return;
        }
        pos += 8;
#if defined(__aarch64__)
        while (pos + 16 <= end) {
            const uint8x16_t v = vld1q_u8(text + pos);
            if (!unicode_neon16_all_digits(v)) {
                break;
            }
            pos += 16;
        }
#endif
    }
    while (pos < end && unicode_ascii_is_digit_byte(text[pos])) {
        ++pos;
    }
}

// WHY (0119): fill contiguous remapped words + lens (no vector<string>).
static void unicode_fill_blob_from_byte_offsets(
        const std::string & text, const std::vector<size_t> & bpe_offsets, bool byte_encode,
        unicode_pretok_blob & out) {
    out.storage.clear();
    out.lens.clear();
    out.lens.reserve(bpe_offsets.size());
    out.use_storage = false;
    size_t start = 0;
    if (!byte_encode) {
        for (size_t len : bpe_offsets) {
            out.lens.push_back(len);
            start += len;
        }
        return;
    }
    if (getenv("LLAMA_BPE_NO_BYTE_ENC_FAST") != nullptr) {
        // Legacy encode still uses N strings then remaps — keep via words path.
        std::vector<std::string> words;
        words.reserve(bpe_offsets.size());
        for (size_t len : bpe_offsets) {
            words.emplace_back(text.data() + start, len);
            start += len;
        }
        words = unicode_byte_encoding_process(words);
        size_t total = 0;
        for (const auto & w : words) {
            total += w.size();
        }
        out.storage.reserve(total);
        out.use_storage = true;
        for (const auto & w : words) {
            out.storage.append(w);
            out.lens.push_back(w.size());
        }
        return;
    }

    const unicode_byte_enc * table = unicode_byte_enc_table();
    const unsigned char * base = (const unsigned char *) text.data();

    // Fast path: all words printable ASCII → view into `text` (no copy, no N strings).
    start = 0;
    bool all_printable = true;
    for (size_t len : bpe_offsets) {
        if (!unicode_word_is_printable_ascii(base + start, len)) {
            all_printable = false;
            break;
        }
        start += len;
    }
    if (all_printable) {
        for (size_t len : bpe_offsets) {
            out.lens.push_back(len);
        }
        return;
    }

    // Mixed: concatenate remapped words into one storage buffer.
    out.storage.reserve(text.size() * 2);
    out.use_storage = true;
    start = 0;
    for (size_t len : bpe_offsets) {
        const unsigned char * p = base + start;
        const size_t before = out.storage.size();
        unicode_byte_enc_append_word(out.storage, table, p, len);
        out.lens.push_back(out.storage.size() - before);
        start += len;
    }
}

// WHY (0121): materialize pretok from codepoint-length offsets into blob (no vector<string>).
// bpe_offsets are lengths in codepoints (same as unicode_regex_split custom scanners).
static void unicode_fill_blob_from_cpt_offsets(
        const std::string & text,
        const std::vector<uint32_t> & cpts,
        const std::vector<size_t> & cpt_byte_off,
        const std::vector<size_t> & bpe_offsets,
        bool had_invalid_utf8,
        bool byte_encode,
        unicode_pretok_blob & out) {
    out.storage.clear();
    out.lens.clear();
    out.lens.reserve(bpe_offsets.size());
    out.use_storage = false;

    // Valid UTF-8 + no byte-encode: words are contiguous byte partitions of `text`.
    if (!byte_encode && !had_invalid_utf8) {
        size_t start = 0;
        for (size_t offset : bpe_offsets) {
            const size_t b0 = cpt_byte_off[start];
            const size_t b1 = cpt_byte_off[start + offset];
            out.lens.push_back(b1 - b0);
            start += offset;
        }
        return;
    }

    // Need a contiguous buffer (remap and/or invalid-UTF8 rebuild).
    out.use_storage = true;
    out.storage.reserve(text.size() * 2);
    const bool fast_enc = byte_encode && (getenv("LLAMA_BPE_NO_BYTE_ENC_FAST") == nullptr);
    const unicode_byte_enc * table = fast_enc ? unicode_byte_enc_table() : nullptr;

    size_t start = 0;
    for (size_t offset : bpe_offsets) {
        std::string raw;
        if (!had_invalid_utf8) {
            const size_t b0 = cpt_byte_off[start];
            const size_t b1 = cpt_byte_off[start + offset];
            raw.assign(text.data() + b0, b1 - b0);
        } else {
            raw.reserve(offset * 2);
            for (size_t i = start; i < start + offset; ++i) {
                raw += unicode_cpt_to_utf8(cpts[i]);
            }
        }

        if (!byte_encode) {
            out.storage.append(raw);
            out.lens.push_back(raw.size());
        } else if (fast_enc) {
            const unsigned char * p = (const unsigned char *) raw.data();
            const size_t blen = raw.size();
            const size_t before = out.storage.size();
            unicode_byte_enc_append_word(out.storage, table, p, blen);
            out.lens.push_back(out.storage.size() - before);
        } else {
            // Legacy A/B: one-word encode via the stock helper.
            std::vector<std::string> one{ std::move(raw) };
            one = unicode_byte_encoding_process(one);
            out.storage.append(one[0]);
            out.lens.push_back(one[0].size());
        }
        start += offset;
    }
}

// WHY (0118): fuse pretok offsets → words with optional byte-encode in one pass.
// Old path built N substrings then N encoded copies (printable words paid twice).
// After 0119: blob fill first, then materialize strings only for callers of unicode_regex_split.
static std::vector<std::string> unicode_words_from_byte_offsets(
        const std::string & text, const std::vector<size_t> & bpe_offsets, bool byte_encode) {
    unicode_pretok_blob blob;
    unicode_fill_blob_from_byte_offsets(text, bpe_offsets, byte_encode, blob);
    std::vector<std::string> bpe_words;
    bpe_words.reserve(blob.lens.size());
    const char * p = blob.base(text);
    size_t off = 0;
    for (size_t len : blob.lens) {
        bpe_words.emplace_back(p + off, len);
        off += len;
    }
    return bpe_words;
}

// Same algorithm as unicode_regex_split_qwen2_ascii_seg but on raw ASCII bytes.
static void unicode_regex_split_qwen2_ascii_bytes_seg(
        const unsigned char * text, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const unsigned char cpt = text[pos];
        const uint8_t c = cls[cpt];

        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = unicode_ascii_tolower(text[pos + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t n2 = unicode_ascii_tolower(text[pos + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        if (!(cpt == '\r' || cpt == '\n' || (c & 4))) {
            const bool next_letter = (pos + 1 < end) && (cls[text[pos + 1]] & 2);
            if ((c & 2) || next_letter) {
                ++pos;
                // 8-wide letter consume (T02-ish) — bytes are proven ASCII
                unicode_ascii_consume_letters_bytes(text, pos, end, cls);
                add_token(pos);
                continue;
            }
        }

        if (c & 4) {
            add_token(++pos);
            continue;
        }

        {
            const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[text[pos + 1]]
                               : (cpt == ' ' ? 0 : c);
            if (c2 & 8) {
                pos += (cpt == ' ');
                while (pos < end && (cls[text[pos]] & 8)) {
                    ++pos;
                }
                while (pos < end && (text[pos] == '\r' || text[pos] == '\n')) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        size_t num_ws = 0;
        size_t last_rn = 0;
        while (pos + num_ws < end && (cls[text[pos + num_ws]] & 1)) {
            const unsigned char w = text[pos + num_ws];
            if (w == '\r' || w == '\n') {
                last_rn = pos + num_ws + 1;
            }
            ++num_ws;
        }

        if (last_rn > 0) {
            pos = last_rn;
            add_token(pos);
            continue;
        }

        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }

        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }

        add_token(++pos);
    }
}

static std::vector<std::string> unicode_regex_split_qwen2_ascii_bytes(const std::string & text, bool byte_encode) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(std::max<size_t>(1, text.size() / 3));
    unicode_regex_split_qwen2_ascii_bytes_seg(
            (const unsigned char *) text.data(), 0, text.size(), bpe_offsets);
    return unicode_words_from_byte_offsets(text, bpe_offsets, byte_encode);
}

// WHY (0115): GPT-2 / Llama3 get the same all-ASCII skip-decode path as Qwen2 (0114).
static void unicode_regex_split_gpt2_ascii_bytes_seg(
        const unsigned char * text, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const unsigned char cpt = text[pos];
        const uint8_t c = cls[cpt];

        // 's|'t|'re|'ve|'m|'ll|'d — case-sensitive (stock GPT-2)
        if (cpt == '\'' && pos + 1 < end) {
            const unsigned char n1 = text[pos + 1];
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const unsigned char n2 = text[pos + 2];
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[text[pos + 1]] : (cpt == ' ' ? 0 : c);
        // <space>?\p{L}+
        if (c2 & 2) {
            pos += (cpt == ' ');
            unicode_ascii_consume_letters_bytes(text, pos, end, cls);
            add_token(pos);
            continue;
        }
        // <space>?\p{N}+
        if (c2 & 4) {
            pos += (cpt == ' ');
            unicode_ascii_consume_digits_bytes(text, pos, end, cls);
            add_token(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+
        if ((c2 & 8)) {
            pos += (cpt == ' ');
            while (pos < end && (cls[text[pos]] & 8)) {
                ++pos;
            }
            add_token(pos);
            continue;
        }

        size_t num_ws = 0;
        while (pos + num_ws < end && (cls[text[pos + num_ws]] & 1)) {
            ++num_ws;
        }
        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }
        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }
        add_token(++pos);
    }
}

static std::vector<std::string> unicode_regex_split_gpt2_ascii_bytes(const std::string & text, bool byte_encode) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(std::max<size_t>(1, text.size() / 3));
    unicode_regex_split_gpt2_ascii_bytes_seg(
            (const unsigned char *) text.data(), 0, text.size(), bpe_offsets);
    return unicode_words_from_byte_offsets(text, bpe_offsets, byte_encode);
}

static void unicode_regex_split_llama3_ascii_bytes_seg(
        const unsigned char * text, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const unsigned char cpt = text[pos];
        const uint8_t c = cls[cpt];

        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = unicode_ascii_tolower(text[pos + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t n2 = unicode_ascii_tolower(text[pos + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || (c & 4))) {
            const bool next_letter = (pos + 1 < end) && (cls[text[pos + 1]] & 2);
            if ((c & 2) || next_letter) {
                ++pos;
                unicode_ascii_consume_letters_bytes(text, pos, end, cls);
                add_token(pos);
                continue;
            }
        }

        // \p{N}{1,3}
        if (c & 4) {
            size_t ini_n = pos;
            while (pos < end && (cls[text[pos]] & 4)) {
                if (++pos - ini_n >= 3) {
                    add_token(pos);
                    ini_n = pos;
                }
            }
            add_token(pos);
            continue;
        }

        // Match stock llama3: leading-space branch uses flags2 for category, but
        // `flags.as_uint()` (current codepoint) for the gate — not flags2.as_uint().
        // ASCII LUT: any cls bit ⇒ as_uint() was nonzero when the LUT was built.
        const uint8_t c_lead = c;
        const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[text[pos + 1]] : (cpt == ' ' ? 0 : c);
        if (!(c2 & (1 | 2 | 4)) && c_lead != 0) {
            pos += (cpt == ' ');
            while (pos < end && (cls[text[pos]] & 8)) {
                ++pos;
            }
            while (pos < end && (text[pos] == '\r' || text[pos] == '\n')) {
                ++pos;
            }
            add_token(pos);
            continue;
        }

        size_t num_ws = 0;
        size_t last_rn = 0;
        while (pos + num_ws < end && (cls[text[pos + num_ws]] & 1)) {
            const unsigned char w = text[pos + num_ws];
            if (w == '\r' || w == '\n') {
                last_rn = pos + num_ws + 1;
            }
            ++num_ws;
        }

        if (last_rn > 0) {
            pos = last_rn;
            add_token(pos);
            continue;
        }
        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }
        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }
        add_token(++pos);
    }
}

static std::vector<std::string> unicode_regex_split_llama3_ascii_bytes(const std::string & text, bool byte_encode) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(std::max<size_t>(1, text.size() / 3));
    unicode_regex_split_llama3_ascii_bytes_seg(
            (const unsigned char *) text.data(), 0, text.size(), bpe_offsets);
    return unicode_words_from_byte_offsets(text, bpe_offsets, byte_encode);
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

// WHY (0123): cpt-slice GPT-2 ASCII scanner (mirrors ascii_bytes_seg) for mixed islands.
static void unicode_regex_split_gpt2_ascii_seg(
        const uint32_t * cpts, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const uint32_t cpt = cpts[pos];
        const uint8_t c = cls[cpt];

        // 's|'t|'re|'ve|'m|'ll|'d — case-sensitive (stock GPT-2)
        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = cpts[pos + 1];
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t n2 = cpts[pos + 2];
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[cpts[pos + 1]] : (cpt == ' ' ? 0 : c);
        // <space>?\p{L}+
        if (c2 & 2) {
            pos += (cpt == ' ');
            while (pos < end && (cls[cpts[pos]] & 2)) {
                ++pos;
            }
            add_token(pos);
            continue;
        }
        // <space>?\p{N}+
        if (c2 & 4) {
            pos += (cpt == ' ');
            while (pos < end && (cls[cpts[pos]] & 4)) {
                ++pos;
            }
            add_token(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+
        if (c2 & 8) {
            pos += (cpt == ' ');
            while (pos < end && (cls[cpts[pos]] & 8)) {
                ++pos;
            }
            add_token(pos);
            continue;
        }

        size_t num_ws = 0;
        while (pos + num_ws < end && (cls[cpts[pos + num_ws]] & 1)) {
            ++num_ws;
        }
        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }
        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }
        add_token(++pos);
    }
}


static void unicode_regex_split_gpt2_unicode_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&] (const size_t pos) -> uint32_t {
        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };
    auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
        return (offset_ini <= pos && pos < offset_end) ? unicode_flags_for_cpt_hot(cpts[pos]) : unicode_cpt_flags{};
    };
    size_t _prev_end = offset_ini;
    auto _add_token = [&] (const size_t end) -> size_t {
        assert(_prev_end <= end && end <= offset_end);
        size_t len = end - _prev_end;
        if (len > 0) {
            bpe_offsets.push_back(len);
        }
        _prev_end = end;
        return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
        const uint32_t cpt = _get_cpt(pos);
        const auto flags = _get_flags(pos);

        // regex: 's|'t|'re|'ve|'m|'ll|'d
        if (cpt == '\'' && pos+1 < offset_end) {
            uint32_t cpt_next = _get_cpt(pos+1);
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += _add_token(pos+2);
                continue;
            }
            if (pos+2 < offset_end) {
                uint32_t cpt_next_next = _get_cpt(pos+2);
                if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                    (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += _add_token(pos+3);
                    continue;
                }
            }
        }

        auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
        // regex: <space>?\p{L}+
        if (flags2.is_letter) {
            pos += (cpt == ' ');
            pos = unicode_consume_letters(cpts, pos, offset_end);
            _add_token(pos);
            continue;
        }
        // regex: <space>?\p{N}+
        if (flags2.is_number) {
            pos += (cpt == ' ');
            pos = unicode_consume_digits(cpts, pos, offset_end);
            _add_token(pos);
            continue;
        }
        // regex: <space>?[^\s\p{L}\p{N}]+
        if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
            pos += (cpt == ' ');
            while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                flags2 = _get_flags(++pos);
            }
            _add_token(pos);
            continue;
        }

        size_t num_whitespaces = 0;
        while (_get_flags(pos+num_whitespaces).is_whitespace) {
            num_whitespaces++;
        }

        // regex: \s+(?!\S)
        if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
            pos += num_whitespaces - 1;
            _add_token(pos);
            continue;
        }

        // regex: \s+
        if (num_whitespaces > 0) {
            pos += num_whitespaces;
            _add_token(pos);
            continue;
        }

        // no matches
        _add_token(++pos);
    }
}

// WHY (0123): GPT-2 mixed islands — ASCII gaps via ascii_seg; letter/number/punct islands
// keep optional leading space (` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`).
static void unicode_regex_split_gpt2_mixed_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    auto is_letter = [](uint32_t c) -> bool {
        if (c < 128u) return unicode_cpt_is_ascii_letter(c);
        return unicode_cpt_flags_from_cpt(c).is_letter;
    };
    auto is_number = [](uint32_t c) -> bool {
        if (c < 128u) return unicode_cpt_is_ascii_digit(c);
        return unicode_cpt_flags_from_cpt(c).is_number;
    };
    size_t i = offset_ini;
    while (i < offset_end) {
        size_t j = i;
        while (j < offset_end && cpts[j] < 128u) {
            ++j;
        }
        if (j == offset_end) {
            unicode_regex_split_gpt2_ascii_seg(cpts.data(), i, offset_end, bpe_offsets);
            return;
        }

        size_t L = j;
        size_t R = j + 1;
        if (is_letter(cpts[j])) {
            while (L > i && is_letter(cpts[L - 1])) {
                --L;
            }
            while (R < offset_end && is_letter(cpts[R])) {
                ++R;
            }
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
        } else if (is_number(cpts[j])) {
            while (L > i && is_number(cpts[L - 1])) {
                --L;
            }
            while (R < offset_end && is_number(cpts[R])) {
                ++R;
            }
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
        } else {
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
            R = j;
            while (R < offset_end) {
                const auto f = unicode_flags_for_cpt_hot(cpts[R]);
                if (f.is_whitespace || f.is_letter || f.is_number || !f.as_uint()) {
                    break;
                }
                ++R;
            }
            if (R <= j) {
                R = j + 1;
            }
        }

        if (L > i) {
            unicode_regex_split_gpt2_ascii_seg(cpts.data(), i, L, bpe_offsets);
        }
        unicode_regex_split_gpt2_unicode_seg(cpts, L, R, bpe_offsets);
        i = R;
    }
}

static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(std::max(offsets.size(), cpts.size() / 3));
    const bool use_ascii = unicode_want_ascii_pretok();

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        if (use_ascii) {
            if (unicode_seg_is_ascii(cpts.data(), offset_ini, offset_end)) {
                unicode_regex_split_gpt2_ascii_seg(cpts.data(), offset_ini, offset_end, bpe_offsets);
            } else {
                unicode_regex_split_gpt2_mixed_seg(cpts, offset_ini, offset_end, bpe_offsets);
            }
            continue;
        }
        unicode_regex_split_gpt2_unicode_seg(cpts, offset_ini, offset_end, bpe_offsets);
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

// WHY (0123): cpt-slice Llama3 ASCII scanner for mixed islands.
static void unicode_regex_split_llama3_ascii_seg(
        const uint32_t * cpts, size_t ini, size_t end, std::vector<size_t> & bpe_offsets) {
    const uint8_t * cls = unicode_ascii_cls_lut();
    size_t prev = ini;
    auto add_token = [&](size_t e) {
        if (e > prev) {
            bpe_offsets.push_back(e - prev);
            prev = e;
        }
    };

    size_t pos = ini;
    while (pos < end) {
        const uint32_t cpt = cpts[pos];
        const uint8_t c = cls[cpt];

        if (cpt == '\'' && pos + 1 < end) {
            const uint32_t n1 = unicode_ascii_tolower(cpts[pos + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                add_token(pos + 2);
                pos = prev;
                continue;
            }
            if (pos + 2 < end) {
                const uint32_t n2 = unicode_ascii_tolower(cpts[pos + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    add_token(pos + 3);
                    pos = prev;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || (c & 4))) {
            const bool next_letter = (pos + 1 < end) && (cls[cpts[pos + 1]] & 2);
            if ((c & 2) || next_letter) {
                ++pos;
                while (pos < end && (cls[cpts[pos]] & 2)) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        // \p{N}{1,3}
        if (c & 4) {
            size_t ini_n = pos;
            while (pos < end && (cls[cpts[pos]] & 4)) {
                if (++pos - ini_n >= 3) {
                    add_token(pos);
                    ini_n = pos;
                }
            }
            add_token(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        {
            const uint8_t c2 = (cpt == ' ' && pos + 1 < end) ? cls[cpts[pos + 1]] : (cpt == ' ' ? 0 : c);
            if (c2 & 8) {
                pos += (cpt == ' ');
                while (pos < end && (cls[cpts[pos]] & 8)) {
                    ++pos;
                }
                while (pos < end && (cpts[pos] == '\r' || cpts[pos] == '\n')) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        size_t num_ws = 0;
        size_t last_rn = 0;
        while (pos + num_ws < end && (cls[cpts[pos + num_ws]] & 1)) {
            const uint32_t w = cpts[pos + num_ws];
            if (w == '\r' || w == '\n') {
                last_rn = pos + num_ws + 1;
            }
            ++num_ws;
        }
        if (last_rn > 0) {
            pos = last_rn;
            add_token(pos);
            continue;
        }
        if (num_ws > 1 && pos + num_ws < end) {
            pos += num_ws - 1;
            add_token(pos);
            continue;
        }
        if (num_ws > 0) {
            pos += num_ws;
            add_token(pos);
            continue;
        }
        add_token(++pos);
    }
}


static void unicode_regex_split_llama3_unicode_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&] (const size_t pos) -> uint32_t {
        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };
    auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
        return (offset_ini <= pos && pos < offset_end) ? unicode_flags_for_cpt_hot(cpts[pos]) : unicode_cpt_flags{};
    };
    size_t _prev_end = offset_ini;
    auto _add_token = [&] (const size_t end) -> size_t {
        assert(_prev_end <= end && end <= offset_end);
        size_t len = end - _prev_end;
        if (len > 0) {
            bpe_offsets.push_back(len);
        }
        _prev_end = end;
        return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
        const uint32_t cpt = _get_cpt(pos);
        const auto flags = _get_flags(pos);

        if (cpt == '\'' && pos+1 < offset_end) {
            uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += _add_token(pos+2);
                continue;
            }
            if (pos+2 < offset_end) {
                uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                    (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += _add_token(pos+3);
                    continue;
                }
            }
        }

        if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
            if (flags.is_letter || _get_flags(pos+1).is_letter) {
                pos++;
                pos = unicode_consume_letters(cpts, pos, offset_end);
                _add_token(pos);
                continue;
            }
        }

        if (flags.is_number) {
            size_t ini = pos;
            while (pos < offset_end) {
                const uint32_t cn = cpts[pos];
                const bool is_num = (cn < 128u) ? unicode_cpt_is_ascii_digit(cn)
                                                : unicode_cpt_flags_from_cpt(cn).is_number;
                if (!is_num) {
                    break;
                }
                if (++pos - ini >= 3 ) {
                    _add_token(pos);
                    ini = pos;
                }
            }
            _add_token(pos);
            continue;
        }

        auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
        if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
            pos += (cpt == ' ');
            while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                flags2 = _get_flags(++pos);
            }
            uint32_t cpt2 = _get_cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') {
                cpt2 = _get_cpt(++pos);
            }
            _add_token(pos);
            continue;
        }

        size_t num_whitespaces = 0;
        size_t last_end_r_or_n = 0;
        while (_get_flags(pos+num_whitespaces).is_whitespace) {
            uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
            if (cpt2 == '\r' || cpt2 == '\n') {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces++;
        }

        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            _add_token(pos);
            continue;
        }

        if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
            pos += num_whitespaces - 1;
            _add_token(pos);
            continue;
        }

        if (num_whitespaces > 0) {
            pos += num_whitespaces;
            _add_token(pos);
            continue;
        }

        _add_token(++pos);
    }
}

// WHY (0123): Llama3 mixed islands — same letter/punct shape as Qwen2 (0122); numbers are \p{N}{1,3}.
static void unicode_regex_split_llama3_mixed_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    auto is_letter = [](uint32_t c) -> bool {
        if (c < 128u) return unicode_cpt_is_ascii_letter(c);
        return unicode_cpt_flags_from_cpt(c).is_letter;
    };
    size_t i = offset_ini;
    while (i < offset_end) {
        size_t j = i;
        while (j < offset_end && cpts[j] < 128u) {
            ++j;
        }
        if (j == offset_end) {
            unicode_regex_split_llama3_ascii_seg(cpts.data(), i, offset_end, bpe_offsets);
            return;
        }

        size_t L = j;
        size_t R = j + 1;
        const bool letter_island = is_letter(cpts[j]);
        if (letter_island) {
            while (L > i && is_letter(cpts[L - 1])) {
                --L;
            }
            while (R < offset_end && is_letter(cpts[R])) {
                ++R;
            }
            if (L > i && L < offset_end && is_letter(cpts[L])) {
                const uint32_t prev = cpts[L - 1];
                if (prev != '\r' && prev != '\n') {
                    const auto pf = unicode_flags_for_cpt_hot(prev);
                    if (!pf.is_letter && !pf.is_number) {
                        --L;
                    }
                }
            }
        } else {
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
            R = j;
            while (R < offset_end) {
                const auto f = unicode_flags_for_cpt_hot(cpts[R]);
                if (f.is_whitespace || f.is_letter || f.is_number || !f.as_uint()) {
                    break;
                }
                ++R;
            }
            while (R < offset_end && (cpts[R] == '\r' || cpts[R] == '\n')) {
                ++R;
            }
            if (R <= j) {
                R = j + 1;
            }
        }

        if (L > i) {
            unicode_regex_split_llama3_ascii_seg(cpts.data(), i, L, bpe_offsets);
        }
        unicode_regex_split_llama3_unicode_seg(cpts, L, R, bpe_offsets);
        i = R;
    }
}

static std::vector<size_t> unicode_regex_split_custom_llama3(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(std::max(offsets.size(), cpts.size() / 3));
    const bool use_ascii = unicode_want_ascii_pretok();

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        if (use_ascii) {
            if (unicode_seg_is_ascii(cpts.data(), offset_ini, offset_end)) {
                unicode_regex_split_llama3_ascii_seg(cpts.data(), offset_ini, offset_end, bpe_offsets);
            } else {
                unicode_regex_split_llama3_mixed_seg(cpts, offset_ini, offset_end, bpe_offsets);
            }
            continue;
        }
        unicode_regex_split_llama3_unicode_seg(cpts, offset_ini, offset_end, bpe_offsets);
    }

    return bpe_offsets;
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
// Match unicode_consume_letters letter predicate (ASCII + \p{L}).
static inline bool unicode_cpt_is_letter_run(uint32_t c) {
    if (c < 128u) {
        return unicode_cpt_is_ascii_letter(c);
    }
    return unicode_cpt_flags_from_cpt(c).is_letter;
}

// Full Unicode Qwen2 scanner on [offset_ini, offset_end).
static void unicode_regex_split_qwen2_unicode_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&] (const size_t pos) -> uint32_t {
        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };

    auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
        return (offset_ini <= pos && pos < offset_end) ? unicode_flags_for_cpt_hot(cpts[pos]) : unicode_cpt_flags{};
    };

    size_t _prev_end = offset_ini;
    auto _add_token = [&] (const size_t end) -> size_t {
        assert(_prev_end <= end && end <= offset_end);
        size_t len = end - _prev_end;
        if (len > 0) {
            bpe_offsets.push_back(len);
        }
        _prev_end = end;
        return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
        const uint32_t cpt = _get_cpt(pos);
        const auto flags = _get_flags(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
        if (cpt == '\'' && pos+1 < offset_end) {
            uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += _add_token(pos+2);
                continue;
            }
            if (pos+2 < offset_end) {
                uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                    (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += _add_token(pos+3);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
            if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                pos++;
                pos = unicode_consume_letters(cpts, pos, offset_end);
                _add_token(pos);
                continue;
            }
        }

        // regex: \p{N}
        if (flags.is_number) {
            pos++;
            _add_token(pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
        auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
        if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
            pos += (cpt == ' ');
            while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                flags2 = _get_flags(++pos);
            }
            uint32_t cpt2 = _get_cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') {
                cpt2 = _get_cpt(++pos);
            }
            _add_token(pos);
            continue;
        }

        size_t num_whitespaces = 0;
        size_t last_end_r_or_n = 0;
        while (_get_flags(pos+num_whitespaces).is_whitespace) {
            uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
            if (cpt2 == '\r' || cpt2 == '\n') {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces++;
        }

        // regex: \s*[\r\n]+
        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            _add_token(pos);
            continue;
        }

        // regex: \s+(?!\S)
        if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
            pos += num_whitespaces - 1;
            _add_token(pos);
            continue;
        }

        // regex: \s+
        if (num_whitespaces > 0) {
            pos += num_whitespaces;
            _add_token(pos);
            continue;
        }

        // no matches
        _add_token(++pos);
    }
}

// WHY (0122): ASCII-majority mixed prompts fail unicode_seg_is_ascii on a few CJK/latin1
// bytes, then paid the full Unicode scanner on the whole MiB. Split into ASCII islands
// (ascii_seg) and non-ASCII islands (unicode_seg). Letter runs expand through \p{L}
// (café / hello世界). Punctuation/emoji islands keep optional leading space and the
// `?[^\s\p{L}\p{N}]+[\r\n]*` span (` ·`, ` 🚀\n`) — splitting those breaks identity.
// Opt-out: LLAMA_BPE_NO_ASCII_PRETOK.
static void unicode_regex_split_qwen2_mixed_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    size_t i = offset_ini;
    while (i < offset_end) {
        size_t j = i;
        while (j < offset_end && cpts[j] < 128u) {
            ++j;
        }
        if (j == offset_end) {
            unicode_regex_split_qwen2_ascii_seg(cpts.data(), i, offset_end, bpe_offsets);
            return;
        }

        size_t L = j;
        size_t R = j + 1;
        // Letter island only when the non-ASCII cpt itself is \p{L} (or ASCII letter
        // glued left). Do NOT start a letter island on punctuation just because the
        // next cpt is a letter — stock prefers ` ?[^\s\p{L}\p{N}]+` for ` ·` / ` 🚀`
        // before a following letter run (`hello ·世界`).
        const bool letter_island = unicode_cpt_is_letter_run(cpts[j]);

        if (letter_island) {
            // [^\r\n\p{L}\p{N}]?\p{L}+ — expand through contiguous letters.
            while (L > i && unicode_cpt_is_letter_run(cpts[L - 1])) {
                --L;
            }
            while (R < offset_end && unicode_cpt_is_letter_run(cpts[R])) {
                ++R;
            }
            if (L > i && L < offset_end && unicode_cpt_is_letter_run(cpts[L])) {
                const uint32_t prev = cpts[L - 1];
                if (prev != '\r' && prev != '\n') {
                    const auto pf = unicode_flags_for_cpt_hot(prev);
                    if (!pf.is_letter && !pf.is_number) {
                        --L;
                    }
                }
            }
        } else {
            // <space>?[^\s\p{L}\p{N}]+[\r\n]* — keep optional space with the specials run.
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
            R = j;
            while (R < offset_end) {
                const auto f = unicode_flags_for_cpt_hot(cpts[R]);
                if (f.is_whitespace || f.is_letter || f.is_number || !f.as_uint()) {
                    break;
                }
                ++R;
            }
            while (R < offset_end && (cpts[R] == '\r' || cpts[R] == '\n')) {
                ++R;
            }
            if (R <= j) {
                R = j + 1;
            }
        }

        if (L > i) {
            unicode_regex_split_qwen2_ascii_seg(cpts.data(), i, L, bpe_offsets);
        }
        unicode_regex_split_qwen2_unicode_seg(cpts, L, R, bpe_offsets);
        i = R;
    }
}

static std::vector<size_t> unicode_regex_split_custom_qwen2(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    // WHY (0110): offsets.size() is usually 1 (whole prompt); word count is ~nbytes/3–6.
    bpe_offsets.reserve(std::max(offsets.size(), cpts.size() / 3));
    const bool use_ascii = unicode_want_ascii_pretok();

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        // WHY (0110): pure-ASCII megaprompts (typical English agent seeds) — LUT path.
        // WHY (0122): mixed → ASCII islands + Unicode islands (same opt-out).
        if (use_ascii) {
            if (unicode_seg_is_ascii(cpts.data(), offset_ini, offset_end)) {
                unicode_regex_split_qwen2_ascii_seg(cpts.data(), offset_ini, offset_end, bpe_offsets);
            } else {
                unicode_regex_split_qwen2_mixed_seg(cpts, offset_ini, offset_end, bpe_offsets);
            }
            continue;
        }

        unicode_regex_split_qwen2_unicode_seg(cpts, offset_ini, offset_end, bpe_offsets);
    }

    return bpe_offsets;
}

// Qwen3.5 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
// Compared to Qwen2, letter-runs also consume Unicode combining marks (\p{M}): [\p{L}\p{M}]+ instead of \p{L}+
static void unicode_regex_split_qwen35_unicode_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&] (const size_t pos) -> uint32_t {
        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
    };
    auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
        return (offset_ini <= pos && pos < offset_end) ? unicode_flags_for_cpt_hot(cpts[pos]) : unicode_cpt_flags{};
    };
    size_t _prev_end = offset_ini;
    auto _add_token = [&] (const size_t end) -> size_t {
        assert(_prev_end <= end && end <= offset_end);
        size_t len = end - _prev_end;
        if (len > 0) {
            bpe_offsets.push_back(len);
        }
        _prev_end = end;
        return len;
    };

    for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
        const uint32_t cpt = _get_cpt(pos);
        const auto flags = _get_flags(pos);

        if (cpt == '\'' && pos+1 < offset_end) {
            uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += _add_token(pos+2);
                continue;
            }
            if (pos+2 < offset_end) {
                uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                    (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += _add_token(pos+3);
                    continue;
                }
            }
        }

        if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
            if (flags.is_letter || flags.is_accent_mark || _get_flags(pos + 1).is_accent_mark || _get_flags(pos+1).is_letter) {
                pos++;
                pos = unicode_consume_letters_or_marks(cpts, pos, offset_end);
                _add_token(pos);
                continue;
            }
        }

        if (flags.is_number) {
            pos++;
            _add_token(pos);
            continue;
        }

        auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
        if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags.as_uint()) {
            pos += (cpt == ' ');
            while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags2.as_uint()) {
                flags2 = _get_flags(++pos);
            }
            uint32_t cpt2 = _get_cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') {
                cpt2 = _get_cpt(++pos);
            }
            _add_token(pos);
            continue;
        }

        size_t num_whitespaces = 0;
        size_t last_end_r_or_n = 0;
        while (_get_flags(pos+num_whitespaces).is_whitespace) {
            uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
            if (cpt2 == '\r' || cpt2 == '\n') {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces++;
        }

        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            _add_token(pos);
            continue;
        }

        if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
            pos += num_whitespaces - 1;
            _add_token(pos);
            continue;
        }

        if (num_whitespaces > 0) {
            pos += num_whitespaces;
            _add_token(pos);
            continue;
        }

        _add_token(++pos);
    }
}

// WHY (0123): Qwen3.5 mixed islands — ASCII gaps reuse Qwen2 ascii_seg (no ASCII is \p{M});
// letter islands expand through \p{L}/\p{M}; punct excludes marks.
static void unicode_regex_split_qwen35_mixed_seg(
        const std::vector<uint32_t> & cpts, size_t offset_ini, size_t offset_end,
        std::vector<size_t> & bpe_offsets) {
    auto is_letter_or_mark = [](uint32_t c) -> bool {
        if (c < 128u) return unicode_cpt_is_ascii_letter(c);
        const auto f = unicode_cpt_flags_from_cpt(c);
        return f.is_letter || f.is_accent_mark;
    };
    size_t i = offset_ini;
    while (i < offset_end) {
        size_t j = i;
        while (j < offset_end && cpts[j] < 128u) {
            ++j;
        }
        if (j == offset_end) {
            unicode_regex_split_qwen2_ascii_seg(cpts.data(), i, offset_end, bpe_offsets);
            return;
        }

        size_t L = j;
        size_t R = j + 1;
        const bool letter_island = is_letter_or_mark(cpts[j]);
        if (letter_island) {
            while (L > i && is_letter_or_mark(cpts[L - 1])) {
                --L;
            }
            while (R < offset_end && is_letter_or_mark(cpts[R])) {
                ++R;
            }
            if (L > i && L < offset_end && is_letter_or_mark(cpts[L])) {
                const uint32_t prev = cpts[L - 1];
                if (prev != '\r' && prev != '\n') {
                    const auto pf = unicode_flags_for_cpt_hot(prev);
                    if (!pf.is_letter && !pf.is_accent_mark && !pf.is_number) {
                        --L;
                    }
                }
            }
        } else {
            if (L > i && cpts[L - 1] == ' ') {
                --L;
            }
            R = j;
            while (R < offset_end) {
                const auto f = unicode_flags_for_cpt_hot(cpts[R]);
                if (f.is_whitespace || f.is_letter || f.is_accent_mark || f.is_number || !f.as_uint()) {
                    break;
                }
                ++R;
            }
            while (R < offset_end && (cpts[R] == '\r' || cpts[R] == '\n')) {
                ++R;
            }
            if (R <= j) {
                R = j + 1;
            }
        }

        if (L > i) {
            unicode_regex_split_qwen2_ascii_seg(cpts.data(), i, L, bpe_offsets);
        }
        unicode_regex_split_qwen35_unicode_seg(cpts, L, R, bpe_offsets);
        i = R;
    }
}

static std::vector<size_t> unicode_regex_split_custom_qwen35(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    // WHY (0116): same reserve heuristic as Qwen2; ASCII segments reuse Qwen2 ascii_seg
    // because no U+0000..007F codepoint is \p{M}.
    bpe_offsets.reserve(std::max(offsets.size(), cpts.size() / 3));
    const bool use_ascii = unicode_want_ascii_pretok();

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        if (use_ascii) {
            if (unicode_seg_is_ascii(cpts.data(), offset_ini, offset_end)) {
                unicode_regex_split_qwen2_ascii_seg(cpts.data(), offset_ini, offset_end, bpe_offsets);
            } else {
                unicode_regex_split_qwen35_mixed_seg(cpts, offset_ini, offset_end, bpe_offsets);
            }
            continue;
        }

        unicode_regex_split_qwen35_unicode_seg(cpts, offset_ini, offset_end, bpe_offsets);
    }

    return bpe_offsets;
}


template <typename CharT>
static std::vector<size_t> unicode_regex_split_stl(const std::basic_string<CharT> & text, const std::basic_string<CharT> & regex, const std::vector<size_t> & offsets) {
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    // Bypass bug in MSVC: https://github.com/ggml-org/llama.cpp/issues/17830
    constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
    constexpr auto regex_flags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, regex_flags);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        std::regex_iterator<BidirIt> end;

        int64_t start_idx = 0;
        while (it != end) {
            std::match_results<BidirIt> match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_kimi_k2(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // Pattern 1: [\p{Han}]+ (Chinese characters)
            if (unicode_cpt_is_han(cpt)) {
                while (unicode_cpt_is_han(_get_cpt(pos))) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }

            // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // Check if current char is a letter OR if current char could be a leading char and next char is a letter
            bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
                                     (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
                                      _get_flags(pos + 1).is_letter && !unicode_cpt_is_han(_get_cpt(pos + 1)));

            if (is_letter_pattern) {
                // Handle optional leading non-letter/non-number character
                bool has_leading_char = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number)) {
                    has_leading_char = true;
                    pos++;
                }

                // Match letter sequence (excluding Han characters)
                bool has_letters = false;
                while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                    has_letters = true;
                    pos++;
                }

                // Only proceed if we found letters (after potentially skipping leading char)
                if (has_letters || (!has_leading_char && _get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++; // consume the first letter if we didn't already

                    // Continue consuming letters
                    while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                        pos++;
                    }

                    // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if (_get_cpt(pos) == '\'' && pos + 1 < offset_end) {
                        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                            pos += 2;
                        } else if (pos + 2 < offset_end) {
                            uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                            if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                                (cpt_next == 'v' && cpt_next_next == 'e') ||
                                (cpt_next == 'l' && cpt_next_next == 'l')) {
                                pos += 3;
                            }
                        }
                    }

                    _add_token(pos);
                    continue;
                } else if (has_leading_char) {
                    // We consumed a leading char but found no letters, backtrack
                    pos--;
                }
            }

            // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                // Match optional [\r\n]*
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            // Count whitespace characters
            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S) (trailing whitespace)
            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // Pattern 8: \s+ (general whitespace)
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // No matches - consume single character
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// AFMOE digit handling: splits digits with leading 1-2 based on total length modulo 3
static std::vector<size_t> unicode_regex_split_custom_afmoe(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);

            // Handle digit sequences with special splitting logic
            if (flags.is_number) {
                size_t digit_start = pos;
                size_t digit_count = 0;

                // Count consecutive digits
                while (_get_flags(pos).is_number && pos < offset_end) {
                    digit_count++;
                    pos++;
                }

                // Split based on total length modulo 3
                size_t remainder = digit_count % 3;
                size_t current = digit_start;

                // Emit leading 1-2 digits if needed
                if (remainder > 0) {
                    _add_token(current + remainder);
                    current += remainder;
                }

                // Emit groups of 3
                while (current < digit_start + digit_count) {
                    _add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            // For non-digits, just move forward
            pos++;
        }

        // Add any remaining content
        if (_prev_end < offset_end) {
            _add_token(offset_end);
        }
    }

    return bpe_offsets;
}

// regex: [^\n]+|[\n]+
// splits text into runs of non-newline characters and runs of newline characters
static std::vector<size_t> unicode_regex_split_custom_newlines(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t pos = offset_ini;
        while (pos < offset_end) {
            const bool is_newline = (cpts[pos] == '\n');
            const size_t run_start = pos;
            while (pos < offset_end && (cpts[pos] == '\n') == is_newline) {
                pos++;
            }
            bpe_offsets.push_back(pos - run_start);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::vector<uint32_t> & cpts, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(cpts, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_llama3(cpts, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen2(cpts, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen35(cpts, offsets);
    } else if (regex_expr == "\\p{Han}+") {
        // K2's first pattern - handle all K2 patterns together
        bpe_offsets = unicode_regex_split_custom_kimi_k2(cpts, offsets);
    } else if (regex_expr == "\\p{AFMoE_digits}") {
        // AFMOE digit pattern - use custom implementation for proper splitting
        bpe_offsets = unicode_regex_split_custom_afmoe(cpts, offsets);
    } else if (regex_expr == "[^\\n]+|[\\n]+") {
        bpe_offsets = unicode_regex_split_custom_newlines(cpts, offsets);
    } else if (regex_expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        // tiny_aya digit grouping pattern from tokenizer.json:
        //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
        // Splits digits into groups of 3 from the right (e.g., 1234567 -> 1, 234, 567)
        // TODO: Revisit this regex, in case there are any subtle tokenization differences with the original regex.
        bpe_offsets = unicode_regex_split_custom_afmoe(cpts, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

// Append UTF-8 encoding of cpt into out without a temporary std::string.
// WHY: pretok word materialization used to `s += unicode_cpt_to_utf8(cpt)` per codepoint.
static void unicode_cpt_append_utf8(std::string & out, uint32_t cpt) {
    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        out.push_back(static_cast<char>(cpt));
        return;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | ((cpt >> 6) & 0x1f)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        return;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | ((cpt >> 12) & 0x0f)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        return;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        out.push_back(static_cast<char>(0xf0 | ((cpt >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cpt & 0x3f)));
        return;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;
    unicode_cpt_append_utf8(result, cpt);
    return result;
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

bool unicode_cpt_is_han(uint32_t cpt) {
    // Han character ranges (Chinese/CJK characters)
    // CJK Unified Ideographs (most common)
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;

    // CJK Extension A
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;

    // CJK Extension B
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;

    // CJK Extension C
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;

    // CJK Extension D
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;

    // CJK Extension E
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;

    // CJK Extension F
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;

    // CJK Compatibility Ideographs
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;

    // CJK Compatibility Ideographs Supplement
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;

    return false;
}

// Internal: when blob_out != nullptr, fill blob and return empty vector; else return words.
static std::vector<std::string> unicode_regex_split_impl(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode,
        unicode_pretok_blob * blob_out);


// WHY (0124): mixed megaprompts still decode the whole string to uint32 before 0122/0123
// cpt islands. Byte-level islands run ascii_bytes_seg on ASCII gaps and only decode
// non-ASCII islands — same boundaries as cpt mixed_seg. Opt-out: LLAMA_BPE_NO_BYTE_MIXED=1
// (falls through to full cpt path). Also disabled when LLAMA_BPE_NO_ASCII_PRETOK=1.
static inline bool unicode_want_byte_mixed() {
    return getenv("LLAMA_BPE_NO_BYTE_MIXED") == nullptr;
}

static size_t unicode_find_non_ascii_byte(const unsigned char * p, size_t i, size_t n) {
    while (i + 8 <= n) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        if (v & 0x8080808080808080ULL) {
            while (i < n && p[i] < 0x80) {
                ++i;
            }
            return i;
        }
        i += 8;
    }
    while (i < n && p[i] < 0x80) {
        ++i;
    }
    return i;
}

static size_t unicode_utf8_lead_at(const unsigned char * p, size_t i, size_t floor) {
    while (i > floor && (p[i] & 0xC0) == 0x80) {
        --i;
    }
    return i;
}

static bool unicode_decode_one_bounded(
        const std::string & text, size_t & offset, size_t end, uint32_t & cpt) {
    if (offset >= end) {
        return false;
    }
    const unsigned char b = (unsigned char) text[offset];
    if (b < 0x80) {
        cpt = b;
        ++offset;
        return true;
    }
    size_t o = offset;
    try {
        cpt = unicode_cpt_from_utf8(text, o);
    } catch (const std::invalid_argument &) {
        return false;
    }
    if (o > end) {
        return false;
    }
    offset = o;
    return true;
}

static bool unicode_decode_range_to_cpts(
        const std::string & text, size_t b0, size_t b1,
        std::vector<uint32_t> & cpts, std::vector<size_t> & cpt_byte_off) {
    cpts.clear();
    cpt_byte_off.clear();
    cpt_byte_off.push_back(0); // relative to island start
    size_t offset = b0;
    while (offset < b1) {
        const size_t remain = b1 - offset;
        const unsigned char * p = (const unsigned char *) text.data() + offset;
        size_t n = 0;
        while (n + 8 <= remain) {
            uint64_t v;
            memcpy(&v, p + n, 8);
            if (v & 0x8080808080808080ULL) {
                break;
            }
            n += 8;
        }
        while (n < remain && p[n] < 0x80) {
            ++n;
        }
        if (n > 0) {
            const size_t base = cpts.size();
            cpts.resize(base + n);
            for (size_t i = 0; i < n; ++i) {
                cpts[base + i] = p[i];
                cpt_byte_off.push_back(offset - b0 + i + 1);
            }
            offset += n;
            continue;
        }
        size_t o = offset;
        uint32_t cpt;
        try {
            cpt = unicode_cpt_from_utf8(text, o);
        } catch (const std::invalid_argument &) {
            return false;
        }
        if (o > b1) {
            return false;
        }
        cpts.push_back(cpt);
        cpt_byte_off.push_back(o - b0);
        offset = o;
    }
    return offset == b1;
}

static void unicode_append_cpt_lens_as_bytes(
        const std::vector<size_t> & cpt_byte_off,
        const std::vector<size_t> & cpt_lens,
        std::vector<size_t> & byte_offsets) {
    size_t start = 0;
    for (size_t clen : cpt_lens) {
        const size_t b0 = cpt_byte_off[start];
        const size_t b1 = cpt_byte_off[start + clen];
        if (b1 > b0) {
            byte_offsets.push_back(b1 - b0);
        }
        start += clen;
    }
}

enum class unicode_byte_mixed_family {
    Qwen2,
    Qwen35,
    Gpt2,
    Llama3,
};

static inline bool unicode_cpt_is_letter_or_mark_run(uint32_t c) {
    if (c < 128u) {
        return unicode_cpt_is_ascii_letter(c);
    }
    const auto f = unicode_cpt_flags_from_cpt(c);
    return f.is_letter || f.is_accent_mark;
}

// Returns false → caller falls through to full cpt decode path.
static bool unicode_regex_split_try_byte_mixed(
        const std::string & text,
        unicode_byte_mixed_family family,
        bool byte_encode,
        unicode_pretok_blob * blob_out,
        std::vector<std::string> * words_out) {
    if (!unicode_want_byte_mixed()) {
        return false;
    }
    const unsigned char * bytes = (const unsigned char *) text.data();
    const size_t n = text.size();
    std::vector<size_t> byte_offsets;
    byte_offsets.reserve(std::max<size_t>(1, n / 3));

    auto is_letter = [&](uint32_t c) -> bool {
        if (family == unicode_byte_mixed_family::Qwen35) {
            return unicode_cpt_is_letter_or_mark_run(c);
        }
        if (c < 128u) {
            return unicode_cpt_is_ascii_letter(c);
        }
        return unicode_cpt_flags_from_cpt(c).is_letter;
    };
    auto is_number = [](uint32_t c) -> bool {
        if (c < 128u) {
            return unicode_cpt_is_ascii_digit(c);
        }
        return unicode_cpt_flags_from_cpt(c).is_number;
    };

    size_t i = 0;
    std::vector<uint32_t> island_cpts;
    std::vector<size_t> island_off;
    std::vector<size_t> island_lens;

    while (i < n) {
        size_t j = unicode_find_non_ascii_byte(bytes, i, n);
        if (j == n) {
            if (family == unicode_byte_mixed_family::Gpt2) {
                unicode_regex_split_gpt2_ascii_bytes_seg(bytes, i, n, byte_offsets);
            } else if (family == unicode_byte_mixed_family::Llama3) {
                unicode_regex_split_llama3_ascii_bytes_seg(bytes, i, n, byte_offsets);
            } else {
                unicode_regex_split_qwen2_ascii_bytes_seg(bytes, i, n, byte_offsets);
            }
            break;
        }
        j = unicode_utf8_lead_at(bytes, j, i);

        size_t j_end = j;
        uint32_t c0;
        if (!unicode_decode_one_bounded(text, j_end, n, c0)) {
            return false;
        }

        size_t L = j;
        size_t R = j_end;
        const bool gpt2 = (family == unicode_byte_mixed_family::Gpt2);

        if (is_letter(c0)) {
            // Expand left through letters.
            while (L > i) {
                const size_t prev = unicode_utf8_lead_at(bytes, L - 1, i);
                size_t t = prev;
                uint32_t pc;
                if (!unicode_decode_one_bounded(text, t, L, pc) || t != L) {
                    return false;
                }
                if (!is_letter(pc)) {
                    break;
                }
                L = prev;
            }
            // Optional prefix
            if (L > i) {
                if (gpt2) {
                    if (bytes[L - 1] == ' ') {
                        --L;
                    }
                } else {
                    const size_t prev = unicode_utf8_lead_at(bytes, L - 1, i);
                    size_t t = prev;
                    uint32_t pc;
                    if (!unicode_decode_one_bounded(text, t, L, pc) || t != L) {
                        return false;
                    }
                    if (pc != '\r' && pc != '\n' && !is_letter(pc) && !is_number(pc)) {
                        if (family == unicode_byte_mixed_family::Qwen35) {
                            const auto pf = unicode_flags_for_cpt_hot(pc);
                            if (!pf.is_accent_mark) {
                                L = prev;
                            }
                        } else {
                            L = prev;
                        }
                    }
                }
            }
            // Expand right
            while (R < n) {
                size_t t = R;
                uint32_t pc;
                if (!unicode_decode_one_bounded(text, t, n, pc)) {
                    return false;
                }
                if (!is_letter(pc)) {
                    break;
                }
                R = t;
            }
        } else if (gpt2 && is_number(c0)) {
            while (L > i) {
                const size_t prev = unicode_utf8_lead_at(bytes, L - 1, i);
                size_t t = prev;
                uint32_t pc;
                if (!unicode_decode_one_bounded(text, t, L, pc) || t != L) {
                    return false;
                }
                if (!is_number(pc)) {
                    break;
                }
                L = prev;
            }
            if (L > i && bytes[L - 1] == ' ') {
                --L;
            }
            while (R < n) {
                size_t t = R;
                uint32_t pc;
                if (!unicode_decode_one_bounded(text, t, n, pc)) {
                    return false;
                }
                if (!is_number(pc)) {
                    break;
                }
                R = t;
            }
        } else {
            // Punctuation / emoji island
            if (L > i && bytes[L - 1] == ' ') {
                --L;
            }
            R = j;
            while (R < n) {
                size_t t = R;
                uint32_t pc;
                if (!unicode_decode_one_bounded(text, t, n, pc)) {
                    return false;
                }
                const auto f = unicode_flags_for_cpt_hot(pc);
                if (family == unicode_byte_mixed_family::Qwen35) {
                    if (f.is_whitespace || f.is_letter || f.is_accent_mark || f.is_number || !f.as_uint()) {
                        break;
                    }
                } else {
                    if (f.is_whitespace || f.is_letter || f.is_number || !f.as_uint()) {
                        break;
                    }
                }
                R = t;
            }
            if (!gpt2) {
                while (R < n && (bytes[R] == '\r' || bytes[R] == '\n')) {
                    ++R;
                }
            }
            if (R <= j) {
                R = j_end;
            }
        }

        if (L > i) {
            if (family == unicode_byte_mixed_family::Gpt2) {
                unicode_regex_split_gpt2_ascii_bytes_seg(bytes, i, L, byte_offsets);
            } else if (family == unicode_byte_mixed_family::Llama3) {
                unicode_regex_split_llama3_ascii_bytes_seg(bytes, i, L, byte_offsets);
            } else {
                unicode_regex_split_qwen2_ascii_bytes_seg(bytes, i, L, byte_offsets);
            }
        }

        if (!unicode_decode_range_to_cpts(text, L, R, island_cpts, island_off)) {
            return false;
        }
        island_lens.clear();
        if (family == unicode_byte_mixed_family::Gpt2) {
            unicode_regex_split_gpt2_unicode_seg(island_cpts, 0, island_cpts.size(), island_lens);
        } else if (family == unicode_byte_mixed_family::Llama3) {
            unicode_regex_split_llama3_unicode_seg(island_cpts, 0, island_cpts.size(), island_lens);
        } else if (family == unicode_byte_mixed_family::Qwen35) {
            unicode_regex_split_qwen35_unicode_seg(island_cpts, 0, island_cpts.size(), island_lens);
        } else {
            unicode_regex_split_qwen2_unicode_seg(island_cpts, 0, island_cpts.size(), island_lens);
        }
        unicode_append_cpt_lens_as_bytes(island_off, island_lens, byte_offsets);
        i = R;
    }

    if (blob_out) {
        unicode_fill_blob_from_byte_offsets(text, byte_offsets, byte_encode, *blob_out);
        if (words_out) {
            words_out->clear();
        }
        return true;
    }
    if (words_out) {
        *words_out = unicode_words_from_byte_offsets(text, byte_offsets, byte_encode);
        return true;
    }
    return false;
}

bool unicode_regex_split_try_blob(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode,
        unicode_pretok_blob & out) {
    // Opt-out for same-binary A/B vs vector<string> materialize.
    if (getenv("LLAMA_BPE_NO_PRETOK_BLOB") != nullptr) {
        return false;
    }
    // WHY (0121): always fill blob (ASCII custom or general cpt path) so mixed Unicode
    // megaprompts also skip ~N std::string pretok words.
    (void) unicode_regex_split_impl(text, regex_exprs, byte_encode, &out);
    return true;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {
    return unicode_regex_split_impl(text, regex_exprs, byte_encode, nullptr);
}

// Internal: when blob_out != nullptr, fill blob and return empty vector; else return words.
static std::vector<std::string> unicode_regex_split_impl(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode,
        unicode_pretok_blob * blob_out) {
    // WHY (0114–0116): all-ASCII custom pretok — skip uint32 decode + cpt_byte_off.
    if (unicode_want_ascii_pretok() && regex_exprs.size() == 1 && unicode_bytes_are_ascii(text)) {
        const std::string & expr = regex_exprs[0];
        std::vector<size_t> bpe_offsets;
        bpe_offsets.reserve(std::max<size_t>(1, text.size() / 3));
        const unsigned char * bytes = (const unsigned char *) text.data();
        bool handled = false;
        if (unicode_is_qwen2_regex_expr(expr) || unicode_is_qwen35_regex_expr(expr)) {
            unicode_regex_split_qwen2_ascii_bytes_seg(bytes, 0, text.size(), bpe_offsets);
            handled = true;
        } else if (unicode_is_gpt2_regex_expr(expr)) {
            unicode_regex_split_gpt2_ascii_bytes_seg(bytes, 0, text.size(), bpe_offsets);
            handled = true;
        } else if (unicode_is_llama3_regex_expr(expr)) {
            unicode_regex_split_llama3_ascii_bytes_seg(bytes, 0, text.size(), bpe_offsets);
            handled = true;
        }
        if (handled) {
            if (blob_out) {
                unicode_fill_blob_from_byte_offsets(text, bpe_offsets, byte_encode, *blob_out);
                return {};
            }
            return unicode_words_from_byte_offsets(text, bpe_offsets, byte_encode);
        }
    }

    // WHY (0124): ASCII-majority mixed — byte islands before full uint32 decode.
    if (unicode_want_ascii_pretok() && regex_exprs.size() == 1) {
        const std::string & expr = regex_exprs[0];
        unicode_byte_mixed_family fam;
        bool try_mixed = false;
        if (unicode_is_qwen2_regex_expr(expr)) {
            fam = unicode_byte_mixed_family::Qwen2;
            try_mixed = true;
        } else if (unicode_is_qwen35_regex_expr(expr)) {
            fam = unicode_byte_mixed_family::Qwen35;
            try_mixed = true;
        } else if (unicode_is_gpt2_regex_expr(expr)) {
            fam = unicode_byte_mixed_family::Gpt2;
            try_mixed = true;
        } else if (unicode_is_llama3_regex_expr(expr)) {
            fam = unicode_byte_mixed_family::Llama3;
            try_mixed = true;
        }
        if (try_mixed) {
            if (blob_out) {
                if (unicode_regex_split_try_byte_mixed(text, fam, byte_encode, blob_out, nullptr)) {
                    return {};
                }
            } else {
                std::vector<std::string> words;
                if (unicode_regex_split_try_byte_mixed(text, fam, byte_encode, nullptr, &words)) {
                    return words;
                }
            }
        }
    }

    // unicode categories
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },
        { "\\p{L}", unicode_cpt_flags::LETTER },
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
        { "\\p{S}", unicode_cpt_flags::SYMBOL },
        { "\\p{Lu}", unicode_cpt_flags::LETTER }, // Uppercase letter
        { "\\p{Ll}", unicode_cpt_flags::LETTER }, // Lowercase letter
        { "\\p{Lt}", unicode_cpt_flags::LETTER }, // Titlecase letter
        { "\\p{Lm}", unicode_cpt_flags::LETTER }, // Modifier letter
        { "\\p{Lo}", unicode_cpt_flags::LETTER }, // Other letter
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C\\\x7E" }, // $+<=>^`|~
    };

    // WHY (0109): decode once and keep byte offsets so valid UTF-8 pretoks can be
    // materialized with substr (no per-cpt utf8 rebuild). Invalid UTF-8 still uses
    // the FFFD rebuild path for identity with unicode_cpts_from_utf8.
    std::vector<uint32_t> cpts;
    std::vector<size_t> cpt_byte_off;
    cpts.reserve(text.size());
    cpt_byte_off.reserve(text.size() + 1);
    cpt_byte_off.push_back(0);
    bool had_invalid_utf8 = false;
    {
        size_t offset = 0;
        while (offset < text.size()) {
            // WHY (0121): mixed megaprompts are still ASCII-majority; skip per-byte
            // unicode_cpt_from_utf8 for runs of bytes < 0x80 (identity: ASCII cpt == byte).
            {
                size_t n = 0;
                const size_t remain = text.size() - offset;
                const unsigned char * p = (const unsigned char *) text.data() + offset;
                while (n + 8 <= remain) {
                    uint64_t v;
                    memcpy(&v, p + n, 8);
                    if (v & 0x8080808080808080ULL) {
                        break;
                    }
                    n += 8;
                }
                while (n < remain && p[n] < 0x80) {
                    ++n;
                }
                if (n > 0) {
                    const size_t base = cpts.size();
                    cpts.resize(base + n);
                    for (size_t i = 0; i < n; ++i) {
                        cpts[base + i] = p[i];
                        cpt_byte_off.push_back(offset + i + 1);
                    }
                    offset += n;
                    continue;
                }
            }
            try {
                cpts.push_back(unicode_cpt_from_utf8(text, offset));
            } catch (const std::invalid_argument &) {
                ++offset;
                cpts.push_back(0xFFFD);
                had_invalid_utf8 = true;
            }
            cpt_byte_off.push_back(offset);
        }
    }

    // WHY (0109): GPT-2/Llama/Qwen custom splitters never read text_collapsed, but the old
    // code still built it whenever any regex mentioned \p{L}/\p{N}/… — a full megaprompt
    // pass before the real scanner. Build lazily only when std::regex fallback needs it.
    std::string text_collapsed;
    bool text_collapsed_ready = false;
    auto ensure_text_collapsed = [&]() {
        if (text_collapsed_ready) {
            return;
        }
        // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
        // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
        text_collapsed.resize(cpts.size());
        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
        text_collapsed_ready = true;
    };

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(cpts, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }
            const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

            if (use_collapsed) {
                ensure_text_collapsed();
                // sanity-check that the original regex does not contain any non-ASCII characters
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    // Match \p{...} Unicode properties of varying lengths
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        // Find the closing brace
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                            const std::string pat = regex_expr.substr(i, closing_brace - i + 1);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                                regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    if (blob_out) {
        unicode_fill_blob_from_cpt_offsets(
                text, cpts, cpt_byte_off, bpe_offsets, had_invalid_utf8, byte_encode, *blob_out);
        return {};
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    if (!had_invalid_utf8) {
        // Valid UTF-8: original byte spans match cpt→utf8 rebuild exactly.
        for (size_t & offset : bpe_offsets) {
            const size_t b0 = cpt_byte_off[start];
            const size_t b1 = cpt_byte_off[start + offset];
            bpe_words.emplace_back(text.data() + b0, b1 - b0);
            start += offset;
        }
    } else {
        for (size_t & offset : bpe_offsets) {
            bpe_words.emplace_back();
            std::string & word = bpe_words.back();
            word.reserve(offset * 2);
            for (size_t i = start; i < start + offset; ++i) {
                unicode_cpt_append_utf8(word, cpts[i]);
            }
            start += offset;
        }
    }

    if (byte_encode) {
        return unicode_byte_encoding_process(bpe_words);
    }

    return bpe_words;
}
