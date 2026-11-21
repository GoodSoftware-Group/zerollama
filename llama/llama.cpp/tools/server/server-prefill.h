#pragma once

// Draft the chat the user is still typing.
//
// The cached text has to be a string prefix of the finished prompt, so the
// slot keeps a real token prefix and the later request only evaluates the
// tail. The last incomplete word is cut, and U+F8FF is appended inside that
// message. The rendered template is then truncated at the sentinel, which
// drops the end-of-turn marker and the assistant header. Those belong to the
// finished request, not the draft.

#include <cstddef>
#include <cstdint>
#include <string>

// U+F8FF in UTF-8. Private use, so a normal chat template emits it unchanged.
inline constexpr const char * LLAMA_PREFILL_END = "\xEF\xA3\xBF";

inline bool llama_prefill_is_space(uint32_t cp) {
    switch (cp) {
        case 0x0009:
        case 0x000A:
        case 0x000B:
        case 0x000C:
        case 0x000D:
        case 0x0020:
        case 0x0085:
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

// Walk one UTF-8 code point backward. `end` is the index just past the code
// point. Invalid bytes are consumed one at a time so a bad sequence cannot
// walk off the start of the string.
inline bool llama_prefill_prev(const std::string & s, size_t end, size_t & start, uint32_t & cp) {
    if (end == 0 || end > s.size()) {
        return false;
    }
    size_t i = end - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) {
        --i;
    }
    const unsigned char b0 = static_cast<unsigned char>(s[i]);
    const size_t n = end - i;
    if (b0 < 0x80 && n == 1) {
        cp = b0;
    } else if ((b0 & 0xE0) == 0xC0 && n == 2) {
        cp = (uint32_t(b0 & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
    } else if ((b0 & 0xF0) == 0xE0 && n == 3) {
        cp = (uint32_t(b0 & 0x0F) << 12) |
             (uint32_t(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
    } else if ((b0 & 0xF8) == 0xF0 && n == 4) {
        cp = (uint32_t(b0 & 0x07) << 18) |
             (uint32_t(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             (uint32_t(static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3F);
    } else {
        i = end - 1;
        cp = static_cast<unsigned char>(s[i]);
    }
    start = i;
    return true;
}

// "two words  " -> "two". A single unfinished word -> "".
inline std::string llama_prefill_before_last_word(std::string text) {
    auto trim_end = [](std::string & s) {
        size_t end = s.size();
        size_t start = 0;
        uint32_t cp = 0;
        while (llama_prefill_prev(s, end, start, cp) && llama_prefill_is_space(cp)) {
            end = start;
        }
        s.resize(end);
    };
    trim_end(text);
    size_t end = text.size();
    size_t start = 0;
    uint32_t cp = 0;
    bool found = false;
    while (llama_prefill_prev(text, end, start, cp)) {
        if (llama_prefill_is_space(cp)) {
            found = true;
            break;
        }
        end = start;
    }
    if (!found) {
        return "";
    }
    text.resize(start);
    trim_end(text);
    return text;
}

#ifdef LLAMA_PREFILL_TEST
#include <cstdio>
#include <cstdlib>
int main() {
    if (std::string(LLAMA_PREFILL_END) != "\xEF\xA3\xBF") {
        std::fputs("sentinel is not U+F8FF\n", stderr);
        return 1;
    }
    auto check = [](const char * in, const char * want) {
        const std::string got = llama_prefill_before_last_word(in);
        if (got != want) {
            std::fprintf(stderr, "before_last_word(%s) = [%s], want [%s]\n", in, got.c_str(), want);
            std::exit(1);
        }
    };
    check("", "");
    check("Summ", "");
    check("Summarize this.\n\nWhat is", "Summarize this.\n\nWhat");
    check("two words  ", "two");
    check("zażółć gęślą", "zażółć");
    std::puts("prefill ok");
    return 0;
}
#endif
