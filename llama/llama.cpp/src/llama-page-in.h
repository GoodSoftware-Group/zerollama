#pragma once

// Fault host expert weights into RAM at load when 3 GiB would remain free.
// Mapped experts otherwise enter one page at a time on the first tokens
// that read them. LLAMA_PAGE_IN=0 skips the read.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

inline constexpr uint64_t LLAMA_PAGE_IN_SPARE = 3ull << 30;
inline constexpr size_t   LLAMA_PAGE_IN_STRIDE = 4096;

// True when reading expert_bytes would leave the spare free.
inline bool llama_page_in_ok(uint64_t expert_bytes, std::optional<uint64_t> available) {
    if (expert_bytes == 0 || !available.has_value()) {
        return false;
    }
    return *available >= expert_bytes + LLAMA_PAGE_IN_SPARE;
}

// MemAvailable from a /proc/meminfo blob, in bytes.
inline std::optional<uint64_t> llama_parse_mem_available(const char * text) {
    if (text == nullptr) {
        return std::nullopt;
    }
    const char * p = std::strstr(text, "MemAvailable:");
    if (p == nullptr) {
        return std::nullopt;
    }
    p += std::strlen("MemAvailable:");
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    char * end = nullptr;
    const unsigned long long kib = std::strtoull(p, &end, 10);
    if (end == p) {
        return std::nullopt;
    }
    return (uint64_t) kib * 1024ull;
}

// One byte per page, folded together so the reads stay.
inline uint8_t llama_page_touch(const uint8_t * data, size_t n) {
    uint8_t acc = 0;
    if (data == nullptr || n == 0) {
        return 0;
    }
    for (size_t off = 0; off < n; off += LLAMA_PAGE_IN_STRIDE) {
        acc ^= data[off];
    }
    acc ^= data[n - 1];
    return acc;
}

// Bytes the system can give out now, or nothing when the platform does not say.
inline std::optional<uint64_t> llama_mem_available() {
#if defined(__linux__)
    std::FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return std::nullopt;
    }
    std::string info;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f) != nullptr) {
        info += buf;
    }
    std::fclose(f);
    return llama_parse_mem_available(info.c_str());
#elif defined(__APPLE__)
    // Free, inactive and speculative pages can be handed out without paging
    // another program's anonymous memory.
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || page == 0) {
        return std::nullopt;
    }
    vm_statistics64_data_t vm = {};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS) {
        return std::nullopt;
    }
    const uint64_t pages = (uint64_t) vm.free_count + (uint64_t) vm.inactive_count + (uint64_t) vm.speculative_count;
    if (pages == 0) {
        return std::nullopt;
    }
    return pages * (uint64_t) page;
#else
    return std::nullopt;
#endif
}

#ifdef LLAMA_PAGE_IN_TEST
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    if (LLAMA_PAGE_IN_SPARE != (3ull << 30)) {
        std::fputs("spare is not 3 GiB\n", stderr);
        return 1;
    }
    if (llama_page_in_ok(0, 100) || llama_page_in_ok(1, std::nullopt)) {
        std::fputs("an empty model or an unknown amount of memory is not read in\n", stderr);
        return 1;
    }
    const uint64_t need = 1000;
    if (!llama_page_in_ok(need, need + LLAMA_PAGE_IN_SPARE)) {
        std::fputs("exactly 3 GiB to spare is enough\n", stderr);
        return 1;
    }
    if (llama_page_in_ok(need, need + LLAMA_PAGE_IN_SPARE - 1)) {
        std::fputs("one byte under the spare is not enough\n", stderr);
        return 1;
    }
    if (llama_parse_mem_available("MemTotal: 8 kB\n")) {
        std::fputs("MemTotal is not MemAvailable\n", stderr);
        return 1;
    }
    const auto parsed = llama_parse_mem_available("MemTotal: 1 kB\nMemAvailable:   12345 kB\n");
    if (!parsed || *parsed != 12345ull * 1024ull) {
        std::fputs("MemAvailable was not read as kibibytes\n", stderr);
        return 1;
    }
    std::vector<uint8_t> pages(5000, 0);
    pages[0] = 1;
    pages[4096] = 2;
    pages[4999] = 4;
    if (llama_page_touch(pages.data(), pages.size()) != (uint8_t) (1 ^ 2 ^ 4)) {
        std::fputs("a page touch missed a page\n", stderr);
        return 1;
    }
    if (llama_page_touch(nullptr, 10) != 0) {
        std::fputs("an empty touch should fold to zero\n", stderr);
        return 1;
    }
    const auto live = llama_mem_available();
    if (!live || *live == 0) {
        std::fputs("this host did not report memory it can give out\n", stderr);
        return 1;
    }
    std::puts("page in ok");
    return 0;
}
#endif
