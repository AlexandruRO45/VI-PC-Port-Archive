// Unit tests for SPIR-V binary word framing (runtime/src/spirv_words.h). CPU-only, no Vulkan instance and no GPU:
// exercises header-length bounds, 4-byte alignment, magic check, truncated stream handling, pre-seeked/non-zero
// stream offset rewind, and exact word round-trip.
#include "../src/spirv_words.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::read_spirv_words;

namespace {

std::string bytes_of(const std::vector<std::uint32_t>& words) {
    return std::string(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(std::uint32_t));
}

int test_rejects_invalid_sizes_and_magic() {
    const std::vector<std::uint32_t> valid = {0x07230203u, 0x00010000u, 0u, 1u, 0u, 0x00010011u};
    const std::string raw = bytes_of(valid);

    for (std::streamoff bad_size : {std::streamoff(-4), std::streamoff(0), std::streamoff(16),
                                    std::streamoff(21), std::streamoff(22), std::streamoff(23)}) {
        std::istringstream in(raw, std::ios::binary);
        EXPECT(read_spirv_words(in, bad_size).empty());
    }

    // Wrong magic (including byte-swapped magic).
    for (std::uint32_t bad_magic : {0x00000000u, 0x03022307u, 0x07230204u}) {
        std::vector<std::uint32_t> words = valid;
        words[0] = bad_magic;
        std::istringstream in(bytes_of(words), std::ios::binary);
        EXPECT(read_spirv_words(in, static_cast<std::streamoff>(words.size() * 4)).empty());
    }
    return 0;
}

int test_rejects_truncated_and_failed_streams() {
    const std::vector<std::uint32_t> five = {0x07230203u, 0x00010000u, 0u, 1u, 0u};
    {
        // Header says 24 bytes, stream only has 20.
        std::istringstream in(bytes_of(five), std::ios::binary);
        EXPECT(read_spirv_words(in, 24).empty());
    }
    {
        // Stream already in fail state before call: seekg(0) fails.
        std::istringstream in(bytes_of(five), std::ios::binary);
        in.setstate(std::ios::failbit);
        EXPECT(read_spirv_words(in, 20).empty());
    }
    return 0;
}

int test_accepts_header_only_and_rewinds_stream() {
    const std::vector<std::uint32_t> header_only = {0x07230203u, 0x00010300u, 42u, 7u, 0u};
    {
        std::istringstream in(bytes_of(header_only), std::ios::binary);
        const auto out = read_spirv_words(in, 20);
        EXPECT(out == header_only);
    }
    const std::vector<std::uint32_t> with_body = {
        0x07230203u, 0x00010300u, 0u, 2u, 0u, 0x00020011u, 0x00000001u
    };
    {
        std::istringstream in(bytes_of(with_body), std::ios::binary);
        in.seekg(8); // Non-zero initial offset must be rewound to 0.
        const auto out = read_spirv_words(in, static_cast<std::streamoff>(with_body.size() * 4));
        EXPECT(out == with_body);
    }
    return 0;
}

} // namespace

int main() {
    if (test_rejects_invalid_sizes_and_magic()) return 1;
    if (test_rejects_truncated_and_failed_streams()) return 1;
    if (test_accepts_header_only_and_rewinds_stream()) return 1;
    std::puts("spirv_words_test: ok");
    return 0;
}
