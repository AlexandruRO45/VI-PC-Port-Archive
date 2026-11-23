// Unit tests for SPIR-V binary word framing (runtime/src/spirv_words.h). CPU-only, no Vulkan instance and no GPU:
// exercises header-length bounds, 4-byte alignment, magic check, truncated stream handling, seekg(0) failure via
// an unseekable streambuf, non-zero stream offset rewind, and exact word round-trip.
#include "../src/spirv_words.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <sstream>
#include <streambuf>
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

// Delivers valid bytes on read while rejecting seekoff, so only !input.seekg(0) fails.
class UnseekableBuf final : public std::streambuf {
public:
    explicit UnseekableBuf(const std::string& data) : data_(data) {
        char* p = data_.data();
        setg(p, p, p + data_.size());
    }

protected:
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }

private:
    std::string data_;
};

int test_rejects_invalid_sizes_and_magic() {
    const std::vector<std::uint32_t> valid = {0x07230203u, 0x00010000u, 0u, 1u, 0u, 0x00010000u};
    const std::string raw = bytes_of(valid);

    for (std::streamoff bad_size : {std::streamoff(-4), std::streamoff(0), std::streamoff(16),
                                    std::streamoff(19), std::streamoff(21), std::streamoff(22), std::streamoff(23)}) {
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

int test_rejects_truncated_and_unseekable_streams() {
    const std::vector<std::uint32_t> five = {0x07230203u, 0x00010000u, 0u, 1u, 0u};
    {
        // Header says 24 bytes, stream only has 20.
        std::istringstream in(bytes_of(five), std::ios::binary);
        EXPECT(read_spirv_words(in, 24).empty());
    }
    {
        // Unseekable stream with valid bytes: fails specifically on seekg(0).
        UnseekableBuf buf(bytes_of(five));
        std::istream in(&buf);
        EXPECT(read_spirv_words(in, 20).empty());
    }
    return 0;
}

int test_accepts_valid_streams_and_rewinds() {
    const std::vector<std::uint32_t> header_only = {0x07230203u, 0x00010300u, 42u, 7u, 0u};
    {
        std::istringstream in(bytes_of(header_only), std::ios::binary);
        EXPECT(read_spirv_words(in, 20) == header_only);
    }
    const std::vector<std::uint32_t> with_nop = {0x07230203u, 0x00010300u, 0u, 1u, 0u, 0x00010000u};
    {
        std::istringstream in(bytes_of(with_nop), std::ios::binary);
        EXPECT(read_spirv_words(in, 24) == with_nop);
    }
    const std::vector<std::uint32_t> with_cap = {
        0x07230203u, 0x00010300u, 0u, 2u, 0u, 0x00020011u, 0x00000001u
    };
    {
        std::istringstream in(bytes_of(with_cap), std::ios::binary);
        in.seekg(8); // Non-zero initial offset must be rewound to 0.
        EXPECT(read_spirv_words(in, static_cast<std::streamoff>(with_cap.size() * 4)) == with_cap);
    }
    return 0;
}

} // namespace

int main() {
    if (test_rejects_invalid_sizes_and_magic()) return 1;
    if (test_rejects_truncated_and_unseekable_streams()) return 1;
    if (test_accepts_valid_streams_and_rewinds()) return 1;
    std::puts("spirv_words_test: ok");
    return 0;
}
