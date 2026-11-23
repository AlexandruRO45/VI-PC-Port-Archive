// CPU-only tests for spirv_capability.h, which the standalone GPU tools use to enable shaderInt64 only for modules
// with 64-bit integers (TEST-468). The modules are hand-built word lists; no Vulkan instance or GPU is needed.
#include "spirv_capability.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void expect(const char* what, bool got, bool want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: got %d, expected %d\n", what, got, want);
    ++failures;
}

// A 5-word header, one OpCapability per entry of `caps`, then OpMemoryModel Logical GLSL450.
std::vector<std::uint32_t> module(std::initializer_list<std::uint32_t> caps) {
    std::vector<std::uint32_t> w{0x07230203u, 0x00010000u, 0u, 1u, 0u};
    for (std::uint32_t c : caps) {
        w.push_back(2u << 16 | 17u);   // OpCapability
        w.push_back(c);
    }
    w.push_back(3u << 16 | 14u);       // OpMemoryModel
    w.push_back(0u);
    w.push_back(1u);
    return w;
}

}  // namespace

int main() {
    using namespace leonida::vk_test;
    constexpr std::uint32_t shader = 1, physical_storage_buffer = 5347, float16_image_amd = 5008;

    // SHADER-423's output: buffer addresses as 32-bit pairs, PhysicalStorageBufferAddresses but no Int64.
    expect("address pairs need no shaderInt64", needs_shader_int64(module({shader, physical_storage_buffer})), false);
    expect("Int64 needs shaderInt64", needs_shader_int64(module({shader, spirv_capability_int64})), true);
    expect("Int64 after other capabilities", needs_shader_int64(module({shader, physical_storage_buffer, 10, 11})), true);
    expect("Int64Atomics implicitly declares Int64", needs_shader_int64(module({shader, spirv_capability_int64_atomics})),
           true);
    expect("Int64ImageEXT implicitly declares Int64",
           needs_shader_int64(module({shader, spirv_capability_int64_image_ext})), true);
    expect("Float16ImageAMD (5008) needs no shaderInt64", needs_shader_int64(module({shader, float16_image_amd})), false);
    // cs_f64_add (vk_compute_test enables shaderFloat64 for it): Float64 without Int64.
    const std::vector<std::uint32_t> f64 = module({shader, spirv_capability_float64});
    expect("Float64", declares_capability(f64, spirv_capability_float64), true);
    expect("Float64 needs no shaderInt64", needs_shader_int64(f64), false);
    expect("declares_any_capability empty set", declares_any_capability(f64, {}), false);
    expect("declares_any_capability match", declares_any_capability(f64, {spirv_capability_int64, spirv_capability_float64}),
           true);

    // Only the leading capability section counts: a later word equal to 11 is an operand, not a capability.
    std::vector<std::uint32_t> later = module({shader});
    later.push_back(2u << 16 | 17u);
    later.push_back(spirv_capability_int64);
    expect("a capability after OpMemoryModel is ignored", needs_shader_int64(later), false);

    const std::vector<std::uint32_t> after_ext{
        0x07230203u, 0x00010000u, 0u, 1u, 0u,
        2u << 16 | 17u, shader,
        2u << 16 | 10u, 0u,   // OpExtension ""
        2u << 16 | 17u, spirv_capability_int64,
    };
    expect("a capability after OpExtension is ignored", needs_shader_int64(after_ext), false);

    // SPIR-V 2.4 section 1 permits OpConditionalCapabilityINTEL (6250, 3 words) alongside OpCapability.
    const std::vector<std::uint32_t> after_cond{
        0x07230203u, 0x00010000u, 0u, 2u, 0u,
        2u << 16 | 17u, shader,
        3u << 16 | 6250u, 1u, physical_storage_buffer,
        2u << 16 | 17u, spirv_capability_int64,
        3u << 16 | 14u, 0u, 1u,
    };
    expect("OpCapability Int64 after OpConditionalCapabilityINTEL", needs_shader_int64(after_cond), true);

    const std::vector<std::uint32_t> cond_only{
        0x07230203u, 0x00010000u, 0u, 2u, 0u,
        2u << 16 | 17u, shader,
        3u << 16 | 6250u, 1u, spirv_capability_int64,
        3u << 16 | 14u, 0u, 1u,
    };
    expect("OpConditionalCapabilityINTEL alone is not unconditional OpCapability", needs_shader_int64(cond_only), false);

    // Null pointers, short headers (< 20 bytes), and truncated capability instructions do not crash or read out of bounds.
    expect("nullptr declares_capability", declares_capability(nullptr, 64, spirv_capability_int64), false);
    expect("nullptr needs_shader_int64", needs_shader_int64(nullptr, 64), false);
    expect("empty", needs_shader_int64(std::vector<std::uint32_t>{}), false);

    const std::vector<std::uint32_t> m = module({shader, spirv_capability_int64});
    for (std::size_t n : {std::size_t{0}, std::size_t{4}, std::size_t{19}, std::size_t{20}}) {
        expect("short header bytes", needs_shader_int64(m.data(), n), false);
    }
    const std::vector<std::uint32_t> truncated{0x07230203u, 0x00010000u, 0u, 1u, 0u, 2u << 16 | 17u};
    expect("truncated OpCapability", needs_shader_int64(truncated), false);
    const std::vector<std::uint32_t> truncated_cond1{0x07230203u, 0x00010000u, 0u, 2u, 0u, 3u << 16 | 6250u};
    expect("truncated OpConditionalCapabilityINTEL (1 word)", needs_shader_int64(truncated_cond1), false);
    const std::vector<std::uint32_t> truncated_cond2{0x07230203u, 0x00010000u, 0u, 2u, 0u, 3u << 16 | 6250u, 1u};
    expect("truncated OpConditionalCapabilityINTEL (2 words)", needs_shader_int64(truncated_cond2), false);

    // Non-SPIR-V magic words (corrupt/text files or reversed-endian streams) are rejected even if word 5 looks like OpCapability.
    const std::vector<std::uint32_t> bad_magic{
        0xdeadbeefu, 0x00010000u, 0u, 1u, 0u, 2u << 16 | 17u, spirv_capability_int64,
    };
    expect("non-SPIR-V magic", needs_shader_int64(bad_magic), false);
    const std::vector<std::uint32_t> swapped_magic{
        0x03022307u, 0x00010000u, 0u, 1u, 0u, 2u << 16 | 17u, spirv_capability_int64,
    };
    expect("byte-swapped SPIR-V magic", needs_shader_int64(swapped_magic), false);

    // The byte form the tools use for std::vector<char>, including a trailing partial word and unaligned storage.
    std::vector<char> bytes(m.size() * 4 + 3, 0);
    std::memcpy(bytes.data(), m.data(), m.size() * 4);
    expect("bytes with a partial word", needs_shader_int64(bytes.data(), bytes.size()), true);

    alignas(8) unsigned char unaligned[128]{};
    std::memcpy(unaligned + 1, m.data(), m.size() * 4);
    expect("unaligned byte buffer", needs_shader_int64(unaligned + 1, m.size() * 4), true);
    expect("unaligned short buffer", needs_shader_int64(unaligned + 1, 19), false);

    if (failures) return 1;
    std::printf("spirv_capability: all checks passed\n");
    return 0;
}
