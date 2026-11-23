// What a SPIR-V module declares, read before the device is created, so a test tool enables only the optional features
// the module needs and reports 77 (unsupported) when the device lacks one, instead of asking for every feature a
// translated shader might use. rdna2spirv.py lowers buffer addresses to 32-bit pairs (SHADER-423), so most memory
// shaders no longer declare Int64; asking for shaderInt64 regardless made the tools fail on devices without it (Intel
// UHD/Arc) before the shader ran (TEST-468).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace leonida::vk_test {

// SPIR-V magic number and capability numbers from spirv/unified1/spirv.h.
inline constexpr std::uint32_t spirv_magic_number = 0x07230203u;
inline constexpr std::uint32_t spirv_capability_float64 = 10;
inline constexpr std::uint32_t spirv_capability_int64 = 11;
inline constexpr std::uint32_t spirv_capability_int64_atomics = 12;
inline constexpr std::uint32_t spirv_capability_int64_image_ext = 5016;

// Whether the module unconditionally declares any capability in `caps`. In SPIR-V section 2.4, OpCapability (2 words)
// and OpConditionalCapabilityINTEL (3 words) share the leading capability section right after the 5-word header: step
// over conditional capabilities so a following OpCapability is still reached, and stop at the first non-capability or
// truncated instruction. Null, non-SPIR-V, or < 20-byte inputs declare nothing; a trailing partial word is ignored.
inline bool declares_any_capability(const void* code, std::size_t bytes,
                                    std::initializer_list<std::uint32_t> caps) {
    if (!code || bytes < 20 || caps.size() == 0) return false;
    const auto* p = static_cast<const unsigned char*>(code);
    const std::size_t words = bytes / 4;
    auto word = [&](std::size_t i) {
        std::uint32_t w;
        std::memcpy(&w, p + 4 * i, 4);
        return w;
    };
    if (word(0) != spirv_magic_number) return false;
    for (std::size_t i = 5; i < words;) {
        const std::uint32_t inst = word(i);
        if (inst == (2u << 16 | 17u)) {          // OpCapability
            if (i + 1 >= words) break;
            const std::uint32_t declared = word(i + 1);
            for (std::uint32_t cap : caps)
                if (declared == cap) return true;
            i += 2;
        } else if (inst == (3u << 16 | 6250u)) { // OpConditionalCapabilityINTEL
            if (i + 2 >= words) break;
            i += 3;
        } else {
            break;
        }
    }
    return false;
}

inline bool declares_any_capability(const std::vector<std::uint32_t>& spv,
                                    std::initializer_list<std::uint32_t> caps) {
    return declares_any_capability(spv.data(), spv.size() * 4, caps);
}

inline bool declares_capability(const void* code, std::size_t bytes, std::uint32_t cap) {
    return declares_any_capability(code, bytes, {cap});
}

inline bool declares_capability(const std::vector<std::uint32_t>& spv, std::uint32_t cap) {
    return declares_capability(spv.data(), spv.size() * 4, cap);
}

// 64-bit integer types need shaderInt64: the Int64 capability, or Int64Atomics / Int64ImageEXT, which implicitly
// declare it in the SPIR-V specification.
inline bool needs_shader_int64(const void* code, std::size_t bytes) {
    return declares_any_capability(
        code, bytes,
        {spirv_capability_int64, spirv_capability_int64_atomics, spirv_capability_int64_image_ext});
}

inline bool needs_shader_int64(const std::vector<std::uint32_t>& spv) {
    return needs_shader_int64(spv.data(), spv.size() * 4);
}

}  // namespace leonida::vk_test
