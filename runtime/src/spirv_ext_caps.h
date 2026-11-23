#pragma once

// Which optional Vulkan extension capabilities does a SPIR-V module declare?
//
// Vulkan requires the matching device feature to be enabled before vkCreateShaderModule accepts a module that declares
// FragmentBarycentricKHR (VK_KHR_fragment_shader_barycentric::fragmentShaderBarycentric),
// FragmentShaderPixelInterlockEXT (VK_EXT_fragment_shader_interlock::fragmentShaderPixelInterlock), or
// ShaderClockKHR (VK_KHR_shader_clock::shaderSubgroupClock / shaderDeviceClock). Without the feature enabled,
// vkCreateShaderModule hits VUID-VkShaderModuleCreateInfo-pCode-08740 (or RuntimeSpirv clock VUIDs 06267 / 06268) or
// an opaque driver failure instead of a clean E_NOTIMPL naming the missing Vulkan feature.
//
// No Vulkan headers; safe on arbitrary or truncated words (returns empty flags on malformed framing).

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

namespace leonida {

struct SpirvExtCaps {
    bool fragment_shader_barycentric = false;      // OpCapability FragmentBarycentricKHR (5284)
    bool fragment_shader_pixel_interlock = false;  // OpCapability FragmentShaderPixelInterlockEXT (5378)
    bool shader_clock = false;                     // OpCapability ShaderClockKHR (5055) or OpReadClockKHR (5056)
    bool shader_subgroup_clock = false;            // OpReadClockKHR with Subgroup scope (3)
    bool shader_device_clock = false;              // OpReadClockKHR with Device scope (1)
};

inline SpirvExtCaps scan_spirv_ext_caps(std::span<const std::uint32_t> words) {
    constexpr std::uint32_t magic = 0x07230203u;
    constexpr std::uint16_t op_capability = 17, op_constant = 43, op_read_clock_khr = 5056;
    constexpr std::uint32_t cap_shader_clock_khr = 5055;
    constexpr std::uint32_t cap_fragment_barycentric_khr = 5284;
    constexpr std::uint32_t cap_fragment_shader_pixel_interlock_ext = 5378;
    constexpr std::uint32_t scope_device = 1;
    constexpr std::uint32_t scope_subgroup = 3;

    SpirvExtCaps caps{};
    if (words.size() < 5 || words[0] != magic) return caps;

    // Collect 32-bit scalar OpConstant values so OpReadClockKHR can distinguish Subgroup (3) vs Device (1) scope.
    std::unordered_map<std::uint32_t, std::uint32_t> constants;
    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t count = words[at] >> 16;
        const std::uint16_t op = static_cast<std::uint16_t>(words[at] & 0xffffu);
        if (count == 0 || at + count > words.size()) return {};  // malformed module framing
        const std::uint32_t* w = &words[at];
        switch (op) {
        case op_capability:
            if (count >= 2) {
                if (w[1] == cap_fragment_barycentric_khr) {
                    caps.fragment_shader_barycentric = true;
                } else if (w[1] == cap_fragment_shader_pixel_interlock_ext) {
                    caps.fragment_shader_pixel_interlock = true;
                } else if (w[1] == cap_shader_clock_khr) {
                    caps.shader_clock = true;
                }
            }
            break;
        case op_constant:
            if (count == 4) constants[w[2]] = w[3];
            break;
        case op_read_clock_khr:
            if (count >= 4) {
                caps.shader_clock = true;
                const auto it = constants.find(w[3]);
                if (it != constants.end()) {
                    if (it->second == scope_subgroup) {
                        caps.shader_subgroup_clock = true;
                    } else if (it->second == scope_device) {
                        caps.shader_device_clock = true;
                    }
                }
            }
            break;
        default:
            break;
        }
        at += count;
    }
    return caps;
}

// Returns the missing Vulkan extension/feature name when a declared capability is unavailable on the device,
// or nullptr if all declared extension capabilities are supported.
inline const char* missing_spirv_ext_cap(const SpirvExtCaps& caps, bool has_barycentric, bool has_pixel_interlock,
                                           bool has_subgroup_clock, bool has_device_clock) {
    if (caps.fragment_shader_barycentric && !has_barycentric) {
        return "VK_KHR_fragment_shader_barycentric (fragmentShaderBarycentric)";
    }
    if (caps.fragment_shader_pixel_interlock && !has_pixel_interlock) {
        return "VK_EXT_fragment_shader_interlock (fragmentShaderPixelInterlock)";
    }
    if (caps.shader_device_clock && !has_device_clock) {
        return "VK_KHR_shader_clock (shaderDeviceClock)";
    }
    if (caps.shader_subgroup_clock && !has_subgroup_clock) {
        return "VK_KHR_shader_clock (shaderSubgroupClock)";
    }
    if (caps.shader_clock && !has_subgroup_clock && !has_device_clock) {
        return "VK_KHR_shader_clock (shaderSubgroupClock / shaderDeviceClock)";
    }
    return nullptr;
}

}  // namespace leonida
