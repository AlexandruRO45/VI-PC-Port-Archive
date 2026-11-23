// leonida::scan_spirv_ext_caps and leonida::missing_spirv_ext_cap (src/spirv_ext_caps.h) over hand-built SPIR-V words.
// No Vulkan, no GPU, no shader files.
#include "../src/spirv_ext_caps.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "spirv_ext_caps_test: FAILED: %s\n", what);
        ++failures;
    }
}

using Words = std::vector<std::uint32_t>;

Words header() { return {0x07230203u, 0x00010500u, 0u, 100u, 0u}; }

void inst(Words& words, std::uint16_t op, std::initializer_list<std::uint32_t> operands) {
    words.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) | op);
    words.insert(words.end(), operands);
}

constexpr std::uint16_t OpCapability = 17, OpTypeInt = 21, OpTypeVector = 23, OpConstant = 43, OpReadClockKHR = 5056;
constexpr std::uint32_t CapShader = 1, CapFloat64 = 10, CapInt64 = 11;
constexpr std::uint32_t CapShaderClockKHR = 5055;
constexpr std::uint32_t CapMeshShadingEXT = 5283;
constexpr std::uint32_t CapFragmentBarycentricKHR = 5284;
constexpr std::uint32_t CapFragmentShaderPixelInterlockEXT = 5378;
constexpr std::uint32_t ScopeDevice = 1, ScopeSubgroup = 3;

Words with_caps(std::initializer_list<std::uint32_t> caps) {
    Words w = header();
    inst(w, OpCapability, {CapShader});
    for (std::uint32_t cap : caps) inst(w, OpCapability, {cap});
    return w;
}

Words clock_module(std::uint32_t scope_literal) {
    Words w = with_caps({CapShaderClockKHR});
    constexpr std::uint32_t uint_t = 1, uvec2_t = 2, scope_c = 3, res_id = 4;
    inst(w, OpTypeInt, {uint_t, 32, 0});
    inst(w, OpTypeVector, {uvec2_t, uint_t, 2});
    inst(w, OpConstant, {uint_t, scope_c, scope_literal});
    inst(w, OpReadClockKHR, {uvec2_t, res_id, scope_c});
    return w;
}

}  // namespace

int main() {
    using leonida::missing_spirv_ext_cap;
    using leonida::scan_spirv_ext_caps;

    // Plain shader and unrelated capabilities need none of the 3 optional extension features.
    {
        const auto caps = scan_spirv_ext_caps(with_caps({CapFloat64, CapInt64, CapMeshShadingEXT}));
        check(!caps.fragment_shader_barycentric, "plain module has no barycentric cap");
        check(!caps.fragment_shader_pixel_interlock, "plain module has no interlock cap");
        check(!caps.shader_clock && !caps.shader_subgroup_clock && !caps.shader_device_clock,
              "plain module has no clock cap");
        check(missing_spirv_ext_cap(caps, false, false, false, false) == nullptr, "plain module never refused");
    }

    // FragmentBarycentricKHR (5284)
    {
        const auto caps = scan_spirv_ext_caps(with_caps({CapFragmentBarycentricKHR}));
        check(caps.fragment_shader_barycentric && !caps.fragment_shader_pixel_interlock && !caps.shader_clock,
              "detects FragmentBarycentricKHR");
        const char* missing = missing_spirv_ext_cap(caps, false, true, true, true);
        check(missing != nullptr && std::strstr(missing, "fragmentShaderBarycentric") != nullptr,
              "missing barycentric names fragmentShaderBarycentric");
        check(missing_spirv_ext_cap(caps, true, false, false, false) == nullptr,
              "barycentric accepted when feature enabled");
    }

    // FragmentShaderPixelInterlockEXT (5378)
    {
        const auto caps = scan_spirv_ext_caps(with_caps({CapFragmentShaderPixelInterlockEXT}));
        check(caps.fragment_shader_pixel_interlock && !caps.fragment_shader_barycentric && !caps.shader_clock,
              "detects FragmentShaderPixelInterlockEXT");
        const char* missing = missing_spirv_ext_cap(caps, true, false, true, true);
        check(missing != nullptr && std::strstr(missing, "fragmentShaderPixelInterlock") != nullptr,
              "missing interlock names fragmentShaderPixelInterlock");
        check(missing_spirv_ext_cap(caps, false, true, false, false) == nullptr,
              "interlock accepted when feature enabled");
    }

    // ShaderClockKHR (5055) without OpReadClockKHR
    {
        const auto caps = scan_spirv_ext_caps(with_caps({CapShaderClockKHR}));
        check(caps.shader_clock && !caps.shader_subgroup_clock && !caps.shader_device_clock,
              "detects bare ShaderClockKHR");
        const char* missing = missing_spirv_ext_cap(caps, true, true, false, false);
        check(missing != nullptr && std::strstr(missing, "VK_KHR_shader_clock") != nullptr,
              "missing clock names VK_KHR_shader_clock");
        check(missing_spirv_ext_cap(caps, false, false, true, false) == nullptr,
              "bare ShaderClockKHR accepted with subgroup clock");
        check(missing_spirv_ext_cap(caps, false, false, false, true) == nullptr,
              "bare ShaderClockKHR accepted with device clock");
    }

    // ShaderClockKHR + OpReadClockKHR with Subgroup scope (3)
    {
        const auto caps = scan_spirv_ext_caps(clock_module(ScopeSubgroup));
        check(caps.shader_clock && caps.shader_subgroup_clock && !caps.shader_device_clock,
              "detects Subgroup-scoped OpReadClockKHR");
        const char* missing = missing_spirv_ext_cap(caps, true, true, false, true);
        check(missing != nullptr && std::strstr(missing, "shaderSubgroupClock") != nullptr,
              "Subgroup clock refused when only device clock is enabled");
        check(missing_spirv_ext_cap(caps, false, false, true, false) == nullptr,
              "Subgroup clock accepted when shaderSubgroupClock is enabled");
    }

    // ShaderClockKHR + OpReadClockKHR with Device scope (1)
    {
        const auto caps = scan_spirv_ext_caps(clock_module(ScopeDevice));
        check(caps.shader_clock && !caps.shader_subgroup_clock && caps.shader_device_clock,
              "detects Device-scoped OpReadClockKHR");
        const char* missing = missing_spirv_ext_cap(caps, true, true, true, false);
        check(missing != nullptr && std::strstr(missing, "shaderDeviceClock") != nullptr,
              "Device clock refused when only subgroup clock is enabled");
        check(missing_spirv_ext_cap(caps, false, false, false, true) == nullptr,
              "Device clock accepted when shaderDeviceClock is enabled");
    }

    // Combined capabilities
    {
        Words w = clock_module(ScopeDevice);
        inst(w, OpCapability, {CapFragmentBarycentricKHR});
        inst(w, OpCapability, {CapFragmentShaderPixelInterlockEXT});
        const auto caps = scan_spirv_ext_caps(w);
        check(caps.fragment_shader_barycentric && caps.fragment_shader_pixel_interlock && caps.shader_clock &&
                  caps.shader_device_clock,
              "detects multiple extension capabilities together");
        check(missing_spirv_ext_cap(caps, true, true, false, true) == nullptr,
              "combined capabilities accepted when all required features are enabled");
        check(missing_spirv_ext_cap(caps, false, true, false, true) != nullptr,
              "combined capabilities refused when barycentric missing");
        check(missing_spirv_ext_cap(caps, true, false, false, true) != nullptr,
              "combined capabilities refused when interlock missing");
        check(missing_spirv_ext_cap(caps, true, true, true, false) != nullptr,
              "combined capabilities refused when device clock missing");
    }

    // Malformed / truncated inputs never crash and return empty flags.
    {
        check(!scan_spirv_ext_caps({}).fragment_shader_barycentric, "empty input");
        Words bad_magic = with_caps({CapFragmentBarycentricKHR});
        bad_magic[0] = 0x12345678u;
        check(!scan_spirv_ext_caps(bad_magic).fragment_shader_barycentric, "bad magic ignored");

        Words zero_count = with_caps({CapFragmentBarycentricKHR});
        zero_count.push_back(OpCapability);  // word count 0
        check(!scan_spirv_ext_caps(zero_count).fragment_shader_barycentric, "zero word count rejected");

        Words truncated = with_caps({CapFragmentBarycentricKHR});
        truncated.push_back((5u << 16) | OpCapability);  // claims 5 words, only 1 left
        check(!scan_spirv_ext_caps(truncated).fragment_shader_barycentric, "truncated instruction rejected");
    }

    if (failures) {
        std::fprintf(stderr, "spirv_ext_caps_test: %d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "spirv_ext_caps_test: OK\n");
    return 0;
}
