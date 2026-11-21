#pragma once

// D3D12 root-signature binary parser (PR A).
//
// Layout (little endian), the serialized D3D12_ROOT_SIGNATURE_DESC:
//   u32 version (1 = 1.0, 2 = 1.1), u32 flags, u32 parameter count,
//   parameters: u32 type, u32 visibility, then
//     descriptor table: u32 range count, ranges of (u32 type, u32 count, u32 base register, u32 space,
//                                                     u32 offset) plus in 1.1 a u32 range flags,
//     constants:        u32 shader register, u32 space, u32 value count,
//     cbv/srv/uav:      u32 shader register, u32 space, (1.1: u32 descriptor flags),
//   u32 static sampler count, samplers of (u32 filter, address u/v/w, f32 mip lod bias, u32 max anisotropy,
//                                          u32 comparison, u32 border, f32 min lod, f32 max lod,
//                                          u32 shader register, u32 space, u32 visibility).
//
// The parser works on a byte slice, never trusts a count past the end of the slice, and reports the first problem
// instead of returning a half-built description. No Vulkan, no COM: the CPU tests build the blobs themselves.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rt::d3d12 {

enum class RootParameterType : std::uint32_t {
    descriptor_table = 0,
    constants_32bit = 1,
    cbv = 2,
    srv = 3,
    uav = 4,
};

enum class ShaderVisibility : std::uint32_t { all = 0, vertex = 1, hull = 2, domain = 3, geometry = 4, pixel = 5 };

enum class DescriptorRangeType : std::uint32_t { srv = 0, uav = 1, cbv = 2, sampler = 3 };

struct DescriptorRange {
    DescriptorRangeType type = DescriptorRangeType::srv;
    std::uint32_t count = 0;  // UINT32_MAX means unbounded
    std::uint32_t base_register = 0;
    std::uint32_t register_space = 0;
    std::uint32_t offset = 0;
    std::uint32_t flags = 0;  // 1.1 only (D3D12_DESCRIPTOR_RANGE_FLAGS)
};

struct RootParameter {
    RootParameterType type = RootParameterType::cbv;
    ShaderVisibility visibility = ShaderVisibility::all;
    std::vector<DescriptorRange> ranges;  // descriptor_table
    std::uint32_t shader_register = 0;
    std::uint32_t register_space = 0;
    std::uint32_t num_32bit_values = 0;  // constants_32bit
    std::uint32_t flags = 0;             // 1.1 root descriptor flags
};

struct StaticSampler {
    std::uint32_t filter = 0;
    std::uint32_t address_u = 0;
    std::uint32_t address_v = 0;
    std::uint32_t address_w = 0;
    float mip_lod_bias = 0.0f;
    std::uint32_t max_anisotropy = 1;
    std::uint32_t comparison = 0;
    std::uint32_t border_color = 0;
    float min_lod = 0.0f;
    float max_lod = 0.0f;
    std::uint32_t shader_register = 0;
    std::uint32_t register_space = 0;
    ShaderVisibility visibility = ShaderVisibility::all;
};

struct RootSignatureDesc {
    std::uint32_t version = 1;  // 1 = 1.0, 2 = 1.1
    std::uint32_t flags = 0;
    std::vector<RootParameter> parameters;
    std::vector<StaticSampler> static_samplers;
};

namespace detail {

class Reader {
public:
    Reader(const void* data, std::size_t size) : data_(static_cast<const std::uint8_t*>(data)), size_(size) {}

    bool u32(std::uint32_t& value) {
        if (offset_ + 4 > size_) return false;
        std::memcpy(&value, data_ + offset_, 4);
        offset_ += 4;
        return true;
    }
    bool f32(float& value) {
        std::uint32_t bits = 0;
        if (!u32(bits)) return false;
        std::memcpy(&value, &bits, 4);
        return true;
    }
    std::size_t remaining() const { return size_ - offset_; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

inline bool known_visibility(std::uint32_t value) { return value <= 5; }
inline bool known_parameter(std::uint32_t value) { return value <= 4; }
inline bool known_range(std::uint32_t value) { return value <= 3; }

}  // namespace detail

inline bool parse_root_signature(const void* data, std::size_t size, RootSignatureDesc& out,
                                 std::string* error = nullptr) {
    out = RootSignatureDesc{};
    if (!data || size < 12) {
        if (error) *error = "stream smaller than the root-signature header";
        return false;
    }
    detail::Reader reader(data, size);
    if (!reader.u32(out.version) || !reader.u32(out.flags)) {
        if (error) *error = "truncated root-signature header";
        return false;
    }
    if (out.version != 1 && out.version != 2) {
        if (error) *error = "unsupported root-signature version";
        return false;
    }
    const bool version_1_1 = out.version == 2;

    std::uint32_t parameter_count = 0;
    if (!reader.u32(parameter_count) || parameter_count > 64) {
        if (error) *error = "invalid root parameter count";
        return false;
    }
    out.parameters.reserve(parameter_count);
    for (std::uint32_t i = 0; i < parameter_count; ++i) {
        RootParameter parameter{};
        std::uint32_t type = 0, visibility = 0;
        if (!reader.u32(type) || !reader.u32(visibility) || !detail::known_parameter(type) ||
            !detail::known_visibility(visibility)) {
            if (error) *error = "invalid root parameter header";
            return false;
        }
        parameter.type = static_cast<RootParameterType>(type);
        parameter.visibility = static_cast<ShaderVisibility>(visibility);
        if (parameter.type == RootParameterType::descriptor_table) {
            std::uint32_t range_count = 0;
            if (!reader.u32(range_count) || range_count > 64) {
                if (error) *error = "invalid descriptor range count";
                return false;
            }
            parameter.ranges.resize(range_count);
            for (auto& range : parameter.ranges) {
                std::uint32_t range_type = 0;
                if (!reader.u32(range_type) || !detail::known_range(range_type)) {
                    if (error) *error = "invalid descriptor range type";
                    return false;
                }
                range.type = static_cast<DescriptorRangeType>(range_type);
                if (!reader.u32(range.count) || !reader.u32(range.base_register) ||
                    !reader.u32(range.register_space) || !reader.u32(range.offset)) {
                    if (error) *error = "truncated descriptor range";
                    return false;
                }
                if (version_1_1 && !reader.u32(range.flags)) {
                    if (error) *error = "truncated 1.1 descriptor range flags";
                    return false;
                }
            }
        } else if (parameter.type == RootParameterType::constants_32bit) {
            if (!reader.u32(parameter.shader_register) || !reader.u32(parameter.register_space) ||
                !reader.u32(parameter.num_32bit_values)) {
                if (error) *error = "truncated 32-bit constants parameter";
                return false;
            }
        } else {
            if (!reader.u32(parameter.shader_register) || !reader.u32(parameter.register_space)) {
                if (error) *error = "truncated root descriptor parameter";
                return false;
            }
            if (version_1_1 && !reader.u32(parameter.flags)) {
                if (error) *error = "truncated 1.1 root descriptor flags";
                return false;
            }
        }
        out.parameters.push_back(std::move(parameter));
    }

    std::uint32_t sampler_count = 0;
    if (!reader.u32(sampler_count) || sampler_count > 64) {
        if (error) *error = "invalid static sampler count";
        return false;
    }
    out.static_samplers.resize(sampler_count);
    for (auto& sampler : out.static_samplers) {
        std::uint32_t visibility = 0;
        if (!reader.u32(sampler.filter) || !reader.u32(sampler.address_u) || !reader.u32(sampler.address_v) ||
            !reader.u32(sampler.address_w) || !reader.f32(sampler.mip_lod_bias) ||
            !reader.u32(sampler.max_anisotropy) || !reader.u32(sampler.comparison) ||
            !reader.u32(sampler.border_color) || !reader.f32(sampler.min_lod) || !reader.f32(sampler.max_lod) ||
            !reader.u32(sampler.shader_register) || !reader.u32(sampler.register_space) ||
            !reader.u32(visibility) || !detail::known_visibility(visibility)) {
            if (error) *error = "truncated static sampler";
            return false;
        }
        sampler.visibility = static_cast<ShaderVisibility>(visibility);
    }
    if (reader.remaining() != 0) {
        if (error) *error = "trailing bytes after the root signature";
        return false;
    }
    return true;
}

}  // namespace rt::d3d12
