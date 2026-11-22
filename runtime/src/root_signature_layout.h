#pragma once

// D3D12 root signature -> Vulkan layout description (PR B).
//
// A descriptor table becomes one descriptor set (its ordinal among the tables is the set index, up to 3); each
// descriptor range inside it becomes a binding. 32-bit constants become one push-constant range per parameter,
// packed from offset 0. Root CBV/SRV/UAV descriptors are 64-bit VA push constants in D3D12; this layer records them
// but leaves the shader contract (which uses SGPR pairs) to the translator, so they only contribute their stage.
//
// The result is plain data: the device turns it into VkDescriptorSetLayouts and a VkPipelineLayout. That split keeps
// the mapping testable without a GPU.

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "d3d12_root_signature.h"

namespace rt::d3d12 {

struct RootLayoutBinding {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    std::uint32_t count = 1;
    VkShaderStageFlags stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
};

struct RootLayoutDesc {
    std::vector<RootLayoutBinding> bindings;
    std::vector<VkPushConstantRange> push_ranges;
    std::uint32_t set_count = 0;
};

namespace detail {

inline VkShaderStageFlags stage_of(ShaderVisibility visibility) {
    switch (visibility) {
        case ShaderVisibility::vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderVisibility::hull: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderVisibility::domain: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case ShaderVisibility::geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case ShaderVisibility::pixel: return VK_SHADER_STAGE_FRAGMENT_BIT;
        default: return VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
}

inline VkDescriptorType descriptor_type(DescriptorRangeType type) {
    switch (type) {
        case DescriptorRangeType::cbv: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorRangeType::uav: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case DescriptorRangeType::sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        default: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }
}

}  // namespace detail

// Returns false with `error` for a shape Vulkan cannot express: more than four descriptor tables, a zero range, or
// push constants past the 128-byte minimum guarantee.
inline bool build_root_layout(const RootSignatureDesc& signature, RootLayoutDesc& out, std::string* error = nullptr) {
    out = RootLayoutDesc{};
    std::uint32_t table_index = 0;
    std::uint32_t push_offset = 0;
    for (const RootParameter& parameter : signature.parameters) {
        if (parameter.type == RootParameterType::descriptor_table) {
            if (table_index >= 4) {
                if (error) *error = "more than four descriptor tables";
                return false;
            }
            for (const DescriptorRange& range : parameter.ranges) {
                if (range.count == 0) {
                    if (error) *error = "descriptor range with zero descriptors";
                    return false;
                }
                RootLayoutBinding binding{};
                binding.set = table_index;
                binding.binding = range.base_register;
                binding.type = detail::descriptor_type(range.type);
                binding.count = range.count == 0xFFFFFFFFu ? 1024u : range.count;
                binding.stage = detail::stage_of(parameter.visibility);
                out.bindings.push_back(binding);
            }
            ++table_index;
        } else if (parameter.type == RootParameterType::constants_32bit) {
            if (!parameter.num_32bit_values) {
                if (error) *error = "32-bit constants parameter with zero values";
                return false;
            }
            VkPushConstantRange range{};
            range.stageFlags = detail::stage_of(parameter.visibility);
            range.offset = push_offset;
            range.size = parameter.num_32bit_values * 4;
            out.push_ranges.push_back(range);
            push_offset += range.size;
            if (push_offset > 128) {
                if (error) *error = "push constants exceed the 128-byte guaranteed minimum";
                return false;
            }
        }
        // root CBV/SRV/UAV: carried as an SGPR pair by the translator, no Vulkan descriptor of its own.
    }
    out.set_count = table_index;
    return true;
}

}  // namespace rt::d3d12
