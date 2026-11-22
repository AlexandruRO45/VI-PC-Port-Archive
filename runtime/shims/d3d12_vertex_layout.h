#pragma once

#include <d3d12.h>
#include <vulkan/vulkan_core.h>
#include <vector>

namespace leonida { struct ShaderEntry; }

namespace rt::d3d12 {

struct VertexLayout {
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputBindingDivisorDescriptionEXT> divisors;
    UINT slots = 0;
    UINT zero_step_slots = 0;
};

HRESULT build_vertex_layout(const D3D12_INPUT_LAYOUT_DESC& layout, const leonida::ShaderEntry& vs,
                            const VkPhysicalDeviceLimits& limits, bool vertex_attribute_divisor, VertexLayout& out);

}  // namespace rt::d3d12
