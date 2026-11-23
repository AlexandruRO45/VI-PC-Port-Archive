#pragma once

#include "../src/vulkan_bindless.h"

namespace leonida::vk_test {

// The checker creates pipelines without binding descriptors or dispatching work.
struct PipelineCheckDescriptors {
    BindlessFeatures features;
    bool compact;
    uint32_t storage_image_limit;

    PipelineCheckDescriptors(const VkPhysicalDeviceVulkan12Features& enabled, const VkPhysicalDeviceLimits& limits,
                             const VkPhysicalDeviceVulkan11Properties& v11, const VkPhysicalDeviceVulkan12Properties& v12) {
        features.update_after_bind = enabled.descriptorBindingSampledImageUpdateAfterBind &&
                                     enabled.descriptorBindingStorageImageUpdateAfterBind &&
                                     enabled.descriptorBindingStorageBufferUpdateAfterBind;
        features.partially_bound = enabled.descriptorBindingPartiallyBound;
        storage_image_limit = features.update_after_bind
            ? std::min(v12.maxPerStageDescriptorUpdateAfterBindStorageImages, v12.maxDescriptorSetUpdateAfterBindStorageImages)
            : std::min(limits.maxPerStageDescriptorStorageImages, limits.maxDescriptorSetStorageImages);
        bool minimum_fits;
        bindless::size_bindings(features, limits, v11, v12, {}, &minimum_fits);
        // Some ordinary sets cannot hold even the runtime's secondary sampled-image arrays. A single
        // descriptor per statically used runtime array suffices to compile these shaders on such devices.
        compact = !features.update_after_bind && !minimum_fits;
    }

    std::optional<uint32_t> layout_mask(uint32_t shader_bindings) const {
        // The shared layout has no multisample fallback images. Keep any MS bindings the shader declares
        // in its own layout, without charging their arrays to shaders that fit the shared layout.
        if (compact || (shader_bindings >> bindless::first_ms_binding)) return shader_bindings;
        return std::nullopt;
    }

    bool fits(uint32_t shader_bindings) const {
        if (!layout_mask(shader_bindings)) return true;
        // At one descriptor per binding, only storage images can exceed a Vulkan minimum (four per stage).
        uint32_t storage = 0;
        for (uint32_t b = 0; b < bindless::binding_count; ++b)
            if ((shader_bindings & (1u << b)) && bindless::bindings[b].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                ++storage;
        return storage <= storage_image_limit;
    }
};

} // namespace leonida::vk_test
