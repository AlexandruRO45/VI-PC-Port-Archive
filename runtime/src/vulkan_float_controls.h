#pragma once

#include <string_view>
#include <vulkan/vulkan.h>

namespace leonida {

// The device property names emitted in the translator's float_controls index field.
inline const VkBool32* float_control_property(const VkPhysicalDeviceFloatControlsProperties& properties,
                                              std::string_view name) {
    if (name == "shaderSignedZeroInfNanPreserveFloat32") return &properties.shaderSignedZeroInfNanPreserveFloat32;
    if (name == "shaderDenormPreserveFloat16") return &properties.shaderDenormPreserveFloat16;
    return nullptr;
}

}  // namespace leonida
