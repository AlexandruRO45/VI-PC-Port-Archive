#pragma once

#include <vulkan/vulkan.h>

#include <cstdio>
#include <mutex>

namespace leonida {

inline VkCompositeAlphaFlagBitsKHR pick_composite_alpha(VkCompositeAlphaFlagsKHR supported) {
    // Preserve opaque or platform-defined composition when available before falling back to pixel alpha.
    constexpr VkCompositeAlphaFlagBitsKHR modes[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    };
    for (const auto alpha : modes) {
        if (!(supported & alpha)) continue;
        if (alpha == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR || alpha == VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
            static std::once_flag warned;
            std::call_once(warned, [alpha] {
                std::fprintf(stderr, "[rt] vulkan: swapchain uses %s alpha, so pixels with alpha below 1 can be translucent\n",
                             alpha == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR ? "premultiplied" : "postmultiplied");
            });
        }
        return alpha;
    }
    return {};
}

}  // namespace leonida
