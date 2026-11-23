// Where Present puts a swapchain's back buffer in the window's image, per DXGI_SCALING (D3D-444). Pure arithmetic,
// no Vulkan calls: dxgi_swapchain.h turns it into the blit, runtime/test/present_scaling_test.cpp checks it.
//
//   stretch               the back buffer fills the window (DXGI_SCALING_STRETCH, the default);
//   none                  unscaled at the top-left corner (top-right in a right-to-left window, WS_EX_LAYOUTRTL),
//                         cropped where the back buffer is larger (DXGI_SCALING_NONE; host DXGI keeps it on an HWND
//                         swapchain and draws unscaled, right-aligned in an RTL window: measured by @akimfomin, #4231);
//   aspect_ratio_stretch  the largest size that keeps the back buffer's aspect ratio, centred
//                         (DXGI_SCALING_ASPECT_RATIO_STRETCH; host DXGI accepts it only for CoreWindow and
//                         composition swapchains, so the layer reaches it only through CreateSwapChainForCoreWindow).
// Where the copy doesn't cover the window, the rest is cleared first to the swapchain's background colour
// (SetBackgroundColor; opaque black by default), as DXGI fills it.
#pragma once

#include <algorithm>
#include <cstdint>

namespace leonida {

enum class PresentScaling : std::uint32_t { stretch = 0, none = 1, aspect_ratio_stretch = 2 };

struct SwapchainBlit {
    std::uint32_t src_x = 0;                      // the copied part's left column in the back buffer
    std::uint32_t src_width = 0, src_height = 0;  // the part of the back buffer copied (from its top row)
    std::uint32_t dst_x = 0, dst_y = 0;           // where it goes in the window's image
    std::uint32_t dst_width = 0, dst_height = 0;
    bool scaled = false;  // the copy changes size (filter it); false: texel for texel
    bool covers = true;   // the copy fills the whole window's image; false: clear the rest first
};

inline SwapchainBlit swapchain_blit(PresentScaling scaling, std::uint32_t src_width, std::uint32_t src_height,
                                std::uint32_t dst_width, std::uint32_t dst_height, bool right_to_left = false) {
    SwapchainBlit r;
    r.src_width = src_width;
    r.src_height = src_height;
    switch (scaling) {
    case PresentScaling::none:
        r.src_width = r.dst_width = std::min(src_width, dst_width);
        r.src_height = r.dst_height = std::min(src_height, dst_height);
        if (right_to_left) {  // the window's origin is its top-right corner: right-aligned, and a wider back
            r.dst_x = dst_width - r.dst_width;            // buffer is cropped to its right part (measured against
            r.src_x = src_width - r.src_width;            // native D3D12 by @akimfomin, #4231)
        }
        break;
    case PresentScaling::aspect_ratio_stretch:
        if (!src_width || !src_height) {  // nothing to copy: the window's image is only cleared
            r.covers = false;
            return r;
        }
        // Width-limited when the back buffer is wider than the window, relative to their heights (64-bit products).
        if (std::uint64_t(src_width) * dst_height >= std::uint64_t(dst_width) * src_height) {
            r.dst_width = dst_width;
            r.dst_height = std::uint32_t((std::uint64_t(dst_width) * src_height + src_width / 2) / src_width);
        } else {
            r.dst_height = dst_height;
            r.dst_width = std::uint32_t((std::uint64_t(dst_height) * src_width + src_height / 2) / src_height);
        }
        r.dst_width = std::min(r.dst_width, dst_width);
        r.dst_height = std::min(r.dst_height, dst_height);
        r.dst_x = (dst_width - r.dst_width) / 2;
        r.dst_y = (dst_height - r.dst_height) / 2;
        break;
    case PresentScaling::stretch:
    default:
        r.dst_width = dst_width;
        r.dst_height = dst_height;
        break;
    }
    r.scaled = r.src_width != r.dst_width || r.src_height != r.dst_height;
    r.covers = r.dst_x == 0 && r.dst_y == 0 && r.dst_width == dst_width && r.dst_height == dst_height;
    return r;
}

}  // namespace leonida
