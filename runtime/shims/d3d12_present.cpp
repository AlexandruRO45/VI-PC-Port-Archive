// The Xbox present path (PresentX) over the shared-device presenter. See d3d12_present.h.
//
// PresentX differs from DXGI's Present in that one call can carry several planes with their own source and
// destination rectangles (the UI composed over the world). The presenter takes the list as planes: a single plane
// that fills the swapchain is a GPU copy, anything else is blitted into place, all in one submission.

#include "d3d12_present.h"

#include "d3d12_com.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>

namespace rt::d3d12 {
namespace {

std::mutex& target_mutex() {
    static std::mutex mutex;
    return mutex;
}

leonida::wsi::Presenter* g_target = nullptr;

void log_once(const char* what, int value) {
    static std::mutex mutex;
    static std::set<int> seen;
    std::lock_guard lock(mutex);
    if (!seen.insert(value).second) return;
    std::fprintf(stderr, "[rt] d3d12: %s (%d)\n", what, value);
}

VkFormat vulkan_format_of(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

bool empty_rect(const RECT& rect) { return !rect.left && !rect.top && !rect.right && !rect.bottom; }

VkRect2D rect_of(const RECT& rect) {
    if (empty_rect(rect)) return {};
    return {{rect.left, rect.top},
            {std::uint32_t(std::max<LONG>(rect.right - rect.left, 0)), std::uint32_t(std::max<LONG>(rect.bottom - rect.top, 0))}};
}

}  // namespace

void set_presentation_target(leonida::wsi::Presenter* presenter) {
    std::lock_guard lock(target_mutex());
    g_target = presenter;
}

leonida::wsi::Presenter* presentation_target() {
    std::lock_guard lock(target_mutex());
    return g_target;
}

HRESULT present_x(ID3D12CommandQueue* queue, std::span<const PresentPlane> planes) {
    if (planes.empty()) return E_INVALIDARG;
    leonida::wsi::Presenter* presenter = presentation_target();
    if (!presenter) {
        std::fprintf(stderr, "[rt] d3d12: PresentX: no swapchain is registered for presentation\n");
        return E_FAIL;
    }

    // The layer's tracker knows the layout each plane's frame left it in, and the presenter restores it after the
    // copy, so the next frame's render target stays in the state the tracker names.
    ID3D12Device* device = nullptr;
    if (queue) queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device));

    constexpr std::size_t kMaxPlanes = 8;
    leonida::wsi::Presenter::Plane list[kMaxPlanes]{};
    std::size_t count = 0;
    for (const PresentPlane& plane : planes) {
        if (!plane.resource) continue;
        const D3D12_RESOURCE_DESC desc = plane.resource->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("PresentX: a plane is not a 2D texture", int(desc.Dimension));
            continue;
        }
        leonida::wsi::Presenter::Plane& out = list[count++];
        out.image = vulkan_image(plane.resource);
        out.format = vulkan_format_of(desc.Format);
        out.extent = {std::uint32_t(desc.Width), desc.Height};
        out.layout = device ? vulkan_image_layout(device, plane.resource) : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        out.source = rect_of(plane.source);
        out.destination = rect_of(plane.destination);
        if (count == kMaxPlanes) {
            if (planes.size() > count) log_once("PresentX: only the first planes are presented", int(planes.size()));
            break;
        }
    }
    if (device) device->Release();
    if (!count) return E_INVALIDARG;
    if (!presenter->present(list, count)) {
        std::fprintf(stderr, "[rt] d3d12: PresentX: presentation failed\n");
        return E_FAIL;
    }
    return S_OK;
}

}  // namespace rt::d3d12
