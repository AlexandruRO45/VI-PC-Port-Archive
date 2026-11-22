// The Xbox present path (PresentX) over the shared-device presenter. See d3d12_present.h.

#include "d3d12_present.h"

#include "d3d12_com.h"

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
    if (planes.empty() || !planes.front().resource) return E_INVALIDARG;
    if (planes.size() > 1) {
        // An Xbox title may present several planes (UI over the world); composing them is future work.
        log_once("PresentX: only the first of several planes is presented", int(planes.size()));
    }
    const PresentPlane& plane = planes.front();
    if (!empty_rect(plane.source) || !empty_rect(plane.destination))
        log_once("PresentX: source/destination rectangles are not composited yet", 1);

    leonida::wsi::Presenter* presenter = presentation_target();
    if (!presenter) {
        std::fprintf(stderr, "[rt] d3d12: PresentX: no swapchain is registered for presentation\n");
        return E_FAIL;
    }

    const D3D12_RESOURCE_DESC desc = plane.resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        std::fprintf(stderr, "[rt] d3d12: PresentX: the present plane is not a 2D texture\n");
        return E_INVALIDARG;
    }

    // The layer's tracker knows the layout the frame left the image in, and the presenter restores it after the copy.
    VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ID3D12Device* device = nullptr;
    if (queue && SUCCEEDED(queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) && device) {
        layout = vulkan_image_layout(device, plane.resource);
        device->Release();
    }
    const VkExtent2D extent{std::uint32_t(desc.Width), desc.Height};
    if (!presenter->present(vulkan_image(plane.resource), vulkan_format_of(desc.Format), extent, layout)) {
        std::fprintf(stderr, "[rt] d3d12: PresentX: presentation failed\n");
        return E_FAIL;
    }
    return S_OK;
}

}  // namespace rt::d3d12
