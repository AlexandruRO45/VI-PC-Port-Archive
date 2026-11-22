// Presentation target shared by the layer's present paths (DXGI swapchain and Xbox PresentX).
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <span>

#include "wsi_presenter.h"

namespace rt::d3d12 {

class SwapChain;

// The process's presentation target: the swapchain that owns the window. An Xbox title has one swapchain, and
// PresentX presents into it, so a single pointer is enough; the swapchain clears it when it is destroyed.
void set_presentation_target(leonida::wsi::Presenter* presenter);
leonida::wsi::Presenter* presentation_target();

// One present plane. The fields the layer uses are here; the rest of D3D12XBOX_PRESENT_PLANE_PARAMETERS is an open
// question (docs section 4.5), so the queue proxy will fill these in once the console layout is pinned down.
struct PresentPlane {
    ID3D12Resource* resource = nullptr;  // the image the frame was rendered into
    RECT source{};                       // what to present; empty means the whole resource
    RECT destination{};                  // where in the swapchain; empty means the whole swapchain
};

// ID3D12CommandQueue::PresentX (Xbox ABI) for a queue whose device renders into the shared swapchain: presents the
// frame through vkQueuePresentKHR on the same VkDevice the frame was rendered on (shared-device WSI). The Xbox
// vtable slot is not measured yet, so this is the body a queue proxy calls; `planes` beyond the first are logged and
// not composited yet (present-plane composition is future work, docs section 4.5).
HRESULT present_x(ID3D12CommandQueue* queue, std::span<const PresentPlane> planes);

}  // namespace rt::d3d12
