#include "dxgi_swapchain.h"

#include "d3d12_com.h"
#include "dxgi_stubs.h"  // generated
#include "dxgi_object.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace rt::dxgi {

namespace {

HRESULT failed(const char* step, VkResult result) {
    std::fprintf(stderr, "[rt] dxgi: Present: %s returned VkResult %d\n", step, int(result));
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        std::fprintf(stderr, "[rt] dxgi: the window changed size; ResizeBuffers is not implemented yet\n");
    return result == VK_ERROR_DEVICE_LOST ? DXGI_ERROR_DEVICE_REMOVED : E_FAIL;
}

class SwapChain final
    : public DxgiObject<SwapChain, IDXGISwapChain3Stubs, IDXGIObject, IDXGIDeviceSubObject, IDXGISwapChain,
                        IDXGISwapChain1, IDXGISwapChain2, IDXGISwapChain3> {
public:
    // {9e47c1d2-3a58-4b0f-8d6e-71c2a9f04b3e}
    static constexpr GUID kTag{0x9e47c1d2, 0x3a58, 0x4b0f, {0x8d, 0x6e, 0x71, 0xc2, 0xa9, 0xf0, 0x4b, 0x3e}};

    SwapChain(leonida::VulkanContext& context, const leonida::PresentTarget& target, ID3D12CommandQueue* queue,
              HWND window, const DXGI_SWAP_CHAIN_DESC1& desc, VkFormat format, IUnknown* parent)
        : DxgiObject(parent), context_(context), queue_(queue), window_(window), desc_(desc),
          swapchain_(context, target, window, {desc.Width, desc.Height, format, desc.BufferCount}) {
        queue_->AddRef();
    }

    ~SwapChain() override {
        rt::d3d12::wait_idle(queue_);       // nothing may still use the images or wait on the semaphores
        for (ID3D12Resource* buffer : buffers_) buffer->Release();  // unregisters them while the images exist
        buffers_.clear();
        queue_->Release();
        // swapchain_ is destroyed after this body.
    }

    HRESULT init() {
        if (!swapchain_.ok()) {
            std::fprintf(stderr, "[rt] dxgi: CreateSwapChainForHwnd: %s\n", swapchain_.error());
            return E_FAIL;
        }
        if (swapchain_.image_count() != desc_.BufferCount) {
            std::fprintf(stderr, "[rt] dxgi: CreateSwapChainForHwnd: the driver made %u images for BufferCount %u\n",
                         swapchain_.image_count(), desc_.BufferCount);
            return DXGI_ERROR_UNSUPPORTED;
        }
        const VkExtent2D extent = swapchain_.extent();
        if (extent.width != desc_.Width || extent.height != desc_.Height) {
            if (desc_.Width || desc_.Height)
                std::fprintf(stderr, "[rt] dxgi: CreateSwapChainForHwnd: %ux%u requested, %ux%u (the client area) used\n",
                             desc_.Width, desc_.Height, extent.width, extent.height);
            desc_.Width = extent.width;
            desc_.Height = extent.height;
        }
        D3D12_RESOURCE_DESC resource{};
        resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resource.Width = extent.width;
        resource.Height = extent.height;
        resource.DepthOrArraySize = 1;
        resource.MipLevels = 1;
        resource.Format = desc_.Format;
        resource.SampleDesc = {1, 0};
        resource.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        for (std::uint32_t i = 0; i < swapchain_.image_count(); ++i) {
            leonida::ResourceBinding binding{};
            binding.image = swapchain_.image(i);
            binding.format = swapchain_.format();
            binding.width = extent.width;
            binding.height = extent.height;
            binding.presentable = true;
            // Swapchain images start UNDEFINED: the game's first PRESENT -> RENDER_TARGET transition discards.
            ID3D12Resource* buffer = nullptr;
            const HRESULT hr = rt::d3d12::create_image_resource(context_, binding, VK_IMAGE_LAYOUT_UNDEFINED, resource,
                                                                IID_PPV_ARGS(&buffer));
            if (FAILED(hr)) return hr;
            buffers_.push_back(buffer);
        }
        return S_OK;
    }

    VkImage image(UINT buffer) const { return buffer < swapchain_.image_count() ? swapchain_.image(buffer) : VK_NULL_HANDLE; }

    static SwapChain* from(IUnknown* object) { return DxgiObject::from<SwapChain>(object); }

    // IDXGIDeviceSubObject: for a D3D12 swapchain the "device" is the command queue.
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override { return queue_->QueryInterface(riid, out); }

    // IDXGISwapChain
    HRESULT STDMETHODCALLTYPE Present(UINT sync_interval, UINT flags) override {
        if (sync_interval > 4) return DXGI_ERROR_INVALID_CALL;
        if (flags & DXGI_PRESENT_TEST) return S_OK;  // never occluded: windowed only
        if (sync_interval != 1) rt::d3d12::stub("IDXGISwapChain::Present(SyncInterval != 1), presented with vsync");
        std::lock_guard lock(mutex_);
        const std::uint32_t image = swapchain_.current();
        const VkResult presented = rt::d3d12::present_on_queue(queue_, swapchain_.queue_present(), swapchain_.handle(),
                                                               image, swapchain_.ready(image));
        if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) return failed("vkQueuePresentKHR", presented);
        ++present_count_;
        const VkResult acquired = swapchain_.acquire_next();
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) return failed("vkAcquireNextImageKHR", acquired);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT buffer, REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (buffer >= buffers_.size()) return DXGI_ERROR_INVALID_CALL;
        return buffers_[buffer]->QueryInterface(riid, out);
    }

    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) override {
        if (fullscreen) *fullscreen = FALSE;
        if (target) *target = nullptr;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        *desc = {};
        desc->BufferDesc.Width = desc_.Width;
        desc->BufferDesc.Height = desc_.Height;
        desc->BufferDesc.Format = desc_.Format;
        desc->SampleDesc = desc_.SampleDesc;
        desc->BufferUsage = desc_.BufferUsage;
        desc->BufferCount = desc_.BufferCount;
        desc->OutputWindow = window_;
        desc->Windowed = TRUE;
        desc->SwapEffect = desc_.SwapEffect;
        desc->Flags = desc_.Flags;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* count) override {
        if (!count) return E_INVALIDARG;
        *count = present_count_;
        return S_OK;
    }

    // IDXGISwapChain1
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        *desc = desc_;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* window) override {
        if (!window) return E_INVALIDARG;
        *window = window_;
        return S_OK;
    }

    // Dirty rectangles and scroll are hints; they do not change what is shown.
    HRESULT STDMETHODCALLTYPE Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        return Present(sync_interval, flags);
    }

    // IDXGISwapChain3
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override {
        std::lock_guard lock(mutex_);
        return swapchain_.current();
    }

private:
    leonida::VulkanContext& context_;
    ID3D12CommandQueue* queue_;
    HWND window_;
    DXGI_SWAP_CHAIN_DESC1 desc_;
    leonida::VulkanSwapchain swapchain_;
    std::vector<ID3D12Resource*> buffers_;
    std::mutex mutex_;  // present and the current image
    UINT present_count_ = 0;
};

}  // namespace

VkFormat vulkan_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;  // flip-model swapchains take only these (no _SRGB back buffers)
    }
}

HRESULT create_swap_chain_for_hwnd(leonida::VulkanContext& context, const leonida::PresentTarget& target,
                                   ID3D12CommandQueue* queue, HWND window, const DXGI_SWAP_CHAIN_DESC1& desc,
                                   REFIID riid, void** out, IUnknown* parent) {
    if (!out) return DXGI_ERROR_INVALID_CALL;
    *out = nullptr;
    if (!queue || !window) return DXGI_ERROR_INVALID_CALL;
    // What DXGI requires of a D3D12 swapchain.
    if (desc.SwapEffect != DXGI_SWAP_EFFECT_FLIP_DISCARD && desc.SwapEffect != DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL)
        return DXGI_ERROR_INVALID_CALL;
    if (desc.BufferCount < 2 || desc.BufferCount > DXGI_MAX_SWAP_CHAIN_BUFFERS || desc.SampleDesc.Count != 1)
        return DXGI_ERROR_INVALID_CALL;
    const VkFormat format = vulkan_format(desc.Format);
    if (format == VK_FORMAT_UNDEFINED) return DXGI_ERROR_INVALID_CALL;
    if (desc.Stereo) {
        rt::d3d12::stub("CreateSwapChainForHwnd(Stereo)");
        return DXGI_ERROR_UNSUPPORTED;
    }
    auto* swapchain = new SwapChain(context, target, queue, window, desc, format, parent);
    HRESULT hr = swapchain->init();
    if (SUCCEEDED(hr)) hr = swapchain->QueryInterface(riid, out);
    swapchain->Release();
    return hr;
}

VkImage vulkan_image(IDXGISwapChain* swapchain, UINT buffer) {
    const SwapChain* s = SwapChain::from(swapchain);
    return s ? s->image(buffer) : VK_NULL_HANDLE;
}

}  // namespace rt::dxgi
