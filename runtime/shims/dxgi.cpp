// dxgi.dll shim: a factory, an adapter and swapchains over the shared-device WSI presenter.
//
// Swapchain back buffers are D3D12 committed render targets the game draws into; Present() hands the image to the
// presenter, which copies it on the GPU into the swapchain image of the window's VkSurfaceKHR and calls
// vkQueuePresentKHR (shared-device WSI, docs/d3d12_vulkan.md section 4.5). ResizeBuffers() recreates the presenter
// for the new size. Everything else (output enumeration, fullscreen, stereo, composition swapchains) is a logged
// stub: the trace will say what the game actually needs.

#include "dxgi.h"

#include <cstdio>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <set>
#include <vector>

#include "d3d12_com.h"
#include "d3d12_present.h"
#include "dxgi_stubs.h"  // generated from dxgi.h/dxgi1_2.h at build time
#include "shims.h"
#include "wsi_presenter.h"

namespace rt::d3d12 {
namespace {

bool guid_equal(REFIID a, REFIID b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

void log_once(const char* what, int value) {
    static std::mutex mutex;
    static std::set<int> seen;
    std::lock_guard lock(mutex);
    if (!seen.insert(value).second) return;
    std::fprintf(stderr, "[rt] dxgi: %s (%d)\n", what, value);
}

VkFormat vulkan_format_of(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

DXGI_FORMAT dxgi_format_of(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

// IUnknown + IDXGIObject for the DXGI objects, in the same style as the D3D12 ones.
template<class Derived, class Stubs, class... Interfaces>
class DxgiObject : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete static_cast<Derived*>(this);
        return left;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (guid_equal(riid, __uuidof(IUnknown)) || ((guid_equal(riid, __uuidof(Interfaces))) || ...)) {
            *out = static_cast<Stubs*>(this);
        } else {
            std::fprintf(stderr,
                         "[rt] dxgi: QueryInterface: unknown IID {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n",
                         static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                         riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT* size, void*) override {
        if (size) *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }

private:
    ULONG refs_ = 1;
};

// One adapter: the host GPU, described generically. A game uses the description for settings only.
class Adapter final : public DxgiObject<Adapter, IDXGIAdapter1Stubs, IDXGIObject, IDXGIAdapter, IDXGIAdapter1> {
public:
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_ADAPTER_DESC* out) override {
        if (!out) return E_INVALIDARG;
        std::memset(out, 0, sizeof(*out));
        out->VendorId = 0x1002;  // AMD: the console's GPU family the shaders are translated for
        out->DeviceId = 0x73BF;
        wcscpy_s(out->Description, L"Leonida (host Vulkan GPU)");
        out->DedicatedVideoMemory = 0;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_ADAPTER_DESC1* out) override {
        if (!out) return E_INVALIDARG;
        DXGI_ADAPTER_DESC desc{};
        GetDesc(&desc);
        std::memset(out, 0, sizeof(*out));
        out->VendorId = desc.VendorId;
        out->DeviceId = desc.DeviceId;
        wcscpy_s(out->Description, desc.Description);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE EnumOutputs(UINT, IDXGIOutput** out) override {
        if (out) *out = nullptr;
        return DXGI_ERROR_NOT_FOUND;  // output enumeration is future work: the window is what presents
    }
    HRESULT STDMETHODCALLTYPE CheckInterfaceSupport(REFGUID, LARGE_INTEGER*) override { return DXGI_ERROR_UNSUPPORTED; }
};

// A window's swapchain: back buffers as D3D12 render targets, presented through the WSI presenter.
class SwapChain final
    : public DxgiObject<SwapChain, IDXGISwapChain1Stubs, IDXGIObject, IDXGIDeviceSubObject, IDXGISwapChain,
                        IDXGISwapChain1> {
public:
    SwapChain(ID3D12Device* device, HWND window, const DXGI_SWAP_CHAIN_DESC1& desc) : desc_(desc), window_(window) {
        if (device) {
            device->AddRef();  // the swapchain keeps the device alive, as IDXGIDeviceSubObject implies
            device_ = device;
        }
        if (!create_presenter()) log_once("swapchain presenter creation failed", 1);
        set_presentation_target(&presenter_);
    }

    ~SwapChain() {
        if (presentation_target() == &presenter_) set_presentation_target(nullptr);
        release_buffers();
        if (device_) device_->Release();
    }

    HRESULT STDMETHODCALLTYPE Present(UINT sync_interval, UINT flags) override {
        if (flags & DXGI_PRESENT_TEST) return S_OK;
        ID3D12Resource* back_buffer = buffer(0);
        if (!back_buffer) {
            std::fprintf(stderr, "[rt] dxgi: Present without a back buffer (GetBuffer was never called)\n");
            return DXGI_ERROR_INVALID_CALL;
        }
        const D3D12_RESOURCE_DESC desc = back_buffer->GetDesc();
        const VkImage image = vulkan_image(back_buffer);
        if (!image) {
            std::fprintf(stderr, "[rt] dxgi: Present: the back buffer is not a Vulkan image\n");
            return DXGI_ERROR_INVALID_CALL;
        }
        const VkImageLayout layout = vulkan_image_layout(device_, back_buffer);
        const VkExtent2D extent{std::uint32_t(desc.Width), desc.Height};
        if (!presenter_.present(image, vulkan_format_of(buffer_format()), extent, layout)) {
            std::fprintf(stderr, "[rt] dxgi: Present failed\n");
            return DXGI_ERROR_DEVICE_REMOVED;
        }
        if (sync_interval > 1) log_once("Present: sync interval above 1 is treated as 1 (FIFO)", int(sync_interval));
        ++presents_;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        return Present(sync_interval, flags);
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT index, REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        ID3D12Resource* resource = buffer(index);
        if (!resource) return DXGI_ERROR_INVALID_CALL;
        return resource->QueryInterface(riid, out);
    }

    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT format,
                                            UINT /*flags*/) override {
        release_buffers();
        if (count) desc_.BufferCount = count;
        if (width) desc_.Width = width;
        if (height) desc_.Height = height;
        if (format != DXGI_FORMAT_UNKNOWN) {
            desc_.Format = format;
        }
        if (!create_presenter()) return E_FAIL;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* out) override {
        if (!out) return E_INVALIDARG;
        std::memset(out, 0, sizeof(*out));
        out->BufferDesc.Width = width();
        out->BufferDesc.Height = height();
        out->BufferDesc.RefreshRate = {60, 1};
        out->BufferDesc.Format = buffer_format();
        out->BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
        out->BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
        out->SampleDesc = desc_.SampleDesc;
        out->BufferUsage = desc_.BufferUsage;
        out->BufferCount = desc_.BufferCount ? desc_.BufferCount : 1;
        out->OutputWindow = window_;
        out->Windowed = TRUE;
        out->SwapEffect = desc_.SwapEffect;
        out->Flags = desc_.Flags;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* out) override {
        if (!out) return E_INVALIDARG;
        *out = desc_;
        out->Width = width();
        out->Height = height();
        out->Format = buffer_format();
        if (!out->BufferCount) out->BufferCount = 1;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) override {
        if (fullscreen) *fullscreen = FALSE;
        if (target) *target = nullptr;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* out) override {
        if (!out) return E_INVALIDARG;
        std::memset(out, 0, sizeof(*out));
        out->Windowed = TRUE;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* out) override {
        if (!out) return E_INVALIDARG;
        *out = window_;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!device_) return E_FAIL;
        return device_->QueryInterface(riid, out);
    }

    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* count) override {
        if (!count) return E_INVALIDARG;
        *count = static_cast<UINT>(presents_);
        return S_OK;
    }

private:
    // The presenter's own format is the fast path (a plain image copy); a format the game asked for explicitly is
    // kept as it is and the presenter blits instead.
    DXGI_FORMAT buffer_format() const {
        if (desc_.Format != DXGI_FORMAT_UNKNOWN) return desc_.Format;
        return dxgi_format_of(presenter_.format());
    }
    UINT width() const { return desc_.Width ? desc_.Width : presenter_.extent().width; }
    UINT height() const { return desc_.Height ? desc_.Height : presenter_.extent().height; }

    bool create_presenter() {
        const VulkanDeviceInfo info = vulkan_device_info(device_);
        if (!info.device) {
            std::fprintf(stderr, "[rt] dxgi: the device is not a Vulkan-backed D3D12 device\n");
            return false;
        }
        const leonida::wsi::Gpu gpu{info.get_instance_proc, info.instance, info.physical, info.device, info.queue,
                                    info.queue_family};
        const VkFormat wanted = desc_.Format == DXGI_FORMAT_UNKNOWN ? VK_FORMAT_R8G8B8A8_UNORM
                                                                   : vulkan_format_of(desc_.Format);
        return presenter_.create(gpu, window_, wanted, VkExtent2D{desc_.Width, desc_.Height});
    }

    void release_buffers() {
        for (ID3D12Resource*& resource : buffers_) {
            if (resource) resource->Release();
            resource = nullptr;
        }
        buffers_.clear();
    }

    ID3D12Resource* buffer(UINT index) {
        if (index >= 8) return nullptr;  // no more than a game ever uses; keeps a bad index from allocating
        if (buffers_.size() <= index) buffers_.resize(index + 1, nullptr);
        if (buffers_[index]) return buffers_[index];
        if (!device_) return nullptr;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width();
        desc.Height = height();
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = buffer_format();
        desc.SampleDesc = {1, 0};
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;
        ID3D12Resource* resource = nullptr;
        const HRESULT hr = device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resource));
        if (FAILED(hr)) {
            std::fprintf(stderr, "[rt] dxgi: back buffer creation failed (0x%08lx)\n", static_cast<unsigned long>(hr));
            return nullptr;
        }
        buffers_[index] = resource;
        return resource;
    }

    ID3D12Device* device_ = nullptr;
    HWND window_ = nullptr;
    DXGI_SWAP_CHAIN_DESC1 desc_{};
    std::vector<ID3D12Resource*> buffers_;
    leonida::wsi::Presenter presenter_;
    UINT64 presents_ = 0;
};

class Factory final
    : public DxgiObject<Factory, IDXGIFactory2Stubs, IDXGIObject, IDXGIFactory, IDXGIFactory1, IDXGIFactory2> {
public:
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT index, IDXGIAdapter** out) override {
        return adapter(index, __uuidof(IDXGIAdapter), reinterpret_cast<void**>(out));
    }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** out) override {
        return adapter(index, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(out));
    }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* out) override {
        if (!out) return E_INVALIDARG;
        *out = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc,
                                              IDXGISwapChain** out) override {
        if (!desc || !out) return E_INVALIDARG;
        *out = nullptr;
        DXGI_SWAP_CHAIN_DESC1 desc1{};
        desc1.Width = desc->BufferDesc.Width;
        desc1.Height = desc->BufferDesc.Height;
        desc1.Format = desc->BufferDesc.Format;
        desc1.SampleDesc = desc->SampleDesc;
        desc1.BufferUsage = desc->BufferUsage;
        desc1.BufferCount = desc->BufferCount;
        desc1.Scaling = DXGI_SCALING_STRETCH;
        desc1.SwapEffect = desc->SwapEffect;
        desc1.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc1.Flags = desc->Flags;
        IDXGISwapChain1* chain = nullptr;
        const HRESULT hr = CreateSwapChainForHwnd(device, desc->OutputWindow, &desc1, nullptr, nullptr, &chain);
        if (FAILED(hr)) return hr;
        const HRESULT queried = chain->QueryInterface(__uuidof(IDXGISwapChain), reinterpret_cast<void**>(out));
        chain->Release();
        return queried;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown* device, HWND window, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                     IDXGISwapChain1** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!device || !window || !desc) return E_INVALIDARG;
        ID3D12Device* d3d12 = nullptr;
        if (FAILED(device->QueryInterface(__uuidof(ID3D12Device), reinterpret_cast<void**>(&d3d12)))) {
            std::fprintf(stderr, "[rt] dxgi: CreateSwapChainForHwnd: the device is not an ID3D12Device\n");
            return DXGI_ERROR_INVALID_CALL;
        }
        auto* chain = new SwapChain(d3d12, window, *desc);
        d3d12->Release();
        return chain->QueryInterface(__uuidof(IDXGISwapChain1), reinterpret_cast<void**>(out));
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                            IDXGISwapChain1** out) override {
        if (out) *out = nullptr;
        stub("IDXGIFactory2::CreateSwapChainForComposition");
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override {
        stub("IDXGIFactory::CreateSoftwareAdapter");
        return E_NOTIMPL;
    }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }

private:
    HRESULT adapter(UINT index, REFIID riid, void** out) {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (index != 0) return DXGI_ERROR_NOT_FOUND;  // one adapter: the host GPU
        auto* adapter = new Adapter();
        return adapter->QueryInterface(riid, out);
    }
};

}  // namespace

HRESULT create_dxgi_factory(REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* factory = new Factory();
    return factory->QueryInterface(riid, out);
}

}  // namespace rt::d3d12

extern "C" HRESULT WINAPI CreateDXGIFactory(REFIID riid, void** out) {
    return rt::d3d12::create_dxgi_factory(riid, out);
}
extern "C" HRESULT WINAPI CreateDXGIFactory1(REFIID riid, void** out) {
    return rt::d3d12::create_dxgi_factory(riid, out);
}
extern "C" HRESULT WINAPI CreateDXGIFactory2(UINT, REFIID riid, void** out) {
    return rt::d3d12::create_dxgi_factory(riid, out);
}
