#include "dxgi_factory.h"

#include "d3d12_com.h"
#include "dxgi_swapchain.h"
#include "dxgi_stubs.h"  // generated
#include "dxgi_object.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace rt::dxgi {

namespace {

class Adapter final : public DxgiObject<Adapter, IDXGIAdapter1Stubs, IDXGIObject, IDXGIAdapter, IDXGIAdapter1> {
public:
    // {3b81f6c0-52e9-4d7a-9c14-a6f08d2e7b55}
    static constexpr GUID kTag{0x3b81f6c0, 0x52e9, 0x4d7a, {0x9c, 0x14, 0xa6, 0xf0, 0x8d, 0x2e, 0x7b, 0x55}};

    Adapter(const Host& host, IUnknown* factory) : DxgiObject(factory) { describe(host); }

    LUID luid() const { return desc_.AdapterLuid; }

    HRESULT STDMETHODCALLTYPE EnumOutputs(UINT, IDXGIOutput** out) override {
        if (!out) return E_INVALIDARG;
        *out = nullptr;
        rt::d3d12::stub("IDXGIAdapter::EnumOutputs (no outputs: windowed only)");
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_ADAPTER_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        static_assert(sizeof(DXGI_ADAPTER_DESC) + sizeof(UINT) <= sizeof(DXGI_ADAPTER_DESC1));
        std::memcpy(desc, &desc_, sizeof(*desc));  // DXGI_ADAPTER_DESC1 is DXGI_ADAPTER_DESC plus Flags
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_ADAPTER_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        *desc = desc_;
        return S_OK;
    }

    // What DXGI answers for D3D12 adapters: the question is a D3D10/11 one.
    HRESULT STDMETHODCALLTYPE CheckInterfaceSupport(REFGUID, LARGE_INTEGER*) override {
        return DXGI_ERROR_UNSUPPORTED;
    }

private:
    void describe(const Host& host) {
        const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            host.get_instance_proc(host.instance, "vkGetPhysicalDeviceProperties2"));
        const auto memory_of = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            host.get_instance_proc(host.instance, "vkGetPhysicalDeviceMemoryProperties"));
        desc_ = {};
        if (!properties2 || !memory_of) return;
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ids};
        properties2(host.physical, &properties);
        const VkPhysicalDeviceProperties& p = properties.properties;
        MultiByteToWideChar(CP_UTF8, 0, p.deviceName, -1, desc_.Description,
                            int(sizeof(desc_.Description) / sizeof(desc_.Description[0])) - 1);
        desc_.VendorId = p.vendorID;
        desc_.DeviceId = p.deviceID;
        if (ids.deviceLUIDValid) {
            std::memcpy(&desc_.AdapterLuid.LowPart, ids.deviceLUID, 4);
            std::memcpy(&desc_.AdapterLuid.HighPart, ids.deviceLUID + 4, 4);
        }
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) desc_.Flags = DXGI_ADAPTER_FLAG_SOFTWARE;
        // Device-local heaps are the adapter's own memory; the rest is system memory it can reach. On an
        // integrated GPU every heap may be device-local, so the split follows the flag, not the device type.
        VkPhysicalDeviceMemoryProperties memory{};
        memory_of(host.physical, &memory);
        for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
            const VkMemoryHeap& heap = memory.memoryHeaps[i];
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                desc_.DedicatedVideoMemory += SIZE_T(heap.size);
            else
                desc_.SharedSystemMemory += SIZE_T(heap.size);
        }
    }

    DXGI_ADAPTER_DESC1 desc_{};
};

class Factory final
    : public DxgiObject<Factory, IDXGIFactory5Stubs, IDXGIObject, IDXGIFactory, IDXGIFactory1, IDXGIFactory2,
                        IDXGIFactory3, IDXGIFactory4, IDXGIFactory5> {
public:
    // {c6d250a7-1e8f-4a39-b47d-0f9b3c58e216}
    static constexpr GUID kTag{0xc6d250a7, 0x1e8f, 0x4a39, {0xb4, 0x7d, 0x0f, 0x9b, 0x3c, 0x58, 0xe2, 0x16}};

    Factory(const Host& host, UINT flags) : host_(host), flags_(flags) {}

    // IDXGIFactory
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT index, IDXGIAdapter** out) override {
        return enum_adapter(index, __uuidof(IDXGIAdapter), reinterpret_cast<void**>(out));
    }

    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND window, UINT) override {
        window_ = window;  // Alt+Enter and friends: nothing to do while presentation is windowed only
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* window) override {
        if (!window) return DXGI_ERROR_INVALID_CALL;
        *window = window_;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc,
                                              IDXGISwapChain** out) override {
        if (!desc || !out) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        if (!desc->Windowed) rt::d3d12::stub("IDXGIFactory::CreateSwapChain(Windowed = FALSE), presented windowed");
        DXGI_SWAP_CHAIN_DESC1 desc1{};
        desc1.Width = desc->BufferDesc.Width;
        desc1.Height = desc->BufferDesc.Height;
        desc1.Format = desc->BufferDesc.Format;
        desc1.SampleDesc = desc->SampleDesc;
        desc1.BufferUsage = desc->BufferUsage;
        desc1.BufferCount = desc->BufferCount;
        desc1.Scaling = DXGI_SCALING_STRETCH;
        desc1.SwapEffect = desc->SwapEffect;
        desc1.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
        desc1.Flags = desc->Flags;
        return create_swap_chain(device, desc->OutputWindow, desc1, __uuidof(IDXGISwapChain),
                                 reinterpret_cast<void**>(out));
    }

    // IDXGIFactory1
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** out) override {
        return enum_adapter(index, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(out));
    }

    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }

    // IDXGIFactory2
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return FALSE; }

    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown* device, HWND window, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
                                                     IDXGIOutput* restrict_to, IDXGISwapChain1** out) override {
        if (!desc || !out) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        if (fullscreen && !fullscreen->Windowed)
            rt::d3d12::stub("IDXGIFactory2::CreateSwapChainForHwnd(fullscreen), presented windowed");
        if (restrict_to) rt::d3d12::stub("IDXGIFactory2::CreateSwapChainForHwnd(pRestrictToOutput), ignored");
        return create_swap_chain(device, window, *desc, __uuidof(IDXGISwapChain1), reinterpret_cast<void**>(out));
    }

    // IDXGIFactory3
    UINT STDMETHODCALLTYPE GetCreationFlags() override { return flags_; }

    // IDXGIFactory4
    HRESULT STDMETHODCALLTYPE EnumAdapterByLuid(LUID luid, REFIID riid, void** out) override {
        if (!out) return E_INVALIDARG;
        *out = nullptr;
        auto* adapter = new Adapter(host_, static_cast<IDXGIFactory5*>(this));
        const LUID ours = adapter->luid();
        HRESULT hr = DXGI_ERROR_NOT_FOUND;
        if (ours.LowPart == luid.LowPart && ours.HighPart == luid.HighPart) hr = adapter->QueryInterface(riid, out);
        adapter->Release();
        return hr;
    }

    // IDXGIFactory5
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(DXGI_FEATURE feature, void* data, UINT size) override {
        if (feature == DXGI_FEATURE_PRESENT_ALLOW_TEARING) {
            if (!data || size != sizeof(BOOL)) return E_INVALIDARG;
            *static_cast<BOOL*>(data) = FALSE;  // FIFO only, for now
            return S_OK;
        }
        return E_INVALIDARG;
    }

private:
    // One adapter: the Vulkan device. A new object per call, holding the factory, as DXGI's do.
    HRESULT enum_adapter(UINT index, REFIID riid, void** out) {
        if (!out) return E_INVALIDARG;
        *out = nullptr;
        if (index != 0) return DXGI_ERROR_NOT_FOUND;
        auto* adapter = new Adapter(host_, static_cast<IDXGIFactory5*>(this));
        const HRESULT hr = adapter->QueryInterface(riid, out);
        adapter->Release();
        return hr;
    }

    // D3D12: the "device" is a command queue, and only the layer's own queues can present.
    HRESULT create_swap_chain(IUnknown* device, HWND window, const DXGI_SWAP_CHAIN_DESC1& desc, REFIID riid,
                              void** out) {
        if (!device) return DXGI_ERROR_INVALID_CALL;
        ID3D12CommandQueue* queue = nullptr;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue)))) {
            std::fprintf(stderr, "[rt] dxgi: CreateSwapChain: the device is not an ID3D12CommandQueue\n");
            return DXGI_ERROR_INVALID_CALL;
        }
        leonida::PresentTarget target{host_.get_instance_proc, host_.instance, host_.physical, 0};
        HRESULT hr = DXGI_ERROR_INVALID_CALL;
        if (!rt::d3d12::queue_family(queue, &target.queue_family))
            std::fprintf(stderr, "[rt] dxgi: CreateSwapChain: the queue is not one of the layer's\n");
        else
            hr = create_swap_chain_for_hwnd(*host_.context, target, queue, window, desc, riid, out,
                                            static_cast<IDXGIFactory5*>(this));
        queue->Release();
        return hr;
    }

    Host host_;
    UINT flags_;
    HWND window_ = nullptr;
};

}  // namespace

HRESULT create_factory(const Host& host, UINT flags, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!host.context || !host.get_instance_proc || !host.instance || !host.physical) return E_INVALIDARG;
    auto* factory = new Factory(host, flags);
    const HRESULT hr = factory->QueryInterface(riid, out);
    factory->Release();
    return hr;
}

}  // namespace rt::dxgi
