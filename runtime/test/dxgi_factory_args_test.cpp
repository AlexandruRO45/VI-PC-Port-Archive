// DXGI methods that used to answer E_NOTIMPL for every call: IDXGIFactory::CreateSoftwareAdapter,
// IDXGIFactory2::GetSharedResourceAdapterLuid and CreateSwapChainForComposition on invalid arguments, and
// IDXGIAdapter3::RegisterHardwareContentProtectionTeardownStatusEvent. The expected answers were measured on the
// host dxgi.dll (RTX 4080 SUPER, Intel UHD 770, WARP adapters) and are checked on the layer's factory and, when
// present, on the host factory too. Calls the host accepts that the layer cannot (a real software adapter DLL, a
// composition swap chain) are not compared. No shaders or game data.
// https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgifactory2-createswapchainforcomposition
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_6.h>

#define EXPECT_HR(expr, want) do { \
    const HRESULT got_ = (expr); \
    if (got_ != (want)) { \
        std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, #expr, \
                     static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
        return 1; \
    } \
} while (0)
#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(IDXGIFactory4* factory, const char* who) {
    // CreateSoftwareAdapter: a null module or a null output is DXGI_ERROR_INVALID_CALL.
    HMODULE self = GetModuleHandleW(nullptr);
    EXPECT(self);
    Ref<IDXGIAdapter> software;
    EXPECT_HR(factory->CreateSoftwareAdapter(nullptr, &software.p), DXGI_ERROR_INVALID_CALL);
    EXPECT(software.p == nullptr);
    EXPECT_HR(factory->CreateSoftwareAdapter(self, nullptr), DXGI_ERROR_INVALID_CALL);
    EXPECT_HR(factory->CreateSoftwareAdapter(nullptr, nullptr), DXGI_ERROR_INVALID_CALL);

    // GetSharedResourceAdapterLuid: a null handle or a null LUID.
    Ref<IDXGIFactory2> factory2;
    EXPECT(SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory2.p))));
    LUID luid{};
    EXPECT_HR(factory2->GetSharedResourceAdapterLuid(nullptr, &luid), DXGI_ERROR_INVALID_CALL);
    EXPECT_HR(factory2->GetSharedResourceAdapterLuid(reinterpret_cast<HANDLE>(0x1234), nullptr),
              DXGI_ERROR_INVALID_CALL);

    // CreateSwapChainForComposition: the page says pDevice, pDesc and ppSwapChain cannot be NULL.
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = desc.Height = 64;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    Ref<IDXGISwapChain1> swapchain;
    EXPECT_HR(factory2->CreateSwapChainForComposition(nullptr, &desc, nullptr, &swapchain.p), DXGI_ERROR_INVALID_CALL);
    EXPECT(swapchain.p == nullptr);
    EXPECT_HR(factory2->CreateSwapChainForComposition(nullptr, nullptr, nullptr, &swapchain.p),
              DXGI_ERROR_INVALID_CALL);
    EXPECT_HR(factory2->CreateSwapChainForComposition(factory, &desc, nullptr, nullptr), DXGI_ERROR_INVALID_CALL);

    // RegisterHardwareContentProtectionTeardownStatusEvent: a null event or cookie is DXGI_ERROR_INVALID_CALL,
    // an event gets S_OK and a non-zero cookie, and the cookie unregisters.
    Ref<IDXGIAdapter1> adapter1;
    EXPECT(SUCCEEDED(factory->EnumAdapters1(0, &adapter1.p)));
    Ref<IDXGIAdapter3> adapter;
    EXPECT(SUCCEEDED(adapter1->QueryInterface(IID_PPV_ARGS(&adapter.p))));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(event);
    DWORD cookie = 77, cookie2 = 0;
    EXPECT_HR(adapter->RegisterHardwareContentProtectionTeardownStatusEvent(nullptr, &cookie), DXGI_ERROR_INVALID_CALL);
    EXPECT(cookie == 77);
    EXPECT_HR(adapter->RegisterHardwareContentProtectionTeardownStatusEvent(event, nullptr), DXGI_ERROR_INVALID_CALL);
    EXPECT_HR(adapter->RegisterHardwareContentProtectionTeardownStatusEvent(event, &cookie), S_OK);
    EXPECT(cookie != 0);
    HANDLE event2 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(event2);
    EXPECT_HR(adapter->RegisterHardwareContentProtectionTeardownStatusEvent(event2, &cookie2), S_OK);
    EXPECT(cookie2 != 0 && cookie2 != cookie);
    adapter->UnregisterHardwareContentProtectionTeardownStatus(cookie);
    adapter->UnregisterHardwareContentProtectionTeardownStatus(cookie2);
    EXPECT(WaitForSingleObject(event, 0) == WAIT_TIMEOUT);  // registering never signals
    CloseHandle(event);
    CloseHandle(event2);
    std::printf("%s: factory and adapter argument answers match the host runtime\n", who);
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    Ref<ID3D12Device> device;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
    Ref<IDXGIFactory4> ours;
    if (FAILED(rt::d3d12::create_dxgi_factory(device.p, IID_PPV_ARGS(&ours.p)))) return 1;
    failures += check(ours.p, "layer");
    Ref<IDXGIFactory4> host;
    if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&host.p))))
        failures += check(host.p, "host");
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
