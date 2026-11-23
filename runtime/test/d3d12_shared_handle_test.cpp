// ID3D12Device::CreateSharedHandle, OpenSharedHandle and OpenSharedHandleByName on calls the host D3D12 runtime
// refuses on their arguments. The expected answers were measured on the host runtime (RTX 4080 SUPER, Intel UHD 770,
// WARP) and are checked on the Vulkan-backed device and, when present, on the host devices too. The Vulkan device
// cannot share objects, so only refusals are compared. No shaders or game data.
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createsharedhandle
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)
#define EXPECT_HR(expr, want) do { \
    const HRESULT got_ = (expr); \
    if (got_ != (want)) { \
        std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, #expr, \
                     static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
        return 1; \
    } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    HANDLE shared = nullptr;
    // CreateSharedHandle: a null object or output is E_INVALIDARG (the page says DXGI_ERROR_INVALID_CALL; the
    // host runtime does not).
    EXPECT_HR(device->CreateSharedHandle(nullptr, nullptr, GENERIC_ALL, nullptr, &shared), E_INVALIDARG);
    EXPECT_HR(device->CreateSharedHandle(fence.p, nullptr, GENERIC_ALL, nullptr, nullptr), E_INVALIDARG);
    // OpenSharedHandle: INVALID_HANDLE_VALUE is E_INVALIDARG; null, an unknown value and a handle to an event
    // (not a shared D3D12 object) are E_HANDLE.
    Ref<ID3D12Resource> opened;
    EXPECT_HR(device->OpenSharedHandle(INVALID_HANDLE_VALUE, IID_PPV_ARGS(&opened.p)), E_INVALIDARG);
    EXPECT_HR(device->OpenSharedHandle(nullptr, IID_PPV_ARGS(&opened.p)), E_HANDLE);
    EXPECT_HR(device->OpenSharedHandle(reinterpret_cast<HANDLE>(0x1234), IID_PPV_ARGS(&opened.p)), E_HANDLE);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(event);
    const HRESULT on_event = device->OpenSharedHandle(event, IID_PPV_ARGS(&opened.p));
    CloseHandle(event);
    EXPECT_HR(on_event, E_HANDLE);
    EXPECT(opened.p == nullptr);
    // OpenSharedHandleByName: a name nothing was shared under is E_HANDLE; an access other than the GENERIC_ALL
    // that CreateSharedHandle accepts is E_INVALIDARG.
    HANDLE by_name = nullptr;
    EXPECT_HR(device->OpenSharedHandleByName(L"leonida_shared_handle_test_nonexistent", GENERIC_ALL, &by_name), E_HANDLE);
    EXPECT_HR(device->OpenSharedHandleByName(nullptr, GENERIC_ALL, &by_name), E_HANDLE);
    EXPECT_HR(device->OpenSharedHandleByName(L"leonida_shared_handle_test_nonexistent", 0, &by_name), E_INVALIDARG);
    std::printf("%s: shared-handle refusals match the host runtime\n", who);
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
