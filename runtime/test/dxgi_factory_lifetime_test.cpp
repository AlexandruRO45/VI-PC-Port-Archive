// The DXGI factory made from a Vulkan-backed D3D12 device (rt::d3d12::create_dxgi_factory) keeps that device alive.
// The factory, its adapters and its swap chains use the device's VulkanContext, VkInstance, VkPhysicalDevice and
// VkQueue; a standalone device owns all of them and destroys them with itself. An app may release its device before
// the factory, so:
//   - creating the factory takes exactly one reference on the device, and releasing the factory gives it back;
//   - an adapter keeps the factory (and so the device) alive after the factory itself is released;
//   - after the app's last reference to the device is gone, the factory still enumerates its adapter (which reads
//     the physical device's properties), finds it by LUID and probes tearing support (a surface on the instance),
//     all on live Vulkan handles.
// Before the fix the factory held no reference, so the last step read a destroyed instance and device.
// Exits 77 when no Vulkan device can be made. No shaders, game data or Xbox DLLs.
#include "../shims/d3d12_com.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

// The object's reference count, without changing it.
ULONG ref_count(IUnknown* object) {
    object->AddRef();
    return object->Release();
}

// The factory answers its adapter and tearing queries; every call reaches the device's Vulkan handles.
int use_factory(IDXGIFactory5* factory) {
    Ref<IDXGIAdapter1> adapter;
    EXPECT(factory->EnumAdapters1(0, &adapter.p) == S_OK && adapter.p);
    DXGI_ADAPTER_DESC1 desc{};
    EXPECT(adapter->GetDesc1(&desc) == S_OK);
    Ref<IDXGIAdapter> by_luid;
    EXPECT(factory->EnumAdapterByLuid(desc.AdapterLuid, IID_PPV_ARGS(&by_luid.p)) == S_OK && by_luid.p);
    BOOL tearing = FALSE;
    EXPECT(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing)) == S_OK);
    return 0;
}

int factory_holds_one_reference() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) return 77;
    const ULONG before = ref_count(device.p);

    Ref<IDXGIFactory5> factory;
    EXPECT(SUCCEEDED(rt::d3d12::create_dxgi_factory(device.p, IID_PPV_ARGS(&factory.p))) && factory.p);
    EXPECT(ref_count(device.p) == before + 1);

    // A second factory takes its own reference.
    {
        Ref<IDXGIFactory5> second;
        EXPECT(SUCCEEDED(rt::d3d12::create_dxgi_factory(device.p, IID_PPV_ARGS(&second.p))) && second.p);
        EXPECT(ref_count(device.p) == before + 2);
    }
    EXPECT(ref_count(device.p) == before + 1);

    factory.reset();
    EXPECT(ref_count(device.p) == before);
    return 0;
}

int factory_outlives_device() {
    Ref<IDXGIFactory5> factory;
    {
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) return 77;
        EXPECT(SUCCEEDED(rt::d3d12::create_dxgi_factory(device.p, IID_PPV_ARGS(&factory.p))) && factory.p);
        // No use before the release: the tearing probe runs once per factory, and it must run after it.
        // The app's last reference: the factory's is the only one left.
        ID3D12Device* raw = device.p;
        device.p = nullptr;
        EXPECT(raw->Release() == 1);
    }
    EXPECT(use_factory(factory.p) == 0);
    return 0;
}

int adapter_outlives_device_and_factory() {
    Ref<IDXGIAdapter1> adapter;
    {
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) return 77;
        Ref<IDXGIFactory5> factory;
        EXPECT(SUCCEEDED(rt::d3d12::create_dxgi_factory(device.p, IID_PPV_ARGS(&factory.p))) && factory.p);
        EXPECT(factory->EnumAdapters1(0, &adapter.p) == S_OK && adapter.p);
    }
    DXGI_ADAPTER_DESC1 desc{};
    EXPECT(adapter->GetDesc1(&desc) == S_OK);
    Ref<IDXGIFactory5> parent;
    EXPECT(adapter->GetParent(IID_PPV_ARGS(&parent.p)) == S_OK && parent.p);
    EXPECT(use_factory(parent.p) == 0);
    return 0;
}

}  // namespace

int main() {
    struct Case {
        const char* name;
        int (*run)();
    };
    constexpr Case cases[] = {
        {"factory_holds_one_reference", factory_holds_one_reference},
        {"factory_outlives_device", factory_outlives_device},
        {"adapter_outlives_device_and_factory", adapter_outlives_device_and_factory},
    };
    int failed = 0;
    for (const Case& c : cases) {
        const int result = c.run();
        if (result == 77) {
            std::printf("SKIP: no Vulkan device\n");
            return 77;
        }
        std::printf("%s: %s\n", c.name, result ? "FAIL" : "ok");
        failed += result ? 1 : 0;
    }
    return failed ? 1 : 0;
}
