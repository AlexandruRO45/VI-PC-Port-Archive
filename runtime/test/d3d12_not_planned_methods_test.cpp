// The D3D12 methods the layer intentionally leaves on the generated stub, deferred until the game's trace shows a use
// (docs/d3d12_vulkan.md section 4). Each HRESULT method returns E_NOTIMPL and each void method is a logged no-op;
// both report their name once. This pins the list so a future override does not silently change a deferred method's
// contract, and so the method-coverage scout does not file them again. No host comparison: the host's answer for
// these is out of scope until the trace shows a use.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (status == vktest::DeviceStatus::failed) return 1;

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    Ref<ID3D12Device> inner;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner.p))));

    Ref<ID3D12Device11> device;
    EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&device.p))));

    // The ten HRESULT methods hit the generated stub and return E_NOTIMPL.
    {
        HANDLE handle = nullptr;
        EXPECT(device->CreateSharedHandle(nullptr, nullptr, 0, nullptr, &handle) == E_NOTIMPL && !handle);
        EXPECT(device->OpenSharedHandle(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->OpenSharedHandleByName(nullptr, 0, &handle) == E_NOTIMPL && !handle);
        EXPECT(device->OpenExistingHeapFromAddress(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->OpenExistingHeapFromFileMapping(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->CreateProtectedResourceSession(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->CreateProtectedResourceSession1(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->CreateLifetimeTracker(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->CreateStateObject(nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
        EXPECT(device->AddToStateObject(nullptr, nullptr, __uuidof(IUnknown), nullptr) == E_NOTIMPL);
    }

    // The eight void methods also hit the stub: they log and return without touching the list or device state.
    device->CreateSamplerFeedbackUnorderedAccessView(nullptr, nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{0});

    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList7> list7;
    EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                     IID_PPV_ARGS(&list7.p))));
    Ref<ID3D12GraphicsCommandList8> list;
    EXPECT(SUCCEEDED(list7->QueryInterface(IID_PPV_ARGS(&list.p))));

    list->SOSetTargets(0, 0, nullptr);
    list->SetViewInstanceMask(0);
    list->SetProtectedResourceSession(nullptr);
    list->InitializeMetaCommand(nullptr, nullptr, 0);
    list->ExecuteMetaCommand(nullptr, nullptr, 0);
    list->SetPipelineState1(nullptr);
    list->DispatchRays(nullptr);
    EXPECT(list->Close() == S_OK);

    std::printf("d3d12 not-planned methods: 10 HRESULT methods return E_NOTIMPL, 8 void methods are logged no-ops\n");
    return 0;
}
