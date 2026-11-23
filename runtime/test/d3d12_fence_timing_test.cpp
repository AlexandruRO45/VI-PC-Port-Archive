// D3D-30 on the Vulkan device, through the Xbox-ABI proxy: SetEventOnMultipleFenceCompletion (ALL and ANY, with an
// event and blocking), fence rewinds with CPU and queue signals, queue GetClockCalibration on both
// paths (calibrated timestamps on the standalone device when it exposes both clock domains, the submitted timestamp
// query otherwise), GetAdapterLuid (must name a DXGI adapter) and GetCustomHeapProperties (compared with the host PC
// runtime's answer when the host adapter is NUMA, as the layer reports itself). No shaders or game data.
// LEONIDA_VK_VALIDATION=1 runs the test-made device under VK_LAYER_KHRONOS_validation; any validation error fails
// the test.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

struct Event {
    explicit Event(bool manual_reset = false) : h(CreateEventW(nullptr, manual_reset, FALSE, nullptr)) {}
    HANDLE h;
    ~Event() { if (h) CloseHandle(h); }
    bool set_within(DWORD ms) const { return WaitForSingleObject(h, ms) == WAIT_OBJECT_0; }
};

UINT64 qpc() {
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    return UINT64(now.QuadPart);
}

// GetClockCalibration twice, 50 ms apart: the CPU stamps bracket the call, and both clocks advance at the same rate.
int check_calibration(ID3D12CommandQueue* queue, double gpu_frequency, const char* path) {
    LARGE_INTEGER qpf{};
    QueryPerformanceFrequency(&qpf);
    UINT64 gpu0 = 0, cpu0 = 0, gpu1 = 0, cpu1 = 0;
    const UINT64 before = qpc();
    EXPECT(SUCCEEDED(queue->GetClockCalibration(&gpu0, &cpu0)));
    const UINT64 after = qpc();
    EXPECT(before <= cpu0 && cpu0 <= after);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const UINT64 before1 = qpc();
    EXPECT(SUCCEEDED(queue->GetClockCalibration(&gpu1, &cpu1)));
    const UINT64 after1 = qpc();
    EXPECT(before1 <= cpu1 && cpu1 <= after1);
    EXPECT(gpu1 > gpu0 && cpu1 > cpu0);
    const double gpu_s = double(gpu1 - gpu0) / gpu_frequency, cpu_s = double(cpu1 - cpu0) / double(qpf.QuadPart);
    std::printf("%s: GPU %.6f s, CPU %.6f s between two calibrations (%.1f us apart)\n", path, gpu_s, cpu_s,
                std::fabs(gpu_s - cpu_s) * 1e6);
    // Each calibration pairs a GPU stamp with a CPU stamp taken during the call, so neither is off by more than
    // that call's own duration: microseconds with calibrated timestamps, a submission round trip for the fallback,
    // which a busy GPU can stretch past a few ms. Allow the two calls' durations, and at least 2 ms.
    const double calls_s = double((after - before) + (after1 - before1)) / double(qpf.QuadPart);
    EXPECT(std::fabs(gpu_s - cpu_s) < (calls_s > 0.002 ? calls_s : 0.002));
    EXPECT(queue->GetClockCalibration(nullptr, &cpu0) == E_INVALIDARG);
    EXPECT(queue->GetClockCalibration(&gpu0, nullptr) == E_INVALIDARG);
    return 0;
}

int check_multiple_fences(ID3D12Device* device, ID3D12CommandQueue* queue) {
    Ref<ID3D12Device1> device1;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
    Ref<ID3D12Fence> a, b;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&a.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&b.p))));
    ID3D12Fence* both[] = {a.p, b.p};
    constexpr auto ALL = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ANY = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;

    // ALL with an event: one fence is not enough.
    {
        const UINT64 values[] = {1, 1};
        Event event;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ALL, event.h)));
        EXPECT(SUCCEEDED(a->Signal(1)));
        EXPECT(!event.set_within(50));
        EXPECT(SUCCEEDED(queue->Signal(b.p, 1)));  // from the GPU
        EXPECT(event.set_within(5000));
    }
    // ANY with an event: the first fence sets it, and the other never reaching its value doesn't matter.
    {
        const UINT64 values[] = {2, 100};
        Event event;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ANY, event.h)));
        EXPECT(!event.set_within(50));
        EXPECT(SUCCEEDED(queue->Signal(a.p, 2)));
        EXPECT(event.set_within(5000));
        EXPECT(!event.set_within(50));  // set once
    }
    // Values already reached: set before the call returns.
    {
        const UINT64 values[] = {2, 1};
        Event all, any;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ALL, all.h)));
        EXPECT(all.set_within(0));
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ANY, any.h)));
        EXPECT(any.set_within(0));
    }
    // A null event blocks: ALL until both fences are signalled from another thread, ANY until one is.
    {
        const UINT64 values[] = {3, 3};
        std::thread signaller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            a->Signal(3);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            b->Signal(3);
        });
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ALL, nullptr)));
        const bool reached = a->GetCompletedValue() >= 3 && b->GetCompletedValue() >= 3;
        signaller.join();
        EXPECT(reached);
    }
    {
        const UINT64 values[] = {4, 4};
        std::thread signaller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            b->Signal(4);
        });
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(both, values, 2, ANY, nullptr)));
        const bool reached = b->GetCompletedValue() >= 4;
        signaller.join();
        EXPECT(reached && a->GetCompletedValue() == 3);
    }
    // A fence released while an event waits on it sets the event, as on the host PC runtime.
    {
        Ref<ID3D12Fence> c;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.p))));
        ID3D12Fence* one[] = {c.p};
        const UINT64 values[] = {1};
        Event event;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(one, values, 1, ALL, event.h)));
        c.reset();
        EXPECT(event.set_within(2000));
    }
    // Bad arguments.
    {
        const UINT64 values[] = {1, 1};
        ID3D12Fence* with_null[] = {a.p, nullptr};
        EXPECT(device1->SetEventOnMultipleFenceCompletion(both, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAGS(2),
                                                           nullptr) == E_INVALIDARG);
        EXPECT(device1->SetEventOnMultipleFenceCompletion(with_null, values, 2, ALL, nullptr) == E_INVALIDARG);
        EXPECT(device1->SetEventOnMultipleFenceCompletion(nullptr, values, 2, ALL, nullptr) == E_INVALIDARG);
        Event event;  // no fences: refused and the event left alone, as on the host PC runtime
        EXPECT(device1->SetEventOnMultipleFenceCompletion(nullptr, nullptr, 0, ALL, event.h) == E_INVALIDARG);
        EXPECT(!event.set_within(0));
    }
    return 0;
}

int check_cpu_rewind(ID3D12Device* device, const char* who) {
    Event old_eight(true), old_twelve, new_eight;
    EXPECT(old_eight.h && old_twelve.h && new_eight.h);
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(10, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(8, old_eight.h)));
    EXPECT(old_eight.set_within(0));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(12, old_twelve.h)));
    EXPECT(SUCCEEDED(fence->Signal(4)));
    const UINT64 rewound = fence->GetCompletedValue();
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(8, new_eight.h)));
    const bool old_retained = old_eight.set_within(0);
    const bool old_twelve_early = old_twelve.set_within(0);
    const bool new_eight_early = new_eight.set_within(50);
    EXPECT(SUCCEEDED(fence->Signal(12)));
    const bool old_twelve_reached = old_twelve_early || old_twelve.set_within(5000);
    const bool new_eight_reached = new_eight_early || new_eight.set_within(5000);
    std::printf("%s: CPU fence 10 -> %llu -> %llu, old event retained %d, new event set early %d\n", who,
                static_cast<unsigned long long>(rewound),
                static_cast<unsigned long long>(fence->GetCompletedValue()), old_retained, new_eight_early);
    EXPECT(rewound == 4 && fence->GetCompletedValue() == 12);
    EXPECT(old_retained && !old_twelve_early && !new_eight_early && old_twelve_reached && new_eight_reached);
    return 0;
}

// Once a queue has a Wait pending, a failure must leave the process without closing live events or destroying
// the blocked queue. The queue's gate is opened before checking the observed results in the normal path.
void require_progress(bool ok, const char* who, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "%s: %s\n", who, what);
    std::fflush(stderr);
    std::_Exit(1);
}

// The host applies CPU signals immediately, even while an earlier queue Signal is held at a gate. When the
// queue reaches its lower value it rewinds the fence, but the event set by the CPU signal stays set.
int check_pending_gpu_rewind(ID3D12Device* device, ID3D12CommandQueue* queue, const char* who) {
    Event old_twenty(true), new_twenty, queue_done;
    EXPECT(old_twenty.h && new_twenty.h && queue_done.h);
    Ref<ID3D12Fence> gate, fence, done;
    for (Ref<ID3D12Fence>* f : {&gate, &fence, &done})
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f->p))));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(20, old_twenty.h)));
    EXPECT(SUCCEEDED(done->SetEventOnCompletion(1, queue_done.h)));

    require_progress(SUCCEEDED(queue->Wait(gate.p, 1)), who, "queue Wait failed");
    require_progress(SUCCEEDED(queue->Signal(fence.p, 10)), who, "queue Signal(10) failed");
    require_progress(SUCCEEDED(queue->Signal(done.p, 1)), who, "queue completion Signal failed");
    require_progress(SUCCEEDED(fence->Signal(15)), who, "CPU Signal(15) failed");
    const UINT64 cpu_first = fence->GetCompletedValue();
    require_progress(SUCCEEDED(fence->Signal(20)), who, "CPU Signal(20) failed");
    const UINT64 cpu_last = fence->GetCompletedValue();
    const bool set_during_gate = old_twenty.set_within(50);

    require_progress(SUCCEEDED(gate->Signal(1)), who, "opening the queue gate failed");
    require_progress(queue_done.set_within(5000), who, "queue did not complete after opening the gate");
    const UINT64 gpu = fence->GetCompletedValue();
    const bool old_retained = old_twenty.set_within(0);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(20, new_twenty.h)));
    const bool new_early = new_twenty.set_within(50);
    EXPECT(SUCCEEDED(fence->Signal(25)));
    const bool new_reached = new_early || new_twenty.set_within(5000);
    if (!old_retained) EXPECT(old_twenty.set_within(5000));
    std::printf("%s: CPU signals %llu -> %llu, queued signal then %llu, old event retained %d, new event set early %d\n",
                who, static_cast<unsigned long long>(cpu_first), static_cast<unsigned long long>(cpu_last),
                static_cast<unsigned long long>(gpu), old_retained, new_early);
    EXPECT(cpu_first == 15 && cpu_last == 20 && gpu == 10 && fence->GetCompletedValue() == 25);
    EXPECT(set_during_gate && old_retained && !new_early && new_reached);
    return 0;
}

int check_queue_rewind(ID3D12Device* device, ID3D12CommandQueue* copy, ID3D12CommandQueue* direct, const char* who) {
    Event copy_done, direct_done;
    EXPECT(copy_done.h && direct_done.h);
    Ref<ID3D12Fence> fence, copy_complete, direct_complete;
    EXPECT(SUCCEEDED(device->CreateFence(10, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_complete.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&direct_complete.p))));
    EXPECT(SUCCEEDED(copy_complete->SetEventOnCompletion(1, copy_done.h)));
    EXPECT(SUCCEEDED(direct_complete->SetEventOnCompletion(1, direct_done.h)));
    require_progress(SUCCEEDED(copy->Signal(fence.p, 4)), who, "COPY Signal(4) failed");
    require_progress(SUCCEEDED(copy->Signal(copy_complete.p, 1)), who, "COPY completion Signal failed");
    require_progress(copy_done.set_within(5000), who, "COPY did not complete");
    const UINT64 rewound = fence->GetCompletedValue();

    require_progress(SUCCEEDED(direct->Wait(fence.p, 8)), who, "DIRECT Wait(8) failed");
    require_progress(SUCCEEDED(direct->Signal(direct_complete.p, 1)), who, "DIRECT completion Signal failed");
    const bool direct_early = direct_done.set_within(50);
    require_progress(SUCCEEDED(fence->Signal(12)), who, "CPU Signal(12) failed");
    require_progress(direct_early || direct_done.set_within(5000), who, "DIRECT did not complete after Signal(12)");
    std::printf("%s: COPY fence value %llu, DIRECT after Wait(8) completed early %d\n", who,
                static_cast<unsigned long long>(rewound), direct_early);
    EXPECT(rewound == 4 && !direct_early);
    return 0;
}

int check_standalone_rewinds(ID3D12Device* device, ID3D12CommandQueue* direct, const char* who) {
    Ref<ID3D12CommandQueue> copy;
    const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_COPY};
    const bool has_copy = SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&copy.p)));
    int failures = check_cpu_rewind(device, who);
    failures += check_pending_gpu_rewind(device, has_copy ? copy.p : direct, who);
    if (has_copy) {
        failures += check_queue_rewind(device, copy.p, direct, who);
    } else {
        std::printf("%s: COPY queue unavailable for the cross-queue rewind case\n", who);
    }
    return failures;
}

int check_luid(ID3D12Device* device) {
    const LUID luid = device->GetAdapterLuid();
    EXPECT(luid.LowPart || luid.HighPart);
    Ref<IDXGIFactory4> factory;
    EXPECT(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
    Ref<IDXGIAdapter1> adapter;
    EXPECT(SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p))));
    DXGI_ADAPTER_DESC1 desc{};
    EXPECT(SUCCEEDED(adapter->GetDesc1(&desc)));
    std::printf("GetAdapterLuid: %08lx:%08lx is the DXGI adapter \"%ls\"\n", luid.HighPart, luid.LowPart,
                desc.Description);
    return 0;
}

int check_custom_heap_properties(ID3D12Device* device) {
    struct Want {
        D3D12_HEAP_TYPE type;
        D3D12_CPU_PAGE_PROPERTY page;
        D3D12_MEMORY_POOL pool;
    };
    const Want table[] = {
        {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L1},
        {D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0},
        {D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0},
    };
    for (const Want& want : table) {
        for (UINT node_mask : {0u, 1u}) {
            const D3D12_HEAP_PROPERTIES got = device->GetCustomHeapProperties(node_mask, want.type);
            EXPECT(got.Type == D3D12_HEAP_TYPE_CUSTOM && got.CPUPageProperty == want.page &&
                   got.MemoryPoolPreference == want.pool && got.CreationNodeMask == 1 && got.VisibleNodeMask == 1);
        }
    }
    const D3D12_HEAP_PROPERTIES custom = device->GetCustomHeapProperties(0, D3D12_HEAP_TYPE_CUSTOM);
    EXPECT(custom.Type == D3D12_HEAP_TYPE(0) && custom.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_UNKNOWN &&
           custom.MemoryPoolPreference == D3D12_MEMORY_POOL_UNKNOWN && custom.CreationNodeMask == 1 &&
           custom.VisibleNodeMask == 1);

    // The host runtime on a NUMA adapter (the answer the layer gives in D3D12_FEATURE_ARCHITECTURE) must agree.
    Ref<ID3D12Device> host;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        std::printf("GetCustomHeapProperties: no host D3D12 device to compare with\n");
        return 0;
    }
    D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
    EXPECT(SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture))));
    if (architecture.UMA) {
        std::printf("GetCustomHeapProperties: the host adapter is UMA; not compared\n");
        return 0;
    }
    for (const Want& want : table) {
        const D3D12_HEAP_PROPERTIES ours = device->GetCustomHeapProperties(0, want.type);
        const D3D12_HEAP_PROPERTIES theirs = host->GetCustomHeapProperties(0, want.type);
        if (std::memcmp(&ours, &theirs, sizeof(ours)) != 0) {
            std::fprintf(stderr, "heap type %d: ours %d/%d/%d, host %d/%d/%d\n", want.type, ours.Type,
                         ours.CPUPageProperty, ours.MemoryPoolPreference, theirs.Type, theirs.CPUPageProperty,
                         theirs.MemoryPoolPreference);
            return 1;
        }
    }
    // What has no custom equivalent: an all-zero description with node 0 in both masks on the host runtime.
    const std::pair<UINT, D3D12_HEAP_TYPE> invalid[] = {
        {0, D3D12_HEAP_TYPE_CUSTOM}, {1, D3D12_HEAP_TYPE_CUSTOM}, {0, D3D12_HEAP_TYPE(99)},
        {2, D3D12_HEAP_TYPE_UPLOAD}};
    for (const auto& [node_mask, type] : invalid) {
        const D3D12_HEAP_PROPERTIES ours = device->GetCustomHeapProperties(node_mask, type);
        const D3D12_HEAP_PROPERTIES theirs = host->GetCustomHeapProperties(node_mask, type);
        if (std::memcmp(&ours, &theirs, sizeof(ours)) != 0) {
            std::fprintf(stderr, "node mask %u, heap type %d: ours %d/%d/%d/%u/%u, host %d/%d/%d/%u/%u\n", node_mask,
                         type, ours.Type, ours.CPUPageProperty, ours.MemoryPoolPreference, ours.CreationNodeMask,
                         ours.VisibleNodeMask, theirs.Type, theirs.CPUPageProperty, theirs.MemoryPoolPreference,
                         theirs.CreationNodeMask, theirs.VisibleNodeMask);
            return 1;
        }
    }
    std::printf("GetCustomHeapProperties: DEFAULT, UPLOAD, READBACK, CUSTOM, an unknown type and node mask 2 match "
                "the host runtime\n");
    return 0;
}

// The proxy over a Vulkan device, the way the game sees it.
HRESULT wrap(ID3D12Device* inner, Ref<ID3D12Device>& device) {
    const HRESULT hr = rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                         reinterpret_cast<void**>(&device.p));
    if (SUCCEEDED(hr)) rt::d3d12::set_device_outer(inner, device.p);
    return hr;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    const double gpu_frequency = 1e9 / double(properties.limits.timestampPeriod);  // GetTimestampFrequency's answer

    int rewind_failures = 0;
    // A device without calibrated timestamps (the test device enables no extensions): the submitted fallback.
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(wrap(inner, device)));
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(!rt::d3d12::queue_uses_calibrated_timestamps(queue.p));
        EXPECT(check_calibration(queue.p, gpu_frequency, "fallback (submitted timestamp query)") == 0);
        EXPECT(check_multiple_fences(device.p, queue.p) == 0);
        rewind_failures += check_cpu_rewind(device.p, "Xbox-ABI proxy");
        rewind_failures += check_pending_gpu_rewind(device.p, queue.p, "Xbox-ABI proxy");
        EXPECT(check_luid(device.p) == 0);
        EXPECT(check_custom_heap_properties(device.p) == 0);
    }

    // The standalone device enables VK_KHR/EXT_calibrated_timestamps where the GPU has it.
    {
        Ref<ID3D12Device> standalone;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(standalone->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        const auto extensions_of = d.instance_fn<PFN_vkEnumerateDeviceExtensionProperties>(
            "vkEnumerateDeviceExtensionProperties");
        std::uint32_t count = 0;
        EXPECT(extensions_of(d.physical, nullptr, &count, nullptr) == VK_SUCCESS);
        std::vector<VkExtensionProperties> extensions(count);
        EXPECT(extensions_of(d.physical, nullptr, &count, extensions.data()) == VK_SUCCESS);
        extensions.resize(count);
        const auto has_extension = [&](const char* name) {
            return std::any_of(extensions.begin(), extensions.end(),
                               [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
        };
        const bool khr = has_extension(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
        const bool ext = has_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
        bool device_clock = false, qpc_clock = false;
        if (khr || ext) {
            // A driver can expose the extension without a Vulkan time domain for Win32 QPC (e.g. llvmpipe in Wine).
            const auto domains_of = d.instance_fn<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                khr ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR" : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
            EXPECT(domains_of);
            EXPECT(domains_of(d.physical, &count, nullptr) == VK_SUCCESS);
            std::vector<VkTimeDomainKHR> domains(count);
            EXPECT(domains_of(d.physical, &count, domains.data()) == VK_SUCCESS);
            domains.resize(count);
            device_clock = std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_DEVICE_KHR) != domains.end();
            qpc_clock = std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR) !=
                        domains.end();
        }
        const bool calibrated = rt::d3d12::queue_uses_calibrated_timestamps(queue.p);
        std::printf("calibrated timestamps: extension %s, DEVICE domain %s, QPC domain %s, queue capability %s\n",
                    khr || ext ? "present" : "missing", device_clock ? "present" : "missing",
                    qpc_clock ? "present" : "missing", calibrated ? "enabled" : "disabled");
        EXPECT(calibrated == (device_clock && qpc_clock));
        const char* clock_path = calibrated ? "calibration-capable queue" : "standalone fallback";
        EXPECT(check_calibration(queue.p, gpu_frequency, clock_path) == 0);
        EXPECT(check_multiple_fences(standalone.p, queue.p) == 0);
        rewind_failures += check_standalone_rewinds(standalone.p, queue.p, "standalone Vulkan");
        EXPECT(check_luid(standalone.p) == 0);
    }

    Ref<IDXGIFactory4> factory;
    if (hostwine::host_is_wine()) {
        std::puts("host fence comparisons skipped under Wine");
    } else if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        std::puts("host fence comparisons skipped because no DXGI factory is available");
    } else {
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string name;
            if (!hostadapter::reference(factory.p, d, warp, &adapter.p, name) ||
                FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
                std::printf("%s D3D12 fence comparison unavailable\n", warp ? "WARP" : "native");
                continue;
            }
            Ref<ID3D12CommandQueue> direct;
            const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
            EXPECT(SUCCEEDED(host->CreateCommandQueue(&desc, IID_PPV_ARGS(&direct.p))));
            rewind_failures += check_standalone_rewinds(host.p, direct.p, name.c_str());
        }
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    if (rewind_failures) return 1;
    std::printf("fences and timing: multiple-fence waits, CPU and queue rewinds, clock calibration, adapter LUID and "
                "custom heap properties all checked\n");
    return 0;
}
