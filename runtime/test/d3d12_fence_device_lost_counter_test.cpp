// D3D12 fence device-loss reporting when vkGetSemaphoreCounterValue first discovers a lost device.
// The injected VkResult is synthetic and never forwarded to the physical Vulkan driver. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
    bool ready(DWORD ms) const { return handle && WaitForSingleObject(handle, ms) == WAIT_OBJECT_0; }
};

namespace driver {
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkGetSemaphoreCounterValue real_counter = nullptr;
PFN_vkWaitSemaphores real_wait = nullptr;
PFN_vkSignalSemaphore real_signal = nullptr;
std::atomic<VkResult> fail_next_counter{VK_SUCCESS};
std::atomic<VkResult> fail_next_wait{VK_SUCCESS};
std::atomic<VkResult> fail_next_signal{VK_SUCCESS};
std::atomic<unsigned> failed_count{0};
std::atomic<unsigned> failed_wait_count{0};
std::atomic<unsigned> foreground_signals{0};
DWORD test_thread = 0;

VKAPI_ATTR VkResult VKAPI_CALL get_counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    // A pending fence may also be polled by the dispatcher thread. Only the explicitly armed
    // synchronous GetCompletedValue call gets the test failure, so it is deterministic.
    if (GetCurrentThreadId() == test_thread) {
        const VkResult injected = fail_next_counter.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            // Even a bogus output counter from a failed query may not be treated as completion.
            *value = 999;
            failed_count.fetch_add(1);
            return injected;
        }
    }
    return real_counter(device, semaphore, value);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info,
                                               std::uint64_t timeout) {
    if (GetCurrentThreadId() == test_thread) {
        const VkResult injected = fail_next_wait.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            failed_wait_count.fetch_add(1);
            return injected;
        }
    }
    return real_wait(device, info, timeout);
}

VKAPI_ATTR VkResult VKAPI_CALL signal_semaphore(VkDevice device, const VkSemaphoreSignalInfo* info) {
    if (GetCurrentThreadId() == test_thread) {
        foreground_signals.fetch_add(1);
        const VkResult injected = fail_next_signal.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) return injected;
    }
    return real_signal(device, info);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_counter);
    if (std::strcmp(name, "vkWaitSemaphores") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (std::strcmp(name, "vkSignalSemaphore") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(signal_semaphore);
    return real_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_instance(instance, name);
}
}  // namespace driver

// RemoveDevice is a logical removal, not a physical GPU failure. Both fences and every registered
// event must complete, including waits registered after removal; a second removal is harmless.
bool remove_device_notifies(ID3D12Device* device, const char* label, bool instrument_signal = false) {
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Device5> device5;
    Event before[3], after[3];  // outlive the fences and their pending callbacks on every exit path
    Ref<ID3D12Fence> a, b;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p))) ||
        FAILED(device->QueryInterface(IID_PPV_ARGS(&device5.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&a.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&b.p)))) return false;
    HANDLE pre[] = {before[0].handle, before[1].handle, before[2].handle};
    HANDLE post[] = {after[0].handle, after[1].handle, after[2].handle};
    for (HANDLE h : pre) if (!h) return false;
    for (HANDLE h : post) if (!h) return false;
    ID3D12Fence* fences[] = {a.p, b.p};
    const UINT64 values[] = {1, 1};
    const auto subscribe = [&](HANDLE* events) {
        return SUCCEEDED(a->SetEventOnCompletion(1, events[0])) &&
               SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2,
                                                                    D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, events[1])) &&
               SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2,
                                                                    D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, events[2]));
    };
    if (device->GetDeviceRemovedReason() != S_OK || a->GetCompletedValue() != 0 || b->GetCompletedValue() != 0 ||
        !subscribe(pre)) return false;
    for (HANDLE h : pre) if (WaitForSingleObject(h, 0) != WAIT_TIMEOUT) return false;
    device5->RemoveDevice();
    const HRESULT reason = device->GetDeviceRemovedReason();
    const UINT64 first = a->GetCompletedValue(), second = b->GetCompletedValue();
    const DWORD pre_status = WaitForMultipleObjects(3, pre, TRUE, 1000);
    const bool post_registered = subscribe(post);
    const DWORD post_status = WaitForMultipleObjects(3, post, TRUE, 1000);
    // WARP accepts signals after removal without lowering the completed value. The calling thread must not
    // signal the Vulkan semaphore once its logical device is lost, even for an increasing value.
    const unsigned signals_before = driver::foreground_signals.load();
    const HRESULT signal1 = a->Signal(1);
    const UINT64 after_signal1 = a->GetCompletedValue();
    const HRESULT signal0 = a->Signal(0);
    const UINT64 after_signal0 = a->GetCompletedValue();
    const HRESULT signal_again = a->Signal(42);
    const UINT64 after_signal_again = a->GetCompletedValue();
    const bool signal_ok = signal1 == S_OK && signal0 == S_OK && signal_again == S_OK &&
                           after_signal1 == UINT64_MAX && after_signal0 == UINT64_MAX &&
                           after_signal_again == UINT64_MAX &&
                           (!instrument_signal || driver::foreground_signals.load() == signals_before);
    std::printf("%s after removal Signal(1/0/42): %08lx/%08lx/%08lx, no foreground Vulkan signal=%s: %s\n", label,
                static_cast<unsigned long>(signal1), static_cast<unsigned long>(signal0),
                static_cast<unsigned long>(signal_again),
                !instrument_signal ? "not checked" : driver::foreground_signals.load() == signals_before ? "yes" : "no",
                signal_ok ? "PASS" : "FAIL");
    device5->RemoveDevice();
    const bool repeated = device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED &&
                          a->GetCompletedValue() == UINT64_MAX && b->GetCompletedValue() == UINT64_MAX;
    const bool ok = reason == DXGI_ERROR_DEVICE_REMOVED && first == UINT64_MAX && second == UINT64_MAX &&
                    pre_status == WAIT_OBJECT_0 && post_registered && post_status == WAIT_OBJECT_0 && repeated && signal_ok;
    std::printf("%s RemoveDevice: reason=%08lx fences=%llu/%llu before=%lu after=%lu repeated=%d: %s\n", label,
                static_cast<unsigned long>(reason), static_cast<unsigned long long>(first),
                static_cast<unsigned long long>(second), pre_status, post_status, int(repeated), ok ? "PASS" : "FAIL");
    return ok;
}

int native_warp_control() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> device;
    Ref<ID3D12Fence> fence;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p))))
        return 77;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;
    Event event;
    if (!event.handle || fence->GetCompletedValue() != 0 ||
        FAILED(fence->SetEventOnCompletion(2, event.handle)) || event.ready(0) ||
        FAILED(fence->Signal(2)) || !event.ready(2000) ||
        fence->GetCompletedValue() != 2 || device->GetDeviceRemovedReason() != S_OK)
        return 1;
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Fence> second;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p))))
        return 1;
    ID3D12Fence* const fences[] = {fence.p, second.p};
    const UINT64 values[] = {2, 3};
    if (FAILED(second->Signal(3)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, nullptr)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, nullptr)))
        return 1;
    std::printf("native WARP: normal fence query/signal/event completed value 2 and two-fence ALL/ANY blocking waits\n");
    return remove_device_notifies(device.p, "native WARP") ? 0 : 1;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int native_result = native_warp_control();
    if (native_result) {
        std::fprintf(stderr, "%s: native WARP %s\n", native_result == 77 ? "SKIP" : "FAIL",
                     native_result == 77 ? "unavailable" : "fence control failed");
        return native_result;
    }
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    driver::real_instance = vk.instance_proc;
    driver::real_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!driver::real_device) return 1;
    driver::real_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        driver::real_device(vk.device, "vkGetSemaphoreCounterValue"));
    if (!driver::real_counter) return 1;
    driver::real_wait = reinterpret_cast<PFN_vkWaitSemaphores>(
        driver::real_device(vk.device, "vkWaitSemaphores"));
    if (!driver::real_wait) return 1;
    driver::real_signal = reinterpret_cast<PFN_vkSignalSemaphore>(
        driver::real_device(vk.device, "vkSignalSemaphore"));
    if (!driver::real_signal) return 1;
    driver::test_thread = GetCurrentThreadId();

    Event pending;
    if (!pending.handle) return 1;
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            fence->GetCompletedValue() != 0 || device->GetDeviceRemovedReason() != S_OK)
            return 1;
        if (FAILED(fence->SetEventOnCompletion(7, pending.handle)) || pending.ready(0)) return 1;

        driver::fail_next_counter.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        const UINT64 ignored = fence->GetCompletedValue();
        const HRESULT still_alive = device->GetDeviceRemovedReason();
        const bool false_completion = pending.ready(0);
        std::printf("Leonida Vulkan: injected VK_ERROR_OUT_OF_HOST_MEMORY with bogus counter 999: "
                    "value=%016llx, device_removed=%08x, pending_event=%d\n",
                    static_cast<unsigned long long>(ignored), static_cast<unsigned>(still_alive),
                    int(false_completion));
        if (ignored != 0 || still_alive != S_OK || false_completion || driver::failed_count.load() != 1)
            ++failures;

        driver::fail_next_counter.store(VK_ERROR_DEVICE_LOST);
        const UINT64 first = fence->GetCompletedValue();
        const HRESULT removed = device->GetDeviceRemovedReason();
        const bool waiter_released = pending.ready(1500);
        const unsigned faults = driver::failed_count.load();
        std::printf("Leonida Vulkan: injected VK_ERROR_DEVICE_LOST on first GetCompletedValue: "
                    "value=%016llx, device_removed=%08x, pending_event=%d, interceptions=%u\n",
                    static_cast<unsigned long long>(first), static_cast<unsigned>(removed),
                    int(waiter_released), faults);
        if (first != UINT64_MAX || removed != DXGI_ERROR_DEVICE_REMOVED ||
            !waiter_released || faults != 2 || driver::fail_next_counter.load() != VK_SUCCESS)
            ++failures;
        // COM releases the fence before its externally owned VkDevice; on failure the cancel
        // path still drains the pending event rather than leaving a dispatcher thread blocked.
    }
    // A blocking host wait can discover device loss after a successful counter query.
    // Unlike an asynchronous wait registered with a HANDLE, it directly invokes vkWaitSemaphores.
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            fence->GetCompletedValue() != 0 || device->GetDeviceRemovedReason() != S_OK)
            return 1;
        driver::fail_next_wait.store(VK_ERROR_DEVICE_LOST);
        const HRESULT blocking = fence->SetEventOnCompletion(7, nullptr);
        const HRESULT removed = device->GetDeviceRemovedReason();
        std::printf("Leonida Vulkan: injected VK_ERROR_DEVICE_LOST on blocking fence wait: "
                    "wait=%08x, device_removed=%08x, interceptions=%u\n",
                    unsigned(blocking), unsigned(removed), driver::failed_wait_count.load());
        if (blocking == S_OK || removed != DXGI_ERROR_DEVICE_REMOVED ||
            driver::failed_wait_count.load() != 1 || driver::fail_next_wait.load() != VK_SUCCESS)
            ++failures;
    }
    // ID3D12Device1::SetEventOnMultipleFenceCompletion has a separate direct blocking Vulkan wait,
    // not the single-fence VulkanFence::wait path. Both ALL and ANY must report device loss to the
    // shared context; a transient OOM error must return failure without falsely removing the device.
    for (const auto flags : {D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY}) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        Ref<ID3D12Device1> device1;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))) ||
            FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p)))) return 1;
        Ref<ID3D12Fence> fence_a, fence_b;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_a.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_b.p)))) return 1;
        ID3D12Fence* const fences[] = {fence_a.p, fence_b.p};
        const UINT64 values[] = {7, 9};
        const unsigned previous_wait_faults = driver::failed_wait_count.load();
        driver::fail_next_wait.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        const HRESULT oom = device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, nullptr);
        const HRESULT still_alive = device->GetDeviceRemovedReason();
        driver::fail_next_wait.store(VK_ERROR_DEVICE_LOST);
        const HRESULT lost = device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, nullptr);
        const HRESULT removed = device->GetDeviceRemovedReason();
        const unsigned delta = driver::failed_wait_count.load() - previous_wait_faults;
        std::printf("Leonida Vulkan: multi-fence blocking %s: transient=%08x, nonfatal_reason=%08x, "
                    "lost_wait=%08x, removed=%08x, interceptions=%u\n",
                    flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY ? "ANY" : "ALL",
                    unsigned(oom), unsigned(still_alive), unsigned(lost), unsigned(removed), delta);
        if (oom == S_OK || still_alive != S_OK || lost == S_OK ||
            removed != DXGI_ERROR_DEVICE_REMOVED || delta != 2 ||
            driver::fail_next_wait.load() != VK_SUCCESS)
            ++failures;
    }
    // Errors on a healthy Vulkan signal must not be hidden by the already-lost fast path.
    for (const VkResult injected : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_DEVICE_LOST}) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        Ref<ID3D12Fence> fence;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;
        const unsigned before = driver::foreground_signals.load();
        driver::fail_next_signal.store(injected);
        const HRESULT result = fence->Signal(1);
        const HRESULT removed = device->GetDeviceRemovedReason();
        const UINT64 completed = fence->GetCompletedValue();
        const HRESULT wanted_reason = injected == VK_ERROR_DEVICE_LOST ? DXGI_ERROR_DEVICE_REMOVED : S_OK;
        const UINT64 wanted_value = injected == VK_ERROR_DEVICE_LOST ? UINT64_MAX : 0;
        std::printf("injected signal %d: hr=%08lx reason=%08lx value=%llu calls=%u\n", int(injected),
                    static_cast<unsigned long>(result), static_cast<unsigned long>(removed),
                    static_cast<unsigned long long>(completed), driver::foreground_signals.load() - before);
        if (result != E_FAIL || removed != wanted_reason || completed != wanted_value ||
            driver::foreground_signals.load() != before + 1 || driver::fail_next_signal.load() != VK_SUCCESS)
            ++failures;
        if (injected == VK_ERROR_DEVICE_LOST) {
            // Synthetic logical-loss policy: subsequent calls complete without another Vulkan signal.
            const unsigned after_failure = driver::foreground_signals.load();
            const HRESULT after_loss = fence->Signal(2);
            const UINT64 after_value = fence->GetCompletedValue();
            std::printf("signal after injected device loss: hr=%08lx value=%llu additional foreground calls=%u\n",
                        static_cast<unsigned long>(after_loss), static_cast<unsigned long long>(after_value),
                        driver::foreground_signals.load() - after_failure);
            if (after_loss != S_OK || after_value != UINT64_MAX ||
                driver::foreground_signals.load() != after_failure) ++failures;
        }
    }
    // Separate logical device: RemoveDevice must mark this device's context lost, as an observed Vulkan loss does.
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        if (!remove_device_notifies(device.p, "Leonida Vulkan", true)) ++failures;
    }
    const unsigned errors = vk.validation_errors();
    std::printf("Vulkan validation errors: %u\n", errors);
    return failures || errors ? 1 : 0;
}
