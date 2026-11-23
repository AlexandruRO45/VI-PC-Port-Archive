// D3D12 fence device-loss reporting when vkGetSemaphoreCounterValue first discovers a lost device.
// The injected VkResult is synthetic and never forwarded to the physical Vulkan driver. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

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
PFN_vkQueueSubmit real_submit = nullptr;
std::atomic<VkSemaphore> pending_progress{VK_NULL_HANDLE};
std::atomic<bool> hold_next_signal{false};
std::atomic<VkResult> fail_next_counter{VK_SUCCESS};
std::atomic<VkResult> fail_next_wait{VK_SUCCESS};
std::atomic<unsigned> failed_count{0};
std::atomic<unsigned> failed_wait_count{0};
DWORD test_thread = 0;

VKAPI_ATTR VkResult VKAPI_CALL get_counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    // A pending fence may also be polled by the dispatcher thread. Only the explicitly armed
    // synchronous GetCompletedValue call gets the test failure, so it is deterministic.
    if (GetCurrentThreadId() == test_thread && semaphore == pending_progress.load()) {
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

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* info,
                                           VkFence fence) {
    if (GetCurrentThreadId() != test_thread || !hold_next_signal.exchange(false))
        return real_submit(queue, count, info, fence);
    if (count != 1 || !info || info->commandBufferCount || !info->signalSemaphoreCount)
        return VK_ERROR_INITIALIZATION_FAILED;
    // Accept the fence-only submission in the shim but leave its real progress semaphore at zero.
    // This keeps the logical signal pending without stalling the physical GPU or depending on scheduling.
    pending_progress.store(info->pSignalSemaphores[0]);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info,
                                               std::uint64_t timeout) {
    const auto progress = pending_progress.load();
    for (std::uint32_t i = 0; progress && i < info->semaphoreCount; ++i) {
        if (info->pSemaphores[i] != progress) continue;
        const VkResult injected = fail_next_wait.exchange(VK_SUCCESS);
        if (injected != VK_SUCCESS) {
            failed_wait_count.fetch_add(1);
            return injected;
        }
        // Re-arm promptly if the test injects a failure after the dispatcher enters this wait.
        constexpr std::uint64_t poll_ns = 10'000'000;
        return real_wait(device, info, timeout < poll_ns ? timeout : poll_ns);
    }
    return real_wait(device, info, timeout);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_counter);
    if (std::strcmp(name, "vkWaitSemaphores") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (std::strcmp(name, "vkQueueSubmit") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_instance(instance, name);
}
}  // namespace driver

rt::d3d12::VulkanDeviceInfo device_info(const vktest::Device& vk) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = driver::get_instance;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    return info;
}

bool queue_pending_signal(ID3D12Device* device, ID3D12Fence* fence, Ref<ID3D12CommandQueue>& queue) {
    const D3D12_COMMAND_QUEUE_DESC desc{};
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p)))) return false;
    driver::pending_progress.store(VK_NULL_HANDLE);
    driver::hold_next_signal.store(true);
    return SUCCEEDED(queue->Signal(fence, 7)) && driver::pending_progress.load() &&
           !driver::hold_next_signal.load();
}

bool observed_wait_fault(unsigned count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (driver::failed_wait_count.load() < count) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

enum class WaitMode { single, all, any };

bool check_blocking_loss(const vktest::Device& vk, WaitMode mode) {
    const auto info = device_info(vk);
    Ref<ID3D12Device> device;
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Fence> first, second;
    Ref<ID3D12CommandQueue> queue;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&first.p))) ||
        !queue_pending_signal(device.p, first.p, queue))
        return false;
    if (mode != WaitMode::single &&
        (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p))) ||
         FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p)))))
        return false;

    Event started, finished;
    if (!started.handle || !finished.handle) return false;
    HRESULT blocking = E_UNEXPECTED;
    std::thread waiter([&] {
        SetEvent(started.handle);
        if (mode == WaitMode::single) {
            blocking = first->SetEventOnCompletion(7, nullptr);
        } else {
            ID3D12Fence* const fences[] = {first.p, second.p};
            const UINT64 values[] = {7, 9};
            const auto flags = mode == WaitMode::all ? D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL
                                                     : D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
            blocking = device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, nullptr);
        }
        SetEvent(finished.handle);
    });

    const bool entered = started.ready(2000);
    const bool early = finished.ready(50);
    const unsigned before = driver::failed_wait_count.load();
    driver::fail_next_wait.store(VK_ERROR_OUT_OF_HOST_MEMORY);
    const bool nonfatal = observed_wait_fault(before + 1) && device->GetDeviceRemovedReason() == S_OK;
    const bool after_oom = finished.ready(50);
    driver::fail_next_wait.store(VK_ERROR_DEVICE_LOST);
    if (!(early || after_oom || finished.ready(3000))) {
        std::fputs("logical fence blocking waiter did not leave after device loss\n", stderr);
        std::_Exit(1);
    }
    waiter.join();
    const bool lost = observed_wait_fault(before + 2) && device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED;
    const char* name = mode == WaitMode::single ? "single" : mode == WaitMode::all ? "ALL" : "ANY";
    std::printf("Leonida Vulkan: logical blocking %s: transient stayed pending=%d, lost wait=%08x, removed=%d\n",
                name, int(nonfatal && !after_oom), unsigned(blocking), int(lost));
    return entered && !early && nonfatal && !after_oom && lost && FAILED(blocking);
}

bool native_warp_control() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> device;
    Ref<ID3D12Fence> fence;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    Event event;
    if (!event.handle || fence->GetCompletedValue() != 0 ||
        FAILED(fence->SetEventOnCompletion(2, event.handle)) || event.ready(0) ||
        FAILED(fence->Signal(2)) || !event.ready(2000) ||
        fence->GetCompletedValue() != 2 || device->GetDeviceRemovedReason() != S_OK)
        return false;
    Ref<ID3D12Device1> device1;
    Ref<ID3D12Fence> second;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&second.p))))
        return false;
    ID3D12Fence* const fences[] = {fence.p, second.p};
    const UINT64 values[] = {2, 3};
    if (FAILED(second->Signal(3)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, nullptr)) ||
        FAILED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, nullptr)))
        return false;
    std::printf("native WARP: normal fence query/signal/event completed value 2 and two-fence ALL/ANY blocking waits\n");
    return true;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!native_warp_control()) {
        std::fprintf(stderr, "SKIP: native WARP unavailable or failed normal fence control\n");
        return 77;
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
    driver::real_submit = reinterpret_cast<PFN_vkQueueSubmit>(driver::real_device(vk.device, "vkQueueSubmit"));
    if (!driver::real_wait || !driver::real_submit) return 1;
    driver::test_thread = GetCurrentThreadId();

    Event pending;
    if (!pending.handle) return 1;
    int failures = 0;
    {
        const auto info = device_info(vk);
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12CommandQueue> queue;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
            fence->GetCompletedValue() != 0 || device->GetDeviceRemovedReason() != S_OK)
            return 1;
        if (FAILED(fence->SetEventOnCompletion(7, pending.handle)) ||
            !queue_pending_signal(device.p, fence.p, queue) || pending.ready(0)) return 1;

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
    for (const auto mode : {WaitMode::single, WaitMode::all, WaitMode::any})
        if (!check_blocking_loss(vk, mode)) ++failures;
    const unsigned errors = vk.validation_errors();
    std::printf("Vulkan validation errors: %u\n", errors);
    return failures || errors ? 1 : 0;
}
