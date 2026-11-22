// A queue's pending Vulkan wait must retain the fence's timeline beyond the last COM reference.
// Synthetic Vulkan interception prevents a use-after-destroy reaching the real driver.
// No game files.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {
namespace hook {
PFN_vkGetInstanceProcAddr get_instance_real = nullptr;
PFN_vkGetDeviceProcAddr get_device_real = nullptr;
PFN_vkCreateSemaphore create_real = nullptr;
PFN_vkDestroySemaphore destroy_real = nullptr;
PFN_vkQueueSubmit submit_real = nullptr;
std::mutex mutex;
std::unordered_set<VkSemaphore> destroyed;
std::unordered_set<VkSemaphore> watched;
std::vector<VkSemaphore> postponed_destroys;
std::atomic<unsigned> invalid_submit{0}, watched_destroyed{0}, premature_destroy{0};
std::atomic<VkSemaphore> last_created{VK_NULL_HANDLE}, monitored_signal{VK_NULL_HANDLE};
std::atomic<bool> gate_waiting{false};
std::atomic<bool> fail_submit_once{false};

VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice device, const VkSemaphoreCreateInfo* info,
                                      const VkAllocationCallbacks* allocator, VkSemaphore* out) {
    const VkResult result = create_real(device, info, allocator, out);
    if (result == VK_SUCCESS && out) {
        last_created = *out;
        std::lock_guard lock(mutex);
        destroyed.erase(*out);  // the implementation may reuse an already-freed handle
        // Queue progress timelines start at zero; these initial-value-one timelines are guest fences.
        const auto* type = info ? reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(info->pNext) : nullptr;
        if (type && type->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
            type->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE && type->initialValue == 1)
            watched.insert(*out);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(mutex);
        if (sem == monitored_signal.load() && gate_waiting.load()) {
            ++premature_destroy;
            postponed_destroys.push_back(sem); // do not break the real Vulkan driver in the red control
            return;
        }
        destroyed.insert(sem);
        if (watched.erase(sem)) ++watched_destroyed;
    }
    destroy_real(device, sem, allocator);
}

void flush_postponed(VkDevice device) {
    std::vector<VkSemaphore> deferred;
    {
        std::lock_guard lock(mutex);
        deferred.swap(postponed_destroys);
    }
    for (VkSemaphore sem : deferred) destroy_real(device, sem, nullptr);  // only after queue idle
}

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, std::uint32_t count,
                                      const VkSubmitInfo* submits, VkFence fence) {
    {
        std::lock_guard lock(mutex);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& s = submits[i];
            for (std::uint32_t j = 0; j < s.waitSemaphoreCount; ++j)
                if (destroyed.count(s.pWaitSemaphores[j])) {
                    ++invalid_submit;
                    // Do not feed an already-destroyed handle to the Vulkan driver (or validation).
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            for (std::uint32_t j = 0; j < s.signalSemaphoreCount; ++j)
                if (destroyed.count(s.pSignalSemaphores[j])) {
                    ++invalid_submit;
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
        }
    }
    if (fail_submit_once.exchange(false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return submit_real(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateSemaphore") == 0) return reinterpret_cast<PFN_vkVoidFunction>(create);
    if (std::strcmp(name, "vkDestroySemaphore") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroy);
    if (std::strcmp(name, "vkQueueSubmit") == 0) return reinterpret_cast<PFN_vkVoidFunction>(submit);
    return get_device_real(device, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (instance && std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return get_instance_real(instance, name);
}
}  // namespace hook

template<typename T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** out() { return &p; }
    void reset() { if (p) p->Release(); p = nullptr; }
};

bool reached(ID3D12Fence* fence, UINT64 value) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < end) {
        if (fence->GetCompletedValue() >= value) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// Run the identical already-satisfied Wait -> Release -> later submission sequence on native WARP.
// The host result proves the guest call sequence is legal and that it must not keep a stale fence handle.
int native_warp() {
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.out())))) return 77;
    Ref<IDXGIAdapter> adapter;
    if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(adapter.out())))) return 77;
    Ref<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.out())))) return 77;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> done;
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(queue.out()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(done.out())))) return 1;
    for (unsigned i = 0; i < 48; ++i) {
        Ref<ID3D12Fence> accepted;
        if (FAILED(device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(accepted.out()))) ||
            FAILED(queue->Wait(accepted.p, 1))) return 1;
        accepted.reset();
        if (FAILED(queue->Signal(done.p, i + 1))) return 1;
    }
    if (!reached(done.p, 48)) return 1;
    std::puts("native WARP: 48/48 released-fence pending waits completed");
    return 0;
}
} // namespace

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "fence lifetime line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    const int warp = native_warp();
    REQUIRE(warp == 0 || warp == 77);  // native WARP optional under Wine
    vktest::Device vk;
    if (!vk.create(false)) return 77;
    hook::get_instance_real = vk.instance_proc;
    hook::get_device_real = vk.device_proc;
    hook::create_real = vk.fn<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    hook::destroy_real = vk.fn<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    hook::submit_real = vk.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    REQUIRE(hook::create_real && hook::destroy_real && hook::submit_real);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = hook::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(device.out()))));
        Ref<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        REQUIRE(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(queue.out()))));
        Ref<ID3D12Fence> done;
        REQUIRE(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(done.out()))));
        for (unsigned i = 0; i < 48; ++i) {
            Ref<ID3D12Fence> accepted;
            REQUIRE(SUCCEEDED(device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(accepted.out()))));
            REQUIRE(SUCCEEDED(queue->Wait(accepted.p, 1)));
            accepted.reset();  // the queue has not submitted the Wait yet
            // A later submission must not refer to the destroyed timeline. Native WARP accepts this sequence.
            if (FAILED(queue->Signal(done.p, i + 1))) {
                std::fprintf(stderr, "queue signal failed at iteration %u, destroyed use=%u\n",
                             i, hook::invalid_submit.load());
                return 1;
            }
        }
        REQUIRE(reached(done.p, 48));
        REQUIRE(hook::invalid_submit.load() == 0);
        // Synthetic vkQueueSubmit failure must leave pending guest-wait retention intact for a retry.
        Ref<ID3D12Fence> retry_wait;
        REQUIRE(SUCCEEDED(device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(retry_wait.out()))));
        REQUIRE(SUCCEEDED(queue->Wait(retry_wait.p, 1)));
        retry_wait.reset();
        hook::fail_submit_once = true;
        REQUIRE(queue->Signal(done.p, 49) == E_FAIL);
        REQUIRE(hook::invalid_submit.load() == 0);
        // #3789's separate baseline bug consumes a failed fence value; retry with a higher value here.
        REQUIRE(SUCCEEDED(queue->Signal(done.p, 50)));
        REQUIRE(reached(done.p, 50));

        // Two independent COM queues may both have a pending Wait on a fence when its COM owner goes away.
        Ref<ID3D12CommandQueue> second;
        REQUIRE(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(second.out()))));
        Ref<ID3D12Fence> shared, second_done;
        REQUIRE(SUCCEEDED(device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(shared.out()))));
        REQUIRE(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(second_done.out()))));
        REQUIRE(SUCCEEDED(queue->Wait(shared.p, 1)));
        REQUIRE(SUCCEEDED(second->Wait(shared.p, 1)));
        shared.reset();
        std::atomic<unsigned> parallel_errors{0};
        std::thread other([&] { if (FAILED(second->Signal(second_done.p, 1))) ++parallel_errors; });
        if (FAILED(queue->Signal(done.p, 51))) ++parallel_errors;
        other.join();
        REQUIRE(parallel_errors == 0);
        REQUIRE(reached(done.p, 51) && reached(second_done.p, 1));
        REQUIRE(hook::invalid_submit.load() == 0);
        second.reset();
        queue.reset();  // wait idle, releasing all submitted-use references
        REQUIRE(hook::watched_destroyed.load() == 50);

        // A queue destroyed without ever submitting its already-satisfied Wait also releases the timeline.
        Ref<ID3D12CommandQueue> pending;
        REQUIRE(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(pending.out()))));
        Ref<ID3D12Fence> unsubmitted;
        REQUIRE(SUCCEEDED(device->CreateFence(1, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(unsubmitted.out()))));
        REQUIRE(SUCCEEDED(pending->Wait(unsubmitted.p, 1)));
        unsubmitted.reset();
        pending.reset();
        REQUIRE(hook::watched_destroyed.load() == 51);

        // Real Vulkan in-flight use: an independent host-signaled timeline keeps the physical VkQueue
        // stalled after Queue::Signal returns. Releasing that signaled fence before the gate opens must
        // not call vkDestroySemaphore while a submitted GPU batch still references it.
        Ref<ID3D12CommandQueue> gated;
        REQUIRE(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(gated.out()))));
        VkSemaphoreTypeCreateInfo gate_type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        gate_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo gate_create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &gate_type};
        VkSemaphore gate = VK_NULL_HANDLE;
        REQUIRE(hook::create_real(vk.device, &gate_create, nullptr, &gate) == VK_SUCCESS);
        const std::uint64_t gate_value = 1;
        VkTimelineSemaphoreSubmitInfo gate_timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        gate_timeline.waitSemaphoreValueCount = 1;
        gate_timeline.pWaitSemaphoreValues = &gate_value;
        const VkPipelineStageFlags gate_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo gate_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &gate_timeline};
        gate_submit.waitSemaphoreCount = 1;
        gate_submit.pWaitSemaphores = &gate;
        gate_submit.pWaitDstStageMask = &gate_stage;
        REQUIRE(hook::submit_real(vk.queue, 1, &gate_submit, VK_NULL_HANDLE) == VK_SUCCESS);
        Ref<ID3D12Fence> in_flight;
        REQUIRE(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(in_flight.out()))));
        hook::monitored_signal = hook::last_created.load();
        hook::gate_waiting = true;
        const HRESULT queued_signal = gated->Signal(in_flight.p, 1);
        in_flight.reset();  // submitted, but Vulkan cannot complete until the host signals gate
        const unsigned destroyed_too_early = hook::premature_destroy.load();
        VkSemaphoreSignalInfo unblock{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        unblock.semaphore = gate;
        unblock.value = 1;
        const auto real_signal = vk.fn<PFN_vkSignalSemaphore>("vkSignalSemaphore");
        if (!real_signal || real_signal(vk.device, &unblock) != VK_SUCCESS) {
            std::fputs("failed to release synthetic GPU gate; exiting without blocking queue teardown\n", stderr);
            std::fflush(stderr);
            std::_Exit(1);
        }
        // The host has now released the gate; a completed GPU signal may legitimately retire here.
        hook::gate_waiting = false;
        REQUIRE(SUCCEEDED(queued_signal));
        REQUIRE(SUCCEEDED(gated->Signal(done.p, 52)));
        REQUIRE(reached(done.p, 52));
        gated.reset();  // waits for the GPU: now it is safe to retire the signaled fence
        hook::flush_postponed(vk.device);
        hook::destroy_real(vk.device, gate, nullptr);
        REQUIRE(destroyed_too_early == 0);
        {
            std::lock_guard lock(hook::mutex);
            REQUIRE(hook::destroyed.count(hook::monitored_signal.load()) == 1);
        }
        hook::monitored_signal = VK_NULL_HANDLE;
        std::puts("in-flight fence: no early destruction, retired after GPU gate released and queue idle");
    }
    std::printf("fence release lifetime: 51 timelines retired, no stale handles; validation=%u\n",
                vktest::validation_error_count());
    return vktest::validation_error_count() ? 1 : 0;
}
