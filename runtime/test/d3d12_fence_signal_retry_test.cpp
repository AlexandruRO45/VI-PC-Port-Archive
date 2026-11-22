// A Vulkan OOM must not consume a D3D12 fence value before its CPU/GPU signal was accepted.
// Synthetic one-shot failures are injected before vkSignalSemaphore/vkQueueSubmit reaches the driver.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <string_view>

#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

namespace {
namespace driver {
PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkSignalSemaphore real_signal = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
std::atomic<bool> fail_signal{false}, fail_submit{false};
std::atomic<unsigned> signal_calls{0}, submit_calls{0};

VKAPI_ATTR VkResult VKAPI_CALL signal(VkDevice device, const VkSemaphoreSignalInfo* info) {
    ++signal_calls;
    if (fail_signal.exchange(false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return real_signal(device, info);
}

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count, const VkSubmitInfo* infos, VkFence fence) {
    ++submit_calls;
    if (fail_submit.exchange(false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return real_submit(queue, count, infos, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkSignalSemaphore") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(signal);
    if (std::strcmp(name, "vkQueueSubmit") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(submit);
    return real_get_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (instance != VK_NULL_HANDLE && std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(instance, name);
}
}  // namespace driver

template<typename T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

bool reached(ID3D12Fence* fence, UINT64 value) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fence->GetCompletedValue() >= value) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}
}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create(false)) return 77;
    driver::real_get_instance = vk.instance_proc;
    driver::real_get_device = vk.device_proc;
    driver::real_signal = vk.fn<PFN_vkSignalSemaphore>("vkSignalSemaphore");
    driver::real_submit = vk.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    EXPECT(driver::real_signal && driver::real_submit);

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        Ref<ID3D12Fence> cpu, gpu, other;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cpu.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&other.p))));
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));

        // CPU: the failed call must not signal or consume value 1.
        driver::fail_signal = true;
        const unsigned before_cpu = driver::signal_calls;
        EXPECT(cpu->Signal(1) == E_FAIL);
        EXPECT(driver::signal_calls == before_cpu + 1);
        EXPECT(cpu->GetCompletedValue() == 0);
        EXPECT(SUCCEEDED(cpu->Signal(1)));
        EXPECT(driver::signal_calls == before_cpu + 2);
        EXPECT(reached(cpu.p, 1));
        EXPECT(SUCCEEDED(cpu->Signal(1)));
        EXPECT(driver::signal_calls == before_cpu + 2);  // accepted duplicates stay ignored
        EXPECT(SUCCEEDED(cpu->Signal(2)));
        EXPECT(reached(cpu.p, 2));
        std::puts("CPU fence: synthetic OOM, identical retry, duplicate, ascending signal ok");

        // GPU: the first submit is rejected before reaching Vulkan; a same-value retry must submit.
        driver::fail_submit = true;
        const unsigned before_gpu = driver::submit_calls;
        EXPECT(queue->Signal(gpu.p, 1) == E_FAIL);
        EXPECT(driver::submit_calls == before_gpu + 1);
        EXPECT(gpu->GetCompletedValue() == 0);
        EXPECT(SUCCEEDED(queue->Signal(gpu.p, 1)));
        EXPECT(driver::submit_calls == before_gpu + 2);
        EXPECT(reached(gpu.p, 1));
        EXPECT(SUCCEEDED(queue->Signal(gpu.p, 1)));
        EXPECT(driver::submit_calls == before_gpu + 2);
        EXPECT(SUCCEEDED(queue->Signal(gpu.p, 2)));
        EXPECT(reached(gpu.p, 2));
        std::puts("GPU fence: synthetic OOM, identical retry, duplicate, ascending signal ok");

        // Cross-origin same-fence: failed GPU value can be retried on CPU and vice versa.
        driver::fail_submit = true;
        EXPECT(queue->Signal(other.p, 1) == E_FAIL);
        EXPECT(SUCCEEDED(other->Signal(1)));
        EXPECT(reached(other.p, 1));
        driver::fail_signal = true;
        EXPECT(other->Signal(2) == E_FAIL);
        EXPECT(SUCCEEDED(queue->Signal(other.p, 2)));
        EXPECT(reached(other.p, 2));
        std::puts("CPU/GPU cross-origin recovery ok");

        // Two D3D12 queues share this fence. An attempted lower value that fails must never undo an accepted
        // higher value. The fence-level transaction also prevents two host submissions of the same value.
        Ref<ID3D12CommandQueue> second;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&second.p))));
        Ref<ID3D12Fence> concurrent;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&concurrent.p))));
        driver::fail_submit = true;
        std::atomic<HRESULT> first_hr{E_PENDING};
        std::thread failing([&] { first_hr.store(queue->Signal(concurrent.p, 3)); });
        failing.join();
        EXPECT(first_hr.load() == E_FAIL);
        EXPECT(SUCCEEDED(second->Signal(concurrent.p, 4)));
        EXPECT(reached(concurrent.p, 4));
        const unsigned before_old = driver::submit_calls;
        EXPECT(SUCCEEDED(queue->Signal(concurrent.p, 3)));
        EXPECT(driver::submit_calls == before_old);  // a superseded failed value cannot roll back 4

        Ref<ID3D12Fence> same;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&same.p))));
        const unsigned before_race = driver::submit_calls;
        std::atomic<HRESULT> a{E_PENDING}, b{E_PENDING};
        std::thread t1([&] { a.store(queue->Signal(same.p, 1)); });
        std::thread t2([&] { b.store(second->Signal(same.p, 1)); });
        t1.join();
        t2.join();
        EXPECT(SUCCEEDED(a.load()) && SUCCEEDED(b.load()));
        EXPECT(driver::submit_calls == before_race + 1);
        EXPECT(reached(same.p, 1));
        std::puts("multi-queue fence: failed lower value, successful higher value, concurrent duplicate ok");
    }
    EXPECT(vk.validation_errors() == 0);
    std::puts("fence signal retry: PASS (0 Vulkan validation errors)");
    return 0;
}
