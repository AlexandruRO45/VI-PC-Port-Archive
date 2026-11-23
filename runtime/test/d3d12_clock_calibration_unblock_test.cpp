// A fallback GetClockCalibration on Q1 must not hold the *device-global* submit lock while waiting for Q1's GPU.
// Test hook parks Q1's vkQueueWaitIdle only until an independent Q2 signal reaches vkQueueSubmit; no driver call is
// made while parked. A watchdog releases the hook at 700 ms, even for an unfixed runtime, to avoid a GPU deadlock.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "clock unblock line %d: %s\n", __LINE__, #x); return 1; } } while(0)

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};

struct Hook {
    PFN_vkGetInstanceProcAddr original_instance = nullptr;
    PFN_vkGetDeviceProcAddr original_device = nullptr;
    PFN_vkQueueWaitIdle original_idle = nullptr;
    PFN_vkQueueSubmit original_submit = nullptr;
    PFN_vkDeviceWaitIdle original_device_idle = nullptr;
    VkQueue queue1 = VK_NULL_HANDLE;
    VkQueue queue2 = VK_NULL_HANDLE;
    std::atomic<bool> armed{false}, parked{false}, saw_q2{false}, timed_out{false};
    std::atomic<unsigned> q2_calls{0}, q1_idle_calls{0};
    std::atomic<unsigned> device_idle_calls{0}, device_idle_during_calibration{0};
};
Hook injected;

VKAPI_ATTR VkResult VKAPI_CALL hooked_idle(VkQueue queue) {
    if (queue == injected.queue1) ++injected.q1_idle_calls;
    if (queue == injected.queue1 && injected.armed.exchange(false)) {
        injected.parked.store(true);
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(700);
        while (!injected.saw_q2.load() && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!injected.saw_q2.load()) injected.timed_out.store(true);
        injected.parked.store(false);
    }
    return injected.original_idle(queue);
}

VKAPI_ATTR VkResult VKAPI_CALL hooked_submit(VkQueue queue, std::uint32_t n, const VkSubmitInfo* info, VkFence fence) {
    if (queue == injected.queue2) { ++injected.q2_calls; injected.saw_q2.store(true); }
    return injected.original_submit(queue, n, info, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL hooked_device_idle(VkDevice device) {
    ++injected.device_idle_calls;
    if (injected.parked.load()) {
        // The old bindless flush reaches vkDeviceWaitIdle while another thread owns Q1's host
        // mutex in vkQueueWaitIdle. Calling a real ICD here violates Vulkan external sync, so
        // report the unsafe attempt WITHOUT touching the driver. The test must still fail.
        ++injected.device_idle_during_calibration;
        return VK_SUCCESS;
    }
    return injected.original_device_idle(device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hooked_get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkQueueWaitIdle") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_idle);
    if (std::strcmp(name, "vkQueueSubmit") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_submit);
    if (std::strcmp(name, "vkDeviceWaitIdle") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_device_idle);
    return injected.original_device(device, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hooked_get_instance(VkInstance instance, const char* name) {
    if (instance && std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_get_device);
    return injected.original_instance(instance, name);
}

bool reaches(ID3D12Fence* fence, UINT64 value, int milliseconds=3000) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < until) {
        if (fence->GetCompletedValue() >= value) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--warp") == 0) {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> adapter;
        Ref<ID3D12Device> device;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        REQUIRE(SUCCEEDED(factory.p->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))));
        REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p))));
        D3D12_COMMAND_QUEUE_DESC desc{};
        Ref<ID3D12CommandQueue> q1, q2;
        Ref<ID3D12Fence> gate, done;
        REQUIRE(SUCCEEDED(device.p->CreateCommandQueue(&desc, IID_PPV_ARGS(&q1.p))));
        REQUIRE(SUCCEEDED(device.p->CreateCommandQueue(&desc, IID_PPV_ARGS(&q2.p))));
        REQUIRE(SUCCEEDED(device.p->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))));
        REQUIRE(SUCCEEDED(device.p->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
        REQUIRE(SUCCEEDED(q1.p->Wait(gate.p, 1)));
        REQUIRE(SUCCEEDED(q1.p->Signal(done.p, 1)));
        std::atomic<HRESULT> calibrated{E_PENDING};
        std::thread caller([&] { UINT64 gpu=0,cpu=0; calibrated=q1.p->GetClockCalibration(&gpu,&cpu); });
        const auto before=std::chrono::steady_clock::now();
        const HRESULT signal_hr=q2.p->Signal(gate.p,1);
        const long signal_ms=long(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-before).count());
        caller.join();
        REQUIRE(SUCCEEDED(signal_hr) && SUCCEEDED(calibrated.load()) && reaches(done.p,1));
        REQUIRE(signal_ms < 500);
        std::printf("native WARP GetClockCalibration future wait: q2 Signal in %ld ms, PASS\n",signal_ms);
        return 0;
    }

    vktest::Device vk;
    vk.queue_count_wanted=2;
    const auto status=vk.initialize();
    if(status==vktest::DeviceStatus::unavailable || vk.queues.size()<2 || vk.queues[0]==vk.queues[1]) return 77;
    REQUIRE(status==vktest::DeviceStatus::ready);
    injected.original_instance=vk.instance_proc;
    injected.original_device=vk.device_proc;
    injected.original_idle=vk.fn<PFN_vkQueueWaitIdle>("vkQueueWaitIdle");
    injected.original_submit=vk.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    injected.original_device_idle=vk.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
    injected.queue1=vk.queues[0]; injected.queue2=vk.queues[1];
    REQUIRE(injected.original_idle && injected.original_submit && injected.original_device_idle);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc=hooked_get_instance;
        info.instance=vk.instance; info.physical=vk.physical; info.device=vk.device;
        info.queue=vk.queue; info.queue_family=vk.family;
        info.command_queues=vk.queues;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        Ref<ID3D12Device> device;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info,IID_PPV_ARGS(&device.p))));
        D3D12_COMMAND_QUEUE_DESC desc{};
        Ref<ID3D12CommandQueue> q1,q2;
        Ref<ID3D12Fence> fence;
        REQUIRE(SUCCEEDED(device.p->CreateCommandQueue(&desc,IID_PPV_ARGS(&q1.p))));
        REQUIRE(SUCCEEDED(device.p->CreateCommandQueue(&desc,IID_PPV_ARGS(&q2.p))));
        REQUIRE(SUCCEEDED(device.p->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
        // The hook is armed only around GetClockCalibration; it parks an otherwise-idle Vulkan queue.
        injected.armed.store(true);
        std::atomic<HRESULT> calibration_hr{E_PENDING};
        std::thread a([&] { UINT64 gpu=0,cpu=0; calibration_hr.store(q1.p->GetClockCalibration(&gpu,&cpu)); });
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(!injected.parked.load() && std::chrono::steady_clock::now()<until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(injected.parked.load());
        const auto before=std::chrono::steady_clock::now();
        std::atomic<HRESULT> signal_hr{E_PENDING};
        std::thread b([&] { signal_hr.store(q2.p->Signal(fence.p,1)); });
        b.join(); a.join();
        const long elapsed=long(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-before).count());
        REQUIRE(SUCCEEDED(signal_hr.load()) && SUCCEEDED(calibration_hr.load()) && reaches(fence.p,1));
        std::printf("calibration fallback: independent Q2 submit after %ld ms; watchdog=%d, validation=%u\n",
                    elapsed,int(injected.timed_out.load()),vk.validation_errors());
        REQUIRE(!injected.timed_out.load() && elapsed<600 && injected.q2_calls==1 && vk.validation_errors()==0);

        if (vk.bindless) {
            // The first submission uploads the initial bindless descriptor table. Inserting
            // sufficiently many distinct S# keys afterwards forces its hash table to grow;
            // the next execution must retire the old buffer while Q1 is calibrated.
            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12DescriptorHeap> sampler_heap;
            REQUIRE(SUCCEEDED(device.p->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&allocator.p))));
            REQUIRE(SUCCEEDED(device.p->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.p,
                                                          nullptr,IID_PPV_ARGS(&list.p))));
            REQUIRE(SUCCEEDED(list.p->Close()));
            ID3D12CommandList* lists[]{list.p};
            q2.p->ExecuteCommandLists(1,lists);  // first bindless table upload
            REQUIRE(SUCCEEDED(q2.p->Signal(fence.p,2)) && reaches(fence.p,2));

            D3D12_DESCRIPTOR_HEAP_DESC heap{};
            heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            heap.NumDescriptors=144;
            REQUIRE(SUCCEEDED(device.p->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&sampler_heap.p))));
            const auto start=sampler_heap.p->GetCPUDescriptorHandleForHeapStart();
            const UINT stride=device.p->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            for(UINT i=0;i<128;++i) {
                D3D12_SAMPLER_DESC s{};
                s.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                s.AddressU=s.AddressV=s.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                s.MipLODBias=float(i)/16.0f;
                s.MaxAnisotropy=1;
                s.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;
                s.MaxLOD=D3D12_FLOAT32_MAX;
                device.p->CreateSampler(&s,{start.ptr+SIZE_T(i)*stride});
            }
            injected.parked.store(false);
            injected.saw_q2.store(false);
            injected.timed_out.store(false);
            injected.armed.store(true);
            injected.device_idle_calls.store(0);
            injected.device_idle_during_calibration.store(0);
            injected.q1_idle_calls.store(0);
            std::atomic<HRESULT> calibration2{E_PENDING};
            std::thread calibration_thread([&] {
                UINT64 gpu=0,cpu=0;
                calibration2.store(q1.p->GetClockCalibration(&gpu,&cpu));
            });
            const auto parked_until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!injected.parked.load() && std::chrono::steady_clock::now()<parked_until)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const bool park_seen=injected.parked.load();
            std::thread flush_thread([&] {
                // Empty list still runs the real device's bindless_prepare callback.
                q2.p->ExecuteCommandLists(1,lists);
            });
            // An artificial gate, not a driver signal: the first phase above already verifies
            // independent Q2 progress. Release the parked Q1 host call safely in both builds.
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            injected.saw_q2.store(true);
            calibration_thread.join();
            flush_thread.join();
            const HRESULT released=q2.p->Signal(fence.p,3);
            const unsigned violations=injected.device_idle_during_calibration.load();
            const unsigned idles=injected.q1_idle_calls.load();
            std::printf("bindless flush + calibration: parked=%d, global idle overlap=%u, Q1 queue idles=%u, watchdog=%d, validation=%u\n",
                        int(park_seen),violations,idles,int(injected.timed_out.load()),vk.validation_errors());
            REQUIRE(park_seen && !injected.timed_out.load() && SUCCEEDED(calibration2.load()) &&
                    SUCCEEDED(released) && reaches(fence.p,3) && violations==0 && idles>=3 &&
                    vk.validation_errors()==0);
        } else {
            std::puts("SKIP bindless flush race: no descriptor-indexing features on this Vulkan device");
        }
    }
    return 0;
}
