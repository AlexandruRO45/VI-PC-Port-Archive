// A synthetic `gta6rt --run` guest for the first-boot runbook (docs/boot.md). Its only device is the
// D3D12XboxCreateDevice it imports from d3d12_xs.dll; the link-time stub fails, so a run that reaches the marker line
// has used the loader's shim registry. It makes one small, deterministic set of device/queue calls and prints one
// marker line. No game files, no shaders, no display, no source from src/ or shims/.
#include <windows.h>
#include <d3d12.h>
#include <unknwn.h>

#include <cstdint>
#include <cstdio>

// Resolved at link time against test/fixtures/d3d12_boot_stub.cpp, at run time by gta6rt's shim registry.
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                      const IID* riid, void** out);

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

bool failed(HRESULT hr, const char* what) {
    if (SUCCEEDED(hr)) return false;
    std::fprintf(stderr, "%s failed (hr=0x%08lx)\n", what, static_cast<unsigned long>(hr));
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the marker reaches the launcher even if teardown fails

    const std::uint32_t params[24] = {};  // the console's 24-word block; the shim logs but does not consume it
    const IID device_iid = __uuidof(ID3D12Device);
    Ref<ID3D12Device> device;
    if (failed(D3D12XboxCreateDevice(nullptr, params, &device_iid, reinterpret_cast<void**>(&device.p)),
               "D3D12XboxCreateDevice") ||
        !device.p)
        return 1;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    Ref<ID3D12CommandQueue> queue;
    if (failed(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p)), "CreateCommandQueue") || !queue.p)
        return 1;

    Ref<ID3D12CommandAllocator> allocator;
    if (failed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p)),
               "CreateCommandAllocator") ||
        !allocator.p)
        return 1;

    Ref<ID3D12GraphicsCommandList> list;
    if (failed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p)),
               "CreateCommandList") ||
        !list.p)
        return 1;
    if (failed(list->Close(), "Close")) return 1;

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);

    Ref<ID3D12Fence> fence;
    if (failed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)), "CreateFence") || !fence.p)
        return 1;
    if (failed(queue->Signal(fence.p, 1), "Signal")) return 1;
    if (failed(queue->Wait(fence.p, 1), "Wait")) return 1;

    // The shim may still have a submitted wait referencing the fence's semaphore. Let the GPU finish before the
    // Refs tear it down, so the test never destroys a VkSemaphore that a queue is still using.
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done) return 1;
    const HRESULT event_hr = fence->SetEventOnCompletion(1, done);
    if (FAILED(event_hr)) {
        CloseHandle(done);
        return 1;
    }
    WaitForSingleObject(done, INFINITE);
    CloseHandle(done);

    std::printf("d3d12 boot guest: PASS, 10 trace calls\n");
    return 0;
}
