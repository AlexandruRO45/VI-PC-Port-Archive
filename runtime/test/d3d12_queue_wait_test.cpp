// D3D-97: cross-queue Wait-before-Signal must make progress. No game files.
#include "../shims/d3d12_com.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

bool wait_for(ID3D12Fence* fence, UINT64 value, DWORD timeout_ms = 3000) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    if (FAILED(fence->SetEventOnCompletion(value, event))) {
        CloseHandle(event);  // no wait was registered
        return false;
    }
    if (WaitForSingleObject(event, timeout_ms) != WAIT_OBJECT_0) {
        // The fence dispatcher can still hold this event, and a queued GPU Wait can keep the queue destructor
        // blocked forever. End the failed test without closing live handles or waiting for blocked queues.
        std::fputs("cross-queue Wait-before-Signal timed out\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
    CloseHandle(event);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) return 77;
    if (argc == 2 && std::strcmp(argv[1], "--timeout-control") == 0) {
        Ref<ID3D12Fence> never_signaled;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&never_signaled.p)))) return 1;
        // Expected to leave the process with code 1 promptly, without destroying the pending event.
        wait_for(never_signaled.p, 1, 20);
        return 0;  // a completed wait here means the control did not exercise the timeout path
    }
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> copy, compute, direct_a, direct_b;
    desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&copy.p)))) return 77;
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&compute.p)))) return 77;
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&direct_a.p))) ||
        FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&direct_b.p)))) return 77;

    Ref<ID3D12Fence> gate_copy, gate_direct, done_copy, done_direct;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate_copy.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate_direct.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done_copy.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done_direct.p)))) return 1;

    // Submission order is deliberately opposite the dependency order.
    const bool submitted = SUCCEEDED(copy->Wait(gate_copy.p, 1)) &&
        SUCCEEDED(copy->Signal(done_copy.p, 1)) &&
        SUCCEEDED(direct_a->Wait(gate_direct.p, 1)) &&
        SUCCEEDED(direct_a->Signal(done_direct.p, 1)) &&
        SUCCEEDED(compute->Signal(gate_copy.p, 2)) &&
        SUCCEEDED(direct_b->Signal(gate_direct.p, 2));
    const bool copy_complete = submitted && wait_for(done_copy.p, 1);
    const bool direct_complete = submitted && wait_for(done_direct.p, 1);
    if (!copy_complete || !direct_complete) {
        // A failed submit can leave a GPU Wait pending. Gate 1 is not a valid fallback once queue signals
        // have claimed 2, so do not destroy queues or fences while their work may still be outstanding.
        std::fputs("cross-queue Wait-before-Signal did not complete\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }

    // D3D-481: a CPU Signal held behind a queue Signal the GPU has not reached is applied by the fence
    // dispatcher. The applied signal must still release a queue whose batch waited on that value.
    Ref<ID3D12Fence> gate2, f2, done2;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate2.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done2.p)))) return 1;
    const bool armed = SUCCEEDED(direct_a->Wait(gate2.p, 1)) &&
        SUCCEEDED(direct_a->Signal(f2.p, 10)) &&   // held behind the gate; claims queue value 10
        SUCCEEDED(f2->Signal(20)) &&               // CPU signal held behind it, applied when f2 reaches 10
        SUCCEEDED(direct_b->Wait(f2.p, 20)) &&
        SUCCEEDED(direct_b->Signal(done2.p, 1)) && // held until a Signal for 20 is submitted
        SUCCEEDED(gate2->Signal(1));
    if (!armed || !wait_for(done2.p, 1)) {
        std::fputs("a queue waiting on a held CPU signal stayed held\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
    std::puts("copy/compute and direct/direct Wait-before-Signal completed");
    return 0;
}
