// Methods of the D3D12 objects (command queue, fence, resource, descriptor heap, query heap, command allocator and
// list) and the device calls that take no description, on arguments the host runtime refuses and on the values it
// reports, compared with the host D3D12 on the hardware adapter and on WARP. A row counts only where the two host
// adapters agree. Not compared: a call the host faults on (a null fence or output), a CPU or queue Signal below the
// fence's value (the host lowers the fence, a Vulkan timeline cannot), GetGPUVirtualAddress and the shader-visible GPU
// descriptor start (this test's Vulkan device has no address arena), ID3D12Resource1. No shaders or game data.
// LEONIDA_PROBE (any value) prints every row.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr std::int64_t kCrash = 0x7FFF0000;

D3D12_RESOURCE_DESC buffer_desc(UINT64 size) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture_desc(UINT w, UINT h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = f;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

bool make_queue(ID3D12Device* d, Ref<ID3D12CommandQueue>& q, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = type;
    return SUCCEEDED(d->CreateCommandQueue(&desc, IID_PPV_ARGS(&q.p)));
}
bool make_buffer(ID3D12Device* d, Ref<ID3D12Resource>& r, D3D12_HEAP_TYPE heap, UINT64 size = 4096,
                 D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    D3D12_RESOURCE_DESC desc = buffer_desc(size);
    desc.Flags = flags;
    const auto state = heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                       : heap == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                          : D3D12_RESOURCE_STATE_COMMON;
    return SUCCEEDED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r.p)));
}
bool make_texture(ID3D12Device* d, Ref<ID3D12Resource>& r, const D3D12_RESOURCE_DESC& desc,
                  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    return SUCCEEDED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r.p)));
}

std::int64_t hr(HRESULT h) { return static_cast<std::int64_t>(static_cast<std::uint32_t>(h)); }
std::int64_t mix(std::int64_t a, std::int64_t b) { return (a * 1000003) ^ b; }

using Fn = std::int64_t (*)(ID3D12Device*);
struct Case {
    const char* name;
    Fn fn;
};

#define CASE(name) {name, [](ID3D12Device* d) -> std::int64_t
#define END }

const Case kCases[] = {
    // ---- command queue
    CASE("queue: GetTimestampFrequency value present") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        UINT64 f = 0; const HRESULT r = q->GetTimestampFrequency(&f);
        return mix(hr(r), f != 0);
    } END,
    CASE("queue: GetClockCalibration values") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        UINT64 g = 0, c = 0; const HRESULT r = q->GetClockCalibration(&g, &c);
        return mix(hr(r), (g != 0) + 2 * (c != 0));
    } END,
    CASE("queue: GetClockCalibration copy queue") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q, D3D12_COMMAND_LIST_TYPE_COPY)) return -1;
        UINT64 g = 0, c = 0; const HRESULT r = q->GetClockCalibration(&g, &c);
        return mix(hr(r), (g != 0) + 2 * (c != 0));
    } END,
    CASE("queue: GetDesc") {
        Ref<ID3D12CommandQueue> q;
        D3D12_COMMAND_QUEUE_DESC desc{}; desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH; desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
        desc.NodeMask = 1;
        if (FAILED(d->CreateCommandQueue(&desc, IID_PPV_ARGS(&q.p)))) return -1;
        const D3D12_COMMAND_QUEUE_DESC got = q->GetDesc();
        return mix(mix(mix(got.Type, got.Priority), got.Flags), got.NodeMask);
    } END,
    CASE("queue: GetDesc node mask 0 becomes") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        return q->GetDesc().NodeMask;
    } END,
    CASE("queue: Signal then completed value") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(5, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        const HRESULT r = q->Signal(f.p, 9);
        if (FAILED(r)) return hr(r);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        f->SetEventOnCompletion(9, e); WaitForSingleObject(e, 5000); CloseHandle(e);
        return mix(hr(r), static_cast<std::int64_t>(f->GetCompletedValue()));
    } END,
    CASE("queue: Wait on a CPU-signalled value") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        Ref<ID3D12Fence> f; Ref<ID3D12Fence> g;
        if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.p)))) return -1;
        const HRESULT w = q->Wait(f.p, 3);
        const HRESULT s = q->Signal(g.p, 1);
        Sleep(30);
        const UINT64 before = g->GetCompletedValue();
        f->Signal(3);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        g->SetEventOnCompletion(1, e); const DWORD got = WaitForSingleObject(e, 5000); CloseHandle(e);
        return mix(mix(hr(w), hr(s)), mix(static_cast<std::int64_t>(before), got == WAIT_OBJECT_0));
    } END,
    CASE("queue: GetDevice wrong iid") {
        Ref<ID3D12CommandQueue> q; if (!make_queue(d, q)) return -1;
        void* out = reinterpret_cast<void*>(1);
        const HRESULT r = q->GetDevice(__uuidof(ID3D12Fence), &out);
        return mix(hr(r), out == nullptr);
    } END,
    // ---- fence
    CASE("fence: GetCompletedValue after create") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        return static_cast<std::int64_t>(f->GetCompletedValue());
    } END,
    CASE("fence: Signal max value") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        const HRESULT r = f->Signal(UINT64_MAX);
        return mix(hr(r), f->GetCompletedValue() == UINT64_MAX);
    } END,
    CASE("fence: SetEventOnCompletion completed value signals") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = f->SetEventOnCompletion(5, e);
        const DWORD w = WaitForSingleObject(e, 0); CloseHandle(e);
        return mix(hr(r), w == WAIT_OBJECT_0);
    } END,
    CASE("fence: SetEventOnCompletion null event completed") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        return hr(f->SetEventOnCompletion(7, nullptr));
    } END,
    CASE("fence: SetEventOnCompletion invalid handle") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        return hr(f->SetEventOnCompletion(1, reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("fence: SetEventOnCompletion pending then Signal") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = f->SetEventOnCompletion(2, e);
        const DWORD before = WaitForSingleObject(e, 10);
        f->Signal(1); const DWORD mid = WaitForSingleObject(e, 10);
        f->Signal(2); const DWORD after = WaitForSingleObject(e, 1000);
        CloseHandle(e);
        return mix(mix(hr(r), before == WAIT_TIMEOUT), mix(mid == WAIT_TIMEOUT, after == WAIT_OBJECT_0));
    } END,
    CASE("fence: Fence1 creation flags") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        Ref<ID3D12Fence1> f1; if (FAILED(f->QueryInterface(IID_PPV_ARGS(&f1.p)))) return -2;
        return f1->GetCreationFlags();
    } END,
    CASE("fence: Fence1 creation flags shared") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&f.p)))) return -1;
        Ref<ID3D12Fence1> f1; if (FAILED(f->QueryInterface(IID_PPV_ARGS(&f1.p)))) return -2;
        return f1->GetCreationFlags();
    } END,
    CASE("fence: SetEventOnCompletion completed value, invalid handle") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        return hr(f->SetEventOnCompletion(3, reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("fence: SetEventOnCompletion closed handle") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr); CloseHandle(e);
        return hr(f->SetEventOnCompletion(1, e));
    } END,
    CASE("fence: SetEventOnCompletion semaphore handle, completed") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        HANDLE e = CreateSemaphoreW(nullptr, 0, 1, nullptr);
        const HRESULT r = f->SetEventOnCompletion(3, e); CloseHandle(e);
        return hr(r);
    } END,
    CASE("fence: SetEventOnCompletion handle without modify access") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        HANDLE ro = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), e, GetCurrentProcess(), &ro, SYNCHRONIZE, FALSE, 0)) return -3;
        const HRESULT r = f->SetEventOnCompletion(3, ro); CloseHandle(ro); CloseHandle(e);
        return hr(r);
    } END,
    CASE("fence: SetEventOnCompletion pseudo handle") {
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(7, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        return hr(f->SetEventOnCompletion(3, GetCurrentProcess()));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion completed, invalid handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(5, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {1};
        return hr(d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion closed handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {1};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr); CloseHandle(e);
        return hr(d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, e));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion semaphore handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(5, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {1};
        HANDLE e = CreateSemaphoreW(nullptr, 0, 1, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, e);
        CloseHandle(e);
        return hr(r);
    } END,
    CASE("device: GetDescriptorHandleIncrementSize types 4 to 9") {
        std::int64_t v = 0;
        for (int t = 4; t < 10; ++t) v = mix(v, d->GetDescriptorHandleIncrementSize(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(t)));
        return v;
    } END,
    CASE("device: GetDescriptorHandleIncrementSize huge type") {
        return d->GetDescriptorHandleIncrementSize(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(0x7fffffff));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion invalid flags and invalid handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {0};
        return hr(d1->SetEventOnMultipleFenceCompletion(fences, values, 1, static_cast<D3D12_MULTIPLE_FENCE_WAIT_FLAGS>(5), reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion zero fences and invalid handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        return hr(d1->SetEventOnMultipleFenceCompletion(nullptr, nullptr, 0, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion foreign fence and invalid handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {5};
        return hr(d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, GetCurrentProcess()));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion no modify access") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {5};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr); HANDLE ro = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), e, GetCurrentProcess(), &ro, SYNCHRONIZE, FALSE, 0)) return -3;
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ro);
        CloseHandle(ro); CloseHandle(e);
        return hr(r);
    } END,
    // ---- resource
    CASE("resource: GetHeapProperties default buffer") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_DEFAULT)) return -1;
        D3D12_HEAP_PROPERTIES p{}; D3D12_HEAP_FLAGS f{};
        const HRESULT h = r->GetHeapProperties(&p, &f);
        return mix(mix(hr(h), p.Type), mix(mix(p.CPUPageProperty, p.MemoryPoolPreference), mix(p.CreationNodeMask, mix(p.VisibleNodeMask, f))));
    } END,
    CASE("resource: GetHeapProperties upload buffer") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_UPLOAD)) return -1;
        D3D12_HEAP_PROPERTIES p{}; D3D12_HEAP_FLAGS f{};
        const HRESULT h = r->GetHeapProperties(&p, &f);
        return mix(mix(hr(h), p.Type), mix(mix(p.CPUPageProperty, p.MemoryPoolPreference), mix(p.CreationNodeMask, mix(p.VisibleNodeMask, f))));
    } END,
    CASE("resource: GetHeapProperties readback buffer") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_READBACK)) return -1;
        D3D12_HEAP_PROPERTIES p{}; D3D12_HEAP_FLAGS f{};
        const HRESULT h = r->GetHeapProperties(&p, &f);
        return mix(mix(hr(h), p.Type), mix(mix(p.CPUPageProperty, p.MemoryPoolPreference), mix(p.CreationNodeMask, mix(p.VisibleNodeMask, f))));
    } END,
    CASE("resource: GetHeapProperties null both") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_DEFAULT)) return -1;
        return hr(r->GetHeapProperties(nullptr, nullptr));
    } END,
    CASE("resource: GetHeapProperties null flags") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_DEFAULT)) return -1;
        D3D12_HEAP_PROPERTIES p{}; return hr(r->GetHeapProperties(&p, nullptr));
    } END,
    CASE("resource: GetHeapProperties null properties") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_DEFAULT)) return -1;
        D3D12_HEAP_FLAGS f{}; return mix(hr(r->GetHeapProperties(nullptr, &f)), f);
    } END,
    CASE("resource: GetHeapProperties texture with flags") {
        Ref<ID3D12Resource> r;
        if (!make_texture(d, r, texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))) return -1;
        D3D12_HEAP_PROPERTIES p{}; D3D12_HEAP_FLAGS f{};
        const HRESULT h = r->GetHeapProperties(&p, &f);
        return mix(mix(hr(h), p.Type), f);
    } END,
    CASE("resource: GetDesc roundtrip texture") {
        Ref<ID3D12Resource> r;
        D3D12_RESOURCE_DESC in = texture_desc(37, 21, DXGI_FORMAT_B8G8R8A8_UNORM); in.MipLevels = 0; in.DepthOrArraySize = 3;
        if (!make_texture(d, r, in)) return -1;
        const D3D12_RESOURCE_DESC o = r->GetDesc();
        return mix(mix(mix(o.Width, o.Height), mix(o.DepthOrArraySize, o.MipLevels)), mix(mix(o.Format, o.Alignment), mix(o.Layout, o.Flags)));
    } END,
    CASE("resource: GetDesc roundtrip buffer alignment") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_DEFAULT, 100)) return -1;
        const D3D12_RESOURCE_DESC o = r->GetDesc();
        return mix(mix(o.Width, o.Alignment), mix(o.Layout, o.Flags));
    } END,
    CASE("resource: Unmap without Map") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_UPLOAD)) return -1;
        r->Unmap(0, nullptr);
        return 0;
    } END,
    CASE("resource: Map twice same pointer") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_UPLOAD)) return -1;
        void* a = nullptr; void* b = nullptr;
        const HRESULT h1 = r->Map(0, nullptr, &a); const HRESULT h2 = r->Map(0, nullptr, &b);
        r->Unmap(0, nullptr); r->Unmap(0, nullptr);
        return mix(mix(hr(h1), hr(h2)), a == b);
    } END,
    CASE("resource: Map subresource past the end") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_UPLOAD)) return -1;
        void* a = nullptr; return hr(r->Map(1, nullptr, &a));
    } END,
    CASE("resource: Map texture upload row-major") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_UPLOAD, 65536)) return -1;
        void* a = nullptr; const D3D12_RANGE rg{100, 50}; return hr(r->Map(0, &rg, &a));
    } END,
    CASE("resource: Map readback range past size") {
        Ref<ID3D12Resource> r; if (!make_buffer(d, r, D3D12_HEAP_TYPE_READBACK, 256)) return -1;
        void* a = nullptr; const D3D12_RANGE rg{0, 4096}; return hr(r->Map(0, &rg, &a));
    } END,
    // ---- descriptor heap
    CASE("descriptor heap: GetDesc") {
        Ref<ID3D12DescriptorHeap> h;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 100, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        if (FAILED(d->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&h.p)))) return -1;
        const auto o = h->GetDesc();
        return mix(mix(o.Type, o.NumDescriptors), mix(o.Flags, o.NodeMask));
    } END,
    CASE("descriptor heap: GetDesc node mask 0") {
        Ref<ID3D12DescriptorHeap> h;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(d->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&h.p)))) return -1;
        return h->GetDesc().NodeMask;
    } END,
    CASE("descriptor heap: CPU start non-zero") {
        Ref<ID3D12DescriptorHeap> h;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(d->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&h.p)))) return -1;
        return h->GetCPUDescriptorHandleForHeapStart().ptr != 0;
    } END,
    CASE("device: GetDescriptorHandleIncrementSize types") {
        std::int64_t v = 0;
        for (int t = 0; t < 4; ++t) v = mix(v, d->GetDescriptorHandleIncrementSize(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(t)) != 0);
        return v;
    } END,
    CASE("device: GetDescriptorHandleIncrementSize invalid type") {
        return d->GetDescriptorHandleIncrementSize(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(7));
    } END,
    // ---- query heap
    CASE("query heap: GetDevice") {
        Ref<ID3D12QueryHeap> q; D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4, 0};
        if (FAILED(d->CreateQueryHeap(&desc, IID_PPV_ARGS(&q.p)))) return -1;
        Ref<ID3D12Device> dev; const HRESULT r = q->GetDevice(IID_PPV_ARGS(&dev.p));
        return mix(hr(r), dev.p == d);
    } END,
    // ---- command allocator and list
    CASE("allocator: Reset twice") {
        Ref<ID3D12CommandAllocator> a; if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        const HRESULT r1 = a->Reset(); const HRESULT r2 = a->Reset();
        return mix(hr(r1), hr(r2));
    } END,
    CASE("allocator: Reset while a list is open") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        return hr(a->Reset());
    } END,
    CASE("list: Close twice") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        const HRESULT r1 = l->Close(); const HRESULT r2 = l->Close();
        return mix(hr(r1), hr(r2));
    } END,
    CASE("list: Reset without Close") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        return hr(l->Reset(a.p, nullptr));
    } END,
    CASE("list: Reset after Close with same allocator") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        l->Close();
        return hr(l->Reset(a.p, nullptr));
    } END,
    CASE("list: Reset with a copy allocator") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12CommandAllocator> c; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&c.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        l->Close();
        return hr(l->Reset(c.p, nullptr));
    } END,
    CASE("list: GetType") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        return l->GetType();
    } END,
    CASE("list: allocator GetDevice") {
        Ref<ID3D12CommandAllocator> a; if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        Ref<ID3D12Device> dev; const HRESULT r = a->GetDevice(IID_PPV_ARGS(&dev.p));
        return mix(hr(r), dev.p == d);
    } END,
    CASE("list: QueryInterface list versions") {
        Ref<ID3D12CommandAllocator> a; Ref<ID3D12GraphicsCommandList> l;
        if (FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p)))) return -1;
        if (FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p)))) return -1;
        std::int64_t v = 0;
        const IID iids[] = {__uuidof(ID3D12GraphicsCommandList1), __uuidof(ID3D12GraphicsCommandList2), __uuidof(ID3D12GraphicsCommandList3),
                            __uuidof(ID3D12GraphicsCommandList4), __uuidof(ID3D12GraphicsCommandList5), __uuidof(ID3D12GraphicsCommandList6),
                            __uuidof(ID3D12GraphicsCommandList7), __uuidof(ID3D12GraphicsCommandList8), __uuidof(ID3D12DebugCommandList),
                            __uuidof(ID3D12CommandList), __uuidof(ID3D12Pageable), __uuidof(ID3D12DeviceChild)};
        for (const IID& iid : iids) {
            void* o = nullptr; const HRESULT r = l->QueryInterface(iid, &o);
            if (o) static_cast<IUnknown*>(o)->Release();
            v = mix(v, hr(r));
        }
        return v;
    } END,
    // ---- device-level calls without a creation argument to refuse
    CASE("device: GetNodeCount") { return d->GetNodeCount(); } END,
    CASE("device: GetDeviceRemovedReason") { return hr(d->GetDeviceRemovedReason()); } END,
    // SetStablePowerState is not a table row: the host answers E_FAIL without Developer Mode, and a failed call
    // leaves the host device removed (every later CreateFence is DXGI_ERROR_DEVICE_REMOVED), so it runs last on a
    // throwaway device (stable_power).
    CASE("device6: SetBackgroundProcessingMode valid") {
        Ref<ID3D12Device6> d6; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d6.p)))) return -2;
        return hr(d6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED, D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr));
    } END,
    CASE("device6: SetBackgroundProcessingMode invalid mode") {
        Ref<ID3D12Device6> d6; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d6.p)))) return -2;
        return hr(d6->SetBackgroundProcessingMode(static_cast<D3D12_BACKGROUND_PROCESSING_MODE>(9), D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr));
    } END,
    CASE("device6: SetBackgroundProcessingMode invalid action") {
        Ref<ID3D12Device6> d6; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d6.p)))) return -2;
        return hr(d6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED, static_cast<D3D12_MEASUREMENTS_ACTION>(9), nullptr, nullptr));
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion all, already complete") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f[2];
        for (auto& x : f) if (FAILED(d->CreateFence(4, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&x.p)))) return -1;
        ID3D12Fence* fences[2] = {f[0].p, f[1].p}; UINT64 values[2] = {3, 4};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, e);
        const DWORD w = WaitForSingleObject(e, 1000); CloseHandle(e);
        return mix(hr(r), w == WAIT_OBJECT_0);
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion any, one pending") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f[2];
        for (auto& x : f) if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&x.p)))) return -1;
        ID3D12Fence* fences[2] = {f[0].p, f[1].p}; UINT64 values[2] = {1, 2};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, e);
        const DWORD before = WaitForSingleObject(e, 10);
        f[1]->Signal(2);
        const DWORD after = WaitForSingleObject(e, 1000); CloseHandle(e);
        return mix(mix(hr(r), before == WAIT_TIMEOUT), after == WAIT_OBJECT_0);
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion all, one pending") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f[2];
        for (auto& x : f) if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&x.p)))) return -1;
        ID3D12Fence* fences[2] = {f[0].p, f[1].p}; UINT64 values[2] = {1, 2};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, e);
        f[0]->Signal(1);
        const DWORD mid = WaitForSingleObject(e, 10);
        f[1]->Signal(2);
        const DWORD after = WaitForSingleObject(e, 1000); CloseHandle(e);
        return mix(mix(hr(r), mid == WAIT_TIMEOUT), after == WAIT_OBJECT_0);
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion zero fences") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(nullptr, nullptr, 0, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, e);
        CloseHandle(e);
        return hr(r);
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion invalid flags") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {0};
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const HRESULT r = d1->SetEventOnMultipleFenceCompletion(fences, values, 1, static_cast<D3D12_MULTIPLE_FENCE_WAIT_FLAGS>(5), e);
        CloseHandle(e);
        return hr(r);
    } END,
    CASE("device1: SetEventOnMultipleFenceCompletion invalid event handle") {
        Ref<ID3D12Device1> d1; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d1.p)))) return -2;
        Ref<ID3D12Fence> f; if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p)))) return -1;
        ID3D12Fence* fences[1] = {f.p}; UINT64 values[1] = {1};
        return hr(d1->SetEventOnMultipleFenceCompletion(fences, values, 1, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, reinterpret_cast<HANDLE>(0x1234)));
    } END,
    CASE("device: QueryInterface of the device chain") {
        const IID iids[] = {__uuidof(ID3D12Device1), __uuidof(ID3D12Device2), __uuidof(ID3D12Device3), __uuidof(ID3D12Device4),
                            __uuidof(ID3D12Device5), __uuidof(ID3D12Device6), __uuidof(ID3D12Device7), __uuidof(ID3D12Device8),
                            __uuidof(ID3D12Device9), __uuidof(ID3D12Device10), __uuidof(ID3D12Device11), __uuidof(ID3D12Object),
                            __uuidof(ID3D12DeviceChild), __uuidof(ID3D12Pageable), __uuidof(IDXGIObject), __uuidof(IDXGIDevice),
                            __uuidof(ID3D12CommandQueue), __uuidof(ID3D12Resource)};
        std::int64_t v = 0;
        for (const IID& iid : iids) {
            void* o = nullptr; const HRESULT r = d->QueryInterface(iid, &o);
            if (o) static_cast<IUnknown*>(o)->Release();
            v = mix(v, hr(r));
        }
        return v;
    } END,
    CASE("device5: CheckDriverMatchingIdentifier unknown id") {
        Ref<ID3D12Device5> d5; if (FAILED(d->QueryInterface(IID_PPV_ARGS(&d5.p)))) return -2;
        D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER id{};
        return d5->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE, &id);
    } END,
};

struct Row {
    std::string name;
    std::int64_t code;
};

std::int64_t guarded(Fn fn, ID3D12Device* d) {
    __try {
        return fn(d);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return kCrash;
    }
}

std::vector<Row> run(ID3D12Device* d) {
    std::vector<Row> rows;
    for (const Case& c : kCases) rows.push_back({c.name, guarded(c.fn, d)});
    return rows;
}
// SetStablePowerState on a fresh device of its own. Only a host that accepts the call (Developer Mode on) gives
// an answer to compare; otherwise the row is not comparable here and is left out.
std::int64_t stable_power(ID3D12Device* d) {
    const HRESULT r = d->SetStablePowerState(TRUE);
    if (SUCCEEDED(r)) d->SetStablePowerState(FALSE);
    return hr(r);
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    const bool probe = GetEnvironmentVariableA("LEONIDA_PROBE", nullptr, 0) != 0;
    Ref<ID3D12Device> ours;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    const std::vector<Row> mine = run(ours.p);
    std::vector<std::vector<Row>> hosts;
    std::vector<std::string> host_names;
    Ref<IDXGIFactory4> factory;
    if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            hosts.push_back(run(host.p));
            host_names.push_back(name);
        }
    }
    int mismatches = 0, compared = 0;
    // SetStablePowerState: every host device is a new one, since a refused call removes it.
    {
        bool any = false, all_accept = true;
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string name;
            if (!factory.p || !hostadapter::reference(factory.p, vk, warp, &adapter.p, name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            any = true;
            if (FAILED(static_cast<HRESULT>(stable_power(host.p)))) all_accept = false;
        }
        if (any && all_accept) {
            ++compared;
            if (stable_power(ours.p) != 0) {
                ++mismatches;
                std::fprintf(stderr, "device: SetStablePowerState: layer refuses, the hosts accept\n");
            }
        } else if (any) {
            std::printf("SetStablePowerState: a host refuses it (no Developer Mode), row not compared\n");
        }
    }
    for (std::size_t i = 0; i < mine.size(); ++i) {
        bool hosts_agree = !hosts.empty();
        for (const auto& h : hosts) hosts_agree = hosts_agree && h[i].code == hosts[0][i].code && h[i].code >= 0;
        const bool differs = hosts_agree && mine[i].code != hosts[0][i].code;
        if (probe) {
            std::printf("%s | layer %llx", mine[i].name.c_str(), static_cast<unsigned long long>(mine[i].code));
            for (const auto& h : hosts) std::printf(" | host %llx", static_cast<unsigned long long>(h[i].code));
            std::printf("%s\n", differs ? "  <-- DIFF" : hosts_agree ? "" : "  (hosts differ)");
        }
        if (!hosts_agree) continue;
        ++compared;
        if (differs) {
            ++mismatches;
            if (!probe)
                std::fprintf(stderr, "%s: layer %llx, host %llx\n", mine[i].name.c_str(),
                             static_cast<unsigned long long>(mine[i].code),
                             static_cast<unsigned long long>(hosts[0][i].code));
        }
    }
    std::printf("object methods: %d rows compared with %zu host adapters, %d differ\n", compared, hosts.size(), mismatches);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return mismatches ? 1 : 0;
}
