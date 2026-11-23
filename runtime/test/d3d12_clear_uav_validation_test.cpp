// Recording validation and closed-list parity for ClearUnorderedAccessViewUint / ClearUnorderedAccessViewFloat.
//
// Compares the Vulkan-backed D3D12 layer against live Windows WARP and the native hardware D3D12 adapter (when not
// running under Wine), and runs a positive GPU readback sanity check on both buffer and texture UAV clears with
// Vulkan validation active. No game data.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return false; } \
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

D3D12_RESOURCE_DESC make_buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC make_texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

// Minimal foreign ID3D12Resource stub (not registered in the device's VulkanContext).
struct ForeignResource final : ID3D12Resource {
    D3D12_RESOURCE_DIMENSION dim = D3D12_RESOURCE_DIMENSION_UNKNOWN;
    explicit ForeignResource(D3D12_RESOURCE_DIMENSION d = D3D12_RESOURCE_DIMENSION_UNKNOWN) : dim(d) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE Map(UINT, const D3D12_RANGE*, void**) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override {
        D3D12_RESOURCE_DESC d = make_buffer_desc(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        d.Dimension = dim;
        return d;
    }
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return 0; }
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT, const D3D12_BOX*, const void*, UINT, UINT) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void*, UINT, UINT, UINT, const D3D12_BOX*) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS*) override {
        return E_NOTIMPL;
    }
};

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12Resource> buf_uav, buf_typed_uav, buf_no_uav;
    Ref<ID3D12Resource> tex_uav, tex_no_uav;
    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_buf_raw{}, cpu_buf_typed{}, cpu_tex{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_buf_raw{}, gpu_buf_typed{}, gpu_tex{};

    bool init(ID3D12Device* dev) {
        device = dev;
        const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto create = [&](const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                                Ref<ID3D12Resource>& out) {
            return SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };
        CHECK(create(make_buffer_desc(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, buf_uav));
        CHECK(create(make_buffer_desc(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, buf_typed_uav));
        CHECK(create(make_buffer_desc(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, buf_no_uav));
        CHECK(create(make_texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, tex_uav));
        CHECK(create(make_texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                     D3D12_RESOURCE_STATE_COMMON, tex_no_uav));

        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpu_heap.p))));
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpu_heap.p))));

        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto cpu0 = cpu_heap->GetCPUDescriptorHandleForHeapStart();
        const auto gpu_cpu0 = gpu_heap->GetCPUDescriptorHandleForHeapStart();
        const auto gpu0 = gpu_heap->GetGPUDescriptorHandleForHeapStart();
        cpu_buf_raw = cpu0;
        cpu_buf_typed = {cpu0.ptr + inc};
        cpu_tex = {cpu0.ptr + 2 * inc};
        gpu_buf_raw = gpu0;
        gpu_buf_typed = {gpu0.ptr + inc};
        gpu_tex = {gpu0.ptr + 2 * inc};

        D3D12_UNORDERED_ACCESS_VIEW_DESC raw_desc{};
        raw_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        raw_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        raw_desc.Buffer.NumElements = 64;
        raw_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(buf_uav.p, nullptr, &raw_desc, cpu_buf_raw);
        device->CreateUnorderedAccessView(buf_uav.p, nullptr, &raw_desc, gpu_cpu0);

        D3D12_UNORDERED_ACCESS_VIEW_DESC typed_desc{};
        typed_desc.Format = DXGI_FORMAT_R32_FLOAT;
        typed_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        typed_desc.Buffer.NumElements = 64;
        device->CreateUnorderedAccessView(buf_typed_uav.p, nullptr, &typed_desc, cpu_buf_typed);
        device->CreateUnorderedAccessView(buf_typed_uav.p, nullptr, &typed_desc, {gpu_cpu0.ptr + inc});

        D3D12_UNORDERED_ACCESS_VIEW_DESC tex_desc{};
        tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tex_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(tex_uav.p, nullptr, &tex_desc, cpu_tex);
        device->CreateUnorderedAccessView(tex_uav.p, nullptr, &tex_desc, {gpu_cpu0.ptr + 2 * inc});
        return true;
    }

    template<class Body>
    HRESULT record_close(D3D12_COMMAND_LIST_TYPE type, Body&& body) const {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&alloc.p))) ||
            FAILED(device->CreateCommandList(0, type, alloc.p, nullptr, IID_PPV_ARGS(&list.p))))
            return E_UNEXPECTED;
        if (type != D3D12_COMMAND_LIST_TYPE_COPY) {
            ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
            list->SetDescriptorHeaps(1, heaps);
        }
        body(list.p);
        return list->Close();
    }
};

// Probes safe host parity cases (valid clears, gpu.ptr == 0, compute/bundle lists, closed-list no-op) on a live D3D12
// device (WARP or hardware). Recreates the WARP device if GetDeviceRemovedReason() != S_OK.
bool check_host_parity(const char* label, bool is_warp, IDXGIAdapter* adapter) {
    Ref<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev.p)))) {
        std::printf("%s: D3D12CreateDevice unavailable, skipping\n", label);
        return true;
    }
    Env e;
    CHECK(e.init(dev.p));
    auto ensure_alive = [&]() -> bool {
        if (dev->GetDeviceRemovedReason() == S_OK) return true;
        if (!is_warp) return false;
        e = Env{};
        dev.reset();
        CHECK(SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev.p))));
        return e.init(dev.p);
    };

    const UINT uval[4] = {0x11223344u, 0x55667788u, 0x99aabbcdu, 0xdeadbeefu};
    const FLOAT fval[4] = {0.25f, 0.5f, 0.75f, 1.0f};

    // Valid buffer and texture clears (both with valid GPU handle and with gpu.ptr == 0).
    CHECK(SUCCEEDED(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
        l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
        l->ClearUnorderedAccessViewFloat(e.gpu_buf_typed, e.cpu_buf_typed, e.buf_typed_uav.p, fval, 0, nullptr);
        l->ClearUnorderedAccessViewUint(e.gpu_tex, e.cpu_tex, e.tex_uav.p, uval, 0, nullptr);
        l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
    })));
    CHECK(ensure_alive());

    CHECK(SUCCEEDED(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
        l->ClearUnorderedAccessViewUint({0}, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
        l->ClearUnorderedAccessViewFloat({0}, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
    })));
    CHECK(ensure_alive());

    // Compute list accepts UAV clears; bundle refuses them.
    CHECK(SUCCEEDED(e.record_close(D3D12_COMMAND_LIST_TYPE_COMPUTE, [&](ID3D12GraphicsCommandList* l) {
        l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
    })));
    CHECK(FAILED(e.record_close(D3D12_COMMAND_LIST_TYPE_BUNDLE, [&](ID3D12GraphicsCommandList* l) {
        l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
    })));
    CHECK(FAILED(e.record_close(D3D12_COMMAND_LIST_TYPE_BUNDLE, [&](ID3D12GraphicsCommandList* l) {
        l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
    })));
    CHECK(ensure_alive());

    // Closed list: calling ClearUnorderedAccessViewUint/Float (even with null pointers) is a no-op.
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
        CHECK(list->Close() == S_OK);
        list->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, nullptr, nullptr, 0, nullptr);
        list->ClearUnorderedAccessViewFloat({0}, {0}, nullptr, nullptr, 0, nullptr);
        CHECK(list->Close() == E_FAIL);
        CHECK(list->Reset(alloc.p, nullptr) == S_OK);
        CHECK(list->Close() == S_OK);
    }
    CHECK(ensure_alive());
    std::printf("%s: host parity passed\n", label);
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    const unsigned errors_before = vk.validation_errors();

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;

        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        Env e;
        EXPECT(e.init(device.p));
        ForeignResource foreign_buf;

        const UINT uval[4] = {0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
        const FLOAT fval[4] = {0.25f, 0.5f, 0.75f, 1.0f};

        // 1. Valid buffer & texture clears, including gpu.ptr == 0 (matches empirical WARP & RTX 5090).
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
            l->ClearUnorderedAccessViewFloat(e.gpu_buf_typed, e.cpu_buf_typed, e.buf_typed_uav.p, fval, 0, nullptr);
            l->ClearUnorderedAccessViewUint(e.gpu_tex, e.cpu_tex, e.tex_uav.p, uval, 0, nullptr);
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
        }) == S_OK);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint({0}, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
            l->ClearUnorderedAccessViewFloat({0}, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
        }) == S_OK);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_COMPUTE, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
        }) == S_OK);

        // 2. Bundle refuses both Uint and Float.
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_BUNDLE, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_BUNDLE, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, fval, 0, nullptr);
        }) == E_INVALIDARG);

        // 3. Null resource, null values, zero CPU handle, unknown-dimension resource.
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, nullptr, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, nullptr, fval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, nullptr, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, nullptr, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, {0}, e.buf_uav.p, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, {0}, e.tex_uav.p, fval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, &foreign_buf, uval, 0, nullptr);
        }) == E_INVALIDARG);

        // 4. Resource without D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_no_uav.p, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_no_uav.p, fval, 0, nullptr);
        }) == E_INVALIDARG);

        // 5. Resource/view kind mismatch: buffer V# with texture resource, image T# with buffer resource.
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.tex_uav.p, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_buf_typed, e.cpu_buf_typed, e.tex_uav.p, fval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewUint(e.gpu_tex, e.cpu_tex, e.buf_uav.p, uval, 0, nullptr);
        }) == E_INVALIDARG);
        EXPECT(e.record_close(D3D12_COMMAND_LIST_TYPE_DIRECT, [&](ID3D12GraphicsCommandList* l) {
            l->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.buf_typed_uav.p, fval, 0, nullptr);
        }) == E_INVALIDARG);

        // 6. Closed-list parity: calling ClearUnorderedAccessViewUint/Float on an already-closed list must be a no-op
        //    and must NOT mutate invalid_ (verified via ExecuteBundle on a closed bundle and Reset on a direct list).
        {
            Ref<ID3D12CommandAllocator> b_alloc, d_alloc;
            Ref<ID3D12GraphicsCommandList> bundle, direct;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&b_alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, b_alloc.p, nullptr,
                                                       IID_PPV_ARGS(&bundle.p))));
            EXPECT(bundle->Close() == S_OK);
            bundle->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, nullptr, nullptr, 0, nullptr);
            bundle->ClearUnorderedAccessViewFloat({0}, {0}, e.buf_no_uav.p, fval, 0, nullptr);
            EXPECT(bundle->Close() == E_FAIL);

            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d_alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d_alloc.p, nullptr,
                                                       IID_PPV_ARGS(&direct.p))));
            direct->ExecuteBundle(bundle.p);  // fails if bundle->invalid_ was mutated
            EXPECT(direct->Close() == S_OK);
            direct->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, nullptr, nullptr, 0, nullptr);
            direct->ClearUnorderedAccessViewFloat({0}, {0}, e.tex_uav.p, nullptr, 0, nullptr);
            EXPECT(direct->Close() == E_FAIL);
            EXPECT(direct->Reset(d_alloc.p, nullptr) == S_OK);
            EXPECT(direct->Close() == S_OK);
        }

        // 7. Positive GPU UAV clear sanity check (buffer + texture readback, 0 Vulkan validation errors).
        {
            Ref<ID3D12CommandQueue> queue;
            Ref<ID3D12Fence> fence;
            const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

            Ref<ID3D12Resource> rb_buf, rb_tex;
            const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
            const auto rb_buf_desc = make_buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
            const auto rb_tex_desc = make_buffer_desc(256 * 8, D3D12_RESOURCE_FLAG_NONE);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_buf_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&rb_buf.p))));
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_tex_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&rb_tex.p))));

            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            ID3D12DescriptorHeap* heaps[] = {e.gpu_heap.p};
            list->SetDescriptorHeaps(1, heaps);
            const UINT buf_val[4] = {0x13579bdfu, 0, 0, 0};
            const FLOAT tex_val[4] = {1.0f, 0.0f, 0.5f, 1.0f};
            list->ClearUnorderedAccessViewUint(e.gpu_buf_raw, e.cpu_buf_raw, e.buf_uav.p, buf_val, 0, nullptr);
            list->ClearUnorderedAccessViewFloat(e.gpu_tex, e.cpu_tex, e.tex_uav.p, tex_val, 0, nullptr);

            D3D12_RESOURCE_BARRIER barriers[2]{};
            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Transition = {e.buf_uav.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Transition = {e.tex_uav.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(2, barriers);
            list->CopyBufferRegion(rb_buf.p, 0, e.buf_uav.p, 0, 256);

            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = rb_tex.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = e.tex_uav.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

            EXPECT(list->Close() == S_OK);
            // Calling on the closed list before ExecuteCommandLists must not corrupt or invalidate it.
            list->ClearUnorderedAccessViewUint({0}, {0}, nullptr, nullptr, 0, nullptr);

            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

            void* mapped = nullptr;
            EXPECT(SUCCEEDED(rb_buf->Map(0, nullptr, &mapped)));
            const auto* words = static_cast<const std::uint32_t*>(mapped);
            for (UINT i = 0; i < 64; ++i) EXPECT(words[i] == 0x13579bdfu);
            rb_buf->Unmap(0, nullptr);

            EXPECT(SUCCEEDED(rb_tex->Map(0, nullptr, &mapped)));
            const auto* bytes = static_cast<const std::uint8_t*>(mapped);
            for (UINT y = 0; y < 8; ++y) {
                for (UINT x = 0; x < 8; ++x) {
                    const std::uint8_t* px = bytes + y * 256 + x * 4;
                    EXPECT(px[0] == 0xff && px[1] == 0x00 && px[2] == 0x80 && px[3] == 0xff);
                }
            }
            rb_tex->Unmap(0, nullptr);
        }
    }

    // 8. Live host parity checks against WARP and the native hardware adapter (skipped on Wine).
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter> warp;
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
                EXPECT(check_host_parity("WARP", true, warp.p));
            }
            Ref<IDXGIAdapter1> hw;
            for (UINT i = 0; factory->EnumAdapters1(i, &hw.p) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                hw->GetDesc1(&desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    EXPECT(check_host_parity("Hardware D3D12", false, hw.p));
                    break;
                }
                hw.reset();
            }
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors\n", validation_errors);
        return 1;
    }
    std::printf("d3d12_clear_uav_validation: all layer, closed-list, GPU readback, and host parity checks passed\n");
    return 0;
}
