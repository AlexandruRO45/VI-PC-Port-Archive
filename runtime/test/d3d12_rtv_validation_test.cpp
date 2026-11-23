// Host-parity validation tests for ID3D12Device::CreateRenderTargetView across typed, typeless, MSAA,
// 1D/2D/3D/array, depth, and buffer resources. No game data or shaders.
// Run on a Vulkan GPU (returns 77 if unavailable); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) {
            p->Release();
            p = nullptr;
        }
    }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

D3D12_RESOURCE_DESC tex_desc(D3D12_RESOURCE_DIMENSION dim, DXGI_FORMAT fmt, UINT64 w, UINT h, UINT16 depth_or_layers,
                             UINT16 mips, UINT samples, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = depth_or_layers;
    d.MipLevels = mips;
    d.Format = fmt;
    d.SampleDesc.Count = samples;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC buf_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

bool create_resource(ID3D12Device* device, const D3D12_RESOURCE_DESC& rd, D3D12_RESOURCE_STATES state,
                     Ref<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                     IID_PPV_ARGS(&out.p)));
}

UINT count_and_clear_errors(ID3D12InfoQueue* iq) {
    if (!iq) return 0;
    UINT errors = 0;
    const UINT64 n = iq->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T len = 0;
        iq->GetMessage(i, nullptr, &len);
        std::vector<char> storage(len);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(iq->GetMessage(i, msg, &len)) &&
            (msg->Severity == D3D12_MESSAGE_SEVERITY_ERROR || msg->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)) {
            ++errors;
        }
    }
    iq->ClearStoredMessages();
    return errors;
}

// Verify a valid RTV creates cleanly, clears on the GPU, and produces 0 host debug errors.
bool verify_valid_on(ID3D12Device* device, ID3D12InfoQueue* iq, const char* label, const D3D12_RESOURCE_DESC& rd,
                     const D3D12_RENDER_TARGET_VIEW_DESC* desc) {
    if (iq) iq->ClearStoredMessages();
    Ref<ID3D12Resource> res;
    if (!create_resource(device, rd, D3D12_RESOURCE_STATE_RENDER_TARGET, res)) {
        std::fprintf(stderr, "valid [%s]: resource creation failed\n", label);
        return false;
    }
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(res.p, desc, handle);
    if (iq) {
        const UINT errs = count_and_clear_errors(iq);
        if (errs != 0) {
            std::fprintf(stderr, "valid [%s]: host debug layer reported %u error(s)\n", label, errs);
            return false;
        }
    }
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    const FLOAT color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    list->ClearRenderTargetView(handle, color, 0, nullptr);
    const HRESULT close_hr = list->Close();
    if (close_hr != S_OK) {
        std::fprintf(stderr, "valid [%s]: list->Close() returned 0x%08lx\n", label, static_cast<unsigned long>(close_hr));
        return false;
    }
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event ev;
    CHECK(ev.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, ev.handle)));
    CHECK(WaitForSingleObject(ev.handle, 5000) == WAIT_OBJECT_0);
    if (iq) {
        const UINT errs = count_and_clear_errors(iq);
        if (errs != 0) {
            std::fprintf(stderr, "valid [%s]: host debug layer reported %u error(s) after execute\n", label, errs);
            return false;
        }
    }
    return true;
}

// Clear a single-sample 2D/2DArray/3D resource through an RTV and read back the specified subresource.
bool clear_and_readback_on(ID3D12Device* device, ID3D12InfoQueue* iq, const char* label, const D3D12_RESOURCE_DESC& rd,
                           const D3D12_RENDER_TARGET_VIEW_DESC* desc, UINT read_subresource, DXGI_FORMAT copy_fmt,
                           std::vector<unsigned char>& out) {
    if (iq) iq->ClearStoredMessages();
    Ref<ID3D12Resource> res;
    if (!create_resource(device, rd, D3D12_RESOURCE_STATE_RENDER_TARGET, res)) {
        std::fprintf(stderr, "readback [%s]: resource creation failed\n", label);
        return false;
    }
    Ref<ID3D12DescriptorHeap> heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(res.p, desc, handle);
    if (iq) {
        const UINT errs = count_and_clear_errors(iq);
        if (errs != 0) {
            std::fprintf(stderr, "readback [%s]: host debug layer reported %u error(s)\n", label, errs);
            return false;
        }
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT num_rows = 0;
    UINT64 row_bytes = 0, total_bytes = 0;
    D3D12_RESOURCE_DESC copy_rd = rd;
    copy_rd.Format = copy_fmt;
    device->GetCopyableFootprints(&copy_rd, read_subresource, 1, 0, &fp, &num_rows, &row_bytes, &total_bytes);
    Ref<ID3D12Resource> rb;
    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    const auto rb_desc = buf_desc(total_bytes);
    CHECK(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb.p))));

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const FLOAT color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    list->ClearRenderTargetView(handle, color, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = res.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = rb.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = res.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = read_subresource;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    const HRESULT close_hr = list->Close();
    if (close_hr != S_OK) {
        std::fprintf(stderr, "readback [%s]: list->Close() returned 0x%08lx\n", label, static_cast<unsigned long>(close_hr));
        return false;
    }
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event ev;
    CHECK(ev.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, ev.handle)));
    CHECK(WaitForSingleObject(ev.handle, 5000) == WAIT_OBJECT_0);
    if (iq) {
        const UINT errs = count_and_clear_errors(iq);
        if (errs != 0) {
            std::fprintf(stderr, "readback [%s]: host debug layer reported %u error(s) after execute\n", label, errs);
            return false;
        }
    }

    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total_bytes)};
    CHECK(SUCCEEDED(rb->Map(0, &range, &mapped)));
    const UINT depth = fp.Footprint.Depth;
    const UINT pitch = fp.Footprint.RowPitch;
    out.resize(static_cast<std::size_t>(row_bytes) * num_rows * depth);
    for (UINT z = 0; z < depth; ++z) {
        for (UINT y = 0; y < num_rows; ++y) {
            const auto* src_row = static_cast<const unsigned char*>(mapped) +
                                  static_cast<std::size_t>(z * num_rows + y) * pitch;
            std::memcpy(out.data() + static_cast<std::size_t>(z * num_rows + y) * row_bytes, src_row,
                        static_cast<std::size_t>(row_bytes));
        }
    }
    const D3D12_RANGE empty{0, 0};
    rb->Unmap(0, &empty);
    return true;
}

bool create_host_warp(Ref<ID3D12Device>& host, Ref<ID3D12InfoQueue>& iq) {
    iq.reset();
    host.reset();
    if (hostwine::host_is_wine()) return false;
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        host.reset();
        return false;
    }
    host->QueryInterface(IID_PPV_ARGS(&iq.p));
    return true;
}

bool ensure_host_warp(Ref<ID3D12Device>& host, Ref<ID3D12InfoQueue>& iq) {
    if (host.p && host->GetDeviceRemovedReason() == S_OK) return true;
    return create_host_warp(host, iq);
}

// Verify an invalid RTV produces a null TargetView on ours (overwriting a valid seed RTV, failing
// ClearRenderTargetView/Close with E_INVALIDARG and 0 Vulkan validation errors), and on host WARP
// (when available) either removes the WARP device or produces at least one D3D12 debug-layer error.
// If WARP removes the device, recreate it cleanly before the next case.
bool verify_invalid(ID3D12Device* ours, Ref<ID3D12Device>& host, Ref<ID3D12InfoQueue>& iq, bool host_available,
                    const char* label, const D3D12_RESOURCE_DESC& rd, D3D12_RESOURCE_STATES initial_state,
                    const D3D12_RENDER_TARGET_VIEW_DESC* desc) {
    // 1. Vulkan backend check: seed slot with a valid RTV first so we verify overwrite with null TargetView.
    {
        Ref<ID3D12Resource> seed_res, target_res;
        const auto seed_rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1,
                                      D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        CHECK(create_resource(ours, seed_rd, D3D12_RESOURCE_STATE_RENDER_TARGET, seed_res));
        if (!create_resource(ours, rd, initial_state, target_res)) {
            std::fprintf(stderr, "invalid [%s]: ours resource creation failed\n", label);
            return false;
        }
        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(ours->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
        const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        ours->CreateRenderTargetView(seed_res.p, nullptr, handle);
        ours->CreateRenderTargetView(target_res.p, desc, handle);

        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(ours->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(ours->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                IID_PPV_ARGS(&list.p))));
        const FLOAT color[4] = {1, 1, 1, 1};
        list->ClearRenderTargetView(handle, color, 0, nullptr);
        const HRESULT close_hr = list->Close();
        if (close_hr != E_INVALIDARG) {
            std::fprintf(stderr, "invalid [%s]: expected E_INVALIDARG from ours list->Close(), got 0x%08lx\n",
                         label, static_cast<unsigned long>(close_hr));
            return false;
        }
    }
    // 2. Host WARP parity check.
    if (host_available) {
        if (!ensure_host_warp(host, iq)) {
            std::fprintf(stderr, "invalid [%s]: failed to recreate host WARP device\n", label);
            return false;
        }
        HRESULT removed = S_OK;
        UINT errs = 0;
        {
            if (iq.p) iq->ClearStoredMessages();
            Ref<ID3D12Resource> host_res;
            if (!create_resource(host.p, rd, initial_state, host_res)) {
                std::fprintf(stderr, "invalid [%s]: host resource creation failed\n", label);
                return false;
            }
            Ref<ID3D12DescriptorHeap> heap;
            const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            CHECK(SUCCEEDED(host->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))));
            const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
            if (iq.p) iq->ClearStoredMessages();
            host->CreateRenderTargetView(host_res.p, desc, handle);
            errs = count_and_clear_errors(iq.p);
            removed = host->GetDeviceRemovedReason();
        }
        if (removed != S_OK) {
            if (!create_host_warp(host, iq)) {
                std::fprintf(stderr, "invalid [%s]: failed to recreate host WARP device after removal (0x%08lx)\n",
                             label, static_cast<unsigned long>(removed));
                return false;
            }
        } else if (iq.p && errs == 0) {
            std::fprintf(stderr, "invalid [%s]: host WARP debug layer accepted view that ours rejected\n", label);
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;

    Ref<ID3D12Debug> debug;
    const bool debug_on = !hostwine::host_is_wine() && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug.p)));
    if (debug_on) debug->EnableDebugLayer();

    Ref<ID3D12Device> ours, host;
    Ref<ID3D12InfoQueue> iq;
    unsigned valid_cases = 0, invalid_cases = 0;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        const bool host_available = create_host_warp(host, iq);

        const auto run_valid = [&](const char* label, const D3D12_RESOURCE_DESC& rd,
                                   const D3D12_RENDER_TARGET_VIEW_DESC* desc) -> bool {
            if (!verify_valid_on(ours.p, nullptr, label, rd, desc)) return false;
            if (host_available) {
                if (!ensure_host_warp(host, iq)) return false;
                if (!verify_valid_on(host.p, iq.p, label, rd, desc)) return false;
            }
            ++valid_cases;
            return true;
        };

        const auto run_valid_readback = [&](const char* label, const D3D12_RESOURCE_DESC& rd,
                                            const D3D12_RENDER_TARGET_VIEW_DESC* desc, UINT read_subresource,
                                            DXGI_FORMAT copy_fmt) -> bool {
            std::vector<unsigned char> actual, expected;
            if (!clear_and_readback_on(ours.p, nullptr, label, rd, desc, read_subresource, copy_fmt, actual))
                return false;
            if (host_available) {
                if (!ensure_host_warp(host, iq)) return false;
                if (!clear_and_readback_on(host.p, iq.p, label, rd, desc, read_subresource, copy_fmt, expected))
                    return false;
                if (actual.size() != expected.size()) {
                    std::fprintf(stderr, "readback [%s]: size mismatch (%zu vs %zu)\n", label, actual.size(),
                                 expected.size());
                    return false;
                }
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (std::abs(int(actual[i]) - int(expected[i])) > 1) {
                        std::fprintf(stderr, "readback [%s]: byte %zu mismatch (ours %u vs WARP %u)\n", label, i,
                                     unsigned(actual[i]), unsigned(expected[i]));
                        return false;
                    }
                }
            }
            ++valid_cases;
            return true;
        };

        const auto run_invalid = [&](const char* label, const D3D12_RESOURCE_DESC& rd,
                                     D3D12_RESOURCE_STATES state,
                                     const D3D12_RENDER_TARGET_VIEW_DESC* desc) -> bool {
            if (!verify_invalid(ours.p, host, iq, host_available, label, rd, state, desc)) return false;
            ++invalid_cases;
            return true;
        };

        constexpr auto kRT = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        constexpr auto kDS = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        constexpr auto kNone = D3D12_RESOURCE_FLAG_NONE;

        // --- Valid RTVs across formats and dimensions ---
        for (const auto fmt : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT,
                               DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT}) {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, fmt, 8, 8, 1, 2, 1, kRT);
            if (!run_valid("2D nullptr desc", rd, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = fmt;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            d.Texture2D = {1, 0};
            if (!run_valid("2D explicit desc", rd, &d)) return 1;
            d.Format = DXGI_FORMAT_UNKNOWN;
            if (!run_valid("2D UNKNOWN format inherits typed", rd, &d)) return 1;
        }
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_TYPELESS, 8, 8, 1, 1, 1, kRT);
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            if (!run_valid("2D typeless -> UNORM", rd, &d)) return 1;
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            if (!run_valid("2D typeless -> SRGB", rd, &d)) return 1;
        }
        // Typed UNORM <-> UNORM_SRGB RTV view casts on RGBA8 and BGRA8 (valid on D3D12 WARP & native HW).
        {
            const auto rdbgra = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_B8G8R8A8_UNORM, 4, 4, 1, 1, 1, kRT);
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            d.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
            if (!run_valid_readback("2D BGRA8_UNORM -> BGRA8_SRGB", rdbgra, &d, 0, DXGI_FORMAT_B8G8R8A8_UNORM)) return 1;

            const auto rdbgra_srgb = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4, 4, 1, 1, 1, kRT);
            d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            if (!run_valid_readback("2D BGRA8_SRGB -> BGRA8_UNORM", rdbgra_srgb, &d, 0, DXGI_FORMAT_B8G8R8A8_UNORM)) return 1;

            const auto rdrgba = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, kRT);
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            if (!run_valid_readback("2D RGBA8_UNORM -> RGBA8_SRGB", rdrgba, &d, 0, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;

            const auto rdrgba_srgb = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4, 4, 1, 1, 1, kRT);
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            if (!run_valid_readback("2D RGBA8_SRGB -> RGBA8_UNORM", rdrgba_srgb, &d, 0, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;
        }
        // 1-sample TEXTURE2DMS and TEXTURE2DMSARRAY RTVs on SampleDesc.Count==1 TEXTURE2D (valid on D3D12).
        {
            const auto rd1x = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 2, 2, 1, kRT);
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            if (!run_valid_readback("2D 1x TEXTURE2DMS RTV", rd1x, &d, 0, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            d.Texture2DMSArray = {1, 1};
            // Subresource index for array slice 1, mip 0 with 2 mips is 2.
            if (!run_valid_readback("2D 1x TEXTURE2DMSARRAY RTV", rd1x, &d, 2, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;
        }
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 4, 3, 1, kRT);
            if (!run_valid("2DArray nullptr desc", rd, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            d.Texture2DArray = {1, 1, 2, 0};
            if (!run_valid("2DArray explicit desc", rd, &d)) return 1;
        }
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 1, 1, 4, kRT);
            if (!run_valid("2DMS nullptr desc", rd, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            if (!run_valid("2DMS explicit desc", rd, &d)) return 1;
        }
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT, 8, 8, 4, 1, 4, kRT);
            if (!run_valid("2DMSArray nullptr desc", rd, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R32_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            d.Texture2DMSArray = {1, 2};
            if (!run_valid("2DMSArray explicit desc", rd, &d)) return 1;
        }
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 4, 3, 1, kRT);
            if (!run_valid("3D nullptr desc", rd, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Texture3D = {1, 1, 1};
            if (!run_valid("3D mip 1 slice 1", rd, &d)) return 1;
            d.Texture3D = {1, 0, UINT_MAX};
            if (!run_valid("3D WSize=UINT_MAX", rd, &d)) return 1;
        }
        // Typeless 3D resource with family-compatible typed RTVs (valid on D3D12).
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, 4, 4, 2, 1, kRT);
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.Texture3D = {0, 0, UINT_MAX};
            if (!run_valid_readback("3D typeless -> UNORM full", rd, &d, 0, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            d.Texture3D = {1, 1, 1};
            if (!run_valid_readback("3D typeless -> SRGB mip 1 slice 1", rd, &d, 1, DXGI_FORMAT_R8G8B8A8_UNORM)) return 1;
        }

        // --- Invalid RTVs ---
        // 1. Missing ALLOW_RENDER_TARGET
        for (const auto fmt : {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM}) {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, fmt, 8, 8, 1, 1, 1, kNone);
            if (!run_invalid("no ALLOW_RENDER_TARGET nullptr", rd, D3D12_RESOURCE_STATE_COMMON, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = fmt;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            if (!run_invalid("no ALLOW_RENDER_TARGET explicit", rd, D3D12_RESOURCE_STATE_COMMON, &d)) return 1;
        }

        // 2. Depth-stencil-only texture
        {
            const auto rd = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_D32_FLOAT, 8, 8, 1, 1, 1, kDS);
            if (!run_invalid("depth-stencil nullptr", rd, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_D32_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            if (!run_invalid("depth-stencil D32 desc", rd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &d)) return 1;
            d.Format = DXGI_FORMAT_R32_FLOAT;
            if (!run_invalid("depth-stencil R32 desc", rd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &d)) return 1;
        }

        // 3. Buffer resource
        {
            const auto rd = buf_desc(256);
            if (!run_invalid("buffer nullptr", rd, D3D12_RESOURCE_STATE_COMMON, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R32_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_BUFFER;
            d.Buffer = {0, 16};
            if (!run_invalid("buffer explicit RTV", rd, D3D12_RESOURCE_STATE_COMMON, &d)) return 1;
        }

        // 4. Dimension & MSAA mismatches
        {
            const auto rd2d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 2, 2, 1, kRT);
            for (const auto vdim : {D3D12_RTV_DIMENSION_UNKNOWN, D3D12_RTV_DIMENSION_BUFFER,
                                    D3D12_RTV_DIMENSION_TEXTURE1D, D3D12_RTV_DIMENSION_TEXTURE1DARRAY,
                                    D3D12_RTV_DIMENSION_TEXTURE3D}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                d.ViewDimension = vdim;
                if (!run_invalid("2D 1x dim mismatch", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d))
                    return 1;
            }

            const auto rdms = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT, 8, 8, 2, 1, 4, kRT);
            for (const auto vdim : {D3D12_RTV_DIMENSION_TEXTURE2D, D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
                                    D3D12_RTV_DIMENSION_TEXTURE1D, D3D12_RTV_DIMENSION_TEXTURE3D}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = DXGI_FORMAT_R32_FLOAT;
                d.ViewDimension = vdim;
                if (!run_invalid("2D 4x MSAA dim mismatch", rdms, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            }

            const auto rd1d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, DXGI_FORMAT_R16G16B16A16_FLOAT, 16, 1, 2, 2, 1, kNone);
            if (!run_invalid("1D no ALLOW_RENDER_TARGET nullptr", rd1d, D3D12_RESOURCE_STATE_COMMON, nullptr)) return 1;
            for (const auto vdim : {D3D12_RTV_DIMENSION_TEXTURE1D, D3D12_RTV_DIMENSION_TEXTURE2D,
                                    D3D12_RTV_DIMENSION_TEXTURE2DARRAY, D3D12_RTV_DIMENSION_TEXTURE3D}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                d.ViewDimension = vdim;
                if (!run_invalid("1D non-RT or dim mismatch", rd1d, D3D12_RESOURCE_STATE_COMMON, &d)) return 1;
            }

            const auto rd3d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 4, 2, 1, kRT);
            for (const auto vdim : {D3D12_RTV_DIMENSION_TEXTURE2D, D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
                                    D3D12_RTV_DIMENSION_TEXTURE1D}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                d.ViewDimension = vdim;
                if (!run_invalid("3D dim mismatch", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            }
        }

        // 5. Format mismatches, depth/BC/typeless view formats, UNKNOWN on typeless
        {
            const auto rd16f = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 1, 1, 1, kRT);
            for (const auto bad_fmt : {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_UINT,
                                       DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_D32_FLOAT,
                                       DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_A8_UNORM}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = bad_fmt;
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                if (!run_invalid("R16G16B16A16_FLOAT bad view format", rd16f, D3D12_RESOURCE_STATE_RENDER_TARGET, &d))
                    return 1;
            }

            const auto rd32f = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT, 8, 8, 1, 1, 1, kRT);
            for (const auto bad_fmt : {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = bad_fmt;
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                if (!run_invalid("R32_FLOAT bad view format", rd32f, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            }

            const auto rdbgra = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_B8G8R8A8_UNORM, 8, 8, 1, 1, 1, kRT);
            for (const auto bad_fmt : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_TYPELESS}) {
                D3D12_RENDER_TARGET_VIEW_DESC d{};
                d.Format = bad_fmt;
                d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                if (!run_invalid("B8G8R8A8_UNORM bad view format", rdbgra, D3D12_RESOURCE_STATE_RENDER_TARGET, &d))
                    return 1;
            }

            const auto rdtypeless = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_TYPELESS, 8, 8, 1, 1, 1, kRT);
            if (!run_invalid("typeless nullptr desc", rdtypeless, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr)) return 1;
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            d.Format = DXGI_FORMAT_UNKNOWN;
            if (!run_invalid("typeless UNKNOWN format", rdtypeless, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Format = DXGI_FORMAT_R16G16B16A16_TYPELESS;
            if (!run_invalid("typeless typeless format", rdtypeless, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Format = DXGI_FORMAT_R32_FLOAT;
            if (!run_invalid("typeless wrong family", rdtypeless, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;

            const auto rd3d_typeless = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R8G8B8A8_TYPELESS, 4, 4, 4, 1, 1, kRT);
            if (!run_invalid("3D typeless nullptr desc", rd3d_typeless, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr)) return 1;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Texture3D = {0, 0, UINT_MAX};
            d.Format = DXGI_FORMAT_UNKNOWN;
            if (!run_invalid("3D typeless UNKNOWN format", rd3d_typeless, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Format = DXGI_FORMAT_R32_FLOAT;
            if (!run_invalid("3D typeless wrong family", rd3d_typeless, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
        }

        // 6. Subresource bounds on typed non-RGBA8 resources (2D, 2DArray, 2DMSArray, 1D, 1DArray, 3D)
        {
            const auto rd2d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 4, 3, 1, kRT);
            D3D12_RENDER_TARGET_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            d.Texture2D = {3, 0};
            if (!run_invalid("2D MipSlice OOB", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2D = {UINT_MAX, 0};
            if (!run_invalid("2D MipSlice UINT_MAX", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2D = {0, 1};
            if (!run_invalid("2D PlaneSlice != 0", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;

            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            d.Texture2DArray = {3, 0, 1, 0};
            if (!run_invalid("2DArray MipSlice OOB", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DArray = {0, 0, 1, 1};
            if (!run_invalid("2DArray PlaneSlice != 0", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DArray = {0, 0, 0, 0};
            if (!run_invalid("2DArray ArraySize == 0", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DArray = {0, 4, 1, 0};
            if (!run_invalid("2DArray FirstArraySlice OOB", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DArray = {0, 3, 2, 0};
            if (!run_invalid("2DArray slice range OOB", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DArray = {0, UINT_MAX, 2, 0};
            if (!run_invalid("2DArray overflow FirstArraySlice", rd2d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;

            const auto rdms = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT, 8, 8, 4, 1, 4, kRT);
            d.Format = DXGI_FORMAT_R32_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            d.Texture2DMSArray = {0, 0};
            if (!run_invalid("2DMSArray ArraySize == 0", rdms, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DMSArray = {4, 1};
            if (!run_invalid("2DMSArray FirstArraySlice OOB", rdms, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DMSArray = {3, 2};
            if (!run_invalid("2DMSArray slice range OOB", rdms, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture2DMSArray = {UINT_MAX, 1};
            if (!run_invalid("2DMSArray overflow FirstArraySlice", rdms, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;

            const auto rd3d = tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 8, 4, 3, 1, kRT);
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            d.Texture3D = {3, 0, 1};
            if (!run_invalid("3D MipSlice OOB", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture3D = {0, 0, 0};
            if (!run_invalid("3D WSize == 0", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            // Mip 1 of depth 4 has depth 2, so FirstWSlice=2 or {1, 2} is out of bounds.
            d.Texture3D = {1, 2, 1};
            if (!run_invalid("3D FirstWSlice >= mip depth", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture3D = {1, 1, 2};
            if (!run_invalid("3D FirstWSlice + WSize > mip depth", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
            d.Texture3D = {0, UINT_MAX, 1};
            if (!run_invalid("3D overflow FirstWSlice", rd3d, D3D12_RESOURCE_STATE_RENDER_TARGET, &d)) return 1;
        }
    }

    if (vk.validation_errors()) return 1;
    std::printf("RTV validation parity: %u valid cases cleared cleanly, %u invalid cases rejected (host WARP: %s, debug layer: %s, Vulkan validation errors: %u)\n",
                valid_cases, invalid_cases, host.p ? "matched" : "unavailable", debug_on ? "verified" : "unavailable",
                vk.validation_errors());
    return 0;
}
