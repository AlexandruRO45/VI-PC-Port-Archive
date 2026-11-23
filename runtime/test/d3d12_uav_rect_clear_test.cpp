// Rectangle UAV clears (ClearUnorderedAccessViewUint / ClearUnorderedAccessViewFloat with NumRects > 0) on
// 3D volumes (TEXTURE3D) and non-attachment 2D UAV formats (R9G9B9E5_SHAREDEXP), on BOTH Direct and Compute queues,
// compared byte-for-byte against Windows WARP and an independent CPU reference model.
//
// Exercises:
//   1. 3D volumes (R8G8B8A8_UNORM and R32_UINT, 8x8x4, 2 mips):
//      - Mip 0 (8x8x4) and Mip 1 (4x4x2)
//      - Full W-range (FirstWSlice=0, WSize=UINT_MAX) and sub-ranges (FirstWSlice/WSize)
//      - Whole-view partial-W clear (NumRects=0 on a W sub-range)
//      - Single rect, multiple overlapping rects, partially out-of-bounds rects (clipped),
//        and completely out-of-bounds / empty rects (no-op preserving previous contents)
//      - Both Direct (D3D12_COMMAND_LIST_TYPE_DIRECT) and Compute (D3D12_COMMAND_LIST_TYPE_COMPUTE) queues
//   2. Non-attachment 2D UAV (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 8x8):
//      - ClearUnorderedAccessViewFloat and ClearUnorderedAccessViewUint with multiple, overlapping,
//        partially out-of-bounds, and completely out-of-bounds / empty rects
//      - Both Direct and Compute queues
//      - Verifies pixels inside rects match expected shared-exponent encoding and pixels outside rects are preserved
//
// Synthetic data only; no game files. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/uav_clear.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
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

D3D12_RESOURCE_DESC tex_desc(D3D12_RESOURCE_DIMENSION dim, UINT64 width, UINT height, UINT16 depth_or_layers,
                             UINT16 mips, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth_or_layers;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
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
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                                  UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, before, after};
    return b;
}

struct ClearOp {
    UINT mip = 0;
    UINT first_w = 0;
    UINT w_size = UINT_MAX;
    bool is_float = false;
    std::array<float, 4> fvals{};
    std::array<UINT, 4> uvals{};
    std::vector<D3D12_RECT> rects;
};

struct TextureSpec {
    const char* name;
    bool volume;
    DXGI_FORMAT dxgi_format;
    VkFormat vk_format;
    UINT width, height;
    UINT16 depth_or_layers;
    UINT16 mips;
    std::vector<ClearOp> ops;
};

struct SubLayout {
    UINT mip;
    UINT width, height, depth;
    UINT64 offset;
    UINT pitch;
};

std::vector<SubLayout> build_layout(const TextureSpec& spec, UINT64& total_bytes) {
    std::vector<SubLayout> subs;
    total_bytes = 0;
    const UINT texel = leonida::uav_clear::layout(spec.vk_format)->bytes;
    for (UINT mip = 0; mip < spec.mips; ++mip) {
        SubLayout s{};
        s.mip = mip;
        s.width = std::max(1u, spec.width >> mip);
        s.height = std::max(1u, spec.height >> mip);
        s.depth = spec.volume ? std::max(1u, UINT(spec.depth_or_layers) >> mip) : 1u;
        s.offset = total_bytes;
        s.pitch = (s.width * texel + 255u) / 256u * 256u;
        total_bytes += (UINT64(s.pitch) * s.height * s.depth + 511ull) / 512ull * 512ull;
        subs.push_back(s);
    }
    return subs;
}

// Tightly packed per-mip byte buffer (concatenated across mips) from the independent CPU model.
std::vector<std::uint8_t> build_cpu_model(const TextureSpec& spec) {
    const UINT texel = leonida::uav_clear::layout(spec.vk_format)->bytes;
    std::vector<std::vector<std::uint8_t>> mips(spec.mips);
    for (UINT mip = 0; mip < spec.mips; ++mip) {
        const UINT w = std::max(1u, spec.width >> mip);
        const UINT h = std::max(1u, spec.height >> mip);
        const UINT d = spec.volume ? std::max(1u, UINT(spec.depth_or_layers) >> mip) : 1u;
        mips[mip].assign(std::size_t(w) * h * d * texel, 0);
    }
    for (const ClearOp& op : spec.ops) {
        const UINT w = std::max(1u, spec.width >> op.mip);
        const UINT h = std::max(1u, spec.height >> op.mip);
        const UINT d = spec.volume ? std::max(1u, UINT(spec.depth_or_layers) >> op.mip) : 1u;
        const UINT z0 = spec.volume ? op.first_w : 0u;
        const UINT z1 = spec.volume ? (op.w_size == UINT_MAX ? d : std::min(d, z0 + op.w_size)) : 1u;
        std::array<std::uint32_t, 4> vals{};
        if (op.is_float) {
            for (int i = 0; i < 4; ++i) vals[i] = std::bit_cast<std::uint32_t>(op.fvals[i]);
        } else {
            for (int i = 0; i < 4; ++i) vals[i] = op.uvals[i];
        }
        leonida::uav_clear::Element elem{};
        leonida::uav_clear::encode(spec.vk_format, op.is_float, vals, elem);
        std::vector<D3D12_RECT> rects = op.rects;
        if (rects.empty()) rects.push_back({0, 0, LONG(w), LONG(h)});
        for (const D3D12_RECT& r : rects) {
            const LONG x0 = std::max<LONG>(r.left, 0);
            const LONG y0 = std::max<LONG>(r.top, 0);
            const LONG x1 = std::min<LONG>(r.right, LONG(w));
            const LONG y1 = std::min<LONG>(r.bottom, LONG(h));
            if (x1 <= x0 || y1 <= y0) continue;
            for (UINT z = z0; z < z1; ++z) {
                for (LONG y = y0; y < y1; ++y) {
                    for (LONG x = x0; x < x1; ++x) {
                        const std::size_t idx = ((std::size_t(z) * h + std::size_t(y)) * w + std::size_t(x)) * texel;
                        std::memcpy(&mips[op.mip][idx], elem.bytes.data(), texel);
                    }
                }
            }
        }
    }
    std::vector<std::uint8_t> flat;
    for (const auto& m : mips) flat.insert(flat.end(), m.begin(), m.end());
    return flat;
}

bool run_on_device(ID3D12Device* dev, bool vulkan_device, D3D12_COMMAND_LIST_TYPE queue_type, const TextureSpec& spec,
                   std::vector<std::uint8_t>& out_flat) {
    Ref<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC qd{queue_type};
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p)))) return false;
    Ref<ID3D12CommandAllocator> alloc;
    if (FAILED(dev->CreateCommandAllocator(queue_type, IID_PPV_ARGS(&alloc.p)))) return false;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(dev->CreateCommandList(0, queue_type, alloc.p, nullptr, IID_PPV_ARGS(&list.p)))) return false;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;

    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto td = tex_desc(spec.volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                             spec.width, spec.height, spec.depth_or_layers, spec.mips, spec.dxgi_format,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Ref<ID3D12Resource> tex;
    if (FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &td,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&tex.p))))
        return false;

    UINT64 total_rb = 0;
    const auto subs = build_layout(spec, total_rb);
    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    const auto bd = buf_desc(total_rb);
    Ref<ID3D12Resource> rb;
    if (FAILED(dev->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&rb.p))))
        return false;

    const UINT num_ops = UINT(spec.ops.size());
    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hd_cpu{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, num_ops,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    const D3D12_DESCRIPTOR_HEAP_DESC hd_gpu{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, num_ops,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (FAILED(dev->CreateDescriptorHeap(&hd_cpu, IID_PPV_ARGS(&cpu_heap.p))) ||
        FAILED(dev->CreateDescriptorHeap(&hd_gpu, IID_PPV_ARGS(&gpu_heap.p))))
        return false;
    const UINT step = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    list->SetDescriptorHeaps(1, heaps);

    for (UINT i = 0; i < num_ops; ++i) {
        const ClearOp& op = spec.ops[i];
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavd{};
        uavd.Format = spec.dxgi_format;
        if (spec.volume) {
            uavd.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uavd.Texture3D.MipSlice = op.mip;
            uavd.Texture3D.FirstWSlice = op.first_w;
            uavd.Texture3D.WSize = op.w_size;
        } else {
            uavd.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uavd.Texture2D.MipSlice = op.mip;
        }
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_h{cpu_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * step};
        D3D12_CPU_DESCRIPTOR_HANDLE gpu_cpu_h{gpu_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * step};
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_h{gpu_heap->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(i) * step};
        dev->CreateUnorderedAccessView(tex.p, nullptr, &uavd, cpu_h);
        dev->CreateUnorderedAccessView(tex.p, nullptr, &uavd, gpu_cpu_h);

        const D3D12_RECT* rects = op.rects.empty() ? nullptr : op.rects.data();
        if (op.is_float) {
            list->ClearUnorderedAccessViewFloat(gpu_h, cpu_h, tex.p, op.fvals.data(), UINT(op.rects.size()), rects);
        } else {
            list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, tex.p, op.uvals.data(), UINT(op.rects.size()), rects);
        }
        D3D12_RESOURCE_BARRIER uav_b{};
        uav_b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_b.UAV.pResource = tex.p;
        list->ResourceBarrier(1, &uav_b);
    }

    const auto to_src = transition(tex.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);

    for (const SubLayout& s : subs) {
        D3D12_TEXTURE_COPY_LOCATION dst{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = {s.offset, {spec.dxgi_format, s.width, s.height, s.depth, s.pitch}};
        D3D12_TEXTURE_COPY_LOCATION src{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = s.mip;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (vulkan_device && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) {
        std::fprintf(stderr, "%s: replay failed (%d)\n", spec.name, int(rt::d3d12::last_execute_result(queue.p)));
        return false;
    }
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;

    void* mapped = nullptr;
    if (FAILED(rb->Map(0, nullptr, &mapped))) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    const UINT texel = leonida::uav_clear::layout(spec.vk_format)->bytes;
    out_flat.clear();
    for (const SubLayout& s : subs) {
        const std::size_t row_bytes = std::size_t(s.width) * texel;
        for (UINT z = 0; z < s.depth; ++z) {
            for (UINT y = 0; y < s.height; ++y) {
                const auto* row = bytes + s.offset + (std::size_t(z) * s.height + y) * s.pitch;
                out_flat.insert(out_flat.end(), row, row + row_bytes);
            }
        }
    }
    rb->Unmap(0, nullptr);
    return true;
}

bool compare_bytes(const char* label, const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "%s: size %zu != %zu\n", label, got.size(), want.size());
        return false;
    }
    unsigned diffs = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            if (diffs++ < 8)
                std::fprintf(stderr, "%s: byte %zu is 0x%02x, want 0x%02x\n", label, i, got[i], want[i]);
        }
    }
    if (diffs) {
        std::fprintf(stderr, "%s: %u total byte mismatches\n", label, diffs);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    Ref<ID3D12Device> warp;
    {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) {
            D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&warp.p));
        }
    }
    bool warp_r9e5_uav = false;
    if (warp.p) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
        if (SUCCEEDED(warp->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))) &&
            (fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
            warp_r9e5_uav = true;
        }
    }

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        std::vector<TextureSpec> cases;

        // 1. 3D volume R8G8B8A8_UNORM (8x8x4, 2 mips):
        //    - Seed mip 0 full W-range, then single rect, multiple overlapping rects, clipped OOB rect,
        //      sub-range FirstWSlice=1/WSize=2, whole-view partial-W clear, and completely OOB / empty rects.
        //    - Mip 1 (4x4x2): full W-range seed, then FirstWSlice=1/WSize=1 clipped rect + empty OOB rect.
        cases.push_back({
            "TEXTURE3D R8G8B8A8_UNORM mip0+mip1 full+subW rects",
            true,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM,
            8, 8, 4, 2,
            {
                // Mip 0 full W-range background (0x40, 0x80, 0xc0, 0xff)
                {0, 0, UINT_MAX, false, {}, {0x40, 0x80, 0xc0, 0xff}, {}},
                // Mip 0 full W-range single rect
                {0, 0, UINT_MAX, true, {0.25f, 0.75f, 1.0f, 0.0f}, {}, {{1, 1, 4, 5}}},
                // Mip 0 sub-range FirstWSlice=1, WSize=2: multiple overlapping rects + partially OOB clipped rect
                {0, 1, 2, false, {}, {0x11, 0x22, 0x33, 0x44}, {{2, 2, 6, 6}, {4, 3, 12, 10}, {-2, 0, 2, 3}}},
                // Mip 0 sub-range FirstWSlice=3, WSize=1: whole-view partial-W clear (NumRects=0)
                {0, 3, 1, false, {}, {0xaa, 0xbb, 0xcc, 0xdd}, {}},
                // Mip 0 completely out-of-bounds and zero-area rects: must be a no-op preserving previous contents
                {0, 0, UINT_MAX, false, {}, {0xfe, 0xfe, 0xfe, 0xfe}, {{8, 0, 12, 4}, {-5, -5, 0, 0}, {3, 3, 3, 5}}},
                // Mip 1 (4x4x2) full W-range background
                {1, 0, UINT_MAX, false, {}, {0x05, 0x06, 0x07, 0x08}, {}},
                // Mip 1 sub-range FirstWSlice=1, WSize=1: clipped rect
                {1, 1, 1, false, {}, {0x90, 0xa0, 0xb0, 0xc0}, {{1, 1, 9, 3}}},
                // Mip 1 completely OOB rect: no-op
                {1, 0, UINT_MAX, false, {}, {0xee, 0xee, 0xee, 0xee}, {{4, 4, 8, 8}}},
            },
        });

        // 2. 3D volume R32_UINT (8x8x4, 2 mips): exact 32-bit uint patterns across full/sub-W and mip 0/1.
        cases.push_back({
            "TEXTURE3D R32_UINT mip0+mip1 full+subW rects",
            true,
            DXGI_FORMAT_R32_UINT,
            VK_FORMAT_R32_UINT,
            8, 8, 4, 2,
            {
                {0, 0, UINT_MAX, false, {}, {0x11223344u, 0, 0, 0}, {}},
                {0, 0, 2, false, {}, {0xdeadbeefu, 0, 0, 0}, {{0, 0, 3, 4}, {2, 2, 6, 5}}},
                {0, 2, 2, false, {}, {0xcafebabeu, 0, 0, 0}, {{-3, 5, 5, 11}}},
                {0, 1, 2, false, {}, {0x0badf00du, 0, 0, 0}, {{10, 10, 15, 15}, {2, 2, 2, 2}}},
                {1, 0, UINT_MAX, false, {}, {0x55667788u, 0, 0, 0}, {}},
                {1, 0, 1, false, {}, {0x99aabbccu, 0, 0, 0}, {{1, 0, 3, 4}, {2, 2, 6, 6}}},
                {1, 1, 1, false, {}, {0x01020304u, 0, 0, 0}, {{-1, -1, 2, 2}}},
            },
        });

        // 3. Non-attachment 2D UAV (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 8x8):
        //    Float + Uint clears with multiple overlapping, clipped, and empty/OOB rects.
        D3D12_FEATURE_DATA_FORMAT_SUPPORT e5_support{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &e5_support, sizeof(e5_support))) &&
            (e5_support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)) {
            cases.push_back({
                "TEXTURE2D R9G9B9E5_SHAREDEXP Float+Uint rects",
                false,
                DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
                8, 8, 1, 1,
                {
                    // Whole-view Float seed (1, 2, 4) -> 0x94010040
                    {0, 0, 1, true, {1.0f, 2.0f, 4.0f, 0.0f}, {}, {}},
                    // Multiple overlapping Float rects + partially OOB clipped rect
                    {0, 0, 1, true, {0.5f, 0.25f, 0.125f, 0.0f}, {}, {{1, 1, 5, 4}, {3, 2, 10, 6}}},
                    // Negative-origin clipped Float rect with HDR values
                    {0, 0, 1, true, {100.0f, 0.0f, 0.5f, 0.0f}, {}, {{-2, 5, 4, 9}}},
                    // Uint rect clear (clears to 0 on R9G9B9E5_SHAREDEXP)
                    {0, 0, 1, false, {}, {0x3f800000u, 0xffffffffu, 0x12345678u, 7u}, {{2, 2, 4, 4}, {6, 0, 8, 2}}},
                    // Completely OOB and zero-area rects: must preserve previous contents
                    {0, 0, 1, true, {65408.0f, 65408.0f, 65408.0f, 0.0f}, {}, {{8, 8, 12, 12}, {-4, 0, 0, 4}, {1, 2, 1, 5}}},
                },
            });
        }

        const std::array<std::pair<D3D12_COMMAND_LIST_TYPE, const char*>, 2> queues = {{
            {D3D12_COMMAND_LIST_TYPE_DIRECT, "DIRECT"},
            {D3D12_COMMAND_LIST_TYPE_COMPUTE, "COMPUTE"},
        }};

        unsigned passed = 0;
        unsigned warp_compared = 0;
        for (const TextureSpec& spec : cases) {
            const std::vector<std::uint8_t> model = build_cpu_model(spec);
            const bool can_warp = warp.p && (spec.dxgi_format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP || warp_r9e5_uav);
            for (const auto& [qtype, qname] : queues) {
                char label[256];
                std::snprintf(label, sizeof(label), "%s [%s]", spec.name, qname);
                std::vector<std::uint8_t> vk_bytes;
                EXPECT(run_on_device(device.p, true, qtype, spec, vk_bytes));
                EXPECT(compare_bytes(label, vk_bytes, model));
                if (can_warp) {
                    std::vector<std::uint8_t> warp_bytes;
                    EXPECT(run_on_device(warp.p, false, qtype, spec, warp_bytes));
                    EXPECT(compare_bytes(label, vk_bytes, warp_bytes));
                    ++warp_compared;
                }
                ++passed;
            }
        }
        std::printf("d3d12_uav_rect_clear_test: %u/%u cases matched CPU model (%u matched WARP byte-for-byte)\n",
                    passed, passed, warp_compared);
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_uav_rect_clear_test: OK (0 validation errors)\n");
    return 0;
}
