// DXR GetRaytracingAccelerationStructurePrebuildInfo contract test.
// Verifies all six D3D12 triangle vertex formats (including 16-bit formats with 4-byte strides),
// ALLOW_COMPACTION and NO_DUPLICATE_ANYHIT_INVOCATION flag forwarding, and clean refusal of
// unsupported vertex formats with zero Vulkan validation errors.
// Synthetic resources only; no game content.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(Ref&& o) noexcept : p(o.p) { o.p = nullptr; }
    Ref& operator=(Ref&& o) noexcept {
        if (this != &o) {
            if (p) p->Release();
            p = o.p;
            o.p = nullptr;
        }
        return *this;
    }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

bool make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE heap_type,
                 D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                 Ref<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap_type;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size < 256 ? 256 : size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                  IID_PPV_ARGS(out.put())));
}

bool execute_and_wait(ID3D12Device5* dev, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList4* cl) {
    if (FAILED(cl->Close())) return false;
    ID3D12CommandList* lists[] = {cl};
    queue->ExecuteCommandLists(1, lists);
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))) ||
        FAILED(queue->Signal(fence.p, 1)))
        return false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) &&
                    WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    return ok;
}

bool is_zero_prebuild(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& info) {
    return info.ResultDataMaxSizeInBytes == 0 &&
           info.ScratchDataSizeInBytes == 0 &&
           info.UpdateScratchDataSizeInBytes == 0;
}

bool is_khr_mode() {
    char buf[32]{};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_RT_AS", buf, static_cast<DWORD>(sizeof(buf)));
    return n > 0 && n < sizeof(buf) && std::strcmp(buf, "khr") == 0;
}

struct FormatCase {
    const char* name;
    DXGI_FORMAT format;
    UINT64 stride;
};

int run_suite(ID3D12Device* base_dev, const char* name, bool is_leonida) {
    Ref<ID3D12Device5> dev;
    if (FAILED(base_dev->QueryInterface(IID_PPV_ARGS(dev.put())))) {
        std::printf("%s: no ID3D12Device5, skipping\n", name);
        return 77;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5)))) {
        std::printf("%s: OPTIONS5 query failed, skipping\n", name);
        return 77;
    }
    if (!is_leonida && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
        std::printf("%s: raytracing not supported (tier=%d), skipping\n", name, static_cast<int>(o5.RaytracingTier));
        return 77;
    }

    // 1. All 6 D3D12-supported DXR triangle vertex formats (including 16-bit formats with 4-byte strides).
    const FormatCase valid_formats[] = {
        {"R32G32B32_FLOAT", DXGI_FORMAT_R32G32B32_FLOAT, 12},
        {"R32G32_FLOAT", DXGI_FORMAT_R32G32_FLOAT, 8},
        {"R16G16B16A16_FLOAT", DXGI_FORMAT_R16G16B16A16_FLOAT, 8},
        {"R16G16_FLOAT", DXGI_FORMAT_R16G16_FLOAT, 4},
        {"R16G16B16A16_SNORM", DXGI_FORMAT_R16G16B16A16_SNORM, 8},
        {"R16G16_SNORM", DXGI_FORMAT_R16G16_SNORM, 4},
    };

    for (const auto& fc : valid_formats) {
        D3D12_RAYTRACING_GEOMETRY_DESC g{};
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g.Triangles.VertexFormat = fc.format;
        g.Triangles.VertexCount = 300;
        g.Triangles.VertexBuffer.StrideInBytes = fc.stride;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = &g;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO arr_info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &arr_info);
        if (arr_info.ResultDataMaxSizeInBytes == 0 || arr_info.ScratchDataSizeInBytes == 0) {
            if (is_leonida && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
                std::printf("%s: neither XDXR nor KHR acceleration structures supported, skipping\n", name);
                return 77;
            }
            std::fprintf(stderr, "%s: valid vertex format %s (stride %llu) returned zero prebuild\n",
                         name, fc.name, static_cast<unsigned long long>(fc.stride));
            return 1;
        }

        const D3D12_RAYTRACING_GEOMETRY_DESC* g_ptr = &g;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        in.ppGeometryDescs = &g_ptr;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO ptr_info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &ptr_info);
        if (ptr_info.ResultDataMaxSizeInBytes != arr_info.ResultDataMaxSizeInBytes ||
            ptr_info.ScratchDataSizeInBytes != arr_info.ScratchDataSizeInBytes ||
            ptr_info.UpdateScratchDataSizeInBytes != arr_info.UpdateScratchDataSizeInBytes) {
            std::fprintf(stderr, "%s: ARRAY vs ARRAY_OF_POINTERS mismatch for %s\n", name, fc.name);
            return 1;
        }
    }

    // 2. Build flags: ALLOW_COMPACTION (alone and combined with PREFER_FAST_TRACE / ALLOW_UPDATE).
    {
        D3D12_RAYTRACING_GEOMETRY_DESC g{};
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g.Triangles.VertexCount = 3000;
        g.Triangles.VertexBuffer.StrideInBytes = 12;

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flag_cases[] = {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
        };
        for (const auto flags : flag_cases) {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.Flags = flags;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = &g;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
                std::fprintf(stderr, "%s: BLAS prebuild with flags=0x%X returned zero\n",
                             name, static_cast<unsigned>(flags));
                return 1;
            }
        }
    }

    // 3. Geometry flags: NO_DUPLICATE_ANYHIT_INVOCATION (alone and combined with OPAQUE).
    {
        const D3D12_RAYTRACING_GEOMETRY_FLAGS geom_flags[] = {
            D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION,
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE | D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION,
        };
        for (const auto gf : geom_flags) {
            D3D12_RAYTRACING_GEOMETRY_DESC g{};
            g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            g.Flags = gf;
            g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            g.Triangles.VertexCount = 3000;
            g.Triangles.VertexBuffer.StrideInBytes = 12;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = &g;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
                std::fprintf(stderr, "%s: BLAS prebuild with geom flags=0x%X returned zero\n",
                             name, static_cast<unsigned>(gf));
                return 1;
            }
        }
    }

    // 4. End-to-end GPU BLAS build sized directly from PrebuildInfo:
    //    - R32G32B32_FLOAT with ALLOW_COMPACTION + NO_DUPLICATE_ANYHIT_INVOCATION (all backends)
    //    - All 6 vertex formats including 4-byte stride R16G16_FLOAT / R16G16_SNORM (native D3D12 & Leonida KHR)
    {
        Ref<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qdesc{};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(dev->CreateCommandQueue(&qdesc, IID_PPV_ARGS(queue.put())))) return 1;

        Ref<ID3D12Resource> vb;
        if (!make_buffer(dev.p, 65536, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ, vb))
            return 1;
        {
            void* ptr = nullptr;
            D3D12_RANGE no_read{0, 0};
            if (FAILED(vb->Map(0, &no_read, &ptr)) || !ptr) return 1;
            std::memset(ptr, 0, 65536);
            D3D12_RANGE written{0, 65536};
            vb->Unmap(0, &written);
        }

        auto build_blas = [&](const char* tag, DXGI_FORMAT fmt, UINT64 stride, UINT vcount,
                              D3D12_RAYTRACING_GEOMETRY_FLAGS gflags,
                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS bflags) -> bool {
            D3D12_RAYTRACING_GEOMETRY_DESC g{};
            g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            g.Flags = gflags;
            g.Triangles.VertexFormat = fmt;
            g.Triangles.VertexCount = vcount;
            g.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
            g.Triangles.VertexBuffer.StrideInBytes = stride;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.Flags = bflags;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = &g;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
                std::fprintf(stderr, "%s: build_blas(%s) prebuild returned zero\n", name, tag);
                return false;
            }

            Ref<ID3D12Resource> as_buf, scratch_buf;
            if (!make_buffer(dev.p, info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, as_buf) ||
                !make_buffer(dev.p, info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_COMMON, scratch_buf)) {
                std::fprintf(stderr, "%s: build_blas(%s) buffer allocation failed\n", name, tag);
                return false;
            }

            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList4> cl;
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) ||
                FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                              IID_PPV_ARGS(cl.put()))))
                return false;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{};
            bd.DestAccelerationStructureData = as_buf->GetGPUVirtualAddress();
            bd.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();
            bd.Inputs = in;
            cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);

            if (!execute_and_wait(dev.p, queue.p, cl.p)) {
                std::fprintf(stderr, "%s: build_blas(%s) GPU execution failed\n", name, tag);
                return false;
            }
            return true;
        };

        if (!build_blas("R32G32B32_FLOAT+ALLOW_COMPACTION+NO_DUP_ANYHIT",
                        DXGI_FORMAT_R32G32B32_FLOAT, 12, 3000,
                        D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION,
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION))
            return 1;

        if (!is_leonida || is_khr_mode()) {
            for (const auto& fc : valid_formats) {
                if (!build_blas(fc.name, fc.format, fc.stride, 300,
                                D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION))
                    return 1;
            }
        }
    }

    if (!is_leonida) {
        std::printf("%s (tier=%d): all 6 vertex formats, ALLOW_COMPACTION, and NO_DUPLICATE_ANYHIT_INVOCATION OK\n",
                    name, static_cast<int>(o5.RaytracingTier));
        return 0;
    }

    // 5. Leonida strict validation: unsupported VertexFormat must return zeroed prebuild info with zero VUIDs.
    {
        const FormatCase invalid_formats[] = {
            {"UNKNOWN", DXGI_FORMAT_UNKNOWN, 12},
            {"R8G8B8A8_UNORM", DXGI_FORMAT_R8G8B8A8_UNORM, 4},
            {"R32G32B32A32_FLOAT", DXGI_FORMAT_R32G32B32A32_FLOAT, 16},
            {"R16G16B16A16_UNORM", DXGI_FORMAT_R16G16B16A16_UNORM, 8},
            {"R16G16_UNORM", DXGI_FORMAT_R16G16_UNORM, 4},
            {"R32_FLOAT", DXGI_FORMAT_R32_FLOAT, 4},
        };

        for (const auto& fc : invalid_formats) {
            D3D12_RAYTRACING_GEOMETRY_DESC g{};
            g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            g.Triangles.VertexFormat = fc.format;
            g.Triangles.VertexCount = 300;
            g.Triangles.VertexBuffer.StrideInBytes = fc.stride;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.NumDescs = 1;
            in.pGeometryDescs = &g;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{0xDEAD, 0xDEAD, 0xDEAD};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            if (!is_zero_prebuild(info)) {
                std::fprintf(stderr, "%s: invalid format %s (ARRAY) did not return zero (%llu, %llu, %llu)\n",
                             name, fc.name, info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes,
                             info.UpdateScratchDataSizeInBytes);
                return 1;
            }

            const D3D12_RAYTRACING_GEOMETRY_DESC* g_ptr = &g;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
            in.ppGeometryDescs = &g_ptr;
            info = {0xDEAD, 0xDEAD, 0xDEAD};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
            if (!is_zero_prebuild(info)) {
                std::fprintf(stderr, "%s: invalid format %s (PTRS) did not return zero (%llu, %llu, %llu)\n",
                             name, fc.name, info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes,
                             info.UpdateScratchDataSizeInBytes);
                return 1;
            }
        }

        // Mixed valid + invalid geometry in one BLAS must also return zeroed prebuild info.
        D3D12_RAYTRACING_GEOMETRY_DESC gs[2]{};
        gs[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        gs[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        gs[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        gs[0].Triangles.VertexCount = 300;
        gs[0].Triangles.VertexBuffer.StrideInBytes = 12;

        gs[1] = gs[0];
        gs[1].Triangles.VertexFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        gs[1].Triangles.VertexBuffer.StrideInBytes = 4;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 2;
        in.pGeometryDescs = gs;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{0xDEAD, 0xDEAD, 0xDEAD};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (!is_zero_prebuild(info)) {
            std::fprintf(stderr, "%s: mixed valid+invalid BLAS did not return zero\n", name);
            return 1;
        }
    }

    std::printf("%s (tier=%d, mode=%s): all 6 vertex formats, ALLOW_COMPACTION, NO_DUPLICATE_ANYHIT_INVOCATION, "
                "and invalid format refusal OK\n",
                name, static_cast<int>(o5.RaytracingTier), is_khr_mode() ? "khr" : "xdxr");
    return 0;
}
} // namespace

int main() {
    // 1. Differential host comparators: Windows WARP and native HW D3D12 (if available and DXR-capable).
    Ref<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
        Ref<IDXGIAdapter> warp;
        Ref<ID3D12Device> warp_dev;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(warp.put()))) &&
            SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(warp_dev.put())))) {
            const int r = run_suite(warp_dev.p, "native WARP", false);
            if (r != 0 && r != 77) return r;
        }

        Ref<IDXGIAdapter1> hw_adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, hw_adapter.put())); ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            hw_adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                hw_adapter.p->Release();
                hw_adapter.p = nullptr;
                continue;
            }
            Ref<ID3D12Device> hw_dev;
            if (SUCCEEDED(D3D12CreateDevice(hw_adapter.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(hw_dev.put())))) {
                char adapter_name[128]{};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapter_name, sizeof(adapter_name), nullptr,
                                    nullptr);
                char label[160]{};
                std::snprintf(label, sizeof(label), "native HW (%s)", adapter_name);
                const int r = run_suite(hw_dev.p, label, false);
                if (r != 0 && r != 77) return r;
                break;
            }
            hw_adapter.p->Release();
            hw_adapter.p = nullptr;
        }
    }

    // 2. Leonida Vulkan D3D12 device.
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(layer.put())))) return 77;
    return run_suite(layer.p, "Leonida Vulkan", true);
}
