// DXR postbuild, copy, TLAS alignment, closed-list, and atomic inline-postbuild parity test.
// Compares live Windows 11 WARP (RaytracingTier 11), native NVIDIA D3D12 (RaytracingTier 12),
// and Leonida Vulkan (XDXR and KHR) with zero Vulkan validation errors.
// Synthetic resources only; no game content.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdint>
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
    T** put() {
        if (p) {
            p->Release();
            p = nullptr;
        }
        return &p;
    }
};

bool make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE heap_type,
                 D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                 Ref<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap_type;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                  IID_PPV_ARGS(out.put())));
}

bool make_list(ID3D12Device5* dev, Ref<ID3D12CommandAllocator>& alloc, Ref<ID3D12GraphicsCommandList4>& cl) {
    return SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) &&
           SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                            IID_PPV_ARGS(cl.put())));
}

bool execute_and_wait(ID3D12Device5* dev, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList4* cl,
                      bool already_closed = false) {
    if (!already_closed && FAILED(cl->Close())) return false;
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

bool write_u32(ID3D12Resource* res, std::uint32_t value) {
    std::uint32_t* p = nullptr;
    D3D12_RANGE no_read{0, 0};
    if (FAILED(res->Map(0, &no_read, reinterpret_cast<void**>(&p))) || !p) return false;
    *p = value;
    D3D12_RANGE written{0, sizeof(value)};
    res->Unmap(0, &written);
    return true;
}

bool read_u32(ID3D12Resource* res, std::uint32_t& out) {
    std::uint32_t* p = nullptr;
    D3D12_RANGE read_range{0, sizeof(out)};
    if (FAILED(res->Map(0, &read_range, reinterpret_cast<void**>(&p))) || !p) return false;
    out = *p;
    D3D12_RANGE no_write{0, 0};
    res->Unmap(0, &no_write);
    return true;
}

struct DeviceEnv {
    Ref<ID3D12Device5> dev;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Resource> vb, inst_buf, scratch, blas_buf, tlas_buf, src_as, post_buf, upload_sentinel, readback_sentinel;
    D3D12_GPU_VIRTUAL_ADDRESS vb_va = 0, inst_va = 0, scr_va = 0, blas_va = 0, tlas_va = 0, src_va = 0, post_va = 0;
    D3D12_RAYTRACING_GEOMETRY_DESC tri{};

    bool init(ID3D12Device* base_dev, const char* name, bool check_strict_validation, int& skip_code) {
        skip_code = 0;
        if (FAILED(base_dev->QueryInterface(IID_PPV_ARGS(dev.put())))) {
            std::printf("%s: no ID3D12Device5, skipping\n", name);
            skip_code = 77;
            return false;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
        if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5)))) {
            std::printf("%s: OPTIONS5 query failed, skipping\n", name);
            skip_code = 77;
            return false;
        }
        if (!check_strict_validation && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
            std::printf("%s: raytracing not supported (tier=%d), skipping\n", name, static_cast<int>(o5.RaytracingTier));
            skip_code = 77;
            return false;
        }

        D3D12_COMMAND_QUEUE_DESC qdesc{};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(dev->CreateCommandQueue(&qdesc, IID_PPV_ARGS(queue.put())))) return false;

        if (!make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ, vb) ||
            !make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ, inst_buf) ||
            !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON, scratch) ||
            !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, blas_buf) ||
            !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, tlas_buf) ||
            !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, src_as) ||
            !make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON, post_buf) ||
            !make_buffer(dev.p, 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ, upload_sentinel) ||
            !make_buffer(dev.p, 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_COPY_DEST, readback_sentinel)) {
            std::fprintf(stderr, "%s: failed to allocate synthetic buffers\n", name);
            skip_code = 1;
            return false;
        }

        // Upload a single triangle and a 16-byte-offset TLAS instance descriptor.
        {
            float* verts = nullptr;
            D3D12_RANGE no_read{0, 0};
            if (FAILED(vb->Map(0, &no_read, reinterpret_cast<void**>(&verts))) || !verts) return false;
            const float tri_verts[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
            std::memcpy(verts, tri_verts, sizeof(tri_verts));
            D3D12_RANGE written{0, sizeof(tri_verts)};
            vb->Unmap(0, &written);

            std::uint8_t* inst_mem = nullptr;
            if (FAILED(inst_buf->Map(0, &no_read, reinterpret_cast<void**>(&inst_mem))) || !inst_mem) return false;
            std::memset(inst_mem, 0, 256);
            auto write_inst = [&](std::size_t offset) {
                auto* inst = reinterpret_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(inst_mem + offset);
                inst->Transform[0][0] = 1.0f;
                inst->Transform[1][1] = 1.0f;
                inst->Transform[2][2] = 1.0f;
                inst->InstanceMask = 0xFF;
                inst->AccelerationStructure = blas_buf->GetGPUVirtualAddress();
            };
            write_inst(0);
            write_inst(16);
            D3D12_RANGE inst_written{0, 256};
            inst_buf->Unmap(0, &inst_written);
        }

        vb_va = vb->GetGPUVirtualAddress();
        inst_va = inst_buf->GetGPUVirtualAddress();
        scr_va = scratch->GetGPUVirtualAddress();
        blas_va = blas_buf->GetGPUVirtualAddress();
        tlas_va = tlas_buf->GetGPUVirtualAddress();
        src_va = src_as->GetGPUVirtualAddress();
        post_va = post_buf->GetGPUVirtualAddress();

        tri = {};
        tri.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        tri.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        tri.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        tri.Triangles.VertexCount = 3;
        tri.Triangles.VertexBuffer.StartAddress = vb_va;
        tri.Triangles.VertexBuffer.StrideInBytes = 12;

        // Verify DXR prebuild works before running GPU builds (handles non-RT Vulkan devices cleanly).
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = &tri;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (info.ResultDataMaxSizeInBytes == 0) {
            if (check_strict_validation && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
                std::printf("%s: neither XDXR nor KHR acceleration structures supported, skipping\n", name);
                skip_code = 77;
                return false;
            }
            std::fprintf(stderr, "%s: valid BLAS prebuild returned zero\n", name);
            skip_code = 1;
            return false;
        }
        return true;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC valid_blas() const {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
        d.DestAccelerationStructureData = blas_va;
        d.ScratchAccelerationStructureData = scr_va;
        d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        d.Inputs.NumDescs = 1;
        d.Inputs.pGeometryDescs = &tri;
        return d;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC valid_tlas() const {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
        d.DestAccelerationStructureData = tlas_va;
        d.ScratchAccelerationStructureData = scr_va;
        d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        d.Inputs.NumDescs = 1;
        d.Inputs.InstanceDescs = inst_va;
        return d;
    }
};

int run_suite(ID3D12Device* base_dev, const char* name, bool check_strict_validation) {
    DeviceEnv env;
    int skip_code = 0;
    if (!env.init(base_dev, name, check_strict_validation, skip_code)) return skip_code;

    // 1. Valid GPU execution parity: BLAS + inline postbuild (CURRENT_SIZE & COMPACTED_SIZE),
    //    standalone EmitPostbuild, and 16-byte-aligned TLAS build.
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(env.dev.p, alloc, cl)) return 1;

        auto blas_desc = env.valid_blas();
        blas_desc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pbs[2] = {
            {env.post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE},
            {env.post_va + 8, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE},
        };
        cl->BuildRaytracingAccelerationStructure(&blas_desc, 2, pbs);

        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = env.blas_buf.p;
        cl->ResourceBarrier(1, &uav);

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC emit_pb{
            env.post_va + 16, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
        cl->EmitRaytracingAccelerationStructurePostbuildInfo(&emit_pb, 1, &env.blas_va);

        auto tlas_desc = env.valid_tlas();
        tlas_desc.Inputs.InstanceDescs = env.inst_va + 16;  // 16-byte aligned non-zero offset
        cl->BuildRaytracingAccelerationStructure(&tlas_desc, 0, nullptr);

        if (!execute_and_wait(env.dev.p, env.queue.p, cl.p)) {
            std::fprintf(stderr, "%s: valid BLAS+postbuild+TLAS GPU execution failed\n", name);
            return 1;
        }
    }

    // 2. Closed DIRECT list parity (verified identically on live WARP, native RTX 5090, and Leonida):
    //    Calling any of the 3 DXR methods on an already-closed DIRECT list must be a no-op and must NOT
    //    poison ExecuteCommandLists (the previously recorded CopyBufferRegion must still execute).
    {
        auto check_closed_noop = [&](const char* label, auto&& closed_call) -> bool {
            constexpr std::uint32_t kSentinel = 0xCAFEBABEu;
            constexpr std::uint32_t kCleared = 0x11111111u;
            if (!write_u32(env.upload_sentinel.p, kSentinel) || !write_u32(env.readback_sentinel.p, kCleared))
                return false;

            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList4> cl;
            if (!make_list(env.dev.p, alloc, cl)) return false;
            cl->CopyBufferRegion(env.readback_sentinel.p, 0, env.upload_sentinel.p, 0, sizeof(kSentinel));
            if (FAILED(cl->Close())) {
                std::fprintf(stderr, "%s: closed-list '%s' first Close() failed\n", name, label);
                return false;
            }

            // Call DXR method on the already-closed DIRECT list.
            closed_call(cl.p);

            // Second Close() must still report E_FAIL (already closed), not S_OK.
            if (cl->Close() != E_FAIL) {
                std::fprintf(stderr, "%s: closed-list '%s' second Close() did not return E_FAIL\n", name, label);
                return false;
            }

            if (!execute_and_wait(env.dev.p, env.queue.p, cl.p, /*already_closed=*/true)) {
                std::fprintf(stderr, "%s: closed-list '%s' ExecuteCommandLists failed\n", name, label);
                return false;
            }
            std::uint32_t got = 0;
            if (!read_u32(env.readback_sentinel.p, got) || got != kSentinel) {
                std::fprintf(stderr, "%s: closed-list '%s' poisoned ExecuteCommandLists (got=0x%08X, want=0x%08X)\n",
                             name, label, got, kSentinel);
                return false;
            }
            return true;
        };

        if (!check_closed_noop("BuildRaytracingAccelerationStructure", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = env.valid_blas();
                cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
            })) return 1;

        if (!check_closed_noop("EmitRaytracingAccelerationStructurePostbuildInfo", [&](ID3D12GraphicsCommandList4* cl) {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                    env.post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
                cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &env.blas_va);
            })) return 1;

        if (!check_closed_noop("CopyRaytracingAccelerationStructure", [&](ID3D12GraphicsCommandList4* cl) {
                cl->CopyRaytracingAccelerationStructure(
                    env.blas_va, env.src_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
            })) return 1;
    }

    // 3. Valid recording parity across WARP, native HW, and Leonida.
    auto expect_ok = [&](const char* label, auto&& record) -> bool {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(env.dev.p, alloc, cl)) return false;
        record(cl.p);
        const HRESULT hr = cl->Close();
        if (FAILED(hr)) {
            std::fprintf(stderr, "%s: valid case '%s' failed Close (0x%08X)\n",
                         name, label, static_cast<unsigned>(hr));
            return false;
        }
        return true;
    };

    if (!expect_ok("Build PERFORM_UPDATE in-place (dst == src)", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_blas();
            d.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            d.SourceAccelerationStructureData = d.DestAccelerationStructureData;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_ok("Build TLAS ARRAY InstanceDescs+16", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.InstanceDescs = env.inst_va + 16;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_ok("Build TLAS ARRAY_OF_POINTERS InstanceDescs+8", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
            d.Inputs.InstanceDescs = env.inst_va + 8;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_ok("Build TLAS NumDescs=0 InstanceDescs=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.NumDescs = 0;
            d.Inputs.InstanceDescs = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_ok("Build TLAS NumDescs=0 InstanceDescs+4 ignored", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.NumDescs = 0;
            d.Inputs.InstanceDescs = env.inst_va + 4;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_ok("Emit TOOLS_VISUALIZATION DestBuffer+8", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                env.post_va + 8, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &env.blas_va);
        })) return 1;

    if (!check_strict_validation) {
        if (env.dev->GetDeviceRemovedReason() != S_OK) {
            std::fprintf(stderr, "%s: device removed (0x%08X)\n",
                         name, static_cast<unsigned>(env.dev->GetDeviceRemovedReason()));
            return 1;
        }
        std::printf("%s: GPU execution, closed-list no-op, and valid recording parity OK\n", name);
        return 0;
    }

    // 4. Leonida strict validation & atomic inline-postbuild checks.
    auto expect_fail = [&](const char* label, auto&& record) -> bool {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(env.dev.p, alloc, cl)) return false;
        record(cl.p);
        const HRESULT hr = cl->Close();
        if (SUCCEEDED(hr)) {
            std::fprintf(stderr, "%s: invalid case '%s' unexpectedly passed Close()\n", name, label);
            return false;
        }
        return true;
    };

    // 4a. Atomic inline postbuild rejection: second postbuild entry invalid must fail Close() and
    //     never allow any partial build/postbuild to execute.
    {
        constexpr std::uint32_t kSentinel = 0xDEADBEEFu;
        constexpr std::uint32_t kCleared = 0x22222222u;
        if (!write_u32(env.upload_sentinel.p, kSentinel) || !write_u32(env.readback_sentinel.p, kCleared)) return 1;

        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(env.dev.p, alloc, cl)) return 1;
        cl->CopyBufferRegion(env.readback_sentinel.p, 0, env.upload_sentinel.p, 0, sizeof(kSentinel));
        auto d = env.valid_blas();
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pbs[2] = {
            {env.post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE},
            {env.post_va + 4, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION},
        };
        cl->BuildRaytracingAccelerationStructure(&d, 2, pbs);
        if (SUCCEEDED(cl->Close())) {
            std::fprintf(stderr, "%s: inline postbuild[1] misaligned unexpectedly passed Close()\n", name);
            return 1;
        }
        if (!execute_and_wait(env.dev.p, env.queue.p, cl.p, /*already_closed=*/true)) return 1;
        std::uint32_t got = 0;
        if (!read_u32(env.readback_sentinel.p, got) || got != kCleared) {
            std::fprintf(stderr, "%s: invalid inline postbuild list was not refused by ExecuteCommandLists\n", name);
            return 1;
        }
    }

    // Inline postbuild DestBuffer=0 and +4 for all 4 InfoTypes (including TOOLS_VISUALIZATION).
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE info_types[] = {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE,
    };
    for (const auto t : info_types) {
        if (!expect_fail("Build inline postbuild[1] DestBuffer=0", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = env.valid_blas();
                const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pbs[2] = {
                    {env.post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE},
                    {0, t},
                };
                cl->BuildRaytracingAccelerationStructure(&d, 2, pbs);
            })) return 1;

        if (!expect_fail("Build inline postbuild[1] DestBuffer+4", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = env.valid_blas();
                const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pbs[2] = {
                    {env.post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE},
                    {env.post_va + 4, t},
                };
                cl->BuildRaytracingAccelerationStructure(&d, 2, pbs);
            })) return 1;

        if (!expect_fail("Emit DestBuffer+4", [&](ID3D12GraphicsCommandList4* cl) {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{env.post_va + 4, t};
                cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &env.blas_va);
            })) return 1;
    }

    // 4b. Build dst == scratch, update src == scratch, and TLAS InstanceDescs alignment.
    if (!expect_fail("Build dst == scratch", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_blas();
            d.ScratchAccelerationStructureData = d.DestAccelerationStructureData;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build PERFORM_UPDATE src == scratch", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_blas();
            d.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            d.SourceAccelerationStructureData = d.ScratchAccelerationStructureData;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS ARRAY Num=1 InstanceDescs=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.InstanceDescs = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS ARRAY Num=1 InstanceDescs+8", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.InstanceDescs = env.inst_va + 8;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS ARRAY_OF_POINTERS Num=1 InstanceDescs=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
            d.Inputs.InstanceDescs = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS ARRAY_OF_POINTERS Num=1 InstanceDescs+4", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = env.valid_tlas();
            d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
            d.Inputs.InstanceDescs = env.inst_va + 4;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    // 4c. CopyRaytracingAccelerationStructure: in-place (dst == src) & SERIALIZE/DESERIALIZE/VIS alignment.
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE copy_modes[] = {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_VISUALIZATION_DECODE_FOR_TOOLS,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE,
    };
    for (const auto m : copy_modes) {
        if (!expect_fail("Copy dst == src", [&](ID3D12GraphicsCommandList4* cl) {
                cl->CopyRaytracingAccelerationStructure(env.blas_va, env.blas_va, m);
            })) return 1;
        if (!expect_fail("Copy dst+128", [&](ID3D12GraphicsCommandList4* cl) {
                cl->CopyRaytracingAccelerationStructure(env.blas_va + 128, env.src_va, m);
            })) return 1;
        if (!expect_fail("Copy src+128", [&](ID3D12GraphicsCommandList4* cl) {
                cl->CopyRaytracingAccelerationStructure(env.blas_va, env.src_va + 128, m);
            })) return 1;
    }

    std::printf("%s: GPU execution, closed-list no-op, atomic postbuild, TLAS & copy validation OK\n", name);
    return 0;
}
}  // namespace

int main() {
    // 1. Live Windows 11 WARP + native HW D3D12 differential comparators.
    Ref<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
        Ref<IDXGIAdapter> warp;
        Ref<ID3D12Device> warp_dev;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(warp.put()))) &&
            SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(warp_dev.put())))) {
            if (warp_dev->GetDeviceRemovedReason() != S_OK) {
                D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(warp_dev.put()));
            }
            const int r = run_suite(warp_dev.p, "native WARP", false);
            if (r != 0 && r != 77) return r;
        }

        Ref<IDXGIAdapter1> hw_adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, hw_adapter.put())); ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            hw_adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
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
        }
    }

    // 2. Leonida Vulkan D3D12 device (XDXR by default, KHR when LEONIDA_RT_AS=khr).
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(layer.put())))) return 77;
    return run_suite(layer.p, "Leonida Vulkan", true);
}
