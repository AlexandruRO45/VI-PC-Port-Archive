// Lanterns, stage 25: the GPU side of the scale frame (lanterns_scale_gpu.h).
#include "lanterns_scale_gpu.h"

#include "lanterns_d3d12.h"
#include "lanterns_scale_prof.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                      const IID* riid, void** out);

using namespace d3d12test;

namespace lanterns::scale {
namespace {

// Shader-visible heap: mesh V#s from slot 0, the rise's tables (set A per parity, set B), two dummy tables root
// signature 2 binds, then the ring (two parities).
constexpr std::uint64_t kMeshSlot = 0, kTableA = 16, kTableB = 32, kDummySrv = 40, kDummyUav = 48, kRingBase = 64;
constexpr std::uint64_t kPoolBytes = 16ull << 20, kPoolChunk = 1ull << 20;
constexpr DXGI_FORMAT kHdr = DXGI_FORMAT_R16G16B16A16_FLOAT, kLdr = DXGI_FORMAT_R8G8B8A8_UNORM,
                      kDepth = DXGI_FORMAT_D32_FLOAT;
constexpr unsigned kComputeSlots = 5;

D3D12_RESOURCE_BARRIER uav_barrier(ID3D12Resource* r) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    return b;
}

}  // namespace

ScaleGpu::~ScaleGpu() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : workers_)
        if (t.joinable()) t.join();
    if (event_) CloseHandle(event_);
}

bool ScaleGpu::create(const World& world, const Shaders& shaders, unsigned threads) {
    world_ = &world;
    threads_ = std::max(1u, threads);
    QueryPerformanceFrequency(&qpf_);
    QueryPerformanceCounter(&qpc0_);
    const std::uint32_t device_params[24] = {};
    const IID device_iid = __uuidof(ID3D12Device);
    const HRESULT created = D3D12XboxCreateDevice(nullptr, device_params, &device_iid, reinterpret_cast<void**>(&device_.p));
    if (FAILED(created) || !device_.p) {
        std::fprintf(stderr, "lanterns: D3D12XboxCreateDevice failed (hr=0x%08lx)\n", static_cast<unsigned long>(created));
        return false;
    }
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(event_);
    CHECK(create_pipelines(shaders));
    CHECK(create_resources());
    plan();
    CHECK(fill_descriptors());
    CHECK(initialize());
    for (unsigned t = 0; t < threads_; ++t) workers_.emplace_back([this, t] { worker(t); });
    return true;
}

bool ScaleGpu::create_pipelines(const Shaders& shaders) {
    // Four graphics root signatures sharing the layout vs_lanterns.asm reads (s[0:1] mesh table, s[2:3] instance
    // table, s4..s19 constants, s[20:21] camera CBV), each with different parameters after it.
    D3D12_DESCRIPTOR_RANGE ranges[4] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
                                        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 0},
                                        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 2, 0, 0},
                                        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0}};
    for (unsigned rs = 0; rs < kRootSignatures; ++rs) {
        D3D12_ROOT_PARAMETER params[6] = {};
        params[0].ParameterType = params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &ranges[0]};
        params[1].DescriptorTable = {1, &ranges[1]};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants = {0, 0, 16};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[3].Descriptor = {1, 0};
        UINT count = 4;
        if (rs == 1) {   // four more constants (b2)
            params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[4].Constants = {2, 0, 4};
            count = 5;
        } else if (rs == 2) {   // two more tables: SRVs t2-t3, a UAV u0
            params[4].ParameterType = params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[4].DescriptorTable = {1, &ranges[2]};
            params[5].DescriptorTable = {1, &ranges[3]};
            count = 6;
        } else if (rs == 3) {   // a second root CBV (b2) and two constants (b3)
            params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            params[4].Descriptor = {2, 0};
            params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[5].Constants = {3, 0, 2};
            count = 6;
        }
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0 = {count, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(desc);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&rs_[rs].p))));
    }
    {   // the compute root signature of stage 3: four constants (s0..s3), a table of five UAVs (s[4:5])
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kComputeSlots, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 4};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize_root_signature(desc);
        CHECK(!blob.empty());
        CHECK(SUCCEEDED(device_->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&compute_rs_.p))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC cdesc{};
        cdesc.pRootSignature = compute_rs_.p;
        cdesc.CS = {shaders.cs_rise.data(), shaders.cs_rise.size()};
        CHECK(SUCCEEDED(device_->CreateComputePipelineState(&cdesc, IID_PPV_ARGS(&rise_.p))));
    }
    root_signatures_ = kRootSignatures + 1;
    const auto base = [&](ID3D12RootSignature* rs, const std::vector<std::uint8_t>& ps, DXGI_FORMAT format, UINT samples) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = rs;
        desc.VS = {shaders.vs.data(), shaders.vs.size()};
        desc.PS = {ps.data(), ps.size()};
        for (auto& target : desc.BlendState.RenderTarget) {
            target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            target.SrcBlend = target.SrcBlendAlpha = D3D12_BLEND_ONE;
            target.DestBlend = target.DestBlendAlpha = D3D12_BLEND_ZERO;
            target.BlendOp = target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            target.LogicOp = D3D12_LOGIC_OP_NOOP;
        }
        desc.SampleMask = 0xffffffffu;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.DepthStencilState.DepthEnable = TRUE;
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = format;
        desc.DSVFormat = kDepth;
        desc.SampleDesc.Count = samples;
        return desc;
    };
    const auto apply_variant = [](D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, unsigned v) {
        static constexpr D3D12_CULL_MODE culls[3] = {D3D12_CULL_MODE_NONE, D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT};
        desc.RasterizerState.CullMode = culls[v % 3];
        switch (variant_flavour(v)) {
        case 1:
            desc.BlendState.RenderTarget[0].BlendEnable = TRUE;
            break;
        case 2:   // counter-clockwise front faces: the swapped cull mode culls the same triangles
            desc.RasterizerState.FrontCounterClockwise = TRUE;
            desc.RasterizerState.CullMode = culls[v % 3 == 1 ? 2 : v % 3 == 2 ? 1 : 0];
            break;
        case 3:
            desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
            desc.RasterizerState.DepthBias = 1;
            break;
        default:
            break;
        }
    };
    const auto make = [&](const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, Ref<ID3D12PipelineState>& out) {
        ++pipelines_;
        return SUCCEEDED(device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&out.p)));
    };
    for (unsigned rs = 0; rs < kRootSignatures; ++rs)
        for (unsigned v = 0; v < kVariants; ++v) {
            auto desc = base(rs_[rs].p, shaders.ps_lit, kHdr, 1);
            apply_variant(desc, v);
            CHECK(make(desc, lit_[rs][v]));
        }
    CHECK(make(base(rs_[0].p, shaders.ps_floor, kHdr, 1), floor_));
    for (unsigned c = 0; c < 3; ++c) {
        auto desc = base(rs_[0].p, shaders.ps_lit, kHdr, 4);
        apply_variant(desc, c);
        CHECK(make(desc, aux_lit_[c]));
        desc = base(rs_[0].p, shaders.ps_lit, kLdr, 1);
        apply_variant(desc, c);
        CHECK(make(desc, top_lit_[c]));
    }
    CHECK(make(base(rs_[0].p, shaders.ps_floor, kHdr, 4), aux_floor_));
    CHECK(make(base(rs_[0].p, shaders.ps_floor, kLdr, 1), top_floor_));
    ++pipelines_;   // the rise
    return true;
}

bool ScaleGpu::create_resources() {
    const Scene& scene = world_->scene;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD, {}, {}, 0, 0};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, {}, {}, 0, 0};
    const auto buffer = [&](const D3D12_HEAP_PROPERTIES& heap, std::uint64_t bytes, D3D12_RESOURCE_STATES state,
                            Ref<ID3D12Resource>& out, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const auto desc = buffer_desc(std::max<std::uint64_t>(bytes, 256), flags);
        return SUCCEEDED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                          IID_PPV_ARGS(&out.p)));
    };
    // Geometry: every mesh's vertices and its fanned faces, as lanterns_gpu.cpp flattens them.
    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;
    for (const Mesh& mesh : scene.meshes) {
        for (const Face& face : mesh.faces) CHECK(face.count >= 3 && face.count <= face.v.size());
        vertex_first_.push_back(vertices.size());
        vertex_count_.push_back(mesh.vertices.size());
        first_index_.push_back(indices.size());
        vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const Face& face : mesh.faces)
            for (unsigned k = 1; k + 1 < face.count; ++k) indices.insert(indices.end(), {face.v[0], face.v[k], face.v[k + 1]});
        index_count_.push_back(indices.size() - first_index_.back());
    }
    // Instance records: the static ones, then (separately) the streamed ones.
    std::vector<Instance> static_records, stream_records;
    for (std::size_t d = 0; d < scene.draws.size(); ++d) {
        const Draw& draw = scene.draws[d];
        const Item& item = world_->items[d];
        if (item.particles) {
            inst_offset_.push_back(0);
        } else if (item.streamed) {
            inst_offset_.push_back(stream_records.size() * sizeof(Instance));
            stream_records.insert(stream_records.end(), draw.instances.begin(), draw.instances.end());
        } else {
            inst_offset_.push_back(static_records.size() * sizeof(Instance));
            static_records.insert(static_records.end(), draw.instances.begin(), draw.instances.end());
        }
    }
    const std::uint64_t vb_bytes = vertices.size() * sizeof(Vertex), ib_bytes = align(indices.size() * 2, 4);
    inst_bytes_ = static_records.size() * sizeof(Instance);
    stream_bytes_ = stream_records.size() * sizeof(Instance);
    up_vb_ = 0;
    up_ib_ = align(up_vb_ + vb_bytes, 256);
    up_inst_ = align(up_ib_ + ib_bytes, 256);
    up_stream_ = align(up_inst_ + inst_bytes_, 256);
    up_zero_ = align(up_stream_ + stream_bytes_, 256);
    up_cam_ = align(up_zero_ + stream_bytes_, 256);   // main, aux, top cameras, 256 bytes apart
    up_extra_ = up_cam_ + 3 * 256;                     // the second root CBV of root signature 3
    up_particles_ = up_extra_ + 256;
    const std::uint64_t upload_bytes = up_particles_ + kParticles * sizeof(Particle);
    CHECK(buffer(upload_heap, upload_bytes, D3D12_RESOURCE_STATE_GENERIC_READ, upload_));
    CHECK(buffer(upload_heap, kPoolBytes, D3D12_RESOURCE_STATE_GENERIC_READ, pool_upload_));
    std::uint8_t* up = nullptr;
    CHECK(SUCCEEDED(upload_->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    std::memset(up, 0, std::size_t(upload_bytes));
    std::memcpy(up + up_vb_, vertices.data(), vb_bytes);
    std::memcpy(up + up_ib_, indices.data(), indices.size() * 2);
    if (inst_bytes_) std::memcpy(up + up_inst_, static_records.data(), inst_bytes_);
    if (stream_bytes_) std::memcpy(up + up_stream_, stream_records.data(), stream_bytes_);
    std::memcpy(up + up_cam_, scene.view_projection.data(), 64);
    std::memcpy(up + up_cam_ + 256, world_->aux_vp.data(), 64);
    std::memcpy(up + up_cam_ + 512, world_->top_vp.data(), 64);
    const auto particles = initial_particles();
    std::memcpy(up + up_particles_, particles.data(), kParticles * sizeof(Particle));
    upload_->Unmap(0, nullptr);
    {   // the filler the copy queue streams: a byte pattern, never read by a shader
        std::uint8_t* pool = nullptr;
        CHECK(SUCCEEDED(pool_upload_->Map(0, nullptr, reinterpret_cast<void**>(&pool))));
        for (std::uint64_t i = 0; i < kPoolBytes; ++i) pool[i] = std::uint8_t(i * 131 + (i >> 12));
        pool_upload_->Unmap(0, nullptr);
    }
    const auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto common = D3D12_RESOURCE_STATE_COMMON, copy_dest = D3D12_RESOURCE_STATE_COPY_DEST;
    CHECK(buffer(default_heap, vb_bytes, copy_dest, vb_));
    CHECK(buffer(default_heap, ib_bytes, copy_dest, ib_));
    CHECK(buffer(default_heap, inst_bytes_, copy_dest, static_inst_));
    for (unsigned p = 0; p < 2; ++p) {
        CHECK(buffer(default_heap, stream_bytes_, common, stream_[p]));
        CHECK(buffer(default_heap, kParticles * sizeof(Particle), common, state_a_[p], uav));
        CHECK(buffer(default_heap, kParticles * sizeof(Instance), common, inst_a_[p], uav));
        CHECK(buffer(readback_heap, kReadbackBytes, copy_dest, readback_[p]));
    }
    CHECK(buffer(default_heap, kPoolBytes, common, pool_));
    CHECK(buffer(default_heap, kParticles * sizeof(Particle), copy_dest, pristine_a_));
    CHECK(buffer(default_heap, kParticles * sizeof(Particle), copy_dest, pristine_b_));
    CHECK(buffer(default_heap, kParticles * sizeof(Particle), common, state_b_, uav));
    CHECK(buffer(default_heap, kParticles * sizeof(Instance), common, inst_b_, uav));

    const auto texture = [&](unsigned width, unsigned height, DXGI_FORMAT format, UINT samples, bool depth,
                             D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = width;
        tex.Height = height;
        tex.DepthOrArraySize = 1;
        tex.MipLevels = 1;
        tex.Format = format;
        tex.SampleDesc.Count = samples;
        tex.Flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = format;
        if (depth) clear.DepthStencil = {1.0f, 0};
        else std::memcpy(clear.Color, kClear.data(), sizeof(clear.Color));
        return SUCCEEDED(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex, state, &clear,
                                                          IID_PPV_ARGS(&out.p)));
    };
    CHECK(texture(kW, kH, kHdr, 1, false, common, main_rt_));
    CHECK(texture(kW, kH, kDepth, 1, true, D3D12_RESOURCE_STATE_DEPTH_WRITE, main_depth_));
    CHECK(texture(kAuxW, kAuxH, kHdr, 4, false, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, aux_msaa_));
    CHECK(texture(kAuxW, kAuxH, kDepth, 4, true, D3D12_RESOURCE_STATE_DEPTH_WRITE, aux_depth_));
    CHECK(texture(kAuxW, kAuxH, kHdr, 1, false, common, aux_resolved_));
    CHECK(texture(kTopW, kTopH, kLdr, 1, false, common, top_rt_));
    CHECK(texture(kTopW, kTopH, kDepth, 1, true, D3D12_RESOURCE_STATE_DEPTH_WRITE, top_depth_));

    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_.p))));
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap_.p))));
    const UINT rtv_step = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const UINT dsv_step = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    const std::uint64_t rtv0 = heap_cpu(rtv_heap_.p), dsv0 = heap_cpu(dsv_heap_.p);
    ID3D12Resource* rts[3] = {main_rt_.p, aux_msaa_.p, top_rt_.p};
    ID3D12Resource* depths[3] = {main_depth_.p, aux_depth_.p, top_depth_.p};
    Pass* passes[3] = {&main_pass_, &aux_pass_, &top_pass_};
    const unsigned widths[3] = {kW, kAuxW, kTopW}, heights[3] = {kH, kAuxH, kTopH};
    for (unsigned k = 0; k < 3; ++k) {
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(rtv0 + k * rtv_step)}, dsv{SIZE_T(dsv0 + k * dsv_step)};
        device_->CreateRenderTargetView(rts[k], nullptr, rtv);
        device_->CreateDepthStencilView(depths[k], nullptr, dsv);
        *passes[k] = {upload_->GetGPUVirtualAddress() + up_cam_ + 256 * k, rtv, dsv, widths[k], heights[k]};
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK(SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&direct_.p))));
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    CHECK(SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&compute_.p))));
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    CHECK(SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&copy_.p))));
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&direct_fence_.p))));
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&compute_fence_.p))));
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence_.p))));
    const auto make_list = [&](D3D12_COMMAND_LIST_TYPE type, Ref<ID3D12CommandAllocator>& alloc,
                               Ref<ID3D12GraphicsCommandList>& list) {
        return SUCCEEDED(device_->CreateCommandAllocator(type, IID_PPV_ARGS(&alloc.p))) &&
               SUCCEEDED(device_->CreateCommandList(0, type, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) &&
               list->Close() == S_OK;
    };
    for (unsigned p = 0; p < 2; ++p) {
        CHECK(make_list(D3D12_COMMAND_LIST_TYPE_COPY, copy_alloc_[p], copy_list_[p]));
        CHECK(make_list(D3D12_COMMAND_LIST_TYPE_COMPUTE, compute_alloc_[p], compute_list_[p]));
        for (unsigned k = 0; k < 3; ++k) CHECK(make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, main_alloc_[p][k], main_list_[p][k]));
        chunk_alloc_[p].resize(threads_);
        chunk_list_[p].resize(threads_);
        for (unsigned t = 0; t < threads_; ++t)
            CHECK(make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, chunk_alloc_[p][t], chunk_list_[p][t]));
    }
    return true;
}

void ScaleGpu::plan() {
    const Scene& scene = world_->scene;
    const auto make = [&](unsigned d, ID3D12PipelineState* pso, unsigned rs, unsigned instances, bool set_b) {
        const Draw& draw = scene.draws[d];
        Call c{};
        c.ring = unsigned(ring_draw_.size());
        c.pso = pso;
        c.rs = rs;
        c.mesh_table = 0;   // filled by fill_descriptors (needs the heap)
        c.index_count = std::uint32_t(index_count_[draw.mesh]);
        c.first_index = std::uint32_t(first_index_[draw.mesh]);
        c.instances = instances;
        std::memcpy(c.constants, draw.model.data(), 48);
        std::memcpy(c.constants + 12, draw.tint.data(), 16);
        ring_draw_.push_back(d);
        ring_set_b_.push_back(set_b ? 1 : 0);
        return c;
    };
    for (unsigned d = 0; d < scene.draws.size(); ++d) {
        const Draw& draw = scene.draws[d];
        const Item& item = world_->items[d];
        ID3D12PipelineState* pso = draw.floor ? floor_.p : lit_[item.rs][item.variant].p;
        main_calls_.push_back(make(d, pso, draw.floor ? 0 : item.rs, unsigned(draw.instances.size()), false));
    }
    for (unsigned d : world_->aux_draws) {
        const Draw& draw = scene.draws[d];
        aux_calls_.push_back(make(d, draw.floor ? aux_floor_.p : aux_lit_[world_->items[d].variant % 3].p, 0,
                                  unsigned(draw.instances.size()), false));
    }
    const unsigned particles = unsigned(scene.draws.size() - 2);   // make_world puts the points before the floor
    aux_calls_.insert(aux_calls_.end() - 1, make(particles, aux_lit_[0].p, 0, kParticles, true));
    for (unsigned d : world_->top_draws) {
        const Draw& draw = scene.draws[d];
        top_calls_.push_back(make(d, draw.floor ? top_floor_.p : top_lit_[world_->items[d].variant % 3].p, 0,
                                  unsigned(draw.instances.size()), false));
    }
    ring_ = ring_draw_.size();
    chunk_begin_.clear();
    for (unsigned t = 0; t <= threads_; ++t) chunk_begin_.push_back(main_calls_.size() * t / threads_);
}

bool ScaleGpu::fill_descriptors() {
    const D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, UINT(kRingBase + 2 * ring_),
                                              D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC staging_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, UINT(2 * ring_),
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap_.p))));
    CHECK(SUCCEEDED(device_->CreateDescriptorHeap(&staging_desc, IID_PPV_ARGS(&staging_heap_.p))));
    srv_cpu_ = heap_cpu(srv_heap_.p);
    srv_gpu_ = heap_gpu(srv_heap_.p);
    staging_cpu_ = heap_cpu(staging_heap_.p);
    step_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CHECK(srv_cpu_ && staging_cpu_ && srv_gpu_ >= 0x10000 && step_ == 32);   // the rise reads its second V# at +0x20
    const auto put = [&](std::uint64_t base, std::uint64_t index, const std::array<std::uint32_t, 4>& v) {
        std::memcpy(reinterpret_cast<void*>(base + index * step_), v.data(), 16);
    };
    const Scene& scene = world_->scene;
    const std::uint64_t vb = vb_->GetGPUVirtualAddress();
    CHECK(scene.meshes.size() <= kTableA);
    for (std::size_t m = 0; m < scene.meshes.size(); ++m)
        put(srv_cpu_, kMeshSlot + m,
            make_vsharp(vb + vertex_first_[m] * sizeof(Vertex), sizeof(Vertex), std::uint32_t(vertex_count_[m])));
    for (unsigned t = 0; t < 3; ++t) {   // tables A0, A1 (8 slots apart), B
        ID3D12Resource* state = t < 2 ? state_a_[t].p : state_b_.p;
        ID3D12Resource* inst = t < 2 ? inst_a_[t].p : inst_b_.p;
        const std::uint64_t at = t < 2 ? kTableA + 8 * t : kTableB;
        put(srv_cpu_, at, make_vsharp(state->GetGPUVirtualAddress(), sizeof(Particle), kParticles));
        put(srv_cpu_, at + 1, make_vsharp(inst->GetGPUVirtualAddress(), sizeof(Instance), kParticles));
    }
    put(srv_cpu_, kDummySrv, make_vsharp(vb, sizeof(Vertex), 1));
    put(srv_cpu_, kDummySrv + 1, make_vsharp(vb, sizeof(Vertex), 1));
    put(srv_cpu_, kDummyUav, make_raw(pool_->GetGPUVirtualAddress(), 256));
    for (auto* calls : {&main_calls_, &aux_calls_, &top_calls_})
        for (Call& c : *calls) c.mesh_table = srv_gpu_ + (kMeshSlot + scene.draws[ring_draw_[c.ring]].mesh) * step_;
    for (unsigned p = 0; p < 2; ++p)
        for (std::uint64_t i = 0; i < ring_; ++i) {
            const unsigned d = ring_draw_[i];
            const Item& item = world_->items[d];
            const auto count = std::uint32_t(scene.draws[d].instances.size());
            std::array<std::uint32_t, 4> v;
            if (ring_set_b_[i]) v = make_vsharp(inst_b_->GetGPUVirtualAddress(), sizeof(Instance), kParticles);
            else if (item.particles) v = make_vsharp(inst_a_[p]->GetGPUVirtualAddress(), sizeof(Instance), kParticles);
            else if (item.streamed)
                v = make_vsharp(stream_[p]->GetGPUVirtualAddress() + inst_offset_[d], sizeof(Instance), count);
            else
                v = make_vsharp(static_inst_->GetGPUVirtualAddress() + inst_offset_[d], sizeof(Instance), count);
            put(staging_cpu_, p * ring_ + i, v);
        }
    return true;
}

bool ScaleGpu::initialize() {
    ID3D12CommandAllocator* alloc = main_alloc_[0][0].p;
    ID3D12GraphicsCommandList* list = main_list_[0][0].p;
    CHECK(SUCCEEDED(alloc->Reset()));
    CHECK(SUCCEEDED(list->Reset(alloc, nullptr)));
    const std::uint64_t vb_bytes = vertex_first_.empty() ? 0 : (vertex_first_.back() + vertex_count_.back()) * sizeof(Vertex);
    list->CopyBufferRegion(vb_.p, 0, upload_.p, up_vb_, vb_bytes);
    list->CopyBufferRegion(ib_.p, 0, upload_.p, up_ib_, align((first_index_.back() + index_count_.back()) * 2, 4));
    if (inst_bytes_) list->CopyBufferRegion(static_inst_.p, 0, upload_.p, up_inst_, inst_bytes_);
    list->CopyBufferRegion(pristine_a_.p, 0, upload_.p, up_particles_, kParticles * sizeof(Particle));
    list->CopyBufferRegion(pristine_b_.p, 0, upload_.p, up_particles_, kParticles * sizeof(Particle));
    const auto c = D3D12_RESOURCE_STATE_COMMON, cd = D3D12_RESOURCE_STATE_COPY_DEST,
               npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, cs = D3D12_RESOURCE_STATE_COPY_SOURCE;
    const D3D12_RESOURCE_BARRIER b[] = {
        transition(vb_.p, cd, npsr), transition(ib_.p, cd, D3D12_RESOURCE_STATE_INDEX_BUFFER),
        transition(static_inst_.p, cd, npsr), transition(pristine_a_.p, cd, c), transition(pristine_b_.p, cd, c),
        transition(main_rt_.p, c, cs), transition(aux_resolved_.p, c, cs), transition(top_rt_.p, c, cs)};
    list->ResourceBarrier(UINT(std::size(b)), b);
    CHECK(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list};
    direct_->ExecuteCommandLists(1, lists);
    Ref<ID3D12Fence> init;
    CHECK(SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&init.p))));
    CHECK(SUCCEEDED(direct_->Signal(init.p, 1)));
    CHECK(wait_fence(init.p, 1, 60000) == Outcome::ok);
    return true;
}

void ScaleGpu::barrier(ID3D12GraphicsCommandList* list, std::initializer_list<D3D12_RESOURCE_BARRIER> barriers,
                       FrameStats& st) {
    TIMED(kResourceBarrier, list->ResourceBarrier(UINT(barriers.size()), barriers.begin()));
    st.barriers += unsigned(barriers.size());
}

void ScaleGpu::begin_pass(ID3D12GraphicsCommandList* list, const Pass& pass, FrameStats&) {
    ID3D12DescriptorHeap* heaps[] = {srv_heap_.p};
    TIMED(kSetDescriptorHeaps, list->SetDescriptorHeaps(1, heaps));
    const D3D12_INDEX_BUFFER_VIEW ibv{ib_->GetGPUVirtualAddress(),
                                      UINT(align((first_index_.back() + index_count_.back()) * 2, 4)), DXGI_FORMAT_R16_UINT};
    TIMED(kSetIndexBufferTopology, list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
    TIMED(kSetIndexBufferTopology, list->IASetIndexBuffer(&ibv));
    const D3D12_VIEWPORT viewport{0, 0, float(pass.width), float(pass.height), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(pass.width), LONG(pass.height)};
    TIMED(kSetTargetsViewports, list->RSSetViewports(1, &viewport));
    TIMED(kSetTargetsViewports, list->RSSetScissorRects(1, &scissor));
    TIMED(kSetTargetsViewports, list->OMSetRenderTargets(1, &pass.rtv, FALSE, &pass.dsv));
}

void ScaleGpu::bind_root_signature(ID3D12GraphicsCommandList* list, unsigned rs, std::uint64_t camera) {
    TIMED(kSetGraphicsRootSignature, list->SetGraphicsRootSignature(rs_[rs].p));
    TIMED(kSetRootCbv, list->SetGraphicsRootConstantBufferView(3, camera));
    if (rs == 1) {
        const UINT k[4] = {1, 2, 3, 4};
        TIMED(kSetRootConstants, list->SetGraphicsRoot32BitConstants(4, 4, k, 0));
    } else if (rs == 2) {
        TIMED(kSetRootTable, list->SetGraphicsRootDescriptorTable(4, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + kDummySrv * step_}));
        TIMED(kSetRootTable, list->SetGraphicsRootDescriptorTable(5, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + kDummyUav * step_}));
    } else if (rs == 3) {
        TIMED(kSetRootCbv, list->SetGraphicsRootConstantBufferView(4, upload_->GetGPUVirtualAddress() + up_extra_));
        const UINT k[2] = {5, 6};
        TIMED(kSetRootConstants, list->SetGraphicsRoot32BitConstants(5, 2, k, 0));
    }
}

void ScaleGpu::record_calls(ID3D12GraphicsCommandList* list, unsigned p, const Pass& pass, const Call* calls,
                            std::size_t n, FrameStats& st) {
    unsigned rs = ~0u;
    ID3D12PipelineState* pso = nullptr;
    for (std::size_t i = 0; i < n; ++i) {
        const Call& c = calls[i];
        if (c.rs != rs) {
            bind_root_signature(list, c.rs, pass.camera);
            rs = c.rs;
            ++st.rs_changes;
        }
        if (c.pso != pso) {
            TIMED(kSetPipelineState, list->SetPipelineState(c.pso));
            pso = c.pso;
            ++st.pso_changes;
        }
        const std::uint64_t slot = kRingBase + p * ring_ + c.ring;
        TIMED(kCopyDescriptors, device_->CopyDescriptorsSimple(
                                    1, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu_ + slot * step_)},
                                    D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(staging_cpu_ + (p * ring_ + c.ring) * step_)},
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        TIMED(kSetRootTable, list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{c.mesh_table}));
        TIMED(kSetRootTable, list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + slot * step_}));
        TIMED(kSetRootConstants, list->SetGraphicsRoot32BitConstants(2, 16, c.constants, 0));
        TIMED(kDraw, list->DrawIndexedInstanced(c.index_count, c.instances, c.first_index, 0, 0));
        ++st.draws;
        ++st.descriptor_copies;
    }
}

void ScaleGpu::compute_steps(ID3D12GraphicsCommandList* list, std::uint64_t table, unsigned steps,
                             ID3D12Resource* state, ID3D12Resource* instances, FrameStats& st) {
    TIMED(kSetComputeRootSignature, list->SetComputeRootSignature(compute_rs_.p));
    ID3D12DescriptorHeap* heaps[] = {srv_heap_.p};
    TIMED(kSetDescriptorHeaps, list->SetDescriptorHeaps(1, heaps));
    const UINT constants[4] = {UINT(kTop), UINT(kRange), kParticles, 0};
    TIMED(kSetRootConstants, list->SetComputeRoot32BitConstants(0, 4, constants, 0));
    TIMED(kSetRootTable, list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu_ + table * step_}));
    TIMED(kSetPipelineState, list->SetPipelineState(rise_.p));
    const D3D12_RESOURCE_BARRIER uav[2] = {uav_barrier(state), uav_barrier(instances)};
    for (unsigned k = 0; k < steps; ++k) {
        TIMED(kDispatch, list->Dispatch(kParticles / 64, 1, 1));
        TIMED(kResourceBarrier, list->ResourceBarrier(2, uav));
        ++st.dispatches;
        st.barriers += 2;
    }
}

bool ScaleGpu::record_chunk(unsigned p, unsigned t) {
    FrameStats& st = worker_stats_[t];
    ID3D12CommandAllocator* alloc = chunk_alloc_[p][t].p;
    ID3D12GraphicsCommandList* list = chunk_list_[p][t].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = alloc->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(alloc, nullptr));
    CHECK(SUCCEEDED(hr));
    begin_pass(list, main_pass_, st);
    record_calls(list, p, main_pass_, main_calls_.data() + chunk_begin_[t], chunk_begin_[t + 1] - chunk_begin_[t], st);
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

bool ScaleGpu::record_copy(unsigned p) {
    FrameStats& st = main_stats_;
    ID3D12GraphicsCommandList* list = copy_list_[p].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = copy_alloc_[p]->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(copy_alloc_[p].p, nullptr));
    CHECK(SUCCEEDED(hr));
    const auto c = D3D12_RESOURCE_STATE_COMMON, cd = D3D12_RESOURCE_STATE_COPY_DEST;
    if (stream_bytes_) {   // poison first, then the records: a frame that reads the poison draws nothing there
        barrier(list, {transition(stream_[p].p, c, cd)}, st);
        TIMED(kCopyBuffer, list->CopyBufferRegion(stream_[p].p, 0, upload_.p, up_zero_, stream_bytes_));
        barrier(list, {transition(stream_[p].p, cd, c)}, st);
        barrier(list, {transition(stream_[p].p, c, cd)}, st);
        TIMED(kCopyBuffer, list->CopyBufferRegion(stream_[p].p, 0, upload_.p, up_stream_, stream_bytes_));
        barrier(list, {transition(stream_[p].p, cd, c)}, st);
        st.copies += 2;
    }
    barrier(list, {transition(pool_.p, c, cd)}, st);
    for (std::uint64_t at = 0; at < kPoolBytes; at += kPoolChunk) {
        TIMED(kCopyBuffer, list->CopyBufferRegion(pool_.p, at, pool_upload_.p, at, kPoolChunk));
        ++st.copies;
    }
    barrier(list, {transition(pool_.p, cd, c)}, st);
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

bool ScaleGpu::record_compute(unsigned p) {
    FrameStats& st = main_stats_;
    ID3D12GraphicsCommandList* list = compute_list_[p].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = compute_alloc_[p]->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(compute_alloc_[p].p, nullptr));
    CHECK(SUCCEEDED(hr));
    const auto c = D3D12_RESOURCE_STATE_COMMON, cd = D3D12_RESOURCE_STATE_COPY_DEST, cs = D3D12_RESOURCE_STATE_COPY_SOURCE,
               u = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier(list, {transition(pristine_a_.p, c, cs), transition(state_a_[p].p, c, cd)}, st);
    TIMED(kCopyBuffer, list->CopyBufferRegion(state_a_[p].p, 0, pristine_a_.p, 0, kParticles * sizeof(Particle)));
    ++st.copies;
    barrier(list, {transition(state_a_[p].p, cd, u), transition(inst_a_[p].p, c, u), transition(pristine_a_.p, cs, c)}, st);
    compute_steps(list, kTableA + 8 * p, kStepsA, state_a_[p].p, inst_a_[p].p, st);
    barrier(list, {transition(state_a_[p].p, u, c), transition(inst_a_[p].p, u, c)}, st);
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

bool ScaleGpu::record_pre(unsigned p) {
    FrameStats& st = main_stats_;
    ID3D12GraphicsCommandList* list = main_list_[p][0].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = main_alloc_[p][0]->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(main_alloc_[p][0].p, nullptr));
    CHECK(SUCCEEDED(hr));
    barrier(list, {transition(main_rt_.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)}, st);
    TIMED(kClearView, list->ClearRenderTargetView(main_pass_.rtv, kClear.data(), 0, nullptr));
    TIMED(kClearView, list->ClearDepthStencilView(main_pass_.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr));
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

bool ScaleGpu::record_bind(unsigned p) {
    FrameStats& st = main_stats_;
    ID3D12GraphicsCommandList* list = main_list_[p][1].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = main_alloc_[p][1]->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(main_alloc_[p][1].p, nullptr));
    CHECK(SUCCEEDED(hr));
    const auto c = D3D12_RESOURCE_STATE_COMMON, npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (stream_bytes_) barrier(list, {transition(stream_[p].p, c, npsr), transition(inst_a_[p].p, c, npsr)}, st);
    else barrier(list, {transition(inst_a_[p].p, c, npsr)}, st);
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

bool ScaleGpu::record_post(unsigned p) {
    FrameStats& st = main_stats_;
    ID3D12GraphicsCommandList* list = main_list_[p][2].p;
    HRESULT hr = S_OK;
    TIMED(kResetAllocator, hr = main_alloc_[p][2]->Reset());
    CHECK(SUCCEEDED(hr));
    TIMED(kResetList, hr = list->Reset(main_alloc_[p][2].p, nullptr));
    CHECK(SUCCEEDED(hr));
    const auto c = D3D12_RESOURCE_STATE_COMMON, cd = D3D12_RESOURCE_STATE_COPY_DEST, cs = D3D12_RESOURCE_STATE_COPY_SOURCE,
               u = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               rt = D3D12_RESOURCE_STATE_RENDER_TARGET, rs = D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
               rd = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    if (stream_bytes_) barrier(list, {transition(stream_[p].p, npsr, c)}, st);
    barrier(list, {transition(main_rt_.p, rt, cs), transition(inst_a_[p].p, npsr, cs),
                   transition(pristine_b_.p, c, cs), transition(state_b_.p, c, cd)}, st);
    // Compute between the passes, on the direct queue: particle set B, which the aux pass draws.
    TIMED(kCopyBuffer, list->CopyBufferRegion(state_b_.p, 0, pristine_b_.p, 0, kParticles * sizeof(Particle)));
    ++st.copies;
    barrier(list, {transition(state_b_.p, cd, u), transition(inst_b_.p, c, u), transition(pristine_b_.p, cs, c)}, st);
    compute_steps(list, kTableB, kStepsB, state_b_.p, inst_b_.p, st);
    barrier(list, {transition(state_b_.p, u, c), transition(inst_b_.p, u, npsr), transition(aux_msaa_.p, rs, rt)}, st);
    // The aux pass: 4x MSAA, resolved.
    TIMED(kClearView, list->ClearRenderTargetView(aux_pass_.rtv, kClear.data(), 0, nullptr));
    TIMED(kClearView, list->ClearDepthStencilView(aux_pass_.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr));
    begin_pass(list, aux_pass_, st);
    record_calls(list, p, aux_pass_, aux_calls_.data(), aux_calls_.size(), st);
    barrier(list, {transition(aux_msaa_.p, rt, rs), transition(aux_resolved_.p, cs, rd)}, st);
    TIMED(kResolve, list->ResolveSubresource(aux_resolved_.p, 0, aux_msaa_.p, 0, DXGI_FORMAT_R16G16B16A16_FLOAT));
    barrier(list, {transition(aux_resolved_.p, rd, cs), transition(top_rt_.p, cs, rt)}, st);
    // The top pass: RGBA8.
    TIMED(kClearView, list->ClearRenderTargetView(top_pass_.rtv, kClear.data(), 0, nullptr));
    TIMED(kClearView, list->ClearDepthStencilView(top_pass_.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr));
    begin_pass(list, top_pass_, st);
    record_calls(list, p, top_pass_, top_calls_.data(), top_calls_.size(), st);
    barrier(list, {transition(top_rt_.p, rt, cs), transition(inst_b_.p, npsr, cs)}, st);
    // Everything to the readback.
    const auto copy_image = [&](ID3D12Resource* src_res, std::uint64_t offset, DXGI_FORMAT format, unsigned w, unsigned h,
                                std::uint64_t pitch) {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback_[p].p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {offset, {format, w, h, 1, UINT(pitch)}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = src_res;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        TIMED(kCopyTexture, list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr));
        ++st.copies;
    };
    copy_image(main_rt_.p, kOffMain, DXGI_FORMAT_R16G16B16A16_FLOAT, kW, kH, kMainPitch);
    copy_image(aux_resolved_.p, kOffAux, DXGI_FORMAT_R16G16B16A16_FLOAT, kAuxW, kAuxH, kAuxPitch);
    copy_image(top_rt_.p, kOffTop, DXGI_FORMAT_R8G8B8A8_UNORM, kTopW, kTopH, kTopPitch);
    TIMED(kCopyBuffer, list->CopyBufferRegion(readback_[p].p, kOffPartA, inst_a_[p].p, 0, kParticles * sizeof(Instance)));
    TIMED(kCopyBuffer, list->CopyBufferRegion(readback_[p].p, kOffPartB, inst_b_.p, 0, kParticles * sizeof(Instance)));
    st.copies += 2;
    barrier(list, {transition(inst_a_[p].p, cs, c), transition(inst_b_.p, cs, c)}, st);
    TIMED(kCloseList, hr = list->Close());
    CHECK(hr == S_OK);
    ++st.lists;
    return true;
}

void ScaleGpu::jitter() {
    if (!jitter_us_) return;
    std::uint64_t x = jitter_state_.fetch_add(0x9e3779b97f4a7c15ull) + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x ^= x >> 31;
    const unsigned us = unsigned(x % (jitter_us_ + 1));
    if (us >= 1000) Sleep(us / 1000);
    else if (us > 50) {   // a short spin: Sleep cannot wait less than a millisecond
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
        while (std::chrono::steady_clock::now() < until) YieldProcessor();
    }
}

void ScaleGpu::worker(unsigned t) {
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t f;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return quit_ || job_frame_ != seen; });
            if (quit_) return;
            f = seen = job_frame_;
        }
        jitter();
        const bool ok = record_chunk(unsigned(f & 1), t);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!ok) worker_failed_ = true;
            ++done_;
        }
        done_cv_.notify_all();
    }
}

void ScaleGpu::log_event(std::uint64_t f, const char* what, std::uint64_t value) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    const double ms = double(now.QuadPart - qpc0_.QuadPart) * 1000.0 / double(qpf_.QuadPart);
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_.size() > 8192) log_.erase(log_.begin(), log_.begin() + 4096);
    log_.push_back({f, ms, what, value});
}

std::string ScaleGpu::frame_log(std::uint64_t f) const {
    std::string out;
    char line[256];
    auto& self = const_cast<ScaleGpu&>(*this);
    std::lock_guard<std::mutex> lock(self.log_mutex_);
    for (const Event& e : log_) {
        if (e.frame + 3 < f || e.frame > f + 1) continue;
        std::snprintf(line, sizeof(line), "frame %llu  %12.3f ms  %s %llu\n", (unsigned long long)e.frame, e.ms, e.what,
                      (unsigned long long)e.value);
        out += line;
    }
    return out;
}

std::string ScaleGpu::fence_status() const {
    char text[256];
    std::snprintf(text, sizeof(text), "fences completed: direct %llu, compute %llu, copy %llu (last frame submitted %llu)",
                  (unsigned long long)direct_fence_->GetCompletedValue(),
                  (unsigned long long)compute_fence_->GetCompletedValue(),
                  (unsigned long long)copy_fence_->GetCompletedValue(), (unsigned long long)job_frame_);
    return text;
}

ScaleGpu::Outcome ScaleGpu::wait_fence(ID3D12Fence* fence, std::uint64_t value, unsigned timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (fence->GetCompletedValue() < value) {
        HRESULT hr = S_OK;
        TIMED(kFenceWaitCpu, hr = fence->SetEventOnCompletion(value, event_));
        if (FAILED(hr)) return Outcome::error;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) return Outcome::hang;
        WaitForSingleObject(event_, DWORD(left.count()));
    }
    return Outcome::ok;
}

ScaleGpu::Outcome ScaleGpu::submit(std::uint64_t f, unsigned timeout_ms) {
    const unsigned p = unsigned(f & 1);
    if (f > 2) {
        const Outcome o = wait_fence(direct_fence_.p, f - 2, timeout_ms);
        if (o != Outcome::ok) return o;
    }
    log_event(f, "record start");
    main_stats_ = FrameStats{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_stats_.assign(threads_, FrameStats{});
        done_ = 0;
        worker_failed_ = false;
        job_frame_ = f;
    }
    cv_.notify_all();
    const bool recorded = record_copy(p) && record_compute(p) && record_pre(p) && record_bind(p) && record_post(p);
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!done_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return done_ == threads_; })) {
            std::fprintf(stderr, "lanterns: frame %llu: the recording threads did not finish in %u ms\n",
                         (unsigned long long)f, timeout_ms);
            return Outcome::hang;
        }
        if (worker_failed_ || !recorded) return Outcome::error;
    }
    log_event(f, "recorded");
    HRESULT hr = S_OK;
    const auto run = [&](ID3D12CommandQueue* queue, std::initializer_list<ID3D12CommandList*> lists) {
        TIMED(kExecuteSmall, queue->ExecuteCommandLists(UINT(lists.size()), lists.begin()));
        main_stats_.executes += 1;
    };
    jitter();
    if (f > 2) TIMED(kQueueWait, hr = copy_->Wait(direct_fence_.p, f - 2));
    run(copy_.p, {copy_list_[p].p});
    TIMED(kSignal, hr = copy_->Signal(copy_fence_.p, f));
    log_event(f, "copy queue: Wait(direct, f-2), Execute, Signal(copy)", f);
    jitter();
    if (f > 2) TIMED(kQueueWait, hr = compute_->Wait(direct_fence_.p, f - 2));
    run(compute_.p, {compute_list_[p].p});
    TIMED(kSignal, hr = compute_->Signal(compute_fence_.p, f));
    log_event(f, "compute queue: Wait(direct, f-2), Execute, Signal(compute)", f);
    jitter();
    run(direct_.p, {main_list_[p][0].p});
    jitter();
    TIMED(kQueueWait, hr = direct_->Wait(copy_fence_.p, f));
    TIMED(kQueueWait, hr = direct_->Wait(compute_fence_.p, f));
    std::vector<ID3D12CommandList*> lists = {main_list_[p][1].p};
    for (unsigned t = 0; t < threads_; ++t) lists.push_back(chunk_list_[p][t].p);
    lists.push_back(main_list_[p][2].p);
    TIMED(kExecuteCommandLists, direct_->ExecuteCommandLists(UINT(lists.size()), lists.data()));
    main_stats_.executes += 1;
    TIMED(kSignal, hr = direct_->Signal(direct_fence_.p, f));
    log_event(f, "direct queue: Execute(pre), Wait(copy f), Wait(compute f), Execute(bind, chunks, post), Signal(direct)", f);
    FrameStats sum = main_stats_;
    for (const FrameStats& w : worker_stats_) {
        sum.draws += w.draws, sum.dispatches += w.dispatches, sum.lists += w.lists, sum.barriers += w.barriers;
        sum.descriptor_copies += w.descriptor_copies, sum.pso_changes += w.pso_changes, sum.rs_changes += w.rs_changes;
    }
    stats_ = sum;
    return Outcome::ok;
}

ScaleGpu::Outcome ScaleGpu::wait(std::uint64_t f, unsigned timeout_ms) {
    const Outcome o = wait_fence(direct_fence_.p, f, timeout_ms);
    log_event(f, o == Outcome::ok ? "direct fence reached" : "direct fence NOT reached", f);
    return o;
}

const std::uint8_t* ScaleGpu::map_readback(std::uint64_t f) {
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(kReadbackBytes)};
    HRESULT hr = S_OK;
    TIMED(kMapUnmap, hr = readback_[f & 1]->Map(0, &range, &mapped));
    return SUCCEEDED(hr) ? static_cast<const std::uint8_t*>(mapped) : nullptr;
}

void ScaleGpu::unmap_readback(std::uint64_t f) {
    const D3D12_RANGE none{0, 0};
    TIMED(kMapUnmap, readback_[f & 1]->Unmap(0, &none));
}

bool ScaleGpu::idle(unsigned timeout_ms) {
    if (!job_frame_) return true;
    return wait_fence(direct_fence_.p, job_frame_, timeout_ms) == Outcome::ok &&
           wait_fence(compute_fence_.p, job_frame_, timeout_ms) == Outcome::ok &&
           wait_fence(copy_fence_.p, job_frame_, timeout_ms) == Outcome::ok;
}

}  // namespace lanterns::scale
