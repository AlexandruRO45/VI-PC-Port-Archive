// Differential synthetic oracle for predicated CopyResource, CopyTextureRegion (upload, readback,
// texture-to-texture, BC, depth/stencil, rgb96, 3D, cross-format), ResolveSubresource / ResolveSubresourceRegion
// (color & depth/stencil), ClearUnorderedAccessViewUint / Float (buffer & 3D/2D textures), AtomicCopyBufferUINT /
// UINT64, and PIPELINE_STATISTICS isolation (D3D-162). Compares Vulkan against Windows WARP.
// --warp-only measures the native reference without Vulkan. No game data.
#include "host_wine.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <dxgi1_4.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { \
    if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return false; } \
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

struct PredCase {
    UINT64 value;
    D3D12_PREDICATION_OP op;
    bool enabled;
    bool executes() const {
        if (!enabled) return true;
        return op == D3D12_PREDICATION_OP_EQUAL_ZERO ? (value != 0) : (value == 0);
    }
};

constexpr PredCase kCorePreds[] = {
    {0, D3D12_PREDICATION_OP_EQUAL_ZERO, true},               // skip
    {UINT64_C(1) << 32, D3D12_PREDICATION_OP_EQUAL_ZERO, true},  // run (high word only)
    {0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO, true},           // run
    {UINT64_MAX, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO, true},  // skip
    {0, D3D12_PREDICATION_OP_EQUAL_ZERO, false},              // unpredicated control
};

bool make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                 Ref<ID3D12Resource>& out, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    const D3D12_HEAP_PROPERTIES props{heap};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p))));
    return true;
}

bool make_tex2d(ID3D12Device* dev, DXGI_FORMAT fmt, UINT w, UINT h, UINT16 array_size, UINT16 mips, UINT samples,
                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out,
                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = array_size;
    desc.MipLevels = mips;
    desc.Format = fmt;
    desc.SampleDesc.Count = samples;
    desc.Flags = flags;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p))));
    return true;
}

bool make_tex3d(ID3D12Device* dev, DXGI_FORMAT fmt, UINT w, UINT h, UINT16 depth,
                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out,
                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = depth;
    desc.MipLevels = 1;
    desc.Format = fmt;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p))));
    return true;
}

bool write_buf(ID3D12Resource* res, const void* bytes, SIZE_T size) {
    void* mapped = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(res->Map(0, &none, &mapped)) && mapped);
    std::memcpy(mapped, bytes, size);
    const D3D12_RANGE written{0, size};
    res->Unmap(0, &written);
    return true;
}

bool read_buf(ID3D12Resource* res, SIZE_T size, std::vector<unsigned char>& out) {
    void* mapped = nullptr;
    const D3D12_RANGE range{0, size};
    CHECK(SUCCEEDED(res->Map(0, &range, &mapped)) && mapped);
    out.assign(static_cast<const unsigned char*>(mapped), static_cast<const unsigned char*>(mapped) + size);
    const D3D12_RANGE none{};
    res->Unmap(0, &none);
    return true;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* res,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
                UINT sub = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = sub;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}

struct CmdCtx {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> pred;

    bool init(ID3D12Device* dev, UINT64 pred_value) {
        const D3D12_COMMAND_QUEUE_DESC qd{};
        CHECK(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))));
        CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
        CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        CHECK(make_buffer(dev, 24, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, pred));
        const UINT64 words[3] = {UINT64_MAX, pred_value, 0};
        CHECK(write_buf(pred.p, words, sizeof(words)));
        return true;
    }

    void begin_pred(const PredCase& pc) {
        if (pc.enabled) list->SetPredication(pred.p, 8, pc.op);
    }

    void end_pred() {
        list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    }

    bool submit_and_wait() {
        CHECK(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        CHECK(ev);
        const HRESULT hr = fence->SetEventOnCompletion(1, ev);
        const DWORD w = SUCCEEDED(hr) ? WaitForSingleObject(ev, 10000) : WAIT_FAILED;
        CloseHandle(ev);
        CHECK(SUCCEEDED(hr) && w == WAIT_OBJECT_0);
        return true;
    }
};

std::vector<unsigned char> fill_pattern(SIZE_T size, unsigned mul, unsigned add) {
    std::vector<unsigned char> v(size);
    for (SIZE_T i = 0; i < size; ++i) v[i] = static_cast<unsigned char>(i * mul + add);
    return v;
}

// Helper: copy all subresources of a 2D/3D texture into a readback buffer and pack exact rows (ignoring padding).
bool readback_texture_packed(ID3D12Device* dev, CmdCtx& ctx, ID3D12Resource* tex,
                             D3D12_RESOURCE_STATES current_state, UINT sub_count,
                             std::vector<unsigned char>& packed_out) {
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(sub_count);
    std::vector<UINT> num_rows(sub_count);
    std::vector<UINT64> row_sizes(sub_count);
    UINT64 total = 0;
    dev->GetCopyableFootprints(&desc, 0, sub_count, 0, fp.data(), num_rows.data(), row_sizes.data(), &total);

    Ref<ID3D12Resource> rb;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    if (current_state != D3D12_RESOURCE_STATE_COPY_SOURCE)
        transition(ctx.list.p, tex, current_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT s = 0; s < sub_count; ++s) {
        D3D12_TEXTURE_COPY_LOCATION src_loc{}, dst_loc{};
        src_loc.pResource = tex;
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src_loc.SubresourceIndex = s;
        dst_loc.pResource = rb.p;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst_loc.PlacedFootprint = fp[s];
        ctx.list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    }
    CHECK(ctx.submit_and_wait());
    std::vector<unsigned char> raw;
    CHECK(read_buf(rb.p, SIZE_T(total), raw));
    packed_out.clear();
    for (UINT s = 0; s < sub_count; ++s) {
        const UINT slices = fp[s].Footprint.Depth;
        const UINT64 slice_pitch = UINT64(num_rows[s]) * fp[s].Footprint.RowPitch;
        for (UINT z = 0; z < slices; ++z) {
            for (UINT y = 0; y < num_rows[s]; ++y) {
                const SIZE_T off = SIZE_T(fp[s].Offset + z * slice_pitch + y * fp[s].Footprint.RowPitch);
                packed_out.insert(packed_out.end(), raw.begin() + off, raw.begin() + off + SIZE_T(row_sizes[s]));
            }
        }
    }
    return true;
}

// 1. CopyResource: buffer <-> buffer
bool test_copy_resource_buffer(ID3D12Device* dev, const PredCase& pc, std::vector<unsigned char>& out) {
    constexpr UINT64 sz = 257;
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    Ref<ID3D12Resource> src_up, seed_up, src, dst, rb;
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, src_up));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, src));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    const auto pat_src = fill_pattern(sz, 37, 11);
    const auto pat_dst = fill_pattern(sz, 13, 173);
    CHECK(write_buf(src_up.p, pat_src.data(), sz) && write_buf(seed_up.p, pat_dst.data(), sz));
    ctx.list->CopyResource(src.p, src_up.p);
    ctx.list->CopyResource(dst.p, seed_up.p);
    transition(ctx.list.p, src.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    ctx.begin_pred(pc);
    ctx.list->CopyResource(dst.p, src.p);
    ctx.end_pred();
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ctx.list->CopyResource(rb.p, dst.p);
    CHECK(ctx.submit_and_wait());
    CHECK(read_buf(rb.p, sz, out));
    CHECK(out == (pc.executes() ? pat_src : pat_dst));
    return true;
}

// 2. CopyResource: 2D textures (RGBA8 multi-mip+array, BC1, D32_FLOAT, D24_UNORM_S8_UINT)
bool test_copy_resource_texture(ID3D12Device* dev, DXGI_FORMAT fmt, UINT w, UINT h, UINT16 layers, UINT16 mips,
                                UINT planes, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    const bool depth = (fmt == DXGI_FORMAT_D32_FLOAT || fmt == DXGI_FORMAT_D24_UNORM_S8_UINT);
    const D3D12_RESOURCE_FLAGS flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_NONE;
    Ref<ID3D12Resource> src, dst;
    CHECK(make_tex2d(dev, fmt, w, h, layers, mips, 1, D3D12_RESOURCE_STATE_COPY_DEST, src, flags));
    CHECK(make_tex2d(dev, fmt, w, h, layers, mips, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst, flags));
    const UINT subs = UINT(layers) * mips * planes;
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(subs);
    std::vector<UINT> num_rows(subs);
    std::vector<UINT64> row_sizes(subs);
    UINT64 total = 0;
    dev->GetCopyableFootprints(&desc, 0, subs, 0, fp.data(), num_rows.data(), row_sizes.data(), &total);
    Ref<ID3D12Resource> src_up, dst_up;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, src_up));
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, dst_up));

    std::vector<unsigned char> buf_src(SIZE_T(total), 0), buf_dst(SIZE_T(total), 0);
    for (UINT s = 0; s < subs; ++s) {
        const bool is_depth_plane = depth && (s < UINT(layers) * mips);
        for (UINT y = 0; y < num_rows[s]; ++y) {
            const SIZE_T off = SIZE_T(fp[s].Offset + y * fp[s].Footprint.RowPitch);
            if (fmt == DXGI_FORMAT_D32_FLOAT) {
                for (UINT x = 0; x < w; ++x) {
                    const float vs = 0.125f + 0.03125f * float((x + y * w) & 7);
                    const float vd = 0.75f - 0.03125f * float((x + y * w) & 7);
                    std::memcpy(buf_src.data() + off + x * 4, &vs, 4);
                    std::memcpy(buf_dst.data() + off + x * 4, &vd, 4);
                }
            } else if (fmt == DXGI_FORMAT_D24_UNORM_S8_UINT && is_depth_plane) {
                for (UINT x = 0; x < w; ++x) {
                    const UINT32 vs = 0x00246800u + (x * 17u + y * 31u) * 256u;
                    const UINT32 vd = 0x00C0A000u + (x * 13u + y * 29u) * 256u;
                    std::memcpy(buf_src.data() + off + x * 4, &vs, 4);
                    std::memcpy(buf_dst.data() + off + x * 4, &vd, 4);
                }
            } else {
                for (SIZE_T b = 0; b < SIZE_T(row_sizes[s]); ++b) {
                    buf_src[off + b] = static_cast<unsigned char>(0x20 + ((s * 53 + y * 17 + b * 7) & 0x7F));
                    buf_dst[off + b] = static_cast<unsigned char>(0xA0 + ((s * 31 + y * 11 + b * 5) & 0x5F));
                }
            }
        }
    }
    CHECK(write_buf(src_up.p, buf_src.data(), SIZE_T(total)));
    CHECK(write_buf(dst_up.p, buf_dst.data(), SIZE_T(total)));
    for (UINT s = 0; s < subs; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dloc{}, sloc{};
        dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dloc.SubresourceIndex = s;
        sloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        sloc.PlacedFootprint = fp[s];
        dloc.pResource = src.p;
        sloc.pResource = src_up.p;
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
        dloc.pResource = dst.p;
        sloc.pResource = dst_up.p;
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    }
    transition(ctx.list.p, src.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ctx.begin_pred(pc);
    ctx.list->CopyResource(dst.p, src.p);
    ctx.end_pred();
    CHECK(readback_texture_packed(dev, ctx, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, subs, out));
    return true;
}

// 3a. CopyTextureRegion upload (2D RGBA8 partial box, BC1 partial box, RGB96, D24S8 depth+stencil, 3D partial slices)
bool test_copy_texture_upload(ID3D12Device* dev, int variant, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    Ref<ID3D12Resource> tex;
    UINT subs = 1;
    if (variant == 0) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 8, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex));
    } else if (variant == 1) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex));
    } else if (variant == 2) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R32G32B32_FLOAT, 4, 2, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex));
    } else if (variant == 3) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_D24_UNORM_S8_UINT, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex,
                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
        subs = 2;
    } else {
        CHECK(make_tex3d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 3, D3D12_RESOURCE_STATE_COPY_DEST, tex));
    }
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(subs);
    std::vector<UINT> num_rows(subs);
    std::vector<UINT64> row_sizes(subs);
    UINT64 total = 0;
    dev->GetCopyableFootprints(&desc, 0, subs, 0, fp.data(), num_rows.data(), row_sizes.data(), &total);

    Ref<ID3D12Resource> seed_up, pred_up;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, pred_up));
    std::vector<unsigned char> seed(SIZE_T(total), 0), payload(SIZE_T(total), 0);
    for (UINT s = 0; s < subs; ++s) {
        const UINT slices = fp[s].Footprint.Depth;
        const UINT64 sp = UINT64(num_rows[s]) * fp[s].Footprint.RowPitch;
        for (UINT z = 0; z < slices; ++z) {
            for (UINT y = 0; y < num_rows[s]; ++y) {
                const SIZE_T off = SIZE_T(fp[s].Offset + z * sp + y * fp[s].Footprint.RowPitch);
                if (variant == 2) {  // RGB96 float
                    for (UINT x = 0; x < 4; ++x) {
                        const float sf[3] = {float(x) + 1.0f, float(y) + 2.0f, 3.5f};
                        const float pf[3] = {float(x) + 10.25f, float(y) + 20.5f, 30.75f};
                        std::memcpy(seed.data() + off + x * 12, sf, 12);
                        std::memcpy(payload.data() + off + x * 12, pf, 12);
                    }
                } else if (variant == 3 && s == 0) {  // D24 depth
                    for (UINT x = 0; x < 4; ++x) {
                        const UINT32 sv = 0x00102000u + (x + y * 4) * 512u;
                        const UINT32 pv = 0x00A0B000u + (x + y * 4) * 1024u;
                        std::memcpy(seed.data() + off + x * 4, &sv, 4);
                        std::memcpy(payload.data() + off + x * 4, &pv, 4);
                    }
                } else {
                    for (SIZE_T b = 0; b < SIZE_T(row_sizes[s]); ++b) {
                        seed[off + b] = static_cast<unsigned char>(0xC0 + ((z * 19 + y * 13 + b * 3) & 0x3F));
                        payload[off + b] = static_cast<unsigned char>(0x10 + ((z * 23 + y * 7 + b * 11) & 0x6F));
                    }
                }
            }
        }
    }
    CHECK(write_buf(seed_up.p, seed.data(), SIZE_T(total)));
    CHECK(write_buf(pred_up.p, payload.data(), SIZE_T(total)));
    for (UINT s = 0; s < subs; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION sloc{seed_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp[s];
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    }
    ctx.begin_pred(pc);
    if (variant == 0) {
        D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sloc{pred_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp[0];
        const D3D12_BOX box{1, 1, 0, 6, 3, 1};
        ctx.list->CopyTextureRegion(&dloc, 2, 1, 0, &sloc, &box);
    } else if (variant == 1) {
        D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sloc{pred_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp[0];
        const D3D12_BOX box{0, 0, 0, 4, 4, 1};
        ctx.list->CopyTextureRegion(&dloc, 4, 4, 0, &sloc, &box);
    } else if (variant == 2) {
        D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sloc{pred_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp[0];
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    } else if (variant == 3) {
        for (UINT s = 0; s < 2; ++s) {
            D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            dloc.SubresourceIndex = s;
            D3D12_TEXTURE_COPY_LOCATION sloc{pred_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            sloc.PlacedFootprint = fp[s];
            ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
        }
    } else {
        D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sloc{pred_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp[0];
        const D3D12_BOX box{0, 0, 0, 4, 4, 2};
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 1, &sloc, &box);
    }
    ctx.end_pred();
    CHECK(readback_texture_packed(dev, ctx, tex.p, D3D12_RESOURCE_STATE_COPY_DEST, subs, out));
    return true;
}

// 3b. CopyTextureRegion readback (RGBA8 + RGB96): verifies inter-row padding stays untouched on TRUE and entire
// readback buffer stays untouched on FALSE!
bool test_copy_texture_readback(ID3D12Device* dev, bool rgb96, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    const DXGI_FORMAT fmt = rgb96 ? DXGI_FORMAT_R32G32B32_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    const UINT w = 4, h = rgb96 ? 2u : 3u;
    Ref<ID3D12Resource> tex;
    CHECK(make_tex2d(dev, fmt, w, h, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex));
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT num_rows = 0;
    UINT64 row_size = 0, total = 0;
    dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &num_rows, &row_size, &total);

    Ref<ID3D12Resource> src_up, seed_up, rb;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, src_up));
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));

    std::vector<unsigned char> tex_bytes(SIZE_T(total), 0);
    std::vector<unsigned char> sentinel(SIZE_T(total), 0xA5);
    for (UINT y = 0; y < num_rows; ++y) {
        const SIZE_T off = SIZE_T(fp.Offset + y * fp.Footprint.RowPitch);
        if (rgb96) {
            for (UINT x = 0; x < w; ++x) {
                const float f[3] = {float(x) + 0.5f, float(y) + 1.25f, float(x + y) + 2.75f};
                std::memcpy(tex_bytes.data() + off + x * 12, f, 12);
            }
        } else {
            for (SIZE_T b = 0; b < SIZE_T(row_size); ++b)
                tex_bytes[off + b] = static_cast<unsigned char>(0x30 + y * 29 + b * 7);
        }
    }
    CHECK(write_buf(src_up.p, tex_bytes.data(), SIZE_T(total)));
    CHECK(write_buf(seed_up.p, sentinel.data(), SIZE_T(total)));

    D3D12_TEXTURE_COPY_LOCATION tloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    tloc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION uploc{src_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    uploc.PlacedFootprint = fp;
    ctx.list->CopyTextureRegion(&tloc, 0, 0, 0, &uploc, nullptr);
    // Pre-seed readback buffer with 0xA5 so we detect both unwanted writes on FALSE and clobbered padding on TRUE.
    ctx.list->CopyBufferRegion(rb.p, 0, seed_up.p, 0, total);
    transition(ctx.list.p, tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);

    ctx.begin_pred(pc);
    D3D12_TEXTURE_COPY_LOCATION rbloc{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    rbloc.PlacedFootprint = fp;
    ctx.list->CopyTextureRegion(&rbloc, 0, 0, 0, &tloc, nullptr);
    ctx.end_pred();
    CHECK(ctx.submit_and_wait());
    CHECK(read_buf(rb.p, SIZE_T(total), out));

    // Verify padding bytes are STILL 0xA5 in every case!
    for (UINT y = 0; y + 1 < num_rows; ++y) {
        const SIZE_T pad_start = SIZE_T(fp.Offset + y * fp.Footprint.RowPitch + row_size);
        const SIZE_T pad_end = SIZE_T(fp.Offset + (y + 1) * fp.Footprint.RowPitch);
        for (SIZE_T p = pad_start; p < pad_end; ++p) CHECK(out[p] == 0xA5);
    }
    return true;
}

// 3c. CopyTextureRegion texture-to-texture:
// variant 0: RGBA8 partial box
// variant 1: BC1 -> R32G32_UINT (block-compressed to uncompressed)
// variant 2: R32G32_UINT -> BC1 (uncompressed to block-compressed)
// variant 3: D32_FLOAT -> R32_FLOAT (depth to color)
// variant 4: R32_FLOAT -> D32_FLOAT (color to depth)
bool test_copy_texture_tex2tex(ID3D12Device* dev, int variant, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    Ref<ID3D12Resource> src, dst;
    if (variant == 0) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 8, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, src));
        CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 8, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    } else if (variant == 1) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, src));
        CHECK(make_tex2d(dev, DXGI_FORMAT_R32G32_UINT, 2, 2, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    } else if (variant == 2) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R32G32_UINT, 2, 2, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, src));
        CHECK(make_tex2d(dev, DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    } else if (variant == 3) {
        CHECK(make_tex2d(dev, DXGI_FORMAT_D32_FLOAT, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, src,
                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
        CHECK(make_tex2d(dev, DXGI_FORMAT_R32_FLOAT, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    } else {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R32_FLOAT, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, src));
        CHECK(make_tex2d(dev, DXGI_FORMAT_D32_FLOAT, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, dst,
                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
    }

    const auto upload_tex = [&](ID3D12Resource* tex, Ref<ID3D12Resource>& up, unsigned mul, unsigned add, bool floats) {
        const D3D12_RESOURCE_DESC d = tex->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT nr = 0;
        UINT64 rs = 0, tot = 0;
        dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, &nr, &rs, &tot);
        CHECK(make_buffer(dev, tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, up));
        std::vector<unsigned char> data(SIZE_T(tot), 0);
        for (UINT y = 0; y < nr; ++y) {
            const SIZE_T off = SIZE_T(fp.Offset + y * fp.Footprint.RowPitch);
            if (floats) {
                for (UINT x = 0; x < d.Width; ++x) {
                    const float v = (float(add) / 256.0f) + 0.015625f * float((x * mul + y * 3) & 7);
                    std::memcpy(data.data() + off + x * 4, &v, 4);
                }
            } else {
                for (SIZE_T b = 0; b < SIZE_T(rs); ++b)
                    data[off + b] = static_cast<unsigned char>(add + (y * 19 + b * mul));
            }
        }
        CHECK(write_buf(up.p, data.data(), SIZE_T(tot)));
        D3D12_TEXTURE_COPY_LOCATION dloc{tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dloc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION sloc{up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        sloc.PlacedFootprint = fp;
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
        return true;
    };

    Ref<ID3D12Resource> up_src, up_dst;
    const bool is_float = (variant == 3 || variant == 4);
    CHECK(upload_tex(src.p, up_src, 7, 32, is_float));
    CHECK(upload_tex(dst.p, up_dst, 13, 192, is_float));
    transition(ctx.list.p, src.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);

    ctx.begin_pred(pc);
    D3D12_TEXTURE_COPY_LOCATION dloc{dst.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    dloc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION sloc{src.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    sloc.SubresourceIndex = 0;
    if (variant == 0) {
        const D3D12_BOX box{1, 1, 0, 5, 3, 1};
        ctx.list->CopyTextureRegion(&dloc, 2, 1, 0, &sloc, &box);
    } else {
        ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    }
    ctx.end_pred();
    CHECK(readback_texture_packed(dev, ctx, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, 1, out));
    return true;
}

// 4a. Color ResolveSubresource & ResolveSubresourceRegion (4x MSAA -> 1x)
bool test_resolve_color(ID3D12Device* dev, bool partial_region, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    const UINT w = partial_region ? 8u : 4u, h = partial_region ? 8u : 4u;
    Ref<ID3D12Resource> msaa, dst;
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, 4, D3D12_RESOURCE_STATE_RENDER_TARGET, msaa,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, 1, D3D12_RESOURCE_STATE_RENDER_TARGET, dst,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));

    Ref<ID3D12DescriptorHeap> rtv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 2;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p))));
    const UINT step = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE h_msaa = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE h_dst{h_msaa.ptr + step};
    dev->CreateRenderTargetView(msaa.p, nullptr, h_msaa);
    dev->CreateRenderTargetView(dst.p, nullptr, h_dst);

    // Exact 1/255 multiples avoid 0.5 UNORM tie-rounding differences between NVIDIA and WARP.
    const float color_msaa[4] = {64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f};
    const float color_dst[4] = {1.0f, 0.0f, 64.0f / 255.0f, 1.0f};
    ctx.list->ClearRenderTargetView(h_msaa, color_msaa, 0, nullptr);
    ctx.list->ClearRenderTargetView(h_dst, color_dst, 0, nullptr);
    transition(ctx.list.p, msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);

    ctx.begin_pred(pc);
    if (partial_region) {
        D3D12_RECT rect{2, 1, 6, 5};
        ctx.list1->ResolveSubresourceRegion(dst.p, 0, 1, 2, msaa.p, 0, &rect, DXGI_FORMAT_R8G8B8A8_UNORM,
                                            D3D12_RESOLVE_MODE_AVERAGE);
    } else {
        ctx.list->ResolveSubresource(dst.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    }
    ctx.end_pred();
    CHECK(readback_texture_packed(dev, ctx, dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, 1, out));
    return true;
}

// 4b. Depth/Stencil ResolveSubresourceRegion (4x MSAA -> 1x, partial rect & whole)
bool test_resolve_depth_stencil(ID3D12Device* dev, bool stencil_plane, D3D12_RESOLVE_MODE mode, bool partial_rect,
                                const PredCase& pc, std::vector<unsigned char>& out_plane) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    const UINT w = 8, h = 6;
    const DXGI_FORMAT res_fmt = stencil_plane ? DXGI_FORMAT_D24_UNORM_S8_UINT : DXGI_FORMAT_D32_FLOAT;
    const DXGI_FORMAT resolve_fmt = stencil_plane ? DXGI_FORMAT_X24_TYPELESS_G8_UINT : DXGI_FORMAT_R32_FLOAT;
    Ref<ID3D12Resource> msaa, dst;
    CHECK(make_tex2d(dev, res_fmt, w, h, 1, 1, 4, D3D12_RESOURCE_STATE_DEPTH_WRITE, msaa,
                     D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
    CHECK(make_tex2d(dev, res_fmt, w, h, 1, 1, 1, D3D12_RESOURCE_STATE_DEPTH_WRITE, dst,
                     D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));

    Ref<ID3D12DescriptorHeap> dsv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    hd.NumDescriptors = 2;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap.p))));
    const UINT step = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    D3D12_CPU_DESCRIPTOR_HANDLE h_msaa = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE h_dst{h_msaa.ptr + step};
    dev->CreateDepthStencilView(msaa.p, nullptr, h_msaa);
    dev->CreateDepthStencilView(dst.p, nullptr, h_dst);

    const D3D12_CLEAR_FLAGS cflags =
        stencil_plane ? (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) : D3D12_CLEAR_FLAG_DEPTH;
    ctx.list->ClearDepthStencilView(h_msaa, cflags, 0.25f, 0x3A, 0, nullptr);
    const D3D12_RECT patch{3, 2, 7, 5};
    ctx.list->ClearDepthStencilView(h_msaa, cflags, 0.5f, 0x5C, 1, &patch);
    ctx.list->ClearDepthStencilView(h_dst, cflags, 0.875f, 0x91, 0, nullptr);

    transition(ctx.list.p, msaa.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_RESOLVE_DEST);

    const UINT sub = stencil_plane ? 1u : 0u;
    ctx.begin_pred(pc);
    if (partial_rect) {
        D3D12_RECT rect{2, 1, 6, 5};
        ctx.list1->ResolveSubresourceRegion(dst.p, sub, 2, 1, msaa.p, sub, &rect, resolve_fmt, mode);
    } else {
        ctx.list1->ResolveSubresourceRegion(dst.p, sub, 0, 0, msaa.p, sub, nullptr, resolve_fmt, mode);
    }
    ctx.end_pred();

    // Read back only the resolved plane so WARP's known D24S8 other-plane quirk doesn't skew comparison.
    const D3D12_RESOURCE_DESC desc = dst->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};
    UINT nr[2]{};
    UINT64 rs[2]{};
    UINT64 total = 0;
    const UINT planes = stencil_plane ? 2u : 1u;
    dev->GetCopyableFootprints(&desc, 0, planes, 0, fp, nr, rs, &total);
    Ref<ID3D12Resource> rb;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION sloc{dst.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    sloc.SubresourceIndex = sub;
    D3D12_TEXTURE_COPY_LOCATION dloc{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dloc.PlacedFootprint = fp[sub];
    ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    CHECK(ctx.submit_and_wait());
    std::vector<unsigned char> raw;
    CHECK(read_buf(rb.p, SIZE_T(total), raw));
    out_plane.clear();
    for (UINT y = 0; y < nr[sub]; ++y) {
        const SIZE_T off = SIZE_T(fp[sub].Offset + y * fp[sub].Footprint.RowPitch);
        out_plane.insert(out_plane.end(), raw.begin() + off, raw.begin() + off + SIZE_T(rs[sub]));
    }
    return true;
}

// 5a. ClearUnorderedAccessViewUint / Float on buffers:
// variant 0: raw R32_TYPELESS whole view (Uint)
// variant 1: typed R8_UINT sub-rect of 3 bytes at offset 1 (Uint, unaligned <4B range)
// variant 2: typed R16G16_FLOAT whole view (Float)
// variant 3: typed R32G32B32A32_UINT sub-rect (Uint, 16-byte elements)
bool test_uav_clear_buffer(ID3D12Device* dev, int variant, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    constexpr UINT64 buf_bytes = 256;
    Ref<ID3D12Resource> seed_up, buf, rb;
    CHECK(make_buffer(dev, buf_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    CHECK(make_buffer(dev, buf_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, buf,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    CHECK(make_buffer(dev, buf_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    const auto seed = fill_pattern(buf_bytes, 19, 0x55);
    CHECK(write_buf(seed_up.p, seed.data(), buf_bytes));

    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    D3D12_DESCRIPTOR_HEAP_DESC chd{};
    chd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    chd.NumDescriptors = 1;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&chd, IID_PPV_ARGS(&cpu_heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC ghd = chd;
    ghd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&ghd, IID_PPV_ARGS(&gpu_heap.p))));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    if (variant == 0) {
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.Buffer.FirstElement = 4;
        uav.Buffer.NumElements = 16;
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else if (variant == 1 || variant == 4) {
        uav.Format = DXGI_FORMAT_R8_UINT;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = 64;
    } else if (variant == 2) {
        uav.Format = DXGI_FORMAT_R16G16_FLOAT;
        uav.Buffer.FirstElement = 4;
        uav.Buffer.NumElements = 12;
    } else {
        uav.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        uav.Buffer.FirstElement = 1;
        uav.Buffer.NumElements = 8;
    }
    dev->CreateUnorderedAccessView(buf.p, nullptr, &uav, cpu_heap->GetCPUDescriptorHandleForHeapStart());
    dev->CreateUnorderedAccessView(buf.p, nullptr, &uav, gpu_heap->GetCPUDescriptorHandleForHeapStart());

    ctx.list->CopyResource(buf.p, seed_up.p);
    transition(ctx.list.p, buf.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    ctx.list->SetDescriptorHeaps(1, heaps);

    const auto gpu_h = gpu_heap->GetGPUDescriptorHandleForHeapStart();
    const auto cpu_h = cpu_heap->GetCPUDescriptorHandleForHeapStart();
    ctx.begin_pred(pc);
    if (variant == 0) {
        const UINT vals[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};
        ctx.list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, buf.p, vals, 0, nullptr);
    } else if (variant == 1) {
        const UINT vals[4] = {0x42u, 0, 0, 0};
        // Unaligned start & end (9 bytes at offset 1) works both predicated and unpredicated;
        // predicated-only <4-byte sub-rect is also exercised when pc.enabled is true (variant 4).
        const D3D12_RECT rect{1, 0, 10, 1};
        ctx.list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, buf.p, vals, 1, &rect);
    } else if (variant == 4) {
        const UINT vals[4] = {0x42u, 0, 0, 0};
        const D3D12_RECT rect{1, 0, 4, 1};  // 3 bytes at offset 1: byte-granular predicated compute path
        ctx.list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, buf.p, vals, 1, &rect);
    } else if (variant == 2) {
        const FLOAT vals[4] = {1.5f, -2.0f, 0.0f, 0.0f};
        ctx.list->ClearUnorderedAccessViewFloat(gpu_h, cpu_h, buf.p, vals, 0, nullptr);
    } else {
        const UINT vals[4] = {0x01020304u, 0x05060708u, 0x090A0B0Cu, 0x0D0E0F10u};
        const D3D12_RECT rects[2] = {{1, 0, 3, 1}, {5, 0, 7, 1}};
        ctx.list->ClearUnorderedAccessViewUint(gpu_h, cpu_h, buf.p, vals, 2, rects);
    }
    ctx.end_pred();
    transition(ctx.list.p, buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ctx.list->CopyResource(rb.p, buf.p);
    CHECK(ctx.submit_and_wait());
    CHECK(read_buf(rb.p, buf_bytes, out));
    return true;
}

// 5b. ClearUnorderedAccessViewUint / Float on 3D volume texture (whole mip) & 2D texture
bool test_uav_clear_texture(ID3D12Device* dev, bool is_vulkan, bool is_3d, bool is_float, const PredCase& pc,
                            std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    Ref<ID3D12Resource> tex;
    if (is_3d) {
        CHECK(make_tex3d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 4, D3D12_RESOURCE_STATE_COPY_DEST, tex,
                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    } else {
        CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, tex,
                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    }
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT nr = 0;
    UINT64 rs = 0, total = 0;
    dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &nr, &rs, &total);
    Ref<ID3D12Resource> seed_up;
    CHECK(make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    const auto seed = fill_pattern(SIZE_T(total), 17, 0x33);
    CHECK(write_buf(seed_up.p, seed.data(), SIZE_T(total)));
    D3D12_TEXTURE_COPY_LOCATION dloc{tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    dloc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION sloc{seed_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    sloc.PlacedFootprint = fp;
    ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
    transition(ctx.list.p, tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    D3D12_DESCRIPTOR_HEAP_DESC chd{};
    chd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    chd.NumDescriptors = 1;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&chd, IID_PPV_ARGS(&cpu_heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC ghd = chd;
    ghd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&ghd, IID_PPV_ARGS(&gpu_heap.p))));

    if (is_3d && is_vulkan) {
        // CreateUnorderedAccessView does not yet implement TEXTURE3D; write the GFX10 T# directly as d3d12_uav_clear_test does.
        namespace g = leonida::gfx10;
        g::Image img{};
        img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
        img.swizzle = g::format_swizzle(VK_FORMAT_R8G8B8A8_UNORM);
        img.type = g::ImageType::image_3d;
        img.width = 4;
        img.height = 4;
        img.depth = 4;
        img.first_mip = 0;
        img.mip_count = 1;
        img.resource_mips = 1;
        const auto words = g::encode(img);
        std::memcpy(reinterpret_cast<void*>(cpu_heap->GetCPUDescriptorHandleForHeapStart().ptr),
                    words.data(), sizeof(words));
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        if (is_3d) {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uav.Texture3D.MipSlice = 0;
            uav.Texture3D.FirstWSlice = 0;
            uav.Texture3D.WSize = 4;
        } else {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Texture2D.MipSlice = 0;
        }
        dev->CreateUnorderedAccessView(tex.p, nullptr, &uav, cpu_heap->GetCPUDescriptorHandleForHeapStart());
        dev->CreateUnorderedAccessView(tex.p, nullptr, &uav, gpu_heap->GetCPUDescriptorHandleForHeapStart());
    }
    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    ctx.list->SetDescriptorHeaps(1, heaps);

    ctx.begin_pred(pc);
    if (is_float) {
        const FLOAT vals[4] = {1.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f};
        ctx.list->ClearUnorderedAccessViewFloat(gpu_heap->GetGPUDescriptorHandleForHeapStart(),
                                                cpu_heap->GetCPUDescriptorHandleForHeapStart(),
                                                tex.p, vals, 0, nullptr);
    } else {
        const UINT vals[4] = {0x11, 0x22, 0x33, 0x44};
        ctx.list->ClearUnorderedAccessViewUint(gpu_heap->GetGPUDescriptorHandleForHeapStart(),
                                               cpu_heap->GetCPUDescriptorHandleForHeapStart(),
                                               tex.p, vals, 0, nullptr);
    }
    ctx.end_pred();
    CHECK(readback_texture_packed(dev, ctx, tex.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 1, out));
    return true;
}

// 6. AtomicCopyBufferUINT & AtomicCopyBufferUINT64
bool test_atomic_copy(ID3D12Device* dev, bool is64, const PredCase& pc, std::vector<unsigned char>& out) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    constexpr UINT64 sz = 64;
    Ref<ID3D12Resource> src_up, seed_up, dep, dst, rb;
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, src_up));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed_up));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, dep));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, dst));
    CHECK(make_buffer(dev, sz, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rb));
    const auto pat_src = fill_pattern(sz, 41, 0x10);
    const auto pat_dst = fill_pattern(sz, 17, 0x90);
    CHECK(write_buf(src_up.p, pat_src.data(), sz) && write_buf(seed_up.p, pat_dst.data(), sz));

    ctx.list->CopyResource(dep.p, src_up.p);
    ctx.list->CopyResource(dst.p, seed_up.p);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

    ID3D12Resource* deps[] = {dep.p};
    const D3D12_SUBRESOURCE_RANGE_UINT64 ranges[] = {{0, {0, sz}}};
    ctx.begin_pred(pc);
    if (is64) ctx.list1->AtomicCopyBufferUINT64(dst.p, 16, src_up.p, 8, 1, deps, ranges);
    else ctx.list1->AtomicCopyBufferUINT(dst.p, 12, src_up.p, 4, 1, deps, ranges);
    ctx.end_pred();

    transition(ctx.list.p, dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ctx.list->CopyResource(rb.p, dst.p);
    CHECK(ctx.submit_and_wait());
    CHECK(read_buf(rb.p, sz, out));
    auto expected = pat_dst;
    if (pc.executes()) {
        if (is64) std::memcpy(expected.data() + 16, pat_src.data() + 8, 8);
        else std::memcpy(expected.data() + 12, pat_src.data() + 4, 4);
    }
    CHECK(out == expected);
    return true;
}

// 7. PIPELINE_STATISTICS isolation (D3D-162): all predicated helpers enclosed in one statistics query yield all zeros.
bool test_statistics_isolation(ID3D12Device* dev, const PredCase& pc) {
    CmdCtx ctx;
    CHECK(ctx.init(dev, pc.value));
    Ref<ID3D12QueryHeap> qh;
    const D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 1, 0};
    CHECK(SUCCEEDED(dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&qh.p))));
    Ref<ID3D12Resource> counters;
    CHECK(make_buffer(dev, 88, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, counters));

    Ref<ID3D12Resource> b_src, b_dst, t_src, t_dst, t_up, t_rb, rgb_tex, rgb_up, rgb_rb, msaa, rdst, uav_buf;
    CHECK(make_buffer(dev, 64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, b_src));
    CHECK(make_buffer(dev, 64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, b_dst));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_SOURCE, t_src));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, t_dst));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R32G32B32_FLOAT, 4, 2, 1, 1, 1, D3D12_RESOURCE_STATE_COPY_DEST, rgb_tex));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 4, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, msaa,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
    CHECK(make_tex2d(dev, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, D3D12_RESOURCE_STATE_RESOLVE_DEST, rdst,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
    CHECK(make_buffer(dev, 64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav_buf,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}, rgb_fp{};
    UINT64 tot = 0, rgb_tot = 0;
    const auto td = t_dst->GetDesc();
    const auto rd = rgb_tex->GetDesc();
    dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &tot);
    dev->GetCopyableFootprints(&rd, 0, 1, 0, &rgb_fp, nullptr, nullptr, &rgb_tot);
    CHECK(make_buffer(dev, tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, t_up));
    CHECK(make_buffer(dev, tot, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, t_rb));
    CHECK(make_buffer(dev, rgb_tot, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, rgb_up));
    CHECK(make_buffer(dev, rgb_tot, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, rgb_rb));

    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    D3D12_DESCRIPTOR_HEAP_DESC chd{};
    chd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    chd.NumDescriptors = 1;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&chd, IID_PPV_ARGS(&cpu_heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC ghd = chd;
    ghd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&ghd, IID_PPV_ARGS(&gpu_heap.p))));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = 16;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(uav_buf.p, nullptr, &uav, cpu_heap->GetCPUDescriptorHandleForHeapStart());
    dev->CreateUnorderedAccessView(uav_buf.p, nullptr, &uav, gpu_heap->GetCPUDescriptorHandleForHeapStart());
    ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
    ctx.list->SetDescriptorHeaps(1, heaps);

    ctx.list->BeginQuery(qh.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
    ctx.begin_pred(pc);
    ctx.list->CopyResource(b_dst.p, b_src.p);
    ctx.list->CopyResource(t_dst.p, t_src.p);
    D3D12_TEXTURE_COPY_LOCATION dloc{t_dst.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION uploc{t_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    uploc.PlacedFootprint = fp;
    ctx.list->CopyTextureRegion(&dloc, 0, 0, 0, &uploc, nullptr);
    D3D12_TEXTURE_COPY_LOCATION rbloc{t_rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    rbloc.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION sloc{t_src.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    ctx.list->CopyTextureRegion(&rbloc, 0, 0, 0, &sloc, nullptr);

    D3D12_TEXTURE_COPY_LOCATION rgb_t{rgb_tex.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION rgb_u{rgb_up.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    rgb_u.PlacedFootprint = rgb_fp;
    ctx.list->CopyTextureRegion(&rgb_t, 0, 0, 0, &rgb_u, nullptr);
    transition(ctx.list.p, rgb_tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION rgb_r{rgb_rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    rgb_r.PlacedFootprint = rgb_fp;
    ctx.list->CopyTextureRegion(&rgb_r, 0, 0, 0, &rgb_t, nullptr);

    ctx.list->ResolveSubresource(rdst.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    const UINT vals[4] = {1, 2, 3, 4};
    ctx.list->ClearUnorderedAccessViewUint(gpu_heap->GetGPUDescriptorHandleForHeapStart(),
                                           cpu_heap->GetCPUDescriptorHandleForHeapStart(),
                                           uav_buf.p, vals, 0, nullptr);
    ID3D12Resource* deps[] = {b_src.p};
    const D3D12_SUBRESOURCE_RANGE_UINT64 ranges[] = {{0, {0, 64}}};
    ctx.list1->AtomicCopyBufferUINT(b_dst.p, 0, b_src.p, 0, 1, deps, ranges);
    ctx.end_pred();
    ctx.list->EndQuery(qh.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
    ctx.list->ResolveQueryData(qh.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, counters.p, 0);
    CHECK(ctx.submit_and_wait());

    std::vector<unsigned char> stats;
    CHECK(read_buf(counters.p, 88, stats));
    for (unsigned w = 0; w < 11; ++w) {
        UINT64 val = UINT64_MAX;
        std::memcpy(&val, stats.data() + w * 8, 8);
        if (val != 0) {
            std::fprintf(stderr, "statistics word %u: %llu, expected 0 (pred executes=%d)\n",
                         w, val, int(pc.executes()));
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool warp_only = false, verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warp-only") == 0) warp_only = true;
        else if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        else return 2;
    }
    if (hostwine::host_is_wine()) return 77;

    vktest::Device vk;
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp_adapter;
    Ref<ID3D12Device> warp, ours;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter.p))) ||
        FAILED(D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p)))) {
        std::fprintf(stderr, "failed to create WARP D3D12 device\n");
        return 1;
    }

    VkPhysicalDeviceDepthStencilResolveProperties ds_resolve{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    if (!warp_only) {
        vk.request_buffer_device_address = true;
        const auto status = vk.initialize();
        if (status == vktest::DeviceStatus::unavailable) {
            std::puts("d3d12_predicated_ops: skipped (no Vulkan device)");
            return 77;
        }
        if (status != vktest::DeviceStatus::ready) return 1;
        if (!vk.conditional_rendering || !vk.storage_buffer_8bit) {
            std::puts("d3d12_predicated_ops: skipped (requires conditional rendering and storageBuffer8BitAccess)");
            return 0;
        }
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &ds_resolve};
        vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props2);

        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.conditional_rendering = vk.conditional_rendering;
        info.storage_buffer_8bit = vk.storage_buffer_8bit;
        info.buffer_device_address = vk.buffer_device_address;
        info.occlusion_query_precise = vk.occlusion_query_precise;
        info.pipeline_statistics_query = vk.pipeline_statistics_query;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    }

    unsigned cases = 0;
    auto compare = [&](const char* label, auto&& fn) -> bool {
        if (verbose) {
            std::fprintf(stderr, "-> %s\n", label);
            std::fflush(stderr);
        }
        std::vector<unsigned char> expected, actual;
        if (!fn(warp.p, expected)) {
            std::fprintf(stderr, "WARP failed: %s\n", label);
            return false;
        }
        if (!warp_only) {
            if (!fn(ours.p, actual)) {
                std::fprintf(stderr, "Vulkan failed: %s\n", label);
                return false;
            }
            if (actual != expected) {
                std::fprintf(stderr, "mismatch (%zu vs %zu bytes): %s\n", actual.size(), expected.size(), label);
                const SIZE_T n = std::min(actual.size(), expected.size());
                for (SIZE_T i = 0; i < n; ++i) {
                    if (actual[i] != expected[i]) {
                        std::fprintf(stderr, "  first diff at byte %zu: got 0x%02x expected 0x%02x\n",
                                     i, unsigned(actual[i]), unsigned(expected[i]));
                        break;
                    }
                }
                return false;
            }
        }
        ++cases;
        return true;
    };

    // Run full predicate matrix on representative cases, and skip+run+highword+off on all variants.
    for (const auto& pc : kCorePreds) {
        if (!compare("CopyResource buffer", [&](ID3D12Device* d, auto& o) { return test_copy_resource_buffer(d, pc, o); }))
            return 1;
        if (!compare("CopyResource tex2d RGBA8 multi-mip+array",
                     [&](ID3D12Device* d, auto& o) { return test_copy_resource_texture(d, DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 2, 2, 1, pc, o); }))
            return 1;
        if (!compare("CopyResource tex2d BC1",
                     [&](ID3D12Device* d, auto& o) { return test_copy_resource_texture(d, DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 1, 1, pc, o); }))
            return 1;
        if (!compare("CopyResource tex2d D32_FLOAT",
                     [&](ID3D12Device* d, auto& o) { return test_copy_resource_texture(d, DXGI_FORMAT_D32_FLOAT, 4, 4, 1, 1, 1, pc, o); }))
            return 1;
        if (!compare("CopyResource tex2d D24_UNORM_S8_UINT",
                     [&](ID3D12Device* d, auto& o) { return test_copy_resource_texture(d, DXGI_FORMAT_D24_UNORM_S8_UINT, 4, 4, 1, 1, 2, pc, o); }))
            return 1;

        for (int v = 0; v < 5; ++v) {
            char label[64];
            std::snprintf(label, sizeof(label), "CopyTextureRegion upload v%d pred=%d", v, int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_copy_texture_upload(d, v, pc, o); }))
                return 1;
        }
        for (bool rgb96 : {false, true}) {
            char label[64];
            std::snprintf(label, sizeof(label), "CopyTextureRegion readback rgb96=%d pred=%d", int(rgb96), int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_copy_texture_readback(d, rgb96, pc, o); }))
                return 1;
        }
        for (int v = 0; v < 5; ++v) {
            char label[64];
            std::snprintf(label, sizeof(label), "CopyTextureRegion tex2tex v%d pred=%d", v, int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_copy_texture_tex2tex(d, v, pc, o); }))
                return 1;
        }
        for (bool partial : {false, true}) {
            char label[64];
            std::snprintf(label, sizeof(label), "ResolveSubresource color partial=%d pred=%d", int(partial), int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_resolve_color(d, partial, pc, o); }))
                return 1;
        }
        for (bool stencil : {false, true}) {
            const VkResolveModeFlags supported =
                stencil ? ds_resolve.supportedStencilResolveModes : ds_resolve.supportedDepthResolveModes;
            for (const auto mode : {D3D12_RESOLVE_MODE_MIN, D3D12_RESOLVE_MODE_MAX, D3D12_RESOLVE_MODE_AVERAGE}) {
                if (stencil && mode == D3D12_RESOLVE_MODE_AVERAGE) continue;
                const VkResolveModeFlags bit = mode == D3D12_RESOLVE_MODE_MIN ? VK_RESOLVE_MODE_MIN_BIT
                                             : mode == D3D12_RESOLVE_MODE_MAX ? VK_RESOLVE_MODE_MAX_BIT
                                                                              : VK_RESOLVE_MODE_AVERAGE_BIT;
                if (!warp_only && !(supported & bit)) continue;
                for (bool partial : {false, true}) {
                    char label[80];
                    std::snprintf(label, sizeof(label), "ResolveSubresourceRegion ds stencil=%d mode=%u partial=%d pred=%d",
                                  int(stencil), unsigned(mode), int(partial), int(pc.executes()));
                    if (!compare(label, [&](ID3D12Device* d, auto& o) {
                            return test_resolve_depth_stencil(d, stencil, mode, partial, pc, o);
                        }))
                        return 1;
                }
            }
        }
        for (int v = 0; v < (pc.enabled ? 5 : 4); ++v) {
            char label[64];
            std::snprintf(label, sizeof(label), "ClearUAV buffer v%d pred=%d", v, int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_uav_clear_buffer(d, v, pc, o); }))
                return 1;
        }
        for (bool is3d : {false, true}) {
            for (bool is_float : {false, true}) {
                char label[64];
                std::snprintf(label, sizeof(label), "ClearUAV texture 3d=%d float=%d pred=%d",
                              int(is3d), int(is_float), int(pc.executes()));
                if (!compare(label, [&](ID3D12Device* d, auto& o) {
                        return test_uav_clear_texture(d, d == ours.p, is3d, is_float, pc, o);
                    }))
                    return 1;
            }
        }
        for (bool is64 : {false, true}) {
            char label[64];
            std::snprintf(label, sizeof(label), "AtomicCopyBuffer 64=%d pred=%d", int(is64), int(pc.executes()));
            if (!compare(label, [&](ID3D12Device* d, auto& o) { return test_atomic_copy(d, is64, pc, o); }))
                return 1;
        }
    }

    if (!warp_only && vk.pipeline_statistics_query) {
        for (const auto& pc : {kCorePreds[0], kCorePreds[1], kCorePreds[2], kCorePreds[3]}) {
            if (!test_statistics_isolation(warp.p, pc)) {
                std::fprintf(stderr, "WARP statistics isolation failed\n");
                return 1;
            }
            if (!test_statistics_isolation(ours.p, pc)) {
                std::fprintf(stderr, "Vulkan statistics isolation failed\n");
                return 1;
            }
            ++cases;
        }
    }

    ours.reset();
    if (!warp_only && vk.validation_errors()) {
        std::fprintf(stderr, "d3d12_predicated_ops: %u Vulkan validation error(s)\n", vk.validation_errors());
        return 1;
    }
    std::printf("d3d12_predicated_ops: %u exact cases (%s)\n", cases,
                warp_only ? "WARP reference" : "Vulkan + WARP differential");
    return 0;
}
