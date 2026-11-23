// CreateDepthStencilView validation and CopyTiles-in-bundle parity against Windows D3D12 (hardware adapter + WARP).
// Verifies valid typed/typeless DSVs across all 4 depth-stencil formats and view dimensions clear cleanly with zero
// Vulkan validation errors, invalid DSVs produce a null TargetView without crashing or emitting validation errors,
// and CopyTiles in a BUNDLE fails Close() on both Leonida and host D3D12/WARP. No game data.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/command_list.h"

#include <dxgi1_4.h>

#include <climits>
#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); return false; } \
} while (0)

#define REQUIRE(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { if (p) { p->Release(); p = nullptr; } return &p; }
};

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
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

D3D12_RESOURCE_DESC tex_desc(D3D12_RESOURCE_DIMENSION dim, UINT64 w, UINT h, UINT16 depth_or_array, UINT16 mips,
                             DXGI_FORMAT fmt, UINT samples, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = depth_or_array;
    d.MipLevels = mips;
    d.Format = fmt;
    d.SampleDesc.Count = samples;
    d.Flags = flags;
    return d;
}

bool create_committed(ID3D12Device* dev, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                      Ref<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                  IID_PPV_ARGS(out.put())));
}

leonida::TargetView read_view(D3D12_CPU_DESCRIPTOR_HANDLE h) {
    leonida::TargetView v{};
    std::memcpy(&v, reinterpret_cast<const void*>(h.ptr), sizeof(v));
    return v;
}

struct ExecEnv {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12DescriptorHeap> dsv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
    UINT64 fence_val = 0;

    bool init(ID3D12Device* dev) {
        device = dev;
        D3D12_COMMAND_QUEUE_DESC qd{};
        CHECK(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))));
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                               IID_PPV_ARGS(list.put()))));
        CHECK(SUCCEEDED(list->Close()));
        CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))));
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(dsv_heap.put()))));
        dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
        return true;
    }

    bool clear_dsv_and_wait(D3D12_CLEAR_FLAGS flags, bool expect_close_ok, bool check_layer_result) {
        CHECK(SUCCEEDED(alloc->Reset()));
        CHECK(SUCCEEDED(list->Reset(alloc.p, nullptr)));
        list->ClearDepthStencilView(dsv, flags, 0.5f, 0x12, 0, nullptr);
        const HRESULT hr = list->Close();
        if (!expect_close_ok) {
            CHECK(FAILED(hr));
            return true;
        }
        CHECK(SUCCEEDED(hr));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (check_layer_result)
            CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        CHECK(SUCCEEDED(queue->Signal(fence.p, ++fence_val)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_val, nullptr)));
        return true;
    }
};

// Valid DSVs: run on both Leonida (with TargetView inspection + GPU clear) and host D3D12/WARP (GPU clear).
bool test_valid_dsvs(ID3D12Device* dev, bool is_layer) {
    ExecEnv env;
    CHECK(env.init(dev));

    // 1. All 4 typed depth-stencil formats: nullptr desc, UNKNOWN format desc, explicit format desc, valid flags.
    struct TypedFmt {
        DXGI_FORMAT fmt;
        bool has_stencil;
    };
    const TypedFmt typed_fmts[] = {
        {DXGI_FORMAT_D16_UNORM, false},
        {DXGI_FORMAT_D24_UNORM_S8_UINT, true},
        {DXGI_FORMAT_D32_FLOAT, false},
        {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true},
    };
    for (const auto& tf : typed_fmts) {
        Ref<ID3D12Resource> res;
        CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 2, tf.fmt, 1,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE, res));
        const D3D12_CLEAR_FLAGS clear_flags =
            tf.has_stencil ? (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) : D3D12_CLEAR_FLAG_DEPTH;

        // nullptr desc on typed DS
        dev->CreateDepthStencilView(res.p, nullptr, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 0 && v.first_layer == 0 && v.layer_count == 1);
        }
        CHECK(env.clear_dsv_and_wait(clear_flags, true, is_layer));

        // explicit desc with DXGI_FORMAT_UNKNOWN on typed DS
        D3D12_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipSlice = 1;
        dev->CreateDepthStencilView(res.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 1 && v.first_layer == 0 && v.layer_count == 1);
        }
        CHECK(env.clear_dsv_and_wait(clear_flags, true, is_layer));

        // explicit desc with matching format
        d.Format = tf.fmt;
        d.Texture2D.MipSlice = 0;
        dev->CreateDepthStencilView(res.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 0 && v.first_layer == 0 && v.layer_count == 1);
        }
        CHECK(env.clear_dsv_and_wait(clear_flags, true, is_layer));

        // READ_ONLY_DEPTH is valid on all 4 formats; READ_ONLY_STENCIL valid when stencil exists.
        d.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
        dev->CreateDepthStencilView(res.p, &d, env.dsv);
        if (is_layer) CHECK(read_view(env.dsv).resource != 0);
        if (tf.has_stencil) {
            d.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
            dev->CreateDepthStencilView(res.p, &d, env.dsv);
            if (is_layer) CHECK(read_view(env.dsv).resource != 0);
            d.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL;
            dev->CreateDepthStencilView(res.p, &d, env.dsv);
            if (is_layer) CHECK(read_view(env.dsv).resource != 0);
        }
    }

    // 2. All 4 typeless depth-stencil families with explicit matching DSV format.
    struct TypelessPair {
        DXGI_FORMAT res_fmt;
        DXGI_FORMAT view_fmt;
        bool has_stencil;
    };
    const TypelessPair typeless_pairs[] = {
        {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM, false},
        {DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, true},
        {DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT, false},
        {DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true},
    };
    for (const auto& tp : typeless_pairs) {
        Ref<ID3D12Resource> res;
        CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, tp.res_fmt, 1,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE, res));
        D3D12_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = tp.view_fmt;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        dev->CreateDepthStencilView(res.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 0 && v.first_layer == 0 && v.layer_count == 1);
        }
        const D3D12_CLEAR_FLAGS clear_flags =
            tp.has_stencil ? (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL) : D3D12_CLEAR_FLAG_DEPTH;
        CHECK(env.clear_dsv_and_wait(clear_flags, true, is_layer));
    }

    // 3. 2D array, 2DMS, 2DMS array (including ArraySize == UINT_MAX).
    {
        Ref<ID3D12Resource> tex2d_arr;
        CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 4, 3, DXGI_FORMAT_D32_FLOAT, 1,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE, tex2d_arr));
        D3D12_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_D32_FLOAT;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MipSlice = 1;
        d.Texture2DArray.FirstArraySlice = 1;
        d.Texture2DArray.ArraySize = 2;
        dev->CreateDepthStencilView(tex2d_arr.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 1 && v.first_layer == 1 && v.layer_count == 2);
        }
        CHECK(env.clear_dsv_and_wait(D3D12_CLEAR_FLAG_DEPTH, true, is_layer));

        d.Texture2DArray.FirstArraySlice = 2;
        d.Texture2DArray.ArraySize = UINT_MAX;
        dev->CreateDepthStencilView(tex2d_arr.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 1 && v.first_layer == 2 && v.layer_count == 2);
        }
        CHECK(env.clear_dsv_and_wait(D3D12_CLEAR_FLAG_DEPTH, true, is_layer));
    }
    {
        Ref<ID3D12Resource> tex2d_ms;
        CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 4, 1, DXGI_FORMAT_D32_FLOAT, 4,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE, tex2d_ms));
        D3D12_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_D32_FLOAT;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        dev->CreateDepthStencilView(tex2d_ms.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 0 && v.first_layer == 0 && v.layer_count == 1);
        }
        CHECK(env.clear_dsv_and_wait(D3D12_CLEAR_FLAG_DEPTH, true, is_layer));

        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray.FirstArraySlice = 1;
        d.Texture2DMSArray.ArraySize = UINT_MAX;
        dev->CreateDepthStencilView(tex2d_ms.p, &d, env.dsv);
        if (is_layer) {
            const auto v = read_view(env.dsv);
            CHECK(v.resource != 0 && v.mip == 0 && v.first_layer == 1 && v.layer_count == 3);
        }
        CHECK(env.clear_dsv_and_wait(D3D12_CLEAR_FLAG_DEPTH, true, is_layer));
    }
    return true;
}

// Invalid DSVs on Leonida: must write a null TargetView, refuse ClearDepthStencilView at Close(), and emit 0 VUIDs.
bool test_invalid_dsvs_layer(ID3D12Device* dev) {
    ExecEnv env;
    CHECK(env.init(dev));

    Ref<ID3D12Resource> buf, tex3d, color_no_ds, typeless_no_ds, tex1d_no_ds, ds_2d, ds_2d_ms, ds_d24, ds_typeless_r32,
        ds_typeless_r24g8;
    CHECK(create_committed(dev, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON, buf));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 1,
                                         D3D12_RESOURCE_FLAG_NONE),
                           D3D12_RESOURCE_STATE_COMMON, tex3d));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 1,
                                         D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                           D3D12_RESOURCE_STATE_COMMON, color_no_ds));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R32_TYPELESS, 1,
                                         D3D12_RESOURCE_FLAG_NONE),
                           D3D12_RESOURCE_STATE_COMMON, typeless_no_ds));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 16, 1, 4, 2, DXGI_FORMAT_R32_TYPELESS, 1,
                                         D3D12_RESOURCE_FLAG_NONE),
                           D3D12_RESOURCE_STATE_COMMON, tex1d_no_ds));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 4, 2, DXGI_FORMAT_D32_FLOAT, 1,
                                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, ds_2d));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 4, 1, DXGI_FORMAT_D32_FLOAT, 4,
                                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, ds_2d_ms));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_D24_UNORM_S8_UINT,
                                         1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, ds_d24));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R32_TYPELESS, 1,
                                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, ds_typeless_r32));
    CHECK(create_committed(dev, tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R24G8_TYPELESS, 1,
                                         D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, ds_typeless_r24g8));

    auto expect_null = [&](const char* label, ID3D12Resource* res, const D3D12_DEPTH_STENCIL_VIEW_DESC* desc) -> bool {
        // Poison slot first so we verify CreateDepthStencilView explicitly zeroes it.
        const leonida::TargetView poison{0xdeadbeefULL, 7, 3, 2};
        std::memcpy(reinterpret_cast<void*>(env.dsv.ptr), &poison, sizeof(poison));
        dev->CreateDepthStencilView(res, desc, env.dsv);
        const auto v = read_view(env.dsv);
        if (v.resource != 0) {
            std::fprintf(stderr, "FAIL invalid DSV accepted: %s (resource=%llu)\n", label,
                         static_cast<unsigned long long>(v.resource));
            return false;
        }
        return env.clear_dsv_and_wait(D3D12_CLEAR_FLAG_DEPTH, false, true);
    };

    D3D12_DEPTH_STENCIL_VIEW_DESC d2d{};
    d2d.Format = DXGI_FORMAT_D32_FLOAT;
    d2d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;

    // Null resource
    CHECK(expect_null("null resource", nullptr, &d2d));

    // Missing ALLOW_DEPTH_STENCIL, BUFFER, TEXTURE3D
    CHECK(expect_null("buffer", buf.p, &d2d));
    CHECK(expect_null("texture3d", tex3d.p, &d2d));
    CHECK(expect_null("color without ALLOW_DEPTH_STENCIL", color_no_ds.p, nullptr));
    CHECK(expect_null("typeless without ALLOW_DEPTH_STENCIL", typeless_no_ds.p, &d2d));

    // ViewDimension / MSAA mismatches
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC d = d2d;
        d.ViewDimension = D3D12_DSV_DIMENSION_UNKNOWN;
        CHECK(expect_null("DSV_DIMENSION_UNKNOWN in non-null desc", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION(99);
        CHECK(expect_null("unknown DSV dimension enum", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        CHECK(expect_null("TEXTURE2DMS on single-sample 2D", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, 1};
        CHECK(expect_null("TEXTURE2DMSARRAY on single-sample 2D", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d.Texture2D = {0};
        CHECK(expect_null("TEXTURE2D on 4x MSAA 2D", ds_2d_ms.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {0, 0, 1};
        CHECK(expect_null("TEXTURE2DARRAY on 4x MSAA 2D", ds_2d_ms.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
        d.Texture1D = {0};
        CHECK(expect_null("TEXTURE1D on 2D", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
        d.Texture1DArray = {0, 0, 1};
        CHECK(expect_null("TEXTURE1DARRAY on 2D", ds_2d.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d.Texture2D = {0};
        CHECK(expect_null("TEXTURE2D on 1D", tex1d_no_ds.p, &d));
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
        d.Texture1D = {0};
        CHECK(expect_null("TEXTURE1D without ALLOW_DEPTH_STENCIL", tex1d_no_ds.p, &d));
    }

    // Format errors
    {
        CHECK(expect_null("nullptr desc on typeless R32", ds_typeless_r32.p, nullptr));
        CHECK(expect_null("nullptr desc on typeless R24G8", ds_typeless_r24g8.p, nullptr));

        D3D12_DEPTH_STENCIL_VIEW_DESC d = d2d;
        d.Format = DXGI_FORMAT_UNKNOWN;
        CHECK(expect_null("UNKNOWN format on typeless R32", ds_typeless_r32.p, &d));
        CHECK(expect_null("UNKNOWN format on typeless R24G8", ds_typeless_r24g8.p, &d));

        d.Format = DXGI_FORMAT_R32_FLOAT;
        CHECK(expect_null("color R32_FLOAT on typeless R32", ds_typeless_r32.p, &d));
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        CHECK(expect_null("color RGBA8 on D32_FLOAT", ds_2d.p, &d));
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        CHECK(expect_null("typeless view format R32_TYPELESS", ds_typeless_r32.p, &d));
        d.Format = DXGI_FORMAT_R24G8_TYPELESS;
        CHECK(expect_null("typeless view format R24G8_TYPELESS", ds_typeless_r24g8.p, &d));
        d.Format = DXGI_FORMAT_D16_UNORM;
        CHECK(expect_null("D16_UNORM on D32_FLOAT", ds_2d.p, &d));
        CHECK(expect_null("D16_UNORM on R32_TYPELESS", ds_typeless_r32.p, &d));
        d.Format = DXGI_FORMAT_D32_FLOAT;
        CHECK(expect_null("D32_FLOAT on D24_UNORM_S8_UINT", ds_d24.p, &d));
        CHECK(expect_null("D32_FLOAT on R24G8_TYPELESS", ds_typeless_r24g8.p, &d));
    }

    // Out-of-range mip / array slices
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC d = d2d;
        d.Texture2D.MipSlice = 2;  // ds_2d has 2 mips (0..1)
        CHECK(expect_null("2D MipSlice == MipLevels", ds_2d.p, &d));

        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {2, 0, 1};
        CHECK(expect_null("2DArray MipSlice out of range", ds_2d.p, &d));
        d.Texture2DArray = {0, 4, 1};
        CHECK(expect_null("2DArray FirstArraySlice == DepthOrArraySize", ds_2d.p, &d));
        d.Texture2DArray = {0, 0, 0};
        CHECK(expect_null("2DArray ArraySize == 0", ds_2d.p, &d));
        d.Texture2DArray = {0, 3, 2};
        CHECK(expect_null("2DArray slice range overflow", ds_2d.p, &d));
        d.Texture2DArray = {0, 2, UINT_MAX - 1};
        CHECK(expect_null("2DArray UINT_MAX-1 overflow", ds_2d.p, &d));

        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {4, 1};
        CHECK(expect_null("2DMSArray FirstArraySlice out of range", ds_2d_ms.p, &d));
        d.Texture2DMSArray = {0, 0};
        CHECK(expect_null("2DMSArray ArraySize == 0", ds_2d_ms.p, &d));
        d.Texture2DMSArray = {3, 2};
        CHECK(expect_null("2DMSArray slice range overflow", ds_2d_ms.p, &d));
    }

    // Flags validation
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC d = d2d;
        d.Flags = D3D12_DSV_FLAGS(0x4);
        CHECK(expect_null("unknown DSV flag bit", ds_2d.p, &d));
        d.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
        CHECK(expect_null("READ_ONLY_STENCIL on D32_FLOAT", ds_2d.p, &d));
        CHECK(expect_null("READ_ONLY_STENCIL on R32_TYPELESS as D32_FLOAT", ds_typeless_r32.p, &d));
    }
    return true;
}

// CopyTiles in a BUNDLE must fail Close() while succeeding on a DIRECT control list.
bool test_copy_tiles_bundle(ID3D12Device* dev, const char* label) {
    Ref<ID3D12Resource> reserved, linear;
    const auto td = buffer_desc(65536);
    CHECK(SUCCEEDED(dev->CreateReservedResource(&td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(reserved.put()))));
    CHECK(create_committed(dev, td, D3D12_RESOURCE_STATE_COPY_SOURCE, linear));

    const D3D12_TILED_RESOURCE_COORDINATE start{0, 0, 0, 0};
    const D3D12_TILE_REGION_SIZE size{1, FALSE, 0, 0, 0};

    Ref<ID3D12CommandAllocator> direct_alloc, bundle_alloc;
    Ref<ID3D12GraphicsCommandList> direct, bundle;
    CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(direct_alloc.put()))));
    CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_alloc.p, nullptr,
                                           IID_PPV_ARGS(direct.put()))));
    direct->CopyTiles(reserved.p, &start, &size, linear.p, 0,
                      D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
    const HRESULT direct_hr = direct->Close();
    if (FAILED(direct_hr)) {
        std::fprintf(stderr, "FAIL CopyTiles direct control on %s: 0x%08lx\n", label,
                     static_cast<unsigned long>(direct_hr));
        return false;
    }

    CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(bundle_alloc.put()))));
    CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_alloc.p, nullptr,
                                           IID_PPV_ARGS(bundle.put()))));
    bundle->CopyTiles(reserved.p, &start, &size, linear.p, 0,
                      D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
    const HRESULT bundle_hr = bundle->Close();
    if (SUCCEEDED(bundle_hr)) {
        std::fprintf(stderr, "FAIL CopyTiles in bundle accepted on %s (expected failure)\n", label);
        return false;
    }
    std::printf("[%s] CopyTiles: direct Close=0x%08lx, bundle Close=0x%08lx\n", label,
                static_cast<unsigned long>(direct_hr), static_cast<unsigned long>(bundle_hr));
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
    REQUIRE(status == vktest::DeviceStatus::ready);
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
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(ours.put()))));
        REQUIRE(test_valid_dsvs(ours.p, true));
        REQUIRE(test_invalid_dsvs_layer(ours.p));
        REQUIRE(test_copy_tiles_bundle(ours.p, "layer"));
    }

    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put()))));
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string name;
            if (!hostadapter::reference(factory.p, vk, warp, adapter.put(), name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(host.put())))) continue;
            REQUIRE(test_valid_dsvs(host.p, false));
            REQUIRE(test_copy_tiles_bundle(host.p, name.c_str()));
        }
    }

    const unsigned validation_errors = vk.validation_errors() - errors_before;
    if (validation_errors) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors\n", validation_errors);
        return 1;
    }
    std::printf("d3d12_dsv_validation_test: PASS (0 Vulkan validation errors)\n");
    return 0;
}
