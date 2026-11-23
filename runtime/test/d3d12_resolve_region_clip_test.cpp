// ResolveSubresourceRegion whose region runs past the destination's edge, against the host (PROBE_WARP=1: WARP).
// NVIDIA and WARP agree: Close fails with E_INVALIDARG when the region runs past the destination resource's top level
// (its Width and Height). The layer closed such a list with S_OK and then refused the resolve at replay, which dropped
// every command of the list, the clears before it too, without a word to the caller. (A region inside the top level
// but past a smaller mip closes on both, and NVIDIA and WARP then write different texels, so it is not checked.)
// Synthetic data only.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};

struct Case {
    const char* name;
    UINT dst_x, dst_y;
    bool has_rect;
    D3D12_RECT rect;
    UINT mip = 0;  // the destination is (4 << mip) square with mip + 1 levels; the resolve goes into mip `mip`
};

ID3D12Resource* texture(ID3D12Device* d, UINT w, UINT h, UINT samples, D3D12_RESOURCE_STATES state, UINT16 mips = 1) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = mips;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = samples;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ID3D12Resource* r = nullptr;
    d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r));
    return r;
}

void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER br{};
    br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br.Transition.pResource = r;
    br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    br.Transition.StateBefore = a;
    br.Transition.StateAfter = b;
    l->ResourceBarrier(1, &br);
}

// Clears a 4x4 4x MSAA source (a red 1x1 at (1,1) on blue-grey) and a 4x4 destination, resolves the region and
// reads the destination back. Returns the close result and the destination texels as hex.
std::string run(ID3D12Device* d, const Case& c) {
    Ref<ID3D12CommandQueue> q;
    Ref<ID3D12CommandAllocator> a;
    Ref<ID3D12GraphicsCommandList1> l;
    Ref<ID3D12Fence> f;
    Ref<ID3D12DescriptorHeap> heap;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q.p))) ||
        FAILED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a.p))) ||
        FAILED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.p, nullptr, IID_PPV_ARGS(&l.p))) ||
        FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.p))) ||
        FAILED(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap.p))))
        return "setup failed";
    Ref<ID3D12Resource> ms, dst, rb;
    ms.p = texture(d, 4, 4, 4, D3D12_RESOURCE_STATE_RENDER_TARGET);
    dst.p = texture(d, 4u << c.mip, 4u << c.mip, 1, D3D12_RESOURCE_STATE_RENDER_TARGET, UINT16(c.mip + 1));
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = 4 * 256;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                               IID_PPV_ARGS(&rb.p));
    if (!ms.p || !dst.p || !rb.p) return "resources failed";
    const UINT inc = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE h0 = heap.p->GetCPUDescriptorHandleForHeapStart(), h1 = h0;
    h1.ptr += inc;
    d->CreateRenderTargetView(ms.p, nullptr, h0);
    D3D12_RENDER_TARGET_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    view.Texture2D.MipSlice = c.mip;
    d->CreateRenderTargetView(dst.p, &view, h1);
    // Every value is exact in UNORM8, so the host and WARP agree byte for byte.
    const float back[4] = {0.2f, 0.4f, 0.6f, 0.8f}, red[4] = {1, 0, 0, 1}, dst_color[4] = {0.4f, 0.4f, 0.4f, 0.4f};
    const D3D12_RECT one{1, 1, 2, 2};
    l.p->ClearRenderTargetView(h0, back, 0, nullptr);
    l.p->ClearRenderTargetView(h0, red, 1, &one);
    l.p->ClearRenderTargetView(h1, dst_color, 0, nullptr);
    barrier(l.p, ms.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    barrier(l.p, dst.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    D3D12_RECT rect = c.rect;
    l.p->ResolveSubresourceRegion(dst.p, c.mip, c.dst_x, c.dst_y, ms.p, 0, c.has_rect ? &rect : nullptr,
                                  DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOLVE_MODE_AVERAGE);
    barrier(l.p, dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION s{}, t{};
    s.pResource = dst.p;
    s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s.SubresourceIndex = c.mip;
    t.pResource = rb.p;
    t.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    t.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 256};
    l.p->CopyTextureRegion(&t, 0, 0, 0, &s, nullptr);
    const HRESULT hr = l.p->Close();
    char head[48];
    std::snprintf(head, sizeof(head), "close 0x%08lx:", static_cast<unsigned long>(hr));
    std::string out = head;
    if (FAILED(hr)) return out;
    ID3D12CommandList* lists[] = {l.p};
    q.p->ExecuteCommandLists(1, lists);
    q.p->Signal(f.p, 1);
    HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    f.p->SetEventOnCompletion(1, e);
    WaitForSingleObject(e, 10000);
    CloseHandle(e);
    const std::uint8_t* p = nullptr;
    D3D12_RANGE range{0, 4 * 256};
    if (FAILED(rb.p->Map(0, &range, reinterpret_cast<void**>(const_cast<std::uint8_t**>(&p))))) return out + " map";
    for (UINT y = 0; y < 4; ++y) {
        out += " ";
        for (UINT x = 0; x < 16; ++x) {
            char hex[4];
            std::snprintf(hex, sizeof(hex), "%02x", p[y * 256 + x]);
            out += hex;
        }
    }
    rb.p->Unmap(0, nullptr);
    return out;
}
}  // namespace

int main() {
    Ref<ID3D12Device> ours, host;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) return 77;
    Ref<IDXGIAdapter> adapter;
    if (std::getenv("PROBE_WARP")) {
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory.p->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))))
            return 77;
    }
    if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return 77;
    const Case cases[] = {
        {"whole source at (0,0), fits", 0, 0, false, {}},
        {"rect (1,1)-(3,3) at (2,2), fits", 2, 2, true, {1, 1, 3, 3}},
        {"whole source at (1,2), 3x2 inside", 1, 2, false, {}},
        {"rect (0,0)-(3,3) at (2,2), 2x2 inside", 2, 2, true, {0, 0, 3, 3}},
        {"rect (1,1)-(4,2) at (3,0), 1x1 inside", 3, 0, true, {1, 1, 4, 2}},
        // Inside the top level but past mip 1: both close, then NVIDIA and WARP write different texels (undefined).
    };
    int bad = 0;
    for (const Case& c : cases) {
        const std::string a = run(ours.p, c), b = run(host.p, c);
        if (a != b) {
            ++bad;
            std::fprintf(stderr, "%s differs:\n  ours: %s\n  host: %s\n", c.name, a.c_str(), b.c_str());
        } else {
            std::printf("%s: matches\n", c.name);
        }
    }
    std::printf("resolve region clip: %d of %zu cases differ\n", bad, sizeof(cases) / sizeof(cases[0]));
    return bad ? 1 : 0;
}
