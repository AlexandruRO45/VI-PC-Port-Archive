// Same-sample colour decompression on the Vulkan layer, the native adapter and WARP.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

constexpr DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr UINT width = 8, height = 6;
constexpr D3D12_RECT detail_rect{2, 1, 4, 4};

struct Case {
    const char* label;
    UINT source_samples, destination_samples;
    UINT x, y;
    const D3D12_RECT* rect;
    bool in_place = false;
    DXGI_FORMAT view_format = format;
    bool accepted = true;
};

bool run_case(ID3D12Device* device, const Case& test) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto make_texture = [&](Ref<ID3D12Resource>& out, UINT samples, D3D12_RESOURCE_STATES initial) {
        texture.SampleDesc.Count = samples;
        return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture, initial, nullptr,
                                                         IID_PPV_ARGS(&out.p)));
    };

    Ref<ID3D12Resource> source, target, resolved;
    CHECK(make_texture(source, test.source_samples, D3D12_RESOURCE_STATE_RENDER_TARGET));
    if (!test.in_place) CHECK(make_texture(target, test.destination_samples, D3D12_RESOURCE_STATE_RENDER_TARGET));

    Ref<ID3D12DescriptorHeap> views;
    const D3D12_DESCRIPTOR_HEAP_DESC view_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&view_desc, IID_PPV_ARGS(&views.p))));
    const auto source_view = views->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE target_view{
        source_view.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    device->CreateRenderTargetView(source.p, nullptr, source_view);
    const FLOAT red[]{1, 0, 0, 1}, green[]{0, 1, 0, 1}, blue[]{0, 0, 1, 1};
    list->ClearRenderTargetView(source_view, red, 0, nullptr);
    list->ClearRenderTargetView(source_view, green, 1, &detail_rect);
    if (!test.in_place) {
        device->CreateRenderTargetView(target.p, nullptr, target_view);
        list->ClearRenderTargetView(target_view, blue, 0, nullptr);
    }

    const auto transition = [&](ID3D12Resource* image, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
        list->ResourceBarrier(1, &barrier);
    };

    transition(source.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    if (!test.in_place)
        transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    ID3D12Resource* output = test.in_place ? source.p : target.p;
    auto output_state = test.in_place ? D3D12_RESOURCE_STATE_RESOLVE_SOURCE : D3D12_RESOURCE_STATE_RESOLVE_DEST;

    auto rect = test.rect ? *test.rect : D3D12_RECT{};
    list1->ResolveSubresourceRegion(output, 0, test.x, test.y, source.p, 0, test.rect ? &rect : nullptr,
                                    test.view_format, D3D12_RESOLVE_MODE_DECOMPRESS);
    if (!test.accepted) {
        const HRESULT closed = list->Close();
        if (closed != E_INVALIDARG) {
            std::fprintf(stderr, "%s: invalid call Close returned %#lx\n", test.label, closed);
            return false;
        }
        return true;
    }

    if (test.destination_samples > 1) {
        CHECK(make_texture(resolved, 1, D3D12_RESOURCE_STATE_RESOLVE_DEST));
        if (output_state != D3D12_RESOURCE_STATE_RESOLVE_SOURCE)
            transition(output, output_state, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        list->ResolveSubresource(resolved.p, 0, output, 0, format);
        output = resolved.p;
        output_state = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    }
    transition(output, output_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto output_desc = output->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&output_desc, 0, 1, 0, &footprint, nullptr, nullptr, &size);

    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = size;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    Ref<ID3D12Resource> readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));

    D3D12_TEXTURE_COPY_LOCATION from{output, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION to{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);

    const auto closed = list->Close();
    if (FAILED(closed)) {
        std::fprintf(stderr, "%s: Close returned %#lx\n", test.label, closed);
        return false;
    }
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    Event event;
    CHECK(event.handle);
    ID3D12CommandList* lists[]{list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 30000) == WAIT_OBJECT_0);

    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, SIZE_T(size)};
    CHECK(SUCCEEDED(readback->Map(0, &read_range, &mapped)) && mapped);
    const UINT left = test.rect ? UINT(rect.left) : 0, top = test.rect ? UINT(rect.top) : 0;
    const UINT right = test.rect ? UINT(rect.right) : width, bottom = test.rect ? UINT(rect.bottom) : height;
    constexpr std::array<unsigned char, 4> full_red{255, 0, 0, 255}, full_green{0, 255, 0, 255},
        full_blue{0, 0, 255, 255};
    unsigned differences = 0;
    const auto* data = static_cast<const unsigned char*>(mapped) + footprint.Offset;
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const bool copied = x >= test.x && y >= test.y && x - test.x < right - left && y - test.y < bottom - top;
            const UINT sx = test.in_place ? x : left + x - test.x, sy = test.in_place ? y : top + y - test.y;
            const bool green_texel = sx >= UINT(detail_rect.left) && sx < UINT(detail_rect.right) &&
                                     sy >= UINT(detail_rect.top) && sy < UINT(detail_rect.bottom);
            const auto& expected = (test.in_place || copied) ? (green_texel ? full_green : full_red) : full_blue;
            const unsigned char* pixel = data + y * footprint.Footprint.RowPitch + 4 * x;
            if (std::memcmp(pixel, expected.data(), 4) && differences++ == 0)
                std::fprintf(stderr, "%s (%u,%u): got %u,%u,%u,%u, expected %u,%u,%u,%u\n", test.label, x, y,
                    pixel[0], pixel[1], pixel[2], pixel[3], expected[0], expected[1], expected[2], expected[3]);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return differences == 0;
}

bool verify(ID3D12Device* device, const char* name, bool layer) {
    constexpr D3D12_RECT source_rect{1, 0, 5, 3}, bad_source{0, 0, 9, 4};
    const Case cases[] = {
        {"1x whole", 1, 1, 0, 0, nullptr},
        {"1x shifted", 1, 1, 3, 2, &source_rect},
        {"4x shifted", 4, 4, 3, 2, &source_rect},
        {"1x in place", 1, 1, 0, 0, nullptr, true},
        {"4x in place", 4, 4, 1, 0, &source_rect, true},
        {"mismatched sample counts", 4, 1, 0, 0, nullptr, false, format, false},
        {"mismatched format", 1, 1, 0, 0, nullptr, false, DXGI_FORMAT_B8G8R8A8_UNORM, false},
        {"out-of-bounds source", 1, 1, 0, 0, &bad_source, false, format, false},
        {"out-of-bounds destination", 4, 4, 5, 4, &source_rect, false, format, false},
    };
    unsigned matched = 0;
    for (const auto& test : cases) {
        const bool passed = run_case(device, test);
        matched += passed;
        if (!passed) std::fprintf(stderr, "%s: %s failed\n", name, test.label);
    }
    // The native runtimes accept Close for this undefined in-place displacement. Only test rejection in the layer.
    if (layer) {
        const Case shifted{"shifted in place", 1, 1, 0, 0, &source_rect, true, format, false};
        if (!run_case(device, shifted)) return false;
    }
    std::printf("%s: %u/%zu DECOMPRESS cases matched\n", name, matched + unsigned(layer),
                std::size(cases) + unsigned(layer));
    return matched == std::size(cases);
}

} // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    const unsigned errors_before = vk.validation_errors();
    bool passed = false;
    {
        Ref<ID3D12Device> backend;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&backend.p)))) return 1;
        passed = verify(backend.p, "Leonida Vulkan", true);
    }

    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return 1;
        unsigned references = 0;
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            std::string name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, name)) continue;
            Ref<ID3D12Device> host;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            ++references;
            passed = verify(host.p, name.c_str(), false) && passed;
        }
        if (!references) return 1;
    }
    return passed && vk.validation_errors() == errors_before ? 0 : 1;
}
