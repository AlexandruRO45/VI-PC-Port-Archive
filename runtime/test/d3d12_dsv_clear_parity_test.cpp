// Synthetic Close and GPU submission parity for DSV clear flags and out-of-range depth: WARP versus Vulkan COM.
// No game data, device debug layer or proprietary shaders.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* name, DXGI_FORMAT format) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 64;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const D3D12_CLEAR_VALUE clear{format, {1.0f, 0}};
    Ref<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                              IID_PPV_ARGS(&texture.p)))) return 1;
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> dsv_heap;
    if (FAILED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap.p)))) return 1;
    const auto handle = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(texture.p, nullptr, handle);
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p)))) return 1;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return 1;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;

    struct Case { const char* label; D3D12_CLEAR_FLAGS flags; float depth; };
    const Case cases[] = {
        {"flags zero", D3D12_CLEAR_FLAGS(0), 1.0f},
        {"unknown flag bits", D3D12_CLEAR_FLAGS(7), 0.5f},
        {"depth above range", D3D12_CLEAR_FLAG_DEPTH, 1.5f},
        {"depth below range", D3D12_CLEAR_FLAG_DEPTH, -0.5f},
        {"depth NaN", D3D12_CLEAR_FLAG_DEPTH, std::numeric_limits<float>::quiet_NaN()},
        {"positive denormal", D3D12_CLEAR_FLAG_DEPTH, std::numeric_limits<float>::denorm_min()},
        {"negative denormal", D3D12_CLEAR_FLAG_DEPTH, -std::numeric_limits<float>::denorm_min()},
        {"valid depth", D3D12_CLEAR_FLAG_DEPTH, 0.25f},
    };
    for (const auto& c : cases) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p)))) return 1;
        list->ClearDepthStencilView(handle, c.flags, c.depth, 7, 0, nullptr);
        const HRESULT hr = list->Close();
        if (FAILED(hr)) {
            std::fprintf(stderr, "%s %s: Close failed %08x\n", name, c.label, unsigned(hr));
            CloseHandle(event);
            return 1;
        }
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const UINT64 value = UINT64(&c - cases) + 1;
        if (FAILED(queue->Signal(fence.p, value)) || FAILED(fence->SetEventOnCompletion(value, event)) ||
            WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) {
            std::fprintf(stderr, "%s %s: GPU submission did not complete\n", name, c.label);
            CloseHandle(event);
            return 1;
        }
    }
    CloseHandle(event);
    std::printf("%s: %zu DSV clear Close and GPU submissions passed\n", name, std::size(cases));
    return 0;
}

// Four-byte D32 depth plane readback, for a full clear followed by a partial rectangle.
// The stencil plane of D32S8 is deliberately not read here; the host/Vulkan clear test covers it.
bool depth_pixels(ID3D12Device* device, float full, float partial, std::vector<std::uint32_t>& pixels) {
    constexpr UINT width = 5, height = 3;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    Ref<ID3D12Resource> image;
    if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                               D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                               IID_PPV_ARGS(&image.p)))) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placement{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &placement, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    if (FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&readback.p)))) return false;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> descriptors;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptors.p)))) return false;
    const auto handle = descriptors->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(image.p, nullptr, handle);
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
    const auto flags = D3D12_CLEAR_FLAGS(D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL);
    list->ClearDepthStencilView(handle, flags, full, 0x42, 0, nullptr);
    const D3D12_RECT rect{1, 1, 4, 3};
    list->ClearDepthStencilView(handle, flags, partial, 0x89, 1, &rect);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = image.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = placement;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool complete = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                          WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!complete) return false;
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    const auto* bytes = static_cast<const unsigned char*>(mapped);
    pixels.resize(width * height);
    for (UINT y = 0; y < height; ++y)
        std::memcpy(pixels.data() + y * width, bytes + placement.Offset + y * placement.Footprint.RowPitch,
                    width * sizeof(std::uint32_t));
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

int compare_depth_pixels(ID3D12Device* warp, ID3D12Device* layer) {
    const std::uint32_t cases[] = {0x3e800000u, 0x80000000u, 0xbf000000u, 0x3fc00000u, 0xff800000u,
                                   0x7f800000u, 0x7fc12345u, 0x00000001u, 0x80000001u};
    unsigned matched = 0;
    for (const auto bits : cases) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        for (unsigned variant = 0; variant != 2; ++variant) {
            const float full = variant ? 0.375f : value;
            const float part = variant ? value : 0.75f;
            std::vector<std::uint32_t> expected, actual;
            if (!depth_pixels(warp, full, part, expected) || !depth_pixels(layer, full, part, actual)) {
                std::fprintf(stderr, "depth readback failed for input %08x variant %u\n", bits, variant);
                return 1;
            }
            if (expected != actual) {
                for (std::size_t i = 0; i < expected.size() && i < actual.size(); ++i)
                    if (expected[i] != actual[i]) {
                        std::fprintf(stderr, "depth %08x variant %u pixel %zu: WARP=%08x Vulkan=%08x\n",
                                     bits, variant, i, expected[i], actual[i]);
                        break;
                    }
                return 1;
            }
            ++matched;
        }
    }
    std::printf("D32S8 depth pixel matrix: %u/%zu WARP/RTX exact full/rectangle clears\n",
                matched, std::size(cases) * 2);
    return 0;
}
}

int main() {
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;
    Ref<ID3D12Device> host;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return 77;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    // Optional portability controls exercise D24-only GPUs and the no-extension path on a GPU
    // that offers both, without manufacturing an unsupported Vulkan extension or capability.
    const auto enabled = [](const char* name) {
        char value[16]{};
        std::size_t length = 0;
        return getenv_s(&length, value, name) == 0 && length > 1;
    };
    const bool force_d24 = enabled("LEONIDA_DSV_TEST_FORCE_D24");
    const bool no_unrestricted = enabled("LEONIDA_DSV_TEST_NO_UNRESTRICTED");
    const DXGI_FORMAT format = !force_d24 && vk.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT)
                                   ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT
                               : vk.depth_format_usable(VK_FORMAT_D24_UNORM_S8_UINT)
                                   ? DXGI_FORMAT_D24_UNORM_S8_UINT : DXGI_FORMAT_UNKNOWN;
    if (format == DXGI_FORMAT_UNKNOWN) {
        std::puts("no transferable D32S8 or D24S8 depth-stencil attachment on this GPU");
        return 77;
    }
    std::printf("parity DSV format: %s; unrestricted depth: %s\n",
                format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ? "D32S8" : "D24S8",
                vk.depth_range_unrestricted && !no_unrestricted ? "enabled" : "disabled/unavailable");
    if (check(host.p, "native WARP", format)) return 1;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.depth_range_unrestricted = vk.depth_range_unrestricted && !no_unrestricted;
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;
    if (check(layer.p, "Leonida Vulkan", format)) return 1;
    if (info.depth_range_unrestricted && format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
        if (compare_depth_pixels(host.p, layer.p)) return 1;
    } else {
        std::puts("D32S8 unrestricted depth readback matrix skipped: device extension/format unavailable");
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    return 0;
}
