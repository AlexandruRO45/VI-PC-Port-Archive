// Depth and stencil copies on COPY and COMPUTE queues, on the Vulkan shim and native D3D12. On a GPU with a
// transfer-only Vulkan family the layer puts COPY queues there, and COMPUTE queues on an async-compute family; neither
// has graphics. There Vulkan allows a copy between a buffer and a depth or stencil aspect only with
// VK_KHR_maintenance10 and the format's *_COPY_ON_{TRANSFER,COMPUTE}_QUEUE feature
// (VUID-vkCmdCopyBufferToImage-commandBuffer-07739 and its siblings). Each case writes known texels, copies them on the
// queue under test (or on DIRECT around it) and reads them back; a case the layer refuses must leave the device usable.
// No game data. LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return Outcome::failed; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT extent = 16;
constexpr UINT row_pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;  // every plane's row fits in 256 bytes

enum class Outcome { matched, refused, failed };

enum class Case {
    depth_upload,      // footprint -> D32 depth on the queue under test
    stencil_upload,    // footprint -> D32S8 stencil plane
    depth_readback,    // D32 depth -> footprint (uploaded on DIRECT)
    depth_to_depth,    // D32 -> D32 (uploaded and read back on DIRECT): vkCmdCopyImage, no buffer
    depth_to_colour,   // D32 -> R32_FLOAT: staged image -> buffer -> image by the layer
};

const char* name(Case c) {
    switch (c) {
    case Case::depth_upload: return "depth upload";
    case Case::stencil_upload: return "stencil upload";
    case Case::depth_readback: return "depth readback";
    case Case::depth_to_depth: return "depth texture copy";
    case Case::depth_to_colour: return "depth to colour copy";
    }
    return "?";
}

const char* name(D3D12_COMMAND_LIST_TYPE type) { return type == D3D12_COMMAND_LIST_TYPE_COPY ? "COPY" : "COMPUTE"; }

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

bool stencil(Case c) { return c == Case::stencil_upload; }
std::uint32_t texel_bits(Case c, UINT x, UINT y) {
    if (stencil(c)) return (y * extent + x) & 0xff;
    const float value = float(y * extent + x) / float(extent * extent);
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
UINT texel_bytes(Case c) { return stencil(c) ? 1 : 4; }

bool buffer(ID3D12Device& device, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
    const D3D12_HEAP_PROPERTIES heap{type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = UINT64(row_pitch) * extent;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&out)));
}

bool texture(ID3D12Device& device, Case c, ComPtr<ID3D12Resource>& out, bool colour = false) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = extent;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = colour ? DXGI_FORMAT_R32_FLOAT : stencil(c) ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = colour ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                     D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&out)));
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}

D3D12_TEXTURE_COPY_LOCATION placed(ID3D12Resource* resource, Case c) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = {0, {stencil(c) ? DXGI_FORMAT_R8_TYPELESS : DXGI_FORMAT_R32_TYPELESS, extent, extent, 1,
                                    row_pitch}};
    return location;
}

void transition(ID3D12GraphicsCommandList& list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list.ResourceBarrier(1, &barrier);
}

// Three steps, each on its own queue: DIRECT (before), `tested` (COPY or COMPUTE: the copy under test), DIRECT
// (after), with the textures in COMMON between them (copy_between).
Outcome run(ID3D12Device& device, Case c, D3D12_COMMAND_LIST_TYPE tested, bool layer) {
    ComPtr<ID3D12Resource> a, b, upload, readback;
    EXPECT(texture(device, c, a));
    if (c == Case::depth_to_depth || c == Case::depth_to_colour) EXPECT(texture(device, c, b, c == Case::depth_to_colour));
    EXPECT(buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload));
    EXPECT(buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback));
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    EXPECT(SUCCEEDED(upload->Map(0, &none, &mapped)));
    for (UINT y = 0; y < extent; ++y)
        for (UINT x = 0; x < extent; ++x) {
            const auto bits = texel_bits(c, x, y);
            std::memcpy(static_cast<std::uint8_t*>(mapped) + y * row_pitch + x * texel_bytes(c), &bits, texel_bytes(c));
        }
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12CommandQueue> direct, other;
    const D3D12_COMMAND_QUEUE_DESC direct_desc{}, other_desc{tested};
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&direct_desc, IID_PPV_ARGS(&direct))));
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&other_desc, IID_PPV_ARGS(&other))));
    ComPtr<ID3D12CommandAllocator> allocators[3];
    ComPtr<ID3D12GraphicsCommandList> lists[3];
    const D3D12_COMMAND_LIST_TYPE types[3] = {D3D12_COMMAND_LIST_TYPE_DIRECT, tested, D3D12_COMMAND_LIST_TYPE_DIRECT};
    for (int i = 0; i < 3; ++i) {
        EXPECT(SUCCEEDED(device.CreateCommandAllocator(types[i], IID_PPV_ARGS(&allocators[i]))));
        EXPECT(SUCCEEDED(device.CreateCommandList(0, types[i], allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i]))));
    }
    const UINT plane = stencil(c) ? 1 : 0;
    const auto in_a = subresource(a.Get(), plane);
    const auto from_upload = placed(upload.Get(), c), to_readback = placed(readback.Get(), c);
    // Each copy moves the textures it touches out of COMMON and back, so every list starts and ends with them in COMMON,
    // the state the next queue's list expects. Explicit barriers: a depth-stencil texture is not promoted implicitly.
    // The buffers stay in their heap's state (GENERIC_READ upload, COPY_DEST readback).
    const auto copy_between = [](ID3D12GraphicsCommandList& list, const D3D12_TEXTURE_COPY_LOCATION& dst,
                                 const D3D12_TEXTURE_COPY_LOCATION& src) {
        const bool dst_texture = dst.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        const bool src_texture = src.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        if (dst_texture) transition(list, dst.pResource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        if (src_texture) transition(list, src.pResource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (dst_texture) transition(list, dst.pResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        if (src_texture) transition(list, src.pResource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    };
    // The copy under test, on the tested queue.
    switch (c) {
    case Case::depth_upload:
    case Case::stencil_upload:
        copy_between(*lists[1].Get(), in_a, from_upload);
        break;
    case Case::depth_readback:
        copy_between(*lists[0].Get(), in_a, from_upload);
        copy_between(*lists[1].Get(), to_readback, in_a);
        break;
    case Case::depth_to_depth:
    case Case::depth_to_colour: {
        copy_between(*lists[0].Get(), in_a, from_upload);
        const auto in_b = subresource(b.Get(), 0);
        copy_between(*lists[1].Get(), in_b, in_a);
        break;
    }
    }
    // The readback, on DIRECT, unless the tested queue did it.
    if (c != Case::depth_readback) {
        ID3D12Resource* result = b ? b.Get() : a.Get();
        copy_between(*lists[2].Get(), to_readback, subresource(result, plane));
    }
    for (auto& list : lists) EXPECT(SUCCEEDED(list->Close()));

    ComPtr<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ID3D12CommandQueue* queues[3] = {direct.Get(), other.Get(), direct.Get()};
    for (UINT i = 0; i < 3; ++i) {
        if (i) EXPECT(SUCCEEDED(queues[i]->Wait(fence.Get(), i)));
        ID3D12CommandList* batch[] = {lists[i].Get()};
        queues[i]->ExecuteCommandLists(1, batch);
        EXPECT(SUCCEEDED(queues[i]->Signal(fence.Get(), i + 1)));
    }
    const Event event;
    EXPECT(event.handle && SUCCEEDED(fence->SetEventOnCompletion(3, event.handle)));
    EXPECT(WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    // The layer refuses a batch it cannot run (ExecuteCommandLists has no result; the shim records it).
    if (layer && rt::d3d12::last_execute_result(other.Get()) != leonida::ReplayResult::success) return Outcome::refused;

    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    bool same = true;
    for (UINT y = 0; y < extent && same; ++y)
        for (UINT x = 0; x < extent && same; ++x) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, static_cast<const std::uint8_t*>(mapped) + y * row_pitch + x * texel_bytes(c),
                        texel_bytes(c));
            if (bits != texel_bits(c, x, y)) {
                std::fprintf(stderr, "%s, %s on %s: texel (%u,%u) read %#x, wrote %#x\n", layer ? "layer" : "native",
                             name(c), name(tested), x, y, bits, texel_bits(c, x, y));
                same = false;
            }
        }
    readback->Unmap(0, &none);
    return same ? Outcome::matched : Outcome::failed;
}

ComPtr<ID3D12Device> native_device() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return device;
    }
}

const char* outcome(Outcome o) { return o == Outcome::matched ? "matched" : o == Outcome::refused ? "refused" : "FAILED"; }

}  // namespace

int main() {
    ComPtr<ID3D12Device> vulkan;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&vulkan)))) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    const auto* info = rt::d3d12::vulkan_device_info(vulkan.Get());
    if (!info) {
        std::puts("Vulkan device info unavailable");
        return 77;
    }
    std::vector<VkQueueFamilyProperties> queues;
    if (const auto families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            info->get_instance_proc(info->instance, "vkGetPhysicalDeviceQueueFamilyProperties"))) {
        std::uint32_t count = 0;
        families(info->physical, &count, nullptr);
        queues.resize(count);
        families(info->physical, &count, queues.data());
    }
    const auto format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(
        info->get_instance_proc(info->instance, "vkGetPhysicalDeviceFormatProperties2"));
    const auto native = native_device();
    int failures = 0;
    bool any_off_graphics = false;
    for (const D3D12_COMMAND_LIST_TYPE tested : {D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_COMPUTE}) {
        // The Vulkan family the layer runs this D3D12 queue type on: COMPUTE without a family of its own shares the
        // graphics one.
        const std::uint32_t family = tested == D3D12_COMMAND_LIST_TYPE_COPY ? info->copy_family
                                     : info->compute_family < queues.size() ? info->compute_family
                                                                            : info->queue_family;
        const VkQueueFlags flags = family < queues.size() ? queues[family].queueFlags : VK_QUEUE_GRAPHICS_BIT;
        const bool off_graphics = !(flags & VK_QUEUE_GRAPHICS_BIT), compute_family = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
        any_off_graphics = any_off_graphics || off_graphics;
        std::printf("%s queues on Vulkan family %u (%s), graphics family %u\n", name(tested), family,
                    off_graphics ? (compute_family ? "compute, no graphics" : "transfer only") : "has graphics",
                    info->queue_family);
        // What the layer must do, from the GPU's own answers (not the layer's): on a family with graphics every case
        // runs; on one without, a copy with a buffer runs only with maintenance10 and the format's feature for that
        // aspect on that kind of family, and is refused otherwise. vkCmdCopyImage (depth_to_depth) always runs.
        const auto has_feature = [&](VkFormat format, bool stencil_aspect) {
#ifdef VK_KHR_MAINTENANCE_10_EXTENSION_NAME
            if (!info->maintenance10 || !format_properties) return false;
            VkFormatProperties3 features{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
            VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &features};
            format_properties(info->physical, format, &properties);
            const VkFormatFeatureFlags2 bit =
                stencil_aspect ? (compute_family ? VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_COMPUTE_QUEUE_BIT_KHR
                                                 : VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_TRANSFER_QUEUE_BIT_KHR)
                               : (compute_family ? VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_COMPUTE_QUEUE_BIT_KHR
                                                 : VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_TRANSFER_QUEUE_BIT_KHR);
            return (features.optimalTilingFeatures & bit) != 0;
#else
            (void)format, (void)stencil_aspect;
            return false;
#endif
        };
        const auto expected = [&](Case c) {
            if (!off_graphics || c == Case::depth_to_depth) return Outcome::matched;
            const bool runs = stencil(c) ? has_feature(VK_FORMAT_D32_SFLOAT_S8_UINT, true)
                                         : has_feature(VK_FORMAT_D32_SFLOAT, false);
            return runs ? Outcome::matched : Outcome::refused;
        };
        for (const Case c : {Case::depth_upload, Case::stencil_upload, Case::depth_readback, Case::depth_to_depth,
                             Case::depth_to_colour}) {
            const Outcome ours = run(*vulkan.Get(), c, tested, true), want = expected(c);
            const Outcome host = native ? run(*native.Get(), c, tested, false) : Outcome::matched;
            std::printf("  %-8s %-22s layer: %s (expected %s)%s%s\n", name(tested), name(c), outcome(ours),
                        outcome(want), native ? ", native D3D12: " : "", native ? outcome(host) : "");
            if (ours != want || host != Outcome::matched) ++failures;
        }
    }
    if (failures) return 1;
    return any_off_graphics ? 0 : 77;  // without a family lacking graphics the rule under test never applies
}
