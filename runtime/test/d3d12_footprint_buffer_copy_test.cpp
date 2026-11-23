// D3D12 CopyTextureRegion between two PLACED_FOOTPRINT buffer resources: native WARP vs Vulkan
// for full, boxed, compressed, volume and typeless copies, with real GPU data and padding checks.
// Completely synthetic; no game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr UINT kResourceBytes = 4096, pitch = 256, width = 4, height = 4;
constexpr UINT64 destination_offset = 512;
struct Result { HRESULT close = E_FAIL; bool finished = false; unsigned wrong = 0; };

Result run(ID3D12Device* device, const char* label) {
    Result result;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = kResourceBytes;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD}, readback{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> src, dst;
    if (FAILED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&src.p))) ||
        FAILED(device->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&dst.p)))) return result;
    unsigned char* mapped = nullptr;
    if (FAILED(src->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return result;
    std::memset(mapped, 0xcd, kResourceBytes);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width * 4; ++x)
            mapped[y * pitch + x] = static_cast<unsigned char>(23 + y * 31 + x * 7);
    src->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = src.p;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, pitch}};
    to.pResource = dst.p;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = {destination_offset, {DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, pitch}};

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC q{};
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return result;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    result.close = list->Close();
    std::printf("%s: footprint buffer->buffer Close 0x%08x\n", label, unsigned(result.close));
    if (FAILED(result.close)) return result;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return result;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return result;
    const bool finished = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                          WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!finished) return result;
    result.finished = true;
    const void* data = nullptr;
    const D3D12_RANGE range{SIZE_T(destination_offset), SIZE_T(destination_offset + height * pitch)};
    if (FAILED(dst->Map(0, &range, const_cast<void**>(&data))) || !data) {result.finished = false; return result;}
    const auto* bytes_out = static_cast<const unsigned char*>(data);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width * 4; ++x)
            result.wrong += bytes_out[destination_offset + y * pitch + x] !=
                            static_cast<unsigned char>(23 + y * 31 + x * 7);
    const D3D12_RANGE none{};
    dst->Unmap(0, &none);
    std::printf("%s: fence complete, mismatched payload bytes %u / %u\n",
                label, result.wrong, width * height * 4);
    return result;
}

struct Variant {
    const char* name;
    DXGI_FORMAT format;
    UINT width, height, depth;
    UINT source_pitch, dest_pitch;
    UINT block_bytes, block_extent;
    D3D12_BOX source_box;
    UINT dest_x, dest_y, dest_z;
    DXGI_FORMAT dest_format = DXGI_FORMAT_UNKNOWN; // uses format when unset
};

constexpr Variant variants[] = {
    {"boxed RGBA8, unequal pitches", DXGI_FORMAT_R8G8B8A8_UNORM,
     4, 4, 1, 512, 256, 4, 1, {1, 1, 0, 3, 3, 1}, 0, 2, 0},
    {"volume RGBA8, two slices", DXGI_FORMAT_R8G8B8A8_UNORM,
     4, 4, 3, 256, 512, 4, 1, {0, 1, 1, 4, 4, 3}, 0, 0, 0},
    {"BC1 aligned block", DXGI_FORMAT_BC1_UNORM,
     8, 8, 1, 256, 512, 8, 4, {4, 0, 0, 8, 4, 1}, 0, 4, 0},
    {"R8 one byte row", DXGI_FORMAT_R8_UNORM,
     4, 4, 1, 256, 256, 1, 1, {1, 2, 0, 2, 3, 1}, 2, 0, 0},
    {"RGB96 packed 12-byte texel", DXGI_FORMAT_R32G32B32_FLOAT,
     4, 4, 1, 256, 512, 12, 1, {0, 0, 0, 4, 4, 1}, 0, 0, 0},
    {"RGBA8 typeless to typed", DXGI_FORMAT_R8G8B8A8_TYPELESS,
     4, 4, 1, 256, 256, 4, 1, {0, 0, 0, 4, 4, 1}, 0, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM},
};

Result run_variant(ID3D12Device* device, const char* backend, const Variant& v) {
    constexpr UINT64 capacity = 32768, source_offset = 512, dest_offset = 1024;
    Result result;
    D3D12_RESOURCE_DESC b{};
    b.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    b.Width = capacity;
    b.Height = 1;
    b.DepthOrArraySize = 1;
    b.MipLevels = 1;
    b.SampleDesc.Count = 1;
    b.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD}, readback{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> src, dst;
    if (FAILED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &b,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&src.p))) ||
        FAILED(device->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &b,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&dst.p)))) return result;
    const UINT blocks_x = (v.width + v.block_extent - 1) / v.block_extent;
    const UINT blocks_y = (v.height + v.block_extent - 1) / v.block_extent;
    const UINT src_slice_pitch = blocks_y * v.source_pitch, dst_slice_pitch = blocks_y * v.dest_pitch;
    std::array<unsigned char, capacity> expected{};
    expected.fill(0xd5);
    unsigned char* mapped = nullptr;
    if (FAILED(src->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return result;
    std::memset(mapped, 0xcd, capacity);
    for (UINT z = 0; z < v.depth; ++z)
        for (UINT row = 0; row < blocks_y; ++row)
            for (UINT col = 0; col < blocks_x * v.block_bytes; ++col)
                mapped[source_offset + z * src_slice_pitch + row * v.source_pitch + col] =
                    static_cast<unsigned char>(17 + z * 23 + row * 37 + col * 3);
    src->Unmap(0, nullptr);
    unsigned char* dest_map = nullptr;
    if (FAILED(dst->Map(0, nullptr, reinterpret_cast<void**>(&dest_map))) || !dest_map) return result;
    std::memset(dest_map, 0xd5, capacity);
    const D3D12_RANGE written{0, SIZE_T(capacity)};
    dst->Unmap(0, &written);

    const UINT rows = (v.source_box.bottom - v.source_box.top + v.block_extent - 1) / v.block_extent;
    const UINT row_bytes = (v.source_box.right - v.source_box.left + v.block_extent - 1) /
                           v.block_extent * v.block_bytes;
    for (UINT z = 0; z < v.source_box.back - v.source_box.front; ++z)
        for (UINT row = 0; row < rows; ++row)
            for (UINT col = 0; col < row_bytes; ++col)
                expected[dest_offset + (v.dest_z + z) * dst_slice_pitch +
                         (v.dest_y / v.block_extent + row) * v.dest_pitch +
                         v.dest_x / v.block_extent * v.block_bytes + col] =
                    static_cast<unsigned char>(17 + (v.source_box.front + z) * 23 +
                                               (v.source_box.top / v.block_extent + row) * 37 +
                                               (v.source_box.left / v.block_extent * v.block_bytes + col) * 3);

    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = src.p;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = {source_offset, {v.format, v.width, v.height, v.depth, v.source_pitch}};
    to.pResource = dst.p;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = {dest_offset, {v.dest_format == DXGI_FORMAT_UNKNOWN ? v.format : v.dest_format,
                                      v.width, v.height, v.depth, v.dest_pitch}};
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC q{};
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return result;
    list->CopyTextureRegion(&to, v.dest_x, v.dest_y, v.dest_z, &from, &v.source_box);
    result.close = list->Close();
    std::printf("%s %s: Close 0x%08x\n", backend, v.name, unsigned(result.close));
    if (FAILED(result.close)) return result;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return result;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return result;
    result.finished = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                      WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!result.finished) return result;
    const void* bytes_out = nullptr;
    const D3D12_RANGE read{0, SIZE_T(capacity)};
    if (FAILED(dst->Map(0, &read, const_cast<void**>(&bytes_out))) || !bytes_out) {
        result.finished = false; return result;
    }
    const auto* actual = static_cast<const unsigned char*>(bytes_out);
    for (UINT64 i = 0; i < capacity; ++i) result.wrong += actual[i] != expected[i];
    const D3D12_RANGE none{};
    dst->Unmap(0, &none);
    std::printf("%s %s: %u / %u payload+padding mismatches\n", backend, v.name, result.wrong,
                unsigned(capacity));
    return result;
}
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> adapter;
    Ref<ID3D12Device> native;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory.p->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&native.p)))) return 77;
    const Result host = run(native.p, "native WARP");
    if (FAILED(host.close) || !host.finished || host.wrong) return 2; // unsupported native candidate
    for (const Variant& v : variants) {
        const Result result = run_variant(native.p, "native WARP", v);
        if (FAILED(result.close) || !result.finished || result.wrong) return 2;
    }

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
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;
    const Result ours = run(layer.p, "Leonida RTX");
    unsigned variant_mismatches = 0;
    for (const Variant& v : variants) {
        const Result result = run_variant(layer.p, "Leonida RTX", v);
        variant_mismatches += FAILED(result.close) || !result.finished || result.wrong;
    }
    const unsigned validation = vk.validation_errors();
    std::printf("comparison: WARP Close=0x%08x RTX Close=0x%08x, validation=%u\n",
                unsigned(host.close), unsigned(ours.close), validation);
    layer.p->Release();
    layer.p = nullptr; // owned VulkanDevice must outlive D3D12 wrapper
    return SUCCEEDED(ours.close) && ours.finished && !ours.wrong && !variant_mismatches && !validation ? 0 : 1;
}
