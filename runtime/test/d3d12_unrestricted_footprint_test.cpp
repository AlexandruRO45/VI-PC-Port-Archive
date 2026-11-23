// D3D12 OPTIONS13 buffer/texture copies, compared with native D3D12 and an independent byte oracle.
// Each case writes a patterned source, copies through a texture, and verifies payload and guard bytes.
#include "../shims/d3d12_com.h"
#include "d3d12_copy_compare.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

constexpr SIZE_T buffer_bytes = 4096;
constexpr unsigned char guard_byte = 0xbd;
using Bytes = std::array<unsigned char, buffer_bytes>;

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression); \
        return false; \
    } \
} while (false)

struct Extent { UINT width, height, depth; };
struct Point { UINT x, y, z; };
struct Layout { UINT64 offset; UINT pitch; };
struct Region {
    Point source, texture, destination;
    Extent extent;
};

struct Case {
    const char* name;
    DXGI_FORMAT format;
    D3D12_RESOURCE_DIMENSION dimension;
    Extent texture, footprint;
    UINT bytes_per_block, block;
    Layout upload, download;
    Region region;
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
    bool direct_only = false;
};

constexpr D3D12_RESOURCE_DIMENSION tex1d = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
constexpr D3D12_RESOURCE_DIMENSION tex2d = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
constexpr D3D12_RESOURCE_DIMENSION tex3d = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
constexpr DXGI_FORMAT rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr Extent rectangle{7, 5, 1};
constexpr Region whole{};  // A null source box copies the whole subresource or footprint.
constexpr Case depth_case{"D16 2D depth plane", DXGI_FORMAT_D16_UNORM, tex2d, rectangle, rectangle, 2, 1,
                           {2, 18}, {6, 22}, whole, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, true};
constexpr Case cases[] = {
    {"RGBA8 legacy", rgba, tex2d, rectangle, rectangle, 4, 1, {512, 256}, {1024, 256}, whole},
    {"RGBA8 offset only", rgba, tex2d, rectangle, rectangle, 4, 1, {4, 256}, {20, 256}, whole},
    {"RGBA8 pitch only", rgba, tex2d, rectangle, rectangle, 4, 1, {512, 36}, {1024, 40}, whole},
    {"RGBA8 offset and pitch", rgba, tex2d, rectangle, rectangle, 4, 1, {12, 36}, {28, 44}, whole},
    {"R8 1D odd bytes", DXGI_FORMAT_R8_UNORM, tex1d, {13, 1, 1}, {13, 1, 1}, 1, 1,
     {3, 19}, {5, 21}, whole},
    {"R8 2D odd rows", DXGI_FORMAT_R8_UNORM, tex2d, rectangle, rectangle, 1, 1, {3, 9}, {5, 11}, whole},
    {"R16 2D two-byte alignment", DXGI_FORMAT_R16_UNORM, tex2d, rectangle, rectangle, 2, 1,
     {2, 18}, {6, 22}, whole},
    {"R8 3D inner box", DXGI_FORMAT_R8_UNORM, tex3d, {9, 7, 5}, {8, 6, 4}, 1, 1, {3, 9}, {5, 11},
     {{2, 1, 1}, {1, 2, 2}, {1, 1, 1}, {5, 3, 2}}},
    {"BC1 2D whole blocks", DXGI_FORMAT_BC1_UNORM, tex2d, {12, 8, 1}, {12, 8, 1}, 8, 4,
     {8, 32}, {24, 40}, whole},
    {"BC3 2D block subregion", DXGI_FORMAT_BC3_UNORM, tex2d, {12, 12, 1}, {16, 12, 1}, 16, 4,
     {16, 80}, {32, 96}, {{4, 0, 0}, {0, 4, 0}, {8, 4, 0}, {8, 8, 1}}},
    // The RGB96 conversion needs compute and depth copies need graphics on a dedicated transfer family.
    {"RGB96 2D repack", DXGI_FORMAT_R32G32B32_UINT, tex2d, rectangle, rectangle, 12, 1,
     {12, 96}, {36, 108}, whole, D3D12_RESOURCE_FLAG_NONE, true},
    depth_case,
};

D3D12_RESOURCE_DESC buffer_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = buffer_bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC texture_desc(const Case& test) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = test.dimension;
    desc.Width = test.texture.width;
    desc.Height = test.texture.height;
    desc.DepthOrArraySize = UINT16(test.texture.depth);
    desc.MipLevels = 1;
    desc.Format = test.format;
    desc.SampleDesc.Count = 1;
    desc.Flags = test.flags;
    return desc;
}

bool resource(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, D3D12_HEAP_TYPE heap,
              D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& result) {
    const D3D12_HEAP_PROPERTIES properties{heap};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                    IID_PPV_ARGS(&result))));
    return true;
}

bool write(ID3D12Resource* resource, const Bytes& bytes) {
    void* mapped = nullptr;
    const D3D12_RANGE none{};
    CHECK(SUCCEEDED(resource->Map(0, &none, &mapped)) && mapped);
    std::memcpy(mapped, bytes.data(), bytes.size());
    const D3D12_RANGE written{0, bytes.size()};
    resource->Unmap(0, &written);
    return true;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer, const Case& test, Layout layout) {
    D3D12_TEXTURE_COPY_LOCATION loc{};
    loc.pResource = buffer;
    loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    const Extent e = test.footprint;
    loc.PlacedFootprint = {layout.offset, {test.format, e.width, e.height, e.depth, layout.pitch}};
    return loc;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* texture) {
    D3D12_TEXTURE_COPY_LOCATION loc{};
    loc.pResource = texture;
    loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    return loc;
}

D3D12_BOX box(Point at, Extent size) {
    return {at.x, at.y, at.z, at.x + size.width, at.y + size.height, at.z + size.depth};
}

SIZE_T region_start(const Case& test, Layout layout, Point position) {
    const SIZE_T rows_per_slice = (test.footprint.height + test.block - 1) / test.block;
    return SIZE_T(layout.offset) + position.z * rows_per_slice * layout.pitch
           + (position.y / test.block) * SIZE_T(layout.pitch)
           + (position.x / test.block) * SIZE_T(test.bytes_per_block);
}

Bytes expected_bytes(const Case& test, const Bytes& source, const Bytes& guard) {
    Bytes expected = guard;
    const Extent size = test.region.extent.width ? test.region.extent : test.texture;
    const SIZE_T row_bytes = (size.width + test.block - 1) / test.block * SIZE_T(test.bytes_per_block);
    for (UINT z = 0; z < size.depth; ++z) {
        for (UINT y = 0; y < size.height; y += test.block) {
            const Point src{test.region.source.x, test.region.source.y + y, test.region.source.z + z};
            const Point dst{test.region.destination.x, test.region.destination.y + y, test.region.destination.z + z};
            std::memcpy(expected.data() + region_start(test, test.download, dst),
                        source.data() + region_start(test, test.upload, src), row_bytes);
        }
    }
    return expected;
}

bool execute(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list, bool layer) {
    if (const HRESULT hr = list->Close(); FAILED(hr)) {
        std::fprintf(stderr, "Close failed (HRESULT 0x%08x)\n", unsigned(hr));
        return false;
    }
    ComPtr<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    if (layer) {
        const auto result = rt::d3d12::last_execute_result(queue);
        if (result != leonida::ReplayResult::success) {
            std::fprintf(stderr, "replay failed (ReplayResult %d)\n", int(result));
            return false;
        }
    }
    CHECK(SUCCEEDED(queue->Signal(fence.Get(), 1)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(event);
    const HRESULT hr = fence->SetEventOnCompletion(1, event);
    const DWORD wait = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
    CloseHandle(event);
    CHECK(SUCCEEDED(hr) && wait == WAIT_OBJECT_0 && SUCCEEDED(device->GetDeviceRemovedReason()));
    return true;
}

bool run(ID3D12Device* device, ID3D12CommandQueue* queue, const Case& test, bool layer) {
    const auto type = queue->GetDesc().Type;
    ComPtr<ID3D12Resource> input, seed, image, output, readback;
    const auto buf = buffer_desc();
    CHECK(resource(device, buf, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, input));
    CHECK(resource(device, buf, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, seed));
    CHECK(resource(device, texture_desc(test), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, image));
    CHECK(resource(device, buf, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, output));
    CHECK(resource(device, buf, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback));

    Bytes source{}, guard{};
    for (SIZE_T i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>((i * 37 + i / 127) % 127 + 1);
    guard.fill(guard_byte);
    CHECK(write(input.Get(), source) && write(seed.Get(), guard));

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list))));

    // Seed the GPU destination, then preserve every byte outside the copied rows.
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(output.Get(), 0, seed.Get(), 0, buffer_bytes);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    transition(list.Get(), image.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

    const auto src = footprint(input.Get(), test, test.upload);
    const auto tex = subresource(image.Get());
    const auto dst = footprint(output.Get(), test, test.download);
    const Region& region = test.region;
    const bool partial = region.extent.width != 0;
    const auto upload_box = box(region.source, region.extent);
    list->CopyTextureRegion(&tex, region.texture.x, region.texture.y, region.texture.z,
                            &src, partial ? &upload_box : nullptr);
    transition(list.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto read_box = box(region.texture, region.extent);
    list->CopyTextureRegion(&dst, region.destination.x, region.destination.y, region.destination.z,
                            &tex, partial ? &read_box : nullptr);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, buffer_bytes);
    CHECK(execute(device, queue, list.Get(), layer));

    const Bytes expected = expected_bytes(test, source, guard);
    Bytes actual{};
    void* data = nullptr;
    const D3D12_RANGE read{0, buffer_bytes}, none{};
    CHECK(SUCCEEDED(readback->Map(0, &read, &data)) && data);
    std::memcpy(actual.data(), data, actual.size());
    readback->Unmap(0, &none);
    const auto [got, wanted] = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (got != actual.end()) {
        std::fprintf(stderr, "byte %zu: got %02x, expected %02x\n",
                     SIZE_T(got - actual.begin()), unsigned(*got), unsigned(*wanted));
        return false;
    }
    return true;
}

bool invalid_placements(ID3D12Device* device) {
    const Case& test = cases[0];
    ComPtr<ID3D12Resource> upload, image;
    CHECK(resource(device, buffer_desc(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload));
    CHECK(resource(device, texture_desc(test), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, image));
    for (Layout layout : {Layout{3, 36}, Layout{4, 29}, Layout{4, 24}}) {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                                  IID_PPV_ARGS(&list))));
        const auto src = footprint(upload.Get(), test, layout);
        const auto dst = subresource(image.Get());
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        CHECK(list->Close() == E_INVALIDARG);
    }
    return true;
}

bool format_supported(ID3D12Device* device, const Case& test) {
    if (test.block == 1 && !test.direct_only) return true;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{test.format};
    const auto required = D3D12_FORMAT_SUPPORT1_TEXTURE2D |
        (test.flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ? D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL : 0);
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
           (support.Support1 & required) == required;
}

bool run_all(ID3D12Device* device, bool layer) {
    const char* backend = layer ? "Vulkan layer" : "Native D3D12";
    constexpr std::array types{D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COPY};
    std::array<ComPtr<ID3D12CommandQueue>, types.size()> queues;
    // Keep DIRECT alive while COPY runs so its completed scratch stays available for reuse.
    for (std::size_t i = 0; i < types.size(); ++i) {
        const D3D12_COMMAND_QUEUE_DESC desc{types[i]};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queues[i]))));
    }
    for (std::size_t i = 0; i < types.size(); ++i) {
        const auto type = types[i];
        const char* queue_name = type == D3D12_COMMAND_LIST_TYPE_COPY ? "COPY" : "DIRECT";
        unsigned passed = 0;
        for (const Case& test : cases) {
            if (test.direct_only && type != D3D12_COMMAND_LIST_TYPE_DIRECT) continue;
            if (!format_supported(device, test)) {
                std::printf("%s %s skipped: %s (format unavailable)\n", backend, queue_name, test.name);
                continue;
            }
            if (!run(device, queues[i].Get(), test, layer)) {
                std::fprintf(stderr, "%s %s failed: %s\n", backend, queue_name, test.name);
                return false;
            }
            ++passed;
        }
        std::printf("%s %s: %u byte-exact round trips with intact guards\n", backend, queue_name, passed);
    }
    // Repeat the two-byte-aligned depth copy after COPY has used the shared scratch.
    if (format_supported(device, depth_case)) {
        CHECK(run(device, queues.front().Get(), depth_case, layer));
        std::printf("%s DIRECT after COPY: byte-exact depth round trip with intact guards\n", backend);
    }
    return true;
}

bool unrestricted(ID3D12Device* device) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 options{};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS13, &options, sizeof(options)))
           && options.UnrestrictedBufferTextureCopyPitchSupported;
}

}  // namespace

int main(int argc, char** argv) {
    const bool layer_only = argc == 2 && std::strcmp(argv[1], "--layer-only") == 0;
    const bool native_only = argc == 2 && std::strcmp(argv[1], "--native-only") == 0;
    const bool warp_only = argc == 2 && std::strcmp(argv[1], "--warp-only") == 0;
    if (argc != 1 && !layer_only && !native_only && !warp_only) return 2;

    if (!native_only && !warp_only) {
        ComPtr<ID3D12Device> layer;
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer)))) {
            std::puts("Vulkan standalone device unavailable");
            return 77;
        }
        copycmp::layer_adapter() = layer->GetAdapterLuid();
        if (!run_all(layer.Get(), true) || !invalid_placements(layer.Get())) return 1;
        if (!unrestricted(layer.Get())) {
            std::fputs("Vulkan layer did not advertise unrestricted footprints in OPTIONS13\n", stderr);
            return 1;
        }
        std::puts("Vulkan layer OPTIONS13: supported; 3 invalid footprints rejected at Close");
    }
    if (!layer_only) {
        if (warp_only && copycmp::host_is_wine()) return 77;
        if (warp_only && !SetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", "warp")) return 1;
        std::string name;
        copycmp::Ptr<ID3D12Device> native;
        if (!copycmp::create_host(native, name) || !unrestricted(native.p)) {
            std::puts("Native D3D12 comparison skipped: no adapter with OPTIONS13 unrestricted footprints");
            return argc == 1 ? 0 : 77;
        }
        std::printf("Native D3D12 adapter: %s\n", name.c_str());
        if (!run_all(native.p, false)) return 1;
    }
    return 0;
}
