// D3D-88: textures placed over the same heap bytes, and aliasing barriers, on the Vulkan device against native D3D12.
//
// The same scenario runs on the layer (a standalone Vulkan device) and on native D3D12 (WARP, and every hardware
// adapter that makes a device), and the answers are compared: which placements succeed or are refused, that the
// heap's reference count doesn't move and that releasing the heap leaves its resources working, and the texels read
// back after each aliasing barrier and clear. Only what D3D12 defines is compared: after an aliasing barrier the
// incoming texture's contents are undefined until it is cleared, so every read follows a clear. On the layer it also
// checks that the overlapping textures really share the heap's memory (the point of placing them).
//
// No game files. Needs a Vulkan GPU; the host comparison needs d3d12.dll (WARP is always there on Windows 10+).
#include "../shims/d3d12_com.h"

#include <dxgi1_6.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

D3D12_RESOURCE_DESC texture_desc(UINT size, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = size;
    desc.Height = size;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

ULONG ref_count(IUnknown* object) {
    object->AddRef();
    return object->Release();
}

constexpr UINT kSize = 256;  // texels per side: 256 KiB of RGBA8, so C at 64 KiB overlaps A and B
constexpr UINT64 kHeapBytes = 4ull << 20;
constexpr UINT64 kSecondPlace = 65536;  // C overlaps the second half of A and B

// What a device did, in an order both kinds of device fill the same way.
struct Answers {
    std::vector<std::string> lines;
    void add(const char* what, long value) {
        char text[160];
        std::snprintf(text, sizeof(text), "%s: %08lx", what, value);
        lines.emplace_back(text);
    }
};

// Everything one device needs to record, run and read back.
class Rig {
public:
    explicit Rig(ID3D12Device* device) : device_(device) {}

    bool init() {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{};
        rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv_desc.NumDescriptors = 8;
        D3D12_HEAP_PROPERTIES readback{};
        readback.Type = D3D12_HEAP_TYPE_READBACK;
        const D3D12_RESOURCE_DESC readback_desc = buffer_desc(UINT64(kSize) * kSize * 4);
        return SUCCEEDED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_.p))) &&
               SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_.p))) &&
               SUCCEEDED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_.p))) &&
               SUCCEEDED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.p, nullptr,
                                                     IID_PPV_ARGS(&list_.p))) &&
               SUCCEEDED(list_->Close()) &&
               SUCCEEDED(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs_.p))) &&
               SUCCEEDED(device_->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&readback_.p)));
    }

    ID3D12GraphicsCommandList* begin() {
        allocator_->Reset();
        list_->Reset(allocator_.p, nullptr);
        return list_.p;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE rtv(UINT slot, ID3D12Resource* target) {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = rtvs_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(slot) * device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        device_->CreateRenderTargetView(target, nullptr, handle);
        return handle;
    }

    static void alias(ID3D12GraphicsCommandList* list, ID3D12Resource* before, ID3D12Resource* after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        barrier.Aliasing.pResourceBefore = before;
        barrier.Aliasing.pResourceAfter = after;
        list->ResourceBarrier(1, &barrier);
    }

    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
                           D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = from;
        barrier.Transition.StateAfter = to;
        list->ResourceBarrier(1, &barrier);
    }

    // Copies `source` (in RENDER_TARGET, left in RENDER_TARGET) to the readback buffer.
    void read_back(ID3D12GraphicsCommandList* list, ID3D12Resource* source) {
        transition(list, source, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        const D3D12_RESOURCE_DESC desc = source->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback_.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = source;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        row_pitch_ = footprint.Footprint.RowPitch;
    }

    bool run() {
        if (FAILED(list_->Close())) return false;
        ID3D12CommandList* lists[] = {list_.p};
        queue_->ExecuteCommandLists(1, lists);
        ++fence_value_;
        return SUCCEEDED(queue_->Signal(fence_.p, fence_value_)) &&
               SUCCEEDED(fence_->SetEventOnCompletion(fence_value_, nullptr));
    }

    // How many texels of the last read-back hold `texel` (4 bytes each).
    UINT count(const std::array<std::uint8_t, 4>& texel) {
        void* data = nullptr;
        D3D12_RANGE range{0, SIZE_T(row_pitch_) * kSize};
        if (FAILED(readback_->Map(0, &range, &data))) return 0;
        UINT n = 0;
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x)
                n += std::memcmp(static_cast<std::uint8_t*>(data) + y * row_pitch_ + x * 4, texel.data(), 4) == 0;
        readback_->Unmap(0, nullptr);
        return n;
    }

private:
    ID3D12Device* device_;
    Ref<ID3D12CommandQueue> queue_;
    Ref<ID3D12Fence> fence_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12DescriptorHeap> rtvs_;
    Ref<ID3D12Resource> readback_;
    UINT64 fence_value_ = 0;
    UINT row_pitch_ = 0;
};

// The scenario. `layer` adds the checks only the Vulkan device can answer. Returns false if it couldn't run at all.
bool scenario(ID3D12Device* device, bool layer, Answers& out, int& layer_failures) {
    Rig rig(device);
    if (!rig.init()) return false;

    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = kHeapBytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
    Ref<ID3D12Heap> heap;
    out.add("CreateHeap (render-target textures only)", device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p)));
    if (!heap.p) return true;

    // A and B are the same texture placed at the same bytes; C, another format, overlaps the second half of them.
    const D3D12_RESOURCE_DESC rgba = texture_desc(kSize, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_RESOURCE_DESC r32 = texture_desc(kSize, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_RESOURCE_DESC plain = texture_desc(kSize, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
    Ref<ID3D12Resource> a, b, c, refused;
    const auto place = [&](UINT64 offset, const D3D12_RESOURCE_DESC& desc, Ref<ID3D12Resource>& r) {
        return device->CreatePlacedResource(heap.p, offset, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                            IID_PPV_ARGS(&r.p));
    };
    out.add("place A at 0", place(0, rgba, a));
    out.add("place B, the same texture, at 0", place(0, rgba, b));
    out.add("place C (R32_FLOAT) over the second half", place(kSecondPlace, r32, c));
    out.add("place a non-render-target texture in a render-target heap", place(0, plain, refused));
    out.add("place running past the heap's end", place(kHeapBytes - kSecondPlace, rgba, refused));
    out.add("place at a 4 KiB offset", place(4096, rgba, refused));
    out.add("heap references after placing three", long(ref_count(heap.p)));
    if (!a.p || !b.p || !c.p) return true;

    if (layer) {
        std::uint64_t mem_a = 0, mem_b = 0, mem_c = 0, at_a = 1, at_b = 1, at_c = 0;
        const bool shared = rt::d3d12::placed_texture_memory(a.p, mem_a, at_a) &&
                            rt::d3d12::placed_texture_memory(b.p, mem_b, at_b) &&
                            rt::d3d12::placed_texture_memory(c.p, mem_c, at_c);
        if (!shared || mem_a != mem_b || mem_a != mem_c || at_a != 0 || at_b != 0 || at_c != kSecondPlace) {
            std::fprintf(stderr, "layer: the placed textures don't share the heap's memory (%d; offsets %llu %llu %llu)\n",
                         int(shared), static_cast<unsigned long long>(at_a), static_cast<unsigned long long>(at_b),
                         static_cast<unsigned long long>(at_c));
            ++layer_failures;
        }
    }

    const float red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1}, blue[4] = {0, 0, 1, 1}, yellow[4] = {1, 1, 0, 1};
    const float seven[4] = {7, 0, 0, 0};
    const std::array<std::uint8_t, 4> red8{255, 0, 0, 255}, green8{0, 255, 0, 255}, blue8{0, 0, 255, 255},
        yellow8{255, 255, 0, 255}, seven32{0x00, 0x00, 0xe0, 0x40};  // 7.0f

    // A is cleared, then B takes the bytes over and is cleared: B reads back its own clear.
    ID3D12GraphicsCommandList* list = rig.begin();
    list->ClearRenderTargetView(rig.rtv(0, a.p), red, 0, nullptr);
    Rig::alias(list, a.p, b.p);
    list->ClearRenderTargetView(rig.rtv(1, b.p), green, 0, nullptr);
    rig.read_back(list, b.p);
    out.add("A -> B, B cleared green: green texels", rig.run() ? long(rig.count(green8)) : -1);

    // Back to A, which is cleared again (its old contents are gone): A reads back blue.
    list = rig.begin();
    Rig::alias(list, b.p, a.p);
    list->ClearRenderTargetView(rig.rtv(0, a.p), blue, 0, nullptr);
    rig.read_back(list, a.p);
    out.add("B -> A, A cleared blue: blue texels", rig.run() ? long(rig.count(blue8)) : -1);

    // C, another format over part of the same bytes: a clear to 7.0 reads back as 7.0 in every texel.
    list = rig.begin();
    Rig::alias(list, a.p, c.p);
    list->ClearRenderTargetView(rig.rtv(2, c.p), seven, 0, nullptr);
    rig.read_back(list, c.p);
    out.add("A -> C, C cleared to 7.0: texels of 7.0", rig.run() ? long(rig.count(seven32)) : -1);

    // A null `after` ("any resource") and a null pair record and run.
    list = rig.begin();
    Rig::alias(list, c.p, nullptr);
    Rig::alias(list, nullptr, nullptr);
    out.add("aliasing barriers with null resources run", rig.run() ? 1 : 0);

    // The heap released while A, B and C live on: A still works, as on D3D12.
    out.add("Release of the heap returns", long(heap.p->Release()));
    heap.p = nullptr;
    list = rig.begin();
    Rig::alias(list, nullptr, a.p);
    list->ClearRenderTargetView(rig.rtv(0, a.p), yellow, 0, nullptr);
    rig.read_back(list, a.p);
    out.add("after the heap's release, A cleared yellow: yellow texels", rig.run() ? long(rig.count(yellow8)) : -1);
    return true;
}

using CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);

}  // namespace

int main() {
    // The layer.
    Answers ours;
    int layer_failures = 0;
    {
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) {
            std::printf("SKIP: no Vulkan device\n");
            return 77;
        }
        if (!scenario(device.p, true, ours, layer_failures)) {
            std::fprintf(stderr, "layer: the scenario couldn't start\n");
            return 1;
        }
    }

    // Native D3D12: WARP and the hardware adapters.
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_device =
        d3d12 ? reinterpret_cast<CreateDeviceFn>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice"))) : nullptr;
    const auto create_factory =
        dxgi ? reinterpret_cast<CreateFactoryFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1"))) : nullptr;
    Ref<IDXGIFactory4> factory;
    if (!create_device || !create_factory || FAILED(create_factory(IID_PPV_ARGS(&factory.p)))) {
        std::fprintf(stderr, "no host D3D12: nothing to compare with\n");
        return 1;
    }
    std::vector<Ref<IUnknown>> owned(8);
    std::vector<std::string> names;
    {
        Ref<IDXGIAdapter> warp;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
            owned[0].p = warp.p;
            warp.p = nullptr;
            names.emplace_back("WARP");
        }
        Ref<IDXGIAdapter1> adapter;
        for (UINT i = 0; names.size() < owned.size() && factory->EnumAdapters1(i, &adapter.p) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                char name[128]{};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
                owned[names.size()].p = adapter.p;
                adapter.p = nullptr;
                names.emplace_back(name);
            }
            adapter.reset();
        }
    }
    int differences = 0, compared = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
        Ref<ID3D12Device> host;
        if (FAILED(create_device(owned[i].p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
        Answers theirs;
        int unused = 0;
        if (!scenario(host.p, false, theirs, unused)) continue;
        ++compared;
        for (std::size_t k = 0; k < ours.lines.size() || k < theirs.lines.size(); ++k) {
            const std::string& mine = k < ours.lines.size() ? ours.lines[k] : std::string("(missing)");
            const std::string& host_line = k < theirs.lines.size() ? theirs.lines[k] : std::string("(missing)");
            if (mine == host_line) continue;
            std::fprintf(stderr, "%s differs: layer \"%s\", host \"%s\"\n", names[i].c_str(), mine.c_str(), host_line.c_str());
            ++differences;
        }
        std::printf("compared with %s: %zu answers\n", names[i].c_str(), theirs.lines.size());
    }
    for (const std::string& line : ours.lines) std::printf("  %s\n", line.c_str());
    if (!compared) {
        std::fprintf(stderr, "no host device made it through the scenario\n");
        return 1;
    }
    if (differences || layer_failures) return 1;
    std::printf("placed textures over the same heap bytes and aliasing barriers matched native D3D12 on %d device(s), "
                "and the layer's textures share the heap's memory\n", compared);
    return 0;
}
