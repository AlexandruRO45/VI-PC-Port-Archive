// ID3D12Device3::OpenExistingHeapFromAddress, OpenExistingHeapFromFileMapping and ID3D12Device5::CreateLifetimeTracker
// on arguments the host D3D12 runtime refuses (null address, a file mapping that is not a handle, a null owner), all
// E_INVALIDARG on the RTX 4080 SUPER, Intel UHD 770 and WARP. Checked on the Vulkan-backed device and, when present,
// on the host devices too. With VK_EXT_external_memory_host, OpenExistingHeapFromAddress on real host memory is
// compared case by case with the host devices (open_cases: RTX 4090 and WARP agree), and a heap is opened on a
// committed range; without it, the device reports EXISTING_HEAPS unsupported and answers an accepted range E_NOTIMPL.
// No shaders or game data.
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device3-openexistingheapfromaddress
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/title_page_watch.h"
#include <dxgi1_4.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)
#define EXPECT_HR(expr, want) do { \
    const HRESULT got_ = (expr); \
    if (got_ != (want)) { \
        std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, #expr, \
                     static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
        return 1; \
    } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device3> device3;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
    Ref<ID3D12Heap> heap;
    EXPECT_HR(device3->OpenExistingHeapFromAddress(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(nullptr, IID_PPV_ARGS(&heap.p)), E_INVALIDARG);
    EXPECT_HR(device3->OpenExistingHeapFromFileMapping(reinterpret_cast<HANDLE>(0x1234), IID_PPV_ARGS(&heap.p)),
              E_INVALIDARG);
    EXPECT(heap.p == nullptr);
    Ref<ID3D12Device5> device5;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5.p)))) {
        Ref<ID3D12LifetimeTracker> tracker;
        EXPECT_HR(device5->CreateLifetimeTracker(nullptr, IID_PPV_ARGS(&tracker.p)), E_INVALIDARG);
        EXPECT(tracker.p == nullptr);
    }
    std::printf("%s: existing-heap and lifetime-tracker refusals match the host runtime\n", who);
    return 0;
}

// The calls the host accepts or refuses on real host memory, one result per case, for the comparison in main().
struct Case {
    const char* name;
    HRESULT hr = S_OK;
    D3D12_HEAP_DESC desc{};
};

std::vector<Case> open_cases(ID3D12Device* device) {
    std::vector<Case> out;
    Ref<ID3D12Device3> device3;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device3.p)))) return out;
    auto run = [&](const char* name, const void* address) {
        Case c{name};
        Ref<ID3D12Heap> heap;
        c.hr = device3->OpenExistingHeapFromAddress(address, IID_PPV_ARGS(&heap.p));
        if (SUCCEEDED(c.hr) && heap.p) c.desc = heap->GetDesc();
        out.push_back(c);
    };
    constexpr SIZE_T k64 = 0x10000;
    // A whole 64 KiB committed read-write allocation, as the title's XMemVirtualAlloc gives it.
    void* whole = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    run("64 KiB committed, at its base", whole);
    run("64 KiB committed, 4 KiB in", static_cast<char*>(whole) + 0x1000);
    // Two committed 64 KiB in one 192 KiB allocation whose last 64 KiB is only reserved.
    char* partial = static_cast<char*>(VirtualAlloc(nullptr, 3 * k64, MEM_RESERVE, PAGE_READWRITE));
    VirtualAlloc(partial, 2 * k64, MEM_COMMIT, PAGE_READWRITE);
    run("192 KiB reserved, first 128 KiB committed", partial);
    void* reserved = VirtualAlloc(nullptr, k64, MEM_RESERVE, PAGE_READWRITE);
    run("64 KiB reserved only", reserved);
    void* readonly = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    run("64 KiB committed read-only", readonly);
    // The console title's graphics regions keep PAGE_WRITECOMBINE (or PAGE_NOCACHE) once XMemVirtualAlloc drops
    // the PAGE_GRAPHICS_* bits.
    void* combined = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE);
    run("64 KiB committed read-write write-combined", combined);
    void* uncached = VirtualAlloc(nullptr, k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_NOCACHE);
    run("64 KiB committed read-write uncached", uncached);
    void* big = VirtualAlloc(nullptr, 32 * k64, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    run("2 MiB committed", big);
    // The heap keeps no hold on the memory the test frees here; every heap above is already released.
    for (void* p : {whole, static_cast<void*>(partial), reserved, readonly, combined, uncached, big}) VirtualFree(p, 0, MEM_RELEASE);
    return out;
}

// Each case answers as the host does: the same HRESULT, and for an opened heap the same description.
int compare_cases(const std::vector<Case>& ours, const std::vector<Case>& host, const char* who) {
    int failures = 0;
    for (size_t i = 0; i < ours.size() && i < host.size(); ++i) {
        const D3D12_HEAP_DESC& a = ours[i].desc;
        const D3D12_HEAP_DESC& b = host[i].desc;
        // The host's Alignment varies from run to run for the same allocation (64 KiB on some runs, 4 MiB on others,
        // on an RX 9070 XT, an RTX 5090 and WARP; found in review), so it is not compared: the layer's must be 64 KiB
        // for an opened heap, and the host's either of the two.
        const bool alignment_ok =
            !SUCCEEDED(ours[i].hr) ||
            (a.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT &&
             (b.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT || b.Alignment == 4ull * 1024 * 1024));
        const bool same = ours[i].hr == host[i].hr && a.SizeInBytes == b.SizeInBytes && alignment_ok &&
                          a.Properties.Type == b.Properties.Type &&
                          a.Properties.CPUPageProperty == b.Properties.CPUPageProperty &&
                          a.Properties.MemoryPoolPreference == b.Properties.MemoryPoolPreference && a.Flags == b.Flags;
        if (!same) {
            std::fprintf(stderr, "vulkan vs %s: \"%s\" differs (hr 0x%08lX vs 0x%08lX)\n", who, ours[i].name,
                         static_cast<unsigned long>(ours[i].hr), static_cast<unsigned long>(host[i].hr));
            ++failures;
        }
    }
    if (!failures) std::printf("vulkan vs %s: %zu OpenExistingHeapFromAddress cases match\n", who, ours.size());
    return failures;
}

// The heap is the caller's memory: CheckFeatureSupport says so, and the heap stays valid while it is held.
// Without the extension the feature is unsupported and an accepted range is E_NOTIMPL.
int check_import(ID3D12Device* device, bool supported) {
    const char* who = "vulkan import";
    D3D12_FEATURE_DATA_EXISTING_HEAPS feature{};
    EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature))));
    EXPECT(bool(feature.Supported) == supported);
    Ref<ID3D12Device3> device3;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
    constexpr SIZE_T size = 4 * 0x10000;
    void* memory = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EXPECT(memory != nullptr);
    {
        Ref<ID3D12Heap> heap;
        const HRESULT hr = device3->OpenExistingHeapFromAddress(memory, IID_PPV_ARGS(&heap.p));
        if (!supported) {
            EXPECT_HR(hr, E_NOTIMPL);
            EXPECT(heap.p == nullptr);
        } else {
            EXPECT_HR(hr, S_OK);
            EXPECT(heap.p != nullptr);
            EXPECT(heap->GetDesc().SizeInBytes == size);
            // The CPU still owns the pages: a write through them is what the GPU will read.
            static_cast<unsigned char*>(memory)[size - 1] = 0x5a;
            // A placed resource in it is refused until the placed path addresses existing heaps.
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = 0x10000;
            buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            Ref<ID3D12Resource> placed;
            EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&placed.p)),
                      E_NOTIMPL);
        }
    }
    EXPECT(VirtualFree(memory, 0, MEM_RELEASE));
    std::printf("%s: existing heap %s\n", who, supported ? "opened on the caller's memory" : "unsupported, as reported");
    return 0;
}

// Xbox device slot 133's backend (rt::d3d12::create_buffer_at_host_address): a buffer on the caller's own pages.
// Its GPU VA is the address itself, and what the CPU writes there is what a GPU copy reads, with no upload: the
// pattern is written after the buffer exists and read back through a READBACK buffer. Checked on a VirtualAlloc range
// and on a view of a pagefile-backed section, the form xmem's physical pages are mapped in. A texture is E_NOTIMPL
// and an address off the import alignment E_INVALIDARG.
int check_title_placed(ID3D12Device* device, const char* kind, void* pages, UINT64 size) {
    const char* who = "vulkan slot-133 buffer";
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    EXPECT_HR(rt::d3d12::create_buffer_at_host_address(device, pages, desc, 0xFFFFFFFFu, IID_PPV_ARGS(&buffer.p)), S_OK);
    EXPECT(buffer->GetGPUVirtualAddress() == reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(pages));
    auto* bytes = static_cast<std::uint8_t*>(pages);
    for (UINT64 i = 0; i < size; ++i) bytes[i] = static_cast<std::uint8_t>(i * 131u + 7u);

    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(readback.p, 0, buffer.p, 0, size);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    EXPECT(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const bool same = std::memcmp(mapped, pages, static_cast<size_t>(size)) == 0;
    readback->Unmap(0, nullptr);
    EXPECT(same);

    D3D12_RESOURCE_DESC texture = desc;
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = 64;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // A texture isn't placed on the pages yet: the title gets an ordinary committed texture instead of a refusal.
    Ref<ID3D12Resource> placed_texture;
    EXPECT_HR(rt::d3d12::create_buffer_at_host_address(device, pages, texture, 0x6u, IID_PPV_ARGS(&placed_texture.p)), S_OK);
    EXPECT(placed_texture.p != nullptr && placed_texture->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D);
    // An address off the buffer alignment can't be bound inside the import: a shadow buffer answers, at the same VA
    // (a driver whose buffer alignment is 1 imports it; the VA is the address either way).
    Ref<ID3D12Resource> fallback;
    desc.Width = size - 1;
    EXPECT_HR(rt::d3d12::create_buffer_at_host_address(device, bytes + 1, desc, 0xFFFFFFFFu, IID_PPV_ARGS(&fallback.p)), S_OK);
    EXPECT(fallback->GetGPUVirtualAddress() == reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(bytes + 1));
    std::printf("%s: %s, %llu bytes: GPU VA is the address, a GPU copy reads the CPU's writes\n", who, kind,
                static_cast<unsigned long long>(size));
    return 0;
}

std::uint8_t pattern_byte(UINT64 i, unsigned seed) {
    return static_cast<std::uint8_t>(i * 131u + (i >> 8) * 7u + seed);
}

// The shadow fallback of slot 133: pages the driver does not import (write-combined, uncached, not all committed)
// still give a buffer whose GPU VA is the address. It is a copy of the pages, made again before every submission:
// a pattern written to the first `committed` bytes after the buffer exists is what a GPU copy in the next
// ExecuteCommandLists reads, and so is a second pattern in the submission after. Bytes past `committed` are not
// committed: they are never read, and the GPU sees zeros there. The same holds where the range was imported after
// all (a Wine that drops the cache modifier), so the caller need not know which path answered.
int check_title_placed_shadow(ID3D12Device* device, const char* kind, void* pages, UINT64 committed, UINT64 size) {
    const char* who = "vulkan slot-133 shadow";
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    EXPECT_HR(rt::d3d12::create_buffer_at_host_address(device, pages, desc, 0xFFFFFFFFu, IID_PPV_ARGS(&buffer.p)), S_OK);
    EXPECT(buffer->GetGPUVirtualAddress() == reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(pages));
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(done != nullptr);
    auto* bytes = static_cast<std::uint8_t*>(pages);
    UINT64 fence_value = 0;
    for (const unsigned seed : {7u, 93u}) {
        // Written after the buffer exists (and, the second time, after a submission read the first pattern).
        for (UINT64 i = 0; i < committed; ++i) bytes[i] = pattern_byte(i, seed);
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        list->CopyBufferRegion(readback.p, 0, buffer.p, 0, size);
        EXPECT(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, done)));
        EXPECT(WaitForSingleObject(done, 30000) == WAIT_OBJECT_0);
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        const auto* got = static_cast<const std::uint8_t*>(mapped);
        UINT64 wrong = size;
        for (UINT64 i = 0; i < size && wrong == size; ++i)
            if (got[i] != (i < committed ? pattern_byte(i, seed) : 0)) wrong = i;
        readback->Unmap(0, nullptr);
        if (wrong != size)
            std::fprintf(stderr, "%s: %s: pattern %u: first wrong byte at %llu\n", who, kind, seed,
                         static_cast<unsigned long long>(wrong));
        EXPECT(wrong == size);
    }
    CloseHandle(done);
    std::printf("%s: %s, %llu bytes (%llu committed): GPU VA is the address, each submission reads the pages' "
                "latest bytes\n",
                who, kind, static_cast<unsigned long long>(size), static_cast<unsigned long long>(committed));
    return 0;
}

int check_title_placed_all(ID3D12Device* device) {
    constexpr SIZE_T size = 4 * 0x10000;
    int failures = 0;
    void* pages = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!pages) return 1;
    failures += check_title_placed(device, "VirtualAlloc range", pages, size);
    // Inside a page, as the title places small UAV buffers (64-byte aligned, found in review on an RTX 5090): the
    // import starts at the page and the buffer is bound at the offset.
    failures += check_title_placed(device, "64 bytes into a page", static_cast<char*>(pages) + 64, 0x1000 - 64);
    VirtualFree(pages, 0, MEM_RELEASE);
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, DWORD(size), nullptr);
    void* view = section ? MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size) : nullptr;
    if (!view) return failures + 1;
    failures += check_title_placed(device, "section view", view, size);
    UnmapViewOfFile(view);
    CloseHandle(section);

    // Cache-modified ranges are not imported (WC imports read wrong bytes on an RTX 5090 in review): a shadow buffer
    // answers at the same VA. Some Wine versions report a requested WC allocation as plain read-write memory; check
    // the protection actually supplied. PAGE_NOCACHE also exercises the fallback on those hosts, rather than
    // dropping the cache-protection check.
    const char* who = "vulkan slot-133 buffer";
    for (DWORD modifier : {DWORD(PAGE_WRITECOMBINE), DWORD(PAGE_NOCACHE)}) {
        void* pages = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | modifier);
        EXPECT(pages != nullptr);
        MEMORY_BASIC_INFORMATION region{};
        EXPECT(VirtualQuery(pages, &region, sizeof(region)) == sizeof(region));
        const bool cache_modified = (region.Protect & (PAGE_WRITECOMBINE | PAGE_NOCACHE)) != 0;
        if (!cache_modified) EXPECT(region.Protect == PAGE_READWRITE);
        const int failed = check_title_placed_shadow(device, modifier == PAGE_WRITECOMBINE ? "write-combined range"
                                                                                           : "uncached range",
                                                     pages, size, size);
        std::printf("%s: cache modifier 0x%lx, reported protection 0x%lx: %s\n", who, modifier, region.Protect,
                    failed ? "FAILED"
                           : cache_modified ? "shadow buffer verified"
                                            : "modifier not reported; read-write import verified");
        failures += failed;
        VirtualFree(pages, 0, MEM_RELEASE);
    }
    {
        // A larger write-combined range, so the per-submission copy the layer logs is measured on more than a few
        // pages (retail places 182 MiB, 74 MiB of it write-combined).
        constexpr SIZE_T large = 64u << 20;
        void* pages = VirtualAlloc(nullptr, large, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE);
        EXPECT(pages != nullptr);
        failures += check_title_placed_shadow(device, "64 MiB write-combined range", pages, large, large);
        VirtualFree(pages, 0, MEM_RELEASE);
    }
    char* partial = static_cast<char*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE));
    EXPECT(partial != nullptr && VirtualAlloc(partial, size / 2, MEM_COMMIT, PAGE_READWRITE) == partial);
    // Pages not committed yet (xmem commits on first touch) are not imported either: the shadow copies the committed
    // half and never reads the other, which a GPU copy sees as zeros.
    failures += check_title_placed_shadow(device, "half-committed range", partial, size / 2, size);
    VirtualFree(partial, 0, MEM_RELEASE);
    return failures;
}

// Submits a GPU copy of all of `buffer` to a READBACK buffer and waits for it: what the GPU read at that submission.
struct GpuReader {
    const char* who = "vulkan slot-133 written pages";
    ID3D12Device* device = nullptr;
    UINT64 size = 0;
    Ref<ID3D12Resource> readback;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    HANDLE done = nullptr;
    UINT64 fence_value = 0;

    ~GpuReader() {
        if (done) CloseHandle(done);
    }
    int init(ID3D12Device* d, UINT64 bytes) {
        device = d;
        size = bytes;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(done != nullptr);
        return 0;
    }
    int read(ID3D12Resource* buffer, std::vector<std::uint8_t>& out) {
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        list->CopyBufferRegion(readback.p, 0, buffer, 0, size);
        EXPECT(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, done)));
        EXPECT(WaitForSingleObject(done, 30000) == WAIT_OBJECT_0);
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        out.assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) + size);
        readback->Unmap(0, nullptr);
        return 0;
    }
};

// The first byte where the GPU's copy differs from `want`, or -1.
long long first_difference(const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want) {
    for (size_t i = 0; i < got.size() && i < want.size(); ++i)
        if (got[i] != want[i]) return static_cast<long long>(i);
    return got.size() == want.size() ? -1 : static_cast<long long>(std::min(got.size(), want.size()));
}

ID3D12Resource* place_shadow(ID3D12Device* device, void* pages, UINT64 size) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* buffer = nullptr;
    if (FAILED(rt::d3d12::create_buffer_at_host_address(device, pages, desc, 0xFFFFFFFFu, IID_PPV_ARGS(&buffer))))
        return nullptr;
    if (buffer->GetGPUVirtualAddress() != reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(pages)) {
        buffer->Release();
        return nullptr;
    }
    return buffer;
}

DWORD protection_of(const void* address) {
    MEMORY_BASIC_INFORMATION region{};
    return VirtualQuery(address, &region, sizeof(region)) && region.State == MEM_COMMIT ? region.Protect : 0;
}

// A shadow is copied again only where the title wrote since the last submission (the shadow's pages are read-only
// between submissions, and a write fault marks them). Checked by what a GPU copy reads after each step, compared byte
// for byte with what the CPU wrote, over the whole buffer:
//   1. a pattern over everything committed, then a submission;
//   2. one 4 KiB page rewritten (plus the last byte of the range), then a submission: the new page arrives and the rest
//      keeps the first pattern;
//   3. nothing written: the same bytes again;
//   4. a page that was read-only after a submission is written by a second thread while the first waits, then a
//      submission.
// `committed` bytes from the start are committed; the rest reads as zeros. The protection after a submission is
// printed: PAGE_READONLY with the range's cache attribute when the writes are watched.
int check_written_pages(ID3D12Device* device, const char* kind, std::uint8_t* pages, UINT64 committed, UINT64 size) {
    const char* who = "vulkan slot-133 written pages";
    Ref<ID3D12Resource> buffer;
    buffer.p = place_shadow(device, pages, size);
    EXPECT(buffer.p != nullptr);
    GpuReader reader;
    if (reader.init(device, size)) return 1;
    std::vector<std::uint8_t> want(size, 0), got;
    for (UINT64 i = 0; i < committed; ++i) want[i] = pages[i] = pattern_byte(i, 11);
    EXPECT(reader.read(buffer.p, got) == 0);
    EXPECT(first_difference(got, want) == -1);
    const DWORD watched = protection_of(pages);

    // Step 2: one page in the middle of the committed part, and the last committed byte.
    const UINT64 page = (committed / 2) & ~UINT64(0xfff);
    for (UINT64 i = page; i < page + 0x1000 && i < committed; ++i) want[i] = pages[i] = pattern_byte(i, 201);
    want[committed - 1] = pages[committed - 1] = 0xa5;
    EXPECT(reader.read(buffer.p, got) == 0);
    long long wrong = first_difference(got, want);
    if (wrong >= 0) std::fprintf(stderr, "%s: %s: one page rewritten: first wrong byte at %lld\n", who, kind, wrong);
    EXPECT(wrong == -1);

    // Step 3: nothing written since.
    EXPECT(reader.read(buffer.p, got) == 0);
    EXPECT(first_difference(got, want) == -1);

    // Step 4: a write from another thread, to the first page.
    HANDLE writer = CreateThread(nullptr, 0, [](void* p) -> DWORD {
        auto* bytes = static_cast<std::uint8_t*>(p);
        for (int i = 0; i < 64; ++i) bytes[i] = std::uint8_t(0xc0 + i);
        return 0;
    }, pages, 0, nullptr);
    EXPECT(writer != nullptr && WaitForSingleObject(writer, 10000) == WAIT_OBJECT_0);
    CloseHandle(writer);
    for (int i = 0; i < 64; ++i) want[i] = std::uint8_t(0xc0 + i);
    EXPECT(reader.read(buffer.p, got) == 0);
    EXPECT(first_difference(got, want) == -1);
    const DWORD after = protection_of(pages);
    buffer.p->Release();
    buffer.p = nullptr;
    // Released: the pages are the title's again, read-write, and a write doesn't fault.
    EXPECT(protection_of(pages) == 0 || (protection_of(pages) & 0xff) == PAGE_READWRITE);
    pages[0] ^= 1;
    std::printf("%s: %s, %llu bytes (%llu committed): a rewritten page, a write from another thread and an unchanged "
                "submission all read back right; protection between submissions 0x%lx, 0x%lx\n",
                who, kind, static_cast<unsigned long long>(size), static_cast<unsigned long long>(committed), watched,
                after);
    return 0;
}

// Pages that change under a live shadow: committed later, decommitted, written after a submission. A reserved range
// whose first half is committed; then a page of the second half is committed and written, and the first 64 KiB is
// decommitted (it reads as zeros again).
int check_commit_changes(ID3D12Device* device) {
    const char* who = "vulkan slot-133 written pages";
    constexpr SIZE_T size = 4 * 0x10000;
    auto* pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE));
    EXPECT(pages != nullptr && VirtualAlloc(pages, size / 2, MEM_COMMIT, PAGE_READWRITE) == pages);
    int failures = 0;
    {
        Ref<ID3D12Resource> buffer;
        buffer.p = place_shadow(device, pages, size);
        EXPECT(buffer.p != nullptr);
        GpuReader reader;
        if (reader.init(device, size)) return 1;
        std::vector<std::uint8_t> want(size, 0), got;
        for (UINT64 i = 0; i < size / 2; ++i) want[i] = pages[i] = pattern_byte(i, 3);
        EXPECT(reader.read(buffer.p, got) == 0);
        EXPECT(first_difference(got, want) == -1);
        // Committed after the shadow was made, then written.
        std::uint8_t* later = pages + 3 * 0x10000;
        EXPECT(VirtualAlloc(later, 0x1000, MEM_COMMIT, PAGE_READWRITE) == later);
        for (UINT64 i = 0; i < 0x1000; ++i) want[3 * 0x10000 + i] = later[i] = pattern_byte(i, 77);
        EXPECT(reader.read(buffer.p, got) == 0);
        long long wrong = first_difference(got, want);
        if (wrong >= 0) std::fprintf(stderr, "%s: page committed later: first wrong byte at %lld\n", who, wrong);
        EXPECT(wrong == -1);
        // Decommitted: zeros, and the rest unchanged.
        EXPECT(VirtualFree(pages, 0x10000, MEM_DECOMMIT));
        std::fill(want.begin(), want.begin() + 0x10000, std::uint8_t(0));
        EXPECT(reader.read(buffer.p, got) == 0);
        wrong = first_difference(got, want);
        if (wrong >= 0) std::fprintf(stderr, "%s: decommitted page: first wrong byte at %lld\n", who, wrong);
        EXPECT(wrong == -1);
        // Committed again and written: it arrives.
        EXPECT(VirtualAlloc(pages, 0x1000, MEM_COMMIT, PAGE_READWRITE) == pages);
        for (UINT64 i = 0; i < 0x1000; ++i) want[i] = pages[i] = pattern_byte(i, 5);
        EXPECT(reader.read(buffer.p, got) == 0);
        EXPECT(first_difference(got, want) == -1);
    }
    std::printf("%s: a page committed after the shadow, a decommitted page and a recommitted one read back right\n",
                who);
    VirtualFree(pages, 0, MEM_RELEASE);
    return failures;
}

// The negative control, and the case it stands for. One pagefile-backed section (SEC_RESERVE, half committed, as xmem
// backs physical pages) is mapped at two addresses, A and B; the shadow is placed at A. A write through B doesn't fault
// at A, so without a mark the page it changed is not copied: the GPU still reads the old bytes. That shows the test
// sees a page the tracking missed. Once the page is reported aliased (what xmem does when it maps one physical page
// at two addresses: leonida::title_page_aliases), it is copied at every submission and the write arrives.
int check_aliased_write(ID3D12Device* device) {
    const char* who = "vulkan slot-133 aliased page";
    constexpr SIZE_T size = 4 * 0x10000;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, DWORD(size),
                                        nullptr);
    EXPECT(section != nullptr);
    auto* a = static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size));
    auto* b = static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size));
    EXPECT(a != nullptr && b != nullptr && VirtualAlloc(a, size / 2, MEM_COMMIT, PAGE_READWRITE) == a);
    const UINT64 committed = size / 2;
    long long unmarked = -1, marked = -1;
    {
        Ref<ID3D12Resource> buffer;
        buffer.p = place_shadow(device, a, size);
        EXPECT(buffer.p != nullptr);
        GpuReader reader;
        if (reader.init(device, size)) return 1;
        std::vector<std::uint8_t> want(size, 0), got;
        for (UINT64 i = 0; i < committed; ++i) want[i] = a[i] = pattern_byte(i, 19);
        EXPECT(reader.read(buffer.p, got) == 0);
        EXPECT(first_difference(got, want) == -1);
        // Through B, the page at offset 0x11000 (in A's second 64 KiB).
        constexpr UINT64 at = 0x11000;
        for (UINT64 i = at; i < at + 0x1000; ++i) want[i] = b[i] = pattern_byte(i, 133);
        EXPECT(std::memcmp(a + at, want.data() + at, 0x1000) == 0);  // the same memory
        EXPECT(reader.read(buffer.p, got) == 0);
        unmarked = first_difference(got, want);
        // Reported aliased: copied from now on.
        leonida::title_page_aliases::set(reinterpret_cast<std::uintptr_t>(a + at), true);
        EXPECT(reader.read(buffer.p, got) == 0);
        marked = first_difference(got, want);
        // And a later write through B, while it stays aliased.
        for (UINT64 i = at; i < at + 64; ++i) want[i] = b[i] = 0x3c;
        EXPECT(reader.read(buffer.p, got) == 0);
        EXPECT(first_difference(got, want) == -1);
        leonida::title_page_aliases::set(reinterpret_cast<std::uintptr_t>(a + at), false);
    }
    UnmapViewOfFile(b);
    UnmapViewOfFile(a);
    CloseHandle(section);
    std::printf("%s: a write through another view of the page, not marked: first stale byte at %lld (expected: in "
                "the page, 69632); marked aliased: %s\n",
                who, unmarked, marked == -1 ? "arrives" : "STILL STALE");
    EXPECT(unmarked == 0x11000);
    EXPECT(marked == -1);
    return 0;
}

// Kernel writes into a watched page. ReadFile writes from kernel mode, so a page a shadow made read-only fails it
// with ERROR_NOACCESS instead of faulting into the layer: the plain call fails (the control). Through the file shims'
// path (leonida::kernel_writes::read_file, what the title's ReadFile import resolves to), the destination is pinned
// and opened first: the read succeeds and its bytes reach the GPU at the next submission, synchronously and with an
// overlapped read on an asynchronous handle.
int check_kernel_write(ID3D12Device* device) {
    const char* who = "vulkan slot-133 kernel write";
    constexpr SIZE_T size = 4 * 0x10000;
    constexpr DWORD chunk = 0x2000;
    wchar_t directory[MAX_PATH]{}, path[MAX_PATH]{};
    EXPECT(GetTempPathW(MAX_PATH, directory) && GetTempFileNameW(directory, L"lsh", 0, path));
    std::vector<std::uint8_t> file_bytes(2 * chunk);
    for (size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = std::uint8_t(i * 29u + 1u);
    {
        const HANDLE out = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        DWORD wrote = 0;
        EXPECT(out != INVALID_HANDLE_VALUE);
        const BOOL ok = WriteFile(out, file_bytes.data(), DWORD(file_bytes.size()), &wrote, nullptr);
        CloseHandle(out);
        EXPECT(ok && wrote == file_bytes.size());
    }
    auto* pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE | PAGE_WRITECOMBINE));
    EXPECT(pages != nullptr && VirtualAlloc(pages, size / 2, MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE) == pages);
    DWORD plain_error = 0, watched = 0;
    bool plain_failed = false;
    {
        Ref<ID3D12Resource> buffer;
        buffer.p = place_shadow(device, pages, size);
        EXPECT(buffer.p != nullptr);
        GpuReader reader;
        if (reader.init(device, size)) return 1;
        std::vector<std::uint8_t> want(size, 0), got;
        for (UINT64 i = 0; i < size / 2; ++i) want[i] = pages[i] = pattern_byte(i, 41);
        EXPECT(reader.read(buffer.p, got) == 0);
        EXPECT(first_difference(got, want) == -1);
        watched = protection_of(pages + 0x4000);

        // The control: a plain ReadFile into a page made read-only by the last submission.
        HANDLE in = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        EXPECT(in != INVALID_HANDLE_VALUE);
        DWORD read = 0;
        plain_failed = !::ReadFile(in, pages + 0x4000, chunk, &read, nullptr);
        plain_error = plain_failed ? GetLastError() : 0;
        if (!plain_failed) std::memcpy(want.data() + 0x4000, file_bytes.data(), chunk);  // not watched after all

        // Through the shims' path: the same read succeeds.
        SetFilePointer(in, 0, nullptr, FILE_BEGIN);
        read = 0;
        const BOOL ok = leonida::kernel_writes::read_file(in, pages + 0x4000, chunk, &read, nullptr);
        CloseHandle(in);
        EXPECT(ok && read == chunk);
        std::memcpy(want.data() + 0x4000, file_bytes.data(), chunk);
        EXPECT(reader.read(buffer.p, got) == 0);
        long long wrong = first_difference(got, want);
        if (wrong >= 0) std::fprintf(stderr, "%s: synchronous read: first wrong byte at %lld\n", who, wrong);
        EXPECT(wrong == -1);

        // Overlapped, on an asynchronous handle, into the next page (read-only again after that submission).
        in = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        EXPECT(in != INVALID_HANDLE_VALUE);
        OVERLAPPED overlapped{};
        overlapped.Offset = chunk;
        overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        BOOL queued = leonida::kernel_writes::read_file(in, pages + 0x8000, chunk, nullptr, &overlapped);
        const bool pending = !queued && GetLastError() == ERROR_IO_PENDING;
        EXPECT(queued || pending);
        // A submission while it may still be in flight leaves the pinned page writable.
        EXPECT(reader.read(buffer.p, got) == 0);
        read = 0;
        EXPECT(GetOverlappedResult(in, &overlapped, &read, TRUE) && read == chunk);
        CloseHandle(overlapped.hEvent);
        CloseHandle(in);
        std::memcpy(want.data() + 0x8000, file_bytes.data() + chunk, chunk);
        EXPECT(reader.read(buffer.p, got) == 0);
        wrong = first_difference(got, want);
        if (wrong >= 0) std::fprintf(stderr, "%s: overlapped read: first wrong byte at %lld\n", who, wrong);
        EXPECT(wrong == -1);
        // Its pin is gone once the read completed: nothing is left pinned.
        EXPECT(leonida::kernel_writes::state().count.load() == 0);
    }
    VirtualFree(pages, 0, MEM_RELEASE);
    DeleteFileW(path);
    std::printf("%s: plain ReadFile into a watched page (protection 0x%lx): %s (error %lu); through the file shims: "
                "synchronous and overlapped reads succeed and reach the GPU\n",
                who, watched, plain_failed ? "fails" : "succeeds", plain_error);
    // The control must show the failure the shims prevent, wherever the page was watched.
    if ((watched & 0xff) == PAGE_READONLY) EXPECT(plain_failed && plain_error == ERROR_NOACCESS);
    return 0;
}

int check_written_pages_all(ID3D12Device* device) {
    int failures = 0;
    constexpr SIZE_T size = 4 * 0x10000;
    for (DWORD modifier : {DWORD(0), DWORD(PAGE_WRITECOMBINE)}) {
        // Half committed, so the import is refused whatever the cache attribute and the shadow answers.
        auto* pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE | modifier));
        if (!pages || VirtualAlloc(pages, size / 2, MEM_COMMIT, PAGE_READWRITE | modifier) != pages) return failures + 1;
        failures += check_written_pages(device, modifier ? "half-committed write-combined range"
                                                         : "half-committed range",
                                        pages, size / 2, size);
        VirtualFree(pages, 0, MEM_RELEASE);
        // Wholly committed and write-combined (not imported), and placed 64 bytes into its first page.
        if (modifier) {
            pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT,
                                                            PAGE_READWRITE | modifier));
            if (!pages) return failures + 1;
            failures += check_written_pages(device, "write-combined range", pages, size, size);
            failures += check_written_pages(device, "write-combined range, 64 bytes in", pages + 64, size - 64,
                                            size - 64);
            VirtualFree(pages, 0, MEM_RELEASE);
        }
    }
    failures += check_commit_changes(device);
    failures += check_aliased_write(device);
    failures += check_kernel_write(device);
    return failures;
}

void print_cases(const std::vector<Case>& cases, const char* who) {
    for (const Case& c : cases)
        std::printf("%s: %-44s hr=0x%08lX size=%llu align=%llu type=%d cpu=%d pool=%d flags=0x%x\n", who, c.name,
                    static_cast<unsigned long>(c.hr), static_cast<unsigned long long>(c.desc.SizeInBytes),
                    static_cast<unsigned long long>(c.desc.Alignment), int(c.desc.Properties.Type),
                    int(c.desc.Properties.CPUPageProperty), int(c.desc.Properties.MemoryPoolPreference),
                    unsigned(c.desc.Flags));
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.host_pointer_alignment = vk.host_pointer_alignment;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        const std::vector<Case> vulkan_cases = open_cases(ours.p);
        print_cases(vulkan_cases, "vulkan");
        failures += check_import(ours.p, vk.host_pointer_alignment != 0);
        if (vk.host_pointer_alignment) failures += check_title_placed_all(ours.p);
        failures += check_written_pages_all(ours.p);
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
                const std::string who = "host " + adapter_name;
                const std::vector<Case> host_cases = open_cases(host.p);
                print_cases(host_cases, who.c_str());
                if (vk.host_pointer_alignment) failures += compare_cases(vulkan_cases, host_cases, who.c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
