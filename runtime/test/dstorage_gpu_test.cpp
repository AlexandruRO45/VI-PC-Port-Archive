// DirectStorage writes physical pages, and Vulkan reads those same pages through slot 133's buffer backend.
// Synthetic data only. Covers file reads, zlib, two CPU aliases, and a remapped resource at the same address.
#include "vk_test_device.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../shims/dstorage_xs_abi.h"
#include "../shims/file_system.h"
#include "../shims/shims.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "dstorage_gpu line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)
#define CHECK_HR(call) do { \
    const HRESULT result = (call); \
    if (FAILED(result)) { \
        std::fprintf(stderr, "dstorage_gpu line %d: %s = 0x%08lx\n", __LINE__, #call, \
                     static_cast<unsigned long>(result)); \
        return 1; \
    } \
} while (0)

namespace {
using namespace dsx;
constexpr size_t kPage = 0x10000;
constexpr size_t kBytes = 4 * kPage;
using Pages = std::array<uint64_t, 4>;
using AllocatePages = BOOL(WINAPI*)(uint64_t, uint64_t*, uint64_t*);
using FreePages = BOOL(WINAPI*)(uint64_t, const uint64_t*);
using ReserveWindow = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using MapPages = void*(WINAPI*)(void*, uint64_t, const uint64_t*);

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() { if (p) p->Release(); p = nullptr; }
    T* operator->() const { return p; }
};

struct PhysicalMemory {
    Pages pages{};
    void* window = nullptr;
    void* alias = nullptr;
    MapPages map = reinterpret_cast<MapPages>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    FreePages free = reinterpret_cast<FreePages>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));

    ~PhysicalMemory() {
        for (void* address : {window, alias}) {
            if (address) {
                map(address, pages.size(), nullptr);
                VirtualFree(address, kBytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
                VirtualFree(address, 0, MEM_RELEASE);
            }
        }
        if (pages[0]) free(2, pages.data());
        if (pages[2]) free(2, pages.data() + 2);
    }

    int initialize() {
        const auto allocate = reinterpret_cast<AllocatePages>(
            rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
        const auto reserve = reinterpret_cast<ReserveWindow>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
        CHECK(allocate && reserve && map && free);
        uint64_t count = 2;
        CHECK(allocate(0x280, &count, pages.data()) && count == 2);
        count = 2;
        CHECK(allocate(0x280, &count, pages.data() + 2) && count == 2);
        constexpr DWORD physical_reserve = MEM_RESERVE | 0x20400000;
        window = reserve(nullptr, kBytes, physical_reserve, 1, PAGE_READWRITE);
        alias = reserve(nullptr, kBytes, physical_reserve, 1, PAGE_READWRITE);
        CHECK(window && alias);
        CHECK(map(window, pages.size(), pages.data()) == window);
        CHECK(map(alias, pages.size(), pages.data()) == alias);
        return 0;
    }
};

D3D12_RESOURCE_DESC buffer_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

// The readback is cleared after every comparison. A later successful comparison must come from a new GPU copy.
struct GpuReadback {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Resource> readback;
    Ref<ID3D12Fence> fence;
    HANDLE done = nullptr;
    UINT64 value = 0;
    ~GpuReadback() { if (done) CloseHandle(done); }

    int initialize(ID3D12Device* device) {
        D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        CHECK_HR(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p)));
        CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p)));
        CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                          IID_PPV_ARGS(&list.p)));
        CHECK_HR(list->Close());
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        const auto desc = buffer_desc();
        CHECK_HR(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                 IID_PPV_ARGS(&readback.p)));
        CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)));
        done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        CHECK(done);
        return 0;
    }

    int compare(ID3D12Resource* source, const std::vector<uint8_t>& expected, const char* stage) {
        CHECK_HR(allocator->Reset());
        CHECK_HR(list->Reset(allocator.p, nullptr));
        list->CopyBufferRegion(readback.p, 0, source, 0, kBytes);
        CHECK_HR(list->Close());
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK_HR(queue->Signal(fence.p, ++value));
        CHECK_HR(fence->SetEventOnCompletion(value, done));
        CHECK(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
        void* mapped = nullptr;
        const D3D12_RANGE range{0, kBytes};
        CHECK_HR(readback->Map(0, &range, &mapped));
        const auto* bytes = static_cast<const uint8_t*>(mapped);
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), bytes);
        const bool same = mismatch.first == expected.end();
        if (!same)
            std::fprintf(stderr, "%s: GPU byte %zu is %u, expected %u\n", stage,
                         static_cast<size_t>(mismatch.first - expected.begin()), unsigned(*mismatch.second),
                         unsigned(*mismatch.first));
        std::memset(mapped, 0, kBytes);
        readback->Unmap(0, nullptr);
        CHECK(same);
        std::printf("%s: %zu GPU bytes match\n", stage, expected.size());
        return 0;
    }
};

// Creating-call IDs are synthetic, as in dstorage_test. The console's IDs are not part of this test.
constexpr GUID kFactory = {0xd5700001, 1, 1, {1, 0, 0, 0, 0, 0, 0, 1}};
constexpr GUID kQueue = {0xd5700002, 1, 1, {1, 0, 0, 0, 0, 0, 0, 2}};
constexpr GUID kFile = {0xd5700003, 1, 1, {1, 0, 0, 0, 0, 0, 0, 3}};

int submit(IDStorageQueueX2* storage, const DSTORAGE_REQUEST& request, IDStorageStatusArrayX* status,
           ID3D12Fence* completion, UINT64 value, GpuReadback& gpu) {
    storage->EnqueueRequest(&request);
    storage->EnqueueStatus(status, 0);
    storage->EnqueueSignal(completion, value);
    // Queue the GPU's wait before DirectStorage starts. The worker signals this real Vulkan-backed fence.
    CHECK_HR(gpu.queue->Wait(completion, value));
    storage->Submit();
    return 0;
}

std::vector<uint8_t> reverse_pages(const std::vector<uint8_t>& bytes) {
    std::vector<uint8_t> reversed(kBytes);
    for (size_t page = 0; page < 4; ++page)
        std::copy_n(bytes.begin() + page * kPage, kPage, reversed.begin() + (3 - page) * kPage);
    return reversed;
}

int run(ID3D12Device* device) {
    PhysicalMemory memory;
    CHECK(memory.initialize() == 0);
    std::memset(memory.window, 0xcd, kBytes);
    std::vector<uint8_t> expected(kBytes, 0xcd);
    const auto desc = buffer_desc();
    Ref<ID3D12Resource> buffer, alias;
    CHECK_HR(rt::d3d12::create_buffer_at_host_address(device, memory.window, desc, IID_PPV_ARGS(&buffer.p)));
    CHECK_HR(rt::d3d12::create_buffer_at_host_address(device, memory.alias, desc, IID_PPV_ARGS(&alias.p)));

    testfiles::ScratchDir scratch("leonida_dstorage_gpu");
    rt::fs::configure(scratch.path(), scratch.path() / "overlay");
    std::vector<uint8_t> file_bytes(kPage + 64);
    for (size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<uint8_t>(i * 131 + 17);
    {
        std::ofstream file(scratch.path() / "input.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
        file.close();
        CHECK(file);
    }
    const auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(
        rt::shims::find("xcrdapi.dll", "DStorageGetFactory"));
    CHECK(get_factory);
    Ref<IDStorageFactoryX2> factory;
    CHECK_HR(get_factory(kFactory, reinterpret_cast<void**>(&factory.p)));
    Ref<IDStorageFileX> file;
    CHECK_HR(factory->OpenFile(L"GAME:\\input.bin", kFile, reinterpret_cast<void**>(&file.p)));
    DSTORAGE_QUEUE_DESC1 qdesc{};
    qdesc.Capacity = kMinQueueCapacity;
    qdesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    qdesc.Options.AutoSubmitDisabled = 1;
    Ref<IDStorageQueueX2> file_queue, memory_queue;
    CHECK_HR(factory->CreateQueue1(&qdesc, kQueue, reinterpret_cast<void**>(&file_queue.p)));
    qdesc.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    qdesc.Priority = DSTORAGE_PRIORITY_REALTIME;
    CHECK_HR(factory->CreateQueue1(&qdesc, kQueue, reinterpret_cast<void**>(&memory_queue.p)));
    Ref<IDStorageStatusArrayX> status;
    CHECK_HR(factory->CreateStatusArray(1, "GPU transfer", kStatusArrayIds[0], reinterpret_cast<void**>(&status.p)));
    Ref<ID3D12Fence> storage_completion;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&storage_completion.p)));
    GpuReadback gpu;
    CHECK(gpu.initialize(device) == 0);

    DSTORAGE_REQUEST request{};
    request.Options.DestinationIsPhysicalPages = 1;
    request.DestinationPageArray = reinterpret_cast<PULONG_PTR>(memory.pages.data());
    request.DestinationPageOffset = static_cast<UINT16>(kPage - 32);
    request.File = file.p;
    request.SourceSize = request.DestinationSize = static_cast<UINT32>(file_bytes.size());
    std::copy(file_bytes.begin(), file_bytes.end(), expected.begin() + request.DestinationPageOffset);
    CHECK(submit(file_queue.p, request, status.p, storage_completion.p, 1, gpu) == 0);
    CHECK(gpu.compare(buffer.p, expected, "file across three physical pages") == 0);
    CHECK_HR(status->GetHResult(0));
    CHECK(gpu.compare(alias.p, expected, "file through the second GPU alias") == 0);

    // zlib.compress(bytes([0xa6]) * 8192). The plain bytes are built independently below.
    const uint8_t compressed[] = {
        0x78, 0x9c, 0xed, 0xc1, 0x01, 0x0d, 0x00, 0x00, 0x00, 0xc2, 0xa0, 0xfe, 0x45, 0x9e, 0xd3, 0x1c,
        0x6e, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x9d, 0x1b, 0xf5, 0xb3, 0xc1, 0x2d};
    request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    request.Options.DestinationIsPhysicalPages = 1;
    request.Options.ZlibDecompress = 1;
    request.DestinationPageArray = reinterpret_cast<PULONG_PTR>(memory.pages.data() + 1);
    request.DestinationPageOffset = 0xf000;
    request.DestinationSize = 8192;
    request.Source = compressed;
    request.SourceSize = sizeof compressed;
    std::fill_n(expected.begin() + kPage + request.DestinationPageOffset, request.DestinationSize, 0xa6);
    CHECK(submit(memory_queue.p, request, status.p, storage_completion.p, 2, gpu) == 0);
    CHECK(gpu.compare(buffer.p, expected, "zlib update after resource creation") == 0);
    CHECK_HR(status->GetHResult(0));
    CHECK(gpu.compare(alias.p, expected, "zlib through the second GPU alias") == 0);

    // Release the imported resource before replacing its CPU mappings. The other alias stays mapped and imported.
    buffer.reset();
    Pages remapped = memory.pages;
    std::reverse(remapped.begin(), remapped.end());
    CHECK(memory.map(memory.window, remapped.size(), remapped.data()) == memory.window);
    expected = reverse_pages(expected);
    CHECK_HR(rt::d3d12::create_buffer_at_host_address(device, memory.window, desc, IID_PPV_ARGS(&buffer.p)));
    for (auto& byte : file_bytes) byte ^= 0x5a;
    request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_MEMORY;
    request.Options.DestinationIsPhysicalPages = 1;
    request.DestinationPageArray = reinterpret_cast<PULONG_PTR>(remapped.data());
    request.DestinationPageOffset = 0xfe00;
    request.DestinationSize = request.SourceSize = static_cast<UINT32>(file_bytes.size());
    request.Source = file_bytes.data();
    std::copy(file_bytes.begin(), file_bytes.end(), expected.begin() + request.DestinationPageOffset);
    CHECK(submit(memory_queue.p, request, status.p, storage_completion.p, 3, gpu) == 0);
    CHECK(gpu.compare(buffer.p, expected, "updated pages remapped at the same VA") == 0);
    CHECK_HR(status->GetHResult(0));
    CHECK(gpu.compare(alias.p, reverse_pages(expected), "original GPU alias after remapping") == 0);
    file_queue->Close();
    memory_queue->Close();
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.host_pointer_alignment || kPage % vk.host_pointer_alignment) {
        std::printf("SKIP: 64 KiB host-memory imports unavailable\n");
        return 77;
    }
    int result = 1;
    {
        Ref<ID3D12Device> device;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.host_pointer_alignment = vk.host_pointer_alignment;
        CHECK_HR(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)));
        result = run(device.p);
    }
    CHECK(vk.validation_errors() == 0);
    if (result == 0) std::puts("dstorage_gpu: PASS (file, zlib, GPU aliases, remapped pages)");
    return result;
}
