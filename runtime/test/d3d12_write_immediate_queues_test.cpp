// CheckFeatureSupport(OPTIONS3) names the list types WriteBufferImmediate works on (WriteBufferImmediateSupportFlags;
// WARP answers DIRECT | BUNDLE | COMPUTE | COPY). Each advertised type is held to it here: a list of that type writes
// in all three modes on a queue of its type, and a bundle's writes land through the DIRECT list that executes it.
// Each of the three lists also copies a word and then overwrites it with a MARKER_OUT write, which runs after the
// copy. The values are read back from a readback buffer the writes target directly. No game files; exits 77 without
// a Vulkan device.
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

bool wait_idle(ID3D12Device* device, ID3D12CommandQueue* queue) {
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) return false;
    return SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr));
}

// One WriteBufferImmediate call with the three modes, at `slot` * 16 bytes into the buffer.
void write_three(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS base, std::uint32_t slot) {
    Ref<ID3D12GraphicsCommandList2> list2;
    if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list2.p)))) return;
    const D3D12_GPU_VIRTUAL_ADDRESS at = base + slot * 16;
    const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER params[] = {
        {at, 0x57420000u | slot << 4 | 0}, {at + 4, 0x57420000u | slot << 4 | 1}, {at + 8, 0x57420000u | slot << 4 | 2}};
    const D3D12_WRITEBUFFERIMMEDIATE_MODE modes[] = {D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT,
                                                     D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN,
                                                     D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT};
    list2->WriteBufferImmediate(3, params, modes);
}

// The word a list copies at `slot` * 16 + 12 and then overwrites with MARKER_OUT.
constexpr std::uint32_t copied = 0x434f5059;  // "COPY"
std::uint32_t marker_after_copy(std::uint32_t slot) { return 0x4d4b0000u | slot; }

bool landed(const std::uint8_t* bytes, std::uint32_t slot) {
    for (std::uint32_t i = 0; i < 3; ++i) {
        std::uint32_t word = 0;
        std::memcpy(&word, bytes + slot * 16 + i * 4, sizeof word);
        if (word != (0x57420000u | slot << 4 | i)) return false;
    }
    return true;
}
}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) return 77;

    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
    const auto advertised = D3D12_COMMAND_LIST_SUPPORT_FLAG_DIRECT | D3D12_COMMAND_LIST_SUPPORT_FLAG_BUNDLE |
                            D3D12_COMMAND_LIST_SUPPORT_FLAG_COMPUTE | D3D12_COMMAND_LIST_SUPPORT_FLAG_COPY;
    check(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options, sizeof options)) &&
              options.WriteBufferImmediateSupportFlags == advertised,
          "OPTIONS3 advertises WriteBufferImmediate on DIRECT, BUNDLE, COMPUTE and COPY lists");

    // Which Vulkan families the COMPUTE and COPY queues below lease from (the type's own family when it has queues).
    if (const auto* info = rt::d3d12::vulkan_device_info(device.p))
        std::printf("Vulkan families: graphics %u, compute %u (%s), copy %u (%s)\n", info->queue_family,
                    info->compute_family, info->compute_queues.empty() ? "no queues, graphics used" : "own queues",
                    info->copy_family, info->copy_queues.empty() ? "no queues, graphics used" : "own queues");

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> readback, upload;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback.p))))
        return 1;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&upload.p))))
        return 1;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return 1;
    std::memset(mapped, 0, 256);
    readback->Unmap(0, nullptr);
    if (FAILED(upload->Map(0, nullptr, &mapped))) return 1;
    std::memcpy(mapped, &copied, sizeof copied);
    upload->Unmap(0, nullptr);
    const D3D12_GPU_VIRTUAL_ADDRESS base = readback->GetGPUVirtualAddress();
    check(base != 0, "the readback buffer has a GPU virtual address");

    // Slots 0, 1 and 2: a DIRECT, a COMPUTE and a COPY list, each on a queue of its own type.
    const D3D12_COMMAND_LIST_TYPE types[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             D3D12_COMMAND_LIST_TYPE_COPY};
    const char* names[] = {"DIRECT", "COMPUTE", "COPY"};
    Ref<ID3D12CommandQueue> queues[3];
    for (std::uint32_t t = 0; t < 3; ++t) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = types[t];
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queues[t].p))) ||
            FAILED(device->CreateCommandAllocator(types[t], IID_PPV_ARGS(&allocator.p))) ||
            FAILED(device->CreateCommandList(0, types[t], allocator.p, nullptr, IID_PPV_ARGS(&list.p)))) {
            std::fprintf(stderr, "could not create a %s queue and list\n", names[t]);
            return 1;
        }
        write_three(list.p, base, t);
        list->CopyBufferRegion(readback.p, t * 16 + 12, upload.p, 0, sizeof copied);
        {
            Ref<ID3D12GraphicsCommandList2> list2;
            if (SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list2.p)))) {
                const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER after{base + t * 16 + 12, marker_after_copy(t)};
                const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
                list2->WriteBufferImmediate(1, &after, &mode);
            }
        }
        char what[96];
        std::snprintf(what, sizeof what, "a %s list with WriteBufferImmediate closes", names[t]);
        check(list->Close() == S_OK, what);
        ID3D12CommandList* batch[] = {list.p};
        queues[t]->ExecuteCommandLists(1, batch);
        std::snprintf(what, sizeof what, "and runs on a %s queue", names[t]);
        check(rt::d3d12::last_execute_result(queues[t].p) == leonida::ReplayResult::success &&
                  wait_idle(device.p, queues[t].p),
              what);
    }

    // Slot 3: a bundle's writes, executed by a DIRECT list.
    Ref<ID3D12CommandAllocator> bundle_allocator, direct_allocator;
    Ref<ID3D12GraphicsCommandList> bundle, direct;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&bundle_allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p, nullptr,
                                         IID_PPV_ARGS(&bundle.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&direct_allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, direct_allocator.p, nullptr,
                                         IID_PPV_ARGS(&direct.p))))
        return 1;
    write_three(bundle.p, base, 3);
    check(bundle->Close() == S_OK, "a bundle with WriteBufferImmediate closes");
    direct->ExecuteBundle(bundle.p);
    check(direct->Close() == S_OK, "the DIRECT list that executes it closes");
    ID3D12CommandList* batch[] = {direct.p};
    queues[0]->ExecuteCommandLists(1, batch);
    check(rt::d3d12::last_execute_result(queues[0].p) == leonida::ReplayResult::success &&
              wait_idle(device.p, queues[0].p),
          "and runs on the DIRECT queue");

    const D3D12_RANGE range{0, 64};
    if (FAILED(readback->Map(0, &range, &mapped))) return 1;
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    check(landed(bytes, 0), "the DIRECT list's three writes landed");
    check(landed(bytes, 1), "the COMPUTE list's three writes landed");
    check(landed(bytes, 2), "the COPY list's three writes landed");
    check(landed(bytes, 3), "the bundle's three writes landed through the DIRECT list");
    for (std::uint32_t t = 0; t < 3; ++t) {
        std::uint32_t word = 0;
        std::memcpy(&word, bytes + t * 16 + 12, sizeof word);
        char what[96];
        std::snprintf(what, sizeof what, "the %s list's MARKER_OUT write landed after its copy to the same word",
                      names[t]);
        check(word == marker_after_copy(t), what);
    }
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);

    std::printf("d3d12_write_immediate_queues_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
