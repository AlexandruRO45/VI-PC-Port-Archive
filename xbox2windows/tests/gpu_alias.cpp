// A title graphics allocation (era_heap, memory type 1) opened as a D3D12 heap with
// era_gpu_alias, addressed through era_title_va. A compute shader reads and writes the
// allocation at translated GPU addresses; the CPU then checks the results through the title's
// own pointer, with no copy or readback buffer in between.
#include "bvh_fixture.h"
#include "era_gpu_alias.h"
#include "era_heap.h"
#include "era_title_va.h"
#include "AliasTransform.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

using namespace bvh_fixture;

namespace {
// XALLOC_ATTRIBUTES: memoryType 1 (graphics) at bit 29, pageSize 1 (64 KB) at bit 14,
// alignment 16 (64 KB) at bit 24.
constexpr std::uint64_t graphics_64k = (1ull << 29) | (1ull << 14) | (16ull << 24);
constexpr std::size_t allocation_bytes = 256 * 1024;
constexpr std::size_t source_offset = 0x1000;
constexpr std::size_t destination_offset = 0x20000;
constexpr Word count = 4096;
constexpr Word guard_words = 64;
constexpr Word sentinel = 0xdeadbeef;

struct TitleAllocation {
    void* base = era_heap::allocate(allocation_bytes, graphics_64k);
    ~TitleAllocation() { era_heap::release_with_attributes(base, graphics_64k); }
};

Word source_word(Word i) { return (i * 2654435761u) ^ 0x13579bdfu; }
Word expected_word(Word i) { return (source_word(i) ^ 0xa5a5a5a5u) + i; }
}

bool check_gpu_alias(ID3D12Device* device)
{
    TitleAllocation title;
    if (!title.base) throw std::runtime_error("era_heap graphics allocation failed");
    auto* base = static_cast<unsigned char*>(title.base);

    era_gpu_alias::Alias alias;
    const HRESULT hr = era_gpu_alias::open(device, base, allocation_bytes, alias);
    if (hr == E_NOTIMPL) {
        std::puts("existing heaps are not supported by this adapter");
        return false;
    }
    check(hr, "Open the title allocation as a D3D12 heap");

    // Failure case: a pointer inside the allocation is not the start of an allocation. open must
    // reject it before calling D3D12, so the debug layer reports no error for this control.
    era_gpu_alias::Alias interior;
    const HRESULT interior_hr = era_gpu_alias::open(device, base + 0x1000, allocation_bytes - 0x1000, interior);
    if (interior_hr != E_INVALIDARG || interior.heap || interior.buffer)
        throw std::runtime_error("an interior pointer was not rejected with E_INVALIDARG");

    era_title_va::Table table;
    const auto title_begin = reinterpret_cast<std::uint64_t>(base);
    const D3D12_GPU_VIRTUAL_ADDRESS gpu_begin = alias.buffer->GetGPUVirtualAddress();
    if (!table.add(title_begin, allocation_bytes, gpu_begin)) throw std::runtime_error("table rejected the alias");
    const std::uint64_t source = table.translate(title_begin + source_offset, count * sizeof(Word));
    const std::uint64_t destination = table.translate(title_begin + destination_offset, count * sizeof(Word));
    if (source != gpu_begin + source_offset || destination != gpu_begin + destination_offset)
        throw std::runtime_error("title address translation differs from the alias offset");

    auto* source_words = reinterpret_cast<Word*>(base + source_offset);
    auto* destination_words = reinterpret_cast<Word*>(base + destination_offset);
    for (Word i = 0; i < count; ++i) {
        source_words[i] = source_word(i);
        destination_words[i] = sentinel;
    }
    for (Word i = count; i < count + guard_words; ++i) destination_words[i] = sentinel;

    Kernel kernel(device, {AliasTransform, sizeof(AliasTransform)});
    ComputeCommands cmd(device, kernel.pipeline.Get());
    auto* list = cmd.list.Get();
    transition(list, alias.buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(kernel.root.Get());
    list->SetComputeRootUnorderedAccessView(0, source);
    list->SetComputeRootUnorderedAccessView(1, destination);
    list->SetComputeRoot32BitConstant(2, count, 0);
    list->Dispatch(count / 64, 1, 1);
    transition(list, alias.buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    cmd.submit_and_wait(device);

    for (Word i = 0; i < count; ++i) {
        equal(destination_words[i], expected_word(i), i, 0);
        equal(source_words[i], source_word(i), i, 1);
    }
    for (Word i = count; i < count + guard_words; ++i) equal(destination_words[i], sentinel, i, 2);
    std::printf("gpu_alias bytes=%zu words=%u source=+0x%zx destination=+0x%zx guard_words=%u "
                "interior_open=0x%08lx mismatches=0\n",
                allocation_bytes, count, source_offset, destination_offset, guard_words,
                static_cast<unsigned long>(interior_hr));
    return true;
}
