#include "ffx_device_init.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
struct ExpectedHeap {
    UINT offset;
    D3D12_DESCRIPTOR_HEAP_TYPE type;
    UINT count;
    D3D12_DESCRIPTOR_HEAP_FLAGS flags;
};
constexpr auto srv = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
constexpr auto visible = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
constexpr auto cpu = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
constexpr ExpectedHeap expected[] = {
    {0x28, srv, 512, cpu}, {0x30, srv, 512, cpu}, {0x38, srv, 512, visible},
    {0x48, srv, 167936, visible}, {0x20, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8, cpu}
};
}

FfxDeviceProbe::FfxDeviceProbe(ID3D12Device* input) : device(input), scratch(0x18f160)
{
    const uint32_t contexts = 1;
    std::memcpy(scratch.data() + 4, &contexts, sizeof contexts);
}

uint32_t FfxDeviceProbe::retain(uintptr_t self)
{
    if (self != reinterpret_cast<uintptr_t>(device) || retained || created)
        throw std::runtime_error("Unexpected device or repeated AddRef");
    const auto result = device->AddRef();
    retained.Attach(device);
    std::puts("device AddRef");
    return result;
}

uint32_t FfxDeviceProbe::create(uintptr_t self, uintptr_t description, uintptr_t iid, uintptr_t output)
{
    if (!retained || self != reinterpret_cast<uintptr_t>(device) || created >= heaps.size())
        throw std::runtime_error("Unexpected device or heap call order");
    const auto& want = expected[created];
    auto* slot = scratch.data() + want.offset;
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    GUID interface_id{};
    std::memcpy(&desc, reinterpret_cast<const void*>(description), sizeof desc);
    std::memcpy(&interface_id, reinterpret_cast<const void*>(iid), sizeof interface_id);
    if (output != reinterpret_cast<uintptr_t>(slot) || interface_id != __uuidof(ID3D12DescriptorHeap) ||
        desc.Type != want.type || desc.NumDescriptors != want.count || desc.Flags != want.flags || desc.NodeMask != 0)
        throw std::runtime_error("Unexpected CreateDescriptorHeap parameters");
    auto& heap = heaps[created];
    const auto hr = device->CreateDescriptorHeap(&desc, interface_id, reinterpret_cast<void**>(heap.GetAddressOf()));
    std::printf("heap=%zu slot=0x%x type=%u count=%u flags=%u node_mask=%u hr=0x%08x\n",
                created + 1, want.offset, unsigned(desc.Type), desc.NumDescriptors, unsigned(desc.Flags),
                desc.NodeMask, unsigned(hr));
    check(hr, "CreateDescriptorHeap in upscaler initialization");
    if (!heap) throw std::runtime_error("CreateDescriptorHeap returned an empty handle");
    const auto readback = heap->GetDesc();
    if (readback.Type != desc.Type || readback.NumDescriptors != desc.NumDescriptors ||
        readback.Flags != desc.Flags || (readback.NodeMask != 0 && readback.NodeMask != 1))
        throw std::runtime_error("Heap description readback mismatch");
    for (size_t i = 0; i < created; ++i)
        if (heap.Get() == heaps[i].Get()) throw std::runtime_error("Repeated live heap");
    const auto result = heap.Get();
    std::memcpy(slot, &result, sizeof result);
    ++created;
    return static_cast<uint32_t>(hr);
}

#ifndef FFX_PACKAGE_DEVICE_SLICE
// Public control for machines without the package. Use the same typed D3D12 bindings.
void run_device_slice(FfxDeviceProbe& probe)
{
    auto device = reinterpret_cast<uintptr_t>(probe.device);
    probe.retain(device);
    std::memcpy(probe.scratch.data() + 8, &device, sizeof device);
    for (const auto& heap : expected) {
        D3D12_DESCRIPTOR_HEAP_DESC desc{heap.type, heap.count, heap.flags, 0};
        probe.create(device, reinterpret_cast<uintptr_t>(&desc),
                     reinterpret_cast<uintptr_t>(&__uuidof(ID3D12DescriptorHeap)),
                     reinterpret_cast<uintptr_t>(probe.scratch.data() + heap.offset));
    }
    // The last 96 bytes of the single-context scratch hold its slot record.
    auto* table = probe.scratch.data() + probe.scratch.size() - 96;
    std::memcpy(probe.scratch.data() + 0x1078, &table, sizeof table);
    auto* records = probe.scratch.data() + 0xbf100;
    std::memcpy(probe.scratch.data() + 0x1070, &records, sizeof records);
}
#endif

void check_ffx_device_init(ID3D12Device* device)
{
#ifdef FFX_PACKAGE_DEVICE_SLICE
    std::puts("source=translated-package");
#else
    std::puts("source=desktop-control");
#endif
    FfxDeviceProbe probe(device);
    run_device_slice(probe);
    ID3D12Device* stored{};
    std::memcpy(&stored, probe.scratch.data() + 8, sizeof stored);
    if (probe.created != probe.heaps.size() || stored != device)
        throw std::runtime_error("Incomplete upscaler device initialization slice");
    for (size_t i = 0; i < probe.created; ++i) {
        ID3D12DescriptorHeap* handle{};
        std::memcpy(&handle, probe.scratch.data() + expected[i].offset, sizeof handle);
        if (handle != probe.heaps[i].Get()) throw std::runtime_error("Heap output slot changed");
    }
    check_backend_slots(probe);
}
