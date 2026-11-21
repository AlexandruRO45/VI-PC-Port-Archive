#include "ffx_recurrent.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
// The Xbox descriptor has an extra word after Flags and a 64-bit tail.
struct XboxTexture {
    uint32_t dimension, reserved;
    uint64_t alignment, width;
    uint32_t height;
    uint16_t layers, mips;
    uint32_t format, samples, quality, layout, flags, extension;
    uint64_t tail;
};
static_assert(sizeof(XboxTexture) == 0x40 && offsetof(XboxTexture, layout) == 0x2c);
constexpr auto texture_flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
constexpr auto initial_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

template<typename T> T read_at(const void* ptr, size_t offset = 0)
{
    T value;
    std::memcpy(&value, static_cast<const uint8_t*>(ptr) + offset, sizeof value);
    return value;
}
template<typename T> void write_at(void* ptr, size_t offset, T value)
{
    std::memcpy(static_cast<uint8_t*>(ptr) + offset, &value, sizeof value);
}
}

uint32_t RecurrentBinding::create(uintptr_t self, uintptr_t heap, uintptr_t flags, uintptr_t description,
                                  const uint64_t* frame)
{
    const auto xbox = read_at<XboxTexture>(reinterpret_cast<void*>(description));
    auto props = read_at<D3D12_HEAP_PROPERTIES>(reinterpret_cast<void*>(heap));
    const auto iid = read_at<IID>(reinterpret_cast<void*>(frame[6]));
    if (self != reinterpret_cast<uintptr_t>(device) || resource || props.Type != D3D12_HEAP_TYPE_DEFAULT ||
        props.CPUPageProperty || props.MemoryPoolPreference || props.CreationNodeMask || props.VisibleNodeMask ||
        flags || frame[4] != initial_state || frame[5] || iid != __uuidof(ID3D12Resource) || !frame[7] ||
        xbox.layout != 0x101 || xbox.extension != 3 || xbox.tail != 0)
        throw std::runtime_error("Unexpected Xbox committed-resource arguments");
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = static_cast<D3D12_RESOURCE_DIMENSION>(xbox.dimension);
    desc.Alignment = xbox.alignment;
    desc.Width = xbox.width;
    desc.Height = xbox.height;
    desc.DepthOrArraySize = xbox.layers;
    desc.MipLevels = xbox.mips;
    desc.Format = static_cast<DXGI_FORMAT>(xbox.format);
    desc.SampleDesc = {xbox.samples, xbox.quality};
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(xbox.flags);
    props.CreationNodeMask = props.VisibleNodeMask = 1;
    result = device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
        static_cast<D3D12_RESOURCE_STATES>(frame[4]), nullptr, iid, reinterpret_cast<void**>(resource.GetAddressOf()));
    write_at(reinterpret_cast<void*>(frame[7]), 0, resource.Get());
    std::printf("recurrent_create hr=0x%08x xbox_layout=0x%x initial_state=0x%x\n",
                unsigned(result), xbox.layout, unsigned(frame[4]));
    return static_cast<uint32_t>(result);
}

#ifndef FFX_PACKAGE_DEVICE_SLICE
// Source-owned control passes the same wire fields to the desktop binding.
void run_recurrent(RecurrentBinding& binding, RecurrentProvider& provider, const RecurrentConfig& config)
{
    auto* scratch = read_at<uint8_t*>(&provider, 0xc8);
    auto* slot = read_at<uint8_t*>(scratch, 0x1078) + provider.slot.index * 96;
    const auto index = read_at<uint32_t>(slot, 4);
    provider.resources[13] = index;
    write_at(slot, 4, index + 1);
    XboxTexture desc{3, 0, 0, config.output_width, config.output_height, 1, 1, 28, 1, 0, 0x101, 5, 3, 0};
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ID3D12Resource* handle = nullptr;
    uint64_t frame[8]{};
    frame[4] = initial_state;
    frame[6] = reinterpret_cast<uintptr_t>(&__uuidof(ID3D12Resource));
    frame[7] = reinterpret_cast<uintptr_t>(&handle);
    check(static_cast<HRESULT>(binding.create(reinterpret_cast<uintptr_t>(binding.device),
          reinterpret_cast<uintptr_t>(&heap), 0, reinterpret_cast<uintptr_t>(&desc), frame)), "Create control texture");
    auto* record = read_at<uint8_t*>(scratch, 0x1070) + 64 * index;
    write_at(record, 0, handle);
    write_at(record, 0x28, 2u);
    write_at(record, 0x2c, 2u);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    const auto self = reinterpret_cast<uintptr_t>(binding.device);
    const auto texture = reinterpret_cast<uintptr_t>(handle);
    const auto stride = binding.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto destination = [&](size_t offset, uint32_t slot_index) {
        return read_at<ID3D12DescriptorHeap*>(scratch, offset)->GetCPUDescriptorHandleForHeapStart().ptr + slot_index * stride;
    };
    binding.create_srv(self, texture, reinterpret_cast<uintptr_t>(&srv), destination(0x28, index));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = srv.Format;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto base = read_at<uint32_t>(slot, 0xc);
    write_at(record, 0x34, base);
    write_at(record, 0x38, 1u);
    for (auto offset : {0x38, 0x30})
        binding.create_uav(self, texture, 0, reinterpret_cast<uintptr_t>(&uav), destination(offset, base));
    write_at(slot, 0xc, base + 1);
}
#endif

void check_ffx_recurrent(ID3D12Device* device)
{
#ifdef FFX_PACKAGE_DEVICE_SLICE
    std::puts("recurrent_source=translated-package");
#else
    std::puts("recurrent_source=desktop-control");
#endif
    for (const auto width : {1920u, 1600u}) {
        RecurrentConfig config{};
        config.render_width = 1280;
        config.render_height = 720;
        config.output_width = width;
        config.output_height = width == 1920 ? 1080 : 900;
        FfxDeviceProbe probe(device);
        run_device_slice(probe);
        RecurrentProvider provider{};
        provider.slot.index = UINT32_MAX;
        provider.resources.fill(UINT32_MAX);
        auto* scratch = probe.scratch.data();
        write_at(&provider, 0xc8, scratch);
        const uint32_t sizes[] = {0, 0, 0, 1};
        if (run_backend_slots(scratch, provider.slot, sizes) || provider.slot.index != 0)
            throw std::runtime_error("Missing recurrent backend context");
        auto* slot = read_at<uint8_t*>(scratch, 0x1078);
        auto* records = read_at<uint8_t*>(scratch, 0x1070);
        if (records != scratch + 0xbf100 || read_at<uint32_t>(slot, 4) != 1)
            throw std::runtime_error("Unexpected initial resource table or cursor");
        RecurrentBinding binding{probe};
        run_recurrent(binding, provider, config);
        auto wanted_indices = std::array<uint32_t, 25>{};
        wanted_indices.fill(UINT32_MAX);
        wanted_indices[13] = 1;
        if (binding.result != S_OK || !binding.resource || provider.resources != wanted_indices ||
            read_at<uint32_t>(slot, 4) != 2 || read_at<ID3D12Resource*>(records, 64) != binding.resource.Get() ||
            read_at<uint32_t>(records, 64+0x28) != 2 || read_at<uint32_t>(records, 64+0x2c) != 2)
            throw std::runtime_error("Recurrent resource index, state or handle mismatch");
        const auto desc = binding.resource->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width ||
            desc.Height != config.output_height || desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
            desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM || desc.Flags != texture_flags ||
            desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN ||
            (desc.Alignment && desc.Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT))
            throw std::runtime_error("Recurrent GetDesc mismatch");
        if (read_at<uint32_t>(records, 64+0x34) != 1 || read_at<uint32_t>(records, 64+0x38) != 1 ||
            read_at<uint32_t>(slot, 0xc) != 2)
            throw std::runtime_error("Recurrent UAV base, count or next slot differs");
        check_recurrent_views(binding);
        std::printf("recurrent_desc=%llux%u format=%u flags=0x%x alignment=%llu context=%u id=13 index=%u next=%u barriers=passed\n",
                    desc.Width, desc.Height, unsigned(desc.Format), unsigned(desc.Flags), desc.Alignment,
                    provider.slot.index, provider.resources[13], read_at<uint32_t>(slot, 4));
    }
}
