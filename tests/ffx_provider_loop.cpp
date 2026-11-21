#include "ffx_provider_loop.h"
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace
{
template <class T> T read(const void* base, size_t offset)
{
    T value;
    std::memcpy(&value, static_cast<const uint8_t*>(base) + offset, sizeof value);
    return value;
}
template <class T> void put(void* base, size_t offset, T value)
{
    std::memcpy(static_cast<uint8_t*>(base) + offset, &value, sizeof value);
}
struct Expected {
    uint32_t id, index, type, format, width, height, usage, state, mode;
    const wchar_t* name;
};
constexpr Expected expected[] = {{13, 1, 2, 10, 1920, 1080, 3, 2, 1, L"FSR4UPSCALER_Recurrent"},
                                 {14, 2, 2, 4, 1920, 1080, 3, 2, 1, L"FSR4UPSCALER_History"},
                                 {15, 3, 2, 4, 1920, 1080, 3, 2, 1, L"FSR4UPSCALER_HistoryReprojected"},
                                 {16, 4, 2, 29, 1920, 1080, 2, 2, 1, L"FSR4UPSCALER_RcasIntermediary"},
                                 {5, 5, 0, 37, 8, 0, 2, 2, 2, L"FSR4UPSCALER_AutoExposure"},
                                 {17, 7, 2, 6, 1, 1, 0, 4, 2, L"FSR4UPSCALER_DefaultExposure"},
                                 {6, 9, 2, 8, 1, 1, 2, 2, 3, L"FSR4UPSCALER_SpdAtomicCounter"},
                                 {7, 11, 2, 28, 80, 45, 3, 2, 1, L"FSR4UPSCALER_Luma_Mip_5"},
                                 {22, 12, 0, 37, 0x1412200, 0, 2, 2, 1, L"FSR4UPSCALER_ScratchBuffer"},
                                 {20, 13, 0, 37, 0x20000, 0, 0, 4, 1, L"FSR4UPSCALER_InitializerBuffer"},
                                 {21, 14, 0, 37, 0x20000, 0, 0, 4, 1, L"FSR4UPSCALER_InitializerBuffer_Upload"},
                                 {23, 15, 2, 23, 1920, 1080, 3, 2, 1, L"FSR4UPSCALER_DebugInformation"}};

DXGI_FORMAT format_for(uint32_t code)
{
    switch (code) {
    case 4:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case 6:
        return DXGI_FORMAT_R32G32_FLOAT;
    case 8:
        return DXGI_FORMAT_R32_UINT;
    case 10:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case 23:
        return DXGI_FORMAT_R16_UNORM;
    case 28:
        return DXGI_FORMAT_R32_FLOAT;
    case 29:
        return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case 37:
        return DXGI_FORMAT_UNKNOWN; // Raw buffer resource.
    default:
        throw std::runtime_error("Unexpected FFX resource format");
    }
}

void create_object(LoopContext& ctx, const LoopDesc& wire, uint32_t index, const D3D12_RESOURCE_DESC& desc,
                   D3D12_RESOURCE_STATES initial)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = wire.heap ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    auto& object = ctx.objects.at(index);
    if (object)
        throw std::runtime_error("Repeated resource index");
    check(ctx.probe.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr,
                                                    IID_PPV_ARGS(&object)),
          "Create provider loop resource");
    check(object->SetName(wire.name), "Name provider loop resource");
    auto actual = object->GetDesc();
    if (actual.Width != desc.Width || actual.Height != desc.Height || actual.Format != desc.Format ||
        actual.Dimension != desc.Dimension || actual.Flags != desc.Flags || actual.MipLevels != desc.MipLevels)
        throw std::runtime_error("Provider D3D12 description mismatch");
    auto* records = read<uint8_t*>(ctx.probe.scratch.data(), 0x1070);
    put(records, index * 64, object.Get());
}
D3D12_RESOURCE_DESC buffer_desc(uint64_t size)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}
uint8_t* slot_of(LoopContext& ctx)
{
    return read<uint8_t*>(ctx.probe.scratch.data(), 0x1078) + 96 * ctx.state.slot.index;
}

void create_views(LoopContext& ctx, const LoopDesc& d, uint32_t index)
{
    auto* resource = ctx.objects.at(index).Get();
    auto* device = ctx.probe.device;
    const auto stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto destination = [&](size_t heap, uint32_t slot) {
        auto handle = ctx.probe.heaps.at(heap)->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += slot * stride;
        return handle;
    };
    const bool buffer = d.type == 0;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    srv.Format = uav.Format = buffer ? DXGI_FORMAT_R32_TYPELESS : resource->GetDesc().Format;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (buffer) {
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        srv.Buffer.NumElements = uav.Buffer.NumElements = d.width / 4;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = d.mips;
    }
    device->CreateShaderResourceView(resource, &srv, destination(0, index));
    if (!(d.usage & 2))
        return;
    auto* slot = slot_of(ctx);
    auto* record = read<uint8_t*>(ctx.probe.scratch.data(), 0x1070) + index * 64;
    const auto base = read<uint32_t>(slot, 0xc);
    put(record, 0x34, base);
    put(record, 0x38, 1u);
    for (size_t heap : {2, 1})
        device->CreateUnorderedAccessView(resource, nullptr, &uav, destination(heap, base));
    // The raw-buffer path advances at both 0x36c0 and 0x3766.
    put(slot, 0xc, base + (buffer ? 2u : 1u));
}
} // namespace

LoopContext::LoopContext(ID3D12Device* device) : probe(device)
{
    run_device_slice(probe);
    auto* scratch = probe.scratch.data();
    put(&state, 0xc8, scratch);
    put(&state, 0x30, loop_callback);
    const uint32_t sizes[] = {0, 0, 0, 1};
    state.slot.index = UINT32_MAX;
    if (run_backend_slots(scratch, state.slot, sizes) || state.slot.index != 0)
        throw std::runtime_error("Unable to initialize provider loop slot");
    state.resources.fill(UINT32_MAX);
    state.mirror.fill(UINT32_MAX);
}

uint32_t LoopContext::create(uintptr_t backend, const LoopDesc& d, uint32_t context, uint32_t* index)
{
    if (calls >= visited.size())
        throw std::runtime_error("Extra provider callback");
    const auto& e = expected[calls];
    std::printf("resource id=%u name=%ls size=%ux%u type=%u format=%u usage=%u state=%u mode=%u heap=%u depth=%u "
                "mips=%u len=%llu\n",
                d.id, d.name, d.width, d.height, d.type, d.format, d.usage, d.state, d.mode, d.heap, d.depth, d.mips,
                d.length);
    if (backend != reinterpret_cast<uintptr_t>(&state) + 8 || context != state.slot.index || d.id != e.id ||
        index != &state.resources.at(e.id) || std::wcscmp(d.name, e.name) || d.type != e.type ||
        d.format != e.format || d.width != e.width || d.height != e.height || d.usage != e.usage ||
        d.state != e.state || d.mode != e.mode || d.heap != (d.id == 21) || d.placed || d.flags || d.depth != 1 ||
        d.mips != 1)
        throw std::runtime_error("Unexpected provider descriptor, order or writeback destination");
    visited[calls++] = d.id;
    if (d.id == fail_id)
        return 7;
    reserve_resource(slot_of(*this), probe.scratch.data(), d, index);
    if (*index != e.index)
        throw std::runtime_error("Unexpected backend resource index");
    auto desc = buffer_desc(d.width);
    if (d.type == 2) {
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Height = d.height;
        desc.DepthOrArraySize = static_cast<UINT16>(d.depth);
        desc.MipLevels = static_cast<UINT16>(d.mips);
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Format = format_for(d.format);
        if (desc.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{desc.Format};
            check(probe.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof support),
                  "Query packed RCAS UAV");
            if (force_packed_rcas || !(support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) ||
                !(support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
                desc.Format = DXGI_FORMAT_R32_UINT;
            std::printf("rcas_resource_format=%u package_format=67\n", unsigned(desc.Format));
        }
    }
    desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>((d.usage & 1) | ((d.usage & 2) ? 4 : 0));
    auto initial = d.state == 2 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    if (d.heap)
        initial = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (d.type == 0)
        initial = D3D12_RESOURCE_STATE_COMMON;
    else if (d.mode != 1)
        initial = D3D12_RESOURCE_STATE_COPY_DEST;
    create_object(*this, d, *index, desc, initial);
    if (d.heap)
        return 0;
    create_views(*this, d, *index);
    return schedule_upload(*this, d);
}

uint32_t LoopContext::make_upload(const LoopDesc& parent, const LoopDesc& upload)
{
    const auto target = state.resources.at(parent.id);
    const auto desc = objects.at(target)->GetDesc();
    if (upload.heap != 1 || upload.usage || upload.state != 20 || upload.id != parent.id ||
        upload.mode != parent.mode || upload.data != parent.data || upload.length != parent.length)
        throw std::runtime_error("Unexpected recursive upload descriptor");
    uint32_t index = 0;
    reserve_resource(slot_of(*this), probe.scratch.data(), upload, &index);
    if (index != target + 1)
        throw std::runtime_error("Unexpected upload slot");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows{};
    UINT64 row_bytes{}, total{};
    probe.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    if (rows != 1 || row_bytes != parent.length || (parent.mode != 2 && parent.mode != 3))
        throw std::runtime_error("Unexpected initialization footprint");
    const float exposure[] = {1, 1};
    if ((parent.mode == 2 &&
         (parent.length != sizeof exposure || !parent.data ||
          std::memcmp(reinterpret_cast<const void*>(parent.data), exposure, sizeof exposure) != 0)) ||
        (parent.mode == 3 && (parent.length != 4 || parent.data != 0)))
        throw std::runtime_error("Unexpected initialization payload");
    create_object(*this, upload, index, buffer_desc(total), D3D12_RESOURCE_STATE_GENERIC_READ);
    auto* staging = objects.at(index).Get();
    void* mapped{};
    D3D12_RANGE no_read{};
    check(staging->Map(0, &no_read, &mapped), "Map initialization upload");
    // The package iterates descriptor.height rows, including zero rows for its raw buffer.
    if (parent.height) {
        if (parent.mode == 2)
            std::memcpy(mapped, reinterpret_cast<const void*>(upload.data), row_bytes);
        else
            std::memset(mapped, static_cast<uint8_t>(upload.data), row_bytes);
    }
    D3D12_RANGE written{0, parent.height ? size_t(row_bytes) : 0};
    staging->Unmap(0, &written);
    ComputeCommands cmd(probe.device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* dst = objects.at(target).Get();
    if (parent.type == 0)
        cmd.list->CopyBufferRegion(dst, 0, staging, 0, parent.width);
    else {
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = staging;
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = footprint;
        to.pResource = dst;
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(cmd.list.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST,
               parent.state == 2 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    cmd.submit_and_wait(probe.device);
    std::printf("upload id=%u source_index=%u target_index=%u rows_written=%u\n", parent.id, index, target,
                parent.height);
    return 0;
}

#ifndef FFX_PACKAGE_DEVICE_SLICE
void reserve_resource(uint8_t* slot, uint8_t* scratch, const LoopDesc& d, uint32_t* index)
{
    *index = read<uint32_t>(slot, 4);
    put(slot, 4, *index + 1);
    auto* record = read<uint8_t*>(scratch, 0x1070) + *index * 64;
    std::memcpy(record + 8, &d.type, 32);
}
uint32_t schedule_upload(LoopContext& ctx, const LoopDesc& d)
{
    if (d.mode == 1)
        return 0;
    auto upload = d;
    upload.heap = 1;
    upload.usage = 0;
    upload.state = 20;
    return ctx.make_upload(d, upload);
}
uint32_t create_loop_context(LoopContext& ctx, void** output, const RecurrentConfig& config)
{
    static constexpr float exposure[] = {1, 1};
    *output = nullptr;
    ctx.state.resources.fill(0);
    for (const auto& e : expected) {
        if (e.id == 23 && !(config.flags & 0x200))
            continue;
        LoopDesc d{};
        d.type = e.type;
        d.format = e.format;
        d.width = e.width;
        d.height = e.height;
        d.depth = d.mips = 1;
        d.usage = e.usage;
        d.state = e.state;
        d.id = e.id;
        d.name = e.name;
        d.heap = e.id == 21;
        d.mode = e.mode;
        if (e.mode == 2) {
            d.length = 8;
            d.data = reinterpret_cast<uintptr_t>(exposure);
        }
        if (e.mode == 3)
            d.length = 4;
        if (ctx.create(reinterpret_cast<uintptr_t>(&ctx.state) + 8, d, ctx.state.slot.index,
                       &ctx.state.resources[e.id]))
            return 3;
    }
    ctx.state.mirror = ctx.state.resources;
    *output = &ctx.state;
    return 0;
}
#endif

void check_ffx_provider_loop(ID3D12Device* device)
{
#ifdef FFX_PACKAGE_DEVICE_SLICE
    std::puts("provider_loop=translated-package");
#else
    std::puts("provider_loop=desktop-control");
#endif
    // Also exercise the software packing route on adapters with native RGB9E5 stores.
    bool supports_native_rcas = false;
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        if (scenario == 3 && !supports_native_rcas)
            break;
        RecurrentConfig config{};
        config.render_width = 1280;
        config.render_height = 720;
        config.output_width = 1920;
        config.output_height = 1080;
        config.flags = (scenario == 1 || scenario == 2) ? 0x200 : 0;
        LoopContext ctx(device);
        ctx.force_packed_rcas = scenario == 3;
        if (scenario == 2)
            ctx.fail_id = 20;
        void* output = device;
#ifdef FFX_LOOP_LOADER
        auto status = check_loop_loader(ctx, &output, config);
#else
        auto status = create_loop_context(ctx, &output, config);
#endif
        const bool failed = ctx.fail_id != 0;
        std::array<uint32_t, 25> wanted{};
        unsigned count = 0, cursor = 1;
        for (const auto& e : expected) {
            if (e.id == 23 && !config.flags)
                continue;
            if (ctx.visited.at(count++) != e.id)
                throw std::runtime_error("Missing provider callback");
            if (e.id == ctx.fail_id)
                break;
            wanted[e.id] = e.index;
            cursor = e.index + (e.mode == 1 ? 1 : 2);
        }
        auto* records = read<uint8_t*>(ctx.probe.scratch.data(), 0x1070);
        for (unsigned i = 1; i < ctx.objects.size(); ++i) {
            auto* object = ctx.objects[i].Get();
            if ((i < cursor) != bool(object) || read<ID3D12Resource*>(records, 64 * i) != object)
                throw std::runtime_error("Resource handle table mismatch");
            for (unsigned j = 1; object && j < i; ++j)
                if (object == ctx.objects[j].Get())
                    throw std::runtime_error("Aliased resource records");
        }
        auto mirror = wanted;
        if (failed)
            mirror.fill(UINT32_MAX);
        if (status != (failed ? 3u : 0u) || output != (failed ? nullptr : &ctx.state) || count != ctx.calls ||
            ctx.state.resources != wanted || ctx.state.mirror != mirror || read<uint32_t>(slot_of(ctx), 4) != cursor)
            throw std::runtime_error("Provider loop result, handle, cursor or indices mismatch");
        std::printf("flags=0x%x failure_id=%u visited=%u next=%u status=%u handle=%s indices=", config.flags,
                    ctx.fail_id, count, cursor, status, output ? "provider" : "null");
        for (const auto& e : expected)
            std::printf(" %u=%u", e.id, ctx.state.resources[e.id]);
        std::puts("");
        if (scenario == 0 || scenario == 3) {
            const auto index = ctx.state.resources.at(16);
            const auto format = ctx.objects.at(index)->GetDesc().Format;
            if (scenario == 3 && format != DXGI_FORMAT_R32_UINT)
                throw std::runtime_error("Packed RCAS control selected the wrong format");
            if (scenario == 0)
                supports_native_rcas = format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
            check_rcas_views(ctx, index, read<uint32_t>(records, 64 * index + 0x34));
            check_ffx_dispatch(ctx, records);
        }
    }
}
