#include "ffx_provider_loop.h"
#include "bvh_fixture.h"
#include "RcasReadNative.h"
#include "RcasReadPacked.h"
#include "RcasWriteNative.h"
#include "RcasWritePacked.h"
#include <algorithm>
#include <cmath>

namespace {
using RGB = std::array<float, 3>;
constexpr UINT texels = 32 * 32;

std::vector<RGB> test_colors()
{
    std::vector<RGB> colors;
    for (int exponent = 0; exponent < 32; ++exponent) {
        const float unit = std::ldexp(1.0f, exponent - 24);
        for (int i = 0; i < 32; ++i) {
            RGB rgb{};
            // Rotate the brightest channel. Probe quantization on both sides
            // of a half step, an exact tie, and a carry into the next exponent.
            rgb[i % 3] = (i < 4 ? 511.25f + 0.25f * i : 256.0f + 5 * i) * unit;
            rgb[(i + 1) % 3] = ((i * 7) + 0.25f * (i % 4)) * unit;
            rgb[(i + 2) % 3] = i % 5 ? unit : -unit;
            colors.push_back(rgb);
        }
    }
    colors[4] = {0, 0, 0};
    colors[5] = {-2, -0.0f, -1};
    colors[6] = {std::ldexp(1.0f, -25), 0, 0};
    colors[7] = {65536, 65408, 0};
    return colors;
}

// Choose the first scale at which all rounded mantissas fit. Double arithmetic
// keeps this oracle independent of the shader's float exponent-bit selection.
uint32_t reference_word(const RGB& rgb, bool ties_up)
{
    for (int e = 0; e < 32; ++e) {
        uint32_t packed = uint32_t(e) << 27;
        bool fits = true;
        for (int c = 0; c < 3; ++c) {
            const double scaled = std::ldexp(std::clamp(double(rgb[c]), 0.0, 65408.0), 24 - e);
            const double rounded = ties_up ? std::floor(scaled + 0.5) : std::nearbyint(scaled);
            if (rounded > 511) { fits = false; break; }
            packed |= uint32_t(rounded) << (9 * c);
        }
        if (fits) return packed;
    }
    throw std::runtime_error("No RGB9E5 encoding for clamped input");
}

void expect_word(uint32_t got, uint32_t want, UINT pixel, UINT field)
{
    if (got != want) {
        std::fprintf(stderr, "RCAS texel=%u field=%u expected=%08x actual=%08x\n", pixel, field, want, got);
        throw std::runtime_error("RCAS texel readback mismatch");
    }
}
}

void check_rcas_views(LoopContext& ctx, uint32_t index, uint32_t uav_slot)
{
    auto* device = ctx.probe.device;
    auto* texture = ctx.objects.at(index).Get();
    auto desc = texture->GetDesc();
    const bool packed = desc.Format == DXGI_FORMAT_R32_UINT;
    if ((!packed && desc.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP) || desc.Width < texels)
        throw std::runtime_error("Unexpected RCAS intermediate texture");
    const auto inputs = test_colors();
    static_assert(sizeof(RGB) == 12);
    auto colors = bvh_fixture::upload(device, inputs);
    const UINT decoded_bytes = 16 * texels;
    auto decoded = make_buffer(device, decoded_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Reuse the provider's actual UAV and SRV. Only the SRV needs copying
    // because the provider stores it in a CPU-only heap.
    constexpr auto type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    const UINT stride = device->GetDescriptorHandleIncrementSize(type);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{type, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> read_heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&read_heap)), "Create RCAS read heap");
    auto srv = ctx.probe.heaps[0]->GetCPUDescriptorHandleForHeapStart();
    srv.ptr += stride * index;
    device->CopyDescriptorsSimple(1, read_heap->GetCPUDescriptorHandleForHeapStart(), srv, type);

    auto* write_heap = ctx.probe.heaps[2].Get();
    auto uav = write_heap->GetGPUDescriptorHandleForHeapStart();
    uav.ptr += stride * uav_slot;
    const D3D12_SHADER_BYTECODE write_code = packed ? D3D12_SHADER_BYTECODE{RcasWritePacked, sizeof(RcasWritePacked)}
                                                     : D3D12_SHADER_BYTECODE{RcasWriteNative, sizeof(RcasWriteNative)};
    const D3D12_SHADER_BYTECODE read_code = packed ? D3D12_SHADER_BYTECODE{RcasReadPacked, sizeof(RcasReadPacked)}
                                                   : D3D12_SHADER_BYTECODE{RcasReadNative, sizeof(RcasReadNative)};
    bvh_fixture::Kernel writer(device, write_code), reader(device, read_code);
    ComputeCommands cmd(device, writer.pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = cmd.list.Get();
    list->SetDescriptorHeaps(1, &write_heap);
    list->SetComputeRootSignature(writer.root.Get());
    list->SetComputeRootDescriptorTable(0, uav);
    list->SetComputeRootShaderResourceView(1, colors->GetGPUVirtualAddress());
    list->Dispatch(texels / 32, 1, 1);
    transition(list, texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    auto* visible_srv = read_heap.Get();
    list->SetDescriptorHeaps(1, &visible_srv);
    list->SetPipelineState(reader.pipeline.Get());
    list->SetComputeRootSignature(reader.root.Get());
    list->SetComputeRootDescriptorTable(0, visible_srv->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootUnorderedAccessView(1, decoded->GetGPUVirtualAddress());
    list->Dispatch(texels / 32, 1, 1);
    transition(list, decoded.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

    desc.Width = texels;
    desc.Height = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT strip{};
    device->GetCopyableFootprints(&desc, 0, 1, decoded_bytes, &strip, nullptr, nullptr, nullptr);
    const size_t bytes = strip.Offset + strip.Footprint.RowPitch;
    auto back = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(back.Get(), 0, decoded.Get(), 0, decoded_bytes);
    transition(list, texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = back.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = strip;
    const D3D12_BOX box{0, 0, 0, texels, 1, 1};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    transition(list, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd.submit_and_wait(device);

    const auto words = bvh_fixture::read(back.Get(), bytes / 4);
    UINT native_half_up = 0;
    for (UINT i = 0; i < texels; ++i) {
        const auto raw = words[strip.Offset / 4 + i];
        const auto canonical = reference_word(inputs[i], false);
        // Direct3D permits round-to-nearest as well as nearest-even on native stores.
        if (!packed && raw != canonical && raw == reference_word(inputs[i], true)) ++native_half_up;
        else expect_word(raw, canonical, i, 4);
        for (unsigned c = 0; c < 4; ++c) {
            const float value = c == 3 ? 1.0f : std::ldexp(float((raw >> (9*c)) & 511), int(raw >> 27) - 24);
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof bits);
            expect_word(words[4*i+c], bits, i, c);
        }
    }
    std::printf("rcas_texels=%u format=%u id=16 index=%u uav_slot=%u native_half_up=%u mismatches=0\n",
                texels, unsigned(desc.Format), index, uav_slot, native_half_up);
}
