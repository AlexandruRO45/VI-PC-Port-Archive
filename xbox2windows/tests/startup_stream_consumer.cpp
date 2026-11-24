#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <wrl/client.h>

#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "startup_stream_draw.h"
#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

// A slot-55 patch is a tagged union with room for the full eight-target blend state.
struct Patch { UINT tag; D3D12_BLEND_DESC blend; };
static_assert(sizeof(Patch) == 0x14c);

static bool derive(void* device, void*& pipeline, bool child)
{
    using Create = HRESULT (WINAPI*)(void*, void*, UINT, const Patch*, REFIID, void**);
    auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[55]);
    Patch preserve{2}, write{2}, sample_mask{3};
    const UINT all_samples = ~0u;
    std::memcpy(&sample_mask.blend, &all_samples, sizeof(all_samples));
    auto& t = preserve.blend.RenderTarget[0];
    t = {TRUE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ZERO,
         D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    write.blend = preserve.blend;
    write.blend.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    write.blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    void* next = nullptr;
    if (create(device, pipeline, 1, nullptr, __uuidof(ID3D12PipelineState), &next) != E_INVALIDARG || next) return false;
    if (FAILED(create(device, pipeline, 1, &preserve, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    if (child) {
        Patch patches[] = {preserve, write, sample_mask};
        if (FAILED(create(device, pipeline, 3, patches, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
        pipeline = next;
    }
    // A grandchild with only a non-blend patch keeps its nearest ancestor's state.
    if (FAILED(create(device, pipeline, 1, &sample_mask, __uuidof(ID3D12PipelineState), &next)) || !next) return false;
    pipeline = next;
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 4 && (argc < 6 || argc > 8) && argc != 10) return 2;
    const bool draw = argc != 4;
    StartupBc7Input bc7{};
    if (argc == 10) {
        bc7 = {argv[6], static_cast<UINT>(std::strtoul(argv[7], nullptr, 10)),
               static_cast<UINT>(std::strtoul(argv[8], nullptr, 10)),
               std::strcmp(argv[9], "tiled") == 0};
        if ((bc7.width != 64 || bc7.height != 64) && (bc7.width != 81 || bc7.height != 69)) return 2;
        if (!bc7.tiled && std::strcmp(argv[9], "linear") != 0) return 2;
    }
    ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    const bool hardware = draw && std::strcmp(argv[4], "hardware") == 0;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], hardware, bc7.blocks != nullptr,
                                                adapter, device, pso)) return rc;
    if (GetEnvironmentVariableW(L"X2W_TEST_TRUNCATED_ONLY", nullptr, 0)) return 0;
    if (argc == 4) return 0;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    UINT export_word = 0;
    if (argc == 7 && !std::isdigit(static_cast<unsigned char>(argv[6][0]))) {
        const std::string mode(argv[6]);
        if (mode != "parent-a8" && mode != "child-a8" && mode != "child-r8") return 2;
        if (!derive(device, pso, mode != "parent-a8")) return 10;
        format = mode == "child-r8" ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_A8_UNORM;
    } else if (argc == 7 || argc == 8) {
        // A render-target format for the translated MRT export, and an optional root constant.
        format = static_cast<DXGI_FORMAT>(std::strtoul(argv[6], nullptr, 10));
        if (format != DXGI_FORMAT_R32_UINT && format != DXGI_FORMAT_R32_SINT &&
            format != DXGI_FORMAT_R32G32B32A32_FLOAT) return 2;
        if (argc == 8) export_word = static_cast<UINT>(std::strtoul(argv[7], nullptr, 0));
    }
    if (GetEnvironmentVariableW(L"X2W_TEST_PLACED_VERTEX", nullptr, 0)) {
        extern void release_placed_vertices(bool);
        const std::string first = std::string(argv[5]) + ".first";
        if (const int rc = check_stream_draw(adapter.Get(), device, pso, first.c_str(), nullptr, format)) return rc;
        // GPU fence and readback are complete before the high range is released.
        release_placed_vertices(false);
        const int rc = check_stream_draw(adapter.Get(), device, pso, argv[5], nullptr, format);
        release_placed_vertices(true);
        return rc;
    }
    return check_stream_draw(adapter.Get(), device, pso, argv[5], bc7.blocks ? &bc7 : nullptr, format, export_word);
}
