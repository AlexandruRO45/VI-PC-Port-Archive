// Differential GetCopyableFootprints fuzzer: seeded random resource descriptions (DXGI formats 1-132, buffers,
// 1D/2D/3D, arrays, mip chains, the depth-stencil flag, subresource ranges, 512-aligned base offsets) asked of the
// layer's device, of the host adapter and of WARP. The answer is the runtime's arithmetic and validation, not a
// driver's, so where the host adapter and WARP agree the layer must give the same answer:
//   - the layer answering where the host refuses, or answering differently, fails the run;
//   - the layer refusing where the host answers is a gap (a format it does not lay out yet): counted, printed with
//     --gaps, not a failure.
// --seeds N --start S --cases N choose the inputs (default: seed 1, 20,000 descriptions). FP_DUMP=1 prints every
// description with the three answers, for deriving a rule over the whole population. An unaligned base offset is
// #2498's; DXGI_FORMAT_420_OPAQUE is skipped because the host runtime fast-fails on some of its descriptions.
// Needs a Vulkan GPU and a host D3D12 adapter (exit 77 otherwise, and under Wine). No game data.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

ComPtr<ID3D12Device> native_device(std::wstring& name) {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        name = desc.Description;
        return device;
    }
}

ComPtr<ID3D12Device> warp_device() {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
        return {};
    return device;
}

struct Answer {
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts;
    std::vector<UINT> rows;
    std::vector<UINT64> row_sizes;
    UINT64 total = 0;
    bool refused() const { return total == UINT64_MAX; }
    bool operator==(const Answer& o) const {
        if (total != o.total || rows != o.rows || row_sizes != o.row_sizes || layouts.size() != o.layouts.size())
            return false;
        for (size_t i = 0; i < layouts.size(); ++i)
            if (std::memcmp(&layouts[i], &o.layouts[i], sizeof layouts[i]) != 0) return false;
        return true;
    }
};

Answer ask(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, UINT first, UINT count, UINT64 base) {
    Answer a;
    a.layouts.assign(count, {});
    a.rows.assign(count, 0xcdcdcdcd);
    a.row_sizes.assign(count, 0xcdcdcdcdcdcdcdcdull);
    a.total = 0xcdcdcdcdcdcdcdcdull;
    device->GetCopyableFootprints(&desc, first, count, base, a.layouts.data(), a.rows.data(), a.row_sizes.data(),
                                  &a.total);
    return a;
}

void describe(const char* what, const D3D12_RESOURCE_DESC& d, UINT first, UINT count, UINT64 base) {
    std::printf("%s: dimension %d format %d %llux%ux%u mips %u flags 0x%x, subresources %u..%u, base %llu\n", what,
                d.Dimension, d.Format, static_cast<unsigned long long>(d.Width), d.Height, d.DepthOrArraySize,
                d.MipLevels, d.Flags, first, first + count - 1, static_cast<unsigned long long>(base));
}

void print(const char* who, const Answer& a) {
    if (a.refused()) {
        std::printf("  %-5s refuses\n", who);
        return;
    }
    std::printf("  %-5s total %llu\n", who, static_cast<unsigned long long>(a.total));
    for (size_t i = 0; i < a.layouts.size() && i < 4; ++i) {
        const auto& l = a.layouts[i];
        std::printf("    [%zu] offset %llu format %d %ux%ux%u pitch %u, %u rows of %llu bytes\n", i,
                    static_cast<unsigned long long>(l.Offset), l.Footprint.Format, l.Footprint.Width,
                    l.Footprint.Height, l.Footprint.Depth, l.Footprint.RowPitch, a.rows[i],
                    static_cast<unsigned long long>(a.row_sizes[i]));
    }
}

}  // namespace

int main(int argc, char** argv) {
    unsigned seeds = 1, start = 1, cases = 20000;
    bool gaps = false;
    for (int i = 1; i < argc; ++i) {
        const auto number = [&](unsigned& out) { if (i + 1 < argc) out = unsigned(std::strtoul(argv[++i], nullptr, 10)); };
        if (!std::strcmp(argv[i], "--seeds")) number(seeds);
        else if (!std::strcmp(argv[i], "--start")) number(start);
        else if (!std::strcmp(argv[i], "--cases")) number(cases);
        else if (!std::strcmp(argv[i], "--gaps")) gaps = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    const bool dump = GetEnvironmentVariableA("FP_DUMP", nullptr, 0) != 0;
    if (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version")) {
        std::puts("SKIP: Wine's D3D12 is not a host reference");
        return 77;
    }
    ComPtr<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer)))) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    std::wstring name;
    ComPtr<ID3D12Device> host = native_device(name);
    ComPtr<ID3D12Device> warp = warp_device();
    if (!host || !warp) {
        std::puts("SKIP: no host D3D12 adapter or no WARP to compare with");
        return 77;
    }
    unsigned asked = 0, wrong = 0, gap_count = 0, hosts_disagree = 0, host_wraps = 0, shown = 0;
    for (unsigned seed = start; seed < start + seeds; ++seed) {
        std::mt19937_64 rng(seed);
        for (unsigned c = 0; c < cases; ++c) {
            D3D12_RESOURCE_DESC d{};
            const int dim = int(rng() % 4);  // buffer, 1D, 2D, 3D
            d.Dimension = D3D12_RESOURCE_DIMENSION(dim == 0 ? 1 : dim + 1);
            d.Format = dim == 0 ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT(1 + rng() % 132);
            const auto size = [&](UINT max) { return UINT(1 + (rng() % 3 == 0 ? rng() % max : rng() % 70)); };
            d.Width = dim == 0 ? 1 + rng() % 100000 : size(dim == 3 ? 4096 : 20000);
            d.Height = dim >= 2 ? size(dim == 3 ? 2400 : 20000) : 1;
            d.DepthOrArraySize = UINT16(dim == 3 ? size(2400) : (dim == 0 ? 1 : 1 + rng() % 6));
            d.MipLevels = UINT16(dim == 0 ? 1 : rng() % 9);  // 0: the full chain
            d.SampleDesc = {1, 0};
            d.Layout = dim == 0 ? D3D12_TEXTURE_LAYOUT_ROW_MAJOR : D3D12_TEXTURE_LAYOUT_UNKNOWN;
            if (rng() % 8 == 0) d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            const UINT mips = d.MipLevels ? d.MipLevels : 13, layers = dim == 3 ? 1 : d.DepthOrArraySize;
            const UINT first = UINT(rng() % (mips * layers * 2 + 1));  // up to two planes, and one past the end
            const UINT count = 1 + UINT(rng() % 8);
            const UINT64 base = rng() % 4 == 0 ? (rng() % 200) * D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT : 0;
            if (d.Format == DXGI_FORMAT_420_OPAQUE) continue;
            ++asked;
            const Answer theirs = ask(host.Get(), d, first, count, base);
            const Answer ours = ask(layer.Get(), d, first, count, base);
            if (dump) {
                const bool warp_ok = !ask(warp.Get(), d, first, count, base).refused();
                std::printf("ALL dim=%d fmt=%d w=%llu h=%u d=%u mips=%u flags=%u first=%u count=%u host=%d warp=%d "
                            "layer=%d same=%d\n", dim, d.Format, static_cast<unsigned long long>(d.Width), d.Height,
                            d.DepthOrArraySize, d.MipLevels, d.Flags, first, count, !theirs.refused(), warp_ok,
                            !ours.refused(), theirs == ours);
            }
            if (theirs == ours) continue;
            if (!(ask(warp.Get(), d, first, count, base) == theirs)) {  // a driver's answer, not the runtime's
                ++hosts_disagree;
                continue;
            }
            // The PC runtime (adapter and WARP alike) computes a subresource's size in 32 bits: a volume subresource
            // of 4 GiB or more wraps (805x777x750 RGBA32: the next mip at 7,608,384,000 - 2^32). The layer keeps
            // the true offsets; those descriptions are not compared.
            bool beyond_32_bits = false;
            for (size_t i = 0; !ours.refused() && i < ours.layouts.size(); ++i)
                beyond_32_bits |= UINT64(ours.layouts[i].Footprint.RowPitch) * ours.rows[i] *
                                      ours.layouts[i].Footprint.Depth > UINT64(UINT32_MAX);
            if (beyond_32_bits) {
                ++host_wraps;
                continue;
            }
            if (ours.refused()) {
                ++gap_count;
                if (gaps) describe("gap", d, first, count, base);
                continue;
            }
            ++wrong;
            if (shown++ < 20) {
                describe("WRONG", d, first, count, base);
                print("host", theirs);
                print("layer", ours);
            }
        }
    }
    std::printf("footprint fuzz (%ls and WARP): %u descriptions, %u wrong, %u gaps (the layer refuses), %u where the "
                "two hosts disagree, %u where the host wraps a 4 GiB subresource\n", name.c_str(), asked, wrong,
                gap_count, hosts_disagree, host_wraps);
    return wrong ? 1 : 0;
}
