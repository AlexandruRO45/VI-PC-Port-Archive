// Sweep of the device queries the layer answers from tables of its own, against Windows D3D12 (the layer's adapter and
// WARP): for every DXGI format, CheckFeatureSupport FORMAT_SUPPORT / FORMAT_INFO / MULTISAMPLE_QUALITY_LEVELS, and for
// grids of texture descriptions GetResourceAllocationInfo and GetCopyableFootprints.
//
// A difference counts only when the hardware adapter and WARP give the same answer and the layer gives another one: where
// the two host implementations disagree the answer is implementation-defined and there is nothing to match.
//
// Which differences fail the run. The layer may know less than the host (it refuses a video format, reports no tiled
// support: those are listed as gaps), but it must never claim more or answer differently:
//   - FORMAT_SUPPORT: a Support1/Support2 bit the host lacks;
//   - MULTISAMPLE_QUALITY_LEVELS: more quality levels than the host;
//   - GetResourceAllocationInfo: a size where the host refuses, or, for the engine-style descriptions (render targets,
//     depth buffers, block-compressed textures with the flags they carry), a smaller size than the host's;
//   - GetCopyableFootprints: any answer that is not the host's (a refusal is a gap, not a wrong answer).
// Each group is one query and format with its smallest description, so one root cause reads as one line.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
// LEONIDA_SWEEP_GAPS=1 also prints the gaps (the layer knows less than the host); LEONIDA_SWEEP_REPORT=1 prints every
// group and exits 0, for exploring a new area.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr int kFormatCount = 192;  // DXGI_FORMAT_UNKNOWN .. DXGI_FORMAT_A4B4G4R4_UNORM (191)
constexpr uint64_t kInvalid = ~0ull;  // GetResourceAllocationInfo's answer for a description D3D12 rejects

// One answer to one query, as comparable words.
using Words = std::vector<uint64_t>;

struct Group {
    unsigned count = 0;
    uint64_t size = kInvalid;  // of the smallest description in the group
    std::string example;
};
using Groups = std::map<std::string, Group>;

struct Results {
    Groups errors, gaps;
    unsigned compared = 0, hosts_disagree = 0;
};

std::string describe(const D3D12_RESOURCE_DESC& d) {
    char text[200];
    std::snprintf(text, sizeof(text), "dim %d %llux%ux%u mips %u format %d samples %u flags 0x%x",
                  int(d.Dimension), static_cast<unsigned long long>(d.Width), d.Height, unsigned(d.DepthOrArraySize),
                  unsigned(d.MipLevels), int(d.Format), d.SampleDesc.Count, unsigned(d.Flags));
    return text;
}

std::string words(const Words& w) {
    std::string out;
    char text[40];
    for (size_t i = 0; i < w.size() && i < 24; ++i) {
        std::snprintf(text, sizeof(text), "%s%llx", i ? " " : "", static_cast<unsigned long long>(w[i]));
        out += text;
    }
    if (w.size() > 24) out += " ...";
    return out;
}

enum class Verdict { same, gap, error };

// Records `ours` against the host's answer (`hardware`, which `warp` confirms when there is a WARP device).
void judge(Results& r, Verdict (*rule)(const Words& ours, const Words& host), const std::string& group, uint64_t size,
           const std::string& what, const Words& ours, const Words& hardware, const Words* warp) {
    if (warp && hardware != *warp) {
        ++r.hosts_disagree;
        return;
    }
    ++r.compared;
    const Verdict verdict = ours == hardware ? Verdict::same : rule(ours, hardware);
    if (verdict == Verdict::same) return;
    Group& g = (verdict == Verdict::error ? r.errors : r.gaps)[group];
    ++g.count;
    if (size < g.size) {
        g.size = size;
        g.example = what + "\n      ours " + words(ours) + "\n      host " + words(hardware);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// FORMAT_SUPPORT, FORMAT_INFO and MULTISAMPLE_QUALITY_LEVELS for every format.
Words format_support(ID3D12Device* d, DXGI_FORMAT f) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT s{};
    s.Format = f;
    const HRESULT hr = d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &s, sizeof(s));
    return {uint64_t(uint32_t(hr)), uint64_t(s.Support1), uint64_t(s.Support2)};
}

Words format_info(ID3D12Device* d, DXGI_FORMAT f) {
    D3D12_FEATURE_DATA_FORMAT_INFO s{};
    s.Format = f;
    const HRESULT hr = d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &s, sizeof(s));
    return {uint64_t(uint32_t(hr)), uint64_t(s.PlaneCount)};
}

Words quality_levels(ID3D12Device* d, DXGI_FORMAT f, UINT samples, D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS flags) {
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS s{};
    s.Format = f;
    s.SampleCount = samples;
    s.Flags = flags;
    const HRESULT hr = d->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &s, sizeof(s));
    return {uint64_t(uint32_t(hr)), uint64_t(s.NumQualityLevels)};
}

// Support the host lacks is an error; support the layer lacks, and a different HRESULT, are gaps.
Verdict support_rule(const Words& ours, const Words& host) {
    return (ours[1] & ~host[1]) || (ours[2] & ~host[2]) ? Verdict::error : Verdict::gap;
}

Verdict info_rule(const Words&, const Words&) { return Verdict::gap; }

Verdict levels_rule(const Words& ours, const Words& host) {
    return ours[1] > host[1] ? Verdict::error : Verdict::gap;
}

void sweep_formats(ID3D12Device* ours, ID3D12Device* hardware, ID3D12Device* warp, Results& r) {
    for (int f = 0; f < kFormatCount; ++f) {
        const DXGI_FORMAT format = DXGI_FORMAT(f);
        char label[64];
        std::snprintf(label, sizeof(label), "format %d", f);
        auto ask = [&](auto&& query, Verdict (*rule)(const Words&, const Words&), const std::string& group,
                       uint64_t size, const std::string& what) {
            const Words w = warp ? query(warp) : Words{};
            judge(r, rule, group, size, what, query(ours), query(hardware), warp ? &w : nullptr);
        };
        ask([&](ID3D12Device* d) { return format_support(d, format); }, support_rule,
            std::string("FORMAT_SUPPORT ") + label, 0, std::string("FORMAT_SUPPORT ") + label);
        ask([&](ID3D12Device* d) { return format_info(d, format); }, info_rule, std::string("FORMAT_INFO ") + label, 0,
            std::string("FORMAT_INFO ") + label);
        for (const UINT samples : {1u, 2u, 4u, 8u, 16u}) {
            for (const auto flags : {D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                                     D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE}) {
                char what[96];
                std::snprintf(what, sizeof(what), "MULTISAMPLE_QUALITY_LEVELS %s samples %u flags %d", label, samples,
                              int(flags));
                ask([&](ID3D12Device* d) { return quality_levels(d, format, samples, flags); }, levels_rule,
                    std::string("MULTISAMPLE_QUALITY_LEVELS ") + label, samples, what);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GetResourceAllocationInfo and GetCopyableFootprints over grids of texture descriptions.
D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth_or_layers,
                            UINT16 mips, DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth_or_layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Flags = flags;
    return desc;
}

Words allocation(ID3D12Device* d, const D3D12_RESOURCE_DESC& desc) {
    const D3D12_RESOURCE_ALLOCATION_INFO info = d->GetResourceAllocationInfo(0, 1, &desc);
    return {info.SizeInBytes, info.Alignment};
}

// An alignment smaller than the host's lets CreatePlacedResource accept an offset the host would refuse, and callers
// are told to take the alignment from GetResourceAllocationInfo. It is an error whenever both answers are valid. A
// larger alignment only wastes space, so it stays a gap.
bool alignment_undersized(const Words& ours, const Words& host) {
    return ours[0] != kInvalid && host[0] != kInvalid && ours[1] < host[1];
}

// A size where the host refuses is an error. A smaller size than the host's would let two placed resources overlap, so it
// is an error for the descriptions an engine makes and a gap for the odd grid (formats without their flag).
Verdict allocation_rule(const Words& ours, const Words& host) {
    if (ours[0] != kInvalid && host[0] == kInvalid) return Verdict::error;
    if (alignment_undersized(ours, host)) return Verdict::error;
    return Verdict::gap;
}

Verdict engine_allocation_rule(const Words& ours, const Words& host) {
    if (ours[0] != kInvalid && (host[0] == kInvalid || ours[0] < host[0])) return Verdict::error;
    if (alignment_undersized(ours, host)) return Verdict::error;
    return Verdict::gap;
}

// A refusal (all ones) is a gap; any other answer that is not the host's is an error.
Verdict footprint_rule(const Words& ours, const Words&) {
    return ours[0] == kInvalid ? Verdict::gap : Verdict::error;
}

bool creatable(ID3D12Device* d, const D3D12_RESOURCE_DESC& desc) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    Ref<ID3D12Resource> resource;
    return SUCCEEDED(d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&resource.p))) &&
           resource.p;
}

unsigned chain_length(const D3D12_RESOURCE_DESC& d) {
    if (d.MipLevels) return d.MipLevels;
    UINT64 extent = d.Width;
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D) extent = extent > d.Height ? extent : d.Height;
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && d.DepthOrArraySize > extent) extent = d.DepthOrArraySize;
    unsigned n = 1;
    while (extent > 1) {
        extent >>= 1;
        ++n;
    }
    return n;
}

Words footprints(ID3D12Device* d, const D3D12_RESOURCE_DESC& desc, UINT count) {
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(count);
    std::vector<UINT> rows(count, 0xdeadbeef);
    std::vector<UINT64> row_sizes(count, 0xdeadbeefdeadbeefull);
    UINT64 total = 0xdeadbeefdeadbeefull;
    d->GetCopyableFootprints(&desc, 0, count, 0, layouts.data(), rows.data(), row_sizes.data(), &total);
    Words out{total};
    for (UINT i = 0; i < count; ++i) {
        const auto& f = layouts[i].Footprint;
        out.insert(out.end(), {layouts[i].Offset, uint64_t(f.Format), f.Width, f.Height, f.Depth, f.RowPitch, rows[i],
                               row_sizes[i]});
    }
    return out;
}

void sweep_textures(ID3D12Device* ours, ID3D12Device* hardware, ID3D12Device* warp, Results& r) {
    struct Extent {
        UINT64 w;
        UINT h;
        UINT16 depth;
    };
    constexpr D3D12_RESOURCE_DIMENSION t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                                       t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const Extent extents[] = {{1, 1, 1}, {3, 5, 2}, {4, 4, 4}, {7, 9, 3}, {64, 64, 4}, {100, 60, 5}, {130, 66, 3}};
    for (int f = 1; f < kFormatCount; ++f) {
        const DXGI_FORMAT format = DXGI_FORMAT(f);
        D3D12_FEATURE_DATA_FORMAT_INFO info{};
        info.Format = format;
        if (FAILED(hardware->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info)))) continue;
        // The video formats (NV12 .. P8, P208 .. V408) are left out of the footprints: GetCopyableFootprints on the
        // hardware runtime aborts the process for them when the count is not what it expects.
        const bool video = (f >= 103 && f <= 113) || (f >= 130 && f <= 132);
        for (const D3D12_RESOURCE_DIMENSION dimension : {t1, t2, t3}) {
            for (const Extent& e : extents) {
                for (const UINT16 mips : {UINT16(1), UINT16(0)}) {
                    for (const UINT16 layers : {UINT16(1), UINT16(3)}) {
                        if (dimension == t3 && layers != 1) continue;
                        const UINT16 depth_or_layers = dimension == t3 ? e.depth : layers;
                        const UINT height = dimension == t1 ? 1 : e.h;
                        const D3D12_RESOURCE_DESC desc =
                            texture(dimension, e.w, height, depth_or_layers, mips, format, 1,
                                    D3D12_RESOURCE_FLAG_NONE);
                        const UINT64 volume = e.w * height * depth_or_layers;
                        const std::string what = describe(desc);
                        char label[48];
                        std::snprintf(label, sizeof(label), "format %d dim %d", f, int(dimension));

                        const Words h_alloc = allocation(hardware, desc);
                        const Words w_alloc = warp ? allocation(warp, desc) : Words{};
                        judge(r, allocation_rule, std::string("GetResourceAllocationInfo ") + label, volume, what,
                              allocation(ours, desc), h_alloc, warp ? &w_alloc : nullptr);

                        // Footprints only of a description every host can create: an invalid one is outside the API's
                        // contract.
                        if (video || h_alloc[0] == kInvalid || !creatable(hardware, desc) ||
                            (warp && !creatable(warp, desc)))
                            continue;
                        const UINT count = chain_length(desc) * (dimension == t3 ? 1 : layers) * info.PlaneCount;
                        if (count > 64) continue;
                        const Words h_foot = footprints(hardware, desc, count);
                        const Words w_foot = warp ? footprints(warp, desc, count) : Words{};
                        judge(r, footprint_rule, std::string("GetCopyableFootprints ") + label, volume, what,
                              footprints(ours, desc, count), h_foot, warp ? &w_foot : nullptr);
                    }
                }
            }
        }
    }
}

// The descriptions an engine really makes: render targets, depth buffers and block-compressed textures at screen and
// shadow-map sizes, with the flags they carry, as arrays and with full mip chains. Sizes only: what a placed resource
// needs from its heap, and where a smaller answer than the host's would let two resources overlap.
void sweep_engine_textures(ID3D12Device* ours, ID3D12Device* hardware, ID3D12Device* warp, Results& r) {
    struct Kind {
        DXGI_FORMAT format;
        D3D12_RESOURCE_FLAGS flags;
    };
    constexpr auto rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ds = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                   uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, none = D3D12_RESOURCE_FLAG_NONE;
    const Kind kinds[] = {
        {DXGI_FORMAT_R8G8B8A8_UNORM, rt | uav}, {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, rt},
        {DXGI_FORMAT_B8G8R8A8_UNORM, rt},       {DXGI_FORMAT_R16G16B16A16_FLOAT, rt | uav},
        {DXGI_FORMAT_R11G11B10_FLOAT, rt | uav}, {DXGI_FORMAT_R10G10B10A2_UNORM, rt},
        {DXGI_FORMAT_R32_FLOAT, rt | uav},      {DXGI_FORMAT_R16_FLOAT, rt | uav},
        {DXGI_FORMAT_R8_UNORM, rt | uav},       {DXGI_FORMAT_R32G32B32A32_FLOAT, rt | uav},
        {DXGI_FORMAT_R16G16_FLOAT, rt | uav},   {DXGI_FORMAT_R32_TYPELESS, ds},
        {DXGI_FORMAT_D32_FLOAT, ds},            {DXGI_FORMAT_D24_UNORM_S8_UINT, ds},
        {DXGI_FORMAT_R24G8_TYPELESS, ds},       {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ds},
        {DXGI_FORMAT_R32G8X24_TYPELESS, ds},    {DXGI_FORMAT_D16_UNORM, ds},
        {DXGI_FORMAT_R16_TYPELESS, ds},         {DXGI_FORMAT_BC1_UNORM, none},
        {DXGI_FORMAT_BC3_UNORM, none},          {DXGI_FORMAT_BC4_UNORM, none},
        {DXGI_FORMAT_BC5_UNORM, none},          {DXGI_FORMAT_BC6H_UF16, none},
        {DXGI_FORMAT_BC7_UNORM, none},          {DXGI_FORMAT_BC7_UNORM_SRGB, none},
    };
    struct Size {
        UINT64 w;
        UINT h;
    };
    const Size sizes[] = {{256, 256}, {512, 256}, {1024, 1024}, {1920, 1080}, {2048, 2048}, {4096, 2048}, {4096, 4096}};
    for (const Kind& k : kinds) {
        for (const Size& s : sizes) {
            for (const UINT16 mips : {UINT16(1), UINT16(0)}) {
                for (const UINT16 layers : {UINT16(1), UINT16(4), UINT16(6), UINT16(16)}) {
                    for (const UINT samples : {1u, 4u}) {
                        if (samples > 1 && (mips != 1 || !(k.flags & (rt | ds)))) continue;
                        const D3D12_RESOURCE_DESC desc =
                            texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, s.w, s.h, layers, mips, k.format, samples,
                                    k.flags);
                        char label[64];
                        std::snprintf(label, sizeof(label), "engine format %d samples %u", int(k.format), samples);
                        const Words h = allocation(hardware, desc);
                        const Words w = warp ? allocation(warp, desc) : Words{};
                        judge(r, engine_allocation_rule, std::string("GetResourceAllocationInfo ") + label,
                              s.w * s.h * layers, describe(desc), allocation(ours, desc), h, warp ? &w : nullptr);
                    }
                }
            }
        }
    }
}

void print(const char* title, const Groups& groups) {
    for (const auto& [name, g] : groups)
        std::printf("  %s %s (%u cases)\n    smallest: %s\n", title, name.c_str(), g.count, g.example.c_str());
}

bool enabled(const char* name) {
    char value[8] = {};
    return GetEnvironmentVariableA(name, value, sizeof(value)) != 0;
}
}  // namespace

// The allocation rules, on answers made up here, so they are checked where there is no Vulkan GPU or no host device.
bool check_allocation_rules() {
    bool ok = true;
    const auto expect = [&](bool good, const char* what) {
        if (!good) {
            std::fprintf(stderr, "FAIL allocation rules: %s\n", what);
            ok = false;
        }
    };
    const Words host{65536, 65536};
    for (const auto rule : {allocation_rule, engine_allocation_rule}) {
        expect(rule({65536, 4096}, host) == Verdict::error, "an alignment below the host's is an error");
        expect(rule({65536, 65536}, host) == Verdict::gap, "an equal answer reaching a rule is a gap");
        expect(rule({65536, 4194304}, host) == Verdict::gap, "an alignment above the host's is a gap");
        expect(rule({kInvalid, 0}, host) == Verdict::gap, "a refusal where the host answers is a gap");
        expect(rule({65536, 65536}, {kInvalid, 0}) == Verdict::error, "an answer where the host refuses is an error");
        expect(rule({65536, 4096}, {kInvalid, 0}) == Verdict::error, "an answer where the host refuses stays an error");
    }
    expect(engine_allocation_rule({32768, 65536}, host) == Verdict::error, "a size below the host's is an error");
    expect(allocation_rule({32768, 65536}, host) == Verdict::gap, "a size below the host's is a gap for the odd grid");
    return ok;
}

int main() {
    if (!check_allocation_rules()) return 1;
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (hostwine::host_is_wine()) return 77;  // every case is a comparison with the host
    const bool report = enabled("LEONIDA_SWEEP_REPORT");
    const bool show_gaps = report || enabled("LEONIDA_SWEEP_GAPS");
    size_t errors = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        Ref<IDXGIAdapter> hw_adapter, warp_adapter;
        Ref<ID3D12Device> hardware, warp;
        std::string hw_name;
        if (hostadapter::reference(factory.p, vk, false, &hw_adapter.p, hw_name))
            D3D12CreateDevice(hw_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hardware.p));
        std::string warp_name;
        if (hostadapter::reference(factory.p, vk, true, &warp_adapter.p, warp_name))
            D3D12CreateDevice(warp_adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p));
        if (!hardware.p && !warp.p) {
            std::fprintf(stderr, "no D3D12 reference device (hardware adapter or WARP)\n");
            return 1;
        }
        // With one reference only, that reference is the answer; with two, they must agree.
        ID3D12Device* primary = hardware.p ? hardware.p : warp.p;
        ID3D12Device* secondary = hardware.p && warp.p ? warp.p : nullptr;
        std::printf("query sweep: layer against %s%s%s\n", hardware.p ? hw_name.c_str() : "WARP",
                    secondary ? " and " : "", secondary ? "WARP" : "");

        Results r;
        sweep_formats(ours.p, primary, secondary, r);
        sweep_textures(ours.p, primary, secondary, r);
        sweep_engine_textures(ours.p, primary, secondary, r);
        std::printf("  %u answers compared, %u skipped where the hosts disagree; %zu wrong, %zu gaps (the layer knows "
                    "less than the host)\n",
                    r.compared, r.hosts_disagree, r.errors.size(), r.gaps.size());
        print("WRONG", r.errors);
        if (show_gaps) print("gap", r.gaps);
        errors = r.errors.size();
    }
    if (errors && !report) return 1;
    if (vk.validation_errors()) return 1;
    return 0;
}
