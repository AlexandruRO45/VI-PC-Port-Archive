// CreateSampler, CreateSampler2 and static-sampler validation parity against Windows D3D12 (WARP) on the
// Vulkan-backed device:
//   - invalid Filter, AddressU/V/W, MaxAnisotropy, ComparisonFunc (with comparison vs non-comparison filters),
//     MipLODBias, NaN MinLOD/MaxLOD, and unknown/incompatible Sampler2 flags leave the descriptor slot untouched;
//   - valid edge cases (ComparisonFunc == NONE / NEVER / ALWAYS, MaxAnisotropy == 0 / 1 / 16, anisotropic mip-point
//     filters, MinLOD > MaxLOD, +/-Inf LOD clamps, boundary MipLODBias [-16, 15.99]) encode valid GFX10 S# words
//     without overflowing S# bitfields or triggering Vulkan validation errors;
//   - every matrix entry's accept/refuse decision matches host WARP for both CreateSampler and CreateSampler2.
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include <dxgi1_4.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

constexpr unsigned char kSentinel = 0xcd;

struct Slots {
    Ref<ID3D12DescriptorHeap> heap;
    SIZE_T start = 0;
    UINT step = 0;
    UINT next = 0;
    static constexpr UINT kCount = 512;

    void reset() {
        heap.reset();
        start = 0;
        step = 0;
        next = 0;
    }

    bool create(ID3D12Device* device) {
        reset();
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kCount,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p)))) return false;
        start = heap->GetCPUDescriptorHandleForHeapStart().ptr;
        step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        return start != 0 && step != 0;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE take() {
        const D3D12_CPU_DESCRIPTOR_HANDLE h{start + SIZE_T(next++ % kCount) * step};
        std::memset(reinterpret_cast<void*>(h.ptr), kSentinel, step);
        return h;
    }

    bool written(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        const auto* p = reinterpret_cast<const unsigned char*>(h.ptr);
        for (UINT i = 0; i < step; ++i)
            if (p[i] != kSentinel) return true;
        return false;
    }

    leonida::gfx10::SamplerWords words(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        leonida::gfx10::SamplerWords w{};
        std::memcpy(w.data(), reinterpret_cast<const void*>(h.ptr), sizeof(w));
        return w;
    }
};

// Host WARP device wrapper that recreates the device whenever an invalid call removes it (e.g. under the D3D12
// debug layer or DXGI_ERROR_INVALID_CALL on CreateSampler*).
struct WarpHost {
    Ref<ID3D12Device11> device;
    Slots slots;
    bool available = false;

    bool init() {
        slots.reset();
        device.reset();
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
        Ref<IDXGIAdapter> warp;
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) return false;
        if (FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p)))) return false;
        if (!slots.create(device.p)) {
            device.reset();
            return false;
        }
        available = true;
        return true;
    }

    bool ensure_alive() {
        if (!device.p || device->GetDeviceRemovedReason() != S_OK) return init();
        return true;
    }
};

D3D12_SAMPLER_DESC base_desc() {
    D3D12_SAMPLER_DESC d{};
    d.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    d.AddressU = d.AddressV = d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    d.MaxLOD = D3D12_FLOAT32_MAX;
    return d;
}

D3D12_SAMPLER_DESC2 to_desc2(const D3D12_SAMPLER_DESC& d, D3D12_SAMPLER_FLAGS flags = D3D12_SAMPLER_FLAG_NONE) {
    D3D12_SAMPLER_DESC2 out{};
    out.Filter = d.Filter;
    out.AddressU = d.AddressU;
    out.AddressV = d.AddressV;
    out.AddressW = d.AddressW;
    out.MipLODBias = d.MipLODBias;
    out.MaxAnisotropy = d.MaxAnisotropy;
    out.ComparisonFunc = d.ComparisonFunc;
    std::memcpy(out.FloatBorderColor, d.BorderColor, sizeof(out.FloatBorderColor));
    out.MinLOD = d.MinLOD;
    out.MaxLOD = d.MaxLOD;
    out.Flags = flags;
    return out;
}

struct Case {
    const char* name;
    D3D12_SAMPLER_DESC desc;
    bool want;
};

std::vector<Case> build_matrix() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<Case> cases;
    const auto add = [&](const char* name, auto mutate, bool want) {
        D3D12_SAMPLER_DESC d = base_desc();
        mutate(d);
        cases.push_back({name, d, want});
    };

    // Baseline valid sampler.
    add("valid base", [](D3D12_SAMPLER_DESC&) {}, true);

    // AddressU / AddressV / AddressW bounds: [WRAP=1 .. MIRROR_ONCE=5].
    add("AddressU=0", [](D3D12_SAMPLER_DESC& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE(0); }, false);
    add("AddressV=0", [](D3D12_SAMPLER_DESC& d) { d.AddressV = D3D12_TEXTURE_ADDRESS_MODE(0); }, false);
    add("AddressW=0", [](D3D12_SAMPLER_DESC& d) { d.AddressW = D3D12_TEXTURE_ADDRESS_MODE(0); }, false);
    add("AddressU=WRAP(1)", [](D3D12_SAMPLER_DESC& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP; }, true);
    add("AddressV=MIRROR_ONCE(5)", [](D3D12_SAMPLER_DESC& d) { d.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE; }, true);
    add("AddressU=6", [](D3D12_SAMPLER_DESC& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE(6); }, false);
    add("AddressV=6", [](D3D12_SAMPLER_DESC& d) { d.AddressV = D3D12_TEXTURE_ADDRESS_MODE(6); }, false);
    add("AddressW=6", [](D3D12_SAMPLER_DESC& d) { d.AddressW = D3D12_TEXTURE_ADDRESS_MODE(6); }, false);
    add("AddressW=UINT_MAX", [](D3D12_SAMPLER_DESC& d) { d.AddressW = D3D12_TEXTURE_ADDRESS_MODE(0xffffffffu); }, false);

    // ComparisonFunc with non-comparison filter: ignored by D3D12.
    add("non-cmp ComparisonFunc=0", [](D3D12_SAMPLER_DESC& d) { d.ComparisonFunc = D3D12_COMPARISON_FUNC(0); }, true);
    add("non-cmp ComparisonFunc=9", [](D3D12_SAMPLER_DESC& d) { d.ComparisonFunc = D3D12_COMPARISON_FUNC(9); }, true);
    add("non-cmp ComparisonFunc=UINT_MAX",
        [](D3D12_SAMPLER_DESC& d) { d.ComparisonFunc = D3D12_COMPARISON_FUNC(0xffffffffu); }, true);

    // ComparisonFunc with comparison filter: 0 (NONE) .. 8 (ALWAYS) accepted; >= 9 refused.
    add("cmp ComparisonFunc=0(NONE)", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        d.ComparisonFunc = D3D12_COMPARISON_FUNC(0);
    }, true);
    add("cmp ComparisonFunc=1(NEVER)", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    }, true);
    add("cmp ComparisonFunc=8(ALWAYS)", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        d.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    }, true);
    add("cmp ComparisonFunc=9", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        d.ComparisonFunc = D3D12_COMPARISON_FUNC(9);
    }, false);
    add("cmp ComparisonFunc=UINT_MAX", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        d.ComparisonFunc = D3D12_COMPARISON_FUNC(0xffffffffu);
    }, false);

    // Filter enum validity: standard, comparison, min/max, anisotropic (mip point & linear), and invalid bits.
    add("Filter=ANISOTROPIC(0x55)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER_ANISOTROPIC; }, true);
    add("Filter=MIN_MAG_ANISO_MIP_POINT(0x54)",
        [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER_MIN_MAG_ANISOTROPIC_MIP_POINT; }, true);
    add("Filter=CMP_MIN_MAG_ANISO_MIP_POINT(0xd4)",
        [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_ANISOTROPIC_MIP_POINT; }, true);
    add("Filter=MIN_MIN_MAG_ANISO_MIP_POINT(0x154)",
        [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER_MINIMUM_MIN_MAG_ANISOTROPIC_MIP_POINT; }, true);
    add("Filter=MAX_MIN_MAG_ANISO_MIP_POINT(0x1d4)",
        [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER_MAXIMUM_MIN_MAG_ANISOTROPIC_MIP_POINT; }, true);
    add("Filter=0x02(bad mip)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x02); }, false);
    add("Filter=0x08(bad mag)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x08); }, false);
    add("Filter=0x20(bad min)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x20); }, false);
    add("Filter=0x40(aniso point)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x40); }, false);
    add("Filter=0x45(aniso min point)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x45); }, false);
    add("Filter=0x50(aniso mag point)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x50); }, false);
    add("Filter=0xc0(cmp aniso point)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0xc0); }, false);
    add("Filter=0x200(high bit)", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0x200); }, false);
    add("Filter=UINT_MAX", [](D3D12_SAMPLER_DESC& d) { d.Filter = D3D12_FILTER(0xffffffffu); }, false);

    // MaxAnisotropy: 0..16 accepted, >= 17 refused (for both point and anisotropic filters).
    add("point MaxAnisotropy=0", [](D3D12_SAMPLER_DESC& d) { d.MaxAnisotropy = 0; }, true);
    add("aniso MaxAnisotropy=0", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 0;
    }, true);
    add("aniso MaxAnisotropy=1", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 1;
    }, true);
    add("aniso MaxAnisotropy=16", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 16;
    }, true);
    add("point MaxAnisotropy=17", [](D3D12_SAMPLER_DESC& d) { d.MaxAnisotropy = 17; }, false);
    add("aniso MaxAnisotropy=17", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 17;
    }, false);
    add("aniso MaxAnisotropy=UINT_MAX", [](D3D12_SAMPLER_DESC& d) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 0xffffffffu;
    }, false);

    // MipLODBias [-16.0, 15.99] and NaN/Inf, plus MinLOD/MaxLOD NaN/Inf/inverted.
    add("MipLODBias=-16.0", [](D3D12_SAMPLER_DESC& d) { d.MipLODBias = -16.0f; }, true);
    add("MipLODBias=15.99", [](D3D12_SAMPLER_DESC& d) { d.MipLODBias = 15.99f; }, true);
    add("MipLODBias=-16.01", [](D3D12_SAMPLER_DESC& d) { d.MipLODBias = -16.01f; }, false);
    add("MipLODBias=16.0", [](D3D12_SAMPLER_DESC& d) { d.MipLODBias = 16.0f; }, false);
    add("MipLODBias=NaN", [=](D3D12_SAMPLER_DESC& d) { d.MipLODBias = qnan; }, false);
    add("MipLODBias=+Inf", [=](D3D12_SAMPLER_DESC& d) { d.MipLODBias = pinf; }, false);
    add("MipLODBias=-Inf", [=](D3D12_SAMPLER_DESC& d) { d.MipLODBias = ninf; }, false);
    add("MinLOD=NaN", [=](D3D12_SAMPLER_DESC& d) { d.MinLOD = qnan; }, false);
    add("MaxLOD=NaN", [=](D3D12_SAMPLER_DESC& d) { d.MaxLOD = qnan; }, false);
    add("MinLOD=+Inf", [=](D3D12_SAMPLER_DESC& d) { d.MinLOD = pinf; }, true);
    add("MinLOD=-Inf", [=](D3D12_SAMPLER_DESC& d) { d.MinLOD = ninf; }, true);
    add("MaxLOD=+Inf", [=](D3D12_SAMPLER_DESC& d) { d.MaxLOD = pinf; }, true);
    add("MaxLOD=-Inf", [=](D3D12_SAMPLER_DESC& d) { d.MaxLOD = ninf; }, true);
    add("MinLOD>MaxLOD", [](D3D12_SAMPLER_DESC& d) { d.MinLOD = 5.0f; d.MaxLOD = 1.0f; }, true);
    return cases;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.sampler_anisotropy = d.enabled.samplerAnisotropy != VK_FALSE;
        info.sampler_filter_minmax = d.sampler_filter_minmax;
        info.custom_border_color = d.custom_border_color;
        info.custom_border_color_without_format = d.custom_border_color_without_format;

        Ref<ID3D12Device11> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Slots slots;
        EXPECT(slots.create(device.p));

        // Null description and null destination handles leave slots untouched without crashing.
        {
            const D3D12_CPU_DESCRIPTOR_HANDLE h1 = slots.take();
            device->CreateSampler(nullptr, h1);
            EXPECT(!slots.written(h1));

            const D3D12_CPU_DESCRIPTOR_HANDLE h2 = slots.take();
            device->CreateSampler2(nullptr, h2);
            EXPECT(!slots.written(h2));

            const D3D12_SAMPLER_DESC valid = base_desc();
            const D3D12_SAMPLER_DESC2 valid2 = to_desc2(valid);
            device->CreateSampler(&valid, D3D12_CPU_DESCRIPTOR_HANDLE{0});
            device->CreateSampler2(&valid2, D3D12_CPU_DESCRIPTOR_HANDLE{0});
        }

        // Unknown CreateSampler2 flags are refused and leave the slot untouched.
        {
            const D3D12_SAMPLER_DESC2 bad_flags = to_desc2(base_desc(), D3D12_SAMPLER_FLAGS(0x4));
            const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
            device->CreateSampler2(&bad_flags, h);
            EXPECT(!slots.written(h));
        }

        // Verify S# encoding details for edge cases that are accepted.
        {
            namespace g = leonida::gfx10;

            // Comparison filter + ComparisonFunc == NONE (0): accepted, no comparison enabled in Vulkan.
            D3D12_SAMPLER_DESC cmp_none = base_desc();
            cmp_none.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
            cmp_none.ComparisonFunc = D3D12_COMPARISON_FUNC(0);
            const D3D12_CPU_DESCRIPTOR_HANDLE h_none = slots.take();
            device->CreateSampler(&cmp_none, h_none);
            EXPECT(slots.written(h_none));
            const g::Sampler s_none = g::decode(slots.words(h_none));
            EXPECT(s_none.compare == g::CompareFunction::never && !s_none.compare_never);
            EXPECT(!g::vulkan_sampler(s_none).compareEnable);

            // Comparison filter + ComparisonFunc == NEVER (1): compareEnable=VK_TRUE, compareOp=NEVER.
            D3D12_SAMPLER_DESC cmp_never = cmp_none;
            cmp_never.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            const D3D12_CPU_DESCRIPTOR_HANDLE h_never = slots.take();
            device->CreateSampler(&cmp_never, h_never);
            EXPECT(slots.written(h_never));
            const g::Sampler s_never = g::decode(slots.words(h_never));
            EXPECT(s_never.compare == g::CompareFunction::never && s_never.compare_never);
            EXPECT(g::vulkan_sampler(s_never).compareEnable &&
                   g::vulkan_sampler(s_never).compareOp == VK_COMPARE_OP_NEVER);

            // Comparison filter + ComparisonFunc == ALWAYS (8): compare == always (7).
            D3D12_SAMPLER_DESC cmp_always = cmp_none;
            cmp_always.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            const D3D12_CPU_DESCRIPTOR_HANDLE h_always = slots.take();
            device->CreateSampler(&cmp_always, h_always);
            EXPECT(slots.written(h_always));
            const g::Sampler s_always = g::decode(slots.words(h_always));
            EXPECT(s_always.compare == g::CompareFunction::always && !s_always.compare_never);

            // Anisotropic filter + MaxAnisotropy == 0: accepted and clamped to 1 in S#.
            D3D12_SAMPLER_DESC aniso0 = base_desc();
            aniso0.Filter = D3D12_FILTER_ANISOTROPIC;
            aniso0.MaxAnisotropy = 0;
            const D3D12_CPU_DESCRIPTOR_HANDLE h_aniso0 = slots.take();
            device->CreateSampler(&aniso0, h_aniso0);
            EXPECT(slots.written(h_aniso0));
            EXPECT(g::decode(slots.words(h_aniso0)).max_anisotropy == 1);
        }

        // Run the full matrix on our device (both CreateSampler and CreateSampler2) and compare against host WARP.
        WarpHost warp;
        const bool have_warp = warp.init();
        const std::vector<Case> matrix = build_matrix();
        std::size_t warp_compared = 0;
        for (const Case& c : matrix) {
            const D3D12_SAMPLER_DESC2 d2 = to_desc2(c.desc);

            const D3D12_CPU_DESCRIPTOR_HANDLE h1 = slots.take();
            device->CreateSampler(&c.desc, h1);
            const bool ours1 = slots.written(h1);

            const D3D12_CPU_DESCRIPTOR_HANDLE h2 = slots.take();
            device->CreateSampler2(&d2, h2);
            const bool ours2 = slots.written(h2);

            if (ours1 != c.want || ours2 != c.want) {
                std::fprintf(stderr, "%s: ours S1=%d S2=%d, want=%d\n", c.name, int(ours1), int(ours2), int(c.want));
                return 1;
            }
            if (ours1) EXPECT(slots.words(h1) == slots.words(h2));

            if (have_warp) {
                EXPECT(warp.ensure_alive());
                const D3D12_CPU_DESCRIPTOR_HANDLE wh1 = warp.slots.take();
                warp.device->CreateSampler(&c.desc, wh1);
                const bool warp1 = warp.slots.written(wh1);

                EXPECT(warp.ensure_alive());
                const D3D12_CPU_DESCRIPTOR_HANDLE wh2 = warp.slots.take();
                warp.device->CreateSampler2(&d2, wh2);
                const bool warp2 = warp.slots.written(wh2);

                if (ours1 != warp1 || ours2 != warp2) {
                    std::fprintf(stderr, "%s: ours S1=%d S2=%d, WARP S1=%d S2=%d\n", c.name, int(ours1), int(ours2),
                                 int(warp1), int(warp2));
                    return 1;
                }
                ++warp_compared;
            }
        }

        if (d.validation_errors()) {
            std::fprintf(stderr, "%u Vulkan validation errors\n", d.validation_errors());
            return 1;
        }
        std::printf("D3D12 sampler validation: %zu matrix cases passed (CreateSampler & CreateSampler2, %zu compared "
                    "with WARP), 0 Vulkan validation errors\n",
                    matrix.size(), warp_compared);
    }
    return 0;
}
