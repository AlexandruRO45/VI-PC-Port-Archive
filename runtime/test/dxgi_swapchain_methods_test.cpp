// IDXGISwapChain1..4 calls that take no description to refuse, compared with the host DXGI on the hardware adapter
// and on WARP: colour space support and selection, HDR metadata, rotation, background colour, the interfaces a
// swap chain answers, the description it reports and the present count. A row counts only where both host adapters
// agree. Not compared: a call the host faults on, GetFrameStatistics (the host's counts depend on the compositor) and
// the order of GetCurrentBackBufferIndex (the host cycles 0, 1, 2..., the layer follows the order Vulkan hands the
// images out). No shaders or game data.
// LEONIDA_PROBE (any value) prints every row.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "../shims/d3d12_com.h"
#include "../shims/dxgi_com.h"
#include "vulkan_loader.h"
#include "../src/vulkan_device_filter.h"

#include <dxgi1_6.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// What the last case saw, in words, for the rows that differ; filled in by note().
std::string& detail() {
    static std::string text;
    return text;
}
template<class... Args>
void note(const char* format, Args... args) {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), format, static_cast<long long>(args)...);
    detail() += buffer;
}

std::int64_t hr(HRESULT h) { return static_cast<std::int64_t>(static_cast<std::uint32_t>(h)); }
std::int64_t mix(std::int64_t a, std::int64_t b) { return (a * 1000003) ^ b; }

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) { return DefWindowProcW(hwnd, msg, wp, lp); }

HWND create_window() {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaDxgiSwapchainMethodsWnd";
    RegisterClassW(&wc);
    RECT rc{0, 0, 128, 128};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExW(0, wc.lpszClassName, L"Leonida DXGI methods", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance,
                           nullptr);
}

struct Target {
    std::string name;
    IDXGIFactory2* factory = nullptr;
    ID3D12CommandQueue* queue = nullptr;
};

// A flip-model swap chain of `format` and `buffers` buffers on a fresh window, handed to `body`.
std::int64_t with_swapchain(const Target& t, DXGI_FORMAT format, UINT buffers,
                            const std::function<std::int64_t(IDXGISwapChain3*, HWND)>& body) {
    HWND hwnd = create_window();
    if (!hwnd) return -1;
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = 128;
    desc.Height = 128;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = buffers;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    Ref<IDXGISwapChain1> sc1;
    std::int64_t result = -2;
    if (SUCCEEDED(t.factory->CreateSwapChainForHwnd(t.queue, hwnd, &desc, nullptr, nullptr, &sc1.p))) {
        Ref<IDXGISwapChain3> sc3;
        result = SUCCEEDED(sc1->QueryInterface(IID_PPV_ARGS(&sc3.p))) ? body(sc3.p, hwnd) : -3;
    }
    DestroyWindow(hwnd);
    return result;
}

struct Case {
    const char* name;
    std::function<std::int64_t(const Target&)> fn;
};

std::vector<Case> make_cases() {
    std::vector<Case> cases;
    const auto add = [&](const char* name, std::function<std::int64_t(const Target&)> fn) {
        cases.push_back({name, std::move(fn)});
    };
    const struct { const char* name; DXGI_FORMAT format; } formats[] = {
        {"B8G8R8A8", DXGI_FORMAT_B8G8R8A8_UNORM}, {"R8G8B8A8", DXGI_FORMAT_R8G8B8A8_UNORM},
        {"R10G10B10A2", DXGI_FORMAT_R10G10B10A2_UNORM}, {"R16G16B16A16F", DXGI_FORMAT_R16G16B16A16_FLOAT}};
    static std::vector<std::string> names;  // case names must outlive the table
    names.reserve(256);
    const auto label = [&](const std::string& s) {
        names.push_back(s);
        return names.back().c_str();
    };
    for (const auto& f : formats) {
        const DXGI_FORMAT format = f.format;
        add(label(std::string("colour spaces: CheckColorSpaceSupport, ") + f.name), [format](const Target& t) {
            return with_swapchain(t, format, 2, [](IDXGISwapChain3* sc, HWND) {
                std::int64_t v = 0;
                for (int cs = 0; cs < 26; ++cs) {
                    UINT flags = 0xdeadbeef;
                    const HRESULT r = sc->CheckColorSpaceSupport(static_cast<DXGI_COLOR_SPACE_TYPE>(cs), &flags);
                    v = mix(v, mix(hr(r), flags));
                    note("%lld:%llx/%llx ", cs, hr(r) & 0xffffffffll, flags);
                }
                return v;
            });
        });
        add(label(std::string("colour spaces: SetColorSpace1 each, ") + f.name), [format](const Target& t) {
            return with_swapchain(t, format, 2, [](IDXGISwapChain3* sc, HWND) {
                std::int64_t v = 0;
                for (int cs = 0; cs < 26; ++cs) {
                    const HRESULT r = sc->SetColorSpace1(static_cast<DXGI_COLOR_SPACE_TYPE>(cs));
                    v = mix(v, hr(r));
                    note("%lld:%llx ", cs, hr(r) & 0xffffffffll);
                }
                return v;
            });
        });
    }
    add("colour spaces: out-of-range values", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            std::int64_t v = 0;
            for (int cs : {26, 27, 28, 29, 30, 31, 32, 33, 34, 40, 50, 64, 100, 255, 256, 0x7fffffff, -1, -2}) {
                UINT flags = 0xdeadbeef;
                const HRESULT c = sc->CheckColorSpaceSupport(static_cast<DXGI_COLOR_SPACE_TYPE>(cs), &flags);
                const HRESULT sh = sc->SetColorSpace1(static_cast<DXGI_COLOR_SPACE_TYPE>(cs));
                v = mix(v, mix(mix(hr(c), flags), hr(sh)));
                note("%lld:%llx/%llx/%llx ", cs, hr(c) & 0xffffffffll, flags, hr(sh) & 0xffffffffll);
            }
            return v;
        });
    });
    add("colour spaces: SetColorSpace1 then GetDesc1 format", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            const HRESULT r = sc->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
            DXGI_SWAP_CHAIN_DESC1 d{};
            sc->GetDesc1(&d);
            return mix(hr(r), d.Format);
        });
    });
    add("hdr: IDXGISwapChain4 is answered", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_R10G10B10A2_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain4> sc4;
            return hr(sc->QueryInterface(IID_PPV_ARGS(&sc4.p)));
        });
    });
    for (const auto& f : formats) add(label(std::string("hdr: SetHDRMetaData types and sizes, ") + f.name), [format = f.format](const Target& t) {
        return with_swapchain(t, format, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain4> sc4;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc4.p)))) return std::int64_t{-4};
            DXGI_HDR_METADATA_HDR10 meta{};
            meta.MaxMasteringLuminance = 1000;
            meta.MaxContentLightLevel = 1000;
            std::int64_t v = 0;
            const auto step = [&](const char* what, HRESULT r) {
                v = mix(v, hr(r));
                detail() += what;
                note(":%llx ", hr(r) & 0xffffffffll);
            };
            step("none/0/null", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, 0, nullptr));
            step("hdr10/40", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(meta), &meta));
            step("hdr10/39", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(meta) - 1, &meta));
            step("hdr10/41", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(meta) + 1, &meta));
            step("hdr10/0/null", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, 0, nullptr));
            step("hdr10/40/null", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(meta), nullptr));
            step("none/40", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, sizeof(meta), &meta));
            step("none/0/ptr", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, 0, &meta));
            step("hdr10plus/40", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10PLUS, sizeof(meta), &meta));
            step("hdr10plus/0", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10PLUS, 0, nullptr));
            step("type7", sc4->SetHDRMetaData(static_cast<DXGI_HDR_METADATA_TYPE>(7), sizeof(meta), &meta));
            meta.MaxMasteringLuminance = 0xffffffffu;
            meta.MinMasteringLuminance = 0xffffu;
            step("hdr10/extreme", sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(meta), &meta));
            return v;
        });
    });
    add("rotation: default and each value", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
            std::int64_t v = mix(hr(sc1->GetRotation(&rotation)), rotation);
            note("default %lld ", rotation);
            for (int r : {0, 1, 2, 3, 4, 9}) {
                const HRESULT set = sc1->SetRotation(static_cast<DXGI_MODE_ROTATION>(r));
                rotation = DXGI_MODE_ROTATION_UNSPECIFIED;
                const HRESULT get = sc1->GetRotation(&rotation);
                v = mix(v, mix(mix(hr(set), hr(get)), rotation));
                note("%lld:%llx/%llx/%lld ", r, hr(set) & 0xffffffffll, hr(get) & 0xffffffffll, rotation);
            }
            return v;
        });
    });
    add("background colour: default, set, get", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            DXGI_RGBA color{9, 9, 9, 9};
            std::int64_t v = mix(hr(sc1->GetBackgroundColor(&color)), mix(mix(static_cast<std::int64_t>(color.r * 100), std::int64_t(color.g * 100)), mix(std::int64_t(color.b * 100), std::int64_t(color.a * 100))));
            note("default %lld,%lld ", static_cast<std::int64_t>(color.r * 100), std::int64_t(color.a * 100));
            const DXGI_RGBA wanted{0.25f, 0.5f, 0.75f, 1.0f};
            v = mix(v, hr(sc1->SetBackgroundColor(&wanted)));
            color = {9, 9, 9, 9};
            v = mix(v, mix(hr(sc1->GetBackgroundColor(&color)), mix(mix(static_cast<std::int64_t>(color.r * 100), std::int64_t(color.g * 100)), mix(std::int64_t(color.b * 100), std::int64_t(color.a * 100)))));
            return v;
        });
    });
    add("background colour: out-of-range components", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            std::int64_t v = 0;
            for (const DXGI_RGBA c : {DXGI_RGBA{2.0f, 0.0f, 0.0f, 1.0f}, DXGI_RGBA{0.0f, -0.1f, 0.0f, 1.0f},
                                      DXGI_RGBA{0.0f, 0.0f, 0.0f, 1.5f}, DXGI_RGBA{1.0f, 1.0f, 1.0f, 1.0f},
                                      DXGI_RGBA{0.0f, 0.0f, 0.0f, 0.0f}}) {
                const HRESULT r = sc1->SetBackgroundColor(&c);
                v = mix(v, hr(r));
                note("%llx ", hr(r) & 0xffffffffll);
            }
            return v;
        });
    });
    add("temporary mono is unsupported", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            return std::int64_t{sc1->IsTemporaryMonoSupported() != FALSE};
        });
    });
    add("interfaces: QueryInterface chain", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            const IID iids[] = {__uuidof(IDXGIObject), __uuidof(IDXGIDeviceSubObject), __uuidof(IDXGISwapChain),
                                __uuidof(IDXGISwapChain1), __uuidof(IDXGISwapChain2), __uuidof(IDXGISwapChain3),
                                __uuidof(IDXGISwapChain4), __uuidof(IDXGIFactory), __uuidof(IDXGIAdapter),
                                __uuidof(ID3D12Resource)};
            std::int64_t v = 0;
            for (const IID& iid : iids) {
                void* o = nullptr;
                const HRESULT r = sc->QueryInterface(iid, &o);
                if (o) static_cast<IUnknown*>(o)->Release();
                v = mix(v, hr(r));
                note("%llx ", hr(r) & 0xffffffffll);
            }
            return v;
        });
    });
    add("device: GetDevice answers the queue", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [&t](IDXGISwapChain3* sc, HWND) {
            Ref<ID3D12CommandQueue> q;
            const HRESULT r = sc->GetDevice(IID_PPV_ARGS(&q.p));
            Ref<ID3D12Device> d;
            const HRESULT wrong = sc->GetDevice(IID_PPV_ARGS(&d.p));
            return mix(mix(hr(r), q.p == t.queue), hr(wrong));
        });
    });
    add("parent: GetParent factory", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGIFactory2> f;
            const HRESULT r = sc->GetParent(IID_PPV_ARGS(&f.p));
            Ref<IDXGIAdapter> a;
            const HRESULT wrong = sc->GetParent(IID_PPV_ARGS(&a.p));
            return mix(hr(r), hr(wrong));
        });
    });
    add("present: GetLastPresentCount", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            UINT c = 99;
            std::int64_t v = mix(hr(sc->GetLastPresentCount(&c)), c);
            for (int i = 0; i < 3; ++i) sc->Present(0, 0);
            v = mix(v, mix(hr(sc->GetLastPresentCount(&c)), c));
            sc->Present(0, DXGI_PRESENT_TEST);
            v = mix(v, mix(hr(sc->GetLastPresentCount(&c)), c));
            return v;
        });
    });
    add("fullscreen: state of a windowed chain", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            BOOL fs = TRUE;
            Ref<IDXGIOutput> out;
            const HRESULT r = sc->GetFullscreenState(&fs, &out.p);
            const HRESULT off = sc->SetFullscreenState(FALSE, nullptr);
            return mix(mix(hr(r), fs), mix(out.p != nullptr, hr(off)));
        });
    });
    add("fullscreen: GetFullscreenDesc of a windowed chain", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            DXGI_SWAP_CHAIN_FULLSCREEN_DESC d{};
            const HRESULT r = sc1->GetFullscreenDesc(&d);
            note("hr %llx windowed %lld ", hr(r) & 0xffffffffll, d.Windowed);
            note("rate %lld/%lld ", d.RefreshRate.Numerator, d.RefreshRate.Denominator);
            note("scaling %lld order %lld ", d.Scaling, d.ScanlineOrdering);
            return mix(mix(hr(r), d.Windowed), mix(mix(d.RefreshRate.Numerator, d.RefreshRate.Denominator),
                                                    mix(d.Scaling, d.ScanlineOrdering)));
        });
    });
    add("GetHwnd and GetCoreWindow", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND hwnd) {
            Ref<IDXGISwapChain1> sc1;
            if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc1.p)))) return std::int64_t{-4};
            HWND got = nullptr;
            const HRESULT r = sc1->GetHwnd(&got);
            void* core = reinterpret_cast<void*>(1);
            const HRESULT c = sc1->GetCoreWindow(__uuidof(IUnknown), &core);
            return mix(mix(hr(r), got == hwnd), mix(hr(c), core == nullptr));
        });
    });
    add("GetDesc1 and GetDesc of a flip chain", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_R8G8B8A8_UNORM, 3, [](IDXGISwapChain3* sc, HWND hwnd) {
            DXGI_SWAP_CHAIN_DESC d{};
            const HRESULT r = sc->GetDesc(&d);
            DXGI_SWAP_CHAIN_DESC1 d1{};
            sc->GetDesc1(&d1);
            note("w %lld fmt %lld bufs %lld ", d.BufferDesc.Width, d.BufferDesc.Format, d.BufferCount);
            note("windowed %lld swap %lld usage %llx ", d.Windowed, d.SwapEffect, d.BufferUsage);
            note("refresh %lld/%lld ", d.BufferDesc.RefreshRate.Numerator, d.BufferDesc.RefreshRate.Denominator);
            note("scanline %lld scaling %lld ", d.BufferDesc.ScanlineOrdering, d.BufferDesc.Scaling);
            note("d1 scaling %lld alpha %lld stereo %lld ", d1.Scaling, d1.AlphaMode, d1.Stereo);
            return mix(mix(mix(hr(r), d.BufferDesc.Width), mix(d.BufferDesc.Format, d.BufferCount)),
                       mix(mix(d.OutputWindow == hwnd, d.Windowed), mix(mix(d.SwapEffect, d.BufferUsage), mix(d1.Scaling, mix(d1.AlphaMode, d1.Stereo)))));
        });
    });
    add("GetBuffer: out of range, wrong interface", [](const Target& t) {
        return with_swapchain(t, DXGI_FORMAT_B8G8R8A8_UNORM, 2, [](IDXGISwapChain3* sc, HWND) {
            Ref<ID3D12Resource> r;
            std::int64_t v = hr(sc->GetBuffer(2, IID_PPV_ARGS(&r.p)));
            Ref<ID3D12Fence> wrong;
            v = mix(v, hr(sc->GetBuffer(0, IID_PPV_ARGS(&wrong.p))));
            Ref<ID3D12Resource> good;
            v = mix(v, hr(sc->GetBuffer(1, IID_PPV_ARGS(&good.p))));
            if (good.p) {
                const D3D12_RESOURCE_DESC d = good->GetDesc();
                v = mix(v, mix(mix(d.Width, d.Height), mix(d.Format, d.Flags)));
            }
            return v;
        });
    });
    return cases;
}

std::vector<std::string>& details() {
    static std::vector<std::string> all;
    return all;
}

std::vector<std::int64_t> run(const Target& t, const std::vector<Case>& cases) {
    std::vector<std::int64_t> rows;
    for (const Case& c : cases) {
        detail().clear();
        rows.push_back(c.fn(t));
        details().push_back(detail());
    }
    return rows;
}
}  // namespace

int main() {
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP: vulkan-1.dll not available\n");
        return 77;
    }
    const auto create_inst = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    const char* inst_exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = inst_exts;
    VkInstance instance = VK_NULL_HANDLE;
    const VkResult created = create_inst(&ici, nullptr, &instance);
    if (created == VK_ERROR_INCOMPATIBLE_DRIVER) {
        std::printf("SKIP: no Vulkan driver\n");
        return 77;
    }
    if (created != VK_SUCCESS) return 1;
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    const auto enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    const auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(instance, "vkGetPhysicalDeviceProperties2"));
    const auto get_qf = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto create_dev = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
    std::uint32_t count = 0;
    enum_devs(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    enum_devs(instance, &count, physicals.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    int best_rank = -1;
    for (auto candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        get_props(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !leonida::vulkan_device_wanted(props.deviceName)) continue;
        std::uint32_t fcount = 0;
        get_qf(candidate, &fcount, nullptr);
        std::vector<VkQueueFamilyProperties> families(fcount);
        get_qf(candidate, &fcount, families.data());
        for (std::uint32_t i = 0; i < fcount; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (const int rank = leonida::vulkan_device_rank(static_cast<int>(props.deviceType), props.vendorID);
                rank > best_rank) {
                best_rank = rank;
                physical = candidate;
                family = i;
            }
            break;
        }
    }
    if (!physical) {
        std::printf("SKIP: no Vulkan 1.3 GPU found\n");
        return leonida::vulkan_device_filter().empty() ? 77 : 1;
    }
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.separateDepthStencilLayouts = VK_TRUE;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char* dev_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &dev_ext;
    VkDevice vk_dev = VK_NULL_HANDLE;
    if (create_dev(physical, &dci, nullptr, &vk_dev) != VK_SUCCESS) return 1;
    VkQueue vk_queue = VK_NULL_HANDLE;
    reinterpret_cast<PFN_vkGetDeviceQueue>(gdpa(vk_dev, "vkGetDeviceQueue"))(vk_dev, family, 0, &vk_queue);
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    get_props2(physical, &props2);
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gipa;
    info.instance = instance;
    info.physical = physical;
    info.device = vk_dev;
    info.queue = vk_queue;
    info.queue_family = family;
    info.separate_depth_stencil_layouts = true;
    Ref<ID3D12Device> dev;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p)))) return 1;
    Ref<IDXGIFactory4> factory;
    if (FAILED(rt::d3d12::create_dxgi_factory(dev.p, IID_PPV_ARGS(&factory.p)))) return 1;
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    Ref<ID3D12CommandQueue> queue;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p)))) return 1;
    Ref<IDXGIFactory2> factory2;
    if (FAILED(factory->QueryInterface(IID_PPV_ARGS(&factory2.p)))) return 1;

    const bool probe = GetEnvironmentVariableA("LEONIDA_PROBE", nullptr, 0) != 0;
    const std::vector<Case> cases = make_cases();
    Target mine{"layer", factory2.p, queue.p};
    const std::vector<std::int64_t> ours = run(mine, cases);
    const std::vector<std::string> our_details = details();
    details().clear();

    std::vector<std::vector<std::int64_t>> hosts;
    std::vector<std::string> host_names;
    std::vector<std::string> host_details;
    Ref<IDXGIFactory4> host_factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&host_factory.p)))) {
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            if (warp) {
                if (FAILED(host_factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
            } else if (FAILED(host_factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p)))) {
                if (FAILED(host_factory->EnumAdapters(0, &adapter.p))) continue;
            }
            Ref<ID3D12Device> host;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            Ref<ID3D12CommandQueue> host_queue;
            if (FAILED(host->CreateCommandQueue(&qd, IID_PPV_ARGS(&host_queue.p)))) continue;
            DXGI_ADAPTER_DESC ad{};
            adapter->GetDesc(&ad);
            char name[128] = {};
            for (int i = 0; i < 127 && ad.Description[i]; ++i) name[i] = static_cast<char>(ad.Description[i] < 128 ? ad.Description[i] : '?');
            Target target{name, host_factory.p, host_queue.p};
            hosts.push_back(run(target, cases));
            host_names.push_back(name);
            if (host_details.empty()) host_details = details();
            details().clear();
        }
    }
    int mismatches = 0, compared = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        bool agree = !hosts.empty();
        for (const auto& h : hosts) agree = agree && h[i] == hosts[0][i];
        const bool differs = agree && ours[i] != hosts[0][i];
        if (probe) {
            std::printf("%s | layer %llx", cases[i].name, static_cast<unsigned long long>(ours[i]));
            for (const auto& h : hosts) std::printf(" | host %llx", static_cast<unsigned long long>(h[i]));
            std::printf("%s\n", differs ? "  <-- DIFF" : agree ? "" : "  (hosts differ)");
        }
        if (!agree) continue;
        ++compared;
        if (differs) {
            ++mismatches;
            if (probe) std::printf("    layer: %s\n    host:  %s\n", our_details[i].c_str(), host_details[i].c_str());
            if (!probe)
                std::fprintf(stderr, "%s: layer %llx, host %llx\n", cases[i].name, static_cast<unsigned long long>(ours[i]),
                             static_cast<unsigned long long>(hosts[0][i]));
        }
    }
    std::printf("dxgi swap chain methods: %d rows compared with %zu host adapters, %d differ\n", compared, hosts.size(),
                mismatches);
    return mismatches ? 1 : 0;
}
