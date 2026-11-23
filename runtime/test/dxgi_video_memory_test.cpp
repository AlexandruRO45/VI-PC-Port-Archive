// Host-parity test for IDXGIAdapter3::QueryVideoMemoryInfo and SetVideoMemoryReservation
// (runtime/shims/dxgi_com.cpp). Compares the Vulkan-backed DXGI layer against live Windows DXGI
// (WARP and the first hardware adapter, each with an open D3D12 device):
//   - valid reservations at 0, AvailableForReservation / 2, and exact AvailableForReservation;
//   - upper-bound rejection (AvailableForReservation + 1 and UINT64_MAX) returning E_INVALIDARG
//     and leaving CurrentReservation unchanged (including when AvailableForReservation == 0);
//   - invalid node_index (> 0) and invalid DXGI_MEMORY_SEGMENT_GROUP returning E_INVALIDARG
//     without disturbing CurrentReservation or the QueryVideoMemoryInfo output struct.
// Exits 77 when neither a host D3D12 device nor a Vulkan device can be made. No shaders or game data.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "vk_test_device.h"
#include "../shims/dxgi_com.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", label, __LINE__, #condition); return 1; } \
} while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

bool is_filled_with(const DXGI_QUERY_VIDEO_MEMORY_INFO& info, unsigned char byte) {
    DXGI_QUERY_VIDEO_MEMORY_INFO expected{};
    std::memset(&expected, byte, sizeof(expected));
    return std::memcmp(&info, &expected, sizeof(info)) == 0;
}

int check_segment_group(const char* label, IDXGIAdapter3* adapter, IDXGIAdapter3* peer,
                        DXGI_MEMORY_SEGMENT_GROUP group) {
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);

    // 0 is always valid, including when AvailableForReservation == 0 (e.g. WARP NON_LOCAL).
    EXPECT(adapter->SetVideoMemoryReservation(0, group, 0) == S_OK);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == 0);

    // Mid-range reservation (AvailableForReservation / 2).
    const UINT64 half = info.AvailableForReservation / 2;
    EXPECT(adapter->SetVideoMemoryReservation(0, group, half) == S_OK);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == half);

    // Out-of-range reservation while CurrentReservation == half must fail with E_INVALIDARG and preserve half.
    EXPECT(adapter->SetVideoMemoryReservation(0, group, info.AvailableForReservation + 1) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(0, group, UINT64_MAX) == E_INVALIDARG);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == half);
    if (peer) {
        DXGI_QUERY_VIDEO_MEMORY_INFO peer_info{};
        EXPECT(peer->QueryVideoMemoryInfo(0, group, &peer_info) == S_OK);
        EXPECT(peer_info.CurrentReservation == half);
    }

    // Exact upper boundary (AvailableForReservation).
    const UINT64 exact = info.AvailableForReservation;
    EXPECT(adapter->SetVideoMemoryReservation(0, group, exact) == S_OK);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == exact);

    // Strictly above AvailableForReservation must fail with E_INVALIDARG and preserve exact.
    EXPECT(adapter->SetVideoMemoryReservation(0, group, info.AvailableForReservation + 1) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(0, group, UINT64_MAX) == E_INVALIDARG);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == exact);
    if (peer) {
        DXGI_QUERY_VIDEO_MEMORY_INFO peer_info{};
        EXPECT(peer->QueryVideoMemoryInfo(0, group, &peer_info) == S_OK);
        EXPECT(peer_info.CurrentReservation == exact);
    }

    // Invalid node_index / group calls must return E_INVALIDARG and leave CurrentReservation untouched.
    EXPECT(adapter->SetVideoMemoryReservation(1, group, 0) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(UINT_MAX, group, 0) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(2), 0) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(-1), 0) == E_INVALIDARG);
    EXPECT(adapter->SetVideoMemoryReservation(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(2), UINT64_MAX) ==
           E_INVALIDARG);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == exact);

    // Restore to 0.
    EXPECT(adapter->SetVideoMemoryReservation(0, group, 0) == S_OK);
    EXPECT(adapter->QueryVideoMemoryInfo(0, group, &info) == S_OK);
    EXPECT(info.CurrentReservation == 0);
    if (peer) {
        DXGI_QUERY_VIDEO_MEMORY_INFO peer_info{};
        EXPECT(peer->QueryVideoMemoryInfo(0, group, &peer_info) == S_OK);
        EXPECT(peer_info.CurrentReservation == 0);
    }
    return 0;
}

int check_adapter(const char* label, IDXGIAdapter3* adapter, IDXGIAdapter3* peer, bool is_layer) {
    if (check_segment_group(label, adapter, peer, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)) return 1;
    if (check_segment_group(label, adapter, peer, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)) return 1;

    // QueryVideoMemoryInfo invalid arguments: E_INVALIDARG and output struct untouched.
    DXGI_QUERY_VIDEO_MEMORY_INFO sentinel{};
    std::memset(&sentinel, 0xaa, sizeof(sentinel));
    EXPECT(adapter->QueryVideoMemoryInfo(1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &sentinel) == E_INVALIDARG);
    EXPECT(is_filled_with(sentinel, 0xaa));

    std::memset(&sentinel, 0xaa, sizeof(sentinel));
    EXPECT(adapter->QueryVideoMemoryInfo(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(2), &sentinel) == E_INVALIDARG);
    EXPECT(is_filled_with(sentinel, 0xaa));

    std::memset(&sentinel, 0xaa, sizeof(sentinel));
    EXPECT(adapter->QueryVideoMemoryInfo(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(-1), &sentinel) == E_INVALIDARG);
    EXPECT(is_filled_with(sentinel, 0xaa));

    EXPECT(adapter->QueryVideoMemoryInfo(1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, nullptr) == E_INVALIDARG);
    EXPECT(adapter->QueryVideoMemoryInfo(0, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(2), nullptr) == E_INVALIDARG);
    if (is_layer) {
        // Native Windows DXGI dereferences nullptr when node_index and group are valid; the layer safely returns
        // E_INVALIDARG.
        EXPECT(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, nullptr) == E_INVALIDARG);
    }

    DXGI_ADAPTER_DESC2 desc{};
    EXPECT(adapter->GetDesc2(&desc) == S_OK);
    char name[128] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
    std::printf("%s: %s passed\n", label, name);
    return 0;
}

int run_host(IDXGIAdapter1* adapter1, const char* label) {
    Ref<IDXGIAdapter3> adapter3;
    if (FAILED(adapter1->QueryInterface(IID_PPV_ARGS(&adapter3.p)))) {
        std::printf("%s: no IDXGIAdapter3, skipped\n", label);
        return 77;
    }
    // WDDM requires an active D3D12 device on the adapter for SetVideoMemoryReservation to accept reservations.
    Ref<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter1, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p)))) {
        std::printf("%s: no D3D12 device, skipped\n", label);
        return 77;
    }
    return check_adapter(label, adapter3.p, nullptr, false);
}

}  // namespace

static void no_dialogs() {
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
}

int main() {
    no_dialogs();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* label = "dxgi_video_memory_test";
    int ran = 0;

    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    Ref<IDXGIFactory4> host_factory;
    if (wine) std::printf("host DXGI: under Wine, skipped\n");
    if (!wine && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&host_factory.p)))) {
        Ref<IDXGIAdapter1> warp;
        if (SUCCEEDED(host_factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
            const int r = run_host(warp.p, "host WARP");
            if (r == 1) return 1;
            ran += r == 0;
        }
        for (UINT i = 0;; ++i) {
            Ref<IDXGIAdapter1> adapter;
            if (host_factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            const int r = run_host(adapter.p, "host hardware adapter");
            if (r == 1) return 1;
            ran += r == 0;
            break;
        }
    }

    vktest::Device d;
    if (d.create()) {
        label = "Vulkan layer";
        leonida::VulkanContext context(d.device_proc, d.device, true);
        Ref<IDXGIFactory4> factory;
        EXPECT(SUCCEEDED(rt::dxgi::create_factory(context, d.instance_proc, d.instance, d.physical, d.device_proc,
                                                  d.queue, d.family, IID_PPV_ARGS(&factory.p))));
        Ref<IDXGIAdapter1> adapter1;
        EXPECT(SUCCEEDED(factory->EnumAdapters1(0, &adapter1.p)));
        Ref<IDXGIAdapter3> adapter3;
        EXPECT(SUCCEEDED(adapter1->QueryInterface(IID_PPV_ARGS(&adapter3.p))));
        DXGI_ADAPTER_DESC1 desc1{};
        EXPECT(SUCCEEDED(adapter1->GetDesc1(&desc1)));
        Ref<IDXGIAdapter3> peer;
        EXPECT(SUCCEEDED(factory->EnumAdapterByLuid(desc1.AdapterLuid, IID_PPV_ARGS(&peer.p))));
        if (check_adapter(label, adapter3.p, peer.p, true)) return 1;
        ++ran;
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
    } else {
        std::printf("Vulkan layer: no device, skipped\n");
    }

    if (!ran) {
        std::printf("SKIP dxgi_video_memory_test: neither a host D3D12 device nor a Vulkan device\n");
        return 77;
    }
    std::printf("dxgi_video_memory_test: %d backend(s) passed\n", ran);
    return 0;
}
