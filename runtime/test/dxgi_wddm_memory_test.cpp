// Compare the shim and native DXGI on the same adapter and within the same process. No window or game files.
#include "../shims/dxgi_com.h"
#include "host_wine.h"
#include "vk_test_device.h"

#include <cstdio>
#include <cstring>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr auto local = DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
constexpr auto nonlocal = DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL;
constexpr UINT64 mib = 1024 * 1024;

struct RestoreReservation {
    IDXGIAdapter3* adapter;
    UINT64 bytes;
    ~RestoreReservation() { adapter->SetVideoMemoryReservation(0, local, bytes); }
};

bool check(bool ok, const char* message) {
    if (!ok) std::fprintf(stderr, "%s\n", message);
    return ok;
}
}  // namespace

int main() {
    if (hostwine::host_is_wine()) return 77;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    const auto get_props = vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2");
    if (!get_props) return 77;
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    get_props(vk.physical, &props);
    if (!id.deviceLUIDValid || props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) return 77;
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    Ref<IDXGIFactory4> host_factory;
    Ref<IDXGIAdapter3> host;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&host_factory.p))) ||
        FAILED(host_factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&host.p)))) return 77;

    DXGI_QUERY_VIDEO_MEMORY_INFO initial{};
    if (FAILED(host->QueryVideoMemoryInfo(0, local, &initial)) || initial.AvailableForReservation < 64 * mib) {
        std::puts("SKIP: native adapter cannot reserve 64 MiB of local memory");
        return 77;
    }
    RestoreReservation restore{host.p, initial.CurrentReservation};
    leonida::VulkanContext context(vk.device_proc, vk.device, true);
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter3> layer, second;
    if (!check(SUCCEEDED(rt::dxgi::create_factory(context, vk.instance_proc, vk.instance, vk.physical, vk.device_proc,
                                                  vk.queue, vk.family, IID_PPV_ARGS(&factory.p))) &&
                   SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&layer.p))) &&
                   SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&second.p))),
               "could not obtain both shim handles by the Vulkan adapter LUID")) return 1;

    auto has_reservation = [](IDXGIAdapter3* adapter, UINT64 bytes) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        return SUCCEEDED(adapter->QueryVideoMemoryInfo(0, local, &info)) && info.CurrentReservation == bytes;
    };
    bool ok = check(SUCCEEDED(host->SetVideoMemoryReservation(0, local, 8 * mib)) &&
                        has_reservation(host.p, 8 * mib),
                    "native DXGI did not accept the 8 MiB control");
    ok &= check(has_reservation(layer.p, 8 * mib) && has_reservation(second.p, 8 * mib),
                "shim handles do not see the native 8 MiB reservation");
    ok &= check(SUCCEEDED(layer->SetVideoMemoryReservation(0, local, 32 * mib)) &&
                    has_reservation(host.p, 32 * mib) && has_reservation(second.p, 32 * mib),
                "native DXGI and the second shim handle do not see the shim's 32 MiB reservation");
    ok &= check(SUCCEEDED(layer->SetVideoMemoryReservation(0, local, initial.CurrentReservation)) &&
                    has_reservation(host.p, initial.CurrentReservation),
                "clearing the shim reservation did not restore the native reservation");

    DXGI_QUERY_VIDEO_MEMORY_INFO host_nonlocal{}, layer_nonlocal{};
    if (SUCCEEDED(host->QueryVideoMemoryInfo(0, nonlocal, &host_nonlocal)) &&
        host_nonlocal.Budget == 0 && host_nonlocal.AvailableForReservation == 0) {
        ok &= check(SUCCEEDED(layer->QueryVideoMemoryInfo(0, nonlocal, &layer_nonlocal)) &&
                        layer_nonlocal.Budget == 0 && layer_nonlocal.CurrentUsage == 0 &&
                        layer_nonlocal.AvailableForReservation == 0 && layer_nonlocal.CurrentReservation == 0,
                    "the shim reported memory in an empty non-local segment");
        constexpr UINT64 request = 32 * mib;
        const HRESULT host_hr = host->SetVideoMemoryReservation(0, nonlocal, request);
        const HRESULT layer_hr = layer->SetVideoMemoryReservation(0, nonlocal, request);
        ok &= check(host_hr == E_INVALIDARG && layer_hr == host_hr &&
                        SUCCEEDED(layer->QueryVideoMemoryInfo(0, nonlocal, &layer_nonlocal)) &&
                        layer_nonlocal.CurrentReservation == 0,
                    "a nonzero reservation on the empty non-local segment was not rejected without changing state");
        if (SUCCEEDED(host_hr)) host->SetVideoMemoryReservation(0, nonlocal, 0);
        if (SUCCEEDED(layer_hr)) layer->SetVideoMemoryReservation(0, nonlocal, 0);
        if (ok) std::puts("empty non-local segment: native and shim reject the same request");
    }
    if (ok) std::puts("local reservation changes are visible through native DXGI and both shim handles");
    return ok && vk.validation_errors() == 0 ? 0 : 1;
}
