// GetGPUVirtualAddress of buffers through the Xbox-ABI device proxy on the Vulkan device. A title builds root
// descriptors and V#s from this address, so a buffer that answers 0 sends the GPU to address 0. Every 1 KiB buffer
// must answer a non-zero, aligned address that no other live buffer shares, whatever its heap and initial state.
// The test makes the device twice: with bufferDeviceAddress (the device the game gets) and without it, where there is no
// GPU address arena and the address must still not be 0.
// No game data or shaders.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT64 kBytes = 1024;
constexpr UINT64 kMinimumAlignment = 256;  // what a V# or a root descriptor may assume of a buffer's address

struct Case {
    const char* name;
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_STATES state;
    D3D12_RESOURCE_FLAGS flags;
};

constexpr Case kCases[] = {
    {"default COMMON", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_NONE},
    {"default COPY_DEST", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE},
    {"default COPY_SOURCE", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_FLAG_NONE},
    {"default GENERIC_READ", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE},
    {"default UAV", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS},
    {"default COMMON + UAV flag", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS},
    {"upload GENERIC_READ", D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE},
    {"readback COPY_DEST", D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE},
};

int run(vktest::Device& d, bool addressable) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = addressable;
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    std::vector<Ref<ID3D12Resource>> buffers(std::size(kCases));
    std::vector<UINT64> addresses;
    for (size_t i = 0; i < std::size(kCases); ++i) {
        const Case& c = kCases[i];
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = c.heap;
        desc.Flags = c.flags;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, c.state, nullptr,
                                                         IID_PPV_ARGS(&buffers[i].p))));
        const UINT64 address = buffers[i]->GetGPUVirtualAddress();
        std::printf("%s: %s -> 0x%llx\n", addressable ? "bda" : "no bda", c.name, (unsigned long long)address);
        if (!address) std::fprintf(stderr, "%s: GetGPUVirtualAddress is 0\n", c.name);
        EXPECT(address != 0);
        EXPECT(address % kMinimumAlignment == 0);
        EXPECT(addressable || address % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT == 0);
        for (const UINT64 other : addresses) EXPECT(address >= other + kBytes || other >= address + kBytes);
        addresses.push_back(address);
    }
    return 0;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.buffer_device_address) {
        std::printf("SKIP: the device has no bufferDeviceAddress\n");
        return 77;
    }
    if (const int result = run(d, true)) return result;
    return run(d, false);
}
