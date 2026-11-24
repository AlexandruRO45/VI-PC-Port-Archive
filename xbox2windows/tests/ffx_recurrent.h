#pragma once
#include "ffx_device_init.h"

struct RecurrentConfig {
    uint64_t type = 0x10000, next = 0;
    uint32_t flags = 0, render_width, render_height, output_width, output_height;
};
static_assert(offsetof(RecurrentConfig, output_width) == 0x1c);

struct RecurrentProvider {
    FfxSlotOutput slot;
    uint8_t pad[0x148 - sizeof(FfxSlotOutput)]{};
    std::array<uint32_t, 25> resources, mirror;
    uint8_t reserved[0x228 - 0x210]{};
    uint32_t render[2]{}, output[2]{};
};
static_assert(offsetof(RecurrentProvider, resources) == 0x148 && offsetof(RecurrentProvider, output) == 0x230);

struct RecurrentBinding {
    explicit RecurrentBinding(FfxDeviceProbe& owner) : probe(owner), device(owner.device) {}
    FfxDeviceProbe& probe;
    ID3D12Device* device;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    HRESULT result = E_PENDING;
    uint32_t create(uintptr_t self, uintptr_t heap, uintptr_t flags, uintptr_t description, const uint64_t* frame);
    D3D12_CPU_DESCRIPTOR_HANDLE srv{}, visible_uav{}, cpu_uav{};
    void create_srv(uintptr_t self, uintptr_t texture, uintptr_t desc, uintptr_t dest);
    void create_uav(uintptr_t self, uintptr_t texture, uintptr_t counter, uintptr_t desc, uintptr_t dest);
};

void check_recurrent_views(const RecurrentBinding& binding);
void run_recurrent(RecurrentBinding& binding, RecurrentProvider& provider, const RecurrentConfig& config);
