#pragma once
#include "desktop_gpu.h"
#include <array>
#include <cstdint>
#include <vector>

struct FfxDeviceProbe {
    explicit FfxDeviceProbe(ID3D12Device* input);
    uint32_t retain(uintptr_t self);
    uint32_t create(uintptr_t self, uintptr_t description, uintptr_t iid, uintptr_t output);

    ID3D12Device* device;
    std::vector<uint8_t> scratch;
    Microsoft::WRL::ComPtr<ID3D12Device> retained;
    std::array<Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>, 5> heaps;
    size_t created = 0;
};

void run_device_slice(FfxDeviceProbe& probe);

struct FfxSlotOutput {
    std::array<uint8_t, 0xec> prefix{};
    uint32_t index;
    uint32_t guard;
};
static_assert(offsetof(FfxSlotOutput, index) == 0xec);
uint32_t run_backend_slots(uint8_t* scratch, FfxSlotOutput& provider, const uint32_t* sizes);
void check_backend_slots(FfxDeviceProbe& probe);
