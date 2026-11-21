#pragma once
#include "ffx_recurrent.h"
#include <stdexcept>

struct LoopDesc {
    uint32_t heap, placed;
    uint64_t allocation[2];
    uint32_t type, format, width, height, depth, mips, flags, usage, state, padding;
    const wchar_t* name;
    uint32_t id, reserved, mode, reserved2;
    uint64_t length, data;
};
static_assert(sizeof(LoopDesc) == 0x68 && offsetof(LoopDesc, id) == 0x48);
inline constexpr uintptr_t loop_callback = 0x2e30;

struct LoopContext {
    explicit LoopContext(ID3D12Device* device);
    FfxDeviceProbe probe;
    RecurrentProvider state{};
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 16> objects;
    std::array<uint32_t, 12> visited{};
    unsigned calls = 0;
    uint32_t fail_id = 0;
    uint32_t create(uintptr_t backend, const LoopDesc& desc, uint32_t context, uint32_t* index);
    uint32_t make_upload(const LoopDesc& parent, const LoopDesc& upload);
};
void reserve_resource(uint8_t* slot, uint8_t* scratch, const LoopDesc& desc, uint32_t* index);
uint32_t schedule_upload(LoopContext& ctx, const LoopDesc& desc);
uint32_t create_loop_context(LoopContext& ctx, void** output, const RecurrentConfig& config);
void check_ffx_provider_loop(ID3D12Device* device);
#ifdef FFX_LOOP_LOADER
uint32_t check_loop_loader(LoopContext& ctx, void** inner, const RecurrentConfig& config);
#endif
