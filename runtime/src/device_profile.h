#pragma once

// Which console the runtime says it is, when LEONIDA_DEVICE_TYPE doesn't name one (shims/xgameruntime.cpp).
//
// The package is the Xbox Series build, one executable for both consoles, so the title reads the console type at run
// time to pick its profile. Series X (Anaconda) has 10 GB of GPU-optimal memory; Series S (Lockhart) is the profile
// a GPU with less VRAM can hold, the 8 GB cards among them (RX 5700 XT, RTX 3060 Ti, RTX 3070). With the GPU's VRAM
// unknown (0), the answer stays Series X, as before.
//
// No Windows, Vulkan or third-party headers.

#include <cstdint>

namespace leonida {

// XSystemDeviceType values (the order of the public documentation).
inline constexpr std::uint32_t device_type_lockhart = 6;   // Xbox Series S
inline constexpr std::uint32_t device_type_anaconda = 7;   // Xbox Series X

// Series X's GPU-optimal memory: a GPU with less VRAM gets the Series S profile.
inline constexpr std::uint64_t series_x_gpu_memory = std::uint64_t(10) << 30;

// The device type for a GPU with `vram` bytes of dedicated video memory (0: unknown).
inline std::uint32_t default_device_type(std::uint64_t vram) {
    return vram && vram < series_x_gpu_memory ? device_type_lockhart : device_type_anaconda;
}

}  // namespace leonida
