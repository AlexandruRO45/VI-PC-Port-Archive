#pragma once
#include <cstdint>

namespace leonida {

// xdxr_rehydrate.comp compiled with: glslc --target-env=vulkan1.2 -O -mfmt=c xdxr_rehydrate.comp -o xdxr_rehydrate.inc
inline constexpr std::uint32_t xdxr_rehydrate_spirv[] =
#include "xdxr_rehydrate.inc"
;

} // namespace leonida
