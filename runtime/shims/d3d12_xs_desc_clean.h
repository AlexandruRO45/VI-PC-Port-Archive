// The PC form of a resource description the retail title builds for the console, for the Xbox-ABI device proxy.
//
// The title's D3D12_RESOURCE_DESC carries console layout words and flag bits the PC device cannot take
// (docs/retail_boot_chain.md, #6982). The slot 25 cleaner returns a copy of the title's description.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdint>

namespace rt::d3d12_xs {

// The D3D12_RESOURCE_FLAGS bits the PC runtime defines; every other bit is a console one.
constexpr std::uint32_t kPcResourceFlagsMask =
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE |
    D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
    D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY | D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY |
    D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE;

inline const auto kXboxTiledResourceLayout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);

// Strip Xbox-only flags and preserve the three supported PC layouts. In addition to 0x101, zebraso measured
// layout words 0x102 and 0x105 reaching slot 25 during a retail boot.
inline D3D12_RESOURCE_DESC pc_resource_desc(const D3D12_RESOURCE_DESC& xbox) {
    D3D12_RESOURCE_DESC pc = xbox;
    if (pc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN &&
        pc.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR &&
        pc.Layout != D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE) {
        pc.Layout = pc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? D3D12_TEXTURE_LAYOUT_ROW_MAJOR
                                                                   : D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    pc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(static_cast<std::uint32_t>(xbox.Flags) & kPcResourceFlagsMask);
    return pc;
}

}  // namespace rt::d3d12_xs
