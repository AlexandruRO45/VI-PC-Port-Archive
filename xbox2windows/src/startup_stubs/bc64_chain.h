#pragma once

#include <d3d12.h>
#include <array>
#include <cstdint>
#include <cstring>

// Fixed non-XOR 64 KiB standard layout, 1024x1024, six BC mips and one slice.
// The optional AMD surface-info and coordinate fixture checks every visible block.
namespace x2w_startup::bc64 {
constexpr UINT bytes = 0x160000, levels = 6, first_tail = 3;
constexpr std::array<UINT, levels> offsets{0x60000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000};

inline bool standard_bc(const D3D12_RESOURCE_DESC& d)
{
    return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Alignment == 0 && d.Width == 1024 &&
           d.SampleDesc.Count == 1 && d.SampleDesc.Quality == 0 && static_cast<UINT>(d.Layout) == 0x109 &&
           d.Flags == D3D12_RESOURCE_FLAG_NONE &&
           (d.Format == DXGI_FORMAT_BC5_UNORM || d.Format == DXGI_FORMAT_BC7_UNORM);
}

inline bool supports(const D3D12_RESOURCE_DESC& d)
{
    return standard_bc(d) && d.Height == 1024 && d.DepthOrArraySize == 1 && d.MipLevels == levels;
}

inline bool supports_nine_mip_tiling(const D3D12_RESOURCE_DESC& d)
{
    return standard_bc(d) && (d.Height == 512 || d.Height == 1024) && d.MipLevels == 9 &&
           (d.DepthOrArraySize == 1 || d.DepthOrArraySize == 2);
}

constexpr UINT side(UINT mip) { return 256 >> mip; }

// Coordinates are 4x4 BC blocks. Each macro tile contains 64x64 blocks.
// Address bits 4..7 select (y0,y1,x0,x1); higher pairs interleave y and x.
constexpr UINT offset(UINT mip, UINT x, UINT y)
{
    const UINT tiles_per_row = (side(mip) + 63) / 64;
    const UINT tile = (y / 64) * tiles_per_row + x / 64;
    UINT within = ((y & 3) << 4) | ((x & 3) << 6);
    for (UINT bit = 2; bit < 6; ++bit) {
        within |= ((y >> bit) & 1) << (4 + bit * 2);
        within |= ((x >> bit) & 1) << (5 + bit * 2);
    }
    return offsets[mip] + tile * 65536 + within;
}

inline void untile(UINT mip, const unsigned char* guest, unsigned char* linear, UINT pitch)
{
    for (UINT y = 0; y < side(mip); ++y)
        for (UINT x = 0; x < side(mip); ++x)
            std::memcpy(linear + static_cast<std::size_t>(y) * pitch + 16 * x, guest + offset(mip, x, y), 16);
}
inline void tile(UINT mip, unsigned char* guest, const unsigned char* linear, UINT pitch)
{
    for (UINT y = 0; y < side(mip); ++y)
        for (UINT x = 0; x < side(mip); ++x)
            std::memcpy(guest + offset(mip, x, y), linear + static_cast<std::size_t>(y) * pitch + 16 * x, 16);
}
} // namespace x2w_startup::bc64
