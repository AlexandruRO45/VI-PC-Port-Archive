namespace bc64 = x2w_startup::bc64;

u64 __stdcall bc64_add_ref(Computer* self) { return InterlockedIncrement(&self->refs); }

long __stdcall bc64_layout(Computer*, unsigned char* out)
{
    if (!out) return E_POINTER;
    std::memset(out, 0, layout_header + 4 * plane_bytes + 8);
    put64(out, bc64::bytes);
    put64(out + 8, 65536);
    put32(out + 0x10, bc64::levels);
    put32(out + 0x14, 1);
    put32(out + 0x1b38, D3D12_RESOURCE_DIMENSION_TEXTURE2D);
    auto* plane = out + layout_header;
    put32(plane, 1);
    put64(plane + 8, bc64::bytes);
    put64(plane + 0x10, bc64::bytes);
    put64(plane + 0x20, 65536);
    put32(plane + 0x28, 16);
    put32(plane + 0x2c, bc64::first_tail);
    put32(plane + 0x30, bc64::levels - bc64::first_tail);
    for (UINT level = 0; level < bc64::levels; ++level) {
        auto* mip = plane + mip_base + level * mip_bytes;
        const UINT side = bc64::side(level);
        const bool tail = level >= bc64::first_tail;
        const UINT padded_rows = tail ? 2 * side : side;
        const UINT span = 16 * side * padded_rows;
        put64(mip, span);
        put64(mip + 8, bc64::offsets[level]);
        put64(mip + 0x10, span);
        put32(mip + 0x18, 4 * side);
        put32(mip + 0x1c, 16 * side);
        put32(mip + 0x20, tail ? 256 : 65536);
        put32(mip + 0x24, side);
        put32(mip + 0x28, padded_rows);
        put32(mip + 0x2c, 1);
        put32(mip + 0x30, side);
        put32(mip + 0x34, side);
        put32(mip + 0x38, 4 * side);
        put32(mip + 0x3c, 4 * side);
        put32(mip + 0x40, 1);
        put32(mip + 0x44, 1);
        put32(mip + 0x48, tail ? 1 : 9);
        put32(mip + 0x60, 1);
        put32(mip + 0x64, tail);
    }
    return S_OK;
}

template<bool into>
long __stdcall bc64_copy(Computer*, unsigned char* target, UINT plane, UINT level, const unsigned char* source,
                         UINT pitch, UINT slice_pitch)
{
    if (plane || level >= bc64::levels || !target || !source) return E_INVALIDARG;
    const UINT side = bc64::side(level);
    const u64 linear_bytes = u64{side - 1} * pitch + 16 * side;
    if (pitch < 16 * side || (slice_pitch && slice_pitch < linear_bytes)) return E_INVALIDARG;
    const auto dest = reinterpret_cast<std::uintptr_t>(target);
    const auto src = reinterpret_cast<std::uintptr_t>(source);
    const auto dest_bytes = into ? bc64::bytes : linear_bytes;
    const auto src_bytes = into ? linear_bytes : bc64::bytes;
    if (dest <= src ? src - dest < dest_bytes : dest - src < src_bytes) return E_INVALIDARG;
    if (!x2w_local::writable(target, dest_bytes) || !x2w_local::readable(source, src_bytes)) return E_FAIL;
    if constexpr (into) {
        bc64::tile(level, target, source, pitch);
        notify_texture_write(target, bc64::bytes);
    } else {
        bc64::untile(level, source, target, pitch);
    }
    return S_OK;
}
u64 __stdcall bc64_size(Computer*) { return bc64::bytes; }
u64 __stdcall bc64_alignment(Computer*) { return 65536; }
u64 __stdcall bc64_offset(Computer*, UINT plane, UINT level)
{
    return !plane && level < bc64::levels ? bc64::offsets[level] : 0;
}
u64 __stdcall bc64_element(Computer*, UINT plane, UINT level, u64 x, UINT y, UINT slice, UINT sample, UINT* bit)
{
    if (bit) *bit = 0;
    if (plane || slice || sample || level >= bc64::levels || x >= bc64::side(level) || y >= bc64::side(level)) return 0;
    return bc64::offset(level, static_cast<UINT>(x), y);
}

long __stdcall bc64_tiling(Computer* self, UINT* tiles, D3D12_PACKED_MIP_INFO* packed, D3D12_TILE_SHAPE* shape,
                           UINT* count, UINT first, D3D12_SUBRESOURCE_TILING* tilings)
{
    const auto& desc = self->desc;
    constexpr UINT tile_side = 256;
    std::array<D3D12_SUBRESOURCE_TILING, bc64::first_tail> standard{};
    UINT slice_tiles = 1;  // The packed mips occupy tile zero of each slice.
    for (UINT level = bc64::first_tail; level-- > 0;) {
        const UINT wide = (static_cast<UINT>(desc.Width >> level) + tile_side - 1) / tile_side;
        const UINT high = ((desc.Height >> level) + tile_side - 1) / tile_side;
        standard[level] = {wide, static_cast<UINT16>(high), 1, slice_tiles};
        slice_tiles += wide * high;
    }

    if (tiles) *tiles = slice_tiles * desc.DepthOrArraySize;
    if (packed) *packed = {static_cast<UINT8>(bc64::first_tail),
                           static_cast<UINT8>(desc.MipLevels - bc64::first_tail), 1, 0};
    if (shape) *shape = {tile_side, tile_side, 1};
    if (count) {
        const UINT subresources = UINT{desc.MipLevels} * desc.DepthOrArraySize;
        const UINT available = first < subresources ? subresources - first : 0;
        const UINT requested = (std::min)(*count, available);
        UINT written = 0;
        for (UINT i = 0; i < requested; ++i) {
            const UINT subresource = first + i;
            const UINT level = subresource % desc.MipLevels;
            if (level >= bc64::first_tail) continue;
            if (tilings) {
                tilings[i] = standard[level];
                tilings[i].StartTileIndexInOverallResource += (subresource / desc.MipLevels) * slice_tiles;
            }
            ++written;
        }
        *count = written;
    }
    return S_OK;
}
