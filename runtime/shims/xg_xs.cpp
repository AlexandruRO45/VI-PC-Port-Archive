// xg_xs.dll: the console's texture layout library. The game executable (1.0.2.1) imports two functions from it.
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is the description's Layout field, which the helper otherwise sets to 0x100 | mode itself; the
//     answer here is written in that form. The HRESULT is only passed back up; no caller branches on it.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4 call sites. Two test the HRESULT and take their failure path when it is negative; two ignore it and call
//     through *out at once. On the computer the callers use vtable slot 1 after they are done (3 sites: Release; no
//     site calls slot 0, AddRef, nor QueryInterface), slot 2 with one out pointer to a large stack structure (2
//     sites: GetResourceLayout), slot 8 with (tiled base, plane, subresource, linear rows, row pitch, slice pitch) in
//     a loop over mips and array slices (1 site: CopyIntoSubresource) and slot 10 with the arguments of
//     ID3D12Device::GetResourceTiling (1 site). No site reads the return value of slots 2, 8 or 10.
//
// The description, the structure slot 2 fills and the layout itself are src/xg_surface.h and src/gfx10_layout.h:
// the GFX10 mip chain and tail measured on the retail texture pools, slice-major arrays as the executable addresses
// them, and the in-block bit order of the S and D swizzles. A description the model cannot lay out still gets a
// computer (E_INVALIDARG; the two callers that do not check would otherwise call through a null pointer, which COM's
// null-on-failure rule would hand them); that one answers every slot with E_NOTIMPL and writes nothing.
//
// LEONIDA_XG_LOG=1 prints every description the title asks about (otherwise the first 24).
#include <windows.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

#include "../src/xg_surface.h"
#include "shims.h"

namespace {

namespace xg = leonida::xg;
namespace gfx10 = leonida::gfx10;

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr size_t kSlots = 32;

void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[96];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: not implemented; E_NOTIMPL (caller %s, logged once)\n", name,
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

bool log_all() {
    static const bool all = [] {
        const char* v = std::getenv("LEONIDA_XG_LOG");
        return v && *v && *v != '0';
    }();
    return all;
}

void log_desc(const char* what, const xg::ResourceDesc& d, const xg::Surface* s, const void* caller) {
    static std::atomic<int> printed{0};
    if (!log_all() && printed.fetch_add(1, std::memory_order_relaxed) >= 24) return;
    char tail[160] = "";
    if (s) {
        std::snprintf(tail, sizeof(tail), " -> mode %u%s, %llu bytes, slice %llu, tail from mip %u", s->plane[0].mode,
                      s->plane[0].exact ? "" : " (stand-in bit order)", static_cast<unsigned long long>(s->size),
                      static_cast<unsigned long long>(s->plane[0].slice_size), s->plane[0].first_tail);
    }
    std::fprintf(stderr,
                 "[rt] xg_xs: %s: dimension %u, %llu x %u x %u, %u mips, format %u, %u samples, layout 0x%x, "
                 "flags 0x%x%s (caller %s)\n",
                 what, d.dimension, static_cast<unsigned long long>(d.width), d.height, d.depth_or_array_size,
                 d.mip_levels, d.format, d.sample_count, d.layout, d.flags, s ? tail : " -> no layout",
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

struct Computer {
    const void* const* vtable;
    std::atomic<ULONG> refs;
    bool valid;  // false: the fallback for a description without a layout
    xg::Surface surface;
};

ULONG __stdcall computer_add_ref(Computer* self) {
    if (!self->valid) return 1;
    return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
}

ULONG __stdcall computer_release(Computer* self) {
    if (!self->valid) return 1;
    const ULONG left = self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (left == 0) delete self;
    return left;
}

// Slot 2: GetResourceLayout(XG_RESOURCE_LAYOUT*).
HRESULT __stdcall computer_layout(Computer* self, xg::ResourceLayout* out) {
    if (!self->valid) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer without a layout, slot ", 2, _ReturnAddress());
        return kNotImpl;
    }
    if (!out) return E_POINTER;
    xg::resource_layout(self->surface, *out);
    return S_OK;
}

// Slot 8: CopyIntoSubresource(tiled resource base, plane, subresource, linear rows, row pitch, slice pitch).
HRESULT __stdcall computer_copy_into(Computer* self, void* base, UINT plane, UINT subresource, const void* rows,
                                     UINT row_pitch, UINT slice_pitch) {
    if (!self->valid) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer without a layout, slot ", 8, _ReturnAddress());
        return kNotImpl;
    }
    const xg::Surface& s = self->surface;
    if (!base || !rows || plane >= s.planes) return E_INVALIDARG;
    const std::uint32_t mip = subresource % s.mips, layer = subresource / s.mips;
    const gfx10::SurfaceLayout& l = s.plane[plane];
    if (layer >= l.layers) return E_INVALIDARG;
    if (!l.exact) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed))
            std::fprintf(stderr, "[rt] xg_xs: CopyIntoSubresource in swizzle mode %u: its bit order is a stand-in "
                                 "(src/gfx10_layout.h), logged once\n", l.mode);
    }
    auto* at = static_cast<std::byte*>(base) + s.plane_base[plane];
    auto* linear = static_cast<std::byte*>(const_cast<void*>(rows));
    if (!gfx10::copy_subresource(l, mip, layer, at, l.size, linear, row_pitch, slice_pitch, true))
        return E_INVALIDARG;
    return S_OK;
}

// Slot 10: GetResourceTiling(num tiles, packed mip info, tile shape, in/out subresource tiling count, first, array).
HRESULT __stdcall computer_tiling(Computer* self, UINT* tiles, xg::PackedMipInfo* packed, xg::TileShape* shape,
                                  UINT* count, UINT first, xg::SubresourceTiling* list) {
    if (!self->valid) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "texture computer without a layout, slot ", 10, _ReturnAddress());
        return kNotImpl;
    }
    leonida::ResourceTiling t;
    if (!xg::resource_tiling(self->surface, t)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "GetResourceTiling for a description without 64 KiB tiling, slot ", 10, _ReturnAddress());
        return kNotImpl;
    }
    if (tiles) *tiles = t.total_tiles;
    if (packed) {
        packed->standard_mips = static_cast<std::uint8_t>(t.standard_mips);
        packed->packed_mips = static_cast<std::uint8_t>(t.packed_mips);
        packed->tiles_for_packed_mips = t.packed_tiles;
        packed->start_tile = t.packed_start;
    }
    if (shape) *shape = {t.shape.width, t.shape.height, t.shape.depth};
    if (count) {
        const UINT total = static_cast<UINT>(t.subresources.size());
        const UINT n = first >= total ? 0 : std::min<UINT>(*count, total - first);
        for (UINT i = 0; list && i < n; ++i) {
            const leonida::SubresourceTiling& r = t.subresources[first + i];
            list[i] = {r.width, static_cast<std::uint16_t>(r.height), static_cast<std::uint16_t>(r.depth), r.start};
        }
        *count = n;
    }
    return S_OK;
}

template <size_t Slot>
HRESULT __stdcall computer_slot(Computer*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress());
    return kNotImpl;
}

template <size_t Slot>
void* slot_function() {
    if constexpr (Slot == 0) return reinterpret_cast<void*>(&computer_add_ref);
    else if constexpr (Slot == 1) return reinterpret_cast<void*>(&computer_release);
    else if constexpr (Slot == 2) return reinterpret_cast<void*>(&computer_layout);
    else if constexpr (Slot == 8) return reinterpret_cast<void*>(&computer_copy_into);
    else if constexpr (Slot == 10) return reinterpret_cast<void*>(&computer_tiling);
    else return reinterpret_cast<void*>(&computer_slot<Slot>);
}

template <size_t... Slot>
std::array<void*, sizeof...(Slot)> make_vtable(std::index_sequence<Slot...>) {
    return {slot_function<Slot>()...};
}

const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
Computer g_fallback{kVtable.data(), {1}, false, {}};

HRESULT __stdcall XGCreateTextureComputer(const xg::ResourceDesc* desc, void** out) {
    if (!out) return E_POINTER;
    *out = &g_fallback;
    if (!desc) return E_INVALIDARG;
    auto* computer = new (std::nothrow) Computer{kVtable.data(), {1}, true, {}};
    if (!computer) return E_OUTOFMEMORY;
    if (!xg::surface(*desc, computer->surface)) {
        delete computer;
        log_desc("XGCreateTextureComputer", *desc, nullptr, _ReturnAddress());
        return E_INVALIDARG;  // see the top of the file: *out keeps the fallback for the callers that do not check
    }
    log_desc("XGCreateTextureComputer", *desc, &computer->surface, _ReturnAddress());
    *out = computer;
    return S_OK;
}

HRESULT __stdcall XGSuggestSwizzleMode(const xg::ResourceDesc* desc, uint32_t /*flags*/, uint32_t* mode) {
    if (!desc || !mode) return E_INVALIDARG;
    *mode = xg::kLayoutBase | xg::suggest_mode(*desc);
    if (log_all()) log_desc("XGSuggestSwizzleMode", *desc, nullptr, _ReturnAddress());
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
