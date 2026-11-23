// Checks the registered xg_xs.dll shim against the call shapes measured in the game executable (runtime/shims/xg_xs.cpp).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. The checks:
// XGSuggestSwizzleMode writes 0x100 | mode into the caller's field; XGCreateTextureComputer answers a description
// with a computer whose slots 2, 8 and 10 (the ones the measured sites call) agree with src/xg_surface.h, and a
// description it cannot lay out with a computer that writes nothing (two measured callers use it without checking
// the HRESULT). The layout rules themselves are test/xg_surface_test.cpp's.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/xg_surface.h"
#include "shims.h"

namespace {

namespace xg = leonida::xg;
namespace gfx10 = leonida::gfx10;

using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using AddRefFn = ULONG(__stdcall*)(void*);
using ReleaseFn = ULONG(__stdcall*)(void*);
using Slot2Fn = HRESULT(__stdcall*)(void*, void*);
using Slot8Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot10Fn = HRESULT(__stdcall*)(void*, uint32_t*, void*, void*, uint32_t*, uint32_t, void*);
constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool all_bytes(const uint8_t* p, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != value) return false;
    return true;
}

void* const* vtable_of(void* computer) { return *static_cast<void* const* const*>(computer); }

}  // namespace

int main() {
    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create != nullptr, "XGCreateTextureComputer is registered");
    check(suggest != nullptr, "XGSuggestSwizzleMode is registered");
    check(rt::shims::owns_dll("xg_xs.dll"), "the registry owns xg_xs.dll");
    if (!create || !suggest) return 1;

    // A 256 x 128 RGBA8 2D texture with its full chain, laid out by the title's own Layout word.
    xg::ResourceDesc desc;
    desc.dimension = 3;
    desc.width = 256;
    desc.height = 128;
    desc.depth_or_array_size = 2;
    desc.mip_levels = 9;
    desc.format = 28;
    desc.sample_count = 1;
    desc.layout = 0xff;  // not a layout word: the suggestion below replaces it, as the title's builder does
    check(suggest(&desc, 0, &desc.layout) == S_OK, "XGSuggestSwizzleMode answers");
    check(desc.layout == (xg::kLayoutBase | gfx10::sw_64kb_d), "XGSuggestSwizzleMode writes 0x100 | SW_64KB_D");
    check(suggest(&desc, 0, nullptr) == E_INVALIDARG, "XGSuggestSwizzleMode(desc, 0, nullptr) is E_INVALIDARG");

    check(create(&desc, nullptr) == E_POINTER, "XGCreateTextureComputer(desc, nullptr) is E_POINTER");
    void* computer = nullptr;
    check(create(&desc, &computer) == S_OK && computer, "XGCreateTextureComputer answers with a computer");
    if (!computer) return 1;
    void* const* vtable = vtable_of(computer);
    xg::Surface surface;
    check(xg::surface(desc, surface), "the model lays the description out");

    std::vector<uint8_t> layout(sizeof(xg::ResourceLayout) + 8, 0xcd);
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout.data()) == S_OK, "slot 2 answers");
    xg::ResourceLayout expected;
    xg::resource_layout(surface, expected);
    check(std::memcmp(layout.data(), &expected, sizeof(expected)) == 0, "slot 2 writes the model's structure");
    check(all_bytes(layout.data() + sizeof(xg::ResourceLayout), 8, 0xcd), "slot 2 writes nothing past 0x1B38 bytes");

    // Slot 8 into slice 1 of mip 2 (subresource 2 + 1 * 9), rows 4 bytes wider than the mip.
    const gfx10::SurfaceLayout& plane = surface.plane[0];
    const gfx10::MipPlacement& mip = plane.mip[2];
    const uint32_t pitch = mip.width * 4 + 4;
    std::vector<uint8_t> rows(size_t(pitch) * mip.height);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = uint8_t(i * 29 + 3);
    std::vector<uint8_t> tiled(size_t(surface.size), 0);
    check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, tiled.data(), 0, 2 + 9, rows.data(), pitch,
                                               pitch * mip.height) == S_OK,
          "slot 8 answers");
    const uint32_t x = mip.width - 1, y = mip.height - 1;
    check(std::memcmp(tiled.data() + gfx10::element_offset(plane, 2, 1, x, y), rows.data() + y * pitch + x * 4, 4) == 0,
          "slot 8 puts the last element of slice 1 of mip 2 where the model does");
    check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, tiled.data(), 1, 0, rows.data(), pitch, 0) == E_INVALIDARG,
          "slot 8 refuses a plane the format does not have");

    uint32_t tiles = 0, count = 0x10;
    xg::PackedMipInfo packed{};
    xg::TileShape shape{};
    xg::SubresourceTiling list[0x10] = {};
    check(reinterpret_cast<Slot10Fn>(vtable[10])(computer, &tiles, &packed, &shape, &count, 0, list) == S_OK,
          "slot 10 answers");
    check(shape.width == 128 && shape.height == 128 && shape.depth == 1, "slot 10's tile shape is RGBA8's 128 x 128");
    check(packed.standard_mips == 2 && count == 0x10 && list[0].width_in_tiles == 2 && list[0].height_in_tiles == 1,
          "slot 10's standard mips and the first subresource's tiles");

    check(reinterpret_cast<AddRefFn>(vtable[0])(computer) == 2, "slot 0 counts a reference");
    check(reinterpret_cast<ReleaseFn>(vtable[1])(computer) == 1, "slot 1 drops it");
    check(reinterpret_cast<ReleaseFn>(vtable[1])(computer) == 0, "slot 1 frees the computer at zero");

    // A description without a layout still gets a computer, which writes nothing.
    uint8_t empty[0x38] = {};
    void* fallback = nullptr;
    check(create(empty, &fallback) == E_INVALIDARG && fallback, "an empty description gets E_INVALIDARG and a computer");
    if (!fallback) return 1;
    vtable = vtable_of(fallback);
    std::memset(layout.data(), 0xcd, layout.size());
    check(reinterpret_cast<Slot2Fn>(vtable[2])(fallback, layout.data()) == kNotImpl &&
              all_bytes(layout.data(), layout.size(), 0xcd),
          "its slot 2 is E_NOTIMPL and writes nothing");
    uint32_t zero = 0, inout = 0x10;
    uint8_t a[12] = {}, b[12] = {}, none[0xc0] = {};
    check(reinterpret_cast<Slot10Fn>(vtable[10])(fallback, &zero, a, b, &inout, 0, none) == kNotImpl && zero == 0 &&
              inout == 0x10 && all_bytes(none, sizeof(none), 0),
          "its slot 10 is E_NOTIMPL and leaves the zero-filled outputs");
    reinterpret_cast<ReleaseFn>(vtable[1])(fallback);
    reinterpret_cast<ReleaseFn>(vtable[1])(fallback);
    check(reinterpret_cast<Slot2Fn>(vtable[31])(fallback, layout.data()) == kNotImpl,
          "it stays usable after more Releases than it was handed out; slot 31 is traced");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
