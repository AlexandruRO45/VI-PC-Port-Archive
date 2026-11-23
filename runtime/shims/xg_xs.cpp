// xg_xs.dll: the console texture-layout helpers.
//
// The retail executable imports two names, and both have measured call sites:
//
//   XGSuggestSwizzleMode  (RVA 0xc4b3ce and 0x4eccca2)
//       rcx = a filled 64-byte layout descriptor, rdx = 0, r8 = 0 or a descriptor. The caller uses only eax as an
//       enum passed on to the layout writer. The console swizzles textures in memory; the host stores them linearly,
//       so the conservative answer is 0 ("linear"), and the descriptor is left as the caller wrote it.
//
//   XGCreateTextureComputer(const void* desc, void** out)  (RVA 0x344235 and 0x979c75)
//       HRESULT-shaped: the second caller tests eax for a negative value. The first caller ignores the HRESULT and
//       dereferences the out pointer immediately, so it would not survive E_NOTIMPL; that path is reported as a
//       boot risk rather than faked with a COM object whose methods are unknown. On failure the out pointer is set
//       to null, the COM convention.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

// No documented PC equivalent and no measured output write: the callers only consume the returned mode.
uint32_t WINAPI XGSuggestSwizzleMode(void* /*layout*/, const void* /*desc*/, const void* /*extra*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xg_xs: XGSuggestSwizzleMode: suggesting linear (0); host textures are not "
                             "swizzled (logged once)\n");
    }
    return 0;
}

HRESULT WINAPI XGCreateTextureComputer(const void* desc, void** out) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xg_xs: XGCreateTextureComputer(%p): E_NOTIMPL; the console texture computer has no "
                             "host equivalent (logged once)\n", desc);
    }
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
);
