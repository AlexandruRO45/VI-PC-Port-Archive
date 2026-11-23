// XCRDAPI.dll: the console's DirectStorage entry point.
//
// The retail executable imports one name, DStorageGetFactory. Its call site (game+0x1d8e4d0) passes an interface id
// (game+0x62fc7d8, 62959d11-d45f-4884-a0a3-f7862ea243ae) and an out pointer, then requires a non-null factory and
// S_OK before calling the factory's vtable slot 8 with 0x02000000 and requiring a non-negative HRESULT. There is no
// dstorage.dll on this PC to forward to.
//
// This is the minimal safe factory: a process-wide object whose observed slot 8 succeeds. Unknown methods return
// E_NOTIMPL and never write through an out pointer; QueryInterface/AddRef/Release follow COM. It is a bring-up
// stopgap, not a DirectStorage implementation: no queue, file or memory is faked.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"
#include "startup_trace.h"

namespace {

struct Factory {
    const void** vtable;
};

HRESULT __stdcall factory_query_interface(Factory* self, const IID* /*iid*/, void** out) {
    if (!out) return E_POINTER;
    *out = self;
    return S_OK;
}

ULONG __stdcall factory_add_ref(Factory* /*self*/) { return 1; }
ULONG __stdcall factory_release(Factory* /*self*/) { return 1; }

// Observed slot 8: SetStagingBufferSize-like, one integer argument, HRESULT. A PC has no staging buffer to size, so
// accepting it is safe.
HRESULT __stdcall factory_set_staging_buffer_size(Factory* /*self*/, uint32_t /*size*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: factory slot 8 (staging buffer size) accepted; no PC staging buffer\n");
    }
    return S_OK;
}

// Every other unmeasured method: refuse, and do not touch an out pointer. One instantiation per slot so the log
// names the method the game actually reached.
template <int Slot>
HRESULT __stdcall factory_not_implemented_slot(Factory* /*self*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: factory slot %d -> E_NOTIMPL (caller %s)\n", Slot,
                     rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    }
    return E_NOTIMPL;
}

// Slot 4 is the one the retail exe reaches right after slot 8. Its arguments are logged once so the call site can be
// resolved; it is still refused until the interface shape is known.
HRESULT __stdcall factory_slot4(Factory* /*self*/, void* rdx, void* r8, void* r9) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: factory slot 4 -> E_NOTIMPL (caller %s rdx=%p r8=%p r9=%p)\n",
                     rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str(), rdx, r8, r9);
    }
    return E_NOTIMPL;
}

const void* g_factory_vtable[] = {
    reinterpret_cast<const void*>(&factory_query_interface),  // 0 IUnknown::QueryInterface
    reinterpret_cast<const void*>(&factory_add_ref),          // 1 IUnknown::AddRef
    reinterpret_cast<const void*>(&factory_release),          // 2 IUnknown::Release
    reinterpret_cast<const void*>(&factory_not_implemented_slot<3>),
    reinterpret_cast<const void*>(&factory_slot4),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<5>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<6>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<7>),
    reinterpret_cast<const void*>(&factory_set_staging_buffer_size),  // 8 observed by the retail exe
    reinterpret_cast<const void*>(&factory_not_implemented_slot<9>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<10>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<11>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<12>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<13>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<14>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<15>),
};

Factory g_factory{g_factory_vtable};

}  // namespace

HRESULT WINAPI DStorageGetFactory(REFIID /*riid*/, void** out) {
    static std::atomic<bool> logged{false};
    if (!out) return E_POINTER;
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: DStorageGetFactory -> software factory (observed slot 8, other methods "
                             "E_NOTIMPL)\n");
    }
    *out = &g_factory;
    return S_OK;
}

SHIM_TABLE(xcrdapi,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory)},
);
