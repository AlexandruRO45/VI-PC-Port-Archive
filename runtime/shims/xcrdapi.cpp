// XCRDAPI.dll: the console's DirectStorage entry point.
//
// The executable imports one function from it (gta6rt --dry-run): DStorageGetFactory.
//
// On the console DStorageGetFactory hands out the factory that opens files for DirectStorage (the console calls it
// XCRDAPI; the PC spells the same entry point dstorage.dll!DStorageGetFactory). The runtime's own shim registry already
// classifies the name as HRESULT-returning (registry.cpp) — the caller expects a factory object through the out
// pointer, and a null factory with a success code trips the caller up.
//
// ASSUMPTION: the factory is not the real DirectStorage one. The D3D12 layer here moves buffers itself, so nothing
// goes through a DirectStorage queue yet; the object returned is a token that answers a non-null pointer and a
// reference count of 1. A caller that only inspects and releases it is served; one that queues a copy through it will
// see the object's methods are missing and should be given the real interface next.
#include <windows.h>

#include <atomic>
#include <cstdio>

#include "shims.h"

namespace {

std::atomic<bool> g_warned{false};

struct FactoryStub {
    // A minimal IUnknown-shaped head: QueryInterface, AddRef, Release. The vtable is shared by every factory this
    // shim hands out, so a caller's Release/AddRef pair is balanced without per-object bookkeeping.
    void** vtable;
    LONG refs;
};

HRESULT __stdcall factory_query_interface(FactoryStub* self, const GUID* /*iid*/, void** out) {
    if (!out) return E_POINTER;
    *out = self;
    return S_OK;
}

ULONG __stdcall factory_add_ref(FactoryStub* self) {
    return static_cast<ULONG>(InterlockedIncrement(&self->refs));
}

ULONG __stdcall factory_release(FactoryStub* self) {
    return static_cast<ULONG>(InterlockedDecrement(&self->refs));
}

void* factory_vtable[] = {
    reinterpret_cast<void*>(&factory_query_interface),
    reinterpret_cast<void*>(&factory_add_ref),
    reinterpret_cast<void*>(&factory_release),
};

HRESULT __stdcall DStorageGetFactory(const GUID* /*riid*/, void** out) {
    if (!out) return E_POINTER;
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] XCRDAPI: DStorageGetFactory hands out a stub with only IUnknown; DirectStorage queueing "
                     "is not wired to the D3D12 layer yet (logged once)\n");
    }
    static FactoryStub factory{factory_vtable, 1};
    *out = &factory;
    return S_OK;
}

}  // namespace

SHIM_TABLE(xcrdapi,
    {"XCRDAPI.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory)},
);
