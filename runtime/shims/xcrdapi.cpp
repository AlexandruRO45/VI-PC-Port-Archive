// XCRDAPI.dll: Xbox Series DirectStorage entry point.
//
// Measured on the retail executable (docs/audio_hal_exe_calls.md, RT-5): one call site passes the address of an IID
// constant and the address of a global out-pointer (`DStorageGetFactory(REFIID, void**)`), then checks the out-pointer
// for null (Stop 5 in docs/retail_boot_chain.md).
//
// No file reads or decompression run through it yet: this returns S_OK with a process-lifetime COM factory stub whose
// IUnknown methods manage reference counts and whose remaining vtable slots (3..63) log once and return E_NOTIMPL.
#include <windows.h>
#include <unknwn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "shims.h"

namespace rt::xcrdapi {
namespace {

constexpr size_t kFactorySlots = 64;
constexpr IID kIDStorageFactory = {0x6924ea0c, 0xc3cd, 0x4826, {0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1}};

struct StubDStorageFactory {
    const void* const* vtable;
    std::atomic<ULONG> refs{0};
};

bool supports_iid(const IID& iid) {
    return IsEqualIID(iid, IID_IUnknown) || IsEqualIID(iid, kIDStorageFactory);
}

HRESULT STDMETHODCALLTYPE factory_query_interface(StubDStorageFactory* self, const IID* riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (!self || !riid) return E_INVALIDARG;
    if (!supports_iid(*riid)) return E_NOINTERFACE;
    self->refs.fetch_add(1, std::memory_order_relaxed);
    *ppv = self;
    return S_OK;
}

ULONG STDMETHODCALLTYPE factory_add_ref(StubDStorageFactory* self) {
    if (!self) return 0;
    return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
}

ULONG STDMETHODCALLTYPE factory_release(StubDStorageFactory* self) {
    if (!self) return 0;
    ULONG prev = self->refs.load(std::memory_order_relaxed);
    while (prev > 0 && !self->refs.compare_exchange_weak(prev, prev - 1, std::memory_order_relaxed)) {}
    return prev > 0 ? prev - 1 : 0;
}

template <size_t Slot>
HRESULT STDMETHODCALLTYPE unimplemented_slot(void* /*self*/) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: IDStorageFactory slot %zu unimplemented -> E_NOTIMPL (logged once)\n",
                     Slot);
    }
    return E_NOTIMPL;
}

template <size_t... Slots>
const void* const* build_vtable(std::index_sequence<Slots...>) {
    static_assert(sizeof...(Slots) + 3 == kFactorySlots);
    static const void* const kVtable[kFactorySlots] = {
        reinterpret_cast<const void*>(&factory_query_interface),
        reinterpret_cast<const void*>(&factory_add_ref),
        reinterpret_cast<const void*>(&factory_release),
        reinterpret_cast<const void*>(&unimplemented_slot<Slots + 3>)...,
    };
    return kVtable;
}

StubDStorageFactory& stub_factory() {
    static StubDStorageFactory factory{build_vtable(std::make_index_sequence<kFactorySlots - 3>{})};
    return factory;
}

HRESULT WINAPI dstorage_get_factory(const IID* riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (!riid) return E_INVALIDARG;
    if (!supports_iid(*riid)) return E_NOINTERFACE;

    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xcrdapi: DStorageGetFactory riid=%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x -> "
                     "stub factory (S_OK)\n",
                     static_cast<unsigned long>(riid->Data1), riid->Data2, riid->Data3,
                     riid->Data4[0], riid->Data4[1], riid->Data4[2], riid->Data4[3],
                     riid->Data4[4], riid->Data4[5], riid->Data4[6], riid->Data4[7]);
    }

    StubDStorageFactory& factory = stub_factory();
    factory.refs.fetch_add(1, std::memory_order_relaxed);
    *ppv = &factory;
    return S_OK;
}

}  // namespace
}  // namespace rt::xcrdapi

SHIM_TABLE(xcrdapi,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&rt::xcrdapi::dstorage_get_factory)},
);
