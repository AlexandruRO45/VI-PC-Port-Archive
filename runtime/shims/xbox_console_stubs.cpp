// Bring-up shims for console-exclusive Xbox Series X|S DLLs:
//   - xcrdapi.dll  (DirectStorage factory)
//   - xg_xs.dll    (Xbox Graphics texture tiling / swizzle)
//   - chad.dll     (Xbox Audio DSP)
//   - xapu.dll     (Xbox Audio Processing Unit)
//   - acphal.dll   (Audio Co-Processor HAL & APU heap)
//
// All entry points log on first call so bring-up traces record call order cleanly.
#include <windows.h>
#include <malloc.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"

namespace {

void log_once(std::atomic<bool>& flag, const char* dll, const char* fn) {
    if (!flag.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] %s: %s called (bring-up stub, logged once)\n", dll, fn);
    }
}

// ---- xcrdapi.dll (DirectStorage) --------------------------------------------------------------------------------

HRESULT WINAPI DStorageGetFactory_stub(REFIID /*riid*/, void** out) {
    static std::atomic<bool> logged{false};
    log_once(logged, "xcrdapi", "DStorageGetFactory -> E_NOTIMPL");
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

// ---- xg_xs.dll (Xbox Graphics texture tiling) -------------------------------------------------------------------

HRESULT WINAPI XGCreateTextureComputer_stub(const void* /*desc*/, void** out) {
    static std::atomic<bool> logged{false};
    log_once(logged, "xg_xs", "XGCreateTextureComputer -> E_NOTIMPL");
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

uint32_t WINAPI XGSuggestSwizzleMode_stub(const void* /*desc*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "xg_xs", "XGSuggestSwizzleMode -> 0 (linear)");
    return 0;
}

// ---- chad.dll (Xbox Audio DSP) ----------------------------------------------------------------------------------

HRESULT WINAPI XDspActivate_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspActivate -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspConnectWithMaximumStreamLimit_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspConnectWithMaximumStreamLimit -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspDeactivate_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspDeactivate -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspDisconnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspDisconnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspSubmitCommand_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspSubmitCommand -> S_OK");
    return S_OK;
}

// ---- xapu.dll (Xbox Audio Processing Unit) ----------------------------------------------------------------------

HRESULT WINAPI XApuConnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuConnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuDisconnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuDisconnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuEnqueueCommand_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuEnqueueCommand -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuDequeueResult_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuDequeueResult -> S_FALSE");
    return S_FALSE;
}

uint32_t WINAPI XApuGetResultCount_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuGetResultCount -> 0");
    return 0;
}

// ---- acphal.dll (Audio Co-Processor HAL) ------------------------------------------------------------------------

HRESULT WINAPI AcpHalCreate_stub(void** out) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalCreate -> E_NOTIMPL");
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

HRESULT WINAPI AcpHalAllocateShapeContexts_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalAllocateShapeContexts -> S_OK");
    return S_OK;
}

HRESULT WINAPI AcpHalReleaseShapeContexts_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalReleaseShapeContexts -> S_OK");
    return S_OK;
}

HRESULT WINAPI ApuCreateHeap_stub(uint32_t /*cached_size*/, uint32_t /*noncached_size*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuCreateHeap -> S_OK");
    return S_OK;
}

HRESULT WINAPI ApuAlloc_stub(void** virtual_address, uint32_t* apu_address, uint32_t size, uint32_t alignment) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuAlloc -> aligned heap");
    const size_t align = (alignment >= 64 && (alignment & (alignment - 1)) == 0) ? alignment : 64;
    void* p = _aligned_malloc(size ? size : 1, align);
    if (!p) return E_OUTOFMEMORY;
    if (virtual_address) *virtual_address = p;
    if (apu_address) *apu_address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) & 0xFFFFFFFFu);
    return S_OK;
}

void WINAPI ApuFree_stub(void* virtual_address) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuFree");
    _aligned_free(virtual_address);
}

BOOL WINAPI ApuIsVirtualAddressValid_stub(const void* virtual_address, uint32_t /*size*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuIsVirtualAddressValid");
    return virtual_address != nullptr ? TRUE : FALSE;
}

uint32_t WINAPI ApuMapVirtualAddress_stub(const void* virtual_address) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuMapVirtualAddress");
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(virtual_address) & 0xFFFFFFFFu);
}

void* WINAPI ApuMapApuAddress_stub(uint32_t /*apu_address*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuMapApuAddress -> nullptr");
    return nullptr;
}

}  // namespace

SHIM_TABLE(xbox_console_stubs,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory_stub)},
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer_stub)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode_stub)},
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate_stub)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit_stub)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate_stub)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect_stub)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand_stub)},
    {"xapu.dll", "XApuConnect", reinterpret_cast<void*>(&XApuConnect_stub)},
    {"xapu.dll", "XApuDisconnect", reinterpret_cast<void*>(&XApuDisconnect_stub)},
    {"xapu.dll", "XApuEnqueueCommand", reinterpret_cast<void*>(&XApuEnqueueCommand_stub)},
    {"xapu.dll", "XApuDequeueResult", reinterpret_cast<void*>(&XApuDequeueResult_stub)},
    {"xapu.dll", "XApuGetResultCount", reinterpret_cast<void*>(&XApuGetResultCount_stub)},
    {"acphal.dll", "AcpHalCreate", reinterpret_cast<void*>(&AcpHalCreate_stub)},
    {"acphal.dll", "AcpHalAllocateShapeContexts", reinterpret_cast<void*>(&AcpHalAllocateShapeContexts_stub)},
    {"acphal.dll", "AcpHalReleaseShapeContexts", reinterpret_cast<void*>(&AcpHalReleaseShapeContexts_stub)},
    {"acphal.dll", "ApuCreateHeap", reinterpret_cast<void*>(&ApuCreateHeap_stub)},
    {"acphal.dll", "ApuAlloc", reinterpret_cast<void*>(&ApuAlloc_stub)},
    {"acphal.dll", "ApuFree", reinterpret_cast<void*>(&ApuFree_stub)},
    {"acphal.dll", "ApuIsVirtualAddressValid", reinterpret_cast<void*>(&ApuIsVirtualAddressValid_stub)},
    {"acphal.dll", "ApuMapVirtualAddress", reinterpret_cast<void*>(&ApuMapVirtualAddress_stub)},
    {"acphal.dll", "ApuMapApuAddress", reinterpret_cast<void*>(&ApuMapApuAddress_stub)},
);
