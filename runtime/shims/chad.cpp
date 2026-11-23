// chad.dll: the console's hardware audio DSP client library (5 imports).
//
// Measured call shapes (static ABI inspection):
//   HRESULT XDspConnectWithMaximumStreamLimit(void* pBuffer, uint32_t bufferSize, uint32_t maxStreams,
//                                             uint32_t maxCommands, uint32_t flags, void** ppContext)
//   HRESULT XDspDisconnect(void* pContext)
//   HRESULT XDspActivate(void* pContext, const DspActivateDesc* pDesc, void** ppOutState, void** ppOutStream)
//   HRESULT XDspDeactivate(void* pStream)
//   HRESULT XDspSubmitCommand(void* pStream, const DspCommand* pCommand, uint32_t* pOutToken)
//
// Session/stream/command bookkeeping lives in the portable header runtime/src/chad_dsp.h (tested by chad_dsp_test).
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "../src/chad_dsp.h"
#include "shims.h"

namespace {

leonida::chad::DspEngine& engine() {
    static leonida::chad::DspEngine instance;
    return instance;
}

HRESULT to_hresult(leonida::chad::Status status) {
    switch (status) {
        case leonida::chad::Status::ok: return S_OK;
        case leonida::chad::Status::pointer: return E_POINTER;
        case leonida::chad::Status::out_of_memory: return E_OUTOFMEMORY;
        case leonida::chad::Status::invalid_arg:
        default: return E_INVALIDARG;
    }
}

void log_first_call(std::atomic<bool>& seen, const char* name, const void* caller) {
    if (seen.exchange(true, std::memory_order_relaxed)) return;
    std::fprintf(stderr, "[rt] chad: %s first call (caller %s)\n", name,
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

HRESULT __stdcall XDspConnectWithMaximumStreamLimit(void* memory_buffer, uint32_t memory_size, uint32_t max_streams,
                                                    uint32_t max_commands, uint32_t flags, void** out_context) {
    static std::atomic<bool> seen{false};
    log_first_call(seen, "XDspConnectWithMaximumStreamLimit", _ReturnAddress());
    return to_hresult(engine().connect(memory_buffer, memory_size, max_streams, max_commands, flags, out_context));
}

HRESULT __stdcall XDspDisconnect(void* context) {
    static std::atomic<bool> seen{false};
    log_first_call(seen, "XDspDisconnect", _ReturnAddress());
    return to_hresult(engine().disconnect(context));
}

HRESULT __stdcall XDspActivate(void* context, const void* desc, void** out_state, void** out_stream) {
    static std::atomic<bool> seen{false};
    log_first_call(seen, "XDspActivate", _ReturnAddress());
    return to_hresult(
        engine().activate(context, static_cast<const leonida::chad::DspActivateDesc*>(desc), out_state, out_stream));
}

HRESULT __stdcall XDspDeactivate(void* stream) {
    static std::atomic<bool> seen{false};
    log_first_call(seen, "XDspDeactivate", _ReturnAddress());
    return to_hresult(engine().deactivate(stream));
}

HRESULT __stdcall XDspSubmitCommand(void* stream, const void* command, uint32_t* out_token) {
    static std::atomic<bool> seen{false};
    log_first_call(seen, "XDspSubmitCommand", _ReturnAddress());
    return to_hresult(
        engine().submit_command(stream, static_cast<const leonida::chad::DspCommand*>(command), out_token));
}

}  // namespace

SHIM_TABLE(chad,
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect)},
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand)},
);
