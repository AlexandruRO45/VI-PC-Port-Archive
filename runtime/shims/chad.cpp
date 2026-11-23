// chad.dll: the console's audio DSP control channel ("chad" = the DSP hardware abstraction layer's host side).
//
// The executable imports five functions (gta6rt --dry-run; all of them are the whole set):
//   XDspActivate, XDspDeactivate, XDspConnectWithMaximumStreamLimit, XDspDisconnect, XDspSubmitCommand
//
// They drive the fixed-function audio DSP of the console: a title opens the device, submits command buffers to it and
// reads results back. A PC has no such DSP; the audio path here builds its own mixes on the CPU, so the channel only
// has to accept work and report success until the audio stage meets the real command shapes.
//
// ASSUMPTION: argument lists and return types are not measured (docs/d3d12_vulkan.md §7 lists the exports only). An
// int-returning HRESULT convention is assumed for the open/submit calls and S_OK (0) is returned. A null-failure
// return would stop the audio device at startup, which is worse here than a call that does nothing. Out parameters
// are not written yet. Every function logs on first use so a boot shows the order the title drives them in.
#include <windows.h>

#include <atomic>
#include <cstdio>

#include "shims.h"

namespace {

std::atomic<bool> g_warned{false};

void warn_once(const char* what) {
    if (!g_warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] chad: the DSP control channel is not implemented; %s and the rest answer S_OK without a "
                     "device (logged once)\n",
                     what);
    }
}

HRESULT __stdcall XDspActivate(const void* /*params*/, void** /*out*/) {
    warn_once("XDspActivate");
    return S_OK;
}

HRESULT __stdcall XDspDeactivate(void* /*dsp*/) { return S_OK; }

HRESULT __stdcall XDspConnectWithMaximumStreamLimit(void* /*dsp*/, uint32_t /*limit*/, void** /*out*/) {
    warn_once("XDspConnectWithMaximumStreamLimit");
    return S_OK;
}

HRESULT __stdcall XDspDisconnect(void* /*stream*/) { return S_OK; }

// A command buffer is submitted and completed immediately; a real DSP would report completion through the result
// queue of xapu/AcpHal.
HRESULT __stdcall XDspSubmitCommand(void* /*stream*/, const void* /*command*/, uint32_t /*size*/) {
    warn_once("XDspSubmitCommand");
    return S_OK;
}

}  // namespace

SHIM_TABLE(chad,
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand)},
);
