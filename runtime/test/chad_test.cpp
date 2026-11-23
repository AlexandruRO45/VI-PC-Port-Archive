// chad.dll shim ABI test: resolves all 5 exports through the shim registry and exercises the full lifecycle.
#include <windows.h>

#include <cstdint>
#include <cstdio>

#include "../src/chad_dsp.h"
#include "shims.h"

namespace {

using ConnectFn = HRESULT(__stdcall*)(void*, uint32_t, uint32_t, uint32_t, uint32_t, void**);
using DisconnectFn = HRESULT(__stdcall*)(void*);
using ActivateFn = HRESULT(__stdcall*)(void*, const void*, void**, void**);
using DeactivateFn = HRESULT(__stdcall*)(void*);
using SubmitFn = HRESULT(__stdcall*)(void*, const void*, uint32_t*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    auto connect = reinterpret_cast<ConnectFn>(rt::shims::find("chad.dll", "XDspConnectWithMaximumStreamLimit"));
    auto disconnect = reinterpret_cast<DisconnectFn>(rt::shims::find("chad.dll", "XDspDisconnect"));
    auto activate = reinterpret_cast<ActivateFn>(rt::shims::find("chad.dll", "XDspActivate"));
    auto deactivate = reinterpret_cast<DeactivateFn>(rt::shims::find("chad.dll", "XDspDeactivate"));
    auto submit = reinterpret_cast<SubmitFn>(rt::shims::find("chad.dll", "XDspSubmitCommand"));

    check(rt::shims::owns_dll("chad.dll"), "the registry owns chad.dll");
    check(connect != nullptr, "XDspConnectWithMaximumStreamLimit is registered");
    check(disconnect != nullptr, "XDspDisconnect is registered");
    check(activate != nullptr, "XDspActivate is registered");
    check(deactivate != nullptr, "XDspDeactivate is registered");
    check(submit != nullptr, "XDspSubmitCommand is registered");
    if (!connect || !disconnect || !activate || !deactivate || !submit) return 1;

    uint8_t memory[4096] = {};
    void* ctx = nullptr;
    check(connect(memory, sizeof(memory), 4, 128, 1, nullptr) == E_POINTER,
          "XDspConnectWithMaximumStreamLimit(null out) -> E_POINTER");
    check(connect(memory, sizeof(memory), 4, 128, 1, &ctx) == S_OK && ctx != nullptr,
          "XDspConnectWithMaximumStreamLimit returns S_OK with a context handle");

    leonida::chad::DspActivateDesc desc{};
    desc.kind = 1;
    desc.block_size = 0x400;
    desc.channel_count = 2;
    void* state = nullptr;
    void* stream = nullptr;
    check(activate(ctx, &desc, &state, &stream) == S_OK && state != nullptr && stream != nullptr,
          "XDspActivate returns S_OK with state and stream handles");

    leonida::chad::DspCommand cmd{};
    cmd.opcode = 1;
    cmd.input_buffer = memory;
    cmd.output_buffer = memory + 1024;
    uint32_t tok1 = 0, tok2 = 0;
    check(submit(stream, &cmd, &tok1) == S_OK && tok1 != 0,
          "XDspSubmitCommand returns S_OK and a non-zero token");
    check(submit(stream, &cmd, &tok2) == S_OK && tok2 > tok1,
          "XDspSubmitCommand advances completion tokens");

    check(deactivate(stream) == S_OK, "XDspDeactivate returns S_OK");
    check(deactivate(stream) == E_INVALIDARG, "double XDspDeactivate returns E_INVALIDARG");
    check(disconnect(ctx) == S_OK, "XDspDisconnect returns S_OK");
    check(disconnect(ctx) == E_INVALIDARG, "double XDspDisconnect returns E_INVALIDARG");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
