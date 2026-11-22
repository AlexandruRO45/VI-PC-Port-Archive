// Checks the registered xinput1_4.dll shim the way a game uses it: every function is resolved through
// rt::shims::find, so the test goes through the same table and calling convention the runtime installs for a title's
// imports. The shim has no physical pad here, so the test injects the state with XInputSetTestState and checks what
// XInputGetState reports, the capabilities, the vibration round trip and the error cases a game branches on.
#include <windows.h>
#include "xinput_shim.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "shims.h"

namespace {

using GetStateFn = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
using SetStateFn = DWORD(WINAPI*)(DWORD, XINPUT_VIBRATION*);
using CapabilitiesFn = DWORD(WINAPI*)(DWORD, DWORD, XINPUT_CAPABILITIES*);
using BatteryFn = DWORD(WINAPI*)(DWORD, BYTE, XINPUT_BATTERY_INFORMATION*);
using SetTestStateFn = void (*)(DWORD, const XINPUT_STATE*, BOOL);
using GetTestVibrationFn = void (*)(DWORD, XINPUT_VIBRATION*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template<typename F>
F resolve(const char* name) {
    void* function = rt::shims::find("xinput1_4.dll", name);
    check(function != nullptr, (std::string("xinput1_4.dll!") + name + " is registered").c_str());
    return reinterpret_cast<F>(function);
}

}  // namespace

int main() {
    const auto get_state = resolve<GetStateFn>("XInputGetState");
    const auto set_state = resolve<SetStateFn>("XInputSetState");
    const auto capabilities = resolve<CapabilitiesFn>("XInputGetCapabilities");
    const auto battery = resolve<BatteryFn>("XInputGetBatteryInformation");
    const auto set_test_state = resolve<SetTestStateFn>("XInputSetTestState");
    const auto get_test_vibration = resolve<GetTestVibrationFn>("XInputGetTestVibration");
    if (!get_state || !set_state || !capabilities || !battery || !set_test_state || !get_test_vibration)
        return 1;

    // A game polls pad 0..3; anything past four is a bad argument, and an empty slot is "not connected".
    XINPUT_STATE state{};
    check(get_state(4, &state) == ERROR_BAD_ARGUMENTS, "XInputGetState(4) reports ERROR_BAD_ARGUMENTS");
    check(get_state(0, nullptr) == ERROR_BAD_ARGUMENTS, "XInputGetState with a null state is rejected");
    set_test_state(1, nullptr, FALSE);
    check(get_state(1, &state) == ERROR_DEVICE_NOT_CONNECTED, "an injected empty slot is not connected");

    // Pad 0 presses A, Start and holds the left stick up and the left trigger half way.
    XINPUT_STATE injected{};
    injected.dwPacketNumber = 42;
    injected.Gamepad.wButtons = XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START;
    injected.Gamepad.bLeftTrigger = 128;
    injected.Gamepad.sThumbLX = 16000;
    set_test_state(0, &injected, TRUE);

    XINPUT_STATE read{};
    check(get_state(0, &read) == ERROR_SUCCESS, "an injected pad 0 reports ERROR_SUCCESS");
    check(read.dwPacketNumber == 42, "the packet number the shim reports is the injected one");
    check(read.Gamepad.wButtons == (XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START), "the buttons match");
    check(read.Gamepad.bLeftTrigger == 128 && read.Gamepad.sThumbLX == 16000, "the trigger and stick match");

    // Vibration is kept per pad and read back through the hook.
    XINPUT_VIBRATION vibration{};
    vibration.wLeftMotorSpeed = 0x4000;
    vibration.wRightMotorSpeed = 0x8000;
    check(set_state(0, &vibration) == ERROR_SUCCESS, "XInputSetState accepts a vibration");
    XINPUT_VIBRATION read_vibration{};
    get_test_vibration(0, &read_vibration);
    check(read_vibration.wLeftMotorSpeed == 0x4000 && read_vibration.wRightMotorSpeed == 0x8000,
          "the vibration round trips through the shim");
    check(set_state(4, &vibration) == ERROR_BAD_ARGUMENTS, "XInputSetState(4) reports ERROR_BAD_ARGUMENTS");
    check(set_state(0, nullptr) == ERROR_BAD_ARGUMENTS, "XInputSetState with a null vibration is rejected");

    // Capabilities: a gamepad with force feedback, which is what the console reports.
    XINPUT_CAPABILITIES caps{};
    check(capabilities(0, XINPUT_FLAG_GAMEPAD, &caps) == ERROR_SUCCESS, "XInputGetCapabilities succeeds");
    check(caps.Type == XINPUT_DEVTYPE_GAMEPAD && caps.SubType == XINPUT_DEVSUBTYPE_GAMEPAD,
          "the capabilities describe a gamepad");
    check((caps.Flags & XINPUT_CAPS_FFB_SUPPORTED) != 0, "force feedback is advertised");

    XINPUT_BATTERY_INFORMATION battery_info{};
    check(battery(0, BATTERY_DEVTYPE_GAMEPAD, &battery_info) == ERROR_SUCCESS, "battery information succeeds");
    check(battery_info.BatteryType == BATTERY_TYPE_WIRED, "the pad is wired");

    if (g_failures) {
        std::fprintf(stderr, "xinput_test: %d failure(s) of %d checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xinput_test: %d checks passed\n", g_checks);
    return 0;
}
