// XInput shim, see xinput_shim.h. State is injected by tests; otherwise a host XInput of an older version is forwarded to.

#include "xinput_shim.h"

#include <cstring>
#include <mutex>

#include "shims.h"

namespace {

constexpr DWORD kMaxControllers = 4;
constexpr DWORD kErrorBadArgument = ERROR_BAD_ARGUMENTS;         // 160: what XInput returns for a bad index
constexpr DWORD kErrorDeviceNotConnected = ERROR_DEVICE_NOT_CONNECTED;  // 1167

struct Controller {
    bool injected = false;
    bool connected = false;
    XINPUT_STATE state{};
    XINPUT_VIBRATION vibration{};
};

std::mutex& mutex() {
    static std::mutex m;
    return m;
}

Controller& controller(DWORD index) {
    static Controller controllers[kMaxControllers];
    return controllers[index];
}

// The host's pad support: xinput1_3.dll (DirectX redistributable, still shipped) or xinput9_1_0.dll (system), never
// this module's own name.
struct HostXInput {
    HMODULE module = nullptr;
    DWORD(WINAPI* get_state)(DWORD, XINPUT_STATE*) = nullptr;
    DWORD(WINAPI* set_state)(DWORD, XINPUT_VIBRATION*) = nullptr;
    DWORD(WINAPI* get_capabilities)(DWORD, DWORD, XINPUT_CAPABILITIES*) = nullptr;
    DWORD(WINAPI* get_battery)(DWORD, BYTE, XINPUT_BATTERY_INFORMATION*) = nullptr;
};

HostXInput& host() {
    static HostXInput host = [] {
        HostXInput h{};
        const HMODULE self = GetModuleHandleA("xinput1_4.dll");
        for (const char* name : {"xinput1_3.dll", "xinput9_1_0.dll"}) {
            HMODULE module = LoadLibraryExA(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!module || module == self) continue;
            h.module = module;
            h.get_state = reinterpret_cast<DWORD(WINAPI*)(DWORD, XINPUT_STATE*)>(GetProcAddress(module, "XInputGetState"));
            h.set_state = reinterpret_cast<DWORD(WINAPI*)(DWORD, XINPUT_VIBRATION*)>(GetProcAddress(module, "XInputSetState"));
            h.get_capabilities = reinterpret_cast<DWORD(WINAPI*)(DWORD, DWORD, XINPUT_CAPABILITIES*)>(
                GetProcAddress(module, "XInputGetCapabilities"));
            h.get_battery = reinterpret_cast<DWORD(WINAPI*)(DWORD, BYTE, XINPUT_BATTERY_INFORMATION*)>(
                GetProcAddress(module, "XInputGetBatteryInformation"));
            break;
        }
        return h;
    }();
    return host;
}

XINPUT_CAPABILITIES gamepad_capabilities() {
    XINPUT_CAPABILITIES capabilities{};
    capabilities.Type = XINPUT_DEVTYPE_GAMEPAD;
    capabilities.SubType = XINPUT_DEVSUBTYPE_GAMEPAD;
    capabilities.Flags = XINPUT_CAPS_FFB_SUPPORTED;
    capabilities.Gamepad.wButtons = 0xFFFF;
    capabilities.Gamepad.bLeftTrigger = 0xFF;
    capabilities.Gamepad.bRightTrigger = 0xFF;
    capabilities.Gamepad.sThumbLX = static_cast<SHORT>(0x7FFF);
    capabilities.Gamepad.sThumbLY = static_cast<SHORT>(0x7FFF);
    capabilities.Gamepad.sThumbRX = static_cast<SHORT>(0x7FFF);
    capabilities.Gamepad.sThumbRY = static_cast<SHORT>(0x7FFF);
    capabilities.Vibration.wLeftMotorSpeed = 0xFFFF;
    capabilities.Vibration.wRightMotorSpeed = 0xFFFF;
    return capabilities;
}

// True when a test injected this controller: the shim's own state wins over the host's pad.
bool injected(DWORD index) {
    std::lock_guard lock(mutex());
    return controller(index).injected;
}

}  // namespace

extern "C" DWORD WINAPI XInputGetState(DWORD index, XINPUT_STATE* state) noexcept {
    if (index >= kMaxControllers) return kErrorBadArgument;
    if (!state) return kErrorBadArgument;
    std::lock_guard lock(mutex());
    Controller& pad = controller(index);
    if (pad.injected) {
        if (!pad.connected) return kErrorDeviceNotConnected;
        *state = pad.state;
        return ERROR_SUCCESS;
    }
    if (host().get_state) return host().get_state(index, state);
    return kErrorDeviceNotConnected;
}

extern "C" DWORD WINAPI XInputSetState(DWORD index, XINPUT_VIBRATION* vibration) noexcept {
    if (index >= kMaxControllers) return kErrorBadArgument;
    if (!vibration) return kErrorBadArgument;
    {
        std::lock_guard lock(mutex());
        controller(index).vibration = *vibration;
    }
    if (injected(index)) return ERROR_SUCCESS;  // a test's pad has no motors to drive
    if (host().set_state) return host().set_state(index, vibration);
    return kErrorDeviceNotConnected;
}

extern "C" DWORD WINAPI XInputGetCapabilities(DWORD index, DWORD flags, XINPUT_CAPABILITIES* capabilities) noexcept {
    if (index >= kMaxControllers || !capabilities) return kErrorBadArgument;
    if (flags & XINPUT_FLAG_GAMEPAD) {
        *capabilities = gamepad_capabilities();
        return ERROR_SUCCESS;
    }
    if (!injected(index) && host().get_capabilities) return host().get_capabilities(index, flags, capabilities);
    *capabilities = gamepad_capabilities();
    return ERROR_SUCCESS;
}

extern "C" void WINAPI XInputEnable(BOOL) noexcept {
    // Rumble suppression is a host concern; the shim keeps reporting state either way, as XInput does after Enable.
}

extern "C" DWORD WINAPI XInputGetBatteryInformation(DWORD index, BYTE device_type,
                                                    XINPUT_BATTERY_INFORMATION* info) noexcept {
    if (index >= kMaxControllers || !info) return kErrorBadArgument;
    if (!injected(index) && host().get_battery) return host().get_battery(index, device_type, info);
    info->BatteryType = BATTERY_TYPE_WIRED;
    info->BatteryLevel = BATTERY_LEVEL_FULL;
    return ERROR_SUCCESS;
}

extern "C" DWORD WINAPI XInputGetKeystroke(DWORD index, DWORD, PXINPUT_KEYSTROKE keystroke) noexcept {
    if (index >= kMaxControllers && index != XUSER_INDEX_ANY) return kErrorBadArgument;
    if (keystroke) std::memset(keystroke, 0, sizeof(*keystroke));
    return ERROR_EMPTY;  // no keyboard/keypad keystrokes are synthesised
}

extern "C" void XInputSetTestState(DWORD index, const XINPUT_STATE* state, BOOL connected) {
    if (index >= kMaxControllers) return;
    std::lock_guard lock(mutex());
    Controller& pad = controller(index);
    pad.injected = true;
    pad.connected = connected != FALSE;
    if (state) pad.state = *state;
    else pad.state = XINPUT_STATE{};
}

extern "C" void XInputGetTestVibration(DWORD index, XINPUT_VIBRATION* vibration) {
    if (index >= kMaxControllers || !vibration) return;
    std::lock_guard lock(mutex());
    *vibration = controller(index).vibration;
}

extern "C" DWORD WINAPI XInputGetStateForTests(DWORD index, void* state) {
    return XInputGetState(index, static_cast<XINPUT_STATE*>(state));
}

extern "C" DWORD WINAPI XInputSetStateForTests(DWORD index, void* vibration) {
    return XInputSetState(index, static_cast<XINPUT_VIBRATION*>(vibration));
}

SHIM_TABLE(xinput,
    {"xinput1_4.dll", "XInputGetState", reinterpret_cast<void*>(&XInputGetState)},
    {"xinput1_4.dll", "XInputSetState", reinterpret_cast<void*>(&XInputSetState)},
    {"xinput1_4.dll", "XInputGetCapabilities", reinterpret_cast<void*>(&XInputGetCapabilities)},
    {"xinput1_4.dll", "XInputEnable", reinterpret_cast<void*>(&XInputEnable)},
    {"xinput1_4.dll", "XInputGetBatteryInformation", reinterpret_cast<void*>(&XInputGetBatteryInformation)},
    {"xinput1_4.dll", "XInputGetKeystroke", reinterpret_cast<void*>(&XInputGetKeystroke)},
    {"xinput1_4.dll", "XInputGetStateForTests", reinterpret_cast<void*>(&XInputGetStateForTests)},
    {"xinput1_4.dll", "XInputSetStateForTests", reinterpret_cast<void*>(&XInputSetStateForTests)},
    {"xinput1_4.dll", "XInputSetTestState", reinterpret_cast<void*>(&XInputSetTestState)},
    {"xinput1_4.dll", "XInputGetTestVibration", reinterpret_cast<void*>(&XInputGetTestVibration)},
);
