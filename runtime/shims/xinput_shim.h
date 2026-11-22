// XInput shim: the pad a game polls through xinput1_4.dll.
//
// The shim keeps the state the game sees. A test (or a future input layer) injects it with XInputSetTestState; when
// nothing is injected the call is forwarded to the host's legacy XInput (xinput1_3.dll, then xinput9_1_0.dll), so a
// physical pad on the PC works as it would for a PC game. That forwarding never loads "xinput1_4.dll", which is this
// module itself when the loader maps it under that name.
//
// This header deliberately does not redeclare XInputGetState and friends: <XInput.h> already declares them
// (with WIN_NOEXCEPT), and the definitions in xinput.cpp match that declaration. Only the test hooks are new.
#pragma once

#include <windows.h>
#include <XInput.h>

// Test hooks (and the PE harness's imports): inject the state a pad reports, and read back the vibration last set.
extern "C" void XInputSetTestState(DWORD index, const XINPUT_STATE* state, BOOL connected);
extern "C" void XInputGetTestVibration(DWORD index, XINPUT_VIBRATION* vibration);
extern "C" DWORD WINAPI XInputGetStateForTests(DWORD index, void* state);
extern "C" DWORD WINAPI XInputSetStateForTests(DWORD index, void* vibration);
