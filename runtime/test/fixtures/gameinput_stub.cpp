// Link-time stand-in for GameInput.dll, so tiny_gameinput.exe gets the import a GDK title has: GameInputCreate by
// ordinal 1 and no name (gameinput_stub.def; docs/gameinput.md, "The import is by ordinal"). Under `gta6rt --run` the
// shim registry answers the import and this DLL never loads; it is built into a folder the loader does not search.
// Test code, not game content.
#include <windows.h>

extern "C" HRESULT __stdcall GameInputCreate(void** out) {
    if (out) *out = nullptr;
    return E_FAIL;
}
