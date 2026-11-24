#include <windows.h>

// Linking with /DELAYLOAD:cpu_delay_provider.dll puts both imports in a delay descriptor.
// delayed_value is imported by name and delayed_ordinal by ordinal 2 (NONAME in the .def).
extern "C" __declspec(dllimport) int delayed_value();
extern "C" __declspec(dllimport) int delayed_ordinal();

extern "C" __declspec(dllexport) int call_delayed() { return delayed_value(); }

extern "C" __declspec(dllexport) int call_delayed_ordinal() { return delayed_ordinal(); }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
