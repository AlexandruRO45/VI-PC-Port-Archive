// The second bundled DLL tiny_reent.dll loads from its DllMain. Its own DllMain loads tiny_reent.dll back, which is
// still inside its DLL_PROCESS_ATTACH at that moment: the call must return that module's handle without running its
// DllMain again. No C runtime. Ordinary test code; no game data.
#include <windows.h>

namespace {
int g_attach_count = 0;
}

extern "C" __declspec(dllexport) int __stdcall tiny_reent_dep_attach_count() { return g_attach_count; }

extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason != DLL_PROCESS_ATTACH) return TRUE;
    ++g_attach_count;
    return LoadLibraryA("tiny_reent.dll") != nullptr;
}
