// A package DLL whose import of vccorlib's GetCmdArguments is by ordinal (130) only, as an import library built with
// NONAME makes it. tiny_cx_args loads it at run time, after the stand-in vccorlib140.dll is loaded, so rt::Loader
// binds the ordinal against that host module.
#include <windows.h>

namespace Platform::Details {
__declspec(dllimport) wchar_t** __cdecl GetCmdArguments(int* argc);
}  // namespace Platform::Details

extern "C" __declspec(dllexport) wchar_t** __cdecl cx_ordinal_arguments(int* argc) {
    return Platform::Details::GetCmdArguments(argc);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
