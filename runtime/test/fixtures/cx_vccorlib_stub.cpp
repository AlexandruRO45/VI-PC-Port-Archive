// A stand-in for the host's vccorlib140.dll (Wine has none): GetCmdArguments under its decorated name and as ordinal
// 130 (the ordinal of Microsoft's x64 vccorlib140.dll 14.52), answering with an argument list no game is given, and one
// export the shims don't serve. Loaded by full path from tiny_cx_args, never mapped by rt::Loader.
#include <windows.h>

namespace Platform::Details {
__declspec(dllexport) wchar_t** __cdecl GetCmdArguments(int* argc) {
    static wchar_t host[] = L"host-copy";
    static wchar_t* argv[] = {host, nullptr};
    if (argc) *argc = 1;
    return argv;
}
}  // namespace Platform::Details

extern "C" __declspec(dllexport) int __cdecl cx_stub_other_export() { return 42; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
