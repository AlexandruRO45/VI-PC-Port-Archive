// Synthetic C++/CX-style startup fixture: vccorlib's Platform::Details::GetCmdArguments, which the main() wrapper of
// a C++/CX program (main(Platform::Array<Platform::String^>^)) calls for its arguments, must give the guest's own
// arguments, the same ones the CRT gives it, and not gta6rt's. A run-time LoadLibrary of vccorlib140.dll must still
// reach the host's copy (or fail where the host has none, as under Wine), not a shim pseudo-handle.
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cwchar>

namespace Platform::Details {
// Exported by vccorlib140.dll as ?GetCmdArguments@Details@Platform@@YAPEAPEA_WPEAH@Z (vccorlib.lib, MSVC 14.44);
// vccorlib.h does not declare it.
__declspec(dllimport) wchar_t** __cdecl GetCmdArguments(int* argc);
}  // namespace Platform::Details

int main(int argc, char**) {
    int cx_argc = -1;
    wchar_t** cx_argv = Platform::Details::GetCmdArguments(&cx_argc);
    wchar_t** crt_wargv = *__p___wargv();
    bool ok = cx_argv && cx_argc == argc && crt_wargv;
    for (int i = 0; ok && i < argc; ++i) ok = cx_argv[i] && crt_wargv[i] && std::wcscmp(cx_argv[i], crt_wargv[i]) == 0;
    ok = ok && !cx_argv[cx_argc];
    // The check script passes "hello" "two words": the host's own arguments never contain them.
    ok = ok && argc == 3 && std::wcscmp(cx_argv[1], L"hello") == 0 && std::wcscmp(cx_argv[2], L"two words") == 0;
    std::printf("tiny_cx_args: GetCmdArguments argc=%d\n", cx_argc);

    if (HMODULE host = LoadLibraryW(L"vccorlib140.dll")) FreeLibrary(host);
    std::printf("tiny_cx_args: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
