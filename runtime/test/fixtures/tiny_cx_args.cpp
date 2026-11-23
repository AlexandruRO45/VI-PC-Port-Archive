// Synthetic C++/CX-style startup fixture: vccorlib's Platform::Details::GetCmdArguments, which the main() wrapper of
// a C++/CX program (main(Platform::Array<Platform::String^>^)) calls for its arguments, must give the guest's own
// arguments, the same ones the CRT gives it, and not the host's, however the guest reaches it:
//   - its own import, by name;
//   - GetProcAddress on the host's vccorlib140.dll, by name and by ordinal;
//   - a package DLL that imports it by ordinal only (cx_ordinal_user.dll), bound by rt::Loader at run time.
// The host's vccorlib140.dll is the stand-in cx_vccorlib_stub (Wine has none), loaded by the full path in
// LEONIDA_TEST_CX_STUB. Its GetCmdArguments answers "host-copy", and its other export must still be its own: the
// shims serve one name, they don't take the DLL over (a bare-name LoadLibrary returns the host module, not a shim
// pseudo-handle).
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cwchar>

namespace Platform::Details {
// Exported by vccorlib140.dll as ?GetCmdArguments@Details@Platform@@YAPEAPEA_WPEAH@Z (vccorlib.lib, MSVC 14.44);
// vccorlib.h does not declare it.
__declspec(dllimport) wchar_t** __cdecl GetCmdArguments(int* argc);
}  // namespace Platform::Details

namespace {

using CmdArguments = wchar_t**(__cdecl*)(int*);
constexpr char kDecorated[] = "?GetCmdArguments@Details@Platform@@YAPEAPEA_WPEAH@Z";
constexpr WORD kOrdinal = 130;  // cx_vccorlib_stub.def

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("tiny_cx_args: %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

// The guest's arguments: what the CRT gave main(), and the two the check script passes.
bool game_arguments(CmdArguments fn, int argc) {
    if (!fn) return false;
    int n = -1;
    wchar_t** argv = fn(&n);
    wchar_t** crt = *__p___wargv();
    if (!argv || !crt || n != argc || argv[n]) return false;
    for (int i = 0; i < n; ++i)
        if (!argv[i] || !crt[i] || std::wcscmp(argv[i], crt[i]) != 0) return false;
    return n == 3 && std::wcscmp(argv[1], L"hello") == 0 && std::wcscmp(argv[2], L"two words") == 0;
}

}  // namespace

int main(int argc, char**) {
    check(game_arguments(&Platform::Details::GetCmdArguments, argc), "import by name");

    wchar_t stub_path[MAX_PATH] = {};
    const DWORD length = GetEnvironmentVariableW(L"LEONIDA_TEST_CX_STUB", stub_path, MAX_PATH);
    check(length > 0 && length < MAX_PATH, "LEONIDA_TEST_CX_STUB names the stand-in vccorlib140.dll");
    HMODULE host = LoadLibraryW(stub_path);
    check(host != nullptr, "the stand-in vccorlib140.dll loads by full path");
    if (host) {
        HMODULE bare = LoadLibraryW(L"vccorlib140.dll");
        check(bare == host, "a bare-name LoadLibrary returns the host module, not a shim handle");
        if (bare) FreeLibrary(bare);

        const auto by_name = reinterpret_cast<CmdArguments>(GetProcAddress(host, kDecorated));
        const auto by_ordinal = reinterpret_cast<CmdArguments>(GetProcAddress(host, MAKEINTRESOURCEA(kOrdinal)));
        check(game_arguments(by_name, argc), "GetProcAddress by name");
        check(game_arguments(by_ordinal, argc), "GetProcAddress by ordinal");
        using Other = int(__cdecl*)();
        const auto other = reinterpret_cast<Other>(GetProcAddress(host, "cx_stub_other_export"));
        check(other && other() == 42, "an export the shims don't serve is the host's");

        // rt::Loader binds this package DLL's ordinal import now, against the loaded stand-in.
        HMODULE user = LoadLibraryW(L"cx_ordinal_user.dll");
        check(user != nullptr, "cx_ordinal_user.dll loads");
        const auto via_import =
            user ? reinterpret_cast<CmdArguments>(GetProcAddress(user, "cx_ordinal_arguments")) : nullptr;
        check(game_arguments(via_import, argc), "a package DLL's import by ordinal");
        FreeLibrary(host);
    }
    std::printf("tiny_cx_args: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
