// A stand-in game executable for `gta6rt --run`: what a real title does in its first second, run through the
// loader instead of the Windows one. Test code, not game content. Run as
//   gta6rt --run tiny_game.exe -- hello 42 [tls] [eh] [exit-code]
// It checks its arguments, a static initializer, the heap, a file round trip, a second thread, the xgameruntime.dll
// imports (answered by the shim registry: the stub DLL it links against would fail them), and that it is the
// process's own image (GetModuleHandle*, GetModuleHandleEx* and GetModuleFileName* name tiny_game.exe, not
// gta6rt.exe), then calls exit() with 0 (42 with `exit-code`) when everything held. Two checks are opt-in, by
// argument (CTest's run_tiny_game_tls and run_tiny_game_eh): `tls`, thread_local on the game's own threads, which
// needs the loader's per-thread TLS (gta6rt starts main on a separate thread); and `eh`, a C++ exception caught
// inside the image.
// On MSVC, `eh` needs more than the unwind tables #176 registers: _CxxThrowException finds the throwing image
// with RtlPcToFileHeader, which the loader redirects in the host MSVC runtime. Arguments, not environment variables,
// so scripts/run_wine_tests.py (which passes a test's command line but not its ENVIRONMENT) runs them too. stdout is
// unbuffered, so a crash still shows how far the run got.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <thread>
#include <vector>

extern "C" __declspec(dllimport) HRESULT __stdcall XGameRuntimeInitialize();
extern "C" __declspec(dllimport) void __stdcall XGameRuntimeUninitialize();
extern "C" __declspec(dllimport) uint32_t __stdcall XSystemGetDeviceType();

extern "C" IMAGE_DOS_HEADER __ImageBase;  // this image's base, wherever it is mapped (the linker defines it)

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("tiny_game: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct StaticInit {
    int value;
    StaticInit() : value(1234) {}
};
StaticInit g_static;

bool has_arg(int argc, char** argv, const char* name) {
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

thread_local int t_counter = 5;

// Whether the file at `path` is this image: the loader copies the headers unchanged, so they match the file's.
bool file_is_this_image(const wchar_t* path) {
    const auto* image = reinterpret_cast<const uint8_t*>(&__ImageBase);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image + __ImageBase.e_lfanew);
    std::vector<uint8_t> file(nt->OptionalHeader.SizeOfHeaders);
    const HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    DWORD read = 0;
    const BOOL ok = ReadFile(f, file.data(), static_cast<DWORD>(file.size()), &read, nullptr);
    CloseHandle(f);
    return ok && read == file.size() && std::memcmp(file.data(), image, file.size()) == 0;
}

// What Windows answers a program about its own image, which a title uses to find its folder: under gta6rt the
// mapped game, not gta6rt.exe, and its file the one it was mapped from.
void check_module_identity() {
    const HMODULE self = reinterpret_cast<HMODULE>(&__ImageBase);
    constexpr DWORD kFromAddress =
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    check(GetModuleHandleW(nullptr) == self, "GetModuleHandleW(NULL) is the game's image, not gta6rt.exe");
    check(GetModuleHandleA(nullptr) == self, "GetModuleHandleA(NULL) is the game's image");
    HMODULE h = nullptr;
    check(GetModuleHandleExW(kFromAddress, reinterpret_cast<LPCWSTR>(&check), &h) && h == self,
          "GetModuleHandleExW(FROM_ADDRESS) on one of the game's functions is the game's image");
    h = nullptr;
    check(GetModuleHandleExA(kFromAddress, reinterpret_cast<LPCSTR>(&g_static), &h) && h == self,
          "GetModuleHandleExA(FROM_ADDRESS) on the game's data is the game's image");
    h = nullptr;
    check(GetModuleHandleExW(0, nullptr, &h) && h == self && FreeLibrary(h),
          "GetModuleHandleExW(0, NULL) is the game's image, and FreeLibrary gives the reference back");
    h = reinterpret_cast<HMODULE>(1);
    SetLastError(ERROR_SUCCESS);
    check(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, nullptr,
                              &h) &&
              GetLastError() == ERROR_INVALID_PARAMETER && h == nullptr,
          "GetModuleHandleExW(PIN | UNCHANGED_REFCOUNT) fails with ERROR_INVALID_PARAMETER and a null handle");

    wchar_t path[1024] = {};
    const DWORD len = GetModuleFileNameW(nullptr, path, 1024);
    if (len == 0 || len >= 1024) {
        check(false, "GetModuleFileNameW(NULL) returns a path");
        return;
    }
    std::printf("tiny_game: GetModuleFileNameW(NULL) = %ls\n", path);
    const wchar_t kName[] = L"\\tiny_game.exe";
    const DWORD name_len = static_cast<DWORD>(std::wcslen(kName));
    const bool full = len > 2 && ((path[1] == L':' && path[2] == L'\\') || (path[0] == L'\\' && path[1] == L'\\'));
    const bool named = len > name_len && CompareStringOrdinal(path + len - name_len, static_cast<int>(name_len), kName,
                                                              static_cast<int>(name_len), TRUE) == CSTR_EQUAL;
    check(full && named && file_is_this_image(path),
          "GetModuleFileNameW(NULL) is the full path of tiny_game.exe, the file this image was mapped from");
    wchar_t again[1024] = {};
    check(GetModuleFileNameW(self, again, 1024) == len && std::wcscmp(again, path) == 0,
          "GetModuleFileNameW(the game's HMODULE) gives the same path");
    check(GetModuleHandleW(path) == self && GetModuleHandleW(L"TINY_GAME.EXE") == self,
          "GetModuleHandleW finds the game by that full path and by its file name in any case");
    SetLastError(ERROR_SUCCESS);
    check(GetModuleHandleW(L"tiny_game") == nullptr && GetLastError() == ERROR_MOD_NOT_FOUND,
          "GetModuleHandleW(\"tiny_game\") adds \".dll\", so it doesn't name the EXE (ERROR_MOD_NOT_FOUND)");

    // Too small a buffer: size - 1 characters and a null, size returned, ERROR_INSUFFICIENT_BUFFER (Vista and later).
    wchar_t cut[1024];
    std::wmemset(cut, L'#', 1024);
    SetLastError(ERROR_SUCCESS);
    const DWORD cut_len = GetModuleFileNameW(nullptr, cut, len);
    check(cut_len == len && GetLastError() == ERROR_INSUFFICIENT_BUFFER && cut[len - 1] == 0 &&
              std::wmemcmp(cut, path, len - 1) == 0 && cut[len] == L'#',
          "GetModuleFileNameW(NULL) truncates to size - 1 characters and a null, returning size with "
          "ERROR_INSUFFICIENT_BUFFER");
    std::wmemset(cut, L'#', 1024);
    SetLastError(ERROR_INVALID_FUNCTION);
    check(GetModuleFileNameW(nullptr, cut, len + 1) == len && GetLastError() == ERROR_SUCCESS &&
              std::wcscmp(cut, path) == 0,
          "GetModuleFileNameW(NULL) with room for the null returns the length and clears the last error");
    std::wmemset(cut, L'#', 1024);
    SetLastError(ERROR_SUCCESS);
    check(GetModuleFileNameW(nullptr, cut, 0) == 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER && cut[0] == L'#',
          "GetModuleFileNameW(NULL, size 0) writes nothing and fails with ERROR_INSUFFICIENT_BUFFER");

    // The A form is the ANSI code page text of the same path, with the same truncation.
    char ansi[1024] = {};
    WideCharToMultiByte(CP_ACP, 0, path, -1, ansi, 1024, nullptr, nullptr);
    const DWORD ansi_len = static_cast<DWORD>(std::strlen(ansi));
    char a[1024];
    std::memset(a, '#', sizeof a);
    check(GetModuleFileNameA(nullptr, a, 1024) == ansi_len && std::strcmp(a, ansi) == 0,
          "GetModuleFileNameA(NULL) is the same path in the ANSI code page");
    std::memset(a, '#', sizeof a);
    SetLastError(ERROR_SUCCESS);
    check(ansi_len > 0 && GetModuleFileNameA(nullptr, a, ansi_len) == ansi_len &&
              GetLastError() == ERROR_INSUFFICIENT_BUFFER && a[ansi_len - 1] == 0 &&
              std::memcmp(a, ansi, ansi_len - 1) == 0,
          "GetModuleFileNameA(NULL) truncates the same way");
    check(GetModuleHandleA(ansi) == self, "GetModuleHandleA finds the game by its full path");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("tiny_game: started, argc=%d\n", argc);
    check(argc >= 3 && std::strcmp(argv[1], "hello") == 0 && std::strcmp(argv[2], "42") == 0,
          "argv is the game's command line (\"hello\" \"42\"), not gta6rt's");
    check(std::strstr(GetCommandLineA(), "hello 42") != nullptr, "GetCommandLineA is the game's command line");
    check(g_static.value == 1234, "a static initializer ran before main");

    std::vector<int> v(1 << 20);
    for (size_t i = 0; i < v.size(); ++i) v[i] = int(i);
    check(v[123456] == 123456, "a 4 MiB heap allocation works");

    char tmp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, tmp);
    std::string path = std::string(tmp) + "leonida_tiny_game_" + std::to_string(GetCurrentProcessId()) + ".txt";
    bool file_ok = false;
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fputs("leonida", f);
        std::fclose(f);
        char back[16] = {};
        if (FILE* r = std::fopen(path.c_str(), "rb")) {
            std::fread(back, 1, 7, r);
            std::fclose(r);
            file_ok = std::strcmp(back, "leonida") == 0;
        }
        std::remove(path.c_str());
    }
    check(file_ok, "a file round trip through the CRT works");

    long long sum = 0;
    std::thread worker([&] { for (int i = 1; i <= 1000; ++i) sum += i; });
    worker.join();
    check(sum == 500500, "a second thread ran and joined");

    check(XGameRuntimeInitialize() == S_OK, "XGameRuntimeInitialize is answered by the gta6rt shim (S_OK)");
    check(XSystemGetDeviceType() == 7, "XSystemGetDeviceType is answered by the gta6rt shim (Series X)");
    XGameRuntimeUninitialize();

    check_module_identity();

    if (has_arg(argc, argv, "tls")) {
        int seen = -1;
        std::thread t([&] { seen = t_counter; t_counter = 9; });
        t.join();
        check(t_counter == 5 && seen == 5, "thread_local starts at its initializer on each thread");
    } else {
        std::printf("tiny_game: SKIP thread_local (argument tls)\n");
    }
    if (has_arg(argc, argv, "eh")) {
        bool caught = false;
        try {
            throw std::string("boom");
        } catch (const std::string& e) {
            caught = e == "boom";
        }
        check(caught, "a C++ exception is thrown and caught inside the image");
    } else {
        std::printf("tiny_game: SKIP C++ exception (argument eh)\n");
    }

    std::printf("tiny_game: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    // Reserve 42 for a successful guest test. The loader itself returns 1 on errors.
    std::exit(g_failures ? 1 : (has_arg(argc, argv, "exit-code") ? 42 : 0));
}
