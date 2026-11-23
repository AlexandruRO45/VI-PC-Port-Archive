// Minimal zero-CRT launcher stub for gta6rt.exe.
//
// Why this stub exists:
// A Windows/Xbox EXE compiled with MSVC /GA emits hardcoded TLS slot-0 accesses
// (`mov rax, gs:[58h]; mov rax, [rax]`) without reading `_tls_index`. If the loader
// itself is a full C++ EXE with `thread_local` / CRT TLS, Windows assigns slot 0 to
// the loader, colliding with the mapped guest EXE.
//
// By placing all C++/CRT/SDL3 runtime code in `gta6rt_core.dll` (loaded dynamically,
// so Windows assigns it TLS slot >= 1) and keeping this tiny stub EXE free of any CRT
// except a minimal unused `.tls` directory, Windows reserves TLS slot 0 cleanly for
// the primary mapped guest EXE on every thread.
#include <windows.h>
#include <shellapi.h>

#pragma section(".tls", read, write)
#pragma section(".tls$ZZZ", read, write)
__declspec(allocate(".tls")) static unsigned char g_tls_start = 0;
__declspec(allocate(".tls$ZZZ")) static unsigned char g_tls_end = 0;

extern "C" ULONG _tls_index = 0;

#pragma comment(linker, "/INCLUDE:_tls_used")
extern "C" const IMAGE_TLS_DIRECTORY64 _tls_used = {
    reinterpret_cast<ULONGLONG>(&g_tls_start),
    reinterpret_cast<ULONGLONG>(&g_tls_end),
    reinterpret_cast<ULONGLONG>(&_tls_index),
    0,
    0,
    0,
};

static wchar_t g_path[32768];

extern "C" void WINAPI stub_entry() {
    DWORD n = GetModuleFileNameW(nullptr, g_path, 32768);
    if (n == 0 || n >= 32768) ExitProcess(1);

    // Replace executable filename with "gta6rt_core.dll"
    DWORD slash = n;
    while (slash > 0 && g_path[slash - 1] != L'\\' && g_path[slash - 1] != L'/') --slash;
    const wchar_t dll_name[] = L"gta6rt_core.dll";
    for (DWORD i = 0; i < sizeof(dll_name) / sizeof(wchar_t); ++i) {
        if (slash + i >= 32768) ExitProcess(1);
        g_path[slash + i] = dll_name[i];
    }

    HMODULE core = LoadLibraryW(g_path);
    if (!core) ExitProcess(1);

    using WMainFn = int(__cdecl*)(int, wchar_t**);
    auto wmain_fn = reinterpret_cast<WMainFn>(GetProcAddress(core, "gta6rt_wmain"));
    if (!wmain_fn) ExitProcess(1);

    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) ExitProcess(1);

    int rc = wmain_fn(argc, argv);
    LocalFree(argv);
    ExitProcess(static_cast<UINT>(rc));
}
