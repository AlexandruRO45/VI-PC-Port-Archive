// Native driver loads the same failing DLL graph that the mapped executable imports.
#include <windows.h>
#include <cstdio>
#include <string>

#if defined(ROLLBACK_NATIVE_DRIVER)
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc == 2) {
        HMODULE module = LoadLibraryW(argv[1]);
        const DWORD error = GetLastError();
        if (module) { FreeLibrary(module); return 2; }
        std::printf("native LoadLibrary error=%lu\n", error);
        return error == ERROR_DLL_INIT_FAILED ? 0 : 3;
    }
    if (argc != 3 || std::wstring(argv[1]) != L"--launch") return 4;
    std::wstring command = L"\"" + std::wstring(argv[2]) + L"\"";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(argv[2], command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process)) return 5;
    const DWORD wait = WaitForSingleObject(process.hProcess, 5000);
    DWORD code = 0;
    if (wait != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 124);
    const bool status = GetExitCodeProcess(process.hProcess, &code) != FALSE;
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    std::printf("native executable exit=0x%08lX\n", code);
    return wait == WAIT_OBJECT_0 && status && code == 0xC0000142u ? 0 : 6;
}
#else
extern "C" __declspec(dllimport) int rollback_b();
int main() {
    std::puts("EXECUTABLE_ENTRY_REACHED");
    return rollback_b();
}
#endif
