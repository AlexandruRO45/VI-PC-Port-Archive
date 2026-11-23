// A run that crashes is kept out of Windows Error Reporting. A WER report of a crashed `gta6rt --run` leaves a process
// with no threads behind that keeps gta6rt.exe open, so the next build fails to link it (LNK1168). gta6rt therefore
// sets SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX at start; LEONIDA_WER=1 keeps the system default.
//
// gta6rt is started with CREATE_DEFAULT_ERROR_MODE, so it does not inherit this test's (or CTest's) error mode: a pass
// means gta6rt set the mode itself. tiny_game reports the mode it runs under through its exit code (error-mode: 43 off,
// 44 on) and crashes with an unhandled access violation (crash), whose code must still be the run's exit code. Whether
// WER then keeps a process behind depends on the machine and the size of the crashed process (the retail boot does,
// tiny_game does not), so this test checks the mode, not the leftover. No game files, no GPU.
#include <windows.h>

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct Run {
    bool exited = false;
    DWORD code = 0;
    DWORD pid = 0;
};

Run run(const std::wstring& gta6rt, const std::wstring& game, const wchar_t* mode, const wchar_t* wer) {
    SetEnvironmentVariableW(L"LEONIDA_WER", wer);  // nullptr removes it; the child inherits the environment
    std::wstring command = L"\"" + gta6rt + L"\" --run \"" + game + L"\" -- hello 42 " + mode;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    Run r;
    if (!CreateProcessW(gta6rt.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_DEFAULT_ERROR_MODE, nullptr,
                        nullptr, &startup, &child)) {
        std::fprintf(stderr, "could not start gta6rt (Win32 error %lu)\n", GetLastError());
        SetEnvironmentVariableW(L"LEONIDA_WER", nullptr);
        return r;
    }
    CloseHandle(child.hThread);
    r.pid = child.dwProcessId;
    const DWORD wait = WaitForSingleObject(child.hProcess, 60000);
    r.exited = wait == WAIT_OBJECT_0 && GetExitCodeProcess(child.hProcess, &r.code);
    if (wait == WAIT_TIMEOUT) TerminateProcess(child.hProcess, 1);
    CloseHandle(child.hProcess);
    SetEnvironmentVariableW(L"LEONIDA_WER", nullptr);
    return r;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tiny_game_wer_test <gta6rt.exe> <tiny_game.exe>\n");
        return 1;
    }
    const std::wstring gta6rt = argv[1], game = argv[2];

    const Run off = run(gta6rt, game, L"error-mode", nullptr);
    check(off.exited && off.code == 43, "a run has SEM_NOGPFAULTERRORBOX set by gta6rt itself (exit 43)");
    if (off.exited && off.code != 43) std::printf("     exit code %lu\n", off.code);

    const Run kept = run(gta6rt, game, L"error-mode", L"1");
    check(kept.exited && kept.code == 44, "LEONIDA_WER=1 keeps the default error mode (exit 44)");
    if (kept.exited && kept.code != 44) std::printf("     exit code %lu\n", kept.code);

    const Run crashed = run(gta6rt, game, L"crash", nullptr);
    check(crashed.exited && crashed.code == static_cast<DWORD>(EXCEPTION_ACCESS_VIOLATION),
          "a crashed run still ends with the exception's code (0xC0000005)");
    if (crashed.exited && crashed.code != static_cast<DWORD>(EXCEPTION_ACCESS_VIOLATION))
        std::printf("     exit code %#lx\n", crashed.code);

    if (failures) {
        std::fprintf(stderr, "tiny_game_wer_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("tiny_game_wer_test: all checks passed\n");
    return 0;
}
