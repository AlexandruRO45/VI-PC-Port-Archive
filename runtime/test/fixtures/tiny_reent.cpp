// A stand-in bundled DLL whose DllMain calls LoadLibrary, as middleware does: on itself (the idiom that pins a module
// against unloading) and on a second bundled DLL it does not import. Windows never re-enters a running DllMain: the
// self-load returns the module's own handle, and the other DLL is loaded and gets its DLL_PROCESS_ATTACH inside the
// call. Run as `gta6rt --init tiny_reent.dll`; check_reent.cmake reads the verdict. A second DLL_PROCESS_ATTACH ends
// the process with exit code 42 at once, so a loader that re-enters fails without running out of stack.
// No C runtime (the entry point is DllMain itself). Ordinary test code; no game data.
#include <windows.h>

namespace {

int g_attach_count = 0;

void say(const char* text) {
    DWORD length = 0, written = 0;
    while (text[length]) ++length;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, length, &written, nullptr);
}

}  // namespace

extern "C" BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, LPVOID) {
    if (reason != DLL_PROCESS_ATTACH) return TRUE;
    if (++g_attach_count != 1) {
        say("tiny_reent: FAIL DllMain(DLL_PROCESS_ATTACH) was entered again while it was running\n");
        TerminateProcess(GetCurrentProcess(), 42);
    }

    const HMODULE again = LoadLibraryA("tiny_reent.dll");
    if (again != self) {
        say("tiny_reent: FAIL LoadLibrary of the module itself did not return its own handle\n");
        return FALSE;
    }

    const HMODULE dep = LoadLibraryA("tiny_reent_dep.dll");
    if (!dep) {
        say("tiny_reent: FAIL LoadLibrary of a second bundled DLL from DllMain failed\n");
        return FALSE;
    }
    using CountFn = int(__stdcall*)();
    const auto dep_count = reinterpret_cast<CountFn>(reinterpret_cast<void*>(GetProcAddress(dep, "tiny_reent_dep_attach_count")));
    if (!dep_count || dep_count() != 1) {
        say("tiny_reent: FAIL the second DLL did not get DLL_PROCESS_ATTACH exactly once inside the call\n");
        return FALSE;
    }
    // The second DLL loaded this one back from its own DllMain; that must not have come here again either.
    if (g_attach_count != 1) {
        say("tiny_reent: FAIL the attach count changed during the nested loads\n");
        return FALSE;
    }
    say("tiny_reent: PASS\n");
    return TRUE;
}
