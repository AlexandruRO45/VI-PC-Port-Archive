// Process-level imports of a mapped executable: its command line and its way out.
//
// `gta6rt --run` calls a mapped EXE's entry point inside the gta6rt process. The EXE's CRT startup asks for its
// arguments and finally calls exit(); both go through the EXE's own import table, so they can be answered here:
//   - argv/argc and the command line are the game's (set by run_entry), not gta6rt's;
//   - exit(), _exit(), quick_exit(), ExitProcess() and abort() from the game end the game's run, not the host:
//     the exit code is recorded, the run is signalled as finished and the calling thread waits for the host's
//     finish_run handoff before invoking native ExitProcess with its TLS intact;
//   - the game's atexit registrations (_crt_atexit, which MSVC's atexit and static destructors reach; msvcrt's
//     atexit and _onexit) go to a table of the run's own instead of the CRT's process-wide one, which the game
//     shares with gta6rt. exit() runs it newest first, as the CRT does, before the run ends; _exit(), _Exit(),
//     quick_exit(), ExitProcess() and abort() do not. _cexit() runs it and returns, _c_exit() returns. A second
//     thread that calls exit(), _exit(), _Exit() or quick_exit() while the first is leaving waits, as with the
//     CRT's exit lock. Not covered yet: at_quick_exit() handlers (quick_exit() runs none) and the thread-local
//     destructor callback (_register_thread_local_exe_atexit_callback); both still reach the host CRT.
// Covered CRTs: the Universal CRT (api-ms-win-crt-runtime-l1-1-0.dll / ucrtbase.dll, what MSVC links; ucrtbased.dll in Debug) and
// msvcrt.dll (MinGW). Calls made inside host DLLs, not through the game's imports, are not seen here.
// The loader matches imports by exact DLL name, so GetCommandLineA/W and ExitProcess are served under every name a
// title can import them by: kernel32.dll (the classic import libraries), kernelbase.dll, and the API sets that the
// umbrella libraries a GDK title links (OneCore_apiset.lib, mincore.lib) bind them to. The retail package's own DLLs
// import ExitProcess from api-ms-win-core-processthreads-l1-1-0.dll.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#include "process_shim.h"
#include "shims.h"

namespace rt::process {
namespace {

std::vector<std::string> g_args;
std::vector<std::wstring> g_wargs;
std::vector<char*> g_argv;
std::vector<wchar_t*> g_wargv;
std::string g_cmdline;
std::wstring g_wcmdline;
int g_argc = 0;
char** g_argv_ptr = nullptr;
wchar_t** g_wargv_ptr = nullptr;

std::atomic<bool> g_running{false};
std::atomic<bool> g_exited{false};
std::atomic<int> g_exit_code{0};
HANDLE g_done = nullptr;
HANDLE g_release_exit = nullptr;
HANDLE g_exit_thread = nullptr;
std::atomic<bool> g_native_exit{false};
std::atomic<DWORD> g_exit_tid{0};
void (*g_prepare_exit)(void*) = nullptr;
void* g_exit_context = nullptr;

std::wstring quote(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring out = L"\"";
    size_t slashes = 0;  // CommandLineToArgvW: backslashes are literal unless they precede a quote
    for (wchar_t c : a) {
        if (c == L'\\') {
            ++slashes;
        } else {
            if (c == L'"') out.append(slashes + 1, L'\\');
            slashes = 0;
        }
        out += c;
    }
    return out.append(slashes, L'\\') + L"\"";  // a trailing run is doubled before the closing quote
}

// The narrow arguments are ANSI code page text, like the CRT's narrow argv; under a UTF-8 activeCodePage CP_ACP is
// UTF-8 anyway. A character the code page has no mapping for becomes the code page's default character (or a
// best-fit one), as in the CRT, instead of failing: std::filesystem::path::string() throws there (RT-84).
std::string narrow(const std::wstring& w) {
    int n = WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_ACP, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

[[noreturn]] void end_run(int code) {
    // A detach callback may call exit again. Preserve the first exit status and let Windows handle reentry.
    if (g_native_exit && g_exit_tid == GetCurrentThreadId()) ExitProcess(static_cast<UINT>(g_exit_code.load()));
    bool expected = false;
    if (g_exited.compare_exchange_strong(expected, true)) {
        g_exit_code = code;
        if (g_prepare_exit) g_prepare_exit(g_exit_context);
        // Keep a real handle even when a worker, rather than the entry thread, requested process exit.
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_exit_thread,
                        SYNCHRONIZE, FALSE, 0);
        g_exit_tid = GetCurrentThreadId();
        if (g_done) SetEvent(g_done);
        if (g_release_exit) {
            WaitForSingleObject(g_release_exit, INFINITE);
            g_native_exit = true;
            // Native ExitProcess kills the other threads before issuing DLL_PROCESS_DETACH on this thread.
            // Neither the host nor an explicit unload can supply that ordering and this thread's TLS together.
            ExitProcess(static_cast<UINT>(g_exit_code.load()));
        }
    }
    // Park until the host's ExitProcess, not ExitThread: Wine < 11.2 can exit 0 if the two race (4a9bc42a).
    if (g_running || g_release_exit) for (;;) Sleep(INFINITE);
    // The run is already over: this is game code running during host shutdown (an atexit handler the game
    // registered with the shared CRT calling exit again). Finish the process with the game's exit code; ending
    // just this thread here would make the host's last thread, and so the process, exit with `code` instead.
    ExitProcess(static_cast<UINT>(g_exit_code.load()));
}

// ---- the game's atexit table
using AtexitFn = void(__cdecl*)();
using OnexitFn = int(__cdecl*)();
std::mutex g_atexit_lock;
std::vector<AtexitFn> g_atexit;
std::atomic<DWORD> g_exiting_thread{0};

bool register_atexit(AtexitFn fn) {
    std::lock_guard<std::mutex> lock(g_atexit_lock);
    try {
        g_atexit.push_back(fn);
    } catch (const std::bad_alloc&) {
        return false;  // the CRT reports a full table as a failed registration
    }
    return true;
}

// Newest first. A handler registered while the handlers run goes on the end and so runs next, as in the UCRT.
void run_atexit_handlers() {
    for (;;) {
        AtexitFn fn;
        {
            std::lock_guard<std::mutex> lock(g_atexit_lock);
            if (g_atexit.empty()) return;
            fn = g_atexit.back();
            g_atexit.pop_back();
        }
        fn();
    }
}

// UCRT: int _crt_atexit(_PVFV), 0 on success. MSVC's atexit() and the static destructors of an EXE reach it.
int __cdecl game_crt_atexit(AtexitFn fn) {
    return fn && register_atexit(fn) ? 0 : -1;
}
// msvcrt.dll (MinGW): int atexit(void (*)(void)) and _onexit_t _onexit(_onexit_t).
int __cdecl game_msvcrt_atexit(AtexitFn fn) { return game_crt_atexit(fn); }
OnexitFn __cdecl game_onexit(OnexitFn fn) {
    // its int result is ignored, as the CRT ignores it
    return fn && register_atexit(reinterpret_cast<AtexitFn>(fn)) ? fn : nullptr;
}

// ---- the way out
// The CRT's exit lock, which exit(), _exit(), _Exit() and quick_exit() all take: the first thread to leave owns it
// and ends the run; any other thread that tries to leave waits forever. The owner may leave again from a handler
// (undefined in C): that ends the run with the new code and skips the remaining handlers.
bool take_exit_lock() {
    const DWORD self = GetCurrentThreadId();
    DWORD owner = 0;
    if (g_exiting_thread.compare_exchange_strong(owner, self)) return true;
    if (owner != self) {
        for (;;) Sleep(INFINITE);
    }
    return false;
}
// exit(): the game's handlers and static destructors, then the end of the run.
void __cdecl game_exit(int code) {
    if (take_exit_lock()) run_atexit_handlers();
    end_run(code);
}
// _exit(), _Exit(), quick_exit(): the end of the run without the atexit handlers.
void __cdecl game_quick_exit(int code) {
    take_exit_lock();
    end_run(code);
}
// _cexit() runs the handlers and flushes the streams, then returns; _c_exit() only returns.
void __cdecl game_cexit() {
    run_atexit_handlers();
    std::fflush(nullptr);
}
void __cdecl game_c_exit() {}
void __stdcall game_exit_process(UINT code) { end_run(static_cast<int>(code)); }
void __cdecl game_abort() {
    std::fprintf(stderr, "[rt] process: the game called abort()\n");
    end_run(3);  // the CRT's abort exit code
}

// ---- arguments
int* __cdecl game_p_argc() { return &g_argc; }
char*** __cdecl game_p_argv() { return &g_argv_ptr; }
wchar_t*** __cdecl game_p_wargv() { return &g_wargv_ptr; }
int __cdecl game_configure_argv(int) { return 0; }  // already parsed: run_entry set them
LPSTR __stdcall game_get_command_line_a() { return g_cmdline.data(); }
LPWSTR __stdcall game_get_command_line_w() { return g_wcmdline.data(); }
char** __cdecl game_p_acmdln_storage() {
    static char* p = nullptr;
    p = g_cmdline.data();
    return &p;
}
wchar_t** __cdecl game_p_wcmdln_storage() {
    static wchar_t* p = nullptr;
    p = g_wcmdline.data();
    return &p;
}

// What the UCRT's _get_narrow/_get_wide_winmain_command_line return, and the CRT's WinMain startup passes as
// lpCmdLine: the command line after the program name and the blanks that follow it. The program name ends at the
// first character up to ' ' outside quotes, the UCRT's own scan. The comparison is unsigned, so a non-ASCII byte of
// the narrow line is part of the name; a DBCS trail byte is never '"' or a blank, so it needs no lead-byte step.
template <typename C>
C* after_program_name(C* p) {
    using U = std::make_unsigned_t<C>;
    bool quoted = false;
    while (U(*p) > U(' ') || (*p && quoted)) {
        if (*p == C('"')) quoted = !quoted;
        ++p;
    }
    while (*p && U(*p) <= U(' ')) ++p;
    return p;
}
char* __cdecl game_get_narrow_winmain_command_line() { return after_program_name(g_cmdline.data()); }
wchar_t* __cdecl game_get_wide_winmain_command_line() { return after_program_name(g_wcmdline.data()); }

// The host CRT's environment tables, read when the game asks (RT-136). _environ and _wenviron are not stable: the host
// CRT builds the wide table only when a wide environment function first runs (a host started through main() has
// none until then), and _putenv/_wputenv reallocate both, so a pointer taken in set_command_line would be null or
// dangling by the time a wide-character startup calls __wgetmainargs. A lookup of a name no one sets makes the CRT
// create the table if it does not exist yet (from the other table or the OS block) without changing it.
// Like the msvcrt.dll original, the returned table is the CRT's own and is only valid until the next _putenv: a CRT
// startup reads it once, before the game has other threads, and a host that edits its environment concurrently with
// a game's startup is not covered.
char** current_environment() {
    if (!_environ) {
        size_t size = 0;
        (void)getenv_s(&size, nullptr, 0, "LEONIDA_RT_ENV_INIT");
    }
    return _environ;
}
wchar_t** current_wide_environment() {
    if (!_wenviron) {
        size_t size = 0;
        (void)_wgetenv_s(&size, nullptr, 0, L"LEONIDA_RT_ENV_INIT");
    }
    return _wenviron;
}

// msvcrt.dll: int __getmainargs(int* argc, char*** argv, char*** env, int expand, void* startinfo)
int __cdecl game_getmainargs(int* argc, char*** argv, char*** env, int, void*) {
    *argc = g_argc;
    *argv = g_argv_ptr;
    if (env) *env = current_environment();
    return 0;
}
int __cdecl game_wgetmainargs(int* argc, wchar_t*** argv, wchar_t*** env, int, void*) {
    *argc = g_argc;
    *argv = g_wargv_ptr;
    if (env) *env = current_wide_environment();
    return 0;
}

struct EntryCall {
    void* entry;
    void (*startup)(void*);
    void* startup_context;
};

DWORD WINAPI game_thread(void* p) {
    const EntryCall call = *static_cast<const EntryCall*>(p);
    bool startup_ok = true;
    try {
        if (call.startup) call.startup(call.startup_context);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        startup_ok = false;
    }
    if (!startup_ok) end_run(1);
    // An EXE entry point (mainCRTStartup and friends) takes the PEB pointer and returns the process exit code.
    auto entry = reinterpret_cast<DWORD(WINAPI*)(void*)>(call.entry);
    const DWORD code = entry(nullptr);
    end_run(static_cast<int>(code));  // returning from the entry point is an exit too
}

}  // namespace

void set_command_line(const std::wstring& program, const std::vector<std::wstring>& args) {
    g_wargs.clear();
    g_wargs.push_back(program);
    g_wargs.insert(g_wargs.end(), args.begin(), args.end());
    g_args.clear();
    g_wcmdline.clear();
    for (auto& a : g_wargs) {
        g_args.push_back(narrow(a));
        if (!g_wcmdline.empty()) g_wcmdline += L' ';
        g_wcmdline += quote(a);
    }
    g_cmdline = narrow(g_wcmdline);
    g_argv.clear();
    g_wargv.clear();
    for (auto& a : g_args) g_argv.push_back(a.data());
    for (auto& a : g_wargs) g_wargv.push_back(a.data());
    g_argv.push_back(nullptr);
    g_wargv.push_back(nullptr);
    g_argc = static_cast<int>(g_args.size());
    g_argv_ptr = g_argv.data();
    g_wargv_ptr = g_wargv.data();
}

int run_entry(void* entry, size_t stack_bytes, void (*startup)(void*), void* startup_context,
              void (*prepare_exit)(void*)) {
    {
        std::lock_guard<std::mutex> lock(g_atexit_lock);
        g_atexit.clear();  // a run never sees the handlers of an earlier one
    }
    g_exiting_thread = 0;
    g_exited = false;
    g_exit_code = 0;
    g_native_exit = false;
    g_exit_tid = 0;
    g_exit_thread = nullptr;
    g_prepare_exit = prepare_exit;
    g_exit_context = startup_context;
    g_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_done) return -1;
    g_release_exit = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_release_exit) {
        CloseHandle(g_done);
        g_done = nullptr;
        return -1;
    }
    g_running = true;
    EntryCall call{entry, startup, startup_context};
    HANDLE thread = CreateThread(nullptr, stack_bytes, game_thread, &call, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!thread) {
        g_running = false;
        CloseHandle(g_done);
        g_done = nullptr;
        CloseHandle(g_release_exit);
        g_release_exit = nullptr;
        return -1;
    }
    // The run ends when any game thread exits the process (exit/ExitProcess/abort) or the entry point returns.
    HANDLE waits[] = {g_done, thread};
    WaitForMultipleObjects(2, waits, FALSE, INFINITE);
    bool expected = false;
    if (g_exited.compare_exchange_strong(expected, true)) {
        // The entry thread exited without calling end_run (e.g. ExitThread); claim g_exited so no worker thread can
        // touch g_done while we close it, and preserve the thread's exit code.
        DWORD thread_code = 0;
        if (GetExitCodeThread(thread, &thread_code)) g_exit_code = static_cast<int>(thread_code);
    } else {
        // Another thread already won g_exited and is publishing g_exit_code + SetEvent(g_done); wait until it finishes
        // so we never close g_done or return before g_exit_code is published.
        WaitForSingleObject(g_done, INFINITE);
    }
    CloseHandle(thread);
    CloseHandle(g_done);
    g_done = nullptr;
    g_running = false;
    return g_exit_code.load();
}

[[noreturn]] void finish_run(int code) {
    if (g_exit_tid) {
        SetEvent(g_release_exit);
        // Normally ExitProcess kills this host thread. If the caller died unexpectedly, retain the exit code.
        if (g_exit_thread) WaitForSingleObject(g_exit_thread, INFINITE);
        else for (;;) Sleep(INFINITE);  // handle duplication failed; retain the live caller's TLS, never race it
    }
    ExitProcess(static_cast<UINT>(code));
}

}  // namespace rt::process

namespace {
using namespace rt::process;
#define UCRT(dll) \
    {dll, "exit", reinterpret_cast<void*>(&game_exit)}, {dll, "_exit", reinterpret_cast<void*>(&game_quick_exit)}, \
    {dll, "_Exit", reinterpret_cast<void*>(&game_quick_exit)}, \
    {dll, "quick_exit", reinterpret_cast<void*>(&game_quick_exit)}, \
    {dll, "_crt_atexit", reinterpret_cast<void*>(&game_crt_atexit)}, \
    {dll, "_cexit", reinterpret_cast<void*>(&game_cexit)}, {dll, "_c_exit", reinterpret_cast<void*>(&game_c_exit)}, \
    {dll, "abort", reinterpret_cast<void*>(&game_abort)}, {dll, "__p___argc", reinterpret_cast<void*>(&game_p_argc)}, \
    {dll, "__p___argv", reinterpret_cast<void*>(&game_p_argv)}, \
    {dll, "__p___wargv", reinterpret_cast<void*>(&game_p_wargv)}, \
    {dll, "_configure_narrow_argv", reinterpret_cast<void*>(&game_configure_argv)}, \
    {dll, "_configure_wide_argv", reinterpret_cast<void*>(&game_configure_argv)}, \
    {dll, "__p__acmdln", reinterpret_cast<void*>(&game_p_acmdln_storage)}, \
    {dll, "__p__wcmdln", reinterpret_cast<void*>(&game_p_wcmdln_storage)}, \
    {dll, "_get_narrow_winmain_command_line", reinterpret_cast<void*>(&game_get_narrow_winmain_command_line)}, \
    {dll, "_get_wide_winmain_command_line", reinterpret_cast<void*>(&game_get_wide_winmain_command_line)}
#define EXIT_PROCESS(dll) {dll, "ExitProcess", reinterpret_cast<void*>(&game_exit_process)}
#define COMMAND_LINE(dll) \
    {dll, "GetCommandLineA", reinterpret_cast<void*>(&game_get_command_line_a)}, \
    {dll, "GetCommandLineW", reinterpret_cast<void*>(&game_get_command_line_w)}
}  // namespace

SHIM_TABLE(process,
    UCRT("api-ms-win-crt-runtime-l1-1-0.dll"),
    UCRT("ucrtbase.dll"),
    UCRT("ucrtbased.dll"),  // the Debug UCRT: a Debug build of the stand-in game (run_tiny_game) links it
    {"msvcrt.dll", "exit", reinterpret_cast<void*>(&game_exit)},
    {"msvcrt.dll", "_exit", reinterpret_cast<void*>(&game_quick_exit)},
    {"msvcrt.dll", "atexit", reinterpret_cast<void*>(&game_msvcrt_atexit)},
    {"msvcrt.dll", "_onexit", reinterpret_cast<void*>(&game_onexit)},
    {"msvcrt.dll", "_cexit", reinterpret_cast<void*>(&game_cexit)},
    {"msvcrt.dll", "_c_exit", reinterpret_cast<void*>(&game_c_exit)},
    {"msvcrt.dll", "abort", reinterpret_cast<void*>(&game_abort)},
    {"msvcrt.dll", "__getmainargs", reinterpret_cast<void*>(&game_getmainargs)},
    {"msvcrt.dll", "__wgetmainargs", reinterpret_cast<void*>(&game_wgetmainargs)},
    EXIT_PROCESS("kernel32.dll"),
    EXIT_PROCESS("kernelbase.dll"),
    EXIT_PROCESS("api-ms-win-core-processthreads-l1-1-0.dll"),
    COMMAND_LINE("kernel32.dll"),
    COMMAND_LINE("kernelbase.dll"),
    COMMAND_LINE("api-ms-win-core-processenvironment-l1-1-0.dll"),
);
