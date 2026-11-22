// Child process transport for the Vulkan pipeline checker's --isolate mode.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace leonida::pipeline_isolate {

using Clock = std::chrono::steady_clock;

struct ProcessResult {
    std::string output;
    std::string error;
    unsigned status = 0;
    int signal = 0;
    bool timed_out = false;
    bool interrupted = false;
};

inline bool deadline_reached(Clock::time_point started, double timeout) {
    return timeout > 0 && std::chrono::duration<double>(Clock::now() - started).count() >= timeout;
}

#ifdef _WIN32

// Quote one argv element for the Windows C runtime (backslashes before quotes and the closing quote
// must be doubled). CreateProcessW receives an explicit application name and never invokes a shell.
inline std::wstring quote_argument(std::wstring_view arg) {
    std::wstring result = L"\"";
    std::size_t slashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        if (c == L'"') result.append(slashes + 1, L'\\');
        result.append(slashes, L'\\');
        slashes = 0;
        result.push_back(c);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

inline std::wstring module_path() {
    std::wstring path(512, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!length) return {};
        if (length < path.size()) {
            path.resize(length);
            return path;
        }
        path.resize(path.size() * 2);
    }
}

// Arguments came through main(char**) using Windows' ANSI code page. Convert them back to the
// UTF-16 required by CreateProcessW, preserving the original path and option boundaries.
inline std::wstring widen(const std::string& text) {
    const int n = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
    if (!n) return {};
    std::wstring result(static_cast<std::size_t>(n), L'\0');
    if (!MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, result.data(), n)) return {};
    result.pop_back();
    return result;
}

struct Handle {
    HANDLE value = nullptr;
    explicit Handle(HANDLE h = nullptr) : value(h) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    explicit operator bool() const { return value && value != INVALID_HANDLE_VALUE; }
    void reset() {
        if (*this) CloseHandle(value);
        value = nullptr;
    }
};

inline ProcessResult windows_error(const char* operation) {
    const auto code = GetLastError();
    ProcessResult result;
    result.error = std::string(operation) + " (Windows error " + std::to_string(code) + ")";
    return result;
}

inline ProcessResult run_process(const std::vector<std::string>& args, double timeout) {
    const auto exe = module_path();
    if (exe.empty()) return windows_error("locating the checker executable");
    auto command = quote_argument(exe);
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto arg = widen(args[i]);
        if (arg.empty() && !args[i].empty()) return windows_error("converting a child argument");
        command += L' ';
        command += quote_argument(arg);
    }

    // A temporary file lets the parent wait with a deadline even if driver workers inherit stdout.
    // stderr stays live for the existing progress and watchdog messages.
    wchar_t folder[MAX_PATH];
    const DWORD length = GetTempPathW(MAX_PATH, folder);
    if (!length || length >= MAX_PATH) return windows_error("finding the child log directory");
    LARGE_INTEGER ticks{};
    QueryPerformanceCounter(&ticks);
    const auto prefix = std::wstring(folder) + L"leonida-pipeline-" + std::to_wstring(GetCurrentProcessId()) +
                        L"-" + std::to_wstring(ticks.QuadPart);
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    Handle log;
    for (unsigned i = 0; i < 32; ++i) {
        const auto filename = prefix + L"-" + std::to_wstring(i) + L".log";
        log.value = CreateFileW(filename.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                &inherit, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (log) break;
        if (GetLastError() != ERROR_FILE_EXISTS) return windows_error("creating the child log");
    }
    if (!log) return windows_error("creating the child log");

    Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr));
    if (!input) return windows_error("opening child stdin");
    Handle err;
    const HANDLE standard_error = GetStdHandle(STD_ERROR_HANDLE);
    if (standard_error && standard_error != INVALID_HANDLE_VALUE &&
        !DuplicateHandle(GetCurrentProcess(), standard_error, GetCurrentProcess(), &err.value, 0, TRUE,
                         DUPLICATE_SAME_ACCESS))
        return windows_error("duplicating child stderr");
    if (!err) {
        err.value = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
        if (!err) return windows_error("opening child stderr");
    }

    Handle job(CreateJobObjectW(nullptr, nullptr));
    if (!job) return windows_error("creating the child job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
        return windows_error("configuring the child job");

    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    if (!bytes) return windows_error("sizing the child handle list");
    // Use suitably aligned storage for the variable-sized thread attribute list.
    std::vector<std::max_align_t> storage((bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes))
        return windows_error("initializing the child handle list");
    const std::unique_ptr<std::remove_pointer_t<LPPROC_THREAD_ATTRIBUTE_LIST>, decltype(&DeleteProcThreadAttributeList)>
        attribute_guard(attributes, DeleteProcThreadAttributeList);
    HANDLE handles[] = {input.value, log.value, err.value};
    if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, sizeof(handles),
                                   nullptr, nullptr))
        return windows_error("setting the child handle list");

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input.value;
    startup.StartupInfo.hStdOutput = log.value;
    startup.StartupInfo.hStdError = err.value;
    startup.lpAttributeList = attributes;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE,
                        CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                        &startup.StartupInfo, &pi))
        return windows_error("starting the checker child");
    Handle process(pi.hProcess), thread(pi.hThread);
    if (!AssignProcessToJobObject(job.value, process.value)) {
        const auto result = windows_error("assigning the checker child to its job");
        TerminateProcess(process.value, 1);
        WaitForSingleObject(process.value, INFINITE);
        return result;
    }
    if (ResumeThread(thread.value) == DWORD(-1)) return windows_error("resuming the checker child");
    thread.reset();

    ProcessResult result;
    const auto started = Clock::now();
    for (;;) {
        const DWORD wait = WaitForSingleObject(process.value, timeout > 0 ? 20 : INFINITE);
        if (wait == WAIT_OBJECT_0) break;
        if (wait != WAIT_TIMEOUT) return windows_error("waiting for the checker child");
        if (deadline_reached(started, timeout)) {
            result.timed_out = true;
            if (!TerminateJobObject(job.value, 1)) return windows_error("terminating the timed-out job");
            if (WaitForSingleObject(process.value, INFINITE) != WAIT_OBJECT_0)
                return windows_error("waiting for the terminated checker child");
            break;
        }
    }
    DWORD status = 0;
    if (!GetExitCodeProcess(process.value, &status)) return windows_error("reading the checker child exit code");
    result.status = status;
    job.reset();  // dispose of any descendants before reusing the cache or log

    LARGE_INTEGER zero{};
    if (!SetFilePointerEx(log.value, zero, nullptr, FILE_BEGIN)) return windows_error("rewinding the child log");
    char buf[4096];
    for (DWORD n = 0;;) {
        if (!ReadFile(log.value, buf, sizeof(buf), &n, nullptr)) return windows_error("reading the child log");
        if (!n) break;
        result.output.append(buf, n);
    }
    return result;
}

#else

inline volatile std::sig_atomic_t interrupted_signal = 0;

inline void remember_signal(int signal) { interrupted_signal = signal; }

class SignalGuard {
public:
    SignalGuard() {
        interrupted_signal = 0;
        struct sigaction action{};
        action.sa_handler = remember_signal;
        sigemptyset(&action.sa_mask);
        have_interrupt_ = sigaction(SIGINT, &action, &old_interrupt_) == 0;
        have_terminate_ = sigaction(SIGTERM, &action, &old_terminate_) == 0;
    }

    ~SignalGuard() { restore(); }

    bool active() const { return have_interrupt_ && have_terminate_; }

    void restore() const {
        if (have_interrupt_) sigaction(SIGINT, &old_interrupt_, nullptr);
        if (have_terminate_) sigaction(SIGTERM, &old_terminate_, nullptr);
    }

private:
    struct sigaction old_interrupt_{}, old_terminate_{};
    bool have_interrupt_ = false, have_terminate_ = false;
};

inline ProcessResult posix_error(const char* operation) {
    const int code = errno;
    ProcessResult result;
    result.error = std::string(operation) + ": " + std::strerror(code);
    return result;
}

inline ProcessResult run_process(const std::vector<std::string>& args, double timeout) {
    const auto close_log = [](std::FILE* file) { std::fclose(file); };
    std::unique_ptr<std::FILE, decltype(close_log)> log(std::tmpfile(), close_log);
    if (!log) return posix_error("creating the child log");
    const int output_fd = fileno(log.get());
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    SignalGuard signals;
    if (!signals.active()) return posix_error("installing child cancellation handlers");
    std::fflush(nullptr);
    const auto pid = fork();
    if (pid < 0) return posix_error("starting the checker child");
    if (!pid) {
        signals.restore();
        if (setpgid(0, 0) < 0 || dup2(output_fd, STDOUT_FILENO) < 0) _exit(127);
        if (output_fd != STDOUT_FILENO) close(output_fd);
        execvp(argv[0], argv.data());
        _exit(127);
    }

    // The child also sets its own group before exec. Attempt it here to close the fork-to-setpgid race.
    setpgid(pid, pid);
    ProcessResult result;
    const auto started = Clock::now();
    for (;;) {
        // Keep its PID allocated until the group is cleaned up, even when the leader already exited.
        siginfo_t info{};
        const int waited = waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT);
        if (!waited && info.si_pid == pid) break;
        if (waited < 0 && errno != EINTR) return posix_error("waiting for the checker child");
        if (interrupted_signal || deadline_reached(started, timeout)) {
            result.interrupted = interrupted_signal != 0;
            result.timed_out = !result.interrupted;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto interrupt = interrupted_signal;
    if (interrupt) result.interrupted = true;
    if (kill(-pid, SIGKILL) < 0 && (result.timed_out || result.interrupted)) kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return posix_error("reaping the checker child");
    }
    if (WIFSIGNALED(status)) result.signal = WTERMSIG(status);
    else if (WIFEXITED(status)) result.status = static_cast<unsigned>(WEXITSTATUS(status));
    if (result.interrupted) result.status = 128u + static_cast<unsigned>(interrupt);

    if (std::fseek(log.get(), 0, SEEK_SET)) return posix_error("rewinding the child log");
    char buf[4096];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof(buf), log.get())) != 0;)
        result.output.append(buf, n);
    if (std::ferror(log.get())) return posix_error("reading the child log");
    return result;
}

#endif

}  // namespace leonida::pipeline_isolate
