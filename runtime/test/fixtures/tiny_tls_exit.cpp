// A genuine C++ thread_local object must be destroyed on the exiting thread before atexit/static destructors.
// check_tls_exit.cmake compares this synthetic executable with itself through gta6rt. No game files.
#include <windows.h>
#include <process.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void say(const char* text) {
    std::printf("tiny_tls_exit: %s\n", text);
    std::fflush(stdout);
}
struct ThreadRecord {
    DWORD owner = GetCurrentThreadId();
    const char* label = "untouched";
    ~ThreadRecord() {
        std::printf("tiny_tls_exit: tls dtor %s %s\n", label,
                    owner == GetCurrentThreadId() ? "same-thread" : "wrong-thread");
        std::fflush(stdout);
    }
};
thread_local ThreadRecord record;
struct StaticRecord {
    ~StaticRecord() { say("static dtor"); }
} static_record;
void at_exit() { say("atexit"); }
DWORD WINAPI exit_worker(void*) {
    record.label = "worker";
    say("touch worker");
    std::exit(24);
}
}  // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "return";
    record.label = "main";
    say("touch main");
    if (std::atexit(at_exit) != 0) return 90;
    if (!std::strcmp(mode, "return")) return 21;
    if (!std::strcmp(mode, "exit")) std::exit(22);
    if (!std::strcmp(mode, "cexit")) {
        _cexit();
        say("after cexit");
        _exit(23);
    }
    if (!std::strcmp(mode, "worker_exit")) {
        HANDLE worker = CreateThread(nullptr, 0, exit_worker, nullptr, 0, nullptr);
        if (!worker) return 91;
        WaitForSingleObject(worker, INFINITE);
        CloseHandle(worker);
        return 92;
    }
    if (!std::strcmp(mode, "cexit_exit")) {
        _cexit();
        say("after cexit");
        static thread_local ThreadRecord later;
        later.label = "later";
        say("touch later");
        std::exit(25);
    }
    if (!std::strcmp(mode, "c_exit_exit")) {
        _c_exit();
        say("after c_exit");
        std::exit(26);
    }
    return 93;
}
