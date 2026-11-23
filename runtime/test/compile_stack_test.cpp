// with_compile_stack (runtime/src/compile_stack.h): a pipeline created on a game thread with a small stack is
// compiled on a thread with at least compile_stack_need bytes of stack, and one with enough stack compiles in place.
// CPU only: the callable stands in for vkCreate*Pipelines and reports the stack it saw. No Vulkan, no game data.
#include "../src/compile_stack.h"

#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct Seen {
    std::size_t left = 0;
    DWORD thread = 0;
    int result = 0;
};

DWORD WINAPI create_pipeline(LPVOID p) {
    auto* seen = static_cast<Seen*>(p);
    seen->result = leonida::with_compile_stack([seen] {
        seen->left = leonida::stack_left();
        seen->thread = GetCurrentThreadId();
        return 42;
    });
    return 0;
}

// Runs create_pipeline on a thread with a `reserve`-byte stack, as a title's CreateThread(dwStackSize) would.
bool on_thread(std::size_t reserve, Seen& seen, DWORD& caller) {
    HANDLE thread = CreateThread(nullptr, reserve, create_pipeline, &seen, STACK_SIZE_PARAM_IS_A_RESERVATION, &caller);
    if (!thread) return false;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return true;
}

}  // namespace

int main() {
    // A 256 KB game thread: the compile moves to a thread with the larger reserve, and the result comes back.
    Seen small;
    DWORD small_caller = 0;
    EXPECT(on_thread(std::size_t(256) << 10, small, small_caller));
    EXPECT(small.result == 42);
    EXPECT(small.left >= leonida::compile_stack_need);
    EXPECT(small.thread != small_caller);

    // A 16 MB game thread already has the room: no extra thread.
    Seen large;
    DWORD large_caller = 0;
    EXPECT(on_thread(std::size_t(16) << 20, large, large_caller));
    EXPECT(large.result == 42);
    EXPECT(large.left >= leonida::compile_stack_need);
    EXPECT(large.thread == large_caller);

    std::printf("compile_stack: ok\n");
    return 0;
}
