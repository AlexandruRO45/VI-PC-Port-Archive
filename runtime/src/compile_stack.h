// A driver compiles a pipeline's shaders on the thread that calls vkCreate*Pipelines, and its compiler can need
// several megabytes of stack: on AMD 26.11.2 more than 1 MB for 26 mesh modules of the corpus and more than 2 MB for
// 10 of 28 large compute modules, while 4 MB was enough for all of them (measured in #4979). gta6rt links with an
// 8 MB stack (#5468), but a thread the title starts with its own dwStackSize keeps that size, and the D3D12 layer
// creates pipelines on whichever thread calls CreateComputePipelineState or replays a command list. On the console
// nothing is compiled there, so a title has no reason to give such a thread a large stack.
//
// with_compile_stack(f) runs f on the calling thread when at least compile_stack_need bytes of its stack are left,
// and otherwise on a short-lived thread with a compile_stack_reserve reservation while the caller waits.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace leonida {

inline constexpr std::size_t compile_stack_need = std::size_t(4) << 20;
inline constexpr std::size_t compile_stack_reserve = std::size_t(8) << 20;

// Bytes of the calling thread's stack left below this frame; SIZE_MAX where that is not known.
inline std::size_t stack_left() {
#ifdef _WIN32
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    const auto here = reinterpret_cast<ULONG_PTR>(&low);
    return here > low ? std::size_t(here - low) : 0;
#else
    return SIZE_MAX;
#endif
}

template<typename F>
auto with_compile_stack(F&& f) -> std::invoke_result_t<F&> {
    using Result = std::invoke_result_t<F&>;
    static_assert(!std::is_void_v<Result> && std::is_default_constructible_v<Result>,
                  "with_compile_stack returns the callable's result");
#ifdef _WIN32
    if (stack_left() < compile_stack_need) {
        struct Call {
            std::remove_reference_t<F>* f;
            Result result{};
        } call{&f};
        const auto run = [](LPVOID p) -> DWORD {
            auto* c = static_cast<Call*>(p);
            c->result = (*c->f)();
            return 0;
        };
        if (HANDLE thread = CreateThread(nullptr, compile_stack_reserve, run, &call,
                                         STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr)) {
            WaitForSingleObject(thread, INFINITE);
            CloseHandle(thread);
            return call.result;
        }
        // No thread: compile here, as before.
    }
#endif
    return f();
}

}  // namespace leonida
