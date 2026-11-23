// Generates fixed trace records for the mapped-file tests and the synthetic writer benchmark.
#include "../shims/d3d12_trace.h"

#include <array>
#include <chrono>
#include <cwchar>
#include <thread>

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) return 2;
    using rt::d3d12_trace::record;
    const bool stderr_mode = std::wcscmp(argv[1], L"stderr") == 0 || std::wcscmp(argv[1], L"bench-stderr") == 0;
    const bool crash = std::wcscmp(argv[1], L"crash") == 0;
    const bool bench = std::wcsncmp(argv[1], L"bench-", 6) == 0;
    const bool small = std::wcscmp(argv[1], L"small") == 0;
    const bool concurrent = std::wcscmp(argv[1], L"concurrent") == 0;
    if (stderr_mode) {
        rt::d3d12_trace::set_enabled(true);
    } else {
        if (argc != 3) return 2;
        const DWORD size = small ? 512 : bench ? 8 * 1024 * 1024 : 128 * 1024;
        if (const DWORD result = rt::d3d12_trace::open_file(argv[2], size)) {
            std::fprintf(stderr, "open failed: %lu\n", result);
            return 3;
        }
    }
    if (bench) {
        constexpr int count = 100'000;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) record("ID3D12CommandQueue::ExecuteCommandLists", "ok");
        const auto recorded = std::chrono::steady_clock::now();
        rt::d3d12_trace::close_file();
        const auto finished = std::chrono::steady_clock::now();
        std::printf("%d records, writer %.3f ms, total %.3f ms\n", count,
                    std::chrono::duration<double, std::milli>(recorded - start).count(),
                    std::chrono::duration<double, std::milli>(finished - start).count());
        return 0;
    }
    record("D3D12XboxCreateDevice", "ok");
    rt::d3d12_trace::result("ID3D12Device::CreateCommandQueue", E_INVALIDARG);
    rt::d3d12_trace::result("ID3D12GraphicsCommandList::SetProtectedResourceSession", E_NOTIMPL);
    if (small) {
        for (int i = 0; i < 40; ++i) record("ID3D12CommandQueue::ExecuteCommandLists", "ok");
        if (!rt::d3d12_trace::enabled()) return 4;
        record("Fixture_AfterLimit", "called");
        if (!rt::d3d12_trace::enabled()) return 5;
    } else if (concurrent) {
        std::array<std::jthread, 4> workers;
        for (unsigned thread = 0; thread < workers.size(); ++thread) {
            workers[thread] = std::jthread([thread] {
                for (unsigned call = 0; call < 128; ++call) {
                    char method[48];
                    std::snprintf(method, sizeof(method), "Fixture_%u_%u", thread, call);
                    record(method);
                }
            });
        }
    }
    if (crash) {
        // The parent waits for this acknowledgement, then kills the process without close_file().
        std::puts("published");
        std::fflush(stdout);
        Sleep(INFINITE);
    }
    rt::d3d12_trace::close_file();
    if (small && rt::d3d12_trace::enabled()) return 6;
    return 0;
}
