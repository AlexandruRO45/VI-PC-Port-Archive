// tiny_guest_m34.exe: the M3 frames and the M4 compute workload as ONE `gta6rt --run` guest, on ONE device, from several
// threads at once (test/run_guest_frame.cmake, tests d3d12_m34_concurrent_guest). A title records and submits from more
// than one thread and keeps a graphics queue, a compute queue and a copy queue busy together; the single-threaded
// guests (tiny_guest_m3, tiny_guest_m4) never have two of them in flight on the layer at the same time.
//
//   1. D3D12XboxCreateDevice once (the import the game uses; the shim returns the Vulkan device behind the proxy).
//   2. Sequentially, on that device: the six M3 frames (test/d3d12_m3_frame_test.cpp), then the M4 workload
//      (test/d3d12_m4_compute_test.cpp). Both are the host tests' bodies, built with LEONIDA_GUEST_LIB, so every pixel
//      of three targets per frame and every word of the compute results is compared with the CPU reference as there.
//   3. Concurrently, on the same device: two threads running the M3 frames `rounds` times each (each with its own
//      direct queue, allocators, lists, targets, queries and fences) and one thread running the M4 workload
//      2 * `rounds` times (its own copy and compute queues), released together by an event. Every run must match.
//
// Two threads of M3 and one of M4 touch the device's shared state together: the GPU address arena, the descriptor
// hash table and the bindless set, shader and pipeline creation, the layout tracker, the fence dispatcher and the
// VkQueues (where the family has fewer queues than D3D12 queues). A race there shows up as a wrong pixel, a wrong
// word, a hung fence (the guest gives up on it after 20 s) or a crash, never as a validation message (the guest has no
// access to the runtime's internals).
//
// Usage: tiny_guest_m34 <folder build.py wrote with the M3 and the M4 shaders and LEONIDA_USER_DATA_SHIFT=4>
//                       [rounds, default 3] [--imprecise-occlusion]
// On success it prints one marker line, after every check. No game data, no shader but the hand-written fixtures.
#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Resolved at link time against test/fixtures/d3d12_xs_stub.cpp (which fails), at run time by gta6rt's shim registry.
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params,
                                                                      const IID* riid, void** out);

// test/d3d12_m3_frame_test.cpp and test/d3d12_m4_compute_test.cpp, built with LEONIDA_M3_GUEST / LEONIDA_M4_GUEST and
// LEONIDA_GUEST_LIB (no main of their own).
int m3_guest_run(ID3D12Device* shared_device, const std::filesystem::path& root_dir, bool first_pass_only,
                 bool precise_occlusion, unsigned long long* samples);
int m4_guest_run(ID3D12Device* shared_device, const std::filesystem::path& root_dir, unsigned* words);

namespace {

struct Tally {
    std::atomic<unsigned> m3_runs{0}, m4_runs{0}, failures{0};
    std::atomic<unsigned long long> samples{0}, words{0};
};

// When each run of the concurrent phase started and ended (QueryPerformanceCounter ticks) and on which thread, so the
// guest can say how many runs really overlapped another thread's run: "three threads at once" is then measured, not
// assumed.
struct Span {
    unsigned thread;
    LONGLONG begin, end;
};
std::mutex g_spans_mutex;
std::vector<Span> g_spans;

LONGLONG now() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the marker reaches the launcher even if the process dies after it
    unsigned rounds = 3;
    bool precise_occlusion = true, usage_ok = argc >= 2;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--imprecise-occlusion") == 0) precise_occlusion = false;
        else if (i == 2 && std::atoi(argv[i]) > 0 && std::atoi(argv[i]) <= 50) rounds = unsigned(std::atoi(argv[i]));
        else usage_ok = false;
    }
    if (!usage_ok) {
        std::fprintf(stderr, "usage: tiny_guest_m34 <folder build.py wrote (spirv/index.json, fssm/*.fssm)> "
                             "[rounds 1..50] [--imprecise-occlusion]\n");
        return 2;
    }
    const std::filesystem::path root_dir = argv[1];

    const std::uint32_t device_params[24] = {};  // the 24-word block the shim logs; it does not create from it
    const IID device_iid = __uuidof(ID3D12Device);
    ID3D12Device* device = nullptr;
    const HRESULT created = D3D12XboxCreateDevice(nullptr, device_params, &device_iid, reinterpret_cast<void**>(&device));
    if (FAILED(created) || !device) {
        std::fprintf(stderr, "D3D12XboxCreateDevice failed (hr=0x%08lx)\n", static_cast<unsigned long>(created));
        return 1;
    }

    Tally tally;
    constexpr unsigned sequential = ~0u;  // the sequential phase records no span
    const auto record = [&](unsigned thread, LONGLONG begin) {
        std::lock_guard lock(g_spans_mutex);
        g_spans.push_back({thread, begin, now()});
    };
    const auto run_m3 = [&](const char* where, bool first_pass_only, unsigned thread) {
        unsigned long long samples = 0;
        const LONGLONG begin = now();
        const int rc = m3_guest_run(device, root_dir, first_pass_only, precise_occlusion, &samples);
        if (thread != sequential) record(thread, begin);
        if (rc != 0) {
            std::fprintf(stderr, "M3 frames (%s) failed with %d\n", where, rc);
            ++tally.failures;
            return;
        }
        ++tally.m3_runs;
        tally.samples += samples;
    };
    const auto run_m4 = [&](const char* where, unsigned thread) {
        unsigned words = 0;
        const LONGLONG begin = now();
        const int rc = m4_guest_run(device, root_dir, &words);
        if (thread != sequential) record(thread, begin);
        if (rc != 0) {
            std::fprintf(stderr, "M4 workload (%s) failed with %d\n", where, rc);
            ++tally.failures;
            return;
        }
        ++tally.m4_runs;
        tally.words += words;
    };

    // 1. One after the other on the shared device: the frames, then the compute workload.
    run_m3("sequential", false, sequential);
    run_m4("sequential", sequential);
    if (tally.failures) return 1;

    // 2. Together: two threads of frames and one of compute, released at the same moment.
    const HANDLE go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!go) return 1;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < 2; ++t)
        threads.emplace_back([&, t] {
            WaitForSingleObject(go, INFINITE);
            for (unsigned r = 0; r < rounds; ++r) run_m3(t == 0 ? "thread A" : "thread B", (r + t) % 3 == 2, t);
        });
    threads.emplace_back([&] {
        WaitForSingleObject(go, INFINITE);
        for (unsigned r = 0; r < 2 * rounds; ++r) run_m4("thread C", 2);
    });
    SetEvent(go);
    for (auto& thread : threads) thread.join();
    CloseHandle(go);
    device->Release();

    const unsigned want_m3 = 1 + 2 * rounds, want_m4 = 1 + 2 * rounds;
    if (tally.failures || tally.m3_runs != want_m3 || tally.m4_runs != want_m4) {
        std::fprintf(stderr, "%u failures, %u of %u M3 runs and %u of %u M4 runs completed\n", tally.failures.load(),
                     tally.m3_runs.load(), want_m3, tally.m4_runs.load(), want_m4);
        return 1;
    }
    // How many concurrent-phase runs overlapped in time with a run of another thread.
    unsigned overlapping = 0;
    for (const Span& a : g_spans)
        for (const Span& b : g_spans)
            if (a.thread != b.thread && a.begin < b.end && b.begin < a.end) {
                ++overlapping;
                break;
            }
    if (overlapping == 0) {
        std::fprintf(stderr, "no run of one thread overlapped a run of another: the concurrent phase did not run "
                             "concurrently\n");
        return 1;
    }
    std::printf("M34 guest: PASS, %u M3 runs and %u M4 runs on one device, %u of them from three threads at once, "
                "every pixel and word exact (%llu samples, %llu words, %u runs overlapped another thread's)\n",
                tally.m3_runs.load(), tally.m4_runs.load(), 4 * rounds, tally.samples.load(), tally.words.load(),
                overlapping);
    return 0;
}
