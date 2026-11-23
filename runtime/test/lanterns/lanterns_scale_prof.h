// Lanterns, stage 25: where the layer spends CPU time. Two instruments, both inside the guest because the pod has no
// perf_event access:
//  - call-class timers: every D3D12 call the stage makes goes through TIMED(class, call), which reads the TSC before
//    and after and adds the difference to the calling thread's counters (one slot per class, no locks); the totals
//    are converted to ns with the TSC rate measured against steady_clock over the run;
//  - a sampling profiler: a thread that, every period, suspends each thread of the process in turn, reads its RIP
//    with GetThreadContext and resumes it. RIPs are grouped by module (gta6rt.exe is the layer) and written as RVAs,
//    which tools symbolize offline against the PDB (llvm-symbolizer). Between suspend and resume the sampler touches
//    only memory it allocated beforehand.
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <intrin.h>
#include <string>
#include <thread>
#include <vector>

namespace lanterns::scale {

enum CallClass : unsigned {
    kDraw, kDispatch, kSetPipelineState, kSetGraphicsRootSignature, kSetComputeRootSignature, kSetRootTable,
    kSetRootConstants, kSetRootCbv, kCopyDescriptors, kResourceBarrier, kExecuteCommandLists, kExecuteSmall, kCloseList,
    kResetList, kResetAllocator, kClearView, kSetTargetsViewports, kSetIndexBufferTopology, kSetDescriptorHeaps,
    kCopyBuffer, kCopyTexture, kResolve, kSignal, kQueueWait, kFenceWaitCpu, kMapUnmap, kClassCount
};

inline const char* call_class_name(unsigned c) {
    static const char* names[kClassCount] = {
        "DrawIndexedInstanced", "Dispatch", "SetPipelineState", "SetGraphicsRootSignature", "SetComputeRootSignature",
        "Set*RootDescriptorTable", "Set*Root32BitConstants", "SetGraphicsRootConstantBufferView",
        "CopyDescriptorsSimple", "ResourceBarrier", "ExecuteCommandLists (direct: main, aux, top passes)",
        "ExecuteCommandLists (copy, compute, direct pre)", "Close", "Reset (list)",
        "Reset (allocator)", "ClearRenderTargetView/DepthStencilView", "OMSetRenderTargets/RSSet*",
        "IASetIndexBuffer/IASetPrimitiveTopology", "SetDescriptorHeaps", "CopyBufferRegion", "CopyTextureRegion",
        "ResolveSubresource", "ID3D12CommandQueue::Signal", "ID3D12CommandQueue::Wait",
        "fence wait on the CPU (SetEventOnCompletion + wait)", "Map/Unmap"};
    return names[c];
}

struct ClassCounters {
    std::uint64_t count[kClassCount] = {}, ticks[kClassCount] = {};
};

class Timers {
public:
    // The calling thread's counters (registered on first use).
    ClassCounters& local();
    bool enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }
    void reset();
    ClassCounters total() const;
    void start_clock();
    double ns_per_tick() const;   // from start_clock() to now
private:
    bool enabled_ = true;
    mutable SRWLOCK lock_ = SRWLOCK_INIT;
    std::vector<ClassCounters*> threads_;
    std::uint64_t tsc0_ = 0;
    LARGE_INTEGER qpc0_{};
};
Timers& timers();

// TIMED(kDraw, list->DrawIndexedInstanced(...)); for calls that return a value, TIMED_R(kClass, hr = queue->Signal(...)).
#define TIMED(cls, call) do { \
    auto& lanterns_tc_ = ::lanterns::scale::timers().local(); \
    const std::uint64_t lanterns_t0_ = __rdtsc(); \
    call; \
    lanterns_tc_.ticks[cls] += __rdtsc() - lanterns_t0_; ++lanterns_tc_.count[cls]; \
} while (0)

class Sampler {
public:
    // Samples every thread of the process but its own every period_us until stop(), up to max_samples.
    bool start(unsigned period_us, std::size_t max_samples);
    void stop();
    // module, RVA, thread index, count: one line per distinct (module, RVA, thread); plus the module table.
    bool write(const std::filesystem::path& file) const;
    std::size_t samples() const { return count_.load(); }
private:
    void run();
    struct Sample { std::uint64_t rip; std::uint32_t thread; };
    std::vector<Sample> samples_;
    std::atomic<std::size_t> count_{0};
    std::atomic<bool> stop_{false};
    unsigned period_us_ = 1000;
    std::thread thread_;
    std::vector<DWORD> thread_ids_;   // index -> Windows thread id, filled by the sampler
};

}  // namespace lanterns::scale