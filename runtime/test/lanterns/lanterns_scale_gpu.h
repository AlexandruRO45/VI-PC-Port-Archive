// Lanterns, stage 25: the GPU side of the scale frame (lanterns_scale.h), through the D3D12 API only.
//
// One frame, f = 1, 2, ... (parity p = f % 2; two frames in flight):
//   COPY queue:    Wait(direct, f - 2); stream[p] <- zeros, barrier, stream[p] <- the instanced draws' records; 16 MiB
//                  of filler into a pool buffer in 1 MiB copies; Signal(copy, f)
//   COMPUTE queue: Wait(direct, f - 2); particle set A[p] <- its first state, then kStepsA rise dispatches with UAV
//                  barriers; Signal(compute, f)
//   DIRECT queue:  [pre: clears]; Wait(copy, f); Wait(compute, f);
//                  [bind, main pass in one list per recording thread, post: particle set B by kStepsB dispatches on the
//                  direct queue, the 4x MSAA aux pass and its resolve, the RGBA8 top pass, copies to readback[p], then
//                  zeros over stream[p] and A[p]]; Signal(direct, f)
// The direct queue's part is submitted first, so its Waits come before the Signals they wait for; the copy and compute
// parts follow, and they also wait for CPU gates: one opens when the CPU starts frame f + 1, the other later, when it
// waits for frame f; copy takes the later one on even frames, compute on odd ones. So the producers of frame f run
// late, and the later one after the other. The CPU opens the late gate only after the direct queue signals that its main
// pass is done, or after kHoldMs. A direct queue that skipped both Waits reads zeros in every frame, and one that skipped
// only its Wait on the copy (or the compute) queue does in every other frame. The CPU waits for the direct queue's frame f - 2 before it records frame f (allocators, descriptor
// ring, readback[p]); the copy and compute queues' frame f - 2 is done by then, since that frame waited for them.
// Because the direct queue zeroes stream[p] and A[p] after use, a frame that does not wait for the copy or compute
// queue draws zeros (nothing) there and differs from frame 1.
// Every draw writes its instance V# into a CPU-only staging heap once (per parity) and copies it each frame, with
// CopyDescriptorsSimple, into a ring in the shader-visible heap; its descriptor table points there. The main pass is
// recorded by worker threads in parallel. The frame's content does not change from frame to frame: every readback
// must be the same bytes.
#pragma once

#include "lanterns_gpu.h"
#include "lanterns_scale.h"

#include <atomic>
#include <condition_variable>
#include <initializer_list>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lanterns::scale {

// The readback of one frame: the main target, the resolved aux target, the top target, then particle sets A and B.
constexpr std::uint64_t kMainPitch = std::uint64_t(kW) * 8, kAuxPitch = std::uint64_t(kAuxW) * 8,
                        kTopPitch = std::uint64_t(kTopW) * 4;
constexpr std::uint64_t kOffMain = 0, kOffAux = kOffMain + kMainPitch * kH, kOffTop = kOffAux + kAuxPitch * kAuxH,
                        kOffPartA = kOffTop + kTopPitch * kTopH,
                        kOffPartB = kOffPartA + kParticles * sizeof(Instance),
                        kReadbackBytes = kOffPartB + kParticles * sizeof(Instance);

struct FrameStats {
    unsigned draws = 0, dispatches = 0, lists = 0, executes = 0, barriers = 0, descriptor_copies = 0, pso_changes = 0,
             rs_changes = 0, copies = 0;
};

class ScaleGpu {
public:
    ~ScaleGpu();
    bool create(const World& world, const Shaders& shaders, unsigned threads);
    // Up to max_us of random delay (a fixed seed) before each queue's submission, the recording threads' start, and
    // each worker's chunk: the frame's content stays the same, its timing across the queues does not.
    void set_jitter(unsigned max_us) { jitter_us_ = max_us; }
    // Records and submits frame f (f = 1, 2, ...; in order). Waits first, at most timeout_ms, for frame f - 2.
    enum class Outcome { ok, hang, error };
    Outcome submit(std::uint64_t f, unsigned timeout_ms);
    // Waits for frame f (at most timeout_ms) and maps its readback.
    Outcome wait(std::uint64_t f, unsigned timeout_ms);
    const std::uint8_t* map_readback(std::uint64_t f);
    void unmap_readback(std::uint64_t f);
    bool idle(unsigned timeout_ms);
    // The completed values of the three fences, and the queue-level log of the last frames.
    std::string fence_status() const;
    std::string frame_log(std::uint64_t f) const;
    const FrameStats& stats() const { return stats_; }
    unsigned pipelines() const { return pipelines_; }
    unsigned root_signatures() const { return root_signatures_; }

private:
    struct Call {
        unsigned ring;
        ID3D12PipelineState* pso;
        unsigned rs;
        std::uint64_t mesh_table;
        std::uint32_t index_count, first_index, instances;
        float constants[16];
    };
    struct Pass {
        std::uint64_t camera;   // root CBV
        D3D12_CPU_DESCRIPTOR_HANDLE rtv, dsv;
        unsigned width, height;
    };
    bool create_pipelines(const Shaders& shaders);
    bool create_resources();
    bool fill_descriptors();
    bool initialize();
    void plan();
    void record_calls(ID3D12GraphicsCommandList* list, unsigned p, const Pass& pass, const Call* calls, std::size_t n,
                      FrameStats& st);
    void bind_root_signature(ID3D12GraphicsCommandList* list, unsigned rs, std::uint64_t camera);
    bool record_chunk(unsigned p, unsigned t);
    bool record_copy(unsigned p), record_compute(unsigned p), record_pre(unsigned p), record_bind(unsigned p),
        record_post(unsigned p);
    void compute_steps(ID3D12GraphicsCommandList* list, std::uint64_t table, unsigned steps, ID3D12Resource* state,
                       ID3D12Resource* instances, FrameStats& st);
    FrameStats main_stats_;
    Outcome wait_fence(ID3D12Fence* fence, std::uint64_t value, unsigned timeout_ms);
    void worker(unsigned t);
    void log_event(std::uint64_t f, const char* what, std::uint64_t value = 0);

    const World* world_ = nullptr;
    unsigned threads_ = 1, pipelines_ = 0, root_signatures_ = 0;
    FrameStats stats_;
    Ref<ID3D12Device> device_;
    Ref<ID3D12RootSignature> rs_[kRootSignatures], compute_rs_;
    Ref<ID3D12PipelineState> lit_[kRootSignatures][kVariants], floor_, aux_lit_[3], aux_floor_, top_lit_[3], top_floor_,
        rise_;
    Ref<ID3D12Resource> vb_, ib_, static_inst_, stream_[2], pool_, upload_, pool_upload_, readback_[2];
    Ref<ID3D12Resource> pristine_a_, pristine_b_, state_a_[2], inst_a_[2], state_b_, inst_b_;
    Ref<ID3D12Resource> main_rt_, main_depth_, aux_msaa_, aux_depth_, aux_resolved_, top_rt_, top_depth_;
    Ref<ID3D12DescriptorHeap> srv_heap_, staging_heap_, rtv_heap_, dsv_heap_;
    Ref<ID3D12CommandQueue> direct_, compute_, copy_;
    Ref<ID3D12Fence> direct_fence_, compute_fence_, copy_fence_;
    // The CPU's gates. Frame f's copy and compute parts each wait for one of them to reach f: the early gate opens
    // when the CPU starts frame f + 1, the late one when it then waits for frame f (or goes idle). On even frames the
    // copy part takes the late gate and compute the early one, on odd frames the other way round, so both always run
    // after the direct queue's part of frame f was submitted, and one of them after the other.
    Ref<ID3D12Fence> gate_early_, gate_late_;
    // The direct queue signals consumed_ = f after frame f's main pass. Before opening the late gate for frame f the
    // CPU waits for it, at most kHoldMs: with every Wait in place it cannot come (the main pass waits for the late
    // producer), so every frame pays kHoldMs; without the Wait on the late producer it comes first.
    Ref<ID3D12Fence> consumed_;
    static constexpr unsigned kHoldMs = 100;
    bool open_gate(ID3D12Fence* gate, std::uint64_t value);
    Ref<ID3D12CommandAllocator> copy_alloc_[2], compute_alloc_[2], main_alloc_[2][3];   // pre, bind, post
    Ref<ID3D12GraphicsCommandList> copy_list_[2], compute_list_[2], main_list_[2][3];
    std::vector<Ref<ID3D12CommandAllocator>> chunk_alloc_[2];
    std::vector<Ref<ID3D12GraphicsCommandList>> chunk_list_[2];
    HANDLE event_ = nullptr;

    std::uint64_t srv_cpu_ = 0, srv_gpu_ = 0, staging_cpu_ = 0;
    UINT step_ = 0;
    std::uint64_t ring_ = 0;   // ring slots per parity
    std::uint64_t up_vb_ = 0, up_ib_ = 0, up_inst_ = 0, up_stream_ = 0, up_zero_ = 0, up_cam_ = 0, up_extra_ = 0,
                  up_particles_ = 0, stream_bytes_ = 0, inst_bytes_ = 0;
    std::vector<std::uint64_t> first_index_, index_count_, vertex_first_, vertex_count_;
    std::vector<std::uint64_t> inst_offset_;   // per draw: its records in static_inst_ or stream_[p]
    std::vector<Call> main_calls_, aux_calls_, top_calls_;
    std::vector<std::uint32_t> ring_draw_;     // per ring slot: the draw whose instance V# it holds
    std::vector<std::uint8_t> ring_set_b_;     // 1: particle set B (the aux pass's points)
    void begin_pass(ID3D12GraphicsCommandList* list, const Pass& pass, FrameStats& st);
    void barrier(ID3D12GraphicsCommandList* list, std::initializer_list<D3D12_RESOURCE_BARRIER> barriers, FrameStats& st);
    std::vector<std::size_t> chunk_begin_;     // main_calls_ split per thread (threads_ + 1 bounds)
    Pass main_pass_{}, aux_pass_{}, top_pass_{};

    // Workers: record chunk t of frame job_frame_.
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_, done_cv_;
    std::uint64_t job_frame_ = 0;
    unsigned done_ = 0;
    bool quit_ = false, worker_failed_ = false;
    std::vector<FrameStats> worker_stats_;
    unsigned jitter_us_ = 0;
    std::atomic<std::uint64_t> jitter_state_{0x9e3779b97f4a7c15ull};
    void jitter();

    struct Event { std::uint64_t frame; double ms; const char* what; std::uint64_t value; };
    std::vector<Event> log_;
    std::mutex log_mutex_;
    LARGE_INTEGER qpc0_{}, qpf_{};
};

}  // namespace lanterns::scale