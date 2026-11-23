// Lanterns stage 11: the acceleration structures and the shadow-ray pass (lanterns_rt.h, cs_lanterns_rt_shadow.asm).
// The BLAS (the prism) and two TLASes (the ten instances; the same with one instance's mask 0, the control) are built
// through ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo and
// ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure on the Gpu's device and queue.
#pragma once

#include "lanterns_gpu.h"
#include "lanterns_rt.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

bool load_rt_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out);

// traced: the TLAS as drawn; masked: one instance's mask 0; swapped: kRtSwapA and kRtSwapB name each other's BLAS;
// stall: a self-referencing BVH only the step cap ends; overflow: one whose stack grows by three a step (see build).
enum class RtRun { traced, masked, swapped, stall, overflow };
constexpr unsigned kRtTlases = 3;
constexpr std::uint32_t kRtPoison = 0xdeadbeefu;    // what the output and step buffers hold before each pass
constexpr std::uint32_t kRtOverflow = 0x80000000u;  // the step word's bit for a lane stopped by a full stack

struct RtSizes {
    std::uint64_t blas = 0, tlas = 0, scratch = 0;
};

class RtShadows {
public:
    // masked: the TLAS instance whose mask is 0 in the control TLAS.
    bool create(Gpu& gpu, const std::vector<std::uint8_t>& shader, const Scene& scene, unsigned masked);
    // Traces the Gpu's last frame through one of the TLASes or fixtures (RtRun). Both buffers are poisoned first.
    bool run(const Scene& scene, RtRun which, std::vector<std::uint32_t>& pixels, std::vector<std::uint32_t>& steps);
    const RtSizes& sizes() const { return sizes_; }

private:
    bool build(const Scene& scene, unsigned masked);

    Gpu* gpu_ = nullptr;
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12Device5> device5_;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_;
    Ref<ID3D12DescriptorHeap> heap_;
    Ref<ID3D12Resource> geometry_, blas_[kRtMeshes], tlas_[kRtTlases], scratch_, output_, steps_, readback_, stall_, poison_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList4> list_;
    Ref<ID3D12Fence> fence_;
    RtSizes sizes_;
    std::uint64_t fence_value_ = 0, heap_cpu_ = 0, heap_gpu_ = 0;
    bool first_run_ = true;
};

}  // namespace lanterns
