// Lanterns stage 22: the six targets, the 5- and 6-target pipelines of ps_lanterns_gbuf6 (with vs_lanterns_full), and the
// decode pass on a Gpu's device and direct queue (lanterns_gbuf6.h).
#pragma once

#include "lanterns_gbuf6.h"
#include "lanterns_gpu.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

struct Gbuf6Shaders {
    std::vector<std::uint8_t> vs_full, ps_gbuf6, cs_decode;
};
bool load_gbuf6_shaders(const std::filesystem::path& dir, Gbuf6Shaders& out);

// One run: every target's bytes (row-major, tightly packed at its own texel size) and the decode output.
struct Gbuf6Result {
    std::array<std::vector<std::uint8_t>, kGbufTargets> targets;
    std::array<unsigned, kGbufTargets> texel_bytes{};
    std::vector<std::uint32_t> decoded;   // 8 words a pixel
};

class Gbuf6 {
public:
    bool create(Gpu& gpu, const Gbuf6Shaders& shaders);
    // Clears all six targets to 0, draws with the 5- or 6-target pipeline, decodes, reads everything back.
    bool run(bool six, Gbuf6Result& out);

private:
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12RootSignature> graphics_root_, compute_root_;
    Ref<ID3D12PipelineState> pso5_, pso6_, decode_;
    Ref<ID3D12DescriptorHeap> heap_, rtv_heap_;
    std::array<Ref<ID3D12Resource>, kGbufTargets> targets_;
    Ref<ID3D12Resource> output_, readback_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    UINT64 fence_value_ = 0;
    std::uint64_t heap_gpu_ = 0;
    std::uint64_t rtv_cpu_ = 0;
    UINT rtv_step_ = 0;
    std::array<std::uint64_t, kGbufTargets> readback_offset_{};
    std::uint64_t decoded_offset_ = 0;
};

}  // namespace lanterns
