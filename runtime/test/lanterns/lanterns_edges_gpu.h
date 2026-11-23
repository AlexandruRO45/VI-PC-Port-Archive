// Lanterns stage 1, the shared edges on the GPU (lanterns_edges.h): its own root signature (eight constants, then a
// table of two buffer SRVs, so the table pointer lands at s8 as in the title's shaders), PSO with additive blending,
// target, list and fence, on a Gpu's device and direct queue.
#pragma once

#include "lanterns_edges.h"
#include "lanterns_gpu.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

struct EdgeShaders {
    std::vector<std::uint8_t> vs, ps;
};
bool load_edge_shaders(const std::filesystem::path& dir, EdgeShaders& out);

class EdgeDraw {
public:
    bool create(Gpu& gpu, const EdgeShaders& shaders);
    // Clears the target to 0, draws the fan's 24 vertices once and reads the target back.
    bool run(std::vector<Color>& pixels);

private:
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_;
    Ref<ID3D12DescriptorHeap> heap_, rtv_heap_;
    Ref<ID3D12Resource> vertices_, alpha_, target_, readback_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    std::uint64_t heap_gpu_ = 0;
    UINT64 fence_value_ = 0;
};

}  // namespace lanterns