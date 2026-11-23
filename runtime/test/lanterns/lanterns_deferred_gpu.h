// Lanterns stage 9: the tiled lighting compute pass on the G-buffer a Gpu (created with R8G8B8A8_UNORM and the stage-9
// shaders) rendered. Its own root signature, heap, list and fence, on the Gpu's device and direct queue.
#pragma once

#include "lanterns_deferred.h"
#include "lanterns_gpu.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

bool load_tiled_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out);

constexpr std::uint32_t kDeferredPoison = 0xdeadbeefu;

class Deferred {
public:
    bool create(Gpu& gpu, const std::vector<std::uint8_t>& shader);
    // Uploads the lights, poisons the three outputs (kDeferredPoison), runs the pass over the Gpu's last frame with
    // these constants (deferred_constants) and reads back the packed pixels, each tile's light count and each tile's
    // list of kept light indices (kLights words a tile, the first count of them written).
    bool run(const std::vector<PointLight>& lights, const std::array<float, 16>& constants,
             std::vector<std::uint32_t>& pixels, std::vector<std::uint32_t>& tile_counts,
             std::vector<std::uint32_t>& tile_lists);
    // The G-buffer's normal (RGB10A2 words) and view depth (R32_FLOAT) targets of the Gpu's last frame.
    bool read_gbuffer(std::vector<std::uint32_t>& normal, std::vector<float>& depth);

private:
    Gpu* gpu_ = nullptr;
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_;
    Ref<ID3D12DescriptorHeap> heap_;
    Ref<ID3D12Resource> lights_, output_, counts_, lists_, poison_, readback_, gbuffer_readback_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    std::uint64_t fence_value_ = 0, heap_gpu_ = 0;
    bool first_run_ = true;
};

}  // namespace lanterns
