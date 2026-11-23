// Lanterns stage 20: the textures, the root signature with its static samplers, and the dispatch of
// cs_lanterns_samplers on a Gpu's device and direct queue (lanterns_samplers.h).
#pragma once

#include "lanterns_gpu.h"
#include "lanterns_samplers.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

constexpr std::uint32_t kSamplerPoison = 0xdeadbeefu;   // what the output holds before the dispatch

bool load_samplers_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out);

class SamplerKernel {
public:
    // Uploads the colour texture, clears the depth texture quadrant by quadrant, and makes the pipeline. The root
    // signature names the three samplers as static samplers, as a title's does; the shader holds them as literals.
    bool create(Gpu& gpu, const std::vector<std::uint8_t>& shader);
    // One dispatch of one group; out[lane] holds the four samples of lane's probe (see sampler_expect).
    bool run(std::array<std::array<std::uint32_t, 4>, kSamplerProbes>& out);

private:
    bool submit_and_wait();

    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_;
    Ref<ID3D12DescriptorHeap> heap_, dsv_heap_;
    Ref<ID3D12Resource> colour_, depth_, staging_, output_, poison_, readback_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    UINT64 fence_value_ = 0;
    std::uint64_t heap_gpu_ = 0;
};

}  // namespace lanterns
