// Lanterns stage 24: the dispatch of cs_lanterns_xy on a Gpu's device and direct queue (lanterns_xy.h).
#pragma once

#include "lanterns_gpu.h"
#include "lanterns_xy.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

bool load_xy_shader(const std::filesystem::path& dir, std::vector<std::uint8_t>& out);

class XyKernel {
public:
    bool create(Gpu& gpu, const std::vector<std::uint8_t>& shader);
    // One Dispatch(kXyGroupsX, kXyGroupsY, 1) into a poisoned output; out holds two words per grid position, row major.
    bool run(std::vector<std::uint32_t>& out);

private:
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_;
    Ref<ID3D12DescriptorHeap> heap_;
    Ref<ID3D12Resource> output_, poison_, readback_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    UINT64 fence_value_ = 0;
    std::uint64_t heap_gpu_ = 0;
};

}  // namespace lanterns
