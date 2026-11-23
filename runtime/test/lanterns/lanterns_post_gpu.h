// Lanterns stage 5: the post-process on the GPU (what it computes: lanterns_post.h). It records on the device and the
// direct queue of a Gpu whose target is R16G16B16A16_FLOAT, after that Gpu's frame: two wave64 compute passes with LDS
// and s_barrier, then a fullscreen pixel shader that reads the scene through a T# and the blur through a V#.
#pragma once

#include "lanterns_gpu.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

struct PostShaders {
    std::vector<std::uint8_t> bloom_h, bloom_h_no_barrier, bloom_v, vs_full, ps_composite;
};
bool load_post_shaders(const std::filesystem::path& dir, PostShaders& out);

enum class PostMode {
    full,         // blur rows, blur columns, composite
    no_barrier,   // the same with cs_lanterns_bloom_h_nobarrier (a measured control)
    skip_blur,    // composite with a zero bloom buffer (the control the reference predicts)
};

class Post {
public:
    // The Gpu must outlive this object (its device owns everything here) and must have been created with the fp16 target.
    bool create(Gpu& gpu, const PostShaders& shaders);
    // One submission on the Gpu's queue, waited for: the passes of the mode and a readback of the final target.
    bool run(PostMode mode, std::vector<Color>& pixels);

private:
    bool create_pipelines(const PostShaders& shaders);
    bool create_resources();
    bool fill_descriptors();
    void record_blur(PostMode mode);

    ID3D12Device* device_ = nullptr;        // the Gpu's
    ID3D12CommandQueue* queue_ = nullptr;   // the Gpu's direct queue
    ID3D12Resource* scene_ = nullptr;       // the Gpu's target
    Ref<ID3D12RootSignature> compute_root_, graphics_root_;
    Ref<ID3D12PipelineState> bloom_h_, bloom_h_no_barrier_, bloom_v_, composite_;
    Ref<ID3D12Resource> rows_, bloom_, zero_, final_, readback_;
    Ref<ID3D12DescriptorHeap> heap_, rtv_heap_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    std::uint64_t heap_gpu_ = 0;
    UINT step_ = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_{};
    UINT64 fence_value_ = 0;
};

}  // namespace lanterns
