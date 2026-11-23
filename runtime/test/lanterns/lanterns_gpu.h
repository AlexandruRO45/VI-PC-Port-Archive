// Lanterns: the GPU side, through the D3D12 API only. The device comes from the D3D12XboxCreateDevice the guest
// imports from d3d12_xs.dll (test/fixtures/tiny_frame.cpp does the same); pipelines come from the shader
// containers fssm_pack.py wrote. See lanterns.cpp for what each stage draws and checks.
#pragma once

#include "../d3d12_test_shaders.h"
#include "lanterns_scene.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lanterns {

using Color = std::array<float, 4>;
using d3d12test::Ref;

// The containers the guest hands to the pipelines, read from the folder fssm_pack.py wrote.
struct Shaders {
    std::vector<std::uint8_t> vs, ps_lit, ps_floor, unknown_vs;
};
bool load_shaders(const std::filesystem::path& dir, Shaders& out);

// The scene's meshes as one vertex array and one 16-bit index array (each face fanned), with where each mesh starts,
// and every draw's instances.
struct Geometry {
    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;
    std::vector<std::uint32_t> first_vertex, vertex_count, first_index, index_count;
    std::vector<Instance> instances;
    std::vector<std::uint32_t> first_instance;
};

class Gpu {
public:
    // Device, pipelines, buffers, and an initial direct-queue submission that copies the uploads into the DEFAULT
    // buffers.
    bool create(const Scene& scene, const Shaders& shaders);
    // Records and submits one frame on the direct queue.
    bool submit_frame(const Scene& scene, bool depth_test);
    // Waits for the last frame and reads back its pixels.
    bool finish_frame(std::vector<Color>& pixels);
    // Waits until the queue is idle (before the objects are released).
    bool idle();

private:
    bool create_pipelines(const Shaders& shaders);
    void bind_draw(const Scene& scene, std::size_t d);
    bool create_resources(const Scene& scene);
    bool initialize(const Scene& scene);
    bool fill_descriptors(const Scene& scene);

    // Declared first, released last: everything below is a child of the device.
    Ref<ID3D12Device> device_;
    Ref<ID3D12RootSignature> root_;
    Ref<ID3D12PipelineState> pso_[2][2];   // [floor][depth test]
    Ref<ID3D12Resource> vertex_buffer_, index_buffer_, instance_buffer_, upload_, readback_, target_, depth_;
    Ref<ID3D12DescriptorHeap> srv_heap_, rtv_heap_, dsv_heap_;
    Ref<ID3D12CommandQueue> queue_;
    Ref<ID3D12CommandAllocator> allocator_;
    Ref<ID3D12GraphicsCommandList> list_;
    Ref<ID3D12Fence> fence_;
    Geometry geometry_;
    std::uint64_t upload_vb_ = 0, upload_ib_ = 0, upload_inst_ = 0, upload_cbv_ = 0;
    std::uint64_t srv_cpu_ = 0, srv_gpu_ = 0;
    UINT srv_step_ = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_{}, dsv_{};
    UINT64 fence_value_ = 0;
};

}  // namespace lanterns
