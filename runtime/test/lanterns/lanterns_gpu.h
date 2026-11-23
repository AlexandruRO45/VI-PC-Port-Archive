// Lanterns: the GPU side, through the D3D12 API only. The device comes from the D3D12XboxCreateDevice the guest
// imports from d3d12_xs.dll (test/fixtures/tiny_frame.cpp does the same); pipelines come from the shader
// containers fssm_pack.py wrote. See lanterns.cpp for what each stage draws and checks.
#pragma once

#include "../d3d12_test_shaders.h"
#include "lanterns_particles.h"
#include "lanterns_post.h"
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
    std::vector<std::uint8_t> vs, ps_lit, ps_floor, unknown_vs, rebuilt_vs, vs_w64, ps_lit_w64, cs_rise;
    std::vector<std::uint8_t> ps_tex_floor, ps_tex_solid;   // stage 6: they replace ps_floor and ps_lit
    std::vector<std::uint8_t> ps_shadow;                    // stage 8: replaces both
};
// stage 1: graphics only; 3 and 5: plus cs_lanterns_rise; 6: plus the textured pixel shaders; 8: plus the shadow
// pixel shader.
bool load_shaders(const std::filesystem::path& dir, int stage, Shaders& out);

// What a frame's readback holds besides the pixels (stages 3 and 5).
struct Readback {
    std::vector<Particle> state;
    std::vector<Instance> instances;   // what cs_lanterns_rise wrote
};

// The scene's meshes as one vertex array and one 16-bit index array (each face fanned), with where each mesh starts,
// and the uploaded instances of the draws that do not take theirs from the compute shader.
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
    // buffers (and the particles' first state, when the shaders include the compute shader).
    // target_format: R32G32B32A32_FLOAT (the other stages) or R16G16B16A16_FLOAT (stage 5's HDR target, which the post-process
    // reads; finish_frame returns its pixels widened to float).
    bool create(const Scene& scene, const Shaders& shaders,
                DXGI_FORMAT target_format = DXGI_FORMAT_R32G32B32A32_FLOAT);
    // Records and submits one frame on the direct queue. With wait_compute != 0 the queue first waits for the
    // compute fence to reach that value (which may be signalled by a submission made after this one).
    bool submit_frame(const Scene& scene, bool depth_test, std::uint64_t wait_compute = 0);
    // Waits for the last frame and reads back its pixels (and, if asked, the particle state and instance records).
    bool finish_frame(std::vector<Color>& pixels, Readback* readback = nullptr);
    // Submits `steps` dispatches of cs_lanterns_rise on the compute queue and signals the fence.
    bool submit_compute(unsigned steps, std::uint64_t& signalled);
    std::uint64_t next_compute_value() const { return compute_value_ + 1; }
    // Stage 6: the textures' SRVs (the floor's mosaic, the solids' squares) and the sampler, before the first frame.
    bool bind_textures(ID3D12Resource* floor, ID3D12Resource* solid);
    // Stage 8: the light (its clip matrix for the depth pass, the shadow pixel shader's three rows after the camera's
    // matrix) and whether the depth pass draws the casters (false: the map stays cleared to 1, the control).
    bool set_light(const std::array<float, 16>& clip, const std::array<float, 12>& rows);
    void set_shadow_casters(bool casters) { shadow_casters_ = casters; }
    // Waits until both queues are idle (before the objects are released).
    bool idle();
    // For the post-process (stage 5), which records on the same device and queue after the frame. Between frames the
    // target is in COPY_SOURCE.
    ID3D12Device* device() const { return device_.p; }
    ID3D12CommandQueue* queue() const { return queue_.p; }
    ID3D12Resource* target() const { return target_.p; }

private:
    bool create_pipelines(const Shaders& shaders);
    bool create_compute(const Shaders& shaders);
    bool create_shadow();                       // lanterns_shadow_gpu.cpp
    void record_shadow_pass(const Scene& scene);   // lanterns_shadow_gpu.cpp
    void bind_draw(const Scene& scene, std::size_t d);
    bool create_resources(const Scene& scene);
    bool initialize(const Scene& scene);
    bool fill_descriptors(const Scene& scene);

    // Declared first, released last: everything below is a child of the device.
    Ref<ID3D12Device> device_;
    Ref<ID3D12RootSignature> root_, compute_root_;
    Ref<ID3D12PipelineState> pso_[2][2];   // [floor][depth test]
    Ref<ID3D12PipelineState> pso_w64_[2];  // [depth test]: the wave64 pair, for the draws marked wave64
    Ref<ID3D12PipelineState> compute_pso_;
    Ref<ID3D12Resource> shadow_map_;
    std::vector<std::uint8_t> vs_;
    Ref<ID3D12PipelineState> shadow_pso_;
    D3D12_CPU_DESCRIPTOR_HANDLE shadow_dsv_{};
    bool shadow_ = false, shadow_casters_ = true, shadow_written_ = false;
    std::uint64_t upload_light_ = 0;
    Ref<ID3D12Resource> vertex_buffer_, index_buffer_, instance_buffer_, upload_, readback_, target_, depth_;
    Ref<ID3D12Resource> particle_state_, particle_instances_;
    Ref<ID3D12DescriptorHeap> srv_heap_, rtv_heap_, dsv_heap_, sampler_heap_;
    Ref<ID3D12CommandQueue> queue_, compute_queue_;
    Ref<ID3D12CommandAllocator> allocator_, compute_allocator_;
    Ref<ID3D12GraphicsCommandList> list_, compute_list_;
    Ref<ID3D12Fence> fence_, compute_fence_;
    Geometry geometry_;
    DXGI_FORMAT target_format_ = DXGI_FORMAT_R32G32B32A32_FLOAT;
    bool compute_ = false, textured_ = false, textures_bound_ = false;
    std::uint64_t srv_texture_slot_ = 0;   // the two texture SRVs (stage 6), after the compute tables
    std::uint64_t upload_vb_ = 0, upload_ib_ = 0, upload_inst_ = 0, upload_cbv_ = 0, upload_particles_ = 0;
    std::uint64_t srv_cpu_ = 0, srv_gpu_ = 0;
    UINT srv_step_ = 0;
    std::uint64_t srv_compute_slot_ = 0;   // the compute table's first slot (after the graphics V#s)
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_{}, dsv_{};
    UINT64 fence_value_ = 0, compute_value_ = 0;
};

}  // namespace lanterns
