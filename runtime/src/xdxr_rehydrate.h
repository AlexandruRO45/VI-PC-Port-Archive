// Records the rehydration of a dehydrated XDXR BVH4 BLAS (xdxr_rehydrate.comp; docs/bvh.md §4a), the GPU side of
// XboxBvhBuilder.dll's RehydrateRaytracingAccelerationStructure*X. The model it follows byte for byte is
// tools/bvh_rehydrate.py.
//
// The dehydrated structure, its leaf count and its level table are in GPU memory, possibly written by work recorded
// before this, so the CPU never reads them: a one-thread pass writes a plan into `plan` (indirect dispatch arguments,
// the header fields, the levels), and every later pass is a vkCmdDispatchIndirect from it. The plan also holds the
// leaf-frame rotation table and the geometry table, which the CPU writes with vkCmdUpdateBuffer before the passes.
// Its size is plan_bytes(geometries); the buffer needs TRANSFER_DST and INDIRECT_BUFFER usage.
//
// The result has exactly the dehydrated size, 64 * (totalSizeIn64BNodes + 1) bytes from the header: every node and
// leaf is written where its packed bytes are. The source and the destination must be different buffers.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "xdxr_rehydrate_spv.h"

namespace leonida {

// One geometry index of the structure: float3 vertices (R32G32B32_FLOAT, `stride` bytes apart), optional u16/u32
// triangle indices and an optional 3x4 row-major transform, all as device addresses.
struct XdxrRehydrateGeometry {
    std::uint64_t vertices = 0;
    std::uint32_t vertex_stride = 12;
    std::uint64_t indices = 0;        // 0 = a triangle list
    std::uint32_t index_type = 0;     // 0 none, 1 R16_UINT, 2 R32_UINT
    std::uint64_t transform = 0;      // 0 = none
};

class XdxrRehydrator {
public:
    // The plan's layout (xdxr_rehydrate.comp PLAN_*).
    static constexpr std::uint32_t max_levels = 32;
    static constexpr std::uint32_t plan_cmds = 16, plan_rot = 1408, plan_geom = 3712;
    static constexpr std::uint32_t status_offset = 0;
    static constexpr std::uint64_t plan_bytes(std::size_t geometries) { return plan_geom + 32ull * geometries; }

    XdxrRehydrator(PFN_vkGetDeviceProcAddr get, VkDevice device) : device_(device) {
        loaded_ = load(get) && make_pipeline();
        rotations_ = rotation_table();
    }
    ~XdxrRehydrator() {
        if (pipeline_) vk_.vkDestroyPipeline(device_, pipeline_, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    XdxrRehydrator(const XdxrRehydrator&) = delete;
    XdxrRehydrator& operator=(const XdxrRehydrator&) = delete;

    bool loaded() const { return loaded_; }

    // Records one rehydration. `src` and `dst` are device addresses; `plan` is the device address of
    // plan_bytes(geometries.size()) bytes at `plan_offset` in `plan_buffer`, 4-byte aligned. The last pass leaves the
    // dependency on the result to the caller's barrier, as the builder's passes do.
    bool record(VkCommandBuffer cb, std::uint64_t src, std::uint64_t dst, std::uint64_t plan, VkBuffer plan_buffer,
                VkDeviceSize plan_offset, const std::vector<XdxrRehydrateGeometry>& geometries) {
        if (!loaded_ || !src || !dst || !plan || !plan_buffer || src == dst || (plan_offset & 3) || (plan & 3))
            return false;
        // The rotation table and the geometry table, before any pass reads them.
        vk_.vkCmdUpdateBuffer(cb, plan_buffer, plan_offset + plan_rot, rotations_.size() * sizeof(float),
                              rotations_.data());
        std::vector<std::uint32_t> table;
        table.reserve(8 * geometries.size());
        for (const XdxrRehydrateGeometry& g : geometries) {
            table.push_back(std::uint32_t(g.vertices));
            table.push_back(std::uint32_t(g.vertices >> 32));
            table.push_back(g.vertex_stride);
            table.push_back(std::uint32_t(g.indices));
            table.push_back(std::uint32_t(g.indices >> 32));
            table.push_back(g.indices ? g.index_type : 0u);
            table.push_back(std::uint32_t(g.transform));
            table.push_back(std::uint32_t(g.transform >> 32));
        }
        for (std::size_t at = 0; at < table.size(); at += 16384) {   // vkCmdUpdateBuffer takes at most 64 KiB
            const std::size_t words = std::min<std::size_t>(16384, table.size() - at);
            vk_.vkCmdUpdateBuffer(cb, plan_buffer, plan_offset + plan_geom + 4 * at, 4 * words, table.data() + at);
        }
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        Push p{};
        p.src = split(src);
        p.dst = split(dst);
        p.plan = split(plan);
        p.geometries = std::uint32_t(geometries.size());
        p.mode = mode_plan;
        vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &p);
        vk_.vkCmdDispatch(cb, 1, 1, 1);
        // The plan is read as dispatch arguments and by the shaders.
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        auto indirect = [&](std::uint32_t mode, std::uint32_t cmd, std::uint32_t level, bool last) {
            Push q = p;
            q.mode = mode;
            q.level = level;
            vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &q);
            vk_.vkCmdDispatchIndirect(cb, plan_buffer, plan_offset + plan_cmds + 12ull * cmd);
            if (!last)
                barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        };
        indirect(mode_clear, 0, 0, false);
        indirect(mode_header, 1, 0, false);
        indirect(mode_leaves, 2, 0, false);
        // Bottom-up: a level's boxes read the nodes of the levels below it. A level the structure doesn't have is a
        // zero-sized dispatch.
        for (std::uint32_t level = 0; level < max_levels; ++level) {
            indirect(mode_box32, 3 + 2 * level, level, false);
            indirect(mode_box16, 4 + 2 * level, level, level + 1 == max_levels);
        }
        return true;
    }

    // Rz * Ry * Rx for each of the 64 rotation indices, the way the model computes them: each angle a multiple of
    // pi/8, every product and every sum rounded to float32 separately (no fused multiply-add).
    static std::vector<float> rotation_table() {
        const double pi = 3.14159265358979323846;
        const double c[5] = {1.0, std::cos(pi / 8), std::cos(pi / 4), std::cos(3 * pi / 8), 0.0};
        std::vector<float> out;
        out.reserve(64 * 9);
        for (int index = 0; index < 64; ++index) {
            const int ix = index & 3, iy = (index >> 2) & 3, iz = (index >> 4) & 3;
            const float cx = float(c[ix]), sx = float(c[4 - ix]), cy = float(c[iy]), sy = float(c[4 - iy]),
                        cz = float(c[iz]), sz = float(c[4 - iz]);
            const float rx[3][3] = {{1, 0, 0}, {0, cx, sx}, {0, -sx, cx}};
            const float ry[3][3] = {{cy, 0, -sy}, {0, 1, 0}, {sy, 0, cy}};
            const float rz[3][3] = {{cz, sz, 0}, {-sz, cz, 0}, {0, 0, 1}};
            float ryx[3][3], r[3][3];
            mul(ry, rx, ryx);
            mul(rz, ryx, r);
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) out.push_back(r[row][col]);
        }
        return out;
    }

private:
    enum : std::uint32_t { mode_plan = 0, mode_clear = 1, mode_header = 2, mode_leaves = 3, mode_box32 = 4, mode_box16 = 5 };
    struct Addr { std::uint32_t lo = 0, hi = 0; };
    struct Push {  // xdxr_rehydrate.comp `Push`
        Addr src, dst, plan;
        std::uint32_t mode = 0, level = 0, geometries = 0, pad = 0;
    };
    static_assert(sizeof(Push) == 40, "push constants must match xdxr_rehydrate.comp");

    static Addr split(std::uint64_t a) { return {std::uint32_t(a), std::uint32_t(a >> 32)}; }

    static void mul(const float a[3][3], const float b[3][3], float out[3][3]) {
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col) {
                volatile float acc = 0.0f;   // volatile: one float32 rounding per product and per sum
                for (int m = 0; m < 3; ++m) {
                    volatile float product = a[r][m] * b[m][col];
                    acc = acc + product;
                }
                out[r][col] = acc;
            }
    }

    void barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                 VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vk_.vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    struct Functions {
#define LEONIDA_XDXR_REHYDRATE_FUNCTIONS(X)                                                                    \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)      \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdBindPipeline) X(vkCmdPushConstants)                \
    X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkCmdPipelineBarrier) X(vkCmdUpdateBuffer)
#define LEONIDA_XDXR_REHYDRATE_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_XDXR_REHYDRATE_FUNCTIONS(LEONIDA_XDXR_REHYDRATE_MEMBER)
#undef LEONIDA_XDXR_REHYDRATE_MEMBER
    };

    bool load(PFN_vkGetDeviceProcAddr get) {
        if (!get || !device_) return false;
        bool all = true;
#define LEONIDA_XDXR_REHYDRATE_LOAD(name)                           \
    vk_.name = reinterpret_cast<PFN_##name>(get(device_, #name));   \
    all = all && vk_.name != nullptr;
        LEONIDA_XDXR_REHYDRATE_FUNCTIONS(LEONIDA_XDXR_REHYDRATE_LOAD)
#undef LEONIDA_XDXR_REHYDRATE_LOAD
#undef LEONIDA_XDXR_REHYDRATE_FUNCTIONS
        return all;
    }

    bool make_pipeline() {
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (vk_.vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(xdxr_rehydrate_spirv);
        module_info.pCode = xdxr_rehydrate_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vk_.vkCreateShaderModule(device_, &module_info, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                      module, "main", nullptr};
        info.layout = layout_;
        const bool ok = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_) == VK_SUCCESS;
        vk_.vkDestroyShaderModule(device_, module, nullptr);
        return ok;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    Functions vk_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<float> rotations_;
    bool loaded_ = false;
};

}  // namespace leonida
