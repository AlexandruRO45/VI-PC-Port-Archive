// ExecuteIndirect(DISPATCH_MESH) without VK_EXT_mesh_shader, for --mesh-compute PSOs.
// The direct path (DispatchMesh) knows the group counts on the CPU: it dispatches the mesh shader as compute, one
// output slot per workgroup, then draws groups * max_primitives * 3 vertices with mesh_compute_vs.h. Here the counts
// are GPU data, so a prepass (mesh_indirect_prepare.comp) turns each record into a VkDispatchIndirectCommand and a
// VkDrawIndirectCommand in a private buffer, and the replay issues one indirect dispatch and one indirect draw per
// record, each record writing its own region of output slots. A region holds capacity_groups slots, fixed when the
// call is prepared; a record with more groups is dropped (zero groups, zero vertices) and the batch logs it when
// its submission retires, instead of writing past the region.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include "mesh_indirect_prepare_spv.h"

namespace leonida {

// The output slots of one ExecuteIndirect call: records * capacity_groups * slot_bytes bytes at most. A budget, not
// a measurement: 64 MiB keeps a call with a few hundred records at a useful capacity without holding a large share
// of a small GPU's memory.
inline constexpr VkDeviceSize mesh_indirect_budget = VkDeviceSize(64) << 20;
// D3D12 caps a DispatchMesh grid at 2^22 thread groups in all.
inline constexpr std::uint32_t mesh_indirect_max_groups = 1u << 22;

// Groups each record's region holds, or 0 when not even one slot per record fits the budget.
inline std::uint32_t mesh_indirect_capacity(std::uint32_t records, std::uint32_t slot_bytes,
                                            VkDeviceSize budget = mesh_indirect_budget) {
    if (!records || !slot_bytes) return 0;
    const VkDeviceSize per_record = budget / records / slot_bytes;
    return std::uint32_t(std::min<VkDeviceSize>(per_record, mesh_indirect_max_groups));
}

struct MeshIndirect {
    VkBuffer commands = VK_NULL_HANDLE;  // records VkDispatchIndirectCommands, then records VkDrawIndirectCommands
    VkDeviceSize draw_offset = 0;        // where the draw commands start in `commands`
    VkDeviceAddress arguments = 0, count = 0, dispatch_out = 0, draw_out = 0, status = 0;
    std::uint32_t stride = 0, records = 0, capacity_groups = 0, max_primitives = 0;
    bool has_count = false;
    VkDeviceAddress output = 0;     // record k's slots start at output + k * region_bytes
    VkDeviceSize region_bytes = 0;  // capacity_groups * slot_bytes
};

class MeshIndirectPreparer {
public:
    MeshIndirectPreparer(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceMemoryProperties& memory)
        : get_(get), device_(device), memory_(memory) {}
    ~MeshIndirectPreparer() {
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
    }
    MeshIndirectPreparer(const MeshIndirectPreparer&) = delete;
    MeshIndirectPreparer& operator=(const MeshIndirectPreparer&) = delete;

    // Retained by the command-buffer slot until its submission's timeline ticket completes; then it reads the status
    // words its prepasses wrote and logs dropped records once per batch.
    struct Batch {
        MeshIndirectPreparer& owner;
        struct Allocation {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            const std::uint32_t* status = nullptr;  // mapped, for the status allocation only
        };
        std::vector<Allocation> allocations;
        explicit Batch(MeshIndirectPreparer& value) : owner(value) {}
        ~Batch() {
            bool dropped = false;
            for (auto& a : allocations) {
                if (a.status && (*a.status & 1u)) dropped = true;
                if (a.status) owner.fn<PFN_vkUnmapMemory>("vkUnmapMemory")(owner.device_, a.memory);
                if (a.buffer) owner.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer")(owner.device_, a.buffer, nullptr);
                if (a.memory) owner.fn<PFN_vkFreeMemory>("vkFreeMemory")(owner.device_, a.memory, nullptr);
            }
            if (dropped)
                std::fprintf(stderr, "[rt] d3d12: ExecuteIndirect(DISPATCH_MESH) through --mesh-compute dropped records "
                                     "with more thread groups than their output region holds\n");
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

        // count == VK_NULL_HANDLE: no count buffer. `output` and `region_bytes` are the caller's (the replay's mesh
        // output scratch); this allocates the command and status buffers.
        bool prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count, VkDeviceSize count_offset,
                     std::uint32_t stride, std::uint32_t records, std::uint32_t capacity_groups,
                     std::uint32_t max_primitives, VkDeviceAddress output, VkDeviceSize region_bytes,
                     MeshIndirect& out) {
            out = {};
            if (!arguments || !stride || stride % 4 || !records || !capacity_groups || !max_primitives || !output ||
                !owner.make_pipeline())
                return false;
            constexpr VkDeviceSize dispatch_bytes = 3 * sizeof(std::uint32_t), draw_bytes = 4 * sizeof(std::uint32_t);
            if (VkDeviceSize(records) > std::numeric_limits<VkDeviceSize>::max() / (dispatch_bytes + draw_bytes))
                return false;
            // The draw commands start 16-byte aligned after the dispatch commands.
            const VkDeviceSize draw_offset = (VkDeviceSize(records) * dispatch_bytes + 15) & ~VkDeviceSize(15);
            const VkDeviceSize bytes = draw_offset + VkDeviceSize(records) * draw_bytes;
            Allocation* commands = allocate(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
            if (!commands) return false;
            const VkBuffer command_buffer = commands->buffer;
            Allocation* status = allocate(sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          true);
            if (!status) return false;
            const VkBuffer status_buffer = status->buffer;
            const auto address = owner.fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress");
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = arguments;
            const VkDeviceAddress args_va = address(owner.device_, &info);
            VkDeviceAddress count_va = 0;
            if (count) {
                info.buffer = count;
                count_va = address(owner.device_, &info);
                if (!count_va || count_offset > UINT64_MAX - count_va) return false;
            }
            info.buffer = command_buffer;
            const VkDeviceAddress commands_va = address(owner.device_, &info);
            info.buffer = status_buffer;
            const VkDeviceAddress status_va = address(owner.device_, &info);
            if (!args_va || !commands_va || !status_va || argument_offset > UINT64_MAX - args_va) return false;
            out.commands = command_buffer;
            out.draw_offset = draw_offset;
            out.arguments = args_va + argument_offset;
            out.count = count ? count_va + count_offset : 0;
            out.dispatch_out = commands_va;
            out.draw_out = commands_va + draw_offset;
            out.status = status_va;
            out.stride = stride;
            out.records = records;
            out.capacity_groups = capacity_groups;
            out.max_primitives = max_primitives;
            out.has_count = count != VK_NULL_HANDLE;
            out.output = output;
            out.region_bytes = region_bytes;
            return true;
        }

    private:
        // A buffer with its own memory, pushed to `allocations` before anything can fail so the batch frees it.
        Allocation* allocate(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags wanted, bool map) {
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = bytes;
            info.usage = usage;
            Allocation a{};
            if (owner.fn<PFN_vkCreateBuffer>("vkCreateBuffer")(owner.device_, &info, nullptr, &a.buffer) != VK_SUCCESS)
                return nullptr;
            allocations.push_back(a);
            Allocation& slot = allocations.back();
            VkMemoryRequirements requirements{};
            owner.fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(owner.device_, slot.buffer,
                                                                                         &requirements);
            std::uint32_t memory_type = UINT32_MAX;
            for (std::uint32_t i = 0; i < owner.memory_.memoryTypeCount; ++i) {
                if (!(requirements.memoryTypeBits & (1u << i))) continue;
                const VkMemoryPropertyFlags flags = owner.memory_.memoryTypes[i].propertyFlags;
                if ((flags & wanted) == wanted) {
                    memory_type = i;
                    break;
                }
                // Device-local is a preference for the commands; host visibility is a requirement for the status.
                if (!map && memory_type == UINT32_MAX) memory_type = i;
            }
            if (memory_type == UINT32_MAX) return nullptr;
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memory_type;
            if (owner.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(owner.device_, &allocation, nullptr, &slot.memory) !=
                    VK_SUCCESS ||
                owner.fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(owner.device_, slot.buffer, slot.memory, 0) !=
                    VK_SUCCESS)
                return nullptr;
            if (map) {
                void* mapped = nullptr;
                if (owner.fn<PFN_vkMapMemory>("vkMapMemory")(owner.device_, slot.memory, 0, VK_WHOLE_SIZE, 0,
                                                             &mapped) != VK_SUCCESS)
                    return nullptr;
                *static_cast<std::uint32_t*>(mapped) = 0;  // host-coherent: visible to the prepass's atomicOr
                slot.status = static_cast<const std::uint32_t*>(mapped);
            }
            return &slot;
        }
    };

    std::unique_ptr<Batch> batch() { return std::make_unique<Batch>(*this); }

    // The prepass, outside any rendering scope: reads the records (and the count) after earlier writes, writes the
    // indirect commands, and makes them visible to the indirect dispatches and draws that follow.
    void emit(VkCommandBuffer cb, const MeshIndirect& job) const {
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr,
                0, nullptr);
        fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const Push push{job.arguments, job.count, job.dispatch_out, job.draw_out, job.status,
                        job.stride / 4, job.records, job.capacity_groups, job.max_primitives,
                        job.has_count ? 1u : 0u};
        fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                                         &push);
        fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, (job.records + 63u) / 64u, 1, 1);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &after, 0,
                nullptr, 0, nullptr);
    }

private:
    struct Push {
        VkDeviceAddress arguments, count, dispatch_out, draw_out, status;
        std::uint32_t stride_words, records, capacity_groups, max_primitives, has_count;
    };
    static_assert(sizeof(Push) == 60 || sizeof(Push) == 64);

    template<typename T> T fn(const char* name) const { return reinterpret_cast<T>(get_(device_, name)); }
    bool make_pipeline() {
        if (pipeline_) return true;
        for (const char* name : {"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements",
             "vkAllocateMemory", "vkFreeMemory", "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
             "vkGetBufferDeviceAddress", "vkCreateShaderModule", "vkDestroyShaderModule", "vkCreatePipelineLayout",
             "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline", "vkCmdPipelineBarrier",
             "vkCmdBindPipeline", "vkCmdPushConstants", "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(mesh_indirect_prepare_spirv);
        shader.pCode = mesh_indirect_prepare_spirv;
        bool ok = fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &shader, nullptr, &module) == VK_SUCCESS;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, std::uint32_t(sizeof(Push))};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) ==
                       VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device_, VK_NULL_HANDLE, 1, &info,
                                                                                 nullptr, &pipeline) == VK_SUCCESS;
        if (module) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, module, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            return false;
        }
        layout_ = layout;
        pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}  // namespace leonida
