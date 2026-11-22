// Synthetic strided byte-copy and repeating-element fill compute helpers executed inside conditional rendering.
// Requires enabled storageBuffer8BitAccess, not shaderInt8.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "predicated_fill_spv.h"
#include "predicated_strided_copy_spv.h"

namespace leonida {

struct PredicatedStridedDispatch {
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::uint32_t src_offset = 0;
    std::uint32_t dst_offset = 0;
    std::uint32_t row_bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t slices = 0;
    std::uint32_t src_row_pitch = 0;
    std::uint32_t src_slice_pitch = 0;
    std::uint32_t dst_row_pitch = 0;
    std::uint32_t dst_slice_pitch = 0;
};

struct PredicatedFillDispatch {
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::uint32_t dst_offset = 0;
    std::uint32_t total_bytes = 0;
    std::uint32_t elem_size = 0;
    std::uint32_t phase = 0;
    std::array<std::uint32_t, 4> pattern{};
};

class PredicatedOps {
public:
    PredicatedOps(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceLimits& limits)
        : get_(get), device_(device), limits_(limits) {}

    ~PredicatedOps() {
        destroy_pipeline(strided_);
        destroy_pipeline(fill_);
    }

    PredicatedOps(const PredicatedOps&) = delete;
    PredicatedOps& operator=(const PredicatedOps&) = delete;

    struct Batch {
        PredicatedOps& owner;
        std::vector<VkDescriptorPool> pools;

        explicit Batch(PredicatedOps& value) : owner(value) {}

        ~Batch() {
            for (auto pool : pools)
                owner.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(owner.device_, pool, nullptr);
        }

        // Prepares byte-granular strided 3D copy windows (slices -> rows -> contiguous byte chunks) within
        // minStorageBufferOffsetAlignment, maxStorageBufferRange, maxComputeWorkGroupCount[0] and uint32 bounds.
        bool prepare_strided(VkBuffer source, VkBuffer destination, VkDeviceSize src_base, VkDeviceSize dst_base,
                             VkDeviceSize row_bytes, std::uint32_t rows, std::uint32_t slices,
                             VkDeviceSize src_row_pitch, VkDeviceSize src_slice_pitch,
                             VkDeviceSize dst_row_pitch, VkDeviceSize dst_slice_pitch,
                             std::vector<PredicatedStridedDispatch>& out) {
            out.clear();
            if (!row_bytes || !rows || !slices) return true;
            if (!source || !destination) return false;
            if (rows == 1) src_row_pitch = dst_row_pitch = 0;
            if (slices == 1) src_slice_pitch = dst_slice_pitch = 0;
            if (rows > 1 && dst_row_pitch < row_bytes) return false;

            const auto span_of = [](VkDeviceSize rb, std::uint32_t r, std::uint32_t s,
                                    VkDeviceSize rp, VkDeviceSize sp, VkDeviceSize& out_span) -> bool {
                VkDeviceSize acc = 0;
                if (s > 1) {
                    if (sp > UINT64_MAX / (s - 1)) return false;
                    acc = VkDeviceSize(s - 1) * sp;
                }
                if (r > 1) {
                    if (rp > (UINT64_MAX - acc) / (r - 1)) return false;
                    acc += VkDeviceSize(r - 1) * rp;
                }
                if (rb > UINT64_MAX - acc) return false;
                out_span = acc + rb;
                return true;
            };
            VkDeviceSize src_total_span = 0, dst_total_span = 0, dst_slice_min = 0;
            if (!span_of(row_bytes, rows, 1, dst_row_pitch, 0, dst_slice_min) ||
                (slices > 1 && dst_slice_pitch < dst_slice_min) ||
                !span_of(row_bytes, rows, slices, src_row_pitch, src_slice_pitch, src_total_span) ||
                !span_of(row_bytes, rows, slices, dst_row_pitch, dst_slice_pitch, dst_total_span) ||
                src_total_span > UINT64_MAX - src_base || dst_total_span > UINT64_MAX - dst_base)
                return false;
            if (source == destination && src_base < dst_base + dst_total_span && dst_base < src_base + src_total_span)
                return false;
            if (!owner.make_strided_pipeline()) return false;

            const auto alignment = std::max<VkDeviceSize>(1, owner.limits_.minStorageBufferOffsetAlignment);
            const VkDeviceSize cap = std::min<VkDeviceSize>(owner.limits_.maxStorageBufferRange, UINT32_MAX);
            const VkDeviceSize threads =
                std::min<VkDeviceSize>(VkDeviceSize(owner.limits_.maxComputeWorkGroupCount[0]) * 64, UINT32_MAX);
            if (!cap || !threads) return false;

            struct Window {
                VkDescriptorBufferInfo source;
                VkDescriptorBufferInfo destination;
                std::uint32_t src_offset, dst_offset, row_bytes, rows, slices;
                std::uint32_t src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch;
            };
            std::vector<Window> windows;

            const auto push_window = [&](VkDeviceSize sb, VkDeviceSize db, VkDeviceSize rb,
                                         std::uint32_t r, std::uint32_t s,
                                         VkDeviceSize srp, VkDeviceSize ssp,
                                         VkDeviceSize drp, VkDeviceSize dsp) -> bool {
                const VkDeviceSize sp_off = sb % alignment;
                const VkDeviceSize dp_off = db % alignment;
                if (sp_off >= cap || dp_off >= cap) return false;
                VkDeviceSize src_span = 0, dst_span = 0;
                if (!span_of(rb, r, s, srp, ssp, src_span) || !span_of(rb, r, s, drp, dsp, dst_span)) return false;
                if (src_span > cap - sp_off || dst_span > cap - dp_off) return false;
                if (rb > threads || r > threads / rb || s > threads / (rb * r)) return false;
                if (windows.size() >= UINT32_MAX / 2) return false;
                windows.push_back({
                    {source, sb - sp_off, sp_off + src_span},
                    {destination, db - dp_off, dp_off + dst_span},
                    static_cast<std::uint32_t>(sp_off),
                    static_cast<std::uint32_t>(dp_off),
                    static_cast<std::uint32_t>(rb),
                    r,
                    s,
                    r > 1 ? static_cast<std::uint32_t>(srp) : 0u,
                    s > 1 ? static_cast<std::uint32_t>(ssp) : 0u,
                    r > 1 ? static_cast<std::uint32_t>(drp) : 0u,
                    s > 1 ? static_cast<std::uint32_t>(dsp) : 0u,
                });
                return true;
            };

            VkDeviceSize one_slice_src = 0, one_slice_dst = 0;
            const bool one_slice_span_ok =
                span_of(row_bytes, rows, 1, src_row_pitch, 0, one_slice_src) &&
                span_of(row_bytes, rows, 1, dst_row_pitch, 0, one_slice_dst);

            for (std::uint32_t z = 0; z < slices;) {
                const VkDeviceSize sb_z = src_base + VkDeviceSize(z) * src_slice_pitch;
                const VkDeviceSize db_z = dst_base + VkDeviceSize(z) * dst_slice_pitch;
                const VkDeviceSize sp_z = sb_z % alignment;
                const VkDeviceSize dp_z = db_z % alignment;

                // Level 1: batch whole slices when 1+ whole slices fit in a single descriptor window and dispatch.
                std::uint32_t max_ns = 0;
                if (one_slice_span_ok && row_bytes <= threads && rows <= threads / row_bytes &&
                    sp_z < cap && dp_z < cap && one_slice_src <= cap - sp_z && one_slice_dst <= cap - dp_z) {
                    const VkDeviceSize slice_bytes = row_bytes * rows;
                    max_ns = std::min<std::uint32_t>(
                        slices - z, static_cast<std::uint32_t>(std::min<VkDeviceSize>(threads / slice_bytes, UINT32_MAX)));
                    if (max_ns > 1 && src_slice_pitch > 0) {
                        const VkDeviceSize fit = 1 + (cap - sp_z - one_slice_src) / src_slice_pitch;
                        max_ns = std::min<std::uint32_t>(max_ns, static_cast<std::uint32_t>(std::min<VkDeviceSize>(fit, UINT32_MAX)));
                    }
                    if (max_ns > 1 && dst_slice_pitch > 0) {
                        const VkDeviceSize fit = 1 + (cap - dp_z - one_slice_dst) / dst_slice_pitch;
                        max_ns = std::min<std::uint32_t>(max_ns, static_cast<std::uint32_t>(std::min<VkDeviceSize>(fit, UINT32_MAX)));
                    }
                }
                if (max_ns >= 1) {
                    if (!push_window(sb_z, db_z, row_bytes, rows, max_ns,
                                     src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch))
                        return false;
                    z += max_ns;
                    continue;
                }

                // Level 2: split slice across rows; Level 3: split single row across contiguous byte windows.
                for (std::uint32_t y = 0; y < rows;) {
                    const VkDeviceSize sb_y = sb_z + VkDeviceSize(y) * src_row_pitch;
                    const VkDeviceSize db_y = db_z + VkDeviceSize(y) * dst_row_pitch;
                    const VkDeviceSize sp_y = sb_y % alignment;
                    const VkDeviceSize dp_y = db_y % alignment;

                    std::uint32_t max_nr = 0;
                    if (row_bytes <= threads && sp_y < cap && dp_y < cap &&
                        row_bytes <= cap - sp_y && row_bytes <= cap - dp_y) {
                        max_nr = std::min<std::uint32_t>(
                            rows - y, static_cast<std::uint32_t>(std::min<VkDeviceSize>(threads / row_bytes, UINT32_MAX)));
                        if (max_nr > 1 && src_row_pitch > 0) {
                            const VkDeviceSize fit = 1 + (cap - sp_y - row_bytes) / src_row_pitch;
                            max_nr = std::min<std::uint32_t>(max_nr, static_cast<std::uint32_t>(std::min<VkDeviceSize>(fit, UINT32_MAX)));
                        }
                        if (max_nr > 1 && dst_row_pitch > 0) {
                            const VkDeviceSize fit = 1 + (cap - dp_y - row_bytes) / dst_row_pitch;
                            max_nr = std::min<std::uint32_t>(max_nr, static_cast<std::uint32_t>(std::min<VkDeviceSize>(fit, UINT32_MAX)));
                        }
                    }
                    if (max_nr >= 1) {
                        if (!push_window(sb_y, db_y, row_bytes, max_nr, 1, src_row_pitch, 0, dst_row_pitch, 0))
                            return false;
                        y += max_nr;
                        continue;
                    }

                    for (VkDeviceSize x = 0; x < row_bytes;) {
                        const VkDeviceSize sb_x = sb_y + x;
                        const VkDeviceSize db_x = db_y + x;
                        const VkDeviceSize sp_x = sb_x % alignment;
                        const VkDeviceSize dp_x = db_x % alignment;
                        if (sp_x >= cap || dp_x >= cap) return false;
                        const VkDeviceSize n = std::min({row_bytes - x, cap - sp_x, cap - dp_x, threads});
                        if (!n) return false;
                        if (!push_window(sb_x, db_x, n, 1, 1, 0, 0, 0, 0)) return false;
                        x += n;
                    }
                    ++y;
                }
                ++z;
            }

            const auto count = static_cast<std::uint32_t>(windows.size());
            const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count * 2};
            VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool_info.maxSets = count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (owner.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(
                    owner.device_, &pool_info, nullptr, &pool) != VK_SUCCESS)
                return false;
            pools.push_back(pool);

            const std::vector<VkDescriptorSetLayout> layouts(count, owner.strided_.set_layout);
            std::vector<VkDescriptorSet> sets(count);
            VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocate.descriptorPool = pool;
            allocate.descriptorSetCount = count;
            allocate.pSetLayouts = layouts.data();
            if (owner.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(
                    owner.device_, &allocate, sets.data()) != VK_SUCCESS)
                return false;

            out.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                VkWriteDescriptorSet writes[2]{};
                const VkDescriptorBufferInfo* infos[] = {&windows[i].source, &windows[i].destination};
                for (std::uint32_t binding = 0; binding < 2; ++binding) {
                    auto& w = writes[binding];
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = sets[i];
                    w.dstBinding = binding;
                    w.descriptorCount = 1;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w.pBufferInfo = infos[binding];
                }
                owner.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(owner.device_, 2, writes, 0, nullptr);
                out.push_back({sets[i], windows[i].src_offset, windows[i].dst_offset, windows[i].row_bytes,
                               windows[i].rows, windows[i].slices, windows[i].src_row_pitch, windows[i].src_slice_pitch,
                               windows[i].dst_row_pitch, windows[i].dst_slice_pitch});
            }
            return true;
        }

        // Appends repeating-element byte-granular fill dispatches onto `out` (supports multiple sub-rects per clear).
        bool append_fill(VkBuffer destination, VkDeviceSize dst_base, VkDeviceSize bytes,
                         const std::uint8_t* pattern, std::uint32_t element_size,
                         std::vector<PredicatedFillDispatch>& out) {
            if (!bytes) return true;
            if (!destination || !pattern || element_size == 0 || element_size > 16 || bytes > UINT64_MAX - dst_base)
                return false;
            if (!owner.make_fill_pipeline()) return false;

            std::array<std::uint32_t, 4> packed{};
            for (std::uint32_t k = 0; k < element_size; ++k)
                packed[k / 4] |= std::uint32_t(pattern[k]) << ((k & 3u) * 8u);

            const auto alignment = std::max<VkDeviceSize>(1, owner.limits_.minStorageBufferOffsetAlignment);
            const VkDeviceSize cap = std::min<VkDeviceSize>(owner.limits_.maxStorageBufferRange, UINT32_MAX);
            // Leave headroom of 16 so (phase + i) inside the shader never wraps uint32.
            const VkDeviceSize threads = std::min<VkDeviceSize>(
                VkDeviceSize(owner.limits_.maxComputeWorkGroupCount[0]) * 64, UINT32_MAX - 16u);
            if (!cap || !threads) return false;

            struct Window {
                VkDescriptorBufferInfo destination;
                std::uint32_t dst_offset, bytes, phase;
            };
            std::vector<Window> windows;
            for (VkDeviceSize done = 0; done < bytes;) {
                const VkDeviceSize d = dst_base + done;
                const VkDeviceSize dp = d % alignment;
                if (dp >= cap) return false;
                const VkDeviceSize n = std::min({bytes - done, cap - dp, threads});
                if (!n) return false;
                if (windows.size() >= UINT32_MAX / 2) return false;
                windows.push_back({{destination, d - dp, dp + n},
                                   static_cast<std::uint32_t>(dp),
                                   static_cast<std::uint32_t>(n),
                                   static_cast<std::uint32_t>(done % element_size)});
                done += n;
            }

            const auto count = static_cast<std::uint32_t>(windows.size());
            const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count};
            VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool_info.maxSets = count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (owner.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(
                    owner.device_, &pool_info, nullptr, &pool) != VK_SUCCESS)
                return false;
            pools.push_back(pool);

            const std::vector<VkDescriptorSetLayout> layouts(count, owner.fill_.set_layout);
            std::vector<VkDescriptorSet> sets(count);
            VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocate.descriptorPool = pool;
            allocate.descriptorSetCount = count;
            allocate.pSetLayouts = layouts.data();
            if (owner.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(
                    owner.device_, &allocate, sets.data()) != VK_SUCCESS)
                return false;

            out.reserve(out.size() + count);
            for (std::uint32_t i = 0; i < count; ++i) {
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = sets[i];
                write.dstBinding = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &windows[i].destination;
                owner.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(owner.device_, 1, &write, 0, nullptr);
                out.push_back({sets[i], windows[i].dst_offset, windows[i].bytes, element_size, windows[i].phase, packed});
            }
            return true;
        }
    };

    std::unique_ptr<Batch> batch() { return std::make_unique<Batch>(*this); }

    void emit_strided(VkCommandBuffer cb, const std::vector<PredicatedStridedDispatch>& jobs) const {
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        for (const auto& job : jobs) {
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    1, &before, 0, nullptr, 0, nullptr);
            fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, strided_.pipeline);
            fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, strided_.layout, 0, 1, &job.set, 0, nullptr);
            const std::uint32_t push[] = {
                job.src_offset, job.dst_offset, job.row_bytes, job.rows, job.slices,
                job.src_row_pitch, job.src_slice_pitch, job.dst_row_pitch, job.dst_slice_pitch,
            };
            fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(
                cb, strided_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            const std::uint32_t total = job.row_bytes * job.rows * job.slices;
            fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, (total - 1u) / 64u + 1u, 1, 1);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                    1, &after, 0, nullptr, 0, nullptr);
        }
    }

    void emit_fill(VkCommandBuffer cb, const std::vector<PredicatedFillDispatch>& jobs) const {
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        for (const auto& job : jobs) {
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    1, &before, 0, nullptr, 0, nullptr);
            fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, fill_.pipeline);
            fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, fill_.layout, 0, 1, &job.set, 0, nullptr);
            const std::uint32_t push[] = {
                job.dst_offset, job.total_bytes, job.elem_size, job.phase,
                job.pattern[0], job.pattern[1], job.pattern[2], job.pattern[3],
            };
            fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(
                cb, fill_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, (job.total_bytes - 1u) / 64u + 1u, 1, 1);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                    1, &after, 0, nullptr, 0, nullptr);
        }
    }

private:
    struct Pipeline {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    template<class T>
    T fn(const char* name) const {
        return reinterpret_cast<T>(get_(device_, name));
    }

    void destroy_pipeline(Pipeline& p) {
        if (p.pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, p.pipeline, nullptr);
        if (p.layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, p.layout, nullptr);
        if (p.set_layout) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, p.set_layout, nullptr);
        p = {};
    }

    bool check_entry_points() const {
        for (const char* name : {"vkCreateDescriptorPool", "vkDestroyDescriptorPool",
                                 "vkAllocateDescriptorSets", "vkUpdateDescriptorSets", "vkCreateShaderModule",
                                 "vkDestroyShaderModule", "vkCreateDescriptorSetLayout",
                                 "vkDestroyDescriptorSetLayout", "vkCreatePipelineLayout", "vkDestroyPipelineLayout",
                                 "vkCreateComputePipelines", "vkDestroyPipeline", "vkCmdPipelineBarrier",
                                 "vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdPushConstants",
                                 "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        return true;
    }

    bool make_pipeline(Pipeline& target, std::uint32_t binding_count, std::uint32_t push_bytes,
                       const std::uint32_t* spirv, std::size_t spirv_bytes) {
        if (target.pipeline) return true;
        if (!check_entry_points()) return false;
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (std::uint32_t i = 0; i < binding_count; ++i)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = binding_count;
        set_info.pBindings = bindings;
        bool ok = fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(
                      device_, &set_info, nullptr, &set) == VK_SUCCESS;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
                       device_, &layout_info, nullptr, &layout) == VK_SUCCESS;
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = spirv_bytes;
        module.pCode = spirv;
        ok = ok && fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(
                       device_, &module, nullptr, &shader) == VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = shader;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                       device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
        if (shader) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, shader, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            if (set) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, set, nullptr);
            return false;
        }
        target = {set, layout, pipeline};
        return true;
    }

    bool make_strided_pipeline() {
        return make_pipeline(strided_, 2, 36, predicated_strided_copy_spirv, sizeof(predicated_strided_copy_spirv));
    }

    bool make_fill_pipeline() {
        return make_pipeline(fill_, 1, 32, predicated_fill_spirv, sizeof(predicated_fill_spirv));
    }

    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceLimits limits_;
    Pipeline strided_;
    Pipeline fill_;
};

}  // namespace leonida
