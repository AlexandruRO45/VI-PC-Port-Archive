// Byte-granular conditional CopyBufferRegion. Requires enabled storageBuffer8BitAccess, not shaderInt8.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <initializer_list>
#include <vector>
#include "predicated_copy_spv.h"
namespace leonida {
struct PredicatedCopyDispatch {
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::uint32_t source = 0, destination = 0, bytes = 0;
};
class PredicatedBufferCopier {
public:
    PredicatedBufferCopier(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceLimits& limits)
        : get_(get), device_(device), limits_(limits) {}
    ~PredicatedBufferCopier() {
        if (!get_ || !device_) return;
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
        if (set_layout_) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, set_layout_, nullptr);
    }
    PredicatedBufferCopier(const PredicatedBufferCopier&)=delete;
    PredicatedBufferCopier& operator=(const PredicatedBufferCopier&)=delete;
    struct Batch {
        PredicatedBufferCopier& owner;
        std::vector<VkDescriptorPool> pools;
        explicit Batch(PredicatedBufferCopier& value):owner(value){}
        ~Batch() {
            if (!owner.get_ || !owner.device_) return;
            const auto destroy = owner.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
            if (!destroy) return;
            for (auto pool : pools) destroy(owner.device_, pool, nullptr);
        }
        bool prepare(VkBuffer source,VkBuffer destination,VkDeviceSize src,VkDeviceSize dst,VkDeviceSize bytes,
                     std::vector<PredicatedCopyDispatch>& out) {
            out.clear();
            if(!bytes)return true;
            if(!source||!destination||bytes>UINT64_MAX-src||bytes>UINT64_MAX-dst)return false;
            if(source==destination && src<dst+bytes && dst<src+bytes)return false;
            if(!owner.make_pipeline())return false;
            // Align each descriptor base down, keeping byte prefixes in push constants. Split at the smaller
            // remaining descriptor range or dispatch limit, so no uint32 shader index can wrap or overread.
            struct Window{VkDescriptorBufferInfo source,destination;std::uint32_t src,dst,bytes;};
            std::vector<Window> windows;
            const auto alignment=std::max<VkDeviceSize>(1,owner.limits_.minStorageBufferOffsetAlignment);
            const VkDeviceSize maximum=owner.limits_.maxStorageBufferRange;
            const VkDeviceSize threads=VkDeviceSize(owner.limits_.maxComputeWorkGroupCount[0])*64;
            for(VkDeviceSize done=0;done<bytes;){
                const auto s=src+done,d=dst+done,sp=s%alignment,dp=d%alignment;
                if(sp>=maximum||dp>=maximum||!threads)return false;
                const auto n=std::min({bytes-done,maximum-sp,maximum-dp,threads});
                if(!n||n>UINT32_MAX)return false;
                windows.push_back({{source,s-sp,sp+n},{destination,d-dp,dp+n},
                                   static_cast<std::uint32_t>(sp),static_cast<std::uint32_t>(dp),static_cast<std::uint32_t>(n)});
                if(windows.size()>UINT32_MAX/2)return false;
                done+=n;
            }
            const auto count=static_cast<std::uint32_t>(windows.size());
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count*2};
            VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool_info.maxSets=count;pool_info.poolSizeCount=1;pool_info.pPoolSizes=&size;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (owner.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(
                    owner.device_, &pool_info, nullptr, &pool) != VK_SUCCESS ||
                pool == VK_NULL_HANDLE)
                return false;
            // The batch is retained with the command buffer until its timeline ticket completes.
            const std::vector<VkDescriptorSetLayout> layouts(count, owner.set_layout_);
            std::vector<VkDescriptorSet> sets(count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocate.descriptorPool = pool; allocate.descriptorSetCount = count; allocate.pSetLayouts = layouts.data();
            bool allocated = owner.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(
                owner.device_, &allocate, sets.data()) == VK_SUCCESS;
            for (VkDescriptorSet s : sets) allocated = allocated && (s != VK_NULL_HANDLE);
            if (!allocated) {
                owner.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(owner.device_, pool, nullptr);
                return false;
            }
            pools.push_back(pool);
            out.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                VkWriteDescriptorSet writes[2]{};
                const VkDescriptorBufferInfo* infos[]={&windows[i].source,&windows[i].destination};
                for(std::uint32_t binding=0;binding<2;++binding){
                    auto& w=writes[binding];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=sets[i];
                    w.dstBinding=binding;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=infos[binding];
                }
                owner.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(owner.device_,2,writes,0,nullptr);
                out.push_back({sets[i],windows[i].src,windows[i].dst,windows[i].bytes});
            }
            return true;
        }
    };
    std::unique_ptr<Batch> batch(){return std::make_unique<Batch>(*this);}
    void emit(VkCommandBuffer cb, const std::vector<PredicatedCopyDispatch>& jobs) const {
        if (!cb || jobs.empty() || !pipeline_ || !layout_ || !get_ || !device_) return;
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const auto bind_pipeline = fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        const auto bind_sets = fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        if (!barrier || !bind_pipeline || !bind_sets || !push_constants || !dispatch) return;
        // Barriers are unconditional even when the dispatch is skipped by conditional rendering. They order
        // transfer producers, dependent copies, and adjacent windows whose aligned descriptors may overlap.
        for (const auto& job : jobs) {
            if (!job.bytes || !job.set) continue;
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            before.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT;
            before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);
            bind_pipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &job.set, 0, nullptr);
            const std::uint32_t push[]={job.source,job.destination,job.bytes};
            push_constants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            dispatch(cb, (job.bytes - 1u) / 64u + 1u, 1, 1);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT;
            after.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
            barrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&after,0,nullptr,0,nullptr);
        }
    }
private:
    template<class T>T fn(const char* name)const{return reinterpret_cast<T>(get_(device_,name));}
    bool make_pipeline() {
        if (pipeline_) return true;
        if (!get_ || !device_) return false;
        // No entry-point assumptions for partial/mock devices: fail before allocating or recording anything.
        for (const char* name : {"vkCreateDescriptorPool", "vkDestroyDescriptorPool",
             "vkAllocateDescriptorSets", "vkUpdateDescriptorSets", "vkCreateShaderModule", "vkDestroyShaderModule",
             "vkCreateDescriptorSetLayout", "vkDestroyDescriptorSetLayout", "vkCreatePipelineLayout",
             "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline",
             "vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdPushConstants", "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkDescriptorSetLayoutBinding bindings[]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 2; set_info.pBindings = bindings;
        bool ok = fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(
                      device_, &set_info, nullptr, &set) == VK_SUCCESS && set != VK_NULL_HANDLE;
        if (!ok) set = VK_NULL_HANDLE;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set;
        layout_info.pushConstantRangeCount = 1; layout_info.pPushConstantRanges = &push;
        if (ok) {
            ok = fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
                     device_, &layout_info, nullptr, &layout) == VK_SUCCESS && layout != VK_NULL_HANDLE;
            if (!ok) layout = VK_NULL_HANDLE;
        }
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = sizeof(predicated_copy_spirv); module.pCode = predicated_copy_spirv;
        if (ok) {
            ok = fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(
                     device_, &module, nullptr, &shader) == VK_SUCCESS && shader != VK_NULL_HANDLE;
            if (!ok) shader = VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module = shader; info.stage.pName = "main";
        info.layout = layout;
        if (ok) {
            ok = fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                     device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS && pipeline != VK_NULL_HANDLE;
            if (!ok) pipeline = VK_NULL_HANDLE;
        }
        if (shader) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, shader, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            if (set) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, set, nullptr);
            return false;
        }
        set_layout_ = set; layout_ = layout; pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceLimits limits_;
    VkDescriptorSetLayout set_layout_=VK_NULL_HANDLE;
    VkPipelineLayout layout_=VK_NULL_HANDLE;
    VkPipeline pipeline_=VK_NULL_HANDLE;
};
}
