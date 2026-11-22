// Synthetic byte-copy, forced descriptor windows and pending-submit lifetime regression. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include <array>
#include <cstring>
#include <cstdio>
#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while(0)
namespace {
PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
unsigned host_barriers = 0;
VkPipelineStageFlags2 host_stages = 0;
VKAPI_ATTR void VKAPI_CALL observe_host(VkCommandBuffer cb, const VkDependencyInfo* info) {
    for (std::uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
        const auto& b = info->pMemoryBarriers[i];
        if (b.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) { ++host_barriers; host_stages |= b.srcStageMask; }
    }
    real_barrier2(cb, info);
}
}
int main() {
    vktest::Device d;
    if(!d.create())return 1;
    if(!d.conditional_rendering || !d.storage_buffer_8bit) {
        std::puts("predicated copy: skipped (requires conditional rendering and storageBuffer8BitAccess)"); return 0;
    }
    using namespace leonida;
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,&memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical,&properties);
    auto limits=properties.limits;
    // Smaller advertised limits force both dispatch and descriptor-window splits on ordinary GPUs.
    limits.maxStorageBufferRange=static_cast<std::uint32_t>(std::max<VkDeviceSize>(256,limits.minStorageBufferOffsetAlignment));
    limits.maxComputeWorkGroupCount[0]=1;
    VulkanContext context(d.device_proc,d.device,true,true);
    context.set_occlusion_resolve_memory(memory);
    context.configure_predication(d.device_proc,memory,properties.limits.maxStorageBufferRange,d.family);
    VulkanQueue queue(context,d.queue,d.family);
    VulkanFence gate(context,0);
    EXPECT(context.loaded() && queue.ok() && gate.semaphore());
    constexpr std::size_t bytes=8207;
    constexpr auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    void *src_data=nullptr,*dst_data=nullptr,*pred_data=nullptr;
    auto src=d.buffer(bytes,usage,&src_data),dst=d.buffer(bytes,usage,&dst_data);
    auto pred=d.buffer(16,usage,&pred_data);
    EXPECT(src&&dst&&pred&&src_data&&dst_data&&pred_data);
    std::array<unsigned char,bytes> input{},expected{};
    for(std::size_t i=0;i<bytes;++i){ input[i]=static_cast<unsigned char>(37*i+11); expected[i]=static_cast<unsigned char>(13*i+173); }
    std::memcpy(src_data,input.data(),bytes); std::memcpy(dst_data,expected.data(),bytes);
    const std::uint64_t predicates[]={UINT64_C(1)<<32,0}; std::memcpy(pred_data,predicates,sizeof(predicates));
    ResourceBinding a{};a.buffer=src;a.buffer_offset=1;a.size=bytes-1;
    ResourceBinding b{};b.buffer=dst;b.buffer_offset=3;b.size=bytes-3;b.host_readable=true;
    ResourceBinding p{};p.buffer=pred;p.size=16;
    context.add_resource(1,a);context.add_resource(2,b);context.add_resource(3,p);
    // D3D-162: capture/rebind and executing/suppressed copy dispatches inside one empty statistics query.
    VkQueryPool statistics = VK_NULL_HANDLE;
    void* statistics_data = nullptr;
    if (d.pipeline_statistics_query) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS; info.queryCount = 1; info.pipelineStatistics = 0x7ff;
        EXPECT(d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool")(d.device, &info, nullptr, &statistics) == VK_SUCCESS);
        context.add_query_pool(4, {statistics, d3d12_query_heap::pipeline_statistics, 1, false, {false}, {}});
        ResourceBinding counters{}; counters.size = 88; counters.host_readable = true;
        counters.buffer = d.buffer(counters.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &statistics_data);
        EXPECT(counters.buffer && statistics_data);
        std::memset(statistics_data, 0xcd, 88); context.add_resource(5, counters);
    }
    CommandList list;
    if (statistics) EXPECT(list.begin_query({4,d3d12_query::pipeline_statistics,0}));
    EXPECT(list.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(list.copy_buffer({2,4,1,2,4096}));  // high-word-only condition; offsets include arena prefixes
    EXPECT(list.copy_buffer({2,6000,2,4,33})); // dependent, non-overlapping copy within one VkBuffer
    EXPECT(list.set_predication({3,8,d3d12_predication::equal_zero}));
    EXPECT(list.copy_buffer({2,0,1,0,3}));    // suppressed: preserves destination prefix exactly
    EXPECT(list.set_predication({3,0,d3d12_predication::not_equal_zero}));
    EXPECT(list.copy_buffer({2,7000,1,0,37}));// suppressed: high word must not be truncated
    EXPECT(list.set_predication({0,0,0}));
    EXPECT(list.copy_buffer({2,5000,1,0,17}));// reset restores ordinary transfer path
    if (statistics) {
        EXPECT(list.end_query({4,d3d12_query::pipeline_statistics,0}));
        EXPECT(list.resolve_queries({4,d3d12_query::pipeline_statistics,0,1,5,0}));
    }
    EXPECT(list.close());
    std::memcpy(expected.data()+7,input.data()+3,4096);
    std::memcpy(expected.data()+6003,expected.data()+7,33);
    std::memcpy(expected.data()+5003,input.data()+1,17);
    const CommandList* pointer=&list;
    // A device without the helper must submit nothing, rather than run a transfer unconditionally.
    EXPECT(queue.execute({&pointer,1})==ReplayResult::unsupported);
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)!=0); // expected includes the not-yet-executed copies
    for(std::size_t i=0;i<bytes;++i) EXPECT(static_cast<unsigned char*>(dst_data)[i]==static_cast<unsigned char>(13*i+173));
    context.configure_predicated_copy(d.device_proc,limits);
    const auto real_submit=context.vk.vkQueueSubmit;
    context.vk.vkQueueSubmit=+[](VkQueue,std::uint32_t,const VkSubmitInfo*,VkFence)->VkResult{return VK_ERROR_OUT_OF_HOST_MEMORY;};
    const auto rejected=queue.execute({&pointer,1});
    context.vk.vkQueueSubmit=real_submit;
    EXPECT(rejected==ReplayResult::unsupported);
    for(std::size_t i=0;i<bytes;++i) EXPECT(static_cast<unsigned char*>(dst_data)[i]==static_cast<unsigned char>(13*i+173));
    queue.wait(gate.semaphore(),1);
    bool submitted=true;
    for(unsigned i=0;i<32;++i) if(queue.execute({&pointer,1})!=ReplayResult::success){submitted=false;break;}
    // Release even on preparation failure, before any destructor can wait for the queue.
    EXPECT(gate.signal(1));EXPECT(queue.idle());EXPECT(submitted);
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)==0 && std::memcmp(src_data,input.data(),bytes)==0);
    for(unsigned i=0;i<32;++i) EXPECT(queue.execute({&pointer,1})==ReplayResult::success);
    EXPECT(queue.idle());
    EXPECT(std::memcmp(dst_data,expected.data(),bytes)==0 && std::memcmp(src_data,input.data(),bytes)==0);
    real_barrier2=context.vk.vkCmdPipelineBarrier2; EXPECT(real_barrier2);
    context.vk.vkCmdPipelineBarrier2=observe_host;
    CommandList ordinary; EXPECT(ordinary.copy_buffer({2,5000,1,0,17}) && ordinary.close());
    const CommandList* plain=&ordinary;
    for(auto kind:{VulkanQueue::QueueKind::direct,VulkanQueue::QueueKind::compute,VulkanQueue::QueueKind::copy}) {
        VulkanQueue typed(context,d.queue,d.family,kind); EXPECT(typed.ok());
        host_barriers=0;host_stages=0;
        if(kind==VulkanQueue::QueueKind::copy) {
            EXPECT(typed.execute({&pointer,1})==ReplayResult::unsupported && host_barriers==0);
            EXPECT(typed.execute({&plain,1})==ReplayResult::success);
            EXPECT(host_stages==VK_PIPELINE_STAGE_2_COPY_BIT);
        } else {
            EXPECT(typed.execute({&pointer,1})==ReplayResult::success);
            EXPECT(host_stages==(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_2_COPY_BIT|
                                 (statistics ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : 0)));
        }
        EXPECT(typed.idle() && host_barriers==1);
        EXPECT((host_stages & ~VulkanQueue::queue_supported_stages(kind))==0);
        EXPECT(std::memcmp(dst_data,expected.data(),bytes)==0);
    }
    context.vk.vkCmdPipelineBarrier2=real_barrier2;
    // Forced-window GPU stress for PredicatedOps (strided upload/readback & phase-continuous repeating-element fill)
    // under the same tiny maxStorageBufferRange + maxComputeWorkGroupCount[0]=1 limits.
    {
        context.enable_copy_scratch(d.device_proc, [&](const VkMemoryRequirements& req) -> VkDeviceMemory {
            for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
                if (!(req.memoryTypeBits & (1u << i))) continue;
                if (!(memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = i;
                VkDeviceMemory mem = VK_NULL_HANDLE;
                if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &ai, nullptr, &mem) == VK_SUCCESS)
                    return mem;
            }
            return VK_NULL_HANDLE;
        });
        constexpr std::uint32_t w = 16, h = 5, pitch = 256, fp_bytes = pitch * h, fill_buf_bytes = 512;
        void *up_ptr = nullptr, *rb_ptr = nullptr, *fill_ptr = nullptr;
        const VkBuffer up_buf = d.buffer(fp_bytes, usage, &up_ptr);
        const VkBuffer rb_buf = d.buffer(fp_bytes, usage, &rb_ptr);
        const VkBuffer fill_buf = d.buffer(fill_buf_bytes, usage, &fill_ptr);
        const VkImage tex = d.image(VK_FORMAT_R8G8B8A8_UNORM, w, h, 1,
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(up_buf && rb_buf && fill_buf && tex && up_ptr && rb_ptr && fill_ptr);
        std::array<unsigned char, fp_bytes> up_init{}, rb_init{}, rb_exp{};
        for (std::size_t i = 0; i < fp_bytes; ++i) {
            up_init[i] = static_cast<unsigned char>(i * 29 + 7);
            rb_init[i] = 0xA5;
        }
        rb_exp = rb_init;
        for (std::uint32_t y = 0; y < h; ++y)
            std::memcpy(rb_exp.data() + y * pitch, up_init.data() + y * pitch, w * 4);
        std::memcpy(up_ptr, up_init.data(), fp_bytes);
        std::memcpy(rb_ptr, rb_init.data(), fp_bytes);
        std::array<unsigned char, fill_buf_bytes> fill_init{}, fill_exp{};
        for (std::size_t i = 0; i < fill_buf_bytes; ++i) fill_init[i] = static_cast<unsigned char>(i * 11 + 0x40);
        fill_exp = fill_init;
        const std::array<std::uint32_t, 4> uav_vals{0x01020304u, 0x05060708u, 0x090A0B0Cu, 0x0D0E0F10u};
        std::uint8_t pat16[16]{};
        std::memcpy(pat16, uav_vals.data(), 16);
        for (std::size_t i = 0; i < 400; ++i) fill_exp[16 + i] = pat16[i % 16];
        std::memcpy(fill_ptr, fill_init.data(), fill_buf_bytes);

        ResourceBinding r_up{}, r_rb{}, r_fill{}, r_tex{};
        r_up.buffer = up_buf; r_up.size = fp_bytes;
        r_rb.buffer = rb_buf; r_rb.size = fp_bytes; r_rb.host_readable = true;
        r_fill.buffer = fill_buf; r_fill.size = fill_buf_bytes; r_fill.host_readable = true;
        r_tex.image = tex; r_tex.format = VK_FORMAT_R8G8B8A8_UNORM; r_tex.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        r_tex.width = w; r_tex.height = h; r_tex.depth = 1; r_tex.mip_levels = 1; r_tex.array_layers = 1; r_tex.samples = 1;
        context.add_resource(10, r_up);
        context.add_resource(11, r_rb);
        context.add_resource(12, r_fill);
        context.add_resource(13, r_tex, VK_IMAGE_LAYOUT_UNDEFINED);

        const CopyLocation loc_up{10, true, 0, 0, w, h, 1, pitch};
        const CopyLocation loc_rb{11, true, 0, 0, w, h, 1, pitch};
        const CopyLocation loc_tex{13, false, 0, 0, 0, 0, 0, 0};
        CommandList ops_list;
        EXPECT(ops_list.set_predication({3, 0, d3d12_predication::equal_zero}));  // true (1<<32)
        EXPECT(ops_list.copy_texture({loc_tex, 0, 0, 0, loc_up, false, {}}));
        EXPECT(ops_list.transition({13, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        EXPECT(ops_list.copy_texture({loc_rb, 0, 0, 0, loc_tex, false, {}}));
        EXPECT(ops_list.clear_unordered_access({12, true, std::uint32_t(VK_FORMAT_R32G32B32A32_UINT), 16, 400, 16, 0, 0, 1},
                                               false, uav_vals, {}));
        EXPECT(ops_list.set_predication({3, 8, d3d12_predication::equal_zero}));  // false (0)
        EXPECT(ops_list.clear_unordered_access({12, true, 0, 0, 512, 4, 0, 0, 1}, false, {0xFFFFFFFFu, 0, 0, 0}, {}));
        EXPECT(ops_list.set_predication({0, 0, 0}));
        EXPECT(ops_list.transition({13, all_subresources, d3d12_state::copy_source, d3d12_state::copy_dest}));
        EXPECT(ops_list.close());

        const CommandList* ops_ptr = &ops_list;
        for (unsigned i = 0; i < 8; ++i) EXPECT(queue.execute({&ops_ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        EXPECT(std::memcmp(rb_ptr, rb_exp.data(), fp_bytes) == 0);
        EXPECT(std::memcmp(fill_ptr, fill_exp.data(), fill_buf_bytes) == 0);
        context.remove_resource(10);
        context.remove_resource(11);
        context.remove_resource(12);
        context.remove_resource(13);
    }
    if (statistics) {
        for (unsigned word=0; word<11; ++word) {
            std::uint64_t value=UINT64_MAX;
            std::memcpy(&value, static_cast<const char*>(statistics_data)+8*word, 8);
            EXPECT(value==0);
        }
        d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(d.device,statistics,nullptr);
    }
    EXPECT(d.validation_errors()==0);
    std::puts("predicated copy integration: failed-submit cleanup and direct/compute/copy host stages passed");
    std::puts("predicated copy: 32 gated + 32 reused submissions; byte offsets, forced windows, copy dependency, rebinding/reset and untouched bytes passed");
}
