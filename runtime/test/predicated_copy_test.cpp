// Preparation-only checks: no Vulkan calls, loader, shaders, GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdio>
#include <cstring>
#include <set>
#define EXPECT(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;} }while(0)
// Non-null capability markers; prepare() never calls these entry points. Nothing is emitted by this test.
template<class T>T available(){return reinterpret_cast<T>(+[](){});}
template<class T>T handle(std::uintptr_t n){return reinterpret_cast<T>(n);}

namespace mock_vk {

constexpr std::uintptr_t kSentinel = 0xdeadbeefu;

struct State {
    std::uintptr_t next_id = 100;
    bool fail_set_layout = false;
    bool fail_pipeline_layout = false;
    bool fail_shader_module = false;
    bool fail_compute_pipeline = false;
    bool fail_create_pool = false;
    bool fail_alloc_sets = false;

    unsigned set_layouts_created = 0;
    unsigned set_layouts_destroyed = 0;
    unsigned pipeline_layouts_created = 0;
    unsigned pipeline_layouts_destroyed = 0;
    unsigned shaders_created = 0;
    unsigned shaders_destroyed = 0;
    unsigned pipelines_created = 0;
    unsigned pipelines_destroyed = 0;
    unsigned pools_created = 0;
    unsigned pools_destroyed = 0;
    unsigned invalid_destroy_calls = 0;
    unsigned barrier_calls = 0;
    unsigned dispatch_calls = 0;
    std::uint32_t last_dispatch_x = 0;

    std::set<std::uintptr_t> live_set_layouts;
    std::set<std::uintptr_t> live_pipeline_layouts;
    std::set<std::uintptr_t> live_shaders;
    std::set<std::uintptr_t> live_pipelines;
    std::set<std::uintptr_t> live_pools;

    void reset() { *this = State{}; }
};

State g_state;

template<class H>
H next_handle() {
    return reinterpret_cast<H>(++g_state.next_id);
}

template<class H>
void record_destroy(std::set<std::uintptr_t>& live, unsigned& destroyed, H h) {
    if (h == VK_NULL_HANDLE) return;
    if (!live.erase(reinterpret_cast<std::uintptr_t>(h))) {
        ++g_state.invalid_destroy_calls;
        return;
    }
    ++destroyed;
}

VKAPI_ATTR VkResult VKAPI_CALL create_set_layout(
    VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    if (g_state.fail_set_layout) {
        *out = handle<VkDescriptorSetLayout>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = next_handle<VkDescriptorSetLayout>();
    g_state.live_set_layouts.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.set_layouts_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_set_layout(
    VkDevice, VkDescriptorSetLayout h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_set_layouts, g_state.set_layouts_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL create_pipeline_layout(
    VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    if (g_state.fail_pipeline_layout) {
        *out = handle<VkPipelineLayout>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = next_handle<VkPipelineLayout>();
    g_state.live_pipeline_layouts.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.pipeline_layouts_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_pipeline_layout(
    VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_pipeline_layouts, g_state.pipeline_layouts_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL create_shader_module(
    VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
    if (g_state.fail_shader_module) {
        *out = handle<VkShaderModule>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = next_handle<VkShaderModule>();
    g_state.live_shaders.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.shaders_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_shader_module(
    VkDevice, VkShaderModule h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_shaders, g_state.shaders_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL create_compute_pipelines(
    VkDevice, VkPipelineCache, std::uint32_t count, const VkComputePipelineCreateInfo*,
    const VkAllocationCallbacks*, VkPipeline* out) {
    if (g_state.fail_compute_pipeline) {
        for (std::uint32_t i = 0; i < count; ++i) out[i] = handle<VkPipeline>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = next_handle<VkPipeline>();
        g_state.live_pipelines.insert(reinterpret_cast<std::uintptr_t>(out[i]));
        ++g_state.pipelines_created;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_pipeline(
    VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_pipelines, g_state.pipelines_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL create_descriptor_pool(
    VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    if (g_state.fail_create_pool) {
        *out = handle<VkDescriptorPool>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = next_handle<VkDescriptorPool>();
    g_state.live_pools.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.pools_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_descriptor_pool(
    VkDevice, VkDescriptorPool h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_pools, g_state.pools_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_descriptor_sets(
    VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    if (g_state.fail_alloc_sets) return VK_ERROR_OUT_OF_POOL_MEMORY;
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = next_handle<VkDescriptorSet>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL update_descriptor_sets(
    VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*) {}

VKAPI_ATTR void VKAPI_CALL cmd_pipeline_barrier(
    VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {
    ++g_state.barrier_calls;
}

VKAPI_ATTR void VKAPI_CALL cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}

VKAPI_ATTR void VKAPI_CALL cmd_bind_descriptor_sets(
    VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
    const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}

VKAPI_ATTR void VKAPI_CALL cmd_push_constants(
    VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t, const void*) {}

VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t, std::uint32_t) {
    ++g_state.dispatch_calls;
    g_state.last_dispatch_x = x;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
    if (!name) return nullptr;
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_descriptor_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_descriptor_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&allocate_descriptor_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&update_descriptor_sets);
    if (std::strcmp(name, "vkCreateShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_shader_module);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_shader_module);
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_set_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_set_layout);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_pipeline_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_pipeline_layout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_compute_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_pipeline);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_pipeline_barrier);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_bind_pipeline);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_bind_descriptor_sets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_push_constants);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_dispatch);
    return nullptr;
}

}  // namespace mock_vk
int main(){
    using namespace leonida;
    ReplayFunctions vk{};
    vk.copy_buffer=available<PFN_vkCmdCopyBuffer>();vk.copy_buffer_to_image=available<PFN_vkCmdCopyBufferToImage>();
    vk.copy_image_to_buffer=available<PFN_vkCmdCopyImageToBuffer>();vk.copy_image=available<PFN_vkCmdCopyImage>();
    vk.begin_conditional_rendering=available<PFN_vkCmdBeginConditionalRenderingEXT>();
    vk.end_conditional_rendering=available<PFN_vkCmdEndConditionalRenderingEXT>();
    vk.prepare_predicate=[](VkBuffer,VkDeviceSize,PredicateCapture& out){out.scratch=handle<VkBuffer>(4);return true;};
    vk.capture_predicate=[](VkCommandBuffer,const PredicateCapture&){};
    ResourceBinding source{},destination{},predicate{};
    source.buffer=handle<VkBuffer>(1);source.size=64;source.buffer_offset=(UINT64_C(1)<<33)+1;
    destination.buffer=handle<VkBuffer>(2);destination.size=64;destination.buffer_offset=(UINT64_C(1)<<34)+3;
    predicate.buffer=handle<VkBuffer>(3);predicate.size=16;
    auto resolve=[&](ResourceId id)->const ResourceBinding*{return id==1?&source:id==2?&destination:id==3?&predicate:nullptr;};
    VkDeviceSize got_src=0,got_dst=0,got_bytes=0;unsigned prepared=0;
    auto copier=[&](VkBuffer a,VkBuffer b,VkDeviceSize src,VkDeviceSize dst,VkDeviceSize bytes,std::vector<PredicatedCopyDispatch>& jobs){
        if(a!=source.buffer||b!=destination.buffer)return false;
        got_src=src;got_dst=dst;got_bytes=bytes;++prepared;
        jobs.push_back({handle<VkDescriptorSet>(5),0,0,static_cast<std::uint32_t>(bytes)});return true;
    };
    PreparedList result;
    auto check=[&](CopyBuffer copy,bool predicated=true){
        CommandList list;
        if(predicated)list.set_predication({3,8,d3d12_predication::equal_zero});
        list.copy_buffer(copy);list.close();
        return prepare(list,resolve,vk,result);
    };
    EXPECT(check({2,3,1,1,7})==ReplayResult::unsupported); // optional capability absent
    vk.prepare_predicated_copy=copier;
    EXPECT(check({2,3,1,1,7})==ReplayResult::unsupported); // emitter absent
    vk.emit_predicated_copy=[](VkCommandBuffer,const std::vector<PredicatedCopyDispatch>&){};
    EXPECT(check({2,3,1,1,7})==ReplayResult::success);
    EXPECT(prepared==1&&got_src==source.buffer_offset+1&&got_dst==destination.buffer_offset+3&&got_bytes==7);
    EXPECT(result.unpredicated_copies==0&&result.commands.back().buffer_copies.empty());
    EXPECT(check({2,63,1,63,1},false)==ReplayResult::success);
    EXPECT(result.commands.back().buffer_copies[0].srcOffset==source.buffer_offset+63);
    EXPECT(result.commands.back().buffer_copies[0].dstOffset==destination.buffer_offset+63&&prepared==1);
    EXPECT(check({2,64,1,64,0})==ReplayResult::success&&prepared==1); // native no-op; no shader/descriptor
    EXPECT(check({2,64,1,0,1})==ReplayResult::invalid_copy);
    EXPECT(check({2,0,1,64,1})==ReplayResult::invalid_copy);
    EXPECT(check({9,0,1,0,1})==ReplayResult::unresolved_resource);
    const auto saved=source;
    source.buffer_offset=UINT64_MAX;
    EXPECT(check({2,0,1,1,1})==ReplayResult::invalid_copy); // offset addition
    EXPECT(check({2,0,1,0,2})==ReplayResult::invalid_copy); // end addition
    source=saved;
    destination.buffer=source.buffer;destination.buffer_offset=source.buffer_offset;
    EXPECT(check({2,1,1,0,7})==ReplayResult::invalid_copy); // physical overlap across resource IDs
    EXPECT(check({2,8,1,0,7})==ReplayResult::success);     // same buffer, disjoint ranges
    destination.buffer=handle<VkBuffer>(2);destination.buffer_offset=0;
    vk.prepare_predicated_copy=[](VkBuffer,VkBuffer,VkDeviceSize,VkDeviceSize,VkDeviceSize,std::vector<PredicatedCopyDispatch>&){return false;};
    EXPECT(check({2,0,1,0,1})==ReplayResult::unsupported); // allocation/preparation failure cannot fall back
    vk.prepare_predicated_copy=copier;
    CommandList resource_copy;
    EXPECT(resource_copy.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(resource_copy.copy_resource({2,1})&&resource_copy.close());
    EXPECT(prepare(resource_copy,resolve,vk,result)==ReplayResult::success&&result.unpredicated_copies==1);
    vk.reset_query_pool=available<PFN_vkCmdResetQueryPool>();vk.begin_query=available<PFN_vkCmdBeginQuery>();
    vk.end_query=available<PFN_vkCmdEndQuery>();vk.write_timestamp=available<PFN_vkCmdWriteTimestamp>();
    vk.copy_query_pool_results=available<PFN_vkCmdCopyQueryPoolResults>();vk.fill_buffer=available<PFN_vkCmdFillBuffer>();
    QueryPool pool{};pool.pool=handle<VkQueryPool>(6);pool.heap_type=d3d12_query_heap::pipeline_statistics;pool.count=1;
    vk.query_pool=[&](QueryHeapId){return &pool;};
    CommandList statistics;
    EXPECT(statistics.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(statistics.begin_query({1,d3d12_query::pipeline_statistics,0}));
    EXPECT(statistics.copy_buffer({2,0,1,0,1}));
    EXPECT(statistics.end_query({1,d3d12_query::pipeline_statistics,0})&&statistics.close());
    const auto before=prepared;
    EXPECT(prepare(statistics,resolve,vk,result)==ReplayResult::unsupported && prepared==before);
    EXPECT(prepare(statistics,resolve,vk,result,true)==ReplayResult::success && prepared==before+1);
    EXPECT(result.statistics_segment_count==2);
    EXPECT(result.commands[2].statistics_end==0 && result.commands[2].statistics_begin==1);
    // Readback visibility follows the actual producer, including mixed command lists.
    destination.host_readable = false;
    EXPECT(check({2,0,1,0,1})==ReplayResult::success && !result.wrote_readback && !result.readback_src_stages);
    destination.host_readable = true;
    EXPECT(check({2,0,1,0,1})==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT(check({2,0,1,0,1},false)==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==VK_PIPELINE_STAGE_2_COPY_BIT);
    CommandList mixed;
    EXPECT(mixed.set_predication({3,0,d3d12_predication::equal_zero}));
    EXPECT(mixed.copy_buffer({2,0,1,0,1}) && mixed.set_predication({0,0,0}));
    EXPECT(mixed.copy_buffer({2,1,1,1,1}) && mixed.close());
    EXPECT(prepare(mixed,resolve,vk,result)==ReplayResult::success && result.wrote_readback);
    EXPECT(result.readback_src_stages==(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_2_COPY_BIT));
    EXPECT(check({2,0,1,0,0})==ReplayResult::success && !result.wrote_readback && !result.readback_src_stages);
    PredicatedBufferCopier missing(+[](VkDevice,const char*)->PFN_vkVoidFunction{return nullptr;},VK_NULL_HANDLE,{});
    std::vector<PredicatedCopyDispatch> jobs;
    EXPECT(!missing.batch()->prepare(source.buffer,destination.buffer,0,0,1,jobs));

    // Regression tests for PredicatedBufferCopier pool cleanup, emit() guards, pipeline partial init, and null guards.
    VkPhysicalDeviceLimits limits{};
    limits.minStorageBufferOffsetAlignment = 16;
    limits.maxStorageBufferRange = 256;
    limits.maxComputeWorkGroupCount[0] = 64;
    const VkDevice mock_dev = handle<VkDevice>(1);
    const VkCommandBuffer mock_cb = handle<VkCommandBuffer>(10);

    // Check A: vkAllocateDescriptorSets failure immediately destroys the newly created pool, and retries succeed.
    {
        mock_vk::g_state.reset();
        PredicatedBufferCopier copier(mock_vk::get_proc, mock_dev, limits);
        mock_vk::g_state.fail_alloc_sets = true;
        auto batch = copier.batch();
        EXPECT(!batch->prepare(source.buffer, destination.buffer, 0, 0, 64, jobs));
        EXPECT(batch->pools.empty() && mock_vk::g_state.pools_created == 1 &&
               mock_vk::g_state.pools_destroyed == 1 && mock_vk::g_state.invalid_destroy_calls == 0);
        mock_vk::g_state.fail_alloc_sets = false;
        EXPECT(batch->prepare(source.buffer, destination.buffer, 0, 0, 64, jobs) &&
               batch->pools.size() == 1 && jobs.size() == 1);

        // Check B: emit() skips zero-byte/null-set jobs and uninitialized copiers safely.
        mock_vk::g_state.barrier_calls = mock_vk::g_state.dispatch_calls = 0;
        copier.emit(mock_cb, {{handle<VkDescriptorSet>(99), 0, 0, 0}, {VK_NULL_HANDLE, 0, 0, 16}});
        EXPECT(mock_vk::g_state.dispatch_calls == 0 && mock_vk::g_state.barrier_calls == 0);
        copier.emit(mock_cb, jobs);
        EXPECT(mock_vk::g_state.dispatch_calls == 1 && mock_vk::g_state.last_dispatch_x == 1 &&
               mock_vk::g_state.barrier_calls == 2);
        PredicatedBufferCopier uninit(mock_vk::get_proc, mock_dev, limits);
        mock_vk::g_state.barrier_calls = mock_vk::g_state.dispatch_calls = 0;
        uninit.emit(mock_cb, jobs);
        EXPECT(mock_vk::g_state.dispatch_calls == 0 && mock_vk::g_state.barrier_calls == 0);
    }
    EXPECT(mock_vk::g_state.pools_created == mock_vk::g_state.pools_destroyed &&
           mock_vk::g_state.pipelines_created == mock_vk::g_state.pipelines_destroyed &&
           mock_vk::g_state.invalid_destroy_calls == 0);

    // Check C: partial make_pipeline() and create-pool failures never destroy sentinel handles and retry cleanly.
    for (int stage = 0; stage < 5; ++stage) {
        mock_vk::g_state.reset();
        if (stage == 0) mock_vk::g_state.fail_set_layout = true;
        if (stage == 1) mock_vk::g_state.fail_pipeline_layout = true;
        if (stage == 2) mock_vk::g_state.fail_shader_module = true;
        if (stage == 3) mock_vk::g_state.fail_compute_pipeline = true;
        if (stage == 4) mock_vk::g_state.fail_create_pool = true;
        {
            PredicatedBufferCopier copier(mock_vk::get_proc, mock_dev, limits);
            EXPECT(!copier.batch()->prepare(source.buffer, destination.buffer, 0, 0, 16, jobs));
            EXPECT(mock_vk::g_state.invalid_destroy_calls == 0);
            if (stage < 4) {
                EXPECT(mock_vk::g_state.set_layouts_created == mock_vk::g_state.set_layouts_destroyed &&
                       mock_vk::g_state.pipeline_layouts_created == mock_vk::g_state.pipeline_layouts_destroyed &&
                       mock_vk::g_state.shaders_created == mock_vk::g_state.shaders_destroyed &&
                       mock_vk::g_state.pipelines_created == mock_vk::g_state.pipelines_destroyed);
            }
            mock_vk::g_state.fail_set_layout = false;
            mock_vk::g_state.fail_pipeline_layout = false;
            mock_vk::g_state.fail_shader_module = false;
            mock_vk::g_state.fail_compute_pipeline = false;
            mock_vk::g_state.fail_create_pool = false;
            EXPECT(copier.batch()->prepare(source.buffer, destination.buffer, 0, 0, 16, jobs));
        }
        EXPECT(mock_vk::g_state.set_layouts_created == mock_vk::g_state.set_layouts_destroyed &&
               mock_vk::g_state.pipeline_layouts_created == mock_vk::g_state.pipeline_layouts_destroyed &&
               mock_vk::g_state.shaders_created == mock_vk::g_state.shaders_destroyed &&
               mock_vk::g_state.pipelines_created == mock_vk::g_state.pipelines_destroyed &&
               mock_vk::g_state.pools_created == mock_vk::g_state.pools_destroyed &&
               mock_vk::g_state.invalid_destroy_calls == 0);
    }

    // Check D: null get_ and null device_ are safe in prepare(), emit(), and destructors.
    {
        PredicatedBufferCopier null_get(nullptr, mock_dev, limits);
        EXPECT(!null_get.batch()->prepare(source.buffer, destination.buffer, 0, 0, 16, jobs));
        EXPECT(null_get.batch()->prepare(source.buffer, destination.buffer, 0, 0, 0, jobs) && jobs.empty());
        null_get.emit(mock_cb, {{handle<VkDescriptorSet>(99), 0, 0, 16}});
        null_get.emit(VK_NULL_HANDLE, jobs);
        PredicatedBufferCopier null_dev(mock_vk::get_proc, VK_NULL_HANDLE, limits);
        EXPECT(!null_dev.batch()->prepare(source.buffer, destination.buffer, 0, 0, 16, jobs));
    }

    std::puts("predicated copy preparation: capability, routing, 64-bit offsets, bounds, overflow, overlap, zero size, failure and statistics-query controls passed");
}
