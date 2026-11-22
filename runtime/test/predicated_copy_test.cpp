// Preparation-only checks: no Vulkan calls, loader, shaders, GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdio>
#define EXPECT(x) do { if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;} }while(0)
// Non-null capability markers; prepare() never calls these entry points. Nothing is emitted by this test.
template<class T>T available(){return reinterpret_cast<T>(+[](){});}
template<class T>T handle(std::uintptr_t n){return reinterpret_cast<T>(n);}
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
    const auto before_rc=prepared;
    EXPECT(prepare(resource_copy,resolve,vk,result)==ReplayResult::success&&result.unpredicated_copies==0&&
           prepared==before_rc+1&&result.commands.back().buffer_copies.empty()&&
           !result.commands.back().predicated_copy.empty());
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
    PredicatedOps missing_ops(+[](VkDevice,const char*)->PFN_vkVoidFunction{return nullptr;},VK_NULL_HANDLE,{});
    std::vector<PredicatedStridedDispatch> strided_jobs;
    std::vector<PredicatedFillDispatch> fill_jobs;
    const std::uint8_t pat[4]={1,2,3,4};
    EXPECT(!missing_ops.batch()->prepare_strided(source.buffer,destination.buffer,0,0,4,2,1,8,16,4,8,strided_jobs));
    EXPECT(!missing_ops.batch()->append_fill(destination.buffer,0,8,pat,4,fill_jobs));

    // Extended routing, capability, MSAA rejection, readback-stage and statistics-split checks for predicated
    // CopyResource(tex), CopyTexture, ResolveSubresource, ClearUnorderedAccess and AtomicCopyBuffer.
    ResourceBinding tex_src{},tex_dst{},tex_msaa{},tex_3d{},buf_uav{};
    tex_src.image=handle<VkImage>(10);tex_src.format=VK_FORMAT_R8G8B8A8_UNORM;tex_src.aspects=VK_IMAGE_ASPECT_COLOR_BIT;
    tex_src.width=4;tex_src.height=4;tex_src.depth=1;tex_src.mip_levels=1;tex_src.array_layers=1;tex_src.samples=1;
    tex_dst=tex_src;tex_dst.image=handle<VkImage>(11);
    tex_msaa=tex_src;tex_msaa.image=handle<VkImage>(12);tex_msaa.samples=4;
    tex_3d=tex_src;tex_3d.image=handle<VkImage>(13);tex_3d.depth=4;tex_3d.volume=true;
    buf_uav.buffer=handle<VkBuffer>(14);buf_uav.size=64;buf_uav.host_readable=true;
    ResourceBinding buf_src{};buf_src.buffer=handle<VkBuffer>(15);buf_src.size=1024;
    ResourceBinding buf_rb{};buf_rb.buffer=handle<VkBuffer>(16);buf_rb.size=1024;buf_rb.host_readable=true;
    auto resolve2=[&](ResourceId id)->const ResourceBinding*{
        switch(id){
        case 1: return &source; case 2: return &destination; case 3: return &predicate;
        case 10: return &tex_src; case 11: return &tex_dst; case 12: return &tex_msaa;
        case 13: return &tex_3d; case 14: return &buf_uav; case 15: return &buf_src; case 16: return &buf_rb;
        default: return nullptr;
        }
    };
    unsigned scratch_calls=0,strided_calls=0,fill_calls=0,copy_calls=0;
    ReplayFunctions vk2=vk;
    vk2.resolve_image=available<PFN_vkCmdResolveImage>();
    vk2.update_buffer=available<PFN_vkCmdUpdateBuffer>();
    vk2.copy_scratch=[&](VkDeviceSize,VkBuffer& b,VkDeviceSize& off){
        b=handle<VkBuffer>(20+scratch_calls++);off=0;return true;
    };
    vk2.prepare_predicated_copy=[&](VkBuffer,VkBuffer,VkDeviceSize,VkDeviceSize,VkDeviceSize bytes,
                                    std::vector<PredicatedCopyDispatch>& out){
        ++copy_calls;out.push_back({handle<VkDescriptorSet>(30),0,0,static_cast<std::uint32_t>(bytes)});return true;
    };
    vk2.prepare_predicated_strided=[&](VkBuffer,VkBuffer,VkDeviceSize,VkDeviceSize,VkDeviceSize rb,
                                       std::uint32_t r,std::uint32_t s,VkDeviceSize,VkDeviceSize,VkDeviceSize,
                                       VkDeviceSize,std::vector<PredicatedStridedDispatch>& out){
        ++strided_calls;out.push_back({handle<VkDescriptorSet>(31),0,0,static_cast<std::uint32_t>(rb),r,s,0,0,0,0});return true;
    };
    vk2.emit_predicated_strided=[](VkCommandBuffer,const std::vector<PredicatedStridedDispatch>&){};
    vk2.prepare_predicated_fill=[&](VkBuffer,VkDeviceSize,VkDeviceSize bytes,const std::uint8_t*,std::uint32_t es,
                                    std::vector<PredicatedFillDispatch>& out){
        ++fill_calls;out.push_back({handle<VkDescriptorSet>(32),0,static_cast<std::uint32_t>(bytes),es,0,{}});return true;
    };
    vk2.emit_predicated_fill=[](VkCommandBuffer,const std::vector<PredicatedFillDispatch>&){};

    // Predicated texture CopyResource: 1x succeeds via scratch+copy; MSAA-to-MSAA is unsupported.
    {
        CommandList l;
        EXPECT(l.set_predication({3,0,d3d12_predication::equal_zero}) && l.copy_resource({11,10}) && l.close());
        EXPECT(prepare(l,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               result.commands[1].staged && result.commands[1].staged_dst && !result.commands[1].predicated_copy.empty());
        ResourceBinding tex_msaa2=tex_msaa; tex_msaa2.image=handle<VkImage>(17);
        auto res_msaa=[&](ResourceId id)->const ResourceBinding*{ return id==17?&tex_msaa2:resolve2(id); };
        CommandList lm;
        EXPECT(lm.set_predication({3,0,d3d12_predication::equal_zero}) && lm.copy_resource({17,12}) && lm.close());
        EXPECT(prepare(lm,res_msaa,vk2,result)==ReplayResult::unsupported);
    }
    // Predicated CopyTexture upload & readback route to strided helper; readback reports COMPUTE_SHADER_BIT.
    {
        const CopyLocation fp_src{15,true,0,0,4,4,1,256}, fp_dst{16,true,0,0,4,4,1,256};
        const CopyLocation t_src{10,false,0,0,0,0,0,0}, t_dst{11,false,0,0,0,0,0,0};
        CommandList up,rb;
        EXPECT(up.set_predication({3,0,d3d12_predication::equal_zero}) && up.copy_texture({t_dst,0,0,0,fp_src,false,{}}) && up.close());
        EXPECT(rb.set_predication({3,0,d3d12_predication::equal_zero}) && rb.copy_texture({fp_dst,0,0,0,t_src,false,{}}) && rb.close());
        ReplayFunctions no_strided=vk2; no_strided.prepare_predicated_strided=nullptr;
        EXPECT(prepare(up,resolve2,no_strided,result)==ReplayResult::unsupported);
        EXPECT(prepare(rb,resolve2,no_strided,result)==ReplayResult::unsupported);
        EXPECT(prepare(up,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               !result.commands[1].predicated_strided.empty() && !result.wrote_readback);
        EXPECT(prepare(rb,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               !result.commands[1].predicated_strided.empty() && result.wrote_readback &&
               result.readback_src_stages==VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }
    // Predicated ResolveSubresource (color) routes to scratch+copy with unpredicated_copies == 0.
    {
        CommandList res;
        EXPECT(res.set_predication({3,0,d3d12_predication::equal_zero}) && res.resolve_subresource({11,0,12,0,{}}) && res.close());
        EXPECT(prepare(res,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               result.commands[1].staged_resolve_saved && result.commands[1].staged_resolve_captured &&
               !result.commands[1].predicated_copy.empty());
    }
    // Predicated ClearUnorderedAccess: <4-byte buffer rect & 3D volume texture route to fill helper + COMPUTE_SHADER_BIT.
    {
        const Rect narrow{1,0,4,1};
        CommandList bclr,tclr;
        EXPECT(bclr.set_predication({3,0,d3d12_predication::equal_zero}) &&
               bclr.clear_unordered_access({14,true,std::uint32_t(VK_FORMAT_R8_UINT),0,64,1,0,0,1},false,{7,0,0,0},
                                           std::span<const Rect>(&narrow,1)) && bclr.close());
        EXPECT(tclr.set_predication({3,0,d3d12_predication::equal_zero}) &&
               tclr.clear_unordered_access({13,false,std::uint32_t(VK_FORMAT_R8G8B8A8_UNORM),0,0,0,0,0,1},false,
                                           {1,2,3,4},{}) && tclr.close());
        EXPECT(prepare(bclr,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               !result.commands[1].predicated_fill.empty() && result.wrote_readback &&
               result.readback_src_stages==VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        EXPECT(prepare(tclr,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               result.commands[1].staged && !result.commands[1].predicated_fill.empty());
    }
    // Predicated AtomicCopyBuffer routes to copy helper + COMPUTE_SHADER_BIT; overlapping physical range rejected.
    {
        ResourceBinding buf_ov{}; buf_ov.buffer=buf_src.buffer; buf_ov.buffer_offset=4; buf_ov.size=64;
        auto res_ov=[&](ResourceId id)->const ResourceBinding*{ return id==18?&buf_ov:resolve2(id); };
        CommandList ac,ac_ov;
        EXPECT(ac.set_predication({3,0,d3d12_predication::equal_zero}) &&
               ac.atomic_copy({16,8,15,8,8,{{15,0,0,64}}}) && ac.close());
        EXPECT(ac_ov.set_predication({3,0,d3d12_predication::equal_zero}) &&
               ac_ov.atomic_copy({18,0,15,0,8,{{15,0,0,64}}}) && ac_ov.close());
        EXPECT(prepare(ac,resolve2,vk2,result)==ReplayResult::success && result.unpredicated_copies==0 &&
               !result.commands[1].predicated_copy.empty() && result.wrote_readback &&
               result.readback_src_stages==VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        EXPECT(prepare(ac_ov,res_ov,vk2,result)==ReplayResult::invalid_copy);
    }
    // Statistics query splitting for predicated strided copy and predicated fill.
    {
        const CopyLocation fp_src{15,true,0,0,4,4,1,256}, t_dst{11,false,0,0,0,0,0,0};
        CommandList st;
        EXPECT(st.set_predication({3,0,d3d12_predication::equal_zero}) &&
               st.begin_query({1,d3d12_query::pipeline_statistics,0}) &&
               st.copy_texture({t_dst,0,0,0,fp_src,false,{}}) &&
               st.clear_unordered_access({14,true,0,0,64,4,0,0,1},false,{1,2,3,4},{}) &&
               st.end_query({1,d3d12_query::pipeline_statistics,0}) && st.close());
        const auto s_before=strided_calls, f_before=fill_calls;
        EXPECT(prepare(st,resolve2,vk2,result,false)==ReplayResult::unsupported &&
               strided_calls==s_before && fill_calls==f_before);
        EXPECT(prepare(st,resolve2,vk2,result,true)==ReplayResult::success && result.statistics_segment_count==3 &&
               result.commands[2].statistics_end==0 && result.commands[2].statistics_begin==1 &&
               result.commands[3].statistics_end==1 && result.commands[3].statistics_begin==2);
    }
    EXPECT(copy_calls==3 && strided_calls==3 && fill_calls==3);
    std::puts("predicated copy preparation: capability, routing, 64-bit offsets, bounds, overflow, overlap, zero size, failure, extended ops and statistics-query controls passed");
}
