// Unit tests for byte-granular conditional buffer copy (runtime/src/vulkan_predicated_copy.h). CPU-only, no
// Vulkan instance and no GPU: exercises zero-byte / overlap / overflow guards, missing proc & pipeline rollback,
// alignment-padded multi-window splitting, descriptor pool/set failure cleanup, emit() sequence, and teardown.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_predicated_copy.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::PredicatedBufferCopier;
using leonida::PredicatedCopyDispatch;

namespace {

struct RecordedDescWrite {
    VkDescriptorSet set;
    std::uint32_t binding;
    VkDescriptorBufferInfo info;
};

struct MockVk {
    const char* missing_fn = nullptr;
    const char* fail_fn = nullptr;

    int create_set_layout = 0, destroy_set_layout = 0;
    int create_pipeline_layout = 0, destroy_pipeline_layout = 0;
    int create_shader = 0, destroy_shader = 0;
    int create_pipeline = 0, destroy_pipeline = 0;
    int create_pool = 0, destroy_pool = 0;
    int alloc_sets = 0;
    std::uintptr_t next_set = 0x500;

    std::vector<RecordedDescWrite> desc_writes;
    std::vector<std::string> ops;
    std::vector<std::array<std::uint32_t, 3>> pushes;
    std::vector<std::uint32_t> dispatches;
};

MockVk* g_mock = nullptr;

template<class H> H make_handle(std::uintptr_t v) { return reinterpret_cast<H>(v); }
bool should_fail(const char* name) { return g_mock->fail_fn && std::strcmp(g_mock->fail_fn, name) == 0; }

VKAPI_ATTR VkResult VKAPI_CALL mock_create_set_layout(
        VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    if (should_fail("vkCreateDescriptorSetLayout")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_set_layout;
    *out = make_handle<VkDescriptorSetLayout>(0x101);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_set_layout(VkDevice, VkDescriptorSetLayout h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_set_layout;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipeline_layout(
        VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    if (should_fail("vkCreatePipelineLayout")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pipeline_layout;
    *out = make_handle<VkPipelineLayout>(0x102);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline_layout(VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pipeline_layout;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader(
        VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
    if (should_fail("vkCreateShaderModule")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_shader;
    *out = make_handle<VkShaderModule>(0x103);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_shader(VkDevice, VkShaderModule h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_shader;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipelines(
        VkDevice, VkPipelineCache, std::uint32_t, const VkComputePipelineCreateInfo*,
        const VkAllocationCallbacks*, VkPipeline* out) {
    if (should_fail("vkCreateComputePipelines")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pipeline;
    *out = make_handle<VkPipeline>(0x104);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pipeline;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pool(
        VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    if (should_fail("vkCreateDescriptorPool")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pool;
    *out = make_handle<VkDescriptorPool>(0x201);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice, VkDescriptorPool h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pool;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_alloc_sets(
        VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    if (should_fail("vkAllocateDescriptorSets")) return VK_ERROR_OUT_OF_POOL_MEMORY;
    ++g_mock->alloc_sets;
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) {
        out[i] = make_handle<VkDescriptorSet>(g_mock->next_set++);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_update_sets(
        VkDevice, std::uint32_t count, const VkWriteDescriptorSet* writes, std::uint32_t, const VkCopyDescriptorSet*) {
    for (std::uint32_t i = 0; i < count; ++i) {
        g_mock->desc_writes.push_back({writes[i].dstSet, writes[i].dstBinding, *writes[i].pBufferInfo});
    }
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_barrier(
        VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
        std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
        std::uint32_t, const VkImageMemoryBarrier*) {
    g_mock->ops.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {
    g_mock->ops.push_back("bind_pipeline");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_sets(
        VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
        const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {
    g_mock->ops.push_back("bind_sets");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_push(
        VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t size, const void* values) {
    g_mock->ops.push_back("push");
    if (size == 12) {
        std::array<std::uint32_t, 3> p{};
        std::memcpy(p.data(), values, 12);
        g_mock->pushes.push_back(p);
    }
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t, std::uint32_t) {
    g_mock->ops.push_back("dispatch");
    g_mock->dispatches.push_back(x);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_proc(VkDevice, const char* name) {
    if (g_mock->missing_fn && std::strcmp(g_mock->missing_fn, name) == 0) return nullptr;
    struct Entry { const char* name; PFN_vkVoidFunction fn; };
    static const Entry table[] = {
        {"vkCreateDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pool)},
        {"vkDestroyDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pool)},
        {"vkAllocateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_sets)},
        {"vkUpdateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_update_sets)},
        {"vkCreateShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader)},
        {"vkDestroyShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader)},
        {"vkCreateDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_set_layout)},
        {"vkDestroyDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_set_layout)},
        {"vkCreatePipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipeline_layout)},
        {"vkDestroyPipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline_layout)},
        {"vkCreateComputePipelines", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipelines)},
        {"vkDestroyPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_barrier)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_pipeline)},
        {"vkCmdBindDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_sets)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_push)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_dispatch)},
    };
    for (const auto& e : table) {
        if (std::strcmp(e.name, name) == 0) return e.fn;
    }
    return nullptr;
}

VkPhysicalDeviceLimits make_limits(VkDeviceSize align, VkDeviceSize max_range, std::uint32_t max_wg_x) {
    VkPhysicalDeviceLimits limits{};
    limits.minStorageBufferOffsetAlignment = align;
    limits.maxStorageBufferRange = static_cast<std::uint32_t>(max_range);
    limits.maxComputeWorkGroupCount[0] = max_wg_x;
    return limits;
}

int test_validation_and_pipeline_rollback() {
    const VkBuffer src = make_handle<VkBuffer>(0x301);
    const VkBuffer dst = make_handle<VkBuffer>(0x302);
    std::vector<PredicatedCopyDispatch> out(1);
    {
        MockVk mock;
        g_mock = &mock;
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 64));
        auto b = copier.batch();
        EXPECT(b->prepare(VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0, 0, out) && out.empty());
        EXPECT(!b->prepare(VK_NULL_HANDLE, dst, 0, 0, 8, out));
        EXPECT(!b->prepare(src, VK_NULL_HANDLE, 0, 0, 8, out));
        EXPECT(!b->prepare(src, dst, UINT64_MAX - 4, 0, 8, out));
        EXPECT(!b->prepare(src, dst, 0, UINT64_MAX - 4, 8, out));
        EXPECT(!b->prepare(src, src, 10, 14, 8, out)); // Overlapping same-buffer range
        EXPECT(mock.create_set_layout == 0);
    }
    {
        MockVk mock;
        mock.missing_fn = "vkCmdDispatch";
        g_mock = &mock;
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 64));
        EXPECT(!copier.batch()->prepare(src, dst, 0, 0, 8, out));
        EXPECT(mock.create_set_layout == 0);
    }
    for (const char* fail : {"vkCreateDescriptorSetLayout", "vkCreatePipelineLayout",
                             "vkCreateShaderModule", "vkCreateComputePipelines"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        {
            PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 64));
            EXPECT(!copier.batch()->prepare(src, dst, 0, 0, 8, out));
        }
        EXPECT(mock.create_set_layout == mock.destroy_set_layout);
        EXPECT(mock.create_pipeline_layout == mock.destroy_pipeline_layout);
        EXPECT(mock.create_shader == mock.destroy_shader);
        EXPECT(mock.create_pipeline == mock.destroy_pipeline);
    }
    return 0;
}

int test_window_limits_and_pool_failures() {
    const VkBuffer src = make_handle<VkBuffer>(0x301);
    const VkBuffer dst = make_handle<VkBuffer>(0x302);
    std::vector<PredicatedCopyDispatch> out;
    {
        MockVk mock;
        g_mock = &mock;
        // sp = 16 >= maximum (16) and dp = 16 >= maximum (16) -> cannot fit even 1 byte in window
        PredicatedBufferCopier small_range(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(32, 16, 64));
        EXPECT(!small_range.batch()->prepare(src, dst, 16, 0, 4, out));
        EXPECT(!small_range.batch()->prepare(src, dst, 0, 16, 4, out));
        EXPECT(mock.create_pool == 0);

        PredicatedBufferCopier zero_wg(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 0));
        EXPECT(!zero_wg.batch()->prepare(src, dst, 0, 0, 4, out));
        EXPECT(mock.create_pool == 0);
    }
    for (const char* fail : {"vkCreateDescriptorPool", "vkAllocateDescriptorSets"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 64));
        EXPECT(!copier.batch()->prepare(src, dst, 0, 0, 8, out));
        EXPECT(mock.create_pool == mock.destroy_pool);
    }
    return 0;
}

int test_alignment_splitting_emit_and_teardown() {
    const VkBuffer src = make_handle<VkBuffer>(0x301);
    const VkBuffer dst = make_handle<VkBuffer>(0x302);
    MockVk mock;
    g_mock = &mock;
    {
        // alignment = 16, max_range = 64, threads = 64.
        // Copy 90 bytes from src=20 (sp=4) to dst=35 (dp=3):
        // window 0: n = min(90, 64-4, 64-3, 64) = 60 -> s=20, d=35, sp=4, dp=3
        // window 1: done=60 -> s=80 (sp=0), d=95 (dp=15), remaining=30 -> n=30
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 64, 1));
        auto b = copier.batch();
        std::vector<PredicatedCopyDispatch> jobs;
        EXPECT(b->prepare(src, dst, 20, 35, 90, jobs));
        EXPECT(jobs.size() == 2);
        EXPECT(jobs[0].source == 4 && jobs[0].destination == 3 && jobs[0].bytes == 60);
        EXPECT(jobs[1].source == 0 && jobs[1].destination == 15 && jobs[1].bytes == 30);

        EXPECT(mock.desc_writes.size() == 4);
        EXPECT(mock.desc_writes[0].set == jobs[0].set && mock.desc_writes[0].binding == 0 &&
               mock.desc_writes[0].info.buffer == src && mock.desc_writes[0].info.offset == 16 &&
               mock.desc_writes[0].info.range == 64);
        EXPECT(mock.desc_writes[1].set == jobs[0].set && mock.desc_writes[1].binding == 1 &&
               mock.desc_writes[1].info.buffer == dst && mock.desc_writes[1].info.offset == 32 &&
               mock.desc_writes[1].info.range == 63);
        EXPECT(mock.desc_writes[2].set == jobs[1].set && mock.desc_writes[2].binding == 0 &&
               mock.desc_writes[2].info.buffer == src && mock.desc_writes[2].info.offset == 80 &&
               mock.desc_writes[2].info.range == 30);
        EXPECT(mock.desc_writes[3].set == jobs[1].set && mock.desc_writes[3].binding == 1 &&
               mock.desc_writes[3].info.buffer == dst && mock.desc_writes[3].info.offset == 80 &&
               mock.desc_writes[3].info.range == 45);

        // Non-overlapping same-buffer copy succeeds and reuses cached pipeline.
        std::vector<PredicatedCopyDispatch> non_overlap;
        EXPECT(b->prepare(src, src, 0, 16, 16, non_overlap));
        EXPECT(mock.create_pipeline == 1);

        copier.emit(make_handle<VkCommandBuffer>(0x401), jobs);
        const std::vector<std::string> expected_ops = {
            "barrier", "bind_pipeline", "bind_sets", "push", "dispatch", "barrier",
            "barrier", "bind_pipeline", "bind_sets", "push", "dispatch", "barrier",
        };
        EXPECT(mock.ops == expected_ops);
        EXPECT(mock.pushes.size() == 2);
        EXPECT((mock.pushes[0] == std::array<std::uint32_t, 3>{4, 3, 60}));
        EXPECT((mock.pushes[1] == std::array<std::uint32_t, 3>{0, 15, 30}));
        EXPECT((mock.dispatches == std::vector<std::uint32_t>{1, 1}));
    }
    EXPECT(mock.destroy_pool == 2);
    EXPECT(mock.destroy_pipeline == 1 && mock.destroy_pipeline_layout == 1 && mock.destroy_set_layout == 1);
    return 0;
}

int test_destination_and_dispatch_limits() {
    const VkBuffer src = make_handle<VkBuffer>(0x301);
    const VkBuffer dst = make_handle<VkBuffer>(0x302);
    {
        // Destination alignment prefix dp=12 > sp=2 caps window 0 at maximum - dp = 52.
        MockVk mock;
        g_mock = &mock;
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 64, 16));
        auto b = copier.batch();
        std::vector<PredicatedCopyDispatch> jobs;
        EXPECT(b->prepare(src, dst, 18, 28, 60, jobs));
        EXPECT(jobs.size() == 2);
        EXPECT(jobs[0].source == 2 && jobs[0].destination == 12 && jobs[0].bytes == 52);
        EXPECT(jobs[1].source == 6 && jobs[1].destination == 0 && jobs[1].bytes == 8);
    }
    {
        // maxComputeWorkGroupCount[0] * 64 = 128 < maxStorageBufferRange = 256 splits at 128;
        // remaining 65 bytes pins multi-workgroup ceil-div (65 - 1) / 64 + 1 = 2.
        MockVk mock;
        g_mock = &mock;
        PredicatedBufferCopier copier(mock_get_proc, make_handle<VkDevice>(0x1), make_limits(16, 256, 2));
        auto b = copier.batch();
        std::vector<PredicatedCopyDispatch> jobs;
        EXPECT(b->prepare(src, dst, 0, 0, 193, jobs));
        EXPECT(jobs.size() == 2);
        EXPECT(jobs[0].bytes == 128 && jobs[1].bytes == 65);
        copier.emit(make_handle<VkCommandBuffer>(0x401), jobs);
        EXPECT((mock.dispatches == std::vector<std::uint32_t>{2, 2}));
    }
    return 0;
}

} // namespace

int main() {
    if (test_validation_and_pipeline_rollback()) return 1;
    if (test_window_limits_and_pool_failures()) return 1;
    if (test_alignment_splitting_emit_and_teardown()) return 1;
    if (test_destination_and_dispatch_limits()) return 1;
    std::puts("vulkan_predicated_copy_test: ok");
    return 0;
}
