// Unit tests for GPU predicate reduction (runtime/src/vulkan_predicate.h). CPU-only, no Vulkan instance and no GPU:
// exercises batch bounds, missing device entry points, pipeline/resource allocation failure cleanup, device-local
// memory preference and fallback, Batch::prepare slot stepping, capture barrier/push-constant/dispatch sequence, and
// destructor teardown.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_predicate.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::PredicateCapture;
using leonida::PredicateReducer;

namespace {

struct MockVk {
    const char* missing_fn = nullptr;
    const char* fail_fn = nullptr;

    std::uint32_t memory_type_bits = 0b111u;
    VkDeviceSize mem_req_size = 256;
    std::uint32_t last_alloc_type = UINT32_MAX;
    VkBufferCreateInfo last_buffer_info{};
    VkDescriptorBufferInfo last_desc_info{};

    int create_set_layout = 0, destroy_set_layout = 0;
    int create_pipeline_layout = 0, destroy_pipeline_layout = 0;
    int create_shader = 0, destroy_shader = 0;
    int create_pipeline = 0, destroy_pipeline = 0;
    int create_buffer = 0, destroy_buffer = 0;
    int alloc_memory = 0, free_memory = 0, bind_memory = 0;
    int create_pool = 0, destroy_pool = 0;
    int alloc_sets = 0, update_sets = 0;

    std::vector<std::string> ops;
    std::vector<std::pair<VkPipelineStageFlags, VkPipelineStageFlags>> barrier_stages;
    VkBufferCopy last_copy{};
    VkBufferMemoryBarrier last_buf_barrier{};
    std::uint32_t last_push_index = UINT32_MAX;
    std::uint32_t last_dispatch[3] = {};
};

MockVk* g_mock = nullptr;

VkDeviceSize non_null_handle(std::uintptr_t v) { return static_cast<VkDeviceSize>(v); }
template<class H> H make_handle(std::uintptr_t v) { return reinterpret_cast<H>(non_null_handle(v)); }

bool should_fail(const char* name) {
    return g_mock->fail_fn && std::strcmp(g_mock->fail_fn, name) == 0;
}

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
VKAPI_ATTR VkResult VKAPI_CALL mock_create_compute_pipelines(
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
VKAPI_ATTR VkResult VKAPI_CALL mock_create_buffer(
        VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    if (should_fail("vkCreateBuffer")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->create_buffer;
    g_mock->last_buffer_info = *info;
    *out = make_handle<VkBuffer>(0x201);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_buffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_buffer;
}
VKAPI_ATTR void VKAPI_CALL mock_get_mem_reqs(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    out->size = g_mock->mem_req_size;
    out->alignment = 16;
    out->memoryTypeBits = g_mock->memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_alloc_memory(
        VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    if (should_fail("vkAllocateMemory")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->alloc_memory;
    g_mock->last_alloc_type = info->memoryTypeIndex;
    *out = make_handle<VkDeviceMemory>(0x202);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_free_memory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->free_memory;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_bind_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    if (should_fail("vkBindBufferMemory")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->bind_memory;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pool(
        VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    if (should_fail("vkCreateDescriptorPool")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pool;
    *out = make_handle<VkDescriptorPool>(0x203);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice, VkDescriptorPool h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pool;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_alloc_sets(
        VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet* out) {
    if (should_fail("vkAllocateDescriptorSets")) return VK_ERROR_OUT_OF_POOL_MEMORY;
    ++g_mock->alloc_sets;
    *out = make_handle<VkDescriptorSet>(0x204);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_update_sets(
        VkDevice, std::uint32_t count, const VkWriteDescriptorSet* writes, std::uint32_t, const VkCopyDescriptorSet*) {
    ++g_mock->update_sets;
    if (count > 0 && writes[0].pBufferInfo) g_mock->last_desc_info = *writes[0].pBufferInfo;
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_barrier(
        VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags,
        std::uint32_t mem_count, const VkMemoryBarrier*, std::uint32_t buf_count,
        const VkBufferMemoryBarrier* bufs, std::uint32_t, const VkImageMemoryBarrier*) {
    g_mock->ops.push_back("barrier");
    g_mock->barrier_stages.emplace_back(src, dst);
    if (buf_count > 0 && bufs) g_mock->last_buf_barrier = bufs[0];
    (void)mem_count;
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_copy(
        VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t count, const VkBufferCopy* regions) {
    g_mock->ops.push_back("copy");
    if (count > 0) g_mock->last_copy = regions[0];
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
    if (size == 4) std::memcpy(&g_mock->last_push_index, values, 4);
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    g_mock->ops.push_back("dispatch");
    g_mock->last_dispatch[0] = x;
    g_mock->last_dispatch[1] = y;
    g_mock->last_dispatch[2] = z;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_proc(VkDevice, const char* name) {
    if (g_mock->missing_fn && std::strcmp(g_mock->missing_fn, name) == 0) return nullptr;
    struct Entry { const char* name; PFN_vkVoidFunction fn; };
    static const Entry table[] = {
        {"vkCreateDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_set_layout)},
        {"vkDestroyDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_set_layout)},
        {"vkCreatePipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipeline_layout)},
        {"vkDestroyPipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline_layout)},
        {"vkCreateShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader)},
        {"vkDestroyShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader)},
        {"vkCreateComputePipelines", reinterpret_cast<PFN_vkVoidFunction>(mock_create_compute_pipelines)},
        {"vkDestroyPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline)},
        {"vkCreateBuffer", reinterpret_cast<PFN_vkVoidFunction>(mock_create_buffer)},
        {"vkDestroyBuffer", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_buffer)},
        {"vkGetBufferMemoryRequirements", reinterpret_cast<PFN_vkVoidFunction>(mock_get_mem_reqs)},
        {"vkAllocateMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_memory)},
        {"vkFreeMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_free_memory)},
        {"vkBindBufferMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_bind_memory)},
        {"vkCreateDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pool)},
        {"vkDestroyDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pool)},
        {"vkAllocateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_sets)},
        {"vkUpdateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_update_sets)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_barrier)},
        {"vkCmdCopyBuffer", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_copy)},
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

VkPhysicalDeviceMemoryProperties make_mem_props(bool include_device_local) {
    VkPhysicalDeviceMemoryProperties props{};
    props.memoryTypeCount = 2;
    props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    props.memoryTypes[1].propertyFlags = include_device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                              : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return props;
}

int test_batch_bounds_and_missing_proc() {
    MockVk mock;
    g_mock = &mock;
    const auto mem = make_mem_props(true);
    {
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), mem, 64);
        EXPECT(!reducer.batch(0));
        EXPECT(!reducer.batch(5)); // 5 > max_range_ (64) / 16
        EXPECT(mock.create_set_layout == 0);
    }
    {
        PredicateReducer huge(mock_get_proc, make_handle<VkDevice>(0x1), mem, UINT64_MAX);
        EXPECT(!huge.batch(std::uint64_t(UINT32_MAX) / 4 + 1));
        EXPECT(mock.create_set_layout == 0);
    }
    {
        mock.missing_fn = "vkCmdDispatch";
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), mem, 64);
        EXPECT(!reducer.batch(1));
        EXPECT(mock.create_set_layout == 0);
    }
    return 0;
}

int test_pipeline_failure_rolls_back_partially_created_objects() {
    const char* stages[] = {
        "vkCreateDescriptorSetLayout",
        "vkCreatePipelineLayout",
        "vkCreateShaderModule",
        "vkCreateComputePipelines",
    };
    for (const char* fail : stages) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        {
            PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), make_mem_props(true), 256);
            EXPECT(!reducer.batch(2));
        }
        EXPECT(mock.create_set_layout == mock.destroy_set_layout);
        EXPECT(mock.create_pipeline_layout == mock.destroy_pipeline_layout);
        EXPECT(mock.create_shader == mock.destroy_shader);
        EXPECT(mock.create_pipeline == mock.destroy_pipeline);
        EXPECT(mock.create_buffer == 0);
    }
    return 0;
}

int test_batch_resource_failures_and_memory_selection() {
    const char* stages[] = {
        "vkCreateBuffer",
        "vkAllocateMemory",
        "vkBindBufferMemory",
        "vkCreateDescriptorPool",
        "vkAllocateDescriptorSets",
    };
    for (const char* fail : stages) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), make_mem_props(true), 256);
        EXPECT(!reducer.batch(2));
        EXPECT(mock.create_buffer == mock.destroy_buffer);
        EXPECT(mock.alloc_memory == mock.free_memory);
        EXPECT(mock.create_pool == mock.destroy_pool);
    }
    {
        MockVk mock;
        mock.memory_type_bits = 0; // No compatible memory type
        g_mock = &mock;
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), make_mem_props(true), 256);
        EXPECT(!reducer.batch(2));
        EXPECT(mock.destroy_buffer == 1);
        EXPECT(mock.alloc_memory == 0);
    }
    {
        MockVk mock;
        mock.memory_type_bits = 0b11u;
        g_mock = &mock;
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), make_mem_props(false), 256);
        auto b = reducer.batch(1);
        EXPECT(b != nullptr);
        EXPECT(mock.last_alloc_type == 0); // Fallback to first matching type when none is device-local
    }
    return 0;
}

int test_batch_prepare_capture_and_teardown() {
    MockVk mock;
    g_mock = &mock;
    const auto mem = make_mem_props(true);
    {
        PredicateReducer reducer(mock_get_proc, make_handle<VkDevice>(0x1), mem, 256);
        auto b1 = reducer.batch(2);
        EXPECT(b1 != nullptr);
        EXPECT(mock.last_alloc_type == 1); // Prefers DEVICE_LOCAL type 1 over HOST_VISIBLE type 0
        EXPECT(mock.last_buffer_info.size == 32);
        EXPECT(mock.last_desc_info.range == 32);
        EXPECT((mock.last_buffer_info.usage & VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT) != 0);

        // Second batch reuses cached pipeline/layouts instead of recreating them.
        auto b2 = reducer.batch(1);
        EXPECT(b2 != nullptr);
        EXPECT(mock.create_pipeline == 1);

        PredicateCapture cap{};
        const VkBuffer src = make_handle<VkBuffer>(0x301);
        EXPECT(!b1->prepare(VK_NULL_HANDLE, 0, cap)); // Rejects null source without consuming a slot
        EXPECT(b1->prepare(src, 48, cap));
        EXPECT(cap.source == src && cap.source_offset == 48 && cap.offset == 0);
        EXPECT(b1->prepare(src, 64, cap));
        EXPECT(cap.offset == 16);
        EXPECT(!b1->prepare(src, 80, cap)); // Batch capacity (2) exhausted

        reducer.capture(make_handle<VkCommandBuffer>(0x401), cap);
        const std::vector<std::string> expected_ops = {
            "barrier", "copy", "barrier", "barrier", "bind_pipeline", "bind_sets", "push", "dispatch" , "barrier"
        };
        EXPECT(mock.ops == expected_ops);
        EXPECT(mock.last_copy.srcOffset == 64 && mock.last_copy.dstOffset == 16 && mock.last_copy.size == 8);
        EXPECT(mock.last_buf_barrier.buffer == src && mock.last_buf_barrier.offset == 64 &&
               mock.last_buf_barrier.size == 8);
        EXPECT(mock.last_push_index == 4); // offset 16 / 4
        EXPECT(mock.last_dispatch[0] == 1 && mock.last_dispatch[1] == 1 && mock.last_dispatch[2] == 1);
        EXPECT(mock.barrier_stages[3].second == VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT);
    }
    EXPECT(mock.destroy_pool == 2 && mock.destroy_buffer == 2 && mock.free_memory == 2);
    EXPECT(mock.destroy_pipeline == 1 && mock.destroy_pipeline_layout == 1 && mock.destroy_set_layout == 1);
    return 0;
}

} // namespace

int main() {
    if (test_batch_bounds_and_missing_proc()) return 1;
    if (test_pipeline_failure_rolls_back_partially_created_objects()) return 1;
    if (test_batch_resource_failures_and_memory_selection()) return 1;
    if (test_batch_prepare_capture_and_teardown()) return 1;
    std::puts("vulkan_predicate_test: ok");
    return 0;
}
