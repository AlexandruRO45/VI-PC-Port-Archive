// Unit tests for GPU indirect dispatch count gating (runtime/src/vulkan_indirect_dispatch_count.h). CPU-only, no
// Vulkan instance and no GPU: exercises argument validation, missing proc / pipeline rollback, memory selection and
// allocation cleanup, BDA overflow guards, multi-chunk emit() push constants/dispatch sizing, and teardown.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_indirect_dispatch_count.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::CountedDispatch;
using leonida::IndirectDispatchCounter;

namespace {

struct PushCopy {
    VkDeviceAddress arguments, count, output;
    std::uint32_t stride_words, first_record, records;
};

struct MockVk {
    const char* missing_fn = nullptr;
    const char* fail_fn = nullptr;
    std::uint32_t memory_type_bits = 0b11u;
    std::uint32_t last_alloc_type = UINT32_MAX;
    VkDeviceSize last_buffer_size = 0;
    VkBufferUsageFlags last_buffer_usage = 0;
    VkDeviceAddress args_va = 0x100000;
    VkDeviceAddress count_va = 0x200000;
    VkDeviceAddress output_va = 0x300000;

    int create_shader = 0, destroy_shader = 0;
    int create_layout = 0, destroy_layout = 0;
    int create_pipeline = 0, destroy_pipeline = 0;
    int create_buffer = 0, destroy_buffer = 0;
    int alloc_memory = 0, free_memory = 0, bind_memory = 0;

    std::vector<std::string> ops;
    std::vector<PushCopy> pushes;
    std::vector<std::uint32_t> dispatches;
};

MockVk* g_mock = nullptr;

template<class H> H make_handle(std::uintptr_t v) { return reinterpret_cast<H>(v); }
bool should_fail(const char* name) { return g_mock->fail_fn && std::strcmp(g_mock->fail_fn, name) == 0; }

VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader(
        VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
    if (should_fail("vkCreateShaderModule")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_shader;
    *out = make_handle<VkShaderModule>(0x101);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_shader(VkDevice, VkShaderModule h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_shader;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_layout(
        VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    if (should_fail("vkCreatePipelineLayout")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_layout;
    *out = make_handle<VkPipelineLayout>(0x102);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_layout(VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_layout;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipelines(
        VkDevice, VkPipelineCache, std::uint32_t, const VkComputePipelineCreateInfo*,
        const VkAllocationCallbacks*, VkPipeline* out) {
    if (should_fail("vkCreateComputePipelines")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pipeline;
    *out = make_handle<VkPipeline>(0x103);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pipeline;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_buffer(
        VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    if (should_fail("vkCreateBuffer")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->create_buffer;
    g_mock->last_buffer_size = info->size;
    g_mock->last_buffer_usage = info->usage;
    *out = make_handle<VkBuffer>(0x201);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_buffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_buffer;
}
VKAPI_ATTR void VKAPI_CALL mock_get_mem_reqs(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    out->size = 256;
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
VKAPI_ATTR VkDeviceAddress VKAPI_CALL mock_get_bda(VkDevice, const VkBufferDeviceAddressInfo* info) {
    if (info->buffer == make_handle<VkBuffer>(0x301)) return g_mock->args_va;
    if (info->buffer == make_handle<VkBuffer>(0x302)) return g_mock->count_va;
    return g_mock->output_va;
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_barrier(
        VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
        std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
        std::uint32_t, const VkImageMemoryBarrier*) {
    g_mock->ops.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {
    g_mock->ops.push_back("bind");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_push(
        VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t size, const void* values) {
    g_mock->ops.push_back("push");
    if (size == sizeof(PushCopy)) {
        PushCopy copy{};
        std::memcpy(&copy, values, sizeof(copy));
        g_mock->pushes.push_back(copy);
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
        {"vkCreateBuffer", reinterpret_cast<PFN_vkVoidFunction>(mock_create_buffer)},
        {"vkDestroyBuffer", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_buffer)},
        {"vkGetBufferMemoryRequirements", reinterpret_cast<PFN_vkVoidFunction>(mock_get_mem_reqs)},
        {"vkAllocateMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_memory)},
        {"vkFreeMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_free_memory)},
        {"vkBindBufferMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_bind_memory)},
        {"vkGetBufferDeviceAddress", reinterpret_cast<PFN_vkVoidFunction>(mock_get_bda)},
        {"vkCreateShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader)},
        {"vkDestroyShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader)},
        {"vkCreatePipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_layout)},
        {"vkDestroyPipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_layout)},
        {"vkCreateComputePipelines", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipelines)},
        {"vkDestroyPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_barrier)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_pipeline)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_push)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_dispatch)},
    };
    for (const auto& e : table) {
        if (std::strcmp(e.name, name) == 0) return e.fn;
    }
    return nullptr;
}

VkPhysicalDeviceMemoryProperties make_mem(bool device_local) {
    VkPhysicalDeviceMemoryProperties props{};
    props.memoryTypeCount = 2;
    props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    props.memoryTypes[1].propertyFlags = device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                      : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return props;
}

VkPhysicalDeviceLimits make_limits(std::uint32_t max_wg_x) {
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = max_wg_x;
    return limits;
}

int test_validation_and_pipeline_rollback() {
    const VkBuffer args = make_handle<VkBuffer>(0x301);
    const VkBuffer cnt = make_handle<VkBuffer>(0x302);
    CountedDispatch out{};
    {
        MockVk mock;
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        auto b = counter.batch();
        EXPECT(!b->prepare(VK_NULL_HANDLE, 0, cnt, 0, 12, 1, out));
        EXPECT(!b->prepare(args, 0, VK_NULL_HANDLE, 0, 12, 1, out));
        EXPECT(!b->prepare(args, 0, cnt, 0, 0, 1, out));
        EXPECT(!b->prepare(args, 0, cnt, 0, 10, 1, out)); // stride not a multiple of 4
        EXPECT(!b->prepare(args, 0, cnt, 0, 12, 0, out));
        EXPECT(mock.create_shader == 0);
    }
    {
        MockVk mock;
        g_mock = &mock;
        IndirectDispatchCounter no_wg(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(0));
        EXPECT(!no_wg.batch()->prepare(args, 0, cnt, 0, 12, 1, out));
        EXPECT(mock.create_shader == 0);
    }
    {
        MockVk mock;
        mock.missing_fn = "vkGetBufferDeviceAddress";
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        EXPECT(!counter.batch()->prepare(args, 0, cnt, 0, 12, 1, out));
        EXPECT(mock.create_shader == 0);
    }
    for (const char* fail : {"vkCreateShaderModule", "vkCreatePipelineLayout", "vkCreateComputePipelines"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        {
            IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
            EXPECT(!counter.batch()->prepare(args, 0, cnt, 0, 12, 1, out));
        }
        EXPECT(mock.create_shader == mock.destroy_shader);
        EXPECT(mock.create_layout == mock.destroy_layout);
        EXPECT(mock.create_pipeline == mock.destroy_pipeline);
    }
    return 0;
}

int test_allocation_failures_and_bda_guards() {
    const VkBuffer args = make_handle<VkBuffer>(0x301);
    const VkBuffer cnt = make_handle<VkBuffer>(0x302);
    CountedDispatch out{};
    for (const char* fail : {"vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        EXPECT(!counter.batch()->prepare(args, 0, cnt, 0, 12, 2, out));
        EXPECT(mock.create_buffer == mock.destroy_buffer);
        EXPECT(mock.alloc_memory == mock.free_memory);
    }
    {
        MockVk mock;
        mock.memory_type_bits = 0;
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        EXPECT(!counter.batch()->prepare(args, 0, cnt, 0, 12, 2, out));
        EXPECT(mock.destroy_buffer == 1 && mock.alloc_memory == 0);
    }
    {
        MockVk mock;
        mock.args_va = 0;
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        EXPECT(!counter.batch()->prepare(args, 0, cnt, 0, 12, 2, out));
        EXPECT(mock.destroy_buffer == 1 && mock.free_memory == 1);
    }
    {
        MockVk mock;
        mock.args_va = UINT64_MAX - 15;
        g_mock = &mock;
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(64));
        EXPECT(!counter.batch()->prepare(args, 16, cnt, 0, 12, 2, out)); // argument_offset overflows BDA
    }
    return 0;
}

int test_prepare_emit_chunks_and_teardown() {
    const VkBuffer args = make_handle<VkBuffer>(0x301);
    const VkBuffer cnt = make_handle<VkBuffer>(0x302);
    MockVk mock;
    g_mock = &mock;
    {
        // maxComputeWorkGroupCount[0] == 1 -> dispatch_limit == 64 records per chunk.
        IndirectDispatchCounter counter(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(1));
        auto b = counter.batch();
        CountedDispatch job{};
        EXPECT(b->prepare(args, 16, cnt, 8, 24, 130, job));
        EXPECT(mock.last_alloc_type == 1);
        EXPECT(mock.last_buffer_size == 130u * 12u);
        EXPECT((mock.last_buffer_usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0);
        EXPECT(job.arguments == mock.args_va + 16 && job.count == mock.count_va + 8 &&
               job.output == mock.output_va && job.stride == 24 && job.max_count == 130);

        counter.emit(make_handle<VkCommandBuffer>(0x401), job);
        const std::vector<std::string> expected_ops = {
            "barrier", "bind", "push", "dispatch", "push", "dispatch", "push", "dispatch", "barrier"
        };
        EXPECT(mock.ops == expected_ops);
        EXPECT(mock.pushes.size() == 3);
        EXPECT(mock.pushes[0].first_record == 0 && mock.pushes[0].records == 64 &&
               mock.pushes[0].stride_words == 6 && mock.pushes[0].count == job.count);
        EXPECT(mock.pushes[1].first_record == 64 && mock.pushes[1].records == 64 &&
               mock.pushes[1].stride_words == 6 && mock.pushes[1].count == job.count &&
               mock.pushes[1].arguments == job.arguments + 64 * 24 &&
               mock.pushes[1].output == job.output + 64 * 12);
        EXPECT(mock.pushes[2].first_record == 128 && mock.pushes[2].records == 2);
        EXPECT((mock.dispatches == std::vector<std::uint32_t>{1, 1, 1}));

        // Large stride forces index_limit + 1 == 8 records per chunk even with large workgroup limit.
        mock.ops.clear();
        mock.pushes.clear();
        mock.dispatches.clear();
        IndirectDispatchCounter wide(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(false), make_limits(1024));
        auto bw = wide.batch();
        CountedDispatch wide_job{};
        EXPECT(bw->prepare(args, 0, cnt, 0, 0x80000000u, 10, wide_job));
        EXPECT(mock.last_alloc_type == 0); // Fallback when no device-local memory exists
        wide.emit(make_handle<VkCommandBuffer>(0x401), wide_job);
        EXPECT(mock.pushes.size() == 2);
        EXPECT(mock.pushes[0].records == 8 && mock.pushes[0].stride_words == 0x20000000u &&
               mock.pushes[0].count == wide_job.count);
        EXPECT(mock.pushes[1].first_record == 8 && mock.pushes[1].records == 2 &&
               mock.pushes[1].arguments == wide_job.arguments + 8ull * 0x80000000u);

        // Multi-workgroup dispatches: max_wg_x == 4 (256 records/chunk), 321 records -> 256 (x=4) + 65 (x=2).
        mock.dispatches.clear();
        IndirectDispatchCounter multi(mock_get_proc, make_handle<VkDevice>(0x1), make_mem(true), make_limits(4));
        auto bm = multi.batch();
        CountedDispatch multi_job{};
        EXPECT(bm->prepare(args, 0, cnt, 0, 12, 321, multi_job));
        multi.emit(make_handle<VkCommandBuffer>(0x401), multi_job);
        EXPECT((mock.dispatches == std::vector<std::uint32_t>{4, 2}));
    }
    EXPECT(mock.destroy_buffer == 3 && mock.free_memory == 3);
    EXPECT(mock.destroy_pipeline == 3 && mock.destroy_layout == 3);
    return 0;
}

} // namespace

int main() {
    if (test_validation_and_pipeline_rollback()) return 1;
    if (test_allocation_failures_and_bda_guards()) return 1;
    if (test_prepare_emit_chunks_and_teardown()) return 1;
    std::puts("vulkan_indirect_dispatch_count_test: ok");
    return 0;
}
