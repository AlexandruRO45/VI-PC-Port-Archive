// Synthetic GPU lifetime test: gate the queue so no predicate arena can retire before all submissions exist.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <set>
#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
namespace mock_vk {

constexpr std::uintptr_t kSentinel = 0xdeadbeefu;

template<class H>
H handle(std::uintptr_t n) {
    return reinterpret_cast<H>(n);
}

struct State {
    std::uintptr_t next_id = 100;
    VkDeviceSize req_size = 64;
    std::uint32_t req_type_bits = 1u;

    bool fail_set_layout = false;
    bool fail_pipeline_layout = false;
    bool fail_shader_module = false;
    bool fail_compute_pipeline = false;
    bool fail_create_buffer = false;
    bool fail_alloc_memory = false;
    bool fail_bind_memory = false;
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
    unsigned buffers_created = 0;
    unsigned buffers_destroyed = 0;
    unsigned memories_created = 0;
    unsigned memories_destroyed = 0;
    unsigned pools_created = 0;
    unsigned pools_destroyed = 0;
    unsigned alloc_memory_calls = 0;
    unsigned invalid_destroy_calls = 0;
    unsigned barrier_calls = 0;
    unsigned copy_calls = 0;
    unsigned dispatch_calls = 0;

    std::set<std::uintptr_t> live_set_layouts;
    std::set<std::uintptr_t> live_pipeline_layouts;
    std::set<std::uintptr_t> live_shaders;
    std::set<std::uintptr_t> live_pipelines;
    std::set<std::uintptr_t> live_buffers;
    std::set<std::uintptr_t> live_memories;
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

VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_pipelines, g_state.pipelines_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(
    VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* out) {
    if (g_state.fail_create_buffer) {
        *out = handle<VkBuffer>(kSentinel);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = next_handle<VkBuffer>();
    g_state.live_buffers.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.buffers_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_buffers, g_state.buffers_destroyed, h);
}

VKAPI_ATTR void VKAPI_CALL get_buffer_memory_requirements(
    VkDevice, VkBuffer, VkMemoryRequirements* out) {
    out->size = g_state.req_size;
    out->alignment = 16;
    out->memoryTypeBits = g_state.req_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(
    VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    ++g_state.alloc_memory_calls;
    if (g_state.fail_alloc_memory) {
        *out = handle<VkDeviceMemory>(kSentinel);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = next_handle<VkDeviceMemory>();
    g_state.live_memories.insert(reinterpret_cast<std::uintptr_t>(*out));
    ++g_state.memories_created;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    record_destroy(g_state.live_memories, g_state.memories_destroyed, h);
}

VKAPI_ATTR VkResult VKAPI_CALL bind_buffer_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g_state.fail_bind_memory ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
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

VKAPI_ATTR void VKAPI_CALL cmd_copy_buffer(
    VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {
    ++g_state.copy_calls;
}

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

VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    ++g_state.dispatch_calls;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
    if (!name) return nullptr;
    if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&create_buffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_buffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&get_buffer_memory_requirements);
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&allocate_memory);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&free_memory);
    if (std::strcmp(name, "vkBindBufferMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&bind_buffer_memory);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&create_descriptor_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&destroy_descriptor_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&allocate_descriptor_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&update_descriptor_sets);
    if (std::strcmp(name, "vkCreateShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&create_shader_module);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&destroy_shader_module);
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&create_set_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&destroy_set_layout);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&create_pipeline_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&destroy_pipeline_layout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&create_compute_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&destroy_pipeline);
    if (std::strcmp(name, "vkCmdCopyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_copy_buffer);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&cmd_pipeline_barrier);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_bind_pipeline);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&cmd_bind_descriptor_sets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&cmd_push_constants);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&cmd_dispatch);
    return nullptr;
}

}  // namespace mock_vk

namespace {
int run_mock_checks() {
    using namespace leonida;
    const VkDevice mock_dev = mock_vk::handle<VkDevice>(1);
    const VkBuffer mock_buf = mock_vk::handle<VkBuffer>(2);
    const VkCommandBuffer mock_cb = mock_vk::handle<VkCommandBuffer>(3);
    VkPhysicalDeviceMemoryProperties mem{};
    mem.memoryTypeCount = 1;
    mem.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Check A: zero VkMemoryRequirements::size and oversized memoryTypeCount > VK_MAX_MEMORY_TYPES.
    {
        mock_vk::g_state.reset();
        mock_vk::g_state.req_size = 0;
        PredicateReducer r(mock_vk::get_proc, mock_dev, mem, 256);
        EXPECT(!r.batch(1));
        EXPECT(mock_vk::g_state.alloc_memory_calls == 0 && mock_vk::g_state.buffers_created == 1 &&
               mock_vk::g_state.buffers_destroyed == 1 && mock_vk::g_state.invalid_destroy_calls == 0);
    }
    {
        mock_vk::g_state.reset();
        mock_vk::g_state.req_type_bits = 0;
        auto mem_oob = mem;
        mem_oob.memoryTypeCount = 64;
        PredicateReducer r_oob(mock_vk::get_proc, mock_dev, mem_oob, 256);
        EXPECT(!r_oob.batch(1));
        EXPECT(mock_vk::g_state.alloc_memory_calls == 0 && mock_vk::g_state.buffers_created == 1 &&
               mock_vk::g_state.buffers_destroyed == 1 && mock_vk::g_state.invalid_destroy_calls == 0);
    }

    // Check B: partial make_pipeline() and batch() failures do not destroy sentinel handles and retry cleanly.
    for (int stage = 0; stage < 9; ++stage) {
        mock_vk::g_state.reset();
        if (stage == 0) mock_vk::g_state.fail_set_layout = true;
        if (stage == 1) mock_vk::g_state.fail_pipeline_layout = true;
        if (stage == 2) mock_vk::g_state.fail_shader_module = true;
        if (stage == 3) mock_vk::g_state.fail_compute_pipeline = true;
        if (stage == 4) mock_vk::g_state.fail_create_buffer = true;
        if (stage == 5) mock_vk::g_state.fail_alloc_memory = true;
        if (stage == 6) mock_vk::g_state.fail_bind_memory = true;
        if (stage == 7) mock_vk::g_state.fail_create_pool = true;
        if (stage == 8) mock_vk::g_state.fail_alloc_sets = true;
        {
            PredicateReducer r(mock_vk::get_proc, mock_dev, mem, 256);
            EXPECT(!r.batch(2));
            EXPECT(mock_vk::g_state.invalid_destroy_calls == 0);
            EXPECT(mock_vk::g_state.buffers_created == mock_vk::g_state.buffers_destroyed &&
                   mock_vk::g_state.memories_created == mock_vk::g_state.memories_destroyed &&
                   mock_vk::g_state.pools_created == mock_vk::g_state.pools_destroyed);
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
            mock_vk::g_state.fail_create_buffer = false;
            mock_vk::g_state.fail_alloc_memory = false;
            mock_vk::g_state.fail_bind_memory = false;
            mock_vk::g_state.fail_create_pool = false;
            mock_vk::g_state.fail_alloc_sets = false;
            auto b = r.batch(2);
            EXPECT(b != nullptr);
        }
        EXPECT(mock_vk::g_state.set_layouts_created == mock_vk::g_state.set_layouts_destroyed &&
               mock_vk::g_state.pipeline_layouts_created == mock_vk::g_state.pipeline_layouts_destroyed &&
               mock_vk::g_state.shaders_created == mock_vk::g_state.shaders_destroyed &&
               mock_vk::g_state.pipelines_created == mock_vk::g_state.pipelines_destroyed &&
               mock_vk::g_state.buffers_created == mock_vk::g_state.buffers_destroyed &&
               mock_vk::g_state.memories_created == mock_vk::g_state.memories_destroyed &&
               mock_vk::g_state.pools_created == mock_vk::g_state.pools_destroyed &&
               mock_vk::g_state.invalid_destroy_calls == 0);
    }

    // Check C: Batch::prepare() 64-bit offset wrap guard and capture() safety guards.
    PredicateCapture valid_cap{};
    {
        mock_vk::g_state.reset();
        PredicateReducer r(mock_vk::get_proc, mock_dev, mem, 256);
        auto b = r.batch(2);
        EXPECT(b != nullptr);

        PredicateCapture cap{};
        EXPECT(!b->prepare(mock_buf, UINT64_MAX - 4, cap) && b->next == 0);
        EXPECT(b->prepare(mock_buf, 8, cap) && cap.offset == 0 && b->next == 1);
        valid_cap = cap;

        PredicateReducer uninit(mock_vk::get_proc, mock_dev, mem, 256);
        mock_vk::g_state.barrier_calls = mock_vk::g_state.copy_calls = mock_vk::g_state.dispatch_calls = 0;
        uninit.capture(mock_cb, cap);
        EXPECT(mock_vk::g_state.barrier_calls == 0 && mock_vk::g_state.copy_calls == 0 &&
               mock_vk::g_state.dispatch_calls == 0);

        auto try_invalid = [&](VkCommandBuffer cb, PredicateCapture bad) {
            mock_vk::g_state.barrier_calls = mock_vk::g_state.copy_calls = mock_vk::g_state.dispatch_calls = 0;
            r.capture(cb, bad);
            return mock_vk::g_state.barrier_calls == 0 && mock_vk::g_state.copy_calls == 0 &&
                   mock_vk::g_state.dispatch_calls == 0;
        };
        EXPECT(try_invalid(VK_NULL_HANDLE, cap));
        { auto bad = cap; bad.source = VK_NULL_HANDLE; EXPECT(try_invalid(mock_cb, bad)); }
        { auto bad = cap; bad.scratch = VK_NULL_HANDLE; EXPECT(try_invalid(mock_cb, bad)); }
        { auto bad = cap; bad.set = VK_NULL_HANDLE; EXPECT(try_invalid(mock_cb, bad)); }
        { auto bad = cap; bad.source_offset = UINT64_MAX - 4; EXPECT(try_invalid(mock_cb, bad)); }
        { auto bad = cap; bad.offset = 4; EXPECT(try_invalid(mock_cb, bad)); }
        { auto bad = cap; bad.offset = VkDeviceSize(1ULL << 34); EXPECT(try_invalid(mock_cb, bad)); }

        mock_vk::g_state.barrier_calls = mock_vk::g_state.copy_calls = mock_vk::g_state.dispatch_calls = 0;
        r.capture(mock_cb, cap);
        EXPECT(mock_vk::g_state.barrier_calls == 4 && mock_vk::g_state.copy_calls == 1 &&
               mock_vk::g_state.dispatch_calls == 1);
    }

    // Check D: null get_ and null device_ guards in batch(), capture(), and destructors.
    {
        PredicateReducer null_get(nullptr, mock_dev, mem, 256);
        EXPECT(!null_get.batch(1));
        null_get.capture(mock_cb, valid_cap);
        PredicateReducer null_dev(mock_vk::get_proc, VK_NULL_HANDLE, mem, 256);
        EXPECT(!null_dev.batch(1));
        null_dev.capture(mock_cb, valid_cap);
    }

    std::puts("predicate capture mock: size/type-count bounds, partial-init sentinel cleanup, offset-wrap, capture and null guards passed");
    return 0;
}
}  // namespace

namespace {
PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
unsigned host_barriers = 0;
VkPipelineStageFlags2 host_source_stages = 0;
VKAPI_ATTR void VKAPI_CALL observe_barrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
    for (std::uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
        const auto& barrier = info->pMemoryBarriers[i];
        if (barrier.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) {
            ++host_barriers;
            host_source_stages |= barrier.srcStageMask;
        }
    }
    real_barrier2(cb, info);
}
}
int main() {
    if (run_mock_checks() != 0) return 1;
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("vk_predicate_capture_test: no Vulkan device, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.conditional_rendering) { std::puts("predicate capture: skipped (no conditional rendering)"); return 0; }
    using namespace leonida;
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    PredicateReducer limits(d.device_proc, d.device, memory, 16);
    EXPECT(!limits.batch(0) && !limits.batch(2));
    PredicateReducer missing(+[](VkDevice, const char*) -> PFN_vkVoidFunction { return nullptr; }, d.device, memory, 16);
    EXPECT(!missing.batch(1));
    auto no_memory = memory; no_memory.memoryTypeCount = 0;
    PredicateReducer allocation_failure(d.device_proc, d.device, no_memory, 16);
    EXPECT(!allocation_failure.batch(1));
    VulkanContext context(d.device_proc, d.device, true, true);
    context.configure_predication(d.device_proc, memory, properties.limits.maxStorageBufferRange, d.family);
    VulkanQueue queue(context, d.queue, d.family);
    VulkanFence gate(context, 0);
    EXPECT(context.loaded() && queue.ok() && gate.semaphore());
    constexpr unsigned count = 32;
    void* mapped = nullptr;
    const VkBuffer predicate = d.buffer(16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &mapped);
    EXPECT(predicate && mapped);
    const std::uint64_t values[] = {UINT64_C(1) << 32, 0};
    std::memcpy(mapped, values, sizeof(values));
    const VkImage image = d.image(VK_FORMAT_R8G8B8A8_UINT, 2, 1, count, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    void* pixels = nullptr;
    const VkBuffer readback = d.buffer(count * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &pixels);
    EXPECT(image && readback && pixels);
    ResourceBinding source{}; source.buffer = predicate; source.size = 16;
    ResourceBinding target{}; target.image = image; target.format = VK_FORMAT_R8G8B8A8_UINT;
    target.width = 2; target.height = 1; target.array_layers = count;
    context.add_resource(1, source);
    context.add_resource(2, target, VK_IMAGE_LAYOUT_UNDEFINED);
    // Each list captures high-only -> zero -> high-only with two offsets, all while no prior batch can finish.
    std::array<CommandList, count> lists;
    queue.wait(gate.semaphore(), 1);
    bool submitted = true;
    for (unsigned i = 0; i < count; ++i) {
        auto& list = lists[i];
        submitted = submitted && list.set_predication({1, 0, d3d12_predication::equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {7, 8, 9, 10});
        submitted = submitted && list.set_predication({1, 8, d3d12_predication::equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {99, 99, 99, 99});
        submitted = submitted && list.set_predication({1, 0, d3d12_predication::not_equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {100, 100, 100, 100});
        submitted = submitted && list.set_predication({0, 0, 0});
        const Rect right{1, 0, 2, 1};
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {float(i), 31, 73, 127}, {&right, 1});
        submitted = submitted && list.close();
        const CommandList* pointer = &list;
        if (!submitted || queue.execute({&pointer, 1}) != ReplayResult::success) { submitted = false; break; }
    }
    // Always release the gate, including on failure, so destructors cannot wait forever on queued work.
    EXPECT(gate.signal(1));
    EXPECT(queue.idle());
    EXPECT(submitted);
    // Completed command-buffer slots must release their old arenas before being reused.
    for (auto& list : lists) {
        const CommandList* pointer = &list;
        EXPECT(queue.execute({&pointer, 1}) == ReplayResult::success);
    }
    EXPECT(queue.idle());
    // Failed submissions must not commit their speculative layout transitions.
    EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CommandList failed;
    EXPECT(failed.transition({2, 0, d3d12_state::render_target, d3d12_state::copy_source}) && failed.close());
    const CommandList* failed_pointer = &failed;
    const auto real_submit = context.vk.vkQueueSubmit;
    context.vk.vkQueueSubmit = +[](VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) -> VkResult {
        return VK_ERROR_OUT_OF_HOST_MEMORY; // no GPU work and no device loss
    };
    const auto failed_result = queue.execute({&failed_pointer, 1});
    context.vk.vkQueueSubmit = real_submit;
    EXPECT(failed_result == ReplayResult::unsupported);
    EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Preserve main's readback-stage collection on every logical queue kind, using real transfer readbacks.
    ResourceBinding output{}; output.buffer = readback; output.size = count * 8; output.host_readable = true;
    context.add_resource(3, output);
    CommandList copy, capture_only, empty;
    EXPECT(copy.copy_buffer({3, 0, 1, 0, 16}) && copy.close());
    EXPECT(capture_only.set_predication({1, 0, d3d12_predication::equal_zero}) && capture_only.close());
    EXPECT(empty.close());
    real_barrier2 = context.vk.vkCmdPipelineBarrier2;
    EXPECT(real_barrier2);
    context.vk.vkCmdPipelineBarrier2 = observe_barrier2;
    for (const auto kind : {VulkanQueue::QueueKind::direct, VulkanQueue::QueueKind::compute, VulkanQueue::QueueKind::copy}) {
        VulkanQueue typed(context, d.queue, d.family, kind);
        EXPECT(typed.ok());
        const CommandList* no_write = &empty;
        host_barriers = 0; host_source_stages = 0;
        EXPECT(typed.execute({&no_write, 1}) == ReplayResult::success);
        EXPECT(host_barriers == 0);
        const CommandList* capture = &capture_only;
        EXPECT(typed.execute({&capture, 1}) == (kind == VulkanQueue::QueueKind::copy
                                                ? ReplayResult::unsupported : ReplayResult::success));
        EXPECT(host_barriers == 0);
        const CommandList* transfer = &copy;
        EXPECT(typed.execute({&transfer, 1}) == ReplayResult::success);
        EXPECT(typed.idle());
        EXPECT(host_barriers == 1 && host_source_stages != 0);
        EXPECT((host_source_stages & ~VulkanQueue::queue_supported_stages(kind)) == 0);
        EXPECT((host_source_stages & ~(VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT)) == 0);
        EXPECT(std::memcmp(pixels, values, sizeof(values)) == 0);
    }
    context.vk.vkCmdPipelineBarrier2 = real_barrier2;
    EXPECT(d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, count};
        context.vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, count};
        region.imageExtent = {2, 1, 1};
        context.vk.vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &region);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        context.vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                      0, 1, &host, 0, nullptr, 0, nullptr);
        return true;
    }));
    const auto* bytes = static_cast<const unsigned char*>(pixels);
    for (unsigned i = 0; i < count; ++i) {
        EXPECT(bytes[i * 8] == 7 && bytes[i * 8 + 1] == 8 && bytes[i * 8 + 2] == 9 && bytes[i * 8 + 3] == 10);
        EXPECT(bytes[i * 8 + 4] == i && bytes[i * 8 + 5] == 31 && bytes[i * 8 + 6] == 73 && bytes[i * 8 + 7] == 127);
    }
    EXPECT(d.validation_errors() == 0);
    std::puts("predicate capture: 32 gated + 32 reused submissions, 192 snapshots, offset rebinding and reset passed");
    std::puts("predicate capture integration: failed-submit tracker rollback and direct/compute/copy readback stages passed");
}
