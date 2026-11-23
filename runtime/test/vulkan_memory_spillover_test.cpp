// CPU-only unit tests for 8 GiB VRAM spillover and 214 MiB BAR1 avoidance across internal Vulkan helpers:
//   - BindlessSet (runtime/src/vulkan_bindless.h): fallback images, device-local table + flush replacement, staging
//   - OcclusionResolve::initialize (runtime/src/vulkan_backend.h)
//   - PredicateReducer::batch (runtime/src/vulkan_predicate.h)
//   - IndirectDispatchCounter::Batch::prepare (runtime/src/vulkan_indirect_dispatch_count.h)
//   - allocate_memory host-staging BAR1 avoidance + VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT pNext preservation
// Uses synthetic Vulkan device procs; no Vulkan loader, no GPU, and no game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_backend.h"
#include "../src/vulkan_bindless.h"
#include "../src/vulkan_indirect_dispatch_count.h"
#include "../src/vulkan_memory.h"
#include "../src/vulkan_predicate.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define EXPECT(condition) do {                                             \
    if (!(condition)) {                                                    \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);       \
        return 1;                                                          \
    }                                                                      \
} while (0)

using namespace leonida;

namespace {

template <typename Handle>
Handle make_handle(std::uintptr_t id) {
    return reinterpret_cast<Handle>(id);
}

struct AllocAttempt {
    std::uint32_t memory_type = UINT32_MAX;
    VkDeviceSize size = 0;
    bool has_device_address_flag = false;
    bool succeeded = false;
};

struct FakeVulkan {
    std::uintptr_t next_id = 1;
    std::unordered_map<std::uintptr_t, VkDeviceSize> buffer_sizes;
    std::unordered_map<std::uintptr_t, VkBufferUsageFlags> buffer_usage;
    std::unordered_set<std::uintptr_t> live_buffers;
    std::unordered_set<std::uintptr_t> live_images;
    std::unordered_set<std::uintptr_t> live_views;
    std::unordered_set<std::uintptr_t> live_samplers;
    std::unordered_set<std::uintptr_t> live_memory;
    std::unordered_map<std::uintptr_t, std::uint32_t> memory_type_of;
    std::unordered_map<std::uintptr_t, std::vector<std::uint8_t>> mapped_storage;
    std::unordered_set<std::uintptr_t> live_pools;
    std::unordered_set<std::uintptr_t> live_set_layouts;
    std::unordered_set<std::uintptr_t> live_pipeline_layouts;
    std::unordered_set<std::uintptr_t> live_pipelines;
    std::unordered_set<std::uintptr_t> live_shader_modules;

    std::uint32_t memory_type_bits = 0xFu;
    std::uint32_t fail_types_mask = 0;
    VkDeviceSize override_buffer_req_size = 0;
    VkResult bind_buffer_result = VK_SUCCESS;
    std::vector<AllocAttempt> attempts;

    bool all_freed() const {
        return live_buffers.empty() && live_images.empty() && live_views.empty() && live_samplers.empty() &&
               live_memory.empty() && live_pools.empty() && live_set_layouts.empty() &&
               live_pipeline_layouts.empty() && live_pipelines.empty() && live_shader_modules.empty();
    }
};

FakeVulkan* g_vk = nullptr;
const VkDevice k_device = make_handle<VkDevice>(0x100);

// GTX 1070-like 8 GiB VRAM + 16 GiB RAM + 214 MiB BAR1, with BAR1 deliberately ordered at type 0 ahead of main
// VRAM (type 1) and system RAM (types 2, 3) to verify BAR1 avoidance does not rely on driver type ordering.
VkPhysicalDeviceMemoryProperties make_no_rebar_bar1_first_memory() {
    constexpr VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkPhysicalDeviceMemoryProperties mp{};
    mp.memoryHeapCount = 3;
    mp.memoryHeaps[0] = {8ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};   // 8 GiB VRAM
    mp.memoryHeaps[1] = {16ull << 30, 0};                                // 16 GiB system RAM
    mp.memoryHeaps[2] = {224395264ull, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}; // 214 MiB BAR1
    mp.memoryTypeCount = 4;
    mp.memoryTypes[0] = {local | host, 2};                               // BAR1
    mp.memoryTypes[1] = {local, 0};                                      // main VRAM
    mp.memoryTypes[2] = {host, 1};                                       // uncached sysmem
    mp.memoryTypes[3] = {host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1};  // cached sysmem
    return mp;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_descriptor_set_layout(
    VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_set_layouts.insert(id);
    *out = make_handle<VkDescriptorSetLayout>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_descriptor_set_layout(
    VkDevice, VkDescriptorSetLayout h, const VkAllocationCallbacks*) {
    g_vk->live_set_layouts.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_descriptor_pool(
    VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_pools.insert(id);
    *out = make_handle<VkDescriptorPool>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_descriptor_pool(
    VkDevice, VkDescriptorPool h, const VkAllocationCallbacks*) {
    g_vk->live_pools.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_descriptor_sets(
    VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
        out[i] = make_handle<VkDescriptorSet>(g_vk->next_id++);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_update_descriptor_sets(
    VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*) {}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_pipeline_layout(
    VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_pipeline_layouts.insert(id);
    *out = make_handle<VkPipelineLayout>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_pipeline_layout(
    VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    g_vk->live_pipeline_layouts.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_shader_module(
    VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_shader_modules.insert(id);
    *out = make_handle<VkShaderModule>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_shader_module(
    VkDevice, VkShaderModule h, const VkAllocationCallbacks*) {
    g_vk->live_shader_modules.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_compute_pipelines(
    VkDevice, VkPipelineCache, std::uint32_t count, const VkComputePipelineCreateInfo*,
    const VkAllocationCallbacks*, VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uintptr_t id = g_vk->next_id++;
        g_vk->live_pipelines.insert(id);
        out[i] = make_handle<VkPipeline>(id);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    g_vk->live_pipelines.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_buffer(
    VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_buffers.insert(id);
    g_vk->buffer_sizes[id] = info->size;
    g_vk->buffer_usage[id] = info->usage;
    *out = make_handle<VkBuffer>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_buffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) {
    const auto id = reinterpret_cast<std::uintptr_t>(h);
    g_vk->live_buffers.erase(id);
    g_vk->buffer_sizes.erase(id);
    g_vk->buffer_usage.erase(id);
}

VKAPI_ATTR void VKAPI_CALL fake_get_buffer_memory_requirements(
    VkDevice, VkBuffer h, VkMemoryRequirements* reqs) {
    const auto it = g_vk->buffer_sizes.find(reinterpret_cast<std::uintptr_t>(h));
    const VkDeviceSize base = it != g_vk->buffer_sizes.end() ? it->second : 256;
    reqs->size = g_vk->override_buffer_req_size ? g_vk->override_buffer_req_size : base;
    reqs->alignment = 256;
    reqs->memoryTypeBits = g_vk->memory_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_bind_buffer_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g_vk->bind_buffer_result;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_image(
    VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_images.insert(id);
    *out = make_handle<VkImage>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_image(VkDevice, VkImage h, const VkAllocationCallbacks*) {
    g_vk->live_images.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR void VKAPI_CALL fake_get_image_memory_requirements(
    VkDevice, VkImage, VkMemoryRequirements* reqs) {
    reqs->size = 4096;
    reqs->alignment = 4096;
    reqs->memoryTypeBits = g_vk->memory_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_bind_image_memory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_image_view(
    VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_views.insert(id);
    *out = make_handle<VkImageView>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_image_view(VkDevice, VkImageView h, const VkAllocationCallbacks*) {
    g_vk->live_views.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_sampler(
    VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*, VkSampler* out) {
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_samplers.insert(id);
    *out = make_handle<VkSampler>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_sampler(VkDevice, VkSampler h, const VkAllocationCallbacks*) {
    g_vk->live_samplers.erase(reinterpret_cast<std::uintptr_t>(h));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_memory(
    VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    bool has_bda = false;
    for (const auto* node = static_cast<const VkBaseInStructure*>(info->pNext); node; node = node->pNext) {
        if (node->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) {
            const auto* flags_info = reinterpret_cast<const VkMemoryAllocateFlagsInfo*>(node);
            has_bda = (flags_info->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) != 0;
        }
    }
    const bool fail = (g_vk->fail_types_mask & (1u << info->memoryTypeIndex)) != 0;
    g_vk->attempts.push_back({info->memoryTypeIndex, info->allocationSize, has_bda, !fail});
    if (fail) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const std::uintptr_t id = g_vk->next_id++;
    g_vk->live_memory.insert(id);
    g_vk->memory_type_of[id] = info->memoryTypeIndex;
    const std::size_t backing = std::min<std::size_t>(std::size_t(info->allocationSize), 16u << 20);
    g_vk->mapped_storage[id].assign(backing, 0);
    *out = make_handle<VkDeviceMemory>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_free_memory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    const auto id = reinterpret_cast<std::uintptr_t>(h);
    g_vk->live_memory.erase(id);
    g_vk->memory_type_of.erase(id);
    g_vk->mapped_storage.erase(id);
}

VKAPI_ATTR VkResult VKAPI_CALL fake_map_memory(
    VkDevice, VkDeviceMemory h, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void** out) {
    auto& storage = g_vk->mapped_storage.at(reinterpret_cast<std::uintptr_t>(h));
    *out = storage.data() + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL fake_get_buffer_device_address(
    VkDevice, const VkBufferDeviceAddressInfo* info) {
    return VkDeviceAddress(0x10000000ull + reinterpret_cast<std::uintptr_t>(info->buffer) * 0x1000ull);
}

VKAPI_ATTR void VKAPI_CALL fake_cmd_pipeline_barrier(
    VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t,
    const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
    const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_copy_buffer(
    VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_update_buffer(
    VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_clear_color_image(
    VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
    const VkImageSubresourceRange*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_bind_descriptor_sets(
    VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
    const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_push_constants(
    VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_device_proc(VkDevice, const char* name) {
#define PROC(fn_name, impl) if (std::strcmp(name, #fn_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(impl)
    PROC(vkCreateDescriptorSetLayout, fake_create_descriptor_set_layout);
    PROC(vkDestroyDescriptorSetLayout, fake_destroy_descriptor_set_layout);
    PROC(vkCreateDescriptorPool, fake_create_descriptor_pool);
    PROC(vkDestroyDescriptorPool, fake_destroy_descriptor_pool);
    PROC(vkAllocateDescriptorSets, fake_allocate_descriptor_sets);
    PROC(vkUpdateDescriptorSets, fake_update_descriptor_sets);
    PROC(vkCreatePipelineLayout, fake_create_pipeline_layout);
    PROC(vkDestroyPipelineLayout, fake_destroy_pipeline_layout);
    PROC(vkCreateShaderModule, fake_create_shader_module);
    PROC(vkDestroyShaderModule, fake_destroy_shader_module);
    PROC(vkCreateComputePipelines, fake_create_compute_pipelines);
    PROC(vkDestroyPipeline, fake_destroy_pipeline);
    PROC(vkCreateBuffer, fake_create_buffer);
    PROC(vkDestroyBuffer, fake_destroy_buffer);
    PROC(vkGetBufferMemoryRequirements, fake_get_buffer_memory_requirements);
    PROC(vkBindBufferMemory, fake_bind_buffer_memory);
    PROC(vkCreateImage, fake_create_image);
    PROC(vkDestroyImage, fake_destroy_image);
    PROC(vkGetImageMemoryRequirements, fake_get_image_memory_requirements);
    PROC(vkBindImageMemory, fake_bind_image_memory);
    PROC(vkCreateImageView, fake_create_image_view);
    PROC(vkDestroyImageView, fake_destroy_image_view);
    PROC(vkCreateSampler, fake_create_sampler);
    PROC(vkDestroySampler, fake_destroy_sampler);
    PROC(vkAllocateMemory, fake_allocate_memory);
    PROC(vkFreeMemory, fake_free_memory);
    PROC(vkMapMemory, fake_map_memory);
    PROC(vkGetBufferDeviceAddress, fake_get_buffer_device_address);
    PROC(vkCmdPipelineBarrier, fake_cmd_pipeline_barrier);
    PROC(vkCmdCopyBuffer, fake_cmd_copy_buffer);
    PROC(vkCmdUpdateBuffer, fake_cmd_update_buffer);
    PROC(vkCmdClearColorImage, fake_cmd_clear_color_image);
    PROC(vkCmdBindPipeline, fake_cmd_bind_pipeline);
    PROC(vkCmdBindDescriptorSets, fake_cmd_bind_descriptor_sets);
    PROC(vkCmdPushConstants, fake_cmd_push_constants);
    PROC(vkCmdDispatch, fake_cmd_dispatch);
#undef PROC
    return nullptr;
}

VulkanDeviceFunctions make_occlusion_vk() {
    VulkanDeviceFunctions vk{};
    vk.vkCreateDescriptorSetLayout = fake_create_descriptor_set_layout;
    vk.vkDestroyDescriptorSetLayout = fake_destroy_descriptor_set_layout;
    vk.vkCreateDescriptorPool = fake_create_descriptor_pool;
    vk.vkDestroyDescriptorPool = fake_destroy_descriptor_pool;
    vk.vkAllocateDescriptorSets = fake_allocate_descriptor_sets;
    vk.vkUpdateDescriptorSets = fake_update_descriptor_sets;
    vk.vkCreatePipelineLayout = fake_create_pipeline_layout;
    vk.vkDestroyPipelineLayout = fake_destroy_pipeline_layout;
    vk.vkCreateShaderModule = fake_create_shader_module;
    vk.vkDestroyShaderModule = fake_destroy_shader_module;
    vk.vkCreateComputePipelines = fake_create_compute_pipelines;
    vk.vkDestroyPipeline = fake_destroy_pipeline;
    vk.vkCreateBuffer = fake_create_buffer;
    vk.vkDestroyBuffer = fake_destroy_buffer;
    vk.vkGetBufferMemoryRequirements = fake_get_buffer_memory_requirements;
    vk.vkBindBufferMemory = fake_bind_buffer_memory;
    vk.vkAllocateMemory = fake_allocate_memory;
    vk.vkFreeMemory = fake_free_memory;
    vk.vkCmdBindPipeline = fake_cmd_bind_pipeline;
    vk.vkCmdBindDescriptorSets = fake_cmd_bind_descriptor_sets;
    vk.vkCmdPushConstants = fake_cmd_push_constants;
    vk.vkCmdDispatch = fake_cmd_dispatch;
    return vk;
}

int test_bindless_spillover_and_bar1_avoidance() {
    const auto mp = make_no_rebar_bar1_first_memory();

    // 1. Normal path: fallbacks + table use main 8 GiB VRAM (type 1), staging uses 16 GiB sysmem (type 2),
    //    and 214 MiB BAR1 (type 0) is never touched even though it is type index 0.
    {
        FakeVulkan vk;
        g_vk = &vk;
        {
            BindlessSet set(fake_get_device_proc, k_device, mp);
            EXPECT(set.ok());
            EXPECT(set.table_in_device_local_memory());
            // 9 single-sample fallback images + 1 table buffer + 1 staging buffer = 11 allocations.
            EXPECT(vk.attempts.size() == 11);
            for (std::size_t i = 0; i < 10; ++i)
                EXPECT(vk.attempts[i].memory_type == 1 && vk.attempts[i].succeeded);
            EXPECT(vk.attempts[10].memory_type == 2 && vk.attempts[10].succeeded);
            for (const auto& a : vk.attempts) EXPECT(a.memory_type != 0);
        }
        EXPECT(vk.all_freed());
    }

    // 2. VRAM exhaustion (both 8 GiB VRAM and 214 MiB BAR1 return VK_ERROR_OUT_OF_DEVICE_MEMORY):
    //    fallback images and table buffer spill cleanly to sysmem (type 2), staging goes straight to sysmem
    //    without trying BAR1, and flush() table replacement/growth also spills cleanly.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 0) | (1u << 1);
        {
            BindlessSet set(fake_get_device_proc, k_device, mp);
            EXPECT(set.ok());
            EXPECT(!set.table_in_device_local_memory());
            // Last attempt is the staging buffer: straight to type 2 (never tried type 0 BAR1).
            EXPECT(!vk.attempts.empty());
            EXPECT(vk.attempts.back().memory_type == 2 && vk.attempts.back().succeeded);
            // The preceding 3 attempts are the table buffer: type 1 (OOM) -> type 0 (OOM) -> type 2 (OK).
            const std::size_t n = vk.attempts.size();
            EXPECT(n >= 4);
            EXPECT(vk.attempts[n - 4].memory_type == 1 && !vk.attempts[n - 4].succeeded);
            EXPECT(vk.attempts[n - 3].memory_type == 0 && !vk.attempts[n - 3].succeeded);
            EXPECT(vk.attempts[n - 2].memory_type == 2 && vk.attempts[n - 2].succeeded);

            // First flush uploads the initial table; then grow the table to force a replacement generation.
            const VkCommandBuffer cb = make_handle<VkCommandBuffer>(0x200);
            EXPECT(set.flush(cb, 1));
            for (std::uint32_t k = 1; k <= 80; ++k) {
                const DescriptorKey key{k, k * 2654435761u, k ^ 0x55555555u, 1, 2, 3, 4, 5};
                EXPECT(set.insert_key(key, 1));
            }
            vk.attempts.clear();
            EXPECT(set.flush(cb, 2));
            EXPECT(set.last_flush_replaced());
            EXPECT(!set.table_in_device_local_memory());
            // New generation: table (1 OOM -> 0 OOM -> 2 OK) + staging (2 OK directly).
            EXPECT(vk.attempts.size() == 4);
            EXPECT(vk.attempts[0].memory_type == 1 && !vk.attempts[0].succeeded);
            EXPECT(vk.attempts[1].memory_type == 0 && !vk.attempts[1].succeeded);
            EXPECT(vk.attempts[2].memory_type == 2 && vk.attempts[2].succeeded);
            EXPECT(vk.attempts[3].memory_type == 2 && vk.attempts[3].succeeded);
            set.release_retired();
            EXPECT(!set.has_retired());
        }
        EXPECT(vk.all_freed());
    }

    // 3. Total OOM fails cleanly with zero leaked handles.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = 0xFu;
        {
            BindlessSet set(fake_get_device_proc, k_device, mp);
            EXPECT(!set.ok());
        }
        EXPECT(vk.all_freed());
    }
    return 0;
}

int test_occlusion_resolve_spillover() {
    const auto mp = make_no_rebar_bar1_first_memory();
    const VulkanDeviceFunctions fns = make_occlusion_vk();

    // 1. Normal path picks main 8 GiB VRAM (type 1), not BAR1 (type 0).
    {
        FakeVulkan vk;
        g_vk = &vk;
        {
            OcclusionResolve helper(fns, k_device);
            EXPECT(helper.initialize(mp));
            EXPECT(vk.attempts.size() == 1);
            EXPECT(vk.attempts[0].memory_type == 1 && vk.attempts[0].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 2. VRAM exhaustion spills cleanly: type 1 (OOM) -> type 0 (OOM) -> type 2 (OK).
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 0) | (1u << 1);
        {
            OcclusionResolve helper(fns, k_device);
            EXPECT(helper.initialize(mp));
            EXPECT(vk.attempts.size() == 3);
            EXPECT(vk.attempts[0].memory_type == 1 && !vk.attempts[0].succeeded);
            EXPECT(vk.attempts[1].memory_type == 0 && !vk.attempts[1].succeeded);
            EXPECT(vk.attempts[2].memory_type == 2 && vk.attempts[2].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 3. Requirement larger than 214 MiB BAR1 skips BAR1 completely without attempting vkAllocateMemory on it.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 1);  // main VRAM OOM; BAR1 would succeed if wrongly tried
        vk.override_buffer_req_size = VkDeviceSize(256) << 20;
        {
            OcclusionResolve helper(fns, k_device);
            EXPECT(helper.initialize(mp));
            EXPECT(vk.attempts.size() == 2);
            EXPECT(vk.attempts[0].memory_type == 1 && !vk.attempts[0].succeeded);
            EXPECT(vk.attempts[1].memory_type == 2 && vk.attempts[1].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 4. Total OOM fails cleanly with zero leaks.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = 0xFu;
        {
            OcclusionResolve helper(fns, k_device);
            EXPECT(!helper.initialize(mp));
        }
        EXPECT(vk.all_freed());
    }
    return 0;
}

int test_predicate_reducer_spillover() {
    const auto mp = make_no_rebar_bar1_first_memory();

    // 1. Normal path picks main 8 GiB VRAM (type 1).
    {
        FakeVulkan vk;
        g_vk = &vk;
        {
            PredicateReducer reducer(fake_get_device_proc, k_device, mp, 65536);
            auto batch = reducer.batch(4);
            EXPECT(batch != nullptr);
            EXPECT(vk.attempts.size() == 1);
            EXPECT(vk.attempts[0].memory_type == 1 && vk.attempts[0].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 2. VRAM exhaustion spills cleanly to sysmem (type 2).
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 0) | (1u << 1);
        {
            PredicateReducer reducer(fake_get_device_proc, k_device, mp, 65536);
            auto batch = reducer.batch(4);
            EXPECT(batch != nullptr);
            EXPECT(vk.attempts.size() == 3);
            EXPECT(vk.attempts[0].memory_type == 1 && !vk.attempts[0].succeeded);
            EXPECT(vk.attempts[1].memory_type == 0 && !vk.attempts[1].succeeded);
            EXPECT(vk.attempts[2].memory_type == 2 && vk.attempts[2].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 3. Total OOM returns null with zero leaks.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = 0xFu;
        {
            PredicateReducer reducer(fake_get_device_proc, k_device, mp, 65536);
            auto batch = reducer.batch(4);
            EXPECT(batch == nullptr);
        }
        EXPECT(vk.all_freed());
    }
    return 0;
}

int test_indirect_dispatch_counter_spillover() {
    const auto mp = make_no_rebar_bar1_first_memory();
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 65535;
    const VkBuffer args = make_handle<VkBuffer>(0x301);
    const VkBuffer count = make_handle<VkBuffer>(0x302);

    // 1. Normal path picks main 8 GiB VRAM (type 1) with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT.
    {
        FakeVulkan vk;
        g_vk = &vk;
        {
            IndirectDispatchCounter counter(fake_get_device_proc, k_device, mp, limits);
            auto batch = counter.batch();
            CountedDispatch out{};
            EXPECT(batch->prepare(args, 0, count, 0, 12, 8, out));
            EXPECT(out.indirect != VK_NULL_HANDLE && out.output != 0);
            EXPECT(vk.attempts.size() == 1);
            EXPECT(vk.attempts[0].memory_type == 1 && vk.attempts[0].has_device_address_flag &&
                   vk.attempts[0].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 2. VRAM exhaustion spills cleanly to sysmem while preserving VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT on every try.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 0) | (1u << 1);
        {
            IndirectDispatchCounter counter(fake_get_device_proc, k_device, mp, limits);
            auto batch = counter.batch();
            CountedDispatch out{};
            EXPECT(batch->prepare(args, 0, count, 0, 12, 8, out));
            EXPECT(out.indirect != VK_NULL_HANDLE && out.output != 0);
            EXPECT(vk.attempts.size() == 3);
            EXPECT(vk.attempts[0].memory_type == 1 && vk.attempts[0].has_device_address_flag &&
                   !vk.attempts[0].succeeded);
            EXPECT(vk.attempts[1].memory_type == 0 && vk.attempts[1].has_device_address_flag &&
                   !vk.attempts[1].succeeded);
            EXPECT(vk.attempts[2].memory_type == 2 && vk.attempts[2].has_device_address_flag &&
                   vk.attempts[2].succeeded);
        }
        EXPECT(vk.all_freed());
    }

    // 3. Total OOM fails cleanly and frees scratch buffers with the batch.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = 0xFu;
        {
            IndirectDispatchCounter counter(fake_get_device_proc, k_device, mp, limits);
            auto batch = counter.batch();
            CountedDispatch out{};
            EXPECT(!batch->prepare(args, 0, count, 0, 12, 8, out));
        }
        EXPECT(vk.all_freed());
    }
    return 0;
}

int test_host_staging_bar1_avoidance_with_device_address() {
    // Mirrors VulkanQueue::stage_khr_instances: HOST_VISIBLE | HOST_COHERENT + DEVICE_ADDRESS_BIT.
    const auto mp = make_no_rebar_bar1_first_memory();
    constexpr VkMemoryPropertyFlags host_flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr,
                                          VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    const VkMemoryRequirements reqs{4096, 256, 0xFu};

    // 1. Normal path lands in sysmem (type 2), never touching 214 MiB BAR1 (type 0).
    {
        FakeVulkan vk;
        g_vk = &vk;
        std::uint32_t chosen = UINT32_MAX;
        const VkDeviceMemory mem = allocate_memory(fake_allocate_memory, k_device, mp, reqs, host_flags, 0, &flags,
                                                   nullptr, &chosen);
        EXPECT(mem != VK_NULL_HANDLE && chosen == 2);
        EXPECT(vk.attempts.size() == 1);
        EXPECT(vk.attempts[0].memory_type == 2 && vk.attempts[0].has_device_address_flag);
        fake_free_memory(k_device, mem, nullptr);
        EXPECT(vk.all_freed());
    }

    // 2. If all sysmem host types fail, BAR1 (type 0) remains available as last-resort fallback.
    {
        FakeVulkan vk;
        g_vk = &vk;
        vk.fail_types_mask = (1u << 2) | (1u << 3);
        std::uint32_t chosen = UINT32_MAX;
        const VkDeviceMemory mem = allocate_memory(fake_allocate_memory, k_device, mp, reqs, host_flags, 0, &flags,
                                                   nullptr, &chosen);
        EXPECT(mem != VK_NULL_HANDLE && chosen == 0);
        EXPECT(vk.attempts.size() == 3);
        EXPECT(vk.attempts[0].memory_type == 2 && !vk.attempts[0].succeeded);
        EXPECT(vk.attempts[1].memory_type == 3 && !vk.attempts[1].succeeded);
        EXPECT(vk.attempts[2].memory_type == 0 && vk.attempts[2].has_device_address_flag &&
               vk.attempts[2].succeeded);
        fake_free_memory(k_device, mem, nullptr);
        EXPECT(vk.all_freed());
    }
    return 0;
}

}  // namespace

int main() {
    if (test_bindless_spillover_and_bar1_avoidance() != 0) return 1;
    if (test_occlusion_resolve_spillover() != 0) return 1;
    if (test_predicate_reducer_spillover() != 0) return 1;
    if (test_indirect_dispatch_counter_spillover() != 0) return 1;
    if (test_host_staging_bar1_avoidance_with_device_address() != 0) return 1;
    std::printf("vulkan_memory_spillover: BindlessSet, OcclusionResolve, PredicateReducer, "
                "IndirectDispatchCounter and BAR1 staging avoidance OK\n");
    return 0;
}
