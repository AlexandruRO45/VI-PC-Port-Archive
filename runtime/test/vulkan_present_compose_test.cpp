// Unit tests for PresentX compute plane composition (runtime/src/vulkan_present_compose.h). CPU-only, no Vulkan
// instance and no GPU: exercises ready() missing-proc and sticky failure caching, ensure_target() creation, reuse,
// resize recreation and failure rollback, target_binding(), record() slot cycling / dispatch sizing, and teardown.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_present_compose.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::PresentComposePush;
using leonida::PresentComposer;
using leonida::VulkanContext;

namespace {

struct MockVk {
    const char* fail_fn = nullptr;

    int create_set_layout = 0, destroy_set_layout = 0;
    int create_pool = 0, destroy_pool = 0;
    int alloc_sets = 0;
    int create_pipeline_layout = 0, destroy_pipeline_layout = 0;
    int create_shader = 0, destroy_shader = 0;
    int create_pipeline = 0, destroy_pipeline = 0;
    int create_image = 0, destroy_image = 0;
    int alloc_memory = 0, free_memory = 0, bind_memory = 0;
    int create_view = 0, destroy_view = 0;

    std::uintptr_t next_image = 0x300;
    std::uintptr_t next_view = 0x400;
    std::vector<VkDescriptorSet> bound_sets;
    std::vector<std::string> ops;
    std::vector<std::pair<VkImageLayout, VkImageLayout>> image_transitions;
    std::uint32_t last_dispatch[3] = {};
    PresentComposePush last_push{};
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
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pool(
        VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    if (should_fail("vkCreateDescriptorPool")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pool;
    *out = make_handle<VkDescriptorPool>(0x102);
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
        out[i] = make_handle<VkDescriptorSet>(0x500 + i);
    }
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_update_sets(
        VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*) {}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipeline_layout(
        VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    if (should_fail("vkCreatePipelineLayout")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_pipeline_layout;
    *out = make_handle<VkPipelineLayout>(0x103);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline_layout(VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pipeline_layout;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader(
        VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
    if (should_fail("vkCreateShaderModule")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_shader;
    *out = make_handle<VkShaderModule>(0x104);
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
    *out = make_handle<VkPipeline>(0x105);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_pipeline;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_image(
        VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage* out) {
    if (should_fail("vkCreateImage")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->create_image;
    *out = make_handle<VkImage>(g_mock->next_image++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_image(VkDevice, VkImage h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_image;
}
VKAPI_ATTR void VKAPI_CALL mock_get_image_mem_reqs(VkDevice, VkImage, VkMemoryRequirements* out) {
    out->size = 4096;
    out->alignment = 64;
    out->memoryTypeBits = 0b11u;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_alloc_memory(
        VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    if (should_fail("vkAllocateMemory")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->alloc_memory;
    *out = make_handle<VkDeviceMemory>(0x201);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_free_memory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->free_memory;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_bind_image_memory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    if (should_fail("vkBindImageMemory")) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++g_mock->bind_memory;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL mock_create_view(
        VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView* out) {
    if (should_fail("vkCreateImageView")) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++g_mock->create_view;
    *out = make_handle<VkImageView>(g_mock->next_view++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_view(VkDevice, VkImageView h, const VkAllocationCallbacks*) {
    if (h) ++g_mock->destroy_view;
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_barrier(
        VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
        std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
        std::uint32_t img_count, const VkImageMemoryBarrier* imgs) {
    g_mock->ops.push_back("barrier");
    if (img_count > 0 && imgs) {
        g_mock->image_transitions.emplace_back(imgs[0].oldLayout, imgs[0].newLayout);
    }
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {
    g_mock->ops.push_back("bind_pipeline");
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_sets(
        VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t count,
        const VkDescriptorSet* sets, std::uint32_t, const std::uint32_t*) {
    g_mock->ops.push_back("bind_sets");
    if (count > 0 && sets) g_mock->bound_sets.push_back(sets[0]);
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_push(
        VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t size, const void* values) {
    g_mock->ops.push_back("push");
    if (size == sizeof(PresentComposePush)) std::memcpy(&g_mock->last_push, values, size);
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    g_mock->ops.push_back("dispatch");
    g_mock->last_dispatch[0] = x;
    g_mock->last_dispatch[1] = y;
    g_mock->last_dispatch[2] = z;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_proc(VkDevice, const char* name) {
    struct Entry { const char* name; PFN_vkVoidFunction fn; };
    static const Entry table[] = {
        {"vkCreateDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_set_layout)},
        {"vkDestroyDescriptorSetLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_set_layout)},
        {"vkCreateDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pool)},
        {"vkDestroyDescriptorPool", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pool)},
        {"vkAllocateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_sets)},
        {"vkUpdateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(mock_update_sets)},
        {"vkCreatePipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipeline_layout)},
        {"vkDestroyPipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline_layout)},
        {"vkCreateShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader)},
        {"vkDestroyShaderModule", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader)},
        {"vkCreateComputePipelines", reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipelines)},
        {"vkDestroyPipeline", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline)},
        {"vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(mock_create_image)},
        {"vkDestroyImage", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_image)},
        {"vkGetImageMemoryRequirements", reinterpret_cast<PFN_vkVoidFunction>(mock_get_image_mem_reqs)},
        {"vkAllocateMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_alloc_memory)},
        {"vkFreeMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_free_memory)},
        {"vkBindImageMemory", reinterpret_cast<PFN_vkVoidFunction>(mock_bind_image_memory)},
        {"vkCreateImageView", reinterpret_cast<PFN_vkVoidFunction>(mock_create_view)},
        {"vkDestroyImageView", reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_view)},
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

VkPhysicalDeviceMemoryProperties make_mem(bool device_local) {
    VkPhysicalDeviceMemoryProperties props{};
    props.memoryTypeCount = 2;
    props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    props.memoryTypes[1].propertyFlags = device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                      : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return props;
}

int test_ready_missing_proc_and_failure_sticky() {
    {
        MockVk mock;
        g_mock = &mock;
        VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(0x1), false);
        ctx.vk.vkCmdDispatch = nullptr;
        PresentComposer composer(ctx, make_mem(true));
        EXPECT(!composer.ready());
        EXPECT(mock.create_set_layout == 0);
        // Sticky failure: even if the pointer is restored, ready() stays false.
        ctx.vk.vkCmdDispatch = mock_cmd_dispatch;
        EXPECT(!composer.ready());
        EXPECT(mock.create_set_layout == 0);
    }
    for (const char* fail : {"vkCreateDescriptorSetLayout", "vkCreateDescriptorPool", "vkAllocateDescriptorSets",
                             "vkCreatePipelineLayout", "vkCreateShaderModule", "vkCreateComputePipelines"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        {
            VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(0x1), false);
            PresentComposer composer(ctx, make_mem(true));
            EXPECT(!composer.ready());
            EXPECT(!composer.ready()); // Does not retry after failure
        }
        EXPECT(mock.create_set_layout == mock.destroy_set_layout);
        EXPECT(mock.create_pool == mock.destroy_pool);
        EXPECT(mock.create_pipeline_layout == mock.destroy_pipeline_layout);
        EXPECT(mock.create_shader == mock.destroy_shader);
        EXPECT(mock.create_pipeline == mock.destroy_pipeline);
    }
    return 0;
}

int test_ensure_target_failures() {
    for (const char* fail : {"vkCreateImage", "vkAllocateMemory", "vkBindImageMemory", "vkCreateImageView"}) {
        MockVk mock;
        mock.fail_fn = fail;
        g_mock = &mock;
        VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(0x1), false);
        PresentComposer composer(ctx, make_mem(true));
        bool changed = false;
        EXPECT(!composer.ensure_target({1280, 720}, changed));
        EXPECT(changed);
        EXPECT(composer.target() == VK_NULL_HANDLE);
        EXPECT(mock.create_image == mock.destroy_image);
        EXPECT(mock.alloc_memory == mock.free_memory);
        EXPECT(mock.create_view == mock.destroy_view);
    }
    {
        MockVk mock;
        g_mock = &mock;
        VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(0x1), false);
        // ensure_target requires VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.
        PresentComposer composer(ctx, make_mem(false));
        bool changed = false;
        EXPECT(!composer.ensure_target({640, 360}, changed));
        EXPECT(mock.destroy_image == 1 && mock.alloc_memory == 0);
    }
    return 0;
}

int test_ready_resize_record_and_teardown() {
    MockVk mock;
    g_mock = &mock;
    {
        VulkanContext ctx(mock_get_proc, make_handle<VkDevice>(0x1), false);
        PresentComposer composer(ctx, make_mem(true));
        EXPECT(composer.ready());
        EXPECT(composer.ready()); // Cached ready state does not recreate pipeline
        EXPECT(mock.create_pipeline == 1);

        bool changed = false;
        EXPECT(composer.ensure_target({19, 10}, changed));
        EXPECT(changed);
        const VkImage first = composer.target();
        EXPECT(first != VK_NULL_HANDLE);

        // Same extent reuses target without recreation.
        EXPECT(composer.ensure_target({19, 10}, changed));
        EXPECT(!changed);
        EXPECT(composer.target() == first);

        // Changed extent destroys previous target and allocates a new one.
        EXPECT(composer.ensure_target({25, 17}, changed));
        EXPECT(changed);
        EXPECT(composer.target() != first);
        EXPECT(mock.destroy_image == 1 && mock.destroy_view == 1 && mock.free_memory == 1);

        const auto binding = composer.target_binding();
        EXPECT(binding.image == composer.target() && binding.format == PresentComposer::target_format &&
               binding.aspects == VK_IMAGE_ASPECT_COLOR_BIT && binding.width == 25 && binding.height == 17);

        PresentComposePush push{};
        push.width = 25;
        push.height = 17;
        push.plane_count = 2;
        composer.record(make_handle<VkCommandBuffer>(0x601), 0, make_handle<VkImageView>(0x701),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, make_handle<VkImageView>(0x702),
                        VK_IMAGE_LAYOUT_GENERAL, push);
        composer.record(make_handle<VkCommandBuffer>(0x601), 3, make_handle<VkImageView>(0x701),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, make_handle<VkImageView>(0x701),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, push);

        const std::vector<std::string> expected_ops = {
            "barrier", "bind_pipeline", "bind_sets", "push", "dispatch", "barrier",
            "barrier", "bind_pipeline", "bind_sets", "push", "dispatch", "barrier",
        };
        EXPECT(mock.ops == expected_ops);
        EXPECT(mock.bound_sets.size() == 2 && mock.bound_sets[0] != mock.bound_sets[1]); // slot 0 vs 3 % 2 == 1
        EXPECT(mock.image_transitions[0].first == VK_IMAGE_LAYOUT_UNDEFINED &&
               mock.image_transitions[0].second == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(mock.image_transitions[1].first == VK_IMAGE_LAYOUT_GENERAL &&
               mock.image_transitions[1].second == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        // (25 + 7) / 8 == 4, (17 + 7) / 8 == 3
        EXPECT(mock.last_dispatch[0] == 4 && mock.last_dispatch[1] == 3 && mock.last_dispatch[2] == 1);
        EXPECT(mock.last_push.width == 25 && mock.last_push.plane_count == 2);
    }
    EXPECT(mock.destroy_view == 2 && mock.destroy_image == 2 && mock.free_memory == 2);
    EXPECT(mock.destroy_pipeline == 1 && mock.destroy_pipeline_layout == 1 &&
           mock.destroy_pool == 1 && mock.destroy_set_layout == 1);
    return 0;
}

} // namespace

int main() {
    if (test_ready_missing_proc_and_failure_sticky()) return 1;
    if (test_ensure_target_failures()) return 1;
    if (test_ready_resize_record_and_teardown()) return 1;
    std::puts("vulkan_present_compose_test: ok");
    return 0;
}
