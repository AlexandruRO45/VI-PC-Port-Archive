// CPU-only: a host allocation failure while replacing a bindless table must leave its old generation usable and
// destroy every temporary Vulkan object once. Synthetic handles and device procedures; no Vulkan loader or GPU.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_bindless.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static bool fail_new = false;
static bool arm_after_set_alloc = false;
void* operator new(std::size_t n) {
    if (fail_new) {
        fail_new = false;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

static std::uintptr_t next_handle = 100;
template <class T> T handle() { return (T)(++next_handle); }

struct Tracked {
    std::array<std::uintptr_t, 64> live{};
    int created = 0, destroyed = 0, invalid_destroy = 0;

    void add(std::uintptr_t id) {
        if (created >= int(live.size()))
            std::abort();
        live[created++] = id;
    }
    void remove(std::uintptr_t id) {
        for (auto& item : live) {
            if (item == id) {
                item = 0;
                ++destroyed;
                return;
            }
        }
        ++invalid_destroy;
    }
    bool contains(std::uintptr_t id) const {
        for (auto item : live)
            if (item == id)
                return true;
        return false;
    }
    bool balanced() const { return created == destroyed && invalid_destroy == 0; }
};

struct Counts {
    Tracked buffers, memory, pools, images, views, samplers, layouts;
    bool balanced() const {
        return buffers.balanced() && memory.balanced() && pools.balanced() && images.balanced() && views.balanced() &&
               samplers.balanced() && layouts.balanced();
    }
} counts;
static VkResult VKAPI_PTR create_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*,
                                        VkDescriptorSetLayout* out) {
    *out = handle<VkDescriptorSetLayout>();
    counts.layouts.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_layout(VkDevice, VkDescriptorSetLayout layout, const VkAllocationCallbacks*) {
    counts.layouts.remove((std::uintptr_t)layout);
}
static VkResult VKAPI_PTR create_pool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
                                      VkDescriptorPool* out) {
    *out = handle<VkDescriptorPool>();
    counts.pools.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_pool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
    counts.pools.remove((std::uintptr_t)pool);
}
static VkResult VKAPI_PTR alloc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    for (unsigned i = 0; i < info->descriptorSetCount; ++i)
        out[i] = handle<VkDescriptorSet>();
    if (arm_after_set_alloc) {
        arm_after_set_alloc = false;
        fail_new = true;
    }
    return VK_SUCCESS;
}
static void VKAPI_PTR update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                  const VkCopyDescriptorSet*) {}
static VkResult VKAPI_PTR create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*,
                                        VkBuffer* out) {
    *out = handle<VkBuffer>();
    counts.buffers.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    counts.buffers.remove((std::uintptr_t)buffer);
}
static void VKAPI_PTR buffer_requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) { *out = {4096, 256, 1}; }
static VkResult VKAPI_PTR bind_buffer(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
static VkResult VKAPI_PTR alloc_memory(VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
                                       VkDeviceMemory* out) {
    void* p = std::malloc(std::size_t(info->allocationSize));
    if (!p)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = (VkDeviceMemory)p;
    counts.memory.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR free_memory(VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
    const auto id = (std::uintptr_t)mem;
    if (counts.memory.contains(id))
        std::free((void*)mem);
    counts.memory.remove(id);
}
static VkResult VKAPI_PTR map_memory(VkDevice, VkDeviceMemory mem, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                     void** out) {
    *out = (void*)mem;
    return VK_SUCCESS;
}
static VkResult VKAPI_PTR create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage* out) {
    *out = handle<VkImage>();
    counts.images.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_image(VkDevice, VkImage image, const VkAllocationCallbacks*) {
    counts.images.remove((std::uintptr_t)image);
}
static void VKAPI_PTR image_requirements(VkDevice, VkImage, VkMemoryRequirements* out) { *out = {4096, 256, 1}; }
static VkResult VKAPI_PTR bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
static VkResult VKAPI_PTR create_view(VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*,
                                      VkImageView* out) {
    *out = handle<VkImageView>();
    counts.views.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_view(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    counts.views.remove((std::uintptr_t)view);
}
static VkResult VKAPI_PTR create_sampler(VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*,
                                         VkSampler* out) {
    *out = handle<VkSampler>();
    counts.samplers.add((std::uintptr_t)*out);
    return VK_SUCCESS;
}
static void VKAPI_PTR destroy_sampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*) {
    counts.samplers.remove((std::uintptr_t)sampler);
}
static void VKAPI_PTR cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                  std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                  std::uint32_t, const VkImageMemoryBarrier*) {}
static void VKAPI_PTR cmd_copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
static void VKAPI_PTR cmd_update(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
static void VKAPI_PTR cmd_clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                                const VkImageSubresourceRange*) {}
static PFN_vkVoidFunction VKAPI_PTR get_proc(VkDevice, const char* name) {
#define MAP(n, f)                                                                                                      \
    if (std::strcmp(name, "vk" #n) == 0)                                                                               \
    return reinterpret_cast<PFN_vkVoidFunction>(&f)
    MAP(CreateDescriptorSetLayout, create_layout);
    MAP(DestroyDescriptorSetLayout, destroy_layout);
    MAP(CreateDescriptorPool, create_pool);
    MAP(DestroyDescriptorPool, destroy_pool);
    MAP(AllocateDescriptorSets, alloc_sets);
    MAP(UpdateDescriptorSets, update_sets);
    MAP(CreateBuffer, create_buffer);
    MAP(DestroyBuffer, destroy_buffer);
    MAP(GetBufferMemoryRequirements, buffer_requirements);
    MAP(BindBufferMemory, bind_buffer);
    MAP(AllocateMemory, alloc_memory);
    MAP(FreeMemory, free_memory);
    MAP(MapMemory, map_memory);
    MAP(CreateImage, create_image);
    MAP(DestroyImage, destroy_image);
    MAP(GetImageMemoryRequirements, image_requirements);
    MAP(BindImageMemory, bind_image);
    MAP(CreateImageView, create_view);
    MAP(DestroyImageView, destroy_view);
    MAP(CreateSampler, create_sampler);
    MAP(DestroySampler, destroy_sampler);
    MAP(CmdPipelineBarrier, cmd_barrier);
    MAP(CmdCopyBuffer, cmd_copy);
    MAP(CmdUpdateBuffer, cmd_update);
    MAP(CmdClearColorImage, cmd_clear);
#undef MAP
    return nullptr;
}
#define EXPECT(condition)                                                                                              \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                                               \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryHeapCount = 1;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryTypes[0].heapIndex = 0;
    memory.memoryHeaps[0].size = 1ull << 30;
    {
        leonida::BindlessFeatures features;
        features.partially_bound = true;
        leonida::BindlessSet set(get_proc, (VkDevice)1, memory, features);
        EXPECT(set.ok());
        leonida::DescriptorKey key{};
        EXPECT(set.insert_key(key, 1));
        EXPECT(set.flush((VkCommandBuffer)1));
        set.table_upload_result(true);
        const VkDescriptorSet old_set = set.set();
        const std::uintptr_t old_pool = counts.pools.live[0];
        EXPECT(old_set && old_pool && set.retired_count() == 0 && set.generations_made() == 0);
        EXPECT(set.insert_key(key, 2) && set.flush_replaces_table());

        // Arm at the replacement set's allocation, immediately before the old generation enters retired_.
        arm_after_set_alloc = true;
        bool threw = false;
        try {
            set.flush((VkCommandBuffer)1, 42);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        EXPECT(!arm_after_set_alloc && !fail_new);
        fail_new = false;
        EXPECT(threw);
        EXPECT(set.set() == old_set && set.retired_count() == 0 && set.generations_made() == 0);
        EXPECT(set.table().lookup(key) == 2 && set.flush_replaces_table());
        EXPECT(counts.pools.contains(old_pool));

        // A retry makes the new generation; the old one stays live until its submission ticket completes.
        EXPECT(set.flush((VkCommandBuffer)1, 42));
        set.table_upload_result(true);
        EXPECT(set.set() != old_set && set.retired_count() == 1 && set.generations_made() == 1);
        set.release_retired(41);
        EXPECT(set.retired_count() == 1 && counts.pools.contains(old_pool));
        set.release_retired(42);
        EXPECT(set.retired_count() == 0 && !counts.pools.contains(old_pool));
    }
    std::printf("buffers %d/%d, memory %d/%d, pools %d/%d\n", counts.buffers.created, counts.buffers.destroyed,
                counts.memory.created, counts.memory.destroyed, counts.pools.created, counts.pools.destroyed);
    EXPECT(counts.balanced()); // every unique handle created by the fake driver was destroyed exactly once
    std::puts("bindless_generation_oom_test: all checks passed");
    return 0;
}
