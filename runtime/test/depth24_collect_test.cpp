// Checks completed scratch batches through prepare(), retire() and collect() without a Vulkan device.
#include "../src/vulkan_depth24.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

namespace {
unsigned allocations_until_failure = 0;
}

void* operator new(std::size_t size) {
    if (allocations_until_failure && --allocations_until_failure == 0) throw std::bad_alloc();
    if (void* storage = std::malloc(size ? size : 1)) return storage;
    throw std::bad_alloc();
}

void operator delete(void* storage) noexcept { std::free(storage); }
void operator delete(void* storage, std::size_t) noexcept { std::free(storage); }

namespace {

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

std::uintptr_t next_handle = 0;
unsigned buffers_created = 0, memories_allocated = 0;
std::array<unsigned, 2> buffer_destroys{}, memory_frees{};

template <typename Handle>
Handle fake_handle(std::uintptr_t id) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(id);
    else return static_cast<Handle>(id);
}

template <typename Info, typename Handle>
VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice, const Info*, const VkAllocationCallbacks*, Handle* out) {
    *out = fake_handle<Handle>(++next_handle);
    return VK_SUCCESS;
}

template <typename Handle>
VKAPI_ATTR void VKAPI_CALL destroy(VkDevice, Handle, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    *out = fake_handle<VkBuffer>(++buffers_created);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    for (std::size_t i = 0; i < buffer_destroys.size(); ++i)
        if (buffer == fake_handle<VkBuffer>(i + 1)) ++buffer_destroys[i];
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    for (std::size_t i = 0; i < memory_frees.size(); ++i)
        if (memory == fake_handle<VkDeviceMemory>(i + 1)) ++memory_frees[i];
}

VKAPI_ATTR VkResult VKAPI_CALL create_pipeline(VkDevice, VkPipelineCache, std::uint32_t,
                                               const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                               VkPipeline* out) {
    *out = fake_handle<VkPipeline>(++next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_set(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet* out) {
    *out = fake_handle<VkDescriptorSet>(++next_handle);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL memory_requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    *out = {5u << 20, 256, 1};
}

VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL semaphore_value(VkDevice, VkSemaphore, std::uint64_t*) { return VK_ERROR_DEVICE_LOST; }
VKAPI_ATTR void VKAPI_CALL update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                      const VkCopyDescriptorSet*) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                    std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                         std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
#define MOCK(vk_name, ...) \
    if (std::strcmp(name, #vk_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(__VA_ARGS__)
    MOCK(vkCreateShaderModule, create<VkShaderModuleCreateInfo, VkShaderModule>);
    MOCK(vkDestroyShaderModule, destroy<VkShaderModule>);
    MOCK(vkCreateDescriptorSetLayout, create<VkDescriptorSetLayoutCreateInfo, VkDescriptorSetLayout>);
    MOCK(vkDestroyDescriptorSetLayout, destroy<VkDescriptorSetLayout>);
    MOCK(vkCreatePipelineLayout, create<VkPipelineLayoutCreateInfo, VkPipelineLayout>);
    MOCK(vkDestroyPipelineLayout, destroy<VkPipelineLayout>);
    MOCK(vkCreateComputePipelines, create_pipeline);
    MOCK(vkDestroyPipeline, destroy<VkPipeline>);
    MOCK(vkCreateDescriptorPool, create<VkDescriptorPoolCreateInfo, VkDescriptorPool>);
    MOCK(vkDestroyDescriptorPool, destroy<VkDescriptorPool>);
    MOCK(vkAllocateDescriptorSets, allocate_set);
    MOCK(vkUpdateDescriptorSets, update_sets);
    MOCK(vkCreateBuffer, create_buffer);
    MOCK(vkDestroyBuffer, destroy_buffer);
    MOCK(vkGetBufferMemoryRequirements, memory_requirements);
    MOCK(vkBindBufferMemory, bind_memory);
    MOCK(vkFreeMemory, free_memory);
    MOCK(vkCmdBindPipeline, bind_pipeline);
    MOCK(vkCmdBindDescriptorSets, bind_sets);
    MOCK(vkCmdPushConstants, push_constants);
    MOCK(vkCmdDispatch, dispatch);
    MOCK(vkGetSemaphoreCounterValue, semaphore_value);
    MOCK(vkDeviceWaitIdle, wait_idle);
#undef MOCK
    return nullptr;
}

bool collect_with_allocation_failure(unsigned fail_on, bool retry) {
    buffers_created = memories_allocated = 0;
    buffer_destroys = memory_frees = {};
    bool failed = false;
    {
        leonida::Depth24Converter converter(get_proc, fake_handle<VkDevice>(1), [](const VkMemoryRequirements&) {
            return fake_handle<VkDeviceMemory>(++memories_allocated);
        });
        EXPECT(converter.loaded());

        // Each job fills its 5 MiB chunk, so both chunks end up in the same retired batch.
        std::vector<leonida::Depth24Job> parts;
        for (unsigned i = 0; i < 2; ++i) {
            leonida::Depth24Job job{};
            job.app = fake_handle<VkBuffer>(99);
            job.app_size = 5u << 20;
            job.width = job.app_row_words = 256;
            job.height = 5120;
            job.slices = 1;
            job.app_slice_words = job.app_row_words * job.height;
            EXPECT(converter.prepare(job, parts));
        }
        converter.retire(VK_NULL_HANDLE, 0);
        allocations_until_failure = fail_on;
        try {
            converter.collect();
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        allocations_until_failure = 0;
        if (retry) {
            EXPECT(failed);
            converter.collect();
        }
    }
    EXPECT(buffers_created == 2 && memories_allocated == 2);
    EXPECT(buffer_destroys[0] == 1 && buffer_destroys[1] == 1);
    EXPECT(memory_frees[0] == 1 && memory_frees[1] == 1);
    return true;
}

}  // namespace

int main() {
    // The old code grows the cache twice. Failing the second growth exposes a partially transferred batch.
    if (!collect_with_allocation_failure(2, false)) return 1;
    // Failing the initial reservation must also leave the batch safe to collect again.
    if (!collect_with_allocation_failure(1, true)) return 1;
    std::puts("depth24_collect_test: ok");
    return 0;
}
