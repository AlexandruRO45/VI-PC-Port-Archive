// CPU-only regression: fence semaphores are destroyed only after their safe global retire ticket.
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

VkSemaphore destroyed = VK_NULL_HANDLE;
std::uint32_t destroy_count = 0;

void destroy_semaphore(VkDevice, VkSemaphore semaphore, const VkAllocationCallbacks*) {
    destroyed = semaphore;
    ++destroy_count;
}

}  // namespace

int main() {
    using namespace leonida;

    DeferredDeletionQueue queue;
    VulkanDeviceFunctions vk{};
    vk.vkDestroySemaphore = destroy_semaphore;

    queue.push(7, DeferredDeletionQueue::Kind::semaphore, 0x1234);
    queue.drain(6, VK_NULL_HANDLE, vk);
    EXPECT(destroy_count == 0);
    queue.drain(7, VK_NULL_HANDLE, vk);
    EXPECT(destroy_count == 1);
    EXPECT(destroyed == reinterpret_cast<VkSemaphore>(std::uintptr_t{0x1234}));

    std::printf("vk_deferred_semaphore_test: ok\n");
    return 0;
}
