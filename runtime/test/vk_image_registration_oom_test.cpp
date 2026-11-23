// Host allocation failures must not leave an image binding without its layout, or replace an existing binding.
// Fake Vulkan handles only: no loader, GPU, or game data.
#include "../src/vulkan_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

namespace {
long fail_after = -1;
int failures = 0;
int destroyed_sparse_images = 0;
VkImage barrier_image = VK_NULL_HANDLE;
VkBuffer updated_buffer = VK_NULL_HANDLE;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

PFN_vkVoidFunction VKAPI_CALL no_proc(VkDevice, const char*) { return nullptr; }
void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) { ++destroyed_sparse_images; }

template<typename Handle> Handle handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(value);
    else return static_cast<Handle>(value);
}

leonida::ResourceBinding image(std::uintptr_t value, std::uint32_t mips = 3, std::uint32_t layers = 2) {
    leonida::ResourceBinding b{};
    b.image = handle<VkImage>(value);
    b.format = VK_FORMAT_R8G8B8A8_UNORM;
    b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    b.mip_levels = mips;
    b.array_layers = layers;
    return b;
}

void ordinary_new_id() {
    int caught = 0, succeeded = 0;
    for (long n = 0; n < 8; ++n) {
        leonida::VulkanContext ctx(no_proc, handle<VkDevice>(1), false);
        fail_after = n;
        bool threw = false;
        try { ctx.add_resource(7, image(101), VK_IMAGE_LAYOUT_GENERAL); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        if (threw) {
            ++caught;
            expect(!ctx.has_resource(7), "failed ordinary registration retained a new ID");
            expect(ctx.layout(7, 0) == VK_IMAGE_LAYOUT_UNDEFINED, "failed ordinary registration retained a layout");
            ctx.add_resource(7, image(101), VK_IMAGE_LAYOUT_GENERAL);
        } else {
            ++succeeded;
        }
        expect(ctx.has_resource(7), "ordinary registration or retry lost ID");
        for (std::uint32_t sub = 0; sub < 6; ++sub)
            expect(ctx.layout(7, sub) == VK_IMAGE_LAYOUT_GENERAL, "ordinary registration or retry lost layout");
    }
    expect(caught >= 2 && succeeded >= 1, "ordinary sweep did not reach both failure and success");
}

void ordinary_replacement() {
    int caught = 0, succeeded = 0;
    for (long n = 0; n < 6; ++n) {
        leonida::VulkanContext ctx(no_proc, handle<VkDevice>(1), false);
        ctx.add_resource(8, image(201, 2, 2), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        fail_after = n;
        bool threw = false;
        try { ctx.add_resource(8, image(202, 3, 2), VK_IMAGE_LAYOUT_GENERAL); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        expect(ctx.has_resource(8), "replacement lost the existing ID");
        if (threw) {
            ++caught;
            for (std::uint32_t sub = 0; sub < 4; ++sub)
                expect(ctx.layout(8, sub) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       "failed replacement changed an existing subresource layout");
            expect(ctx.layout(8, 4) == VK_IMAGE_LAYOUT_UNDEFINED,
                   "failed replacement changed the existing subresource count");
            ctx.add_resource(8, image(202, 3, 2), VK_IMAGE_LAYOUT_GENERAL);
        } else {
            ++succeeded;
        }
        for (std::uint32_t sub = 0; sub < 6; ++sub)
            expect(ctx.layout(8, sub) == VK_IMAGE_LAYOUT_GENERAL, "replacement or retry lost a new layout");
    }
    expect(caught >= 1 && succeeded >= 1, "replacement sweep did not reach both failure and success");
}

void sparse_new_id() {
    int caught = 0, succeeded = 0;
    for (long n = 0; n < 9; ++n) {
        leonida::VulkanContext ctx(no_proc, handle<VkDevice>(1), false);
        ctx.vk.vkDestroyImage = destroy_image;
        auto sparse = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
        sparse->image = handle<VkImage>(301);
        fail_after = n;
        bool threw = false;
        try { ctx.add_sparse_resource(9, image(301), sparse); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        if (threw) {
            ++caught;
            expect(!ctx.has_resource(9), "failed sparse registration retained a new ID");
            expect(sparse.use_count() == 1, "failed sparse registration retained ownership");
            ctx.add_sparse_resource(9, image(301), sparse);
        } else {
            ++succeeded;
        }
        expect(ctx.has_resource(9), "sparse registration or retry lost ID");
        expect(sparse.use_count() == 2, "sparse registration or retry lost ownership");
        for (std::uint32_t sub = 0; sub < 6; ++sub)
            expect(ctx.layout(9, sub) == VK_IMAGE_LAYOUT_UNDEFINED,
                   "sparse registration or retry had unexpected initial layout");
    }
    expect(caught >= 3 && succeeded >= 1, "sparse sweep did not reach map and tracker failures plus success");
    expect(destroyed_sparse_images == 9, "sparse images were not destroyed exactly once");
}

void sparse_replacement() {
    int caught = 0, succeeded = 0;
    for (long n = 0; n < 6; ++n) {
        leonida::VulkanContext ctx(no_proc, handle<VkDevice>(1), false);
        ctx.vk.vkDestroyImage = destroy_image;
        auto old = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
        old->image = handle<VkImage>(401);
        ctx.add_sparse_resource(10, image(401, 2, 2), old);
        ctx.add_resource(10, image(401, 2, 2), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        auto replacement = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
        replacement->image = handle<VkImage>(402);
        fail_after = n;
        bool threw = false;
        try { ctx.add_sparse_resource(10, image(402, 3, 2), replacement); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        expect(ctx.has_resource(10), "sparse replacement lost existing ID");
        expect(old.use_count() == 2, "sparse replacement lost existing owner");
        expect(replacement.use_count() == 1, "duplicate sparse owner replaced the existing owner");
        if (threw) {
            ++caught;
            for (std::uint32_t sub = 0; sub < 4; ++sub)
                expect(ctx.layout(10, sub) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       "failed sparse replacement changed the prior layout");
            expect(ctx.layout(10, 4) == VK_IMAGE_LAYOUT_UNDEFINED,
                   "failed sparse replacement changed the subresource count");
            ctx.add_sparse_resource(10, image(402, 3, 2), replacement);
        } else {
            ++succeeded;
        }
        for (std::uint32_t sub = 0; sub < 6; ++sub)
            expect(ctx.layout(10, sub) == VK_IMAGE_LAYOUT_UNDEFINED,
                   "sparse replacement or retry lost the new layout");
    }
    expect(caught >= 1 && succeeded >= 1, "sparse replacement sweep did not reach failure and success");
    expect(destroyed_sparse_images == 21, "sparse replacements were not destroyed exactly once");
}

// A real VulkanQueue replay resolves the binding after the failed replacement. This catches an eager binding
// overwrite even when the old LayoutTracker entry happens to survive the same allocation failure.
VkResult VKAPI_CALL create_pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* out) {
    *out = handle<VkCommandPool>(1); return VK_SUCCESS;
}
void VKAPI_CALL destroy_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VkResult VKAPI_CALL create_semaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* out) {
    *out = handle<VkSemaphore>(1); return VK_SUCCESS;
}
void VKAPI_CALL destroy_semaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}
VkResult VKAPI_CALL counter(VkDevice, VkSemaphore, std::uint64_t* out) { *out = UINT64_MAX; return VK_SUCCESS; }
VkResult VKAPI_CALL idle(VkQueue) { return VK_SUCCESS; }
VkResult VKAPI_CALL allocate_commands(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* out) {
    *out = handle<VkCommandBuffer>(1); return VK_SUCCESS;
}
VkResult VKAPI_CALL reset_command(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VkResult VKAPI_CALL begin_command(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VkResult VKAPI_CALL end_command(VkCommandBuffer) { return VK_SUCCESS; }
VkResult VKAPI_CALL submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) { return VK_SUCCESS; }
void VKAPI_CALL record_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                               std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                               std::uint32_t count, const VkImageMemoryBarrier* images) {
    if (count) barrier_image = images[0].image;
}
void VKAPI_CALL record_update(VkCommandBuffer, VkBuffer buffer, VkDeviceSize, VkDeviceSize, const void*) {
    updated_buffer = buffer;
}
PFN_vkVoidFunction VKAPI_CALL queue_proc(VkDevice, const char* name) {
#define PROC(vk_name, implementation) \
    if (std::strcmp(name, #vk_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(implementation)
    PROC(vkCreateCommandPool, create_pool); PROC(vkDestroyCommandPool, destroy_pool);
    PROC(vkCreateSemaphore, create_semaphore); PROC(vkDestroySemaphore, destroy_semaphore);
    PROC(vkGetSemaphoreCounterValue, counter); PROC(vkQueueWaitIdle, idle);
    PROC(vkAllocateCommandBuffers, allocate_commands); PROC(vkResetCommandBuffer, reset_command);
    PROC(vkBeginCommandBuffer, begin_command); PROC(vkEndCommandBuffer, end_command);
    PROC(vkQueueSubmit, submit); PROC(vkCmdPipelineBarrier, record_barrier);
    PROC(vkCmdUpdateBuffer, record_update);
#undef PROC
    return nullptr;
}

void replacement_binding_replay() {
    leonida::VulkanContext ctx(queue_proc, handle<VkDevice>(1), false);
    ctx.add_resource(11, image(501), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    fail_after = 0;
    bool threw = false;
    try { ctx.add_resource(11, image(502), VK_IMAGE_LAYOUT_GENERAL); }
    catch (const std::bad_alloc&) { threw = true; }
    fail_after = -1;
    expect(threw, "replacement fault did not reach layout allocation");
    if (!threw) return;
    leonida::CommandList list;
    expect(list.transition({11, 0, leonida::d3d12_state::copy_source, leonida::d3d12_state::copy_dest}) && list.close(),
           "binding observation command list invalid");
    barrier_image = VK_NULL_HANDLE;
    {
        leonida::VulkanQueue queue(ctx, handle<VkQueue>(1), 0);
        expect(queue.ok(), "fake queue did not initialize");
        const leonida::CommandList* commands[] = {&list};
        expect(queue.execute(commands) == leonida::ReplayResult::success, "binding observation replay failed");
    }
    expect(barrier_image == handle<VkImage>(501), "failed replacement changed the image used by replay");
}

void sparse_replacement_binding_replay() {
    leonida::VulkanContext ctx(queue_proc, handle<VkDevice>(1), false);
    ctx.vk.vkDestroyImage = destroy_image;
    auto old = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
    old->image = handle<VkImage>(511);
    ctx.add_sparse_resource(14, image(511), old);
    ctx.add_resource(14, image(511), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    auto replacement = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
    replacement->image = handle<VkImage>(512);
    fail_after = 0;
    bool threw = false;
    try { ctx.add_sparse_resource(14, image(512), replacement); }
    catch (const std::bad_alloc&) { threw = true; }
    fail_after = -1;
    expect(threw, "sparse replacement fault did not reach registration allocation");
    if (!threw) return;
    leonida::CommandList list;
    expect(list.transition({14, 0, leonida::d3d12_state::copy_source, leonida::d3d12_state::copy_dest}) && list.close(),
           "sparse binding observation command list invalid");
    barrier_image = VK_NULL_HANDLE;
    {
        leonida::VulkanQueue queue(ctx, handle<VkQueue>(1), 0);
        expect(queue.ok(), "fake sparse replacement queue did not initialize");
        const leonida::CommandList* commands[] = {&list};
        expect(queue.execute(commands) == leonida::ReplayResult::success,
               "sparse binding observation replay failed");
    }
    expect(barrier_image == handle<VkImage>(511),
           "failed sparse replacement changed the image used by replay");
}

void replacement_buffer_va_replay() {
    leonida::VulkanContext ctx(queue_proc, handle<VkDevice>(1), false);
    leonida::ResourceBinding old{};
    old.buffer = handle<VkBuffer>(601);
    old.va = 0x2000;
    old.size = 64;
    ctx.add_resource(12, old);
    fail_after = 0;
    bool threw = false;
    try { ctx.add_resource(12, image(602), VK_IMAGE_LAYOUT_GENERAL); }
    catch (const std::bad_alloc&) { threw = true; }
    fail_after = -1;
    expect(threw, "buffer replacement fault did not reach layout allocation");
    if (!threw) return;
    leonida::CommandList list;
    expect(list.write_immediate({old.va, 0x12345678, leonida::write_immediate_mode::normal}) && list.close(),
           "VA observation command list invalid");
    updated_buffer = VK_NULL_HANDLE;
    {
        leonida::VulkanQueue queue(ctx, handle<VkQueue>(1), 0);
        expect(queue.ok(), "fake VA queue did not initialize");
        const leonida::CommandList* commands[] = {&list};
        expect(queue.execute(commands) == leonida::ReplayResult::success, "failed replacement lost old buffer VA");
    }
    expect(updated_buffer == old.buffer, "failed replacement changed the buffer found by VA");
}

void sparse_retry_replay() {
    const int destroyed_before = destroyed_sparse_images;
    {
        leonida::VulkanContext ctx(queue_proc, handle<VkDevice>(1), false);
        ctx.vk.vkDestroyImage = destroy_image;
        auto sparse = std::make_shared<leonida::SparseResource>(ctx.device(), ctx.vk);
        sparse->image = handle<VkImage>(701);
        auto binding = image(701);
        binding.not_zeroed = true;
        fail_after = 0;
        bool threw = false;
        try { ctx.add_sparse_resource(13, binding, sparse); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        expect(threw && !ctx.has_resource(13), "sparse retry did not start from an unregistered ID");
        ctx.add_sparse_resource(13, binding, sparse);
        leonida::CommandList list;
        expect(list.transition({13, 0, leonida::d3d12_state::common, leonida::d3d12_state::copy_dest}) && list.close(),
               "sparse retry command list invalid");
        barrier_image = VK_NULL_HANDLE;
        {
            leonida::VulkanQueue queue(ctx, handle<VkQueue>(1), 0);
            expect(queue.ok(), "fake sparse queue did not initialize");
            const leonida::CommandList* commands[] = {&list};
            expect(queue.execute(commands) == leonida::ReplayResult::success, "sparse retry replay failed");
        }
        expect(barrier_image == binding.image, "sparse retry replay used the wrong image");
        expect(ctx.layout(13, 0) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               "sparse retry image had no active layout-tracker entry");
    }
    expect(destroyed_sparse_images == destroyed_before + 1, "sparse retry image was not destroyed exactly once");
}
}  // namespace

void* operator new(std::size_t size) {
    // MSVC Debug's empty-vector iterator proxy is allocated in a noexcept constructor. Skip that small
    // bookkeeping allocation and fail the subsequent layout storage or map node instead.
    if (size > 2 * sizeof(void*)) {
        if (fail_after == 0) { fail_after = -1; throw std::bad_alloc{}; }
        if (fail_after > 0) --fail_after;
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    ordinary_new_id();
    ordinary_replacement();
    sparse_new_id();
    sparse_replacement();
    replacement_binding_replay();
    sparse_replacement_binding_replay();
    replacement_buffer_va_replay();
    sparse_retry_replay();
    if (failures) return 1;
    std::puts("vk_image_registration_oom_test: ok");
    return 0;
}
