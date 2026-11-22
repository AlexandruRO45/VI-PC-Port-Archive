// Concurrent helper reconfiguration with a command buffer recorded using the previous generation.
// A hooked vkQueueSubmit parks after VulkanQueue::execute() has released resource_mutex_.
// Reconfiguration must neither call vkDeviceWaitIdle during that submit nor destroy the old
// scratch/pipeline resources before the pending submission reaches the GPU.
// The submitted D24 and BC1 copies are verified against CPU data under Vulkan validation.
// No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr get_real = nullptr;
PFN_vkQueueSubmit submit_real = nullptr;
PFN_vkDeviceWaitIdle wait_real = nullptr;
PFN_vkDestroyBuffer destroy_buffer_real = nullptr;
PFN_vkDestroyDescriptorPool destroy_pool_real = nullptr;

std::mutex mutex;
std::condition_variable cv;
bool armed = false;
bool entered = false;
bool released = false;
bool in_submit = false;
bool overlapped_idle = false;
bool destroyed_early = false;

void arm() {
    std::lock_guard lock(mutex);
    armed = true;
    entered = false;
    released = false;
    in_submit = false;
    overlapped_idle = false;
    destroyed_early = false;
}

bool wait_for_submit() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(5), [] { return entered; });
}

void release() {
    {
        std::lock_guard lock(mutex);
        released = true;
    }
    cv.notify_all();
}

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* infos, VkFence fence) {
    {
        std::unique_lock lock(mutex);
        if (armed) {
            armed = false;
            entered = true;
            in_submit = true;
            cv.notify_all();
            // The real Vulkan entry point is delayed until reconfiguration publishes its new helper.
            cv.wait(lock, [] { return released; });
            in_submit = false;
            // A pre-fix destructor may already have freed resources recorded by this command buffer.
            // Do not submit invalid GPU handles to the driver when demonstrating that negative control.
            if (overlapped_idle || destroyed_early) return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    return submit_real(queue, count, infos, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkDevice device) {
    {
        std::lock_guard lock(mutex);
        if (in_submit) {
            overlapped_idle = true;
            cv.notify_all();
            // Negative control: no physical driver concurrency and no GPU submission of freed handles.
            return VK_SUCCESS;
        }
    }
    return wait_real(device);
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* alloc) {
    {
        std::lock_guard lock(mutex);
        if (in_submit) destroyed_early = true;
    }
    destroy_buffer_real(device, buffer, alloc);
}

VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice device, VkDescriptorPool pool,
                                       const VkAllocationCallbacks* alloc) {
    {
        std::lock_guard lock(mutex);
        if (in_submit) destroyed_early = true;
    }
    destroy_pool_real(device, pool, alloc);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(submit);
    if (n == "vkDeviceWaitIdle") return reinterpret_cast<PFN_vkVoidFunction>(wait_idle);
    if (n == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(destroy_buffer);
    if (n == "vkDestroyDescriptorPool") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pool);
    return get_real(device, name);
}

void install(const vktest::Device& d) {
    get_real = d.device_proc;
    submit_real = d.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    wait_real = d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
    destroy_buffer_real = d.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    destroy_pool_real = d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
}
} // namespace driver

leonida::ResourceBinding buffer_binding(VkBuffer buffer, VkDeviceSize bytes, bool readback = false) {
    leonida::ResourceBinding b{};
    b.buffer = buffer;
    b.size = bytes;
    b.host_readable = readback;
    return b;
}

// Both cases require the real Vulkan GPU: the parked command buffer references actual resources.
int run_case(vktest::Device& d, bool depth) {
    using namespace leonida;

    // D3D12 placed footprints require 256-byte row pitches (64 R32 texels).
    const std::uint32_t w = depth ? 64u : 16u;
    constexpr std::uint32_t h = 16;
    const VkDeviceSize depth_bytes = VkDeviceSize(w) * h * 4;
    constexpr VkDeviceSize bc_bytes = 4096;
    void *upload_ptr = nullptr, *read_ptr = nullptr;
    const VkBuffer upload = depth ?
        d.buffer(depth_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT, &upload_ptr) : VK_NULL_HANDLE;
    const VkBuffer readback =
        d.buffer(depth ? depth_bytes : bc_bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT, &read_ptr);
    if (!readback || (depth && !upload)) return 1;
    if (depth) {
        for (std::uint32_t i = 0; i < w * h; ++i)
            static_cast<std::uint32_t*>(upload_ptr)[i] = ((i * 1279u + 17) & 0xffffffu) | 0x7f000000u;
    }
    const VkImage image = d.image(depth ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_BC1_RGBA_UNORM_BLOCK, w, h, 1,
                                   depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!image) return 1;

    VkPhysicalDeviceMemoryProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        d.physical, &properties);
    const auto allocate = [&](const VkMemoryRequirements& requirements) -> VkDeviceMemory {
        for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i)) ||
                !(properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            info.allocationSize = requirements.size;
            info.memoryTypeIndex = i;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &info, nullptr, &mem) == VK_SUCCESS)
                return mem;
            break;
        }
        return VK_NULL_HANDLE;
    };

    {
        VulkanContext context(driver::get_device, d.device, true);
        if (!context.loaded()) return 1;
        if (depth) context.enable_depth24(driver::get_device, allocate);
        else context.enable_copy_scratch(driver::get_device, allocate);

        ResourceBinding image_binding{};
        image_binding.image = image;
        image_binding.format = depth ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        image_binding.aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT;
        image_binding.width = w;
        image_binding.height = h;
        image_binding.mip_levels = 1;
        image_binding.array_layers = 1;
        image_binding.depth = 1;
        image_binding.samples = 1;
        if (depth) image_binding.depth24 = image_binding.depth24_as_float = true;
        context.add_resource(1, image_binding, VK_IMAGE_LAYOUT_UNDEFINED);
        if (depth) context.add_resource(2, buffer_binding(upload, depth_bytes));
        context.add_resource(3, buffer_binding(readback, depth ? depth_bytes : bc_bytes, true));

        CommandList list;
        CopyLocation tex{1, false, 0, 0, 0, 0, 0, 0};
        if (depth) {
            CopyLocation src{2, true, 0, 0, w, h, 1, w * 4};
            CopyLocation dst{3, true, 0, 0, w, h, 1, w * 4};
            if (!list.copy_texture({tex, 0, 0, 0, src, false, {}}) ||
                !list.copy_texture({dst, 0, 0, 0, tex, false, {}})) return 1;
        } else {
            // This zero-initializes BC1 through the copy_scratch fallback before reading the image back.
            CopyLocation dst{3, true, 0, 0, w, h, 1, 256};
            if (!list.transition({1, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}) ||
                !list.copy_texture({dst, 0, 0, 0, tex, false, {}})) return 1;
        }
        if (!list.close()) return 1;

        VulkanQueue queue(context, d.queue, d.family);
        if (!queue.ok()) return 1;

        driver::arm();
        std::atomic<ReplayResult> result{ReplayResult::unsupported};
        std::thread recording([&] {
            const CommandList* ptr = &list;
            result.store(queue.execute({&ptr, 1}));
        });
        const bool entered = driver::wait_for_submit();
        std::atomic<bool> reconfigured{false};
        std::thread replacing;
        if (entered) {
            replacing = std::thread([&] {
                if (depth) context.enable_depth24(driver::get_device, allocate);
                else context.enable_copy_scratch(driver::get_device, allocate);
                reconfigured.store(true);
                driver::cv.notify_all();
            });
        }
        bool done_before_release = false;
        if (entered) {
            std::unique_lock lock(driver::mutex);
            done_before_release = driver::cv.wait_for(lock, std::chrono::seconds(3), [&] {
                return reconfigured.load() || driver::overlapped_idle;
            }) && reconfigured.load();
        }
        driver::release();
        recording.join();
        if (replacing.joinable()) replacing.join();
        bool overlap = false, early_free = false;
        {
            std::lock_guard lock(driver::mutex);
            overlap = driver::overlapped_idle;
            early_free = driver::destroyed_early;
        }
        if (!entered || !done_before_release || overlap || early_free ||
            result.load() != ReplayResult::success || !queue.idle()) {
            std::fprintf(stderr, "%s: submit=%d reconfig-before-release=%d wait-overlap=%d early-free=%d result=%d\n",
                         depth ? "depth24" : "scratch", int(entered), int(done_before_release),
                         int(overlap), int(early_free), int(result.load()));
            return 1;
        }

        if (depth) {
            auto* src = static_cast<std::uint32_t*>(upload_ptr);
            auto* dst = static_cast<std::uint32_t*>(read_ptr);
            for (std::uint32_t i = 0; i < w * h; ++i)
                if (dst[i] != (src[i] & 0xffffffu)) {
                    std::fprintf(stderr, "depth24: incorrect result at %u: %08x vs %08x\n",
                                 i, dst[i], src[i] & 0xffffffu);
                    return 1;
                }
        } else {
            for (std::uint32_t y = 0; y < h / 4; ++y)
                for (std::uint32_t x = 0; x < w / 4 * 8; ++x)
                    if (static_cast<std::uint8_t*>(read_ptr)[y * 256 + x] != 0) {
                        std::fprintf(stderr, "scratch: BC1 zero-fill incorrect at block row %u byte %u\n", y, x);
                        return 1;
                    }
        }
        std::printf("%s: reconfigured during parked submit; old generation alive; GPU data correct\n",
                    depth ? "depth24" : "scratch");
    }
    return 0;
}

} // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    driver::install(d);
    if (d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT)) EXPECT(run_case(d, true) == 0);
    else std::printf("depth24: D32_SFLOAT_S8_UINT unavailable, skipped\n");
    if (d.bc) EXPECT(run_case(d, false) == 0);
    else std::printf("scratch: BC1 compression unavailable, skipped\n");
    EXPECT(d.validation_errors() == 0);
    std::printf("vk_helper_reconfigure_test: ok\n");
    return 0;
}
