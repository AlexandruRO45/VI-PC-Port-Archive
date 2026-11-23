// Synthetic regression test for the DXGI swapchain buffer-count contract (D3D-389), with no GPU and no
// window: VulkanSwapchain (runtime/shims/dxgi_swapchain.h) runs on a fake Vulkan function table, so
// vkGetSwapchainImagesKHR can report more images than the app's requested BufferCount.
//
// DXGI rule: GetDesc/GetDesc1 report the BufferCount the app created with, and GetCurrentBackBufferIndex is
// always below it, one stable resource per index. The shim must not expose images the driver added above
// minImageCount, and must hide a caps.minImageCount above the request. When the native count differs, the
// exposed buffers form an offscreen ring; each present blits the current slot into the acquired native image.
// The fake also executes the FLIP_SEQUENTIAL image copies to check the resulting pixels without a compositor.
#include "../shims/dxgi_swapchain.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::uintptr_t kSurface = 0x5100;
constexpr std::uintptr_t kSwapchainBase = 0x5300;
constexpr std::uintptr_t kFence = 0x5102;
constexpr std::uintptr_t kPool = 0x5103;
constexpr std::uintptr_t kCmd = 0x5104;
constexpr std::uintptr_t kNativeBase = 0x6000;
constexpr std::uintptr_t kOffscreenBase = 0x7000;

enum class Pixel : std::uint8_t { unset, blue, green, yellow, magenta, red };
struct PixelImage {
    VkExtent2D extent;
    std::vector<Pixel> pixels;
};

struct Fake {
    std::uint32_t min_image_count = 2;
    std::uint32_t max_image_count = 0;
    std::uint32_t returned_images = 2;
    std::uint32_t create_min_image_count = 0;
    std::uint32_t acquire_cursor = 0;
    std::uint32_t last_acquired = 0;
    std::uint32_t presented_image = UINT32_MAX;
    VkImage blit_src = VK_NULL_HANDLE;
    VkImage blit_dst = VK_NULL_HANDLE;
    unsigned presents = 0;
    unsigned created_offscreen = 0;
    unsigned swapchain_creations = 0, fail_history_create_in = 0, fail_ring_create_in = 0;
    VkSwapchainKHR active_swapchain = VK_NULL_HANDLE, last_old_swapchain = VK_NULL_HANDLE;
    bool fail_swapchain_create = false;
    bool fail_acquire = false;
    VkExtent2D surface_extent{64, 64};
    bool compose = false;
    bool invalid_copy = false;
    VkResult next_present = VK_SUCCESS;
    std::unordered_map<VkImage, PixelImage> pixel_images;
    std::vector<Pixel> blitted, visible;
};
Fake g;

VkImage native_image(std::uint32_t i) {
    return reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kNativeBase + i));
}

template <typename T>
PFN_vkVoidFunction as_function(T fn) {
    return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

VKAPI_ATTR VkResult VKAPI_CALL create_win32_surface(VkInstance, const VkWin32SurfaceCreateInfoKHR*,
                                                    const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    if (out) *out = reinterpret_cast<VkSurfaceKHR>(kSurface);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_surface(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL surface_support(VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32* out) {
    if (out) *out = VK_TRUE;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_caps(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* caps) {
    if (!caps) return VK_ERROR_INITIALIZATION_FAILED;
    std::memset(caps, 0, sizeof(*caps));
    caps->minImageCount = g.min_image_count;
    caps->maxImageCount = g.max_image_count;
    caps->currentExtent = g.surface_extent;
    caps->minImageExtent = {1, 1};
    caps->maxImageExtent = {4096, 4096};
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_formats(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                               VkSurfaceFormatKHR* formats) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (formats && *count) {
        formats[0].format = VK_FORMAT_B8G8R8A8_UNORM;
        formats[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_present_modes(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                     VkPresentModeKHR* modes) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (modes && *count) modes[0] = VK_PRESENT_MODE_FIFO_KHR;
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* props) {
    std::memset(props, 0, sizeof(*props));
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = 1ull << 30;
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}
VKAPI_ATTR VkResult VKAPI_CALL create_swapchain(VkDevice, const VkSwapchainCreateInfoKHR* info,
                                                const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    if (!info || !out || info->oldSwapchain != g.active_swapchain) return VK_ERROR_INITIALIZATION_FAILED;
    ++g.swapchain_creations;
    g.create_min_image_count = info->minImageCount;
    g.last_old_swapchain = info->oldSwapchain;
    // Vulkan retires oldSwapchain even when creation of its replacement fails.
    g.active_swapchain = VK_NULL_HANDLE;
    if (g.fail_swapchain_create) {
        g.fail_swapchain_create = false;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = g.active_swapchain = reinterpret_cast<VkSwapchainKHR>(kSwapchainBase + g.swapchain_creations);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_swapchain(VkDevice, VkSwapchainKHR swapchain, const VkAllocationCallbacks*) {
    if (g.active_swapchain == swapchain) g.active_swapchain = VK_NULL_HANDLE;
}
VKAPI_ATTR VkResult VKAPI_CALL get_images(VkDevice, VkSwapchainKHR, std::uint32_t* count, VkImage* images) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (images) {
        for (std::uint32_t i = 0; i < g.returned_images && i < *count; ++i) images[i] = native_image(i);
    }
    *count = g.returned_images;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL acquire_next_image(VkDevice, VkSwapchainKHR swapchain, std::uint64_t, VkSemaphore, VkFence,
                                                  std::uint32_t* out) {
    if (!out || g.returned_images == 0) return VK_ERROR_INITIALIZATION_FAILED;
    if (std::exchange(g.fail_acquire, false)) return VK_ERROR_OUT_OF_DATE_KHR;
    if (swapchain != g.active_swapchain) return VK_ERROR_OUT_OF_DATE_KHR;
    g.last_acquired = g.acquire_cursor % g.returned_images;
    ++g.acquire_cursor;
    *out = g.last_acquired;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue, const VkPresentInfoKHR* info) {
    if (!info || info->swapchainCount != 1 || !info->pImageIndices) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult status = g.next_present;
    g.next_present = VK_SUCCESS;
    if (status != VK_SUCCESS) return status;
    g.presented_image = info->pImageIndices[0];
    ++g.presents;
    if (g.compose) g.visible = g.blitted;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                            VkFence* out) {
    if (out) *out = reinterpret_cast<VkFence>(kFence);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice, VkFence, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, std::uint32_t, const VkFence*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL device_wait_idle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*,
                                            VkImage* out) {
    if (!info || !out) return VK_ERROR_INITIALIZATION_FAILED;
    constexpr auto history_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    auto& failing = info->usage == history_usage ? g.fail_history_create_in : g.fail_ring_create_in;
    if (failing && --failing == 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kOffscreenBase + g.created_offscreen++));
    if (g.compose) {
        const auto width = info->extent.width, height = info->extent.height;
        g.pixel_images.emplace(*out, PixelImage{{width, height}, std::vector<Pixel>(std::size_t(width) * height)});
    }
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage image, const VkAllocationCallbacks*) {
    g.pixel_images.erase(image);
}
VKAPI_ATTR void VKAPI_CALL image_mem_req(VkDevice, VkImage, VkMemoryRequirements* req) {
    std::memset(req, 0, sizeof(*req));
    req->size = 4096;
    req->alignment = 1;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_mem(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                         VkDeviceMemory* out) {
    if (out) *out = reinterpret_cast<VkDeviceMemory>(kOffscreenBase);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL bind_image_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_cmd_pool(VkDevice, const VkCommandPoolCreateInfo*,
                                               const VkAllocationCallbacks*, VkCommandPool* out) {
    if (out) *out = reinterpret_cast<VkCommandPool>(kPool);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_cmd_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL alloc_cmd_bufs(VkDevice, const VkCommandBufferAllocateInfo*,
                                              VkCommandBuffer* out) {
    if (out) *out = reinterpret_cast<VkCommandBuffer>(kCmd);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_cmd_buf(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL begin_cmd_buf(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL end_cmd_buf(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, std::uint32_t,
                                       const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                       std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL cmd_blit(VkCommandBuffer, VkImage src, VkImageLayout, VkImage dst, VkImageLayout,
                                     std::uint32_t count, const VkImageBlit* regions, VkFilter) {
    g.blit_src = src;
    g.blit_dst = dst;
    if (!g.compose) return;
    const auto image = g.pixel_images.find(src);
    const auto extent = g.surface_extent;
    if (image == g.pixel_images.end() || image->second.extent.width != extent.width ||
        image->second.extent.height != extent.height || count != 1 ||
        regions[0].srcOffsets[1].x != int(extent.width) || regions[0].srcOffsets[1].y != int(extent.height) ||
        regions[0].dstOffsets[1].x != int(extent.width) || regions[0].dstOffsets[1].y != int(extent.height)) {
        g.invalid_copy = true;
        return;
    }
    g.blitted = image->second.pixels;
}
VKAPI_ATTR void VKAPI_CALL cmd_copy(VkCommandBuffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                     VkImageLayout dst_layout, std::uint32_t count, const VkImageCopy* regions) {
    if (!g.compose) return;
    const auto from = g.pixel_images.find(src), to = g.pixel_images.find(dst);
    const bool readable = src_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || src_layout == VK_IMAGE_LAYOUT_GENERAL;
    const bool writable = dst_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL || dst_layout == VK_IMAGE_LAYOUT_GENERAL;
    if (from == g.pixel_images.end() || to == g.pixel_images.end() || src == dst ||
        !readable || !writable) {
        g.invalid_copy = true;
        return;
    }
    const auto fits = [](VkExtent2D image, VkOffset3D start, VkExtent3D size) {
        return start.x >= 0 && start.y >= 0 && start.z == 0 && size.depth == 1 && size.width && size.height &&
               std::uint64_t(start.x) + size.width <= image.width && std::uint64_t(start.y) + size.height <= image.height;
    };
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& r = regions[i];
        if (!fits(from->second.extent, r.srcOffset, r.extent) || !fits(to->second.extent, r.dstOffset, r.extent)) {
            g.invalid_copy = true;
            return;
        }
        for (std::uint32_t y = 0; y < r.extent.height; ++y) {
            const auto source = (std::size_t(r.srcOffset.y) + y) * from->second.extent.width + r.srcOffset.x;
            const auto dest = (std::size_t(r.dstOffset.y) + y) * to->second.extent.width + r.dstOffset.x;
            std::copy_n(from->second.pixels.data() + source, r.extent.width, to->second.pixels.data() + dest);
        }
    }
}
VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    return VK_SUCCESS;
}

struct Entry {
    const char* name;
    PFN_vkVoidFunction fn;
};
const Entry kEntries[] = {
    {"vkCreateWin32SurfaceKHR", as_function(create_win32_surface)},
    {"vkDestroySurfaceKHR", as_function(destroy_surface)},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", as_function(surface_support)},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", as_function(surface_caps)},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", as_function(surface_formats)},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", as_function(surface_present_modes)},
    {"vkGetPhysicalDeviceMemoryProperties", as_function(memory_properties)},
    {"vkCreateSwapchainKHR", as_function(create_swapchain)},
    {"vkDestroySwapchainKHR", as_function(destroy_swapchain)},
    {"vkGetSwapchainImagesKHR", as_function(get_images)},
    {"vkAcquireNextImageKHR", as_function(acquire_next_image)},
    {"vkQueuePresentKHR", as_function(queue_present)},
    {"vkCreateFence", as_function(create_fence)},
    {"vkDestroyFence", as_function(destroy_fence)},
    {"vkWaitForFences", as_function(wait_for_fences)},
    {"vkResetFences", as_function(reset_fences)},
    {"vkDeviceWaitIdle", as_function(device_wait_idle)},
    {"vkCreateImage", as_function(create_image)},
    {"vkDestroyImage", as_function(destroy_image)},
    {"vkGetImageMemoryRequirements", as_function(image_mem_req)},
    {"vkAllocateMemory", as_function(alloc_mem)},
    {"vkFreeMemory", as_function(free_mem)},
    {"vkBindImageMemory", as_function(bind_image_mem)},
    {"vkCreateCommandPool", as_function(create_cmd_pool)},
    {"vkDestroyCommandPool", as_function(destroy_cmd_pool)},
    {"vkAllocateCommandBuffers", as_function(alloc_cmd_bufs)},
    {"vkResetCommandBuffer", as_function(reset_cmd_buf)},
    {"vkBeginCommandBuffer", as_function(begin_cmd_buf)},
    {"vkEndCommandBuffer", as_function(end_cmd_buf)},
    {"vkCmdPipelineBarrier", as_function(cmd_barrier)},
    {"vkCmdBlitImage", as_function(cmd_blit)},
    {"vkCmdCopyImage", as_function(cmd_copy)},
    {"vkQueueSubmit", as_function(queue_submit)},
};

PFN_vkVoidFunction lookup(const char* name) {
    for (const Entry& entry : kEntries)
        if (name && std::strcmp(name, entry.name) == 0) return entry.fn;
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc(VkInstance, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice, const char* name) { return lookup(name); }

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// Requests `requested` buffers while faking `native` images and a surface minimum of `minimum`.
int run_case(const char* label, std::uint32_t requested, std::uint32_t native, std::uint32_t minimum,
             bool expect_offscreen) {
    g = Fake{};
    g.min_image_count = minimum;
    g.returned_images = native;
    std::mutex mutex;
    leonida::VulkanSwapchain swapchain(get_instance_proc, reinterpret_cast<VkInstance>(1),
                                      reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                      reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 0, mutex);
    if (!swapchain.create(reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1)), 64, 64,
                          VK_FORMAT_B8G8R8A8_UNORM, requested))
        return fail("VulkanSwapchain::create");

    // GetDesc/GetDesc1 read image_count(): the app's request, never the native count.
    if (swapchain.image_count() != requested) {
        std::fprintf(stderr, "FAIL: %s: exposed BufferCount %u, requested %u (native %u)\n", label,
                     swapchain.image_count(), requested, native);
        return 1;
    }
    const std::uint32_t expected_min = minimum > requested ? minimum : requested;
    if (g.create_min_image_count != expected_min) {
        std::fprintf(stderr, "FAIL: %s: vkCreateSwapchainKHR minImageCount %u, expected %u\n", label,
                     g.create_min_image_count, expected_min);
        return 1;
    }
    if (swapchain.is_offscreen() != expect_offscreen)
        return fail(expect_offscreen ? "expected the offscreen ring" : "expected direct native images");

    if (expect_offscreen) {
        if (g.created_offscreen != requested) return fail("offscreen ring size != requested BufferCount");
        std::vector<VkImage> exposed;
        for (std::uint32_t i = 0; i < requested; ++i) {
            const leonida::ResourceBinding binding = swapchain.resource_binding(i);
            if (!binding.image || binding.presentable) return fail("offscreen backbuffer binding is not offscreen");
            if (binding.image != swapchain.image(i)) return fail("binding/image disagree");
            for (VkImage other : exposed)
                if (other == binding.image) return fail("two exposed backbuffers share one image");
            exposed.push_back(binding.image);
        }
        if (swapchain.current_back_buffer_index() >= requested)
            return fail("initial current index out of range");

        std::uint32_t index = 0;
        if (!swapchain.acquire_next_image(index)) return fail("first acquire");
        constexpr unsigned kFrames = 20;
        for (unsigned frame = 0; frame < kFrames; ++frame) {
            const std::uint32_t current = swapchain.current_back_buffer_index();
            if (current >= requested) return fail("GetCurrentBackBufferIndex >= BufferCount");
            if (current != frame % requested) return fail("current index does not cycle over the exposed ring");
            const std::uint32_t acquired = g.last_acquired;
            g.blit_src = g.blit_dst = VK_NULL_HANDLE;
            g.presented_image = UINT32_MAX;
            if (!swapchain.present(current)) return fail("present");
            if (g.blit_src != exposed[current]) return fail("present did not blit the current backbuffer");
            if (g.blit_dst != native_image(acquired)) return fail("present blitted to the wrong acquired image");
            if (g.presented_image != acquired) return fail("presented an image other than the acquired one");
            if (frame + 1 < kFrames && !swapchain.acquire_next_image(index)) return fail("acquire");
        }
        // ResizeBuffers(count=0) keeps the exposed BufferCount, not the native image count.
        if (!swapchain.resize(64, 64)) return fail("resize");
        if (swapchain.image_count() != requested) return fail("resize(count=0) changed the exposed BufferCount");
        if (!swapchain.is_offscreen()) return fail("resize dropped the offscreen ring");
        return 0;
    }

    // Direct path: exposed index and native image index are the same, and nothing is blitted.
    if (swapchain.current_back_buffer_index() != 0) return fail("direct initial index");
    std::uint32_t index = 0;
    if (!swapchain.acquire_next_image(index)) return fail("direct first acquire");
    for (unsigned frame = 0; frame < 9; ++frame) {
        const std::uint32_t current = swapchain.current_back_buffer_index();
        if (current >= requested) return fail("direct GetCurrentBackBufferIndex >= BufferCount");
        if (current != g.last_acquired) return fail("direct current index is not the acquired image");
        if (swapchain.resource_binding(current).image != native_image(current))
            return fail("direct binding is not the native image");
        g.blit_src = g.blit_dst = VK_NULL_HANDLE;
        g.presented_image = UINT32_MAX;
        if (!swapchain.present(current)) return fail("direct present");
        if (g.blit_src || g.blit_dst) return fail("direct present took the blit path");
        if (g.presented_image != current) return fail("direct present index");
        if (frame + 1 < 9 && !swapchain.acquire_next_image(index)) return fail("direct acquire");
    }
    return 0;
}

struct Case {
    const char* label;
    std::uint32_t requested;
    std::uint32_t native;
    std::uint32_t minimum;
    bool offscreen;
};

int run_present_history() {
    using namespace leonida;
    g = Fake{};
    g.surface_extent = {128, 128};
    g.compose = true;
    std::mutex mutex;
    VulkanSwapchain swapchain(get_instance_proc, reinterpret_cast<VkInstance>(1), reinterpret_cast<VkPhysicalDevice>(2),
                              get_device_proc, reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 0, mutex);
    if (!swapchain.create(reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1)), 128, 128,
                          VK_FORMAT_B8G8R8A8_UNORM, 2, false, true) || !swapchain.is_offscreen())
        return fail("FLIP_SEQUENTIAL needs an offscreen ring even when the native image count matches");

    const auto acquire = [&]() -> PixelImage* {
        std::uint32_t native_index = 0;
        if (!swapchain.acquire_next_image(native_index)) return nullptr;
        const auto index = swapchain.current_back_buffer_index();
        const auto image = g.pixel_images.find(swapchain.image(index));
        return index < swapchain.image_count() && image != g.pixel_images.end() ? &image->second : nullptr;
    };
    const auto present = [&](const PresentRegions* regions = nullptr) {
        return swapchain.present(swapchain.current_back_buffer_index(), regions) && !g.invalid_copy &&
               g.visible.size() == std::size_t(g.surface_extent.width) * g.surface_extent.height;
    };
    const auto fill = [](PixelImage& image, Rect rect, Pixel color) {
        for (std::int32_t y = rect.top; y < rect.bottom; ++y)
            std::fill_n(image.pixels.data() + std::size_t(y) * image.extent.width + rect.left,
                        rect.right - rect.left, color);
    };
    const auto at = [](std::uint32_t x, std::uint32_t y) { return g.visible[y * g.surface_extent.width + x]; };

    const Rect small_dirty[]{{4, 4, 12, 12}};
    const PresentRegions partial{small_dirty, std::nullopt};
    auto* frame = acquire();
    if (!frame) return fail("acquire full frame");
    if (present(&partial) || g.invalid_copy || g.presents) return fail("a new chain requires a full frame first");
    fill(*frame, {0, 0, 64, 128}, Pixel::blue);
    fill(*frame, {64, 0, 128, 128}, Pixel::green);
    if (!present() || at(12, 48) != Pixel::blue || at(116, 48) != Pixel::green)
        return fail("full Present seeds the frame history");

    frame = acquire();
    if (!frame) return fail("acquire right-scroll frame");
    const Rect dirty_right[]{{44, 0, 76, 128}};
    fill(*frame, dirty_right[0], Pixel::yellow);
    const auto right = present_scroll_copy({{32, 0, 92, 128}, 32, 0}, 128, 128);
    const PresentRegions scroll_right{dirty_right, right};
    if (!right || !present(&scroll_right) || at(12, 48) != Pixel::blue || at(58, 48) != Pixel::yellow ||
        at(84, 48) != Pixel::blue || at(116, 48) != Pixel::green)
        return fail("right scroll uses the previous frame and dirty pixels win the overlap");
    if (frame->pixels != g.visible) return fail("the submitted backbuffer keeps every pixel of the composed frame");

    frame = acquire();
    if (!frame) return fail("acquire frame to fail");
    fill(*frame, {0, 0, 128, 128}, Pixel::red);
    g.next_present = VK_ERROR_OUT_OF_DATE_KHR;
    if (present() || g.invalid_copy) return fail("injected present failure");

    frame = acquire();
    if (!frame || frame->pixels != g.visible)
        return fail("the reacquired backbuffer preserves its composed pixels across the ring");
    const Rect dirty_left[]{{20, 16, 48, 80}};
    fill(*frame, dirty_left[0], Pixel::magenta);
    const auto left = present_scroll_copy({{0, 0, 60, 128}, -32, 0}, 128, 128);
    const PresentRegions scroll_left{dirty_left, left};
    if (!left || !present(&scroll_left) || at(8, 48) != Pixel::blue || at(14, 48) != Pixel::yellow ||
        at(30, 48) != Pixel::magenta || at(30, 8) != Pixel::yellow || at(55, 48) != Pixel::blue ||
        at(70, 48) != Pixel::yellow || at(116, 48) != Pixel::green)
        return fail("left scroll uses only the last successful frame and preserves pixels outside the hints");
    const auto clipped = present_scroll_copy({{0, 0, 64, 128}, -80, 0}, 128, 128);
    if (!clipped || clipped->source.left != 80 || clipped->source.right != 128 || clipped->dst_x != 0)
        return fail("scroll clips its source and keeps the matching destination");

    const auto first_buffer = swapchain.image(0), second_buffer = swapchain.image(1);
    const auto old_count = swapchain.image_count();
    const auto old_images = g.pixel_images.size();
    const auto old_creations = g.swapchain_creations;
    struct StageFailure { unsigned ring, history; Pixel marker; const char* name; };
    const StageFailure failures[] = {
        {1, 0, Pixel::red, "first ring allocation"},
        {old_count, 0, Pixel::green, "last ring allocation"},
        {0, 2, Pixel::blue, "second history allocation"},
    };
    std::vector<Pixel> retained;
    for (const auto& failure : failures) {
        g.fail_ring_create_in = failure.ring;
        g.fail_history_create_in = failure.history;
        if (swapchain.resize(96, 80) || g.fail_ring_create_in || g.fail_history_create_in ||
            g.swapchain_creations != old_creations || g.pixel_images.size() != old_images ||
            swapchain.image_count() != old_count || swapchain.extent().width != 128 ||
            swapchain.extent().height != 128 || swapchain.image(0) != first_buffer || swapchain.image(1) != second_buffer) {
            std::fprintf(stderr, "FAIL: %s must keep the old swapchain and free staged images\n", failure.name);
            return 1;
        }
        frame = acquire();
        if (!frame) return fail("acquire after failed resize");
        retained = g.visible;
        fill(*frame, small_dirty[0], failure.marker);
        if (!present(&partial) || at(8, 8) != failure.marker || at(30, 48) != Pixel::magenta ||
            at(70, 48) != Pixel::yellow || at(116, 48) != Pixel::green)
            return fail("after a failed resize, dirty updates still compose with the old history");
    }
    frame = acquire();
    if (!frame || frame->pixels != retained || !present() || g.visible != retained)
        return fail("a full Present without drawing uses the preserved pixels of the reacquired backbuffer");

    g.surface_extent = {96, 80};
    if (!swapchain.resize(96, 80)) return fail("retry resize FLIP_SEQUENTIAL chain");
    frame = acquire();
    if (!frame) return fail("acquire after resize");
    const auto presents = g.presents;
    if (present(&partial) || g.invalid_copy || g.presents != presents)
        return fail("resize discards the history and requires a new full frame");
    fill(*frame, {0, 0, 96, 80}, Pixel::green);
    if (!present() || !std::all_of(g.visible.begin(), g.visible.end(), [](Pixel pixel) { return pixel == Pixel::green; }))
        return fail("the full frame after resize seeds history at the new size");

    // Use a non-default last successful count so a zero-count retry cannot pass by falling back to two.
    g.returned_images = 3;
    if (!swapchain.resize(96, 80, VK_FORMAT_UNDEFINED, 3) || swapchain.image_count() != 3)
        return fail("set up three exposed buffers for the native swapchain failure");
    g.fail_swapchain_create = true;
    if (swapchain.resize(88, 72) || g.fail_swapchain_create || !g.last_old_swapchain || g.active_swapchain ||
        !g.pixel_images.empty() || swapchain.image_count() || acquire())
        return fail("native creation failure leaves no usable retired swapchain or staged images");
    g.surface_extent = {88, 72};
    if (!swapchain.resize(88, 72) || g.last_old_swapchain || swapchain.image_count() != 3)
        return fail("retry native creation without the retired swapchain and preserve the requested count");
    const auto before_acquire_failure = g.presents;
    g.fail_acquire = true;
    if (acquire() || present() || g.presents != before_acquire_failure)
        return fail("a failed acquire must not present an unacquired native image");
    frame = acquire();
    if (!frame) return fail("acquire after retrying native creation");
    fill(*frame, {0, 0, 88, 72}, Pixel::red);
    if (!present() || at(80, 60) != Pixel::red) return fail("present after retrying native creation");
    return 0;
}

}  // namespace

int main() {
    const Case cases[] = {
        // Vulkan returns more images than minImageCount: the app asked for 2 and must see 2.
        {"driver returns 4 for BufferCount 2", 2, 4, 2, true},
        // The surface minimum exceeds the request (2): still expose the app's 2.
        {"minImageCount 3 for BufferCount 2", 2, 3, 3, true},
        // Control: the app asks for more than minImageCount; native and exposed counts agree, no blit.
        {"BufferCount 3 above minImageCount 2", 3, 3, 2, false},
    };
    for (const Case& c : cases) {
        if (const int result = run_case(c.label, c.requested, c.native, c.minimum, c.offscreen)) return result;
    }
    if (const int result = run_present_history()) return result;
    std::puts("dxgi_swapchain_count_test: PASS (buffer counts, scroll, dirty overlap, failed present, resize rollback)");
    return 0;
}
