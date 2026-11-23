// A Windows Vulkan guest for the macOS host: one Win32 window, a Win32 surface, a swapchain cleared to a cycling colour
// for about three seconds, and a readback of the last frame's first pixel. Exits 0 when that pixel is the clear colour,
// else the number of the failing step. Vulkan comes only from vulkan-1.dll's vkGetInstanceProcAddr, as in Leonida.
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>

__declspec(dllimport) PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char*);

#define FN(name) PFN_##name name
#define LOAD(inst, name) name = (PFN_##name)vkGetInstanceProcAddr(inst, #name)
FN(vkCreateInstance); FN(vkEnumeratePhysicalDevices); FN(vkGetPhysicalDeviceQueueFamilyProperties);
FN(vkCreateDevice); FN(vkGetDeviceQueue); FN(vkCreateWin32SurfaceKHR); FN(vkGetPhysicalDeviceSurfaceFormatsKHR);
FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR); FN(vkGetPhysicalDeviceSurfaceSupportKHR); FN(vkCreateSwapchainKHR);
FN(vkGetSwapchainImagesKHR); FN(vkAcquireNextImageKHR); FN(vkQueuePresentKHR); FN(vkCreateCommandPool);
FN(vkAllocateCommandBuffers); FN(vkBeginCommandBuffer); FN(vkEndCommandBuffer); FN(vkCmdPipelineBarrier);
FN(vkCmdClearColorImage); FN(vkCmdCopyImageToBuffer); FN(vkQueueSubmit); FN(vkQueueWaitIdle); FN(vkCreateSemaphore);
FN(vkCreateBuffer); FN(vkGetBufferMemoryRequirements); FN(vkAllocateMemory); FN(vkBindBufferMemory); FN(vkMapMemory);
FN(vkGetPhysicalDeviceMemoryProperties); FN(vkDeviceWaitIdle); FN(vkEnumerateDeviceExtensionProperties);

static void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &b);
}

static int run(void) {
    HINSTANCE hinst = GetModuleHandleW(NULL);
    WNDCLASSEXW wc = {sizeof wc, 0, DefWindowProcW, 0, 0, hinst, NULL, NULL, NULL, NULL, L"leonida", NULL};
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, L"leonida", L"Leonida macOS host: Vulkan from a Windows guest", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, 640, 360, NULL, NULL, hinst, NULL);
    if (!hwnd) return 1;
    ShowWindow(hwnd, SW_SHOW);

    LOAD(NULL, vkCreateInstance);
    const char* iext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vk_clear", 1, NULL, 0, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 2, iext};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst)) return 2;
    LOAD(inst, vkEnumeratePhysicalDevices); LOAD(inst, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(inst, vkCreateDevice); LOAD(inst, vkGetDeviceQueue); LOAD(inst, vkCreateWin32SurfaceKHR);
    LOAD(inst, vkGetPhysicalDeviceSurfaceFormatsKHR); LOAD(inst, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD(inst, vkGetPhysicalDeviceSurfaceSupportKHR); LOAD(inst, vkCreateSwapchainKHR);
    LOAD(inst, vkGetSwapchainImagesKHR); LOAD(inst, vkAcquireNextImageKHR); LOAD(inst, vkQueuePresentKHR);
    LOAD(inst, vkCreateCommandPool); LOAD(inst, vkAllocateCommandBuffers); LOAD(inst, vkBeginCommandBuffer);
    LOAD(inst, vkEndCommandBuffer); LOAD(inst, vkCmdPipelineBarrier); LOAD(inst, vkCmdClearColorImage);
    LOAD(inst, vkCmdCopyImageToBuffer); LOAD(inst, vkQueueSubmit); LOAD(inst, vkQueueWaitIdle);
    LOAD(inst, vkCreateSemaphore); LOAD(inst, vkCreateBuffer); LOAD(inst, vkGetBufferMemoryRequirements);
    LOAD(inst, vkAllocateMemory); LOAD(inst, vkBindBufferMemory); LOAD(inst, vkMapMemory);
    LOAD(inst, vkGetPhysicalDeviceMemoryProperties); LOAD(inst, vkDeviceWaitIdle);
    LOAD(inst, vkEnumerateDeviceExtensionProperties);

    VkWin32SurfaceCreateInfoKHR sci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, NULL, 0, hinst, hwnd};
    VkSurfaceKHR surface;
    if (vkCreateWin32SurfaceKHR(inst, &sci, NULL, &surface)) return 3;

    VkPhysicalDevice gpu;
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &gpu);
    if (!n) return 4;
    VkBool32 can_present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(gpu, 0, surface, &can_present);
    if (!can_present) return 5;

    const char* dext[2] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    uint32_t ndext = 1, have = 0;
    VkExtensionProperties props[256];
    have = 256;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &have, props);
    for (uint32_t i = 0; i < have; ++i)
        if (!strcmp(props[i].extensionName, "VK_KHR_portability_subset")) dext[ndext++] = "VK_KHR_portability_subset";
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, ndext, dext};
    VkDevice dev;
    if (vkCreateDevice(gpu, &dci, NULL, &dev)) return 6;
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps);
    VkSurfaceFormatKHR fmt;
    n = 1;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &n, &fmt);
    VkSwapchainCreateInfoKHR swci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swci.surface = surface;
    swci.minImageCount = caps.minImageCount < 2 ? 2 : caps.minImageCount;
    swci.imageFormat = fmt.format;
    swci.imageColorSpace = fmt.colorSpace;
    swci.imageExtent = caps.currentExtent;
    swci.imageArrayLayers = 1;
    swci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swci.clipped = VK_TRUE;
    VkSwapchainKHR swapchain;
    if (vkCreateSwapchainKHR(dev, &swci, NULL, &swapchain)) return 7;
    VkImage images[8];
    n = 8;
    vkGetSwapchainImagesKHR(dev, swapchain, &n, images);
    printf("vk_clear: %ux%u swapchain, %u images, format %d\n", swci.imageExtent.width, swci.imageExtent.height, n,
           fmt.format);

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                   VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VkCommandPool pool;
    vkCreateCommandPool(dev, &pci, NULL, &pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &cai, &cb);
    VkSemaphoreCreateInfo semi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acquired, done;
    vkCreateSemaphore(dev, &semi, NULL, &acquired);
    vkCreateSemaphore(dev, &semi, NULL, &done);

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer readback;
    vkCreateBuffer(dev, &bci, NULL, &readback);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, readback, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
    uint32_t type = 0;
    for (; type < mp.memoryTypeCount; ++type)
        if ((req.memoryTypeBits >> type & 1) && (mp.memoryTypes[type].propertyFlags &
                                                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                                                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            break;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
    VkDeviceMemory mem;
    if (vkAllocateMemory(dev, &mai, NULL, &mem)) return 8;
    vkBindBufferMemory(dev, readback, mem, 0);
    unsigned char* pixel;
    vkMapMemory(dev, mem, 0, 4, 0, (void**)&pixel);

    const int frames = 180;
    VkClearColorValue color = {{0}};
    for (int f = 0; f < frames; ++f) {
        MSG msg;
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        float t = (float)f / frames;  // a blue-to-orange ramp, exact in 8 bits at the end
        color.float32[0] = t;
        color.float32[1] = 0.25f + 0.25f * t;
        color.float32[2] = 1.0f - t;
        color.float32[3] = 1.0f;
        uint32_t index;
        if (vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index)) return 9;
        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb, &begin);
        barrier(cb, images[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &all);
        if (f == frames - 1) {
            barrier(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy copy = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
            vkCmdCopyImageToBuffer(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
            barrier(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        } else {
            barrier(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
        vkEndCommandBuffer(cb);
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &acquired, &wait_stage, 1, &cb, 1, &done};
        if (vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)) return 10;
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &done, 1, &swapchain, &index, NULL};
        if (vkQueuePresentKHR(queue, &pi)) return 11;
        vkQueueWaitIdle(queue);
    }
    vkDeviceWaitIdle(dev);

    // B8G8R8A8 or R8G8B8A8, UNORM or SRGB: compare against the final colour in both channel orders, with slack for sRGB.
    int r = pixel[2], g = pixel[1], b = pixel[0];
    if (fmt.format == VK_FORMAT_R8G8B8A8_UNORM || fmt.format == VK_FORMAT_R8G8B8A8_SRGB) { r = pixel[0]; b = pixel[2]; }
    printf("vk_clear: last frame's first pixel r=%d g=%d b=%d\n", r, g, b);
    if (r < 200 || b > 40 || g < 100) return 12;
    printf("vk_clear: OK\n");
    return 0;
}

int main(void) { return run(); }
