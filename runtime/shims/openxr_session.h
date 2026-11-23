#pragma once

// VR mode over OpenXR (LEONIDA_VR=openxr, docs/vr.md). Built only with -DLEONIDA_OPENXR=ON, which links the Khronos
// OpenXR loader into the D3D12 layer; without it the layer is unchanged.
//
// The session follows XR_KHR_vulkan_enable2. The runtime has to see the Vulkan instance and device being made, so
// create_vulkan_device_standalone (d3d12_vulkan_device.cpp) asks it in three steps:
//   1. create(), before any Vulkan object: the XrInstance (with XR_KHR_vulkan_enable2 only) and the headset's
//      XrSystemId. The runtime's extensions are logged in one line, including the ones later work needs for
//      eye-tracked foveated rendering (vr_config.h, ExtensionSupport); none of those is enabled yet.
//   2. create_vulkan_instance() / physical_device() / create_vulkan_device(): xrCreateVulkanInstanceKHR,
//      xrGetVulkanGraphicsDevice2KHR (the GPU the headset is on) and xrCreateVulkanDeviceKHR. The runtime adds the
//      Vulkan extensions it needs to the layer's own lists.
//   3. start(), once the D3D12 device and its queue exist: the XrSession on the layer's first graphics queue and a
//      thread that runs the frame loop (xrWaitFrame, xrBeginFrame, xrEndFrame) at the headset's rate, apart from
//      the game's presents.
// Every step can fail, for example with no runtime installed, no headset connected or a runtime without
// XR_KHR_vulkan_enable2: the reason is logged once per process, the step returns null or false, and the caller
// carries on with the flat window as without VR. Nothing here throws.
//
// What the frame loop shows comes from a LayerSource. The first one is the virtual cinema (make_cinema()): a
// PresentMirror that copies each frame the window presenter shows into a headset swapchain, shown as a quad layer
// in front of the viewer while the window keeps showing it too. Later sources (projection layers for stereo, and
// foveation with an eye-gaze action) plug into the same loop.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <mutex>

namespace leonida {
class PresentMirror;
}

namespace rt::vr {

class OpenXrSession {
public:
    // Step 1. Null when LEONIDA_VR is not "openxr" (silently when unset or off), or when OpenXR cannot be used.
    static std::unique_ptr<OpenXrSession> create();
    ~OpenXrSession();
    OpenXrSession(const OpenXrSession&) = delete;
    OpenXrSession& operator=(const OpenXrSession&) = delete;

    // Step 2. Each failure is logged; the caller then drops the session and makes the object itself.
    VkResult create_vulkan_instance(PFN_vkGetInstanceProcAddr get, const VkInstanceCreateInfo& info, VkInstance* out);
    VkPhysicalDevice physical_device(VkInstance instance);
    VkResult create_vulkan_device(PFN_vkGetInstanceProcAddr get, VkPhysicalDevice physical,
                                  const VkDeviceCreateInfo& info, VkDevice* out);

    // Step 3. `queue_mutex` guards every use of the queue (the layer's submit mutex): OpenXR calls that may use it
    // (xrBeginFrame, xrEndFrame, xrAcquireSwapchainImage, xrReleaseSwapchainImage) take it too. False (logged) when
    // the session cannot be made.
    bool start(PFN_vkGetInstanceProcAddr get, VkInstance instance, VkPhysicalDevice physical, VkDevice device,
               std::uint32_t queue_family, std::uint32_t queue_index, std::mutex& queue_mutex);

    // The virtual cinema for one window presenter (VulkanPresenter::set_mirror), or null before start(). The newest
    // cinema is the one shown.
    std::shared_ptr<leonida::PresentMirror> make_cinema();

    // How many times this process reported that VR is unavailable: at most once, whatever failed. For tests.
    static unsigned unavailable_reports();

    struct Impl;

private:
    OpenXrSession();
    std::shared_ptr<Impl> impl_;
};

}  // namespace rt::vr
