// VR mode over OpenXR: the session, its frame loop and the virtual cinema (openxr_session.h, docs/vr.md).
#include "openxr_session.h"

#include "../src/vr_config.h"
#include "../src/vulkan_present.h"

#define XR_USE_GRAPHICS_API_VULKAN 1
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static_assert(leonida::vr::vk_format_r8g8b8a8_unorm == std::int64_t(VK_FORMAT_R8G8B8A8_UNORM));
static_assert(leonida::vr::vk_format_r8g8b8a8_srgb == std::int64_t(VK_FORMAT_R8G8B8A8_SRGB));
static_assert(leonida::vr::vk_format_b8g8r8a8_unorm == std::int64_t(VK_FORMAT_B8G8R8A8_UNORM));
static_assert(leonida::vr::vk_format_b8g8r8a8_srgb == std::int64_t(VK_FORMAT_B8G8R8A8_SRGB));

namespace rt::vr {

namespace {

std::atomic<unsigned> g_unavailable_reports{0};
std::atomic<bool> g_unavailable_said{false};

// "VR unavailable" goes to the log once per process, whatever the reason; every caller then shows the window only.
void report_unavailable(const char* format, ...) {
    if (g_unavailable_said.exchange(true)) return;
    char why[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(why, sizeof(why), format, args);
    va_end(args);
    std::fprintf(stderr, "[rt] vr: VR unavailable: %s; showing the window only\n", why);
    ++g_unavailable_reports;
}

std::string result_text(XrInstance instance, XrResult result) {
    char name[XR_MAX_RESULT_STRING_SIZE] = {};
    if (instance != XR_NULL_HANDLE && XR_SUCCEEDED(xrResultToString(instance, result, name))) return name;
    // Without an instance xrResultToString cannot be asked; these are the results the calls before one give.
    switch (result) {
    case XR_ERROR_RUNTIME_UNAVAILABLE: return "XR_ERROR_RUNTIME_UNAVAILABLE";
    case XR_ERROR_RUNTIME_FAILURE: return "XR_ERROR_RUNTIME_FAILURE";
    case XR_ERROR_INITIALIZATION_FAILED: return "XR_ERROR_INITIALIZATION_FAILED";
    case XR_ERROR_API_VERSION_UNSUPPORTED: return "XR_ERROR_API_VERSION_UNSUPPORTED";
    case XR_ERROR_EXTENSION_NOT_PRESENT: return "XR_ERROR_EXTENSION_NOT_PRESENT";
    case XR_ERROR_OUT_OF_MEMORY: return "XR_ERROR_OUT_OF_MEMORY";
    default: return "XrResult " + std::to_string(int(result));
    }
}

template <typename F>
F instance_fn(XrInstance instance, const char* name) {
    PFN_xrVoidFunction fn = nullptr;
    if (XR_FAILED(xrGetInstanceProcAddr(instance, name, &fn))) return nullptr;
    return reinterpret_cast<F>(fn);
}

// What the frame loop shows: the layers of one frame, in the session's reference space. Called by the frame loop
// with the source mutex held, which stays held until xrEndFrame has used them.
class LayerSource {
public:
    virtual ~LayerSource() = default;
    virtual void layers(XrSpace space, std::vector<const XrCompositionLayerBaseHeader*>& out) = 0;
};

}  // namespace

struct OpenXrSession::Impl {
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace space = XR_NULL_HANDLE;
    XrEnvironmentBlendMode blend = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    std::uint32_t max_width = 0, max_height = 0;
    leonida::vr::ExtensionSupport extensions;
    leonida::vr::CinemaConfig cinema;
    std::vector<std::int64_t> formats;  // the runtime's swapchain formats (VkFormat), its order of preference

    PFN_xrGetVulkanGraphicsRequirements2KHR graphics_requirements = nullptr;
    PFN_xrCreateVulkanInstanceKHR create_vk_instance = nullptr;
    PFN_xrGetVulkanGraphicsDevice2KHR vk_graphics_device = nullptr;
    PFN_xrCreateVulkanDeviceKHR create_vk_device = nullptr;

    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;
    std::mutex* queue_mutex = nullptr;

    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> lost{false};  // the session ended (headset gone, runtime closed) or was shut down
    std::mutex source_mutex;        // source, and a cinema's swapchain while the frame loop may use it
    LayerSource* source = nullptr;

    void set_source(LayerSource* s) {
        std::lock_guard lock(source_mutex);
        source = s;
    }
    void clear_source(LayerSource* s) {
        std::lock_guard lock(source_mutex);
        if (source == s) source = nullptr;
    }

    void run();

    void shutdown() {
        stop = true;
        if (thread.joinable()) thread.join();
        std::lock_guard lock(source_mutex);
        lost = true;
        // Destroying the session destroys its swapchains too; a cinema that is still alive then has nothing to free.
        if (space != XR_NULL_HANDLE) xrDestroySpace(space);
        if (session != XR_NULL_HANDLE) xrDestroySession(session);
        if (instance != XR_NULL_HANDLE) xrDestroyInstance(instance);
        space = XR_NULL_HANDLE;
        session = XR_NULL_HANDLE;
        instance = XR_NULL_HANDLE;
    }
};

// The frame loop: session state changes, then one xrWaitFrame / xrBeginFrame / xrEndFrame per headset frame, with
// the current source's layers. It runs at the headset's rate and never waits for the game: a cinema shows the last
// frame it was given until the next one arrives.
void OpenXrSession::Impl::run() {
    bool running = false;
    bool warned_frame = false;
    std::vector<const XrCompositionLayerBaseHeader*> layers;
    while (!stop) {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(instance, &event) == XR_SUCCESS) {
            if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                const XrSessionState state = reinterpret_cast<const XrEventDataSessionStateChanged&>(event).state;
                if (state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                    begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    const XrResult r = xrBeginSession(session, &begin);
                    running = XR_SUCCEEDED(r);
                    if (running)
                        std::fprintf(stderr, "[rt] vr: headset session running\n");
                    else
                        std::fprintf(stderr, "[rt] vr: xrBeginSession failed (%s)\n", result_text(instance, r).c_str());
                } else if (state == XR_SESSION_STATE_STOPPING) {
                    xrEndSession(session);
                    running = false;
                    std::fprintf(stderr, "[rt] vr: headset session stopped\n");
                } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
                    lost = true;
                }
            } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                lost = true;
            }
            event = {XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (lost) {
            std::fprintf(stderr, "[rt] vr: the headset session ended; the window carries on\n");
            return;
        }
        if (!running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState frame{XR_TYPE_FRAME_STATE};
        XrResult r = xrWaitFrame(session, &wait, &frame);
        if (XR_SUCCEEDED(r)) {
            XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
            std::lock_guard queue(*queue_mutex);
            r = xrBeginFrame(session, &begin);
        }
        if (XR_FAILED(r)) {
            if (!warned_frame) std::fprintf(stderr, "[rt] vr: frame loop: %s\n", result_text(instance, r).c_str());
            warned_frame = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        std::lock_guard lock(source_mutex);
        layers.clear();
        if (frame.shouldRender && source) source->layers(space, layers);
        XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
        end.displayTime = frame.predictedDisplayTime;
        end.environmentBlendMode = blend;
        end.layerCount = std::uint32_t(layers.size());
        end.layers = layers.empty() ? nullptr : layers.data();
        std::lock_guard queue(*queue_mutex);
        r = xrEndFrame(session, &end);
        if (XR_FAILED(r) && !warned_frame) {
            std::fprintf(stderr, "[rt] vr: xrEndFrame: %s\n", result_text(instance, r).c_str());
            warned_frame = true;
        }
    }
}

namespace {

// The virtual cinema: each frame the window shows is copied into a headset swapchain of the same size, shown as a
// quad layer (vr_config.h, place_screen). The copy is recorded into the presenter's own command buffer, between its
// write of the window's image and the present, so the headset gets exactly the window's frame and the game's pacing
// is untouched. The swapchain image is acquired when the copy is recorded and released once it is submitted; the
// frame loop shows the last released image.
class Cinema final : public leonida::PresentMirror, public LayerSource {
public:
    explicit Cinema(std::shared_ptr<OpenXrSession::Impl> s) : s_(std::move(s)) {
        barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(s_->device_proc(s_->device, "vkCmdPipelineBarrier"));
        copy_ = reinterpret_cast<PFN_vkCmdCopyImage>(s_->device_proc(s_->device, "vkCmdCopyImage"));
        s_->set_source(this);
    }

    ~Cinema() override {
        s_->clear_source(this);
        std::lock_guard lock(s_->source_mutex);
        if (swapchain_ != XR_NULL_HANDLE && !s_->lost) xrDestroySwapchain(swapchain_);
    }

    void record(VkCommandBuffer commands, VkImage image, VkExtent2D extent, VkFormat format) override {
        if (failed_ || s_->lost || !barrier_ || !copy_) return;
        const std::int64_t want = leonida::vr::cinema_swapchain_format(format, s_->formats);
        if (!want) {
            if (!warned_format_)
                std::fprintf(stderr, "[rt] vr: the headset has no swapchain format for the window's format %d (sRGB "
                                     "8-bit with the same channel order); the headset shows nothing\n",
                             int(format));
            warned_format_ = true;
            return;
        }
        if ((s_->max_width && extent.width > s_->max_width) || (s_->max_height && extent.height > s_->max_height)) {
            if (!warned_size_)
                std::fprintf(stderr, "[rt] vr: the window (%ux%u) is larger than the headset's largest swapchain "
                                     "(%ux%u); the headset shows nothing until it is smaller\n",
                             extent.width, extent.height, s_->max_width, s_->max_height);
            warned_size_ = true;
            return;
        }
        if (swapchain_ == XR_NULL_HANDLE || extent.width != extent_.width || extent.height != extent_.height ||
            want != format_) {
            if (!make_swapchain(extent, want)) {
                failed_ = true;
                return;
            }
        }
        if (state_ == Image::none) {
            XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            XrResult r;
            {
                std::lock_guard queue(*s_->queue_mutex);
                r = xrAcquireSwapchainImage(swapchain_, &acquire, &index_);
            }
            if (XR_FAILED(r)) return fail("xrAcquireSwapchainImage", r);
            state_ = Image::acquired;
        }
        if (state_ == Image::acquired) {
            // Normally at once; a compositor that still reads the image makes this frame skip the headset, and the
            // next one waits again.
            XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wait.timeout = 100'000'000;  // ns
            const XrResult r = xrWaitSwapchainImage(swapchain_, &wait);
            if (r == XR_TIMEOUT_EXPIRED) return;
            if (XR_FAILED(r)) return fail("xrWaitSwapchainImage", r);
            state_ = Image::ready;
        }
        // A waited image is in COLOR_ATTACHMENT_OPTIMAL (XR_KHR_vulkan_enable2, usage COLOR_ATTACHMENT) and goes back so.
        const VkImage target = images_[index_].image;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_copy.srcAccessMask = 0;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = target;
        to_copy.subresourceRange = range;
        barrier_(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                 1, &to_copy);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {extent.width, extent.height, 1};
        copy_(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
              &region);
        VkImageMemoryBarrier back = to_copy;
        back.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        back.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        back.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier_(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                 1, &back);
        pending_release_ = true;
    }

    // With the queue mutex held (PresentMirror).
    void submitted() override {
        if (!pending_release_) return;
        pending_release_ = false;
        state_ = Image::none;
        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        const XrResult r = xrReleaseSwapchainImage(swapchain_, &release);
        if (XR_FAILED(r)) return fail("xrReleaseSwapchainImage", r);
        has_content_ = true;
    }

    // With the source mutex held (LayerSource).
    void layers(XrSpace space, std::vector<const XrCompositionLayerBaseHeader*>& out) override {
        if (swapchain_ == XR_NULL_HANDLE || !has_content_) return;
        quad_.space = space;
        out.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad_));
    }

private:
    enum class Image { none, acquired, ready };

    void fail(const char* what, XrResult r) {
        std::fprintf(stderr, "[rt] vr: %s failed (%s); the headset shows nothing from now on\n", what,
                     result_text(s_->instance, r).c_str());
        failed_ = true;
    }

    // A swapchain the window's size, replacing the last one. The frame loop does not show it before its first image
    // is released.
    bool make_swapchain(VkExtent2D extent, std::int64_t format) {
        {
            std::lock_guard lock(s_->source_mutex);
            if (swapchain_ != XR_NULL_HANDLE) xrDestroySwapchain(swapchain_);
            swapchain_ = XR_NULL_HANDLE;
            images_.clear();
            has_content_ = false;
            state_ = Image::none;
        }
        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        info.format = format;
        info.sampleCount = 1;
        info.width = extent.width;
        info.height = extent.height;
        info.faceCount = 1;
        info.arraySize = 1;
        info.mipCount = 1;
        XrSwapchain made = XR_NULL_HANDLE;
        XrResult r = xrCreateSwapchain(s_->session, &info, &made);
        if (XR_FAILED(r)) {
            fail("xrCreateSwapchain", r);
            return false;
        }
        std::uint32_t count = 0;
        r = xrEnumerateSwapchainImages(made, 0, &count, nullptr);
        std::vector<XrSwapchainImageVulkan2KHR> images(count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
        if (XR_SUCCEEDED(r) && count)
            r = xrEnumerateSwapchainImages(made, count, &count,
                                           reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data()));
        if (XR_FAILED(r) || !count) {
            xrDestroySwapchain(made);
            fail("xrEnumerateSwapchainImages", r);
            return false;
        }
        const leonida::vr::QuadPlacement placed = leonida::vr::place_screen(s_->cinema, extent.width, extent.height);
        std::lock_guard lock(s_->source_mutex);
        swapchain_ = made;
        images_ = std::move(images);
        extent_ = extent;
        format_ = format;
        quad_ = {XR_TYPE_COMPOSITION_LAYER_QUAD};
        quad_.layerFlags = 0;  // opaque: the window's alpha is not a coverage mask
        quad_.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        quad_.subImage.swapchain = made;
        quad_.subImage.imageRect = {{0, 0}, {std::int32_t(extent.width), std::int32_t(extent.height)}};
        quad_.pose.orientation = {placed.pose.orientation.x, placed.pose.orientation.y, placed.pose.orientation.z,
                                  placed.pose.orientation.w};
        quad_.pose.position = {placed.pose.position.x, placed.pose.position.y, placed.pose.position.z};
        quad_.size = {placed.width_m, placed.height_m};
        std::fprintf(stderr, "[rt] vr: virtual cinema: %ux%u frame on a %.2f x %.2f m screen %.2f m away (%.0f degrees wide)\n",
                     extent.width, extent.height, double(placed.width_m), double(placed.height_m),
                     double(s_->cinema.distance_m), double(leonida::vr::horizontal_fov_degrees(s_->cinema)));
        return true;
    }

    std::shared_ptr<OpenXrSession::Impl> s_;
    PFN_vkCmdPipelineBarrier barrier_ = nullptr;
    PFN_vkCmdCopyImage copy_ = nullptr;
    // Written by the presenting thread only (under the presenter's lock); swapchain_ and quad_ change under the
    // source mutex too, because the frame loop reads them.
    XrSwapchain swapchain_ = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkan2KHR> images_;
    VkExtent2D extent_{};
    std::int64_t format_ = 0;
    XrCompositionLayerQuad quad_{XR_TYPE_COMPOSITION_LAYER_QUAD};
    Image state_ = Image::none;
    std::uint32_t index_ = 0;
    bool pending_release_ = false;
    std::atomic<bool> has_content_{false};
    bool failed_ = false;
    bool warned_format_ = false;
    bool warned_size_ = false;
};

}  // namespace

OpenXrSession::OpenXrSession() : impl_(std::make_shared<Impl>()) {}

OpenXrSession::~OpenXrSession() { impl_->shutdown(); }

unsigned OpenXrSession::unavailable_reports() { return g_unavailable_reports.load(); }

std::unique_ptr<OpenXrSession> OpenXrSession::create() {
    using namespace leonida::vr;
    const std::optional<std::string> mode_text = process_env("LEONIDA_VR");
    const Mode mode = parse_mode(mode_text ? mode_text->c_str() : nullptr);
    if (mode == Mode::off) return nullptr;
    if (mode == Mode::unknown) {
        report_unavailable("LEONIDA_VR=%s is not a VR mode (the one there is: openxr)", mode_text->c_str());
        return nullptr;
    }
    std::unique_ptr<OpenXrSession> self(new OpenXrSession());
    Impl& x = *self->impl_;
    const CinemaSettings settings = cinema_settings_from(process_env);
    for (const std::string& warning : settings.warnings) std::fprintf(stderr, "[rt] vr: %s\n", warning.c_str());
    x.cinema = settings.config;

    // The runtime's extensions, in one line, before anything is enabled.
    std::uint32_t count = 0;
    XrResult r = xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
    std::vector<XrExtensionProperties> properties(count, {XR_TYPE_EXTENSION_PROPERTIES});
    if (XR_SUCCEEDED(r) && count) r = xrEnumerateInstanceExtensionProperties(nullptr, count, &count, properties.data());
    if (XR_FAILED(r)) {
        report_unavailable("no OpenXR runtime (%s)", result_text(XR_NULL_HANDLE, r).c_str());
        return nullptr;
    }
    std::vector<std::string> names;
    for (std::uint32_t i = 0; i < count; ++i) names.emplace_back(properties[i].extensionName);
    x.extensions = extension_support(names);
    std::fprintf(stderr, "[rt] vr: OpenXR runtime extensions: %s\n", extension_summary(x.extensions).c_str());
    if (!x.extensions.vulkan_enable2) {
        report_unavailable("the OpenXR runtime has no XR_KHR_vulkan_enable2");
        return nullptr;
    }

    const char* enabled[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
    XrInstanceCreateInfo create{XR_TYPE_INSTANCE_CREATE_INFO};
    std::snprintf(create.applicationInfo.applicationName, XR_MAX_APPLICATION_NAME_SIZE, "gta6rt");
    std::snprintf(create.applicationInfo.engineName, XR_MAX_ENGINE_NAME_SIZE, "Leonida");
    create.applicationInfo.apiVersion = XR_API_VERSION_1_0;  // every runtime takes 1.0; nothing newer is used
    create.enabledExtensionCount = 1;
    create.enabledExtensionNames = enabled;
    r = xrCreateInstance(&create, &x.instance);
    if (XR_FAILED(r)) {
        x.instance = XR_NULL_HANDLE;
        report_unavailable("xrCreateInstance failed (%s)", result_text(XR_NULL_HANDLE, r).c_str());
        return nullptr;
    }
    XrInstanceProperties instance_properties{XR_TYPE_INSTANCE_PROPERTIES};
    if (XR_SUCCEEDED(xrGetInstanceProperties(x.instance, &instance_properties)))
        std::fprintf(stderr, "[rt] vr: OpenXR runtime: %s %u.%u.%u\n", instance_properties.runtimeName,
                     unsigned(XR_VERSION_MAJOR(instance_properties.runtimeVersion)),
                     unsigned(XR_VERSION_MINOR(instance_properties.runtimeVersion)),
                     unsigned(XR_VERSION_PATCH(instance_properties.runtimeVersion)));

    XrSystemGetInfo system{XR_TYPE_SYSTEM_GET_INFO};
    system.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    r = xrGetSystem(x.instance, &system, &x.system);
    if (XR_FAILED(r)) {
        report_unavailable(r == XR_ERROR_FORM_FACTOR_UNAVAILABLE ? "no headset connected (%s)" : "xrGetSystem failed (%s)",
                           result_text(x.instance, r).c_str());
        return nullptr;
    }
    XrSystemProperties system_properties{XR_TYPE_SYSTEM_PROPERTIES};
    if (XR_SUCCEEDED(xrGetSystemProperties(x.instance, x.system, &system_properties))) {
        x.max_width = system_properties.graphicsProperties.maxSwapchainImageWidth;
        x.max_height = system_properties.graphicsProperties.maxSwapchainImageHeight;
        std::fprintf(stderr, "[rt] vr: headset: %s (swapchains up to %ux%u)\n", system_properties.systemName,
                     x.max_width, x.max_height);
    }

    x.graphics_requirements =
        instance_fn<PFN_xrGetVulkanGraphicsRequirements2KHR>(x.instance, "xrGetVulkanGraphicsRequirements2KHR");
    x.create_vk_instance = instance_fn<PFN_xrCreateVulkanInstanceKHR>(x.instance, "xrCreateVulkanInstanceKHR");
    x.vk_graphics_device = instance_fn<PFN_xrGetVulkanGraphicsDevice2KHR>(x.instance, "xrGetVulkanGraphicsDevice2KHR");
    x.create_vk_device = instance_fn<PFN_xrCreateVulkanDeviceKHR>(x.instance, "xrCreateVulkanDeviceKHR");
    if (!x.graphics_requirements || !x.create_vk_instance || !x.vk_graphics_device || !x.create_vk_device) {
        report_unavailable("the runtime lists XR_KHR_vulkan_enable2 but does not give its functions");
        return nullptr;
    }
    // Required before xrCreateSession. The layer makes a Vulkan 1.3 instance (1.2 devices with the KHR extensions
    // are fine as well); a runtime that needs newer is refused.
    XrGraphicsRequirementsVulkan2KHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    r = x.graphics_requirements(x.instance, x.system, &requirements);
    if (XR_FAILED(r)) {
        report_unavailable("xrGetVulkanGraphicsRequirements2KHR failed (%s)", result_text(x.instance, r).c_str());
        return nullptr;
    }
    if (requirements.minApiVersionSupported > XR_MAKE_VERSION(1, 3, 0)) {
        report_unavailable("the runtime needs Vulkan %u.%u",
                           unsigned(XR_VERSION_MAJOR(requirements.minApiVersionSupported)),
                           unsigned(XR_VERSION_MINOR(requirements.minApiVersionSupported)));
        return nullptr;
    }

    // The runtime's preferred blend mode for a stereo headset (opaque on a VR headset).
    std::uint32_t blend_count = 0;
    if (XR_SUCCEEDED(xrEnumerateEnvironmentBlendModes(x.instance, x.system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                      0, &blend_count, nullptr)) &&
        blend_count) {
        std::vector<XrEnvironmentBlendMode> modes(blend_count);
        if (XR_SUCCEEDED(xrEnumerateEnvironmentBlendModes(x.instance, x.system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                          blend_count, &blend_count, modes.data())) &&
            blend_count)
            x.blend = modes[0];
    }
    return self;
}

VkResult OpenXrSession::create_vulkan_instance(PFN_vkGetInstanceProcAddr get, const VkInstanceCreateInfo& info,
                                               VkInstance* out) {
    Impl& x = *impl_;
    XrVulkanInstanceCreateInfoKHR create{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    create.systemId = x.system;
    create.pfnGetInstanceProcAddr = get;
    create.vulkanCreateInfo = &info;
    VkResult vk = VK_ERROR_INITIALIZATION_FAILED;
    *out = VK_NULL_HANDLE;
    const XrResult r = x.create_vk_instance(x.instance, &create, out, &vk);
    if (XR_FAILED(r) || vk != VK_SUCCESS) {
        report_unavailable("xrCreateVulkanInstanceKHR failed (%s, VkResult %d)", result_text(x.instance, r).c_str(),
                           int(vk));
        return vk != VK_SUCCESS ? vk : VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

VkPhysicalDevice OpenXrSession::physical_device(VkInstance instance) {
    Impl& x = *impl_;
    XrVulkanGraphicsDeviceGetInfoKHR get{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    get.systemId = x.system;
    get.vulkanInstance = instance;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    const XrResult r = x.vk_graphics_device(x.instance, &get, &physical);
    if (XR_FAILED(r) || !physical) {
        report_unavailable("xrGetVulkanGraphicsDevice2KHR failed (%s)", result_text(x.instance, r).c_str());
        return VK_NULL_HANDLE;
    }
    return physical;
}

VkResult OpenXrSession::create_vulkan_device(PFN_vkGetInstanceProcAddr get, VkPhysicalDevice physical,
                                             const VkDeviceCreateInfo& info, VkDevice* out) {
    Impl& x = *impl_;
    XrVulkanDeviceCreateInfoKHR create{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    create.systemId = x.system;
    create.pfnGetInstanceProcAddr = get;
    create.vulkanPhysicalDevice = physical;
    create.vulkanCreateInfo = &info;
    VkResult vk = VK_ERROR_INITIALIZATION_FAILED;
    *out = VK_NULL_HANDLE;
    const XrResult r = x.create_vk_device(x.instance, &create, out, &vk);
    if (XR_FAILED(r) || vk != VK_SUCCESS) {
        report_unavailable("xrCreateVulkanDeviceKHR failed (%s, VkResult %d)", result_text(x.instance, r).c_str(),
                           int(vk));
        return vk != VK_SUCCESS ? vk : VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

bool OpenXrSession::start(PFN_vkGetInstanceProcAddr get, VkInstance instance, VkPhysicalDevice physical,
                          VkDevice device, std::uint32_t queue_family, std::uint32_t queue_index,
                          std::mutex& queue_mutex) {
    Impl& x = *impl_;
    XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    binding.instance = instance;
    binding.physicalDevice = physical;
    binding.device = device;
    binding.queueFamilyIndex = queue_family;
    binding.queueIndex = queue_index;
    XrSessionCreateInfo create{XR_TYPE_SESSION_CREATE_INFO};
    create.next = &binding;
    create.systemId = x.system;
    XrResult r = xrCreateSession(x.instance, &create, &x.session);
    if (XR_FAILED(r)) {
        x.session = XR_NULL_HANDLE;
        report_unavailable("xrCreateSession failed (%s)", result_text(x.instance, r).c_str());
        return false;
    }
    // LOCAL: the origin is where the head was when the session started, -Z ahead, Y up.
    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    space.poseInReferenceSpace.orientation.w = 1.0f;
    r = xrCreateReferenceSpace(x.session, &space, &x.space);
    if (XR_FAILED(r)) {
        x.space = XR_NULL_HANDLE;
        report_unavailable("xrCreateReferenceSpace failed (%s)", result_text(x.instance, r).c_str());
        return false;
    }
    std::uint32_t count = 0;
    r = xrEnumerateSwapchainFormats(x.session, 0, &count, nullptr);
    x.formats.resize(count);
    if (XR_SUCCEEDED(r) && count) r = xrEnumerateSwapchainFormats(x.session, count, &count, x.formats.data());
    if (XR_FAILED(r)) {
        report_unavailable("xrEnumerateSwapchainFormats failed (%s)", result_text(x.instance, r).c_str());
        return false;
    }
    x.formats.resize(count);
    x.device = device;
    x.device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get(instance, "vkGetDeviceProcAddr"));
    x.queue_mutex = &queue_mutex;
    if (!x.device_proc) {
        report_unavailable("vkGetDeviceProcAddr is missing");
        return false;
    }
    x.thread = std::thread([&x] { x.run(); });
    std::fprintf(stderr, "[rt] vr: OpenXR session on queue family %u; frames go to the headset and the window\n",
                 queue_family);
    return true;
}

std::shared_ptr<leonida::PresentMirror> OpenXrSession::make_cinema() {
    if (impl_->session == XR_NULL_HANDLE || !impl_->queue_mutex || impl_->lost) return nullptr;
    return std::make_shared<Cinema>(impl_);
}

}  // namespace rt::vr
