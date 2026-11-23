// Opt-in Khronos validation for the standalone Vulkan test programs. With LEONIDA_VK_VALIDATION set (and not "0") the
// instance enables VK_LAYER_KHRONOS_validation and VK_EXT_debug_utils (plus VK_KHR_portability_enumeration when the
// loader advertises it, for MoltenVK on macOS), prepare_device() adds VK_KHR_portability_subset when the physical
// device advertises it, every message goes to stderr, and validation-type errors are counted so the program can fail
// on them (exit_status). The layer's own variables still apply, e.g. VK_KHRONOS_VALIDATION_GPUAV_ENABLE=1 for
// GPU-assisted validation of what a shader does at run time.
// Include after vulkan_loader.h (VK_NO_PROTOTYPES): every call goes through the vkGetInstanceProcAddr it loaded.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR static_cast<VkInstanceCreateFlags>(0x00000001u)
#endif

namespace leonida::vk_test {

inline void add_unique_name(std::vector<const char*>& names, const char* wanted) {
    for (const char* existing : names) {
        if (existing && std::strcmp(existing, wanted) == 0) return;
    }
    names.push_back(wanted);
}

// Enables VK_KHR_portability_enumeration and VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR when the loader lists
// the extension, so a portability ICD such as MoltenVK can be enumerated behind libvulkan on macOS. Safe no-op when
// the loader (or a direct ICD dylib) does not advertise the extension, and avoids duplicating an entry already in
// `extensions`. `extensions` must outlive vkCreateInstance.
inline bool enable_portability_enumeration(PFN_vkGetInstanceProcAddr gipa, VkInstanceCreateInfo& ici,
                                           std::vector<const char*>& extensions) {
    if (!gipa) return false;
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    if (!enumerate) return false;
    uint32_t count = 0;
    if (enumerate(nullptr, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkExtensionProperties> available(count);
    const VkResult res = enumerate(nullptr, &count, available.data());
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) return false;
    available.resize(count);
    for (const auto& ext : available) {
        if (std::strcmp(ext.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            add_unique_name(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            ici.enabledExtensionCount = uint32_t(extensions.size());
            ici.ppEnabledExtensionNames = extensions.data();
            return true;
        }
    }
    return false;
}

// True when `pd` advertises VK_KHR_portability_subset (VUID-VkDeviceCreateInfo-pProperties-04451).
inline bool has_portability_subset(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice pd) {
    if (!gipa || !pd) return false;
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(instance, "vkEnumerateDeviceExtensionProperties"));
    if (!enumerate) return false;
    uint32_t count = 0;
    if (enumerate(pd, nullptr, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkExtensionProperties> available(count);
    const VkResult res = enumerate(pd, nullptr, &count, available.data());
    if (res != VK_SUCCESS && res != VK_INCOMPLETE) return false;
    available.resize(count);
    for (const auto& ext : available) {
        if (std::strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0) return true;
    }
    return false;
}

class OptInValidation {
public:
    // Before vkCreateInstance: adds the layer, the extension and a messenger that also covers instance creation to
    // ici (keeping what it already enables). False, after a message, when validation was asked for but the layer
    // isn't installed: the caller should fail instead of running unchecked. This object must outlive the call.
    bool prepare(PFN_vkGetInstanceProcAddr gipa, VkInstanceCreateInfo& ici) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at startup
#endif
        const char* env = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        on_ = env && std::strcmp(env, "0") != 0;
        if (!on_) return true;
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            gipa(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
        uint32_t count = 0;
        if (enumerate) enumerate(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        if (count) enumerate(&count, available.data());
        bool found = false;
        for (const auto& properties : available) found = found || !std::strcmp(properties.layerName, layer_);
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer_);
            return false;
        }
        layers_.assign(ici.ppEnabledLayerNames, ici.ppEnabledLayerNames + ici.enabledLayerCount);
        add_unique_name(layers_, layer_);
        extensions_.assign(ici.ppEnabledExtensionNames, ici.ppEnabledExtensionNames + ici.enabledExtensionCount);
        add_unique_name(extensions_, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        enable_portability_enumeration(gipa, ici, extensions_);
        info_.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info_.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        info_.pfnUserCallback = on_message;
        info_.pNext = ici.pNext;
        ici.pNext = &info_;
        ici.enabledLayerCount = uint32_t(layers_.size());
        ici.ppEnabledLayerNames = layers_.data();
        ici.enabledExtensionCount = uint32_t(extensions_.size());
        ici.ppEnabledExtensionNames = extensions_.data();
        return true;
    }

    // Before vkCreateDevice: enables VK_KHR_portability_subset when validation is on and the physical device
    // advertises it (VUID-VkDeviceCreateInfo-pProperties-04451 on MoltenVK). This object must outlive vkCreateDevice.
    void prepare_device(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice pd,
                        VkDeviceCreateInfo& dci) {
        if (!on_ || !has_portability_subset(gipa, instance, pd)) return;
        device_extensions_.assign(dci.ppEnabledExtensionNames,
                                  dci.ppEnabledExtensionNames + dci.enabledExtensionCount);
        add_unique_name(device_extensions_, "VK_KHR_portability_subset");
        dci.enabledExtensionCount = uint32_t(device_extensions_.size());
        dci.ppEnabledExtensionNames = device_extensions_.data();
    }

    // After vkCreateInstance: the messenger for everything that follows. The program may exit without destroying it.
    VkResult attach(PFN_vkGetInstanceProcAddr gipa, VkInstance instance) {
        if (!on_) return VK_SUCCESS;
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (!create) return VK_ERROR_EXTENSION_NOT_PRESENT;
        info_.pNext = nullptr;
        return create(instance, &info_, nullptr, &messenger_);
    }

    bool enabled() const { return on_; }
    static unsigned errors() { return error_count(); }

    // The exit status with validation counted: a run that reported a validation error fails (1) even when its own
    // check passed or it reported itself unsupported (77); any other failure code is kept.
    static int exit_status(int rc) {
        if (!errors()) return rc;
        std::fprintf(stderr, "%u validation errors\n", errors());
        return rc != 0 && rc != 77 ? rc : 1;
    }

private:
    static unsigned& error_count() {
        static unsigned count = 0;
        return count;
    }

    // Only validation-type errors count: a broken third-party layer must not fail the test.
    static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT types,
                                                     const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
        const bool validation = (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0;
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && validation) ++error_count();
        std::fprintf(stderr, "%s: %s\n", validation ? "validation" : "vulkan (not counted)",
                     data && data->pMessage ? data->pMessage : "(no message)");
        return VK_FALSE;
    }

    bool on_ = false;
    const char* layer_ = "VK_LAYER_KHRONOS_validation";
    std::vector<const char*> layers_, extensions_, device_extensions_;
    VkDebugUtilsMessengerCreateInfoEXT info_{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

}  // namespace leonida::vk_test
