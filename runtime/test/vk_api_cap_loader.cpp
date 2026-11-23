// A stand-in for the Vulkan loader (libvulkan.so.1) that makes the real one look older, so the test tools can be run
// against a Vulkan 1.0 or 1.1 instance on a machine whose GPU is newer (TEST-1110).
//
//   LEONIDA_VK_API_CAP=1.0|1.1|1.2   the version to pretend (default: 1.0)
//   LEONIDA_VK_REAL_LOADER=<path>    the real loader (default: libvulkan.so.1 from the system paths)
//
// Put the directory of this library first on LD_LIBRARY_PATH: the tools dlopen("libvulkan.so.1") and take
// vkGetInstanceProcAddr from it. Every call goes on to the real loader (and to the layers, such as
// VK_LAYER_KHRONOS_validation), except that
//   - vkCreateInstance asks for at most the capped apiVersion, so the layers judge the tool by that version;
//   - vkGetPhysicalDeviceProperties reports at most the capped apiVersion;
//   - with 1.0, the core 1.1 instance entry points (vkGetPhysicalDeviceFeatures2 and its siblings) are null, as
//     on a Vulkan 1.0 loader.
// A tool that chains Vulkan 1.1/1.2 structures without looking at apiVersion then shows a validation error, or
// crashes on the null function; a tool that checks exits 77.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

PFN_vkGetInstanceProcAddr real_gipa = nullptr;
VkInstance last_instance = VK_NULL_HANDLE;
std::uint32_t cap = VK_API_VERSION_1_0;

void init() {
    if (real_gipa) return;
    if (const char* v = std::getenv("LEONIDA_VK_API_CAP")) {
        if (!std::strcmp(v, "1.1")) cap = VK_API_VERSION_1_1;
        else if (!std::strcmp(v, "1.2")) cap = VK_API_VERSION_1_2;
    }
    const char* path = std::getenv("LEONIDA_VK_REAL_LOADER");
    void* library = dlopen(path && *path ? path : "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "vk_api_cap_loader: cannot load the real loader: %s\n", dlerror());
        std::exit(1);
    }
    real_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
}

VKAPI_ATTR VkResult VKAPI_CALL capped_create_instance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* alloc,
                                                      VkInstance* instance) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    if (info->pApplicationInfo) app = *info->pApplicationInfo;
    if (app.apiVersion == 0 || app.apiVersion > cap) app.apiVersion = cap;
    VkInstanceCreateInfo changed = *info;
    changed.pApplicationInfo = &app;
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(real_gipa(nullptr, "vkCreateInstance"));
    const VkResult result = create(&changed, alloc, instance);
    if (result == VK_SUCCESS) last_instance = *instance;
    return result;
}

VKAPI_ATTR void VKAPI_CALL capped_properties(VkPhysicalDevice device, VkPhysicalDeviceProperties* properties) {
    const auto real = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        real_gipa(last_instance, "vkGetPhysicalDeviceProperties"));
    real(device, properties);
    if (properties->apiVersion > cap) properties->apiVersion = cap;
}

// The instance-level functions Vulkan 1.1 made core.
bool core11_instance_function(const char* name) {
    static const char* const names[] = {
        "vkGetPhysicalDeviceFeatures2", "vkGetPhysicalDeviceProperties2", "vkGetPhysicalDeviceFormatProperties2",
        "vkGetPhysicalDeviceImageFormatProperties2", "vkGetPhysicalDeviceQueueFamilyProperties2",
        "vkGetPhysicalDeviceMemoryProperties2", "vkGetPhysicalDeviceSparseImageFormatProperties2",
        "vkGetPhysicalDeviceExternalBufferProperties", "vkGetPhysicalDeviceExternalFenceProperties",
        "vkGetPhysicalDeviceExternalSemaphoreProperties", "vkEnumeratePhysicalDeviceGroups",
        "vkEnumerateInstanceVersion"};
    for (const char* n : names)
        if (!std::strcmp(n, name)) return true;
    return false;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance,
                                                                                            const char* name) {
    init();
    if (!std::strcmp(name, "vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(capped_create_instance);
    if (!std::strcmp(name, "vkGetPhysicalDeviceProperties")) return reinterpret_cast<PFN_vkVoidFunction>(capped_properties);
    if (cap == VK_API_VERSION_1_0 && core11_instance_function(name)) return nullptr;
    return real_gipa(instance, name);
}
