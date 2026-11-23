// CPU-only unit tests for OptInValidation and its portability helpers (vk_validation.h).
// Uses synthetic vkGetInstanceProcAddr callbacks so no Vulkan loader, layer or GPU is required.
#define VK_NO_PROTOTYPES
#include "vk_validation.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const char* what) {
    if (cond) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool contains_once(const char* const* names, std::uint32_t count, const char* wanted) {
    unsigned matches = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (names[i] && std::strcmp(names[i], wanted) == 0) ++matches;
    }
    return matches == 1;
}

struct FakeVulkanState {
    std::vector<const char*> layers;
    std::vector<const char*> instance_extensions;
    std::vector<const char*> device_extensions;
};

FakeVulkanState g_state;

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_layers(std::uint32_t* count, VkLayerProperties* props) {
    if (!props) {
        *count = std::uint32_t(g_state.layers.size());
        return VK_SUCCESS;
    }
    const std::uint32_t n = *count < g_state.layers.size() ? *count : std::uint32_t(g_state.layers.size());
    for (std::uint32_t i = 0; i < n; ++i) {
        props[i] = {};
        std::snprintf(props[i].layerName, sizeof props[i].layerName, "%s", g_state.layers[i]);
    }
    const bool incomplete = n < g_state.layers.size();
    *count = n;
    return incomplete ? VK_INCOMPLETE : VK_SUCCESS;
}

VkResult fill_extensions(const std::vector<const char*>& src, std::uint32_t* count, VkExtensionProperties* props) {
    if (!props) {
        *count = std::uint32_t(src.size());
        return VK_SUCCESS;
    }
    const std::uint32_t n = *count < src.size() ? *count : std::uint32_t(src.size());
    for (std::uint32_t i = 0; i < n; ++i) {
        props[i] = {};
        std::snprintf(props[i].extensionName, sizeof props[i].extensionName, "%s", src[i]);
    }
    const bool incomplete = n < src.size();
    *count = n;
    return incomplete ? VK_INCOMPLETE : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_instance_extensions(const char* layer_name, std::uint32_t* count,
                                                                  VkExtensionProperties* props) {
    if (layer_name) {
        *count = 0;
        return VK_SUCCESS;
    }
    return fill_extensions(g_state.instance_extensions, count, props);
}

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_device_extensions(VkPhysicalDevice, const char* layer_name,
                                                                std::uint32_t* count, VkExtensionProperties* props) {
    if (layer_name) {
        *count = 0;
        return VK_SUCCESS;
    }
    return fill_extensions(g_state.device_extensions, count, props);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_gipa(VkInstance, const char* name) {
    if (std::strcmp(name, "vkEnumerateInstanceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_enumerate_layers);
    if (std::strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_enumerate_instance_extensions);
    if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_enumerate_device_extensions);
    return nullptr;
}

void clear_validation_env() {
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_VALIDATION", "");
#else
    unsetenv("LEONIDA_VK_VALIDATION");
#endif
}

void set_validation_env(const char* value) {
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_VALIDATION", value);
#else
    setenv("LEONIDA_VK_VALIDATION", value, 1);
#endif
}

}  // namespace

int main() {
    using leonida::vk_test::OptInValidation;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* saved_ptr = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    const bool had_saved = saved_ptr != nullptr;
    const std::string saved_value = had_saved ? saved_ptr : "";

    int dummy_pd_storage = 0;
    const auto dummy_pd = reinterpret_cast<VkPhysicalDevice>(&dummy_pd_storage);
    int dummy_inst_storage = 0;
    const auto dummy_inst = reinterpret_cast<VkInstance>(&dummy_inst_storage);

    // 1. Validation unset and "0" are complete no-ops even when portability extensions exist.
    g_state = {{"VK_LAYER_KHRONOS_validation"}, {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME},
               {"VK_KHR_portability_subset"}};
    for (int mode = 0; mode < 2; ++mode) {
        if (mode == 0) clear_validation_env();
        else set_validation_env("0");
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        OptInValidation val;
        check(val.prepare(fake_gipa, ici), "prepare succeeds when validation is off");
        check(!val.enabled(), "enabled() is false when validation is off");
        check(ici.flags == 0 && ici.enabledLayerCount == 0 && ici.enabledExtensionCount == 0 && ici.pNext == nullptr,
              "ici is untouched when validation is off");
        val.prepare_device(fake_gipa, dummy_inst, dummy_pd, dci);
        check(dci.enabledExtensionCount == 0, "dci is untouched when validation is off");
    }

    // 2. Missing validation layer fails prepare().
    set_validation_env("1");
    g_state = {{}, {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME}, {}};
    {
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        OptInValidation val;
        check(!val.prepare(fake_gipa, ici), "prepare fails when validation is requested and layer is absent");
    }

    // 3. Loader without VK_KHR_portability_enumeration: enables validation + debug_utils only, no portability bit.
    g_state = {{"VK_LAYER_KHRONOS_validation"}, {"VK_KHR_surface"}, {"VK_KHR_shader_fma"}};
    {
        const char* caller_ext = "VK_KHR_surface";
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &caller_ext;
        OptInValidation val;
        check(val.prepare(fake_gipa, ici), "prepare succeeds without portability enumeration");
        check((ici.flags & VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR) == 0,
              "portability flag is not set when loader does not advertise the extension");
        check(ici.enabledLayerCount == 1 &&
                  contains_once(ici.ppEnabledLayerNames, ici.enabledLayerCount, "VK_LAYER_KHRONOS_validation"),
              "validation layer enabled once");
        check(ici.enabledExtensionCount == 2 &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount, "VK_KHR_surface") &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount,
                                VK_EXT_DEBUG_UTILS_EXTENSION_NAME),
              "caller extension preserved and debug_utils added");

        const char* fma = "VK_KHR_shader_fma";
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &fma;
        val.prepare_device(fake_gipa, dummy_inst, dummy_pd, dci);
        check(dci.enabledExtensionCount == 1 && dci.ppEnabledExtensionNames == &fma,
              "prepare_device is a no-op when VK_KHR_portability_subset is not advertised");
    }

    // 4. Loader with VK_KHR_portability_enumeration and device with VK_KHR_portability_subset (macOS MoltenVK).
    g_state = {{"VK_LAYER_KHRONOS_validation"},
               {"VK_EXT_metal_surface", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME},
               {"VK_KHR_shader_fma", "VK_KHR_portability_subset"}};
    {
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.flags = 0x4u;  // caller flag preserved alongside ENUMERATE_PORTABILITY_BIT_KHR
        OptInValidation val;
        check(val.prepare(fake_gipa, ici), "prepare succeeds with portability enumeration");
        check(ici.flags == (0x4u | VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR),
              "ENUMERATE_PORTABILITY_BIT_KHR added while keeping existing flags");
        check(ici.enabledExtensionCount == 2 &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount,
                                VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount,
                                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME),
              "debug_utils and VK_KHR_portability_enumeration both enabled");

        const char* fma = "VK_KHR_shader_fma";
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &fma;
        val.prepare_device(fake_gipa, dummy_inst, dummy_pd, dci);
        check(dci.enabledExtensionCount == 2 &&
                  contains_once(dci.ppEnabledExtensionNames, dci.enabledExtensionCount, "VK_KHR_shader_fma") &&
                  contains_once(dci.ppEnabledExtensionNames, dci.enabledExtensionCount, "VK_KHR_portability_subset"),
              "prepare_device keeps caller extensions and adds VK_KHR_portability_subset");
    }

    // 5. Deduplication when the caller already enabled the layer or extensions.
    {
        const char* pre_layers[] = {"VK_LAYER_KHRONOS_validation"};
        const char* pre_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = pre_layers;
        ici.enabledExtensionCount = 2;
        ici.ppEnabledExtensionNames = pre_exts;
        OptInValidation val;
        check(val.prepare(fake_gipa, ici), "prepare succeeds with pre-populated entries");
        check(ici.enabledLayerCount == 1 &&
                  contains_once(ici.ppEnabledLayerNames, ici.enabledLayerCount, "VK_LAYER_KHRONOS_validation"),
              "no duplicate validation layer");
        check(ici.enabledExtensionCount == 2 &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount,
                                VK_EXT_DEBUG_UTILS_EXTENSION_NAME) &&
                  contains_once(ici.ppEnabledExtensionNames, ici.enabledExtensionCount,
                                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME),
              "no duplicate instance extensions");

        const char* pre_dev_exts[] = {"VK_KHR_portability_subset"};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = pre_dev_exts;
        val.prepare_device(fake_gipa, dummy_inst, dummy_pd, dci);
        check(dci.enabledExtensionCount == 1 &&
                  contains_once(dci.ppEnabledExtensionNames, dci.enabledExtensionCount, "VK_KHR_portability_subset"),
              "no duplicate VK_KHR_portability_subset");
    }

    if (had_saved) set_validation_env(saved_value.c_str());
    else clear_validation_env();

    if (failures) {
        std::fprintf(stderr, "%d vk_validation test(s) failed\n", failures);
        return 1;
    }
    std::printf("vk_validation: all cases passed\n");
    return 0;
}
