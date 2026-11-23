// Unit tests for the OpFmaKHR width gate in vk_shader_fma.h (no Vulkan instance, no GPU). A module built with
// `rdna2spirv.py --fma khr` needs shaderFmaFloat16, shaderFmaFloat32 or shaderFmaFloat64 for each OpFmaKHR result width;
// a module that declares Float64 for other doubles but has only f32 OpFmaKHRs needs only shaderFmaFloat32.
#include "vk_shader_fma.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t word0(std::uint32_t op, std::uint32_t count) { return count << 16 | op; }

// A minimal module: header, OpCapability FMAKHR (+ Float64), float types, and OpFmaKHRs of the given result types.
// Ids: 1 = f32, 2 = f64, 3 = v2f32, 4 = v2f64, 5 = f16, 6 = v2f16; FMA operands are ids 10..12.
std::vector<std::uint32_t> module(bool fma_cap, bool float64_cap, const std::vector<std::uint32_t>& fma_types) {
    std::vector<std::uint32_t> w = {0x07230203, 0x00010500, 0, 100, 0};
    if (fma_cap) w.insert(w.end(), {word0(17, 2), 6030});
    if (float64_cap) w.insert(w.end(), {word0(17, 2), 10});
    w.insert(w.end(), {
        word0(22, 3), 1, 32,
        word0(22, 3), 2, 64,
        word0(23, 4), 3, 1, 2,
        word0(23, 4), 4, 2, 2,
        word0(22, 3), 5, 16,
        word0(23, 4), 6, 5, 2,
    });
    std::uint32_t id = 20;
    for (const std::uint32_t t : fma_types) w.insert(w.end(), {word0(4427, 6), t, id++, 10, 11, 12});
    return w;
}

bool accepts(const leonida::vk_test::ShaderFma& fma, const std::vector<std::uint32_t>& m) {
    return fma.accepts(m.data(), m.size() * sizeof(m.front()), "test.spv");
}

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
struct MockVkState {
    VkResult count_result = VK_SUCCESS;
    VkResult list_result = VK_SUCCESS;
    bool list_fma = true;
    VkBool32 f16 = VK_FALSE;
    VkBool32 f32 = VK_FALSE;
    VkBool32 f64 = VK_FALSE;
} g_mock;

VKAPI_ATTR VkResult VKAPI_CALL mock_enumerate(
    VkPhysicalDevice, const char*, std::uint32_t* count, VkExtensionProperties* props) {
    if (!props) {
        *count = g_mock.list_fma ? 1u : 0u;
        return g_mock.count_result;
    }
    if (g_mock.list_result != VK_SUCCESS) return g_mock.list_result;
    if (g_mock.list_fma && *count >= 1) {
        std::memset(&props[0], 0, sizeof(props[0]));
        std::snprintf(props[0].extensionName, sizeof(props[0].extensionName), "%s", VK_KHR_SHADER_FMA_EXTENSION_NAME);
        *count = 1;
        return VK_SUCCESS;
    }
    *count = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_features2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* f2) {
    for (auto* s = static_cast<VkBaseOutStructure*>(f2 ? f2->pNext : nullptr); s; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR) {
            auto* fma = reinterpret_cast<VkPhysicalDeviceShaderFmaFeaturesKHR*>(s);
            fma->shaderFmaFloat16 = g_mock.f16;
            fma->shaderFmaFloat32 = g_mock.f32;
            fma->shaderFmaFloat64 = g_mock.f64;
        }
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_gipa(VkInstance, const char* name) {
    if (!name) return nullptr;
    if (!std::strcmp(name, "vkEnumerateDeviceExtensionProperties"))
        return reinterpret_cast<PFN_vkVoidFunction>(&mock_enumerate);
    if (!std::strcmp(name, "vkGetPhysicalDeviceFeatures2"))
        return reinterpret_cast<PFN_vkVoidFunction>(&mock_features2);
    return nullptr;
}
#endif

}  // namespace

int main() {
    using leonida::vk_test::ShaderFma;
    using leonida::vk_test::spirv_declares_capability;
    using leonida::vk_test::spirv_fma_widths;

    bool f16 = false, f32 = false, f64 = false;
    auto m = module(true, true, {1});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(f32 && !f64, "an f32 OpFmaKHR is width 32");
    m = module(true, true, {2});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(!f32 && f64, "an f64 OpFmaKHR is width 64");
    m = module(true, true, {4, 3});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(f32 && f64, "vector results take their component width");

    m = module(true, false, {5});
    spirv_fma_widths(m.data(), m.size() * 4, f16, f32, f64);
    expect(f16 && !f32 && !f64, "an f16 OpFmaKHR is width 16");
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(!f32 && !f64, "4-arg spirv_fma_widths does not misclassify f16 as f32");

    m = module(true, false, {6});
    spirv_fma_widths(m.data(), m.size() * 4, f16, f32, f64);
    expect(f16 && !f32 && !f64, "a v2f16 OpFmaKHR takes its f16 component width");

    m = module(true, false, {99});
    spirv_fma_widths(m.data(), m.size() * 4, f16, f32, f64);
    expect(!f16 && f32 && !f64, "an unknown OpFmaKHR result type defaults to width 32");

    ShaderFma only16;
    only16.float16 = true;
    ShaderFma only32;
    only32.float32 = true;
    ShaderFma only64;
    only64.float64 = true;
    ShaderFma both;
    both.float32 = both.float64 = true;
    ShaderFma all3;
    all3.float16 = all3.float32 = all3.float64 = true;
    const ShaderFma none;

    expect(accepts(none, module(false, true, {})), "a module without FMAKHR needs nothing");
    expect(accepts(only32, module(true, false, {1})), "f32 OpFmaKHR on a shaderFmaFloat32 device");
    expect(accepts(only32, module(true, true, {1})), "Float64 elsewhere + f32 OpFmaKHR needs only shaderFmaFloat32");
    expect(!accepts(only32, module(true, true, {2})), "f64 OpFmaKHR is refused without shaderFmaFloat64");
    expect(!accepts(only32, module(true, true, {1, 2})), "mixed widths need both");
    expect(accepts(only64, module(true, true, {2})), "f64 OpFmaKHR on a shaderFmaFloat64 device");
    expect(!accepts(only64, module(true, false, {1})), "f32 OpFmaKHR is refused without shaderFmaFloat32");
    expect(accepts(both, module(true, true, {1, 2})), "both widths on a device with both");
    expect(!accepts(none, module(true, false, {1})), "no VK_KHR_shader_fma refuses any OpFmaKHR");
    expect(!accepts(only64, module(true, false, {})), "FMAKHR without an OpFmaKHR keeps needing shaderFmaFloat32");
    expect(only16.supported() && only32.supported() && only64.supported() && !none.supported(),
           "supported() covers float16, float32 and float64");

    expect(accepts(only16, module(true, false, {5})), "f16 OpFmaKHR on a shaderFmaFloat16 device");
    expect(accepts(only16, module(true, false, {6})), "v2f16 OpFmaKHR on a shaderFmaFloat16 device");
    expect(!accepts(only32, module(true, false, {5})), "f16 OpFmaKHR is refused on a shaderFmaFloat32-only device");
    expect(!accepts(both, module(true, false, {6})), "v2f16 OpFmaKHR is refused without shaderFmaFloat16");
    expect(!accepts(only16, module(true, false, {5, 1})), "mixed f16+f32 OpFmaKHRs need both features");
    expect(!accepts(only16, module(true, false, {})), "FMAKHR without an OpFmaKHR still needs shaderFmaFloat32");
    expect(accepts(all3, module(true, true, {5, 1, 2})), "all three widths pass when float16/32/64 are enabled");

    // Malformed / boundary SPIR-V inputs.
    expect(!spirv_declares_capability(nullptr, 24, 6030), "null data declares no capability");
    expect(!spirv_declares_capability(m.data(), 16, 6030), "header shorter than 20 bytes declares no capability");
    expect(!spirv_declares_capability(m.data(), 22, 6030), "unaligned byte count declares no capability");
    f16 = f32 = f64 = true;
    spirv_fma_widths(nullptr, 24, f16, f32, f64);
    expect(!f16 && !f32 && !f64, "null data resets all fma width flags");

    auto bad_magic = module(true, true, {1, 2, 5});
    bad_magic[0] = 0xDEADBEEFu;
    expect(!spirv_declares_capability(bad_magic.data(), bad_magic.size() * 4, 6030), "non-SPIR-V magic is rejected");
    f16 = f32 = f64 = true;
    spirv_fma_widths(bad_magic.data(), bad_magic.size() * 4, f16, f32, f64);
    expect(!f16 && !f32 && !f64, "non-SPIR-V magic reports no fma widths");

    const std::vector<std::uint32_t> trunc_cap = {0x07230203, 0x00010500, 0, 10, 0, word0(17, 2)};
    expect(!spirv_declares_capability(trunc_cap.data(), trunc_cap.size() * 4, 6030),
           "truncated OpCapability does not read past end");
    const std::vector<std::uint32_t> huge_count = {
        0x07230203, 0x00010500, 0, 10, 0, word0(17, 0xffff), 6030, word0(17, 2), 6030};
    expect(!spirv_declares_capability(huge_count.data(), huge_count.size() * 4, 6030),
           "oversized instruction word count stops capability scan safely");
    auto trunc_fma = module(true, false, {});
    trunc_fma.insert(trunc_fma.end(), {word0(4427, 6), 1, 20});  // missing operands
    spirv_fma_widths(trunc_fma.data(), trunc_fma.size() * 4, f16, f32, f64);
    expect(!f16 && !f32 && !f64, "truncated OpFmaKHR stops width scan safely");

    // Null name and null link_after guards.
    m = module(true, false, {1});
    expect(!none.accepts(m.data(), m.size() * 4, nullptr), "accepts() handles null module name safely");
    ShaderFma link_check = only32;
    link_check.link_after(nullptr);

    // ShaderFma::query() state reset and feature population.
    int dummy_inst_storage = 1, dummy_pd_storage = 2;
    const auto inst = reinterpret_cast<VkInstance>(&dummy_inst_storage);
    const auto pd = reinterpret_cast<VkPhysicalDevice>(&dummy_pd_storage);

    ShaderFma queried;
    queried.float16 = queried.float32 = queried.float64 = true;
    queried.query(nullptr, inst, pd);
    expect(!queried.supported(), "query(nullptr, ...) clears stale flags");

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    queried.float16 = queried.float32 = queried.float64 = true;
    queried.query(&mock_gipa, inst, VK_NULL_HANDLE);
    expect(!queried.supported(), "query(..., VK_NULL_HANDLE) clears stale flags");

    g_mock = {VK_ERROR_INITIALIZATION_FAILED, VK_SUCCESS, true, VK_TRUE, VK_TRUE, VK_TRUE};
    queried.float16 = queried.float32 = queried.float64 = true;
    queried.query(&mock_gipa, inst, pd);
    expect(!queried.supported(), "failed extension count query clears stale flags");

    g_mock = {VK_SUCCESS, VK_INCOMPLETE, true, VK_TRUE, VK_TRUE, VK_TRUE};
    queried.float16 = queried.float32 = queried.float64 = true;
    queried.query(&mock_gipa, inst, pd);
    expect(!queried.supported(), "failed extension list query clears stale flags");

    g_mock = {VK_SUCCESS, VK_SUCCESS, false, VK_TRUE, VK_TRUE, VK_TRUE};
    queried.float16 = queried.float32 = queried.float64 = true;
    queried.query(&mock_gipa, inst, pd);
    expect(!queried.supported(), "unlisted VK_KHR_shader_fma leaves features disabled");

    g_mock = {VK_SUCCESS, VK_SUCCESS, true, VK_TRUE, VK_TRUE, VK_FALSE};
    queried.query(&mock_gipa, inst, pd);
    expect(queried.float16 && queried.float32 && !queried.float64 && queried.features.pNext == nullptr,
           "query() records shaderFmaFloat16/32/64 from physical device features");

    g_mock = {VK_SUCCESS, VK_SUCCESS, true, VK_FALSE, VK_TRUE, VK_TRUE};
    queried.query(&mock_gipa, inst, pd);
    expect(!queried.float16 && queried.float32 && queried.float64,
           "re-querying a second physical device replaces prior feature flags");

    VkBaseOutStructure tail{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr};
    VkBaseOutStructure head{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &tail};
    queried.link_after(&head);
    expect(head.pNext == reinterpret_cast<VkBaseOutStructure*>(&queried.features) &&
               queried.features.pNext == &tail,
           "link_after() splices features into pNext chain when supported()");
#endif

    if (failures) return 1;
    std::puts("vk_shader_fma: all checks passed");
    return 0;
}
