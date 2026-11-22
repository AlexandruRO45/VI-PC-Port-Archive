// G-buffer sandbox: draws a tools/sandbox_scene.py scene through a translated deferred material pair (a VS that
// pulls its own vertices and a G-buffer PS) and reads every target back.
//
//   vk_gbuffer <scene.bin> <vs.spv> <ps.spv> --link L0,L1,.. [--size W H] [--cam x y z yaw pitch] [--fov deg]
//              [--dump prefix]
//
// The pair (chosen by tools/gbuffer.py) reads user SGPRs, not vertex buffers. The VS:
//   s8  table A: +0x6e0 V# frame constants (0x52c: 1.0 turns off a depth bias), +0x1780 V# a 12-byte record store
//   s9  table B (per draw): +0x60 V# viewport constants (width, height at 0x44), +0x80 V# object constants
//       (world 4x4 at 0x00, previous world at 0x40, row vectors), +0xa0 V# view constants (view-projection at
//       0x160, previous view-projection at 0x1a0)
//   s11 vertex stream base. The shader builds the stream V# itself; the stream starts with a header of attribute
//       words (offset << 7 | stride << 3 | format): 0x0 position, 0x8 normal, 0xc uv; 0x60 position bias xyz,
//       axis mode, scale xyz, uv scale (per field). Positions are three fields: 24-bit in dword 0 bits 8-31,
//       24-bit in dword 0 bits 0-7 plus dword 1 bits 0-15, 16-bit in dword 1 bits 16-31; the VS weights world
//       row 2 by the first and row 0 by the third, so they hold model z, y, x. Normals 10:10 octahedral with bit
//       31 set, uvs float2.
//   s14 index V# slot (>= 0: indexed), s18 table of index V#s (R32_UINT, stride 4). The draw is non-indexed:
//       the vertex id selects the index.
// The PS: s8 table A (+0x6e0 frame constants, +0xac0 V# of a dword table indexed by s14 that picks the T#),
// s9 table B (+0x20 V# material bytes, +0x60 viewport constants), s15 the T# table, s16 + 0x520 the S#.
// Its targets: mrt0..3 R8G8B8A8_UNORM, mrt4 R16G16_SFLOAT (docs/gbuffer.md).
//
// The game links a PS input to a VS output by semantic (SPI_PS_INPUT_CNTL); the translator numbers VS outputs
// by param and PS inputs by attribute, so --link gives, per PS attribute, the VS param it reads, and the PS's
// Input Location decorations are rewritten to it.
//
// Prints counts only: covered pixels, distinct albedo colours, decoded normals facing the camera, and motion
// vectors (zero for a still camera). Nothing here is game data: the scene and the shaders come from the user's
// work folder; --dump writes PPMs of the targets there.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"
#include "sandbox_common.h"

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);   \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

using namespace leonida::sandbox;

// Rewrites the Location of every Input variable of a SPIR-V module: location l becomes link[l].
bool relink_inputs(std::vector<uint32_t>& w, const std::vector<uint32_t>& link) {
    if (w.size() < 5 || w[0] != 0x07230203u) return false;
    std::set<uint32_t> inputs;
    for (size_t i = 5; i < w.size();) {
        const uint32_t op = w[i] & 0xFFFF, n = w[i] >> 16;
        if (!n || i + n > w.size()) return false;
        if (op == 59 && n >= 4 && w[i + 3] == 1) inputs.insert(w[i + 2]);   // OpVariable ... Input
        i += n;
    }
    for (size_t i = 5; i < w.size();) {
        const uint32_t op = w[i] & 0xFFFF, n = w[i] >> 16;
        if (op == 71 && n == 4 && w[i + 2] == 30 && inputs.count(w[i + 1])) {   // OpDecorate id Location l
            if (w[i + 3] >= link.size()) return false;
            w[i + 3] = link[w[i + 3]];
        }
        i += n;
    }
    return true;
}

uint32_t oct10(const float n[3]) {
    const float l1 = std::fabs(n[0]) + std::fabs(n[1]) + std::fabs(n[2]);
    float u = l1 > 0 ? n[0] / l1 : 0, v = l1 > 0 ? n[1] / l1 : 0;
    if (l1 > 0 && n[2] < 0) {
        const float pu = (1 - std::fabs(v)) * (u >= 0 ? 1 : -1), pv = (1 - std::fabs(u)) * (v >= 0 ? 1 : -1);
        u = pu; v = pv;
    }
    auto q = [](float x) { return uint32_t(std::clamp(std::lround((x + 1) * 511.5f), 0l, 1023l)); };
    return 0x80000000u | q(u) | (q(v) << 10);
}

// The PS's 24-bit octahedral normal from mrt1 (g, b, a bytes; 12 bits each axis), as a unit vector.
void decode_normal(const uint8_t* p, float n[3]) {
    const uint32_t q = p[1] | (uint32_t(p[2]) << 8) | (uint32_t(p[3]) << 16);
    float x = float(q & 0xFFF) / 2047.5f - 1, y = float(q >> 12) / 2047.5f - 1, z = 1 - std::fabs(x) - std::fabs(y);
    const float t = std::max(-z, 0.f);
    x += x >= 0 ? -t : t; y += y >= 0 ? -t : t;
    const float l = std::sqrt(x * x + y * y + z * z);
    n[0] = x / l; n[1] = y / l; n[2] = z / l;
}

float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vk_gbuffer scene.bin vs.spv ps.spv --link L0,L1,.. [--size W H] [--cam x y z yaw pitch] "
                             "[--fov deg] [--dump prefix]\n");
        return 2;
    }
    Scene scene;
    if (!load_scene(argv[1], scene)) { std::fprintf(stderr, "bad scene file %s\n", argv[1]); return 2; }
    uint32_t W = 640, H = 360;
    std::vector<uint32_t> link;
    const char* dump = nullptr;
    Camera cam{};
    cam.pos[0] = scene.center[0] - 40; cam.pos[1] = scene.center[1] - 40; cam.pos[2] = scene.center[2] + 25;
    cam.yaw = 0.785398f; cam.pitch = -0.35f;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--link") && i + 1 < argc) {
            for (const char* p = argv[++i]; *p;) {
                char* end; link.push_back(uint32_t(std::strtoul(p, &end, 10)));
                if (end == p) { std::fprintf(stderr, "bad --link\n"); return 2; }
                p = *end == ',' ? end + 1 : end;
            }
        }
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!std::strcmp(argv[i], "--size") && i + 2 < argc) { W = std::atoi(argv[++i]); H = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--fov") && i + 1 < argc) cam.fov = float(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--cam") && i + 5 < argc) {
            for (int k = 0; k < 3; ++k) cam.pos[k] = float(std::atof(argv[++i]));
            cam.yaw = float(std::atof(argv[++i])) * 3.14159265f / 180; cam.pitch = float(std::atof(argv[++i])) * 3.14159265f / 180;
        } else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (link.empty()) { std::fprintf(stderr, "give --link\n"); return 2; }
    if (!W || !H || W > 8192 || H > 8192) { std::fprintf(stderr, "bad --size\n"); return 2; }
    auto vs_code = read_spv(argv[2]), ps_code = read_spv(argv[3]);
    if (vs_code.empty() || ps_code.empty()) { std::fprintf(stderr, "cannot read the shaders\n"); return 2; }
    if (!relink_inputs(ps_code, link)) { std::fprintf(stderr, "cannot relink the PS inputs\n"); return 2; }
    std::fprintf(stderr, "scene: %zu draws, %zu meshes, %zu textures, %zu triangles\n", scene.draws.size(),
                 scene.meshes.size(), scene.textures.size(), scene.indices.size() / 3);

    gipa = load_vulkan();
    if (!gipa) { std::fprintf(stderr, "the Vulkan loader was not found\n"); return 77; }
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { std::fprintf(stderr, "unsupported: no Vulkan instance\n"); return 77; }
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkCreateDevice); LI(vkGetDeviceProcAddr); LI(vkGetPhysicalDeviceFeatures2);
    LI(vkEnumerateDeviceExtensionProperties);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    if (!n) { std::fprintf(stderr, "unsupported: no Vulkan physical device\n"); return 77; }
    std::vector<VkPhysicalDevice> pds(n);
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    int best = -1;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (int r = leonida::vk_test::device_rank(pp.deviceType, pp.vendorID); r > best) { pd = p; props = pp; best = r; }
    }
    constexpr uint32_t push_words = 36, push_bytes = push_words * 4;
    if (props.limits.maxPushConstantsSize < push_bytes) { std::fprintf(stderr, "unsupported: push constants\n"); return 77; }
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    // The PS reads its f16 interpolants per vertex (PerVertexKHR); the VS subtracts BaseVertex (shaderDrawParameters).
    uint32_t en = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en); vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    bool has_bary = false;
    for (const auto& e : exts) has_bary |= !std::strcmp(e.extensionName, "VK_KHR_fragment_shader_barycentric");
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; q12.pNext = has_bary ? &bq : nullptr;
    VkPhysicalDeviceVulkan11Features q11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; q11.pNext = &q12;
    VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; qf2.pNext = &q11;
    vkGetPhysicalDeviceFeatures2(pd, &qf2);
    if (!has_bary || !bq.fragmentShaderBarycentric || !q11.shaderDrawParameters || !q12.bufferDeviceAddress ||
        !q12.runtimeDescriptorArray || !q12.descriptorIndexing || !q12.shaderSampledImageArrayNonUniformIndexing ||
        !qf2.features.shaderInt64) {
        std::fprintf(stderr, "unsupported: needs VK_KHR_fragment_shader_barycentric, shaderDrawParameters, "
                             "bufferDeviceAddress, descriptor indexing and shaderInt64\n");
        return 77;
    }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    if (qi == qn) { std::fprintf(stderr, "unsupported: no graphics queue\n"); return 77; }

    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    fb.fragmentShaderBarycentric = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext = &fb;
    f12.bufferDeviceAddress = VK_TRUE; f12.runtimeDescriptorArray = VK_TRUE; f12.descriptorIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext = &f12;
    f11.shaderDrawParameters = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f11;
    f2.features.shaderInt64 = VK_TRUE;
    const char* dev_ext[1] = {"VK_KHR_fragment_shader_barycentric"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dev_ext;
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LD(n) auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory);
    LD(vkBindImageMemory); LD(vkCreateImageView); LD(vkCreateRenderPass); LD(vkCreateFramebuffer);
    LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline);
    LD(vkCmdDraw); LD(vkCmdEndRenderPass); LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage);
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCreateDescriptorSetLayout); LD(vkCreateDescriptorPool); LD(vkAllocateDescriptorSets);
    LD(vkUpdateDescriptorSets); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b); if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext = &fi; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m; r = vkAllocateMemory(dev, &ai, nullptr, &m); if (r) return r;
        r = vkBindBufferMemory(dev, *b, m, 0); if (r) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };
    auto make_image = [&](uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage* img, VkImageView* view) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
        VkResult r = vkCreateImage(dev, &ii, nullptr, img); if (r) return r;
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, *img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; r = vkAllocateMemory(dev, &ai, nullptr, &m); if (r) return r;
        r = vkBindImageMemory(dev, *img, m, 0); if (r) return r;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = *img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {aspect, 0, 1, 0, 1};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };

    // Heap: the Xbox GPU VA window, laid out as the pair loads it (see the header).
    const uint32_t ndraw = uint32_t(scene.draws.size()), nmesh = uint32_t(scene.meshes.size());
    const uint32_t table_a = 0x0, frame_va = 0x2000, material_va = 0x2800, store_va = 0x2900, view_va = 0x4000,
                   viewport_va = 0x4200, bytes_va = 0x4300, sampler_va = 0x4400;
    const uint64_t layout_end = 0x8000 + uint64_t(ndraw) * 0x1a0 + uint64_t(nmesh) * 0x20;
    if (layout_end > 0xFFFFFFFFull) { std::fprintf(stderr, "scene too large for a 32-bit heap\n"); return 2; }
    const uint32_t table_b = 0x8000, object_va = table_b + ndraw * 0x100, tsharp_va = object_va + ndraw * 0x80,
                   index_vsharp_va = tsharp_va + ndraw * 0x20, data_va = index_vsharp_va + nmesh * 0x20;
    std::vector<uint32_t> stream_va(nmesh), index_va(nmesh);
    uint64_t at = data_va;
    for (uint32_t m = 0; m < nmesh; ++m) {
        const Mesh& me = scene.meshes[m];
        if (0x80 + uint64_t(me.vertex_count) * 12 >= (1u << 25)) {   // attribute word: offset << 7 in 32 bits
            std::fprintf(stderr, "mesh %u has too many vertices for the stream header\n", m);
            return 2;
        }
        stream_va[m] = uint32_t(at); at += (0x80 + uint64_t(me.vertex_count) * 20 + 255) & ~uint64_t(255);
        index_va[m] = uint32_t(at); at += (uint64_t(me.index_count) * 4 + 255) & ~uint64_t(255);
    }
    if (at + 0x1000 > 0xFFFFFFFFull) { std::fprintf(stderr, "scene too large for a 32-bit heap\n"); return 2; }
    const VkDeviceSize heap_bytes = at + 0x1000;
    VkBuffer heap; void* heapp;
    CHECK(make_buffer(heap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heapp));
    std::memset(heapp, 0, size_t(heap_bytes));
    auto heap_u32 = [&](uint32_t va) { return reinterpret_cast<uint32_t*>(static_cast<char*>(heapp) + va); };
    auto heap_f32 = [&](uint32_t va) { return reinterpret_cast<float*>(static_cast<char*>(heapp) + va); };
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bdai.buffer = heap;
    const VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &bdai);
    auto vsharp = [&](uint32_t at_va, uint32_t base, uint32_t bytes) {   // raw: stride 0, OOB_SELECT 3
        uint32_t* p = heap_u32(at_va); p[0] = base; p[1] = 0; p[2] = bytes; p[3] = 3u << 28;
    };
    // Table A.
    vsharp(table_a + 0x6e0, frame_va, 0x600);
    vsharp(table_a + 0xac0, material_va, 0x10);
    { uint32_t* p = heap_u32(table_a + 0x1780); p[0] = store_va; p[1] = 12u << 16; p[2] = 64; p[3] = 0x11014facu; }
    *heap_f32(frame_va + 0x52c) = 1.0f;
    heap_f32(viewport_va + 0x44)[0] = float(W); heap_f32(viewport_va + 0x44)[1] = float(H);
    // T# keys: unique words per texture with the RGBA8 UNORM format in word 1.
    const uint32_t fmt_word = leonida::gfx10::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM) << 20;
    auto tsharp = [&](uint32_t image) {
        std::array<uint32_t, 8> t{0x5A000000u | image, fmt_word, 0x7FF7FFu, 0, 0, 0, 0, 0};
        return t;
    };
    const uint32_t nimg = uint32_t(scene.textures.size()) + 1;   // image 0: 1x1 white for untextured draws
    leonida::DescriptorTable table(2 * (nimg + 2));
    for (uint32_t i = 0; i < nimg; ++i) table.insert(tsharp(i).data(), i);
    // A linear, wrapping sampler; the S# is read from memory (s16 + 0x520).
    uint32_t ssharp[8] = {1024, 12582912, 418381824, 0, 0, 0, 0, 0};
    std::memcpy(heap_u32(sampler_va), ssharp, 16);
    table.insert(ssharp, 0);
    for (uint32_t d = 0; d < ndraw; ++d) {
        const Draw& dr = scene.draws[d];
        const uint32_t b = table_b + d * 0x100, o = object_va + d * 0x80;
        vsharp(b + 0x20, bytes_va, 0x20);
        vsharp(b + 0x60, viewport_va, 0x50);
        vsharp(b + 0x80, o, 0x80);
        vsharp(b + 0xa0, view_va, 0x1e0);
        const auto ts = tsharp(dr.texture < 0 ? 0 : uint32_t(dr.texture) + 1);
        std::memcpy(heap_u32(tsharp_va + d * 0x20), ts.data(), 32);
        float* ob = heap_f32(o);
        for (int m = 0; m < 2; ++m)   // world, then previous world (the same: nothing moves)
            for (int r = 0; r < 4; ++r) for (int j = 0; j < 4; ++j) ob[m * 16 + r * 4 + j] = j < 3 ? dr.rows[r][j] : r == 3;
    }
    // Vertex streams and index V#s.
    for (uint32_t m = 0; m < nmesh; ++m) {
        const Mesh& me = scene.meshes[m];
        const float* v = scene.vertices.data() + size_t(me.first_vertex) * 8;
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (uint32_t i = 0; i < me.vertex_count; ++i)
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], v[i * 8 + k]); hi[k] = std::max(hi[k], v[i * 8 + k]); }
        // The VS takes model x from the 16-bit field, y from the split 24-bit field and z from the 24-bit field in
        // dword 0 (it weights world row 0 by the third decoded value and row 2 by the first).
        const float range[3] = {65535.f, 16777215.f, 16777215.f};
        float scale[3];
        for (int k = 0; k < 3; ++k) scale[k] = hi[k] > lo[k] ? (hi[k] - lo[k]) / range[k] : 1.f;
        const uint32_t pos_off = 0x80, nrm_off = pos_off + me.vertex_count * 8, uv_off = nrm_off + me.vertex_count * 4;
        uint32_t* hdr = heap_u32(stream_va[m]);
        hdr[0] = (pos_off << 7) | (8u << 3);
        hdr[2] = (nrm_off << 7) | (4u << 3);
        hdr[3] = (uv_off << 7) | (8u << 3) | 2u;   // format 2: float2
        float* hf = heap_f32(stream_va[m] + 0x60);
        hf[0] = lo[2]; hf[1] = lo[1]; hf[2] = lo[0]; hdr[0x6c / 4] = 0;   // bias per field, axis mode 0
        hf[4] = scale[2]; hf[5] = scale[1]; hf[6] = scale[0]; hf[7] = 1.f;   // scale per field, uv scale
        uint32_t* pos = heap_u32(stream_va[m] + pos_off);
        uint32_t* nrm = heap_u32(stream_va[m] + nrm_off);
        float* uv = heap_f32(stream_va[m] + uv_off);
        for (uint32_t i = 0; i < me.vertex_count; ++i) {
            const float* p = v + i * 8;
            uint32_t qv[3];
            for (int k = 0; k < 3; ++k)
                qv[k] = uint32_t(std::clamp(std::lround((p[k] - lo[k]) / scale[k]), 0l, long(range[k])));
            pos[i * 2] = (qv[2] << 8) | (qv[1] & 0xFF);
            pos[i * 2 + 1] = ((qv[1] >> 8) & 0xFFFF) | (qv[0] << 16);
            nrm[i] = oct10(p + 3);
            uv[i * 2] = p[6]; uv[i * 2 + 1] = p[7];
        }
        std::memcpy(heap_u32(index_va[m]), scene.indices.data() + me.first_index, size_t(me.index_count) * 4);
        uint32_t* iv = heap_u32(index_vsharp_va + m * 0x20);
        iv[0] = index_va[m]; iv[1] = 4u << 16; iv[2] = me.index_count; iv[3] = 0x11014facu;   // R32_UINT, stride 4
    }
    view_projection(cam, float(W) / float(H), heap_f32(view_va + 0x160));
    std::memcpy(heap_f32(view_va + 0x1a0), heap_f32(view_va + 0x160), 64);   // previous = current: still camera
    const auto gpu_table = table.build_gpu_buffer(true);
    VkBuffer tb; void* tbp;
    CHECK(make_buffer(gpu_table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
    std::memcpy(tbp, gpu_table.data(), gpu_table.size() * 4);

    // Textures.
    std::vector<VkImage> timg(nimg); std::vector<VkImageView> tview(nimg);
    VkDeviceSize up_bytes = 4;
    for (const auto& t : scene.textures) up_bytes += t.rgba.size();
    VkBuffer up; void* upp;
    CHECK(make_buffer(up_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up, &upp));
    std::vector<VkDeviceSize> up_off(nimg);
    { const uint32_t white = 0xFFFFFFFFu; std::memcpy(upp, &white, 4); VkDeviceSize o = 4;
      for (uint32_t i = 1; i < nimg; ++i) { up_off[i] = o; std::memcpy(static_cast<char*>(upp) + o, scene.textures[i - 1].rgba.data(), scene.textures[i - 1].rgba.size()); o += scene.textures[i - 1].rgba.size(); } }
    for (uint32_t i = 0; i < nimg; ++i)
        CHECK(make_image(i ? scene.textures[i - 1].w : 1, i ? scene.textures[i - 1].h : 1, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &timg[i], &tview[i]));
    VkSamplerCreateInfo sci = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(
        leonida::gfx10::SamplerWords{ssharp[0], ssharp[1], ssharp[2], ssharp[3]}));
    sci.maxLod = 0;   // the scene stores mip 0 only
    VkSampler smp; CHECK(vkCreateSampler(dev, &sci, nullptr, &smp));

    // Targets: the G-buffer (docs/gbuffer.md) and D32 depth.
    constexpr int NT = 5;
    const VkFormat tfmt[NT] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                               VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16_SFLOAT};
    const VkFormat dfmt = VK_FORMAT_D32_SFLOAT;
    VkImage rt[NT + 1]; VkImageView rtv[NT + 1];
    for (int t = 0; t < NT; ++t)
        CHECK(make_image(W, H, tfmt[t], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &rt[t], &rtv[t]));
    CHECK(make_image(W, H, dfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &rt[NT], &rtv[NT]));
    VkAttachmentDescription ad[NT + 1]{};
    VkAttachmentReference car[NT];
    for (int t = 0; t <= NT; ++t) {
        ad[t].format = t < NT ? tfmt[t] : dfmt; ad[t].samples = VK_SAMPLE_COUNT_1_BIT; ad[t].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ad[t].storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad[t].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ad[t].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; ad[t].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ad[t].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        if (t < NT) car[t] = {uint32_t(t), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkAttachmentReference dar{NT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = NT; sd.pColorAttachments = car; sd.pDepthStencilAttachment = &dar;
    VkSubpassDependency dep{0, VK_SUBPASS_EXTERNAL,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT};
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = NT + 1; rpi.pAttachments = ad; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = NT + 1; fbi.pAttachments = rtv; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fbuf; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fbuf));

    // Descriptor set: binding 0 images[], 1 samplers[], 2 the descriptor hash table.
    VkDescriptorSetLayoutBinding b[3]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, nimg, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 3; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[3]{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, nimg}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    std::vector<VkDescriptorImageInfo> iinfo(nimg);
    for (uint32_t i = 0; i < nimg; ++i) iinfo[i] = {VK_NULL_HANDLE, tview[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sinfo{smp};
    VkDescriptorBufferInfo tinfo{tb, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[3]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; }
    wr[0].dstBinding = 0; wr[0].descriptorCount = nimg; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = iinfo.data();
    wr[1].dstBinding = 1; wr[1].descriptorCount = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &sinfo;
    wr[2].dstBinding = 2; wr[2].descriptorCount = 1; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tinfo;
    vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);

    // Pipeline: no vertex inputs (the VS pulls its own), depth test, no culling, five colour targets.
    auto module = [&](const std::vector<uint32_t>& code, VkShaderModule* m) {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs; CHECK(module(vs_code, &vs)); CHECK(module(ps_code, &fs));
    VkPipelineShaderStageCreateInfo st[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, float(H), float(W), -float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};   // D3D clip space: y up
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cba[NT]{};
    for (auto& c : cba) c.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = NT; cb.pAttachments = cba;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = st; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &dss; gpi.pColorBlendState = &cb;
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da,
                       VkPipelineStageFlags ss, VkPipelineStageFlags dst) {
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = from; ib.newLayout = to; ib.image = img; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ib.srcAccessMask = sa; ib.dstAccessMask = da; ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, ss, dst, 0, 0, nullptr, 0, nullptr, 1, &ib);
    };
    const uint32_t tbytes[NT + 1] = {4, 4, 4, 4, 4, 4};
    VkBuffer rb[NT + 1]; void* rbp[NT + 1];
    for (int t = 0; t <= NT; ++t) CHECK(make_buffer(VkDeviceSize(W) * H * tbytes[t], VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb[t], &rbp[t]));

    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
    for (uint32_t i = 0; i < nimg; ++i) {
        barrier(timg[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy c{}; c.bufferOffset = up_off[i]; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {i ? scene.textures[i - 1].w : 1, i ? scene.textures[i - 1].h : 1, 1};
        vkCmdCopyBufferToImage(cmd, up, timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        barrier(timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    VkClearValue clear[NT + 1]{}; clear[NT].depthStencil = {1, 0};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fbuf; rbi.renderArea = sc; rbi.clearValueCount = NT + 1; rbi.pClearValues = clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[push_words] = {};
    pc[32] = uint32_t(heap_base); pc[33] = uint32_t(heap_base >> 32);
    pc[8] = table_a;
    pc[16] = sampler_va - 0x520;
    for (uint32_t d = 0; d < ndraw; ++d) {
        const uint32_t m = scene.draws[d].mesh;
        pc[9] = table_b + d * 0x100;
        pc[11] = stream_va[m];          // s12 (high address bits) stays 0
        pc[14] = 0;                     // VS: index V# slot; PS: byte offset of the T# index in table A + 0xac0
        pc[15] = tsharp_va + d * 0x20;  // PS: T# table
        pc[18] = index_vsharp_va + m * 0x20;
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes, pc);
        vkCmdDraw(cmd, scene.meshes[m].index_count, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
    for (int t = 0; t <= NT; ++t) {
        VkBufferImageCopy rc{}; rc.imageSubresource = {VkImageAspectFlags(t < NT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT), 0, 0, 1};
        rc.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, rt[t], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb[t], 1, &rc);
    }
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(q));

    // Counts over the pixels the draw covered (depth < 1).
    const auto* t0 = static_cast<const uint8_t*>(rbp[0]);
    const auto* t1 = static_cast<const uint8_t*>(rbp[1]);
    const auto* t2 = static_cast<const uint8_t*>(rbp[2]);
    const auto* t4 = static_cast<const uint16_t*>(rbp[4]);
    const auto* dz = static_cast<const float*>(rbp[NT]);
    float fwd[3]; cam.forward(fwd);
    size_t covered = 0, written = 0, facing = 0, still = 0;
    float motion_max = 0;
    std::set<uint32_t> albedo;
    for (size_t i = 0; i < size_t(W) * H; ++i) {
        if (!(dz[i] < 1)) continue;
        ++covered;
        written += std::abs(int(t2[i * 4 + 3]) - 128) <= 1;   // mrt2.a = 0.5 on every pixel the PS shades
        albedo.insert(t0[i * 4] | (t0[i * 4 + 1] << 8) | (t0[i * 4 + 2] << 16));
        float nn[3]; decode_normal(t1 + i * 4, nn);
        facing += nn[0] * fwd[0] + nn[1] * fwd[1] + nn[2] * fwd[2] < 0;
        const float mx = half_to_float(t4[i * 2]), my = half_to_float(t4[i * 2 + 1]);
        const float mm = std::sqrt(mx * mx + my * my);
        motion_max = std::max(motion_max, std::isfinite(mm) ? mm : 1e30f);
        still += mm < 0.1f;
    }
    std::printf("vk_gbuffer: draws=%u pixels=%u covered=%zu shaded=%zu albedo_colours=%zu normals_facing=%zu "
                "motion_still=%zu motion_max=%.3f\n", ndraw, W * H, covered, written, albedo.size(), facing, still, motion_max);
    if (dump) {
        std::vector<uint8_t> img(size_t(W) * H * 4);
        const std::string pre = dump;
        bool ok = write_ppm((pre + "_albedo.ppm").c_str(), t0, W, H, false);
        for (size_t i = 0; i < size_t(W) * H; ++i) {
            float nn[3]; decode_normal(t1 + i * 4, nn);
            for (int k = 0; k < 3; ++k) img[i * 4 + k] = dz[i] < 1 ? uint8_t(std::lround((nn[k] * 0.5f + 0.5f) * 255)) : 0;
        }
        ok = ok && write_ppm((pre + "_normal.ppm").c_str(), img.data(), W, H, false);
        ok = ok && write_ppm((pre + "_mrt2.ppm").c_str(), t2, W, H, false);
        ok = ok && write_ppm((pre + "_mrt3.ppm").c_str(), static_cast<const uint8_t*>(rbp[3]), W, H, false);
        if (!ok) { std::fprintf(stderr, "cannot write %s_*.ppm\n", dump); return 1; }
    }
    return 0;
}
