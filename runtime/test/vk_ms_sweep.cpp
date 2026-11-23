// Runs one translated mesh shader with the runtime's bindless set 0 and a host heap, on whichever GPU the loader gives
// it, and writes what it produced. It is the per-shader worker of tools/ms_sweep.py, the mesh-shader sibling of
// vk_ps_sweep.
//
//   vk_ms_sweep native  <ms.spv> <ps.spv> -        <out.raw> <width> <height> <groups> 0 0
//   vk_ms_sweep compute <ms.spv> <ps.spv> <vs.spv> <out.raw> <width> <height> <groups> <slot-bytes> <max-primitives>
//
// native   the module is a VK_EXT_mesh_shader mesh shader, drawn with vkCmdDrawMeshTasksEXT(groups, 1, 1) and the
//          pixel shader. If the device has meshShaderQueries, the primitives the mesh workgroups generated are
//          counted (stdout "native_prims N"; "native_prims n/a" otherwise).
// compute  the module was translated with rdna2spirv.py --mesh-compute: it is dispatched as a compute shader over
//          the same groups and writes its slots (slot-bytes and max-primitives are the shader's "mesh_compute" layout
//          in index.json) into an output buffer; vs.spv (tools/mesh_compute_vs.py) then draws them with the same
//          pixel shader, as the runtime's fallback does.
// Inputs follow the contract in docs/design.md and are identical on every device and in both modes:
//   set 0       the BindlessSet of runtime/src/vulkan_bindless.h with only its fallbacks
//   push        s0..s31 from VKS_SGPRS, then the heap base (dwords 32..33); in compute mode the output address (36..37)
//   heap        VKS_HEAP_KB KiB (default 1024), zero unless VKS_HEAP_FILL=<offset>,<bytes>,<seed> fills a range with
//               floats in [0.25, 1). The sweep leaves it zero: the heap holds the shaders' descriptors, so any data
//               in it is read as a wild address and resets the device
// Output: the colour target (width * height * 16 bytes of RGBA32F, cleared to -1 so an unwritten pixel is
// recognisable), then in compute mode the output buffer (groups * slot-bytes), zero except for the two UINT32_MAX count sentinels in each header before dispatch.
//
// Exit codes: 0 ran; 77 the device lacks a feature the module needs (stdout says which); 1 a Vulkan call failed
// (stderr says which); 3 the fence did not signal in VKS_TIMEOUT_S seconds (default 20); 4 the device was lost;
// 5 pipeline creation failed (stderr "PIPELINE <result>").
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "../src/vulkan_bindless.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ == VK_ERROR_DEVICE_LOST) {                                                \
            std::fprintf(stderr, "DEVICE_LOST %s (line %d)\n", #x, __LINE__);            \
            return 4;                                                                    \
        }                                                                                \
        if (r_ != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, int(r_), __LINE__);  \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

static std::vector<uint32_t> capabilities(const std::vector<uint32_t>& code) {
    std::vector<uint32_t> caps;
    for (std::size_t i = 5; i + 1 < code.size();) {
        const uint32_t words = code[i] >> 16, op = code[i] & 0xffffu;
        if (!words) break;
        if (op == 17) caps.push_back(code[i + 1]);
        i += words;
    }
    return caps;
}

static bool parse_words(const char* s, uint32_t* out, uint32_t max) {
    for (uint32_t i = 0; *s; ++i) {
        if (i == max) return false;
        char* end;
        out[i] = uint32_t(std::strtoull(s, &end, 0));
        if (end == s) return false;
        s = end;
        if (*s == ',') ++s;
    }
    return true;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const double t_start = now_s();
    if (argc != 11) {
        std::fprintf(stderr, "usage: vk_ms_sweep native|compute ms.spv ps.spv vs.spv|- out.raw width height groups slot-bytes max-prims\n");
        return 2;
    }
    const bool compute = !std::strcmp(argv[1], "compute");
    if (!compute && std::strcmp(argv[1], "native")) return 2;
    const uint32_t W = uint32_t(std::atoi(argv[6])), H = uint32_t(std::atoi(argv[7]));
    const uint32_t groups = uint32_t(std::atoi(argv[8]));
    const uint64_t slot = std::strtoull(argv[9], nullptr, 0);
    const uint32_t max_prims = uint32_t(std::atoi(argv[10]));
    if (!W || !H || !groups || (compute && (slot < 8 || slot > SIZE_MAX / groups || !max_prims || !std::strcmp(argv[4], "-")))) return 2;
    uint32_t pc[38] = {};
    if (const char* s = std::getenv("VKS_SGPRS"); s && !parse_words(s, pc, 32)) {
        std::fprintf(stderr, "VKS_SGPRS: expected up to 32 comma-separated words\n");
        return 2;
    }
    const char* heap_kb_env = std::getenv("VKS_HEAP_KB");
    const uint64_t heap_bytes = (heap_kb_env && std::atoi(heap_kb_env) > 0 ? uint64_t(std::atoi(heap_kb_env)) : 1024u) << 10;
    uint32_t fill[3] = {};   // offset, bytes, seed
    bool fill_heap = false;
    if (const char* s = std::getenv("VKS_HEAP_FILL"); s && *s) {
        if (!parse_words(s, fill, 3) || fill[0] % 4 || fill[1] % 4 || uint64_t(fill[0]) + fill[1] > heap_bytes) {
            std::fprintf(stderr, "VKS_HEAP_FILL: expected offset,bytes,seed inside the heap\n");
            return 2;
        }
        fill_heap = true;
    }
    const char* timeout_env = std::getenv("VKS_TIMEOUT_S");
    const uint64_t timeout_ns = (timeout_env && std::atoi(timeout_env) > 0 ? uint64_t(std::atoi(timeout_env)) : 20u) * 1000000000ull;

    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkGetPhysicalDeviceProperties2);
    LOADI(vkGetPhysicalDeviceFeatures2); LOADI(vkGetPhysicalDeviceQueueFamilyProperties);
    LOADI(vkGetPhysicalDeviceMemoryProperties); LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr);
    LOADI(vkEnumerateDeviceExtensionProperties);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    if (chosen >= pds.size()) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDevice pd = pds[chosen];
    VkPhysicalDeviceVulkan11Properties props11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties props12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    props11.pNext = &props12;
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &props11};
    vkGetPhysicalDeviceProperties2(pd, &props);
    std::fprintf(stderr, "device: %s\n", props.properties.deviceName);
    constexpr uint32_t pc_bytes = 152;
    if (props.properties.limits.maxPushConstantsSize < pc_bytes) {
        std::printf("unsupported: push constants %u bytes\n", props.properties.limits.maxPushConstantsSize);
        return 77;
    }
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !((qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT))) qi++;
    if (qi == qn) return 1;

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has_ext = [&](const char* e) {
        for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true;
        return false;
    };

    std::vector<uint32_t> ms_code = read_spv(argv[2]);
    if (ms_code.size() < 6) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have.pNext = &have12;
    have12.pNext = &mesh_have;
    vkGetPhysicalDeviceFeatures2(pd, &have);

    std::string missing;
    auto need = [&](bool ok, const char* what) {
        if (ok) return;
        if (!missing.empty()) missing += ", ";
        missing += what;
    };
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    std::vector<const char*> enable_exts;
    for (const uint32_t cap : capabilities(ms_code)) {
        switch (cap) {
            case 10: need(have.features.shaderFloat64, "shaderFloat64"); enable.features.shaderFloat64 = VK_TRUE; break;
            case 11: need(have.features.shaderInt64, "shaderInt64"); enable.features.shaderInt64 = VK_TRUE; break;
            case 32: need(have.features.shaderClipDistance, "shaderClipDistance"); enable.features.shaderClipDistance = VK_TRUE; break;
            case 33: need(have.features.shaderCullDistance, "shaderCullDistance"); enable.features.shaderCullDistance = VK_TRUE; break;
            case 5283: need(!compute, "a mesh capability in a --mesh-compute module"); break;
        }
    }
    if (!compute) need(has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME) && mesh_have.meshShader, "VK_EXT_mesh_shader");
    need(have12.bufferDeviceAddress && have12.runtimeDescriptorArray, "bufferDeviceAddress or runtimeDescriptorArray");
    need(have.features.shaderInt64 && have.features.fragmentStoresAndAtomics &&
             have.features.shaderSampledImageArrayDynamicIndexing,
         "shaderInt64, fragmentStoresAndAtomics or shaderSampledImageArrayDynamicIndexing");
    if (!missing.empty()) {
        std::printf("unsupported: %s\n", missing.c_str());
        return 77;
    }
    enable.features.shaderInt64 = VK_TRUE;
    enable.features.fragmentStoresAndAtomics = VK_TRUE;
    enable.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    enable.features.shaderStorageImageArrayDynamicIndexing = have.features.shaderStorageImageArrayDynamicIndexing;
    const bool prim_query = !compute && mesh_have.meshShaderQueries;
    if (!compute) {
        enable_exts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        mesh.meshShader = VK_TRUE;
        mesh.meshShaderQueries = prim_query ? VK_TRUE : VK_FALSE;
    }
    leonida::BindlessFeatures flags{
        have12.descriptorBindingSampledImageUpdateAfterBind && have12.descriptorBindingStorageImageUpdateAfterBind &&
            have12.descriptorBindingStorageBufferUpdateAfterBind,
        have12.descriptorBindingPartiallyBound == VK_TRUE};
    en12.descriptorIndexing = have12.descriptorIndexing;
    en12.descriptorBindingSampledImageUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingStorageImageUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingStorageBufferUpdateAfterBind = flags.update_after_bind;
    en12.descriptorBindingPartiallyBound = flags.partially_bound;
    en12.runtimeDescriptorArray = VK_TRUE;
    en12.bufferDeviceAddress = VK_TRUE;
    en12.pNext = nullptr;
    if (!compute) en12.pNext = &mesh;
    enable.pNext = &en12;

    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enable;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable_exts.size()); dci.ppEnabledExtensionNames = enable_exts.data();
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateRenderPass); LOADD(vkCreateFramebuffer);
    LOADD(vkCreateShaderModule); LOADD(vkCreatePipelineLayout); LOADD(vkCreateGraphicsPipelines);
    LOADD(vkCreateComputePipelines); LOADD(vkCmdDispatch); LOADD(vkCmdPipelineBarrier);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBeginRenderPass); LOADD(vkCmdBindPipeline); LOADD(vkCmdDraw); LOADD(vkCmdEndRenderPass);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit); LOADD(vkCreateBuffer);
    LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory); LOADD(vkMapMemory);
    LOADD(vkGetBufferDeviceAddress); LOADD(vkCmdBindDescriptorSets); LOADD(vkCmdPushConstants);
    LOADD(vkCreateFence); LOADD(vkWaitForFences); LOADD(vkCreateQueryPool); LOADD(vkCmdResetQueryPool);
    LOADD(vkCmdBeginQuery); LOADD(vkCmdEndQuery); LOADD(vkGetQueryPoolResults);
    PFN_vkCmdDrawMeshTasksEXT draw_mesh = nullptr;
    if (!compute) draw_mesh = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT"));
    VkQueue q;
    vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };

    const double t_device = now_s();
    constexpr uint32_t sampled_total = 4096 + 1024 + 256 + 256, sampler_total = 1024, storage_total = 1024 + 13 * 256;
    const auto fits = [&](const leonida::BindlessFeatures& f) {
        const VkPhysicalDeviceLimits& l = props.properties.limits;
        if (f.update_after_bind)
            return props12.maxPerStageDescriptorUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxPerStageDescriptorUpdateAfterBindStorageImages >= storage_total &&
                   props12.maxPerStageUpdateAfterBindResources >= sampled_total + sampler_total + storage_total + 1 &&
                   props12.maxDescriptorSetUpdateAfterBindSampledImages >= sampled_total &&
                   props12.maxDescriptorSetUpdateAfterBindSamplers >= sampler_total &&
                   props12.maxDescriptorSetUpdateAfterBindStorageImages >= storage_total;
        return l.maxPerStageDescriptorSampledImages >= sampled_total && l.maxPerStageDescriptorSamplers >= sampler_total &&
               l.maxPerStageDescriptorStorageImages >= storage_total &&
               l.maxPerStageResources >= sampled_total + sampler_total + storage_total + 1 &&
               l.maxDescriptorSetSampledImages >= sampled_total && l.maxDescriptorSetSamplers >= sampler_total &&
               l.maxDescriptorSetStorageImages >= storage_total;
    };
    if (!fits(flags)) flags = {};
    if (!fits(flags)) {
        flags.update_after_bind = en12.descriptorBindingSampledImageUpdateAfterBind;
        flags.partially_bound = en12.descriptorBindingPartiallyBound;
        leonida::bindless::size_bindings(flags, props.properties.limits, props11, props12);
    }
    std::optional<leonida::BindlessSet> set;
    set.emplace(vkGetDeviceProcAddr, dev, mp, flags);
    if (!set->ok() || !set->layout() || !set->set()) {
        std::printf("unsupported: cannot create the bindless set\n");
        return 77;
    }

    auto host_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, bool address, VkBuffer* b, void** mapped,
                           uint64_t* addr) -> VkResult {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = usage | (address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u);
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b);
        if (r != VK_SUCCESS) return r;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateFlagsInfo af{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        af.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext = address ? &af : nullptr; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory mem;
        if ((r = vkAllocateMemory(dev, &ai, nullptr, &mem)) != VK_SUCCESS) return r;
        if ((r = vkBindBufferMemory(dev, *b, mem, 0)) != VK_SUCCESS) return r;
        if ((r = vkMapMemory(dev, mem, 0, size, 0, mapped)) != VK_SUCCESS) return r;
        std::memset(*mapped, 0, size_t(size));
        if (address) {
            VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            da.buffer = *b;
            *addr = vkGetBufferDeviceAddress(dev, &da);
        }
        return VK_SUCCESS;
    };

    // Heap.
    VkBuffer heap;
    void* hp = nullptr;
    uint64_t base = 0;
    CHECK(host_buffer(heap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &heap, &hp, &base));
    if (fill_heap) {
        uint32_t x = fill[2] * 2654435761u + 0x9e3779b9u;
        for (uint32_t o = fill[0]; o < fill[0] + fill[1]; o += 4) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            const uint32_t bits = 0x3e800000u + (x % 0x01000000u);   // floats in [0.25, 1)
            std::memcpy(static_cast<char*>(hp) + o, &bits, 4);
        }
    }
    pc[32] = uint32_t(base); pc[33] = uint32_t(base >> 32);

    // The fallback's output buffer.
    VkBuffer outbuf = VK_NULL_HANDLE;
    void* outp = nullptr;
    const uint64_t out_bytes = uint64_t(groups) * slot;
    if (compute) {
        uint64_t out_addr = 0;
        CHECK(host_buffer(out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &outbuf, &outp, &out_addr));
        const uint32_t header[] = {UINT32_MAX, UINT32_MAX};
        for (uint32_t g = 0; g < groups; ++g)
            std::memcpy(static_cast<char*>(outp) + g * slot, header, sizeof(header));
        pc[36] = uint32_t(out_addr); pc[37] = uint32_t(out_addr >> 32);
    }

    // One colour target.
    const VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImage img;
    VkImageView view;
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &img));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m;
        CHECK(vkAllocateMemory(dev, &ai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, img, m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
    }
    VkAttachmentDescription ad{};
    ad.format = fmt; ad.samples = VK_SAMPLE_COUNT_1_BIT;
    ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{};
    sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sd.colorAttachmentCount = 1; sd.pColorAttachments = &ref;
    VkSubpassDependency dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1; rpi.pAttachments = &ad; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    VkRenderPass rp;
    CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = 1; fbi.pAttachments = &view; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb;
    CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));

    // Pipelines.
    auto module = [&](const std::vector<uint32_t>& code, VkShaderModule* m) {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    std::vector<uint32_t> ps_code = read_spv(argv[3]);
    if (ps_code.size() < 6) { std::fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    VkShaderModule msm, fs, vs = VK_NULL_HANDLE;
    CHECK(module(ms_code, &msm));
    CHECK(module(ps_code, &fs));
    if (compute) {
        std::vector<uint32_t> vs_code = read_spv(argv[4]);
        if (vs_code.size() < 6) { std::fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
        CHECK(module(vs_code, &vs));
    }
    const VkShaderStageFlags pcs = VK_SHADER_STAGE_FRAGMENT_BIT |
                                   (compute ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT : VK_SHADER_STAGE_MESH_BIT_EXT);
    VkPushConstantRange pcr{pcs, 0, pc_bytes};
    const VkDescriptorSetLayout set_layout = set->layout();
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));

    VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = compute ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_MESH_BIT_EXT;
    stages[0].module = compute ? vs : msm; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = stages; gpi.pVertexInputState = compute ? &vin : nullptr;
    gpi.pInputAssemblyState = compute ? &ia : nullptr;
    gpi.pViewportState = &vps; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb;
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe, cpipe = VK_NULL_HANDLE;
    {
        const VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "PIPELINE %d\n", int(r));
            return r == VK_ERROR_DEVICE_LOST ? 4 : 5;
        }
    }
    if (compute) {
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = msm; cpi.stage.pName = "main";
        cpi.layout = pl;
        const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &cpipe);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "PIPELINE %d\n", int(r));
            return r == VK_ERROR_DEVICE_LOST ? 4 : 5;
        }
    }

    const double t_pipe = now_s();
    const VkDeviceSize target_bytes = VkDeviceSize(W) * H * 16;
    VkBuffer buf;
    void* bufp = nullptr;
    CHECK(host_buffer(target_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &buf, &bufp, nullptr));

    VkQueryPool qp = VK_NULL_HANDLE;
    if (prim_query) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT; qpi.queryCount = 1;
        CHECK(vkCreateQueryPool(dev, &qpi, nullptr, &qp));
    }
    VkCommandPoolCreateInfo cpi2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi2.queueFamilyIndex = qi;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpi2, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    if (!set->flush(cmd)) {
        std::fprintf(stderr, "FAILED set->flush\n");
        return 1;
    }
    if (!set->fallbacks_ready()) set->record_fallback_init(cmd);
    const VkDescriptorSet ds = set->set();
    if (compute) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pl, pcs, 0, pc_bytes, pc);
        vkCmdDispatch(cmd, groups, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    if (prim_query) vkCmdResetQueryPool(cmd, qp, 0, 1);
    VkClearValue clear{};
    for (float& f : clear.color.float32) f = -1;
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = 1; rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pl, pcs, 0, pc_bytes, pc);
    if (prim_query) vkCmdBeginQuery(cmd, qp, 0, 0);
    if (compute) vkCmdDraw(cmd, groups * max_prims * 3, 1, 0, 0);
    else draw_mesh(cmd, groups, 1, 1);
    if (prim_query) vkCmdEndQuery(cmd, qp, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy reg{};
    reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    reg.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
    CHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, fence));
    const VkResult waited = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);
    if (waited == VK_TIMEOUT) {
        std::fprintf(stderr, "TIMEOUT\n");
        return 3;
    }
    CHECK(waited);
    const double t_draw = now_s();
    if (!compute) {
        uint64_t count = 0;
        if (prim_query) {
            CHECK(vkGetQueryPoolResults(dev, qp, 0, 1, sizeof(count), &count, sizeof(count),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            std::printf("native_prims %llu\n", (unsigned long long)count);
        } else {
            std::printf("native_prims n/a\n");
        }
    }
    std::ofstream output(argv[5], std::ios::binary);
    output.write(static_cast<const char*>(bufp), std::streamsize(target_bytes));
    if (compute) output.write(static_cast<const char*>(outp), std::streamsize(out_bytes));
    output.close();
    if (!output) {
        std::fprintf(stderr, "failed to write %s\n", argv[5]);
        return 1;
    }
    std::fprintf(stderr, "ok %s %ux%u groups %u -> %s (device %.2fs, pipelines %.2fs, draw %.2fs)\n", argv[1], W, H,
                 groups, argv[5], t_device - t_start, t_pipe - t_device, t_draw - t_pipe);
    return 0;
}
