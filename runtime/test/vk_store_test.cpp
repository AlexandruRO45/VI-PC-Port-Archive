// Headless Vulkan test for translated typed image stores, loads and atomics (runtime/test/shaders/cs_imgstore.asm,
// cs_imgload.asm, cs_imgload_d16.asm and cs_imgatomic.asm).
// Usage: vk_store_test cs.spv [load|load-d16|atomic]
// The shader builds three T#s (word1 FORMAT = 56 rgba8 unorm, 20 r32 uint, 21 r32 sint) and stores to three 8x8
// images. The translator picks the float / uint / sint view (set 0, bindings 8 / 9 / 10 for 2D) from the FORMAT
// field, and the bindless index from the descriptor hash table (binding 2). As in the runtime, each image has its
// own index in the 2D arrays (1, 2, 3; index 0 is the fallback) and its view in its own format at that index of the
// sampled array (binding 0), which typed image_load fetches from. Checks every texel.
// "load" (cs_imgload.asm): the uint and sint images are filled before the dispatch, and the shader loads both
// texels with image_load and stores their sum into the sint image. The device is made without
// shaderStorageImageReadWithoutFormat, and a module that declares StorageImageReadWithoutFormat fails (SHADER-151).
// "atomic" (cs_imgatomic.asm, TEST-184): binds an 8x8 2D R32_UINT storage image at binding 4 (index 1) and a 4x4x4
// 3D R32_UINT storage image at binding 7 (index 2), pre-fills both, runs all 10 image_atomic_* ops and checks every
// texel of both images. Requires neither storage-image without-format feature.
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);       \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static uint32_t key_hash(const uint32_t k[8]) {     // same FNV-1a as the translator's desc_lookup
    uint32_t h = 2166136261u;
    for (int i = 0; i < 8; i++) h = (h ^ k[i]) * 16777619u;
    return h;
}

static void table_insert(std::vector<uint32_t>& t, const uint32_t key[8], uint32_t index) {
    uint32_t mask = t[0], h = key_hash(key);
    for (uint32_t k = 0; k < 8; k++) {
        uint32_t base = ((h + k) & mask) * 10;
        bool used = false;
        for (int j = 0; j < 8; j++) used |= t[base + 1 + j] != 0;
        if (used) continue;
        for (int j = 0; j < 8; j++) t[base + 1 + j] = key[j];
        t[base + 9] = index;
        return;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vk_store_test cs.spv\n"); return 2; }
    gipa = load_vulkan();
    if (!gipa) return 1;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkCreateDevice);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetDeviceProcAddr);
    LOADI(vkGetPhysicalDeviceMemoryProperties); LOADI(vkGetPhysicalDeviceFeatures);
    LOADI(vkGetPhysicalDeviceFormatProperties);
    const bool packed_load = argc > 2 && !std::strcmp(argv[2], "load-d16");
    const bool load_mode = packed_load || (argc > 2 && !std::strcmp(argv[2], "load"));
    const bool atomic_mode = argc > 2 && !std::strcmp(argv[2], "atomic");
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    VkPhysicalDevice pd = chosen < pds.size() ? pds[chosen] : VK_NULL_HANDLE;
    if (!pd) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // The typed stores need format-less storage image writes; atomic mode only uses explicit R32ui storage images.
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(pd, &have);
    if (props.limits.maxPushConstantsSize < 144) {
        std::printf("unsupported: need 144 push constant bytes, device has %u\n", props.limits.maxPushConstantsSize);
        return 77;
    }
    if ((!atomic_mode && !have.shaderStorageImageWriteWithoutFormat) || !have.shaderInt64) {
        std::printf("unsupported: shaderStorageImageWriteWithoutFormat=%u shaderInt64=%u\n",
                    have.shaderStorageImageWriteWithoutFormat, have.shaderInt64);
        return 77;
    }
    if (atomic_mode) {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R32_UINT, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)) {
            std::printf("unsupported: VK_FORMAT_R32_UINT lacks VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT\n");
            return 77;
        }
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.runtimeDescriptorArray = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderStorageImageWriteWithoutFormat = atomic_mode ? VK_FALSE : VK_TRUE;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.size() < 20 || code.size() % 4) { std::fprintf(stderr, "%s: not a SPIR-V module\n", argv[1]); return 1; }
    if (load_mode) {
        // Typed image_load fetches through the sampled arrays, so it must not need a feature Intel Xe-LP lacks
        // (SHADER-151). shaderStorageImageReadWithoutFormat stays off; a module declaring it would be invalid here.
        std::vector<uint32_t> words(code.size() / 4);
        std::memcpy(words.data(), code.data(), code.size());
        for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
            const uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
            // OpCapability (17) StorageImageReadWithoutFormat (55; spirv/unified1/spirv.h)
            if (op == 17 && count == 2 && i + 1 < words.size() && words[i + 1] == 55) {
                std::printf("FAILED: the module declares StorageImageReadWithoutFormat\n");
                return 1;
            }
            i += count;
        }
    }
    f2.features.shaderInt64 = VK_TRUE;
    f2.pNext = &f12;
    leonida::vk_test::ShaderFma fma;      // OpFmaKHR (rdna2spirv.py --fma khr)
    fma.query(gipa, inst, pd);
    if (!fma.accepts(code.data(), code.size(), argv[1])) return 77;
    fma.link_after(&f2);
    const char* fma_extension = fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    if (fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindBufferMemory); LOADD(vkMapMemory); LOADD(vkCreateShaderModule); LOADD(vkCreateImage);
    LOADD(vkGetImageMemoryRequirements); LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateSampler);
    LOADD(vkCreateDescriptorSetLayout); LOADD(vkCreatePipelineLayout); LOADD(vkCreateComputePipelines);
    LOADD(vkCreateDescriptorPool); LOADD(vkAllocateDescriptorSets); LOADD(vkUpdateDescriptorSets);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBindPipeline); LOADD(vkCmdBindDescriptorSets); LOADD(vkCmdPushConstants); LOADD(vkCmdDispatch);
    LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyImageToBuffer); LOADD(vkCmdCopyBufferToImage);
    LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit);
    LOADD(vkQueueWaitIdle);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b);
        if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m;
        if ((r = vkAllocateMemory(dev, &mai, nullptr, &m))) return r;
        if ((r = vkBindBufferMemory(dev, *b, m, 0))) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };

    if (atomic_mode) {
        // 2D (8x8) and 3D (4x4x4) R32_UINT storage image atomics (cs_imgatomic.asm, TEST-184).
        // Set 0: binding 0 = sampled 2D, 1 = sampler, 2 = hash table, 4 = 2D r32ui (%simages), 7 = 3D r32ui (%simages3d).
        constexpr uint32_t W2 = 8, D3 = 4, N = 4;
        std::vector<uint32_t> table(1 + 16 * 10 + 10, 0);
        table[0] = 15;
        const uint32_t t2d[8] = {0x1000, 20u << 20, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
        const uint32_t t3d[8] = {0x2000, 20u << 20, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22};
        table_insert(table, t2d, 1);
        table_insert(table, t3d, 2);
        VkBuffer tb; void* tbp;
        CHECK(make_buffer(table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
        std::memcpy(tbp, table.data(), table.size() * 4);

        auto make_r32ui = [&](VkImageType type, VkImageViewType vtype, VkExtent3D ext, VkImage* out_img,
                              VkImageView* out_view) -> VkResult {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = type; ii.format = VK_FORMAT_R32_UINT; ii.extent = ext;
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            VkResult r = vkCreateImage(dev, &ii, nullptr, out_img);
            if (r) return r;
            VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, *out_img, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkDeviceMemory m;
            if ((r = vkAllocateMemory(dev, &mai, nullptr, &m))) return r;
            if ((r = vkBindImageMemory(dev, *out_img, m, 0))) return r;
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = *out_img; vi.viewType = vtype; vi.format = VK_FORMAT_R32_UINT;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            return vkCreateImageView(dev, &vi, nullptr, out_view);
        };
        VkImage fb2d, img2d, fb3d, img3d;
        VkImageView fbv2d, view2d, fbv3d, view3d;
        CHECK(make_r32ui(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, {1, 1, 1}, &fb2d, &fbv2d));
        CHECK(make_r32ui(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, {W2, W2, 1}, &img2d, &view2d));
        CHECK(make_r32ui(VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, {1, 1, 1}, &fb3d, &fbv3d));
        CHECK(make_r32ui(VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, {D3, D3, D3}, &img3d, &view3d));
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        VkSampler sampler; CHECK(vkCreateSampler(dev, &sci, nullptr, &sampler));

        VkDescriptorSetLayoutBinding b[5]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[3] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[4] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 5; dli.pBindings = b;
        VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
        VkDescriptorPoolSize ps[4] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * N},
        };
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
        VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
        VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
        VkDescriptorBufferInfo tbi{tb, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo sii[N], s2i[N], s3i[N];
        for (uint32_t e = 0; e < N; e++) {
            sii[e] = {VK_NULL_HANDLE, fbv2d, VK_IMAGE_LAYOUT_GENERAL};
            s2i[e] = {VK_NULL_HANDLE, e == 1 ? view2d : fbv2d, VK_IMAGE_LAYOUT_GENERAL};
            s3i[e] = {VK_NULL_HANDLE, e == 2 ? view3d : fbv3d, VK_IMAGE_LAYOUT_GENERAL};
        }
        VkDescriptorImageInfo smpi{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkWriteDescriptorSet wr[5]{};
        for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; w.descriptorCount = 1; }
        wr[0].dstBinding = 0; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = sii; wr[0].descriptorCount = N;
        wr[1].dstBinding = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &smpi;
        wr[2].dstBinding = 2; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tbi;
        wr[3].dstBinding = 4; wr[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[3].pImageInfo = s2i; wr[3].descriptorCount = N;
        wr[4].dstBinding = 7; wr[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[4].pImageInfo = s3i; wr[4].descriptorCount = N;
        vkUpdateDescriptorSets(dev, 5, wr, 0, nullptr);

        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 144};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
        cpi.layout = pl;
        VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

        // Initial and expected 2D (8x8) texels (row-major: idx = 8*y + x).
        const uint32_t init2d[W2 * W2] = {
            // Row 0: single-lane ops (add, sub, and, or, xor, smin, smax, umin)
            0x00001000u, 100u, 0xF0F00FF0u, 0xF0F00FF0u, 0xF0F00FF0u, 5u, 0xFFFFFFF0u, 0x80000000u,
            // Row 1: zeroed slots recording Row 0 glc return values
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            // Row 2: umax, cmpswap hit, cmpswap miss, sub wrap, non-glc add, non-glc cmpswap, canary, neg smin
            0x7FFFFFFFu, 0x11223344u, 0x11223344u, 10u, 50u, 0xCAFE0001u, 0x62626262u, 0xFFFFFFF0u,
            // Row 3: zeroed slots recording Row 2 glc return / preserved vdata values
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            // Row 4: 64-lane contention
            1000u, 5000u, 0u, 0xFFFFFFFFu, 0xA5A5A5A5u, 100u, 0xFFFFFF9Cu, 0xFFFFFFFFu,
            // Row 5: inactive-lane canaries (lanes 0..15 OFF)
            0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x77777777u,
            // Row 6: 48 active lanes (lanes 16..63 ON)
            0u, 100u, 0xFFFFFFFFu, 0x00000055u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x77777777u,
            // Row 7: zeroed slots recording lane 0's preserved vdata after inactive glc ops
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        };
        const uint32_t want2d[W2 * W2] = {
            0x00001234u, 63u, 0x00F00FF0u, 0xF0FAAFF5u, 0x0FF0F0F0u, 0xFFFFFFFBu, 2u, 0x7FFFFFFFu,
            0x00001000u, 100u, 0xF0F00FF0u, 0xF0F00FF0u, 0xF0F00FF0u, 5u, 0xFFFFFFF0u, 0x80000000u,
            0x80000000u, 0x55667788u, 0x11223344u, 0xFFFFFFF1u, 57u, 0xCAFE0002u, 0x62626262u, 0xFFFFFFF0u,
            0x7FFFFFFFu, 0x11223344u, 0x11223344u, 10u, 7u, 0xCAFE0002u, 0x11223344u, 0xFFFFFFF0u,
            3080u, 2920u, 0xFFFFFFFFu, 0xFFFF0000u, 0xA5A5A5E5u, 0xFFFFFFE0u, 31u, 10u,
            0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x77777777u,
            48u, 52u, 0x00FF00FFu, 0x00FF0055u, 0x80000000u, 0x7FFFFFFFu, 5u, 0xBAD00000u,
            1u, 1u, 0x00FF00FFu, 0x00FF0000u, 0x80000000u, 0x7FFFFFFFu, 5u, 0xBAD00000u,
        };
        // Initial and expected 3D (4x4x4) texels (idx = 16*z + 4*y + x). Untouched texels keep canary 0x3D000000 | idx.
        uint32_t init3d[D3 * D3 * D3], want3d[D3 * D3 * D3];
        for (uint32_t i = 0; i < D3 * D3 * D3; i++) init3d[i] = want3d[i] = 0x3D000000u | i;
        auto set3 = [&](uint32_t x, uint32_t y, uint32_t z, uint32_t iv, uint32_t wv) {
            const uint32_t idx = (z * D3 + y) * D3 + x;
            init3d[idx] = iv;
            want3d[idx] = wv;
        };
        auto set3_glc = [&](uint32_t x, uint32_t y, uint32_t iv, uint32_t wv) {
            set3(x, y, 0, iv, wv);
            set3(x, y, 2, 0u, iv);
        };
        set3_glc(0, 0, 10u, 15u);
        set3_glc(1, 0, 50u, 32u);
        set3_glc(2, 0, 0xFF00AA55u, 0x0F00A055u);
        set3_glc(3, 0, 0x00110022u, 0xAA11BB22u);
        set3_glc(0, 1, 0x12345678u, 0x12CB5687u);
        set3_glc(1, 1, 3u, 0xFFFFFFFCu);
        set3_glc(2, 1, 0xFFFFFFF6u, 4u);
        set3_glc(3, 1, 0x80000001u, 0x7FFFFFFEu);
        set3_glc(0, 2, 0x7FFFFFFEu, 0x80000001u);
        set3_glc(1, 2, 0x13579BDFu, 0x2468ACE0u);
        set3_glc(2, 2, 0x13579BDFu, 0x13579BDFu);
        set3(1, 2, 1, 200u, 2280u);
        set3(2, 3, 1, 0u, 0x8000001Fu);
        set3(0, 0, 3, 0x88888888u, 0x88888888u);
        set3(1, 0, 3, 0x99999999u, 0x99999999u);
        set3(2, 0, 3, 0u, 48u);
        set3(3, 0, 3, 0x99999999u, 0xBEEF3D3Du);

        VkBuffer up2, up3, rb2, rb3;
        void *up2p, *up3p, *rb2p, *rb3p;
        CHECK(make_buffer(sizeof(init2d), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up2, &up2p));
        CHECK(make_buffer(sizeof(init3d), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up3, &up3p));
        CHECK(make_buffer(sizeof(init2d), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb2, &rb2p));
        CHECK(make_buffer(sizeof(init3d), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb3, &rb3p));
        std::memcpy(up2p, init2d, sizeof(init2d));
        std::memcpy(up3p, init3d, sizeof(init3d));

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi;
        VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(cmd, &cbi));
        VkImage all_imgs[4] = {fb2d, fb3d, img2d, img3d};
        VkImageMemoryBarrier ib[4]{};
        for (int k = 0; k < 4; k++) {
            ib[k].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; ib[k].image = all_imgs[k];
            ib[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ib[k].newLayout = k < 2 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ib[k].dstAccessMask = k < 2 ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                                        : VK_ACCESS_TRANSFER_WRITE_BIT;
            ib[k].srcQueueFamilyIndex = ib[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 4, ib);
        VkBufferImageCopy bic2{}, bic3{};
        bic2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic2.imageExtent = {W2, W2, 1};
        bic3.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic3.imageExtent = {D3, D3, D3};
        vkCmdCopyBufferToImage(cmd, up2, img2d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic2);
        vkCmdCopyBufferToImage(cmd, up3, img3d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic3);
        for (int k = 2; k < 4; k++) {
            ib[k].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib[k].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, ib + 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        uint32_t pc[36] = {};
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 144, pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        for (int k = 2; k < 4; k++) {
            ib[k].oldLayout = VK_IMAGE_LAYOUT_GENERAL; ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            ib[k].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            ib[k].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, ib + 2);
        vkCmdCopyImageToBuffer(cmd, img2d, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb2, 1, &bic2);
        vkCmdCopyImageToBuffer(cmd, img3d, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb3, 1, &bic3);
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(q));

        const uint32_t* got2 = static_cast<const uint32_t*>(rb2p);
        const uint32_t* got3 = static_cast<const uint32_t*>(rb3p);
        uint32_t bad2 = 0, bad3 = 0;
        for (uint32_t y = 0; y < W2; y++)
            for (uint32_t x = 0; x < W2; x++) {
                const uint32_t i = y * W2 + x;
                if (got2[i] != want2d[i]) {
                    if (bad2 < 8)
                        std::fprintf(stderr, "2D (%u,%u): got 0x%08x, want 0x%08x\n", x, y, got2[i], want2d[i]);
                    bad2++;
                }
            }
        for (uint32_t z = 0; z < D3; z++)
            for (uint32_t y = 0; y < D3; y++)
                for (uint32_t x = 0; x < D3; x++) {
                    const uint32_t i = (z * D3 + y) * D3 + x;
                    if (got3[i] != want3d[i]) {
                        if (bad3 < 8)
                            std::fprintf(stderr, "3D (%u,%u,%u): got 0x%08x, want 0x%08x\n", x, y, z, got3[i], want3d[i]);
                        bad3++;
                    }
                }
        std::printf("2d r32ui atomics: %u bad\n", bad2);
        std::printf("3d r32ui atomics: %u bad\n", bad3);
        return (bad2 || bad3) ? 1 : 0;
    }

    // T#s exactly as the shader builds them (cs_imgstore.asm): word0 = 0x1000 * (k + 1), word1 = FORMAT << 20,
    // words 2..7 = 0x11 * (k + 1).
    const uint32_t fmts[3] = {56, 20, 21};
    const VkFormat vkf[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
    std::vector<uint32_t> table(1 + 16 * 10 + 10, 0);
    table[0] = 15;
    for (uint32_t k = 0; k < 3; k++) {
        uint32_t t[8] = {0x1000 * (k + 1), fmts[k] << 20};
        for (int j = 2; j < 8; j++) t[j] = 0x11 * (k + 1);
        table_insert(table, t, k + 1);     // its index in the 2D arrays; 0 is the fallback, as in the runtime
    }
    VkBuffer tb; void* tbp;
    CHECK(make_buffer(table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
    std::memcpy(tbp, table.data(), table.size() * 4);

    const uint32_t W = 8;
    VkImage img[3]; VkImageView view[3];
    for (int k = 0; k < 3; k++) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = vkf[k]; ii.extent = {W, W, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &img[k]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img[k], &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m));
        CHECK(vkBindImageMemory(dev, img[k], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img[k]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = vkf[k];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view[k]));
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler sampler; CHECK(vkCreateSampler(dev, &sci, nullptr, &sampler));
    // Set 0: binding 0 = sampled images, 1 = samplers, 2 = hash table, 8/9/10 = float/uint/sint 2D storage image arrays.
    // Four elements each: index 0 stands for the runtime's fallback, image k is at index k + 1. Binding 0 holds each
    // image's view in its own format (typed image_load reads the uint and sint ones through integer aliases); every
    // element of a storage array holds the image of its class, so all of them are valid descriptors.
    constexpr uint32_t N = 4;
    VkDescriptorSetLayoutBinding b[6]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (int k = 0; k < 3; k++) b[3 + k] = {uint32_t(8 + k), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, N, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[4] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, N},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * N},
    };
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo tbi{tb, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo sii[N] = {{VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[0], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[1], VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, view[2], VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorImageInfo smpi{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorImageInfo dii[3][N];
    VkWriteDescriptorSet wr[6]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; w.descriptorCount = 1; }
    wr[0].dstBinding = 0; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = sii;
    wr[0].descriptorCount = N;
    wr[1].dstBinding = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &smpi;
    wr[2].dstBinding = 2; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tbi;
    for (int k = 0; k < 3; k++) {
        for (uint32_t e = 0; e < N; e++) dii[k][e] = {VK_NULL_HANDLE, view[k], VK_IMAGE_LAYOUT_GENERAL};
        wr[3 + k].dstBinding = 8 + k; wr[3 + k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[3 + k].pImageInfo = dii[k]; wr[3 + k].descriptorCount = N;
    }
    vkUpdateDescriptorSets(dev, 6, wr, 0, nullptr);

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 144};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    VkBuffer rb[3]; void* rbp[3];
    for (int k = 0; k < 3; k++) CHECK(make_buffer(W * W * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb[k], &rbp[k]));
    // Load mode: what the uint and sint images hold before the dispatch, texel (x, y) at i = 8y + x.
    // The last row of the D16 fixture exercises both sides of each 16-bit boundary.
    struct PackedLoad { uint32_t u; int32_t s, sum; };
    constexpr PackedLoad boundary[] = {
        {65535, 0, 65535}, {65536, 0, 65535}, {65537, 0, 65535}, {0, 32767, 32767},
        {0, 32768, 32767}, {0, -32768, -32768}, {0, -32769, -32768}, {65536, -32769, 32767},
    };
    static_assert(sizeof(boundary) / sizeof(boundary[0]) == W);
    VkBuffer fill[2]{}; void* fillp[2]{};
    if (load_mode) {
        for (int j = 0; j < 2; j++)
            CHECK(make_buffer(W * W * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &fill[j], &fillp[j]));
        for (uint32_t i = 0; i < W * W; i++) {
            const bool edge = packed_load && i / W == W - 1;
            static_cast<uint32_t*>(fillp[0])[i] = edge ? boundary[i % W].u : 100 + i;
            static_cast<int32_t*>(fillp[1])[i] = edge ? boundary[i % W].s : -2 * int32_t(i);
        }
    }
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkImageMemoryBarrier ib[3]{};
    for (int k = 0; k < 3; k++) {
        ib[k].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; ib[k].image = img[k];
        ib[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        ib[k].srcQueueFamilyIndex = ib[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (load_mode && k > 0) {   // filled by a copy first
            ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 3, ib);
    if (load_mode) {
        for (int k = 1; k < 3; k++) {
            VkBufferImageCopy bic{};
            bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
            vkCmdCopyBufferToImage(cmd, fill[k - 1], img[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
            ib[k].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib[k].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            ib[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 2, ib + 1);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[36] = {};
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 144, pc);
    vkCmdDispatch(cmd, 1, 1, 1);
    for (int k = 0; k < 3; k++) {
        ib[k].oldLayout = VK_IMAGE_LAYOUT_GENERAL; ib[k].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        ib[k].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; ib[k].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, ib);
    for (int k = 0; k < 3; k++) {
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {W, W, 1};
        vkCmdCopyImageToBuffer(cmd, img[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb[k], 1, &bic);
    }
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));

    uint32_t bad[3] = {};
    for (uint32_t y = 0; y < W; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint32_t i = y * W + x;
            const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * i;
            int er = int(std::lround(x / 7.0 * 255)), eg = int(std::lround(y / 7.0 * 255));
            if (std::abs(c[0] - er) > 1 || std::abs(c[1] - eg) > 1 || std::abs(c[2] - 128) > 1 || c[3] != 255) bad[0]++;
            const bool edge = packed_load && y == W - 1;
            if (static_cast<const uint32_t*>(rbp[1])[i] != (edge ? boundary[x].u : 100 + i)) bad[1]++;
            // load: the uint texel (100 + 8y + x) plus the sint one (-2 (8y + x)), both read with image_load
            const int32_t want_s = edge ? boundary[x].sum : load_mode ? 100 - int32_t(i) : -int32_t(i);
            if (static_cast<const int32_t*>(rbp[2])[i] != want_s) bad[2]++;
        }
    const uint8_t* c = static_cast<const uint8_t*>(rbp[0]) + 4 * (3 * W + 5);
    std::printf("rgba8 unorm (float view): %u bad, texel(5,3) = %u %u %u %u\n", bad[0], c[0], c[1], c[2], c[3]);
    std::printf("r32 uint (uint view): %u bad, texel(5,3) = %u\n", bad[1], static_cast<const uint32_t*>(rbp[1])[3 * W + 5]);
    std::printf("r32 sint (sint view): %u bad, texel(5,3) = %d\n", bad[2], static_cast<const int32_t*>(rbp[2])[3 * W + 5]);
    return (bad[0] || bad[1] || bad[2]) ? 1 : 0;
}
