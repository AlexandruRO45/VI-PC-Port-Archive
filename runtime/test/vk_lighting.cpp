// Full-screen pass runner for translated lighting shaders (tools/lighting_pass.py drives it).
//
//   vk_lighting <plan.txt>
//
// The plan is a text file, one directive per line ('#' starts a comment):
//   size W H                       render-target size
//   vs <file.spv> / ps <file.spv>  the translated pair (the VS draws one full-screen triangle from SV_VertexID)
//   heap <file>                    raw bytes of the GPU VA window the shaders' table pointers and V#s address
//   push <sgpr> <value>            user SGPR s0..s31 (push-constant words 0..31; words 32/33 are the heap base)
//   image <type> <vkformat> <w> <h> <d> <file|-> <va> [<va> ...]
//                                  an image: type 2d, 2darray, cube, 3d or 1d; d is the layer count (array, cube: 6)
//                                  or the depth (3d); the file holds every layer (or slice) of mip 0, tightly packed,
//                                  or '-' for zeros. Its T# (gfx10::encode) is written into the heap at every va and
//                                  mapped to the image in the descriptor hash table.
//   sampler <w0> <w1> <w2> <w3>    an S# (a literal the shader carries, or one in the heap) mapped to its sampler
//   target <vkformat>              one colour target per line, mrt0 first
//   out <prefix>                   readback: <prefix>_t<N>.bin holds target N's raw bytes, row-major
//
// It prints one line: the device, the size and the byte count of every target. Nothing here is game data: the
// shaders come from the user's work folder and the plan only from what the tool wrote.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"
#include "../src/vulkan_bindless.h"

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

struct ImageSpec {
    std::string type;
    VkFormat format;
    uint32_t w, h, d;
    std::string file;
    std::vector<uint32_t> vas;
};

struct Plan {
    uint32_t W = 0, H = 0;
    std::string vs, ps, heap, out;
    std::vector<std::pair<uint32_t, uint32_t>> push;
    std::vector<ImageSpec> images;
    std::vector<std::array<uint32_t, 4>> samplers;
    std::vector<VkFormat> targets;
};

std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

bool parse(const char* path, Plan& p) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (const auto h = line.find('#'); h != std::string::npos) line.resize(h);
        std::istringstream s(line);
        std::string op;
        if (!(s >> op)) continue;
        bool ok = true;
        if (op == "size") ok = bool(s >> p.W >> p.H);
        else if (op == "vs") ok = bool(s >> p.vs);
        else if (op == "ps") ok = bool(s >> p.ps);
        else if (op == "heap") ok = bool(s >> p.heap);
        else if (op == "out") ok = bool(s >> p.out);
        else if (op == "push") {
            uint32_t r, v; ok = bool(s >> r >> std::hex >> v) && r < 32;
            if (ok) p.push.push_back({r, v});
        } else if (op == "image") {
            ImageSpec im; int fmt;
            ok = bool(s >> im.type >> fmt >> im.w >> im.h >> im.d >> im.file);
            im.format = VkFormat(fmt);
            for (std::string v; ok && s >> v;) im.vas.push_back(uint32_t(std::strtoul(v.c_str(), nullptr, 0)));
            ok = ok && im.w && im.h && im.d && im.w <= 16384 && im.h <= 16384 && im.d <= 2048 &&
                 (im.type == "2d" || im.type == "2darray" || im.type == "cube" || im.type == "3d" || im.type == "1d");
            if (ok) p.images.push_back(im);
        } else if (op == "sampler") {
            std::array<uint32_t, 4> w{};
            for (auto& x : w) ok = ok && bool(s >> std::hex >> x);
            if (ok) p.samplers.push_back(w);
        } else if (op == "target") {
            int fmt; ok = bool(s >> fmt);
            if (ok) p.targets.push_back(VkFormat(fmt));
        } else ok = false;
        if (!ok) { std::fprintf(stderr, "%s:%d: bad line\n", path, n); return false; }
    }
    return p.W && p.H && p.W <= 8192 && p.H <= 8192 && !p.vs.empty() && !p.ps.empty() && !p.heap.empty() &&
           !p.out.empty() && !p.targets.empty() && p.targets.size() <= 8;
}

uint32_t texel_bytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_UNORM: return 2;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_UINT: return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
        default: return 0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: vk_lighting plan.txt\n"); return 2; }
    Plan plan;
    if (!parse(argv[1], plan)) { std::fprintf(stderr, "bad plan %s\n", argv[1]); return 2; }
    const uint32_t W = plan.W, H = plan.H;
    auto read_spv = [](const std::string& p) {
        const auto b = read_file(p);
        std::vector<uint32_t> w(b.size() / 4);
        std::memcpy(w.data(), b.data(), w.size() * 4);
        return w;
    };
    const auto vs_code = read_spv(plan.vs), ps_code = read_spv(plan.ps);
    if (vs_code.empty() || ps_code.empty()) { std::fprintf(stderr, "cannot read the shaders\n"); return 2; }
    const auto heap_bytes_in = read_file(plan.heap);
    if (heap_bytes_in.empty()) { std::fprintf(stderr, "cannot read the heap\n"); return 2; }
    for (const auto& im : plan.images) {
        if (!texel_bytes(im.format)) { std::fprintf(stderr, "unsupported image format %d\n", int(im.format)); return 2; }
        for (uint32_t va : im.vas)
            if (uint64_t(va) + 32 > heap_bytes_in.size()) { std::fprintf(stderr, "T# va 0x%x outside the heap\n", va); return 2; }
    }
    for (VkFormat t : plan.targets)
        if (!texel_bytes(t)) { std::fprintf(stderr, "unsupported target format %d\n", int(t)); return 2; }

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
    VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features q11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; q11.pNext = &q12;
    VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; qf2.pNext = &q11;
    vkGetPhysicalDeviceFeatures2(pd, &qf2);
    if (!q11.shaderDrawParameters || !q12.bufferDeviceAddress || !q12.runtimeDescriptorArray || !q12.descriptorIndexing ||
        !q12.shaderSampledImageArrayNonUniformIndexing || !qf2.features.shaderInt64 ||
        !qf2.features.shaderSampledImageArrayDynamicIndexing) {
        std::fprintf(stderr, "unsupported: needs shaderDrawParameters, bufferDeviceAddress, descriptor indexing and shaderInt64\n");
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
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE; f12.runtimeDescriptorArray = VK_TRUE; f12.descriptorIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext = &f12;
    f11.shaderDrawParameters = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f11;
    f2.features.shaderInt64 = VK_TRUE; f2.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    f2.features.shaderStorageImageArrayDynamicIndexing = qf2.features.shaderStorageImageArrayDynamicIndexing;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LD(n) auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory);
    LD(vkBindImageMemory); LD(vkCreateImageView); LD(vkCreateRenderPass); LD(vkCreateFramebuffer);
    LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline);
    LD(vkCmdDraw); LD(vkCmdEndRenderPass); LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage);
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
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

    leonida::BindlessSet set(vkGetDeviceProcAddr, dev, mp);
    if (!set.ok()) { std::fprintf(stderr, "cannot create the bindless set\n"); return 1; }

    // The heap: one buffer the shaders reach through the base in push-constant words 32/33.
    VkBuffer heap; void* heapp;
    CHECK(make_buffer(heap_bytes_in.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heapp));
    std::memcpy(heapp, heap_bytes_in.data(), heap_bytes_in.size());
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bdai.buffer = heap;
    const VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &bdai);

    // Images: create, upload, register in the set, write the T# at each va and map it in the hash table.
    VkDeviceSize up_bytes = 16;
    std::vector<VkDeviceSize> up_off(plan.images.size());
    std::vector<std::vector<char>> data(plan.images.size());
    for (size_t i = 0; i < plan.images.size(); ++i) {
        const auto& im = plan.images[i];
        const VkDeviceSize bytes = VkDeviceSize(im.w) * (im.type == "1d" ? 1 : im.h) * im.d * texel_bytes(im.format);
        data[i] = im.file == "-" ? std::vector<char>(bytes, 0) : read_file(im.file);
        if (data[i].size() != bytes) { std::fprintf(stderr, "image %zu: %s holds %zu bytes, expected %llu\n", i, im.file.c_str(), data[i].size(), (unsigned long long)bytes); return 2; }
        up_off[i] = up_bytes; up_bytes += (bytes + 15) & ~VkDeviceSize(15);
    }
    VkBuffer up; void* upp;
    CHECK(make_buffer(up_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up, &upp));
    for (size_t i = 0; i < plan.images.size(); ++i) std::memcpy(static_cast<char*>(upp) + up_off[i], data[i].data(), data[i].size());
    std::vector<VkImage> imgs(plan.images.size());
    for (size_t i = 0; i < plan.images.size(); ++i) {
        const auto& im = plan.images[i];
        const bool is3d = im.type == "3d", is1d = im.type == "1d", layered = im.type == "2darray" || im.type == "cube";
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = is3d ? VK_IMAGE_TYPE_3D : is1d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
        ii.format = im.format; ii.extent = {im.w, is1d ? 1 : im.h, is3d ? im.d : 1}; ii.mipLevels = 1;
        ii.arrayLayers = layered ? im.d : 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (im.type == "cube") ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &imgs[i]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, imgs[i], &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &ai, nullptr, &m)); CHECK(vkBindImageMemory(dev, imgs[i], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = imgs[i]; vi.format = im.format;
        vi.viewType = is3d ? VK_IMAGE_VIEW_TYPE_3D : is1d ? VK_IMAGE_VIEW_TYPE_1D : layered ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layered ? im.d : 1};
        VkImageView view; CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        const auto dim = is3d ? leonida::ImageDimension::dim_3d : is1d ? leonida::ImageDimension::dim_1d
                         : layered ? leonida::ImageDimension::dim_2d_array : leonida::ImageDimension::dim_2d;
        const uint32_t index = set.add_image(dim, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (!index) { std::fprintf(stderr, "image %zu: the set is full\n", i); return 1; }
        leonida::gfx10::Image t{};
        t.address = (uint64_t(i) + 1) << 24;   // unique per image, so every T# is its own key
        t.format = leonida::gfx10::format_from_vulkan(im.format);
        t.type = is3d ? leonida::gfx10::ImageType::image_3d : is1d ? leonida::gfx10::ImageType::image_1d
                 : im.type == "cube" ? leonida::gfx10::ImageType::cube
                 : layered ? leonida::gfx10::ImageType::image_2d_array : leonida::gfx10::ImageType::image_2d;
        t.width = im.w; t.height = is1d ? 1 : im.h; t.depth = is3d ? im.d : 1; t.layer_count = layered ? im.d : 1;
        t.swizzle = leonida::gfx10::format_swizzle(im.format);
        const auto words = leonida::gfx10::encode(t);
        for (uint32_t va : im.vas) std::memcpy(static_cast<char*>(heapp) + va, words.data(), 32);
        set.insert_key(leonida::make_tsharp_key(words.data()), index);
    }
    for (const auto& s : plan.samplers) {
        VkSamplerCreateInfo sci = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(
            leonida::gfx10::SamplerWords{s[0], s[1], s[2], s[3]}));
        VkSampler smp; CHECK(vkCreateSampler(dev, &sci, nullptr, &smp));
        const uint32_t index = set.add_sampler(smp);
        if (!index) { std::fprintf(stderr, "the sampler array is full\n"); return 1; }
        set.insert_key(leonida::make_ssharp_key(s.data()), index);
    }

    // Targets and the pass.
    const size_t NT = plan.targets.size();
    std::vector<VkImage> rt(NT); std::vector<VkImageView> rtv(NT);
    for (size_t t = 0; t < NT; ++t) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = plan.targets[t]; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &rt[t]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, rt[t], &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &ai, nullptr, &m)); CHECK(vkBindImageMemory(dev, rt[t], m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = rt[t]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = plan.targets[t];
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &rtv[t]));
    }
    std::vector<VkAttachmentDescription> ad(NT);
    std::vector<VkAttachmentReference> car(NT);
    for (size_t t = 0; t < NT; ++t) {
        ad[t] = {};
        ad[t].format = plan.targets[t]; ad[t].samples = VK_SAMPLE_COUNT_1_BIT; ad[t].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ad[t].storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad[t].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ad[t].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; ad[t].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ad[t].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        car[t] = {uint32_t(t), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = uint32_t(NT); sd.pColorAttachments = car.data();
    VkSubpassDependency dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = uint32_t(NT); rpi.pAttachments = ad.data(); rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = uint32_t(NT); fbi.pAttachments = rtv.data(); fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fbuf; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fbuf));

    auto module = [&](const std::vector<uint32_t>& code, VkShaderModule* m) {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs; CHECK(module(vs_code, &vs)); CHECK(module(ps_code, &fs));
    VkPipelineShaderStageCreateInfo st[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes};
    const VkDescriptorSetLayout dsl = set.layout();
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
    std::vector<VkPipelineColorBlendAttachmentState> cba(NT);
    for (auto& c : cba) { c = {}; c.colorWriteMask = 0xF; }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = uint32_t(NT); cb.pAttachments = cba.data();
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = st; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &dss; gpi.pColorBlendState = &cb;
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    std::vector<VkBuffer> rb(NT); std::vector<void*> rbp(NT);
    for (size_t t = 0; t < NT; ++t)
        CHECK(make_buffer(VkDeviceSize(W) * H * texel_bytes(plan.targets[t]), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb[t], &rbp[t]));

    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
    set.record_fallback_init(cmd);
    for (size_t i = 0; i < plan.images.size(); ++i) {
        const auto& im = plan.images[i];
        const bool layered = im.type == "2darray" || im.type == "cube", is3d = im.type == "3d", is1d = im.type == "1d";
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.image = imgs[i];
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layered ? im.d : 1};
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
        VkBufferImageCopy c{}; c.bufferOffset = up_off[i];
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layered ? im.d : 1};
        c.imageExtent = {im.w, is1d ? 1 : im.h, is3d ? im.d : 1};
        vkCmdCopyBufferToImage(cmd, up, imgs[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    if (!set.flush(cmd)) { std::fprintf(stderr, "cannot flush the bindless set\n"); return 1; }
    std::vector<VkClearValue> clear(NT);
    for (auto& c : clear) c = {};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fbuf; rbi.renderArea = sc; rbi.clearValueCount = uint32_t(NT); rbi.pClearValues = clear.data();
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    const VkDescriptorSet ds = set.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    uint32_t pc[push_words] = {};
    for (const auto& [r, v] : plan.push) pc[r] = v;
    pc[32] = uint32_t(heap_base); pc[33] = uint32_t(heap_base >> 32);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes, pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    for (size_t t = 0; t < NT; ++t) {
        VkBufferImageCopy rc{}; rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rc.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, rt[t], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb[t], 1, &rc);
    }
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));
    set.table_upload_result(true);

    std::printf("vk_lighting: device=\"%s\" size=%ux%u", props.deviceName, W, H);
    for (size_t t = 0; t < NT; ++t) {
        const size_t bytes = size_t(W) * H * texel_bytes(plan.targets[t]);
        const std::string path = plan.out + "_t" + std::to_string(t) + ".bin";
        std::ofstream o(path, std::ios::binary);
        o.write(static_cast<const char*>(rbp[t]), std::streamsize(bytes));
        if (!o) { std::fprintf(stderr, "\ncannot write %s\n", path.c_str()); return 1; }
        std::printf(" t%zu=%zu", t, bytes);
    }
    std::printf("\n");
    return 0;
}
