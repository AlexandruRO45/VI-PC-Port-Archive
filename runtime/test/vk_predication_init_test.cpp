// GPU regression for SetPredication suppressing internal lazy committed-texture zero-fill (detail::initialize)
// and unconditional XDXR BLAS/copy/postbuild dispatches.
// With a suppressing predicate (equal_zero == 0):
//   1. TransitionBarrier from COMMON -> COPY_SOURCE on a dirty UNDEFINED color image still zero-fills it.
//   2. TextureBarrier from COMMON -> COPY_SOURCE on a dirty UNDEFINED color image still zero-fills it.
//   3. ClearRenderTarget on a dirty UNDEFINED image still zero-fills it while suppressing the explicit clear.
//   4. ClearDepthStencil on a dirty UNDEFINED D32_SFLOAT_S8_UINT image still zero-fills both planes while suppressing the clear.
//   5. ClearUnorderedAccess on a dirty UNDEFINED UAV image still zero-fills it while suppressing the UAV clear.
//   6. CopyTexture of a 4x4 box into a dirty UNDEFINED 16x16 image still zero-fills surrounding texels.
//   7. XDXR BuildAccelerationStructure, CopyAccelerationStructure, and AccelerationStructurePostbuild still execute
//      while predicated out.
// Exits 77 without a Vulkan 1.3 + VK_EXT_conditional_rendering device. No external shaders or game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

constexpr std::uint32_t kTex = 16;
constexpr std::uint32_t kDirtyWord = 0xdeadbeefu;

bool prefill_color_u32(vktest::Device& d, VkImage img, std::uint32_t word) {
    const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto clear = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    if (!barrier || !clear) return false;
    return d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = img;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &to_dst);
        VkClearColorValue v{};
        v.uint32[0] = v.uint32[1] = v.uint32[2] = v.uint32[3] = word;
        clear(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &to_dst.subresourceRange);
        return true;
    });
}

bool prefill_color_f32(vktest::Device& d, VkImage img, float r, float g, float b, float a) {
    const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto clear = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    if (!barrier || !clear) return false;
    return d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = img;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &to_dst);
        VkClearColorValue v{};
        v.float32[0] = r;
        v.float32[1] = g;
        v.float32[2] = b;
        v.float32[3] = a;
        clear(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &to_dst.subresourceRange);
        return true;
    });
}

bool prefill_depth_stencil(vktest::Device& d, VkImage img, float depth, std::uint32_t stencil) {
    const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto clear = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    if (!barrier || !clear) return false;
    return d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = img;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &to_dst);
        const VkClearDepthStencilValue v{depth, stencil};
        clear(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &to_dst.subresourceRange);
        return true;
    });
}

ResourceBinding make_image_binding(VkImage img, VkFormat fmt, VkImageAspectFlags aspects, bool color_att = true) {
    ResourceBinding r{};
    r.image = img;
    r.format = fmt;
    r.aspects = aspects;
    r.width = kTex;
    r.height = kTex;
    r.mip_levels = 1;
    r.array_layers = 1;
    r.color_attachment = color_att;
    return r;
}

ResourceBinding make_buf_binding(VkBuffer buf, VkDeviceSize size, VkDeviceSize offset = 0, std::uint64_t va = 0,
                                 bool host_readable = false) {
    ResourceBinding r{};
    r.buffer = buf;
    r.buffer_offset = offset;
    r.size = size;
    r.va = va;
    r.host_readable = host_readable;
    return r;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no Vulkan 1.3 device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.conditional_rendering) {
        std::printf("SKIP: no VK_EXT_conditional_rendering\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);

    VulkanContext ctx(d.device_proc, d.device, true, true);
    EXPECT(ctx.loaded());
    ctx.configure_predication(d.device_proc, memory, props.limits.maxStorageBufferRange, d.family);
    VulkanQueue queue(ctx, d.queue, d.family);
    EXPECT(queue.ok());

    constexpr std::uint32_t kRowPitch = 256;
    constexpr std::uint32_t kReadbackBytes = kTex * kRowPitch;

    void* pred_map = nullptr;
    VkBuffer pred_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &pred_map);
    void* rb_map = nullptr;
    VkBuffer rb_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb_map);
    void* rb_stencil_map = nullptr;
    VkBuffer rb_stencil_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb_stencil_map);
    void* up_map = nullptr;
    VkBuffer up_buf = d.buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up_map);
    EXPECT(pred_buf && pred_map && rb_buf && rb_map && rb_stencil_buf && rb_stencil_map && up_buf && up_map);

    // Offset 0 holds 0 -> SetPredication(equal_zero) suppresses predicated commands.
    std::memset(pred_map, 0, 64);

    constexpr ResourceId kPred = 1, kRb = 2, kRbStencil = 3, kUp = 4;
    ctx.add_resource(kPred, make_buf_binding(pred_buf, 64));
    ctx.add_resource(kRb, make_buf_binding(rb_buf, kReadbackBytes, 0, 0, true));
    ctx.add_resource(kRbStencil, make_buf_binding(rb_stencil_buf, kReadbackBytes, 0, 0, true));
    ctx.add_resource(kUp, make_buf_binding(up_buf, 4096));

    const auto read_color_texel = [&](const void* map, std::uint32_t x, std::uint32_t y) -> std::uint32_t {
        const auto* bytes = static_cast<const std::uint8_t*>(map);
        std::uint32_t v = 0;
        std::memcpy(&v, bytes + std::size_t(y) * kRowPitch + std::size_t(x) * 4, 4);
        return v;
    };

    // 1. TransitionBarrier on dirty UNDEFINED image under suppressing predicate must zero-fill it.
    {
        VkImage img1 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
        EXPECT(img1 && prefill_color_u32(d, img1, kDirtyWord));
        constexpr ResourceId kImg1 = 10;
        ctx.add_resource(kImg1, make_image_binding(img1, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, false),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        EXPECT(list.transition({kImg1, all_subresources, d3d12_state::common, d3d12_state::copy_source}));
        EXPECT(list.set_predication({0, 0, 0}));
        CopyTexture ct{};
        ct.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct.src = {kImg1, false, 0, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(ct));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y)
            for (std::uint32_t x = 0; x < kTex; ++x) {
                const std::uint32_t got = read_color_texel(rb_map, x, y);
                if (got != 0u) {
                    std::fprintf(stderr, "subtest 5 (%u,%u) = 0x%08x\n", x, y, got);
                    EXPECT(got == 0u);
                }
            }
    }
    std::printf("1/7 TransitionBarrier lazy zero-fill under SetPredication: ok\n");

    // 2. TextureBarrier on dirty UNDEFINED image under suppressing predicate must zero-fill it.
    {
        VkImage img2 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
        EXPECT(img2 && prefill_color_u32(d, img2, kDirtyWord));
        constexpr ResourceId kImg2 = 11;
        ctx.add_resource(kImg2, make_image_binding(img2, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, false),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        TextureBarrier tb{};
        tb.sync_before = d3d12_sync::none;
        tb.sync_after = d3d12_sync::copy;
        tb.access_before = d3d12_access::no_access;
        tb.access_after = d3d12_access::copy_source;
        tb.layout_before = d3d12_layout::common;
        tb.layout_after = d3d12_layout::copy_source;
        tb.resource = kImg2;
        tb.range = {all_subresources, 0, 0, 0, 0, 0};
        EXPECT(list.barrier(tb));
        EXPECT(list.set_predication({0, 0, 0}));
        CopyTexture ct{};
        ct.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct.src = {kImg2, false, 0, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(ct));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y)
            for (std::uint32_t x = 0; x < kTex; ++x) {
                const std::uint32_t got = read_color_texel(rb_map, x, y);
                if (got != 0u) {
                    std::fprintf(stderr, "subtest 5 (%u,%u) = 0x%08x\n", x, y, got);
                    EXPECT(got == 0u);
                }
            }
    }
    std::printf("2/7 TextureBarrier lazy zero-fill under SetPredication: ok\n");

    // 3. ClearRenderTarget on dirty UNDEFINED image: initial zero-fill runs while explicit clear is suppressed.
    {
        VkImage img3 = d.image(VK_FORMAT_R8G8B8A8_UNORM, kTex, kTex, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        EXPECT(img3 && prefill_color_f32(d, img3, 0.25f, 0.5f, 0.75f, 1.0f));
        constexpr ResourceId kImg3 = 12;
        ctx.add_resource(kImg3, make_image_binding(img3, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, true),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        EXPECT(list.clear_render_target({kImg3, 0, 0, 1}, {1.0f, 0.0f, 0.0f, 1.0f}));
        EXPECT(list.set_predication({0, 0, 0}));
        EXPECT(list.transition({kImg3, all_subresources, d3d12_state::render_target, d3d12_state::copy_source}));
        CopyTexture ct{};
        ct.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct.src = {kImg3, false, 0, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(ct));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y)
            for (std::uint32_t x = 0; x < kTex; ++x) {
                const std::uint32_t got = read_color_texel(rb_map, x, y);
                if (got != 0u) {
                    std::fprintf(stderr, "subtest 5 (%u,%u) = 0x%08x\n", x, y, got);
                    EXPECT(got == 0u);
                }
            }
    }
    std::printf("3/7 ClearRenderTarget lazy zero-fill under SetPredication: ok\n");

    // 4. ClearDepthStencil on dirty UNDEFINED D32_SFLOAT_S8_UINT image: zero-fill runs while clear is suppressed.
    if (d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT)) {
        VkImage img4 = d.image(VK_FORMAT_D32_SFLOAT_S8_UINT, kTex, kTex, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        EXPECT(img4 && prefill_depth_stencil(d, img4, 0.75f, 0x5au));
        constexpr ResourceId kImg4 = 13;
        ctx.add_resource(kImg4,
                         make_image_binding(img4, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, false),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_map, 0xcc, kReadbackBytes);
        std::memset(rb_stencil_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        EXPECT(list.clear_depth_stencil({kImg4, 0, 0, 1}, clear_depth | clear_stencil, 0.25f, 0x33u));
        EXPECT(list.set_predication({0, 0, 0}));
        EXPECT(list.transition({kImg4, all_subresources, d3d12_state::depth_write, d3d12_state::copy_source}));
        CopyTexture ct_d{};
        ct_d.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct_d.src = {kImg4, false, 0, 0, 0, 0, 0, 0};
        CopyTexture ct_s{};
        ct_s.dst = {kRbStencil, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct_s.src = {kImg4, false, 1, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(ct_d));
        EXPECT(list.copy_texture(ct_s));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y) {
            const auto* srow = static_cast<const std::uint8_t*>(rb_stencil_map) + std::size_t(y) * kRowPitch;
            for (std::uint32_t x = 0; x < kTex; ++x) {
                EXPECT(read_color_texel(rb_map, x, y) == 0u);
                EXPECT(srow[x] == 0u);
            }
        }
        std::printf("4/7 ClearDepthStencil lazy zero-fill under SetPredication: ok\n");
    } else {
        std::printf("4/7 skipped: D32_SFLOAT_S8_UINT unavailable\n");
    }

    // 5. ClearUnorderedAccess on dirty UNDEFINED UAV image: initial zero-fill runs while UAV clear is suppressed.
    {
        VkImage img5 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        EXPECT(img5 && prefill_color_u32(d, img5, kDirtyWord));
        constexpr ResourceId kImg5 = 14;
        ctx.add_resource(kImg5, make_image_binding(img5, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, true),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        std::memset(rb_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        const UavView uav{kImg5, false, VK_FORMAT_R32_UINT, 0, 0, 0, 0, 0, 1};
        EXPECT(list.clear_unordered_access(uav, false, {0x12345678u, 0x12345678u, 0x12345678u, 0x12345678u}));
        EXPECT(list.set_predication({0, 0, 0}));
        EXPECT(list.transition({kImg5, all_subresources, d3d12_state::unordered_access, d3d12_state::copy_source}));
        CopyTexture ct{};
        ct.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        ct.src = {kImg5, false, 0, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(ct));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y)
            for (std::uint32_t x = 0; x < kTex; ++x) {
                const std::uint32_t got = read_color_texel(rb_map, x, y);
                if (got != 0u) {
                    std::fprintf(stderr, "subtest 5 (%u,%u) = 0x%08x\n", x, y, got);
                    EXPECT(got == 0u);
                }
            }
    }
    std::printf("5/7 ClearUnorderedAccess lazy zero-fill under SetPredication: ok\n");

    // 6. CopyTexture partial box into dirty UNDEFINED image under predication still zero-fills untouched texels.
    {
        VkImage img6 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
        EXPECT(img6 && prefill_color_u32(d, img6, kDirtyWord));
        constexpr ResourceId kImg6 = 15;
        ctx.add_resource(kImg6, make_image_binding(img6, VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, false),
                         VK_IMAGE_LAYOUT_UNDEFINED);
        for (std::uint32_t y = 0; y < 4; ++y) {
            auto* row = reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(up_map) + std::size_t(y) * kRowPitch);
            for (std::uint32_t x = 0; x < 4; ++x) row[x] = 0x11223344u;
        }
        std::memset(rb_map, 0xcc, kReadbackBytes);

        CommandList list;
        EXPECT(list.set_predication({kPred, 0, d3d12_predication::equal_zero}));
        CopyTexture up{};
        up.dst = {kImg6, false, 0, 0, 0, 0, 0, 0};
        up.src = {kUp, true, 0, 0, 4, 4, 1, kRowPitch};
        up.has_box = true;
        up.box = {0, 0, 0, 4, 4, 1};
        EXPECT(list.copy_texture(up));
        EXPECT(list.set_predication({0, 0, 0}));
        EXPECT(list.transition({kImg6, all_subresources, d3d12_state::copy_dest, d3d12_state::copy_source}));
        CopyTexture rb{};
        rb.dst = {kRb, true, 0, 0, kTex, kTex, 1, kRowPitch};
        rb.src = {kImg6, false, 0, 0, 0, 0, 0, 0};
        EXPECT(list.copy_texture(rb));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        for (std::uint32_t y = 0; y < kTex; ++y)
            for (std::uint32_t x = 0; x < kTex; ++x)
                if (x >= 4 || y >= 4) EXPECT(read_color_texel(rb_map, x, y) == 0u);
    }
    std::printf("6/7 CopyTexture partial-box lazy zero-fill under SetPredication: ok\n");

    // 7. XDXR BLAS build, copy, and postbuild under suppressing predication must execute unconditionally.
    if (d.buffer_device_address) {
        VulkanContext xdxr_ctx(d.device_proc, d.device, true, true, false, true);
        EXPECT(xdxr_ctx.loaded() && xdxr_ctx.xdxr_acceleration_structures());
        xdxr_ctx.configure_predication(d.device_proc, memory, props.limits.maxStorageBufferRange, d.family);
        VulkanQueue xdxr_queue(xdxr_ctx, d.queue, d.family);
        EXPECT(xdxr_queue.ok());

        XdxrLayout expected{};
        EXPECT(expected.compute(1, false) && expected.size > 0 && expected.size <= 4096 && expected.scratch <= 4096);

        constexpr VkDeviceSize kArenaBytes = 32768;
        constexpr VkDeviceSize kOffVerts = 0;
        constexpr VkDeviceSize kOffScratch = 256;
        constexpr VkDeviceSize kOffDst = 4096;
        constexpr VkDeviceSize kOffClone = 8192;
        constexpr VkDeviceSize kOffPostbuild = 12288;
        constexpr VkDeviceSize kOffPred = 16384;
        constexpr VkDeviceSize kOffRb = 20480;
        constexpr std::uint64_t kBaseVa = 0x10000000ull;

        void* arena_map = nullptr;
        VkBuffer arena = d.buffer(kArenaBytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  &arena_map);
        EXPECT(arena && arena_map);
        const std::uint64_t arena_addr = d.device_address(arena);
        EXPECT(arena_addr != 0);
        xdxr_ctx.set_heap_base(arena_addr - kBaseVa);

        std::memset(arena_map, 0, kArenaBytes);
        const float tri[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        std::memcpy(static_cast<std::uint8_t*>(arena_map) + kOffVerts, tri, sizeof(tri));
        std::memset(static_cast<std::uint8_t*>(arena_map) + kOffRb, 0xcc, 12288);

        constexpr ResourceId kXPred = 20, kXVerts = 21, kXScratch = 22, kXDst = 23, kXClone = 24, kXPost = 25,
                             kXRb = 26;
        xdxr_ctx.add_resource(kXPred, make_buf_binding(arena, 256, kOffPred));
        xdxr_ctx.add_resource(kXVerts, make_buf_binding(arena, 256, kOffVerts, kBaseVa + kOffVerts));
        xdxr_ctx.add_resource(kXScratch, make_buf_binding(arena, 4096, kOffScratch, kBaseVa + kOffScratch));
        xdxr_ctx.add_resource(kXDst, make_buf_binding(arena, 4096, kOffDst, kBaseVa + kOffDst));
        xdxr_ctx.add_resource(kXClone, make_buf_binding(arena, 4096, kOffClone, kBaseVa + kOffClone));
        xdxr_ctx.add_resource(kXPost, make_buf_binding(arena, 256, kOffPostbuild, kBaseVa + kOffPostbuild));
        xdxr_ctx.add_resource(kXRb, make_buf_binding(arena, 12288, kOffRb, 0, true));

        CommandList list;
        EXPECT(list.set_predication({kXPred, 0, d3d12_predication::equal_zero}));
        BuildAccelerationStructure build{};
        build.dst_va = kBaseVa + kOffDst;
        build.scratch_va = kBaseVa + kOffScratch;
        RtTriangleGeometry geom{};
        geom.vertex_va = kBaseVa + kOffVerts;
        geom.vertex_count = 3;
        geom.vertex_stride = 12;
        geom.vertex_format = VK_FORMAT_R32G32B32_SFLOAT;
        build.triangles.push_back(geom);
        EXPECT(list.build_acceleration_structure(build));
        EXPECT(list.uav_barrier({0}));
        CopyAccelerationStructure copy{};
        copy.src_va = kBaseVa + kOffDst;
        copy.dst_va = kBaseVa + kOffClone;
        copy.mode = 0;
        EXPECT(list.copy_acceleration_structure(copy));
        EXPECT(list.uav_barrier({0}));
        AccelerationStructurePostbuild post{};
        post.dest_va = kBaseVa + kOffPostbuild;
        post.type = 3;
        post.sources.push_back(kBaseVa + kOffDst);
        EXPECT(list.acceleration_structure_postbuild(post));
        EXPECT(list.uav_barrier({0}));
        EXPECT(list.set_predication({0, 0, 0}));
        EXPECT(list.copy_buffer({kXRb, 0, kXDst, 0, expected.size}));
        EXPECT(list.copy_buffer({kXRb, 4096, kXClone, 0, expected.size}));
        EXPECT(list.copy_buffer({kXRb, 8192, kXPost, 0, 8}));
        EXPECT(list.close());
        const CommandList* ptr = &list;
        EXPECT(xdxr_queue.execute({&ptr, 1}) == ReplayResult::success);
        EXPECT(xdxr_queue.idle());

        const auto* rb = static_cast<const std::uint8_t*>(arena_map) + kOffRb;
        std::uint64_t reported_size = 0;
        std::memcpy(&reported_size, rb + 8192, sizeof(reported_size));
        EXPECT(reported_size == expected.size);
        bool any_nonzero = false;
        for (std::uint32_t i = 0; i < expected.size; ++i) any_nonzero |= rb[i] != 0;
        EXPECT(any_nonzero);
        EXPECT(std::memcmp(rb, rb + 4096, expected.size) == 0);
        std::printf("7/7 XDXR BLAS build, copy, and postbuild under SetPredication: ok\n");
    } else {
        std::printf("7/7 skipped: bufferDeviceAddress unavailable\n");
    }

    if (d.validation) {
        std::printf("validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    std::printf("vk_predication_init_test: ok\n");
    return 0;
}
