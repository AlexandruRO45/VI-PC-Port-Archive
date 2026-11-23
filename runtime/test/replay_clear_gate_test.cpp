// Replay with vkCmdClearColorImage / vkCmdClearDepthStencilImage unset: a list that needs one of them, as a
// whole-view clear or as the implicit zero-clear of a never-used (UNDEFINED) image, is refused as unsupported
// instead of calling a null pointer (#634 finding by @bobcatchris). No Vulkan loader, GPU, shaders or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {
unsigned clears = 0, attachment_clears = 0;
std::vector<VkImageMemoryBarrier> g_undefined_barriers;
std::vector<VkImageSubresourceRange> g_depth_clear_ranges;
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t image_count, const VkImageMemoryBarrier* images) {
    for (std::uint32_t i = 0; i < image_count; ++i)
        if (images[i].oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) g_undefined_barriers.push_back(images[i]);
}
VKAPI_ATTR void VKAPI_CALL clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
    std::uint32_t, const VkImageSubresourceRange*) { ++clears; }
VKAPI_ATTR void VKAPI_CALL clear_depth(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
    std::uint32_t range_count, const VkImageSubresourceRange* ranges) {
    ++clears;
    for (std::uint32_t i = 0; i < range_count; ++i) g_depth_clear_ranges.push_back(ranges[i]);
}
VKAPI_ATTR void VKAPI_CALL copy_buf(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL copy_img(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
    std::uint32_t, const VkImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout,
    std::uint32_t, const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer,
    std::uint32_t, const VkBufferImageCopy*) {}
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t, const VkClearAttachment*, std::uint32_t,
    const VkClearRect*) { ++attachment_clears; }

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

leonida::ResourceBinding image(VkFormat format, VkImageAspectFlags aspects) {
    leonida::ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    b.format = format;
    b.aspects = aspects;
    b.width = b.height = 4;
    b.depth = b.mip_levels = b.array_layers = b.samples = 1;
    b.color_attachment = aspects == VK_IMAGE_ASPECT_COLOR_BIT;
    return b;
}

// Everything a clear can use, minus the two image clears; each case adds back what it needs.
leonida::ReplayFunctions functions() {
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    return vk;
}

leonida::ReplayResult run(const leonida::CommandList& list, const leonida::ResourceBinding& binding,
                          const leonida::ReplayFunctions& vk, leonida::LayoutTracker* tracker = nullptr) {
    clears = attachment_clears = 0;
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &binding; }, vk, tracker);
}
} // namespace

int main() {
    using leonida::ReplayResult;
    const auto color = image(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const auto depth = image(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    const leonida::Rect rect{1, 1, 3, 3};

    // A whole-view render-target clear runs vkCmdClearColorImage.
    leonida::CommandList whole_color;
    expect(whole_color.clear_render_target({1, 0, 0, 1}, {0.25f, 0.5f, 0.75f, 1}) && whole_color.close(),
           "record a whole-view colour clear");
    auto vk = functions();
    expect(run(whole_color, color, vk) == ReplayResult::unsupported, "whole-view colour clear without clear_color_image");
    vk.clear_color_image = clear_color;
    expect(run(whole_color, color, vk) == ReplayResult::success && clears == 1, "whole-view colour clear with it");

    // A whole-view depth clear runs vkCmdClearDepthStencilImage.
    leonida::CommandList whole_depth;
    expect(whole_depth.clear_depth_stencil({1, 0, 0, 1}, leonida::clear_depth, 1.0f, 0) && whole_depth.close(),
           "record a whole-view depth clear");
    vk = functions();
    expect(run(whole_depth, depth, vk) == ReplayResult::unsupported,
           "whole-view depth clear without clear_depth_stencil_image");
    vk.clear_depth_stencil_image = clear_depth;
    expect(run(whole_depth, depth, vk) == ReplayResult::success && clears == 1, "whole-view depth clear with it");

    // A rectangle clear goes through vkCmdClearAttachments, but on a never-used image the zero-clear that
    // initializes it first needs vkCmdClearColorImage (the case that crashed in detail::initialize).
    leonida::CommandList rect_color;
    expect(rect_color.clear_render_target({1, 0, 0, 1}, {1, 0, 0, 1}, std::span<const leonida::Rect>(&rect, 1)) &&
           rect_color.close(), "record a rectangle colour clear");
    vk = functions();
    expect(run(rect_color, color, vk) == ReplayResult::success && attachment_clears == 1,
           "rectangle clear without a tracker needs no image clear");
    leonida::LayoutTracker tracker;
    tracker.track(1, color, VK_IMAGE_LAYOUT_UNDEFINED);
    expect(run(rect_color, color, vk, &tracker) == ReplayResult::unsupported,
           "rectangle clear on an UNDEFINED image without clear_color_image");
    expect(tracker.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED, "a refused list leaves the tracker untouched");
    vk.clear_color_image = clear_color;
    expect(run(rect_color, color, vk, &tracker) == ReplayResult::success && clears == 1 && attachment_clears == 1,
           "rectangle clear on an UNDEFINED image: zero-clear, then the rectangle");
    // Once initialized, the image no longer needs the zero-clear.
    vk.clear_color_image = nullptr;
    expect(run(rect_color, color, vk, &tracker) == ReplayResult::success && attachment_clears == 1,
           "rectangle clear on an initialized image needs no image clear");

    // not_zeroed images (D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) are only moved out of UNDEFINED, never cleared.
    auto not_zeroed = color;
    not_zeroed.not_zeroed = true;
    leonida::LayoutTracker fresh;
    fresh.track(1, not_zeroed, VK_IMAGE_LAYOUT_UNDEFINED);
    expect(run(rect_color, not_zeroed, vk, &fresh) == ReplayResult::success && clears == 0,
           "not_zeroed image: no zero-clear needed");

    // Multi-mip and multi-layer depth/stencil images leave UNDEFINED across the whole aspect on first use, so
    // uploading mips or slices one at a time never emits a later UNDEFINED transition that resets shared HiZ state.
    {
        leonida::ResourceBinding upload_buf{};
        upload_buf.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
        upload_buf.size = 65536;

        auto make_vk = [&](bool separate) {
            auto f = functions();
            f.clear_color_image = clear_color;
            f.clear_depth_stencil_image = clear_depth;
            f.copy_buffer = copy_buf;
            f.copy_image = copy_img;
            f.copy_buffer_to_image = copy_b2i;
            f.copy_image_to_buffer = copy_i2b;
            f.separate_depth_stencil_layouts = separate;
            return f;
        };

        auto run_uploads = [&](const leonida::ResourceBinding& img, const leonida::ReplayFunctions& f,
                               leonida::LayoutTracker& tr, const std::vector<std::uint32_t>& subresources,
                               const std::vector<std::pair<std::uint32_t, std::uint32_t>>& dims) {
            clears = 0;
            g_undefined_barriers.clear();
            g_depth_clear_ranges.clear();
            leonida::CommandList list;
            for (std::size_t i = 0; i < subresources.size(); ++i) {
                const leonida::CopyLocation dst{1, false, subresources[i], 0, 0, 0, 0, 0};
                const leonida::CopyLocation src{2, true, 0, 0, dims[i].first, dims[i].second, 1, 256};
                expect(list.copy_texture({dst, 0, 0, 0, src, false, {}}), "record per-subresource upload");
            }
            expect(list.close(), "close upload list");
            const auto resolve = [&](leonida::ResourceId id) -> const leonida::ResourceBinding* {
                if (id == 1) return &img;
                if (id == 2) return &upload_buf;
                return nullptr;
            };
            return leonida::replay(list, VK_NULL_HANDLE, resolve, f, &tr);
        };

        // 1. 3-mip D32_FLOAT: uploading mips 0, 1, 2 emits one UNDEFINED barrier and one depth zero-clear over all 3 mips.
        auto d32_3mips = depth;
        d32_3mips.width = d32_3mips.height = 16;
        d32_3mips.mip_levels = 3;
        leonida::LayoutTracker tr_mips(true);
        tr_mips.track(1, d32_3mips, VK_IMAGE_LAYOUT_UNDEFINED);
        expect(run_uploads(d32_3mips, make_vk(true), tr_mips, {0, 1, 2}, {{16, 16}, {8, 8}, {4, 4}}) ==
                   ReplayResult::success,
               "3-mip D32_FLOAT sequential uploads succeed");
        expect(g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
                   g_undefined_barriers[0].subresourceRange.baseMipLevel == 0 &&
                   g_undefined_barriers[0].subresourceRange.levelCount == 3 &&
                   g_undefined_barriers[0].subresourceRange.baseArrayLayer == 0 &&
                   g_undefined_barriers[0].subresourceRange.layerCount == 1,
               "3-mip D32_FLOAT leaves UNDEFINED once across all 3 mips");
        expect(g_depth_clear_ranges.size() == 1 && g_depth_clear_ranges[0].baseMipLevel == 0 &&
                   g_depth_clear_ranges[0].levelCount == 3,
               "3-mip D32_FLOAT zero-clears all 3 mips together");
        for (std::uint32_t mip = 0; mip < 3; ++mip)
            expect(tr_mips.layout(1, mip) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   "3-mip D32_FLOAT tracker records TRANSFER_DST for every mip");

        // 2. not_zeroed 3-mip D32_FLOAT: still leaves UNDEFINED once across all 3 mips, with zero clears.
        auto d32_nz = d32_3mips;
        d32_nz.not_zeroed = true;
        leonida::LayoutTracker tr_nz(true);
        tr_nz.track(1, d32_nz, VK_IMAGE_LAYOUT_UNDEFINED);
        expect(run_uploads(d32_nz, make_vk(true), tr_nz, {0, 1, 2}, {{16, 16}, {8, 8}, {4, 4}}) == ReplayResult::success,
               "not_zeroed 3-mip D32_FLOAT uploads succeed");
        expect(clears == 0 && g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.levelCount == 3,
               "not_zeroed 3-mip D32_FLOAT leaves UNDEFINED once across all 3 mips without clearing");

        // 3. 2-layer D24S8 with separate depth/stencil layouts: each aspect initializes all layers once, independently.
        auto d24s8_2layers = image(VK_FORMAT_D24_UNORM_S8_UINT,
                                   VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
        d24s8_2layers.width = 20;
        d24s8_2layers.height = 12;
        d24s8_2layers.array_layers = 2;
        leonida::LayoutTracker tr_ds(true);
        tr_ds.track(1, d24s8_2layers, VK_IMAGE_LAYOUT_UNDEFINED);
        expect(run_uploads(d24s8_2layers, make_vk(true), tr_ds, {0, 1}, {{20, 12}, {20, 12}}) == ReplayResult::success,
               "2-layer D24S8 depth uploads succeed");
        expect(g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
                   g_undefined_barriers[0].subresourceRange.layerCount == 2,
               "2-layer D24S8 depth plane leaves UNDEFINED once across both layers");
        expect(tr_ds.layout(1, 0) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   tr_ds.layout(1, 1) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   tr_ds.layout(1, 2) == VK_IMAGE_LAYOUT_UNDEFINED &&
                   tr_ds.layout(1, 3) == VK_IMAGE_LAYOUT_UNDEFINED,
               "2-layer D24S8 depth init keeps stencil plane UNDEFINED until used");
        expect(run_uploads(d24s8_2layers, make_vk(true), tr_ds, {2, 3}, {{20, 12}, {20, 12}}) == ReplayResult::success,
               "2-layer D24S8 stencil uploads succeed");
        expect(g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT &&
                   g_undefined_barriers[0].subresourceRange.layerCount == 2,
               "2-layer D24S8 stencil plane leaves UNDEFINED once across both layers");

        // 4. Partially initialized depth image (mip 0 already in SHADER_READ_ONLY): uploading mip 1 does not clobber mip 0.
        leonida::LayoutTracker tr_partial(true);
        tr_partial.track(1, d32_3mips, VK_IMAGE_LAYOUT_UNDEFINED);
        tr_partial.set(1, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        expect(run_uploads(d32_3mips, make_vk(true), tr_partial, {1}, {{8, 8}}) == ReplayResult::success,
               "partially initialized D32_FLOAT mip 1 upload succeeds");
        expect(g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.baseMipLevel == 1 &&
                   g_undefined_barriers[0].subresourceRange.levelCount == 1,
               "partially initialized D32_FLOAT initializes only mip 1");
        expect(tr_partial.layout(1, 0) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                   tr_partial.layout(1, 1) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   tr_partial.layout(1, 2) == VK_IMAGE_LAYOUT_UNDEFINED,
               "partially initialized D32_FLOAT preserves mip 0 and leaves mip 2 untouched");

        // 5. Colour multi-mip image still initializes only the touched mip.
        auto color_3mips = color;
        color_3mips.width = color_3mips.height = 16;
        color_3mips.mip_levels = 3;
        leonida::LayoutTracker tr_color(true);
        tr_color.track(1, color_3mips, VK_IMAGE_LAYOUT_UNDEFINED);
        expect(run_uploads(color_3mips, make_vk(true), tr_color, {0}, {{16, 16}}) == ReplayResult::success,
               "3-mip colour mip 0 upload succeeds");
        expect(g_undefined_barriers.size() == 1 &&
                   g_undefined_barriers[0].subresourceRange.baseMipLevel == 0 &&
                   g_undefined_barriers[0].subresourceRange.levelCount == 1 &&
                   tr_color.layout(1, 1) == VK_IMAGE_LAYOUT_UNDEFINED &&
                   tr_color.layout(1, 2) == VK_IMAGE_LAYOUT_UNDEFINED,
               "3-mip colour image initializes only mip 0");
    }

    if (failures) return 1;
    std::printf("replay_clear_gate_test: whole-view, implicit zero-clears and multi-subresource depth init OK\n");
    return 0;
}
