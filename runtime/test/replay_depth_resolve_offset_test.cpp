// Shifted depth/stencil resolves prepare an intermediate before emitting commands. Verify the stencil-only
// render and copy, with distinct source and destination array slices. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>

namespace {
using namespace leonida;

template<class T>
T handle(std::uintptr_t n) { return reinterpret_cast<T>(n); }

struct Trace {
    unsigned barriers = 0, begins = 0, ends = 0, copies = 0, transfer_resolves = 0;
    VkRect2D render_area{};
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkRenderingAttachmentInfo stencil{};
    bool has_depth = false;
    VkImage copy_src = VK_NULL_HANDLE, copy_dst = VK_NULL_HANDLE;
    VkImageCopy region{};
} trace;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) { ++trace.barriers; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo* info) {
    ++trace.begins;
    trace.render_area = info->renderArea;
    trace.has_depth = info->pDepthAttachment != nullptr;
    if (info->pStencilAttachment) trace.stencil = *info->pStencilAttachment;
}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) { ++trace.ends; }
VKAPI_ATTR void VKAPI_CALL begin_pass(VkCommandBuffer, const VkRenderPassBeginInfo* info, VkSubpassContents) {
    ++trace.begins;
    trace.render_area = info->renderArea;
    trace.framebuffer = info->framebuffer;
}
VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkImage source, VkImageLayout, VkImage dest, VkImageLayout,
                                std::uint32_t count, const VkImageCopy* regions) {
    trace.copies += count;
    trace.copy_src = source;
    trace.copy_dst = dest;
    if (count == 1) trace.region = regions[0];
}
VKAPI_ATTR void VKAPI_CALL resolve_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                         std::uint32_t, const VkImageResolve*) { ++trace.transfer_resolves; }

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool untouched() { return !trace.barriers && !trace.begins && !trace.ends && !trace.copies && !trace.transfer_resolves; }
}  // namespace

int main() {
    constexpr VkImageAspectFlagBits stencil = VK_IMAGE_ASPECT_STENCIL_BIT;
    ResourceBinding source{};
    source.image = handle<VkImage>(1);
    source.format = VK_FORMAT_D24_UNORM_S8_UINT;
    source.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | stencil;
    source.width = 8;
    source.height = 6;
    source.depth = source.mip_levels = 1;
    source.array_layers = 2;
    source.samples = 4;
    auto dest = source;
    dest.image = handle<VkImage>(2);
    dest.width = 16;
    dest.height = 12;
    dest.mip_levels = 2;
    dest.array_layers = 3;
    dest.samples = 1;
    const auto scratch = handle<VkImage>(3);
    const auto source_view = handle<VkImageView>(4), dest_view = handle<VkImageView>(5);
    const auto scratch_view = handle<VkImageView>(6);

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &source;
        return id == 2 ? &dest : nullptr;
    };
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.copy_image = copy;
    vk.resolve_image = resolve_image;
    vk.image_view = [&](const ResourceBinding& binding, const VkImageSubresourceRange& range) {
        const bool from_source = binding.image == source.image;
        expect(range.aspectMask == stencil && range.baseMipLevel == (from_source ? 0u : 1u) &&
               range.baseArrayLayer == (from_source ? 1u : 2u), "render view selects the source or destination slice");
        return from_source ? source_view : dest_view;
    };
    const auto run = [&](const ResolveRegion& region, bool depth_only = false) {
        trace = {};
        CommandList list;
        expect(list.resolve_subresource({2, depth_only ? 5u : 11u, 1, depth_only ? 1u : 3u, region}) &&
               list.close(), "record a closed resolve list");
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    constexpr auto maximum = d3d12_resolve_mode::maximum;
    const ResolveRegion shifted{3, 2, Rect{2, 1, 7, 5}, maximum};
    expect(run(shifted) == ReplayResult::unsupported && untouched(),
           "a shifted resolve without an intermediate hook stops before emission");
    vk.depth_resolve_scratch = [](VkFormat, VkImageAspectFlags, VkExtent2D, VkImage&, VkImageView&) { return false; };
    expect(run(shifted) == ReplayResult::unsupported && untouched(),
           "intermediate allocation failure stops before emission");
    vk.depth_resolve_scratch = [](VkFormat, VkImageAspectFlags, VkExtent2D, VkImage&, VkImageView&) { return true; };
    expect(run(shifted) == ReplayResult::unsupported && untouched(), "incomplete intermediate stops before emission");

    unsigned allocations = 0;
    vk.depth_resolve_scratch = [&](VkFormat format, VkImageAspectFlags aspect, VkExtent2D extent,
                                   VkImage& image, VkImageView& view) {
        ++allocations;
        expect(format == source.format && aspect == stencil && extent.width == 8 && extent.height == 6,
               "the intermediate matches the source format, plane and extent");
        image = scratch;
        view = scratch_view;
        return true;
    };
    expect(run(shifted) == ReplayResult::success && allocations == 1, "shifted resolve gets one intermediate");
    expect(trace.begins == 1 && trace.ends == 1 && !trace.transfer_resolves && !trace.has_depth &&
           trace.stencil.imageView == source_view && trace.stencil.resolveImageView == scratch_view &&
           trace.stencil.resolveMode == VK_RESOLVE_MODE_MAX_BIT,
           "dynamic rendering resolves stencil only into the intermediate");
    expect(trace.render_area.offset.x == 2 && trace.render_area.offset.y == 1 &&
           trace.render_area.extent.width == 5 && trace.render_area.extent.height == 4,
           "render area uses the source rectangle");
    const auto& r = trace.region;
    expect(trace.copies == 1 && trace.copy_src == scratch && trace.copy_dst == dest.image &&
           r.srcSubresource.aspectMask == stencil && r.srcSubresource.mipLevel == 0 &&
           r.srcSubresource.baseArrayLayer == 0 && r.srcSubresource.layerCount == 1 &&
           r.dstSubresource.aspectMask == stencil && r.dstSubresource.mipLevel == 1 &&
           r.dstSubresource.baseArrayLayer == 2 && r.dstSubresource.layerCount == 1 &&
           r.srcOffset.x == 2 && r.srcOffset.y == 1 && r.dstOffset.x == 3 && r.dstOffset.y == 2 &&
           r.extent.width == 5 && r.extent.height == 4 && r.extent.depth == 1,
           "the copy preserves aspect, offsets and the original destination slice");

    expect(run({4, 2, Rect{2, 1, 7, 5}, maximum}) == ReplayResult::invalid_copy && allocations == 1 && untouched(),
           "a shifted rectangle past the destination edge fails before allocating or emitting");
    expect(run({0, 0, std::nullopt, maximum}) == ReplayResult::success && allocations == 1 &&
           trace.begins == 1 && trace.ends == 1 && trace.stencil.resolveImageView == dest_view && !trace.copies,
           "an unshifted resolve keeps the direct path");

    // On drivers that resolve layer zero in dynamic rendering, shifted depth-only array resolves
    // still need the render pass with a framebuffer that binds the scratch image, not the final target.
    source.format = dest.format = VK_FORMAT_D32_SFLOAT;
    source.aspects = dest.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    vk.image_view = [&](const ResourceBinding& binding, const VkImageSubresourceRange& range) {
        expect(binding.image == source.image && range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
               range.baseArrayLayer == 1, "the source view selects depth from array layer one");
        return source_view;
    };
    vk.depth_resolve_scratch = [&](VkFormat format, VkImageAspectFlags aspect, VkExtent2D extent,
                                   VkImage& image, VkImageView& view) {
        expect(format == VK_FORMAT_D32_SFLOAT && aspect == VK_IMAGE_ASPECT_DEPTH_BIT &&
               extent.width == 8 && extent.height == 6, "depth scratch has the source extent and plane");
        image = scratch;
        view = scratch_view;
        return true;
    };
    vk.begin_render_pass = begin_pass;
    vk.end_render_pass = end;
    const auto framebuffer = handle<VkFramebuffer>(7);
    bool fail_framebuffer = true;
    vk.depth_resolve_framebuffer = [&](const ResourceBinding& from, VkImageView from_view, VkImageLayout from_layout,
                                       const ResourceBinding& to, VkImageView to_view, VkImageLayout to_layout,
                                       VkResolveModeFlagBits mode, VkExtent2D extent, VkRenderPass& pass) {
        expect(from.image == source.image && from_view == source_view && to.image == scratch &&
               to_view == scratch_view && from_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
               to_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && mode == VK_RESOLVE_MODE_MAX_BIT &&
               extent.width == 8 && extent.height == 6, "the render pass binds depth source and scratch views");
        pass = handle<VkRenderPass>(8);
        return fail_framebuffer ? VK_NULL_HANDLE : framebuffer;
    };
    expect(run(shifted, true) == ReplayResult::unsupported && untouched(),
           "scratch framebuffer failure stops before emission");
    fail_framebuffer = false;
    expect(run(shifted, true) == ReplayResult::success && trace.begins == 1 && trace.ends == 1 &&
           trace.framebuffer == framebuffer && trace.copies == 1 && trace.copy_src == scratch &&
           trace.region.dstSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
           trace.region.dstSubresource.baseArrayLayer == 2,
           "shifted depth uses the render-pass workaround then copies to the selected destination layer");

    if (failures) return 1;
    std::puts("depth/stencil resolve offset: failures, stencil copy and layered depth render pass verified");
    return 0;
}
