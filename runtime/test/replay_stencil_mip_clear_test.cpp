// A ClearDepthStencilView with STENCIL and rects on a mip past 0 writes the stencil by copies, not through the
// attachment: NVIDIA's Vulkan driver on Blackwell sends a stencil attachment view of mip > 0 to mip 0 (issue 5256). Depth
// stays an attachment clear; mip 0, no transfer scratch, or a predicated clear (conditional rendering doesn't cover
// copies) keeps the attachment path for both aspects. Checks the recorded Vulkan calls. No Vulkan loader, GPU, shaders
// or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
VkImageAspectFlags cleared_aspects = 0;
unsigned attachment_clears = 0, fills = 0;
std::uint32_t fill_data = 0;
VkDeviceSize fill_offset = 0, fill_size = 0;
std::vector<VkBufferImageCopy> copies;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear_depth(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
    std::uint32_t, const VkImageSubresourceRange*) {}
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* a, std::uint32_t,
    const VkClearRect*) {
    if (n == 1) cleared_aspects = a[0].aspectMask;
    ++attachment_clears;
}
VKAPI_ATTR void VKAPI_CALL fill(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data) {
    ++fills;
    fill_offset = offset;
    fill_size = size;
    fill_data = data;
}
VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t n,
    const VkBufferImageCopy* regions) {
    copies.assign(regions, regions + n);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

leonida::ResourceBinding depth_stencil() {
    leonida::ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    b.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    b.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    b.width = b.height = 8;
    b.depth = b.samples = 1;
    b.mip_levels = 3;
    b.array_layers = 2;
    return b;
}

VKAPI_ATTR void VKAPI_CALL begin_conditional(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {}
VKAPI_ATTR void VKAPI_CALL end_conditional(VkCommandBuffer) {}

leonida::ReplayFunctions functions(bool scratch) {
    leonida::ReplayFunctions vk{};
    vk.begin_conditional_rendering = begin_conditional;
    vk.end_conditional_rendering = end_conditional;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, leonida::PredicateCapture& out) {
        out = {source, offset, reinterpret_cast<VkBuffer>(std::uintptr_t{3}), 0, VK_NULL_HANDLE};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const leonida::PredicateCapture&) {};
    vk.pipeline_barrier = barrier;
    vk.clear_depth_stencil_image = clear_depth;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.fill_buffer = fill;
    vk.copy_buffer_to_image = copy;
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    if (scratch)
        vk.copy_scratch = [](VkDeviceSize, VkBuffer& buffer, VkDeviceSize& offset) {
            buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{7});
            offset = 6;  // not a multiple of 4: the copies must start at 8
            return true;
        };
    return vk;
}

bool run(std::uint32_t mip, std::uint32_t flags, bool scratch, bool predicated = false) {
    leonida::CommandList list;
    const leonida::Rect rects[] = {{1, 1, 3, 2}, {0, 0, 1, 1}};
    if (predicated && !list.set_predication({2, 0, leonida::d3d12_predication::equal_zero})) return false;
    if (!list.clear_depth_stencil({1, mip, 1, 1}, flags, 0.5f, 0x77, rects) || !list.close()) return false;
    const auto binding = depth_stencil();
    leonida::ResourceBinding predicate{};
    predicate.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
    predicate.size = 8;
    cleared_aspects = 0;
    attachment_clears = fills = 0;
    copies.clear();
    return leonida::replay(list, VK_NULL_HANDLE,
                           [&](leonida::ResourceId id) { return id == 2 ? &predicate : &binding; },
                           functions(scratch)) == leonida::ReplayResult::success;
}
} // namespace

int main() {
    const std::uint32_t both = leonida::clear_depth | leonida::clear_stencil;

    // Mip 1, depth and stencil: depth through the attachment, stencil filled and copied into mip 1, layer 1.
    expect(run(1, both, true), "mip 1 depth+stencil replays");
    expect(attachment_clears == 1 && cleared_aspects == VK_IMAGE_ASPECT_DEPTH_BIT, "mip 1: the attachment clears depth only");
    expect(fills == 1 && fill_data == 0x77777777u && fill_offset == 8 && fill_size % 4 == 0 && fill_size >= 2,
           "mip 1: the scratch holds the stencil value, from a 4-byte-aligned offset");
    expect(copies.size() == 2, "mip 1: one copy per rect");
    if (copies.size() == 2) {
        const VkBufferImageCopy& a = copies[0];
        expect(a.imageSubresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT && a.imageSubresource.mipLevel == 1 &&
                   a.imageSubresource.baseArrayLayer == 1 && a.imageSubresource.layerCount == 1,
               "mip 1: the copy targets the stencil of mip 1, layer 1");
        expect(a.imageOffset.x == 1 && a.imageOffset.y == 1 && a.imageExtent.width == 2 && a.imageExtent.height == 1 &&
                   a.imageExtent.depth == 1 && a.bufferOffset == 8,
               "mip 1: the copy covers the rect");
    }

    // Mip 1, stencil only: no attachment clear at all.
    expect(run(1, leonida::clear_stencil, true), "mip 1 stencil-only replays");
    expect(attachment_clears == 0 && fills == 1 && copies.size() == 2, "mip 1 stencil only: copies, no attachment");

    // Mip 1, depth only: unchanged.
    expect(run(1, leonida::clear_depth, true), "mip 1 depth-only replays");
    expect(attachment_clears == 1 && cleared_aspects == VK_IMAGE_ASPECT_DEPTH_BIT && fills == 0 && copies.empty(),
           "mip 1 depth only: the attachment path");

    // Mip 0: the attachment path clears both.
    expect(run(0, both, true), "mip 0 replays");
    expect(attachment_clears == 1 && cleared_aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) &&
               fills == 0 && copies.empty(),
           "mip 0: both aspects through the attachment");

    // No transfer scratch: the attachment path, as before.
    expect(run(1, both, false), "mip 1 without scratch replays");
    expect(attachment_clears == 1 && cleared_aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) &&
               copies.empty(),
           "mip 1 without scratch: the attachment path");

    // Under SetPredication: the attachment path for both aspects, which conditional rendering covers. A copy would
    // write the stencil of a clear the predicate skips while its depth half is skipped.
    expect(run(1, both, true, true), "predicated mip 1 replays");
    expect(attachment_clears == 1 && cleared_aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) &&
               fills == 0 && copies.empty(),
           "predicated mip 1: no fill or copy, both aspects through the attachment");

    if (failures) return 1;
    std::puts("replay_stencil_mip_clear_test: OK");
    return 0;
}
