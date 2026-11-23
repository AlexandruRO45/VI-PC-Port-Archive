// Block-compressed footprint copies in vulkan_replay.h must reject copies that run past the block-padded mip edge
// or end inside a block without reaching the mip edge (VUID-vkCmdCopyBufferToImage-imageExtent-00207 / 00208 and
// VUID-vkCmdCopyImageToBuffer-imageExtent-00207 / 00208), while keeping valid padded and mip-edge copies intact.
// No Vulkan loader, GPU or game data: fake Vulkan callbacks capture the emitted copy regions.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

std::vector<VkBufferImageCopy> b2i_copies;
std::vector<VkBufferImageCopy> i2b_copies;

VKAPI_ATTR void VKAPI_CALL noop_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                        std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL noop_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}

VKAPI_ATTR void VKAPI_CALL noop_copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                           std::uint32_t, const VkImageCopy*) {}

VKAPI_ATTR void VKAPI_CALL rec_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) b2i_copies.push_back(regions[i]);
}

VKAPI_ATTR void VKAPI_CALL rec_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) i2b_copies.push_back(regions[i]);
}

template<typename T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

int checks = 0;
int failures = 0;
void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace leonida;

    ReplayFunctions vk{};
    vk.pipeline_barrier = noop_barrier;
    vk.copy_buffer = noop_copy_buffer;
    vk.copy_buffer_to_image = rec_copy_b2i;
    vk.copy_image_to_buffer = rec_copy_i2b;
    vk.copy_image = noop_copy_image;

    ResourceBinding buf{};
    buf.buffer = handle<VkBuffer>(1);
    buf.size = 65536;
    buf.host_readable = true;

    const auto make_bc = [](std::uintptr_t id, VkFormat format, std::uint32_t width, std::uint32_t height,
                            std::uint32_t mips = 1) {
        ResourceBinding img{};
        img.image = handle<VkImage>(id);
        img.format = format;
        img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        img.width = width;
        img.height = height;
        img.mip_levels = mips;
        img.array_layers = 1;
        img.depth = 1;
        img.samples = 1;
        return img;
    };

    const ResourceBinding bc1_8x8 = make_bc(2, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 8, 4);  // mips: 8x8, 4x4, 2x2, 1x1
    const ResourceBinding bc1_7x5 = make_bc(3, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 7, 5, 1);
    const ResourceBinding bc7_7x5 = make_bc(4, VK_FORMAT_BC7_UNORM_BLOCK, 7, 5, 1);

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case 1: return &buf;
        case 2: return &bc1_8x8;
        case 3: return &bc1_7x5;
        case 4: return &bc7_7x5;
        default: return nullptr;
        }
    };

    const auto run_copy = [&](const CopyTexture& copy) {
        b2i_copies.clear();
        i2b_copies.clear();
        CommandList list;
        if (!list.copy_texture(copy) || !list.close()) return ReplayResult::invalid_copy;
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    const auto tex_loc = [](ResourceId id, std::uint32_t sub = 0) {
        return CopyLocation{id, false, sub, 0, 0, 0, 0, 0};
    };
    const auto fp_loc = [](std::uint32_t w, std::uint32_t h, std::uint32_t pitch = 256) {
        return CopyLocation{1, true, 0, 0, w, h, 1, pitch};
    };

    // --- Valid BC1 / BC7 footprint uploads & readbacks ---

    // 1. 8x8 BC1 whole upload (exact block multiple).
    expect(run_copy({tex_loc(2), 0, 0, 0, fp_loc(8, 8), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 8 && b2i_copies[0].imageExtent.height == 8,
           "8x8 BC1 whole upload succeeds with 8x8 extent");

    // 2. 7x5 BC1 whole upload from 8x8 block-padded footprint clamps cleanly to mip edge (7x5).
    expect(run_copy({tex_loc(3), 0, 0, 0, fp_loc(8, 8), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 7 && b2i_copies[0].imageExtent.height == 5,
           "7x5 BC1 padded 8x8 upload clamps to 7x5 mip edge");

    // 3. 7x5 BC1 partial edge-block upload (4x4 footprint at (4, 4)) clamps cleanly to 3x1 (4+3==7, 4+1==5).
    expect(run_copy({tex_loc(3), 4, 4, 0, fp_loc(4, 4), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageOffset.x == 4 && b2i_copies[0].imageOffset.y == 4 &&
               b2i_copies[0].imageExtent.width == 3 && b2i_copies[0].imageExtent.height == 1,
           "7x5 BC1 edge block at (4,4) clamps to 3x1");

    // 4. 7x5 BC1 interior whole-block upload (4x4 at (0, 0)).
    expect(run_copy({tex_loc(3), 0, 0, 0, fp_loc(4, 4), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 4 && b2i_copies[0].imageExtent.height == 4,
           "7x5 BC1 interior 4x4 block upload succeeds");

    // 5. 7x5 BC1 readback with block-padded box {0,0,0, 8,8,1} clamps to 7x5.
    expect(run_copy({fp_loc(8, 8), 0, 0, 0, tex_loc(3), true, {0, 0, 0, 8, 8, 1}}) == ReplayResult::success &&
               i2b_copies.size() == 1 && i2b_copies[0].imageExtent.width == 7 && i2b_copies[0].imageExtent.height == 5,
           "7x5 BC1 readback with padded 8x8 box clamps to 7x5");

    // 6. 7x5 BC1 readback with exact mip-edge box {0,0,0, 7,5,1} (also used by staged ReadFromSubresource).
    expect(run_copy({fp_loc(8, 8), 0, 0, 0, tex_loc(3), true, {0, 0, 0, 7, 5, 1}}) == ReplayResult::success &&
               i2b_copies.size() == 1 && i2b_copies[0].imageExtent.width == 7 && i2b_copies[0].imageExtent.height == 5,
           "7x5 BC1 readback with exact 7x5 mip-edge box succeeds");

    // 7. 7x5 BC1 edge-block readback with padded {4,4,0, 8,8,1} and exact {4,4,0, 7,5,1}.
    expect(run_copy({fp_loc(4, 4), 0, 0, 0, tex_loc(3), true, {4, 4, 0, 8, 8, 1}}) == ReplayResult::success &&
               i2b_copies.size() == 1 && i2b_copies[0].imageExtent.width == 3 && i2b_copies[0].imageExtent.height == 1,
           "7x5 BC1 edge-block readback with padded 4..8 box clamps to 3x1");
    expect(run_copy({fp_loc(4, 4), 0, 0, 0, tex_loc(3), true, {4, 4, 0, 7, 5, 1}}) == ReplayResult::success &&
               i2b_copies.size() == 1 && i2b_copies[0].imageExtent.width == 3 && i2b_copies[0].imageExtent.height == 1,
           "7x5 BC1 edge-block readback with exact 4..7 x 4..5 box succeeds");

    // 8. 8x8 BC1 mip 3 (1x1) and mip 2 (2x2) padded 4x4 uploads clamp to 1x1 and 2x2.
    expect(run_copy({tex_loc(2, 3), 0, 0, 0, fp_loc(4, 4), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 1 && b2i_copies[0].imageExtent.height == 1,
           "1x1 BC1 mip upload from 4x4 footprint clamps to 1x1");
    expect(run_copy({tex_loc(2, 2), 0, 0, 0, fp_loc(4, 4), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 2 && b2i_copies[0].imageExtent.height == 2,
           "2x2 BC1 mip upload from 4x4 footprint clamps to 2x2");

    // 9. 7x5 BC7 (16-byte blocks) whole upload from 8x8 footprint clamps cleanly to 7x5.
    expect(run_copy({tex_loc(4), 0, 0, 0, fp_loc(8, 8), false, {}}) == ReplayResult::success &&
               b2i_copies.size() == 1 && b2i_copies[0].imageExtent.width == 7 && b2i_copies[0].imageExtent.height == 5,
           "7x5 BC7 padded 8x8 upload clamps to 7x5");

    // --- Invalid out-of-bounds BC footprint uploads & readbacks (must NOT silently truncate) ---

    expect(run_copy({tex_loc(2), 0, 0, 0, fp_loc(16, 16), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "8x8 BC1 rejects oversized 16x16 upload");
    expect(run_copy({tex_loc(2), 4, 0, 0, fp_loc(8, 8), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "8x8 BC1 rejects 8x8 upload at x=4 (runs past width 8)");
    expect(run_copy({tex_loc(2), 0, 4, 0, fp_loc(8, 8), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "8x8 BC1 rejects 8x8 upload at y=4 (runs past height 8)");
    expect(run_copy({tex_loc(3), 0, 0, 0, fp_loc(12, 8), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "7x5 BC1 rejects oversized 12x8 upload (past padded width 8)");
    expect(run_copy({tex_loc(3), 0, 0, 0, fp_loc(8, 12), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "7x5 BC1 rejects oversized 8x12 upload (past padded height 8)");
    expect(run_copy({tex_loc(3), 4, 0, 0, fp_loc(8, 4), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "7x5 BC1 rejects 8x4 upload at x=4 (past padded width 8)");
    expect(run_copy({tex_loc(2, 3), 0, 0, 0, fp_loc(8, 4), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "1x1 BC1 mip rejects 8x4 upload (past padded width 4)");
    expect(run_copy({fp_loc(12, 8), 0, 0, 0, tex_loc(2), true, {0, 0, 0, 12, 8, 1}}) == ReplayResult::invalid_copy &&
               i2b_copies.empty(),
           "8x8 BC1 rejects oversized 12x8 readback box");
    expect(run_copy({fp_loc(12, 8), 0, 0, 0, tex_loc(3), true, {0, 0, 0, 12, 8, 1}}) == ReplayResult::invalid_copy &&
               i2b_copies.empty(),
           "7x5 BC1 rejects oversized 12x8 readback box");
    expect(run_copy({tex_loc(4), 4, 0, 0, fp_loc(8, 4), false, {}}) == ReplayResult::invalid_copy && b2i_copies.empty(),
           "7x5 BC7 rejects 8x4 upload at x=4");

    // --- Invalid interior non-block-aligned BC extents (VUID-vkCmdCopyBufferToImage-imageExtent-00207 / 00208) ---

    expect(run_copy({tex_loc(2), 0, 0, 0, fp_loc(8, 8), true, {0, 0, 0, 3, 4, 1}}) == ReplayResult::invalid_copy &&
               b2i_copies.empty(),
           "8x8 BC1 upload rejects interior unaligned width 3 (VUID-00207)");
    expect(run_copy({tex_loc(2), 0, 0, 0, fp_loc(8, 8), true, {0, 0, 0, 4, 3, 1}}) == ReplayResult::invalid_copy &&
               b2i_copies.empty(),
           "8x8 BC1 upload rejects interior unaligned height 3 (VUID-00208)");
    expect(run_copy({fp_loc(8, 8), 0, 0, 0, tex_loc(2), true, {0, 0, 0, 6, 4, 1}}) == ReplayResult::invalid_copy &&
               i2b_copies.empty(),
           "8x8 BC1 readback rejects interior unaligned width 6 (VUID-00207)");
    expect(run_copy({fp_loc(8, 8), 0, 0, 0, tex_loc(2), true, {0, 0, 0, 4, 6, 1}}) == ReplayResult::invalid_copy &&
               i2b_copies.empty(),
           "8x8 BC1 readback rejects interior unaligned height 6 (VUID-00208)");
    expect(run_copy({tex_loc(3), 0, 0, 0, fp_loc(8, 8), true, {0, 0, 0, 6, 4, 1}}) == ReplayResult::invalid_copy &&
               b2i_copies.empty(),
           "7x5 BC1 upload rejects interior unaligned width 6 (< mip width 7)");
    expect(run_copy({fp_loc(8, 8), 0, 0, 0, tex_loc(3), true, {0, 0, 0, 6, 4, 1}}) == ReplayResult::invalid_copy &&
               i2b_copies.empty(),
           "7x5 BC1 readback rejects interior unaligned width 6 (< mip width 7)");
    expect(run_copy({tex_loc(3), 4, 0, 0, fp_loc(8, 8), true, {0, 0, 0, 2, 4, 1}}) == ReplayResult::invalid_copy &&
               b2i_copies.empty(),
           "7x5 BC1 upload at x=4 rejects unaligned width 2 (4+2 != 7)");

    std::printf("replay_bc_copy_bounds_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
