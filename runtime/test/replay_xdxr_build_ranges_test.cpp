// XDXR build range checks and replay ordering with synthetic VAs. No driver or GPU is used.
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>

int main() {
    using namespace leonida;

    BufferVaIndex<ResourceId> buffers;
    ReplayFunctions vk{};
    vk.xdxr_check = [&](const BuildAccelerationStructure& build) {
        return VulkanContext::xdxr_build_supported(build, buffers);
    };
    unsigned records = 0;
    vk.xdxr_record = [&](VkCommandBuffer, const BuildAccelerationStructure&) { ++records; };
    const auto resolve = [](ResourceId) -> const ResourceBinding* { return nullptr; };

    unsigned failures = 0;
    const auto run = [&](const BuildAccelerationStructure& build, ReplayResult expected, const char* what) {
        CommandList list;
        if (!list.build_acceleration_structure(build) || !list.close()) {
            std::fprintf(stderr, "FAIL: %s: could not make a closed command list\n", what);
            ++failures;
            return;
        }
        const unsigned before = records;
        const auto actual = replay(list, VK_NULL_HANDLE, resolve, vk);
        const unsigned expected_records = expected == ReplayResult::success ? 1 : 0;
        if (actual != expected || records - before != expected_records) {
            std::fprintf(stderr, "FAIL: %s: replay result %d (expected %d), recorded %u (expected %u)\n",
                         what, int(actual), int(expected), records - before, expected_records);
            ++failures;
        }
    };

    constexpr std::uint64_t dst_base = 0x10000, dst_offset = 1024;
    constexpr std::uint64_t scratch_base = 0x20000, scratch_offset = 256;
    constexpr std::uint64_t src_base = 0x30000, src_offset = 256;
    constexpr std::uint64_t instances_base = 0x40000, instances_offset = 16;
    constexpr ResourceId dst = 1, scratch = 2, src = 3, other = 4, instances = 5;

    BuildAccelerationStructure build{};
    build.dst_va = dst_base + dst_offset;
    build.scratch_va = scratch_base + scratch_offset;
    build.triangles.push_back({0x50000, 6, 12, VK_FORMAT_R32G32B32_SFLOAT});
    XdxrLayout layout;
    if (!layout.compute(2, false)) return 1;
    const auto success = ReplayResult::success;
    const auto unsupported = ReplayResult::unsupported;
    const auto dst_size = dst_offset + layout.size;
    const auto scratch_size = scratch_offset + layout.scratch;

    buffers.insert(dst_base, dst_size, dst);
    buffers.insert(scratch_base, scratch_size, scratch);
    run(build, success, "BLAS accepts exact destination and scratch spans at nonzero offsets");

    buffers.insert(dst_base, dst_size - 1, dst);
    buffers.insert(dst_base + dst_size - 1, 256, other);
    run(build, unsupported, "BLAS destination cannot continue into an adjacent buffer");
    buffers.erase(dst_base + dst_size - 1, other);
    buffers.erase(dst_base, dst);
    run(build, unsupported, "BLAS rejects an unregistered destination");

    buffers.insert(dst_base, dst_size, dst);
    buffers.insert(build.dst_va, layout.size - 1, other);
    run(build, success, "BLAS accepts an enclosing destination when the nearer alias is short");
    buffers.erase(build.dst_va, other);

    buffers.insert(scratch_base, scratch_size - 1, scratch);
    run(build, unsupported, "BLAS rejects a scratch span one byte short");
    buffers.erase(scratch_base, scratch);
    run(build, unsupported, "BLAS rejects an unregistered scratch address");
    buffers.insert(scratch_base, scratch_size, scratch);

    // The source is used only for an out-of-place refit.
    build.src_va = src_base + src_offset;
    run(build, success, "a fresh build ignores an unused source address");
    build.update = true;
    buffers.insert(src_base, src_offset + layout.size, src);
    run(build, unsupported, "refit requires ALLOW_UPDATE even with valid spans");
    buffers.erase(src_base, src);
    build.build_flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    run(build, unsupported, "out-of-place refit rejects an unregistered source");
    buffers.insert(src_base, src_offset + layout.size - 1, src);
    run(build, unsupported, "out-of-place refit rejects a short source");
    buffers.insert(src_base, src_offset + layout.size, src);
    run(build, success, "out-of-place refit accepts an exact source span");
    build.src_va = dst_base + src_offset;
    run(build, success, "out-of-place refit accepts disjoint offsets in one buffer");
    build.src_va = build.dst_va;
    run(build, success, "refit accepts an explicitly in-place source");
    build.src_va = 0;
    run(build, success, "refit accepts an implicit in-place source");

    build.update = false;
    build.build_flags = VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
    build.triangles.front().vertex_count = 12;
    XdxrLayout full_boxes;
    if (!layout.compute(4, false, true) || !full_boxes.compute(4, false) ||
        layout.size >= full_boxes.size) return 1;
    buffers.insert(scratch_base, scratch_offset + layout.scratch, scratch);
    buffers.insert(dst_base, dst_offset + layout.size, dst);
    run(build, success, "BOX16 accepts the smaller exact destination span");
    build.build_flags = 0;
    run(build, unsupported, "BOX32 requires more space than the BOX16 span");

    build = {};
    build.is_tlas = true;
    build.dst_va = dst_base + dst_offset;
    build.scratch_va = scratch_base + scratch_offset;
    if (!layout.compute(0, true)) return 1;
    buffers.insert(dst_base, dst_offset + layout.size, dst);
    buffers.insert(scratch_base, scratch_offset + layout.scratch, scratch);
    run(build, success, "empty TLAS needs no instance buffer");

    build.instance_count = 3;
    if (!layout.compute(build.instance_count, true)) return 1;
    buffers.insert(dst_base, dst_offset + layout.size, dst);
    buffers.insert(scratch_base, scratch_offset + layout.scratch, scratch);
    run(build, unsupported, "nonempty TLAS needs a nonnull instance address");
    build.instances_va = instances_base + instances_offset;
    run(build, unsupported, "TLAS rejects an unregistered instance address");
    constexpr std::uint64_t instance_bytes = 64;
    const auto instances_size = instances_offset + instance_bytes * build.instance_count;
    buffers.insert(instances_base, instances_size - 1, instances);
    run(build, unsupported, "TLAS rejects an incomplete final instance descriptor");
    buffers.insert(instances_base, instances_size, instances);
    run(build, success, "TLAS accepts an exact array of contiguous instance descriptors");

    if (failures) return 1;
    std::puts("XDXR build ranges: exact spans replay and invalid ranges stop before recording");
    return 0;
}
