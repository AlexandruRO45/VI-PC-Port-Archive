// Unit tests for queue family selection (runtime/src/queue_families.h). CPU-only, no Vulkan instance and no GPU:
// exercises dedicated async compute/DMA copy selection, queueCount == 0 skipping, 1x1x1 granularity rules,
// graphics fallback, and sparse_capable_queue_families filtering.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/queue_families.h"

#include <array>
#include <cstdint>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::QueueFamilies;
using leonida::pick_queue_families;
using leonida::sparse_capable_queue_families;

namespace {

VkQueueFamilyProperties make_family(VkQueueFlags flags, std::uint32_t count = 1,
                                    VkExtent3D granularity = {1, 1, 1}) {
    VkQueueFamilyProperties f{};
    f.queueFlags = flags;
    f.queueCount = count;
    f.minImageTransferGranularity = granularity;
    return f;
}

constexpr VkQueueFlags k_gfx = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
constexpr VkQueueFlags k_cmp = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
constexpr VkQueueFlags k_dma = VK_QUEUE_TRANSFER_BIT;

int test_dedicated_async_and_dma() {
    const std::array families = {
        make_family(k_gfx),
        make_family(k_cmp),
        make_family(k_dma),
    };
    const QueueFamilies q = pick_queue_families(families, 0);
    EXPECT(q.direct == 0);
    EXPECT(q.compute == 1);
    EXPECT(q.copy == 2);
    return 0;
}

int test_first_matching_family_wins_and_skips_zero_count() {
    const std::array families = {
        make_family(k_gfx),
        make_family(k_cmp, 0),
        make_family(k_dma, 0),
        make_family(k_cmp, 2),
        make_family(k_cmp, 4),
        make_family(k_dma, 1),
        make_family(k_dma, 2),
    };
    const QueueFamilies q = pick_queue_families(families, 0);
    EXPECT(q.direct == 0);
    EXPECT(q.compute == 3);
    EXPECT(q.copy == 5);
    return 0;
}

int test_dma_requires_1x1x1_granularity() {
    const std::array families = {
        make_family(k_gfx),
        make_family(k_dma, 1, {2, 1, 1}),
        make_family(k_dma, 1, {1, 2, 1}),
        make_family(k_dma, 1, {1, 1, 2}),
        make_family(k_dma, 1, {0, 0, 0}),
        make_family(k_dma, 1, {1, 1, 1}),
    };
    const QueueFamilies q = pick_queue_families(families, 0);
    EXPECT(q.direct == 0);
    EXPECT(q.compute == 0);
    EXPECT(q.copy == 5);
    return 0;
}

int test_copy_falls_back_to_fine_grained_async_compute() {
    const std::array families = {
        make_family(k_gfx),
        make_family(k_dma, 1, {4, 4, 1}),
        make_family(k_cmp, 1, {1, 1, 1}),
    };
    const QueueFamilies q = pick_queue_families(families, 0);
    EXPECT(q.direct == 0);
    EXPECT(q.compute == 2);
    EXPECT(q.copy == 2);
    return 0;
}

int test_copy_falls_back_to_graphics_when_async_compute_is_coarse() {
    const std::array families = {
        make_family(k_dma, 1, {2, 2, 1}),
        make_family(k_cmp, 1, {1, 2, 1}),
        make_family(k_gfx, 1, {1, 1, 1}),
    };
    const QueueFamilies q = pick_queue_families(families, 2);
    EXPECT(q.direct == 2);
    EXPECT(q.compute == 1);
    EXPECT(q.copy == 2);
    return 0;
}

int test_no_dedicated_families_or_empty_span() {
    const std::array families = {
        make_family(k_gfx),
        make_family(k_gfx),
    };
    const QueueFamilies q = pick_queue_families(families, 1);
    EXPECT(q.direct == 1);
    EXPECT(q.compute == 1);
    EXPECT(q.copy == 1);

    const QueueFamilies empty = pick_queue_families({}, 3);
    EXPECT(empty.direct == 3);
    EXPECT(empty.compute == 3);
    EXPECT(empty.copy == 3);
    return 0;
}

int test_sparse_capable_disabled_or_unsupported_direct() {
    const std::array families = {
        make_family(k_gfx | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_cmp),
        make_family(k_dma),
    };
    const QueueFamilies picked = pick_queue_families(families, 0);

    const QueueFamilies no_sparse = sparse_capable_queue_families(families, picked, false);
    EXPECT(no_sparse.direct == 0 && no_sparse.compute == 1 && no_sparse.copy == 2);

    const QueueFamilies out_of_range = sparse_capable_queue_families(families, {5, 1, 2}, true);
    EXPECT(out_of_range.direct == 5 && out_of_range.compute == 1 && out_of_range.copy == 2);

    const std::array no_direct_sparse = {
        make_family(k_gfx),
        make_family(k_cmp),
        make_family(k_dma),
    };
    const QueueFamilies kept = sparse_capable_queue_families(no_direct_sparse, picked, true);
    EXPECT(kept.direct == 0 && kept.compute == 1 && kept.copy == 2);
    return 0;
}

int test_sparse_capable_filters_non_sparse_and_oob_families() {
    const std::array both_sparse = {
        make_family(k_gfx | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_cmp | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_dma | VK_QUEUE_SPARSE_BINDING_BIT),
    };
    const QueueFamilies kept = sparse_capable_queue_families(both_sparse, {0, 1, 2}, true);
    EXPECT(kept.direct == 0 && kept.compute == 1 && kept.copy == 2);

    const std::array only_compute_sparse = {
        make_family(k_gfx | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_cmp | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_dma),
    };
    const QueueFamilies copy_fallback = sparse_capable_queue_families(only_compute_sparse, {0, 1, 2}, true);
    EXPECT(copy_fallback.direct == 0 && copy_fallback.compute == 1 && copy_fallback.copy == 0);

    const std::array only_copy_sparse = {
        make_family(k_gfx | VK_QUEUE_SPARSE_BINDING_BIT),
        make_family(k_cmp),
        make_family(k_dma | VK_QUEUE_SPARSE_BINDING_BIT),
    };
    const QueueFamilies compute_fallback = sparse_capable_queue_families(only_copy_sparse, {0, 1, 2}, true);
    EXPECT(compute_fallback.direct == 0 && compute_fallback.compute == 0 && compute_fallback.copy == 2);

    const QueueFamilies oob_fallback = sparse_capable_queue_families(both_sparse, {0, 9, 10}, true);
    EXPECT(oob_fallback.direct == 0 && oob_fallback.compute == 0 && oob_fallback.copy == 0);
    return 0;
}

}  // namespace

int main() {
    if (test_dedicated_async_and_dma() != 0) return 1;
    if (test_first_matching_family_wins_and_skips_zero_count() != 0) return 1;
    if (test_dma_requires_1x1x1_granularity() != 0) return 1;
    if (test_copy_falls_back_to_fine_grained_async_compute() != 0) return 1;
    if (test_copy_falls_back_to_graphics_when_async_compute_is_coarse() != 0) return 1;
    if (test_no_dedicated_families_or_empty_span() != 0) return 1;
    if (test_sparse_capable_disabled_or_unsupported_direct() != 0) return 1;
    if (test_sparse_capable_filters_non_sparse_and_oob_families() != 0) return 1;
    std::printf("queue_families_test: ok\n");
    return 0;
}
