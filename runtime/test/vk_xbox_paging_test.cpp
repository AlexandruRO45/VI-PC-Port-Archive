// The Xbox page mapper (src/xbox_paging.h) on a real device: a 64 GiB sparse arena with a fixed device address,
// physical pages from 64 MiB device-local chunks, and RegisterPagePoolX / CopyPageMappingsBatchX /
// ClearPageMappingsX semantics, each call one vkQueueBindSparse ordered on the queue (bind_sparse_ordered).
//
// Checks (FAIL = exit 1): remaps between GPU reads return the newly mapped page's data, in order; the arena's device
// address never changes; two arena pages aliasing one physical page see each other's writes; a cleared page reads 0
// (where the device guarantees it, residencyNonResidentStrict);
// contiguous runs coalesce into one bind and scattered ones do not; refused calls bind nothing; a destroyed arena,
// a released page and a held or dropped bind each leave the pool able to free its chunk, and only then; a bind on
// another queue's timeline is refused; no validation errors (LEONIDA_VK_VALIDATION=1).
// Measurements (INFO lines): bind round-trip cost per call by page count, contiguous and scattered; and whether a
// remap delays GPU work that was queued before it (the gap a bind puts between two submissions on the queue).
// No device, no sparse residency for buffers, or less than 64 GiB of sparse address space: SKIP, exit 77.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/xbox_paging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace leonida;
using namespace leonida::xbox;

namespace {

int g_failures = 0;
#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

std::uint32_t pattern(std::uint32_t k) { return 0xA5000000u | (k * 0x01010101u & 0x00FFFFFFu) | k; }

// GPU work outside VulkanQueue::execute(), ordered on the queue's timeline: it waits for everything the queue has
// submitted (binds included), and the queue's next bind waits for it.
struct Gpu {
    vktest::Device& d;
    VulkanContext& context;
    VulkanQueue& queue;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::uint64_t value = 0;

    Gpu(vktest::Device& device, VulkanContext& ctx, VulkanQueue& q) : d(device), context(ctx), queue(q) {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = d.family;
        d.fn<PFN_vkCreateCommandPool>("vkCreateCommandPool")(d.device, &info, nullptr, &pool);
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        d.fn<PFN_vkCreateSemaphore>("vkCreateSemaphore")(d.device, &semaphore, nullptr, &timeline);
    }
    ~Gpu() {
        d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
        if (pool) d.fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(d.device, pool, nullptr);
        if (timeline) d.fn<PFN_vkDestroySemaphore>("vkDestroySemaphore")(d.device, timeline, nullptr);
    }

    // Records with `body`, submits, and returns the value `timeline` reaches when it is done.
    template<typename Body>
    std::uint64_t submit(Body&& body) {
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        d.fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(d.device, &alloc, &cb);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        d.fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(cb, &begin);
        body(cb);
        d.fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(cb);
        const std::uint64_t after = queue.submitted_ticket();
        const VkSemaphore wait_semaphore = queue.timeline();
        const std::uint64_t signal = ++value;
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = after ? 1 : 0;
        values.pWaitSemaphoreValues = &after;
        values.signalSemaphoreValueCount = 1;
        values.pSignalSemaphoreValues = &signal;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
        info.waitSemaphoreCount = after ? 1 : 0;
        info.pWaitSemaphores = &wait_semaphore;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cb;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &timeline;
        {
            std::lock_guard lock(context.submit_mutex());
            EXPECT(d.fn<PFN_vkQueueSubmit>("vkQueueSubmit")(d.queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS);
        }
        queue.wait(timeline, signal);  // the queue's next bind or submission comes after this
        retired_.push_back({cb, signal});
        return signal;
    }

    void wait(std::uint64_t v) {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline;
        info.pValues = &v;
        EXPECT(d.fn<PFN_vkWaitSemaphores>("vkWaitSemaphores")(d.device, &info, 10'000'000'000ull) == VK_SUCCESS);
        for (auto it = retired_.begin(); it != retired_.end();) {
            if (it->second > v) { ++it; continue; }
            d.fn<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(d.device, pool, 1, &it->first);
            it = retired_.erase(it);
        }
    }

    void wait_queue(std::uint64_t ticket) {
        const VkSemaphore s = queue.timeline();
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &s;
        info.pValues = &ticket;
        EXPECT(d.fn<PFN_vkWaitSemaphores>("vkWaitSemaphores")(d.device, &info, 10'000'000'000ull) == VK_SUCCESS);
    }

    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> retired_;
};

struct DeviceBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

DeviceBuffer device_local(vktest::Device& d, const VkPhysicalDeviceMemoryProperties& props, VkDeviceSize size) {
    DeviceBuffer out;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    d.fn<PFN_vkCreateBuffer>("vkCreateBuffer")(d.device, &info, nullptr, &out.buffer);
    VkMemoryRequirements req{};
    d.fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(d.device, out.buffer, &req);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i)) ||
            !(props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            continue;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = i;
        if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &alloc, nullptr, &out.memory) == VK_SUCCESS)
            break;
    }
    d.fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(d.device, out.buffer, out.memory, 0);
    return out;
}

void destroy(vktest::Device& d, DeviceBuffer& b) {
    if (b.buffer) d.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer")(d.device, b.buffer, nullptr);
    if (b.memory) d.fn<PFN_vkFreeMemory>("vkFreeMemory")(d.device, b.memory, nullptr);
    b = {};
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_sparse = true;
    d.request_buffer_device_address = true;
    if (!d.create()) {
        std::printf("SKIP: no usable Vulkan device\n");
        return 77;
    }
    if (!d.sparse || !(d.families[d.family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) {
        std::printf("SKIP: no sparse residency for buffers on the graphics queue\n");
        return 77;
    }
    VkPhysicalDeviceProperties dev_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &dev_props);
    VkPhysicalDeviceMemoryProperties mem_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                  &mem_props);
    std::printf("INFO device %s, driver 0x%x\n", dev_props.deviceName, dev_props.driverVersion);
    constexpr VkDeviceSize kArena = 64ull << 30;  // the console VAs below sit at 37 and 40 GiB
    if (dev_props.limits.sparseAddressSpaceSize < kArena) {
        std::printf("SKIP: %llu bytes of sparse address space, the test needs 64 GiB\n",
                    static_cast<unsigned long long>(dev_props.limits.sparseAddressSpaceSize));
        return 77;
    }
    const bool strict = dev_props.sparseProperties.residencyNonResidentStrict == VK_TRUE;

    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(queue.ok());

    // The arena. The pool is made after it (it needs the arena's memory types) but must outlive it.
    std::optional<PagePool> pool_storage;
    PagedRegion region;
    if (!region.create(d.device, d.device_proc, kArena)) {
        std::printf("FAIL the 64 GiB sparse arena could not be created\n");
        return 1;
    }
    const std::uint32_t page_type = pick_page_memory_type(mem_props, region.memory_type_bits());
    std::printf("INFO arena %llu GiB at device address 0x%llx; memoryTypeBits 0x%x; page memory type %u\n",
                (unsigned long long)(region.size() >> 30), (unsigned long long)region.address(),
                region.memory_type_bits(), page_type);
    EXPECT(page_type != UINT32_MAX);
    PagePool& pool = pool_storage.emplace(d.device, d.device_proc, page_type);
    EXPECT(pool.ok());
    PageMapper mapper(region, pool);
    const BindQueue submit{queue.timeline(), [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds,
                                                 BindQueue::Submitted submitted) {
        return queue.bind_sparse_ordered(buffer, binds, std::move(submitted));
    }};
    const auto address_now = [&] {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = region.buffer();
        return d.fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress")(d.device, &info);
    };
    const VkDeviceAddress arena_address = region.address();

    Gpu gpu(d, context, queue);
    void* readback_map = nullptr;
    VkBuffer readback = d.buffer(kPageSize * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_map);
    auto* rb = static_cast<std::uint32_t*>(readback_map);
    const auto read_page = [&](std::uint64_t page, std::uint32_t slot = 0) {
        const std::uint64_t v = gpu.submit([&](VkCommandBuffer cb) {
            VkBufferCopy copy{page * kPageSize, slot * kPageSize, kPageSize};
            d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, region.buffer(), readback, 1, &copy);
        });
        gpu.wait(v);
        const std::uint32_t* words = rb + slot * (kPageSize / 4);
        return std::make_pair(words[0], words[kPageSize / 4 - 1]);
    };

    // 16 physical pages, one pool, filled with distinct patterns through a scratch arena window (one batch call).
    std::vector<PageId> pages;
    EXPECT(pool.allocate(16, pages));
    const std::uint32_t handle = mapper.register_page_pool(pages);
    constexpr std::uint64_t kScratch = 1000;  // arena page index of the fill window
    {
        const PageMappingRange all{PageRangeType::incrementing, 16, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &all, 1};
        const auto before = mapper.stats();
        EXPECT(mapper.copy_page_mappings_batch(handle, {&batch, 1}, submit));
        EXPECT(mapper.stats().binds - before.binds == 1);  // 16 contiguous pages of one chunk: one bind
        const std::uint64_t v = gpu.submit([&](VkCommandBuffer cb) {
            for (std::uint32_t k = 0; k < 16; ++k)
                d.fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, region.buffer(), (kScratch + k) * kPageSize,
                                                              kPageSize, pattern(k));
        });
        gpu.wait(v);
    }

    // Remaps between reads take effect in order; the device address stays put.
    constexpr std::uint64_t kConsolePage = (37ull << 30) / kPageSize;  // a console VA at 37 GiB
    for (std::uint32_t k : {5u, 9u, 2u, 9u, 15u}) {
        const PageMappingRange one{PageRangeType::incrementing, 1, k};
        const PageMappingBatch batch{kConsolePage * kPageSize, &one, 1};
        EXPECT(mapper.copy_page_mappings_batch(handle, {&batch, 1}, submit));
        const auto [first, last] = read_page(kConsolePage);
        if (first != pattern(k) || last != pattern(k))
            std::printf("FAIL remap to pool page %u read 0x%08x/0x%08x, want 0x%08x\n", k, first, last, pattern(k));
        EXPECT(first == pattern(k) && last == pattern(k));
        EXPECT(address_now() == arena_address);
    }
    std::printf("INFO 5 remaps of VA 0x%llx (pool pages 5,9,2,9,15) read back in order; device address fixed\n",
                (unsigned long long)(kConsolePage * kPageSize));

    // Aliasing: a second arena page on pool page 3; a write through one is read through the other.
    {
        const PageMappingRange r{PageRangeType::incrementing, 1, 3};
        const PageMappingBatch a{(kConsolePage + 8) * kPageSize, &r, 1}, b{(kConsolePage + 9) * kPageSize, &r, 1};
        const PageMappingBatch both[2]{a, b};
        EXPECT(mapper.copy_page_mappings_batch(handle, both, submit));
        const std::uint64_t v = gpu.submit([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, region.buffer(), (kConsolePage + 8) * kPageSize,
                                                          kPageSize, 0xC0FFEE00u);
        });
        gpu.wait(v);
        const auto [first, last] = read_page(kConsolePage + 9);
        EXPECT(first == 0xC0FFEE00u && last == 0xC0FFEE00u);
        std::printf("INFO aliasing: write via VA+8 pages, read via VA+9 pages -> 0x%08x\n", first);
    }

    // Clear: unbound pages read 0 where non-resident reads are strict; otherwise their value is undefined.
    EXPECT(mapper.clear_page_mappings(kConsolePage * kPageSize, 1, submit));
    {
        const auto [first, last] = read_page(kConsolePage);
        if (strict) EXPECT(first == 0 && last == 0);
        std::printf("INFO cleared page reads 0x%08x/0x%08x\n", first, last);
    }
    // Refused calls bind nothing.
    {
        const auto submits = mapper.stats().submits;
        EXPECT(!mapper.copy_page_mappings_batch(handle + 99, {}, submit));  // unknown pool
        const PageMappingRange bad{PageRangeType::incrementing, 2, 15};     // runs past the pool's 16 pages
        const PageMappingBatch batch{kConsolePage * kPageSize, &bad, 1};
        EXPECT(!mapper.copy_page_mappings_batch(handle, {&batch, 1}, submit));
        const PageMappingRange one{PageRangeType::incrementing, 1, 0};
        const PageMappingRange two{PageRangeType::incrementing, 2, 0};
        const PageMappingBatch unaligned{kConsolePage * kPageSize + 4096, &one, 1};
        EXPECT(!mapper.copy_page_mappings_batch(handle, {&unaligned, 1}, submit));
        const PageMappingBatch past_end{region.size(), &one, 1};
        EXPECT(!mapper.copy_page_mappings_batch(handle, {&past_end, 1}, submit));
        const PageMappingBatch across_end{region.size() - kPageSize, &two, 1};
        EXPECT(!mapper.copy_page_mappings_batch(handle, {&across_end, 1}, submit));
        EXPECT(!mapper.clear_page_mappings(kPageSize, UINT64_MAX, submit));  // the end wraps around
        EXPECT(!mapper.clear_page_mappings(region.size() + kPageSize, 1, submit));
        // A pool that lists a page the pool has taken back, no page, or a page it never had.
        std::vector<PageId> stale;
        EXPECT(pool.allocate(1, stale));
        pool.release(stale);
        for (PageId id : {stale[0], kNoPage, PageId(0x00ffffff)}) {
            const auto listed = mapper.register_page_pool({id});
            const PageMappingBatch at{kConsolePage * kPageSize, &one, 1};
            EXPECT(!mapper.copy_page_mappings_batch(listed, {&at, 1}, submit));
            EXPECT(mapper.unregister_page_pool(listed));
        }
        EXPECT(mapper.stats().submits == submits);
        EXPECT(!mapper.clear_page_mappings(kScratch * kPageSize, 1, BindQueue{}));  // no way to submit
    }

    // Calls a pool or arena refuses instead of crashing on.
    {
        const auto in_use = pool.pages_in_use();
        const PageId garbage[]{PageId(0x00ffffff), kNoPage, PageId(UINT32_MAX)};
        pool.release(garbage);
        EXPECT(pool.pages_in_use() == in_use);
        EXPECT(pool.memory(garbage[0]) == VK_NULL_HANDLE && !pool.allocated(garbage[0]));
        pool.set_priority_hint(std::nanf(""));
        PagePool unusable(d.device, d.device_proc, UINT32_MAX);
        std::vector<PageId> none;
        EXPECT(!unusable.ok() && !unusable.allocate(1, none) && none.empty());
        EXPECT(!region.create(d.device, d.device_proc, kPageSize));  // already created
        PagedRegion empty, huge;
        EXPECT(!empty.create(d.device, d.device_proc, 0));
        EXPECT(!huge.create(d.device, d.device_proc, ~VkDeviceSize(0)));
        EXPECT(address_now() == arena_address);
    }

    // Coalescing and bind cost.
    std::vector<PageId> big;
    EXPECT(pool.allocate(1024, big));
    const std::uint32_t big_pool = mapper.register_page_pool(big);
    std::vector<PageId> reversed(big.rbegin(), big.rend());
    const std::uint32_t scattered_pool = mapper.register_page_pool(reversed);
    std::printf("INFO pool: %u chunk(s), %.0f MiB reserved, %llu pages in use\n", pool.chunk_count(),
                double(pool.bytes_reserved()) / (1 << 20), (unsigned long long)pool.pages_in_use());

    constexpr std::uint64_t kBench = (40ull << 30) / kPageSize;
    std::printf("INFO bind round trip (median of 15; CPU time from the call to the bind's timeline signal):\n");
    for (std::uint32_t n : {1u, 16u, 256u, 1024u}) {
        for (bool contiguous : {true, false}) {
            const std::uint32_t pool_handle = contiguous ? big_pool : scattered_pool;
            std::vector<double> ms;
            std::uint64_t binds = 0;
            for (int rep = 0; rep < 15; ++rep) {
                // Every call must really change the mapping (an identical remap is a no-op with no bind): alternate
                // between two pool offsets, or, when the run covers the whole pool, clear it first.
                const PageMappingRange r{PageRangeType::incrementing, n, rep % 2 ? 0u : (n < 1024 ? n : 0u)};
                if (n == 1024) {
                    EXPECT(mapper.clear_page_mappings(kBench * kPageSize, n, submit));
                    gpu.wait_queue(queue.submitted_ticket());
                }
                const PageMappingBatch batch{kBench * kPageSize, &r, 1};
                const auto before = mapper.stats();
                const auto t0 = std::chrono::steady_clock::now();
                EXPECT(mapper.copy_page_mappings_batch(pool_handle, {&batch, 1}, submit));
                gpu.wait_queue(queue.submitted_ticket());
                const auto t1 = std::chrono::steady_clock::now();
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                binds = mapper.stats().binds - before.binds;
            }
            if (contiguous) EXPECT(binds == 1);
            else if (n > 1) EXPECT(binds == n);
            const double m = median(ms);
            std::printf("INFO   %4u pages %-10s %4llu bind(s): %8.3f ms  (%.2f us/page)\n", n,
                        contiguous ? "contiguous" : "scattered", (unsigned long long)binds, m, m * 1000.0 / n);
        }
    }
    EXPECT(mapper.clear_page_mappings(kBench * kPageSize, 1024, submit));

    // Does a remap delay work queued before it?
    // W1 and W2 each copy 128 MiB eight times between device-local buffers. Timestamps give the gap between W1's end
    // and W2's start, with nothing between them and with a 64-page scattered remap between them. The bind is ordered
    // after W1 by design (it waits on the queue timeline), so it cannot overlap W1; the question is its GPU-side cost.
    {
        constexpr VkDeviceSize kWork = 128ull << 20;
        DeviceBuffer a = device_local(d, mem_props, kWork), b = device_local(d, mem_props, kWork);
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 4;
        VkQueryPool queries = VK_NULL_HANDLE;
        d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool")(d.device, &qp, nullptr, &queries);
        const auto work = [&](std::uint32_t q0) {
            return gpu.submit([&](VkCommandBuffer cb) {
                if (q0 == 0) d.fn<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(cb, queries, 0, 4);
                d.fn<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries,
                                                                      q0);
                for (int i = 0; i < 8; ++i) {
                    VkBufferCopy c{0, 0, kWork};
                    d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, i % 2 ? b.buffer : a.buffer,
                                                                  i % 2 ? a.buffer : b.buffer, 1, &c);
                    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                                                            &barrier, 0, nullptr, 0, nullptr);
                }
                d.fn<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                                                      queries, q0 + 1);
            });
        };
        const double period_ns = dev_props.limits.timestampPeriod;
        const auto measure = [&](bool remap, double& w1_ms, double& gap_ms, double& call_ms) {
            work(0);
            call_ms = 0;
            if (remap) {
                const PageMappingRange r{PageRangeType::incrementing, 64, 0};
                const PageMappingBatch batch{kBench * kPageSize, &r, 1};
                const auto t0 = std::chrono::steady_clock::now();
                EXPECT(mapper.copy_page_mappings_batch(scattered_pool, {&batch, 1}, submit));
                call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            gpu.wait(work(2));
            std::uint64_t ts[4]{};
            d.fn<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(d.device, queries, 0, 4, sizeof(ts), ts,
                                                                     sizeof(std::uint64_t),
                                                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            w1_ms = double(ts[1] - ts[0]) * period_ns / 1e6;
            gap_ms = double(ts[2] - ts[1]) * period_ns / 1e6;
            if (remap) EXPECT(mapper.clear_page_mappings(kBench * kPageSize, 64, submit));
        };
        std::vector<double> base_gap, remap_gap, calls, w1s;
        for (int rep = 0; rep < 9; ++rep) {
            double w1 = 0, gap = 0, call = 0;
            measure(false, w1, gap, call);
            base_gap.push_back(gap);
            w1s.push_back(w1);
            measure(true, w1, gap, call);
            remap_gap.push_back(gap);
            calls.push_back(call);
        }
        std::printf("INFO queued-work test: W1 runs %.2f ms on the GPU\n", median(w1s));
        std::printf("INFO   gap W1 end -> W2 start: %.3f ms without a remap, %.3f ms with a 64-page scattered remap "
                    "between them (median of 9)\n",
                    median(base_gap), median(remap_gap));
        std::printf("INFO   the remap call returned on the CPU in %.3f ms while W1 was still running (no CPU stall)\n",
                    median(calls));
        EXPECT(median(calls) < median(w1s));  // the CPU call does not wait for queued GPU work
        d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(d.device, queries, nullptr);
        d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
        destroy(d, a);
        destroy(d, b);
    }

    // Free everything: chunks go back once nothing is mapped and the last bind is complete.
    EXPECT(mapper.clear_page_mappings(kScratch * kPageSize, 16, submit));
    EXPECT(mapper.clear_page_mappings((kConsolePage + 8) * kPageSize, 2, submit));
    gpu.wait_queue(queue.submitted_ticket());
    pool.release(pages);
    pool.release(big);
    const VkDeviceSize freed = pool.trim(queue.completed_ticket());
    std::printf("INFO trim freed %.0f MiB; %u chunk(s) left\n", double(freed) / (1 << 20), pool.chunk_count());
    EXPECT(pool.chunk_count() == 0);

    // A released page that is still mapped is not handed out again until it is unmapped.
    {
        std::vector<PageId> held;
        EXPECT(pool.allocate(1, held));
        const auto held_pool = mapper.register_page_pool(held);
        const PageMappingRange range{PageRangeType::incrementing, 1, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &range, 1};
        EXPECT(mapper.copy_page_mappings_batch(held_pool, {&batch, 1}, submit));
        pool.release(held);
        std::vector<PageId> next;
        EXPECT(pool.allocate(1, next));
        EXPECT(next[0] != held[0]);
        EXPECT(mapper.clear_page_mappings(kScratch * kPageSize, 1, submit));
        EXPECT(mapper.unregister_page_pool(held_pool));
        gpu.wait_queue(queue.submitted_ticket());
        pool.release(next);
        EXPECT(pool.trim(queue.completed_ticket()) == kChunkSize);
    }

    // trim() keeps a chunk until the last bind that referenced it has completed.
    {
        std::vector<PageId> one;
        EXPECT(pool.allocate(1, one));
        const auto one_pool = mapper.register_page_pool(one);
        const PageMappingRange range{PageRangeType::incrementing, 1, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &range, 1};
        EXPECT(mapper.copy_page_mappings_batch(one_pool, {&batch, 1}, submit));
        EXPECT(mapper.clear_page_mappings(kScratch * kPageSize, 1, submit));
        const std::uint64_t last = queue.submitted_ticket();
        pool.release(one);
        EXPECT(pool.trim(last - 1) == 0);
        gpu.wait_queue(last);
        EXPECT(pool.trim(queue.completed_ticket()) == kChunkSize);
        EXPECT(mapper.unregister_page_pool(one_pool));
    }

    // A bind behind a Wait whose Signal is not submitted yet is held with the queue's work, in queue order, and
    // trim() keeps its chunk until it has a ticket.
    {
        EXPECT(pool.chunk_count() == 0);
        VulkanFence gate(context, 0), behind(context, 0);
        std::uint64_t last_ticket = 0;
        const BindQueue watched{queue.timeline(), [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds,
                                                      BindQueue::Submitted submitted) {
            return queue.bind_sparse_ordered(buffer, binds, [&, submitted](std::uint64_t ticket) {
                last_ticket = ticket;
                submitted(ticket);
            });
        }};
        std::vector<PageId> one;
        EXPECT(pool.allocate(1, one));
        const auto one_pool = mapper.register_page_pool(one);
        const PageMappingRange range{PageRangeType::incrementing, 1, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &range, 1};
        queue.wait(gate.semaphore(), 1);
        EXPECT(queue.signal(behind.semaphore(), 7));  // held: the Wait ahead of it has no Signal yet
        const std::uint64_t before = queue.submitted_ticket();
        EXPECT(mapper.copy_page_mappings_batch(one_pool, {&batch, 1}, watched));
        EXPECT(mapper.clear_page_mappings(kScratch * kPageSize, 1, watched));
        pool.release(one);
        EXPECT(queue.submitted_ticket() == before && last_ticket == 0);  // both binds held behind the Signal
        EXPECT(pool.trim(UINT64_MAX) == 0);
        EXPECT(gate.signal(1));
        EXPECT(last_ticket == queue.submitted_ticket() && last_ticket > before + 1);  // the Signal first, then both
        gpu.wait_queue(queue.submitted_ticket());
        EXPECT(behind.completed() == 7);
        EXPECT(pool.trim(queue.completed_ticket()) == kChunkSize);
        EXPECT(mapper.unregister_page_pool(one_pool));
    }

    // A held bind dropped because the fence it waits on is released lets its chunk go. The clear behind it does not
    // wait on that fence, so it is released and submitted.
    {
        std::vector<PageId> one;
        EXPECT(pool.allocate(1, one));
        const auto one_pool = mapper.register_page_pool(one);
        const PageMappingRange range{PageRangeType::incrementing, 1, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &range, 1};
        const std::uint64_t before = queue.submitted_ticket();
        {
            VulkanFence gate(context, 0);
            queue.wait(gate.semaphore(), 1);
            EXPECT(mapper.copy_page_mappings_batch(one_pool, {&batch, 1}, submit));
            EXPECT(mapper.clear_page_mappings(kScratch * kPageSize, 1, submit));
        }
        EXPECT(queue.submitted_ticket() == before + 1);  // the clear only
        gpu.wait_queue(queue.submitted_ticket());
        pool.release(one);
        EXPECT(pool.trim(queue.completed_ticket()) == kChunkSize);
        EXPECT(mapper.unregister_page_pool(one_pool));
    }

    // A pool's binds go through one queue: a bind on another queue's timeline is refused before anything is bound.
    {
        VulkanQueue other(context, d.queue, d.family);
        EXPECT(other.ok() && other.timeline() != queue.timeline());
        const BindQueue via_other{other.timeline(), [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds,
                                                        BindQueue::Submitted submitted) {
            return other.bind_sparse_ordered(buffer, binds, std::move(submitted));
        }};
        std::vector<PageId> one;
        EXPECT(pool.allocate(1, one));
        const auto one_pool = mapper.register_page_pool(one);
        const PageMappingRange range{PageRangeType::incrementing, 1, 0};
        const PageMappingBatch batch{kScratch * kPageSize, &range, 1};
        const auto submits = mapper.stats().submits;
        EXPECT(!mapper.copy_page_mappings_batch(one_pool, {&batch, 1}, via_other));
        EXPECT(mapper.stats().submits == submits && other.submitted_ticket() == 0);
        EXPECT(region.mapped(kScratch) == kNoPage);
        EXPECT(mapper.unregister_page_pool(one_pool));
        pool.release(one);
        EXPECT(pool.trim(queue.completed_ticket()) == kChunkSize);
    }

    // A region destroyed with a page still mapped gives its chunk back to trim(), as one cleared first does.
    for (bool clear : {true, false}) {
        std::vector<PageId> one;
        EXPECT(pool.allocate(1, one));
        {
            PagedRegion temporary;
            EXPECT(temporary.create(d.device, d.device_proc, kPageSize));
            PageMapper m(temporary, pool);
            const auto one_pool = m.register_page_pool(one);
            const PageMappingRange range{PageRangeType::incrementing, 1, 0};
            const PageMappingBatch batch{0, &range, 1};
            EXPECT(m.copy_page_mappings_batch(one_pool, {&batch, 1}, submit));
            if (clear) EXPECT(m.clear_page_mappings(0, 1, submit));
            gpu.wait_queue(queue.submitted_ticket());
            EXPECT(m.unregister_page_pool(one_pool));
        }
        pool.release(one);
        const VkDeviceSize freed_after = pool.trim(queue.completed_ticket());
        std::printf("INFO region destroyed (%s): trim freed %.0f MiB, %llu bytes still reserved\n",
                    clear ? "cleared first" : "page still mapped", double(freed_after) / (1 << 20),
                    (unsigned long long)pool.bytes_reserved());
        EXPECT(freed_after == kChunkSize);
    }

    const auto stats = mapper.stats();
    std::printf("INFO mapper: %llu calls, %llu submits, %llu binds, %llu pages\n", (unsigned long long)stats.calls,
                (unsigned long long)stats.submits, (unsigned long long)stats.binds, (unsigned long long)stats.pages);
    if (d.validation) {
        std::printf("INFO validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    queue.idle();
    std::printf(g_failures ? "FAILED (%d)\n" : "PASS\n", g_failures);
    return g_failures ? 1 : 0;
}
