// Xbox physical pages (src/xbox_physical.h): CPU halves on a pagefile section mapped at title VAs over placeholders,
// write-fault dirty tracking, and coherence with the GPU halves (xbox_paging.h) through the sparse arena.
//
// Checks (FAIL = exit 1): one physical page mapped at two VAs shows the same bytes; a write marks exactly that page
// dirty, and taking the dirty list write-protects it again; CPU writes reach the GPU after an upload, read through
// the arena; a GPU write reaches the CPU half after a readback; no validation errors (LEONIDA_VK_VALIDATION=1).
// Measurements (INFO): cost of a first write to a clean page (the fault); upload cost per page.
// No device, no sparse residency for buffers, or less than 64 GiB of sparse address space: SKIP, exit 77. Windows
// only.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/xbox_paging.h"
#include "../src/xbox_physical.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// GPU work outside VulkanQueue::execute(), ordered on the queue's timeline (see vk_xbox_paging_test.cpp).
struct Gpu {
    vktest::Device& d;
    VulkanContext& context;
    VulkanQueue& queue;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::uint64_t value = 0;
    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> retired;

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
        queue.wait(timeline, signal);
        retired.push_back({cb, signal});
        return signal;
    }

    void wait(std::uint64_t v) {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline;
        info.pValues = &v;
        EXPECT(d.fn<PFN_vkWaitSemaphores>("vkWaitSemaphores")(d.device, &info, 10'000'000'000ull) == VK_SUCCESS);
        for (auto it = retired.begin(); it != retired.end();) {
            if (it->second > v) { ++it; continue; }
            d.fn<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(d.device, pool, 1, &it->first);
            it = retired.erase(it);
        }
    }
};

bool all_bytes(const void* p, std::uint8_t value, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i)
        if (b[i] != value) return false;
    return true;
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
    constexpr VkDeviceSize kArena = 64ull << 30;
    if (dev_props.limits.sparseAddressSpaceSize < kArena) {
        std::printf("SKIP: %llu bytes of sparse address space, the test needs 64 GiB\n",
                    static_cast<unsigned long long>(dev_props.limits.sparseAddressSpaceSize));
        return 77;
    }
    VkPhysicalDeviceMemoryProperties mem_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                  &mem_props);
    VulkanContext context(d.device_proc, d.device, true);
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(context.loaded() && queue.ok());

    // The pool is made after the arena (it needs the arena's memory types) but must outlive it.
    std::optional<PagePool> pool_storage;
    PagedRegion region;
    EXPECT(region.create(d.device, d.device_proc, kArena));
    PagePool& pool =
        pool_storage.emplace(d.device, d.device_proc, pick_page_memory_type(mem_props, region.memory_type_bits()));
    PageMapper mapper(region, pool);
    const SubmitBinds submit = [&](VkBuffer buffer, std::span<const VkSparseMemoryBind> binds) {
        return queue.bind_sparse_ordered(buffer, binds);
    };

    PhysicalPageStore store(1024);
    if (!store.ok()) {
        std::printf("FAIL the physical page store (section + placeholder APIs) could not be created\n");
        return 1;
    }
    PhysicalGpu gpu_half(d.device, d.device_proc, pool, store, mem_props);
    EXPECT(gpu_half.ok());
    Gpu gpu(d, context, queue);

    // Pages the store does not have are refused, and the GPU half allocates nothing for them.
    {
        EXPECT(store.data(1024) == nullptr && !store.allocated(1024) && !store.allocated(1000));
        store.mark_dirty(1024);
        const auto in_use = pool.pages_in_use();
        const PhysicalPage absent[]{1000, 5000};
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
            EXPECT(!gpu_half.record_upload(cb, absent));
            EXPECT(!gpu_half.record_readback(cb, absent));
        }));
        EXPECT(gpu_half.gpu_page(1000) == kNoPage);
        EXPECT(pool.pages_in_use() == in_use && gpu_half.pages_uploaded() == 0);
        EXPECT(!gpu_half.finish_readback(absent));
    }

    // --- Double mapping: one physical page at two VAs ----------------------------------------------------------
    std::vector<PhysicalPage> phys;
    EXPECT(store.allocate(4, phys));
    auto* r1 = static_cast<std::uint8_t*>(store.reserve(8));
    auto* r2 = static_cast<std::uint8_t*>(store.reserve(8));
    EXPECT(r1 && r2);
    EXPECT(store.map(r1, phys));                               // r1 pages 0..3 = P0..P3
    const PhysicalPage one[1]{phys[1]};
    EXPECT(store.map(r2 + 2 * kPageSize, one));                // r2 page 2 = P1 as well
    // A VA inside a page is refused, as MapViewOfFile3 refuses it, and r1 page 0 still shows P0.
    const PhysicalPage other[1]{phys[3]};
    store.data(phys[0])[0] = 0xA5;
    for (std::size_t offset : {std::size_t(1), std::size_t(4096), std::size_t(kPageSize - 1)})
        EXPECT(!store.map(r1 + offset, other));
    EXPECT(r1[0] == 0xA5);
    store.data(phys[0])[0] = 0;
    EXPECT(store.take_dirty().empty());                        // fresh pages start clean
    const auto faults0 = store.write_faults();
    std::memset(r1 + 1 * kPageSize, 0x11, kPageSize);          // write P1 through r1
    EXPECT(all_bytes(r2 + 2 * kPageSize, 0x11, kPageSize));    // ... and read it through r2
    EXPECT(store.write_faults() - faults0 == 1);               // one fault for the page, then plain writes
    {
        const auto dirty = store.take_dirty();
        EXPECT(dirty.size() == 1 && dirty[0] == phys[1]);
        std::printf("INFO double mapping: P%u written via VA %p, read via VA %p; dirty list {P%u}\n", phys[1],
                    static_cast<void*>(r1 + kPageSize), static_cast<void*>(r2 + 2 * kPageSize),
                    dirty.empty() ? 0u : dirty[0]);
    }
    r2[2 * kPageSize + 5] = 0x22;                              // a write through the other view is tracked too
    EXPECT(r1[kPageSize + 5] == 0x22);
    {
        const auto dirty = store.take_dirty();
        EXPECT(dirty.size() == 1 && dirty[0] == phys[1]);
    }

    // --- CPU write -> GPU read through the arena -----------------------------------------------------------------
    for (std::uint32_t i = 0; i < 4; ++i) std::memset(r1 + i * kPageSize, int(0x40 + i), kPageSize);
    const auto dirty = store.take_dirty();
    EXPECT(dirty.size() == 4);
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, dirty)); }));
    std::vector<PageId> gpu_pages;
    for (PhysicalPage p : phys) gpu_pages.push_back(gpu_half.gpu_page(p));
    const std::uint32_t pool_handle = mapper.register_page_pool(gpu_pages);
    constexpr std::uint64_t kVa = 12ull << 30;
    const PageMappingRange all{PageRangeType::incrementing, 4, 0};
    const PageMappingBatch batch{kVa, &all, 1};
    EXPECT(mapper.copy_page_mappings_batch(pool_handle, {&batch, 1}, submit));
    void* readback_map = nullptr;
    VkBuffer readback = d.buffer(4 * kPageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback_map);
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
        const VkBufferCopy copy{kVa, 0, 4 * kPageSize};
        d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, region.buffer(), readback, 1, &copy);
    }));
    bool cpu_to_gpu = true;
    for (std::uint32_t i = 0; i < 4; ++i)
        cpu_to_gpu = cpu_to_gpu && all_bytes(static_cast<std::uint8_t*>(readback_map) + i * kPageSize,
                                             std::uint8_t(0x40 + i), kPageSize);
    EXPECT(cpu_to_gpu);
    std::printf("INFO CPU writes to P%u..P%u read back through arena VA 0x%llx after one upload: %s\n", phys[0],
                phys[3], (unsigned long long)kVa, cpu_to_gpu ? "match" : "MISMATCH");

    // --- GPU write -> CPU read -----------------------------------------------------------------------------------
    const PhysicalPage p2[1]{phys[2]};
    gpu.wait(gpu.submit([&](VkCommandBuffer cb) {
        d.fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, region.buffer(), kVa + 2 * kPageSize, kPageSize, 0x5A5A5A5Au);
        EXPECT(gpu_half.record_readback(cb, p2));
    }));
    const PhysicalPage p3[1]{phys[3]};
    EXPECT(!gpu_half.finish_readback(p3));  // not the pages that were read back
    EXPECT(gpu_half.finish_readback(p2));
    EXPECT(!gpu_half.finish_readback(p2));  // finished once
    EXPECT(all_bytes(r1 + 2 * kPageSize, 0x5A, kPageSize));
    EXPECT(store.take_dirty().empty());  // a readback updates the CPU half without making it dirty
    std::printf("INFO GPU fill of arena page -> readback -> CPU view of P%u reads 0x%02x\n", phys[2],
                r1[2 * kPageSize]);

    // --- Pages written on both sides, failed uploads and new GPU halves ------------------------------------------
    {
        // A page the CPU wrote since its last upload is not read back over: its writes would be lost.
        r1[2 * kPageSize] = 0x33;
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(!gpu_half.record_readback(cb, p2)); }));
        EXPECT(r1[2 * kPageSize] == 0x33);
        // A failed upload leaves its pages dirty: P2 with a page the store does not have fails as a whole.
        auto taken = store.take_dirty();
        EXPECT(taken.size() == 1 && taken[0] == phys[2]);
        taken.push_back(5000);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(!gpu_half.record_upload(cb, taken)); }));
        const auto again = store.take_dirty();
        EXPECT(again.size() == 1 && !again.empty() && again[0] == phys[2]);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, again)); }));
        // A CPU write between recording a readback and finishing it drops the readback.
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_readback(cb, p2)); }));
        r1[2 * kPageSize + 1] = 0x44;
        EXPECT(!gpu_half.finish_readback(p2));
        EXPECT(r1[2 * kPageSize] == 0x33 && r1[2 * kPageSize + 1] == 0x44);
        const auto written = store.take_dirty();
        EXPECT(written.size() == 1);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, written)); }));
        // A page with no GPU half is not read back; a new GPU half is marked dirty, so its first upload gives it
        // the CPU half's contents instead of whatever its pool page held.
        std::vector<PhysicalPage> fresh;
        EXPECT(store.allocate(1, fresh));
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(!gpu_half.record_readback(cb, fresh)); }));
        EXPECT(gpu_half.gpu_page(fresh[0]) != kNoPage);
        const auto first = store.take_dirty();
        EXPECT(first.size() == 1 && !first.empty() && first[0] == fresh[0]);
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, first)); }));
    }

    // --- Unmapping a view leaves the other mapping alone ----------------------------------------------------------
    const PhysicalPage none[1]{PhysicalPageStore::kUnmapped};
    EXPECT(store.map(r2 + 2 * kPageSize, none));
    EXPECT(r1[kPageSize] == 0x41);

    // --- Costs -----------------------------------------------------------------------------------------------------
    {
        constexpr std::uint32_t kN = 256;
        std::vector<PhysicalPage> many;
        EXPECT(store.allocate(kN, many));
        auto* r3 = static_cast<std::uint8_t*>(store.reserve(kN));
        if (!r3 || !store.map(r3, many)) {
            std::printf("FAIL mapping %u pages into a fresh reservation\n", kN);
            return 1;  // the writes below would fault on an unmapped range
        }
        (void)store.take_dirty();
        const auto f0 = store.write_faults();
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < kN; ++i) r3[std::size_t(i) * kPageSize] = std::uint8_t(i);
        const auto t1 = std::chrono::steady_clock::now();
        EXPECT(store.write_faults() - f0 == kN);
        const double fault_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kN;
        const auto taken0 = std::chrono::steady_clock::now();
        const auto pages = store.take_dirty();
        const auto taken1 = std::chrono::steady_clock::now();
        EXPECT(pages.size() == kN);
        const auto u0 = std::chrono::steady_clock::now();
        gpu.wait(gpu.submit([&](VkCommandBuffer cb) { EXPECT(gpu_half.record_upload(cb, pages)); }));
        const auto u1 = std::chrono::steady_clock::now();
        std::printf("INFO first write to a clean page (fault + unprotect): %.2f us; take_dirty of %u pages: %.3f ms\n",
                    fault_us, kN, std::chrono::duration<double, std::milli>(taken1 - taken0).count());
        std::printf("INFO upload of %u dirty pages (%.0f MiB: memcpy to staging + GPU copy + wait): %.3f ms (%.2f "
                    "GB/s)\n",
                    kN, double(kN * kPageSize) / (1 << 20),
                    std::chrono::duration<double, std::milli>(u1 - u0).count(),
                    double(kN * kPageSize) / std::chrono::duration<double>(u1 - u0).count() / 1e9);
    }

    EXPECT(mapper.clear_page_mappings(kVa, 4, submit));
    queue.idle();
    if (d.validation) {
        std::printf("INFO validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    std::printf(g_failures ? "FAILED (%d)\n" : "PASS\n", g_failures);
    return g_failures ? 1 : 0;
}
