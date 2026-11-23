// Device-lost diagnostics (runtime/src/vulkan_breadcrumbs.h), without a device loss:
//   1. the state of a list from its two marker words (submitted / started / completed) and the D3D12 operations
//      recorded for it, on synthetic data;
//   2. the report format on synthetic entries;
//   3. on a GPU with VK_AMD_buffer_marker: a list that ran reads "completed"; a list held behind a Wait on a fence
//      that is not signalled reads "submitted" (the GPU has not started it, whatever the CPU submitted) until the
//      fence is signalled, then "completed"; a ring that wraps forgets the lists whose slots were reused;
//   4. fallbacks: no marker command, or a context that was never configured, leaves the breadcrumbs off and
//      execute() unchanged.
// The words are read after the queue is idle; on a real loss they are best effort (no memory dependency to the
// host is possible then), which the report says. Skips the GPU part (exit 77 when nothing else ran) without a
// Vulkan 1.3 device or the extension. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

std::string report_of(const std::vector<BreadcrumbRing::Entry>& entries) {
#ifdef _MSC_VER
    std::FILE* f = nullptr;
    if (tmpfile_s(&f) != 0 || !f) return {};
#else
    std::FILE* f = std::tmpfile();
    if (!f) return {};
#endif
    BreadcrumbRing::write_report(f, entries);
    std::rewind(f);
    std::string out;
    char buffer[512];
    std::size_t n = 0;
    while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, n);
    std::fclose(f);
    return out;
}

const BreadcrumbRing::Entry* find(const std::vector<BreadcrumbRing::Entry>& entries, std::uintptr_t list) {
    for (const auto& e : entries)
        if (e.list_id == list) return &e;
    return nullptr;
}

int synthetic() {
    // 1. A word holds the tag only once the GPU wrote it: an older tag (slot reuse) or 0 is "not yet".
    EXPECT(breadcrumb_state(0, 0, 7) == BreadcrumbState::submitted);
    EXPECT(breadcrumb_state(3, 3, 7) == BreadcrumbState::submitted);  // the previous list of the slot
    EXPECT(breadcrumb_state(7, 3, 7) == BreadcrumbState::started);
    EXPECT(breadcrumb_state(7, 7, 7) == BreadcrumbState::completed);
    EXPECT(std::strcmp(breadcrumb_state_name(BreadcrumbState::started), "started (not completed)") == 0);

    CommandList list;
    EXPECT(list.copy_buffer({2, 0, 1, 0, 16}));
    EXPECT(list.uav_barrier({2}));
    EXPECT(list.close());
    std::vector<std::uint32_t> ops;
    std::uint32_t op = 0;
    for (const auto& command : list.commands())
        if (breadcrumb_op_of(command, op)) ops.push_back(op);
    EXPECT(ops.size() == 2);
    EXPECT(ops[0] == breadcrumb_op::copy_buffer_region && ops[1] == breadcrumb_op::resource_barrier);
    EXPECT(breadcrumb_op::copy_buffer_region == 7 && breadcrumb_op::resource_barrier == 15);  // D3D12 values

    // 2. The report starts at the last completed list before the first one that did not complete.
    std::vector<BreadcrumbRing::Entry> entries = {
        {1, 0x10, {breadcrumb_op::copy_buffer_region}, BreadcrumbState::completed},
        {2, 0x20, {breadcrumb_op::dispatch}, BreadcrumbState::completed},
        {3, 0x30, std::vector<std::uint32_t>(10, breadcrumb_op::draw_indexed_instanced), BreadcrumbState::started},
        {4, 0x40, {}, BreadcrumbState::submitted},
    };
    const std::string text = report_of(entries);
    EXPECT(text.find("2 of the last 4 lists did not complete") != std::string::npos);
    EXPECT(text.find("list 0x10") == std::string::npos);
    EXPECT(text.find("list 0x20 on queue 0, tag 2: completed, 1 operations: Dispatch") != std::string::npos);
    EXPECT(text.find("list 0x30 on queue 0, tag 3: started (not completed), 10 operations: DrawIndexedInstanced") !=
           std::string::npos);
    EXPECT(text.find(", ...") != std::string::npos);  // more than 8 operations are cut
    EXPECT(text.find("list 0x40 on queue 0, tag 4: submitted (not started), 0 operations") != std::string::npos);
    EXPECT(text.find("may be incomplete") != std::string::npos);
    entries.resize(2);
    EXPECT(report_of(entries).find("every one of the 2 lists kept completed") != std::string::npos);
    EXPECT(report_of({}).find("no command list was submitted") != std::string::npos);

    // Without GPU breadcrumbs: the last 16 submitted lists, their progress UNKNOWN, with names and queue.
    std::vector<BreadcrumbRing::Entry> cpu(20);
    for (std::uint32_t i = 0; i < cpu.size(); ++i) {
        cpu[i].tag = i + 1;
        cpu[i].list_id = 0x100 + i;
        cpu[i].queue_id = 0x9000;
        cpu[i].state = BreadcrumbState::unknown;
    }
    cpu.back().list_name = L"shadow pass \u00e9";  // UTF-8 in the report
    cpu.back().queue_name = L"direct";
    const std::string unknown = report_of(cpu);
    EXPECT(unknown.find("GPU progress UNKNOWN (no VK_AMD_buffer_marker); the last 16 of the 20 lists submitted") !=
           std::string::npos);
    EXPECT(unknown.find("list 0x103 ") == std::string::npos && unknown.find("list 0x104 ") != std::string::npos);
    EXPECT(unknown.find("list 0x113 \"shadow pass \xc3\xa9\" on queue 0x9000 \"direct\", tag 20: submitted, GPU "
                        "progress UNKNOWN") != std::string::npos);
    EXPECT(unknown.find("completed") == std::string::npos && unknown.find("not started") == std::string::npos);
    // A submission that failed leaves nothing behind; the next one is kept.
    {
        BreadcrumbRing ring;
        ring.initialize_cpu_only();
        std::vector<BreadcrumbRing::Mark> lost{ring.mark(list)};
        ring.submitted(std::move(lost), false);
        EXPECT(ring.snapshot().empty());
        std::vector<BreadcrumbRing::Mark> kept{ring.mark(list)};
        const std::uint32_t tag = kept[0].tag;
        ring.submitted(std::move(kept), true);
        const auto kept_entries = ring.snapshot();
        EXPECT(kept_entries.size() == 1 && kept_entries[0].tag == tag &&
               kept_entries[0].state == BreadcrumbState::unknown);
    }
    std::printf("breadcrumbs: state, operations and report on synthetic data ok\n");
    return 0;
}

// The marker buffer's sharing mode, read from the create call: several queue families write it (direct, compute and
// copy queues), with no ownership transfer, so it is CONCURRENT among them when there are two or more.
namespace sharing {
VkSharingMode mode = VK_SHARING_MODE_MAX_ENUM;
std::vector<std::uint32_t> families;
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer*) {
    mode = info->sharingMode;
    families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;  // the create info is all this test needs
}
VKAPI_ATTR void VKAPI_CALL write_marker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, std::uint32_t) {}
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**) {
    return VK_ERROR_MEMORY_MAP_FAILED;
}
}  // namespace sharing

int marker_buffer_sharing() {
    BreadcrumbFunctions fn;
    fn.create_buffer = sharing::create_buffer;
    fn.write_marker = sharing::write_marker;
    fn.map = sharing::map;
    const VkPhysicalDeviceMemoryProperties memory{};
    const std::uint32_t three[] = {0, 2, 1};
    const std::uint32_t one[] = {0};
    for (const auto& [families, want] : {std::pair{std::span<const std::uint32_t>(three), VK_SHARING_MODE_CONCURRENT},
                                         std::pair{std::span<const std::uint32_t>(one), VK_SHARING_MODE_EXCLUSIVE},
                                         std::pair{std::span<const std::uint32_t>(), VK_SHARING_MODE_EXCLUSIVE}}) {
        BreadcrumbRing ring;
        EXPECT(!ring.initialize(VK_NULL_HANDLE, memory, fn, families));  // the fake create fails, after the call
        EXPECT(sharing::mode == want);
        const std::vector<std::uint32_t> expected(want == VK_SHARING_MODE_EXCLUSIVE ? 0 : 3);
        const bool listed = want == VK_SHARING_MODE_EXCLUSIVE ||
                            (sharing::families.size() == 3 && sharing::families[0] == 0 &&
                             sharing::families[1] == 2 && sharing::families[2] == 1);
        EXPECT(sharing::families.size() == expected.size() && listed);
    }
    std::printf("breadcrumbs: marker buffer CONCURRENT among several queue families, EXCLUSIVE for one ok\n");
    return 0;
}

}  // namespace

int main() {
    if (synthetic() || marker_buffer_sharing()) return 1;

    vktest::Device d;
    d.request_buffer_marker = true;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_breadcrumbs_test: no Vulkan 1.3 device, GPU part skipped\n");
        return 0;
    }

    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer src = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(src && dst);
    ResourceBinding src_rb{};
    src_rb.buffer = src;
    src_rb.size = 64;
    ResourceBinding dst_rb{};
    dst_rb.buffer = dst;
    dst_rb.size = 64;
    dst_rb.host_readable = true;
    CommandList a, b;
    EXPECT(a.copy_buffer({2, 0, 1, 0, 16}));
    EXPECT(a.close());
    EXPECT(b.copy_buffer({2, 16, 1, 16, 16}));
    EXPECT(b.uav_barrier({2}));
    EXPECT(b.close());
    const CommandList* pa = &a;
    const CommandList* pb = &b;

    // 4. Fallbacks: a context that was not configured has no breadcrumbs; one without the marker command refuses;
    //    without the marker extension the ring keeps the CPU's view only.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(context.breadcrumbs() == nullptr);
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        BreadcrumbRing ring;
        BreadcrumbFunctions none;  // no vkCmdWriteBufferMarkerAMD
        EXPECT(!ring.initialize(d.device, d.memory_properties(), none));
        EXPECT(ring.snapshot().empty());
        std::printf("breadcrumbs: off without configuration or marker command ok\n");
    }
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(!context.configure_breadcrumbs(d.memory_properties(), false, false));  // CPU only
        BreadcrumbRing* ring = context.breadcrumbs();
        EXPECT(ring && !ring->gpu());
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        ring->label_queue(&queue, L"copy queue");
        ring->label_list(pa, L"list A");
        ring->label_list(pb, L"");  // unnamed
        const CommandList* both[] = {&a, &b};
        EXPECT(queue.execute(both) == ReplayResult::success);
        EXPECT(queue.idle());
        const auto entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        for (const auto& e : entries) {
            EXPECT(e.state == BreadcrumbState::unknown);  // never "completed" without GPU writes
            EXPECT(e.queue_id == reinterpret_cast<std::uintptr_t>(&queue) && e.queue_name == L"copy queue");
        }
        EXPECT(entries[0].list_name == L"list A" && entries[1].list_name.empty());
        ring->label_list(pa, L"");  // a later submission without a name carries none
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        EXPECT(ring->snapshot().back().list_name.empty());
        // A queue that never names itself (the device's staging queue) and lands at the address of a named queue
        // that was destroyed must not carry that queue's name.
        std::optional<VulkanQueue> reused;
        reused.emplace(context, d.queue, d.family);
        ring->label_queue(&*reused, L"destroyed queue");
        reused.reset();
        reused.emplace(context, d.queue, d.family);  // the same storage, so the same address
        EXPECT(reused->ok());
        EXPECT(reused->execute({&pa, 1}) == ReplayResult::success);
        EXPECT(reused->idle());
        EXPECT(ring->snapshot().back().queue_name.empty());
        reused.reset();
        std::printf("breadcrumbs: CPU only without VK_AMD_buffer_marker: submitted lists, queue and names, progress "
                    "UNKNOWN ok\n");
    }

    if (!d.buffer_marker) {
        std::fprintf(stderr, "vk_breadcrumbs_test: no VK_AMD_buffer_marker, GPU breadcrumbs skipped\n");
        return d.validation_errors() ? 1 : 0;
    }

    // 3. The GPU's view of two lists.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(context.configure_breadcrumbs(d.memory_properties(), true, false));
        EXPECT(context.breadcrumbs()->gpu());
        BreadcrumbRing* ring = context.breadcrumbs();
        EXPECT(ring);
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        VulkanFence fence(context, 0);
        EXPECT(fence.semaphore() != VK_NULL_HANDLE);

        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        auto entries = ring->snapshot();
        EXPECT(entries.size() == 1);
        EXPECT(entries[0].list_id == reinterpret_cast<std::uintptr_t>(pa));
        EXPECT(entries[0].state == BreadcrumbState::completed);
        EXPECT(entries[0].ops.size() == 1 && entries[0].ops[0] == breadcrumb_op::copy_buffer_region);

        // A Wait for a fence value no Signal has reached holds the batch on the CPU (QueueHolds): nothing reached
        // vkQueueSubmit, so the ring has nothing for it. The Signal releases it; the ring then marks the held copy
        // under the list the application passed, with that list's name.
        ring->label_list(pb, L"held list");
        queue.wait(fence.semaphore(), 1);
        EXPECT(queue.execute({&pb, 1}) == ReplayResult::success);
        entries = ring->snapshot();
        EXPECT(entries.size() == 1 && !find(entries, reinterpret_cast<std::uintptr_t>(pb)));
        EXPECT(report_of(entries).find("every one of the 1 lists kept completed") != std::string::npos);

        EXPECT(fence.signal(1));
        EXPECT(queue.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        const auto* released = find(entries, reinterpret_cast<std::uintptr_t>(pb));
        EXPECT(released && released->state == BreadcrumbState::completed && released->list_name == L"held list");
        EXPECT(released->ops.size() == 2 && released->ops[1] == breadcrumb_op::resource_barrier);
        ring->label_list(pb, L"");
        std::printf("breadcrumbs: completed / held behind an unsignalled fence, then released under its own id and "
                    "name / completed ok\n");

        // Held batches keep the names they had at ExecuteCommandLists, while the application's objects were alive:
        //   1. X is held, then X and its queue are renamed: the breadcrumb keeps the old names;
        //   2. the application releases X and a new list Y takes its storage, is named and runs at once on another
        //      queue before X's batch is released: X's breadcrumb must not take Y's name;
        //   3. a second batch, Z, is held on the other queue behind another fence; Z is released first, then X:
        //      each keeps its own name, and the ring lists them in the order they reached vkQueueSubmit.
        {
            VulkanQueue other(context, d.queue, d.family);
            EXPECT(other.ok());
            VulkanFence fence_x(context, 0), fence_z(context, 0);
            EXPECT(fence_x.semaphore() != VK_NULL_HANDLE && fence_z.semaphore() != VK_NULL_HANDLE);
            std::optional<CommandList> storage;
            storage.emplace(b);  // X: b's two operations
            const CommandList* px = &*storage;
            ring->label_queue(&queue, L"queue one");
            ring->label_list(px, L"X");
            queue.wait(fence_x.semaphore(), 1);
            EXPECT(queue.execute({&px, 1}) == ReplayResult::success);  // held
            ring->label_list(px, L"X renamed");                         // 1.
            ring->label_queue(&queue, L"queue one renamed");
            storage.reset();                                            // 2. X released ...
            storage.emplace(a);                                         //    ... Y (a's one operation) in its place
            EXPECT(&*storage == px);
            ring->label_list(px, L"Y");
            EXPECT(other.execute({&px, 1}) == ReplayResult::success);   //    Y is not held: it runs now
            EXPECT(other.idle());
            const CommandList* pz = pa;                                 // 3.
            ring->label_list(pz, L"Z");
            other.wait(fence_z.semaphore(), 1);
            EXPECT(other.execute({&pz, 1}) == ReplayResult::success);  // held
            ring->label_list(pz, L"Z renamed");
            EXPECT(fence_z.signal(1));  // releases Z first
            EXPECT(other.idle());
            EXPECT(fence_x.signal(1));  // then X
            EXPECT(queue.idle());
            entries = ring->snapshot();
            EXPECT(entries.size() >= 3);
            const BreadcrumbRing::Entry& y = entries[entries.size() - 3];
            const BreadcrumbRing::Entry& z = entries[entries.size() - 2];
            const BreadcrumbRing::Entry& x = entries[entries.size() - 1];
            EXPECT(y.list_name == L"Y" && y.ops.size() == 1 && y.queue_id == reinterpret_cast<std::uintptr_t>(&other));
            EXPECT(z.list_name == L"Z" && z.list_id == reinterpret_cast<std::uintptr_t>(pz) && z.ops.size() == 1);
            EXPECT(x.list_name == L"X" && x.list_id == reinterpret_cast<std::uintptr_t>(px) && x.ops.size() == 2);
            EXPECT(x.queue_name == L"queue one" && x.queue_id == reinterpret_cast<std::uintptr_t>(&queue));
            EXPECT(y.tag < z.tag && z.tag < x.tag);  // marked when submitted, not when held
            for (const auto* e : {&y, &z, &x}) EXPECT(e->state == BreadcrumbState::completed);
            ring->label_list(px, L"");
            ring->label_list(pz, L"");
            ring->label_queue(&queue, L"");
            std::printf("breadcrumbs: held batches keep the names they had when held (renamed, address reused, "
                        "released out of order) ok\n");
        }

        // Two lists in one ExecuteCommandLists get a slot each, and the ring keeps the newest kSlots lists.
        const CommandList* both[] = {&a, &b};
        for (std::uint32_t i = 0; i < BreadcrumbRing::kSlots / 2; ++i)
            EXPECT(queue.execute(both) == ReplayResult::success);
        EXPECT(queue.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == BreadcrumbRing::kSlots);
        for (const auto& e : entries) EXPECT(e.state == BreadcrumbState::completed);
        EXPECT(entries.back().list_id == reinterpret_cast<std::uintptr_t>(pb));
        EXPECT(entries[entries.size() - 2].list_id == reinterpret_cast<std::uintptr_t>(pa));
        std::printf("breadcrumbs: %u lists across a ring wrap ok\n", BreadcrumbRing::kSlots + 2);

        // Two queues on two threads through the full ring: every list gets its own slot and tag, the snapshot is
        // in marking order, and the lists whose slots were reused are gone.
        VulkanQueue second(context, d.queue, d.family);
        EXPECT(second.ok());
        constexpr std::uint32_t per_thread = 400;
        bool ok_a = true, ok_b = true;
        std::thread ta([&] {
            for (std::uint32_t i = 0; i < per_thread; ++i)
                ok_a = ok_a && queue.execute({&pa, 1}) == ReplayResult::success;
        });
        std::thread tb([&] {
            for (std::uint32_t i = 0; i < per_thread; ++i)
                ok_b = ok_b && second.execute({&pb, 1}) == ReplayResult::success;
        });
        ta.join();
        tb.join();
        EXPECT(ok_a && ok_b);
        EXPECT(queue.idle() && second.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == BreadcrumbRing::kSlots);
        std::uint32_t from_a = 0, from_b = 0;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            EXPECT(entries[i].state == BreadcrumbState::completed);
            if (i) EXPECT(entries[i].tag > entries[i - 1].tag);  // marking order, no tag twice
            if (i >= entries.size() - 2 * per_thread) {
                from_a += entries[i].list_id == reinterpret_cast<std::uintptr_t>(pa);
                from_b += entries[i].list_id == reinterpret_cast<std::uintptr_t>(pb);
            }
        }
        EXPECT(from_a == per_thread && from_b == per_thread);  // the newest 800 are the two threads' lists
        std::printf("breadcrumbs: two queues on two threads, %u lists, ordered and recycled ok\n", 2 * per_thread);
    }

    const unsigned errors = d.validation_errors();
    std::printf("breadcrumbs: %u validation errors\n", errors);
    return errors ? 1 : 0;
}
