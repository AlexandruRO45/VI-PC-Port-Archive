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
#include <string>
#include <thread>
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
    std::printf("breadcrumbs: state, operations and report on synthetic data ok\n");
    return 0;
}

}  // namespace

int main() {
    if (synthetic()) return 1;

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

        queue.wait(fence.semaphore(), 1);
        EXPECT(queue.execute({&pb, 1}) == ReplayResult::success);  // vkQueueSubmit succeeded: CPU knowledge only
        entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        const auto* held = find(entries, reinterpret_cast<std::uintptr_t>(pb));
        EXPECT(held && held->state == BreadcrumbState::submitted);  // held before its first command
        EXPECT(find(entries, reinterpret_cast<std::uintptr_t>(pa))->state == BreadcrumbState::completed);
        const std::string text = report_of(entries);
        EXPECT(text.find("1 of the last 2 lists did not complete") != std::string::npos);
        EXPECT(text.find(": submitted (not started), 2 operations: CopyBufferRegion, ResourceBarrier") != std::string::npos);

        EXPECT(fence.signal(1));
        EXPECT(queue.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        EXPECT(find(entries, reinterpret_cast<std::uintptr_t>(pb))->state == BreadcrumbState::completed);
        std::printf("breadcrumbs: completed / submitted behind an unsignalled fence / completed ok\n");

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
