// vk_pipeline_check's memory budget for large modules (large_module_gate.h), without a Vulkan device.
#include "large_module_gate.h"

#include <atomic>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

using leonida::LargeModuleGate;

static int budget_and_count() {
    LargeModuleGate gate(10, 3);
    CHECK(gate.admits(4));
    gate.enter(4);
    CHECK(gate.admits(4));
    gate.enter(4);
    CHECK(!gate.admits(4));   // 12 > 10
    CHECK(gate.admits(2));    // 10 fits exactly
    gate.enter(2);
    CHECK(!gate.admits(0));   // three in: max_inflight
    gate.leave(4);
    CHECK(gate.admits(4));
    gate.leave(4);
    gate.leave(2);
    CHECK(gate.inflight() == 0);
    return 0;
}

static int oversized_module_still_gets_in_alone() {
    // Before: a module whose estimate exceeds the whole budget would wait forever.
    LargeModuleGate gate(10, 4);
    CHECK(gate.admits(100));
    gate.enter(100);
    CHECK(!gate.admits(1));   // nothing joins it
    gate.leave(100);
    CHECK(gate.admits(1));
    return 0;
}

static int max_inflight_zero_means_one() {
    LargeModuleGate gate(100, 0);
    CHECK(gate.max_inflight() == 1);
    gate.enter(1);
    CHECK(!gate.admits(1));
    gate.leave(1);
    return 0;
}

static int ticket_releases() {
    LargeModuleGate gate(10, 2);
    {
        LargeModuleGate::Ticket a(gate, 6);
        CHECK(gate.inflight() == 1);
        LargeModuleGate::Ticket moved = std::move(a);   // the moved-from ticket must not leave twice
        CHECK(gate.inflight() == 1);
        LargeModuleGate::Ticket b;
        b = LargeModuleGate::Ticket(gate, 4);
        CHECK(gate.inflight() == 2);
        b.release();
        CHECK(gate.inflight() == 1);
    }
    CHECK(gate.inflight() == 0);
    return 0;
}

static int threads_never_exceed_the_budget() {
    constexpr uint64_t kBudget = 10;
    LargeModuleGate gate(kBudget, 3);
    std::atomic<uint64_t> used{0};
    std::atomic<unsigned> in{0};
    std::atomic<bool> over{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 200; ++i) {
                const uint64_t est = 1 + uint64_t((t + i) % 5);   // 1..5
                LargeModuleGate::Ticket ticket(gate, est);
                const uint64_t u = used += est;
                const unsigned n = ++in;
                if (n > 3 || (n > 1 && u > kBudget)) over = true;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                --in;
                used -= est;
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(!over);
    CHECK(gate.inflight() == 0);
    return 0;
}

static int estimate_and_memory_query() {
    CHECK(leonida::large_module_estimate(4u << 20) == (uint64_t(4) << 20) * leonida::kBytesPerSpirvByte);
#if defined(__linux__) || defined(_WIN32)
    CHECK(leonida::available_memory_bytes() > 0);   // Linux: MemAvailable; Windows: ullAvailPhys
#endif
    leonida::release_free_memory();   // must be callable anywhere
    return 0;
}

#if defined(__linux__)
#include <filesystem>
#include <fstream>

static void write_file(const std::filesystem::path& p, const char* text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

static int cgroup_limits_are_read_up_the_tree() {
    // Before: only MemAvailable was read, so under systemd-run -p MemoryMax=... (or in a container) the budget was
    // the whole machine's and the checker could be killed by its own cgroup.
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("lmg_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    write_file(root / "self", "0::/user.slice/app.scope\n");
    write_file(root / "cg/user.slice/memory.max", "10000\n");
    write_file(root / "cg/user.slice/memory.current", "4000\n");          // room 6000
    write_file(root / "cg/user.slice/app.scope/memory.max", "max\n");     // no limit of its own
    write_file(root / "cg/user.slice/app.scope/memory.current", "3000\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 6000);
    write_file(root / "cg/user.slice/app.scope/memory.max", "5000\n");    // room 2000, the tighter one
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "cg/user.slice/app.scope/memory.current", "7000\n"); // over its limit: no room
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 0);
    // A mount rooted at a constrained cgroup still exposes its budget at the mount root.
    write_file(root / "cg/user.slice/app.scope/memory.max", "max\n");
    write_file(root / "cg/memory.max", "5000\n");
    write_file(root / "cg/memory.current", "3000\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "self", "0::/\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "cg/memory.max", "max\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == UINT64_MAX);
    write_file(root / "self", "12:memory:/legacy\n");                      // cgroup v1 only: no limit read
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == UINT64_MAX);
    CHECK(leonida::cgroup_memory_room((root / "missing").string(), (root / "cg").string()) == UINT64_MAX);
    fs::remove_all(root);
    return 0;
}
#endif

static std::vector<size_t> drain(leonida::StartOrder& order, bool admit) {
    std::vector<size_t> started;
    size_t k;
    bool admitted;
    while (order.next(k, admitted, [&](uint64_t) { return admit; })) started.push_back(k);
    return started;
}

static int start_order_largest_first() {
    // Index:                     0   1    2  3    4   5
    const std::vector<uint64_t> sizes{10, 500, 30, 700, 30, 5};
    leonida::StartOrder all_small(sizes, 0);   // no large threshold: one list, by size
    CHECK(all_small.large_count() == 0);
    CHECK((drain(all_small, true) == std::vector<size_t>{3, 1, 2, 4, 0, 5}));   // equal sizes keep index order

    leonida::StartOrder admitted(sizes, 100);   // 1 and 3 are large
    CHECK(admitted.large_count() == 2);
    CHECK((drain(admitted, true) == std::vector<size_t>{3, 1, 2, 4, 0, 5}));

    // While the gate is full, the small modules go first; the large ones start once nothing else is left.
    leonida::StartOrder held(sizes, 100);
    CHECK((drain(held, false) == std::vector<size_t>{2, 4, 0, 5, 3, 1}));

    leonida::StartOrder empty(std::vector<uint64_t>{}, 100);
    size_t k = 99;
    bool was_admitted = true;
    CHECK(!empty.next(k, was_admitted, [](uint64_t) { return true; }));
    return 0;
}

// Sizes 1000, 900, 800, 700 are large (threshold 100), 40..10 small; the budget holds the 1000-byte module alone.
static const std::vector<uint64_t> kMixed{1000, 900, 800, 700, 40, 30, 20, 10};
static constexpr uint64_t kMixedBudget = 1000 * leonida::kBytesPerSpirvByte;

static int start_order_takes_the_gate_place_as_it_hands_out() {
    // Four workers each take a module before any of them reaches its own work: one large module gets the gate's
    // room, the other three take small work instead of queueing behind it.
    LargeModuleGate gate(kMixedBudget, 4);
    leonida::StartOrder order(kMixed, 100);
    auto admit = [&](uint64_t bytes) { return gate.try_enter(leonida::large_module_estimate(bytes)); };
    std::vector<size_t> got;
    std::vector<bool> admissions;
    for (int worker = 0; worker < 4; ++worker) {
        size_t k;
        bool admitted;
        CHECK(order.next(k, admitted, admit));
        got.push_back(k);
        admissions.push_back(admitted);
    }
    CHECK((got == std::vector<size_t>{0, 4, 5, 6}));
    CHECK((admissions == std::vector<bool>{true, false, false, false}));
    CHECK(gate.inflight() == 1);
    {
        auto ticket = LargeModuleGate::Ticket::adopt(gate, leonida::large_module_estimate(kMixed[0]));
        CHECK(ticket.held());
    }   // the 1000-byte module is done
    CHECK(gate.inflight() == 0);
    size_t k;
    bool admitted;
    CHECK(order.next(k, admitted, admit) && k == 1 && admitted);   // the room is free again: the next large one
    CHECK(gate.inflight() == 1);
    CHECK(order.next(k, admitted, admit) && k == 7 && !admitted);   // 800 does not fit beside 900
    CHECK(order.next(k, admitted, admit) && k == 2 && !admitted);   // no small one left: it waits at the gate
    CHECK(gate.inflight() == 1);
    LargeModuleGate::Ticket::adopt(gate, leonida::large_module_estimate(kMixed[1])).release();
    CHECK(gate.inflight() == 0);
    return 0;
}

static int threads_start_small_work_while_the_gate_is_full() {
    // The same four workers as real threads: each takes one module, then all meet before working on it.
    LargeModuleGate gate(kMixedBudget, 4);
    leonida::StartOrder order(kMixed, 100);
    auto admit = [&](uint64_t bytes) { return gate.try_enter(leonida::large_module_estimate(bytes)); };
    std::vector<std::atomic<int>> seen(kMixed.size());
    std::atomic<int> first_large{0}, first_small{0}, arrived{0};
    std::atomic<bool> over{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            bool first = true;
            size_t k;
            bool admitted;
            while (order.next(k, admitted, admit)) {
                seen[k]++;
                const bool large = kMixed[k] > 100;
                LargeModuleGate::Ticket ticket;
                if (admitted) ticket = LargeModuleGate::Ticket::adopt(gate, leonida::large_module_estimate(kMixed[k]));
                if (first) {
                    ++(large ? first_large : first_small);
                    ++arrived;
                    while (arrived.load() < 4) std::this_thread::yield();
                    first = false;
                }
                if (large && !admitted) ticket = LargeModuleGate::Ticket(gate, leonida::large_module_estimate(kMixed[k]));
                if (gate.inflight() > 1) over = true;   // no two of these large modules fit the budget together
                std::this_thread::sleep_for(std::chrono::milliseconds(large ? 5 : 1));
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(first_large.load() == 1);
    CHECK(first_small.load() == 3);
    for (const auto& s : seen) CHECK(s.load() == 1);
    CHECK(!over);
    CHECK(gate.inflight() == 0);
    return 0;
}

static int start_order_hands_out_each_index_once_across_threads() {
    std::vector<uint64_t> sizes(2000);
    for (size_t i = 0; i < sizes.size(); ++i) sizes[i] = (i * 7919) % 1000;
    leonida::StartOrder order(sizes, 900);
    std::vector<std::atomic<int>> seen(sizes.size());
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            size_t k;
            bool admitted;
            while (order.next(k, admitted, [&](uint64_t) { return t % 2 == 0; })) seen[k]++;
        });
    for (auto& th : threads) th.join();
    for (const auto& s : seen) CHECK(s.load() == 1);
    return 0;
}

int main() {
#if defined(__linux__)
    if (int r = cgroup_limits_are_read_up_the_tree()) return r;
#endif
    if (int r = budget_and_count()) return r;
    if (int r = oversized_module_still_gets_in_alone()) return r;
    if (int r = max_inflight_zero_means_one()) return r;
    if (int r = ticket_releases()) return r;
    if (int r = threads_never_exceed_the_budget()) return r;
    if (int r = estimate_and_memory_query()) return r;
    if (int r = start_order_largest_first()) return r;
    if (int r = start_order_hands_out_each_index_once_across_threads()) return r;
    if (int r = start_order_takes_the_gate_place_as_it_hands_out()) return r;
    if (int r = threads_start_small_work_while_the_gate_is_full()) return r;
    std::puts("large_module_gate: ok");
    return 0;
}
