// Regression test for TLS-slot-0 safety and per-instance pump ownership in src/shader_library.h and src/queue_holds.h.
//
// When a mapped Xbox executable owns static TLS slot 0 (TEB->ThreadLocalStoragePointer[0]), host functions reached on
// guest threads must neither execute MSVC's _Init_thread_header dynamic static guards (which read _Init_thread_epoch
// from slot 0) nor read/write host thread_local storage (which aliases slot 0).
//
// On Windows x64, this test swaps TEB->ThreadLocalStoragePointer[0] on worker threads to zeroed, non-zero (0x01) and
// PAGE_READONLY scratch pages before calling leonida::zlib_inflate (fixed Huffman) and leonida::QueueHolds. On every
// platform it also verifies same-instance re-entrancy, cross-instance independence, exception safety and concurrency.
// No game files and no GPU are involved.
#include "../src/queue_holds.h"
#include "../src/shader_library.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) && defined(_M_X64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#endif

namespace {

std::atomic<int> g_checks{0};
std::atomic<int> g_failures{0};

void check(bool condition, const char* what) {
    g_checks.fetch_add(1, std::memory_order_relaxed);
    if (!condition) {
        g_failures.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

// Fixed-Huffman zlib stream of "hello hello hello leonida" (synthetic test vector from shader_library_test.cpp).
const std::vector<std::uint8_t> kFixedZlib = {0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x40, 0x26, 0x73,
                                              0x52, 0xf3, 0xf3, 0x32, 0x53, 0x12, 0x01, 0x7a, 0xaf, 0x09, 0x79};
const std::vector<std::uint8_t> kFixedExpected = [] {
    const std::string s = "hello hello hello leonida";
    return std::vector<std::uint8_t>(s.begin(), s.end());
}();

using Holds = leonida::QueueHolds<int>;

struct ReentrantHolder final : Holds::Holder {
    Holds& holds;
    int stage = 0;
    int releases = 0;
    int discards = 0;
    bool reentered_remove = false;

    explicit ReentrantHolder(Holds& h) : holds(h) {}

    bool release_ready() override {
        if (stage == 0) {
            stage = 1;
            ++releases;
            // Re-entrant pump_ready() on the same instance must set dirty_ and return without deadlocking.
            holds.pump_ready();
            return true;
        }
        if (stage == 1) {
            stage = 2;
            ++releases;
            holds.held_changed(-1);
            return true;
        }
        return false;
    }

    void discard_waiting_on(int sem) override {
        ++discards;
        if (!reentered_remove) {
            reentered_remove = true;
            // Re-entrant remove_fence() during discard must return immediately instead of deadlocking pump_mutex_.
            holds.remove_fence(sem);
        }
    }

    std::uint64_t oldest_held_seq() override { return stage < 2 ? 1 : 0; }
};

struct CrossInstanceHolder final : Holds::Holder {
    Holds* peer = nullptr;
    int releases = 0;

    explicit CrossInstanceHolder(Holds* p) : peer(p) {}

    bool release_ready() override {
        if (releases > 0) return false;
        ++releases;
        if (peer) peer->pump_ready();
        return true;
    }

    void discard_waiting_on(int) override {}
    std::uint64_t oldest_held_seq() override { return 0; }
};

struct CountingHolder final : Holds::Holder {
    std::atomic<int> pending{0};
    std::atomic<int> releases{0};
    std::atomic<int> discards{0};

    bool release_ready() override {
        int cur = pending.load(std::memory_order_acquire);
        while (cur > 0) {
            if (pending.compare_exchange_weak(cur, cur - 1, std::memory_order_acq_rel)) {
                releases.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

    void discard_waiting_on(int) override { discards.fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t oldest_held_seq() override { return pending.load(std::memory_order_acquire) > 0 ? 1 : 0; }
};

void test_portable_behavior() {
    // 1. Fixed-Huffman inflate works and uses compile-time tables.
    {
        std::vector<std::uint8_t> out;
        check(leonida::zlib_inflate(kFixedZlib, out) && out == kFixedExpected,
              "portable fixed-Huffman zlib_inflate matches expected bytes");
    }

    // 2. Same-instance re-entrancy in pump_ready() and remove_fence().
    {
        Holds holds;
        ReentrantHolder holder(holds);
        holds.attach(&holder);
        holds.add_fence(1, 0);
        holds.held_changed(1);
        holds.pump_ready();
        check(holder.releases == 2, "re-entrant pump_ready() runs follow-up pass without deadlocking");

        holds.held_changed(1);
        holds.remove_fence(1);
        check(holder.discards == 1 && holder.reentered_remove,
              "re-entrant remove_fence() during discard_waiting_on does not deadlock");
        holds.held_changed(-1);
        holds.detach(&holder);
    }

    // 3. Cross-instance pumping on the same thread: pumping holds_a must NOT suppress holds_b.
    {
        Holds holds_a, holds_b;
        CrossInstanceHolder holder_b(nullptr);
        CrossInstanceHolder holder_a(&holds_b);
        holds_b.attach(&holder_b);
        holds_a.attach(&holder_a);
        holds_b.held_changed(1);
        holds_a.held_changed(1);
        holds_a.pump_ready();
        check(holder_a.releases == 1 && holder_b.releases == 1,
              "pumping a second QueueHolds instance inside a release on the same thread still runs");
        holds_a.held_changed(-1);
        holds_b.held_changed(-1);
        holds_a.detach(&holder_a);
        holds_b.detach(&holder_b);
    }

    // 4. Exception safety: if an after_held barrier throws, pump_owner_ must be cleared so later pumps work.
    {
        Holds holds;
        CountingHolder holder;
        holds.attach(&holder);
        bool caught = false;
        try {
            holds.after_held(&holds, [] { throw std::runtime_error("boom"); });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught, "throwing after_held callback propagates");

        holder.pending.store(1, std::memory_order_release);
        holds.held_changed(1);
        holds.pump_ready();
        check(holder.releases.load(std::memory_order_acquire) == 1,
              "pump_ready() still executes on the same thread after an earlier barrier threw");
        holds.held_changed(-1);
        holds.detach(&holder);
    }

    // 5. Concurrent multi-threaded pump_ready() + fixed-Huffman inflate.
    {
        Holds holds;
        CountingHolder holder;
        holds.attach(&holder);
        constexpr int kThreads = 4;
        constexpr int kIters = 200;
        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&] {
                for (int i = 0; i < kIters; ++i) {
                    std::vector<std::uint8_t> out;
                    if (!leonida::zlib_inflate(kFixedZlib, out) || out != kFixedExpected) {
                        check(false, "concurrent fixed-Huffman zlib_inflate failed");
                        return;
                    }
                    holder.pending.fetch_add(1, std::memory_order_acq_rel);
                    holds.held_changed(1);
                    holds.pump_ready();
                    holds.held_changed(-1);
                }
            });
        }
        for (auto& w : workers) w.join();
        check(holder.releases.load(std::memory_order_acquire) == kThreads * kIters,
              "concurrent pump_ready() releases every held item");
        holds.detach(&holder);
    }
}

#if defined(_WIN32) && defined(_M_X64)
// Ensure this test binary has a static .tls directory so TEB->ThreadLocalStoragePointer[0] exists.
thread_local std::uint32_t g_tls_slot0_anchor = 0x13572468u;

void** module_tls_slots() noexcept { return reinterpret_cast<void**>(__readgsqword(0x58)); }

struct ScopedTlsSlot0Poison {
    void** slots = nullptr;
    void* saved = nullptr;

    explicit ScopedTlsSlot0Poison(void* scratch_page) noexcept {
        slots = module_tls_slots();
        if (slots) {
            saved = slots[0];
            slots[0] = scratch_page;
        }
    }

    ~ScopedTlsSlot0Poison() noexcept {
        if (slots) slots[0] = saved;
    }

    ScopedTlsSlot0Poison(const ScopedTlsSlot0Poison&) = delete;
    ScopedTlsSlot0Poison& operator=(const ScopedTlsSlot0Poison&) = delete;
};

bool all_bytes_equal(const void* page, std::size_t size, std::uint8_t value) {
    const auto* p = static_cast<const std::uint8_t*>(page);
    for (std::size_t i = 0; i < size; ++i)
        if (p[i] != value) return false;
    return true;
}

void test_windows_tls_slot0_poisoned() {
    check(g_tls_slot0_anchor == 0x13572468u, "main thread TLS slot 0 is active");
    check(module_tls_slots() != nullptr, "TEB->ThreadLocalStoragePointer is non-null");

    constexpr std::size_t kPageBytes = 65536;
    void* zero_page = VirtualAlloc(nullptr, kPageBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    void* nonzero_page = VirtualAlloc(nullptr, kPageBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    void* ro_page = VirtualAlloc(nullptr, kPageBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    check(zero_page && nonzero_page && ro_page, "allocated TLS scratch pages");
    if (!zero_page || !nonzero_page || !ro_page) return;

    std::memset(nonzero_page, 0x01, kPageBytes);
    DWORD old_protect = 0;
    check(VirtualProtect(ro_page, kPageBytes, PAGE_READONLY, &old_protect) != FALSE,
          "marked read-only TLS scratch page PAGE_READONLY");

    // Run on fresh worker threads so fixed-Huffman inflate and QueueHolds execute with slot 0 poisoned.
    std::thread first_use([&] {
        if (g_tls_slot0_anchor != 0x13572468u) {
            check(false, "worker thread TLS slot 0 anchor mismatch");
            return;
        }

        // Case A: zeroed TLS slot 0 (breaks MSVC _Init_thread_header dynamic static guards on main).
        {
            std::vector<std::uint8_t> out;
            bool inflate_ok = false;
            Holds holds;
            CountingHolder holder;
            holds.attach(&holder);
            holds.add_fence(7, 0);
            holder.pending.store(1, std::memory_order_release);
            holds.held_changed(1);
            {
                ScopedTlsSlot0Poison poison(zero_page);
                inflate_ok = leonida::zlib_inflate(kFixedZlib, out) && out == kFixedExpected;
                holds.pump_ready();
                holds.remove_fence(7);
            }
            holds.held_changed(-1);
            holds.detach(&holder);
            check(inflate_ok, "fixed-Huffman zlib_inflate succeeds under zeroed TLS slot 0");
            check(holder.releases.load(std::memory_order_acquire) == 1 &&
                      holder.discards.load(std::memory_order_acquire) == 1,
                  "QueueHolds::pump_ready and remove_fence run under zeroed TLS slot 0");
            check(all_bytes_equal(zero_page, kPageBytes, 0x00),
                  "zeroed guest TLS slot 0 page stays untouched");
        }

        // Case B: non-zero (0x01) TLS slot 0 (made thread_local in_pump() return true and skip pumping on main).
        {
            Holds holds;
            CountingHolder holder;
            holds.attach(&holder);
            holds.add_fence(8, 0);
            holder.pending.store(1, std::memory_order_release);
            holds.held_changed(1);
            {
                ScopedTlsSlot0Poison poison(nonzero_page);
                holds.pump_ready();
                holds.remove_fence(8);
            }
            holds.held_changed(-1);
            holds.detach(&holder);
            check(holder.releases.load(std::memory_order_acquire) == 1 &&
                      holder.discards.load(std::memory_order_acquire) == 1,
                  "QueueHolds::pump_ready and remove_fence do not skip work under 0x01 guest TLS slot 0");
            check(all_bytes_equal(nonzero_page, kPageBytes, 0x01),
                  "0x01 guest TLS slot 0 page stays untouched");
        }

        // Case C: PAGE_READONLY TLS slot 0 (any write to host thread_local in slot 0 would access-violate).
        {
            std::vector<std::uint8_t> out;
            bool inflate_ok = false;
            Holds holds;
            ReentrantHolder holder(holds);
            holds.attach(&holder);
            holds.add_fence(9, 0);
            holds.held_changed(1);
            {
                ScopedTlsSlot0Poison poison(ro_page);
                inflate_ok = leonida::zlib_inflate(kFixedZlib, out) && out == kFixedExpected;
                holds.pump_ready();
                holds.held_changed(1);
                holds.remove_fence(9);
                holds.held_changed(-1);
            }
            holds.detach(&holder);
            check(inflate_ok && holder.releases == 2 && holder.discards == 1,
                  "zlib_inflate and QueueHolds never write to PAGE_READONLY guest TLS slot 0");
        }
    });
    first_use.join();

    // Case D: concurrent workers all running with PAGE_READONLY TLS slot 0.
    {
        Holds holds;
        CountingHolder holder;
        holds.attach(&holder);
        constexpr int kThreads = 4;
        constexpr int kIters = 100;
        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&] {
                if (g_tls_slot0_anchor != 0x13572468u) {
                    check(false, "concurrent worker TLS slot 0 anchor mismatch");
                    return;
                }
                ScopedTlsSlot0Poison poison(ro_page);
                for (int i = 0; i < kIters; ++i) {
                    std::vector<std::uint8_t> out;
                    if (!leonida::zlib_inflate(kFixedZlib, out) || out != kFixedExpected) {
                        check(false, "concurrent poisoned-TLS fixed-Huffman zlib_inflate failed");
                        return;
                    }
                    holder.pending.fetch_add(1, std::memory_order_acq_rel);
                    holds.held_changed(1);
                    holds.pump_ready();
                    holds.held_changed(-1);
                }
            });
        }
        for (auto& w : workers) w.join();
        check(holder.releases.load(std::memory_order_acquire) == kThreads * kIters,
              "concurrent poisoned-TLS workers release every held item");
        holds.detach(&holder);
    }

    VirtualFree(zero_page, 0, MEM_RELEASE);
    VirtualFree(nonzero_page, 0, MEM_RELEASE);
    VirtualFree(ro_page, 0, MEM_RELEASE);
}
#endif

}  // namespace

int main() {
#if defined(_WIN32) && defined(_M_X64)
    // Run the poisoned-TLS checks first so first-use paths have not been primed on the main thread.
    test_windows_tls_slot0_poisoned();
#endif
    test_portable_behavior();
    std::printf("runtime_src_tls_safe_test: %d checks, %d failures\n", g_checks.load(), g_failures.load());
    return g_failures.load() == 0 ? 0 : 1;
}
