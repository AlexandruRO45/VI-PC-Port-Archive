// compile_gate.h: the byte budget that bounds concurrent pipeline compiles. Synthetic weights only; no GPU.
#include "../src/compile_gate.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

constexpr std::uint64_t MiB = 1ull << 20;
constexpr std::uint64_t GiB = 1ull << 30;

void wait_until(const std::atomic<bool>& flag) {
    for (int i = 0; i < 2000 && !flag.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Holds the gate's weight in flight for `hold`, recording the peak number in flight across all callers.
void run_compiles(leonida::CompileGate& gate, unsigned threads, std::uint64_t weight, std::atomic<unsigned>& peak) {
    std::atomic<unsigned> now{0};
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < threads; ++i)
        pool.emplace_back([&] {
            const auto ticket = gate.enter(weight);
            const unsigned n = now.fetch_add(1) + 1;
            for (unsigned p = peak.load(); n > p && !peak.compare_exchange_weak(p, n);) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            now.fetch_sub(1);
        });
    for (auto& t : pool) t.join();
}

}  // namespace

int main() {
    // The budget: LEONIDA_COMPILE_BUDGET_MIB wins when it is a whole number; otherwise half the free memory, >= 2 GiB.
    check(leonida::compile_budget(nullptr, 20 * GiB) == 10 * GiB, "no override: half of the free physical memory");
    check(leonida::compile_budget(nullptr, 1 * GiB) == 2 * GiB, "no override: never below 2 GiB");
    check(leonida::compile_budget("", 20 * GiB) == 10 * GiB, "an empty override is ignored");
    check(leonida::compile_budget("512", 20 * GiB) == 512 * MiB, "the override is in MiB");
    check(leonida::compile_budget("0", 20 * GiB) == 0, "0 turns the gate off");
    check(leonida::compile_budget("12abc", 20 * GiB) == 10 * GiB, "a non-number override falls back to the default");
    check(leonida::compile_budget("-5", 20 * GiB) == 10 * GiB, "a negative override falls back to the default");

    // The parser keeps missing, invalid and an explicit value apart, and checks the range before scaling to bytes.
    using S = leonida::BudgetSetting;
    const auto kind = [](const char* s) { return leonida::parse_budget_mib(s).kind; };
    const auto mib = [](const char* s) { return leonida::parse_budget_mib(s).mib; };
    check(kind(nullptr) == S::missing && kind("") == S::missing && kind(" \t ") == S::missing,
          "unset, empty and whitespace-only are missing");
    check(kind("0") == S::value && mib("0") == 0, "an explicit 0 is a value, not missing");
    check(kind(" 512\t") == S::value && mib(" 512\t") == 512, "spaces and tabs around the number are allowed");
    check(kind(" -5") == S::invalid, "a sign after leading whitespace is invalid");
    check(kind("+5") == S::invalid, "a plus sign is invalid");
    check(kind("1e3") == S::invalid && kind("512MB") == S::invalid, "exponents and suffixes are invalid");
    check(kind("5 12") == S::invalid, "whitespace inside the number is invalid");
    check(kind("17592186044415") == S::value && mib("17592186044415") == leonida::kMaxBudgetMib,
          "the largest MiB count that fits in bytes is accepted");
    check(kind("17592186044416") == S::invalid, "one past the largest is invalid, not wrapped");
    check(kind("18446744073709551616") == S::invalid, "a number past 64 bits is invalid");
    check(leonida::compile_budget(leonida::parse_budget_mib("17592186044415"), 20 * GiB) == (leonida::kMaxBudgetMib << 20),
          "the largest accepted value scales without overflow");
    check(leonida::compile_budget("17592186044416", 20 * GiB) == 10 * GiB,
          "an out-of-range value falls back to the default instead of wrapping");

    // The weight: 1 KiB per byte of SPIR-V, at least 64 MiB.
    check(leonida::compile_weight(8 * MiB) == 8 * GiB, "an 8 MiB module weighs 8 GiB");
    check(leonida::compile_weight(4096) == 64 * MiB, "a small module weighs the 64 MiB minimum");

    {   // A module heavier than the whole budget still compiles when it is alone.
        leonida::CompileGate gate(1 * GiB);
        const auto ticket = gate.enter(5 * GiB);
        check(!ticket.waited() && gate.in_flight() == 1, "an oversized compile is admitted when nothing is in flight");
    }
    {   // The weights in flight stay inside the budget: 4 units of budget, 8 compiles of 2 units, at most 2 at once.
        leonida::CompileGate gate(4 * GiB);
        std::atomic<unsigned> peak{0};
        run_compiles(gate, 8, 2 * GiB, peak);
        check(peak.load() == 2, "eight 2 GiB compiles under a 4 GiB budget run two at a time");
        check(gate.in_use() == 0 && gate.in_flight() == 0, "every ticket gives its weight back");
    }
    {   // Budget 0: no gate, everything runs at once.
        leonida::CompileGate gate(0);
        std::atomic<unsigned> peak{0};
        run_compiles(gate, 6, 8 * GiB, peak);
        check(peak.load() == 6, "a 0 budget admits every compile at once");
    }
    {   // Arrival order: a heavy compile queued behind a running one is not overtaken by a later light one.
        leonida::CompileGate gate(4 * GiB);
        auto first = std::make_unique<leonida::CompileGate::Ticket>(gate.enter(3 * GiB));
        std::atomic<bool> heavy_queued{false}, heavy_in{false}, light_in{false};
        std::atomic<int> order{0}, heavy_rank{0}, light_rank{0};
        std::thread heavy([&] {
            heavy_queued = true;
            const auto t = gate.enter(3 * GiB);
            heavy_rank = ++order;
            heavy_in = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        });
        wait_until(heavy_queued);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let the heavy one block in enter()
        std::thread light([&] {
            const auto t = gate.enter(1 * GiB);  // would fit next to `first`, but arrived after the heavy one
            light_rank = ++order;
            light_in = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        check(!light_in.load(), "a light compile waits behind an earlier heavy one even when it would fit");
        first.reset();
        heavy.join();
        light.join();
        check(heavy_rank.load() == 1 && light_rank.load() == 2, "waiters are admitted in arrival order");
    }
    {   // A ticket that had to wait says so (the runtime logs the budget the first time).
        leonida::CompileGate gate(1 * GiB);
        auto first = std::make_unique<leonida::CompileGate::Ticket>(gate.enter(1 * GiB));
        std::atomic<bool> waited{false};
        std::thread t([&] { waited = gate.enter(1 * GiB).waited(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        first.reset();
        t.join();
        check(waited.load(), "a compile that had to wait reports it");
    }

    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
