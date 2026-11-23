#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace leonida {

// Bounds the host memory that concurrent pipeline compiles take. A driver compile of a large compute module can
// commit about a kilobyte of host memory per byte of SPIR-V (8.3-9.7 MB modules took 8-10 GB each on NVIDIA,
// #8186), and the retail title starts up to 11 such compiles at once on its job threads. Unbounded, a cold boot on a
// 16-32 GB machine runs out of commit and the driver fails the compiles with VK_ERROR_UNKNOWN (#7154).
//
// Each compile enters with a weight (its estimated commit) and waits while the weights in flight would go over the
// budget. Three rules keep it safe:
//   - a compile is always admitted when nothing is in flight, so a module heavier than the whole budget still
//     compiles (alone) instead of waiting forever;
//   - waiters are admitted in arrival order, so a heavy compile is not starved by a stream of light ones;
//   - a budget of 0 admits everything (no gate).
class CompileGate {
public:
    explicit CompileGate(std::uint64_t budget_bytes) : budget_(budget_bytes) {}
    CompileGate(const CompileGate&) = delete;
    CompileGate& operator=(const CompileGate&) = delete;

    // Releases its weight when it goes out of scope.
    class Ticket {
    public:
        Ticket(CompileGate* gate, std::uint64_t weight, bool waited) : gate_(gate), weight_(weight), waited_(waited) {}
        Ticket(Ticket&& other) noexcept : gate_(other.gate_), weight_(other.weight_), waited_(other.waited_) {
            other.gate_ = nullptr;
        }
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        Ticket& operator=(Ticket&&) = delete;
        ~Ticket() {
            if (gate_) gate_->leave(weight_);
        }
        // True when this compile had to wait for others to finish.
        bool waited() const { return waited_; }

    private:
        CompileGate* gate_;
        std::uint64_t weight_;
        bool waited_;
    };

    Ticket enter(std::uint64_t weight) {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::uint64_t turn = next_turn_++;
        bool waited = false;
        while (!(turn == serving_ && admits(weight))) {
            waited = true;
            changed_.wait(lock);
        }
        ++serving_;
        in_use_ += weight;
        ++in_flight_;
        changed_.notify_all();  // the next waiter in line may fit too
        return Ticket(this, weight, waited);
    }

    std::uint64_t budget() const { return budget_; }
    std::uint64_t in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return in_use_;
    }
    unsigned in_flight() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return in_flight_;
    }

private:
    bool admits(std::uint64_t weight) const {
        return budget_ == 0 || in_flight_ == 0 || in_use_ + weight <= budget_;
    }

    void leave(std::uint64_t weight) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            in_use_ -= weight;
            --in_flight_;
        }
        changed_.notify_all();
    }

    const std::uint64_t budget_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::uint64_t in_use_ = 0;
    unsigned in_flight_ = 0;
    std::uint64_t next_turn_ = 0;  // handed out on entry
    std::uint64_t serving_ = 0;    // the turn that may be admitted next
};

// The estimated host commit of compiling a module of `spirv_bytes` bytes: 1 KiB per byte of SPIR-V (the measured
// ratio for the 8-10 MB modules; smaller modules measured below it, so this errs on the safe side), at least 64 MiB.
inline std::uint64_t compile_weight(std::uint64_t spirv_bytes) {
    constexpr std::uint64_t kPerByte = 1024;
    constexpr std::uint64_t kMinimum = 64ull << 20;
    const std::uint64_t weight = spirv_bytes * kPerByte;
    return weight < kMinimum ? kMinimum : weight;
}

// The gate's budget in bytes. `env` is LEONIDA_COMPILE_BUDGET_MIB: a whole number of MiB, where 0 means no gate.
// Without it (or when it isn't a whole number) the budget is half of `available_physical` bytes, at least 2 GiB.
inline std::uint64_t compile_budget(const char* env, std::uint64_t available_physical) {
    if (env && *env) {
        char* end = nullptr;
        const unsigned long long mib = std::strtoull(env, &end, 10);
        if (end && *end == '\0' && *env != '-') return static_cast<std::uint64_t>(mib) << 20;
    }
    constexpr std::uint64_t kFloor = 2ull << 30;
    const std::uint64_t half = available_physical / 2;
    return half < kFloor ? kFloor : half;
}

}  // namespace leonida
