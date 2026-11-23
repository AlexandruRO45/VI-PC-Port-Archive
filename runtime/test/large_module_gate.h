// How many large SPIR-V modules vk_pipeline_check lets into the driver at once, by a memory budget.
//
// Measured on RADV GFX1201 (Mesa 25.3.6) with the Mesa shader cache off: a 5.7 MB translated ray-tracing compute
// module peaks at 1.96 GB of process memory compiling alone (~330 bytes per SPIR-V byte), a 0.36 MB one at 94 MB.
// Compiling the large ones strictly one at a time left the other threads idle: 40 modules (27 over 4 MiB) on 6
// threads took 308 s exclusively and 75 s in parallel, 7.0 GB and 8.7 GB peak. kBytesPerSpirvByte keeps a margin
// over the measurement.
//
// The process grew past one module's peak even one at a time because glibc keeps freed memory in per-thread arenas
// (6 threads: 6.2 GB; MALLOC_ARENA_MAX=1: 2.1 GB, same modules). release_free_memory() hands it back after each
// large module.
//
// Windows vendor drivers (calibrated on NVIDIA 610.88 RTX 2080 Ti and 621.01 RTX 5060 Ti with --wave64-emulate) peak
// several times higher per SPIR-V byte: 2.5-2.7 MiB mesh modules peak ~2.8 GiB commit alone (~1,040 B/B) and
// ~12.5 GiB across 4 threads (~1,150 B/B), while a 15.03 MiB compute module peaked at 24.6 GB (~1,670 B/B).
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__GLIBC__) || defined(_WIN32)
#include <malloc.h>
#endif

namespace leonida {

#if defined(_WIN32)
inline constexpr uint64_t kBytesPerSpirvByte = 2048;
#else
inline constexpr uint64_t kBytesPerSpirvByte = 512;
#endif

inline uint64_t large_module_estimate(uint64_t spirv_bytes) {
    return spirv_bytes > UINT64_MAX / kBytesPerSpirvByte ? UINT64_MAX : spirv_bytes * kBytesPerSpirvByte;
}

#if defined(__linux__)
// The first unsigned number in a file ("max" and unreadable files give no value).
inline bool read_u64_file(const std::string& path, uint64_t& value) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long v = 0;
    const bool ok = std::fscanf(f, "%llu", &v) == 1;
    std::fclose(f);
    value = v;
    return ok;
}

// The room left under the cgroup v2 memory.max of this process and of every cgroup above it (a container or
// systemd-run -p MemoryMax=...), or UINT64_MAX when none sets a limit. cgroup_root is a parameter for the tests.
inline uint64_t cgroup_memory_room(const std::string& self_cgroup_file = "/proc/self/cgroup",
                                   const std::string& cgroup_root = "/sys/fs/cgroup") {
    std::FILE* f = std::fopen(self_cgroup_file.c_str(), "r");
    if (!f) return UINT64_MAX;
    char line[4096];
    std::string rel;
    while (std::fgets(line, sizeof(line), f)) {
        std::string l(line);
        if (l.rfind("0::", 0) == 0) {   // the cgroup v2 line
            rel = l.substr(3);
            while (!rel.empty() && (rel.back() == '\n' || rel.back() == '\r')) rel.pop_back();
            break;
        }
    }
    std::fclose(f);
    uint64_t room = UINT64_MAX;
    for (std::string dir = rel; !dir.empty();) {
        // In a cgroup namespace, "/" can be a constrained cgroup mounted as the visible root.
        const std::string base = dir == "/" ? cgroup_root : cgroup_root + dir;
        uint64_t max = 0, current = 0;
        if (read_u64_file(base + "/memory.max", max) && read_u64_file(base + "/memory.current", current))
            room = std::min(room, max > current ? max - current : 0);
        const auto slash = dir.find_last_of('/');
        if (dir == "/" || slash == std::string::npos) break;
        dir.resize(slash == 0 ? 1 : slash);
    }
    return room;
}
#endif

// Windows commit headroom: the smaller of free physical RAM and remaining commit limit (pagefile or job cap).
// Clamps to at least 1 byte when known, matching Linux's convention where 0 means "cannot tell".
inline uint64_t windows_available_memory(uint64_t avail_phys, uint64_t avail_pagefile) {
    return (std::max)(uint64_t(1), (std::min)(avail_phys, avail_pagefile));
}

// Memory the OS could give this process now, or 0 when it can't tell (then the caller keeps the old exclusive gate).
inline uint64_t available_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status)
               ? windows_available_memory(uint64_t(status.ullAvailPhys), uint64_t(status.ullAvailPageFile))
               : 0;
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { found = true; break; }
    std::fclose(f);
    if (!found) return 0;
    const uint64_t room = cgroup_memory_room();
    return std::max<uint64_t>(1, std::min(uint64_t(kb) * 1024, room));   // 0 would mean "unknown"
#else
    return 0;
#endif
}

// Return memory freed by the driver's compiler to the OS (glibc keeps it in the arena of the thread that freed it;
// Windows CRT heaps can likewise retain decommitted/free blocks across large allocations).
inline void release_free_memory() {
#if defined(__GLIBC__)
    malloc_trim(0);
#elif defined(_WIN32)
    _heapmin();
#endif
}

inline bool oversized_module_exceeds_memory(uint64_t estimate_bytes, uint64_t available_bytes) {
    return available_bytes > 0 && estimate_bytes > available_bytes;
}

inline std::string format_oversized_module_warning(std::string_view name, uint64_t estimate_bytes,
                                                   uint64_t available_bytes, bool exclusive_among_all) {
    char nums[128];
    std::snprintf(nums, sizeof(nums), " needs about %.1f GB, %.1f GB available: it compiles alone%s and may not fit\n",
                  estimate_bytes / 1e9, available_bytes / 1e9,
                  exclusive_among_all ? "" : " among large modules");
    std::string out = "  ";
    out.append(name);
    out.append(nums);
    return out;
}

// Lets large modules in while their estimates fit the budget and fewer than max_inflight are in. One always gets in
// when none is, so a module bigger than the whole budget still compiles (alone among the large ones).
class LargeModuleGate {
public:
    using AvailableMemoryFn = uint64_t (*)();

    LargeModuleGate(uint64_t budget, unsigned max_inflight, AvailableMemoryFn live_available = nullptr)
        : budget_(budget), active_budget_(budget), max_inflight_(max_inflight ? max_inflight : 1),
          live_available_(live_available) {}

    bool admits(uint64_t estimate) const {
        std::lock_guard<std::mutex> g(mu_);
        return admits_locked(estimate);
    }
    void enter(uint64_t estimate) {
        std::unique_lock<std::mutex> l(mu_);
        if (inflight_ == 0) refresh_budget_locked();
        cv_.wait(l, [&] { return admits_locked(estimate); });
        used_ = estimate > UINT64_MAX - used_ ? UINT64_MAX : used_ + estimate;
        ++inflight_;
    }
    void leave(uint64_t estimate) {
        {
            std::lock_guard<std::mutex> g(mu_);
            used_ = estimate >= used_ ? 0 : used_ - estimate;
            --inflight_;
            if (inflight_ == 0) refresh_budget_locked();
        }
        cv_.notify_all();
    }
    unsigned inflight() const {
        std::lock_guard<std::mutex> g(mu_);
        return inflight_;
    }
    uint64_t budget() const { return budget_; }
    uint64_t active_budget() const {
        std::lock_guard<std::mutex> g(mu_);
        return active_budget_;
    }
    unsigned max_inflight() const { return max_inflight_; }

    // Holds a place from construction until destruction (or release()), then hands freed memory back.
    class Ticket {
    public:
        Ticket() = default;
        Ticket(LargeModuleGate& gate, uint64_t estimate) : gate_(&gate), estimate_(estimate) { gate.enter(estimate); }
        Ticket(Ticket&& o) noexcept : gate_(o.gate_), estimate_(o.estimate_) { o.gate_ = nullptr; }
        Ticket& operator=(Ticket&& o) noexcept {
            if (this != &o) { release(); gate_ = o.gate_; estimate_ = o.estimate_; o.gate_ = nullptr; }
            return *this;
        }
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        ~Ticket() { release(); }
        void release() {
            if (!gate_) return;
            gate_->leave(estimate_);
            gate_ = nullptr;
            release_free_memory();
        }
    private:
        LargeModuleGate* gate_ = nullptr;
        uint64_t estimate_ = 0;
    };

private:
    void refresh_budget_locked() {
        if (!live_available_) return;
        const uint64_t live = live_available_();
        if (live > 0) active_budget_ = (std::min)(budget_, live / 4 * 3);
    }
    bool admits_locked(uint64_t estimate) const {
        if (inflight_ == 0) return true;
        if (inflight_ >= max_inflight_) return false;
        if (estimate > active_budget_ || used_ > active_budget_ - estimate) return false;
        if (live_available_) {
            const uint64_t live = live_available_();
            if (live > 0 && estimate > live / 4 * 3) return false;
        }
        return true;
    }

    const uint64_t budget_;
    uint64_t active_budget_;
    const unsigned max_inflight_;
    const AvailableMemoryFn live_available_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t used_ = 0;
    unsigned inflight_ = 0;
};

}  // namespace leonida
