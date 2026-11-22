// Optional Windows commit cap for the pipeline checker. The job covers the checker and any
// processes it starts. Windows counts committed memory, so a large address reservation alone
// does not consume this budget.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

namespace leonida::pipeline_budget {

inline constexpr std::size_t mib = 1024 * 1024;
inline constexpr std::size_t minimum_auto_mib = 256;

struct Budget {
    std::size_t bytes = 0;
    bool automatic = false;
};

inline std::optional<Budget> parse(std::string_view value) {
    if (value == "auto") return Budget{0, true};
    if (value.empty()) return std::nullopt;
    std::size_t count = 0;
    const auto end = value.data() + value.size();
    const auto [ptr, error] = std::from_chars(value.data(), end, count);
    if (error != std::errc{} || ptr != end || !count || count > std::numeric_limits<std::size_t>::max() / mib)
        return std::nullopt;
    return Budget{count * mib, false};
}

// Keep 25% headroom against both reported commit availability and physical memory availability.
// Round down to MiB without overflowing even when the OS reports data wider than size_t.
inline std::size_t automatic_budget(std::uint64_t available_commit, std::uint64_t available_phys) {
    const auto available = available_commit < available_phys ? available_commit : available_phys;
    const auto share = available - available / 4 - (available % 4 != 0);
    const auto largest = std::numeric_limits<std::size_t>::max() / mib;
    const auto count = share / mib < largest ? share / mib : largest;
    return count < minimum_auto_mib ? 0 : static_cast<std::size_t>(count) * mib;
}

#ifdef _WIN32
struct Error {
    const char* operation;
    DWORD code;
};

class CommitJob {
public:
    CommitJob() = default;
    CommitJob(const CommitJob&) = delete;
    CommitJob& operator=(const CommitJob&) = delete;
    ~CommitJob() {
        if (job_) CloseHandle(job_);
    }

    // Apply before loading Vulkan so driver allocations and child processes are subject to the cap.
    // Create an unnamed, non-inheritable job without breakaway flags. Process membership, not an
    // inherited handle, keeps descendants subject to the limit.
    std::optional<Error> apply(Budget budget) {
        if (job_) return Error{"job already configured", ERROR_ALREADY_EXISTS};
        if (budget.automatic) {
            MEMORYSTATUSEX memory{};
            memory.dwLength = sizeof(memory);
            if (!GlobalMemoryStatusEx(&memory)) return Error{"GlobalMemoryStatusEx", GetLastError()};
            budget.bytes = automatic_budget(memory.ullAvailPageFile, memory.ullAvailPhys);
            if (!budget.bytes) return Error{"insufficient available memory for the automatic cap", ERROR_NOT_ENOUGH_MEMORY};
        }
        if (!budget.bytes) return Error{"zero commit limit", ERROR_INVALID_PARAMETER};

        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        if (!job) return Error{"CreateJobObjectW", GetLastError()};
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
        limits.JobMemoryLimit = budget.bytes;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            const auto error = GetLastError();
            CloseHandle(job);
            return Error{"SetInformationJobObject", error};
        }
        // Check that the OS accepted the actual byte limit before joining. Some compatibility
        // layers acknowledge SetInformationJobObject without storing the memory limit.
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION installed{};
        if (!QueryInformationJobObject(job, JobObjectExtendedLimitInformation, &installed, sizeof(installed), nullptr)) {
            const auto error = GetLastError();
            CloseHandle(job);
            return Error{"QueryInformationJobObject", error};
        }
        if (!(installed.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_JOB_MEMORY) ||
            installed.JobMemoryLimit != budget.bytes) {
            CloseHandle(job);
            return Error{"verify job memory limit", ERROR_NOT_SUPPORTED};
        }
        if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
            const auto error = GetLastError();
            CloseHandle(job);
            return Error{"AssignProcessToJobObject", error};
        }
        job_ = job;
        limit_ = budget.bytes;
        return std::nullopt;
    }

    std::size_t limit() const { return limit_; }

    // Windows can account for reservations or rejected allocation attempts in this high-water mark.
    std::optional<std::size_t> peak_job_memory_used() const {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        if (!job_ || !QueryInformationJobObject(job_, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr))
            return std::nullopt;
        return info.PeakJobMemoryUsed;
    }

private:
    HANDLE job_ = nullptr;
    std::size_t limit_ = 0;
};
#endif

}  // namespace leonida::pipeline_budget
