// A hard cap on vk_pipeline_check's committed memory on Windows, so one runaway driver compile fails inside the
// checker instead of exhausting the system's commit limit.
//
// Measured on Windows 11, RTX 3080 (NVIDIA 621.14), 32 GB RAM: one wave64-emulated compute module of the 1.0.2.1
// corpus compiling alone took the checker to about 69 GB of private bytes. The system hit its commit limit, other
// programs failed with 0xC000012D (STATUS_COMMITMENT_LIMIT) and the display driver reset. large_module_gate.h only
// estimates a module's memory before it enters the driver; it cannot stop one that allocates far past the estimate.
//
// The cap puts the process into a job object with JOB_OBJECT_LIMIT_PROCESS_MEMORY. Past the limit the process's
// allocations fail, so the driver returns an error or the checker dies: make_game.py's pipelines stage and --isolate
// already handle a dead checker (the module is named in --inflight-file and retried alone, then skipped). Processes
// the checker starts (--isolate children) are in the same job, and the limit applies to each of them on its own.
// Off Windows these functions do nothing: a Linux container or systemd-run -p MemoryMax=... already caps the
// process, and large_module_gate.h reads that limit.
#pragma once

#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace leonida {

// The default cap: the machine's physical memory. A compile needing more than that would page the system to a
// standstill even where the page file can still back it.
inline uint64_t default_process_memory_cap() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (GlobalMemoryStatusEx(&status)) return status.ullTotalPhys;
#endif
    return 0;
}

// Caps the current process at `bytes` of committed memory. 0 = no cap. Returns false with `error` set when the job
// object can't be set up; the caller decides whether to go on without the cap.
inline bool apply_process_memory_cap(uint64_t bytes, std::string& error) {
    error.clear();
    if (bytes == 0) return true;
#ifdef _WIN32
    const auto failed = [&](const char* api) {
        error = std::string(api) + " failed (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    };
    if (bytes > static_cast<uint64_t>(SIZE_MAX)) bytes = SIZE_MAX;
    // The handle stays open for the life of the process: closing the last handle to a job ends its limits only
    // with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, which is not set, but keeping it open is the documented pattern.
    const HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) return failed("CreateJobObjectW");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = static_cast<SIZE_T>(bytes);
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits)) {
        const bool ok = failed("SetInformationJobObject");
        CloseHandle(job);
        return ok;
    }
    // Nested jobs (Windows 8 and later) let this work when a terminal, CI runner or make_game.py already put the
    // process in a job.
    if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
        const bool ok = failed("AssignProcessToJobObject");
        CloseHandle(job);
        return ok;
    }
    return true;
#else
    (void)bytes;
    return true;
#endif
}

}  // namespace leonida
