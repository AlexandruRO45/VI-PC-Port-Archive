#pragma once

#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace rt::d3d12_trace {

inline std::atomic<bool> tracing{false};

namespace detail {

inline constexpr char truncated_line[] = "[rt] d3d12 trace file reached its limit; later records are unavailable\n";
inline constexpr DWORD file_capacity = 64u * 1024u * 1024u;
inline std::mutex mutex;
inline unsigned long long sequence = 0;
inline HANDLE file = INVALID_HANDLE_VALUE;
inline char* view = nullptr;
inline std::size_t capacity = 0;
inline std::size_t written = 0;
inline bool full = false;

// The first byte of the next line is kept zero. Write the rest of the new line, including its
// newline, before replacing that zero. A process killed during the copy leaves the old terminator.
inline void append(const char* line, std::size_t length) noexcept {
    char* start = view + written;
    start[length] = '\0';
    std::memcpy(start + 1, line + 1, length - 1);
    std::atomic_ref<char>(*start).store(line[0], std::memory_order_release);
    written += length;
}

}  // namespace detail

// Open before running guest code. Return a Win32 error and keep any pre-existing file intact.
inline DWORD open_file(const wchar_t* path, DWORD capacity = detail::file_capacity) noexcept {
    std::lock_guard lock(detail::mutex);
    if (detail::view || !path || !*path || capacity <= sizeof(detail::truncated_line))
        return ERROR_INVALID_PARAMETER;
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return GetLastError();
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, capacity, nullptr);
    if (!mapping) {
        const DWORD error = GetLastError();
        CloseHandle(file);
        return error;
    }
    auto* view = static_cast<char*>(MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, capacity));
    const DWORD error = view ? ERROR_SUCCESS : GetLastError();
    CloseHandle(mapping);  // the view keeps the mapping alive until close_file()
    if (!view) {
        CloseHandle(file);
        return error;
    }
    view[0] = '\0';
    detail::file = file;
    detail::view = view;
    detail::capacity = capacity;
    detail::written = 0;
    detail::full = false;
    detail::sequence = 0;
    tracing.store(true, std::memory_order_release);
    return ERROR_SUCCESS;
}

inline void close_file() noexcept {
    std::lock_guard lock(detail::mutex);
    if (!detail::view) return;
    tracing.store(false, std::memory_order_relaxed);
    const bool flushed = !detail::written || FlushViewOfFile(detail::view, detail::written) != FALSE;
    UnmapViewOfFile(detail::view);
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(detail::written);
    if (!SetFilePointerEx(detail::file, end, nullptr, FILE_BEGIN) || !SetEndOfFile(detail::file))
        std::fprintf(stderr, "[rt] d3d12 trace file could not be trimmed; it retains its terminator\n");
    if (!flushed) std::fprintf(stderr, "[rt] d3d12 trace file could not be flushed\n");
    CloseHandle(detail::file);
    detail::file = INVALID_HANDLE_VALUE;
    detail::view = nullptr;
}

inline void set_enabled(bool enable) noexcept {
    tracing.store(enable, std::memory_order_relaxed);
}

inline bool enabled() noexcept {
    return tracing.load(std::memory_order_relaxed);
}

// Callers supply fixed method names and outcomes only. Keep one ordered record per call across threads.
inline void record(const char* method, const char* outcome = "called") noexcept {
    if (!enabled()) return;
    const std::lock_guard lock(detail::mutex);
    if (!enabled()) return;  // the mapped sink may have closed while this caller waited
    if (!detail::view) {
        std::fprintf(stderr, "[d3d12-trace] #%llu %s %s\n", ++detail::sequence, method, outcome);
        std::fflush(stderr);
        return;
    }
    if (detail::full) return;
    char line[512];
    const int size = std::snprintf(line, sizeof(line), "[d3d12-trace] #%llu %s %s\n", detail::sequence + 1,
                                   method, outcome);
    if (size <= 0 || std::size_t(size) >= sizeof(line) ||
        std::size_t(size) > detail::capacity - detail::written - sizeof(detail::truncated_line)) {
        detail::append(detail::truncated_line, sizeof(detail::truncated_line) - 1);
        detail::full = true;
        tracing.store(false, std::memory_order_relaxed);
        return;
    }
    detail::append(line, std::size_t(size));
    ++detail::sequence;
}

inline HRESULT result(const char* method, HRESULT code) noexcept {
    record(method, code == E_NOTIMPL ? "unsupported" : FAILED(code) ? "failed" : "ok");
    return code;
}

}  // namespace rt::d3d12_trace
