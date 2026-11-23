#pragma once
// What the D3D12 layer's slot-133 shadow buffers (ShadowPages in d3d12_vulkan_device.cpp) need from the rest of the
// runtime to see every write to the title's pages.
//
// A shadow notices the title's writes by making the buffer's pages read-only between submissions: a write faults,
// the layer's handler makes the page writable again and the page is copied at the next submission. Two kinds of
// write don't fault there, and are reported here instead:
//
// 1. Aliases (title_page_aliases). xmem maps one physical page at as many VAs as the title asks
//    (XMemMapPhysicalPages): a write through one VA changes what the others read without touching them. xmem
//    reports each 64 KiB page mapped at more than one address; a shadow copies those at every submission.
//
// 2. Kernel writes (kernel_writes). A read-only page makes ReadFile fail with ERROR_NOACCESS instead of faulting
//    into a user-mode handler. The file shims (read_file and the rest, below) pin the destination before the kernel
//    writes it: the layer makes its pages writable and doesn't protect them again until the pin ends, when the
//    call returns or, for an overlapped read, once its OVERLAPPED says it completed. A read that still fails with
//    ERROR_NOACCESS is retried once after the destination is opened again.
//
//    Any other host code that has the kernel write title memory (a DirectStorage queue that ReadFiles straight into
//    a request's destination, for one) does the same: kernel_writes::read_file instead of ::ReadFile, or
//    begin_write(destination, bytes, overlapped) before the call and end_write(id, pending) after it. A user-mode
//    copy into title memory needs nothing: it faults like the title's own writes.
//
// The shadows themselves are brought up to date in one place, DeviceState::refresh_shadows in
// d3d12_vulkan_device.cpp, which VulkanQueue::execute_now calls (VulkanContext::set_before_execute) before every
// submission of command lists.
//
// Header-only, so the D3D12 layer, xmem and the file shims share one state in a process without linking each other.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace leonida::title_page_aliases {

constexpr std::uintptr_t kPageSize = 0x10000;

struct State {
    std::shared_mutex mutex;
    std::unordered_set<std::uintptr_t> aliased;  // 64 KiB-aligned VAs
    std::atomic<std::uint64_t> generation{0};
};

inline State& state() {
    static State s;
    return s;
}

// The 64 KiB page at `va` (rounded down) shows a page that is also mapped elsewhere (true), or no longer does.
inline void set(std::uintptr_t va, bool aliased) {
    State& s = state();
    va &= ~(kPageSize - 1);
    std::unique_lock lock(s.mutex);
    const bool changed = aliased ? s.aliased.insert(va).second : s.aliased.erase(va) != 0;
    if (changed) s.generation.fetch_add(1, std::memory_order_release);
}

// Bumped by every change of set().
inline std::uint64_t generation() { return state().generation.load(std::memory_order_acquire); }

// Calls `f(va)` for each aliased 64 KiB page overlapping [begin, end).
template<class F>
void for_each_in(std::uintptr_t begin, std::uintptr_t end, F&& f) {
    State& s = state();
    std::shared_lock lock(s.mutex);
    if (s.aliased.empty()) return;
    for (std::uintptr_t va = begin & ~(kPageSize - 1); va < end; va += kPageSize)
        if (s.aliased.count(va)) f(va);
}

}  // namespace leonida::title_page_aliases

namespace leonida::kernel_writes {

// The layer's side: makes every watched page in [begin, end) writable. Set once the first shadow exists.
using Opener = void (*)(std::uintptr_t begin, std::uintptr_t end);

struct Pin {
    std::uintptr_t begin, end;
    const OVERLAPPED* overlapped;  // null: ends when the call returns
    std::uint64_t id;
};

struct State {
    std::atomic<Opener> opener{nullptr};
    std::mutex mutex;
    std::vector<Pin> pins;
    std::uint64_t next_id = 1;
    std::atomic<std::size_t> count{0};  // pins.size(), read without the lock
};

inline State& state() {
    static State s;
    return s;
}

inline void set_opener(Opener opener) { state().opener.store(opener, std::memory_order_release); }

// Whether an overlapped request is still in flight. Its OVERLAPPED is the title's and may be gone by now: a read that
// faults counts as completed (the title doesn't free an OVERLAPPED it is still waiting on).
inline bool overlapped_pending(const OVERLAPPED* overlapped) {
    __try {
        return overlapped->Internal == static_cast<ULONG_PTR>(STATUS_PENDING);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// The pinned ranges overlapping [begin, end), for the layer's refresh. Overlapped pins whose request completed are
// dropped first.
template<class F>
void for_each_pin(std::uintptr_t begin, std::uintptr_t end, F&& f) {
    State& s = state();
    if (!s.count.load(std::memory_order_acquire)) return;
    std::lock_guard lock(s.mutex);
    std::erase_if(s.pins, [](const Pin& pin) { return pin.overlapped && !overlapped_pending(pin.overlapped); });
    s.count.store(s.pins.size(), std::memory_order_release);
    for (const Pin& pin : s.pins)
        if (pin.begin < end && begin < pin.end) f((std::max)(pin.begin, begin), (std::min)(pin.end, end));
}

// Before the kernel writes [buffer, buffer + bytes): pins the range (until end_write(), or until `overlapped` completes
// when it is not null) and opens it. Returns the pin's id, 0 when nothing was pinned.
inline std::uint64_t begin_write(const void* buffer, std::size_t bytes, const OVERLAPPED* overlapped) {
    State& s = state();
    const Opener opener = s.opener.load(std::memory_order_acquire);
    if (!opener || !buffer || !bytes) return 0;
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer);
    std::uint64_t id = 0;
    {
        std::lock_guard lock(s.mutex);
        id = s.next_id++;
        s.pins.push_back({begin, begin + bytes, overlapped, id});
        s.count.store(s.pins.size(), std::memory_order_release);
    }
    // After the pin: a refresh that runs from here on leaves the range writable.
    opener(begin, begin + bytes);
    return id;
}

// The call is over: a synchronous pin ends. An overlapped pin whose request is still pending stays until it completes.
inline void end_write(std::uint64_t id, bool pending) {
    if (!id) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    std::erase_if(s.pins, [id, pending](const Pin& pin) { return pin.id == id && !(pending && pin.overlapped); });
    s.count.store(s.pins.size(), std::memory_order_release);
}

// Opens [buffer, buffer + bytes) again, for a retry after ERROR_NOACCESS. false: no shadow watches anything.
inline bool reopen(const void* buffer, std::size_t bytes) {
    const Opener opener = state().opener.load(std::memory_order_acquire);
    if (!opener || !buffer || !bytes) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer);
    opener(begin, begin + bytes);
    return true;
}

// The file shims' reads into title memory (file_system.cpp): ReadFile, ReadFileEx and ReadFileScatter with their
// destination pinned and opened first, and retried once if it fails with ERROR_NOACCESS anyway.
inline BOOL read_file(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped) {
    const std::uint64_t pin = begin_write(buffer, bytes, overlapped);
    BOOL ok = ::ReadFile(file, buffer, bytes, read, overlapped);
    if (!ok && GetLastError() == ERROR_NOACCESS && reopen(buffer, bytes))
        ok = ::ReadFile(file, buffer, bytes, read, overlapped);
    const DWORD error = GetLastError();
    end_write(pin, !ok && error == ERROR_IO_PENDING);
    SetLastError(error);
    return ok;
}

inline BOOL read_file_ex(HANDLE file, LPVOID buffer, DWORD bytes, LPOVERLAPPED overlapped,
                         LPOVERLAPPED_COMPLETION_ROUTINE routine) {
    const std::uint64_t pin = begin_write(buffer, bytes, overlapped);
    BOOL ok = ::ReadFileEx(file, buffer, bytes, overlapped, routine);
    if (!ok && GetLastError() == ERROR_NOACCESS && reopen(buffer, bytes))
        ok = ::ReadFileEx(file, buffer, bytes, overlapped, routine);
    const DWORD error = GetLastError();
    end_write(pin, bool(ok));  // queued: pending until the OVERLAPPED says otherwise
    SetLastError(error);
    return ok;
}

// Each segment is one system page.
inline BOOL read_file_scatter(HANDLE file, FILE_SEGMENT_ELEMENT segments[], DWORD bytes, LPDWORD reserved,
                              LPOVERLAPPED overlapped) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    std::vector<std::uint64_t> pins;
    const DWORD count = segments ? (bytes + info.dwPageSize - 1) / info.dwPageSize : 0;
    for (DWORD i = 0; i < count; ++i)
        pins.push_back(begin_write(segments[i].Buffer, info.dwPageSize, overlapped));
    BOOL ok = ::ReadFileScatter(file, segments, bytes, reserved, overlapped);
    if (!ok && GetLastError() == ERROR_NOACCESS && count) {
        for (DWORD i = 0; i < count; ++i) reopen(segments[i].Buffer, info.dwPageSize);
        ok = ::ReadFileScatter(file, segments, bytes, reserved, overlapped);
    }
    const DWORD error = GetLastError();
    for (const std::uint64_t pin : pins) end_write(pin, bool(ok) || error == ERROR_IO_PENDING);
    SetLastError(error);
    return ok;
}

}  // namespace leonida::kernel_writes
