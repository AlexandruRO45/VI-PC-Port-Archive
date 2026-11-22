#include "era_query.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>

using namespace std::literals;
using Byte = unsigned char;

namespace {
struct ReleaseRegion {
    void operator()(Byte* p) const { VirtualFree(p, 0, MEM_RELEASE); }
};
using Region = std::unique_ptr<Byte, ReleaseRegion>;

struct QueryBuffer {
    MEMORY_BASIC_INFORMATION info;
    std::array<Byte, 16> tail;
};
static_assert(offsetof(QueryBuffer, tail) == era_query::tail_offset && sizeof(QueryBuffer) == 64);

// The post-query predicates of the four BVH callers, as modeled in memory_query_check.cpp.
bool caller_accepts(const QueryBuffer& buffer, std::size_t returned, bool source)
{
    return returned && (buffer.tail[0] & 3) && (buffer.info.State & 0x1000) &&
           (!source || (buffer.info.Protect & 6));
}

std::size_t page_size()
{
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize;
}

// Query as the callers do: a zeroed 64-byte buffer. A sentinel after the tail byte shows
// that only +0x30 is written there.
std::size_t ask(const void* address, QueryBuffer& buffer, HANDLE process = GetCurrentProcess())
{
    buffer = {};
    std::fill(buffer.tail.begin() + 1, buffer.tail.end(), Byte{0xa5});
    return era_query::query(process, address, &buffer, sizeof(buffer));
}

bool sentinels_intact(const QueryBuffer& buffer)
{
    return std::all_of(buffer.tail.begin() + 1, buffer.tail.end(), [](Byte b) { return b == 0xa5; });
}

// Bytes 0x00-0x2F must equal a plain VirtualQueryEx result for every page state.
bool fields_checks()
{
    const auto page = page_size();
    Region region(static_cast<Byte*>(VirtualAlloc(nullptr, 4 * page, MEM_RESERVE, PAGE_NOACCESS)));
    Byte* base = region.get();
    if (!base || !VirtualAlloc(base, 3 * page, MEM_COMMIT, PAGE_READWRITE)) return false;
    DWORD previous = 0;
    if (!VirtualProtect(base + page, page, PAGE_READONLY, &previous) ||
        !VirtualProtect(base + 2 * page, page, PAGE_NOACCESS, &previous)) return false;
    unsigned compared = 0;
    for (int registered = 0; registered < 2; ++registered) {
        if (registered && !era_query::register_gpu_region(base, 4 * page)) return false;
        for (std::size_t offset : {std::size_t{0}, page, 2 * page, 3 * page}) {
            MEMORY_BASIC_INFORMATION plain{};
            QueryBuffer buffer;
            if (VirtualQueryEx(GetCurrentProcess(), base + offset, &plain, sizeof(plain)) != sizeof(plain) ||
                ask(base + offset, buffer) != sizeof(plain) ||
                std::memcmp(&plain, &buffer.info, sizeof(plain)) != 0 || !sentinels_intact(buffer)) return false;
            ++compared;
        }
    }
    era_query::unregister_gpu_region(base);
    std::printf("field_comparisons=%u\n", compared);
    return true;
}

// The tail flag follows registration only; the caller predicates then follow State and Protect.
bool tail_checks()
{
    const auto page = page_size();
    Region region(static_cast<Byte*>(VirtualAlloc(nullptr, 4 * page, MEM_RESERVE, PAGE_NOACCESS)));
    Byte* base = region.get();
    if (!base || !VirtualAlloc(base, 3 * page, MEM_COMMIT, PAGE_READWRITE)) return false;
    QueryBuffer buffer;

    // Negative control: before registration the handler, like plain VirtualQueryEx, fails the tail check.
    if (!ask(base, buffer) || buffer.tail[0] != 0 || caller_accepts(buffer, sizeof(buffer.info), false))
        return false;

    if (!era_query::register_gpu_region(base, 2 * page)) return false;
    struct Case { std::size_t offset; Byte tail; };
    const Case cases[]{{0, 1}, {page / 2, 1}, {2 * page - 1, 1}, {2 * page, 0}, {3 * page, 0}};
    for (const auto& c : cases) {
        const auto written = ask(base + c.offset, buffer);
        if (!written || buffer.tail[0] != c.tail || !sentinels_intact(buffer)) return false;
    }

    // Registered read/write: both predicate shapes accept.
    auto written = ask(base, buffer);
    if (!caller_accepts(buffer, written, false) || !caller_accepts(buffer, written, true)) return false;
    // Registered read-only: Protect & 6 still holds for the source check.
    DWORD previous = 0;
    if (!VirtualProtect(base, page, PAGE_READONLY, &previous)) return false;
    written = ask(base, buffer);
    if (!caller_accepts(buffer, written, false) || !caller_accepts(buffer, written, true)) return false;
    // Registered no-access: the common check accepts, the source check rejects.
    if (!VirtualProtect(base, page, PAGE_NOACCESS, &previous)) return false;
    written = ask(base, buffer);
    if (!caller_accepts(buffer, written, false) || caller_accepts(buffer, written, true)) return false;
    // Registered but decommitted: State fails both.
    if (!VirtualFree(base, page, MEM_DECOMMIT)) return false;
    written = ask(base, buffer);
    if (buffer.info.State != MEM_RESERVE || caller_accepts(buffer, written, false) ||
        caller_accepts(buffer, written, true)) return false;
    // Committed read/write but outside the registered range: the tail check fails.
    written = ask(base + 2 * page, buffer);
    if (buffer.info.State != MEM_COMMIT || caller_accepts(buffer, written, false)) return false;

    // A real handle to this process counts as the current process.
    HANDLE self = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId());
    if (!self) return false;
    written = ask(base + page, buffer, self);
    CloseHandle(self);
    if (!written || buffer.tail[0] != 1) return false;

    // After unregistering, the flag clears again.
    if (!era_query::unregister_gpu_region(base)) return false;
    if (!ask(base + page, buffer) || buffer.tail[0] != 0) return false;
    std::printf("tail_cases=%zu predicate_cases=6\n", std::size(cases));
    return true;
}

bool registry_checks()
{
    alignas(64) static Byte block[0x400];
    const auto at = [](std::size_t offset) { return static_cast<const void*>(block + offset); };
    // Invalid input.
    if (era_query::register_gpu_region(nullptr, 16) || era_query::register_gpu_region(at(0), 0) ||
        era_query::register_gpu_region(reinterpret_cast<const void*>(~std::uintptr_t{0} - 4), 16))
        return false;
    if (!era_query::register_gpu_region(at(0x100), 0x100)) return false;
    // Exact, front, back, containing and contained overlaps are refused.
    struct Range { std::size_t offset, size; };
    const Range overlaps[]{{0x100, 0x100}, {0xc0, 0x80}, {0x1c0, 0x80}, {0x80, 0x200}, {0x140, 0x20}};
    for (const auto& r : overlaps) {
        if (era_query::register_gpu_region(at(r.offset), r.size)) return false;
    }
    // Adjacent ranges are accepted.
    if (!era_query::register_gpu_region(at(0x200), 0x100) ||
        !era_query::register_gpu_region(at(0x000), 0x100)) return false;
    // Unregistering needs the exact base.
    if (era_query::unregister_gpu_region(at(0x180)) || !era_query::unregister_gpu_region(at(0x100)) ||
        era_query::unregister_gpu_region(at(0x100))) return false;
    // The freed range can be registered again.
    const bool again = era_query::register_gpu_region(at(0x100), 0x100);
    const bool cleaned = era_query::unregister_gpu_region(at(0x000)) && era_query::unregister_gpu_region(at(0x100)) &&
                         era_query::unregister_gpu_region(at(0x200));
    std::printf("overlap_rejections=%zu\n", std::size(overlaps));
    return again && cleaned;
}

// A buffer shorter than MEMORY_BASIC_INFORMATION fails without writing, like VirtualQueryEx.
bool short_checks()
{
    std::array<Byte, sizeof(MEMORY_BASIC_INFORMATION)> buffer{};
    buffer.fill(0xd7);
    SetLastError(ERROR_SUCCESS);
    const auto written = era_query::query(GetCurrentProcess(), buffer.data(), buffer.data(), buffer.size() - 1);
    const auto error = GetLastError();
    const bool untouched = std::all_of(buffer.begin(), buffer.end(), [](Byte b) { return b == 0xd7; });
    // Exactly 0x30 bytes succeeds and writes no tail byte.
    std::array<Byte, sizeof(MEMORY_BASIC_INFORMATION) + 1> exact{};
    exact.back() = 0x5a;
    const auto exact_written = era_query::query(GetCurrentProcess(), exact.data(), exact.data(), exact.size() - 1);
    std::printf("short_query_bytes=%zu last_error=%lu\n", written, error);
    for (const std::size_t length : {buffer.size() - 1, buffer.size(), sizeof(QueryBuffer)}) {
        SetLastError(ERROR_SUCCESS);
        const auto native = VirtualQueryEx(GetCurrentProcess(), exact.data(), nullptr, length);
        const DWORD native_error = GetLastError();
        SetLastError(ERROR_SUCCESS);
        const auto result = era_query::query(GetCurrentProcess(), exact.data(), nullptr, length);
        const DWORD shim_error = GetLastError();
        std::printf("null_query_length=%zu native_error=%lu shim_error=%lu\n", length, native_error, shim_error);
        if (native || result || native_error == ERROR_SUCCESS || shim_error != native_error) return false;
    }
    return written == 0 && error == ERROR_BAD_LENGTH && untouched &&
           exact_written == sizeof(MEMORY_BASIC_INFORMATION) && exact.back() == 0x5a;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 2;
    }
    const std::string_view scenario = argv[1];
    bool passed;
    if (scenario == "fields"sv) {
        passed = fields_checks();
    } else if (scenario == "tail"sv) {
        passed = tail_checks();
    } else if (scenario == "registry"sv) {
        passed = registry_checks();
    } else if (scenario == "short"sv) {
        passed = short_checks();
    } else {
        return 2;
    }
    std::printf("Desktop XMemVirtualQuery handler %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
