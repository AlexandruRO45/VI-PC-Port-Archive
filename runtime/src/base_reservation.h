#pragma once

// Keeping a relocation-less game image's preferred range free until the loader maps it there.
//
// An EXE without a .reloc directory can only run at its ImageBase (0x140000000 by default). Windows and x86-64 Wine
// leave that range alone, but arm64 Wine on macOS (CrossOver's arm64 bottles, whose x86-64 code runs under FEX) puts
// its own start-up allocations at 0x100000000 and up, over that range, before gta6rt's first line runs. So gta6rt
// starts itself again, suspended, reserves [ImageBase, ImageBase + SizeOfImage) in that child before any of its code
// runs, and resumes it; the child's loader then commits the image inside the reservation (commit_reserved), so
// nothing else can take the range in between. The child is told by LEONIDA_BASE_RESERVED=<base>:<size> (hex), which
// also stops it from starting itself again. Measured on CrossOver Preview 20260821 (arm64 bottle, M2 Max): in a
// suspended child 0x140000000-0x170000000 is free and reservable; macOS maps its own regions above that (the arm64
// shared region starts at 0x180000000), so a range there can't be had either way.
//
// Nothing changes where the range is free at start-up (Windows, x86-64 Wine) or where the image has relocations.

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace rt::base_reservation {

inline constexpr wchar_t kVariable[] = L"LEONIDA_BASE_RESERVED";

struct Range {
    std::uint64_t base = 0;
    std::uint64_t size = 0;
};

namespace detail {

inline bool valid_range(const Range& r) {
    return r.base != 0 && (r.base & 0xffffu) == 0 && r.size != 0 &&
           r.size <= (std::numeric_limits<std::uint64_t>::max)() - r.base;
}

inline bool parse_hex_u64(const wchar_t*& p, std::uint64_t& out) {
    out = 0;
    std::size_t digits = 0;
    constexpr std::uint64_t kMax = (std::numeric_limits<std::uint64_t>::max)();
    for (; *p; ++p) {
        std::uint64_t d = 0;
        const wchar_t ch = *p;
        if (ch >= L'0' && ch <= L'9') d = static_cast<std::uint64_t>(ch - L'0');
        else if (ch >= L'a' && ch <= L'f') d = static_cast<std::uint64_t>(ch - L'a' + 10);
        else if (ch >= L'A' && ch <= L'F') d = static_cast<std::uint64_t>(ch - L'A' + 10);
        else break;
        if (out > (kMax - d) / 16) return false;
        out = out * 16 + d;
        ++digits;
    }
    return digits > 0;
}

class ScopedEnvVar {
public:
    explicit ScopedEnvVar(const wchar_t* name) : name_(name) {
        DWORD cap = GetEnvironmentVariableW(name_, nullptr, 0);
        if (cap == 0) {
            had_value_ = (GetLastError() != ERROR_ENVVAR_NOT_FOUND);
            return;
        }
        for (;;) {
            saved_.resize(cap);
            SetLastError(ERROR_SUCCESS);
            const DWORD written = GetEnvironmentVariableW(name_, saved_.data(), cap);
            if (written == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
                had_value_ = false;
                saved_.clear();
                return;
            }
            if (written < cap) {
                had_value_ = true;
                return;
            }
            cap = written;
        }
    }
    ~ScopedEnvVar() { SetEnvironmentVariableW(name_, had_value_ ? saved_.data() : nullptr); }
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const wchar_t* name_;
    bool had_value_ = false;
    std::vector<wchar_t> saved_;
};

inline bool module_file_name(std::vector<wchar_t>& out) {
    out.resize(MAX_PATH);
    for (;;) {
        SetLastError(ERROR_SUCCESS);
        const DWORD length = GetModuleFileNameW(nullptr, out.data(), static_cast<DWORD>(out.size()));
        if (!length) return false;
        if (length < out.size()) return true;
        if (out.size() >= 32768) return false;
        out.resize(out.size() > 16384 ? 32768 : out.size() * 2);
    }
}

}  // namespace detail

// The range this process was started with (LEONIDA_BASE_RESERVED), if any.
inline std::optional<Range> inherited() {
    wchar_t value[64] = {};
    const DWORD length = GetEnvironmentVariableW(kVariable, value, 64);
    if (!length || length >= 64) return std::nullopt;
    Range r{};
    const wchar_t* p = value;
    if (!detail::parse_hex_u64(p, r.base) || *p != L':') return std::nullopt;
    ++p;
    if (!detail::parse_hex_u64(p, r.size) || *p != L'\0') return std::nullopt;
    if (!detail::valid_range(r)) return std::nullopt;
    return r;
}

// Commits [base, base + size) for the loader when it lies inside the reservation this process was started with and
// that reservation is still in place; returns null otherwise (the caller then allocates as usual). The reservation
// is the allocation, so VirtualFree(base, 0, MEM_RELEASE) frees it as it frees any image.
inline void* commit_reserved(std::uint64_t base, std::uint64_t size) {
    if (!size || size > (std::numeric_limits<SIZE_T>::max)()) return nullptr;
    const auto r = inherited();
    if (!r || base != r->base || size > r->size) return nullptr;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<void*>(base), &info, sizeof info) || info.State != MEM_RESERVE ||
        reinterpret_cast<std::uint64_t>(info.AllocationBase) != base ||
        reinterpret_cast<std::uint64_t>(info.BaseAddress) != base ||
        static_cast<std::uint64_t>(info.RegionSize) < size)
        return nullptr;
    return VirtualAlloc(reinterpret_cast<void*>(base), static_cast<SIZE_T>(size), MEM_COMMIT, PAGE_READWRITE);
}

// The preferred range of `image` when it has no relocations and that range is not free in this process; nullopt
// when the image can be mapped as it is (or is not a readable PE32+, which the loader reports itself).
inline std::optional<Range> needed(const std::filesystem::path& image) {
    std::ifstream file(image, std::ios::binary);
    IMAGE_DOS_HEADER dos{};
    if (!file.read(reinterpret_cast<char*>(&dos), sizeof dos) || dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < 0)
        return std::nullopt;
    IMAGE_NT_HEADERS64 nt{};
    if (!file.seekg(dos.e_lfanew) || !file.read(reinterpret_cast<char*>(&nt), sizeof nt) ||
        nt.Signature != IMAGE_NT_SIGNATURE || nt.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC ||
        !nt.OptionalHeader.SizeOfImage)
        return std::nullopt;
    const auto& oh = nt.OptionalHeader;
    if (oh.NumberOfRvaAndSizes > IMAGE_DIRECTORY_ENTRY_BASERELOC &&
        oh.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC].Size)
        return std::nullopt;  // relocatable: the loader moves it if the base is taken
    const Range r{oh.ImageBase, (static_cast<std::uint64_t>(oh.SizeOfImage) + 0xffff) & ~std::uint64_t{0xffff}};
    if (!detail::valid_range(r)) return std::nullopt;
    const std::uint64_t end = r.base + r.size;
    for (std::uint64_t at = r.base; at < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof info)) return r;
        if (info.State != MEM_FREE) return r;
        const auto region_base = reinterpret_cast<std::uint64_t>(info.BaseAddress);
        const auto region_size = static_cast<std::uint64_t>(info.RegionSize);
        if (!region_size) return r;
        if (region_size > (std::numeric_limits<std::uint64_t>::max)() - region_base) break;
        const std::uint64_t next = region_base + region_size;
        if (next <= at) return r;
        at = next;
    }
    return std::nullopt;
}

// Runs this program again with the same command line, suspended, reserves `r` in the new process, resumes it and
// returns its exit code. On failure prints why and returns 1 (the child, if any, is ended unrun).
inline int relaunch(const Range& r) {
    if (!detail::valid_range(r) || r.size > (std::numeric_limits<SIZE_T>::max)()) {
        std::fprintf(stderr, "error: invalid reservation range %#llx-%#llx\n",
                     static_cast<unsigned long long>(r.base), static_cast<unsigned long long>(r.size));
        return 1;
    }
    std::vector<wchar_t> self;
    if (!detail::module_file_name(self)) {
        std::fprintf(stderr, "error: GetModuleFileName failed (%lu)\n", GetLastError());
        return 1;
    }
    wchar_t marker[64];
    std::swprintf(marker, 64, L"%llx:%llx", static_cast<unsigned long long>(r.base),
                  static_cast<unsigned long long>(r.size));
    detail::ScopedEnvVar restore_env(kVariable);
    // Inherited by the child: it tells it to commit inside the reservation and not to relaunch again.
    if (!SetEnvironmentVariableW(kVariable, marker)) {
        std::fprintf(stderr, "error: setting %ls failed (%lu)\n", kVariable, GetLastError());
        return 1;
    }
    std::wstring command_line = GetCommandLineW();
    STARTUPINFOW startup{sizeof startup};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : {startup.hStdInput, startup.hStdOutput, startup.hStdError})
        if (h && h != INVALID_HANDLE_VALUE) SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    PROCESS_INFORMATION process{};
    std::fflush(stdout);
    std::fflush(stderr);
    if (!CreateProcessW(self.data(), command_line.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr,
                        &startup, &process)) {
        std::fprintf(stderr, "error: starting gta6rt again to reserve the image range failed (%lu)\n", GetLastError());
        return 1;
    }
    const auto abandon = [&process] {  // the child never ran: end it and release its handles
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
    };
    if (!VirtualAllocEx(process.hProcess, reinterpret_cast<void*>(r.base), static_cast<SIZE_T>(r.size), MEM_RESERVE,
                        PAGE_NOACCESS)) {
        const DWORD error = GetLastError();
        abandon();
        std::fprintf(stderr,
                     "error: the image has no relocations and its range %#llx-%#llx is taken, also in a new "
                     "process (VirtualAllocEx %lu)\n",
                     static_cast<unsigned long long>(r.base), static_cast<unsigned long long>(r.base + r.size), error);
        return 1;
    }
    if (ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {  // still suspended: the wait would never end
        const DWORD error = GetLastError();
        abandon();
        std::fprintf(stderr, "error: resuming the relaunched gta6rt failed (%lu)\n", error);
        return 1;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}

}  // namespace rt::base_reservation
