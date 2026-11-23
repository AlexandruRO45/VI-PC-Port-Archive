// Hooks on a function of the loaded game image, by redirecting its direct callers (docs/modding.md, "Day one").
//
// Patching inside a function needs an instruction decoder to move its prologue. Redirecting the callers does not:
// every `call rel32` (E8) and `jmp rel32` (E9) in the image's executable sections whose target is the function is
// pointed at a thunk instead. The thunk saves the argument registers (rcx, rdx, r8, r9, xmm0-3), passes one 32-bit
// argument through a C++ function, restores the rest and jumps to the original. A call through a pointer (vtable,
// import slot) is not redirected, which suits RAGE's archetype lookup: a static function called directly.
//
// A byte pattern can look like E8/E9 + rel32 inside another instruction; it is only rewritten when its target is
// exactly the hooked function, so a false match must also point there. x64 Windows only.
#pragma once

#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace leonida::hooks {

using RemapFn = uint32_t (*)(uint32_t);

// True when `dst` is reachable from a 5-byte rel32 branch whose next instruction is at `next_ip`.
inline bool rel32_reaches(uintptr_t next_ip, uintptr_t dst) {
    constexpr uintptr_t kMaxPos = static_cast<uintptr_t>(INT32_MAX);
    constexpr uintptr_t kMaxNeg = static_cast<uintptr_t>(INT32_MAX) + 1u;
    return dst >= next_ip ? (dst - next_ip <= kMaxPos) : (next_ip - dst <= kMaxNeg);
}

// True when [image_base + offset, image_base + offset + length) is committed readable memory in the same allocation
// as `image_base` (works for both VirtualAlloc-mapped images and OS SEC_IMAGE mappings).
inline bool readable_image_span(const uint8_t* image_base, uint64_t offset, uint64_t length) {
    if (!image_base || length == 0) return false;
    if (UINT64_MAX - offset < length) return false;
    const uint64_t end_off = offset + length;
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(image_base);
    if (end_off > UINTPTR_MAX || UINTPTR_MAX - base_addr < static_cast<uintptr_t>(end_off)) return false;

    MEMORY_BASIC_INFORMATION base_mbi{};
    if (VirtualQuery(image_base, &base_mbi, sizeof(base_mbi)) == 0) return false;
    if (base_mbi.State != MEM_COMMIT || !base_mbi.AllocationBase) return false;
    const void* const alloc_base = base_mbi.AllocationBase;

    uintptr_t cur = base_addr + static_cast<uintptr_t>(offset);
    const uintptr_t end = base_addr + static_cast<uintptr_t>(end_off);
    while (cur < end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(reinterpret_cast<const void*>(cur), &mbi, sizeof(mbi)) == 0) return false;
        if (mbi.State != MEM_COMMIT || mbi.AllocationBase != alloc_base) return false;
        const DWORD low_prot = mbi.Protect & 0xFFu;
        if (low_prot == 0 || low_prot == PAGE_NOACCESS || (mbi.Protect & PAGE_GUARD) != 0) return false;
        const uintptr_t region_base = reinterpret_cast<uintptr_t>(mbi.BaseAddress);
        if (mbi.RegionSize == 0) return false;
        if (UINTPTR_MAX - region_base < mbi.RegionSize) return true;
        const uintptr_t region_end = region_base + mbi.RegionSize;
        if (region_end <= cur) return false;
        cur = region_end;
    }
    return true;
}

// Executable memory reachable via rel32 from every branch whose next-instruction address lies in
// [min_next_ip, max_next_ip]. nullptr if the range cannot be spanned by a single rel32 target or none is free.
inline uint8_t* alloc_near_range(uintptr_t min_next_ip, uintptr_t max_next_ip, std::size_t size) {
    if (min_next_ip == 0 || max_next_ip < min_next_ip || size == 0) return nullptr;
    if (max_next_ip - min_next_ip > 0xFFFFFFFFULL) return nullptr;
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uintptr_t gran = si.dwAllocationGranularity ? si.dwAllocationGranularity : 0x10000u;
    const uintptr_t mid = min_next_ip + (max_next_ip - min_next_ip) / 2;
    const uintptr_t center = mid & ~(gran - 1);
    auto try_at = [&](uintptr_t at) -> uint8_t* {
        if (at < gran) return nullptr;
        if (!rel32_reaches(min_next_ip, at) || !rel32_reaches(max_next_ip, at)) return nullptr;
        void* p = VirtualAlloc(reinterpret_cast<void*>(at), size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        if (!p) return nullptr;
        const uintptr_t p_addr = reinterpret_cast<uintptr_t>(p);
        if (!rel32_reaches(min_next_ip, p_addr) || !rel32_reaches(max_next_ip, p_addr)) {
            VirtualFree(p, 0, MEM_RELEASE);
            return nullptr;
        }
        return static_cast<uint8_t*>(p);
    };
    if (uint8_t* p = try_at(center)) return p;
    constexpr uintptr_t kMaxSpan = static_cast<uintptr_t>(INT32_MAX) + 1u;
    for (uintptr_t delta = gran; delta <= kMaxSpan; delta += gran) {
        bool any_reachable = false;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && center < delta) continue;
            if (side == 1 && UINTPTR_MAX - center < delta) continue;
            const uintptr_t at = side ? center + delta : center - delta;
            if (at >= gran && rel32_reaches(min_next_ip, at) && rel32_reaches(max_next_ip, at)) {
                any_reachable = true;
                if (uint8_t* p = try_at(at)) return p;
            }
        }
        if (!any_reachable) break;
    }
    return nullptr;
}

// Executable memory within +-2 GiB of `close_to`, so a rel32 branch from the image reaches it. nullptr if none is free.
inline uint8_t* alloc_near(const void* close_to, std::size_t size) {
    if (!close_to) return nullptr;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(close_to);
    return alloc_near_range(addr, addr, size);
}

// The thunk: argument `arg` (0 = rcx, 1 = rdx; the low 32 bits) goes through `fn`, then the original runs with every
// other argument register as the caller left it. Stack: the caller's call leaves rsp = 8 mod 16; four pushes and
// 0x68 bytes make it 16-aligned for the call to `fn`, with its 32-byte shadow space and room for xmm0-3.
inline uint8_t* make_remap_thunk_in_range(uintptr_t min_next_ip, uintptr_t max_next_ip, int arg, RemapFn fn,
                                          const void* original) {
    if (min_next_ip == 0 || max_next_ip < min_next_ip || !fn || !original || (arg != 0 && arg != 1)) return nullptr;
    uint8_t* t = alloc_near_range(min_next_ip, max_next_ip, 4096);
    if (!t) return nullptr;
    uint8_t* p = t;
    auto put = [&](std::initializer_list<uint8_t> bytes) { for (uint8_t b : bytes) *p++ = b; };
    auto put64 = [&](uint64_t v) { std::memcpy(p, &v, 8); p += 8; };
    put({0x51, 0x52, 0x41, 0x50, 0x41, 0x51});        // push rcx, rdx, r8, r9
    put({0x48, 0x83, 0xEC, 0x68});                    // sub rsp, 0x68
    put({0xF3, 0x0F, 0x7F, 0x44, 0x24, 0x20});        // movdqu [rsp+0x20], xmm0
    put({0xF3, 0x0F, 0x7F, 0x4C, 0x24, 0x30});        // movdqu [rsp+0x30], xmm1
    put({0xF3, 0x0F, 0x7F, 0x54, 0x24, 0x40});        // movdqu [rsp+0x40], xmm2
    put({0xF3, 0x0F, 0x7F, 0x5C, 0x24, 0x50});        // movdqu [rsp+0x50], xmm3
    if (arg == 1) put({0x89, 0xD1});                   // mov ecx, edx
    put({0x48, 0xB8}); put64(reinterpret_cast<uint64_t>(fn));         // mov rax, fn
    put({0xFF, 0xD0});                                 // call rax
    put({0xF3, 0x0F, 0x6F, 0x44, 0x24, 0x20});        // movdqu xmm0, [rsp+0x20]
    put({0xF3, 0x0F, 0x6F, 0x4C, 0x24, 0x30});        // movdqu xmm1, [rsp+0x30]
    put({0xF3, 0x0F, 0x6F, 0x54, 0x24, 0x40});        // movdqu xmm2, [rsp+0x40]
    put({0xF3, 0x0F, 0x6F, 0x5C, 0x24, 0x50});        // movdqu xmm3, [rsp+0x50]
    put({0x48, 0x83, 0xC4, 0x68});                    // add rsp, 0x68
    put({0x41, 0x59, 0x41, 0x58, 0x5A, 0x59});        // pop r9, r8, rdx, rcx
    put({0x89, static_cast<uint8_t>(arg == 0 ? 0xC1 : 0xC2)});         // mov ecx|edx, eax (the remapped value)
    put({0x48, 0xB8}); put64(reinterpret_cast<uint64_t>(original));   // mov rax, original
    put({0xFF, 0xE0});                                 // jmp rax
    DWORD old = 0;
    if (!VirtualProtect(t, 4096, PAGE_EXECUTE_READ, &old)) {
        VirtualFree(t, 0, MEM_RELEASE);
        return nullptr;
    }
    FlushInstructionCache(GetCurrentProcess(), t, p - t);
    return t;
}

inline uint8_t* make_remap_thunk(const void* close_to, int arg, RemapFn fn, const void* original) {
    if (!close_to) return nullptr;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(close_to);
    return make_remap_thunk_in_range(addr, addr, arg, fn, original);
}

using ResultFn = uint32_t (*)(uint32_t result, uint32_t arg0);

// A thunk that runs the original and then passes (its 32-bit result, its first argument) through `fn`, returning
// what `fn` returns: for a getter such as "the view mode of context N". Only for functions with at most four
// register arguments (a stack argument would be one slot off) and an integer result.
// Stack: rsp = 8 mod 16 at entry; one push and 0x20 bytes of shadow space make it 16-aligned for both calls.
inline uint8_t* make_result_thunk_in_range(uintptr_t min_next_ip, uintptr_t max_next_ip, ResultFn fn,
                                           const void* original) {
    if (min_next_ip == 0 || max_next_ip < min_next_ip || !fn || !original) return nullptr;
    uint8_t* t = alloc_near_range(min_next_ip, max_next_ip, 4096);
    if (!t) return nullptr;
    uint8_t* p = t;
    auto put = [&](std::initializer_list<uint8_t> bytes) { for (uint8_t b : bytes) *p++ = b; };
    auto put64 = [&](uint64_t v) { std::memcpy(p, &v, 8); p += 8; };
    put({0x51});                                       // push rcx (the first argument, for fn)
    put({0x48, 0x83, 0xEC, 0x20});                     // sub rsp, 0x20
    put({0x48, 0xB8}); put64(reinterpret_cast<uint64_t>(original));   // mov rax, original
    put({0xFF, 0xD0});                                 // call rax (rcx, rdx, r8, r9 still the caller's)
    put({0x48, 0x8B, 0x54, 0x24, 0x20});               // mov rdx, [rsp+0x20] (the saved first argument)
    put({0x89, 0xC1});                                 // mov ecx, eax (the original's result)
    put({0x48, 0xB8}); put64(reinterpret_cast<uint64_t>(fn));         // mov rax, fn
    put({0xFF, 0xD0});                                 // call rax
    put({0x48, 0x83, 0xC4, 0x20});                     // add rsp, 0x20
    put({0x59, 0xC3});                                 // pop rcx ; ret
    DWORD old = 0;
    if (!VirtualProtect(t, 4096, PAGE_EXECUTE_READ, &old)) {
        VirtualFree(t, 0, MEM_RELEASE);
        return nullptr;
    }
    FlushInstructionCache(GetCurrentProcess(), t, p - t);
    return t;
}

inline uint8_t* make_result_thunk(const void* close_to, ResultFn fn, const void* original) {
    if (!close_to) return nullptr;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(close_to);
    return make_result_thunk_in_range(addr, addr, fn, original);
}

namespace detail {
struct PatchedBranch {
    uint8_t* disp_ptr;
    int32_t orig_rel;
};
}  // namespace detail

// Points every E8/E9 rel32 in [begin, begin + size) whose target is `target` at `thunk`. The range must be writable.
// Returns how many were rewritten.
inline std::size_t redirect_direct_branches(uint8_t* begin, std::size_t size, const uint8_t* target,
                                            const uint8_t* thunk,
                                            std::vector<detail::PatchedBranch>* patches = nullptr) {
    if (!begin || !target || !thunk || size < 5) return 0;
    const uintptr_t thunk_addr = reinterpret_cast<uintptr_t>(thunk);
    std::size_t n = 0;
    for (std::size_t i = 0; i + 5 <= size; ++i) {
        if (begin[i] != 0xE8 && begin[i] != 0xE9) continue;
        int32_t rel;
        std::memcpy(&rel, begin + i + 1, 4);
        const uint8_t* next = begin + i + 5;
        const uintptr_t next_addr = reinterpret_cast<uintptr_t>(next);
        const uintptr_t dst_addr = next_addr + static_cast<uintptr_t>(static_cast<intptr_t>(rel));
        if (dst_addr != reinterpret_cast<uintptr_t>(target)) continue;
        if (!rel32_reaches(next_addr, thunk_addr)) continue;
        if (patches) patches->push_back({begin + i + 1, rel});
        const int32_t rel2 = static_cast<int32_t>(static_cast<uint32_t>(thunk_addr - next_addr));
        std::memcpy(begin + i + 1, &rel2, 4);
        ++n;
        i += 4;
    }
    return n;
}

struct HookResult {
    bool ok = false;
    std::size_t redirected = 0;
    const char* error = "";
};

// Hooks the function at `target_rva` of the mapped image at `base`: one thunk, then every direct call or jump to it
// in the image's executable sections. Runs before any image code (main.cpp), so no thread is executing the bytes.
// Validates `target_rva` as code of the mapped image at `base`, builds the thunk with `make` and points the
// image's direct calls and jumps to the target at it.
template <class MakeThunk>
inline HookResult redirect_image_calls(uint8_t* base, uint32_t target_rva, MakeThunk make) {
    HookResult r;
    if (!base || !readable_image_span(base, 0, sizeof(IMAGE_DOS_HEADER)))
        return r.error = "not a mapped PE image", r;
    MEMORY_BASIC_INFORMATION base_mbi{};
    if (VirtualQuery(base, &base_mbi, sizeof(base_mbi)) == 0 || !base_mbi.AllocationBase)
        return r.error = "not a mapped PE image", r;
    const void* const image_alloc_base = base_mbi.AllocationBase;

    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < static_cast<LONG>(sizeof(IMAGE_DOS_HEADER)))
        return r.error = "not a mapped PE image", r;
    const uint64_t lfanew = static_cast<uint64_t>(dos->e_lfanew);
    if (lfanew > UINT64_MAX - sizeof(IMAGE_NT_HEADERS64) ||
        !readable_image_span(base, lfanew, sizeof(IMAGE_NT_HEADERS64)))
        return r.error = "not a mapped PE64 image", r;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64) ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return r.error = "not a mapped PE64 image", r;
    const uint64_t image_size = nt->OptionalHeader.SizeOfImage;
    const uint64_t opt_end =
        lfanew + offsetof(IMAGE_NT_HEADERS64, OptionalHeader) +
        static_cast<uint64_t>(nt->FileHeader.SizeOfOptionalHeader);
    const uint64_t sec_bytes =
        static_cast<uint64_t>(nt->FileHeader.NumberOfSections) * sizeof(IMAGE_SECTION_HEADER);
    if (image_size == 0 || opt_end > image_size || sec_bytes > image_size - opt_end)
        return r.error = "not a mapped PE64 image", r;
    const uint64_t sec_table_end = opt_end + sec_bytes;
    if (!readable_image_span(base, lfanew, sec_table_end - lfanew))
        return r.error = "not a mapped PE64 image", r;
    if (target_rva >= image_size) return r.error = "the RVA is outside the image", r;

    const IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
    // Match pe_loader.cpp's effective section span (VirtualSize, falling back to SizeOfRawData when 0), clamped to SizeOfImage.
    auto exec_section_span = [&](const IMAGE_SECTION_HEADER& s, uint64_t& va, uint64_t& len) -> bool {
        if (!(s.Characteristics & IMAGE_SCN_MEM_EXECUTE)) return false;
        va = s.VirtualAddress;
        const uint64_t raw = s.Misc.VirtualSize ? static_cast<uint64_t>(s.Misc.VirtualSize)
                                                : static_cast<uint64_t>(s.SizeOfRawData);
        if (raw == 0 || va >= image_size) return false;
        len = raw > image_size - va ? image_size - va : raw;
        return len > 0;
    };

    bool in_code = false;
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        uint64_t va = 0, len = 0;
        if (exec_section_span(sec[i], va, len) && target_rva >= va && target_rva < va + len)
            in_code = true;
    }
    if (!in_code) return r.error = "the RVA is not in an executable section", r;

    bool have_branch_range = false;
    uintptr_t min_next_ip = 0;
    uintptr_t max_next_ip = 0;
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        uint64_t va = 0, len = 0;
        if (!exec_section_span(sec[i], va, len)) continue;
        if (!readable_image_span(base, va, len))
            return r.error = "executable section is not mapped", r;
        if (len >= 5) {
            const uintptr_t sec_first_next = reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(va) + 5u;
            const uintptr_t sec_last_next = reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(va + len);
            if (!have_branch_range) {
                min_next_ip = sec_first_next;
                max_next_ip = sec_last_next;
                have_branch_range = true;
            } else {
                if (sec_first_next < min_next_ip) min_next_ip = sec_first_next;
                if (sec_last_next > max_next_ip) max_next_ip = sec_last_next;
            }
        }
    }

    uint8_t* target = base + target_rva;
    if (!have_branch_range) {
        min_next_ip = reinterpret_cast<uintptr_t>(target);
        max_next_ip = min_next_ip;
    }

    auto free_thunk = [&](uint8_t* t) {
        if (t && reinterpret_cast<const void*>(t) != image_alloc_base) VirtualFree(t, 0, MEM_RELEASE);
    };

    uint8_t* thunk = nullptr;
    if constexpr (std::is_invocable_v<MakeThunk, uint8_t*, uintptr_t, uintptr_t>) {
        thunk = make(target, min_next_ip, max_next_ip);
    } else {
        thunk = make(target);
    }
    if (!thunk) return r.error = "no memory within 2 GiB of the image for the thunk", r;
    const uintptr_t thunk_addr = reinterpret_cast<uintptr_t>(thunk);
    if (!rel32_reaches(min_next_ip, thunk_addr) || !rel32_reaches(max_next_ip, thunk_addr)) {
        free_thunk(thunk);
        return r.error = "no memory within 2 GiB of the image for the thunk", r;
    }

    struct TouchedSection {
        uint8_t* begin;
        std::size_t size;
        std::size_t first_patch;
        std::size_t patch_count;
    };
    std::vector<detail::PatchedBranch> patches;
    std::vector<TouchedSection> touched;

    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        uint64_t va = 0, len = 0;
        if (!exec_section_span(sec[i], va, len)) continue;
        uint8_t* begin = base + va;
        const std::size_t size = static_cast<std::size_t>(len);
        DWORD old = 0;
        if (!VirtualProtect(begin, size, PAGE_EXECUTE_READWRITE, &old)) {
            bool rollback_ok = true;
            for (const auto& ts : touched) {
                DWORD rb_old = 0;
                if (!VirtualProtect(ts.begin, ts.size, PAGE_EXECUTE_READWRITE, &rb_old)) {
                    rollback_ok = false;
                    continue;
                }
                for (std::size_t k = 0; k < ts.patch_count; ++k) {
                    const auto& pb = patches[ts.first_patch + k];
                    std::memcpy(pb.disp_ptr, &pb.orig_rel, 4);
                }
                DWORD rb_ignored = 0;
                VirtualProtect(ts.begin, ts.size, rb_old, &rb_ignored);
                FlushInstructionCache(GetCurrentProcess(), ts.begin, ts.size);
            }
            if (rollback_ok) free_thunk(thunk);
            r.redirected = 0;
            return r.error = "VirtualProtect failed", r;
        }
        const std::size_t first_patch = patches.size();
        const std::size_t count = redirect_direct_branches(begin, size, target, thunk, &patches);
        if (count > 0) touched.push_back({begin, size, first_patch, count});
        r.redirected += count;
        DWORD ignored = 0;
        VirtualProtect(begin, size, old, &ignored);
        FlushInstructionCache(GetCurrentProcess(), begin, size);
    }

    if (r.redirected == 0) free_thunk(thunk);
    r.ok = true;
    return r;
}

// The argument-remapping hook (archetype lookup): argument `arg` of every direct call goes through `fn`.
inline HookResult hook_image_calls(uint8_t* base, uint32_t target_rva, RemapFn fn, int arg) {
    if (!fn) return HookResult{false, 0, "hook callback is null"};
    if (arg != 0 && arg != 1) return HookResult{false, 0, "only argument 0 or 1 can be remapped"};
    return redirect_image_calls(
        base, target_rva, [&](uint8_t* target, uintptr_t min_next_ip, uintptr_t max_next_ip) {
            return make_remap_thunk_in_range(min_next_ip, max_next_ip, arg, fn, target);
        });
}

// The result-override hook (view-mode getter): every direct call's result goes through `fn` with the first argument.
inline HookResult hook_image_results(uint8_t* base, uint32_t target_rva, ResultFn fn) {
    if (!fn) return HookResult{false, 0, "hook callback is null"};
    return redirect_image_calls(
        base, target_rva, [&](uint8_t* target, uintptr_t min_next_ip, uintptr_t max_next_ip) {
            return make_result_thunk_in_range(min_next_ip, max_next_ip, fn, target);
        });
}

}  // namespace leonida::hooks
