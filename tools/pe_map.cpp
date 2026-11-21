// Map an x64 PE image without LoadLibrary and bind its imports through a host resolver.
// Scope: DIR64 relocations, ordinary imports by name or ordinal, section protections, .pdata
// registration, TLS callbacks and the entry point. TLS data templates and delay imports are rejected.
// Header fields are checked against the file or SizeOfImage before they are used.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// Keep in step with STATUS in tests/cpu/native_map_check.py.
enum MapStatus : std::uint32_t {
    MAP_OK = 0,
    MAP_IO = 1,
    MAP_FORMAT = 2,
    MAP_MEMORY = 3,
    MAP_RELOCATION = 4,
    MAP_UNRESOLVED_IMPORT = 5,
    MAP_UNSUPPORTED = 6,
    MAP_PROTECT = 7,
    MAP_FUNCTION_TABLE = 8,
    MAP_INIT_FAILED = 9,
};

// Returns the address bound to one import, or null to reject it. name is null for an ordinal.
using Resolver = void*(__cdecl*)(const char* dll, const char* name, std::uint32_t ordinal);

namespace {

constexpr std::uint64_t MAX_FILE = 1ull << 30;

struct Failure {
    std::uint32_t* status;
    char* detail;
    std::size_t capacity;

    template <typename... Args>
    void* operator()(MapStatus code, const char* format, Args... args) const {
        if (status) *status = code;
        if (detail && capacity) std::snprintf(detail, capacity, format, args...);
        return nullptr;
    }
};

bool inside(std::uint64_t offset, std::uint64_t size, std::uint64_t limit) {
    return offset <= limit && size <= limit - offset;
}

const IMAGE_DATA_DIRECTORY& directory(const IMAGE_NT_HEADERS64* nt, int index) {
    static const IMAGE_DATA_DIRECTORY empty{};
    return nt->OptionalHeader.NumberOfRvaAndSizes > static_cast<DWORD>(index)
        ? nt->OptionalHeader.DataDirectory[index] : empty;
}

// A view of the mapped image that rejects any range outside SizeOfImage.
struct Image {
    std::uint8_t* base;
    std::uint64_t size;

    template <typename T>
    T* at(std::uint64_t rva, std::uint64_t count = 1) const {
        return inside(rva, sizeof(T) * count, size) ? reinterpret_cast<T*>(base + rva) : nullptr;
    }

    const char* string(std::uint64_t rva) const {
        if (rva >= size) return nullptr;
        const auto* start = base + rva;
        const bool terminated = std::memchr(start, 0, static_cast<std::size_t>(size - rva)) != nullptr;
        return terminated ? reinterpret_cast<const char*>(start) : nullptr;
    }

    const IMAGE_NT_HEADERS64* headers() const {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        return reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    }
};

DWORD protection(DWORD flags) {
    const bool exec = flags & IMAGE_SCN_MEM_EXECUTE, write = flags & IMAGE_SCN_MEM_WRITE;
    const bool read = flags & IMAGE_SCN_MEM_READ;
    if (exec) return write ? PAGE_EXECUTE_READWRITE : read ? PAGE_EXECUTE_READ : PAGE_EXECUTE;
    return write ? PAGE_READWRITE : read ? PAGE_READONLY : PAGE_NOACCESS;
}

// True if rva lies inside a section mapped as executable.
bool executable(const IMAGE_NT_HEADERS64* nt, std::uint64_t rva) {
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if ((s.Characteristics & IMAGE_SCN_MEM_EXECUTE) && rva >= s.VirtualAddress && rva - s.VirtualAddress < span)
            return true;
    }
    return false;
}

// True if every page of [address, address + size) is committed and currently allows reads.
// Image bounds alone do not make bytes readable: section protections apply, and the image may
// change them at run time, so the live protection is queried at each read, including on release.
// The query is not atomic with the read that follows: another thread reprotecting the same pages
// in between is not handled.
bool readable(const void* address, std::size_t size) {
    auto* cursor = static_cast<const std::uint8_t*>(address);
    const auto* end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cursor, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
        const DWORD access = info.Protect & 0xff;
        if ((info.Protect & PAGE_GUARD) || access == PAGE_NOACCESS || access == PAGE_EXECUTE) return false;
        cursor = static_cast<const std::uint8_t*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

constexpr std::uint64_t MAX_TLS_CALLBACKS = 256;
constexpr const char* TOO_MANY_TLS_CALLBACKS = "more than 256 TLS callbacks";

// TLS callbacks of a mapped, relocated image in table order. Returns null or the reason the list is rejected.
const char* tls_callbacks(const Image& image, const IMAGE_NT_HEADERS64* nt, std::vector<PIMAGE_TLS_CALLBACK>& out) {
    const auto& dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (!dir.Size) return nullptr;
    const auto* tls = image.at<IMAGE_TLS_DIRECTORY64>(dir.VirtualAddress);
    if (!tls) return "TLS directory outside image";
    // Sections are already protected here, both after mapping and on release.
    if (!readable(tls, sizeof(IMAGE_TLS_DIRECTORY64))) return "TLS directory not readable";
    // Each value is read once after its check, so the validated value is the one used.
    const std::uint64_t list = tls->AddressOfCallBacks;
    if (!list) return nullptr;
    const auto base = reinterpret_cast<std::uint64_t>(image.base);
    for (std::uint64_t i = 0; i <= MAX_TLS_CALLBACKS; ++i) {
        const std::uint64_t va = list + 8 * i;
        const auto* slot = va >= base ? image.at<std::uint64_t>(va - base) : nullptr;
        if (!slot) return "TLS callback list outside image";
        if (!readable(slot, sizeof(std::uint64_t))) return "TLS callback list not readable";
        const std::uint64_t callback = *slot;
        if (!callback) return nullptr;
        if (i == MAX_TLS_CALLBACKS) return TOO_MANY_TLS_CALLBACKS;
        if (callback < base || callback - base >= image.size || !executable(nt, callback - base))
            return "TLS callback outside executable section";
        out.push_back(reinterpret_cast<PIMAGE_TLS_CALLBACK>(callback));
    }
    return nullptr;
}

// Notify the image as the Windows loader does for LoadLibrary/FreeLibrary: TLS callbacks in table
// order, then the entry point, with a null reserved argument. Returns the entry point's result.
// Measured on Windows 11 build 26300: LoadLibrary sends nothing to a non-DLL image, neither TLS
// callbacks nor the entry point (for an EXE the entry point is the program itself).
BOOL notify(const Image& image, const IMAGE_NT_HEADERS64* nt, const std::vector<PIMAGE_TLS_CALLBACK>& callbacks,
            DWORD reason) {
    if (!(nt->FileHeader.Characteristics & IMAGE_FILE_DLL)) return TRUE;
    for (auto callback : callbacks) callback(image.base, reason, nullptr);
    if (!nt->OptionalHeader.AddressOfEntryPoint) return TRUE;
    using Entry = BOOL(WINAPI*)(HINSTANCE, DWORD, LPVOID);
    const auto entry = reinterpret_cast<Entry>(image.base + nt->OptionalHeader.AddressOfEntryPoint);
    return entry(reinterpret_cast<HINSTANCE>(image.base), reason, nullptr);
}

bool read_file(const wchar_t* path, std::vector<std::uint8_t>& raw) {
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER length{};
    DWORD read = 0;
    bool ok = GetFileSizeEx(file, &length) && length.QuadPart > 0
        && static_cast<std::uint64_t>(length.QuadPart) <= MAX_FILE;
    if (ok) {
        try {
            raw.resize(static_cast<std::size_t>(length.QuadPart));
        } catch (const std::bad_alloc&) {
            ok = false;
        }
    }
    ok = ok && ReadFile(file, raw.data(), static_cast<DWORD>(raw.size()), &read, nullptr) && read == raw.size();
    CloseHandle(file);
    return ok;
}

}  // namespace

extern "C" __declspec(dllexport) void* pe_map_image(const wchar_t* path, Resolver resolve,
                                                    std::uint32_t* status, char* detail,
                                                    std::size_t capacity) {
    const Failure fail{status, detail, capacity};
    if (status) *status = MAP_OK;
    if (detail && capacity) detail[0] = '\0';

    std::vector<std::uint8_t> raw;
    if (!read_file(path, raw)) return fail(MAP_IO, "cannot read image");
    if (raw.size() < sizeof(IMAGE_DOS_HEADER)) return fail(MAP_FORMAT, "truncated DOS header");
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(raw.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0
        || !inside(dos->e_lfanew, sizeof(IMAGE_NT_HEADERS64), raw.size()))
        return fail(MAP_FORMAT, "bad DOS header");
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(raw.data() + dos->e_lfanew);
    const auto& opt = nt->OptionalHeader;
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64
        || opt.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return fail(MAP_FORMAT, "not an x64 PE32+ image");
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    const std::uint64_t table_end =
        reinterpret_cast<const std::uint8_t*>(sections + nt->FileHeader.NumberOfSections) - raw.data();
    if (nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64)
        || dos->e_lfanew + sizeof(IMAGE_NT_HEADERS64) > opt.SizeOfHeaders || table_end > opt.SizeOfHeaders
        || opt.SizeOfHeaders > opt.SizeOfImage || opt.SizeOfHeaders > raw.size())
        return fail(MAP_FORMAT, "section table outside headers");

    if (directory(nt, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT).Size) return fail(MAP_UNSUPPORTED, "delay imports");
    if (opt.AddressOfEntryPoint && !executable(nt, opt.AddressOfEntryPoint))
        return fail(MAP_FORMAT, "entry point outside executable section");

    // Map away from the preferred base so the relocation path always runs.
    auto allocate = [&] {
        return static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, opt.SizeOfImage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    };
    auto* base = allocate();
    if (base == reinterpret_cast<std::uint8_t*>(opt.ImageBase)) {
        auto* other = allocate();
        VirtualFree(base, 0, MEM_RELEASE);
        base = other;
    }
    if (!base) return fail(MAP_MEMORY, "cannot reserve image");
    const Image image{base, opt.SizeOfImage};
    auto release = [&](void* result) {
        VirtualFree(base, 0, MEM_RELEASE);
        return result;
    };

    std::memcpy(base, raw.data(), opt.SizeOfHeaders);
    const WORD section_count = nt->FileHeader.NumberOfSections;
    for (WORD i = 0; i < section_count; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        const char* name = reinterpret_cast<const char*>(s.Name);
        if (s.VirtualAddress < opt.SizeOfHeaders || !inside(s.VirtualAddress, span, opt.SizeOfImage))
            return release(fail(MAP_FORMAT, "section %.8s outside image", name));
        // Like the Windows loader, check raw data against the file by two rules only: no raw range
        // may reach 4 GiB, and the last section's raw data must end inside the file. Raw data of an
        // earlier section may lie past the end of the file; the missing bytes stay zero.
        const std::uint64_t raw_end = std::uint64_t{s.PointerToRawData} + s.SizeOfRawData;
        if (s.SizeOfRawData && raw_end > 0xffffffffull)
            return release(fail(MAP_FORMAT, "section %.8s raw data reaches 4 GiB", name));
        if (s.SizeOfRawData && i + 1 == section_count && raw_end > raw.size())
            return release(fail(MAP_FORMAT, "last section %.8s raw data runs past the file", name));
        // A section without a file offset stays zero-filled even when SizeOfRawData is set; it does
        // not take the headers at file offset 0.
        if (!s.PointerToRawData || !s.SizeOfRawData) continue;
        // Read whole 0x200-byte file sectors: round the start of the raw data down and its end up.
        const std::uint64_t first = s.PointerToRawData & ~std::uint64_t{0x1ff};
        std::uint64_t end = (raw_end + 0x1ff) & ~std::uint64_t{0x1ff};
        if (end > raw.size()) end = raw.size();
        if (first >= end) continue;
        const std::uint64_t length = end - first < span ? end - first : span;
        std::memcpy(base + s.VirtualAddress, raw.data() + first, static_cast<std::size_t>(length));
    }

    // TLS data needs a slot in each thread's ThreadLocalStoragePointer array, which ntdll owns.
    // Only an empty template is accepted. Unlike Windows, which assigns an index even for an
    // empty template, pe_map leaves AddressOfIndex as linked.
    const auto& tls = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (tls.Size) {
        const auto* t = tls.Size >= sizeof(IMAGE_TLS_DIRECTORY64)
            ? image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress) : nullptr;
        if (!t) return release(fail(MAP_FORMAT, "TLS directory outside image"));
        if (t->EndAddressOfRawData != t->StartAddressOfRawData || t->SizeOfZeroFill)
            return release(fail(MAP_UNSUPPORTED, "TLS data"));
    }

    // Writes below (relocations, IAT) must land in sections, never in the headers or in the
    // directory that drives them, so validated structures cannot change while they are walked.
    auto writable = [&](std::uint64_t rva, const IMAGE_DATA_DIRECTORY& driver) {
        const std::uint64_t start = driver.VirtualAddress, end = start + driver.Size;
        const bool overlaps = rva + 8 > start && rva < end;
        return rva >= opt.SizeOfHeaders && image.at<std::uint64_t>(rva) && !overlaps;
    };

    const auto delta = reinterpret_cast<std::uint64_t>(base) - opt.ImageBase;
    const auto& relocs = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
    if (nt->FileHeader.Characteristics & IMAGE_FILE_RELOCS_STRIPPED)
        return release(fail(MAP_RELOCATION, "relocations stripped"));
    if (!inside(relocs.VirtualAddress, relocs.Size, opt.SizeOfImage))
        return release(fail(MAP_RELOCATION, "relocation directory outside image"));
    for (std::uint64_t offset = 0; offset + sizeof(IMAGE_BASE_RELOCATION) <= relocs.Size;) {
        const auto* block = image.at<IMAGE_BASE_RELOCATION>(relocs.VirtualAddress + offset);
        const DWORD page = block->VirtualAddress, length = block->SizeOfBlock;
        if (length < sizeof(IMAGE_BASE_RELOCATION) || length > relocs.Size - offset)
            return release(fail(MAP_RELOCATION, "bad relocation block"));
        const auto* entry = reinterpret_cast<const WORD*>(block + 1);
        const DWORD count = (length - sizeof(IMAGE_BASE_RELOCATION)) / sizeof(WORD);
        for (DWORD k = 0; k < count; ++k) {
            const WORD type = entry[k] >> 12;
            const std::uint64_t target = page + std::uint64_t{entry[k] & 0xfffu};
            if (type == IMAGE_REL_BASED_ABSOLUTE) continue;
            if (type != IMAGE_REL_BASED_DIR64 || !writable(target, relocs))
                return release(fail(MAP_RELOCATION, "unsupported relocation at RVA %#llx",
                                    static_cast<unsigned long long>(target)));
            *image.at<std::uint64_t>(target) += delta;
        }
        offset += length;
    }

    const auto& imports = directory(nt, IMAGE_DIRECTORY_ENTRY_IMPORT);
    if (imports.Size) {
        const auto descriptors = imports.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR);
        if (!descriptors) return release(fail(MAP_FORMAT, "import directory smaller than one descriptor"));
        if (!resolve) return release(fail(MAP_UNRESOLVED_IMPORT, "imports present without a resolver"));
        const auto* desc = image.at<IMAGE_IMPORT_DESCRIPTOR>(imports.VirtualAddress, descriptors);
        if (!desc) return release(fail(MAP_FORMAT, "import directory outside image"));
        for (std::uint64_t d = 0; d < descriptors && desc[d].Name; ++d) {
            const char* name_in_image = image.string(desc[d].Name);
            if (!name_in_image) return release(fail(MAP_FORMAT, "import DLL name outside image"));
            // Keep a copy: an IAT write below may legally overlap the string in the image.
            const std::string dll_name(name_in_image);
            const char* dll = dll_name.c_str();
            const DWORD lookup = desc[d].OriginalFirstThunk ? desc[d].OriginalFirstThunk : desc[d].FirstThunk;
            for (std::uint64_t i = 0;; ++i) {
                const auto* word = image.at<std::uint64_t>(lookup + 8 * i);
                if (!word || !writable(desc[d].FirstThunk + 8 * i, imports))
                    return release(fail(MAP_FORMAT, "import thunk for %s outside image", dll));
                if (!*word) break;
                const bool by_ordinal = IMAGE_SNAP_BY_ORDINAL64(*word);
                const auto ordinal = by_ordinal ? static_cast<std::uint32_t>(IMAGE_ORDINAL64(*word)) : 0u;
                const auto* hint = by_ordinal ? nullptr : image.at<WORD>(*word & 0x7fffffff);
                const char* name = hint ? image.string((*word & 0x7fffffff) + sizeof(WORD)) : nullptr;
                if (!by_ordinal && !name) return release(fail(MAP_FORMAT, "import name for %s outside image", dll));
                void* bound = resolve(dll, name, ordinal);
                if (!bound && name) return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!%s", dll, name));
                if (!bound)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!#%u", dll,
                                        static_cast<unsigned>(ordinal)));
                *image.at<std::uint64_t>(desc[d].FirstThunk + 8 * i) = reinterpret_cast<std::uint64_t>(bound);
            }
        }
    }

    // Record the actual base in the mapped headers, as the Windows loader does.
    reinterpret_cast<IMAGE_NT_HEADERS64*>(base + dos->e_lfanew)->OptionalHeader.ImageBase =
        reinterpret_cast<std::uint64_t>(base);
    DWORD old = 0;
    if (!VirtualProtect(base, opt.SizeOfHeaders, PAGE_READONLY, &old))
        return release(fail(MAP_PROTECT, "cannot protect headers"));
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if (span && !VirtualProtect(base + s.VirtualAddress, span, protection(s.Characteristics), &old))
            return release(fail(MAP_PROTECT, "cannot protect section %.8s", reinterpret_cast<const char*>(s.Name)));
    }
    FlushInstructionCache(GetCurrentProcess(), base, opt.SizeOfImage);

    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) {
        auto* table = image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress, pdata.Size / sizeof(RUNTIME_FUNCTION));
        if (!table || !RtlAddFunctionTable(table, pdata.Size / sizeof(RUNTIME_FUNCTION),
                                           reinterpret_cast<DWORD64>(base)))
            return release(fail(MAP_FUNCTION_TABLE, "cannot register .pdata"));
    }

    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    const auto* mapped_nt = image.headers();
    if (const char* reason = tls_callbacks(image, mapped_nt, callbacks)) {
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        const bool limit = reason == TOO_MANY_TLS_CALLBACKS;
        return release(fail(limit ? MAP_UNSUPPORTED : MAP_FORMAT, "%s", reason));
    }
    if (!notify(image, mapped_nt, callbacks, DLL_PROCESS_ATTACH)) {
        // Measured on Windows: a failed attach is followed by a full detach before the image is freed.
        // The callback list collected before attach is reused here; only pe_map_release re-reads it.
        notify(image, mapped_nt, callbacks, DLL_PROCESS_DETACH);
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        return release(fail(MAP_INIT_FAILED, "entry point returned FALSE"));
    }
    return base;
}

// Only valid for a base returned by pe_map_image, whose headers were validated and are read-only.
extern "C" __declspec(dllexport) void* pe_map_export(void* mapped, const char* name) {
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    const auto& exports = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    const auto* dir = exports.Size ? image.at<IMAGE_EXPORT_DIRECTORY>(exports.VirtualAddress) : nullptr;
    if (!dir) return nullptr;
    const auto* names = image.at<DWORD>(dir->AddressOfNames, dir->NumberOfNames);
    const auto* ordinals = image.at<WORD>(dir->AddressOfNameOrdinals, dir->NumberOfNames);
    const auto* functions = image.at<DWORD>(dir->AddressOfFunctions, dir->NumberOfFunctions);
    if (!names || !ordinals || !functions) return nullptr;
    for (DWORD i = 0; i < dir->NumberOfNames; ++i) {
        const char* exported = image.string(names[i]);
        if (!exported || std::strcmp(exported, name) != 0) continue;
        if (ordinals[i] >= dir->NumberOfFunctions) return nullptr;
        const DWORD rva = functions[ordinals[i]];
        // Forwarded exports point back into the export directory; they are not code here.
        if (rva >= exports.VirtualAddress && rva - exports.VirtualAddress < exports.Size) return nullptr;
        // An empty slot (RVA 0) names no code; returning the image base would be wrong.
        return rva && rva < image.size ? image.base + rva : nullptr;
    }
    return nullptr;
}

extern "C" __declspec(dllexport) void pe_map_release(void* mapped) {
    auto* base = static_cast<std::uint8_t*>(mapped);
    const auto* nt = Image{base, 0}.headers();
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    // The list is re-read, as Windows does; if it was overwritten or made unreadable,
    // the entry point is still notified.
    if (tls_callbacks(image, nt, callbacks)) callbacks.clear();
    notify(image, nt, callbacks, DLL_PROCESS_DETACH);
    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(base + pdata.VirtualAddress));
    VirtualFree(base, 0, MEM_RELEASE);
}
