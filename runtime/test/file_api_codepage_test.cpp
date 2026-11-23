// Filename inputs use the process's ANSI/OEM mode. Returned names are compared directly with native APIs:
// Windows 11's FindFirst/NextFileA still returns ANSI names after SetFileApisToOEM.
// Compare the actual file shim registry with native Windows over original synthetic files.
// No game data. A code-page pair with no distinct round-tripping filename character skips (77).
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "file_system.h"
#include "shims.h"

namespace {
int checks = 0, failures = 0;
std::string context;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::printf("FAIL [%s] %s\n", context.c_str(), message);
    }
}

// Restore process-wide state on every normal return and exception path.
struct ProcessState {
    bool ansi = ::AreFileApisANSI() != FALSE;
    std::filesystem::path directory = std::filesystem::current_path();
    ~ProcessState() {
        if (ansi) ::SetFileApisToANSI(); else ::SetFileApisToOEM();
        ::SetCurrentDirectoryW(directory.c_str());
        rt::fs::configure({}, {});
    }
};

std::string encode(const std::wstring& text, UINT code_page) {
    const int count = ::WideCharToMultiByte(code_page, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (!count) return {};
    std::string bytes(static_cast<size_t>(count), '\0');
    if (!::WideCharToMultiByte(code_page, 0, text.c_str(), -1, bytes.data(), count, nullptr, nullptr)) return {};
    bytes.pop_back();
    return bytes;
}

bool round_trips(wchar_t character, UINT code_page, std::string& bytes) {
    bytes = encode(std::wstring(1, character), code_page);
    wchar_t decoded[3]{};
    return !bytes.empty() &&
           ::MultiByteToWideChar(code_page, 0, bytes.c_str(), -1, decoded, 3) == 2 &&
           decoded[0] == character;
}

wchar_t distinct_character() {
    if (::GetACP() == ::GetOEMCP()) return 0;
    // A real character on both detected pages, rather than a fixture tied to CP1252/CP437 or a best-fit '?'.
    for (unsigned value = 0xa0; value < 0xd800; ++value) {
        std::string ansi, oem;
        if (round_trips(static_cast<wchar_t>(value), ::GetACP(), ansi) &&
            round_trips(static_cast<wchar_t>(value), ::GetOEMCP(), oem) && ansi != oem)
            return static_cast<wchar_t>(value);
    }
    return 0;
}

template<class Fn>
Fn resolve(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("kernel32.dll", name));
}

struct Api {
    decltype(&::CreateFileA) create;
    decltype(&::GetFileAttributesA) attributes;
    decltype(&::GetFileAttributesExA) attributes_ex;
    decltype(&::FindFirstFileA) first;
    decltype(&::FindFirstFileExA) first_ex;
    decltype(&::FindFirstFileW) first_w;
    decltype(&::FindNextFileA) next;
    decltype(&::FindClose) close;
    decltype(&::DeleteFileA) remove;
    decltype(&::CreateDirectoryA) mkdir;
    decltype(&::CreateDirectoryExA) mkdir_ex;
    decltype(&::RemoveDirectoryA) rmdir;
    decltype(&::MoveFileA) move;
    decltype(&::MoveFileExA) move_ex;
    decltype(&::CopyFileA) copy;
    decltype(&::CopyFileExA) copy_ex;
    decltype(&::SetFileAttributesA) set_attributes;
};

const Api native{::CreateFileA, ::GetFileAttributesA, ::GetFileAttributesExA, ::FindFirstFileA, ::FindFirstFileExA,
                 ::FindFirstFileW, ::FindNextFileA, ::FindClose, ::DeleteFileA, ::CreateDirectoryA,
                 ::CreateDirectoryExA, ::RemoveDirectoryA, ::MoveFileA, ::MoveFileExA, ::CopyFileA,
                 ::CopyFileExA, ::SetFileAttributesA};

Api mapped_api() {
    return {resolve<decltype(native.create)>("CreateFileA"), resolve<decltype(native.attributes)>("GetFileAttributesA"),
            resolve<decltype(native.attributes_ex)>("GetFileAttributesExA"), resolve<decltype(native.first)>("FindFirstFileA"),
            resolve<decltype(native.first_ex)>("FindFirstFileExA"), resolve<decltype(native.first_w)>("FindFirstFileW"),
            resolve<decltype(native.next)>("FindNextFileA"), resolve<decltype(native.close)>("FindClose"),
            resolve<decltype(native.remove)>("DeleteFileA"), resolve<decltype(native.mkdir)>("CreateDirectoryA"),
            resolve<decltype(native.mkdir_ex)>("CreateDirectoryExA"), resolve<decltype(native.rmdir)>("RemoveDirectoryA"),
            resolve<decltype(native.move)>("MoveFileA"), resolve<decltype(native.move_ex)>("MoveFileExA"),
            resolve<decltype(native.copy)>("CopyFileA"), resolve<decltype(native.copy_ex)>("CopyFileExA"),
            resolve<decltype(native.set_attributes)>("SetFileAttributesA")};
}

void write_file(const std::filesystem::path& path, const char* bytes = "original") {
    std::filesystem::create_directories(path.parent_path());
    const HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(h != INVALID_HANDLE_VALUE, "fixture file created");
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    check(::WriteFile(h, bytes, static_cast<DWORD>(std::strlen(bytes)), &written, nullptr) &&
              written == std::strlen(bytes), "fixture bytes written");
    ::CloseHandle(h);
}

std::string read_file(const std::filesystem::path& path) {
    const HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                   OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {};
    char bytes[32]{};
    DWORD count = 0;
    ::ReadFile(h, bytes, sizeof(bytes), &count, nullptr);
    ::CloseHandle(h);
    return std::string(bytes, count);
}

void compare_success(BOOL actual_native, BOOL actual_mapped, const char* api) {
    check(actual_native != FALSE, api);  // a failed native setup must not make equal failures pass
    check(actual_mapped == actual_native, api);
}

std::vector<std::string> list_after_wide_first(const Api& api, const std::wstring& pattern) {
    WIN32_FIND_DATAW first{};
    const HANDLE h = api.first_w(pattern.c_str(), &first);
    check(h != INVALID_HANDLE_VALUE, "wide enumeration control starts");
    if (h == INVALID_HANDLE_VALUE) return {};
    std::vector<std::string> names;
    WIN32_FIND_DATAA next{};
    while (api.next(h, &next)) names.emplace_back(next.cFileName);
    check(::GetLastError() == ERROR_NO_MORE_FILES, "narrow enumeration reaches its end");
    api.close(h);
    std::sort(names.begin(), names.end());
    return names;
}

void exercise(const Api& mapped, const std::filesystem::path& root, wchar_t character, int pass, bool console) {
    const bool oem = pass == 1;
    if (oem) ::SetFileApisToOEM(); else ::SetFileApisToANSI();
    const UINT cp = oem ? ::GetOEMCP() : ::GetACP();
    context = std::string(oem ? "OEM" : "ANSI") + (console ? " console" : " host") + " pass " + std::to_string(pass);
    check((::AreFileApisANSI() == FALSE) == oem, "native file API mode selected");

    const std::wstring sub = L"case" + std::to_wstring(pass) + (console ? L"-console" : L"-host");
    const auto oracle = root / sub / L"native";
    const auto package = root / sub / L"package";
    const auto overlay = root / sub / L"overlay";
    const std::wstring marker(1, character);
    const auto leaf = [&](const wchar_t* stem) { return std::wstring(stem) + marker + L".bin"; };
    for (const auto& folder : {oracle, package}) {
        for (const auto* stem : {L"read", L"write", L"attrs", L"delete", L"move", L"moveex", L"list0", L"list1", L"list2"})
            write_file(folder / leaf(stem));
        std::filesystem::create_directories(folder / leaf(L"template"));
        std::filesystem::create_directories(folder / leaf(L"rmdir"));
        ::SetFileAttributesW((folder / leaf(L"template")).c_str(), FILE_ATTRIBUTE_HIDDEN);
    }
    rt::fs::configure(package, overlay);
    // Relative host paths keep this test independent of how the user's TEMP folder is encoded.
    const std::wstring native_base = sub + L"\\native\\";
    const std::wstring mapped_base = console ? L"GAME:\\" : sub + L"\\package\\";
    const auto n = [&](const wchar_t* stem) { return encode(native_base + leaf(stem), cp); };
    const auto m = [&](const wchar_t* stem) { return encode(mapped_base + leaf(stem), cp); };
    const auto changed = [&](const wchar_t* stem) { return (console ? overlay : package) / leaf(stem); };

    HANDLE hn = native.create(n(L"read").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    HANDLE hm = mapped.create(m(L"read").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    check(hn != INVALID_HANDLE_VALUE, "native CreateFileA read");
    check(hm != INVALID_HANDLE_VALUE, "mapped CreateFileA read");
    char bn[16]{}, bm[16]{};
    DWORD nn = 0, nm = 0;
    if (hn != INVALID_HANDLE_VALUE) { ::ReadFile(hn, bn, sizeof(bn), &nn, nullptr); ::CloseHandle(hn); }
    if (hm != INVALID_HANDLE_VALUE) { ::ReadFile(hm, bm, sizeof(bm), &nm, nullptr); ::CloseHandle(hm); }
    check(nn == 8 && nm == nn && std::memcmp(bn, bm, nn) == 0, "CreateFileA reads the same bytes");
    check(!console || !std::filesystem::exists(overlay / leaf(L"read")), "read does not create an overlay copy");

    const DWORD an = native.attributes(n(L"read").c_str()), am = mapped.attributes(m(L"read").c_str());
    check(an != INVALID_FILE_ATTRIBUTES && an == am, "GetFileAttributesA matches native");
    WIN32_FILE_ATTRIBUTE_DATA dn{}, dm{};
    compare_success(native.attributes_ex(n(L"read").c_str(), GetFileExInfoStandard, &dn),
                    mapped.attributes_ex(m(L"read").c_str(), GetFileExInfoStandard, &dm), "GetFileAttributesExA");
    check(dn.dwFileAttributes == dm.dwFileAttributes && dn.nFileSizeHigh == dm.nFileSizeHigh &&
              dn.nFileSizeLow == dm.nFileSizeLow, "GetFileAttributesExA reports the same attributes and size");

    for (bool extended : {false, true}) {
        WIN32_FIND_DATAA fn{}, fm{};
        hn = extended ? native.first_ex(n(L"read").c_str(), FindExInfoStandard, &fn, FindExSearchNameMatch, nullptr, 0)
                      : native.first(n(L"read").c_str(), &fn);
        hm = extended ? mapped.first_ex(m(L"read").c_str(), FindExInfoStandard, &fm, FindExSearchNameMatch, nullptr, 0)
                      : mapped.first(m(L"read").c_str(), &fm);
        check(hn != INVALID_HANDLE_VALUE, extended ? "native FindFirstFileExA" : "native FindFirstFileA");
        check(hm != INVALID_HANDLE_VALUE, extended ? "mapped FindFirstFileExA" : "mapped FindFirstFileA");
        check(fn.cFileName[0] && std::string(fm.cFileName) == std::string(fn.cFileName),
              "enumeration returns the same filename bytes as native");
        if (hn != INVALID_HANDLE_VALUE) native.close(hn);
        if (hm != INVALID_HANDLE_VALUE) mapped.close(hm);
    }
    const auto listed_native = list_after_wide_first(native, native_base + L"list*");
    const auto listed_mapped = list_after_wide_first(mapped, mapped_base + L"list*");
    check(listed_native.size() == 2 && listed_mapped == listed_native, "FindNextFileA matches native filename bytes");

    hn = native.create(n(L"write").c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    hm = mapped.create(m(L"write").c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    check(hn != INVALID_HANDLE_VALUE && hm != INVALID_HANDLE_VALUE, "CreateFileA write opens both paths");
    nn = nm = 0;
    if (hn != INVALID_HANDLE_VALUE) { ::WriteFile(hn, "updated!", 8, &nn, nullptr); ::CloseHandle(hn); }
    if (hm != INVALID_HANDLE_VALUE) { ::WriteFile(hm, "updated!", 8, &nm, nullptr); ::CloseHandle(hm); }
    check(nn == 8 && nm == nn && read_file(changed(L"write")) == read_file(oracle / leaf(L"write")),
          "CreateFileA writes the expected filename");
    check(!console || read_file(package / leaf(L"write")) == "original", "write-open preserves the package bytes");

    compare_success(native.set_attributes(n(L"attrs").c_str(), FILE_ATTRIBUTE_HIDDEN),
                    mapped.set_attributes(m(L"attrs").c_str(), FILE_ATTRIBUTE_HIDDEN), "SetFileAttributesA");
    check(::GetFileAttributesW(changed(L"attrs").c_str()) == ::GetFileAttributesW((oracle / leaf(L"attrs")).c_str()),
          "SetFileAttributesA changes the expected filename");
    check(!console || (::GetFileAttributesW((package / leaf(L"attrs")).c_str()) & FILE_ATTRIBUTE_HIDDEN) == 0,
          "attribute change preserves package attributes");

    compare_success(native.mkdir(n(L"newdir").c_str(), nullptr), mapped.mkdir(m(L"newdir").c_str(), nullptr),
                    "CreateDirectoryA");
    check(std::filesystem::is_directory(changed(L"newdir")), "CreateDirectoryA uses the expected filename");
    compare_success(native.mkdir_ex(n(L"template").c_str(), n(L"newdirex").c_str(), nullptr),
                    mapped.mkdir_ex(m(L"template").c_str(), m(L"newdirex").c_str(), nullptr), "CreateDirectoryExA");
    check(std::filesystem::is_directory(changed(L"newdirex")) &&
              ::GetFileAttributesW(changed(L"newdirex").c_str()) ==
                  ::GetFileAttributesW((oracle / leaf(L"newdirex")).c_str()), "CreateDirectoryExA resolves both names");

    compare_success(native.copy(n(L"read").c_str(), n(L"copied").c_str(), TRUE),
                    mapped.copy(m(L"read").c_str(), m(L"copied").c_str(), TRUE), "CopyFileA");
    check(read_file(changed(L"copied")) == "original", "CopyFileA resolves both names");
    compare_success(native.copy_ex(n(L"read").c_str(), n(L"copiedex").c_str(), nullptr, nullptr, nullptr, 0),
                    mapped.copy_ex(m(L"read").c_str(), m(L"copiedex").c_str(), nullptr, nullptr, nullptr, 0), "CopyFileExA");
    check(read_file(changed(L"copiedex")) == "original", "CopyFileExA resolves both names");
    compare_success(native.move(n(L"move").c_str(), n(L"moved").c_str()),
                    mapped.move(m(L"move").c_str(), m(L"moved").c_str()), "MoveFileA");
    check(read_file(changed(L"moved")) == "original", "MoveFileA resolves both names");
    compare_success(native.move_ex(n(L"moveex").c_str(), n(L"movedex").c_str(), 0),
                    mapped.move_ex(m(L"moveex").c_str(), m(L"movedex").c_str(), 0), "MoveFileExA");
    check(read_file(changed(L"movedex")) == "original", "MoveFileExA resolves both names");

    compare_success(native.remove(n(L"delete").c_str()), mapped.remove(m(L"delete").c_str()), "DeleteFileA");
    check(native.attributes(n(L"delete").c_str()) == INVALID_FILE_ATTRIBUTES &&
              mapped.attributes(m(L"delete").c_str()) == INVALID_FILE_ATTRIBUTES, "DeleteFileA hides the expected name");
    // Prepare this directory through the wide API, so a wrong CreateDirectoryA and RemoveDirectoryA cannot
    // accidentally cancel each other's filename-encoding error.
    compare_success(native.rmdir(n(L"rmdir").c_str()), mapped.rmdir(m(L"rmdir").c_str()), "RemoveDirectoryA");
    const auto mapped_attributes_w = resolve<decltype(&::GetFileAttributesW)>("GetFileAttributesW");
    check(mapped_attributes_w && mapped_attributes_w((mapped_base + leaf(L"rmdir")).c_str()) == INVALID_FILE_ATTRIBUTES,
          "RemoveDirectoryA hides the expected directory");
    if (console) {
        for (const auto* stem : {L"read", L"delete", L"move", L"moveex"})
            check(read_file(package / leaf(stem)) == "original", "copy/move/delete preserve original package files");
        for (const auto* stem : {L"copied", L"copiedex", L"moved", L"movedex", L"newdir", L"newdirex"})
            check(!std::filesystem::exists(package / leaf(stem)), "mutations create no package entries");
        check(std::filesystem::is_directory(package / leaf(L"rmdir")), "directory removal preserves the package folder");
    }
}

}  // namespace

int main() {
    ProcessState restore;
    try {
        const wchar_t character = distinct_character();
        if (!character) {
            std::printf("SKIP: no distinct round-tripping filename character for ACP %u and OEM %u\n", ::GetACP(), ::GetOEMCP());
            return 77;
        }
        std::printf("INFO ACP=%u OEM=%u character=U+%04X\n", ::GetACP(), ::GetOEMCP(), unsigned(character));
        const Api mapped = mapped_api();
        check(mapped.create && mapped.attributes && mapped.attributes_ex && mapped.first && mapped.first_ex &&
                  mapped.first_w && mapped.next && mapped.close && mapped.remove && mapped.mkdir && mapped.mkdir_ex &&
                  mapped.rmdir && mapped.move && mapped.move_ex && mapped.copy && mapped.copy_ex && mapped.set_attributes,
              "all narrow file APIs resolve through the production registry");
        if (failures) return 1;
        const auto root = std::filesystem::temp_directory_path() /
                          ("leonida_file_api_codepage_" + std::to_string(::GetCurrentProcessId()));
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        check(::SetCurrentDirectoryW(root.c_str()) != FALSE, "enter synthetic fixture directory");
        if (failures) return 1;
        for (int pass = 0; pass != 3; ++pass)
            for (bool console : {false, true})
                exercise(mapped, root, character, pass, console);
        ::SetCurrentDirectoryW(restore.directory.c_str());
        rt::fs::configure({}, {});
        std::filesystem::remove_all(root);
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
