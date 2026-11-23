// Original stand-in run both natively and mapped. All paths and bytes come from check_current_directory.cmake.
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) ++failures;
    std::printf("tiny_current_directory: %s %s\n", ok ? "PASS" : "FAIL", what);
}

std::wstring current_directory() {
    const DWORD size = GetCurrentDirectoryW(0, nullptr);
    std::vector<wchar_t> text(size ? size : 1);
    if (!GetCurrentDirectoryW(static_cast<DWORD>(text.size()), text.data())) return {};
    return text.data();
}

std::string ansi(const std::wstring& wide) {
    const int size = WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (!size) return {};
    std::string out(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, out.data(), size, nullptr, nullptr);
    out.pop_back();
    return out;
}

bool same_path(const std::wstring& a, const std::wstring& b) { return _wcsicmp(a.c_str(), b.c_str()) == 0; }

void check_buffers(const std::wstring& expected) {
    const DWORD need_w = static_cast<DWORD>(expected.size() + 1);
    SetLastError(9876);
    check(GetCurrentDirectoryW(0, nullptr) == need_w && GetLastError() == 9876, "W size query");
    for (const DWORD capacity : {0ul, 1ul, need_w - 1, need_w, need_w + 1}) {
        std::vector<wchar_t> buffer(need_w + 2, L'#');
        SetLastError(9876);
        const DWORD result = GetCurrentDirectoryW(capacity, buffer.data());
        const DWORD error = GetLastError();
        const bool fits = capacity >= need_w;
        check(result == (fits ? need_w - 1 : need_w) && error == 9876 &&
                  (fits ? same_path(buffer.data(), expected) && buffer[need_w] == L'#'
                        : std::all_of(buffer.begin(), buffer.end(), [](wchar_t c) { return c == L'#'; })),
              "W buffer length, contents and LastError");
    }
    const std::string expected_a = ansi(expected);
    const DWORD need_a = static_cast<DWORD>(expected_a.size() + 1);
    SetLastError(9876);
    check(GetCurrentDirectoryA(0, nullptr) == need_a && GetLastError() == 9876, "A size query");
    for (const DWORD capacity : {0ul, 1ul, need_a - 1, need_a, need_a + 1}) {
        std::vector<char> buffer(need_a + 2, '#');
        SetLastError(9876);
        const DWORD result = GetCurrentDirectoryA(capacity, buffer.data());
        const DWORD error = GetLastError();
        const bool fits = capacity >= need_a;
        check(result == (fits ? need_a - 1 : need_a) && error == 9876 &&
                  (fits ? _stricmp(buffer.data(), expected_a.c_str()) == 0 && buffer[need_a] == '#'
                        : std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == '#'; })),
              "A buffer length, contents and LastError");
    }
}

std::string read(const wchar_t* path) {
    const HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    char bytes[64]{};
    DWORD count = 0;
    const BOOL ok = ReadFile(file, bytes, sizeof(bytes), &count, nullptr);
    CloseHandle(file);
    return ok ? std::string(bytes, count) : std::string();
}

bool write(const wchar_t* path, DWORD disposition, const char* text) {
    const HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, disposition, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    if (disposition == OPEN_EXISTING) SetFilePointer(file, 0, nullptr, FILE_END);
    DWORD count = 0;
    const DWORD size = static_cast<DWORD>(std::char_traits<char>::length(text));
    const BOOL ok = WriteFile(file, text, size, &count, nullptr);
    CloseHandle(file);
    return ok && count == size;
}

void rejected(const wchar_t* path, DWORD expected_error, const std::wstring& retained) {
    SetLastError(9876);
    const BOOL ok = SetCurrentDirectoryW(path);
    const DWORD error = GetLastError();
    check(!ok && error == expected_error && same_path(current_directory(), retained),
          "failed directory change preserves the current directory and native error");
}

struct DeviceOpen {
    bool ok;
    DWORD error, type;
};
struct DeviceCase {
    const wchar_t* name;
    DWORD access, disposition;
};
const DeviceCase device_cases[] = {
    {L"NUL", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING},
    {L"NUL:", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING},
    {L"CON", GENERIC_READ, OPEN_EXISTING},
    {L"CON", GENERIC_WRITE, CREATE_ALWAYS},
    {L"CONOUT$", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING},
    {L"CONOUT$", GENERIC_READ | GENERIC_WRITE, CREATE_ALWAYS},
    {L"CONIN$", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING},
    {L"CONIN$", GENERIC_READ | GENERIC_WRITE, CREATE_ALWAYS},
    {L"NUL.txt", GENERIC_READ | GENERIC_WRITE, CREATE_ALWAYS},
};

DeviceOpen open_device(const DeviceCase& test, bool narrow, const char* phase) {
    const HANDLE h = narrow
        ? CreateFileA(ansi(test.name).c_str(), test.access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, test.disposition, 0, nullptr)
        : CreateFileW(test.name, test.access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, test.disposition, 0, nullptr);
    const DeviceOpen result{h != INVALID_HANDLE_VALUE, h == INVALID_HANDLE_VALUE ? GetLastError() : 0,
                            h == INVALID_HANDLE_VALUE ? 0 : GetFileType(h)};
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    std::printf("tiny_current_directory: device %s %ls %c disposition=%lu ok=%d error=%lu type=%lu\n",
                phase, test.name, narrow ? 'A' : 'W', test.disposition, result.ok, result.error, result.type);
    return result;
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 4) return 90;
    const std::wstring saved = current_directory();
    std::wstring root = argv[1], host = argv[2], scratch = argv[3];
    std::replace(root.begin(), root.end(), L'/', L'\\');
    std::replace(host.begin(), host.end(), L'/', L'\\');
    std::replace(scratch.begin(), scratch.end(), L'/', L'\\');
    // The mapped overlay has never had a T: write. Current main already makes this root available on first use.
    check(SetCurrentDirectoryW(scratch.c_str()) && same_path(current_directory(), scratch),
          "review control: enter the fresh scratch root before any scratch write");
    check(SetCurrentDirectoryW(saved.c_str()), "return to host after fresh scratch control");
    std::vector<DeviceOpen> devices_before;
    for (const auto& test : device_cases)
        for (bool narrow : {false, true}) devices_before.push_back(open_device(test, narrow, "before"));
    const std::wstring data = root + L"\\data";
    check(read((data + L"\\data.txt").c_str()) == "package", "absolute read before changing directory");
    SetLastError(9876);
    const BOOL changed = SetCurrentDirectoryW(data.c_str());
    const DWORD change_error = GetLastError();
    check(changed && change_error == 9876, "enter directory using W");
    if (!changed) return 1;  // The baseline must fail here without running relative writes in the host folder.
    check_buffers(data);
    size_t device_index = 0;
    for (const auto& test : device_cases) {
        for (bool narrow : {false, true}) {
            const DeviceOpen before = devices_before[device_index++];
            const DeviceOpen after = open_device(test, narrow, "after");
            check(before.ok == after.ok && before.error == after.error && before.type == after.type,
                  "review regression: relative device names keep their native result and handle type");
        }
    }
    // Modern Windows treats NUL.txt as an ordinary file, while older hosts may still recognize a DOS device.
    // The actual native open above supplies that distinction, rather than the fixture hardcoding the OS version.
    if (devices_before.back().ok && devices_before.back().type == FILE_TYPE_DISK) {
        check(write((host + L"\\NUL.txt").c_str(), CREATE_ALWAYS, "host-lookalike") &&
                  write(L"NUL.txt", CREATE_ALWAYS, "ordinary") && read(L"NUL.txt") == "ordinary" &&
                  read((host + L"\\NUL.txt").c_str()) == "host-lookalike",
              "review control: ordinary NUL.txt stays relative and does not overwrite its host counterpart");
    }
    const std::wstring nt_host = L"\\??\\" + host + L"\\host.txt";
    check(GetFileAttributesW(nt_host.c_str()) != INVALID_FILE_ATTRIBUTES &&
              GetFileAttributesA(ansi(nt_host).c_str()) != INVALID_FILE_ATTRIBUTES && read(nt_host.c_str()) == "host",
          "review regression: NT absolute host paths keep their native attributes and readable bytes");
    check(read(L"data.txt") == "package", "relative read reaches package");
    check(GetFileAttributesA("data.txt") != INVALID_FILE_ATTRIBUTES, "relative ANSI attributes");
    check(write(L"data.txt", OPEN_EXISTING, "+overlay"), "relative write keeps existing bytes");
    check(read(L"data.txt") == "package+overlay", "relative read sees written copy");
    check(write(L"new.txt", CREATE_NEW, "new"), "relative new file");
    WIN32_FIND_DATAW found{};
    HANDLE find = FindFirstFileW(L"*.txt", &found);
    int data_count = 0, new_count = 0;
    if (find != INVALID_HANDLE_VALUE) {
        do {
            if (_wcsicmp(found.cFileName, L"data.txt") == 0) ++data_count;
            if (_wcsicmp(found.cFileName, L"new.txt") == 0) ++new_count;
        } while (FindNextFileW(find, &found));
        FindClose(find);
    }
    check(data_count == 1 && new_count == 1, "relative listing contains each visible name once");
    find = FindFirstFileW(L"*.", &found);
    int plain_count = 0, dotted_count = 0;
    if (find != INVALID_HANDLE_VALUE) {
        do {
            if (_wcsicmp(found.cFileName, L"plain") == 0) ++plain_count;
            if (_wcsicmp(found.cFileName, L"data.txt") == 0) ++dotted_count;
        } while (FindNextFileW(find, &found));
        FindClose(find);
    }
    check(plain_count == 1 && dotted_count == 0, "review control: trailing-dot wildcard mask keeps native matching");
    check(DeleteFileA("deleted.txt") && GetFileAttributesW(L"deleted.txt") == INVALID_FILE_ATTRIBUTES,
          "relative delete hides a package file");
    rejected(L"missing", ERROR_FILE_NOT_FOUND, data);
    rejected(L"missing\\child", ERROR_PATH_NOT_FOUND, data);
    rejected(L"data.txt", ERROR_DIRECTORY, data);
    rejected(L"data.txt\\child", ERROR_PATH_NOT_FOUND, data);
    rejected(L"", ERROR_INVALID_NAME, data);
    rejected((host + L"\\missing").c_str(), ERROR_FILE_NOT_FOUND, data);
    check(SetCurrentDirectoryA("sub") && same_path(current_directory(), data + L"\\sub"), "relative ANSI directory change");
    check(SetCurrentDirectoryW(L"..") && same_path(current_directory(), data), "parent directory change");
    check(SetCurrentDirectoryW(L"sub/./..") && same_path(current_directory(), data), "normalize relative separators and dots");
    for (const wchar_t* path : {L"sub.", L"sub ", L"sub. ", L"sub.\\", L"sub\\.. ", L"sub\\..."}) {
        check(SetCurrentDirectoryW(data.c_str()) && SetCurrentDirectoryW(path) &&
                  same_path(current_directory(), data + L"\\sub"),
              "review regression: final dot and space spelling normalizes the directory identity");
    }
    check(SetCurrentDirectoryW(data.c_str()) && SetCurrentDirectoryW(L"sub.\\child") &&
              same_path(current_directory(), data + L"\\sub\\child"),
          "review regression: a single intermediate trailing dot normalizes");
    check(SetCurrentDirectoryW(data.c_str()), "return after directory spelling controls");
    for (const wchar_t* path : {L"sub \\child", L"sub. \\child", L"sub..\\child", L"sub...\\child"})
        rejected(path, ERROR_PATH_NOT_FOUND, data);
    rejected(L"sub \\", ERROR_FILE_NOT_FOUND, data);
    const std::wstring rooted = data.substr(data.find(L':') + 1);
    check(SetCurrentDirectoryW(rooted.c_str()) && same_path(current_directory(), data), "rooted path keeps current drive");
    check(CreateDirectoryW(L"caf\u00e9", nullptr) && SetCurrentDirectoryW(L"caf\u00e9"), "wide directory name");
    check_buffers(data + L"\\caf\u00e9");
    check(SetCurrentDirectoryW(data.c_str()), "return from wide directory");
    check(read((host + L"\\host.txt").c_str()) == "host", "absolute host read while in console directory");
    check(write((host + L"\\host.txt").c_str(), OPEN_EXISTING, "+host"), "absolute host write remains on host");

    int alias_index = 0;
    for (const wchar_t* suffix : {L".", L" ", L". "}) {
        for (bool narrow : {false, true}) {
            const std::wstring name = root + L"\\review-alias-" + std::to_wstring(alias_index) + (narrow ? L"-a" : L"-w");
            const std::wstring alias = name + suffix, moved = name + L"-moved";
            check(SetCurrentDirectoryW(name.c_str()), "enter canonical package directory for alias lock control");
            SetLastError(0);
            const BOOL renamed = narrow ? MoveFileA(ansi(alias).c_str(), ansi(moved).c_str())
                                        : MoveFileW(alias.c_str(), moved.c_str());
            check(!renamed && GetLastError() == ERROR_SHARING_VIOLATION,
                  "review regression: an alias cannot move the canonical current directory");
            const BOOL entered = narrow ? SetCurrentDirectoryA(ansi(alias).c_str()) : SetCurrentDirectoryW(alias.c_str());
            check(entered && same_path(current_directory(), name),
                  "review regression: A/W aliased directory changes report the canonical spelling");
            SetLastError(0);
            const BOOL removed = narrow ? RemoveDirectoryA(ansi(name).c_str()) : RemoveDirectoryW(name.c_str());
            check(!removed && GetLastError() == ERROR_SHARING_VIOLATION,
                  "review regression: an aliased current directory cannot be removed by its canonical name");
            check(GetFileAttributesW(L".") != INVALID_FILE_ATTRIBUTES, "review regression: locked directory remains visible");
            check(SetCurrentDirectoryW(data.c_str()), "leave alias control directory");
        }
        ++alias_index;
    }

    const std::wstring empty = root + L"\\empty";
    check(SetCurrentDirectoryW(empty.c_str()), "enter empty package directory");
    SetLastError(0);
    check(!MoveFileW(empty.c_str(), empty.c_str()) && GetLastError() == ERROR_SHARING_VIOLATION,
          "a move onto the current directory's identical name is still locked");
    SetLastError(0);
    check(!MoveFileW(empty.c_str(), (root + L"\\EMPTY").c_str()) && GetLastError() == ERROR_SHARING_VIOLATION,
          "a case-only rename of the current directory is locked");
    SetLastError(0);
    check(!MoveFileExW(empty.c_str(), empty.c_str(), MOVEFILE_REPLACE_EXISTING) && GetLastError() == ERROR_SHARING_VIOLATION,
          "MoveFileEx replacement does not bypass the current-directory lock");
    SetLastError(0);
    check(!MoveFileW(root.c_str(), root.c_str()) && GetLastError() == ERROR_ACCESS_DENIED,
          "a move onto an ancestor's identical name is still locked");
    SetLastError(0);
    check(!RemoveDirectoryW(empty.c_str()) && GetLastError() == ERROR_SHARING_VIOLATION, "current directory cannot be removed");
    SetLastError(0);
    check(!MoveFileW(empty.c_str(), (root + L"\\moved-empty").c_str()) && GetLastError() == ERROR_SHARING_VIOLATION,
          "current directory cannot be moved");
    check(SetCurrentDirectoryW(data.c_str()) && RemoveDirectoryW(empty.c_str()), "leaving directory releases its lock");
    rejected(empty.c_str(), ERROR_FILE_NOT_FOUND, data);
    check(CreateDirectoryA("new-dir", nullptr) && SetCurrentDirectoryA("new-dir"), "enter new writable directory");
    SetLastError(0);
    check(!MoveFileW(L".", L".") && GetLastError() == ERROR_SHARING_VIOLATION,
          "a relative same-name move of the writable current directory is locked");
    SetLastError(0);
    check(!MoveFileW(L".", L"..\\moved-dir") && GetLastError() == ERROR_SHARING_VIOLATION,
          "new current directory cannot be moved");
    check(SetCurrentDirectoryW(L"..") && MoveFileW(L"new-dir", L"moved-dir"), "leaving writable directory releases its lock");
    check(SetCurrentDirectoryA(ansi(host).c_str()) && same_path(current_directory(), host), "absolute host change exits console mode");
    check(read(L"host.txt") == "host+host", "relative read after host change stays on host");
    check(SetCurrentDirectoryW(saved.c_str()), "restore original host directory");
    std::printf("tiny_current_directory: %d checks, %d failures\n", checks, failures);
    if (!failures) std::puts("tiny_current_directory: OK");
    return failures ? 1 : 0;
}
