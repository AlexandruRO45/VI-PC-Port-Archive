#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>

extern "C" __declspec(dllimport) char** __cdecl __p__pgmptr();
extern "C" __declspec(dllimport) wchar_t** __cdecl __p__wpgmptr();

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) return 1;
    const DWORD capacity = GetFullPathNameW(argv[1], 0, nullptr, nullptr);
    if (!capacity) return 2;
    std::wstring expected_wide(capacity, L'\0');
    const DWORD length = GetFullPathNameW(argv[1], capacity, expected_wide.data(), nullptr);
    if (!length || length >= capacity) return 2;
    expected_wide.resize(length);
    const int size = WideCharToMultiByte(CP_ACP, 0, expected_wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (!size) return 2;
    std::string expected(static_cast<size_t>(size), '\0');
    if (!WideCharToMultiByte(CP_ACP, 0, expected_wide.c_str(), -1, expected.data(), size, nullptr, nullptr)) return 3;
    char* narrow = nullptr;
    wchar_t* wide = nullptr;
    if (_get_pgmptr(&narrow) != 0 || _get_wpgmptr(&wide) != 0 || !narrow || !wide ||
        std::strcmp(narrow, expected.c_str()) != 0 || std::wcscmp(wide, expected_wide.c_str()) != 0 ||
        *__p__pgmptr() != narrow || *__p__wpgmptr() != wide)
        return 4;
    std::puts("tiny_pgmptr: PASS");
    return 0;
}
