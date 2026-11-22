// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed. Arguments stay wide, so one the ANSI code page can't represent is kept intact in __wargv and no
// longer throws (RT-84: path::string() raised std::system_error, 0xC0000409, on such a character). No game files.
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <string>
#include <vector>

#include "process_shim.h"
#include "shims.h"

namespace {
int failures = 0;

wchar_t** wargv() {
    using Fn = wchar_t*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___wargv"))();
}

char** argv() {
    using Fn = char*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___argv"))();
}

void round_trip(const std::vector<std::wstring>& args, const char* what) {
    rt::process::set_command_line(L"game.exe", args);
    using Fn = LPWSTR(WINAPI*)();
    const LPWSTR line = reinterpret_cast<Fn>(rt::shims::find("kernel32.dll", "GetCommandLineW"))();
    int count = 0;
    LPWSTR* parsed = CommandLineToArgvW(line, &count);
    bool ok = parsed && count == int(args.size()) + 1;
    for (int i = 0; ok && i < count; ++i) ok = wargv()[i] && wcscmp(parsed[i], wargv()[i]) == 0;
    for (size_t i = 0; ok && i < args.size(); ++i) ok = args[i] == wargv()[i + 1];   // exactly what main.cpp passed
    for (int i = 0; ok && i < count; ++i) ok = argv()[i] != nullptr;                  // and a narrow argv for each
    if (!ok) {
        std::fwprintf(stderr, L"FAIL %hs: GetCommandLineW [%ls] parses to %d argument(s)\n", what, line, count);
        ++failures;
    }
    LocalFree(parsed);
}
}  // namespace

int main() {
    round_trip({L"plain", L"with space"}, "plain and spaced arguments");
    round_trip({L"C:\\My Games\\", L"next"}, "a quoted argument ending in a backslash");
    round_trip({L"a\\\"b"}, "a backslash before a quote");
    round_trip({L"\\\\server\\share\\x y"}, "a UNC path with a space");

    // Non-ASCII arguments, whatever the ANSI code page: one that 1252 has (e), and Greek and CJK ones that it and
    // most code pages lack. Before RT-84, main.cpp's path::string() threw on the last two before the game started.
    round_trip({L"caf\u00e9"}, "a Latin-1 argument");
    round_trip({L"\u03b1", L"\u4e2d\u6587 x"}, "arguments outside the ANSI code page");

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact\n");
    return 0;
}
