// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed. Arguments stay wide, including characters the ANSI code page cannot represent. No game files.
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

void round_trip(const std::vector<std::wstring>& args, const char* what) {
    rt::process::set_command_line(L"game.exe", args);
    using Fn = LPWSTR(WINAPI*)();
    const LPWSTR line = reinterpret_cast<Fn>(rt::shims::find("kernel32.dll", "GetCommandLineW"))();
    int count = 0;
    LPWSTR* parsed = CommandLineToArgvW(line, &count);
    bool ok = parsed && count == int(args.size()) + 1;
    for (int i = 0; ok && i < count; ++i) ok = wargv()[i] && wcscmp(parsed[i], wargv()[i]) == 0;
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

    // main.cpp passes the game's wide arguments straight through. The CJK character is outside 1252 and the
    // accented one outside 1253, so the old path::string() conversion threw on one locale or the other.
    const std::wstring cafe = L"caf\u00e9";
    const std::wstring cjk = L"\u4e2d";
    round_trip({cafe, cjk}, "non-ASCII arguments");
    if (!wargv()[1] || cafe != wargv()[1] || !wargv()[2] || cjk != wargv()[2]) {
        std::fprintf(stderr, "FAIL: __wargv does not hold the original wide arguments\n");
        ++failures;
    }

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact\n");
    return 0;
}
