// Source-built CRT-free provider with a NONAME export. No Xbox/game data.
#include <windows.h>
#include <cstdint>

// Debug /RTC inserts CRT initialization calls that a /NODEFAULTLIB fixture
// cannot serve. Keep this synthetic fixture CRT-free in Debug and Release.
#pragma runtime_checks("", off)

extern "C" BOOL WINAPI ordinal_provider_entry(HINSTANCE, DWORD, void*) { return TRUE; }

extern "C" std::uint64_t mix_args(std::uint64_t a, std::uint64_t b, std::uint64_t* words) {
    words[0] = a;
    words[1] = b;
    words[2] = a - b + 11;
#ifdef ALTER_RETURN
    return words[2] + 1;
#else
    return words[2];
#endif
}
