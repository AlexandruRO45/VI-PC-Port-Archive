// Source-built CRT-free EXE importing the provider only by ordinal 77.
#include <cstdint>

// Debug /RTC inserts CRT initialization calls that a /NODEFAULTLIB fixture
// cannot serve. Keep this synthetic fixture CRT-free in Debug and Release.
#pragma runtime_checks("", off)

extern "C" __declspec(dllimport) std::uint64_t mix_args(
    std::uint64_t a, std::uint64_t b, std::uint64_t* output);

extern "C" std::uint32_t ordinal_exe_entry() {
    std::uint64_t words[3] = {};
    const std::uint64_t result = mix_args(5, 3, words);
    if (words[0] != 5 || words[1] != 3 || words[2] != 13) return 0xBAD00001u;
    return static_cast<std::uint32_t>(result);  // original 13; altered provider 14
}
