// LEONIDA_PROBE (shims/startup_trace.h): with the variable set and a trace open, a breakpoint at each listed RVA of an
// image logs the registers and lets the instruction run, re-armed for the next hit. A synthetic three-instruction
// function stands in for the image; no game files. A separate executable because the variables are read once.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

int count(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
    return n;
}

}  // namespace

int main() {
    // `mov rax, rcx` (3 bytes), `add rax, 1` (4 bytes, at RVA 3), `ret`: f(x) = x + 1.
    const unsigned char code[] = {0x48, 0x89, 0xC8, 0x48, 0x83, 0xC0, 0x01, 0xC3};
    auto* image = static_cast<unsigned char*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READ));
    check(image != nullptr, "an executable page for the synthetic image");
    if (!image) return 1;
    DWORD old = 0;
    VirtualProtect(image, 4096, PAGE_READWRITE, &old);
    std::memcpy(image, code, sizeof code);
    VirtualProtect(image, 4096, PAGE_EXECUTE_READ, &old);
    using Fn = unsigned long long (*)(unsigned long long);
    const auto f = reinterpret_cast<Fn>(image);

    // Before the first exception: both variables are read once. RVA 3 is `add`; 0x5000 is outside the image.
    SetEnvironmentVariableW(L"LEONIDA_PROBE", L"3, 5000");
    SetEnvironmentVariableW(L"LEONIDA_PROBE_MAX", L"2");
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path = std::wstring(temp) + L"probe_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());
    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    rt::startup_trace::install_probes(image, 4096);
    check(image[3] == 0xCC, "the probe is armed at RVA 3");

    bool results_ok = true;
    for (unsigned long long x = 41; x < 44; ++x) results_ok = results_ok && f(x) == x + 1;
    check(results_ok, "the probed instruction still runs, three calls in a row (re-armed after each)");
    check(image[3] == 0xCC, "the probe is armed again after the last hit");
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    in.close();
    DeleteFileW(path.c_str());
    check(count(log, "\"kind\":\"probe\"") == 2, "LEONIDA_PROBE_MAX=2: two of the three hits are logged");
    check(log.find("\"hit\":1,\"rax\":\"29\",") != std::string::npos,
          "the first hit has rax = 0x29 (`mov rax, rcx` ran, `add` not yet)");
    check(log.find("\"hit\":2,\"rax\":\"2a\",") != std::string::npos, "the second hit sees the next argument");
    check(count(log, "\"kind\":\"exception\"") == 0, "no probe hit reaches the exception log");

    std::printf("%s\n", g_failed ? "probe_test: FAILED" : "probe_test: OK");
    return g_failed ? 1 : 0;
}
