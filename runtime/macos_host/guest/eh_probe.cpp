// C++ exception cases beyond tiny_game's same-frame throw, for the macOS host's unwinder. Exits 0 when all hold.
#include <cstdio>
#include <stdexcept>
#include <string>

static int g_dtors = 0;
struct Guard {
    ~Guard() { ++g_dtors; }
};
__declspec(noinline) static void deep(int n) {
    Guard g;  // destroyed while the exception passes through this frame
    std::string s(64, 'x');
    if (n == 0) throw std::runtime_error("deep " + s.substr(0, 2));
    deep(n - 1);
}
__declspec(noinline) static void throw_int() { throw 7; }

int main() {
    int fails = 0;
    try { deep(3); } catch (const std::exception& e) {
        fails += std::string(e.what()) != "deep xx";
    }
    fails += (g_dtors != 4) * 10;
    int got = 0;
    try { throw_int(); } catch (int v) { got = v; }
    fails += (got != 7) * 100;
    bool any = false;
    try { throw std::string("s"); } catch (...) { any = true; }
    fails += !any * 1000;
    std::printf("eh_probe: %s (dtors=%d fails=%d)\n", fails ? "FAIL" : "OK", g_dtors, fails);
    return fails ? 1 : 0;
}
