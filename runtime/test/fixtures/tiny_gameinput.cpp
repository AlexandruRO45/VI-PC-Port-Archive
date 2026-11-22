// A mapped program that imports GameInputCreate the way a GDK title does: from GameInput.dll, by ordinal 1, with no
// name in its import table (docs/gameinput.md, "The import is by ordinal"). Run under `gta6rt --run`, the loader must
// bind that import to the GameInput shim; check_loaded_gameinput.cmake reads the binding from the loader's report,
// since a host GameInput.dll would answer the calls below as well. Ordinary test code; no game data.
#include <windows.h>
#include <unknwn.h>

#include <cstdio>

extern "C" __declspec(dllimport) HRESULT __stdcall GameInputCreate(void** game_input);

namespace {

int fail(const char* stage, unsigned long value) {
    std::printf("tiny_gameinput: FAIL %s (0x%08lx)\n", stage, value);
    return 1;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    void* first = nullptr;
    HRESULT hr = GameInputCreate(&first);
    if (hr != S_OK || !first) return fail("GameInputCreate", static_cast<unsigned long>(hr));
    // The shim hands out one instance per process; each call adds a reference.
    void* second = nullptr;
    hr = GameInputCreate(&second);
    if (hr != S_OK || second != first) return fail("a second GameInputCreate returns the same instance", static_cast<unsigned long>(hr));
    static_cast<IUnknown*>(second)->Release();
    static_cast<IUnknown*>(first)->Release();
    std::puts("tiny_gameinput: PASS");
    return 0;
}
