/* AMD64 Windows caller probe. With no DLL argument, run synthetic controls. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

typedef HRESULT (WINAPI *Configure)(void *);
static unsigned seen;

static HRESULT WINAPI control(void *desc) {
    uint32_t expected[8] = {0};
    expected[1] = 3 + seen++;
    return memcmp(desc, expected, sizeof(expected)) ? E_INVALIDARG :
           (expected[1] == 3 ? S_OK : E_NOTIMPL);
}

static HRESULT call(Configure configure, unsigned count) {
    _Alignas(16) uint32_t words[8] = {0};
    words[1] = count;
    return configure(words);
}

int wmain(int argc, wchar_t **argv) {
    if (sizeof(void *) != 8 || argc > 2) return 2;
    if (argc == 1) {
        return call(control, 3) == S_OK && call(control, 4) == E_NOTIMPL ? 0 : 1;
    }
    HMODULE dll = LoadLibraryW(argv[1]);
    if (!dll) return 2;
    Configure configure = (Configure)GetProcAddress(dll, "D3D12ConfigureVirtualMemoryX");
    if (!configure) {
        FreeLibrary(dll);
        return 2;
    }
    int failed = 0;
    for (unsigned count = 3; count <= 4; ++count) {
        HRESULT result = call(configure, count);
        printf("count %u: HRESULT 0x%08lx\n", count, (unsigned long)result);
        if (FAILED(result)) failed = 1;
    }
    FreeLibrary(dll);
    return failed;
}

