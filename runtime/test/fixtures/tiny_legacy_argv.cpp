// Original synthetic caller of the actual legacy CRT argument imports, not a replacement CRT.
#include <windows.h>
#include <cstdio>

extern "C" __declspec(dllimport) int __cdecl __getmainargs(int*, char***, char***, int, void*);
extern "C" __declspec(dllimport) int __cdecl __wgetmainargs(int*, wchar_t***, wchar_t***, int, void*);

template <class C>
void dump(const C* text) {
    if (!text) { std::printf("null"); return; }
    for (; *text; ++text) {
        if constexpr (sizeof(C) == 1) std::printf("%02x", static_cast<unsigned char>(*text));
        else std::printf("%04x", static_cast<unsigned short>(*text));
    }
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 2 || (argv[1][0] != L'A' && argv[1][0] != L'W')) return 20;
    // Reconfiguration can invalidate the incoming argv table. Retain the requested sequence before any call.
    char mode[8]{};
    int length = 0;
    for (; length < 7 && argv[1][length]; ++length) mode[length] = static_cast<char>(argv[1][length]);
    if (length < 2 || argv[1][length]) return 20;
    struct StartupInfo { int newmode; } info{};
    for (int step = 1; mode[step]; ++step) {
        if (mode[step] != '0' && mode[step] != '1') return 20;
        const int expand = mode[step] - '0';
        int count = -1, rc = -1;
        if (mode[0] == 'A') {
            char** values = nullptr;
            char** environment = nullptr;
            rc = __getmainargs(&count, &values, &environment, expand, &info);
            std::printf("A step=%d expand=%d rc=%d argc=%d env=%d term=%d\n",
                        step, expand, rc, count, environment != nullptr,
                        values && count >= 0 && count < 256 && !values[count]);
            if (rc || !values || count < 1 || count >= 256) return 21;
            for (int i = 0; i < count; ++i) {
                std::printf("arg%d=", i); dump(values[i]); std::puts("");
            }
        } else {
            wchar_t** values = nullptr;
            wchar_t** environment = nullptr;
            rc = __wgetmainargs(&count, &values, &environment, expand, &info);
            std::printf("W step=%d expand=%d rc=%d argc=%d env=%d term=%d\n",
                        step, expand, rc, count, environment != nullptr,
                        values && count >= 0 && count < 256 && !values[count]);
            if (rc || !values || count < 1 || count >= 256) return 22;
            for (int i = 0; i < count; ++i) {
                std::printf("arg%d=", i); dump(values[i]); std::puts("");
            }
        }
    }
    return 0;
}
