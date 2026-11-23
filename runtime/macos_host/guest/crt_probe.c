// The UCRT pieces a retail game imports that differ from the host libc (32-bit long, UTF-16 wchar_t, the MS rand),
// plus GameInputCreate by ordinal. Exit code 0, or the number of the first check that failed.
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

static int __cdecl by_value(const void* a, const void* b) { return *(const int*)a - *(const int*)b; }

int main(void) {
    volatile float x = 0.5f;
    if (fabsf(sinf(x) - 0.4794255f) > 1e-6f || fabsf(powf(x, 2.0f) - 0.25f) > 1e-7f) return 1;
    if (wcslen(L"abc") != 3 || wcscmp(L"abc", L"abd") >= 0 || _wcsnicmp(L"ABC", L"abd", 2)) return 2;
    wchar_t w[16];
    if (swprintf_s(w, 16, L"%s=%d", L"n", 42) != 4 || wcscmp(w, L"n=42")) return 3;
    if (strtol("99999999999", NULL, 10) != 0x7FFFFFFF || sizeof(long) != 4) return 4;
    int v[] = {3, 1, 2};
    qsort(v, 3, sizeof v[0], by_value);
    if (v[0] != 1 || v[2] != 3) return 5;
    srand(1);
    if (rand() != 41) return 6;  // the MS generator's first value for seed 1
    if (_stricmp("Hello", "hELLO") || toupper('a') != 'A') return 7;
    __time64_t now = _time64(NULL);
    struct tm t;
    if (_gmtime64_s(&t, &now) || t.tm_year < 120) return 8;
    HANDLE m = CreateMutexA(NULL, FALSE, NULL);
    if (!m || WaitForSingleObject(m, 0) != WAIT_OBJECT_0 || !ReleaseMutex(m)) return 9;
    typedef HRESULT(WINAPI * CreateFn)(void**);
    CreateFn create = (CreateFn)GetProcAddress(LoadLibraryA("GameInput.dll"), MAKEINTRESOURCEA(1));
    void* input = NULL;
    if (!create || create(&input) != S_OK || !input) return 10;
    return 0;
}
