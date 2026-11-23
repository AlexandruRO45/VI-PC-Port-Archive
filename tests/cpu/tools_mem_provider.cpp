// Test-only recording provider for xmem.dll!XMemIsToolsMemAvailable (#870). It is not the desktop
// xmem.dll shim. Synthetic behavior only:
//   TOOLS_MEM_RESULT  "0" returns FALSE (desktop without a tools partition; proposed desktop policy),
//                     "1" returns TRUE, "flip" alternates FALSE/TRUE per call (negative control).
//                     Anything else, or unset, returns 2, which the caller rejects (fail closed).
//   TOOLS_MEM_RECORD  file that receives one "query <n> <result>" line per call. Unset returns 2.
// The signature BOOL XMemIsToolsMemAvailable(void) is an assumption; the caller passes no arguments.
#include <windows.h>

static volatile LONG calls;

static bool equals(const wchar_t* left, const wchar_t* right)
{
    while (*left && *left == *right)
        ++left, ++right;
    return *left == *right;
}

static bool record(LONG call, BOOL result)
{
    wchar_t path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"TOOLS_MEM_RECORD", path, MAX_PATH);
    if (!length || length >= MAX_PATH)
        return false;
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    char line[] = "query 0 0\n";
    line[6] = static_cast<char>('0' + call % 10);
    line[8] = static_cast<char>('0' + result % 10);
    DWORD written = 0;
    const bool ok = WriteFile(file, line, sizeof(line) - 1, &written, nullptr) && written == sizeof(line) - 1;
    CloseHandle(file);
    return ok;
}

extern "C" BOOL WINAPI XMemIsToolsMemAvailable()
{
    const LONG call = InterlockedIncrement(&calls);
    wchar_t mode[8];
    const DWORD length = GetEnvironmentVariableW(L"TOOLS_MEM_RESULT", mode, 8);
    BOOL result = 2;
    if (length && length < 8) {
        if (equals(mode, L"0"))
            result = FALSE;
        else if (equals(mode, L"1"))
            result = TRUE;
        else if (equals(mode, L"flip"))
            result = (call & 1) ? FALSE : TRUE;
    }
    if (!record(call, result))
        return 2;
    return result;
}

// Number of XMemIsToolsMemAvailable calls in this process. Exported by both provider variants.
extern "C" LONG WINAPI tools_mem_calls()
{
    return calls;
}
