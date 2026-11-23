// Exercises the host's placeholder memory (VirtualAlloc2, MapViewOfFile3), thread pool and processor information the
// way gta6rt's xmem and xtaskqueue shims use them. Exit code 0, or the number of the first check that failed.
#include <windows.h>

typedef PVOID(WINAPI* VirtualAlloc2Fn)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
typedef PVOID(WINAPI* MapViewOfFile3Fn)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                        ULONG);
typedef BOOL(WINAPI* UnmapViewOfFile2Fn)(HANDLE, PVOID, ULONG);

static volatile LONG g_work, g_timer;
static void CALLBACK on_work(PTP_CALLBACK_INSTANCE i, PVOID ctx, PTP_WORK w) { InterlockedAdd(&g_work, (LONG)(INT_PTR)ctx); }
static void CALLBACK on_timer(PTP_CALLBACK_INSTANCE i, PVOID ctx, PTP_TIMER t) { InterlockedIncrement(&g_timer); }

int main(void) {
    HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
    VirtualAlloc2Fn va2 = (VirtualAlloc2Fn)GetProcAddress(kb, "VirtualAlloc2");
    MapViewOfFile3Fn map3 = (MapViewOfFile3Fn)GetProcAddress(kb, "MapViewOfFile3");
    UnmapViewOfFile2Fn unmap2 = (UnmapViewOfFile2Fn)GetProcAddress(kb, "UnmapViewOfFile2");
    if (!va2 || !map3 || !unmap2) return 1;

    // A 2 MiB-aligned placeholder, split in two; the second half becomes ordinary memory, the first a section view.
    MEM_ADDRESS_REQUIREMENTS req = {0};
    req.Alignment = 0x200000;
    MEM_EXTENDED_PARAMETER p = {0};
    p.Type = MemExtendedParameterAddressRequirements;
    p.Pointer = &req;
    char* base = va2(GetCurrentProcess(), NULL, 0x400000, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, &p, 1);
    if (!base || ((UINT_PTR)base & 0x1FFFFF)) return 2;
    MEMORY_BASIC_INFORMATION info;
    if (!VirtualQuery(base, &info, sizeof info) || info.State != MEM_RESERVE || info.RegionSize != 0x400000) return 3;
    if (!VirtualFree(base + 0x200000, 0x200000, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return 4;
    char* hi = va2(GetCurrentProcess(), base + 0x200000, 0x200000, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                   PAGE_READWRITE, NULL, 0);
    if (hi != base + 0x200000) return 5;
    hi[0x1FFFFF] = 7;
    if (!VirtualQuery(base, &info, sizeof info) || info.State != MEM_RESERVE || info.RegionSize != 0x200000) return 6;

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT, 0, 0x20000, NULL);
    if (!section) return 7;
    if (!VirtualFree(base, 0x10000, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return 8;
    char* v = map3(section, GetCurrentProcess(), base, 0x10000, 0x10000, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
    if (v != base) return 9;
    v[5] = 42;
    char* w = map3(section, GetCurrentProcess(), NULL, 0x10000, 0x10000, 0, PAGE_READWRITE, NULL, 0);
    if (!w || w[5] != 42) return 10;  // two views of one page see the same bytes
    if (!unmap2(GetCurrentProcess(), v, MEM_PRESERVE_PLACEHOLDER)) return 11;
    if (!VirtualQuery(base, &info, sizeof info) || info.State != MEM_RESERVE) return 12;

    // Thread pool: three work items, then a timer firing every 5 ms until cancelled.
    PTP_WORK work = CreateThreadpoolWork(on_work, (PVOID)5, NULL);
    for (int i = 0; i < 3; ++i) SubmitThreadpoolWork(work);
    WaitForThreadpoolWorkCallbacks(work, FALSE);
    CloseThreadpoolWork(work);
    if (g_work != 15) return 13;
    PTP_TIMER timer = CreateThreadpoolTimer(on_timer, NULL, NULL);
    LARGE_INTEGER due;
    due.QuadPart = -10000;  // 1 ms from now
    FILETIME ft = {due.LowPart, (DWORD)due.HighPart};
    SetThreadpoolTimer(timer, &ft, 5, 0);
    Sleep(60);
    SetThreadpoolTimer(timer, NULL, 0, 0);
    WaitForThreadpoolTimerCallbacks(timer, TRUE);
    LONG fired = g_timer;
    Sleep(20);
    CloseThreadpoolTimer(timer);
    if (fired < 3 || g_timer != fired) return 14;

    DWORD len = 0;
    if (GetLogicalProcessorInformation(NULL, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return 15;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION cpu[65];
    if (len > sizeof cpu || !GetLogicalProcessorInformation(cpu, &len)) return 16;
    int cores = 0;
    for (DWORD i = 0; i < len / sizeof cpu[0]; ++i) cores += cpu[i].Relationship == RelationProcessorCore;
    return cores > 0 ? 0 : 17;
}
