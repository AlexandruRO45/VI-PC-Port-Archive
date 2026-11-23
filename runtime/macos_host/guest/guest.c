// A Windows x64 test guest with no CRT, for the macOS host prototype. Built with clang for
// x86_64-pc-windows-msvc and lld-link (build.sh). Exits 0 when every check held, else the number of the first failure.
typedef void* HANDLE;
typedef unsigned long DWORD;
typedef int BOOL;
typedef unsigned long long SIZE_T;
typedef long long LARGE_INTEGER;
#define WINAPI __stdcall
#define IMPORT __declspec(dllimport)

IMPORT HANDLE WINAPI GetStdHandle(DWORD);
IMPORT BOOL WINAPI WriteFile(HANDLE, const void*, DWORD, DWORD*, void*);
IMPORT void WINAPI ExitProcess(unsigned);
IMPORT DWORD WINAPI GetLastError(void);
IMPORT HANDLE WINAPI GetProcessHeap(void);
IMPORT void* WINAPI HeapAlloc(HANDLE, DWORD, SIZE_T);
IMPORT BOOL WINAPI HeapFree(HANDLE, DWORD, void*);
IMPORT void* WINAPI VirtualAlloc(void*, SIZE_T, DWORD, DWORD);
IMPORT BOOL WINAPI VirtualFree(void*, SIZE_T, DWORD);
IMPORT DWORD WINAPI TlsAlloc(void);
IMPORT void* WINAPI TlsGetValue(DWORD);
IMPORT BOOL WINAPI TlsSetValue(DWORD, void*);
IMPORT HANDLE WINAPI CreateThread(void*, SIZE_T, DWORD(WINAPI*)(void*), void*, DWORD, DWORD*);
IMPORT DWORD WINAPI WaitForSingleObject(HANDLE, DWORD);
IMPORT BOOL WINAPI CloseHandle(HANDLE);
IMPORT void WINAPI Sleep(DWORD);
IMPORT BOOL WINAPI QueryPerformanceCounter(LARGE_INTEGER*);
IMPORT HANDLE WINAPI CreateFileW(const unsigned short*, DWORD, DWORD, void*, DWORD, DWORD, HANDLE);
IMPORT BOOL WINAPI ReadFile(HANDLE, void*, DWORD, DWORD*, void*);

static DWORD str_len(const char* s) { DWORD n = 0; while (s[n]) ++n; return n; }
static void say(const char* s) { DWORD w; WriteFile(GetStdHandle((DWORD)-11), s, str_len(s), &w, 0); }

static void* teb_self(void) { void* p; __asm__ volatile("movq %%gs:0x30, %0" : "=r"(p)); return p; }

static DWORD g_tls;
static volatile int g_counter = 7;  // a writable static (.data)
static volatile int* volatile g_counter_ptr = &g_counter;  // an absolute address: needs a DIR64 relocation

static DWORD WINAPI worker(void* arg) {
    // Own TEB, own TLS slot value.
    if (teb_self() == arg) return 101;
    if (TlsGetValue(g_tls) != 0) return 102;
    TlsSetValue(g_tls, (void*)0x2222);
    Sleep(1);
    if (TlsGetValue(g_tls) != (void*)0x2222) return 103;
    g_counter += 1;
    return 55;
}

static unsigned run(void) {
    say("guest: hello from Windows x64 code\n");
    void* teb = teb_self();
    if (!teb) return 1;
    if (g_counter != 7 || *g_counter_ptr != 7) return 2;

    HANDLE heap = GetProcessHeap();
    unsigned char* p = HeapAlloc(heap, 0, 4096);
    if (!p) return 3;
    for (int i = 0; i < 4096; ++i) p[i] = (unsigned char)i;
    if (p[4095] != 255) return 4;
    if (!HeapFree(heap, 0, p)) return 5;

    unsigned char* v = VirtualAlloc(0, 1 << 20, 0x3000 /* COMMIT|RESERVE */, 0x04 /* READWRITE */);
    if (!v) return 6;
    v[(1 << 20) - 1] = 9;
    if (!VirtualFree(v, 0, 0x8000 /* RELEASE */)) return 7;

    g_tls = TlsAlloc();
    if (g_tls == 0xFFFFFFFFu) return 8;
    TlsSetValue(g_tls, (void*)0x1111);
    DWORD tid;
    HANDLE t = CreateThread(0, 0, worker, teb, 0, &tid);
    if (!t) return 9;
    if (WaitForSingleObject(t, 0xFFFFFFFFu) != 0) return 10;
    CloseHandle(t);
    if (TlsGetValue(g_tls) != (void*)0x1111) return 11;
    if (g_counter != 8) return 12;
    if (teb_self() != teb) return 13;

    LARGE_INTEGER a, b;
    QueryPerformanceCounter(&a);
    Sleep(2);
    QueryPerformanceCounter(&b);
    if (b <= a) return 14;

    static const unsigned short path[] = {'g','u','e','s','t','_','p','r','o','b','e','.','t','x','t',0};
    HANDLE f = CreateFileW(path, 0x80000000 /* GENERIC_READ */, 1, 0, 3 /* OPEN_EXISTING */, 0, 0);
    if (f == (HANDLE)-1) return 15;
    char buf[16];
    DWORD got = 0;
    if (!ReadFile(f, buf, sizeof buf, &got, 0) || got != 5 || buf[0] != 'p' || buf[4] != 'e') return 16;
    CloseHandle(f);

    say("guest: all checks passed\n");
    return 0;
}

void mainCRTStartup(void) { ExitProcess(run()); }
