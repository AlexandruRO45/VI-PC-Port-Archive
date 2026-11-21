// Engine-simulation harness: threads, clocks, file I/O, a deliberately missing import, device + window + WSI.
#include <windows.h>

#pragma function(memset)
extern "C" void* __cdecl memset(void* destination, int value, size_t count) {
    volatile unsigned char* out = static_cast<unsigned char*>(destination);
    for (size_t i = 0; i < count; ++i) out[i] = static_cast<unsigned char>(value);
    return destination;
}

extern "C" __declspec(dllimport) LPVOID WINAPI VirtualAlloc(LPVOID, SIZE_T, DWORD, DWORD);
extern "C" __declspec(dllimport) HANDLE WINAPI CreateThread(LPSECURITY_ATTRIBUTES, SIZE_T, LPTHREAD_START_ROUTINE,
                                                            LPVOID, DWORD, LPDWORD);
extern "C" __declspec(dllimport) DWORD WINAPI WaitForSingleObject(HANDLE, DWORD);
extern "C" __declspec(dllimport) BOOL WINAPI QueryPerformanceCounter(LARGE_INTEGER*);
extern "C" __declspec(dllimport) BOOL WINAPI QueryPerformanceFrequency(LARGE_INTEGER*);
extern "C" __declspec(dllimport) ULONGLONG WINAPI GetTickCount64(void);
extern "C" __declspec(dllimport) HANDLE WINAPI CreateFileW(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD,
                                                           HANDLE);
extern "C" __declspec(dllimport) BOOL WINAPI ReadFile(HANDLE, LPVOID, DWORD, LPDWORD, LPOVERLAPPED);
extern "C" __declspec(dllimport) BOOL WINAPI WriteFile(HANDLE, LPCVOID, DWORD, LPDWORD, LPOVERLAPPED);
extern "C" __declspec(dllimport) BOOL WINAPI CloseHandle(HANDLE);
extern "C" __declspec(dllimport) HWND WINAPI CreateWindowExW(DWORD, LPCWSTR, LPCWSTR, DWORD, int, int, int, int, HWND,
                                                             HMENU, HINSTANCE, LPVOID);
extern "C" __declspec(dllimport) int WINAPI wsprintfA(LPSTR, LPCSTR, ...);
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12CreateDevice(IUnknown*, int, const GUID&, void**);
extern "C" __declspec(dllimport) int __cdecl LeonidaRunAnimation(HWND);
// Deliberately not implemented anywhere: the loader must log it and bind a no-op stub.
extern "C" __declspec(dllimport) DWORD WINAPI XInputGetStateForTests(DWORD, void*);

static DWORD WINAPI worker(void* parameter) {
    int* counter = static_cast<int*>(parameter);
    for (int i = 0; i < 1000; ++i) ++*counter;
    return 0;
}

extern "C" void HarnessEntry();
extern "C" void HarnessEntry() {
    void* memory = VirtualAlloc(nullptr, 1u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    int counter = 0;
    HANDLE thread = CreateThread(nullptr, 0, worker, &counter, 0, nullptr);
    if (thread) WaitForSingleObject(thread, 5000);

    LARGE_INTEGER frequency{}, qpc{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&qpc);

    char buffer[64]{};
    DWORD read = 0;
    HANDLE input = CreateFileW(L"engine_input.txt", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (input != INVALID_HANDLE_VALUE) {
        ReadFile(input, buffer, 32, &read, nullptr);
        CloseHandle(input);
    }

    XInputGetStateForTests(0, nullptr);  // missing: stub + missing_imports.log

    HWND hwnd = CreateWindowExW(0, L"LeonidaGameClass", L"Leonida - engine simulation", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, nullptr, nullptr, nullptr, nullptr);
    GUID id{};
    void* device = nullptr;
    D3D12CreateDevice(nullptr, 0, id, &device);

    char line[256]{};
    const int length = wsprintfA(line, "thread=%d bytes=%lu text=%s qpc=%I64d tick=%I64u\n", counter,
                                 static_cast<unsigned long>(read), buffer, static_cast<long long>(qpc.QuadPart),
                                 static_cast<unsigned long long>(GetTickCount64()));
    HANDLE output = CreateFileW(L"engine_sim.log", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
    if (output != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(output, line, DWORD(length), &written, nullptr);
        CloseHandle(output);
    }
    if (memory && hwnd) LeonidaRunAnimation(hwnd);
}
