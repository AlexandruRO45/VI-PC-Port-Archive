// Test harness image for the PE loader: imports VirtualAlloc (kernel32), CreateWindowExW (user32),
// D3D12CreateDevice and LeonidaRunAnimation (d3d12). Custom entry, no CRT.
#include <windows.h>

extern "C" __declspec(dllimport) LPVOID WINAPI VirtualAlloc(LPVOID, SIZE_T, DWORD, DWORD);
extern "C" __declspec(dllimport) HWND WINAPI CreateWindowExW(DWORD, LPCWSTR, LPCWSTR, DWORD, int, int, int, int, HWND,
                                                             HMENU, HINSTANCE, LPVOID);
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12CreateDevice(IUnknown*, int, const GUID&, void**);
extern "C" __declspec(dllimport) int __cdecl LeonidaRunAnimation(HWND);

extern "C" void HarnessEntry();
extern "C" void HarnessEntry() {
    void* memory = VirtualAlloc(nullptr, 1u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    HWND hwnd = CreateWindowExW(0, L"LeonidaGameClass", L"Leonida - PE harness (Vulkan WSI)", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, nullptr, nullptr, nullptr, nullptr);
    if (!hwnd) return;
    ShowWindow(hwnd, SW_SHOW);
    GUID id{};
    void* device = nullptr;
    D3D12CreateDevice(nullptr, 0, id, &device);
    if (!memory) return;
    LeonidaRunAnimation(hwnd);
}
