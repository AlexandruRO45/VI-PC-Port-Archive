// Link-time stand-in for the console's xmem.dll. Under `gta6rt --run` the shim registry serves these imports.
// The stub fails allocations and mappings so that a test cannot pass by invoking it directly.
#include <windows.h>

#include <cstddef>
#include <cstdint>

extern "C" __declspec(dllexport) void* __stdcall XMemAlloc(size_t, uint64_t) { return nullptr; }
extern "C" __declspec(dllexport) void __stdcall XMemFree(void*, uint64_t) {}
extern "C" __declspec(dllexport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, uint64_t, DWORD) { return nullptr; }
extern "C" __declspec(dllexport) BOOL WINAPI XMemAllocatePhysicalPages(uint64_t, uint64_t*, uint64_t*) { return FALSE; }
extern "C" __declspec(dllexport) void* WINAPI XMemMapPhysicalPages(void*, uint64_t, const uint64_t*) { return nullptr; }
extern "C" __declspec(dllexport) BOOL WINAPI XMemMapPhysicalPagesScatter(void* const*, uint32_t, const uint64_t*) {
    return FALSE;
}
extern "C" __declspec(dllexport) BOOL WINAPI XMemFreePhysicalPages(uint64_t, const uint64_t*) { return FALSE; }
