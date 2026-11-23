// Compiled only into xmem_test. The runtime has no hook setter or callback storage.
#pragma once

#ifdef LEONIDA_XMEM_TEST_HOOKS
#include <windows.h>

namespace rt::shims::testing {

struct XmemMemoryHooks {
    BOOL(WINAPI* virtual_free)(LPVOID, SIZE_T, DWORD) = nullptr;
    SIZE_T(WINAPI* virtual_query)(LPCVOID, PMEMORY_BASIC_INFORMATION, SIZE_T) = nullptr;
    BOOL(WINAPI* unmap_view2)(HANDLE, PVOID, ULONG) = nullptr;
};

// Changes the callbacks under the physical-memory mutex; null members call the native API.
// Callbacks run with that mutex held and must not re-enter an XMem shim.
void set_xmem_memory_hooks(XmemMemoryHooks hooks);

}  // namespace rt::shims::testing
#endif
