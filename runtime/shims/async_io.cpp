// Asynchronous file I/O and completion ports, see async_io.h.

#include "async_io.h"

#include <cstdio>

#include "shims.h"

namespace {

HMODULE kernel() {
    static const HMODULE module = GetModuleHandleA("kernel32.dll");
    return module;
}

template<typename F>
F host(const char* name) {
    return reinterpret_cast<F>(GetProcAddress(kernel(), name));
}

void log_once(const char* what) {
    static bool logged = false;
    if (logged) return;
    logged = true;
    std::fprintf(stderr, "[rt] kernel32: %s\n", what);
}

}  // namespace

extern "C" HANDLE WINAPI CreateFile2(LPCWSTR name, DWORD access, DWORD share, DWORD creation,
                                     LPCREATEFILE2_EXTENDED_PARAMETERS parameters) {
    using Fn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, DWORD, LPCREATEFILE2_EXTENDED_PARAMETERS);
    if (const Fn fn = host<Fn>("CreateFile2")) return fn(name, access, share, creation, parameters);
    // An older host: build the CreateFileW call the extended parameters describe.
    using CreateFileWFn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
    const CreateFileWFn create_file = host<CreateFileWFn>("CreateFileW");
    if (!create_file) {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
        return INVALID_HANDLE_VALUE;
    }
    const DWORD flags = parameters ? parameters->dwFileFlags : 0;
    return create_file(name, access, share, parameters ? parameters->lpSecurityAttributes : nullptr, creation, flags,
                       parameters ? parameters->hTemplateFile : nullptr);
}

extern "C" BOOL WINAPI ReadFileScatter(HANDLE file, FILE_SEGMENT_ELEMENT* segments, DWORD bytes, LPDWORD reserved,
                                       LPOVERLAPPED overlapped) {
    using Fn = BOOL(WINAPI*)(HANDLE, FILE_SEGMENT_ELEMENT*, DWORD, LPDWORD, LPOVERLAPPED);
    if (const Fn fn = host<Fn>("ReadFileScatter")) return fn(file, segments, bytes, reserved, overlapped);
    log_once("ReadFileScatter is not available on this host");
    SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return FALSE;
}

extern "C" BOOL WINAPI WriteFileGather(HANDLE file, FILE_SEGMENT_ELEMENT* segments, DWORD bytes, LPDWORD reserved,
                                       LPOVERLAPPED overlapped) {
    using Fn = BOOL(WINAPI*)(HANDLE, FILE_SEGMENT_ELEMENT*, DWORD, LPDWORD, LPOVERLAPPED);
    if (const Fn fn = host<Fn>("WriteFileGather")) return fn(file, segments, bytes, reserved, overlapped);
    log_once("WriteFileGather is not available on this host");
    SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return FALSE;
}

extern "C" HANDLE WINAPI CreateIoCompletionPort(HANDLE file, HANDLE existing, ULONG_PTR key, DWORD threads) {
    using Fn = HANDLE(WINAPI*)(HANDLE, HANDLE, ULONG_PTR, DWORD);
    if (const Fn fn = host<Fn>("CreateIoCompletionPort")) return fn(file, existing, key, threads);
    log_once("CreateIoCompletionPort is not available on this host");
    SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return nullptr;
}

extern "C" BOOL WINAPI GetQueuedCompletionStatus(HANDLE port, LPDWORD bytes, PULONG_PTR key,
                                                 LPOVERLAPPED* overlapped, DWORD timeout) {
    using Fn = BOOL(WINAPI*)(HANDLE, LPDWORD, PULONG_PTR, LPOVERLAPPED*, DWORD);
    if (const Fn fn = host<Fn>("GetQueuedCompletionStatus")) return fn(port, bytes, key, overlapped, timeout);
    log_once("GetQueuedCompletionStatus is not available on this host");
    SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return FALSE;
}

extern "C" BOOL WINAPI PostQueuedCompletionStatus(HANDLE port, DWORD bytes, ULONG_PTR key,
                                                  LPOVERLAPPED overlapped) {
    using Fn = BOOL(WINAPI*)(HANDLE, DWORD, ULONG_PTR, LPOVERLAPPED);
    if (const Fn fn = host<Fn>("PostQueuedCompletionStatus")) return fn(port, bytes, key, overlapped);
    log_once("PostQueuedCompletionStatus is not available on this host");
    SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
    return FALSE;
}

SHIM_TABLE(kernel32_async,
    {"kernel32.dll", "CreateFile2", reinterpret_cast<void*>(&CreateFile2)},
    {"kernel32.dll", "ReadFileScatter", reinterpret_cast<void*>(&ReadFileScatter)},
    {"kernel32.dll", "WriteFileGather", reinterpret_cast<void*>(&WriteFileGather)},
    {"kernel32.dll", "CreateIoCompletionPort", reinterpret_cast<void*>(&CreateIoCompletionPort)},
    {"kernel32.dll", "GetQueuedCompletionStatus", reinterpret_cast<void*>(&GetQueuedCompletionStatus)},
    {"kernel32.dll", "PostQueuedCompletionStatus", reinterpret_cast<void*>(&PostQueuedCompletionStatus)},
);
