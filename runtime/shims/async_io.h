// Asynchronous file I/O and completion ports: the calls a streaming engine (RAGE included) makes through kernel32.
//
// The file and port handles stay host handles: the shim maps the console-visible entry points onto the host's
// implementations, which already give the overlapped semantics a title expects. Resolving the host function through
// GetProcAddress (never a direct call) matters when this module is linked into a shim DLL that exports the same
// names, because a direct call would land back in the shim.
#pragma once

#include <windows.h>

extern "C" HANDLE WINAPI CreateFile2(LPCWSTR name, DWORD access, DWORD share, DWORD creation,
                                     LPCREATEFILE2_EXTENDED_PARAMETERS parameters);
extern "C" BOOL WINAPI ReadFileScatter(HANDLE file, FILE_SEGMENT_ELEMENT* segments, DWORD bytes,
                                       LPDWORD reserved, LPOVERLAPPED overlapped);
extern "C" BOOL WINAPI WriteFileGather(HANDLE file, FILE_SEGMENT_ELEMENT* segments, DWORD bytes,
                                       LPDWORD reserved, LPOVERLAPPED overlapped);
extern "C" HANDLE WINAPI CreateIoCompletionPort(HANDLE file, HANDLE existing, ULONG_PTR key, DWORD threads);
extern "C" BOOL WINAPI GetQueuedCompletionStatus(HANDLE port, LPDWORD bytes, PULONG_PTR key,
                                                 LPOVERLAPPED* overlapped, DWORD timeout);
extern "C" BOOL WINAPI PostQueuedCompletionStatus(HANDLE port, DWORD bytes, ULONG_PTR key,
                                                  LPOVERLAPPED overlapped);
