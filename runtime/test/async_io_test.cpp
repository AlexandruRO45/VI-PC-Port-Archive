// Checks the registered kernel32 asynchrony shims through the shim table: CreateFile2 opens an overlapped file,
// WriteFileGather writes two page-sized segments in one call, a completion port reports it, ReadFileScatter reads the
// pages back into two more segments, and PostQueuedCompletionStatus/GetQueuedCompletionStatus round-trip a packet.
// The file lives in the working directory and is removed at the end.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "shims.h"

namespace {

using CreateFile2Fn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, DWORD, LPCREATEFILE2_EXTENDED_PARAMETERS);
using ReadFileScatterFn = BOOL(WINAPI*)(HANDLE, FILE_SEGMENT_ELEMENT*, DWORD, LPDWORD, LPOVERLAPPED);
using WriteFileGatherFn = BOOL(WINAPI*)(HANDLE, FILE_SEGMENT_ELEMENT*, DWORD, LPDWORD, LPOVERLAPPED);
using CreatePortFn = HANDLE(WINAPI*)(HANDLE, HANDLE, ULONG_PTR, DWORD);
using GetQueuedFn = BOOL(WINAPI*)(HANDLE, LPDWORD, PULONG_PTR, LPOVERLAPPED*, DWORD);
using PostQueuedFn = BOOL(WINAPI*)(HANDLE, DWORD, ULONG_PTR, LPOVERLAPPED);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template<typename F>
F resolve(const char* name) {
    void* function = rt::shims::find("kernel32.dll", name);
    const std::string what = std::string("kernel32.dll!") + name + " is registered";
    check(function != nullptr, what.c_str());
    return reinterpret_cast<F>(function);
}

struct Page {
    void* memory = nullptr;
    Page() { memory = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }
    ~Page() {
        if (memory) VirtualFree(memory, 0, MEM_RELEASE);
    }
    std::uint8_t* bytes() const { return static_cast<std::uint8_t*>(memory); }
};

// Enters `count` pages as the segment array the calls take; the array ends with a null element.
FILE_SEGMENT_ELEMENT* segments(FILE_SEGMENT_ELEMENT* out, const Page* const* pages, int count) {
    for (int i = 0; i < count; ++i) out[i].Buffer = pages[i]->bytes();
    out[count].Buffer = nullptr;
    return out;
}

}  // namespace

int main() {
    const auto create_file2 = resolve<CreateFile2Fn>("CreateFile2");
    const auto read_scatter = resolve<ReadFileScatterFn>("ReadFileScatter");
    const auto write_gather = resolve<WriteFileGatherFn>("WriteFileGather");
    const auto create_port = resolve<CreatePortFn>("CreateIoCompletionPort");
    const auto get_queued = resolve<GetQueuedFn>("GetQueuedCompletionStatus");
    const auto post_queued = resolve<PostQueuedFn>("PostQueuedCompletionStatus");
    if (!create_file2 || !read_scatter || !write_gather || !create_port || !get_queued || !post_queued) return 1;

    const wchar_t* const path = L"async_io_test_file.bin";
    DeleteFileW(path);

    // CreateFile2 with the parameters a streaming engine passes: an overlapped file.
    CREATEFILE2_EXTENDED_PARAMETERS parameters{};
    parameters.dwSize = sizeof(parameters);
    parameters.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    parameters.dwFileFlags = FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING;  // scatter/gather needs unbuffered I/O
    const HANDLE file = create_file2(path, GENERIC_READ | GENERIC_WRITE, 0, CREATE_ALWAYS, &parameters);
    check(file != INVALID_HANDLE_VALUE, "CreateFile2 opens an overlapped, unbuffered file");
    if (file == INVALID_HANDLE_VALUE) {
        return 1;
    }
    // Unbuffered writes cannot grow the file, so its length is set before the gather.
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = 8192;
    check(SetFileInformationByHandle(file, FileEndOfFileInfo, &eof, sizeof(eof)) != FALSE,
          "the file is extended to the two pages it will hold");

    // The port the engine waits on; the file's completions report the key given here.
    const HANDLE port = create_port(file, nullptr, 1, 1);
    check(port != nullptr, "CreateIoCompletionPort associates the file with a new port");
    if (!port) {
        CloseHandle(file);
        return 1;
    }

    // Gather-write two pages with different patterns in one call.
    Page first, second;
    check(first.memory && second.memory, "two page-aligned buffers");
    for (int i = 0; i < 4096; ++i) {
        first.bytes()[i] = std::uint8_t(i & 0xFF);
        second.bytes()[i] = std::uint8_t(0xFF - (i & 0xFF));
    }
    const Page* const written[2] = {&first, &second};
    FILE_SEGMENT_ELEMENT write_segments[3];
    OVERLAPPED write_overlapped{};
    const BOOL started = write_gather(file, segments(write_segments, written, 2), 8192, nullptr, &write_overlapped);
    check(started || GetLastError() == ERROR_IO_PENDING, "WriteFileGather starts an overlapped write of two pages");

    DWORD bytes = 0;
    ULONG_PTR key = 0;
    LPOVERLAPPED completed = nullptr;
    check(get_queued(port, &bytes, &key, &completed, 5000) != FALSE, "the gathered write completes");
    check(bytes == 8192, "the completion reports both pages");
    check(key == 1, "the completion carries the file's key");
    check(completed == &write_overlapped, "the completion carries the OVERLAPPED the write was started with");

    // Scatter-read the pages back into two new ones and check the patterns.
    Page read_first, read_second;
    const Page* const read_pages[2] = {&read_first, &read_second};
    FILE_SEGMENT_ELEMENT read_segments[3];
    OVERLAPPED read_overlapped{};
    const BOOL read_started = read_scatter(file, segments(read_segments, read_pages, 2), 8192, nullptr,
                                           &read_overlapped);
    check(read_started || GetLastError() == ERROR_IO_PENDING, "ReadFileScatter starts an overlapped read of two pages");
    check(get_queued(port, &bytes, &key, &completed, 5000) != FALSE, "the scattered read completes");
    check(bytes == 8192, "the read completion reports both pages");
    check(std::memcmp(read_first.bytes(), first.bytes(), 4096) == 0, "the first page round trips");
    check(std::memcmp(read_second.bytes(), second.bytes(), 4096) == 0, "the second page round trips");

    // A packet posted by hand, the way a thread pool wakes its worker.
    check(post_queued(port, 1234, 7, nullptr) != FALSE, "PostQueuedCompletionStatus accepts a packet");
    bytes = 0;
    key = 0;
    completed = nullptr;
    check(get_queued(port, &bytes, &key, &completed, 5000) != FALSE, "the posted packet is dequeued");
    check(bytes == 1234 && key == 7 && completed == nullptr,
          "the packet carries the posted bytes, key and a null OVERLAPPED");

    CloseHandle(port);
    CloseHandle(file);
    DeleteFileW(path);

    if (g_failures) {
        std::fprintf(stderr, "async_io_test: %d failure(s) of %d checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("async_io_test: %d checks passed\n", g_checks);
    return 0;
}
