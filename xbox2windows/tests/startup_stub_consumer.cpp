// Source-built consumer for the startup trap stubs. Imports by name from chad.dll, xg_xs.dll and
// xmem.dll and calls the export the first argument selects. (XMemTransferMemory and
// D3D12ConfigureVirtualMemoryX have real implementations in the shims and their own tests.)
//   none     -> returns 0 without calling a stub (control)
//   chad     -> XDspActivate(0x11, 0x22, 0x33, 0x44)
//   xg_xs    -> XGSuggestSwizzleMode(0x55, 0, 0, 0)
//   physical -> XMemAllocatePhysicalPages/XMemFreePhysicalPages round trip (no trap expected)
//   dstorage -> DirectStorage multi-page RFC 1950 zlib request and failure controls (Refs #1144)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <array>
#include <fstream>
#include <cstring>
#include <vector>

using u64 = std::uint64_t;

extern "C" __declspec(dllimport) u64 XDspActivate(u64, u64, u64, u64);
extern "C" __declspec(dllimport) u64 XGSuggestSwizzleMode(u64, u64, u64, u64);
extern "C" __declspec(dllimport) int XMemAllocatePhysicalPages(u64 flags, std::size_t* count, u64* pages);
extern "C" __declspec(dllimport) int XMemFreePhysicalPages(std::size_t count, u64* pages);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                                              ULONGLONG attributes, DWORD protection);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count,
                                                                   const ULONG_PTR* tokens);

namespace {

uint32_t adler32_calc(const unsigned char* data, size_t size) {
    uint32_t a = 1, b = 0;
    while (size) {
        size_t chunk = size < 5552 ? size : 5552;
        size -= chunk;
        while (chunk--) {
            a += *data++;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return (b << 16) | a;
}

std::vector<unsigned char> make_stored_zlib(const unsigned char* data, size_t size) {
    std::vector<unsigned char> out;
    out.push_back(0x78);
    out.push_back(0x01);
    size_t pos = 0;
    while (pos < size) {
        size_t chunk = (size - pos > 32768) ? 32768 : (size - pos);
        bool last = (pos + chunk == size);
        out.push_back(last ? 0x01 : 0x00);
        out.push_back(static_cast<unsigned char>(chunk & 0xFF));
        out.push_back(static_cast<unsigned char>((chunk >> 8) & 0xFF));
        unsigned short check = static_cast<unsigned short>(chunk ^ 0xFFFF);
        out.push_back(static_cast<unsigned char>(check & 0xFF));
        out.push_back(static_cast<unsigned char>((check >> 8) & 0xFF));
        out.insert(out.end(), data + pos, data + pos + chunk);
        pos += chunk;
    }
    uint32_t adler = adler32_calc(data, size);
    out.push_back(static_cast<unsigned char>((adler >> 24) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>(adler & 0xFF));
    return out;
}

struct BitWriter {
    std::vector<unsigned char> bytes;
    unsigned bit_buf = 0;
    int bit_cnt = 0;

    void put_bit(int b) {
        bit_buf |= (b & 1) << bit_cnt++;
        if (bit_cnt == 8) {
            bytes.push_back(static_cast<unsigned char>(bit_buf));
            bit_buf = 0;
            bit_cnt = 0;
        }
    }

    void put_code_msb(unsigned code, int bits) {
        for (int i = bits - 1; i >= 0; --i) {
            put_bit((code >> i) & 1);
        }
    }

    void flush() {
        if (bit_cnt > 0) {
            bytes.push_back(static_cast<unsigned char>(bit_buf));
            bit_buf = 0;
            bit_cnt = 0;
        }
    }
};

std::vector<unsigned char> make_fixed_deflate_zlib(unsigned char val, size_t count) {
    BitWriter bw;
    bw.bytes.push_back(0x78);
    bw.bytes.push_back(0x01);
    bw.put_bit(1);
    bw.put_bit(1);
    bw.put_bit(0);
    bw.put_code_msb(48 + val, 8);

    size_t remaining = count - 1;
    while (remaining >= 258) {
        bw.put_code_msb(197, 8); // symbol 285 = length 258
        bw.put_code_msb(0, 5);   // distance 1
        remaining -= 258;
    }
    while (remaining >= 3) {
        size_t len = remaining > 10 ? 10 : remaining;
        unsigned sym = static_cast<unsigned>(257 + (len - 3));
        bw.put_code_msb(sym - 256, 7);
        bw.put_code_msb(0, 5);
        remaining -= len;
    }
    while (remaining > 0) {
        bw.put_code_msb(48 + val, 8);
        --remaining;
    }
    bw.put_code_msb(0, 7); // end of block
    bw.flush();

    std::vector<unsigned char> plain(count, val);
    uint32_t a = adler32_calc(plain.data(), count);
    bw.bytes.push_back(static_cast<unsigned char>((a >> 24) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>((a >> 16) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>((a >> 8) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>(a & 0xFF));
    return bw.bytes;
}

struct Object {
    void** vtable;
};

using DStorageGetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CloseFileFn = void(__stdcall*)(void*);

struct SubmitGate {
    void** vtable = nullptr;
    HANDLE entered = nullptr;
    HANDLE resume = nullptr;
    volatile LONG resumed = 0;
};

long __stdcall hold_worker(void* fence, u64) {
    auto& gate = *static_cast<SubmitGate*>(fence);
    SetEvent(gate.entered);
    if (WaitForSingleObject(gate.resume, 5000) != WAIT_OBJECT_0) return E_FAIL;
    InterlockedExchange(&gate.resumed, 1);
    return S_OK;
}

bool dstorage_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");

    HMODULE xcrdapi = LoadLibraryA("XCRDAPI.dll");
    if (!xcrdapi) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(xcrdapi, "DStorageGetFactory"));
    if (!get_factory) return false;

    void* factory_ptr = nullptr;
    if (get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    const size_t plain_size = 128 * 1024;
    std::vector<unsigned char> plain(plain_size);
    for (size_t i = 0; i < plain_size; ++i) plain[i] = static_cast<unsigned char>((i % 251) ^ 0x5a);
    const auto comp = make_stored_zlib(plain.data(), plain_size);

    const char* cases[] = { "multipage", "short", "missing_grant", "linear_short" };
    for (const char* mode : cases) {
        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "Status", nullptr, &status_ptr) != 0 || !status_ptr)
            return false;
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        auto* queue = static_cast<Object*>(queue_ptr);

        if (std::strcmp(mode, "linear_short") == 0) {
            std::vector<unsigned char> dest(64 * 1024, 0xAA);
            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | 0x400000;
            *reinterpret_cast<void**>(request + 0x08) = dest.data();
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = static_cast<unsigned>(dest.size());
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
            if (hr == 0) return false;
            for (unsigned char b : dest) {
                if (b != 0xAA) return false;
            }
        } else {
            size_t page_count = 2;
            u64 pages[2] = {};
            if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 2) return false;

            ULONG_PTR tokens[2] = { pages[0], pages[1] };
            unsigned destination_size = 128 * 1024;

            void* va = XMemVirtualAlloc(nullptr, 2 * 65536, 0x20402000, 0x21, PAGE_READWRITE);
            if (!va) return false;
            if (XMemMapPhysicalPages(va, 2, tokens) != va) return false;
            std::memset(va, 0xAA, 128 * 1024);

            if (std::strcmp(mode, "short") == 0) {
                destination_size = 64 * 1024;
            } else if (std::strcmp(mode, "missing_grant") == 0) {
                tokens[1] = 0;
            }

            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | (1ull << 21) | 0x400000;
            *reinterpret_cast<ULONG_PTR**>(request + 0x08) = tokens;
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = destination_size;
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);

            if (std::strcmp(mode, "multipage") == 0) {
                if (hr != 0) return false;
                if (std::memcmp(va, plain.data(), plain_size) != 0) return false;
            } else {
                if (hr == 0) return false;
                const auto* bytes = static_cast<const unsigned char*>(va);
                for (size_t i = 0; i < 128 * 1024; ++i) {
                    if (bytes[i] != 0xAA) return false;
                }
            }

            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
        }
    }

    // Test chunked streaming reads across 4 requests on a file source (Refs #1136)
    {
        const size_t total_chunks = 4;
        const size_t chunk_dest_size = 64 * 1024;
        const size_t total_plain_size = total_chunks * chunk_dest_size; // 256 KiB
        std::vector<unsigned char> full_plain(total_plain_size);
        for (size_t i = 0; i < total_plain_size; ++i) full_plain[i] = static_cast<unsigned char>((i % 239) ^ 0xa5);
        const auto full_comp = make_stored_zlib(full_plain.data(), total_plain_size);

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // Padded to 600 KiB so offset 500 KiB can read zeroes from file and test zlib continuation bounds
        std::vector<unsigned char> file_bytes(600 * 1024, 0);
        std::memcpy(file_bytes.data(), full_comp.data(), full_comp.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, static_cast<unsigned>(total_chunks), "ChunkedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        // The last four bytes are zero padding. Only the first request writes;
        // the short read fails, and the existing empty-request path succeeds.
        const u64 offsets[]{file_bytes.size() - 4, file_bytes.size() - 3, UINT64_MAX};
        const unsigned counts[]{4, 4, 0};
        unsigned char destinations[3][6];
        std::memset(destinations, 0xa5, sizeof(destinations));
        for (unsigned i = 0; i < 3; ++i) {
            alignas(8) unsigned char req[0x48]{};
            *reinterpret_cast<void**>(req + 0x08) = destinations[i] + 1;
            *reinterpret_cast<unsigned*>(req + 0x18) = 4;
            *reinterpret_cast<void**>(req + 0x20) = file_obj;
            *reinterpret_cast<u64*>(req + 0x28) = offsets[i];
            *reinterpret_cast<unsigned*>(req + 0x30) = counts[i];
            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        for (int i = 0; i < 200 && !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2); ++i) Sleep(10);
        bool bounds_ok = true;
        for (unsigned i = 0; i < 3; ++i) {
            const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
            bounds_ok &= hr == (i == 1 ? E_FAIL : S_OK);
            for (unsigned j = 0; j < 6; ++j)
                bounds_ok &= destinations[i][j] == (i == 0 && j >= 1 && j <= 4 ? 0 : 0xa5);
        }
        if (!bounds_ok) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        std::vector<std::vector<unsigned char>> chunks(total_chunks, std::vector<unsigned char>(chunk_dest_size, 0));
        for (size_t k = 0; k < total_chunks; ++k) {
            unsigned char req[0x48] = {};
            *reinterpret_cast<u64*>(req + 0x00) = 1; // zlib file request
            *reinterpret_cast<void**>(req + 0x08) = chunks[k].data();
            *reinterpret_cast<u64*>(req + 0x10) = 0;
            *reinterpret_cast<unsigned*>(req + 0x18) = static_cast<unsigned>(chunk_dest_size);
            *reinterpret_cast<void**>(req + 0x20) = file_obj;
            *reinterpret_cast<u64*>(req + 0x28) = k * chunk_dest_size; // offset stepped by destination size
            *reinterpret_cast<unsigned*>(req + 0x30) = static_cast<unsigned>(full_comp.size());
            *reinterpret_cast<unsigned*>(req + 0x38) = static_cast<unsigned>(total_plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, static_cast<unsigned>(k));
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(total_chunks - 1))) break;
            Sleep(10);
        }
        bool ok = true;
        for (size_t k = 0; k < total_chunks; ++k) {
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(k))) ok = false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, static_cast<unsigned>(k));
            if (hr != 0) ok = false;
            if (std::memcmp(chunks[k].data(), full_plain.data() + k * chunk_dest_size, chunk_dest_size) != 0) ok = false;
        }
        if (!ok) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        // Negative control: chunk request with out-of-bounds or wrapping offset must fail and leave destination untouched
        void* neg_status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ChunkNegStatus", nullptr, &neg_status_ptr) != 0 || !neg_status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* neg_status = static_cast<Object*>(neg_status_ptr);

        std::vector<unsigned char> dest1(64 * 1024, 0xEE); // sentinel
        std::vector<unsigned char> dest2(64 * 1024, 0xDD); // sentinel for wrap test

        // Piece 1 with invalid offset (500 KiB > 256 KiB)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 500 * 1024;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(total_plain_size);

        // Piece 2 with near-wrapping offset (UINT64_MAX - 0xFFFF)
        unsigned char req2[0x48] = {};
        *reinterpret_cast<u64*>(req2 + 0x00) = 1;
        *reinterpret_cast<void**>(req2 + 0x08) = dest2.data();
        *reinterpret_cast<unsigned*>(req2 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req2 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req2 + 0x28) = 0xFFFFFFFFFFFF0000ull;
        *reinterpret_cast<unsigned*>(req2 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req2 + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req2);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 1)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0) ||
            !reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 1)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr1 = reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 0);
        long hr2 = reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 1);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (hr1 == 0 || hr2 == 0) return false;
        for (unsigned char b : dest1) {
            if (b != 0xEE) return false;
        }
        for (unsigned char b : dest2) {
            if (b != 0xDD) return false;
        }
    }

    // Control: standalone valid stream sharing cache-key fields decodes its own bytes (Refs #1214)
    {
        const size_t stream1_plain_size = 64 * 1024;
        const size_t stream2_plain_size = 64 * 1024;
        std::vector<unsigned char> plain1(stream1_plain_size, 0x33);
        std::vector<unsigned char> plain2(stream2_plain_size, 0x77);
        const auto comp1 = make_stored_zlib(plain1.data(), stream1_plain_size);
        const auto comp2 = make_stored_zlib(plain2.data(), stream2_plain_size);
        const unsigned shared_compressed_size = static_cast<unsigned>(comp1.size());

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        const size_t offset2 = 128 * 1024;
        std::vector<unsigned char> file_bytes(offset2 + comp2.size(), 0);
        std::memcpy(file_bytes.data(), comp1.data(), comp1.size());
        std::memcpy(file_bytes.data() + offset2, comp2.data(), comp2.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 3, "SharedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest1_0(32 * 1024, 0);
        std::vector<unsigned char> dest1_1(32 * 1024, 0);
        std::vector<unsigned char> dest2(64 * 1024, 0);

        // Stream 1 piece 0: offset 0, compressed size S, destination 32 KiB
        unsigned char req1_0[0x48] = {};
        *reinterpret_cast<u64*>(req1_0 + 0x00) = 1;
        *reinterpret_cast<void**>(req1_0 + 0x08) = dest1_0.data();
        *reinterpret_cast<unsigned*>(req1_0 + 0x18) = 32 * 1024;
        *reinterpret_cast<void**>(req1_0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1_0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req1_0 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1_0 + 0x38) = static_cast<unsigned>(stream1_plain_size);

        // Stream 1 piece 1: offset 32 KiB, compressed size S, destination 32 KiB
        unsigned char req1_1[0x48] = {};
        *reinterpret_cast<u64*>(req1_1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1_1 + 0x08) = dest1_1.data();
        *reinterpret_cast<unsigned*>(req1_1 + 0x18) = 32 * 1024;
        *reinterpret_cast<void**>(req1_1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1_1 + 0x28) = 32 * 1024;
        *reinterpret_cast<unsigned*>(req1_1 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1_1 + 0x38) = static_cast<unsigned>(stream1_plain_size);

        // Stream 2: offset 128 KiB, exact same compressed size S, destination 64 KiB
        unsigned char req2[0x48] = {};
        *reinterpret_cast<u64*>(req2 + 0x00) = 1;
        *reinterpret_cast<void**>(req2 + 0x08) = dest2.data();
        *reinterpret_cast<unsigned*>(req2 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req2 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req2 + 0x28) = offset2;
        *reinterpret_cast<unsigned*>(req2 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req2 + 0x38) = static_cast<unsigned>(stream2_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1_0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1_1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req2);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 2);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr0 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
        long hr1 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1);
        long hr2 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 2);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (hr0 != 0 || hr1 != 0 || hr2 != 0) return false;
        for (unsigned char b : dest1_0) {
            if (b != 0x33) return false;
        }
        for (unsigned char b : dest1_1) {
            if (b != 0x33) return false;
        }
        for (unsigned char b : dest2) {
            if (b != 0x77) return false;
        }
    }

    // Control: standalone stream with damaged checksum at matching offset fails and leaves destination unchanged (Refs #1214)
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp1 = make_fixed_deflate_zlib(0x33, stream_plain_size);
        auto comp_bad = make_fixed_deflate_zlib(0x77, stream_plain_size);
        // Corrupt the Adler-32 checksum in comp_bad:
        comp_bad.back() ^= 0xFF;
        const unsigned shared_compressed_size = static_cast<unsigned>(comp1.size());

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        const size_t offset_bad = 64 * 1024; // Matching continuation offset (0 + 64 KiB)
        std::vector<unsigned char> file_bytes(offset_bad + comp_bad.size(), 0);
        std::memcpy(file_bytes.data(), comp1.data(), comp1.size());
        std::memcpy(file_bytes.data() + offset_bad, comp_bad.data(), comp_bad.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "DamagedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest1(64 * 1024, 0);
        std::vector<unsigned char> dest_bad(64 * 1024, 0xBB); // Sentinel

        // Stream 1 piece 0: offset 0, compressed size S, destination 64 KiB (inflates 128 KiB of 0x33, caches stream)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // Standalone stream at matching continuation offset 64 KiB with damaged checksum
        unsigned char req_bad[0x48] = {};
        *reinterpret_cast<u64*>(req_bad + 0x00) = 1;
        *reinterpret_cast<void**>(req_bad + 0x08) = dest_bad.data();
        *reinterpret_cast<unsigned*>(req_bad + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_bad + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_bad + 0x28) = offset_bad;
        *reinterpret_cast<unsigned*>(req_bad + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req_bad + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_bad);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr0 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
        long hr_bad = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // Stream 1 piece 0 must succeed; the damaged standalone stream must fail with non-zero HRESULT
        if (hr0 != 0 || hr_bad == 0) return false;
        for (unsigned char b : dest1) {
            if (b != 0x33) return false;
        }
        // Destination buffer of the damaged stream must remain completely unchanged (sentinel 0xBB)
        for (unsigned char b : dest_bad) {
            if (b != 0xBB) return false;
        }
    }

    // Test short final destination in a multi-piece stream vs invalid oversized decode targeting physical pages (Refs #1144)
    {
        // Multi-piece stream: total plain size = 100 KiB (piece 0 = 64 KiB, piece 1 = 36 KiB short final piece)
        const size_t total_plain_size = 100 * 1024;
        std::vector<unsigned char> full_plain(total_plain_size);
        for (size_t i = 0; i < total_plain_size; ++i) full_plain[i] = static_cast<unsigned char>((i % 251) ^ 0x3c);
        const auto full_comp = make_stored_zlib(full_plain.data(), total_plain_size);

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        std::vector<unsigned char> file_bytes(256 * 1024, 0);
        std::memcpy(file_bytes.data(), full_comp.data(), full_comp.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        // Allocate 2 physical pages for the valid multi-piece stream: page 0 (piece 0) and page 1 (piece 1)
        size_t page_count = 2;
        u64 pages[2] = {};
        if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        ULONG_PTR tokens_all[2] = { pages[0], pages[1] };
        void* va = XMemVirtualAlloc(nullptr, 2 * 65536, 0x20402000, 0x21, PAGE_READWRITE);
        if (!va || XMemMapPhysicalPages(va, 2, tokens_all) != va) {
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        // Prefill both physical pages with guard byte 0xAA
        std::memset(va, 0xAA, 2 * 65536);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ShortFinalStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        // Piece 0: full 64 KiB physical page destination
        ULONG_PTR tokens0[1] = { pages[0] };
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = 1 | (1ull << 21); // zlib | page_destination
        *reinterpret_cast<ULONG_PTR**>(req0 + 0x08) = tokens0;
        *reinterpret_cast<u64*>(req0 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req0 + 0x38) = static_cast<unsigned>(total_plain_size);

        // Piece 1: short final destination below 64 KiB (36 KiB) with 1 page grant; neighboring token left null
        ULONG_PTR tokens1[2] = { pages[1], 0 };
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1 | (1ull << 21); // zlib | page_destination
        *reinterpret_cast<ULONG_PTR**>(req1 + 0x08) = tokens1;
        *reinterpret_cast<u64*>(req1 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 36 * 1024; // Short destination capacity (< 64 KiB)
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 64 * 1024; // Stepped by previous chunk size
        *reinterpret_cast<unsigned*>(req1 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) break;
            Sleep(10);
        }
        bool valid_complete = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) &&
                              reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1);
        long hr0 = valid_complete ? reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0) : -1;
        long hr1 = valid_complete ? reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1) : -1;

        bool valid_ok = valid_complete && hr0 == 0 && hr1 == 0;
        const auto* page0_bytes = static_cast<const unsigned char*>(va);
        const auto* page1_bytes = static_cast<const unsigned char*>(va) + 65536;

        // Check piece 0 exact match
        if (valid_ok && std::memcmp(page0_bytes, full_plain.data(), 64 * 1024) != 0) valid_ok = false;
        // Check piece 1 exact 36 KiB match
        if (valid_ok && std::memcmp(page1_bytes, full_plain.data() + 64 * 1024, 36 * 1024) != 0) valid_ok = false;
        // Check guard bytes in page 1 beyond 36 KiB remain intact (0xAA)
        if (valid_ok) {
            for (size_t i = 36 * 1024; i < 65536; ++i) {
                if (page1_bytes[i] != 0xAA) {
                    valid_ok = false;
                    break;
                }
            }
        }
        // Neighboring token in tokens1 must have remained untouched (0)
        if (valid_ok && tokens1[1] != 0) valid_ok = false;

        // Negative control: invalid oversized decode targeting short physical page destination
        // (100 KiB standalone stream into 36 KiB destination with only 1 page grant)
        void* neg_status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "OversizedNegStatus", nullptr, &neg_status_ptr) != 0 || !neg_status_ptr) {
            valid_ok = false;
        }
        auto* neg_status = static_cast<Object*>(neg_status_ptr);

        // Reset page 1 to guard byte 0x55
        std::memset(static_cast<unsigned char*>(va) + 65536, 0x55, 65536);
        ULONG_PTR tokens_neg[2] = { pages[1], 0 }; // Neighboring grant is 0

        unsigned char req_oversized[0x48] = {};
        *reinterpret_cast<u64*>(req_oversized + 0x00) = 1 | (1ull << 21) | 0x400000; // zlib | page_destination | memory_source
        *reinterpret_cast<ULONG_PTR**>(req_oversized + 0x08) = tokens_neg;
        *reinterpret_cast<u64*>(req_oversized + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req_oversized + 0x18) = 36 * 1024; // Capacity 36 KiB < 100 KiB decode
        *reinterpret_cast<const unsigned char**>(req_oversized + 0x20) = full_comp.data();
        *reinterpret_cast<u64*>(req_oversized + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req_oversized + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req_oversized + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_oversized);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0)) break;
            Sleep(10);
        }
        bool neg_complete = reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0);
        long hr_neg = neg_complete ? reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 0) : 0;

        // Oversized decode must fail with non-zero HRESULT
        bool neg_ok = neg_complete && hr_neg != 0;
        // All 64 KiB of destination page must remain completely untouched (guard byte 0x55)
        for (size_t i = 0; i < 65536; ++i) {
            if (page1_bytes[i] != 0x55) {
                neg_ok = false;
                break;
            }
        }
        // Neighboring token must remain untouched (0)
        if (tokens_neg[1] != 0) neg_ok = false;

        XMemMapPhysicalPages(va, 2, nullptr);
        VirtualFree(va, 0, MEM_RELEASE);
        XMemFreePhysicalPages(2, pages);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (!valid_ok || !neg_ok) return false;
    }

    // Control: valid chunked submission followed by a separate Submit with same-size non-zlib read at matching offset fails cleanly (Refs #1214)
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp = make_fixed_deflate_zlib(0x44, stream_plain_size);
        const unsigned comp_size = static_cast<unsigned>(comp.size());
        const size_t offset_cont = 64 * 1024; // Matching continuation offset (0 + 64 KiB)

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // File layout:
        // Offset 0: valid compressed stream (size comp_size)
        // Offset 64 KiB: non-zlib bytes (0xEE), size comp_size
        std::vector<unsigned char> file_bytes(offset_cont + comp_size, 0xEE);
        std::memcpy(file_bytes.data(), comp.data(), comp_size);

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr1 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ChunkedStatus1", nullptr, &status_ptr1) != 0 || !status_ptr1) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status1 = static_cast<Object*>(status_ptr1);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest0(64 * 1024, 0);
        std::vector<unsigned char> dest1(64 * 1024, 0);

        // Piece 0: offset 0, compressed size S, destination 64 KiB
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = 1;
        *reinterpret_cast<void**>(req0 + 0x08) = dest0.data();
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req0 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // Piece 1: offset 64 KiB, compressed size S, destination 64 KiB
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = offset_cont;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // First Submit: valid 2-slice chunked submission
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 1)) break;
            Sleep(10);
        }
        bool comp1_ok = reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0) &&
                        reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 1);
        long hr0 = comp1_ok ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 0) : -1;
        long hr1 = comp1_ok ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 1) : -1;

        if (hr0 != 0 || hr1 != 0) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        for (unsigned char b : dest0) if (b != 0x44) return false;
        for (unsigned char b : dest1) if (b != 0x44) return false;

        // Separate Submit: same queue and same live file, same-size non-zlib read at matching offset 64 KiB
        void* status_ptr2 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "SeparateStatus", nullptr, &status_ptr2) != 0 || !status_ptr2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status2 = static_cast<Object*>(status_ptr2);

        std::vector<unsigned char> dest_unrelated(64 * 1024, 0x99); // Sentinel

        unsigned char req_unrelated[0x48] = {};
        *reinterpret_cast<u64*>(req_unrelated + 0x00) = 1; // zlib decompression requested
        *reinterpret_cast<void**>(req_unrelated + 0x08) = dest_unrelated.data();
        *reinterpret_cast<unsigned*>(req_unrelated + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_unrelated + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_unrelated + 0x28) = offset_cont; // Matching continuation offset
        *reinterpret_cast<unsigned*>(req_unrelated + 0x30) = comp_size; // Same size
        *reinterpret_cast<unsigned*>(req_unrelated + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_unrelated);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status2, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0)) break;
            Sleep(10);
        }
        bool comp2_ok = reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0);
        long hr_unrelated = comp2_ok ? reinterpret_cast<GetHResultFn>(status2->vtable[4])(status2, 0) : 0;

        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // The separate read must fail with non-zero HRESULT and leave its sentinel untouched
        if (!comp2_ok || hr_unrelated == 0) return false;
        for (unsigned char b : dest_unrelated) {
            if (b != 0x99) return false;
        }
    }

    // Control: deterministic boundary test across two Submit() calls (Refs #1214, #1235)
    // First Submit(): compressed stream into undersized destination buffer (no continuation in this batch).
    // Second Submit(): matching non-zlib request at the continuation offset, queued while the first batch is gated.
    // Both requests must fail cleanly, and both destination sentinel buffers must remain untouched.
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp = make_fixed_deflate_zlib(0x5A, stream_plain_size);
        const unsigned comp_size = static_cast<unsigned>(comp.size());
        const size_t offset_cont = 64 * 1024; // Matching continuation offset (0 + 64 KiB)

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // File layout:
        // Offset 0: valid compressed stream (size comp_size)
        // Offset 64 KiB: non-zlib bytes (0xEE), size comp_size
        std::vector<unsigned char> file_bytes(offset_cont + comp_size, 0xEE);
        std::memcpy(file_bytes.data(), comp.data(), comp_size);

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr1 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "UndersizedStatus", nullptr, &status_ptr1) != 0 || !status_ptr1) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status1 = static_cast<Object*>(status_ptr1);

        void* status_ptr2 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "MatchingStatus", nullptr, &status_ptr2) != 0 || !status_ptr2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status2 = static_cast<Object*>(status_ptr2);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest_undersized(64 * 1024, 0xA5); // Sentinel 0xA5
        std::vector<unsigned char> dest_matching(64 * 1024, 0xB6);   // Sentinel 0xB6

        // First Submit: compressed stream into undersized destination buffer (64 KiB < 128 KiB uncompressed)
        // with NO continuation requests queued in this batch.
        unsigned char req_undersized[0x48] = {};
        *reinterpret_cast<u64*>(req_undersized + 0x00) = 1;
        *reinterpret_cast<void**>(req_undersized + 0x08) = dest_undersized.data();
        *reinterpret_cast<unsigned*>(req_undersized + 0x18) = 64 * 1024; // Undersized capacity
        *reinterpret_cast<void**>(req_undersized + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_undersized + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req_undersized + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req_undersized + 0x38) = static_cast<unsigned>(stream_plain_size);

        // The signal callback holds the worker at the head of the first batch. It uses static
        // storage so even a failed test can exit without abandoning a pointer to the stack.
        static SubmitGate gate;
        static void* fence_vtable[11] = {};
        fence_vtable[10] = reinterpret_cast<void*>(&hold_worker);
        gate.vtable = fence_vtable;
        gate.entered = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        gate.resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!gate.entered || !gate.resume) return false;
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &gate, 1);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_undersized);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        if (WaitForSingleObject(gate.entered, 3000) != WAIT_OBJECT_0) {
            SetEvent(gate.resume);
            return false;
        }

        // Immediately in second Submit: matching non-zlib request at continuation offset
        unsigned char req_matching[0x48] = {};
        *reinterpret_cast<u64*>(req_matching + 0x00) = 1; // zlib decompression requested
        *reinterpret_cast<void**>(req_matching + 0x08) = dest_matching.data();
        *reinterpret_cast<unsigned*>(req_matching + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_matching + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_matching + 0x28) = offset_cont; // Matching continuation offset
        *reinterpret_cast<unsigned*>(req_matching + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req_matching + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_matching);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status2, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        SetEvent(gate.resume);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0) &&
                reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0)) break;
            Sleep(10);
        }
        bool complete1 = reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0);
        bool complete2 = reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0);
        long hr1 = complete1 ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 0) : 0;
        long hr2 = complete2 ? reinterpret_cast<GetHResultFn>(status2->vtable[4])(status2, 0) : 0;
        if (complete1 && complete2) {
            CloseHandle(gate.resume);
            CloseHandle(gate.entered);
        }

        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // Both requests must fail cleanly
        if (!complete1 || hr1 == 0 || !InterlockedCompareExchange(&gate.resumed, 0, 0)) return false;
        if (!complete2 || hr2 == 0) return false;

        // Both sentinels must remain completely untouched
        for (unsigned char b : dest_undersized) {
            if (b != 0xA5) return false;
        }
        for (unsigned char b : dest_matching) {
            if (b != 0xB6) return false;
        }
    }

    return true;
}

// The Python fixture supplies absolute ranges from each synthetic file's RDS5
// directory. Keep one real queue and one open file object per input for all batches.
bool dstorage_directory_batches()
{
    struct Request { unsigned batch, file; unsigned long long offset; unsigned length; };
    std::vector<Request> requests;
    Request r{};
    while (std::scanf("%u%u%llu%u", &r.batch, &r.file, &r.offset, &r.length) == 4) {
        if (r.file >= 2 || !r.length || requests.size() >= 32) return false;
        requests.push_back(r);
    }
    if (requests.empty() || !std::feof(stdin)) return false;
    HMODULE library = LoadLibraryW(L"XCRDAPI.dll");
    if (!library) return false;
    auto factory_export = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(library, "DStorageGetFactory"));
    void* object = nullptr;
    if (!factory_export || factory_export(nullptr, &object) || !object) return false;
    auto* factory = static_cast<Object*>(object);
    void* files[2]{};
    const wchar_t* names[]{L"small.ttpf", L"large.ttpf"};
    for (unsigned i = 0; i < 2; ++i)
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, names[i], nullptr, &files[i])) return false;
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, nullptr, nullptr, &object)) return false;
    auto* queue = static_cast<Object*>(object);
    const unsigned count = static_cast<unsigned>(requests.size());
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, count, "directory", nullptr, &object)) return false;
    auto* status = static_cast<Object*>(object);

    constexpr unsigned page = 4096, guard = 16;
    std::vector<std::array<unsigned char, page + 2 * guard>> output(count);
    for (unsigned i = 0; i < count; ++i) {
        const auto& request = requests[i];
        output[i].fill(0xa5);
        alignas(8) unsigned char record[0x48]{};
        const u64 fields[]{1, reinterpret_cast<u64>(output[i].data() + guard), 0, page,
                           reinterpret_cast<u64>(files[request.file]), request.offset, request.length};
        std::memcpy(record, fields, sizeof(fields));
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, record);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
        if (i + 1 == count || requests[i + 1].batch != request.batch)
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    }
    const auto deadline = GetTickCount64() + 5000;
    while (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, count - 1)) {
        if (GetTickCount64() >= deadline) return false;
        Sleep(1);
    }
    std::ofstream readback("readback.bin", std::ios::binary);
    for (unsigned i = 0; i < count; ++i) {
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::printf("%08lx\n", static_cast<unsigned long>(hr));
        readback.write(reinterpret_cast<const char*>(output[i].data()), output[i].size());
    }
    for (void* file : files)
        reinterpret_cast<CloseFileFn>(static_cast<Object*>(file)->vtable[3])(file);
    readback.close();
    return readback.good();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const char* which = argv[1];
    if (!std::strcmp(which, "none")) return 0;
    if (!std::strcmp(which, "chad")) return static_cast<int>(XDspActivate(0x11, 0x22, 0x33, 0x44));
    if (!std::strcmp(which, "xg_xs")) return static_cast<int>(XGSuggestSwizzleMode(0x55, 0, 0, 0));
    if (!std::strcmp(which, "physical")) {
        std::size_t count = 4;
        u64 pages[4] = {};
        if (!XMemAllocatePhysicalPages(0x20400000ull /* 64 KiB pages */, &count, pages) || count != 4) return 3;
        for (u64 page : pages)
            if (!page) return 4;
        if (!XMemFreePhysicalPages(count, pages)) return 5;
        if (XMemFreePhysicalPages(count, pages)) return 6;  // double free must be refused
        std::puts("physical ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage-directory")) return dstorage_directory_batches() ? 0 : 8;
    if (!std::strcmp(which, "dstorage")) {
        if (!dstorage_checks()) return 7;
        std::puts("dstorage ok");
        return 0;
    }
    return 2;
}
