// Fixed-size source-built directory batch host. The executable computes both request ranges.
#include "quad_frames.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdint>

using Word = std::uint64_t;
extern "C" __declspec(dllimport) HRESULT WINAPI DStorageGetFactory(const GUID*, void**);

namespace {
struct Object { void** slots; };
using Ptr = Object*;
template<unsigned slot, class Result, class... Args>
Result invoke(Ptr self, Args... args)
{
    return reinterpret_cast<Result(WINAPI*)(Ptr, Args...)>(self->slots[slot])(self, args...);
}
struct Inputs { Word first, second, end, first_bias; };
Inputs inputs{};
Ptr file = nullptr, queue = nullptr, status = nullptr;
constexpr unsigned page = 4096;
std::array<unsigned char, 2 * page> blocks;
unsigned count = 0;
FILE* trace = nullptr;

class CloseBatch {
public:
    ~CloseBatch() {
        if (file) invoke<3, void>(file);
        for (Ptr object : {file, status, queue})
            if (object) invoke<2, unsigned>(object);
        if (trace) std::fclose(trace);
    }
};
}

extern "C" __declspec(dllexport) const Inputs* directory_input()
{
    // Read the first three compressed boundary words verbatim, after the raw metadata boundary.
    std::ifstream input("fixture.ttpf", std::ios::binary);
    char tag[4]{};
    unsigned boundaries = 0;
    if (!input.read(tag, 4) || std::memcmp(tag, "RDS5", 4) ||
        !input.read(reinterpret_cast<char*>(&boundaries), sizeof(boundaries)) || boundaries < 4) return nullptr;
    input.seekg(24);
    if (!input.read(reinterpret_cast<char*>(&inputs), 3 * sizeof(Word)) ||
        std::fread(&inputs.first_bias, sizeof(Word), 1, stdin) != 1) return nullptr;
    void* result = nullptr;
    if (FAILED(DStorageGetFactory(nullptr, &result)) || !result) return nullptr;
    auto factory = static_cast<Ptr>(result);
    const GUID* iid = nullptr;
    const HRESULT file_result = invoke<4, HRESULT>(factory, L"fixture.ttpf", iid, &file);
    const HRESULT queue_result = invoke<10, HRESULT>(factory, static_cast<const void*>(nullptr), iid, &queue);
    const HRESULT status_result = invoke<5, HRESULT>(factory, 2u, "BC pages", iid, &status);
    invoke<2, unsigned>(factory);
    trace = std::fopen("ranges.txt", "w");
    if (FAILED(file_result) || FAILED(queue_result) || FAILED(status_result) || !trace) {
        CloseBatch cleanup;
        return nullptr;
    }
    blocks.fill(0xa5);
    return &inputs;
}

extern "C" __declspec(dllexport) unsigned enqueue_page(Word offset, unsigned size)
{
    if (count >= 2 || !size) return 0;
    alignas(8) unsigned char record[0x48]{};
    const Word fields[]{1, reinterpret_cast<Word>(blocks.data() + page * count), 0, page,
                        reinterpret_cast<Word>(file), offset, size};
    std::memcpy(record, fields, sizeof(fields));
    invoke<3, void>(queue, record);
    invoke<4, void>(queue, status, count++);
    std::fprintf(trace, "request %llu %u\n", static_cast<unsigned long long>(offset), size);
    return size;
}

extern "C" __declspec(dllexport) unsigned draw_pages()
{
    CloseBatch cleanup;
    if (count != 2) return 3;
    invoke<6, void>(queue);
    const auto deadline = GetTickCount64() + 5000;
    while (!invoke<3, bool>(status, 1u)) {
        if (GetTickCount64() >= deadline) return 4;
        Sleep(1);
    }
    bool failed = false;
    for (unsigned i = 0; i < count; ++i) {
        const auto hr = invoke<4, HRESULT>(status, i);
        std::fprintf(trace, "status %08lx\n", static_cast<unsigned long>(hr));
        failed |= FAILED(hr);
    }
    std::ofstream pages("pages.bin", std::ios::binary);
    if (!pages.write(reinterpret_cast<const char*>(blocks.data()), blocks.size())) return 5;
    if (failed) return 6;
    QuadFrames frames;
    const HRESULT created = frames.create(128);
    if (FAILED(created)) return 7;
    char adapter[320]{};
    if (FAILED(frames.adapter(adapter, sizeof(adapter)))) return 7;
    std::array<unsigned char, 128 * 128 * 4> pixels;
    TextureImage image{blocks.data(), blocks.size(), 64, 128, DXGI_FORMAT_BC7_UNORM};
    FrameResult result{};
    const HRESULT drawn = frames.draw(&image, pixels.data(), pixels.size(), result);
    const ULONG refs = frames.release();
    std::fprintf(trace, "adapter %s\nfence %llu\nrelease %lu\n", adapter, result.fence, refs);
    if (FAILED(drawn) || refs) return 8;
    return std::ofstream("pixels.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(pixels.data()), pixels.size()) ? 0 : 9;
}
