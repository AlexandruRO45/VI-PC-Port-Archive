// Lanterns: small D3D12 helpers the GPU files share. The heap-handle calls through slots 9/10 and the V# encoding are
// ported from test/d3d12_translated_draw_test.cpp (M1.5), as lanterns_gpu.cpp first had them.
#pragma once

#include "../d3d12_test_shaders.h"
#include "../../src/gfx10_descriptors.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "lanterns: %s:%d: %s\n", __FILE__, __LINE__, #condition); return false; } \
} while (0)

namespace lanterns {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

// The heap's first CPU handle (slot 9) and, for a shader-visible heap, its first GPU handle (slot 10).
inline std::uint64_t heap_cpu(ID3D12DescriptorHeap* heap) {
    return reinterpret_cast<XboxHeapHandleFn>(d3d12test::slot(heap, 9))(heap);
}
inline std::uint64_t heap_gpu(ID3D12DescriptorHeap* heap) {
    return reinterpret_cast<XboxHeapHandleFn>(d3d12test::slot(heap, 10))(heap);
}

inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::uint64_t align(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// Waits until the fence reaches value, for at most kFenceWaitMs: a lost signal or a hung GPU fails the check with a
// line saying so, instead of blocking the guest until run_lanterns.py's timeout kills it. It polls
// GetCompletedValue, so it needs no event object.
constexpr ULONGLONG kFenceWaitMs = 30000;
template <class Fence>
bool wait_fence(Fence& fence, UINT64 value) {
    const ULONGLONG deadline = GetTickCount64() + kFenceWaitMs;
    for (UINT64 now = fence->GetCompletedValue(); now < value; now = fence->GetCompletedValue()) {
        if (GetTickCount64() > deadline) {
            std::fprintf(stderr, "lanterns: a fence stayed at %llu for %llu ms, waiting for %llu\n",
                         static_cast<unsigned long long>(now), kFenceWaitMs, static_cast<unsigned long long>(value));
            return false;
        }
        Sleep(1);
    }
    return true;
}

// A V# over records of 'stride' bytes, format 32_32_32_32_FLOAT, structured out-of-range checks (an idxen access).
inline std::array<std::uint32_t, 4> make_vsharp(std::uint64_t base, std::uint32_t stride, std::uint32_t records) {
    leonida::gfx10::Buffer buffer;
    buffer.address = base;
    buffer.stride = stride;
    buffer.num_records = records;
    buffer.format = 77;  // GFX10_FORMAT_32_32_32_32_FLOAT
    buffer.oob_select = leonida::gfx10::OobSelect::structured;
    return leonida::gfx10::encode(buffer);
}

// A raw V# (stride 0: offen addressing checked against a byte count).
inline std::array<std::uint32_t, 4> make_raw(std::uint64_t base, std::uint32_t bytes) {
    leonida::gfx10::Buffer buffer;
    buffer.address = base;
    buffer.num_records = bytes;
    buffer.format = 77;
    buffer.oob_select = leonida::gfx10::OobSelect::raw;
    return leonida::gfx10::encode(buffer);
}

}  // namespace lanterns
