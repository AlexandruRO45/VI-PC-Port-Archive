// Helpers for the D3D12 pipeline tests: wrap a synthetic FSSM program in a shader container shaped like the game's
// (DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads), serialize a
// root signature with the host PC runtime, and write a translator-style index next to compiled test modules.
// No game data: the programs are a few marker words, and the modules are GLSL written to the translator's contract.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace d3d12test {

inline void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// zlib stored stream (RFC 1950 / RFC 1951 section 3.2.4), splitting payloads larger than 65,535 bytes across multiple
// non-compressed blocks so multi-64 KiB synthetic FSSM containers inflate cleanly.
inline std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    static constexpr std::size_t kMaxBlockBytes = 65535;
    const std::size_t block_count = data.empty() ? 1 : (data.size() + kMaxBlockBytes - 1) / kMaxBlockBytes;
    std::vector<std::uint8_t> out;
    out.reserve(2 + block_count * 5 + data.size() + 4);
    out.push_back(0x78);
    out.push_back(0x01);
    std::size_t offset = 0;
    do {
        const std::size_t chunk = std::min<std::size_t>(data.size() - offset, kMaxBlockBytes);
        const bool last = (offset + chunk == data.size());
        const auto length = std::uint16_t(chunk);
        const auto nlen = std::uint16_t(~length);
        out.insert(out.end(), {std::uint8_t(last ? 0x01 : 0x00), std::uint8_t(length), std::uint8_t(length >> 8),
                               std::uint8_t(nlen), std::uint8_t(nlen >> 8)});
        out.insert(out.end(), data.begin() + std::ptrdiff_t(offset), data.begin() + std::ptrdiff_t(offset + chunk));
        offset += chunk;
    } while (offset < data.size());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// A synthetic FSSM program: the magic, then `words` (distinct per test shader so their hashes differ).
inline std::vector<std::uint8_t> fssm_program(std::initializer_list<std::uint32_t> words) {
    std::vector<std::uint8_t> out = {'F', 'S', 'S', 'M'};
    for (const std::uint32_t w : words) put32(out, w);
    return out;
}

inline std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

inline std::vector<std::uint8_t> serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!d3d12) return {};
    struct ModuleGuard {
        HMODULE module;
        ~ModuleGuard() { FreeLibrary(module); }
    } const module_guard{d3d12};
    const auto fn = reinterpret_cast<SerializeFn>(
        reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")));
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        struct BlobGuard {
            ID3DBlob* ptr;
            ~BlobGuard() { ptr->Release(); }
        } const blob_guard{blob};
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
    }
    return out;
}

inline D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

inline D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

template<class T>
struct Ref {
    T* p = nullptr;

    Ref() noexcept = default;
    explicit Ref(T* ptr) noexcept : p(ptr) {}
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& other) noexcept : p(std::exchange(other.p, nullptr)) {}
    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) {
            reset();
            p = std::exchange(other.p, nullptr);
        }
        return *this;
    }
    ~Ref() { reset(); }

    void reset() noexcept {
        if (T* old = std::exchange(p, nullptr)) old->Release();
    }
    T* operator->() const noexcept { return p; }
};

inline void* slot(void* object, unsigned index) {
    if (!object) return nullptr;
    void** const vtable = *reinterpret_cast<void***>(object);
    return vtable ? vtable[index] : nullptr;
}

}  // namespace d3d12test
