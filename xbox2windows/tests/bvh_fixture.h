#pragma once
#include "desktop_gpu.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bvh_fixture {
using Microsoft::WRL::ComPtr;
using Word = std::uint32_t;
using Row = std::array<Word,4>;
struct Kernel {
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    Kernel(ID3D12Device* device, D3D12_SHADER_BYTECODE shader) {
        check(device->CreateRootSignature(0, shader.pShaderBytecode, shader.BytecodeLength, IID_PPV_ARGS(&root)), "Create BVH fixture root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.Get();
        desc.CS = shader;
        check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create BVH fixture pipeline");
    }
};
template<class T>
inline ComPtr<ID3D12Resource> upload(ID3D12Device* device, const std::vector<T>& rows) {
    const size_t bytes = rows.size()*sizeof(T);
    auto buffer = make_buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE empty{};
    check(buffer->Map(0,&empty,&mapped),"Upload BVH fixture input");
    std::memcpy(mapped,rows.data(),bytes);
    buffer->Unmap(0,nullptr);
    return buffer;
}
inline std::vector<Word> read(ID3D12Resource* buffer, size_t words) {
    const D3D12_RANGE range{0,words*sizeof(Word)}, empty{};
    void* mapped = nullptr;
    check(buffer->Map(0,&range,&mapped),"Map BVH fixture results");
    auto* begin = static_cast<const Word*>(mapped);
    std::vector<Word> result(begin,begin+words);
    buffer->Unmap(0,&empty);
    return result;
}
inline void equal(Word actual, Word expected, unsigned record, unsigned field) {
    if (actual == expected) return;
    std::fprintf(stderr,"record=%u field=%u expected=%08x actual=%08x\n",record,field,expected,actual);
    throw std::runtime_error("BVH fixture result differs from CPU reference");
}
}
