#include "desktop_gpu.h"
#include "CopyBlocks.h"
#include "FillBlocks.h"
#include "NumberBlocks.h"
#include "CopySpan.h"
#include "ResetBounds.h"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
enum class Operation { copy, fill, number, span };
constexpr UINT capacity = 384;
using Words = std::array<UINT, capacity>;
using Pipelines = std::array<ComPtr<ID3D12PipelineState>, 5>;
constexpr UINT seed = 0xfffffffcu;
constexpr UINT sentinel = 0xcdfa1538u;

struct Case {
    Operation op;
    UINT count;
    UINT groups;
    bool reset;
    Words expected;
};

Case prepare_case(Operation op, UINT count, UINT groups, bool reset, Words& source)
{
    Case result{op, count, groups, reset, {}};
    result.expected.fill(sentinel);
    for (UINT i = 0; i < capacity; ++i) source[i] = (0x7133u + i * 97u) ^ (i << 15);
    source[1] = count;
    const UINT active_words = op == Operation::span ? 16 * (count + 1) : 4 * count;
    if (active_words > capacity || groups == 0) throw std::runtime_error("Invalid buffer fixture");
    for (UINT i = 0; i < active_words; ++i) {
        switch (op) {
        case Operation::copy: case Operation::span: result.expected[i] = source[i]; break;
        case Operation::fill: result.expected[i] = seed; break;
        case Operation::number: result.expected[i] = seed + i; break;
        }
    }
    if (reset) {
        std::fill_n(result.expected.begin() + 4, 3, 0x7f7fffffu);
        std::fill_n(result.expected.begin() + 7, 3, 0xff7fffffu);
    }
    return result;
}
}

void check_buffer_ops(ID3D12Device* device)
{
    const D3D12_SHADER_BYTECODE code[] = {{CopyBlocks, sizeof(CopyBlocks)}, {FillBlocks, sizeof(FillBlocks)},
        {NumberBlocks, sizeof(NumberBlocks)}, {CopySpan, sizeof(CopySpan)}, {ResetBounds, sizeof(ResetBounds)}};
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, CopyBlocks, sizeof(CopyBlocks), IID_PPV_ARGS(&root)), "Create buffer ops root");
    Pipelines pipelines;
    for (size_t i = 0; i < pipelines.size(); ++i) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.Get();
        desc.CS = code[i];
        check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipelines[i])), "Create buffer operation PSO");
    }
    constexpr size_t case_count = 26;
    auto upload = make_buffer(device, (case_count + 1) * sizeof(Words), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = make_buffer(device, sizeof(Words), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, case_count * sizeof(Words), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    void* pointer = nullptr;
    const D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &pointer), "Map buffer fixture input");
    auto* upload_words = static_cast<Words*>(pointer);
    std::vector<Case> cases;
    cases.reserve(case_count);
    for (auto op : {Operation::copy, Operation::fill, Operation::number})
        for (UINT count : {0u, 1u, 31u, 32u, 33u, 65u})
            cases.push_back(prepare_case(op, count, 3, false, upload_words[cases.size()]));
    for (UINT groups : {1u, 2u})
        for (UINT count : {0u, 16u})
            for (bool reset : {false, true})
                cases.push_back(prepare_case(Operation::span, count, groups, reset, upload_words[cases.size()]));
    upload_words[case_count].fill(sentinel);
    upload->Unmap(0, nullptr);

    ComputeCommands commands(device, pipelines[0].Get());
    auto* list = commands.list.Get();
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
    auto output_state = D3D12_RESOURCE_STATE_COMMON;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        transition(list, output.Get(), output_state, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(output.Get(), 0, upload.Get(), case_count * sizeof(Words), sizeof(Words));
        transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(pipelines[static_cast<size_t>(c.op)].Get());
        list->SetComputeRootShaderResourceView(0, upload->GetGPUVirtualAddress() + i * sizeof(Words));
        std::array<UINT, 16> settings{c.groups * 32, c.op == Operation::span ? static_cast<UINT>(c.reset) : c.count, seed};
        list->SetComputeRoot32BitConstants(2, static_cast<UINT>(settings.size()), settings.data(), 0);
        list->Dispatch(c.groups, 1, 1);
        if (c.reset) {
            D3D12_RESOURCE_BARRIER order{};
            order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            order.UAV.pResource = output.Get();
            list->ResourceBarrier(1, &order);
            list->SetPipelineState(pipelines.back().Get());
            list->Dispatch(1, 1, 1);
        }
        transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        output_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->CopyBufferRegion(readback.Get(), i * sizeof(Words), output.Get(), 0, sizeof(Words));
    }
    commands.submit_and_wait(device);

    const D3D12_RANGE range{0, case_count * sizeof(Words)};
    check(readback->Map(0, &range, &pointer), "Map buffer operation results");
    const auto* actual_words = static_cast<const Words*>(pointer);
    bool pass = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto mismatch = std::mismatch(c.expected.begin(), c.expected.end(), actual_words[i].begin());
        if (mismatch.first != c.expected.end()) {
            std::fprintf(stderr, "operation=%u count=%u groups=%u reset=%u word=%zu expected=%08x actual=%08x\n",
                         static_cast<unsigned>(c.op), c.count, c.groups, static_cast<unsigned>(c.reset),
                         static_cast<size_t>(mismatch.first - c.expected.begin()), *mismatch.first, *mismatch.second);
            pass = false;
            break;
        }
    }
    readback->Unmap(0, &empty);
    if (!pass) throw std::runtime_error("Buffer operation disagrees with CPU result");
    std::printf("buffer_cases=%zu words_per_case=%u mismatches=0\n", cases.size(), capacity);
}
