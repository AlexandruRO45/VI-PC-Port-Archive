#pragma once
#include "bvh_fixture.h"

enum class ScratchInput { bypass, zero, pattern };

struct DispatchScratch {
    DispatchScratch(ID3D12Device* device, ID3D12Resource* provider_scratch, ScratchInput input);
    void produce(ID3D12GraphicsCommandList* list);
    void bind_consumer(ID3D12GraphicsCommandList* list) const;
    void finish(ID3D12GraphicsCommandList* list);
    void verify() const;
    float expected_history_fraction(unsigned x, unsigned y) const;

    ScratchInput mode;
    ID3D12Resource* work;
    bvh_fixture::Kernel tensor, padding;
    Microsoft::WRL::ComPtr<ID3D12Resource> upload, back, trace;
};
