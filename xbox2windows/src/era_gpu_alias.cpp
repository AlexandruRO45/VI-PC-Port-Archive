#include "era_gpu_alias.h"

namespace era_gpu_alias {
HRESULT open(ID3D12Device* device, void* base, std::uint64_t size, Alias& alias) noexcept
{
    alias = {};
    if (!device || !base || size == 0) {
        return E_INVALIDARG;
    }
    // OpenExistingHeapFromAddress requires the base of an allocation (debug layer message 1088
    // otherwise). Reject anything else here so an invalid pointer never reaches D3D12.
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(base, &info, sizeof(info)) != sizeof(info) || info.AllocationBase != base ||
        info.State != MEM_COMMIT) {
        return E_INVALIDARG;
    }
    D3D12_FEATURE_DATA_EXISTING_HEAPS existing{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing, sizeof(existing))) ||
        !existing.Supported) {
        return E_NOTIMPL;
    }
    Microsoft::WRL::ComPtr<ID3D12Device3> device3;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device3)))) {
        return E_NOTIMPL;
    }
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    HRESULT hr = device3->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        return hr;
    }
    if (heap->GetDesc().SizeInBytes < size) {
        return E_INVALIDARG;
    }
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Existing heaps are cross-adapter heaps, which only accept cross-adapter resources.
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    hr = device->CreatePlacedResource(heap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                      IID_PPV_ARGS(&buffer));
    if (FAILED(hr)) {
        return hr;
    }
    alias.heap = heap;
    alias.buffer = buffer;
    alias.size = size;
    return S_OK;
}
}
