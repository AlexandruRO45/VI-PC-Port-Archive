#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>

#include "pso_compute_embedded.h"
#include "pso_compute_stripped.h"

using Microsoft::WRL::ComPtr;

int main()
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::fputs("Could not create the D3D12 WARP device\n", stderr);
        return 1;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS = {pso_compute_stripped, sizeof(pso_compute_stripped)};
    ComPtr<ID3D12PipelineState> pipeline;
    const HRESULT missing = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline));
    if (missing != E_INVALIDARG || pipeline) {
        std::fprintf(stderr, "Unexpected result without a root signature: 0x%08lx\n", missing);
        return 1;
    }

    ComPtr<ID3D12RootSignature> signature;
    const HRESULT root = device->CreateRootSignature(0, pso_compute_embedded, sizeof(pso_compute_embedded),
                                                     IID_PPV_ARGS(&signature));
    if (FAILED(root)) {
        std::fprintf(stderr, "Could not create the fixture root signature: 0x%08lx\n", root);
        return 1;
    }
    desc.pRootSignature = signature.Get();
    const HRESULT supplied = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline));
    std::printf("without_root=0x%08lx with_matching_root=0x%08lx\n", missing, supplied);
    return FAILED(supplied) || !pipeline ? 1 : 0;
}
