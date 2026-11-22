#include <d3d12.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstddef>
#include <string_view>

using Microsoft::WRL::ComPtr;

int main(int argc, char** argv)
{
    static_assert(sizeof(void*) == 8, "Use the x64 developer shell");
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants.Num32BitValues = 4;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC input{};
    input.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    input.Desc_1_0.NumParameters = 1;
    input.Desc_1_0.pParameters = &parameter;
    const bool bvh_flags = argc == 2 && std::string_view(argv[1]) == "bvh-flags";
    static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_0) +
                  offsetof(D3D12_ROOT_SIGNATURE_DESC, Flags) == 0x28);
    if (bvh_flags) {
        D3D12_ROOT_PARAMETER1 root_parameter{};
        root_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        root_parameter.Constants.Num32BitValues = 4;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = 1;
        desc.Desc_1_1.pParameters = &root_parameter;
        ComPtr<ID3DBlob> control;
        HRESULT plain = D3D12SerializeVersionedRootSignature(&desc, &control, nullptr);
        std::printf("v1.1 control=0x%08lx", static_cast<unsigned long>(plain));
        bool rejected = SUCCEEDED(plain) && control.Get();
        // Each console bit is tested alone before the pair, so a change in one bit is visible on its own.
        for (const unsigned flags : {0x01000000u, 0x02000000u, 0x03000000u}) {
            ComPtr<ID3DBlob> observed;
            desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
            HRESULT tagged = D3D12SerializeVersionedRootSignature(&desc, &observed, nullptr);
            std::printf(" flags_%08x=0x%08lx", flags, static_cast<unsigned long>(tagged));
            rejected = rejected && tagged == E_INVALIDARG && !observed.Get();
        }
        std::puts("");
        return rejected ? 0 : 1;
    }
    ComPtr<ID3DBlob> blob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&input, &blob, nullptr);
    if (FAILED(hr)) {
        std::fprintf(stderr, "Serialize failed (0x%08lx)\n", static_cast<unsigned long>(hr));
        return 1;
    }

    ComPtr<ID3D12RootSignatureDeserializer> reader;
    hr = D3D12CreateRootSignatureDeserializer(
        blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&reader));
    if (FAILED(hr)) {
        std::fprintf(stderr, "Deserialize failed (0x%08lx)\n", static_cast<unsigned long>(hr));
        return 1;
    }
    const auto* desc = reader->GetRootSignatureDesc();
    if (desc->Flags != input.Desc_1_0.Flags) return 1;
    if (desc->NumParameters != 1 ||
        desc->pParameters[0].ParameterType != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
        desc->pParameters[0].Constants.Num32BitValues != 4) {
        std::fputs("Root constants changed after serialization\n", stderr);
        return 1;
    }
    return 0;
}
